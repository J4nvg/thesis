"""Stage 3: one ``(model_variant, paradigm, seed, stage)`` backtest, end to end.

What :func:`run_stage` does, in order (plan §5.3, §5.4, §8 P3):

1. resolve the stage identity ``hash(resolved stage config + upstream hashes +
   seed)`` and **skip** the stage when the store says it is already complete
   under that identity;
2. ``seed_everything(seed)`` and snapshot ``config.yaml`` / ``env.json``;
3. build the forecaster from the spec, the resolved parameters (the experiment
   entry's overrides, else ``tuning/<model>/best_params.json``, else the spec's
   legacy defaults) and a :class:`~strikecast.models.spec.RunContext`;
4. run :class:`~strikecast.backtest.engine.ExpandingWindowBacktest` under the
   paradigm's grouping, streaming :class:`PredictionSet` parts to disk through
   :class:`~strikecast.store.run_store.PersistHook`;
5. reload the parts, compute the ``evaluate_long`` views and write them;
6. mark the stage complete.

Behaviour notes, all legacy-faithful and all flagged in ``docs/REFACTOR_PLAN.md``
§4:

* **Which targets.** ``cv`` runs on ``target_cv_view`` from ``cv_start_frac``;
  ``test`` runs on ``target_full`` from ``train_val_end`` (F17, F19). Both come
  from the stage config, never from a constant here.
* **F80.** For the diff family's ``cv`` stage the model-space list is
  ``Diff(full targets)`` split at ``train_val_end`` -- one step longer than
  differencing the level CV view -- so it is passed to the engine explicitly.
* **F81.** The adapter preset is per paradigm per stage: the count family's
  Global CV takes the 200-sample median (``for_cv``) while its Activity and
  Local CV go through ``for_test``. :meth:`StageConfig.adapter_for` owns that
  table; this module only reads it.
* **F6 / F67.** The MASE/RMSSE denominators come from the stage's
  ``naive_scales`` block: train for every count/diff stage, train+val for the
  hurdle's test stage.
* **Activity views.** The legacy scripts call ``evaluate_long`` WITHOUT
  ``regions_activity`` for the Global paradigm and WITH it for Activity and
  Local (``_regression_GBDT.py``:988/1207/1308 vs :1478ff). Reproduced exactly,
  so a Global stage writes four metric views and the other two write six.
* **Resume.** A complete stage under the same hash is skipped. A *partial*
  stage restarts from fold 0 and overwrites its own parts: the engine cannot
  start mid-schedule today (see ``RunStore.resume_point``), so the resume point
  is logged rather than honoured. Nothing is lost, only recomputed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strikecast.backtest.engine import ExpandingWindowBacktest
from strikecast.backtest.grouping import partition, take
from strikecast.backtest.hooks import ProgressHook
from strikecast.backtest.predictions import PredictionSet
from strikecast.backtest.protocols import SINGLE_CHANNEL, FoldResult
from strikecast.backtest.schedule import schedule_from_config
from strikecast.pipeline import data_stage as _data_stage
from strikecast.pipeline.context import (
    get_spec,
    make_fold_hook,
    make_run_context,
    make_tracker,
    resolve_store_root,
    track,
    tracker_tags,
)
from strikecast.seeds import record_env, seed_everything
from strikecast.store import RunKey, RunStore, stage_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from darts import TimeSeries

    from strikecast.config.schema import ExperimentConfig, Paradigm, StageConfig
    from strikecast.models.spec import ModelSpec, RunContext
    from strikecast.pipeline.data_stage import DataArtifacts

__all__ = [
    "MissingTunedParams",
    "StageOutcome",
    "TunedParamsStale",
    "check_tuned_features",
    "make_forecaster",
    "plan_selections",
    "require_tuned_params",
    "resolve_params",
    "run_experiment",
    "run_stage",
    "selection_groups",
    "stage_targets",
]

logger = logging.getLogger(__name__)

#: ``evaluate``'s metric set per experiment family. Everything that forecasts a
#: count -- the count and diff families -- uses ``evaluate_long``; the composite
#: families are P5 and are mapped here so the wiring is already right.
METRIC_SET_BY_EXPERIMENT: dict[str, str] = {"hurdle": "hurdle", "damage": "classification"}


# --------------------------------------------------------------------------- #
# outcome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageOutcome:
    """What one stage produced (or skipped)."""

    run_key: RunKey
    stage: str
    stage_hash: str
    skipped: bool
    n_folds: int = 0
    n_rows: int = 0
    metric_paths: dict[str, Path] = field(default_factory=dict)
    params_source: str = ""

    @property
    def label(self) -> str:
        key = self.run_key
        return f"{key.experiment}/{key.model}/{key.paradigm}/seed={key.seed}/{self.stage}"


# --------------------------------------------------------------------------- #
# parameters and forecasters
# --------------------------------------------------------------------------- #
def resolve_params(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    model_name: str,
    store: RunStore,
    features_hash: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Parameters for one model, and where they came from.

    Order (``ModelEntry`` docstring): the experiment entry's explicit
    ``params`` win, then the stored ``best_params.json`` of the tuning stage
    mapped through ``spec.from_best_params`` (which is the identity for the
    GBDTs and the ``_build_lstm_from_best`` mapping for the RNNs), then the
    spec's legacy defaults.

    ``features_hash`` is the selection the run will train on
    (``data.features.hash``). When given, the stored study must have been tuned
    on the same selection (:func:`check_tuned_features`, plan "figure feature
    selection" §7): a mismatch raises :class:`TunedParamsStale` instead of
    silently running a study's optimum on a feature space it never saw.
    ``None`` (Chronos-2, which has no selection) skips the check.
    """
    entry = cfg.model_entry(model_name)
    if entry.params:
        return dict(entry.params), "config"

    directory = store.tuning_dir(cfg.name, model_name)
    if (directory / "best_params.json").is_file():
        from strikecast.tuning import load_best_params  # noqa: PLC0415

        if features_hash is not None:
            check_tuned_features(cfg, model_name, directory, features_hash)
        best = load_best_params(directory)
        return dict(spec.from_best_params(best)), "tuned"

    if spec.tunable:
        logger.warning(
            "model %r is tunable but %s/best_params.json does not exist; "
            "running on the spec defaults",
            model_name,
            directory,
        )
    return dict(spec.defaults), "defaults"


class TunedParamsStale(RuntimeError):
    """A study's ``best_params.json`` (or an unfinished study) belongs to a
    different feature selection than the one this run trains on.

    ``best_params.json`` and the Optuna SQLite file are keyed on the model name
    only, not on the selection, so after a re-selection (figure protocol, plan
    §7) an old study would be resumed by ``load_if_exists`` and old optima
    reused by cv/test. ``strikecast tune ... --force`` archives the directory
    as ``<model>.stale-<hash>`` and tunes afresh.
    """


def recorded_features_hash(directory: Path) -> str | None:
    """``provenance.features_hash`` of ``<directory>/best_params.json``, or ``None``.

    Studies tuned by this pipeline carry it (``tune_stage._record_feature_provenance``);
    the imported thesis params (``scripts/import_golden_params.py``, audit B1)
    carry a provenance block without it.
    """
    import json  # noqa: PLC0415

    path = directory / "best_params.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    value = provenance.get("features_hash") if isinstance(provenance, dict) else None
    return str(value) if value else None


def selection_protocol(cfg: ExperimentConfig) -> str:
    """``"legacy"`` or ``"figure"``: the protocol of ``cfg``'s (narrowed) selection."""
    return str(cfg.feature_selection.protocol)


def check_tuned_features(
    cfg: ExperimentConfig, model_name: str, directory: Path, features_hash: str
) -> None:
    """Refuse a ``best_params.json`` tuned on another selection (plan §7).

    * recorded ``features_hash`` != ``features_hash`` -> :class:`TunedParamsStale`;
    * no recorded hash (the imported thesis params) -> accepted only under the
      legacy protocol (``legacy=<family>``), where they reproduce the thesis;
      under the figure protocol they were tuned on the thesis selection, which
      no figure run trains on.
    """
    recorded = recorded_features_hash(directory)
    hint = (
        f"Re-tune with `strikecast tune experiment={cfg.name} model={model_name} --force` "
        f"(it archives {directory.name} as {directory.name}.stale-<hash>, nothing is deleted)."
    )
    if recorded is not None:
        if recorded != features_hash:
            raise TunedParamsStale(
                f"{directory}/best_params.json was tuned on feature selection "
                f"{recorded} but this run trains on {features_hash}. {hint}"
            )
        return
    if selection_protocol(cfg) != "legacy":
        raise TunedParamsStale(
            f"{directory}/best_params.json records no features_hash: these are imported "
            "thesis params, tuned on the thesis selection, and are valid only under the "
            f"legacy protocol (legacy={cfg.name}). This run uses the figure protocol "
            f"(selection {features_hash}). {hint}"
        )


class MissingTunedParams(RuntimeError):
    """A tunable model reached ``cv``/``test`` without its ``best_params.json``."""


def require_tuned_params(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    model_name: str,
    params_source: str,
    store: RunStore,
    *,
    allow_default_params: bool = False,
) -> None:
    """Refuse to run a tunable model on the spec defaults (audit C17).

    Before this check a ``cv``/``test`` job that started before its ``tune``
    finished ran on the defaults with only a WARNING, completed, and was then
    skipped forever by the job generator. ``allow_default_params`` (CLI
    ``--allow-default-params``) is the explicit opt-out: the ``--benchmark``
    pilot and ad-hoc laptop runs use it. The stage identity includes the
    parameters, so such a stage is recomputed once tuned parameters exist.
    """
    if allow_default_params or not spec.tunable or params_source != "defaults":
        return
    raise MissingTunedParams(
        f"model {model_name!r} is tunable but "
        f"{store.tuning_dir(cfg.name, model_name)}/best_params.json does not exist. "
        f"Run `strikecast tune experiment={cfg.name} model={model_name}` first (or "
        "import the thesis' params with scripts/import_golden_params.py), or pass "
        "--allow-default-params to run on the spec defaults deliberately."
    )


def _fallback_builder(cfg: ExperimentConfig, spec: ModelSpec, model_name: str):
    """The F5 ``NaiveMean`` fallback, from the spec or from the experiment entry."""
    entry = cfg.model_entry(model_name)
    if entry.fallback == "naive_mean":
        def _naive_mean() -> Any:
            from darts.models import NaiveMean  # noqa: PLC0415

            return NaiveMean()

        return _naive_mean
    return spec.fallback


def make_forecaster(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    model_name: str,
    params: dict[str, Any],
    ctx: RunContext,
    *,
    preset: str,
) -> Any:
    """Build the adapter the engine drives.

    ``preset`` is one of ``for_cv`` / ``for_test`` / ``for_tuning`` (F55, F56,
    F81) and only applies to :class:`GlobalDartsForecaster`; the local adapter
    has no post-processing to preset, exactly as the legacy ``is_local`` branch
    has none.
    """
    from strikecast.models.adapters import (  # noqa: PLC0415
        GlobalDartsForecaster,
        LocalDartsForecaster,
    )

    def builder() -> Any:
        return spec.build(params, ctx)

    if spec.kind == "local":
        return LocalDartsForecaster(
            builder, fallback=_fallback_builder(cfg, spec, model_name)
        )
    if spec.kind == "global":
        factory = getattr(GlobalDartsForecaster, preset, None)
        if factory is None:
            raise ValueError(
                f"unknown adapter preset {preset!r}; expected for_cv, for_test or for_tuning"
            )
        return factory(builder, is_neural=spec.is_neural)
    if spec.kind == "composite":
        raise ValueError(
            f"model {model_name!r} is a composite: it has no single darts adapter. "
            "`run_stage` dispatches composites to "
            "`strikecast.pipeline.composite_stage.run_composite_stage`, which builds "
            "the forecaster through `strikecast.models.hurdle` instead."
        )
    if spec.kind == "chronos":
        # Audit C15: a Chronos-2 "model" is an AutoGluon predictor fit once on
        # the AutoGluon frame, not a darts adapter built per group. `run_stage`
        # dispatches it to `chronos_stage.run_chronos_stage`, which builds the
        # `Chronos2Forecaster` (strikecast.models.chronos) from the data there.
        raise ValueError(
            f"model {model_name!r} is a Chronos-2 fixed predictor: `run_stage` "
            "dispatches it to `strikecast.pipeline.chronos_stage.run_chronos_stage`, "
            "which fits/loads the AutoGluon predictor and builds the Chronos2Forecaster."
        )
    raise ValueError(f"model {model_name!r} has kind {spec.kind!r}, which run_stage cannot drive")


def _transform(cfg: ExperimentConfig) -> Any:
    if cfg.transform.kind == "diff":
        from strikecast.transforms.diff import Diff  # noqa: PLC0415

        return Diff()
    from strikecast.transforms.identity import Identity  # noqa: PLC0415

    return Identity()


# --------------------------------------------------------------------------- #
# series selection
# --------------------------------------------------------------------------- #
def stage_targets(data: DataArtifacts, stage_name: str) -> list[TimeSeries]:
    """The LEVEL-space target list of a stage.

    ``cv`` -> ``target_cv_view`` (the series truncated at ``train_val_end``),
    ``test`` -> ``target_full``. Exactly the two lists the legacy scripts hand
    ``run_expanding_cv`` and ``run_final_test``.
    """
    if stage_name == "cv":
        return list(data.bundle.target_cv_view)
    if stage_name == "test":
        return list(data.bundle.target_full)
    raise KeyError(f"unknown stage {stage_name!r}; expected 'cv' or 'test'")


def model_targets_for(
    cfg: ExperimentConfig, data: DataArtifacts, stage_name: str
) -> list[TimeSeries] | None:
    """The explicit model-space override, or ``None`` to let the engine derive it.

    Only the diff family's CV stage needs one (F80): the legacy script
    differences the FULL target list and only then takes the CV view, so its
    model-space list is one step longer than ``Diff.forward(level CV view)``.
    Reproduced by :func:`strikecast.data.series.model_space_parts`, which
    differences the bundle's un-encoded full target list and re-runs the legacy
    encode+split on it -- literally what ``_diff_regression.py`` lines 234-239
    do. The figure protocol's ``diff_l2`` selector fits on the ``target_train``
    of the same construction, so the two can never drift apart.
    """
    if cfg.transform.kind != "diff" or stage_name != "cv":
        return None

    if data.bundle.raw is None:
        raise ValueError(
            "the diff family's CV stage needs the bundle's un-encoded lists (F80); "
            "this bundle has raw=None"
        )
    from strikecast.data.series import model_space_parts  # noqa: PLC0415

    return list(model_space_parts(data.bundle)["target_cv_view"])


def _covariates(
    spec: ModelSpec, data: DataArtifacts
) -> tuple[list[TimeSeries], list[TimeSeries]]:
    """Past and future covariates for one model (F28: RNNs take the raw past)."""
    bundle = data.bundle
    past = bundle.raw_past_covs if spec.needs_raw_past_covs else bundle.past_covs
    return list(past), list(bundle.future_covs)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _naive_scales(
    data: DataArtifacts, stage_cfg: StageConfig
) -> tuple[dict[str, float], dict[str, float]]:
    from strikecast.evaluation.metrics import compute_naive_scales  # noqa: PLC0415

    bundle = data.bundle
    cfgn = stage_cfg.naive_scales
    if cfgn.fit_on == "train_val":
        targets = [
            tr.append(vl) for tr, vl in zip(bundle.target_train, bundle.target_val, strict=True)
        ]
    else:
        targets = list(bundle.target_train)
    return compute_naive_scales(targets, bundle.region_names, seasonality=cfgn.seasonality)


def _metric_set(cfg: ExperimentConfig) -> str:
    return METRIC_SET_BY_EXPERIMENT.get(cfg.name, "count")


def _evaluate(
    cfg: ExperimentConfig,
    data: DataArtifacts,
    stage_cfg: StageConfig,
    paradigm: str,
    preds: PredictionSet,
) -> dict[str, Any]:
    from strikecast.evaluation.aggregate import evaluate  # noqa: PLC0415

    mae_scales, rmse_scales = _naive_scales(data, stage_cfg)
    # The legacy Global evaluations pass no activity map and therefore write no
    # per-activity views; the Activity and Local ones do. Preserved verbatim.
    activity = None if paradigm == "global" else (data.activity_by_region or None)
    return evaluate(
        preds,
        mae_scales,
        rmse_scales,
        activity_by_region=activity,
        metric_set=_metric_set(cfg),  # type: ignore[arg-type]
        threshold=cfg.calibration.threshold,
    )


# --------------------------------------------------------------------------- #
# persistence helpers
# --------------------------------------------------------------------------- #
def _persist_fold_preds(
    hook: Any,
    folds: Sequence[Any],
    fold_preds: dict[str, list[list[TimeSeries]]],
) -> None:
    """Replay already-computed fold predictions through a :class:`PersistHook`.

    The grouped paradigms run one engine per group, so no single engine sees the
    whole region list and the hook cannot be attached to the loop. Replaying the
    restored-order result through the same hook keeps the on-disk layout (one
    part per retrain window) identical to the streamed case.
    """
    channels = list(fold_preds)
    for i, fold in enumerate(folds):
        predictions = {
            channel: [per_region[i] for per_region in fold_preds[channel]]
            for channel in channels
        }
        hook.on_fold(FoldResult(fold=fold, predictions=predictions), {})
    hook.close()


def _naive_fold_preds(
    model_name: str,
    targets: Sequence[TimeSeries],
    stage_cfg: StageConfig,
    start_frac: float,
) -> dict[str, list[list[TimeSeries]]]:
    """``naive_last`` / ``naive_weekly`` as the engine-shaped bundle.

    The naive baselines never fit anything, so they bypass the engine entirely,
    exactly as the legacy ``naive_collect_long`` does. They are also
    region-independent, so every paradigm produces the same numbers; the run
    store still keys them by paradigm so a leaderboard can quote them per view.
    """
    from strikecast.backtest.naive import NAIVE_METHODS  # noqa: PLC0415

    try:
        fn = NAIVE_METHODS[model_name]
    except KeyError:
        raise KeyError(
            f"{model_name!r} is not a naive method; known: {sorted(NAIVE_METHODS)}"
        ) from None
    return {
        SINGLE_CHANNEL: [
            fn(ts, start_frac, stage_cfg.horizon, stage_cfg.predict_stride) for ts in targets
        ]
    }


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def run_stage(
    cfg: ExperimentConfig,
    model_name: str,
    paradigm: Paradigm | str,
    seed: int,
    stage_name: str,
    data: DataArtifacts,
    *,
    store: RunStore | None = None,
    tracker: Any = None,
    force: bool = False,
    progress_every: int = 0,
    allow_default_params: bool = False,
    max_folds: int | None = None,
) -> StageOutcome:
    """Run (or skip) one ``(model, paradigm, seed, stage)`` backtest.

    ``allow_default_params`` lets a tunable model without ``best_params.json``
    run on the spec defaults; without it that is an error (audit C17, see
    :func:`require_tuned_params`). ``max_folds`` truncates the schedule to its
    first ``max_folds`` folds -- the ``--benchmark`` pilot and the legacy
    verification job (audit D9) run one or two retrain windows. It is part of
    the stage identity when set, so a truncated stage is never mistaken for a
    complete one, and it is absent from the identity when not set, so every
    existing stage hash is unchanged.
    """
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    spec = get_spec(model_name, cfg.name)
    extra = {"allow_default_params": allow_default_params, "max_folds": max_folds}
    if spec.kind == "composite":
        # The hurdle and damage families run two/four panels, several output
        # channels and a calibration step; `composite_stage` owns all of that
        # and returns the same `StageOutcome` (plan §8 P5).
        from strikecast.pipeline.composite_stage import run_composite_stage  # noqa: PLC0415

        return run_composite_stage(
            cfg,
            model_name,
            str(paradigm),
            seed,
            stage_name,
            data,  # type: ignore[arg-type]
            store=store,
            tracker=tracker,
            force=force,
            progress_every=progress_every,
            **_accepted(run_composite_stage, extra),
        )
    if spec.kind == "chronos":
        # Chronos-2 (audit C15): one AutoGluon predictor, fit once, on the
        # AutoGluon frame; `chronos_stage` owns the fit/reload and the metrics.
        from strikecast.pipeline.chronos_stage import run_chronos_stage  # noqa: PLC0415

        return run_chronos_stage(
            cfg,
            model_name,
            str(paradigm),
            seed,
            stage_name,
            data,
            store=store,
            tracker=tracker,
            force=force,
            progress_every=progress_every,
            **_accepted(run_chronos_stage, extra),
        )
    stage_cfg = cfg.stage(stage_name)
    backtest_cfg = stage_cfg.backtest(cfg.split)
    key = RunKey(cfg.name, model_name, str(paradigm), int(seed))
    try:
        params, params_source = resolve_params(
            cfg, spec, model_name, store, features_hash=data.features.hash
        )
        require_tuned_params(
            cfg, spec, model_name, params_source, store,
            allow_default_params=allow_default_params,
        )
    except (MissingTunedParams, TunedParamsStale) as exc:
        # Recorded so `submit_all.py status` shows WHY the job failed.
        store.fail_stage(key, stage_name, f"{type(exc).__name__}: {exc}")
        raise

    resolved_stage = {
        "experiment": cfg.name,
        "model": model_name,
        "paradigm": str(paradigm),
        "stage": stage_name,
        "backtest": backtest_cfg.model_dump(mode="json"),
        "adapter": stage_cfg.adapter_for(paradigm),  # type: ignore[arg-type]
        "naive_scales": stage_cfg.naive_scales.model_dump(mode="json"),
        "transform": cfg.transform.model_dump(mode="json"),
        "common_kwargs": cfg.common_kwargs.model_dump(mode="json"),
        "params": params,
        "device": cfg.device_for(model_name),
        "threads": cfg.threads,
    }
    if max_folds is not None:
        resolved_stage["max_folds"] = int(max_folds)
    digest = stage_hash(resolved_stage, list(data.upstream), int(seed))

    if not force and store.is_complete(key, stage_name, digest):
        logger.info("skip %s/%s: complete under the same identity", key.relative(), stage_name)
        return StageOutcome(key, stage_name, digest, skipped=True, params_source=params_source)

    resume = store.resume_point(key, stage_name)
    if resume:
        logger.info(
            "%s/%s: %d folds are already on disk, but the engine cannot start "
            "mid-schedule; recomputing from fold 0 and overwriting the parts",
            key.relative(),
            stage_name,
            resume,
        )

    owns_tracker = tracker is None
    tracker = tracker if tracker is not None else make_tracker(cfg)

    seed_everything(int(seed))
    ctx = make_run_context(cfg, model_name, int(seed), data)

    store.start_stage(
        key, stage_name, digest, params_source=params_source, max_folds=max_folds
    )
    try:
        store.write_config(key, cfg.model_dump(mode="json"))
        env = record_env()
        tracked_id = None
        try:
            tracked_id = tracker.start(
                key,
                cfg.model_dump(mode="json"),
                tags=tracker_tags(cfg, spec),
                stage=stage_name,
            )
        except Exception as exc:  # pragma: no cover - the mirror never fails a run
            logger.warning("tracker.start failed: %s", exc)
        if tracked_id:
            env = {**env, "tracker_run_id": tracked_id}
        store.write_env(key, env)
        if tracked_id:
            # §5.5: the run directory must name its mirror. The store owns the
            # merge, so `env.json` keeps the id of every stage, not just this one.
            store.record_tracker_run_id(key, tracked_id, stage=stage_name)

        level_targets = stage_targets(data, stage_name)
        override = model_targets_for(cfg, data, stage_name)
        transform = _transform(cfg)
        model_space = override if override is not None else transform.forward(list(level_targets))
        folds = schedule_from_config(model_space[0], backtest_cfg)
        if max_folds is not None:
            folds = folds[: int(max_folds)]

        persist = store.persist_hook(
            key,
            stage_name,
            backtest_cfg.retrain_stride,
            actuals=level_targets,
            region_names=data.region_names,
        )
    except BaseException as exc:
        # Anything between start_stage and the backtest must not leave the
        # stage `running` either (C21).
        if isinstance(exc, KeyboardInterrupt):
            store.interrupt_stage(key, stage_name, str(exc) or type(exc).__name__)
        else:
            store.fail_stage(key, stage_name, f"{type(exc).__name__}: {exc}")
        raise

    try:
        if spec.kind == "naive":
            fold_preds = _naive_fold_preds(
                model_name, level_targets, stage_cfg, backtest_cfg.start_frac
            )
            if max_folds is not None:
                fold_preds = {
                    channel: [list(region[: len(folds)]) for region in regions]
                    for channel, regions in fold_preds.items()
                }
            _persist_fold_preds(persist, folds, fold_preds)
        else:
            _run_backtest(
                cfg=cfg,
                spec=spec,
                model_name=model_name,
                params=params,
                ctx=ctx,
                paradigm=str(paradigm),
                stage_cfg=stage_cfg,
                backtest_cfg=backtest_cfg,
                data=data,
                level_targets=level_targets,
                model_space=override,
                transform=transform,
                persist=persist,
                tracker=tracker,
                stage_name=stage_name,
                progress_every=progress_every,
                max_folds=max_folds,
            )

        preds = store.load_predictions(key, stage_name, legacy_order=True)
        views = _evaluate(cfg, data, stage_cfg, str(paradigm), preds)
        metric_paths = store.write_metrics(key, stage_name, views)
        store.complete_stage(key, stage_name)

        track(tracker, "log_tables", views, stage=stage_name)
        track(tracker, "log_artifact", store.predictions_dir(key, stage_name), "predictions")
        track(tracker, "log_artifact", store.metrics_dir(key, stage_name), "metrics")
    except KeyboardInterrupt as exc:
        # SIGTERM/SIGUSR1 (strikecast.pipeline.interrupt) or Ctrl-C: the parts on
        # disk are intact; the stage is `interrupted`, not `failed` (audit C21).
        store.interrupt_stage(key, stage_name, str(exc) or type(exc).__name__)
        if owns_tracker:
            track(tracker, "finish", "failed")
        raise
    except Exception as exc:
        store.fail_stage(key, stage_name, f"{type(exc).__name__}: {exc}")
        if owns_tracker:
            track(tracker, "finish", "failed")
        raise
    else:
        if owns_tracker:
            track(tracker, "finish")

    logger.info(
        "%s/%s: %d folds, %d rows, %d metric views",
        key.relative(),
        stage_name,
        len(folds),
        len(preds),
        len(metric_paths),
    )
    return StageOutcome(
        key,
        stage_name,
        digest,
        skipped=False,
        n_folds=len(folds),
        n_rows=len(preds),
        metric_paths=metric_paths,
        params_source=params_source,
    )


def _run_backtest(
    *,
    cfg: ExperimentConfig,
    spec: ModelSpec,
    model_name: str,
    params: dict[str, Any],
    ctx: RunContext,
    paradigm: str,
    stage_cfg: StageConfig,
    backtest_cfg: Any,
    data: DataArtifacts,
    level_targets: list[TimeSeries],
    model_space: list[TimeSeries] | None,
    transform: Any,
    persist: Any,
    tracker: Any,
    stage_name: str,
    progress_every: int,
    max_folds: int | None = None,
) -> None:
    """Drive the engine under the paradigm's grouping.

    One group (``paradigm='global'``) streams straight through the
    :class:`PersistHook`. Several groups run one engine each -- which is exactly
    what the legacy ``*_per_activity`` / ``*_per_region`` wrappers do -- and the
    restored-order result is replayed through the same hook afterwards.
    """
    from strikecast.backtest.grouping import restore_order  # noqa: PLC0415

    preset = stage_cfg.adapter_for(paradigm)  # type: ignore[arg-type]
    past_covs, future_covs = _covariates(spec, data)
    groups = partition(data.region_names, paradigm, data.activity_by_region)  # type: ignore[arg-type]

    hooks: list[Any] = []
    if progress_every:
        hooks.append(ProgressHook(progress_every))

    if len(groups) == 1:
        fold_hook = make_fold_hook(
            tracker,
            _fold_metrics_fn(cfg, data, stage_cfg, paradigm, level_targets),
            prefix=f"{stage_name}/",
            every=max(1, backtest_cfg.retrain_stride or 1),
        )
        engine = ExpandingWindowBacktest(
            backtest_cfg, transform, hooks=[persist, *hooks, *([fold_hook] if fold_hook else [])]
        )
        forecaster = make_forecaster(cfg, spec, model_name, params, ctx, preset=preset)
        for i, _ in enumerate(
            engine.iter_folds(
                forecaster,
                level_targets,
                past_covs,
                future_covs,
                model_targets=model_space,
            )
        ):
            if max_folds is not None and i + 1 >= int(max_folds):
                break
        persist.close()  # flush the folds since the last retrain boundary
        return

    results = []
    for group in groups:
        logger.info("group %s (%d regions)", group.label, len(group.indices))
        engine = ExpandingWindowBacktest(backtest_cfg, transform, hooks=hooks)
        forecaster = make_forecaster(cfg, spec, model_name, params, ctx, preset=preset)
        args = (
            forecaster,
            [level_targets[i] for i in group.indices],
            take(past_covs, group.indices),
            take(future_covs, group.indices),
        )
        group_targets = None if model_space is None else [model_space[i] for i in group.indices]
        if max_folds is None:
            collected = engine.run(*args, model_targets=group_targets)
        else:
            collected = _run_first_folds(engine, args, group_targets, int(max_folds))
        results.append((group, collected))

    fold_preds = restore_order(results, len(level_targets))
    folds = schedule_from_config(
        (model_space or transform.forward(list(level_targets)))[0], backtest_cfg
    )
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    _persist_fold_preds(persist, folds, fold_preds)


def _run_first_folds(
    engine: ExpandingWindowBacktest,
    args: tuple[Any, ...],
    model_targets: list[TimeSeries] | None,
    max_folds: int,
) -> dict[str, list[list[TimeSeries]]]:
    """``engine.run`` stopped after ``max_folds`` folds (benchmark/verification)."""
    forecaster, targets = args[0], args[1]
    collected: dict[str, list[list[TimeSeries]]] = {
        channel: [[] for _ in targets] for channel in forecaster.channels
    }
    for i, result in enumerate(engine.iter_folds(*args, model_targets=model_targets)):
        for channel, region_preds in result.predictions.items():
            for r_idx, pred in enumerate(region_preds):
                collected[channel][r_idx].append(pred)
        if i + 1 >= max_folds:
            break
    return collected


def _accepted(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``kwargs`` that ``fn`` accepts (and that is not a default no-op).

    The composite and Chronos stages are owned by other modules and grow their
    keyword arguments independently; forwarding only what a signature names
    keeps the dispatch working while they do. A non-default value that cannot
    be forwarded is an error rather than silently dropped.
    """
    import inspect  # noqa: PLC0415

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        params = {}
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    out: dict[str, Any] = {}
    defaults = {"allow_default_params": False, "max_folds": None}
    for name, value in kwargs.items():
        if accepts_any or name in params:
            if value != defaults.get(name, object()):
                out[name] = value
        elif value != defaults.get(name, object()):
            raise TypeError(
                f"{getattr(fn, '__qualname__', fn)} does not accept {name}={value!r}"
            )
    return out


def _fold_metrics_fn(
    cfg: ExperimentConfig,
    data: DataArtifacts,
    stage_cfg: StageConfig,
    paradigm: str,
    level_targets: list[TimeSeries],
):
    """``cumulative -> global metrics``, for the tracker's per-fold mirror."""

    def _metrics(cumulative: dict[str, list[list[TimeSeries]]]) -> dict[str, Any]:
        preds = PredictionSet.from_fold_preds(
            level_targets, cumulative, data.region_names
        )
        views = _evaluate(cfg, data, stage_cfg, paradigm, preds)
        return dict(views["global"])

    return _metrics


# --------------------------------------------------------------------------- #
# selection routing
# --------------------------------------------------------------------------- #
def selection_groups(
    cfg: ExperimentConfig, models: Sequence[str] | None = None
) -> dict[str | None, list[str]]:
    """``selection key -> model names`` of the requested models.

    :meth:`ExperimentConfig.selection_groups` owns the routing
    (``ModelEntry.selection``, plan "figure feature selection" §5); ``None`` is
    the top-level ``feature_selection``. Empty groups are dropped.
    """
    names = list(models) if models else None
    return {key: list(group) for key, group in cfg.selection_groups(names).items() if group}


def plan_selections(
    cfg: ExperimentConfig,
    store: RunStore,
    *,
    data: DataArtifacts | None = None,
    models: Sequence[str] | None = None,
) -> list[tuple[ExperimentConfig, DataArtifacts, list[str]]]:
    """``(narrowed cfg, its data, its models)`` per selection group.

    Each group's data comes from ONE ``prepare_data(cfg.for_selection(key))``,
    so e.g. the count family's Poisson GBDTs train on the ``count_poisson``
    selection and its Tweedie GBDTs (and the RNNs riding along) on
    ``count_tweedie``. An explicit ``data`` holds one selection, so it is
    accepted only when every requested model routes to the same key; anything
    else would silently train a model on another branch's feature space.
    """
    groups = selection_groups(cfg, models)
    if data is not None:
        if len(groups) > 1:
            listing = "; ".join(
                f"{key or '<feature_selection>'}: {names}" for key, names in groups.items()
            )
            raise ValueError(
                f"an explicit data= holds ONE feature selection, but the requested models "
                f"of {cfg.name!r} route to {len(groups)} ({listing}). Pass data=None to "
                "prepare one per selection, or restrict models= to one group."
            )
        key, names = next(iter(groups.items()), (None, []))
        return [(cfg.for_selection(key), data, names)]
    plan: list[tuple[ExperimentConfig, DataArtifacts, list[str]]] = []
    for key, names in groups.items():
        group_cfg = cfg.for_selection(key)
        plan.append((group_cfg, _data_stage.prepare_data(group_cfg, store), names))
    return plan


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def run_experiment(
    cfg: ExperimentConfig,
    *,
    data: DataArtifacts | None = None,
    models: Sequence[str] | None = None,
    paradigms: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    stages: Sequence[str] = ("cv", "test"),
    store: RunStore | None = None,
    force: bool = False,
    progress_every: int = 0,
    allow_default_params: bool = False,
    max_folds: int | None = None,
) -> list[StageOutcome]:
    """Run the cartesian product of models x paradigms x seeds x stages.

    Two rules of §5.4 live here:

    * the **CV stage stays single-seed** -- it is a tuning-and-selection view,
      and the thesis ran it once, under the tuning seed. Seeds sweep the test
      stage only;
    * a **deterministic** model (``stochastic=False``: the naives, linear,
      ARIMA) runs once, under the first seed, and is broadcast across seeds by
      the report (§7.1).

    Models run per selection group (:func:`plan_selections`), each under its
    narrowed config and its own data, so the stage identity (``data.upstream``)
    carries the selection the model actually trained on.
    """
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    paradigm_names = list(paradigms) if paradigms else [str(p) for p in cfg.paradigm_names]
    seed_list = [int(s) for s in (seeds if seeds else cfg.seeds.eval_seeds)]
    tuning_seed = int(cfg.seeds.tuning_seed)

    outcomes: list[StageOutcome] = []
    for group_cfg, group_data, model_names in plan_selections(
        cfg, store, data=data, models=models
    ):
        for model_name in model_names:
            spec = get_spec(model_name, group_cfg.name)
            for paradigm in paradigm_names:
                for stage_name in stages:
                    if stage_name == "cv":
                        stage_seeds = [tuning_seed]
                    elif not spec.stochastic:
                        stage_seeds = seed_list[:1]
                    else:
                        stage_seeds = seed_list
                    for seed in stage_seeds:
                        outcomes.append(
                            run_stage(
                                group_cfg,
                                model_name,
                                paradigm,
                                seed,
                                stage_name,
                                group_data,
                                store=store,
                                force=force,
                                progress_every=progress_every,
                                allow_default_params=allow_default_params,
                                max_folds=max_folds,
                            )
                        )
    return outcomes
