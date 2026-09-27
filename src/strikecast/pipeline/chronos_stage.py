"""The Chronos-2 family through the pipeline (audit C15, plan §8 P5).

``_chronos2.py`` does not speak darts, so it cannot go through
:func:`strikecast.pipeline.data_stage.prepare_data`'s bundle/feature-selection
path or :func:`strikecast.pipeline.run_stage.make_forecaster`. This module is
the Chronos counterpart of both, and :mod:`~strikecast.pipeline.run_stage`,
:mod:`~strikecast.pipeline.tune_stage` and
:mod:`~strikecast.pipeline.data_stage` dispatch to it on ``spec.kind ==
"chronos"`` / ``data.panel_variant == "chronos"``, the same way they dispatch
the hurdle/damage composites to :mod:`~strikecast.pipeline.composite_stage`.

What runs where
---------------
``prepare_chronos_data``
    ``_chronos2.py`` §1-§3: :func:`~strikecast.data.panel.build_panel_legacy_chronos`
    then :func:`~strikecast.data.autogluon.panel_to_long_frames`. NO feature
    selection and no window features (F28, F123: 27 known + 84 past covariates,
    the 84th being the stray ``index`` column). The expdecay switch
    (``series.window.expdecay``) does not apply: the Chronos frame has no
    ``ewm_*`` columns.
``tune_chronos_model``
    ``_chronos2.py`` §8, via :func:`strikecast.tuning.autogluon_runner.tune_chronos`.
    Writes ``<store>/chronos2/tuning/chronos2_fine_tuned/{optuna.sqlite3,
    best_params.json, trials.csv}`` like every other study.
``run_chronos_stage``
    ``_chronos2.py`` §7/§9/§10 for ONE ``(model, seed)``: fit the predictor
    once (or reload it), run the 164-fold rolling backtest on the shared
    engine (``retrain_stride=None``), persist the parts, and write the
    ``chronos_evaluate_long`` views. The fitted AutoGluon predictor is kept at
    ``<run>/artifacts/predictor/`` (with ``artifacts/predictor.json`` recording
    the fit identity) because the importance stage needs it (``_chronos2.py:686``).

The family has no ``cv`` stage (F131): a ``cv`` request is skipped with a log
line instead of raising, so ``strikecast run experiment=chronos2`` with the
CLI's default ``stage=cv,test`` does the right thing.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strikecast.data.cache import content_hash
from strikecast.pipeline.context import (
    get_spec,
    make_run_context,
    make_tracker,
    resolve_store_root,
    track,
    tracker_tags,
)
from strikecast.pipeline.data_stage import DataArtifacts, FeatureSets
from strikecast.seeds import record_env, seed_everything
from strikecast.store import RunKey, RunStore, stage_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strikecast.config.schema import ExperimentConfig
    from strikecast.data.autogluon import LongFrames
    from strikecast.data.load import Inputs
    from strikecast.pipeline.run_stage import StageOutcome
    from strikecast.pipeline.tune_stage import TuneOutcome

__all__ = [
    "PREDICTOR_DIRNAME",
    "ChronosData",
    "is_chronos_family",
    "load_stage_predictor",
    "predictor_dir",
    "prepare_chronos_data",
    "run_chronos_stage",
    "tune_chronos_model",
]

logger = logging.getLogger(__name__)

#: ``<run>/artifacts/<PREDICTOR_DIRNAME>/`` holds the fitted TimeSeriesPredictor.
PREDICTOR_DIRNAME = "predictor"
#: ``<run>/artifacts/<PREDICTOR_META>``: the identity the predictor was fit under.
PREDICTOR_META = "predictor.json"
#: The only stage the family has (F131).
CHRONOS_STAGES = ("test",)


def is_chronos_family(cfg: ExperimentConfig) -> bool:
    """``data.panel_variant == "chronos"``: the AutoGluon frame, not a darts bundle."""
    return getattr(cfg.data, "panel_variant", None) == "chronos"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChronosData(DataArtifacts):
    """The Chronos family's data stage output.

    A :class:`DataArtifacts` so that everything that only forwards the object
    (``run_experiment``, ``tune_experiment``, the CLI) keeps working; ``bundle``
    is ``None`` because there is no darts bundle. ``features`` records the
    covariates AutoGluon sees (``source="all"``: no selection, F28).
    """

    frames: LongFrames | None = field(default=None, repr=False)
    target: str = ""
    future_covariates: tuple[str, ...] = ()
    past_covariates: tuple[str, ...] = ()
    train_frac: float = 0.70
    test_frac: float = 0.20
    seasonality: int = 7

    @property
    def region_names(self) -> list[str]:  # type: ignore[override]
        if self.frames is None:
            return []
        return list(self.frames.item_ids)

    @cached_property
    def tsdf(self) -> Any:
        """The ``TimeSeriesDataFrame`` of ``_chronos2.py:220``. Imports AutoGluon."""
        from strikecast.data.autogluon import to_timeseries_dataframe  # noqa: PLC0415

        return to_timeseries_dataframe(self._frames())

    @property
    def n_steps(self) -> int:
        panel = self._frames().panel()
        return int(panel.num_timesteps_per_item().min())

    @cached_property
    def train_data(self) -> Any:
        """``train_data`` of ``_chronos2.py:235``: the first ``n - round(0.2 n)`` steps.

        ``int(round(TEST_FRAC * n))`` = 169 on the real panel, leaving 678
        training steps (F34/F126).
        """
        from strikecast.data.autogluon import test_split_length  # noqa: PLC0415

        test_size = test_split_length(self.n_steps, self.test_frac)
        train, _ = self.tsdf.train_test_split(prediction_length=test_size)
        return train

    def scales(self) -> tuple[dict[str, float], dict[str, float]]:
        """``MAE_SCALES`` / ``RMSE_SCALES`` (``_chronos2.py:293-300``): train only."""
        from strikecast.data.autogluon import scales_for_stage  # noqa: PLC0415

        mae, rmse = scales_for_stage(
            self._frames(), self.target, train_frac=self.train_frac, seasonality=self.seasonality
        )
        return dict(mae), dict(rmse)

    def _frames(self) -> LongFrames:
        if self.frames is None:
            raise ValueError("this ChronosData carries no frames")
        return self.frames


def _panel_hash(cfg: ExperimentConfig) -> str:
    """The data stage's panel identity, so a data-file change moves this one too."""
    from strikecast.pipeline import data_stage  # noqa: PLC0415

    try:
        return data_stage._panel_hash(cfg)
    except Exception:  # pragma: no cover - defensive: the private helper changed shape
        return content_hash("panel", cfg.data.model_dump(mode="json"))


def prepare_chronos_data(
    cfg: ExperimentConfig,
    store: RunStore | None = None,
    *,
    inputs: Inputs | None = None,
    **_: Any,
) -> ChronosData:
    """``_chronos2.py`` §1-§3: the panel, the AutoGluon frames, the covariate split.

    Nothing is cached in the store: the panel takes seconds, and the frame is
    rebuilt from it bit-for-bit (``tests/golden/test_chronos_equality.py``
    pins its shape against the stored sidecar). ``store`` is accepted for
    signature parity with :func:`~strikecast.pipeline.data_stage.prepare_data`.
    """
    from strikecast.data.autogluon import panel_to_long_frames  # noqa: PLC0415
    from strikecast.data.covariates import split_covariates  # noqa: PLC0415
    from strikecast.data.panel import build_panel_legacy_chronos  # noqa: PLC0415

    if inputs is None:
        from strikecast.data.load import load_inputs  # noqa: PLC0415

        inputs = load_inputs(cfg.data.fixed_dir, cfg.data.dataset_dir)

    target = cfg.data.target
    result = build_panel_legacy_chronos(
        inputs, target, low_prevalence_ratio=cfg.data.low_prevalence_ratio
    )
    frames = panel_to_long_frames(result.panel)
    split = split_covariates(result.panel, list(result.global_weather_columns), target)
    future = [c for c in split.future_covariates]
    # AutoGluon's view: every frame column that is neither the target nor a
    # known covariate is a past covariate -- including `index` (F123).
    past = [c for c in frames.values.columns if c != target and c not in set(future)]

    panel_hash = _panel_hash(cfg)
    series_hash = content_hash(
        "chronos_frame",
        panel_hash,
        cfg.series.split.model_dump(mode="json"),
        target,
        list(frames.values.columns),
    )
    features = FeatureSets(
        past_keep=list(past),
        future_keep=list(future),
        source="all",
        hash=content_hash("chronos_features", list(past), list(future)),
    )
    logger.info(
        "chronos data: %d items, %d known + %d past covariates",
        len(frames.item_ids),
        len(future),
        len(past),
    )
    return ChronosData(
        bundle=None,  # type: ignore[arg-type]
        features=features,
        panel_hash=panel_hash,
        series_hash=series_hash,
        activity_by_region=dict(inputs.activity_by_region),
        panel=None,
        frames=frames,
        target=target,
        future_covariates=tuple(future),
        past_covariates=tuple(past),
        train_frac=float(cfg.split.train),
        test_frac=float(cfg.split.test),
    )


def _require_chronos_data(data: Any) -> ChronosData:
    if not isinstance(data, ChronosData):
        raise TypeError(
            "the Chronos-2 family needs the AutoGluon frames: prepare the data with "
            "strikecast.pipeline.chronos_stage.prepare_chronos_data (data_stage."
            f"prepare_data does this for data.panel_variant='chronos'), got {type(data)!r}"
        )
    return data


# --------------------------------------------------------------------------- #
# predictor location
# --------------------------------------------------------------------------- #
def predictor_dir(store: RunStore, key: RunKey) -> Path:
    """``<run>/artifacts/predictor/``: where a stage's fitted predictor lives."""
    return store.artifacts_dir(key) / PREDICTOR_DIRNAME


def load_stage_predictor(store: RunStore, key: RunKey) -> Any:
    """Load the predictor a completed Chronos test stage fitted (for importance)."""
    from autogluon.timeseries import TimeSeriesPredictor  # noqa: PLC0415

    path = predictor_dir(store, key)
    if not (path / "predictor.pkl").exists():
        raise FileNotFoundError(
            f"no fitted Chronos-2 predictor at {path}; run the test stage first "
            f"(strikecast run experiment={key.experiment} model={key.model} stage=test "
            f"seed={key.seed})"
        )
    return TimeSeriesPredictor.load(str(path))


def _fit_identity(builder: Any, seed: int, data: ChronosData) -> str:
    return content_hash(
        "chronos_fit",
        builder.hyperparameters,
        int(seed),
        builder.num_val_windows,
        builder.eval_metric,
        builder.prediction_length,
        list(data.upstream),
    )


def _obtain_predictor(
    store: RunStore,
    key: RunKey,
    builder: Any,
    data: ChronosData,
    seed: int,
    *,
    force: bool,
) -> tuple[Any, bool]:
    """Reload the stage's predictor when it was fit under the same identity, else fit.

    ``_chronos2.py`` reuses ``checkpoints/<name>`` whenever the DIRECTORY exists
    (§7, §9). The run store keys it instead: ``artifacts/predictor.json``
    records the fit identity, and a predictor fit under different
    hyper-parameters, seed or data is discarded and re-fit.
    Returns ``(predictor, reused)``.
    """
    from strikecast.models.chronos import load_or_fit_predictor  # noqa: PLC0415

    path = predictor_dir(store, key)
    meta_path = store.artifacts_dir(key) / PREDICTOR_META
    identity = _fit_identity(builder, seed, data)
    reuse = False
    if not force and path.exists() and meta_path.exists():
        import json  # noqa: PLC0415

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            reuse = meta.get("fit_identity") == identity
        except (OSError, ValueError):
            reuse = False
    if not reuse:
        # Meta first: a fit killed midway must never look like a finished one.
        if meta_path.exists():
            meta_path.unlink()
        if path.exists():
            shutil.rmtree(path)

    predictor = load_or_fit_predictor(
        path,
        builder=builder,
        train_data=None if reuse else data.train_data,
        target=data.target,
        known_covariates_names=list(data.future_covariates),
        force=False,
    )
    if not reuse:
        store.write_artifact(
            key,
            PREDICTOR_META,
            {
                "fit_identity": identity,
                "hyperparameters": builder.hyperparameters,
                "random_seed": int(seed),
                "num_val_windows": builder.num_val_windows,
                "prediction_length": builder.prediction_length,
                "eval_metric": builder.eval_metric,
                "target": data.target,
                "known_covariates": list(data.future_covariates),
                "n_past_covariates": len(data.past_covariates),
                "train_steps": int(data.n_steps) - _test_size(data),
            },
        )
    return predictor, reuse


def _test_size(data: ChronosData) -> int:
    from strikecast.data.autogluon import test_split_length  # noqa: PLC0415

    return test_split_length(data.n_steps, data.test_frac)


def _internal_leaderboard(predictor: Any) -> list[dict[str, Any]]:
    try:
        board = predictor.leaderboard()
    except Exception as exc:  # pragma: no cover - informational only
        logger.warning("predictor.leaderboard() failed: %s", exc)
        return []
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in board.to_dict(orient="records")
    ]


# --------------------------------------------------------------------------- #
# store compatibility (the store and run_stage grow keywords independently)
# --------------------------------------------------------------------------- #
def _start_stage(store: RunStore, key: RunKey, stage: str, digest: str, **extra: Any) -> None:
    import inspect  # noqa: PLC0415

    params = inspect.signature(store.start_stage).parameters
    store.start_stage(key, stage, digest, **{k: v for k, v in extra.items() if k in params})


def _interrupt_stage(store: RunStore, key: RunKey, stage: str, reason: str) -> None:
    mark = getattr(store, "interrupt_stage", None)
    if mark is not None:
        mark(key, stage, reason)
    else:  # pragma: no cover - older store
        store.fail_stage(key, stage, f"interrupted: {reason}")


def _require_tuned(
    cfg: ExperimentConfig,
    spec: Any,
    model_name: str,
    params_source: str,
    store: RunStore,
    key: RunKey,
    stage_name: str,
    *,
    allow_default_params: bool,
) -> None:
    """``run_stage.require_tuned_params`` (audit C17), recorded as a failed stage."""
    try:
        from strikecast.pipeline.run_stage import require_tuned_params  # noqa: PLC0415
    except ImportError:  # pragma: no cover - older run_stage
        return
    try:
        require_tuned_params(
            cfg, spec, model_name, params_source, store,
            allow_default_params=allow_default_params,
        )
    except Exception as exc:
        store.fail_stage(key, stage_name, f"{type(exc).__name__}: {exc}")
        raise


# --------------------------------------------------------------------------- #
# the run stage
# --------------------------------------------------------------------------- #
def run_chronos_stage(
    cfg: ExperimentConfig,
    model_name: str,
    paradigm: str,
    seed: int,
    stage_name: str,
    data: Any,
    *,
    store: RunStore | None = None,
    tracker: Any = None,
    force: bool = False,
    progress_every: int = 0,
    allow_default_params: bool = False,
    max_folds: int | None = None,
) -> StageOutcome:
    """Run (or skip) one Chronos-2 ``(model, seed)`` test stage.

    ``allow_default_params`` and ``max_folds`` mean what they mean for
    :func:`strikecast.pipeline.run_stage.run_stage`: without the first, a
    ``chronos2_fine_tuned`` run with no ``tuning/.../best_params.json`` is an
    error (audit C17; D4 re-tunes it); the second keeps only the first N of the
    164 folds (benchmark / verification) and is part of the stage identity.
    """
    from strikecast.backtest.engine import ExpandingWindowBacktest  # noqa: PLC0415
    from strikecast.backtest.hooks import ProgressHook  # noqa: PLC0415
    from strikecast.evaluation.metrics import chronos_evaluate_long  # noqa: PLC0415
    from strikecast.models.chronos import (  # noqa: PLC0415
        Chronos2Builder,
        Chronos2Forecaster,
        level_targets_from_frames,
    )
    from strikecast.pipeline.run_stage import StageOutcome, resolve_params  # noqa: PLC0415
    from strikecast.transforms.identity import Identity  # noqa: PLC0415

    store = store if store is not None else RunStore(resolve_store_root(cfg))
    key = RunKey(cfg.name, model_name, str(paradigm), int(seed))
    if stage_name not in cfg.stages:
        logger.warning(
            "skip %s/%s: the Chronos-2 family has no %r stage (F131; stages: %s)",
            key.relative(),
            stage_name,
            stage_name,
            sorted(cfg.stages),
        )
        return StageOutcome(key, stage_name, "", skipped=True, params_source="n/a")
    if str(paradigm) != "global":
        raise ValueError(
            f"Chronos-2 is one predictor over all 20 items: paradigm must be 'global' "
            f"(F133), got {paradigm!r}"
        )

    data = _require_chronos_data(data)
    spec = get_spec(model_name, cfg.name)
    stage_cfg = cfg.stage(stage_name)
    backtest_cfg = stage_cfg.backtest(cfg.split)
    params, params_source = resolve_params(cfg, spec, model_name, store)
    _require_tuned(cfg, spec, model_name, params_source, store, key, stage_name,
                   allow_default_params=allow_default_params)

    resolved_stage = {
        "experiment": cfg.name,
        "model": model_name,
        "paradigm": str(paradigm),
        "stage": stage_name,
        "backtest": backtest_cfg.model_dump(mode="json"),
        "naive_scales": stage_cfg.naive_scales.model_dump(mode="json"),
        "params": params,
        "family": "chronos",
    }
    if max_folds is not None:
        resolved_stage["max_folds"] = int(max_folds)
    digest = stage_hash(resolved_stage, list(data.upstream), int(seed))
    if not force and store.is_complete(key, stage_name, digest):
        logger.info("skip %s/%s: complete under the same identity", key.relative(), stage_name)
        return StageOutcome(key, stage_name, digest, skipped=True, params_source=params_source)

    owns_tracker = tracker is None
    tracker = tracker if tracker is not None else make_tracker(cfg)

    seed_everything(int(seed))
    ctx = make_run_context(cfg, model_name, int(seed))

    _start_stage(store, key, stage_name, digest, params_source=params_source, max_folds=max_folds)
    store.write_config(key, cfg.model_dump(mode="json"))
    env = record_env()
    tracked_id = None
    try:
        tracked_id = tracker.start(
            key, cfg.model_dump(mode="json"), tags=tracker_tags(cfg, spec), stage=stage_name
        )
    except Exception as exc:  # pragma: no cover - the mirror never fails a run
        logger.warning("tracker.start failed: %s", exc)
    if tracked_id:
        env = {**env, "tracker_run_id": tracked_id}
    store.write_env(key, env)
    if tracked_id:
        store.record_tracker_run_id(key, tracked_id, stage=stage_name)

    try:
        builder = spec.build(params, ctx)
        if not isinstance(builder, Chronos2Builder):
            raise TypeError(f"{model_name} did not build a Chronos2Builder: {type(builder)!r}")
        predictor, reused = _obtain_predictor(store, key, builder, data, int(seed), force=force)
        store.write_artifact(key, "internal_leaderboard.json", _internal_leaderboard(predictor))

        regions = data.region_names
        tsdf = data.tsdf
        forecaster = Chronos2Forecaster(
            predictor,
            tsdf,
            target=data.target,
            known_covariates=list(data.future_covariates),
            region_names=regions,
            horizon=backtest_cfg.horizon,
        )
        level_targets = level_targets_from_frames(tsdf, data.target, regions)
        persist = store.persist_hook(
            key,
            stage_name,
            backtest_cfg.retrain_stride,
            actuals=level_targets,
            region_names=regions,
        )
        hooks: list[Any] = [persist]
        if progress_every:
            hooks.append(ProgressHook(progress_every))
        engine = ExpandingWindowBacktest(backtest_cfg, Identity(), hooks=hooks)
        n_folds = 0
        folds = engine.iter_folds(forecaster, level_targets)
        if max_folds is not None:
            from itertools import islice  # noqa: PLC0415

            folds = islice(folds, int(max_folds))
        for _ in folds:
            n_folds += 1
        persist.close()

        preds = store.load_predictions(key, stage_name, legacy_order=True)
        mae_scales, rmse_scales = data.scales()
        views = chronos_evaluate_long(preds.legacy_frame(), mae_scales, rmse_scales)
        metric_paths = store.write_metrics(key, stage_name, views)
        store.complete_stage(key, stage_name)

        track(tracker, "log_tables", views, stage=stage_name)
        track(tracker, "log_artifact", store.predictions_dir(key, stage_name), "predictions")
        track(tracker, "log_artifact", store.metrics_dir(key, stage_name), "metrics")
    except KeyboardInterrupt as exc:
        # SIGTERM/SIGUSR1 (strikecast.pipeline.interrupt) or Ctrl-C (audit C21).
        _interrupt_stage(store, key, stage_name, str(exc) or type(exc).__name__)
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
        "%s/%s: %d folds, %d rows (predictor %s)",
        key.relative(),
        stage_name,
        n_folds,
        len(preds),
        "reused" if reused else "fitted",
    )
    return StageOutcome(
        key,
        stage_name,
        digest,
        skipped=False,
        n_folds=n_folds,
        n_rows=len(preds),
        metric_paths=metric_paths,
        params_source=params_source,
    )


# --------------------------------------------------------------------------- #
# the tune stage
# --------------------------------------------------------------------------- #
def tune_chronos_model(
    cfg: ExperimentConfig,
    model_name: str,
    data: Any,
    *,
    store: RunStore | None = None,
    tracker: Any = None,
    n_trials: int | None = None,
    force: bool = False,
    fit: Any = None,
    score: Any = None,
) -> TuneOutcome:
    """``_chronos2.py`` §8 for one model: the 12-trial AutoGluon fine-tune study.

    ``fit``/``score`` are test seams forwarded to
    :func:`~strikecast.tuning.autogluon_runner.make_chronos_run_trial`.
    """
    from strikecast.pipeline.tune_stage import (  # noqa: PLC0415
        TuneOutcome,
        _TuningKey,
        trial_callback,
    )
    from strikecast.tuning import load_best_params  # noqa: PLC0415
    from strikecast.tuning.autogluon_runner import (  # noqa: PLC0415
        settings_from_config,
        tune_chronos,
    )

    store = store if store is not None else RunStore(resolve_store_root(cfg))
    spec = get_spec(model_name, cfg.name)
    directory = store.tuning_dir(cfg.name, model_name)
    best_path = directory / "best_params.json"

    if not spec.tunable:
        logger.info("%s has no search space; nothing to tune", model_name)
        return TuneOutcome(model_name, dict(spec.defaults), None, 0, True, "not tunable")
    if best_path.is_file() and not force:
        logger.info("%s: %s exists; not re-tuning", model_name, best_path)
        return TuneOutcome(
            model_name, load_best_params(directory), None, 0, True,
            "best_params.json exists", best_path,
        )

    data = _require_chronos_data(data)
    settings = settings_from_config(cfg)
    seed_everything(int(settings.tuning_seed))

    owns_tracker = tracker is None
    tracker = tracker if tracker is not None else make_tracker(cfg)
    try:
        track(
            tracker,
            "start",
            _TuningKey(cfg.name, model_name, int(settings.tuning_seed)),
            cfg.model_dump(mode="json"),
            tags=tracker_tags(cfg, spec),
            stage="tune",
        )
        result = tune_chronos(
            spec,
            data.train_data,
            target=data.target,
            known_covariates=list(data.future_covariates),
            settings=settings,
            directory=directory,
            n_trials=n_trials,
            callbacks=[trial_callback(tracker)],
            fit=fit,
            score=score,
        )
    finally:
        if owns_tracker:
            track(tracker, "finish")

    logger.info(
        "%s: best %s=%s after %d trials", model_name, settings.metric, result.best_value,
        result.n_trials,
    )
    return TuneOutcome(
        model_name,
        dict(result.best_params),
        result.best_value,
        result.n_trials,
        False,
        "",
        result.best_params_path,
    )
