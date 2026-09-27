"""Stage 2: one Optuna study per model variant.

The legacy objective, verbatim (``_regression_GBDT.py::make_gbm_objective``,
``_diff_regression.py::make_gbm_objective``)::

    params  = suggester(trial, objective_kind)
    builder = lambda: build_gbm_from_params(variant, params)
    for step, cumulative in enumerate(run_expanding_cv_iter(builder, target_for_cv,
                                      CV_START_VAL, ...)):
        last = _score_fold_preds(cumulative, target_for_cv, region_names, "RMSSE_mean")
        trial.report(last, step=step); if trial.should_prune(): raise TrialPruned
    return float(last)

is reassembled here from parts that already exist: ``spec.search_space`` is the
suggester, :class:`~strikecast.backtest.engine.ExpandingWindowBacktest.iter_folds`
is ``run_expanding_cv_iter``, and
:class:`~strikecast.backtest.hooks.PruningHook` is the report/prune/return
block. ``strikecast.tuning.optuna_runner.tune`` owns the study itself (SQLite,
``load_if_exists``, the TPE seed, the median pruner) and writes
``best_params.json`` and ``trials.csv``.

Fixed by §1 and §5.4, not by this module:

* tuning runs **once**, under ``seeds.tuning_seed`` and the **Global**
  paradigm; Activity and Local reuse the Global-tuned configuration (F14);
* the objective metric has no default (F39); the experiment YAML writes it out;
* the adapter preset is ``for_tuning``: the 200-sample median WITHOUT the
  ``exp`` log-link (F56), which is what the legacy iterator applied -- for the
  GBDTs. The RNN objectives were different (audit 2026-09-26 B7, see
  :func:`trial_protocol`): ``run_expanding_cv`` (``for_cv``, the ``exp``
  log-link applied), scored once on all folds, no per-fold report; their only
  pruning is the PyTorch-Lightning callback on ``train_loss`` that the RNN
  search space adds to the trainer;
* ``n_trials`` is the TOTAL the study should end with, so an interrupted study
  is continued rather than restarted (F11 records what resuming does not
  restore: the TPE RNG state).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strikecast.backtest.engine import ExpandingWindowBacktest
from strikecast.backtest.hooks import PruningHook
from strikecast.backtest.predictions import PredictionSet
from strikecast.pipeline import data_stage as _data_stage
from strikecast.pipeline.context import (
    get_spec,
    make_run_context,
    make_tracker,
    resolve_store_root,
    track,
    tracker_tags,
)
from strikecast.pipeline.run_stage import (
    _covariates,
    _metric_set,
    _naive_scales,
    make_forecaster,
    model_targets_for,
    stage_targets,
)
from strikecast.seeds import seed_everything
from strikecast.store import RunStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

    from strikecast.config.schema import ExperimentConfig
    from strikecast.pipeline.data_stage import DataArtifacts

__all__ = ["TuneOutcome", "trial_callback", "trial_protocol", "tune_experiment", "tune_model"]

logger = logging.getLogger(__name__)

#: The stage the Optuna objective scores. Every legacy study cross-validates.
TUNING_STAGE = "cv"

#: The paradigm every legacy study tuned under (§1 "Seeds", F14).
TUNING_PARADIGM = "global"


@dataclass(frozen=True)
class TuneOutcome:
    """What one study produced, or why it was skipped."""

    model: str
    best_params: dict[str, Any]
    best_value: float | None
    n_trials: int
    skipped: bool
    reason: str = ""
    path: Path | None = None


class _Settings:
    """``TuningLike`` view of a pydantic ``TuningConfig`` plus the seed config.

    ``strikecast.tuning.optuna_runner.tune`` reads a flat object (see its
    ``TuningLike`` protocol); ``TuningConfig`` splits the same values across two
    models, so this adapts one to the other without either side importing the
    other.
    """

    def __init__(self, cfg: ExperimentConfig, kind: str = "gbdt") -> None:
        tuning = cfg.tuning
        if tuning is None:
            raise ValueError(f"experiment {cfg.name!r} has no tuning block")
        self.metric = tuning.objective
        self.tuning_seed = int(cfg.seeds.tuning_seed)
        self.direction = tuning.direction
        self.n_warmup_steps = (
            tuning.pruner.n_warmup_steps if tuning.pruner.kind == "median" else 0
        )
        # `TuningConfig.n_trials` is keyed by model kind; `tune()` reads a flat
        # field and lets `spec.n_trials` win over it, which is the legacy order.
        self.n_trials: int | None = tuning.n_trials.get(kind)
        self.timeout = None if tuning.timeout_s is None else float(tuning.timeout_s)
        self.catch: tuple[type[BaseException], ...] = ()
        self.gc_after_trial = False
        self.show_progress_bar = False
        self.n_jobs = 1


def _score_fn(
    cfg: ExperimentConfig,
    data: DataArtifacts,
    stage_cfg: Any,
    level_targets: list[Any],
    metric: str,
):
    """``_score_fold_preds`` bound to this experiment's targets and scales.

    The score is computed on the CUMULATIVE predictions of folds 0..step, on
    the LEVEL-space targets, with the same naive scales the CV stage's metrics
    use -- which is what makes the pruner's view and the leaderboard's view the
    same number.
    """
    from strikecast.evaluation.aggregate import evaluate  # noqa: PLC0415

    mae_scales, rmse_scales = _naive_scales(data, stage_cfg)
    region_names = data.region_names
    metric_set = _metric_set(cfg)

    def _score(cumulative: dict[str, list[list[Any]]]) -> float:
        preds = PredictionSet.from_fold_preds(level_targets, cumulative, region_names)
        views = evaluate(
            preds,
            mae_scales,
            rmse_scales,
            metric_set=metric_set,  # type: ignore[arg-type]
            threshold=cfg.calibration.threshold,
        )
        return float(views["global"][metric])

    return _score


def trial_protocol(spec: Any) -> tuple[str, bool]:
    """``(adapter preset, per-fold pruning)`` of the legacy objective (B7).

    GBDT objectives (``make_gbm_objective`` in ``_regression_GBDT.py`` /
    ``_diff_regression.py``) iterate ``run_expanding_cv_iter`` -- the
    ``for_tuning`` preset, no ``exp`` (F56) -- and report the running score to
    Optuna after every fold (:class:`PruningHook`).

    RNN objectives (``make_nn_objective``, ``_regression_LSTM.py:1080-1106``,
    ``_diff_regression.py:1163-1173``) call ``run_expanding_cv`` -- the
    ``for_cv`` preset, so the Poisson/Tweedie log-link RNNs are scored on
    ``exp``-transformed (count-scale) predictions -- and return one score over
    all folds, never reporting per fold: the MedianPruner only sees the
    per-epoch ``train_loss`` of the PL pruning callback. Reporting fold scores
    as well would mix two quantities on one Optuna step axis.
    """
    if getattr(spec, "is_neural", False):
        return "for_cv", False
    return "for_tuning", True


def trial_callback(tracker: Any) -> Callable[[Any, Any], None]:
    """An Optuna callback that mirrors each finished trial (§5.5).

    ``study.optimize(callbacks=[...])`` calls this after **every** finished
    trial, so the tune run's W&B curve grows as the study runs and survives an
    interrupted study -- unlike replaying ``trials_dataframe()`` afterwards,
    which is what this stage did before ``tune()`` grew a ``callbacks``
    argument and which logged nothing at all when a study crashed at trial 49
    (the very failure mode F11 is about).

    Pruned and failed trials have no value; they are skipped rather than logged
    as NaN, because the tracker mirrors what the study recorded. ``params`` is
    the trial's own parameter dict, which is complete even for a pruned trial
    (:func:`~strikecast.tuning.optuna_runner.make_objective` suggests
    everything before the first fold runs).

    Every call goes through :func:`track`, which swallows tracker errors: a
    callback that raised would abort ``optimize`` and take the study with it.
    """

    def _callback(study: Any, trial: Any) -> None:
        value = getattr(trial, "value", None)
        if value is None or value != value:  # None, or NaN: a pruned/failed trial
            return
        track(
            tracker,
            "log_trial",
            int(trial.number),
            float(value),
            dict(getattr(trial, "params", {}) or {}),
        )

    return _callback


def tune_model(
    cfg: ExperimentConfig,
    model_name: str,
    data: DataArtifacts,
    *,
    store: RunStore | None = None,
    tracker: Any = None,
    n_trials: int | None = None,
    force: bool = False,
) -> TuneOutcome:
    """Run (or resume, or skip) the study of one model variant."""
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    spec = get_spec(model_name, cfg.name)
    if spec.kind == "chronos":
        # Chronos-2 (audit C15): the objective is AutoGluon's internal
        # validation MASE of a fresh fine-tune per trial, not the shared CV
        # backtest (F129); `tuning/autogluon_runner.py` owns that study.
        from strikecast.pipeline.chronos_stage import tune_chronos_model  # noqa: PLC0415

        return tune_chronos_model(
            cfg, model_name, data, store=store, tracker=tracker, n_trials=n_trials, force=force
        )
    directory = store.tuning_dir(cfg.name, model_name)
    best_path = directory / "best_params.json"

    if not spec.tunable:
        logger.info("%s has no search space; nothing to tune", model_name)
        return TuneOutcome(model_name, dict(spec.defaults), None, 0, True, "not tunable")

    if best_path.is_file() and not force:
        from strikecast.tuning import load_best_params  # noqa: PLC0415

        logger.info("%s: %s exists; not re-tuning (Appendix C)", model_name, best_path)
        return TuneOutcome(
            model_name, load_best_params(directory), None, 0, True, "best_params.json exists",
            best_path,
        )

    settings = _Settings(cfg, "rnn" if spec.is_neural else "gbdt")
    stage_cfg = cfg.stage(TUNING_STAGE)
    backtest_cfg = stage_cfg.backtest(cfg.split)
    level_targets = stage_targets(data, TUNING_STAGE)
    override = model_targets_for(cfg, data, TUNING_STAGE)
    past_covs, future_covs = _covariates(spec, data)
    score = _score_fn(cfg, data, stage_cfg, level_targets, settings.metric)

    owns_tracker = tracker is None
    tracker = tracker if tracker is not None else make_tracker(cfg)

    from strikecast.transforms.diff import Diff  # noqa: PLC0415
    from strikecast.transforms.identity import Identity  # noqa: PLC0415

    def _transform() -> Any:
        return Diff() if cfg.transform.kind == "diff" else Identity()

    def run_trial(params: dict[str, Any], trial: Any) -> float:
        # Every trial starts from the same RNG state, exactly as a fresh
        # process would: the legacy scripts seed once at import and build one
        # model per trial. The sampler's own state lives in the study.
        seed_everything(int(cfg.seeds.tuning_seed))
        ctx = make_run_context(cfg, model_name, int(cfg.seeds.tuning_seed))
        preset, per_fold_pruning = trial_protocol(spec)
        forecaster = make_forecaster(cfg, spec, model_name, dict(params), ctx, preset=preset)
        if not per_fold_pruning:
            # B7: the RNN objective -- all folds, one score, no trial.report.
            engine = ExpandingWindowBacktest(backtest_cfg, _transform(), hooks=[])
            cumulative = engine.run(
                forecaster, level_targets, past_covs, future_covs, model_targets=override
            )
            if not any(any(region) for region in cumulative.values()):
                raise RuntimeError(f"{model_name}: the schedule produced no folds to score")
            return float(score(cumulative))
        hook = PruningHook(score, trial, metric_name=settings.metric)
        engine = ExpandingWindowBacktest(backtest_cfg, _transform(), hooks=[hook])
        for _ in engine.iter_folds(
            forecaster, level_targets, past_covs, future_covs, model_targets=override
        ):
            pass
        if hook.last_value is None:
            raise RuntimeError(f"{model_name}: the schedule produced no folds to score")
        return float(hook.last_value)

    from strikecast.tuning import tune as _tune  # noqa: PLC0415

    try:
        track(
            tracker,
            "start",
            _TuningKey(cfg.name, model_name, int(cfg.seeds.tuning_seed)),
            cfg.model_dump(mode="json"),
            tags=tracker_tags(cfg, spec),
            stage="tune",
        )
        result = _tune(
            spec,
            run_trial,
            settings,
            directory / "optuna.sqlite3",
            study_name=model_name,
            n_trials=n_trials,
            out_dir=directory,
            callbacks=[trial_callback(tracker)],
        )
    finally:
        if owns_tracker:
            track(tracker, "finish")

    _record_feature_provenance(result.best_params_path, cfg, data)
    logger.info(
        "%s: best %s=%s after %d trials (%d pruned)",
        model_name,
        settings.metric,
        result.best_value,
        result.n_trials,
        result.n_pruned,
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


def _record_feature_provenance(path: Path, cfg: ExperimentConfig, data: DataArtifacts) -> None:
    """Add the features a study was tuned on to its ``best_params.json``.

    Publication runs re-tune on the ``leaky7`` features (decision D4) while the
    imported thesis params were tuned on ``expdecay7`` (audit A1/B1); both end
    up as ``<store>/<exp>/tuning/<model>/best_params.json``. The ``provenance``
    block (``expdecay`` mode, window config, upstream hashes) lets a reader --
    and a cv/test guard -- tell them apart. Extra key; ``load_best_params``
    ignores it.
    """
    import json  # noqa: PLC0415

    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"] = {
        "expdecay": cfg.series.window.expdecay,
        "window": cfg.series.window.model_dump(mode="json"),
        "panel_hash": data.panel_hash,
        "series_hash": data.series_hash,
        "features_hash": data.features.hash,
        "features_source": data.features.source,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


class _TuningKey:
    """A run key for a study, which has no paradigm and no eval seed.

    The tracker takes the four coordinates by attribute (Stream C duck-types
    them), and a study is global and runs under the tuning seed by definition
    (§1, F14), so those are the values it carries.
    """

    __slots__ = ("experiment", "model", "paradigm", "seed")

    def __init__(self, experiment: str, model: str, seed: int = 42) -> None:
        self.experiment = experiment
        self.model = model
        self.paradigm = TUNING_PARADIGM
        self.seed = seed


def tune_experiment(
    cfg: ExperimentConfig,
    *,
    data: DataArtifacts | None = None,
    models: Sequence[str] | None = None,
    store: RunStore | None = None,
    n_trials: int | None = None,
    force: bool = False,
) -> list[TuneOutcome]:
    """Tune every (tunable) model of an experiment, once."""
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    if data is None:
        data = _data_stage.prepare_data(cfg, store)
    names = list(models) if models else cfg.model_names
    return [
        tune_model(cfg, name, data, store=store, n_trials=n_trials, force=force)
        for name in names
    ]
