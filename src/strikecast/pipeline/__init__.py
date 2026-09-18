"""The orchestration layer: config in, run store out (plan §5.3, §8 P3).

Four stages, each independently callable and each resumable through the run
store:

``data_stage``
    config -> panel -> :class:`~strikecast.data.series.SeriesBundle` ->
    feature selection, cached under ``runs/<experiment>/shared/``.
``tune_stage``
    one Optuna study per model variant, on SQLite, scored by the CV fold loop
    with the pruning hook; writes ``best_params.json``.
``run_stage``
    one ``(model, paradigm, seed, stage)`` backtest: seed, build, run, persist,
    evaluate, mark complete.
``report_stage``
    the cross-model and cross-seed tables under ``report/``.

Nothing in this package imports Hydra; the CLI (:mod:`strikecast.cli`) composes
a :class:`~strikecast.config.schema.ExperimentConfig` and hands it in.
"""

from .context import NoopTracker, RunContext, make_run_context, make_tracker, resolve_store_root
from .data_stage import DataArtifacts, FeatureSets, prepare_data
from .report_stage import report

# NOTE: the `run_stage` FUNCTION is deliberately not re-exported here -- that
# name must keep pointing at the `run_stage` MODULE, so that
# `from strikecast.pipeline import run_stage` gives the module (which is what
# the CLI and the tests import). Use `from strikecast.pipeline.run_stage import
# run_stage` for the function.
from .run_stage import StageOutcome, run_experiment
from .tune_stage import TuneOutcome, tune_experiment, tune_model

__all__ = [
    "DataArtifacts",
    "FeatureSets",
    "NoopTracker",
    "RunContext",
    "StageOutcome",
    "TuneOutcome",
    "make_run_context",
    "make_tracker",
    "prepare_data",
    "report",
    "resolve_store_root",
    "run_experiment",
    "tune_experiment",
    "tune_model",
]
