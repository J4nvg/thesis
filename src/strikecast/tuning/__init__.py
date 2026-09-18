"""Optuna tuning (plan §5.3, §8 P3).

One study per model variant, stored in SQLite under
``runs/<experiment>/tuning/<model_variant>/optuna.sqlite3`` so an interrupted
study resumes instead of starting over. The legacy scripts pickled
``(best_params, study)`` only after all trials had finished.
"""

from .optuna_runner import (
    BEST_PARAMS_KEYS,
    RunTrial,
    TuningResult,
    TuningSettings,
    find_tuning_dir,
    load_best_params,
    load_tuning_meta,
    make_objective,
    tune,
)

__all__ = [
    "BEST_PARAMS_KEYS",
    "RunTrial",
    "TuningResult",
    "TuningSettings",
    "find_tuning_dir",
    "load_best_params",
    "load_tuning_meta",
    "make_objective",
    "tune",
]
