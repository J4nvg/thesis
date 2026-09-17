"""Expanding-window backtest: schedule, engine, hooks, grouping, predictions.

Phase 2 of ``docs/REFACTOR_PLAN.md``. The contracts every module here codes
against live in :mod:`strikecast.backtest.protocols`.
"""

from .engine import ExpandingWindowBacktest
from .grouping import Group, Paradigm, RunOne, partition, restore_order, run_grouped, take
from .hooks import CallbackHook, ProgressHook, PruningHook
from .naive import (
    NAIVE_METHODS,
    naive_fold_preds,
    naive_last_fold_preds,
    naive_prediction_set,
    naive_weekly_fold_preds,
)
from .predictions import COLUMNS, LEGACY_COLUMNS, PredictionSet
from .protocols import (
    SINGLE_CHANNEL,
    Fold,
    FoldHook,
    FoldResult,
    Forecaster,
    TargetTransform,
)
from .schedule import fold_schedule, schedule_from_config

__all__ = [
    "COLUMNS",
    "LEGACY_COLUMNS",
    "NAIVE_METHODS",
    "SINGLE_CHANNEL",
    "CallbackHook",
    "ExpandingWindowBacktest",
    "Fold",
    "FoldHook",
    "FoldResult",
    "Forecaster",
    "Group",
    "Paradigm",
    "PredictionSet",
    "ProgressHook",
    "PruningHook",
    "RunOne",
    "TargetTransform",
    "fold_schedule",
    "naive_fold_preds",
    "naive_last_fold_preds",
    "naive_prediction_set",
    "naive_weekly_fold_preds",
    "partition",
    "restore_order",
    "run_grouped",
    "schedule_from_config",
    "take",
]
