"""The Optuna objective's scalar: one global metric from one fold-pred bundle.

Port of ``_score_fold_preds``, which exists twice, byte-identical apart from
its default metric:

* ``_regression_GBDT.py`` lines 985-990 -- ``metric="MASE_mean"``
* ``_diff_regression.py`` lines 1120-1124 -- ``metric="RMSSE_mean"``
* (``_regression_LSTM.py`` line 1021 carries the GBDT copy.)

Plan F39: both Optuna objectives pass ``RMSSE_mean`` explicitly, so neither
default ever fired and the disagreement has no effect on any recorded run. The
port keeps ``RMSSE_mean`` as the default -- the value every recorded run
actually used -- and the plan's treatment stands: the tuning objective becomes
an explicit config field with no default at the config layer, so this default
is a convenience for direct callers only.

Behaviour-preservation notes
----------------------------

``Q1`` **The scales are the caller's, not the stage's.** The legacy function
    read the module globals ``MAE_SCALES`` / ``RMSE_SCALES``, which were
    computed once from the TRAIN split and reused for tuning, validation CV and
    test alike. They are parameters here; passing the train-only scales
    reproduces every tuned run.

``Q2`` **No activity map, so no activity views.** The legacy call is
    ``evaluate_long(long_df, MAE_SCALES, RMSE_SCALES)`` with three arguments,
    so the objective never paid for the two activity views. Preserved.

``Q3`` **The score is a mean over regions** (``MASE_mean`` / ``RMSSE_mean`` come
    from ``per_region``), so tuning optimises unweighted per-region scaled
    error, not pooled error. A region whose naive scale is degenerate drops out
    of the objective entirely (NaN is skipped by ``mean``), and if EVERY region
    is degenerate the objective is NaN and Optuna will reject the trial.

``Q4`` **The frame is rebuilt from the fold predictions on every call**, once
    per pruning report in the iterative objective. Kept: the cost is in the
    metric functions, and skipping it would change which trials the
    ``MedianPruner`` sees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from strikecast.backtest.predictions import PredictionSet

from .aggregate import evaluate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from darts import TimeSeries

__all__ = ["score_fold_preds"]


def score_fold_preds(
    fold_preds: Sequence[Sequence[TimeSeries]] | Mapping[str, Sequence[Sequence[TimeSeries]]],
    target_list: Sequence[TimeSeries],
    region_names: Sequence[str],
    mae_scales: Mapping[str, float],
    rmse_scales: Mapping[str, float],
    metric: str = "RMSSE_mean",
) -> float:
    """Run the metric pipeline on a fold-pred bundle and return one global metric.

    ``fold_preds`` is the legacy ``[[pred per fold] per region]`` list (or the
    ``{channel: ...}`` mapping a composite forecaster produces, which must then
    hold exactly one channel). ``target_list`` is the level-space actual series
    the predictions are scored against -- ``target_for_cv`` for the validation
    stage, ``target_full`` for test -- and it is the same list the legacy
    ``collect_predictions_long`` call received.

    Returns ``float(evaluate(...)["global"][metric])``. ``metric`` is looked up
    in the global row, so any of ``MAE``, ``RMSE``, ``MedAE``, ``ME``,
    ``PoissonDev``, ``TweedieDev``, ``ZeroAcc``, ``n``, ``MASE_mean``,
    ``MASE_median`` or ``RMSSE_mean`` works; an unknown name raises
    ``KeyError``, as it did in the legacy code.
    """
    pred_set = PredictionSet.from_fold_preds(target_list, fold_preds, region_names)
    result = evaluate(pred_set, mae_scales, rmse_scales)
    return float(result["global"][metric])
