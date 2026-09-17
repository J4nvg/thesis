"""Fold hooks: progress logging, arbitrary callbacks, Optuna pruning.

A hook is anything with ``on_fold(result, cumulative)`` (see
:class:`~strikecast.backtest.protocols.FoldHook`). The engine calls every hook
after it has folded the new predictions into ``cumulative``, so a hook always
sees the fold it was called for. Raising from a hook aborts the run -- that is
how :class:`PruningHook` stops a trial.

``cumulative`` is ``channel -> list[list[TimeSeries]]``, outer by region, inner
by fold so far. For a single-output forecaster
``cumulative[SINGLE_CHANNEL]`` is exactly the ``all_fold_preds`` bundle the
legacy ``run_expanding_cv_iter`` yields, so a legacy scoring function
(``_score_fold_preds`` -> ``collect_predictions_long`` -> ``evaluate_long``)
can be wrapped unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from darts import TimeSeries

    from .protocols import FoldResult

__all__ = ["CallbackHook", "ProgressHook", "PruningHook"]

logger = logging.getLogger(__name__)


class ProgressHook:
    """Log one line every fold, or every ``every`` folds.

    ``every=1`` logs every fold; ``every=7`` logs one line per retrain window
    when ``predict_stride=1`` and ``retrain_stride=7``. Folds are counted by
    ``Fold.index``, so fold 0 is always logged.
    """

    def __init__(
        self,
        every: int = 1,
        *,
        level: int = logging.INFO,
        log: logging.Logger | None = None,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.every = every
        self.level = level
        self.logger = log if log is not None else logger

    def on_fold(
        self,
        result: FoldResult,
        cumulative: dict[str, list[list[TimeSeries]]],
    ) -> None:
        fold = result.fold
        if fold.index % self.every:
            return
        n_regions = len(next(iter(cumulative.values()))) if cumulative else 0
        self.logger.log(
            self.level,
            "   fold %d  t0=%d  cutoff=%s  retrain=%s  (%d channels x %d regions)",
            fold.index,
            fold.t0,
            fold.cutoff,
            fold.retrain,
            len(result.predictions),
            n_regions,
        )


class CallbackHook:
    """Adapt a plain function to the :class:`FoldHook` protocol.

    ``fn(result, cumulative)`` is called after every fold; its return value is
    ignored. Anything it raises propagates and aborts the run.
    """

    def __init__(
        self,
        fn: Callable[[FoldResult, dict[str, list[list[TimeSeries]]]], Any],
    ) -> None:
        self.fn = fn

    def on_fold(
        self,
        result: FoldResult,
        cumulative: dict[str, list[list[TimeSeries]]],
    ) -> None:
        self.fn(result, cumulative)


class PruningHook:
    """Report a running score to Optuna after every fold and prune on request.

    Reproduces the cadence of the legacy objectives verbatim
    (``_regression_GBDT.py::make_gbm_objective`` and
    ``_diff_regression.py::make_gbm_objective``)::

        for step, cumulative_fold_preds in enumerate(run_expanding_cv_iter(...)):
            last_score = _score_fold_preds(cumulative_fold_preds, ..., metric="RMSSE_mean")
            trial.report(last_score, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(last_score)

    That is: every fold, no warm-up, ``step`` is the 0-based fold counter
    (``Fold.index``), the score is computed on the CUMULATIVE predictions of
    folds 0..step (so it is a running, not a per-fold, score), the check comes
    immediately after the report, and the trial's final value is the score of
    the last fold. :attr:`last_value` holds that score so the objective can
    ``return hook.last_value`` after the loop.

    ``score_fn`` receives ``cumulative`` and returns a float -- the legacy
    ``_score_fold_preds`` bound to its target list and region names.
    ``metric_name`` is the label that score carries (``"RMSSE_mean"`` in both
    legacy objectives); it is recorded and logged, never used to select the
    metric, because ``score_fn`` already did that.

    ``optuna`` is imported inside :meth:`on_fold`, so this module imports
    without optuna installed.
    """

    def __init__(
        self,
        score_fn: Callable[[dict[str, list[list[TimeSeries]]]], float],
        trial: Any,
        metric_name: str = "RMSSE_mean",
    ) -> None:
        self.score_fn = score_fn
        self.trial = trial
        self.metric_name = metric_name
        self.last_value: float | None = None

    def on_fold(
        self,
        result: FoldResult,
        cumulative: dict[str, list[list[TimeSeries]]],
    ) -> None:
        import optuna  # noqa: PLC0415  -- lazy: the engine must not need optuna

        value = float(self.score_fn(cumulative))
        self.last_value = value
        self.trial.report(value, step=result.fold.index)
        if self.trial.should_prune():
            logger.info(
                "pruned at fold %d (%s=%.6f)", result.fold.index, self.metric_name, value
            )
            raise optuna.TrialPruned()
