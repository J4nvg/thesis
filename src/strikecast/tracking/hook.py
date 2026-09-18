"""The bridge from the backtest engine to a tracker (plan §5.5).

§5.5 asks for "per fold: running global metrics with ``step=fold``".
:class:`TrackerFoldHook` is the
:class:`~strikecast.backtest.protocols.FoldHook` that does it: the engine hands
it the cumulative predictions after every fold, exactly as it hands them to
:class:`~strikecast.backtest.hooks.PruningHook`, it scores them with the caller's
``metrics_fn`` and mirrors the result with :meth:`Tracker.log_fold`.

The hook scores the CUMULATIVE predictions, not the fold's own, so the curve
W&B shows is the same running score the Optuna pruner sees -- the two must not
tell different stories about the same run. ``metrics_fn`` is normally the
legacy ``_score_fold_preds``/``evaluate_long`` global row bound to its target
list and region names.

This module imports neither wandb nor the engine; it only needs the hook
signature, which is structural.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

__all__ = ["TrackerFoldHook", "numeric_metrics"]

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from darts import TimeSeries

    from strikecast.backtest.protocols import FoldResult

    from .protocol import Tracker

logger = logging.getLogger(__name__)


def numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep the float-able entries of a metrics row.

    An ``evaluate_long`` global row carries labels (``model``, ``paradigm``)
    next to its numbers, and W&B would chart those as NaN. Booleans are
    numbers in python but are dropped too: none of the metric views produce
    one, so a bool here means a flag column slipped in.
    """
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


class TrackerFoldHook:
    """Mirror running global metrics to a :class:`Tracker`, one point per fold.

    Parameters
    ----------
    tracker:
        Anything satisfying :class:`~strikecast.tracking.protocol.Tracker`;
        a :class:`~strikecast.tracking.noop.NoopTracker` makes the hook inert.
    metrics_fn:
        ``cumulative -> Mapping[str, float]``. Receives the engine's
        ``channel -> per region -> per fold`` bundle and returns the running
        global metrics for the folds seen so far. Returning ``None`` or an
        empty mapping skips that fold.
    prefix:
        Prepended to every metric key, e.g. ``"cv/"`` so the CV and test stages
        of one run do not share a chart. Empty by default.
    every:
        Score only every ``every``-th fold (fold 0 always scores). Scoring is
        not free -- it re-evaluates all folds so far -- so a long test stage can
        mirror once per retrain window with ``every=7`` while the engine still
        runs every fold.

    Failures inside ``metrics_fn`` are logged and swallowed: tracking must not
    be able to abort a run. That is the opposite of
    :class:`~strikecast.backtest.hooks.PruningHook`, whose exceptions are load
    bearing.
    """

    def __init__(
        self,
        tracker: Tracker,
        metrics_fn: Callable[[dict[str, list[list[TimeSeries]]]], Mapping[str, Any] | None],
        *,
        prefix: str = "",
        every: int = 1,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.tracker = tracker
        self.metrics_fn = metrics_fn
        self.prefix = prefix
        self.every = every
        self.last_metrics: dict[str, float] | None = None

    # -- FoldHook ------------------------------------------------------------
    def on_fold(
        self,
        result: FoldResult,
        cumulative: dict[str, list[list[TimeSeries]]],
    ) -> None:
        fold = result.fold
        if fold.index % self.every:
            return
        try:
            raw = self.metrics_fn(cumulative)
        except Exception as exc:
            logger.warning("fold metrics failed at fold %d (%s); not mirrored", fold.index, exc)
            return
        if not raw:
            return
        metrics = numeric_metrics(raw)
        if self.prefix:
            metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
        self.last_metrics = metrics
        self.tracker.log_fold(fold.index, metrics)
