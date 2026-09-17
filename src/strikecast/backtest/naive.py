"""Zero-parameter baselines, ported verbatim from the legacy evaluation tools.

Source: ``src/evaluation_tools.py`` lines 158-235
(``naive_last_historical_forecasts``, ``naive_weekly_historical_forecasts``,
``naive_historical_forecasts``, ``naive_collect_long``). The arithmetic, the
loop bounds and the start-index rules are copied unchanged; only the names and
the return container differ.

The two baselines the thesis reports:

``naive_last``
    ``y_hat(t0 + h) = y(t0 - 1)`` for every step of the horizon (persistence).
``naive_weekly``
    ``y_hat(t0 + h) = y(t0 + h - 7)`` (seasonal naive; the skill-score
    reference in every leaderboard).

``naive_historical_forecasts`` -- here :func:`naive_fold_preds` -- is a third,
separate implementation used only by the two pre-hurdle notebooks. It offers
``last`` / ``last_non_negative`` / ``non_negative_mean`` and is NOT the
function behind ``naive_last``: it has its own start rule and its own ``t0 ==
0`` guard (see Q1).

Behaviour-preservation notes (plan §6):

* ``Q1`` the three start rules genuinely differ and are preserved separately:
  ``naive_last`` uses ``max(int(start_frac * n), 1)``, ``naive_weekly`` uses
  ``max(int(start_frac * n), 7)``, and :func:`naive_fold_preds` uses a plain
  ``int(start_frac * n)`` plus an ``if t0 > 0`` guard that emits a 0.0
  forecast at ``t0 == 0``. The model runners
  (``run_expanding_cv`` / ``run_final_test``) use the plain
  ``int(start_frac * n_total)``. With the real fractions
  (``CV_START_VAL``/``TRAIN_VAL_END`` on 847- and 676-point series) all three
  land on the same index, so the folds line up and the leaderboards compare
  like with like -- but the ``max()`` clamps mean a short series or a small
  ``start_frac`` would silently give the naives FEWER folds than the models
  they are benchmarked against. See report flag F40.
* ``Q2`` ``naive_weekly`` reads ``vals[t0 - 7 : t0 - 7 + horizon]``. Because
  the start index is clamped at 7 this slice is always in bounds and always
  full length; it is never shortened or padded.
* ``Q3`` ``naive_collect_long`` dispatches through a two-entry dict, so any
  method other than ``"naive_last"`` / ``"naive_weekly"`` raises ``KeyError``.
  Preserved.
* ``Q4`` ``_chronos2.py::naive_rolling_long`` is a fourth implementation of the
  same two baselines, built directly on the AutoGluon frame. It is NOT ported
  and NOT used here; see report flag F41 for how it differs (no ``max()``
  clamp, a per-horizon ``continue`` instead, so early folds emit partial rows
  and the fold counter still advances).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from darts import TimeSeries

from .predictions import PredictionSet

__all__ = [
    "NAIVE_METHODS",
    "naive_fold_preds",
    "naive_last_fold_preds",
    "naive_prediction_set",
    "naive_weekly_fold_preds",
]


def _window(actual_ts: TimeSeries, t0: int, horizon: int) -> pd.Index:
    """``actual_ts.time_index[t0 : t0 + horizon]``; the cast is for the type checker."""
    return cast(pd.Index, actual_ts.time_index[t0 : t0 + horizon])


def naive_last_fold_preds(
    actual_ts: TimeSeries, start_frac: float, horizon: int, stride: int
) -> list[TimeSeries]:
    """``y_hat(t+h) = y(t-1)``, repeated across the horizon.

    Verbatim ``src/evaluation_tools.py::naive_last_historical_forecasts``.
    Start index ``max(int(start_frac * n), 1)`` (Q1).
    """
    n = len(actual_ts)
    vals = actual_ts.values().ravel()
    start_idx = max(int(start_frac * n), 1)
    preds: list[TimeSeries] = []
    for t0 in range(start_idx, n - horizon + 1, stride):
        last_val = float(vals[t0 - 1])
        values = np.full((horizon, 1), last_val, dtype=float)
        preds.append(TimeSeries.from_times_and_values(_window(actual_ts, t0, horizon), values))
    return preds


def naive_weekly_fold_preds(
    actual_ts: TimeSeries, start_frac: float, horizon: int, stride: int
) -> list[TimeSeries]:
    """``y_hat(t+h) = y(t+h-7)``: same-day-last-week seasonal naive.

    Verbatim ``src/evaluation_tools.py::naive_weekly_historical_forecasts``.
    Start index ``max(int(start_frac * n), 7)`` (Q1, Q2).
    """
    n = len(actual_ts)
    vals = actual_ts.values().ravel()
    start_idx = max(int(start_frac * n), 7)
    preds: list[TimeSeries] = []
    for t0 in range(start_idx, n - horizon + 1, stride):
        lagged = vals[t0 - 7 : t0 - 7 + horizon].reshape(-1, 1).astype(float)
        preds.append(TimeSeries.from_times_and_values(_window(actual_ts, t0, horizon), lagged))
    return preds


def naive_fold_preds(
    actual_ts: TimeSeries,
    start_frac: float,
    horizon: int,
    stride: int,
    method: str = "last",
) -> list[TimeSeries]:
    """Persistence baseline with three variants, used by the pre-hurdle notebooks.

    Verbatim ``src/evaluation_tools.py::naive_historical_forecasts``.

    ``method="last"``
        The most recent observed value regardless of sign
        (``event_classifiers_prehurdle`` behaviour).
    ``method="last_non_negative"``
        The most recent non-negative historical observation
        (``regressor_part_prehurdle`` behaviour).
    ``method="non_negative_mean"``
        The mean of all non-negative historical observations.

    Note this is NOT :func:`naive_last_fold_preds`: the start index is a plain
    ``int(start_frac * n_total)`` and ``t0 == 0`` yields a constant 0.0
    forecast rather than being skipped (Q1).
    """
    n_total = len(actual_ts)
    start_idx = int(start_frac * n_total)
    preds: list[TimeSeries] = []

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        pred_val = 0.0
        if t0 > 0:
            if method == "last":
                pred_val = float(actual_ts.values()[t0 - 1, 0])
            elif method in ("last_non_negative", "non_negative_mean"):
                past_vals = actual_ts.values()[:t0, 0]
                non_neg = past_vals[past_vals >= 0]
                if len(non_neg) > 0:
                    pred_val = (
                        float(non_neg[-1])
                        if method == "last_non_negative"
                        else float(np.mean(non_neg))
                    )
            else:
                raise ValueError(f"Unknown naive_fold_preds method: {method!r}")

        values = np.full((horizon, 1), pred_val, dtype=float)
        preds.append(TimeSeries.from_times_and_values(_window(actual_ts, t0, horizon), values))

    return preds


#: The dispatch table of ``naive_collect_long`` (Q3). A method outside these
#: two keys raises ``KeyError``, exactly as the legacy lookup does.
NAIVE_METHODS = {
    "naive_last": naive_last_fold_preds,
    "naive_weekly": naive_weekly_fold_preds,
}


def naive_prediction_set(
    target_list: Sequence[TimeSeries],
    region_names: Sequence[str],
    method: str,
    start_frac: float,
    horizon: int,
    stride: int,
) -> PredictionSet:
    """``naive_collect_long``, returning a :class:`~.predictions.PredictionSet`.

    Verbatim ``src/evaluation_tools.py::naive_collect_long`` except that the
    long frame is wrapped. ``method`` is ``"naive_last"`` or ``"naive_weekly"``
    (Q3); ``horizon`` and ``stride`` stay required so nothing couples back to a
    notebook constant.

    ``PredictionSet.legacy_frame()`` on the result is identical to the legacy
    ``(long_df, fold_preds)`` tuple's first element, dtypes and row order
    included. The fold predictions themselves are reachable through
    :data:`NAIVE_METHODS` when a caller needs them.
    """
    fn = NAIVE_METHODS[method]
    fold_preds_list = [fn(ts, start_frac, horizon, stride) for ts in target_list]
    return PredictionSet.from_fold_preds(target_list, fold_preds_list, region_names)
