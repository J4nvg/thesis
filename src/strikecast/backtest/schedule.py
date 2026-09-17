"""The expanding-window fold schedule.

One function, one loop. Every legacy runner -- ``run_expanding_cv``,
``run_expanding_cv_iter``, ``run_final_test`` (count family),
``run_expanding_cv`` / ``run_expanding_cv_iter`` / ``run_final_test_diff``
(diff family), ``run_hurdle_cv`` / ``run_final_test`` (hurdle),
``run_damage_cv`` / ``run_final_test`` (damage) and
``chronos2_rolling_long`` (fixed predictor) -- opens with exactly this::

    n_total   = len(reference)
    start_idx = int(start_frac * n_total)
    for t0 in range(start_idx, n_total - horizon + 1, predict_stride):
        retrain = (t0 - start_idx) % retrain_stride == 0
        cutoff  = reference.time_index[t0]

``int()`` truncates towards zero, which is what makes ``start_frac =
TRAIN_FRAC + VAL_FRAC = 0.7999999999999999`` (flag F17) meaningful: it can land
one index below ``int(0.8 * n)``. The schedule reproduces that truncation
verbatim rather than rounding.

The reference series is always the MODEL-space one (the diffed list for the
diff family), because that is the list the legacy runners take ``ref_diff`` /
``ref_ts`` from. ``cutoff`` is the first timestamp NOT in the training or
context window: darts 0.43's ``drop_after(cutoff)`` drops ``cutoff`` itself, so
both windows end at ``time_index[t0 - 1]``.

Not in scope here: the naive baselines in ``src/evaluation_tools.py`` clamp the
start index (``max(int(start_frac * n), 1)`` for ``naive_last`` and
``max(int(start_frac * n), 7)`` for ``naive_weekly``). Those clamps never bind
on the real panel (the smallest start index used is 591, the CV start on the 676-step view), but they are a
different formula and belong to the naive port, not to this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from .protocols import Fold

if TYPE_CHECKING:
    from collections.abc import Sequence

    from darts import TimeSeries

    from strikecast.config.schema import BacktestConfig

__all__ = ["fold_schedule", "schedule_from_config"]


def fold_schedule(
    n_total: int,
    start_frac: float,
    horizon: int,
    predict_stride: int = 1,
    retrain_stride: int | None = 7,
    time_index: Sequence[pd.Timestamp] | pd.Index | None = None,
) -> list[Fold]:
    """Return the folds of one expanding-window backtest.

    Parameters
    ----------
    n_total:
        Length of the MODEL-space reference series (``len(target_list[0])``).
    start_frac:
        Fraction of ``n_total`` at which the first fold starts. Truncated with
        ``int()``, exactly as the legacy runners do.
    horizon:
        Forecast horizon in steps (``OUTPUT_CHUNK_LEN``, 7 everywhere).
    predict_stride:
        Distance in steps between two consecutive forecast origins
        (``CV_STRIDE`` / ``predict_stride``, 1 everywhere).
    retrain_stride:
        Retrain every this many *steps* after the start index. ``None`` means
        "never retrain": every fold gets ``retrain=False``. This is the
        Chronos-2 case, where the predictor is fit once before the loop.
    time_index:
        The reference series' time index. When given, ``Fold.cutoff`` is
        ``time_index[t0]``. When ``None`` -- which only happens in pure-index
        unit tests -- every ``cutoff`` is ``pandas.NaT``, so a schedule built
        without an index can be compared on ``(t0, retrain)`` but never drives
        a real ``drop_after``.

    Notes
    -----
    ``retrain_stride`` of 0 and non-positive ``predict_stride`` are rejected
    here. The legacy code crashes on both (``ZeroDivisionError`` and
    ``range()``'s ``ValueError``); this only turns the crash into a message.
    """
    if predict_stride < 1:
        raise ValueError(f"predict_stride must be >= 1, got {predict_stride}")
    if retrain_stride is not None and retrain_stride < 1:
        raise ValueError(f"retrain_stride must be >= 1 or None, got {retrain_stride}")

    start_idx = int(start_frac * n_total)

    folds: list[Fold] = []
    for index, t0 in enumerate(range(start_idx, n_total - horizon + 1, predict_stride)):
        retrain = retrain_stride is not None and (t0 - start_idx) % retrain_stride == 0
        cutoff = cast("pd.Timestamp", pd.NaT if time_index is None else time_index[t0])
        folds.append(Fold(index=index, t0=t0, cutoff=cutoff, retrain=retrain))
    return folds


def schedule_from_config(reference: TimeSeries, cfg: BacktestConfig) -> list[Fold]:
    """``fold_schedule`` driven by a :class:`BacktestConfig` and a darts series.

    ``reference`` must be the MODEL-space series the engine slices on (the
    legacy ``ref_ts`` / ``ref_diff``), i.e. ``model_targets[0]``.
    """
    return fold_schedule(
        n_total=len(reference),
        start_frac=cfg.start_frac,
        horizon=cfg.horizon,
        predict_stride=cfg.predict_stride,
        retrain_stride=cfg.retrain_stride,
        time_index=reference.time_index,
    )
