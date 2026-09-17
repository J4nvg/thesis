"""Unit tests for `strikecast.backtest.schedule`.

Every expected fold list here is hand-computed from the legacy loop::

    start_idx = int(start_frac * n_total)
    for t0 in range(start_idx, n_total - horizon + 1, predict_stride):
        retrain = (t0 - start_idx) % retrain_stride == 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strikecast.backtest.schedule import fold_schedule, schedule_from_config
from strikecast.config.schema import BacktestConfig


def as_pairs(folds) -> list[tuple[int, bool]]:
    return [(f.t0, f.retrain) for f in folds]


# --------------------------------------------------------------------------- #
# the standard configuration: daily predictions, weekly retrains
# --------------------------------------------------------------------------- #
def test_daily_predict_weekly_retrain() -> None:
    """n=40, start_frac=0.5 -> start_idx=20, folds t0=20..33, retrain at 20 and 27."""
    folds = fold_schedule(n_total=40, start_frac=0.5, horizon=7)

    assert [f.t0 for f in folds] == list(range(20, 34))
    assert len(folds) == 14
    assert [f.t0 for f in folds if f.retrain] == [20, 27]
    assert [f.index for f in folds] == list(range(14))


def test_retrain_flag_is_relative_to_the_start_index() -> None:
    """The modulo is on ``t0 - start_idx``, not on ``t0``: fold 0 always retrains."""
    folds = fold_schedule(n_total=41, start_frac=0.5, horizon=7)  # start_idx = 20
    assert folds[0].t0 == 20
    assert folds[0].retrain is True
    assert as_pairs(folds)[:8] == [
        (20, True), (21, False), (22, False), (23, False),
        (24, False), (25, False), (26, False), (27, True),
    ]


def test_retrain_stride_none_never_retrains() -> None:
    """The Chronos-2 case: a fixed predictor, fit once before the loop."""
    folds = fold_schedule(n_total=40, start_frac=0.5, horizon=7, retrain_stride=None)

    assert len(folds) == 14
    assert not any(f.retrain for f in folds)


def test_retrain_stride_one_retrains_every_fold() -> None:
    folds = fold_schedule(n_total=30, start_frac=0.5, horizon=7, retrain_stride=1)
    assert all(f.retrain for f in folds)


def test_predict_stride_interacts_with_retrain_stride() -> None:
    """predict_stride=2, retrain_stride=7: offsets are even, so 7 is never hit."""
    folds = fold_schedule(
        n_total=60, start_frac=0.5, horizon=7, predict_stride=2, retrain_stride=7
    )
    assert [f.t0 for f in folds] == list(range(30, 54, 2))
    # offsets 0, 2, 4, ... 14, ...  ->  divisible by 7 at offsets 0 and 14
    assert [f.t0 for f in folds if f.retrain] == [30, 44]


# --------------------------------------------------------------------------- #
# horizon boundary
# --------------------------------------------------------------------------- #
def test_last_fold_fits_exactly_one_horizon() -> None:
    """The last origin is n_total - horizon, so the window ends on the last step."""
    folds = fold_schedule(n_total=20, start_frac=0.5, horizon=7)
    assert folds[-1].t0 == 13  # 13 + 7 = 20 = n_total


def test_start_index_on_the_boundary_gives_one_fold() -> None:
    # int(0.65 * 20) = 13 = n_total - horizon
    folds = fold_schedule(n_total=20, start_frac=0.65, horizon=7)
    assert as_pairs(folds) == [(13, True)]


def test_start_index_past_the_boundary_gives_no_folds() -> None:
    # int(0.7 * 20) = 14 > n_total - horizon = 13
    assert fold_schedule(n_total=20, start_frac=0.7, horizon=7) == []


def test_horizon_longer_than_the_series_gives_no_folds() -> None:
    assert fold_schedule(n_total=5, start_frac=0.0, horizon=7) == []


# --------------------------------------------------------------------------- #
# int() truncation (flag F17)
# --------------------------------------------------------------------------- #
def test_train_val_end_truncation_matches_int_not_round() -> None:
    """``TRAIN_FRAC + VAL_FRAC`` is 0.7999999999999999, and ``int()`` truncates.

    On the real panel (n=847) the artefact does not bite -- both 0.8 and
    0.7999999999999999 truncate to 677 -- but it does at n=855, and the legacy
    runners always use the un-rounded sum.
    """
    train_val_end = 0.70 + 0.10
    assert train_val_end == 0.7999999999999999
    assert train_val_end != 0.8

    assert fold_schedule(855, train_val_end, horizon=7)[0].t0 == 683
    assert fold_schedule(855, 0.8, horizon=7)[0].t0 == 684

    # the real panel length: no difference, recorded so a change is noticed
    assert fold_schedule(847, train_val_end, horizon=7)[0].t0 == 677
    assert fold_schedule(847, 0.8, horizon=7)[0].t0 == 677


def test_cv_start_frac_truncation_on_the_real_cv_view() -> None:
    """CV_START_VAL = 0.7 / 0.7999999999999999 = 0.875 on the 676-step CV view."""
    cv_start_frac = 0.70 / (0.70 + 0.10)
    assert cv_start_frac == 0.875
    folds = fold_schedule(676, cv_start_frac, horizon=7)
    assert folds[0].t0 == 591
    assert len(folds) == 79
    assert sum(f.retrain for f in folds) == 12


# --------------------------------------------------------------------------- #
# cutoffs
# --------------------------------------------------------------------------- #
def test_cutoff_is_time_index_at_t0() -> None:
    index = pd.date_range("2023-01-01", periods=40, freq="D")
    folds = fold_schedule(n_total=40, start_frac=0.5, horizon=7, time_index=index)

    assert folds[0].cutoff == pd.Timestamp("2023-01-21")  # index[20]
    assert all(f.cutoff == index[f.t0] for f in folds)


def test_cutoff_is_nat_without_a_time_index() -> None:
    """Documented fallback: an index-only schedule carries ``pandas.NaT``."""
    folds = fold_schedule(n_total=40, start_frac=0.5, horizon=7)
    assert all(f.cutoff is pd.NaT for f in folds)


# --------------------------------------------------------------------------- #
# config entry point
# --------------------------------------------------------------------------- #
def test_schedule_from_config_matches_fold_schedule() -> None:
    darts = pytest.importorskip("darts")

    index = pd.date_range("2023-01-01", periods=40, freq="D")
    series = darts.TimeSeries.from_times_and_values(index, np.arange(40.0).reshape(-1, 1))
    cfg = BacktestConfig(start_frac=0.5, horizon=7, predict_stride=1, retrain_stride=7)

    from_cfg = schedule_from_config(series, cfg)
    direct = fold_schedule(40, 0.5, 7, 1, 7, time_index=index)

    assert from_cfg == direct
    assert from_cfg[0].cutoff == index[20]


def test_backtest_config_defaults_and_strictness() -> None:
    cfg = BacktestConfig(start_frac=0.875)
    assert (cfg.horizon, cfg.predict_stride, cfg.retrain_stride) == (7, 1, 7)
    assert BacktestConfig(start_frac=0.8, retrain_stride=None).retrain_stride is None

    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        BacktestConfig(start_frac=0.8, stride=1)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("predict_stride", [0, -1])
def test_non_positive_predict_stride_rejected(predict_stride: int) -> None:
    with pytest.raises(ValueError, match="predict_stride"):
        fold_schedule(40, 0.5, 7, predict_stride=predict_stride)


@pytest.mark.parametrize("retrain_stride", [0, -1])
def test_non_positive_retrain_stride_rejected(retrain_stride: int) -> None:
    with pytest.raises(ValueError, match="retrain_stride"):
        fold_schedule(40, 0.5, 7, retrain_stride=retrain_stride)
