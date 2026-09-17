"""`PredictionSet` unit tests, including a level-D check against the legacy collector.

The golden test here is `test_legacy_frame_matches_collect_predictions_long`:
it builds synthetic fold predictions whose last folds run PAST the end of the
actual series, so the legacy ``if t in actual_map`` filter actually fires, and
compares `PredictionSet.legacy_frame()` with
``src.evaluation_tools.collect_predictions_long`` via `assert_frame_equal`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from darts import TimeSeries  # noqa: E402

from strikecast.backtest.predictions import COLUMNS, LEGACY_COLUMNS, PredictionSet  # noqa: E402
from strikecast.backtest.protocols import SINGLE_CHANNEL  # noqa: E402

N_REGIONS = 3
N_FOLDS = 5
HORIZON = 7
N_POINTS = 40
STRIDE = 3
START = pd.Timestamp("2024-01-01")


def _actuals() -> list[TimeSeries]:
    """Three regions, 40 daily points, distinct deterministic values."""
    index = pd.date_range(START, periods=N_POINTS, freq="D")
    rng = np.random.default_rng(0)
    return [
        TimeSeries.from_times_and_values(
            index, (rng.integers(0, 5, size=N_POINTS) + r).astype(float).reshape(-1, 1)
        )
        for r in range(N_REGIONS)
    ]


def _fold_preds(actuals: list[TimeSeries], *, overrun: int = 0) -> list[list[TimeSeries]]:
    """Five folds per region, the last ``overrun`` steps past the actual series end.

    The fold origins are placed so that the final fold's horizon extends
    ``overrun`` days beyond the last actual date; those steps must be dropped
    by the collector.
    """
    per_region: list[list[TimeSeries]] = []
    last_t0 = N_POINTS - HORIZON + overrun
    first_t0 = last_t0 - (N_FOLDS - 1) * STRIDE
    for r, actual in enumerate(actuals):
        folds: list[TimeSeries] = []
        for f, t0 in enumerate(range(first_t0, last_t0 + 1, STRIDE)):
            index = pd.date_range(actual.time_index[0] + pd.Timedelta(days=t0), periods=HORIZON)
            values = (np.arange(HORIZON, dtype=float) + 10 * f + 100 * r).reshape(-1, 1)
            folds.append(TimeSeries.from_times_and_values(index, values))
        per_region.append(folds)
    return per_region


@pytest.fixture
def actuals() -> list[TimeSeries]:
    return _actuals()


@pytest.fixture
def region_names() -> list[str]:
    return ["alpha", "bravo", "charlie"]


# --------------------------------------------------------------------------
# Golden: legacy_frame == collect_predictions_long
# --------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize("overrun", [0, 4])
def test_legacy_frame_matches_collect_predictions_long(
    legacy_src, actuals, region_names, overrun
) -> None:
    fold_preds = _fold_preds(actuals, overrun=overrun)
    expected = legacy_src.collect_predictions_long(actuals, fold_preds, region_names)

    ps = PredictionSet.from_fold_preds(actuals, fold_preds, region_names)
    ps.assert_equal_legacy(expected)


@pytest.mark.golden
def test_actual_map_filter_actually_drops_rows(legacy_src, actuals, region_names) -> None:
    """Guards the guard: with ``overrun=4`` the last fold really is truncated."""
    full = legacy_src.collect_predictions_long(actuals, _fold_preds(actuals), region_names)
    over = legacy_src.collect_predictions_long(
        actuals, _fold_preds(actuals, overrun=4), region_names
    )
    assert len(over) < len(full)
    assert len(full) == N_REGIONS * N_FOLDS * HORIZON
    # 4 steps of the last fold, 1 of the second-to-last (stride 3), per region
    assert len(over) == N_REGIONS * (N_FOLDS * HORIZON - 5)


@pytest.mark.golden
def test_dict_input_single_channel_matches_legacy(legacy_src, actuals, region_names) -> None:
    fold_preds = _fold_preds(actuals, overrun=4)
    expected = legacy_src.collect_predictions_long(actuals, fold_preds, region_names)
    ps = PredictionSet.from_fold_preds(actuals, {SINGLE_CHANNEL: fold_preds}, region_names)
    ps.assert_equal_legacy(expected)


# --------------------------------------------------------------------------
# Schema, origin_date, channels
# --------------------------------------------------------------------------


def test_schema_and_row_order(actuals, region_names) -> None:
    ps = PredictionSet.from_fold_preds(actuals, _fold_preds(actuals), region_names)
    assert list(ps.frame.columns) == list(COLUMNS)
    assert ps.channels == (SINGLE_CHANNEL,)
    assert len(ps) == N_REGIONS * N_FOLDS * HORIZON
    # region-major, then fold, then horizon
    assert ps.frame["region"].tolist()[: N_FOLDS * HORIZON] == [region_names[0]] * (
        N_FOLDS * HORIZON
    )
    first = ps.frame.head(HORIZON + 1)
    assert first["fold"].tolist() == [0] * HORIZON + [1]
    assert first["horizon"].tolist() == list(range(1, HORIZON + 1)) + [1]


def test_origin_date_is_first_predicted_date(actuals, region_names) -> None:
    fold_preds = _fold_preds(actuals)
    ps = PredictionSet.from_fold_preds(actuals, fold_preds, region_names)
    for r, region in enumerate(region_names):
        sub = ps.frame[ps.frame["region"] == region]
        for f, pred in enumerate(fold_preds[r]):
            rows = sub[sub["fold"] == f]
            assert (rows["origin_date"] == pred.time_index[0]).all()
            # horizon 1's date IS the origin date
            assert rows.iloc[0]["date"] == pred.time_index[0]


def test_origin_date_is_derivable_from_fold(actuals, region_names) -> None:
    """`fold` and `origin_date` carry the same information (plan §5.2)."""
    ps = PredictionSet.from_fold_preds(actuals, _fold_preds(actuals), region_names)
    mapping = ps.frame.groupby("fold")["origin_date"].nunique()
    assert (mapping == 1).all()


def test_multi_channel_blocks_and_accessors(actuals, region_names) -> None:
    fold_preds = _fold_preds(actuals)
    doubled = [[ts * 2 for ts in region] for region in fold_preds]
    ps = PredictionSet.from_fold_preds(
        actuals, {"prob": fold_preds, "count": doubled}, region_names
    )
    assert ps.channels == ("prob", "count")
    assert len(ps) == 2 * N_REGIONS * N_FOLDS * HORIZON

    prob = ps.for_channel("prob")
    assert prob.channels == ("prob",)
    pd.testing.assert_frame_equal(
        prob.legacy_frame("prob"), ps.legacy_frame("prob"), check_dtype=True
    )
    np.testing.assert_allclose(
        ps.legacy_frame("count")["y_pred"].to_numpy(),
        2 * ps.legacy_frame("prob")["y_pred"].to_numpy(),
    )
    assert list(ps.legacy_frame("prob").columns) == list(LEGACY_COLUMNS)

    with pytest.raises(KeyError):
        ps.for_channel("hurdle")
    with pytest.raises(KeyError):
        ps.legacy_frame("hurdle")


def test_empty_run_gives_typed_empty_frame(actuals, region_names) -> None:
    ps = PredictionSet.from_fold_preds(actuals, [[], [], []], region_names)
    assert len(ps) == 0
    assert list(ps.frame.columns) == list(COLUMNS)
    assert ps.channels == ()


def test_rejects_wrong_columns() -> None:
    with pytest.raises(ValueError, match="expects columns"):
        PredictionSet(pd.DataFrame({"region": [], "fold": []}))


# --------------------------------------------------------------------------
# Persistence and concat
# --------------------------------------------------------------------------


def test_parquet_round_trip_is_exact(tmp_path, actuals, region_names) -> None:
    ps = PredictionSet.from_fold_preds(actuals, _fold_preds(actuals, overrun=4), region_names)
    path = ps.to_parquet(tmp_path / "nested" / "preds.parquet")
    assert path.exists()
    assert not (tmp_path / "nested" / "preds.parquet.tmp").exists()
    back = PredictionSet.from_parquet(path)
    pd.testing.assert_frame_equal(back.frame, ps.frame, check_dtype=True)


def test_concat_preserves_order_and_channels(actuals, region_names) -> None:
    a = PredictionSet.from_fold_preds(actuals, _fold_preds(actuals), region_names)
    b = PredictionSet.from_fold_preds(
        actuals, {"count": _fold_preds(actuals)}, region_names
    )
    both = PredictionSet.concat([a, b])
    assert both.channels == (SINGLE_CHANNEL, "count")
    assert len(both) == len(a) + len(b)
    assert both.frame.index.tolist() == list(range(len(both)))
    pd.testing.assert_frame_equal(
        both.legacy_frame(SINGLE_CHANNEL), a.legacy_frame(), check_dtype=True
    )


def test_concat_of_nothing_is_empty() -> None:
    assert len(PredictionSet.concat([])) == 0
