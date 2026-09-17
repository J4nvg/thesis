"""Unit tests for ``strikecast.transforms``.

The diff transform is checked against a VERBATIM copy of
``_diff_regression.py::_diff_to_level`` (lines 660-674) pasted below, so the
test fails the moment the port drifts from the legacy expression.

Fixtures are local on purpose: ``tests/conftest.py`` is owned by another work
stream and this file must not depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.dataprocessing.transformers import Diff as DartsDiff

from strikecast.backtest.protocols import TargetTransform
from strikecast.transforms import Diff, Identity, get_transform

N_DAYS = 60
REGIONS = ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Verbatim legacy reference (_diff_regression.py lines 660-674). Do not edit.
# ---------------------------------------------------------------------------
def _diff_to_level(diff_pred_ts, level_anchor_ts):
    """Un-diff a single fold-prediction: anchor = last actual level before the
    forecast window. Returns a TimeSeries on the same time index as the input.
    """
    first_pred_time = diff_pred_ts.time_index[0]
    # find the timestamp immediately before the first predicted timestamp
    anchor_idx = level_anchor_ts.time_index.get_loc(first_pred_time) - 1
    anchor_value = float(level_anchor_ts.values()[anchor_idx, 0])

    diff_vals = diff_pred_ts.values().ravel()
    level_vals = anchor_value + np.cumsum(diff_vals)
    return TimeSeries.from_times_and_values(
        diff_pred_ts.time_index,
        level_vals.reshape(-1, 1),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def level_series() -> list[TimeSeries]:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-03-01", periods=N_DAYS, freq="D")
    out = []
    for i, region in enumerate(REGIONS):
        statics = pd.DataFrame({"region": [region], "tier": [float(i)]})
        values = rng.poisson(2.0 + i, N_DAYS).astype(float).reshape(-1, 1)
        out.append(
            TimeSeries.from_times_and_values(
                dates, values, columns=["y"], static_covariates=statics
            )
        )
    return out


# ---------------------------------------------------------------------------
# Protocol conformance / factory
# ---------------------------------------------------------------------------
def test_both_transforms_satisfy_the_frozen_protocol() -> None:
    assert isinstance(Identity(), TargetTransform)
    assert isinstance(Diff(), TargetTransform)


@pytest.mark.parametrize(
    ("name", "cls"),
    [("identity", Identity), ("diff", Diff), ("IDENTITY", Identity), (" Diff ", Diff)],
)
def test_get_transform_resolves_names(name: str, cls: type) -> None:
    t = get_transform(name)
    assert isinstance(t, cls)
    assert t.name == cls.name


def test_get_transform_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown target transform"):
        get_transform("log")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_identity_forward_is_a_new_list_of_the_same_objects(
    level_series: list[TimeSeries],
) -> None:
    out = Identity().forward(level_series)
    assert out is not level_series
    assert all(a is b for a, b in zip(out, level_series, strict=True))


def test_identity_inverse_returns_pred_unchanged(level_series: list[TimeSeries]) -> None:
    pred = level_series[0][10:17]
    assert Identity().inverse(pred, level_series[0]) is pred


# ---------------------------------------------------------------------------
# Diff.forward: how the legacy builds the diffed targets
# ---------------------------------------------------------------------------
def test_diff_forward_matches_the_legacy_darts_transformer(
    level_series: list[TimeSeries],
) -> None:
    legacy = DartsDiff(lags=1, dropna=True).fit_transform(list(level_series))
    ours = Diff().forward(level_series)
    assert len(ours) == len(legacy)
    for a, b in zip(ours, legacy, strict=True):
        assert a.time_index.equals(b.time_index)
        np.testing.assert_allclose(a.values(), b.values())


def test_diff_forward_equals_timeseries_diff_defaults(
    level_series: list[TimeSeries],
) -> None:
    """``Diff(lags=1, dropna=True)`` == ``ts.diff()`` with darts 0.43 defaults."""
    for src, out in zip(level_series, Diff().forward(level_series), strict=True):
        expected = src.diff()
        assert out.time_index.equals(expected.time_index)
        np.testing.assert_allclose(out.values(), expected.values())


def test_diff_forward_drops_the_first_timestamp(level_series: list[TimeSeries]) -> None:
    for src, out in zip(level_series, Diff().forward(level_series), strict=True):
        assert len(out) == len(src) - 1
        assert out.time_index[0] == src.time_index[1]
        assert out.time_index[-1] == src.time_index[-1]
        np.testing.assert_allclose(
            out.values().ravel(), np.diff(src.values().ravel())
        )


def test_diff_forward_keeps_static_covariates(level_series: list[TimeSeries]) -> None:
    """The models need the encoded statics for their encodings (F54)."""
    for src, out in zip(level_series, Diff().forward(level_series), strict=True):
        assert out.static_covariates is not None
        pd.testing.assert_frame_equal(out.static_covariates, src.static_covariates)


def test_diff_forward_does_not_mutate_the_input(level_series: list[TimeSeries]) -> None:
    before = [ts.values().copy() for ts in level_series]
    Diff().forward(level_series)
    for ts, vals in zip(level_series, before, strict=True):
        np.testing.assert_allclose(ts.values(), vals)


def test_diff_forward_is_stateless_across_calls(level_series: list[TimeSeries]) -> None:
    """A fresh darts transformer per call, exactly like the legacy script."""
    t = Diff()
    a, b = t.forward(level_series), t.forward(level_series)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x.values(), y.values())


# ---------------------------------------------------------------------------
# Diff.inverse: verbatim port + round trips
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("start", [1, 5, 20, N_DAYS - 7])
def test_diff_inverse_matches_the_verbatim_legacy_function(
    level_series: list[TimeSeries], start: int
) -> None:
    transform = Diff()
    diffed = transform.forward(level_series)
    for level, dseries in zip(level_series, diffed, strict=True):
        pred = dseries.slice_n_points_after(level.time_index[start], 5)
        expected = _diff_to_level(pred, level)
        got = transform.inverse(pred, level)
        assert got.time_index.equals(expected.time_index)
        np.testing.assert_allclose(got.values(), expected.values(), atol=0, rtol=0)


def test_diff_round_trip_recovers_the_level_series(
    level_series: list[TimeSeries],
) -> None:
    """True differences un-diffed on the full level anchor give back the levels."""
    transform = Diff()
    diffed = transform.forward(level_series)
    for level, dseries in zip(level_series, diffed, strict=True):
        for start in (1, 13, 40):
            pred = dseries.slice_n_points_after(level.time_index[start], 7)
            got = transform.inverse(pred, level)
            expected = level.values().ravel()[start : start + 7]
            np.testing.assert_allclose(got.values().ravel(), expected, atol=1e-9)


def test_diff_inverse_anchor_is_the_value_one_step_before_the_window(
    level_series: list[TimeSeries],
) -> None:
    level = level_series[0]
    transform = Diff()
    dseries = transform.forward(level_series)[0]
    start = 25
    pred = dseries.slice_n_points_after(level.time_index[start], 3)
    # Replace the values by a known ramp so the anchor is isolated.
    ramp = TimeSeries.from_times_and_values(
        pred.time_index, np.array([[1.0], [2.0], [3.0]])
    )
    anchor = float(level.values()[start - 1, 0])
    got = transform.inverse(ramp, level)
    np.testing.assert_allclose(
        got.values().ravel(), anchor + np.array([1.0, 3.0, 6.0]), atol=1e-12
    )


def test_diff_inverse_uses_the_full_level_series_not_a_view(
    level_series: list[TimeSeries],
) -> None:
    """The legacy CV passes ``target_level_list[r_idx]``: the FULL level series.

    A truncated context that still contains the first predicted timestamp gives
    the same answer; one that does not raises. Pinning both makes the contract
    ("context is the full series") explicit.
    """
    level = level_series[0]
    transform = Diff()
    dseries = transform.forward(level_series)[0]
    start = 30
    pred = dseries.slice_n_points_after(level.time_index[start], 4)

    truncated_ok = level[: start + 1]
    np.testing.assert_allclose(
        transform.inverse(pred, truncated_ok).values(),
        transform.inverse(pred, level).values(),
    )

    too_short = level[: start - 3]
    with pytest.raises(KeyError):
        transform.inverse(pred, too_short)


def test_diff_inverse_wraps_to_the_last_level_value_at_index_zero(
    level_series: list[TimeSeries],
) -> None:
    """F50: ``get_loc(first) - 1 == -1`` indexes the LAST level value, not an error.

    Legacy ``_diff_to_level`` uses a plain Python ``- 1`` on the positional
    index, so a prediction starting at the very first level timestamp anchors on
    the END of the series. Unreachable in the legacy schedule (the diff series
    starts one step after the level series, so the earliest possible prediction
    start is level index 1), but it is what the code does and it is preserved.
    """
    level = level_series[0]
    transform = Diff()
    pred = TimeSeries.from_times_and_values(
        level.time_index[:3], np.array([[1.0], [1.0], [1.0]])
    )
    got = transform.inverse(pred, level)
    last_value = float(level.values()[-1, 0])
    np.testing.assert_allclose(
        got.values().ravel(), last_value + np.array([1.0, 2.0, 3.0])
    )
    # ... and it agrees with the verbatim legacy function, which is the point.
    np.testing.assert_allclose(got.values(), _diff_to_level(pred, level).values())


def test_diff_inverse_output_shape_and_index(level_series: list[TimeSeries]) -> None:
    level = level_series[0]
    transform = Diff()
    dseries = transform.forward(level_series)[0]
    pred = dseries.slice_n_points_after(level.time_index[9], 7)
    got = transform.inverse(pred, level)
    assert got.time_index.equals(pred.time_index)
    assert got.n_components == 1
    assert got.n_samples == 1
    assert got.values().shape == (7, 1)
    # The legacy rebuild drops static covariates and renames the component.
    assert got.static_covariates is None


def test_diff_inverse_preserves_float_dtype(level_series: list[TimeSeries]) -> None:
    level = level_series[0]
    transform = Diff()
    dseries = transform.forward(level_series)[0]
    pred = dseries.slice_n_points_after(level.time_index[4], 3)
    assert transform.inverse(pred, level).values().dtype == np.float64


# ---------------------------------------------------------------------------
# Split-after-diff vs diff-after-split (the legacy ordering)
# ---------------------------------------------------------------------------
def test_legacy_orders_diff_before_split_real_panel_length() -> None:
    """``forward`` on the FULL list, then split -- never split then diff.

    Reproduced on the real panel length (847 days, printed by
    ``_diff_regression.ipynb``: ``original len 847`` / ``diffed len 846`` /
    ``CV view length 676`` / ``target_for_cv_diff len 676``).

    * level 847 -> ``split_before(0.7999...)`` -> 676, ending on day 675
    * diff 846  -> ``split_before(0.7999...)`` -> 676, ending on day 676
      (the legacy ordering: diff the full series, then split)
    * level CV view 676 -> diff -> 675, ending on day 675
      (the ordering the legacy does NOT use)

    So the diffed CV view is the same LENGTH as the level CV view but covers one
    day more, which is exactly why the un-diff anchor lookup must be done
    against the full level series. Pinned so the ordering cannot silently
    change. Note this contradicts the sentence in ``backtest/protocols.py``
    saying the model-space reference "is one step shorter than the level
    series": true for the test-stage full series, false for the CV view (F51).
    """
    train_val_end = 0.70 + 0.10
    n_days = 847
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    level = TimeSeries.from_times_and_values(
        dates, np.arange(n_days, dtype=float).reshape(-1, 1)
    )

    level_cv = level.split_before(train_val_end)[0]
    diff_full = Diff().forward([level])[0]
    diff_then_split = diff_full.split_before(train_val_end)[0]
    split_then_diff = Diff().forward([level_cv])[0]

    assert (len(level), len(diff_full)) == (847, 846)
    assert len(level_cv) == 676
    assert len(diff_then_split) == 676
    assert len(split_then_diff) == 675
    assert diff_then_split.time_index[-1] == level_cv.time_index[-1] + pd.Timedelta(
        days=1
    )
    assert split_then_diff.time_index[-1] == level_cv.time_index[-1]
