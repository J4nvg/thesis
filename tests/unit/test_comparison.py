"""Unit tests for `strikecast.evaluation.comparison` (plan sec. 7.2).

The frames are synthetic and small: a handful of regions over a contiguous run
of forecast origins, with model B built from model A plus a controlled error, so
every statistic has an answer that can be derived by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from strikecast.backtest.predictions import COLUMNS
from strikecast.evaluation.comparison import (
    JOIN_KEYS,
    DMResult,
    block_bootstrap_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    compare_pair,
    diebold_mariano,
    loss_values,
    mean_difference,
    moving_block_indices,
    paired_differences,
    pairwise_table,
    per_origin_mean,
    write_pairwise,
)

REGIONS = ("r0", "r1")
HORIZONS = (1, 2, 3)
N_ORIGINS = 40


def make_frame(err: np.ndarray | float, *, seed: int = 0, channel: str = "y_pred") -> pd.DataFrame:
    """A long PredictionSet-shaped frame; `err` is added to the truth to forecast.

    `err` is either a scalar or an array indexed the same way the rows are
    generated (region-major, then origin, then horizon).
    """
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2024-01-01", periods=N_ORIGINS, freq="D")
    rows = []
    for region in REGIONS:
        for fold, origin in enumerate(origins):
            for h in HORIZONS:
                rows.append(
                    {
                        "region": region,
                        "fold": fold,
                        "horizon": h,
                        "date": origin + pd.Timedelta(days=h - 1),
                        "y_true": float(rng.poisson(2.0)),
                        "y_pred": np.nan,
                        "origin_date": origin,
                        "channel": channel,
                    }
                )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    errors = np.asarray(err, dtype=float)
    if errors.ndim == 0:
        errors = np.full(len(frame), float(err))
    frame["y_pred"] = frame["y_true"].to_numpy(float) + errors
    return frame


@pytest.fixture
def frame_a() -> pd.DataFrame:
    return make_frame(0.5, seed=1)


@pytest.fixture
def frame_b() -> pd.DataFrame:
    """Same truth as `frame_a` (same `seed`), a strictly larger error."""
    return make_frame(1.5, seed=1)


# --------------------------------------------------------------------------- #
# losses and pairing
# --------------------------------------------------------------------------- #
def test_squared_and_absolute_losses(frame_a: pd.DataFrame) -> None:
    assert np.allclose(loss_values(frame_a, "squared"), 0.25)
    assert np.allclose(loss_values(frame_a, "absolute"), 0.5)


def test_an_unknown_loss_is_refused(frame_a: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown loss"):
        loss_values(frame_a, "pinball")


def test_paired_differences_join_on_region_origin_and_horizon(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    diff = paired_differences(frame_a, frame_b)
    assert list(diff.columns) == [*JOIN_KEYS, "loss_a", "loss_b", "d"]
    assert len(diff) == len(REGIONS) * N_ORIGINS * len(HORIZONS)
    assert np.allclose(diff["d"], 0.25 - 2.25)


def test_pairing_only_keeps_the_intersection(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> None:
    truncated = frame_b[frame_b["origin_date"] < "2024-01-11"]
    diff = paired_differences(frame_a, truncated)
    assert diff["origin_date"].nunique() == 10


def test_pairing_disjoint_frames_raises(frame_a: pd.DataFrame) -> None:
    shifted = frame_a.assign(origin_date=frame_a["origin_date"] + pd.Timedelta(days=365))
    with pytest.raises(ValueError, match="share no forecast instance"):
        paired_differences(frame_a, shifted)


def test_a_multi_channel_frame_must_be_narrowed(frame_a: pd.DataFrame) -> None:
    doubled = pd.concat([frame_a, frame_a.assign(channel="count")], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        paired_differences(doubled, doubled)
    diff = paired_differences(doubled, doubled, channel="y_pred")
    assert np.allclose(diff["d"], 0.0)


def test_fold_is_not_a_join_key(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> None:
    """A run whose fold numbering starts elsewhere still pairs on origin_date."""
    renumbered = frame_b.assign(fold=frame_b["fold"] + 100)
    diff = paired_differences(frame_a, renumbered)
    assert len(diff) == len(REGIONS) * N_ORIGINS * len(HORIZONS)


def test_mean_difference_orients_the_percentage_so_positive_means_a_wins(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    head = mean_difference(paired_differences(frame_a, frame_b))
    assert head["mean_loss_a"] == pytest.approx(0.25)
    assert head["mean_loss_b"] == pytest.approx(2.25)
    assert head["mean_diff"] == pytest.approx(-2.0)
    assert head["pct_improvement"] == pytest.approx(100.0 * (2.25 - 0.25) / 2.25)
    assert head["rmse_a"] == pytest.approx(0.5)
    assert head["pct_improvement_rmse"] == pytest.approx(100.0 * (1.5 - 0.5) / 1.5)


def test_per_origin_mean_collapses_regions_and_horizons(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    series = per_origin_mean(paired_differences(frame_a, frame_b))
    assert len(series) == N_ORIGINS
    assert series.index.is_monotonic_increasing
    assert np.allclose(series.to_numpy(), -2.0)


# --------------------------------------------------------------------------- #
# block bootstrap
# --------------------------------------------------------------------------- #
def test_moving_blocks_are_contiguous_and_the_right_length() -> None:
    rng = np.random.default_rng(0)
    idx = moving_block_indices(20, 7, rng)
    assert idx.size == 20
    assert idx.min() >= 0 and idx.max() < 20
    # every full block of 7 is a run of consecutive integers
    for start in range(0, 14, 7):
        block = idx[start : start + 7]
        assert np.all(np.diff(block) == 1)


def test_a_block_length_of_one_is_the_iid_bootstrap() -> None:
    rng = np.random.default_rng(0)
    idx = moving_block_indices(20, 1, rng)
    assert idx.size == 20


def test_a_block_longer_than_the_sample_is_clamped() -> None:
    rng = np.random.default_rng(0)
    idx = moving_block_indices(5, 50, rng)
    assert idx.size == 5
    assert np.all(np.diff(idx) == 1)


def test_the_bootstrap_resamples_origins_not_rows(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    diff = paired_differences(frame_a, frame_b)
    out = block_bootstrap_ci(diff, n_boot=50, seed=0)
    assert out["n_origins"] == N_ORIGINS
    assert out["estimate"] == pytest.approx(-2.0)
    # the difference is constant, so every resample gives the same mean
    assert out["ci_lo"] == pytest.approx(-2.0)
    assert out["ci_hi"] == pytest.approx(-2.0)


def test_the_bootstrap_interval_brackets_a_noisy_estimate() -> None:
    rng = np.random.default_rng(3)
    a = make_frame(rng.normal(0.0, 1.0, size=len(REGIONS) * N_ORIGINS * len(HORIZONS)), seed=1)
    b = make_frame(rng.normal(0.0, 2.0, size=len(REGIONS) * N_ORIGINS * len(HORIZONS)), seed=1)
    diff = paired_differences(a, b)
    out = block_bootstrap_ci(diff, n_boot=300, seed=0)
    assert out["ci_lo"] < out["estimate"] < out["ci_hi"]
    assert out["ci_hi"] < 0.0  # B really is worse
    assert out["boot_se"] > 0.0


def test_the_bootstrap_is_deterministic_given_the_seed() -> None:
    rng = np.random.default_rng(5)
    n = len(REGIONS) * N_ORIGINS * len(HORIZONS)
    a = make_frame(rng.normal(size=n), seed=1)
    b = make_frame(rng.normal(size=n) * 2, seed=1)
    diff = paired_differences(a, b)
    first = block_bootstrap_ci(diff, n_boot=100, seed=11)
    again = block_bootstrap_ci(diff, n_boot=100, seed=11)
    other = block_bootstrap_ci(diff, n_boot=100, seed=12)
    assert first == again
    assert first["ci_lo"] != other["ci_lo"]


def test_a_single_origin_has_no_interval(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> None:
    diff = paired_differences(frame_a, frame_b)
    one = diff[diff["origin_date"] == diff["origin_date"].min()]
    out = block_bootstrap_ci(one, n_boot=10, seed=0)
    assert out["n_origins"] == 1
    assert math.isnan(out["ci_lo"]) and math.isnan(out["ci_hi"])


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def test_dm_default_lag_is_horizon_minus_one() -> None:
    d = np.random.default_rng(0).normal(size=60)
    assert diebold_mariano(d, horizon=7).lag == 6
    assert diebold_mariano(d, horizon=1).lag == 0
    assert diebold_mariano(d, horizon=7, lag=2).lag == 2


def test_dm_with_lag_zero_and_no_hln_is_the_plain_t_ratio() -> None:
    d = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.25, 0.0, 0.15])
    res = diebold_mariano(d, horizon=1, hln=False)
    var = ((d - d.mean()) ** 2).mean()  # gamma_0, the ddof=0 variance
    assert res.stat == pytest.approx(d.mean() / math.sqrt(var / d.size))


def test_dm_detects_a_clear_difference_and_ignores_a_null_one() -> None:
    rng = np.random.default_rng(0)
    null = rng.normal(0.0, 1.0, size=200)
    real = rng.normal(-1.0, 0.2, size=200)
    assert diebold_mariano(null, horizon=7).p_value > 0.05
    assert diebold_mariano(real, horizon=7).p_value < 1e-6
    assert diebold_mariano(real, horizon=7).stat < 0  # A has the lower loss


def test_hln_shrinks_the_statistic_and_uses_the_t_distribution() -> None:
    d = np.random.default_rng(1).normal(-0.3, 1.0, size=50)
    plain = diebold_mariano(d, horizon=7, hln=False)
    corrected = diebold_mariano(d, horizon=7, hln=True)
    n, h = plain.n, 7
    factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert corrected.stat == pytest.approx(plain.stat * factor)
    assert abs(corrected.stat) < abs(plain.stat)
    assert corrected.p_value > plain.p_value


def test_bartlett_and_truncated_weights_differ() -> None:
    d = np.random.default_rng(2).normal(size=120)
    bartlett = diebold_mariano(d, horizon=7, weights="bartlett")
    truncated = diebold_mariano(d, horizon=7, weights="truncated")
    assert bartlett.hac_var != truncated.hac_var
    assert bartlett.weights == "bartlett"


def test_an_unknown_weight_scheme_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown HAC weights"):
        diebold_mariano([1.0, 2.0], horizon=1, weights="parzen")


def test_the_hac_variance_grows_with_positive_autocorrelation() -> None:
    rng = np.random.default_rng(4)
    white = rng.normal(size=400)
    ar = np.zeros(400)
    for t in range(1, 400):
        ar[t] = 0.8 * ar[t - 1] + rng.normal()
    assert diebold_mariano(ar, horizon=7).hac_var > diebold_mariano(ar, horizon=1).hac_var
    assert diebold_mariano(white, horizon=7).hac_var == pytest.approx(
        diebold_mariano(white, horizon=1).hac_var, rel=0.5
    )


def test_dm_on_a_degenerate_series_is_nan() -> None:
    res = diebold_mariano([0.5], horizon=7)
    assert isinstance(res, DMResult)
    assert math.isnan(res.stat) and math.isnan(res.p_value)
    zero_var = diebold_mariano([1.0, 1.0, 1.0, 1.0], horizon=1)
    assert math.isnan(zero_var.stat)


# --------------------------------------------------------------------------- #
# Cliff's delta
# --------------------------------------------------------------------------- #
def test_cliffs_delta_is_plus_one_when_a_always_wins() -> None:
    assert cliffs_delta([-1.0, -2.0, -0.5]) == pytest.approx(1.0)


def test_cliffs_delta_is_minus_one_when_b_always_wins() -> None:
    assert cliffs_delta([1.0, 2.0, 0.5]) == pytest.approx(-1.0)


def test_cliffs_delta_counts_ties_as_zero() -> None:
    assert cliffs_delta([-1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert cliffs_delta([-1.0, 0.0]) == pytest.approx(0.5)


def test_cliffs_delta_of_nothing_is_nan() -> None:
    assert math.isnan(cliffs_delta([]))


def test_cliffs_delta_magnitudes_follow_romano() -> None:
    assert cliffs_delta_magnitude(0.10) == "negligible"
    assert cliffs_delta_magnitude(0.20) == "small"
    assert cliffs_delta_magnitude(0.40) == "medium"
    assert cliffs_delta_magnitude(-0.90) == "large"
    assert cliffs_delta_magnitude(float("nan")) == "undefined"


# --------------------------------------------------------------------------- #
# the summary table
# --------------------------------------------------------------------------- #
def test_compare_pair_answers_all_four_questions(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    summary = compare_pair(frame_a, frame_b, label_a="A", label_b="B", n_boot=50)
    assert summary.model_a == "A" and summary.model_b == "B"
    assert summary.scope == "pooled"
    assert summary.horizon is None
    assert summary.dm_lag == max(HORIZONS) - 1
    assert summary.n_origins == N_ORIGINS
    assert summary.mean_diff == pytest.approx(-2.0)
    assert summary.pct_improvement > 0  # A is better
    assert summary.cliffs_delta_a_better == pytest.approx(1.0)
    assert summary.cliffs_magnitude == "large"


def test_compare_pair_per_horizon_uses_that_horizons_lag(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    summary = compare_pair(frame_a, frame_b, horizon=3, n_boot=20)
    assert summary.horizon == 3
    assert summary.dm_lag == 2
    assert summary.scope == "h=3"
    assert summary.n_instances == len(REGIONS) * N_ORIGINS


def test_compare_pair_on_an_absent_horizon_raises(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="no shared instances at horizon"):
        compare_pair(frame_a, frame_b, horizon=99, n_boot=10)


def test_compare_pair_is_antisymmetric(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> None:
    ab = compare_pair(frame_a, frame_b, n_boot=20)
    ba = compare_pair(frame_b, frame_a, n_boot=20)
    assert ab.mean_diff == pytest.approx(-ba.mean_diff)
    assert ab.cliffs_delta_a_better == pytest.approx(-ba.cliffs_delta_a_better)


def test_pairwise_table_covers_every_unordered_pair(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    frames = {"A": frame_a, "B": frame_b, "C": make_frame(2.5, seed=1)}
    table = pairwise_table(frames, n_boot=20)
    assert len(table) == 3  # AB, AC, BC
    assert set(zip(table["model_a"], table["model_b"], strict=True)) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    }
    assert (table["pct_improvement"] > 0).all()


def test_pairwise_table_can_emit_both_directions(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    table = pairwise_table({"A": frame_a, "B": frame_b}, n_boot=20, ordered=True)
    assert len(table) == 2
    assert table["mean_diff"].tolist() == pytest.approx(
        [-table["mean_diff"].iloc[1], -table["mean_diff"].iloc[0]]
    )


def test_pairwise_table_over_several_horizons(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    table = pairwise_table({"A": frame_a, "B": frame_b}, horizons=[None, 1, 2, 3], n_boot=10)
    assert len(table) == 4
    assert table["scope"].tolist() == ["pooled", "h=1", "h=2", "h=3"]


def test_the_absolute_loss_is_supported_end_to_end(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> None:
    summary = compare_pair(frame_a, frame_b, loss="absolute", n_boot=20)
    assert summary.loss == "absolute"
    assert summary.mean_loss_a == pytest.approx(0.5)
    assert summary.mean_loss_b == pytest.approx(1.5)


def test_write_pairwise_round_trips(frame_a: pd.DataFrame, frame_b: pd.DataFrame, tmp_path) -> None:
    table = pairwise_table({"A": frame_a, "B": frame_b}, n_boot=10)
    path = write_pairwise(table, tmp_path / "report" / "pairwise_squared.csv")
    back = pd.read_csv(path)
    assert list(back.columns) == list(table.columns)
    assert len(back) == len(table)
