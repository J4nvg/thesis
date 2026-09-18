"""Unit tests for `strikecast.evaluation.seeds` (plan sec. 7.1)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from strikecast.evaluation.leaderboard import MetricRow, metric_frame
from strikecast.evaluation.seeds import (
    CI_COLUMNS,
    GROUP_KEYS,
    aggregate_seeds,
    bootstrap_mean_interval,
    broadcast_deterministic,
    leaderboard_ci,
    t_interval,
    write_leaderboard_ci,
)

SEEDS = (42, 1, 2, 3, 4)


def _rows(model: str, values: dict[int, float], *, stage: str = "test") -> list[MetricRow]:
    return [
        MetricRow("gbdt", model, "global", seed, stage, {"MAE": v, "RMSE": v * 2})
        for seed, v in values.items()
    ]


# --------------------------------------------------------------------------- #
# intervals
# --------------------------------------------------------------------------- #
def test_t_interval_matches_the_textbook_formula() -> None:
    values = [1.0, 1.2, 0.9, 1.1, 1.05]
    lo, hi = t_interval(values)
    arr = np.asarray(values)
    half = scipy_stats.t.ppf(0.975, 4) * arr.std(ddof=1) / math.sqrt(5)
    assert lo == pytest.approx(arr.mean() - half)
    assert hi == pytest.approx(arr.mean() + half)


def test_t_interval_is_undefined_for_fewer_than_two_values() -> None:
    assert all(math.isnan(x) for x in t_interval([1.0]))
    assert all(math.isnan(x) for x in t_interval([]))


def test_t_interval_widens_as_alpha_shrinks() -> None:
    values = [1.0, 1.2, 0.9, 1.1, 1.05]
    lo95, hi95 = t_interval(values, alpha=0.05)
    lo99, hi99 = t_interval(values, alpha=0.01)
    assert lo99 < lo95 and hi99 > hi95


def test_t_interval_ignores_nan_values() -> None:
    assert t_interval([1.0, 2.0, float("nan")]) == pytest.approx(t_interval([1.0, 2.0]))


def test_bootstrap_interval_is_deterministic_given_the_seed() -> None:
    values = [1.0, 1.2, 0.9, 1.1, 1.05]
    a = bootstrap_mean_interval(values, n_boot=200, seed=7)
    b = bootstrap_mean_interval(values, n_boot=200, seed=7)
    c = bootstrap_mean_interval(values, n_boot=200, seed=8)
    assert a == b
    assert a != c


def test_bootstrap_interval_cannot_leave_the_observed_range() -> None:
    values = [1.0, 1.2, 0.9, 1.1, 1.05]
    lo, hi = bootstrap_mean_interval(values, n_boot=500, seed=0)
    assert min(values) <= lo <= hi <= max(values)


def test_bootstrap_interval_is_undefined_for_one_value() -> None:
    assert all(math.isnan(x) for x in bootstrap_mean_interval([1.0]))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_reports_mean_sd_n_and_range_per_cell() -> None:
    values = {42: 1.0, 1: 1.2, 2: 0.9, 3: 1.1, 4: 1.05}
    frame = aggregate_seeds(_rows("lightgbm", values))
    assert list(frame.columns) == list(CI_COLUMNS)
    mae = frame[frame["metric"] == "MAE"].iloc[0]
    arr = np.fromiter(values.values(), float)
    assert mae["n"] == 5
    assert mae["mean"] == pytest.approx(arr.mean())
    assert mae["std"] == pytest.approx(arr.std(ddof=1))
    assert mae["min"] == pytest.approx(arr.min())
    assert mae["max"] == pytest.approx(arr.max())
    assert mae["t_lo"] < mae["mean"] < mae["t_hi"]
    assert not mae["deterministic"]


def test_aggregation_is_per_experiment_model_paradigm_stage_metric() -> None:
    rows = _rows("a", {42: 1.0, 1: 1.1}) + _rows("b", {42: 2.0, 1: 2.1})
    rows += _rows("a", {42: 0.5, 1: 0.6}, stage="cv")
    frame = aggregate_seeds(rows)
    assert len(frame) == 3 * 2  # (a/test, b/test, a/cv) x (MAE, RMSE)
    assert set(GROUP_KEYS).issubset(frame.columns)
    a_cv = frame[(frame["model"] == "a") & (frame["stage"] == "cv") & (frame["metric"] == "MAE")]
    assert a_cv.iloc[0]["mean"] == pytest.approx(0.55)


def test_a_single_seed_gives_n_one_and_nan_spread() -> None:
    frame = aggregate_seeds(_rows("a", {42: 1.0}))
    row = frame[frame["metric"] == "MAE"].iloc[0]
    assert row["n"] == 1
    assert row["mean"] == pytest.approx(1.0)
    assert math.isnan(row["std"])
    assert math.isnan(row["t_lo"]) and math.isnan(row["t_hi"])
    assert not row["deterministic"]


def test_the_aggregation_accepts_the_long_frame_as_well_as_rows() -> None:
    rows = _rows("a", {42: 1.0, 1: 1.2})
    assert aggregate_seeds(rows).equals(aggregate_seeds(metric_frame(rows)))


def test_aggregating_nothing_gives_an_empty_typed_frame() -> None:
    frame = aggregate_seeds([])
    assert frame.empty
    assert list(frame.columns) == list(CI_COLUMNS)


def test_expected_seeds_and_the_seed_list_are_recorded() -> None:
    frame = aggregate_seeds(_rows("a", {42: 1.0, 2: 1.2}), eval_seeds=SEEDS)
    row = frame[frame["metric"] == "MAE"].iloc[0]
    assert row["expected_seeds"] == 5
    assert row["n"] == 2
    assert row["seeds"] == "2;42"


# --------------------------------------------------------------------------- #
# deterministic models (plan sec. 5.4)
# --------------------------------------------------------------------------- #
def test_a_deterministic_model_is_broadcast_across_the_seed_list() -> None:
    rows = _rows("arima", {42: 1.85})
    out = broadcast_deterministic(rows, stochastic={"arima": False}, eval_seeds=SEEDS)
    assert len(out) == 5
    assert {r.seed for r in out} == set(SEEDS)
    assert {r.metrics["MAE"] for r in out} == {1.85}


def test_broadcast_leaves_a_stochastic_model_alone() -> None:
    rows = _rows("lightgbm", {42: 1.0})
    out = broadcast_deterministic(rows, stochastic={"lightgbm": True}, eval_seeds=SEEDS)
    assert len(out) == 1


def test_a_model_absent_from_the_stochastic_map_is_not_broadcast() -> None:
    rows = _rows("mystery", {42: 1.0})
    out = broadcast_deterministic(rows, stochastic={"arima": False}, eval_seeds=SEEDS)
    assert len(out) == 1


def test_broadcast_never_overwrites_a_real_run() -> None:
    rows = _rows("arima", {42: 1.85, 1: 1.90})
    out = broadcast_deterministic(rows, stochastic={"arima": False}, eval_seeds=SEEDS)
    by_seed = {r.seed: r.metrics["MAE"] for r in out}
    assert by_seed[42] == 1.85
    assert by_seed[1] == 1.90
    assert by_seed[2] == 1.90  # broadcast from the lowest seed present, which is 1


def test_broadcast_is_a_no_op_without_a_map_or_a_seed_list() -> None:
    rows = _rows("arima", {42: 1.85})
    assert broadcast_deterministic(rows, stochastic=None, eval_seeds=SEEDS) == rows
    assert broadcast_deterministic(rows, stochastic={"arima": False}, eval_seeds=None) == rows


def test_a_deterministic_model_gets_a_point_value_not_a_zero_width_interval() -> None:
    rows = broadcast_deterministic(
        _rows("arima", {42: 1.85}), stochastic={"arima": False}, eval_seeds=SEEDS
    )
    frame = aggregate_seeds(rows, stochastic={"arima": False}, eval_seeds=SEEDS)
    row = frame[frame["metric"] == "MAE"].iloc[0]
    assert row["deterministic"]
    assert row["mean"] == pytest.approx(1.85)
    assert math.isnan(row["std"])
    assert math.isnan(row["t_lo"]) and math.isnan(row["t_hi"])
    assert math.isnan(row["boot_lo"]) and math.isnan(row["boot_hi"])


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #
def _store(root: Path) -> Path:
    for seed, mae in zip(SEEDS, (0.75, 0.76, 0.74, 0.77, 0.755), strict=True):
        for model in ("lightgbm_tuned", "arima"):
            value = mae if model == "lightgbm_tuned" else 0.81
            path = (
                root
                / "gbdt"
                / model
                / "global"
                / f"seed={seed}"
                / "test"
                / "metrics"
                / "global.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"MAE": value, "MASE_mean": value * 4}), encoding="utf-8")
            if model == "arima" and seed != 42:
                path.unlink()  # deterministic: ran once
    return root


def test_leaderboard_ci_writes_the_report_file(tmp_path: Path) -> None:
    root = _store(tmp_path / "runs")
    frame = leaderboard_ci(
        root,
        experiment="gbdt",
        stages=["test"],
        eval_seeds=SEEDS,
        stochastic={"arima": False, "lightgbm_tuned": True},
        n_boot=100,
    )
    out = root / "gbdt" / "report" / "leaderboard_ci.csv"
    assert out.exists()
    back = pd.read_csv(out)
    assert list(back.columns) == list(CI_COLUMNS)
    lgb = frame[(frame["model"] == "lightgbm_tuned") & (frame["metric"] == "MAE")].iloc[0]
    arima = frame[(frame["model"] == "arima") & (frame["metric"] == "MAE")].iloc[0]
    assert lgb["n"] == 5 and not lgb["deterministic"]
    assert arima["n"] == 5 and arima["deterministic"]  # broadcast from its one run
    assert arima["mean"] == pytest.approx(0.81)


def test_write_leaderboard_ci_creates_the_parent_directory(tmp_path: Path) -> None:
    frame = aggregate_seeds(_rows("a", {42: 1.0, 1: 1.1}))
    path = write_leaderboard_ci(frame, tmp_path / "deep" / "report" / "leaderboard_ci.csv")
    assert path.exists()
    assert len(pd.read_csv(path)) == len(frame)
