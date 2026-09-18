"""Unit tests for `strikecast.evaluation.leaderboard`.

Two halves: synthetic run-store trees built in `tmp_path`, and one golden test
that points the same builder at `golden/results/<family>/global_*.json` and
demands the thesis `leaderboard.csv` back.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strikecast.evaluation.leaderboard import (
    LEADERBOARD_KEYS,
    STAGES,
    MetricRow,
    build_leaderboard,
    build_leaderboard_from_rows,
    collect_metric_rows,
    discover_metric_files,
    golden_metric_rows,
    leaderboard_rows,
    metric_frame,
    read_global_metrics,
    write_leaderboard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "golden" / "results"

METRIC_ORDER = ("MAE", "RMSE", "MedAE", "ME", "ZeroAcc", "n", "MASE_mean", "RMSSE_mean")


def _metrics(mae: float, mase: float, n: int = 100) -> dict[str, float]:
    """A `global.json` payload with the legacy key order."""
    return {
        "MAE": mae,
        "RMSE": mae * 2.0,
        "MedAE": mae / 2.0,
        "ME": -mae / 4.0,
        "ZeroAcc": 0.86,
        "n": n,
        "MASE_mean": mase,
        "RMSSE_mean": mase / 2.0,
    }


def _write_stage(
    root: Path,
    experiment: str,
    model: str,
    paradigm: str,
    seed: int,
    stage: str,
    metrics: dict[str, float],
) -> Path:
    path = root / experiment / model / paradigm / f"seed={seed}" / stage / "metrics" / "global.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """Two models x two paradigms x two seeds x two stages, plus decoys."""
    root = tmp_path / "runs"
    for model, base in (("lightgbm_tuned", 0.40), ("catboost_tuned", 0.45)):
        for paradigm, bump in (("global", 0.0), ("local", 0.03)):
            for seed in (42, 1):
                for stage, scale in (("cv", 1.0), ("test", 1.8)):
                    mae = (base + bump) * scale + 0.001 * seed
                    mase = (1.5 + bump * 10) * scale + 0.01 * seed
                    _write_stage(root, "gbdt", model, paradigm, seed, stage, _metrics(mae, mase))
    # decoys that must never be walked into
    (root / "gbdt" / "shared").mkdir(parents=True, exist_ok=True)
    (root / "gbdt" / "shared" / "panel.abc.parquet").write_bytes(b"")
    (root / "gbdt" / "report").mkdir(parents=True, exist_ok=True)
    (root / "gbdt" / "tuning" / "lightgbm_tuned").mkdir(parents=True, exist_ok=True)
    (root / "gbdt" / "lightgbm_tuned" / "global" / "notaseed").mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def test_discovery_finds_every_stage_and_skips_the_reserved_directories(store: Path) -> None:
    found = list(discover_metric_files(store))
    assert len(found) == 2 * 2 * 2 * 2
    assert {row[0] for row in found} == {"gbdt"}
    assert {row[1] for row in found} == {"lightgbm_tuned", "catboost_tuned"}
    assert {row[3] for row in found} == {42, 1}
    assert {row[4] for row in found} == {"cv", "test"}
    assert all(p.name == "global.json" for *_, p in found)


def test_discovery_is_sorted_deterministically(store: Path) -> None:
    keys = [(e, m, p, s, STAGES.index(st)) for e, m, p, s, st, _ in discover_metric_files(store)]
    assert keys == sorted(keys)


def test_discovery_of_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert list(discover_metric_files(tmp_path / "nope")) == []


def test_a_directory_that_is_not_seed_shaped_is_ignored(store: Path) -> None:
    bad = store / "gbdt" / "lightgbm_tuned" / "global" / "notaseed" / "cv" / "metrics"
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "global.json").write_text("{}", encoding="utf-8")
    assert len(list(discover_metric_files(store))) == 16


def test_collect_filters_on_seed_model_and_paradigm(store: Path) -> None:
    rows = collect_metric_rows(store, seeds=[42], models=["lightgbm_tuned"], paradigms=["global"])
    assert len(rows) == 2
    assert {r.stage for r in rows} == {"cv", "test"}
    assert all(r.seed == 42 for r in rows)


def test_collect_restricted_to_one_stage(store: Path) -> None:
    rows = collect_metric_rows(store, stages=["test"])
    assert len(rows) == 8
    assert {r.stage for r in rows} == {"test"}


def test_read_global_metrics_preserves_key_order(tmp_path: Path) -> None:
    path = tmp_path / "global.json"
    payload = {"MAE": 1.0, "RMSE": 2.0, "n": 3}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert list(read_global_metrics(path)) == ["MAE", "RMSE", "n"]


def test_read_global_metrics_refuses_a_non_object(tmp_path: Path) -> None:
    path = tmp_path / "global.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="must hold an object"):
        read_global_metrics(path)


# --------------------------------------------------------------------------- #
# the frame
# --------------------------------------------------------------------------- #
def test_leaderboard_columns_are_split_paradigm_model_then_the_metric_order(store: Path) -> None:
    frame = build_leaderboard(store, experiment="gbdt", seed=42)
    assert tuple(frame.columns) == (*LEADERBOARD_KEYS, *METRIC_ORDER)


def test_leaderboard_holds_one_row_per_model_paradigm_stage(store: Path) -> None:
    frame = build_leaderboard(store, experiment="gbdt", seed=42)
    assert len(frame) == 8
    assert frame.duplicated(subset=list(LEADERBOARD_KEYS)).sum() == 0


def test_leaderboard_selects_a_single_seed(store: Path) -> None:
    a = build_leaderboard(store, experiment="gbdt", seed=42)
    b = build_leaderboard(store, experiment="gbdt", seed=1)
    assert len(a) == len(b) == 8
    assert not np.allclose(a["MAE"].to_numpy(), b["MAE"].to_numpy())


def test_leaderboard_sorts_by_split_first_then_mase_mean(store: Path) -> None:
    frame = build_leaderboard(store, experiment="gbdt", seed=42)
    assert frame["split"].tolist() == ["cv"] * 4 + ["test"] * 4
    for split in ("cv", "test"):
        sub = frame[frame["split"] == split]["MASE_mean"].tolist()
        assert sub == sorted(sub)


def test_a_missing_metric_becomes_nan_and_sorts_last(tmp_path: Path) -> None:
    rows = [
        MetricRow("e", "a", "global", 42, "cv", _metrics(0.4, 2.0)),
        MetricRow("e", "b", "global", 42, "cv", {"MAE": 0.1}),
    ]
    frame = build_leaderboard_from_rows(rows)
    assert frame["model"].tolist() == ["a", "b"]
    assert pd.isna(frame.loc[1, "MASE_mean"])


def test_leaderboard_rows_put_the_keys_first(store: Path) -> None:
    rows = leaderboard_rows(collect_metric_rows(store, seeds=[42]))
    assert all(tuple(list(r)[:3]) == LEADERBOARD_KEYS for r in rows)


def test_write_leaderboard_round_trips_without_an_index(store: Path, tmp_path: Path) -> None:
    frame = build_leaderboard(store, experiment="gbdt", seed=42)
    path = write_leaderboard(frame, tmp_path / "report" / "leaderboard.csv")
    back = pd.read_csv(path)
    assert list(back.columns) == list(frame.columns)
    assert len(back) == len(frame)


# --------------------------------------------------------------------------- #
# the long frame the seed aggregation consumes
# --------------------------------------------------------------------------- #
def test_metric_frame_is_long_and_covers_every_seed(store: Path) -> None:
    frame = metric_frame(collect_metric_rows(store))
    assert list(frame.columns) == [
        "experiment",
        "model",
        "paradigm",
        "seed",
        "stage",
        "metric",
        "value",
    ]
    assert len(frame) == 16 * len(METRIC_ORDER)
    assert set(frame["seed"]) == {42, 1}


def test_metric_frame_drops_non_numeric_metrics() -> None:
    rows = [MetricRow("e", "m", "global", 42, "cv", {"MAE": 1.0, "note": "hi", "flag": True})]
    frame = metric_frame(rows)
    assert frame["metric"].tolist() == ["MAE"]


def test_metric_frame_of_nothing_is_an_empty_typed_frame() -> None:
    frame = metric_frame([])
    assert frame.empty
    assert "value" in frame.columns


# --------------------------------------------------------------------------- #
# golden: the thesis leaderboards, rebuilt
# --------------------------------------------------------------------------- #
@pytest.mark.golden
@pytest.mark.parametrize("family", ["gbdt", "diff"])
def test_the_builder_reproduces_the_golden_leaderboard(family: str) -> None:
    """The golden `global_*.json` layout carries split/paradigm/model, so the
    same builder rebuilds `leaderboard.csv` exactly.

    Tolerance is CSV float round-trip only (`repr` is 17 significant digits,
    pandas writes fewer), hence `rtol=1e-12`, not an arithmetic allowance.
    """
    directory = GOLDEN / family
    if not (directory / "leaderboard.csv").exists():
        pytest.skip(f"golden artefacts for {family} not available")
    got = build_leaderboard_from_rows(golden_metric_rows(directory))
    expected = pd.read_csv(directory / "leaderboard.csv")

    assert list(got.columns) == list(expected.columns)
    assert got[list(LEADERBOARD_KEYS)].equals(expected[list(LEADERBOARD_KEYS)])
    numeric = [c for c in expected.columns if c not in LEADERBOARD_KEYS]
    np.testing.assert_allclose(
        got[numeric].to_numpy(float),
        expected[numeric].to_numpy(float),
        rtol=1e-12,
        atol=0.0,
        equal_nan=True,
    )


@pytest.mark.golden
def test_the_seed_42_row_of_a_run_store_matches_the_golden_row() -> None:
    """One golden row, replayed through the run-store reader.

    `golden_metric_rows` and `collect_metric_rows` must agree, so a stage that
    writes `global.json` under the plan 5.3 layout lands on the thesis numbers.
    """
    directory = GOLDEN / "gbdt"
    target = directory / "global_test_global_lightgbm_poisson_tuned.json"
    if not target.exists():
        pytest.skip("golden artefacts for gbdt not available")
    rows = golden_metric_rows(directory)
    row = next(
        r
        for r in rows
        if (r.stage, r.paradigm, r.model) == ("test", "global", "lightgbm_poisson_tuned")
    )
    expected = pd.read_csv(directory / "leaderboard.csv")
    wanted = expected[
        (expected["split"] == "test")
        & (expected["paradigm"] == "global")
        & (expected["model"] == "lightgbm_poisson_tuned")
    ].iloc[0]
    for metric in ("MAE", "RMSE", "MASE_mean", "RMSSE_mean", "n"):
        assert row.metrics[metric] == pytest.approx(float(wanted[metric]), rel=1e-12)


@pytest.mark.golden
def test_the_golden_reader_splits_paradigm_from_model_even_when_both_say_global() -> None:
    directory = GOLDEN / "gbdt"
    if not directory.exists():
        pytest.skip("golden artefacts not available")
    rows = golden_metric_rows(directory)
    assert {r.paradigm for r in rows} == {"global", "activity", "local"}
    assert all(not r.model.startswith(("global_", "activity_", "local_")) for r in rows)
    assert {r.stage for r in rows} == {"cv", "test"}


@pytest.mark.golden
def test_a_run_store_built_from_golden_rows_rebuilds_the_same_leaderboard(tmp_path: Path) -> None:
    """Round trip: golden JSON -> plan 5.3 tree -> `build_leaderboard(seed=42)`."""
    directory = GOLDEN / "gbdt"
    if not (directory / "leaderboard.csv").exists():
        pytest.skip("golden artefacts not available")
    root = tmp_path / "runs"
    for row in golden_metric_rows(directory):
        _write_stage(root, "gbdt", row.model, row.paradigm, 42, row.stage, dict(row.metrics))
    got = build_leaderboard(root, experiment="gbdt", seed=42)
    expected = pd.read_csv(directory / "leaderboard.csv")
    assert got[list(LEADERBOARD_KEYS)].equals(expected[list(LEADERBOARD_KEYS)])
    np.testing.assert_allclose(
        got["MASE_mean"].to_numpy(float),
        expected["MASE_mean"].to_numpy(float),
        rtol=1e-12,
    )
