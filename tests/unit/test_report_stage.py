"""Unit tests for `strikecast.pipeline.report_stage` (plan sec. 7).

Everything here runs against a SYNTHETIC run store built in `tmp_path`: a few
`global.json` metric files and a few prediction parts, written by hand in the
plan sec. 5.3 layout. No pipeline runs, no real data, no darts -- the report
stage only reads what a finished stage left behind, and the point of these
tests is that it keeps producing a report when the store is partial, which is
what the store on a laptop actually looks like.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from strikecast.backtest.predictions import COLUMNS, PredictionSet  # noqa: E402
from strikecast.config.schema import (  # noqa: E402
    ExperimentConfig,
    ModelEntry,
    SeedConfig,
    StoreConfig,
)
from strikecast.pipeline import report_stage  # noqa: E402
from strikecast.store import RunKey, RunStore  # noqa: E402

EXPERIMENT = "diff"
REGIONS = ("r0", "r1", "r2", "r3")
HORIZONS = (1, 2, 3)
N_ORIGINS = 24
SEEDS = (42, 1, 2)
ACTIVITY = {"r0": 1, "r1": 1, "r2": 2, "r3": 3}

#: model -> (error scale, is it stochastic)
MODELS = {
    "lightgbm": (0.5, True),
    "lstm_w7": (0.9, True),
    "naive_last": (1.6, False),
}


# --------------------------------------------------------------------------- #
# synthetic store
# --------------------------------------------------------------------------- #
def _frame(scale: float, *, seed: int, channel: str = "y_pred") -> pd.DataFrame:
    """A long PredictionSet-shaped frame whose error size is `scale`."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2024-01-01", periods=N_ORIGINS, freq="D")
    rows = []
    for region in REGIONS:
        for fold, origin in enumerate(origins):
            for h in HORIZONS:
                y = float(rng.poisson(3.0))
                rows.append(
                    {
                        "region": region,
                        "fold": fold,
                        "horizon": h,
                        "date": origin + pd.Timedelta(days=h - 1),
                        "y_true": y,
                        "y_pred": y + rng.normal(0.0, scale),
                        "origin_date": origin,
                        "channel": channel,
                    }
                )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _metrics(frame: pd.DataFrame, seed: int) -> dict[str, float]:
    """A `global.json` payload with the legacy key order, derived from `frame`."""
    err = frame["y_pred"].to_numpy(float) - frame["y_true"].to_numpy(float)
    mae = float(np.abs(err).mean())
    return {
        "MAE": mae,
        "RMSE": float(np.sqrt((err**2).mean())),
        "MedAE": float(np.median(np.abs(err))),
        "ME": float(err.mean()),
        "ZeroAcc": 0.8,
        "n": int(len(frame)),
        "MASE_mean": mae * 1.1,
        "RMSSE_mean": mae * 0.9,
    }


def _write_run(
    store: RunStore,
    model: str,
    paradigm: str,
    seed: int,
    stage: str,
    frame: pd.DataFrame,
    *,
    predictions: bool = True,
) -> None:
    key = RunKey(EXPERIMENT, model, paradigm, seed)
    if predictions:
        store.part_writer(key, stage).write(PredictionSet(frame), 0, N_ORIGINS - 1)
    path = store.metrics_dir(key, stage) / "global.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_metrics(frame, seed)), encoding="utf-8")


def _write_series_manifest(store: RunStore, activity: dict[str, int]) -> Path:
    """The activity map the per-tier breakdown reads out of the store."""
    bundle = store.experiment_dir(EXPERIMENT) / "shared" / "series.deadbeef"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = bundle / "manifest.json"
    manifest.write_text(
        json.dumps({"region_names": list(activity), "activity_by_region": activity}),
        encoding="utf-8",
    )
    return manifest


def _cfg(root: Path, models: list[str], seeds: tuple[int, ...] = SEEDS) -> ExperimentConfig:
    return ExperimentConfig(
        name=EXPERIMENT,
        models=[ModelEntry(name=m) for m in models],
        store=StoreConfig(root=str(root)),
        seeds=SeedConfig(eval_seeds=list(seeds)),
    )


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    """Three models x one paradigm x three seeds, test stage, plus a CV stage."""
    store = RunStore(tmp_path / "runs")
    for model, (scale, _) in MODELS.items():
        for seed in SEEDS:
            frame = _frame(scale, seed=seed)
            _write_run(store, model, "global", seed, "test", frame)
        _write_run(store, model, "global", 42, "cv", _frame(scale * 1.2, seed=42))
    _write_series_manifest(store, ACTIVITY)
    return store


@pytest.fixture
def cfg(store: RunStore) -> ExperimentConfig:
    return _cfg(store.root, list(MODELS))


def _report(cfg: ExperimentConfig, store: RunStore, **kwargs) -> dict[str, Path]:
    """`report` with a cheap bootstrap; the arithmetic is tested elsewhere."""
    kwargs.setdefault("n_boot", 25)
    return report_stage.report(cfg, store=store, **kwargs)


# --------------------------------------------------------------------------- #
# the full set of plan sec. 7 outputs
# --------------------------------------------------------------------------- #
def test_report_writes_every_plan_7_output(cfg, store) -> None:
    written = _report(cfg, store)
    assert set(written) == {
        "leaderboard",
        "leaderboard_ci",
        "pairwise_rmse",
        "pairwise_mae",
        "cd_rmse",
        "cd_mae",
        "family_comparison",
        "summary",
    }
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0
    assert written["summary"].name == "summary.md"
    assert written["cd_rmse"].suffix == ".svg"
    assert "<svg" in written["cd_rmse"].read_text(encoding="utf-8")[:2000]


def test_the_pairwise_table_covers_every_breakdown(cfg, store) -> None:
    written = _report(cfg, store)
    table = pd.read_csv(written["pairwise_rmse"])
    assert set(table["breakdown"]) == {"global", "horizon", "tier"}
    # three models -> three unordered pairs, one global row each
    assert len(table[table["breakdown"] == "global"]) == 3
    assert set(table[table["breakdown"] == "horizon"]["scope"]) == {"h=1", "h=2", "h=3"}
    # the activity map has tiers 1, 2 and 3
    assert set(table[table["breakdown"] == "tier"]["tier"].astype(float)) == {1.0, 2.0, 3.0}
    labels = set(table["model_a"]) | set(table["model_b"])
    assert labels == {f"{m}@global" for m in MODELS}


def test_the_pairwise_table_has_the_plan_7_2_columns(cfg, store) -> None:
    written = _report(cfg, store)
    table = pd.read_csv(written["pairwise_mae"])
    for column in (
        "mean_diff",
        "pct_improvement",
        "ci_lo",
        "ci_hi",
        "dm_p",
        "dm_lag",
        "cliffs_delta_a_better",
        "cliffs_magnitude",
    ):
        assert column in table.columns
    assert (table["loss"] == "absolute").all()
    pooled = table[table["breakdown"] == "global"]
    assert pooled["dm_p"].between(0.0, 1.0).all()


def test_the_better_model_wins_by_the_expected_sign(cfg, store) -> None:
    written = _report(cfg, store)
    table = pd.read_csv(written["pairwise_rmse"])
    row = table[
        (table["breakdown"] == "global")
        & (table["model_a"] == "lightgbm@global")
        & (table["model_b"] == "naive_last@global")
    ].iloc[0]
    assert row["mean_diff"] < 0.0  # A's loss minus B's: A is better
    assert row["pct_improvement"] > 0.0
    assert row["cliffs_delta_a_better"] > 0.0


def test_family_comparison_pairs_families_on_the_same_seed(cfg, store) -> None:
    written = _report(cfg, store)
    table = pd.read_csv(written["family_comparison"])
    assert set(table["aggregate"]) == {"mean", "best"}
    assert {"gbdt", "rnn", "naive"} == set(table["family_a"]) | set(table["family_b"])
    test_rows = table[(table["stage"] == "test") & (table["metric"] == "MAE")]
    stochastic_pair = test_rows[
        (test_rows["family_a"] == "gbdt") & (test_rows["family_b"] == "rnn")
    ]
    assert not stochastic_pair.empty
    assert (stochastic_pair["n_seeds"] == len(SEEDS)).all()
    assert stochastic_pair["t_lo"].notna().all()
    assert (stochastic_pair["mean_diff"] < 0).all()  # the GBDT is the better one


def test_summary_renders_the_paper_tables(cfg, store) -> None:
    written = _report(cfg, store)
    text = written["summary"].read_text(encoding="utf-8")
    assert "# Report: diff" in text
    assert "## 1. Leaderboard (Table 4 layout)" in text
    assert "## 2. Pairwise differences" in text
    assert "## 3. Family comparison" in text
    assert "## 4. Rank statistics" in text
    assert "Diebold-Mariano p-values" in text
    assert "Friedman chi2" in text
    assert "cd_rmse.svg" in text
    assert "lightgbm@global" in text


def test_summary_marks_a_deterministic_model_instead_of_a_zero_width_interval(
    cfg, store
) -> None:
    written = _report(cfg, store)
    text = written["summary"].read_text(encoding="utf-8")
    ci = pd.read_csv(written["leaderboard_ci"])
    naive = ci[(ci["model"] == "naive_last") & (ci["stage"] == "test")]
    assert naive["deterministic"].all()
    assert naive["t_lo"].isna().all()
    assert "(det.)" in text
    # a stochastic model keeps its interval
    gbm = ci[(ci["model"] == "lightgbm") & (ci["stage"] == "test")]
    assert not gbm["deterministic"].any()
    assert gbm["t_lo"].notna().any()


# --------------------------------------------------------------------------- #
# graceful degradation (plan sec. 7, Q1)
# --------------------------------------------------------------------------- #
def test_an_empty_store_still_produces_a_report(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    cfg = _cfg(store.root, ["lightgbm"])
    written = _report(cfg, store)
    assert written["leaderboard"].is_file()
    frame = pd.read_csv(written["leaderboard"])
    assert frame.empty
    assert list(frame.columns) == ["split", "paradigm", "model"]
    text = written["summary"].read_text(encoding="utf-8")
    assert "No leaderboard rows" in text
    assert "Not available" in text


def test_a_single_model_store_reports_without_pairwise_tables(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    _write_run(store, "lightgbm", "global", 42, "test", _frame(0.5, seed=42))
    cfg = _cfg(store.root, ["lightgbm"], seeds=(42,))
    written = _report(cfg, store)
    assert "pairwise_rmse" not in written
    assert "cd_rmse" not in written
    assert written["leaderboard"].is_file() and written["summary"].is_file()
    text = written["summary"].read_text(encoding="utf-8")
    assert "1 run(s) with stored `test` predictions" in text
    assert len(pd.read_csv(written["leaderboard"])) == 1


def test_a_single_seed_store_reports_point_values(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    for model, (scale, _) in MODELS.items():
        _write_run(store, model, "global", 42, "test", _frame(scale, seed=42))
    cfg = _cfg(store.root, list(MODELS), seeds=(42,))
    written = _report(cfg, store)
    ci = pd.read_csv(written["leaderboard_ci"])
    stochastic = ci[ci["model"] == "lightgbm"]
    assert (stochastic["n"] == 1).all()
    assert stochastic["std"].isna().all()  # n=1: no spread, and no fake interval
    assert stochastic["t_lo"].isna().all()
    family = pd.read_csv(written["family_comparison"])
    assert (family["n_seeds"] == 1).all()
    assert family["t_lo"].isna().all()


def test_metrics_without_predictions_still_give_a_leaderboard(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    for model, (scale, _) in MODELS.items():
        _write_run(
            store, model, "global", 42, "test", _frame(scale, seed=42), predictions=False
        )
    cfg = _cfg(store.root, list(MODELS), seeds=(42,))
    written = _report(cfg, store)
    assert "pairwise_rmse" not in written
    assert len(pd.read_csv(written["leaderboard"])) == 3
    assert "family_comparison" in written  # metrics alone are enough for sec. 7.2 item 5


def test_two_models_have_no_critical_difference_diagram(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    for model in ("lightgbm", "naive_last"):
        _write_run(store, model, "global", 42, "test", _frame(MODELS[model][0], seed=42))
    cfg = _cfg(store.root, ["lightgbm", "naive_last"], seeds=(42,))
    written = _report(cfg, store)
    assert "pairwise_rmse" in written
    assert "cd_rmse" not in written
    text = written["summary"].read_text(encoding="utf-8")
    assert "needs at least 3 models" in text


def test_a_missing_activity_map_only_drops_the_tier_breakdown(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    for model, (scale, _) in MODELS.items():
        _write_run(store, model, "global", 42, "test", _frame(scale, seed=42))
    cfg = _cfg(store.root, list(MODELS), seeds=(42,))
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"fixed_dir": str(tmp_path / "nowhere")})}
    )
    written = _report(cfg, store)
    table = pd.read_csv(written["pairwise_rmse"])
    assert set(table["breakdown"]) == {"global", "horizon"}


def test_a_multi_channel_run_is_narrowed_to_one_channel(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    _write_run(store, "lightgbm", "global", 42, "test", _frame(0.5, seed=42))
    composite = pd.concat(
        [
            _frame(0.7, seed=7, channel="hurdle"),
            _frame(1.9, seed=8, channel="prob"),
            _frame(1.1, seed=9, channel="count"),
        ],
        ignore_index=True,
    )
    _write_run(store, "hurdle", "global", 42, "test", composite)
    cfg = _cfg(store.root, ["lightgbm", "hurdle"], seeds=(42,))
    written = _report(cfg, store)
    table = pd.read_csv(written["pairwise_rmse"])
    pooled = table[table["breakdown"] == "global"]
    assert len(pooled) == 1  # one pair, not three fanned-out channels
    row = pooled.iloc[0]
    assert row["model_a"] == "hurdle@global"
    # the hurdle product is the channel compared, so the loss matches scale 0.7
    assert row["mean_loss_a"] == pytest.approx(0.7**2, rel=0.3)


def test_an_unknown_channel_run_is_skipped_with_a_warning(tmp_path: Path, caplog) -> None:
    store = RunStore(tmp_path / "runs")
    _write_run(store, "lightgbm", "global", 42, "test", _frame(0.5, seed=42))
    damage = pd.concat(
        [_frame(0.7, seed=7, channel="health"), _frame(1.9, seed=8, channel="energy")],
        ignore_index=True,
    )
    _write_run(store, "damage", "global", 42, "test", damage)
    cfg = _cfg(store.root, ["lightgbm", "damage"], seeds=(42,))
    with caplog.at_level("WARNING"):
        written = _report(cfg, store)
    assert "pairwise_rmse" not in written  # only one comparable run is left
    assert any("none is a comparison default" in r.getMessage() for r in caplog.records)


def test_an_explicit_channel_selects_a_component(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    _write_run(store, "lightgbm", "global", 42, "test", _frame(0.5, seed=42))
    damage = pd.concat(
        [_frame(0.7, seed=7, channel="health"), _frame(1.9, seed=8, channel="energy")],
        ignore_index=True,
    )
    _write_run(store, "damage", "global", 42, "test", damage)
    cfg = _cfg(store.root, ["lightgbm", "damage"], seeds=(42,))
    written = _report(cfg, store, channel="health")
    assert "pairwise_rmse" in written


# --------------------------------------------------------------------------- #
# selection and options
# --------------------------------------------------------------------------- #
def test_models_and_paradigms_restrict_what_is_compared(cfg, store) -> None:
    written = _report(cfg, store, models=["lightgbm", "naive_last"])
    table = pd.read_csv(written["pairwise_rmse"])
    labels = set(table["model_a"]) | set(table["model_b"])
    assert labels == {"lightgbm@global", "naive_last@global"}


def test_comparisons_false_writes_the_leaderboards_only(cfg, store) -> None:
    written = _report(cfg, store, comparisons=False)
    assert set(written) == {"leaderboard", "leaderboard_ci", "summary"}


def test_the_comparison_stage_can_be_chosen(cfg, store) -> None:
    written = _report(cfg, store, comparison_stage="cv")
    assert "comparison stage: `cv`" in written["summary"].read_text(encoding="utf-8")
    assert "pairwise_rmse" in written


def test_the_report_is_deterministic(cfg, store) -> None:
    first = _report(cfg, store)
    first_text = {name: path.read_text(encoding="utf-8") for name, path in first.items()
                  if path.suffix in {".csv", ".md"}}
    second = _report(cfg, store)
    for name, text in first_text.items():
        assert second[name].read_text(encoding="utf-8") == text


def test_report_rewrites_into_the_same_directory(cfg, store) -> None:
    written = _report(cfg, store)
    assert {p.parent for p in written.values()} == {store.report_dir(EXPERIMENT)}
