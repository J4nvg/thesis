"""`strikecast.verification.legacy` + `scripts/verify_legacy.py` (audit 2026-09-26 D9).

The comparison, the tolerance levels, the case list and the report are tested
on small frames and on the real configs. The end-to-end laptop run (diff
`linear` + `naive_last`, one retrain window, on the real data; ~3 minutes, most
of it the panel build) is opt-in: `STRIKECAST_VERIFY_E2E=1`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from strikecast.verification import legacy

REPO = Path(__file__).resolve().parents[2]


def frame(n_folds=3, *, shift=0.0, y_true_shift=0.0, regions=("a", "b")):
    rows = []
    for region in regions:
        for fold in range(n_folds):
            for h in range(1, 8):
                rows.append({
                    "region": region, "fold": fold, "horizon": h,
                    "date": pd.Timestamp("2024-08-05") + pd.Timedelta(days=fold + h - 1),
                    "y_true": float(fold + h) + y_true_shift, "y_pred": float(fold) + shift,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def test_identical_frames_pass_at_level_e() -> None:
    out = legacy.compare_frames(frame(), frame(), max_folds=None, level="E", tol=0.0)
    assert out["status"] == "PASS" and out["max_abs_dy"] == 0.0 and out["folds"] == 3


def test_the_golden_frame_is_cut_to_the_folds_that_ran() -> None:
    out = legacy.compare_frames(frame(2), frame(5), max_folds=2, level="E", tol=0.0)
    assert out["status"] == "PASS" and out["rows"] == 2 * 2 * 7


def test_a_prediction_outside_the_tolerance_fails() -> None:
    out = legacy.compare_frames(frame(shift=0.3), frame(), max_folds=None, level="F", tol=0.5)
    assert out["status"] == "PASS"
    out = legacy.compare_frames(frame(shift=0.7), frame(), max_folds=None, level="F", tol=0.5)
    assert out["status"] == "FAIL" and out["max_abs_dy"] == pytest.approx(0.7)


def test_report_level_never_fails_but_flags_a_large_mae_change() -> None:
    small = legacy.compare_frames(frame(shift=0.01), frame(), max_folds=None, level="R", tol=None)
    assert small["status"] == "REPORT"
    big = legacy.compare_frames(frame(shift=5.0), frame(), max_folds=None, level="R", tol=None)
    assert big["status"] == "SUSPECT" and big["rel_mae_diff"] > legacy.SUSPECT_REL_MAE


def test_y_true_and_keys_are_always_exact() -> None:
    out = legacy.compare_frames(frame(y_true_shift=1.0), frame(), max_folds=None, level="R", tol=None)
    assert out["status"] == "FAIL" and "y_true" in out["note"]
    other = frame()
    other["date"] = other["date"] + pd.Timedelta(days=1)
    out = legacy.compare_frames(frame(), other, max_folds=None, level="R", tol=None)
    assert out["status"] == "FAIL" and "date" in out["note"]
    out = legacy.compare_frames(frame(2), frame(3), max_folds=None, level="E", tol=0.0)
    assert out["status"] == "FAIL" and "schedule" in out["note"]


def test_golden_probability_frames_use_y_prob() -> None:
    golden = frame().rename(columns={"y_pred": "y_prob"})
    out = legacy.compare_frames(frame(), golden, max_folds=None, level="R", tol=None)
    assert out["status"] == "REPORT"


# --------------------------------------------------------------------------- #
# tolerances, golden files, cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("experiment", "model", "family", "kind", "level"),
    [
        ("diff", "naive_last", "naive", "naive", "E"),
        ("diff", "linear", "linear", "global", "E"),
        ("diff", "arima", "arima", "local", "F"),
        ("diff", "lightgbm", "lightgbm", "global", "F"),
        ("count", "catboost_tweedie", "catboost", "global", "E"),
        ("diff", "catboost", "catboost", "global", "R"),
        ("count", "xgboost_tweedie", "xgboost", "global", "R"),
        ("count", "lstm_w7", "lstm", "global", "R"),
        ("chronos2", "chronos2_zero_shot", "chronos", "chronos", "F"),
        ("chronos2", "chronos2_fine_tuned", "chronos", "chronos", "R"),
        ("hurdle", "hurdle", "hurdle", "composite", "R"),
    ],
)
def test_tolerance_levels(experiment, model, family, kind, level) -> None:
    got_level, tol, why = legacy.tolerance_for(experiment, model, family, kind)
    assert got_level == level and why
    assert (tol is None) == (level == "R")


def test_golden_paths_follow_the_legacy_file_names(tmp_path) -> None:
    for rel in (
        "gbdt/predictions_long_test_local_catboost_tweedie_tuned.parquet",
        "lstm/predictions_long_test_activity_gru_w7_tuned.parquet",
        "diff/predictions_long_test_global_naive_last.parquet",
        "chronos2/predictions_long_chronos2_zero_shot.parquet",
        "finalhurdle/global_cv_hurdle_preds.parquet",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    roots = [tmp_path]
    assert legacy.golden_path("count", "catboost_tweedie", "local", "test", "catboost", roots)
    assert legacy.golden_path("count", "gru_w7", "activity", "test", "gru", roots)
    assert legacy.golden_path("diff", "naive_last", "global", "test", "naive", roots)
    assert legacy.golden_path("chronos2", "chronos2_zero_shot", "global", "test", "chronos", roots)
    assert legacy.golden_path("hurdle", "hurdle", "global", "cv", "hurdle", roots)
    assert legacy.golden_path("hurdle", "hurdle", "local", "cv", "hurdle", roots) is None


def test_the_cases_are_the_thesis_matrix() -> None:
    pytest.importorskip("hydra")
    pytest.importorskip("darts")
    cases = legacy.cases_for("count", overrides=["legacy=count"])
    pairs = {(c.model, c.paradigm) for c in cases}
    assert ("catboost_tweedie", "local") in pairs
    assert ("lstm_w7", "activity") in pairs and ("lstm_w7", "local") not in pairs
    assert len(cases) == 6 * 3 + 15 * 2
    assert all(c.stage == "test" for c in cases)
    gpu = legacy.cases_for("count", resource="gpu", overrides=["legacy=count"])
    assert {c.family for c in gpu} == {"lstm", "gru"}
    hurdle = legacy.cases_for("hurdle", overrides=["legacy=hurdle"])
    assert {(c.model, c.stage) for c in hurdle} == {("hurdle", "cv")}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_the_report_orders_failures_first(tmp_path) -> None:
    directory = legacy.results_dir(tmp_path)
    directory.mkdir(parents=True)
    for status, model in (("PASS", "linear"), ("FAIL", "arima"), ("REPORT", "lstm_w7")):
        result = legacy.Result("diff", model, "global", "test", "E", 0.0, status,
                               folds=7, max_abs_dy=0.1, note=f"{model} note | pipe")
        (directory / f"{model}.json").write_text(json.dumps(result.__dict__))
    csv_path, md_path, counts = legacy.write_report(tmp_path)
    table = pd.read_csv(csv_path)
    assert list(table["status"]) == ["FAIL", "REPORT", "PASS"]
    assert counts["FAIL"] == 1 and counts["PASS"] == 1
    text = md_path.read_text()
    assert "| FAIL | diff | arima |" in text and "pipe" in text and "note | pipe" not in text


def test_the_script_insists_on_pythonhashseed_zero(tmp_path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_legacy.py"), "report",
         "--store-root", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 2 and "PYTHONHASHSEED=0" in done.stderr


@pytest.mark.skipif(
    os.environ.get("STRIKECAST_VERIFY_E2E") != "1",
    reason="set STRIKECAST_VERIFY_E2E=1 (real data, ~3 min): diff linear + naive_last, 1 window",
)
def test_end_to_end_on_the_real_data(tmp_path) -> None:
    root = tmp_path / "runs_verify"
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_legacy.py"), "all",
         "--experiment=diff", "--models=linear,naive_last", "--windows=1",
         "--store-root", str(root)],
        env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True,
        timeout=1800, cwd=REPO,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    table = pd.read_csv(root / "_verification" / "verification.csv")
    assert set(table["status"]) == {"PASS"} and len(table) == 2
