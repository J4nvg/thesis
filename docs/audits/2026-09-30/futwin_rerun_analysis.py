"""Analyse the future-covariate-window sensitivity rerun (plan: FUTWIN_RERUN.md, same folder).

Read-only. Run on the laptop after pulling the cluster store:

    rsync -a uvt:reruns/thesis/runs_sensitivity_futwin/ runs_sensitivity_futwin/
    uv run python docs/audits/2026-09-30/futwin_rerun_analysis.py \
        | tee docs/audits/2026-09-30/futwin_rerun_output.txt

Self-test without cluster results (both "runs" are the publication run, so every delta is 0):

    uv run python docs/audits/2026-09-30/futwin_rerun_analysis.py \
        --lags87 runs_publication_20260929 --control runs_publication_20260929

Sections: [R0] inventory, [R1] control reproduces the publication run, [R2] Day-1 penalty per
configuration (2,7) vs (8,7) on the same 158 target dates, [R3] family-level horizon tests,
[R4] overall accuracy change (all target dates, leaderboard metric), [R5] verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import horizon_pipeline as hp  # noqa: E402

MODELS = ("lightgbm_poisson", "lightgbm_tweedie", "xgboost_poisson", "xgboost_tweedie",
          "catboost_poisson", "catboost_tweedie")
PARADIGMS = ("global", "activity", "local")
H = hp.H
pd.set_option("display.width", 200)


def test_dir(store: Path, model: str, paradigm: str) -> Path:
    return store / "count" / model / paradigm / "seed=42" / "test"


def complete(store: Path, model: str, paradigm: str) -> bool:
    state = store / "count" / model / paradigm / "seed=42" / "state.json"
    try:
        return json.loads(state.read_text())["test"]["status"] == "complete"
    except (OSError, KeyError, ValueError):
        return False


def load(store: Path, model: str, paradigm: str) -> pd.DataFrame:
    p = pd.read_parquet(test_dir(store, model, paradigm) / "predictions")
    p = p[p.channel == "y_pred"].copy()
    p["yhat"] = p.y_pred.clip(lower=0)  # metrics.py:249
    return p


def per_horizon(p: pd.DataFrame, dates=None) -> pd.DataFrame:
    if dates is not None:
        p = p[p.date.isin(dates)]
    e = p.y_true - p.yhat
    return pd.DataFrame({"RMSE": np.sqrt((e ** 2).groupby(p.horizon).mean()),
                         "MAE": e.abs().groupby(p.horizon).mean()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub", default=str(ROOT / "runs_publication_20260929"))
    ap.add_argument("--lags87", default=str(ROOT / "runs_sensitivity_futwin" / "lags_8_7"))
    ap.add_argument("--control", default=str(ROOT / "runs_sensitivity_futwin" / "control_2_7"))
    a = ap.parse_args()
    pub, s87, ctl = Path(a.pub), Path(a.lags87), Path(a.control)

    hp.section("R0", "Inventory")
    pairs = [(m, p) for m in MODELS for p in PARADIGMS]
    done = [(m, p) for m, p in pairs if complete(s87, m, p)]
    missing = [f"{m}@{p}" for m, p in pairs if (m, p) not in done]
    print(f"publication store: {pub}\n(8,7) store:       {s87}  -> {len(done)}/18 complete")
    if missing:
        print("NOT complete (check logs/slurm/sensitivity_futwin on the cluster):", ", ".join(missing))
    if not done:
        raise SystemExit("nothing to analyse yet")

    hp.section("R1", "Control: catboost_tweedie@global through the switch with (2,7) == publication run?")
    if complete(ctl, "catboost_tweedie", "global"):
        c, p = load(ctl, "catboost_tweedie", "global"), load(pub, "catboost_tweedie", "global")
        key = ["region", "origin_date", "horizon"]
        mrg = c.merge(p, on=key, suffixes=("_ctl", "_pub"))
        diff = float((mrg.y_pred_ctl - mrg.y_pred_pub).abs().max())
        ok = len(mrg) == len(p) == len(c) and diff < 1e-9
        print(f"rows {len(c)} vs {len(p)}, max |pred diff| = {diff:.3g} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("FAIL means the rerun differs from the publication run for a reason other than the "
                  "window (params, selection, library versions): do NOT interpret R2-R4 before fixing it.")
    else:
        print("control not complete: R2-R4 are unverified against a same-code rerun")

    p0 = load(pub, *done[0])
    common = set.intersection(*[set(p0.loc[p0.horizon == h, "date"]) for h in H])
    print(f"\ncommon target dates: {len(common)}")

    hp.section("R2", "Day-1 penalty = error(Day 1) / mean error(Days 2-7) - 1, same target dates (%)")
    rows, mats = [], {w: {m: {} for m in hp.METRICS} for w in ("2,7", "8,7")}
    for m, par in done:
        k = f"{m}@{par}"
        for w, store in (("2,7", pub), ("8,7", s87)):
            ph = per_horizon(load(store, m, par), common)
            for met in hp.METRICS:
                mats[w][met][k] = ph[met]
        row: dict[str, Any] = {"config": k}
        for met in hp.METRICS:
            for w in ("2,7", "8,7"):
                s = mats[w][met][k]
                row[f"{met} pen {w}"] = 100 * (s[1] / s[2:].mean() - 1)
            row[f"{met} d2<d1 (8,7)"] = bool(mats["8,7"][met][k][2] < mats["8,7"][met][k][1])
        rows.append(row)
    t = pd.DataFrame(rows)
    print(t.round(2).to_string(index=False))
    med = t[[c for c in t.columns if "pen" in c]].median()
    print("\nmedian over configurations:", "  ".join(f"{c} {v:+.2f}" for c, v in med.items()))

    hp.section("R3", "Family-level horizon tests, same target dates: Friedman + Day 2 vs Day 1")
    for w in ("2,7", "8,7"):
        for met in hp.METRICS:
            M = pd.DataFrame(mats[w][met]).T[H]
            fr = hp.friedman(M)
            rel = 100 * M.div(M[1], axis=0) - 100
            print(f"window {w} {met}: N={fr['N']} Friedman chi2={fr['chi2']:.1f} p={fr['p']:.3g} W={fr['W']:.2f} | "
                  f"Day2<Day1 {(M[2] < M[1]).sum()}/{len(M)} | median % vs Day 1, Days 2-7: "
                  + " ".join(f"{rel[h].median():+.1f}" for h in H[1:]))

    hp.section("R4", "Overall accuracy (all target dates, all horizons pooled) and per horizon, (8,7) - (2,7)")
    rows = []
    for m, par in done:
        a27, a87 = load(pub, m, par), load(s87, m, par)
        r: dict[str, Any] = {"config": f"{m}@{par}"}
        for w, d in (("2,7", a27), ("8,7", a87)):
            e = d.y_true - d.yhat
            r[f"RMSE {w}"], r[f"MAE {w}"] = float(np.sqrt((e ** 2).mean())), float(e.abs().mean())
        r["dRMSE %"] = 100 * (r["RMSE 8,7"] / r["RMSE 2,7"] - 1)
        r["dMAE %"] = 100 * (r["MAE 8,7"] / r["MAE 2,7"] - 1)
        ph27, ph87 = per_horizon(a27), per_horizon(a87)
        for h in (1, 7):
            r[f"dRMSE h{h} %"] = 100 * (ph87.RMSE[h] / ph27.RMSE[h] - 1)
        rows.append(r)
    t4 = pd.DataFrame(rows)
    print(t4.round(4).to_string(index=False))
    print("\nmedian:", "  ".join(f"{c} {t4[c].median():+.2f}" for c in t4.columns if c.startswith("d")))

    hp.section("R5", "Verdict (pre-registered in FUTWIN_RERUN.md)")
    p27, p87 = med["MAE pen 2,7"], med["MAE pen 8,7"]
    r27, r87 = med["RMSE pen 2,7"], med["RMSE pen 8,7"]
    shrink_mae = 1 - p87 / p27 if p27 > 0 else float("nan")
    shrink_rmse = 1 - r87 / r27 if r27 > 0 else float("nan")
    print(f"median Day-1 penalty MAE {p27:+.2f}% -> {p87:+.2f}% (shrinks {shrink_mae:.0%}); "
          f"RMSE {r27:+.2f}% -> {r87:+.2f}% (shrinks {shrink_rmse:.0%})")
    if len(done) < 18:
        print("INCOMPLETE: verdict is provisional until 18/18 stages are complete.")
    if shrink_mae >= 2 / 3 and shrink_rmse >= 2 / 3:
        print("CONFIRMED: the Day-1 penalty is the covariate-window artefact. See FUTWIN_RERUN.md 'If confirmed'.")
    elif shrink_mae <= 1 / 3 and shrink_rmse <= 1 / 3:
        print("REJECTED: widening the window does not remove the penalty. See FUTWIN_RERUN.md 'If rejected'.")
    else:
        print("PARTIAL: the window explains part of the penalty. See FUTWIN_RERUN.md 'If partial'.")


if __name__ == "__main__":
    main()
