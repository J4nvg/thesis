"""Analyse the feature-ablation sensitivity rerun (design: README.md, same folder).

Read-only and partial-store tolerant. Run on the laptop after pulling the cluster stores:

    rsync -a uvt:reruns/thesis/runs_feature_ablation/ runs_feature_ablation/
    uv run python docs/feature_ablation/analysis.py

stdout is also written to <out>/analysis_output.txt. Self-test without cluster results (every
variant's store is the publication store, so only seed 42 exists and every delta is 0):

    uv run python docs/feature_ablation/analysis.py --selftest --n-boot 200

Sections: [R0] inventory + metrics, [R1] tuned control reproduces the publication run,
[R2] selector value (vs all, vs random draws), [R3] single groups vs core, [R4] leave-one-out
vs selected (fixed and re-selected), [R5] pre-registered reading + round-2 commands.

Conventions of results.csv (one row per contrast x loss x seed scope):
- ``delta = variant - reference`` in metric units (RMSE for loss "squared", MAE for loss
  "absolute"); negative means the variant has the lower error. ``pct = 100 * delta / reference``.
- ``ci_lo``/``ci_hi``: 95% moving-block bootstrap over forecast origins (block 7) of that delta;
  for "squared" the same resamples as compare_pair's MSE interval, mapped to RMSE (a replicate's
  RMSE difference has the sign of its MSE difference, so "CI excludes 0" is the same verdict).
- ``dm_p``: HLN-corrected Diebold-Mariano on the per-origin loss differential; ``p_holm``: Holm
  within (run, section, loss, seed scope). ``cliffs_delta``: paired, positive = variant better.
- Seeds: per-seed rows pair variant seed s with reference seed s (compare_pair). The "pooled" row
  averages each instance's loss over the seeds both sides completed, then runs the same tests on
  that seed-averaged differential (origins stay whole, so a resample carries all seeds of an
  origin). compare_pair itself cannot stack seeds: its join keys (region, origin_date, horizon)
  must be unique per frame.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strikecast.evaluation.comparison import (
    DEFAULT_BLOCK_LENGTH,
    block_bootstrap_ci,
    cliffs_delta,
    compare_pair,
    diebold_mariano,
    mean_difference,
    moving_block_indices,
    paired_differences,
    per_origin_mean,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELFTEST_OUT = Path(tempfile.gettempdir()) / "feature_ablation_selftest"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

#: Group slugs in display order, and their category labels (evaluation/importance.py).
GROUPS: tuple[str, ...] = ("strikes", "spatial", "conflict", "comms", "macro", "missile", "cyber",
                           "weather", "calendar")
GROUP_LABELS: dict[str, str] = {
    "strikes": "Autoregressive strikes", "spatial": "Spatial / static", "conflict": "Conflict & damage",
    "comms": "Comms / diplo / aid", "macro": "Macroeconomic", "missile": "Missile / launch",
    "cyber": "Cyber", "weather": "Weather / geomag.", "calendar": "Calendar",
}
PAST_GROUPS = GROUPS[:7]
DRAWS = tuple(f"random_d{d}" for d in range(1, 6))
UNIFORM, STRATIFIED = DRAWS[:3], DRAWS[3:]
LOSSES = {"squared": "rmse", "absolute": "mae"}
KEY = ["region", "origin_date", "horizon"]


# --------------------------------------------------------------------------- #
# helpers: output, variant table, store access
# --------------------------------------------------------------------------- #
class Tee:
    """stdout to the terminal and to a file."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def section(tag: str, title: str) -> None:
    print(f"\n{'=' * 100}\n[{tag}] {title}\n{'=' * 100}")


def read_variants(path: Path) -> pd.DataFrame:
    v = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    v["seeds"] = v.seeds.map(lambda s: tuple(int(x) for x in s.split(",")))
    return v.set_index("variant", drop=False)


def parse_runs(items: Sequence[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        for tok in item.replace(",", " ").split():
            model, _, paradigm = tok.partition("@")
            out.append((model, paradigm or "global"))
    return out


class Stores:
    """Where each (variant, model, paradigm, seed) lives. ``selftest``: always the publication run."""

    def __init__(self, root: Path, pub: Path, selftest: bool) -> None:
        self.root, self.pub, self.selftest = root, pub, selftest

    def run_dir(self, variant: str, model: str, paradigm: str, seed: int) -> Path:
        store = self.pub if self.selftest else self.root / variant
        return store / "count" / model / paradigm / f"seed={seed}"

    def pub_dir(self, model: str, paradigm: str, seed: int = 42) -> Path:
        return self.pub / "count" / model / paradigm / f"seed={seed}"

    def extra_variants(self) -> list[str]:
        """Round-2 ``path_*`` stores, which are not rows of variants.tsv."""
        if self.selftest or not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.glob("path_*") if p.is_dir())


def complete(run: Path) -> bool:
    try:
        return json.loads((run / "state.json").read_text())["test"]["status"] == "complete"
    except (OSError, KeyError, ValueError):
        return False


@functools.cache
def load(run: Path) -> pd.DataFrame:
    p = pd.read_parquet(run / "test" / "predictions")
    p = p[p.channel == "y_pred"].copy()
    p["y_pred_raw"] = p.y_pred
    p["y_pred"] = p.y_pred.clip(lower=0)  # metrics.py:249: the leaderboard scores clipped predictions
    return p.reset_index(drop=True)


def feature_space(run: Path) -> dict[str, Any]:
    try:
        return json.loads((run / "feature_space.json").read_text())
    except (OSError, ValueError):
        return {}


def per_horizon(p: pd.DataFrame) -> pd.DataFrame:
    e = p.y_true - p.y_pred
    return pd.DataFrame({"rmse": np.sqrt((e ** 2).groupby(p.horizon).mean()),
                         "mae": e.abs().groupby(p.horizon).mean()})


def reference_rmse(pub: Path) -> float:
    """RMSE of diff naive_weekly (global, test, seed 42): the SkillScore reference (compute.py:86)."""
    path = pub / "diff" / "naive_weekly" / "global" / "seed=42" / "test" / "metrics" / "global.json"
    try:
        return float(json.loads(path.read_text())["RMSE"])
    except (OSError, KeyError, ValueError):
        board = pub / "_figures" / "master_leaderboard.csv"
        m = pd.read_csv(board)
        return float(m.loc[(m.Modelname == "diff") & (m.model == "naive_weekly"), "rmse"].iloc[0])


def stored_metrics(run: Path) -> dict[str, float]:
    try:
        return json.loads((run / "test" / "metrics" / "global.json").read_text())
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- #
# paired contrasts
# --------------------------------------------------------------------------- #
def metric_ci(diff: pd.DataFrame, loss: str, n_boot: int, seed: int = 0,
              alpha: float = 0.05) -> tuple[float, float]:
    """CI of the metric difference (RMSE or MAE), same origin resamples as block_bootstrap_ci."""
    g = diff.groupby("origin_date")
    sa, sb = g.loss_a.sum().sort_index().to_numpy(float), g.loss_b.sum().sort_index().to_numpy(float)
    cnt = g.loss_a.count().sort_index().to_numpy(float)
    n = sa.size
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = moving_block_indices(n, DEFAULT_BLOCK_LENGTH, rng)
        a, b, c = sa[idx].sum(), sb[idx].sum(), cnt[idx].sum()
        reps[i] = (math.sqrt(a / c) - math.sqrt(b / c)) if loss == "squared" else (a - b) / c
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _metric(mean_loss: float, loss: str) -> float:
    return math.sqrt(max(mean_loss, 0.0)) if loss == "squared" else mean_loss


def summarise(diff: pd.DataFrame, loss: str, n_boot: int) -> dict[str, Any]:
    """compare_pair's answer on an already built differential (used for the seed-pooled row)."""
    head = mean_difference(diff)
    boot = block_bootstrap_ci(diff, n_boot=n_boot)
    po = per_origin_mean(diff)
    dm = diebold_mariano(po, int(pd.to_numeric(diff.horizon).max()))
    return {"mean_loss_a": head["mean_loss_a"], "mean_loss_b": head["mean_loss_b"],
            "dm_p": dm.p_value, "cliffs_delta": cliffs_delta(po), "n_pairs": len(diff),
            "n_origins": boot["n_origins"]}


def contrast(
    frames_a: dict[int, pd.DataFrame], frames_b: dict[int, pd.DataFrame], *, section_tag: str,
    variant: str, reference: str, n_boot: int, note: str = "", pooled: bool = True,
    losses: Sequence[str] = tuple(LOSSES),
) -> list[dict[str, Any]]:
    """Per-seed rows (compare_pair) plus, if asked, the seed-pooled row, per loss."""
    seeds = sorted(set(frames_a) & set(frames_b), key=lambda s: (s != 42, s))
    rows: list[dict[str, Any]] = []
    for loss in losses:
        met = LOSSES[loss]
        per_seed, diffs = [], []
        for s in seeds:
            a, b = frames_a[s], frames_b[s]
            cp = compare_pair(a, b, label_a=variant, label_b=reference, loss=loss, n_boot=n_boot)
            diff = paired_differences(a, b, loss=loss)
            lo, hi = metric_ci(diff, loss, n_boot)
            va, vb = _metric(cp.mean_loss_a, loss), _metric(cp.mean_loss_b, loss)
            row = {"section": section_tag, "variant": variant, "reference": reference, "seed": str(s),
                   "loss": loss, "metric": met, "value_variant": va, "value_reference": vb,
                   "delta": va - vb, "pct": 100 * (va - vb) / vb if vb else np.nan,
                   "ci_lo": lo, "ci_hi": hi, "dm_p": cp.dm_p, "cliffs_delta": cp.cliffs_delta_a_better,
                   "n_pairs": cp.n_instances, "n_origins": cp.n_origins, "n_seeds": 1, "note": note}
            per_seed.append(row)
            diffs.append(diff.assign(seed=s))
        rows += per_seed
        if pooled and diffs:
            stacked = pd.concat(diffs).groupby(KEY, as_index=False)[["loss_a", "loss_b", "d"]].mean()
            sm = summarise(stacked, loss, n_boot)
            lo, hi = metric_ci(stacked, loss, n_boot)
            va, vb = _metric(sm["mean_loss_a"], loss), _metric(sm["mean_loss_b"], loss)
            rows.append({"section": section_tag, "variant": variant, "reference": reference,
                         "seed": "pooled", "loss": loss, "metric": met, "value_variant": va,
                         "value_reference": vb, "delta": va - vb,
                         "pct": 100 * (va - vb) / vb if vb else np.nan, "ci_lo": lo, "ci_hi": hi,
                         "dm_p": sm["dm_p"], "cliffs_delta": sm["cliffs_delta"],
                         "n_pairs": sm["n_pairs"], "n_origins": sm["n_origins"],
                         "n_seeds": len(seeds), "note": note,
                         "delta_seed_mean": float(np.mean([r["delta"] for r in per_seed])),
                         "seeds_sig_better": sum(r["ci_hi"] < 0 for r in per_seed),
                         "seeds_sig_worse": sum(r["ci_lo"] > 0 for r in per_seed)})
    return rows


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values; NaN stays NaN and does not count towards m."""
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = np.minimum(1.0, np.maximum.accumulate([(m - i) * v for i, v in enumerate(ok.to_numpy())]))
    out = pd.Series(np.nan, index=p.index)
    out[ok.index] = adj
    return out


def verdict(row: pd.Series | dict | None) -> str:
    """'better' / 'worse' / 'n.s.' / 'n/a' for the variant, from the delta CI."""
    if row is None:
        return "n/a"
    lo, hi = row["ci_lo"], row["ci_hi"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return "better" if hi < 0 else "worse" if lo > 0 else "n.s."


def show(res: pd.DataFrame, section_tag: str, seeds: Iterable[str] | None = None) -> None:
    t = res[res.section == section_tag]
    if seeds is not None:
        t = t[t.seed.isin(list(seeds))]
    if t.empty:
        print("(nothing to compare yet)")
        return
    t = t.assign(verdict=t.apply(verdict, axis=1))
    cols = ["variant", "reference", "seed", "metric", "value_variant", "value_reference", "delta",
            "pct", "ci_lo", "ci_hi", "dm_p", "p_holm", "cliffs_delta", "verdict", "note"]
    fmt = {**dict.fromkeys(("value_variant", "value_reference"), "{:.5f}"),
           **dict.fromkeys(("delta", "ci_lo", "ci_hi"), "{:+.5f}"),
           "pct": "{:+.2f}", "dm_p": "{:.3g}", "p_holm": "{:.3g}", "cliffs_delta": "{:+.3f}"}
    for met in ("rmse", "mae"):
        sub = t[t.metric == met][cols].drop(columns="metric")
        for c, f in fmt.items():
            sub[c] = sub[c].map(lambda x, f=f: "nan" if pd.isna(x) else f.format(x))
        sub = sub.rename(columns={"value_variant": "variant_val", "value_reference": "ref_val"})
        print(f"\n-- {met.upper()} (delta = variant - reference; negative = variant better)")
        print(sub.to_string(index=False))


# --------------------------------------------------------------------------- #
# per run (model@paradigm)
# --------------------------------------------------------------------------- #
def analyse_run(model: str, paradigm: str, stores: Stores, variants: pd.DataFrame,
                n_boot: int, ref_rmse: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run = f"{model}@{paradigm}"
    names = list(variants.index) + stores.extra_variants()
    expected = {v: (variants.seeds[v] if v in variants.index else (42,)) for v in names}

    section("R0", f"{run}: inventory (expected from variants.tsv vs complete test stages)")
    print(f"root: {stores.pub if stores.selftest else stores.root}"
          + ("   [SELFTEST: every variant is the publication run, only seed 42 exists]"
             if stores.selftest else ""))
    done: dict[str, dict[int, Path]] = {}
    missing = []
    for v, seeds in expected.items():
        for s in seeds:
            d = stores.run_dir(v, model, paradigm, s)
            if complete(d):
                done.setdefault(v, {})[s] = d
            else:
                missing.append(f"{v}/s{s}")
    n_exp = sum(len(s) for s in expected.values())
    n_done = sum(len(s) for s in done.values())
    print(f"{n_done}/{n_exp} jobs complete, {len(done)}/{len(expected)} variants with >= 1 seed")
    if missing:
        print(f"NOT complete ({len(missing)}; check `bash scripts/slurm/feature_ablation.sh --status`):")
        print("  " + ", ".join(missing))

    # metrics + per-horizon of every completed job
    mrows, hrows = [], []
    for v, seeds in done.items():
        for s, d in sorted(seeds.items()):
            p = load(d)
            e = p.y_true - p.y_pred
            rmse, mae = float(np.sqrt((e ** 2).mean())), float(e.abs().mean())
            stored = stored_metrics(d)
            if stored and abs(stored.get("RMSE", rmse) - rmse) > 1e-9:
                print(f"WARNING {v}/s{s}: recomputed RMSE {rmse:.6f} != stored {stored['RMSE']:.6f}")
            fs = feature_space(d)
            row = {"run": run, "variant": v, "family": variants.family.get(v, "path"), "seed": s,
                   "mae": mae, "rmse": rmse, "skill": 1 - rmse / ref_rmse, "n_rows": len(p),
                   "n_past_pairs": fs.get("n_past_pairs", np.nan),
                   "n_future": len(fs["future_keep"]) if "future_keep" in fs else np.nan,
                   "calendar_encoders": fs.get("calendar_encoders", np.nan),
                   "kept_groups": ";".join(fs.get("kept_groups", [])) if fs else "",
                   "mode": fs.get("mode", "")}
            for g in PAST_GROUPS:
                row[f"pairs_{g}"] = fs.get("pairs_by_group", {}).get(g, np.nan)
            mrows.append(row)
            ph = per_horizon(p).reset_index()
            hrows.append(ph.assign(run=run, variant=v, seed=s))
    metrics = pd.DataFrame(mrows)
    horizons = (pd.concat(hrows)[["run", "variant", "seed", "horizon", "rmse", "mae"]]
                if hrows else pd.DataFrame(columns=["run", "variant", "seed", "horizon", "rmse", "mae"]))
    if metrics.empty:
        print("nothing to analyse yet")
        return metrics, horizons, pd.DataFrame()
    agg = (metrics.groupby("variant", sort=False)
           .agg(seeds=("seed", "size"), rmse=("rmse", "mean"), mae=("mae", "mean"),
                skill=("skill", "mean"), pairs=("n_past_pairs", "first"), groups=("kept_groups", "first")))
    agg["groups"] = agg.groups.map(lambda g: "all 9" if len(str(g).split(";")) == len(GROUPS) else g)
    print(f"\nper-variant test metrics, seed mean (SkillScore = 1 - RMSE / {ref_rmse:.6f}, the diff "
          "naive_weekly RMSE):")
    agg.insert(1, "rank_rmse", agg.rmse.rank(method="min").astype(int))
    print(agg.to_string(float_format=lambda x: f"{x:.5f}"))

    def frames(v: str) -> dict[int, pd.DataFrame]:
        return {s: load(d) for s, d in done.get(v, {}).items()}

    rows: list[dict[str, Any]] = []

    section("R1", f"{run}: control `selected_tuned` (tuned params, seed 42) == publication run?")
    pub_run = stores.pub_dir(model, paradigm)
    if not complete(pub_run):
        print(f"publication run {pub_run} not complete: R1 cannot be checked")
    elif 42 not in done.get("selected_tuned", {}):
        print("selected_tuned/s42 not complete: R2-R4 are unverified against a same-code rerun")
    else:
        pub_pred = load(pub_run)
        pub_m = stored_metrics(pub_run)
        for v, expect_equal in (("selected_tuned", True), ("selected", False)):
            if 42 not in done.get(v, {}):
                continue
            c = load(done[v][42])
            mrg = c.merge(pub_pred, on=KEY, suffixes=("_v", "_pub"))
            diff = float((mrg.y_pred_raw_v - mrg.y_pred_raw_pub).abs().max())
            same_rows = len(mrg) == len(c) == len(pub_pred)
            if expect_equal:
                ok = same_rows and diff < 1e-12
                print(f"selected_tuned: rows {len(c)} vs {len(pub_pred)}, max |dy_pred| = {diff:.3g} "
                      f"-> {'PASS' if ok else 'FAIL'}")
                if not ok:
                    print("FAIL: the switch is not inert under mode=selected (or params/selection/library "
                          "versions differ). Do NOT interpret R2-R4 before fixing it.")
            else:
                print(f"selected (default params): max |dy_pred| vs publication = {diff:.3g} "
                      + ("(0 is expected only in the selftest)" if stores.selftest
                         else "(expected > 0: default vs tuned params)" if diff > 0
                         else "(UNEXPECTED 0: are the defaults really in use?)"))
        board = stores.pub / "_figures" / "master_leaderboard.csv"
        if pub_m and board.is_file():
            m = pd.read_csv(board)
            hit = m[(m.Modelname == "gbdt") & (m.paradigm == paradigm) & (m.model == model)]
            if not hit.empty:
                mine = 1 - pub_m["RMSE"] / ref_rmse
                print(f"SkillScore check, publication {run}: recomputed {mine:.5f}, leaderboard "
                      f"{float(hit.SkillScore.iloc[0]):.5f} (rmse {pub_m['RMSE']:.5f}, mae {pub_m['MAE']:.5f})")

    section("R2", f"{run}: selector value (reference = selected, default params; 3 seeds + pooled)")
    sel = frames("selected")
    if not sel:
        print("selected not complete: R2 and R4 need it")
    else:
        for v in ("all", *DRAWS, "core"):
            if v in done:
                rows += contrast(frames(v), sel, section_tag="R2", variant=v, reference="selected",
                                 n_boot=n_boot, note="stratified" if v in STRATIFIED else "")
        draws_done = [v for v in DRAWS if v in done]
        if draws_done:
            # median draw: per seed by that seed's metric; pooled by the seed-mean metric
            for loss, met in LOSSES.items():
                m = metrics[metrics.variant.isin(draws_done)]
                for s in sorted(set(m.seed) & set(sel), key=lambda s: (s != 42, s)):
                    ms = m[m.seed == s].sort_values(met)
                    med = ms.variant.iloc[(len(ms) - 1) // 2]
                    rows += contrast({s: load(done[med][s])}, {s: sel[s]}, section_tag="R2",
                                     variant="random_median", reference="selected", n_boot=n_boot,
                                     note=med, pooled=False, losses=(loss,))
                pm = m.groupby("variant")[met].mean().sort_values()
                med = pm.index[(len(pm) - 1) // 2]
                rows += [r for r in contrast(frames(med), sel, section_tag="R2", variant="random_median",
                                             reference="selected", n_boot=n_boot, note=med, losses=(loss,))
                         if r["seed"] == "pooled"]
    res = pd.DataFrame(rows)
    if not res.empty:
        res = _holm(res)
        show(res, "R2")
        _selector_descriptives(metrics)

    section("R3", f"{run}: single groups (core + one group) vs core, seed 42")
    rows3 = []
    core = frames("core")
    for g in GROUPS:
        for v, ref in ((f"only_{g}", "core"), (f"reselect_only_{g}", "core"),
                       (f"reselect_only_{g}", f"only_{g}")):
            fa, fb = frames(v), (core if ref == "core" else frames(ref))
            fa, fb = {k: x for k, x in fa.items() if k == 42}, {k: x for k, x in fb.items() if k == 42}
            if fa and fb:
                rows3 += contrast(fa, fb, section_tag="R3", variant=v, reference=ref, n_boot=n_boot,
                                  note=g, pooled=False)
    res = _append(res, rows3)
    show(res, "R3")

    section("R4", f"{run}: leave-one-out from the selected 100 vs selected, seed 42 (fixed and re-selected)")
    rows4 = []
    sel42 = {k: x for k, x in sel.items() if k == 42}
    for g in GROUPS:
        for v in (f"drop_{g}", f"reselect_drop_{g}"):
            fa = {k: x for k, x in frames(v).items() if k == 42}
            if fa and sel42:
                rows4 += contrast(fa, sel42, section_tag="R4", variant=v, reference="selected",
                                  n_boot=n_boot, note=g, pooled=False)
    res = _append(res, rows4)
    show(res, "R4")
    if not res.empty:
        _replaceable(res)

    section("R5", f"{run}: pre-registered reading (primary RMSE, secondary MAE; README 'Pre-registered reading')")
    _reading(res, metrics, missing)
    if not res.empty:
        res.insert(0, "run", run)
    return metrics, horizons, res


def _holm(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    res["p_holm"] = np.nan
    for _, idx in res.groupby(["section", "loss", "seed"]).groups.items():
        res.loc[idx, "p_holm"] = holm(res.loc[idx, "dm_p"])
    return res


def _append(res: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return res
    return _holm(pd.concat([res, pd.DataFrame(rows)], ignore_index=True))


def _get(res: pd.DataFrame, variant: str, reference: str, metric: str, seed: str) -> pd.Series | None:
    if res.empty:
        return None
    t = res[(res.variant == variant) & (res.reference == reference) & (res.metric == metric)
            & (res.seed == seed)]
    return None if t.empty else t.iloc[0]


def _selector_descriptives(metrics: pd.DataFrame) -> None:
    print("\n-- descriptive: selected vs the 5 random draws (lower is better)")
    for met in ("rmse", "mae"):
        m = metrics[metrics.variant.isin(("selected", *DRAWS))]
        if m.empty or "selected" not in set(m.variant):
            continue
        for s in [*sorted(set(m.seed), key=lambda s: (s != 42, s)), "mean"]:
            sub = m if s == "mean" else m[m.seed == s]
            v = sub.groupby("variant")[met].mean()
            draws = v.reindex(DRAWS).dropna()
            if "selected" not in v or draws.empty:
                continue
            uni, strat = v.reindex(UNIFORM).dropna(), v.reindex(STRATIFIED).dropna()
            beats = bool((v["selected"] < draws).all())
            gap = uni.mean() - v["selected"] if len(uni) and len(strat) else np.nan
            share = (uni.mean() - strat.mean()) / gap if abs(gap) > 1e-9 else np.nan
            print(f"{met.upper()} seed {s}: selected {v['selected']:.4f} | draws {draws.min():.4f}.."
                  f"{draws.max():.4f} ({len(draws)}/5) -> beats all {'YES' if beats else 'no'} | "
                  f"uniform mean {uni.mean():.4f}, stratified mean {strat.mean():.4f}; share of the "
                  f"selector's gain over uniform explained by group composition = {share:.2f}")


def _replaceable(res: pd.DataFrame) -> None:
    print("\n-- replaceable groups: drop_g worse than selected (CI > 0) while reselect_drop_g ~ selected "
          "(CI includes 0)")
    for met in ("rmse", "mae"):
        out = []
        for g in GROUPS:
            d, r = _get(res, f"drop_{g}", "selected", met, "42"), _get(res, f"reselect_drop_{g}", "selected", met, "42")
            if d is None:
                continue
            flag = verdict(d) == "worse" and r is not None and verdict(r) == "n.s."
            out.append(f"{g}: drop {verdict(d)}, reselect_drop {verdict(r)}"
                       + (" -> REPLACEABLE" if flag else ""))
        print(f"{met.upper()}: " + " | ".join(out))


def _fmt(row: pd.Series | None) -> str:
    return "n/a" if row is None else f"{row.delta:+.4f} {verdict(row)}"


def _reading(res: pd.DataFrame, metrics: pd.DataFrame, missing: list[str]) -> None:
    if missing:
        print(f"INCOMPLETE ({len(missing)} jobs missing): every statement below is provisional.")
    if res.empty:
        print("no contrasts yet")
        return
    for met in ("rmse", "mae"):
        tag = "primary" if met == "rmse" else "secondary"
        print(f"\n## {met.upper()} ({tag})")
        # selector adds value: selected beats all (pooled CI excl. 0, every seed same direction)
        r = _get(res, "all", "selected", met, "pooled")
        if r is None:
            print("selector adds value: n/a (all or selected missing)")
        else:
            ok = verdict(r) == "worse" and r.seeds_sig_worse == r.n_seeds
            print(f"selector adds value (selected beats all): {'YES' if ok else 'NO'} -- all - selected = "
                  f"{r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], {int(r.seeds_sig_worse)}/{int(r.n_seeds)} "
                  "seeds with CI > 0")
        # selector beats chance
        m = metrics[metrics.variant.isin(("selected", *DRAWS))].groupby("variant")[met].mean()
        draws = m.reindex(DRAWS).dropna()
        r = _get(res, "random_median", "selected", met, "pooled")
        if "selected" in m and len(draws) and r is not None:
            beats_all = bool((m["selected"] < draws).all())
            dm_ok = r.delta > 0 and r.dm_p < 0.05
            print(f"selector beats chance: beats all {len(draws)} draws {'YES' if beats_all else 'NO'} "
                  f"(descriptive); vs median draw ({r.note}) delta {r.delta:+.4f}, DM p = {r.dm_p:.3g} -> "
                  f"{'YES' if beats_all and dm_ok else 'NO'}")
            uni, strat = m.reindex(UNIFORM).dropna(), m.reindex(STRATIFIED).dropna()
            if len(uni) and len(strat):
                gap = uni.mean() - m["selected"]
                share = (uni.mean() - strat.mean()) / gap if gap > 1e-9 else np.nan
                read = ("n/a: uniform draws are not worse than selected" if not np.isfinite(share)
                        else "group composition carries most of the selector's value" if share >= 0.5
                        else "stratified ~ uniform: the value is in the within-group picks" if share <= 0.2
                        else "both composition and within-group picks contribute")
                print(f"  stratified vs uniform: mean {strat.mean():.4f} vs {uni.mean():.4f}; share of the "
                      f"gap to selected closed by stratifying = {share:.2f} -> {read} (descriptive)")
        else:
            print("selector beats chance: n/a (draws or selected missing)")
        # groups
        print("group verdicts (only_g vs core | drop_g vs selected | reselect_drop_g vs selected, seed 42):")
        for g in GROUPS:
            o = _get(res, f"only_{g}", "core", met, "42")
            d = _get(res, f"drop_{g}", "selected", met, "42")
            rd = _get(res, f"reselect_drop_{g}", "selected", met, "42")
            signal = verdict(o) == "better" and verdict(d) == "worse"
            repl = verdict(d) == "worse" and verdict(rd) == "n.s."
            if d is None:
                label = (f"{'helps' if verdict(o) == 'better' else 'no gain'} alone; no leave-one-out"
                         + (" (not in the selected 100)" if g in ("missile", "cyber") else ""))
            else:
                label = ("CARRIES SIGNAL" if signal else "helps alone only" if verdict(o) == "better"
                         else "") + (", REPLACEABLE" if repl else "")
            print(f"  {g:9s} only {_fmt(o):18s} | drop {_fmt(d):18s} | reselect_drop {_fmt(rd):18s} {label}")

    # round 2: cumulative path in order of single-group RMSE gain over core
    print("\n## Round 2: cumulative path, groups ordered by single-group RMSE gain over core (only_g, seed 42)")
    order = []
    for g in GROUPS:
        o = _get(res, f"only_{g}", "core", "rmse", "42")
        if o is not None:
            order.append((o.delta, GROUPS.index(g), g, verdict(o), o.p_holm))
    order.sort()
    order = [(d, g, v, p) for d, _, g, v, p in order]
    if len(order) < len(GROUPS):
        print(f"only {len(order)}/{len(GROUPS)} only_* variants complete: the order below is provisional")
    for rank, (dlt, g, vd, ph) in enumerate(order, 1):
        print(f"  {rank}. {g:9s} delta RMSE vs core {dlt:+.4f} ({vd}, Holm p {ph:.3g})")
    gs = [g for _, g, _, _ in order]
    if len(gs) >= 2:
        print("\nsubmit (cluster login node, ~/reruns/thesis):")
        for k in range(2, min(8, len(gs)) + 1):
            print(f"  bash scripts/slurm/feature_ablation.sh --combo k{k} {','.join(gs[:k])}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "runs_feature_ablation"))
    ap.add_argument("--pub", default=str(ROOT / "runs_publication_20260929"))
    ap.add_argument("--runs", nargs="+", default=["catboost_tweedie@global"],
                    help="model@paradigm items, space- or comma-separated")
    ap.add_argument("--variants-file", default=str(HERE / "variants.tsv"))
    ap.add_argument("--out", default=None,
                    help=f"output folder (default {HERE / 'output'}; --selftest: {SELFTEST_OUT})")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true",
                    help="every variant = the publication store: only seed 42, every delta 0, R1 PASS")
    a = ap.parse_args()
    out = Path(a.out) if a.out else (SELFTEST_OUT if a.selftest else HERE / "output")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "analysis_output.txt")

    variants = read_variants(Path(a.variants_file))
    stores = Stores(Path(a.root), Path(a.pub), a.selftest)
    ref_rmse = reference_rmse(stores.pub)
    print(f"feature ablation analysis: {len(variants)} variants, "
          f"{int(variants.seeds.map(len).sum())} jobs per run, n_boot {a.n_boot}")
    ms, hs, rs = [], [], []
    for model, paradigm in parse_runs(a.runs):
        m, h, r = analyse_run(model, paradigm, stores, variants, a.n_boot, ref_rmse)
        ms.append(m), hs.append(h), rs.append(r)
    metrics = pd.concat(ms, ignore_index=True)
    res = pd.concat(rs, ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.10g")
    pd.concat(hs, ignore_index=True).to_csv(out / "per_horizon.csv", index=False, float_format="%.10g")
    if not res.empty:
        res.insert(res.columns.get_loc("dm_p") + 1, "p_holm", res.pop("p_holm"))
    res.to_csv(out / "results.csv", index=False, float_format="%.10g")
    print(f"\nwrote {out}/results.csv ({len(res)} rows), metrics.csv ({len(metrics)}), "
          "per_horizon.csv, analysis_output.txt")


if __name__ == "__main__":
    main()
