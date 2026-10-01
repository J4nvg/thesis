"""Figures of the feature-ablation rerun, from the CSVs analysis.py writes (README.md, same folder).

    uv run python docs/feature_ablation/figures.py              # output/ -> figures/
    uv run python docs/feature_ablation/figures.py --selftest   # the analysis --selftest output

Each figure is written as SVG (strikecast.reporting.style.save_svg: byte-stable) and PNG, next to
the CSV it is drawn from. The work is split so reporting/build.py can import it unchanged later
(PROMOTING_FIGURES.md): ``*_frame`` functions turn the analysis CSVs into one tidy frame per
figure, ``plot_*`` functions take that frame and return a matplotlib Figure (no I/O, no rc
mutation; call them inside ``thesis_style()``).

F1 selector value, F2 single groups vs core, F3 leave-one-out vs selected, F4 per-horizon RMSE/MAE,
F5 cumulative path (round 2; skipped until path_* variants exist).

Colours: the dataviz reference palette's first three categorical slots (validated all-pairs for
colour-vision deficiency); aqua sits below 3:1 contrast on white, so every series also differs
by marker shape and carries a text label.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strikecast.reporting.style import save_svg, thesis_style

HERE = Path(__file__).resolve().parent
SELFTEST_DIR = Path(tempfile.gettempdir()) / "feature_ablation_selftest"
log = logging.getLogger("feature_ablation.figures")

GROUPS: tuple[str, ...] = ("strikes", "spatial", "conflict", "comms", "macro", "missile", "cyber",
                           "weather", "calendar")
GROUP_LABELS: dict[str, str] = {
    "strikes": "Autoregressive strikes", "spatial": "Spatial / static", "conflict": "Conflict & damage",
    "comms": "Comms / diplo / aid", "macro": "Macroeconomic", "missile": "Missile / launch",
    "cyber": "Cyber", "weather": "Weather / geomag.", "calendar": "Calendar",
}
FUTURE_GROUPS = ("weather", "calendar")
DRAWS = tuple(f"random_d{d}" for d in range(1, 6))
METRICS = {"rmse": "RMSE", "mae": "MAE"}

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
#: kind -> (colour, marker) for F1; fixed/re-selected for F2/F3; F4 series.
KIND_STYLE = {"selected": (BLUE, "o"), "all": (INK2, "s"), "uniform": (ORANGE, "D"),
              "stratified": (AQUA, "^")}
ABL_STYLE = {"fixed": (BLUE, "o"), "reselected": (ORANGE, "D")}
SERIES_STYLE = {"selected": (BLUE, "o", "-"), "all": (ORANGE, "s", "--"), "core": (AQUA, "^", ":")}


# --------------------------------------------------------------------------- #
# frames: analysis CSVs -> one tidy frame per figure
# --------------------------------------------------------------------------- #
def _seed_order(s: Any) -> tuple[int, int]:
    s = str(s)
    return (0, 0) if s == "42" else (1, int(s)) if s.isdigit() else (2, 0)


def selector_frame(metrics: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """F1: one row per (variant, metric, seed) plus a "mean" row carrying the pooled CI.

    The CI of the "mean" row is the pooled 95% block-bootstrap CI of (variant - selected), drawn
    around the selected seed mean, so a bar that does not cross the reference line is significant.
    """
    names = ["selected", "all", *DRAWS]
    m = metrics[metrics.variant.isin(names)]
    if m.empty or "selected" not in set(m.variant):
        return pd.DataFrame()
    rows = []
    for met in METRICS:
        ref = float(m.loc[m.variant == "selected", met].mean())
        for v in names:
            sub = m[m.variant == v]
            if sub.empty:
                continue
            kind = ("selected" if v == "selected" else "all" if v == "all"
                    else "stratified" if v in DRAWS[3:] else "uniform")
            label = {"selected": "Selected top-100", "all": "All 1785 pairs"}.get(
                v, f"Random draw {v[-1]} ({kind})")
            for r in sub.itertuples():
                rows.append({"variant": v, "label": label, "kind": kind, "metric": met,
                             "seed": str(r.seed), "value": getattr(r, met), "ci_lo": np.nan,
                             "ci_hi": np.nan, "reference": ref})
            ci = results[(results.variant == v) & (results.reference == "selected")
                         & (results.metric == met) & (results.seed == "pooled")] if not results.empty else results
            lo, hi = (ref + float(ci.ci_lo.iloc[0]), ref + float(ci.ci_hi.iloc[0])) if len(ci) else (np.nan, np.nan)
            rows.append({"variant": v, "label": label, "kind": kind, "metric": met, "seed": "mean",
                         "value": float(sub[met].mean()), "ci_lo": lo, "ci_hi": hi, "reference": ref})
    return pd.DataFrame(rows)


def _annotation(metrics: pd.DataFrame, variant: str, minus: str | None = None) -> str:
    """Pair counts from feature_space.json: past pairs, or future components for weather/calendar."""
    sub = metrics[(metrics.variant == variant) & (metrics.seed == 42)]
    if sub.empty or not np.isfinite(sub.n_past_pairs.iloc[0]):
        return ""
    r = sub.iloc[0]
    if minus is not None:
        base = metrics[(metrics.variant == minus) & (metrics.seed == 42)]
        if base.empty:
            return ""
        dp = int(r.n_past_pairs - base.n_past_pairs.iloc[0])
        df = int(r.n_future - base.n_future.iloc[0]) if np.isfinite(r.n_future) else 0
        parts = [f"{dp:+d} pairs" if dp else "", f"{df:+d} future" if df else ""]
        return ", ".join(p for p in parts if p) or "no change"
    parts = [f"{int(r.n_past_pairs)} pairs" if r.n_past_pairs else "",
             f"{int(r.n_future)} future" if np.isfinite(r.n_future) and r.n_future else "",
             "enc." if r.calendar_encoders is True or str(r.calendar_encoders) == "True" else ""]
    return " + ".join(p for p in parts if p) or "no inputs"


def forest_frame(results: pd.DataFrame, metrics: pd.DataFrame, *, section: str) -> pd.DataFrame:
    """F2 (section R3: only_* and reselect_only_* vs core) or F3 (R4: drop_* / reselect_drop_* vs
    selected). One row per (group, flavour, metric), seed 42."""
    if results.empty:
        return pd.DataFrame()
    if section == "R3":
        pairs = {"fixed": ("only_{g}", "core"), "reselected": ("reselect_only_{g}", "core")}
    else:
        pairs = {"fixed": ("drop_{g}", "selected"), "reselected": ("reselect_drop_{g}", "selected")}
    rows = []
    for g in GROUPS:
        for flavour, (pat, ref) in pairs.items():
            v = pat.format(g=g)
            hit = results[(results.section == section) & (results.variant == v)
                          & (results.reference == ref) & (results.seed == "42")]
            for r in hit.itertuples():
                minus = "selected" if (section == "R4" and flavour == "fixed") else None
                rows.append({"group": g, "label": GROUP_LABELS[g], "flavour": flavour, "variant": v,
                             "reference": ref, "metric": r.metric, "delta": r.delta, "ci_lo": r.ci_lo,
                             "ci_hi": r.ci_hi, "p_holm": r.p_holm, "value_reference": r.value_reference,
                             "annotation": _annotation(metrics, v, minus) if not metrics.empty else ""})
    return pd.DataFrame(rows)


def horizon_frame(per_horizon: pd.DataFrame, variants: Sequence[str] = ("selected", "all", "core")) -> pd.DataFrame:
    """F4: seed mean and seed range of the per-horizon metric, long over metric."""
    ph = per_horizon[per_horizon.variant.isin(variants)]
    if ph.empty:
        return pd.DataFrame()
    long = ph.melt(id_vars=["variant", "seed", "horizon"], value_vars=list(METRICS), var_name="metric")
    out = (long.groupby(["variant", "metric", "horizon"]).value
           .agg(mean="mean", lo="min", hi="max", n_seeds="size").reset_index())
    out["order"] = out.variant.map({v: i for i, v in enumerate(variants)})
    return out.sort_values(["order", "metric", "horizon"]).drop(columns="order").reset_index(drop=True)


def path_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """F5: round-2 path_* variants plus core / selected / all as references (seed 42)."""
    m = metrics[metrics.seed == 42]
    path = m[m.variant.str.startswith("path_")].copy()
    if path.empty:
        return pd.DataFrame()
    path["k"] = path.kept_groups.fillna("").map(lambda s: len([g for g in str(s).split(";") if g]))
    path = path.sort_values("k")
    # k = 1 is the best single group of the shortest path: an only_* variant of round 1
    first = [g for g in str(path.kept_groups.iloc[0]).split(";") if g]
    singles = m[m.variant.isin([f"only_{g}" for g in first])].sort_values("rmse")
    if len(singles):
        lead = singles.iloc[[0]].assign(k=1)
        path = pd.concat([lead, path[path.k > 1]], ignore_index=True)
    added, prev = [], set()
    for s in path.kept_groups.fillna(""):
        cur = {g for g in str(s).split(";") if g}
        added.append(",".join(g for g in GROUPS if g in cur - prev))
        prev = cur
    path["added"] = added
    path["role"] = "path"
    refs = m[m.variant.isin(("core", "selected", "all"))].assign(role="reference", added="", k=np.nan)
    cols = ["variant", "role", "k", "added", "kept_groups", "rmse", "mae", "n_past_pairs"]
    return pd.concat([path[cols], refs[cols]], ignore_index=True)


# --------------------------------------------------------------------------- #
# plots: tidy frame in, Figure out
# --------------------------------------------------------------------------- #
def _axes_style(ax: Any) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelcolor=INK)


def plot_selector(frame: pd.DataFrame, *, title: str = "") -> Any:
    """F1: per variant, seed points (small, hollow) and the seed mean (large) with the pooled CI."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    order = list(dict.fromkeys(frame.label))
    y = {lab: i for i, lab in enumerate(order[::-1])}
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(order) + 1.9), sharey=True)
    for ax, met in zip(axes, METRICS, strict=True):
        sub = frame[frame.metric == met]
        _axes_style(ax)
        if sub.empty:
            ax.set_axis_off()
            continue
        ax.axvline(sub.reference.iloc[0], color=BLUE, linewidth=1.2, linestyle="--", zorder=1)
        for r in sub.itertuples():
            colour, marker = KIND_STYLE[r.kind]
            if r.seed == "mean":
                if np.isfinite(r.ci_lo):
                    ax.hlines(y[r.label], r.ci_lo, r.ci_hi, color=colour, linewidth=2, zorder=2)
                ax.plot(r.value, y[r.label], marker=marker, markersize=9, color=colour,
                        markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=4)
            else:
                ax.plot(r.value, y[r.label] + 0.22, marker=marker, markersize=5, linestyle="none",
                        markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.2, zorder=3)
        ax.set_xlabel(f"Test {METRICS[met]} (lower is better)", color=INK)
        ax.set_yticks(list(y.values()), list(y.keys()))
    handles = [Line2D([], [], marker="o", markersize=5, markerfacecolor="none", markeredgecolor=INK2,
                      linestyle="none", label="one seed"),
               Line2D([], [], marker="o", markersize=9, color=INK2, linestyle="-", linewidth=2,
                      label="seed mean, 95% CI of the difference to selected"),
               Line2D([], [], color=BLUE, linestyle="--", label="selected (seed mean)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.55, -0.01))
    fig.suptitle(title or "Selector value: selected top-100 vs all pairs vs random 100", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_forest(frame: pd.DataFrame, *, reference: str, title: str = "") -> Any:
    """F2/F3: delta vs the reference per group, fixed and re-selected side by side, RMSE | MAE.

    Filled marker: 95% CI excludes 0; hollow: it does not. Negative = better than the reference.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    groups = [g for g in GROUPS if g in set(frame.group)]
    y = {g: i for i, g in enumerate(groups[::-1])}
    both = {g for g, s in frame.groupby("group") if s.flavour.nunique() > 1}
    shift = {"fixed": 0.17, "reselected": -0.17}

    def ypos(group: str, flavour: str) -> float:  # one flavour only: on the label line
        return y[group] + (shift[flavour] if group in both else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(groups) + 2.1), sharey=True)
    for ax, met in zip(axes, METRICS, strict=True):
        sub = frame[frame.metric == met]
        _axes_style(ax)
        ax.axvline(0, color=INK2, linewidth=1, zorder=1)
        for r in sub.itertuples():
            colour, marker = ABL_STYLE[r.flavour]
            yy = ypos(r.group, r.flavour)
            sig = np.isfinite(r.ci_lo) and (r.ci_lo > 0 or r.ci_hi < 0)
            ax.hlines(yy, r.ci_lo, r.ci_hi, color=colour, linewidth=2, zorder=2)
            ax.plot(r.delta, yy, marker=marker, markersize=8, linestyle="none", zorder=3,
                    markerfacecolor=colour if sig else "white", markeredgecolor=colour,
                    markeredgewidth=1.5)
        ax.set_xlabel(f"Δ {METRICS[met]} vs {reference}\n(negative = better)", color=INK)
        ax.set_yticks(list(y.values()), [GROUP_LABELS[g] for g in y])
        lo, hi = ax.get_xlim()
        if hi - lo < 1e-9:  # every delta 0 (selftest): give the zero line some room
            ax.set_xlim(-0.01, 0.01)
    # pair counts as a text column right of the MAE panel
    ann = frame[frame.metric == "rmse"]
    for r in ann.itertuples():
        if r.annotation:
            axes[1].annotate(r.annotation, xy=(1.02, ypos(r.group, r.flavour)),
                             xycoords=("axes fraction", "data"), va="center", fontsize=9,
                             color=INK2)
    handles = [Line2D([], [], marker=ABL_STYLE[f][1], color=ABL_STYLE[f][0], markersize=8,
                      linewidth=2, label=lab)
               for f, lab in (("fixed", "fixed (edit the feature set)"),
                              ("reselected", "re-selected (selector re-picks 100)"))
               if f in set(frame.flavour)]
    handles.append(Line2D([], [], marker="o", markersize=8, markerfacecolor="white",
                          markeredgecolor=INK2, linestyle="none", label="hollow: 95% CI includes 0"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, color=INK)
    fig.tight_layout(rect=(0, 0.07, 0.9, 1))
    return fig


def plot_horizon(frame: pd.DataFrame, *, title: str = "") -> Any:
    """F4: per-horizon seed mean (line) and seed range (band), RMSE | MAE, direct labels."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = {"selected": "Selected top-100", "all": "All pairs", "core": "Core only"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, met in zip(axes, METRICS, strict=True):
        sub = frame[frame.metric == met]
        ax.grid(color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for v, s in sub.groupby("variant", sort=False):
            colour, marker, ls = SERIES_STYLE.get(v, (INK2, "o", "-"))
            ax.fill_between(s.horizon, s.lo, s.hi, color=colour, alpha=0.15, linewidth=0)
            ax.plot(s.horizon, s["mean"], color=colour, marker=marker, markersize=6, linewidth=2,
                    linestyle=ls, label=labels.get(v, v))
            ax.annotate(labels.get(v, v), xy=(s.horizon.iloc[-1], s["mean"].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                        color=INK2)
        ax.set_xlabel("Horizon (days ahead)", color=INK)
        ax.set_ylabel(f"Test {METRICS[met]}", color=INK)
        ax.set_xticks(sorted(sub.horizon.unique()))
        ax.set_xlim(right=sub.horizon.max() + 1.6)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title or "Accuracy per horizon (line: seed mean, band: seed range)", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_path(frame: pd.DataFrame, *, title: str = "") -> Any:
    """F5: RMSE | MAE along the cumulative path; core / selected / all as reference lines."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    path = frame[frame.role == "path"]
    refs = frame[frame.role == "reference"].set_index("variant")
    ref_style = {"core": (AQUA, ":"), "selected": (BLUE, "--"), "all": (ORANGE, "-.")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, met in zip(axes, METRICS, strict=True):
        ax.grid(color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.plot(path.k, path[met], color=INK2, marker="o", markersize=7, linewidth=2)
        for r in path.itertuples():
            ax.annotate(f"+{r.added}", xy=(r.k, getattr(r, met)), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none",
                              "alpha": 0.85})
        for v, (colour, ls) in ref_style.items():
            if v in refs.index:
                val = float(refs.loc[v, met])
                ax.axhline(val, color=colour, linestyle=ls, linewidth=1.3)
                ax.annotate(v, xy=(1.0, val), xycoords=("axes fraction", "data"), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.set_xlabel("Groups in the model (cumulative, by single-group gain)", color=INK)
        ax.set_ylabel(f"Test {METRICS[met]}", color=INK)
        ax.set_xticks(sorted(path.k.unique()))
    fig.suptitle(title or "Round 2: cumulative group path (seed 42)", color=INK)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def _save(fig: Any, frame: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{name}.csv", index=False, float_format="%.10g", lineterminator="\n")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    save_svg(fig, out / f"{name}.svg")
    print(f"wrote {name}.svg/.png/.csv")


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        print(f"missing {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"seed": str} if path.name == "results.csv" else None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=None, help=f"analysis output (default {HERE / 'output'})")
    ap.add_argument("--out", default=None, help=f"figure folder (default {HERE / 'figures'})")
    ap.add_argument("--run", default=None, help="model@paradigm (default: the first in results.csv)")
    ap.add_argument("--selftest", action="store_true", help=f"read {SELFTEST_DIR} (analysis --selftest)")
    a = ap.parse_args()
    inp = Path(a.inp) if a.inp else (SELFTEST_DIR if a.selftest else HERE / "output")
    out = Path(a.out) if a.out else (inp / "figures" if a.selftest else HERE / "figures")

    results, metrics = _read(inp / "results.csv"), _read(inp / "metrics.csv")
    per_h = _read(inp / "per_horizon.csv")
    if metrics.empty:
        raise SystemExit(f"no metrics in {inp}: run analysis.py first")
    run = a.run or str(metrics.run.iloc[0])
    metrics = metrics[metrics.run == run]
    results = results[results.run == run] if not results.empty else results
    per_h = per_h[per_h.run == run] if not per_h.empty else per_h
    print(f"figures for {run}: {inp} -> {out}")

    with thesis_style():
        f1 = selector_frame(metrics, results)
        if f1.empty:
            print("skip F1: selected / all / random_* not in metrics.csv")
        else:
            _save(plot_selector(f1), f1, out, "F1_selector_value")
        for tag, sec, ref, name, title in (
            ("F2", "R3", "core", "F2_single_groups",
             "Single groups: core (own lags + statics) plus one group vs core, seed 42"),
            ("F3", "R4", "selected", "F3_leave_one_out",
             "Leave one group out of the selected 100 vs selected, seed 42"),
        ):
            fr = forest_frame(results, metrics, section=sec)
            if fr.empty:
                print(f"skip {tag}: no {sec} contrasts in results.csv")
            else:
                _save(plot_forest(fr, reference=ref, title=title), fr, out, name)
        f4 = horizon_frame(per_h) if not per_h.empty else pd.DataFrame()
        if f4.empty:
            print("skip F4: selected / all / core not in per_horizon.csv")
        else:
            _save(plot_horizon(f4), f4, out, "F4_per_horizon")
        f5 = path_frame(metrics)
        if f5.empty:
            print("skip F5: no path_* variants yet (round 2: see analysis R5 for the --combo commands)")
        else:
            _save(plot_path(f5), f5, out, "F5_cumulative_path")


if __name__ == "__main__":
    main()
