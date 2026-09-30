"""Prototype horizon figure with the significant pairs marked (proposal, 2026-09-30).

Read-only on the run store; writes only the preview figures next to this file.

    uv run python docs/audits/2026-09-30/horizon_figure.py

One panel per metric. Each grey line is one of the top-20 configurations, expressed as the %
change of its error relative to its own Day-1 error (the within-configuration change the paired
tests evaluate); the blue line is the median over configurations. Brackets mark the comparisons
of the pre-specified family (each day vs Day 1, each day vs the previous day) that are
significant after Holm (Wilcoxon signed-rank); the label is the Holm-adjusted p.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import horizon_pipeline as hp  # noqa: E402

from strikecast.reporting.style import save_svg, thesis_style  # noqa: E402

ACCENT = "#2a78d6"      # single series -> categorical slot 1
CONFIG = "#b9b7ae"      # de-emphasised context lines
INK = "#0b0b0b"
INK_2 = "#52514e"
BASELINE = "#c3c2b7"
GRID = "#e1e0d9"


def p_label(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def draw_panel(ax, M: pd.DataFrame, metric: str, *, xlabel: bool = True) -> None:
    rel = 100 * M.div(M[1], axis=0) - 100
    x = hp.H
    for _, row in rel.iterrows():
        ax.plot(x, row.values, color=CONFIG, lw=0.8, alpha=0.8, zorder=1)
    med = rel.median()
    ax.axhline(0, color=BASELINE, lw=1, zorder=0)
    ax.plot(x, med.values, color=ACCENT, lw=2, zorder=3)
    ax.plot(x, med.values, "o", ms=7, color=ACCENT, mec="white", mew=1.5, zorder=4)

    # significant comparisons as brackets above the data
    ph = hp.pairwise(M, "wilcoxon")
    sig = ph[ph.sig].copy()
    sig["a"] = sig.comparison.str.extract(r"h(\d) vs")[0].astype(int)
    sig["b"] = sig.comparison.str.extract(r"vs h(\d)")[0].astype(int)
    sig["lo"], sig["hi"] = sig[["a", "b"]].min(axis=1), sig[["a", "b"]].max(axis=1)
    sig = sig.assign(span=sig.hi - sig.lo).sort_values(["span", "lo"])
    top = rel.values.max()
    span_y = rel.values.max() - rel.values.min()
    step, tick = 0.17 * span_y, 0.03 * span_y
    levels: list[list[tuple[int, int]]] = []
    for _, s in sig.iterrows():
        lvl = next((i for i, used in enumerate(levels)
                    if all(s.hi <= lo or s.lo >= hi for lo, hi in used)), len(levels))
        if lvl == len(levels):
            levels.append([])
        levels[lvl].append((s.lo, s.hi))
        y = top + step * (lvl + 0.6)
        x0, x1 = s.lo + 0.06, s.hi - 0.06
        ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=INK_2, lw=1, zorder=5,
                solid_capstyle="butt")
        ax.text((x0 + x1) / 2, y + 0.4 * tick, p_label(s.p_holm), ha="center", va="bottom",
                fontsize=7.5, color=INK_2)
    ax.set_ylim(rel.values.min() - 0.08 * span_y, top + step * (len(levels) + 0.9))

    fr = hp.friedman(M)
    ax.set_title(f"{metric}", loc="left", fontsize=10, fontweight="bold", color=INK, pad=14)
    ax.text(0, 1.02, f"Friedman $\\chi^2$(6) = {fr['chi2']:.1f}, {p_label(fr['p'])}, "
            f"Kendall's W = {fr['W']:.2f}", transform=ax.transAxes, fontsize=8, color=INK_2,
            va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Day {h}" for h in x])
    ax.set_xlim(0.7, 7.3)
    ax.set_xlabel("Forecast horizon" if xlabel else "", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Change in {metric}\nvs Day 1 (%)", color=INK_2, fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.8, ls="-")
    ax.grid(False, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def figure(Ms: dict[str, pd.DataFrame], suptitle: str | None = None):
    # thesis text block: A4 with 42 mm margins = 126 mm = 4.96 in, so draw at print size
    fig, axes = plt.subplots(2, 1, figsize=(4.96, 6.4))
    for i, (ax, m) in enumerate(zip(axes, hp.METRICS, strict=True)):
        draw_panel(ax, Ms[m], m, xlabel=i == len(axes) - 1)
    handles = [plt.Line2D([], [], color=CONFIG, lw=0.8, label="One configuration (top 20)"),
               plt.Line2D([], [], color=ACCENT, lw=2, marker="o", ms=7, mec="white", mew=1.5,
                          label="Median"),
               plt.Line2D([], [], color=INK_2, lw=1, label="Significant, Holm-adjusted")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.5, -0.005))
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97 if suptitle else 1), h_pad=1.6)
    return fig


def main() -> None:
    lb = pd.read_csv(hp.RUNS / "_figures/master_leaderboard.csv")
    top20 = lb.head(20)
    preds = hp.load_preds(top20)
    p0 = next(iter(preds.values()))
    common = set.intersection(*[set(p0.loc[p0.horizon == h, "date"]) for h in hp.H])
    designs = {
        "A": ("Design A (current): each horizon scored on its own 164 target dates", hp.matrices_from_preds(preds)),
        "B": (f"Design B: every horizon scored on the same {len(common)} target dates",
              hp.matrices_from_preds(preds, dates=common)),
    }
    for tag, (title, Ms) in designs.items():
        with thesis_style():
            fig = figure(Ms, suptitle=title)
            fig.savefig(HERE / f"horizon_sig_design{tag}.png", dpi=200, bbox_inches="tight")
            save_svg(fig, HERE / f"horizon_sig_design{tag}.svg")
        print(f"wrote horizon_sig_design{tag}.png/.svg")




# --------------------------------------------------------------------------- family version
FAMILY_STYLE = {  # categorical slots 1-2 (validated pair)
    "other": ("#2a78d6", "Other models"),
    "GBDT": ("#eb6834", "Gradient-boosted trees"),
}


def family_panel(ax, M: pd.DataFrame, fam: dict[str, str], metric: str, *, xlabel: bool) -> None:
    rel_all = 100 * M.div(M[1], axis=0) - 100
    lo_all, hi_all = rel_all.values.min(), rel_all.values.max()
    span_y = hi_all - lo_all
    ax.axhline(0, color=BASELINE, lw=1, zorder=0)
    stats_txt = []
    n_brackets = 0
    for g, (color, name) in FAMILY_STYLE.items():
        S = M[[fam[i] == g for i in M.index]]
        rel = rel_all.loc[S.index]
        for _, row in rel.iterrows():
            ax.plot(hp.H, row.values, color=color, lw=0.7, alpha=0.28, zorder=1)
        med = rel.median()
        ax.plot(hp.H, med.values, color=color, lw=2, zorder=3)
        fr = hp.friedman(S)
        stats_txt.append(f"{name.lower()} {p_label(fr['p'])}")
        # post-hoc only where the family's Friedman test rejects
        ph = hp.pairwise(S, "wilcoxon") if fr["p"] < hp.ALPHA else None
        for h in hp.H[1:]:
            sig = ph is not None and bool(ph.loc[ph.comparison == f"h{h} vs h1", "sig"].iloc[0])
            ax.plot(h, med[h], "o", ms=6.5, mec=color, mew=1.6, zorder=4,
                    mfc=color if sig else "white")
        if ph is not None:  # adjacent-day steps: brackets below the data, family colour, staggered
            for _, s in ph[ph.sig & ~ph.comparison.str.endswith("vs h1")].iterrows():
                a, b = (int(c) for c in s.comparison.replace("h", "").split(" vs "))
                y = lo_all - (0.08 + 0.15 * (n_brackets % 2)) * span_y
                tick = 0.03 * span_y
                x0, x1 = min(a, b) + 0.06, max(a, b) - 0.06
                ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color=color, lw=1, zorder=5)
                ax.text((x0 + x1) / 2, y - 0.4 * tick, p_label(s.p_holm), ha="center", va="top",
                        fontsize=7, color=INK_2)
                n_brackets += 1
        ax.text(7.3, med[7], f"{name}\n(n = {len(S)})", fontsize=7.5, color=INK, va="center",
                ha="left", linespacing=1.1, clip_on=False)
    levels = min(n_brackets, 2)
    ax.set_ylim(lo_all - (0.08 + 0.15 * levels) * span_y, hi_all + 0.08 * span_y)
    ax.set_title(metric, loc="left", fontsize=10, fontweight="bold", color=INK, pad=14)
    ax.text(0, 1.02, "Friedman: " + "; ".join(stats_txt), transform=ax.transAxes, fontsize=8,
            color=INK_2, va="bottom")
    ax.set_xticks(hp.H)
    ax.set_xticklabels([f"Day {h}" for h in hp.H])
    ax.set_xlim(0.7, 7.3)
    ax.set_xlabel("Forecast horizon" if xlabel else "", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Change in {metric}\nvs Day 1 (%)", color=INK_2, fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.8, ls="-")
    ax.grid(False, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def family_figure(Ms: dict[str, pd.DataFrame], fam: dict[str, str]):
    fig, axes = plt.subplots(2, 1, figsize=(4.96, 6.6))
    for i, (ax, m) in enumerate(zip(axes, hp.METRICS, strict=True)):
        family_panel(ax, Ms[m], fam, m, xlabel=i == 1)
    handles = [
        plt.Line2D([], [], color=INK_2, marker="o", ls="", ms=6.5, mfc=INK_2, mec=INK_2,
                   label="Differs from Day 1"),
        plt.Line2D([], [], color=INK_2, marker="o", ls="", ms=6.5, mfc="white", mec=INK_2, mew=1.6,
                   label="Does not differ"),
        plt.Line2D([], [], color=INK_2, lw=1, label="Differs from previous day"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.6)
    return fig


def main_family() -> None:
    lb = pd.read_csv(hp.RUNS / "_figures/master_leaderboard.csv")
    top20 = lb.head(10)
    preds = hp.load_preds(top20)
    p0 = next(iter(preds.values()))
    common = set.intersection(*[set(p0.loc[p0.horizon == h, "date"]) for h in hp.H])
    Ms = hp.matrices_from_preds(preds, dates=common)
    fam = {hp.label(r): ("GBDT" if r.Modelname == "gbdt" else "other") for _, r in top20.iterrows()}
    with thesis_style():
        fig = family_figure(Ms, fam)
        fig.savefig(HERE / "horizon_family_designB.png", dpi=200, bbox_inches="tight")
        save_svg(fig, HERE / "horizon_family_designB.svg")
    print("wrote horizon_family_designB.png/.svg")


if __name__ == "__main__":
    main()
    main_family()
