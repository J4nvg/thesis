"""Horizon boxplots with the significant pairs marked, thesis scoring vs same target dates (2026-09-30).

Read-only on the run store; writes only the figures next to this file.

    uv run python docs/audits/2026-09-30/horizon_boxplots.py

Same form as the thesis figure (top-20 configurations, one box per horizon, absolute RMSE / MAE),
plus the paired structure the tests use: each thin grey line is one configuration. Brackets mark
the comparisons of the pre-specified family (each day vs Day 1, each day vs the previous day) that
are significant after Holm (Wilcoxon signed-rank, horizon_pipeline.pairwise); stars give the
Holm-adjusted p, the arrow the direction (the later day has a higher / lower error).

- horizon_boxplots_AvsB: left column scores each horizon on its own 164 target dates (the thesis),
  right column scores every horizon on the same 158 dates. Rows share the y-axis.
- horizon_boxplots_B: the right column alone at thesis print width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import horizon_pipeline as hp  # noqa: E402

from strikecast.reporting.style import save_svg, thesis_style  # noqa: E402

ACCENT = "#2a78d6"      # single series -> categorical slot 1
BOX_FILL = "#d4e4f7"    # light step of the accent hue
CONFIG = "#b9b7ae"      # de-emphasised context lines
INK = "#0b0b0b"
INK_2 = "#52514e"
BASELINE = "#c3c2b7"
GRID = "#e1e0d9"


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*"


def significant_pairs(M: pd.DataFrame) -> pd.DataFrame:
    ph = hp.pairwise(M, "wilcoxon")
    sig = ph[ph.sig].copy()
    sig["a"] = sig.comparison.str.extract(r"h(\d) vs")[0].astype(int)   # later day
    sig["b"] = sig.comparison.str.extract(r"vs h(\d)")[0].astype(int)   # earlier day
    return sig.assign(span=sig.a - sig.b).sort_values(["span", "b"])


def bracket_levels(sig: pd.DataFrame) -> list[int]:
    """Stack brackets so no two on the same level overlap (shared end points count as overlap)."""
    levels: list[list[tuple[int, int]]] = []
    out = []
    for _, s in sig.iterrows():
        lvl = next((i for i, used in enumerate(levels)
                    if all(s.a < lo or s.b > hi for lo, hi in used)), len(levels))
        if lvl == len(levels):
            levels.append([])
        levels[lvl].append((s.b, s.a))
        out.append(lvl)
    return out


def draw_panel(ax, M: pd.DataFrame, metric: str, *, top: float, step: float, xlabel: bool,
               ylabel: bool) -> None:
    x = np.array(hp.H)
    for _, row in M.iterrows():
        ax.plot(x, row.values, color=CONFIG, lw=0.6, alpha=0.7, zorder=1)
    ax.boxplot([M[h].values for h in hp.H], positions=x, widths=0.5, patch_artist=True,
               showfliers=True, zorder=2,
               boxprops=dict(facecolor=BOX_FILL, edgecolor=ACCENT, lw=1, alpha=0.75),
               medianprops=dict(color=ACCENT, lw=2),
               whiskerprops=dict(color=ACCENT, lw=1), capprops=dict(color=ACCENT, lw=1),
               flierprops=dict(marker="o", ms=3.5, mfc="white", mec=ACCENT, mew=0.8))

    sig = significant_pairs(M)
    tick = 0.25 * step
    for (_, s), lvl in zip(sig.iterrows(), bracket_levels(sig), strict=True):
        y = top + step * (lvl + 0.5)
        x0, x1 = s.b + 0.08, s.a - 0.08
        ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=INK_2, lw=0.9, zorder=5,
                solid_capstyle="butt")
        arrow = "↑" if s.mean_diff > 0 else "↓"
        ax.text((x0 + x1) / 2, y + 0.05 * step, f"{arrow}{stars(s.p_holm)}", ha="center",
                va="bottom", fontsize=7.5, color=INK, linespacing=1)

    fr = hp.friedman(M)
    p = fr["p"]
    ax.text(0, 1.015, f"{metric}   Friedman $\\chi^2$(6) = {fr['chi2']:.1f}, "
            f"{'p < 0.001' if p < 0.001 else f'p = {p:.3f}'}, W = {fr['W']:.2f}",
            transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in x])
    ax.set_xlim(0.45, 7.55)
    ax.set_xlabel("Forecast horizon (days ahead)" if xlabel else "", color=INK_2, fontsize=8.5)
    ax.set_ylabel(metric if ylabel else "", color=INK_2, fontsize=8.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8, ls="-")
    ax.grid(False, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=0)


def row_limits(Ms: list[pd.DataFrame]) -> tuple[float, float, float, int]:
    lo = min(M.values.min() for M in Ms)
    hi = max(M.values.max() for M in Ms)
    n_lvl = max((max(bracket_levels(significant_pairs(M)), default=-1) + 1) for M in Ms)
    step = 0.07 * (hi - lo)
    return lo, hi, step, n_lvl


def legend(fig, y: float) -> None:
    handles = [
        plt.Line2D([], [], color=CONFIG, lw=0.8, label="One configuration (top 20)"),
        plt.matplotlib.patches.Patch(facecolor=BOX_FILL, edgecolor=ACCENT, label="Median, IQR, 1.5 IQR"),
        plt.Line2D([], [], color=INK_2, lw=0.9,
                   label="Holm-significant pair: ↑ later day worse, ↓ better;  "
                         "* p<.05  ** p<.01  *** p<.001"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
               handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.5, y))


def figure_AvsB(A: dict[str, pd.DataFrame], B: dict[str, pd.DataFrame], n_common: int):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharey="row")
    for r, m in enumerate(hp.METRICS):
        lo, hi, step, n_lvl = row_limits([A[m], B[m]])
        for c, Ms in enumerate((A, B)):
            draw_panel(axes[r, c], Ms[m], m, top=hi, step=step, xlabel=r == 1, ylabel=c == 0)
        axes[r, 0].set_ylim(lo - 0.04 * (hi - lo), hi + step * (n_lvl + 0.4))
    axes[0, 0].set_title("As in the thesis: each day scored on its own\n164 target dates",
                         fontsize=9, fontweight="bold", color=INK, loc="left", pad=16)
    axes[0, 1].set_title(f"Same {n_common} target dates for every day\n(only the lead time differs)",
                         fontsize=9, fontweight="bold", color=INK, loc="left", pad=16)
    legend(fig, -0.005)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.4, w_pad=1.0)
    return fig


def figure_B(B: dict[str, pd.DataFrame]):
    # thesis text block: A4 with 42 mm margins = 126 mm = 4.96 in, so draw at print size
    fig, axes = plt.subplots(2, 1, figsize=(4.96, 6.2))
    for r, (ax, m) in enumerate(zip(axes, hp.METRICS, strict=True)):
        lo, hi, step, n_lvl = row_limits([B[m]])
        draw_panel(ax, B[m], m, top=hi, step=step, xlabel=r == 1, ylabel=True)
        ax.set_ylim(lo - 0.04 * (hi - lo), hi + step * (n_lvl + 0.4))
    handles = [
        plt.Line2D([], [], color=CONFIG, lw=0.8, label="One configuration (top 20)"),
        plt.Line2D([], [], color=INK_2, lw=0.9,
                   label="Holm-significant: ↑ later day worse, ↓ better;  * p<.05  ** p<.01"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7,
               handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.6)
    return fig


def main() -> None:
    lb = pd.read_csv(hp.RUNS / "_figures/master_leaderboard.csv")
    preds = hp.load_preds(lb.head(20))
    p0 = next(iter(preds.values()))
    common = set.intersection(*[set(p0.loc[p0.horizon == h, "date"]) for h in hp.H])
    A, B = hp.matrices_from_preds(preds), hp.matrices_from_preds(preds, dates=common)
    for tag, Ms in (("A thesis scoring", A), ("B same dates", B)):
        for m in hp.METRICS:
            sig = significant_pairs(Ms[m])
            print(f"{tag:17s} {m:4s} medians " + " ".join(f"{v:.3f}" for v in Ms[m].median())
                  + "  | Holm-significant: " + (", ".join(
                      f"{r.comparison} ({r.median_pct:+.2f}%, p={r.p_holm:.3f})" for _, r in sig.iterrows())
                      or "none"))
    with thesis_style():
        for name, fig in (("horizon_boxplots_AvsB", figure_AvsB(A, B, len(common))),
                          ("horizon_boxplots_B", figure_B(B))):
            fig.savefig(HERE / f"{name}.png", dpi=200, bbox_inches="tight")
            save_svg(fig, HERE / f"{name}.svg")
            plt.close(fig)
            print(f"wrote {name}.png/.svg")


if __name__ == "__main__":
    main()
