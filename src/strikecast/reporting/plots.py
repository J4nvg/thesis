"""RESULTS figures of the thesis, ported from ``results/analyse_results.ipynb``.

Each function takes already-computed frames and returns a matplotlib
``Figure``; the caller draws it inside :func:`~strikecast.reporting.style.thesis_style`
and writes it with :func:`~strikecast.reporting.style.save_svg`. The global
``sns.set_theme`` / ``rcParams`` mutations of the notebook are replaced by that
scoped style (audit C5); layout, sizes and colours follow the cells.
"""

from __future__ import annotations

import itertools
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from strikecast.reporting.compute import model_key

__all__ = [
    "LEVEL_LABELS",
    "NAIVE_COLOR",
    "calibration_prcurve",
    "cd_label",
    "ShareFacet",
    "feature_importance_grid",
    "importance_share_heatmap",
    "model_colors",
    "per_region_grid",
    "prauc_prevalence_scatter",
    "top20_horizon_boxplot",
    "top5_horizon_lines",
]

#: Cell 12 / 18.
NAIVE_COLOR = (50 / 255, 50 / 255, 50 / 255, 0.5)
LEVEL_LABELS = {1: "Low activity", 2: "Medium activity", 3: "High activity"}


def model_colors(rows: pd.DataFrame) -> dict[str, tuple[float, ...]]:
    """Cell 12 ``MODEL_COLORS``: ``tab20`` cycled over ``<model>_<paradigm>``, naive grey."""
    import seaborn as sns  # noqa: PLC0415

    pal = sns.color_palette("tab20")
    cycle = itertools.cycle(pal)
    colors = {model_key(r): (*next(cycle), 1) for _, r in rows.iterrows()}
    colors["naive_weekly_local"] = NAIVE_COLOR
    return colors


# --------------------------------------------------------------------------- #
# cell 21: per-region bars by activity tier (F12, F13)
# --------------------------------------------------------------------------- #
def per_region_grid(
    mat: pd.DataFrame,
    metric: str,
    tiers_by_region: Mapping[str, int],
    colors: Mapping[str, Any],
    *,
    tiers: Sequence[int] = (3, 2, 1),
    exclude: Sequence[str] = ("arima_global",),
) -> Any:
    import matplotlib.patches as mpatches  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    models = [c for c in mat.columns if c not in exclude]
    n = len(models)
    width = 0.7 / n
    handles = [mpatches.Patch(facecolor=colors[m], label=m) for m in models]
    fig, axes = plt.subplots(len(tiers), 1, figsize=(15, 16), squeeze=False)
    sub_mat = mat[models]
    for ri, tier in enumerate(tiers):
        ax = axes[ri][0]
        regions = [r for r in sub_mat.index if tiers_by_region.get(r, 0) == tier]
        regions = sorted(regions, key=lambda r: sub_mat.loc[r].mean(), reverse=True)
        for gi, region in enumerate(regions):
            vals = sub_mat.loc[region, models].sort_values()
            for k, mdl in enumerate(vals.index):
                x = gi + (k - (n - 1) / 2) * width
                ax.bar(x, vals[mdl], width=width * 0.9, color=colors[mdl], edgecolor="none")
                ax.annotate(
                    f"{vals[mdl]:.2f}", (x, vals[mdl]), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom", rotation=90,
                    fontsize=11, fontweight="bold",
                )
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=45, ha="right", fontweight="bold")
        ax.margins(y=0.24)
        ax.set_title(f"{LEVEL_LABELS.get(tier, tier)} — {metric}", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=10)
    fig.legend(handles=handles, title="Model", loc="upper center", ncol=n,
               bbox_to_anchor=(0.5, 1.0), fontsize=10, title_fontsize=11)
    fig.suptitle(f"Per-region {metric} by activity tier", fontsize=17, y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# --------------------------------------------------------------------------- #
# cell 26: top-5 metrics over horizons (F18)
# --------------------------------------------------------------------------- #
def top5_horizon_lines(
    per_horizon: Mapping[str, pd.DataFrame],
    colors: Mapping[str, Any],
    metrics: Sequence[str] = ("RMSE", "MAE"),
) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    df = None
    for name, df in per_horizon.items():
        for ax, metric in zip(axes, metrics, strict=True):
            sns.lineplot(data=df, x="horizon", y=metric, label=name, marker="o",
                         color=colors[name], ax=ax)
    for ax, metric in zip(axes, metrics, strict=True):
        ax.set_title(f"{metric}", fontsize=14)
        ax.set_xlabel("Horizon", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        if df is not None:
            ax.set_xticks(df["horizon"].unique())
        ax.grid(True, linestyle="--", alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    fig.suptitle("Comparing Model Performance over Horizons", fontsize=18, y=0.96)
    fig.legend(handles, labels, title="Models", bbox_to_anchor=(0.5, 0.91), loc="upper center",
               ncol=3, fontsize="small", title_fontsize="small")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.12, hspace=0.3)
    return fig


# --------------------------------------------------------------------------- #
# cell 27: top-20 metric distribution by horizon (F19)
# --------------------------------------------------------------------------- #
def top20_horizon_boxplot(per_horizon: Sequence[pd.DataFrame], metric: str = "RMSE") -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    combined = pd.concat([df[["horizon", metric]] for df in per_horizon], ignore_index=True)
    fig = plt.figure(figsize=(8, 11))
    ax1 = fig.add_subplot(2, 1, 1)
    sns.boxplot(data=combined, x="horizon", y=metric, hue="horizon", palette="tab20",
                legend=False, ax=ax1)
    ax1.set_title(f"{metric} Boxplot by Horizon", fontsize=14)
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel(metric)
    ax2 = fig.add_subplot(2, 1, 2)
    sns.histplot(data=combined, x=metric, hue="horizon", kde=True, element="step", alpha=0.3,
                 palette="tab20", ax=ax2)
    ax2.set_title(f"{metric} Distribution by Horizon", fontsize=14)
    ax2.set_xlabel(metric)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# cells 34, 36: hurdle classifier (F16, F17)
# --------------------------------------------------------------------------- #
def calibration_prcurve(curves: Mapping[str, Any]) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    u, c = curves["uncal"], curves["cal"]
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    ax1.plot(u["prob_pred"], u["prob_true"], marker="o", label="Uncalibrated (SPE)")
    ax1.plot(c["prob_pred"], c["prob_true"], marker="s", label="Beta Calibrated")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2.plot(u["recall"], u["precision"], label=f"Uncalibrated (AUC = {u['pr_auc']:.3f})")
    ax2.plot(c["recall"], c["precision"], label=f"Beta Calibrated (AUC = {c['pr_auc']:.3f})")
    base = curves["prevalence"]
    ax2.axhline(base, linestyle="--", color="gray", label=f"Baseline (Prevalence = {base:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def prauc_prevalence_scatter(metrics_df: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=metrics_df, x="Prevalence", y="PR-AUC", hue="Delta", palette="viridis",
                    s=150, edgecolor="black", ax=ax)
    ax.plot([0, 1], [0, 1], color="red", linestyle="--",
            label="No-Skill Baseline (PR-AUC = Prevalence)")
    for _, row in metrics_df.iterrows():
        ax.text(row["Prevalence"] + 0.015, row["PR-AUC"], row["Region"], fontsize=9,
                fontweight="bold", alpha=0.8)
    ax.set_title("Model PR-AUC vs. Regional Prevalence", fontsize=14, pad=15)
    ax.set_xlabel("Baseline Prevalence (Fraction of days with Events)", fontsize=12)
    ax.set_ylabel("Model PR-AUC Score", fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# cells 49-52: feature importance (F20, F21, F23)
# --------------------------------------------------------------------------- #
def feature_importance_grid(
    importancedict: Mapping[str, Mapping[str, pd.DataFrame]], title: str, top_n: int = 15
) -> Any:
    """Cell 49 ``plot_feature_importances``: rows = groups, columns = metrics."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    row_keys = list(importancedict)
    col_keys: list[str] = []
    for inner in importancedict.values():
        for k in inner:
            if k not in col_keys:
                col_keys.append(k)
    nrows, ncols = len(row_keys), len(col_keys)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 8 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=22, y=1.02)
    for i, r_key in enumerate(row_keys):
        for j, c_key in enumerate(col_keys):
            ax = axes[i, j]
            if c_key not in importancedict[r_key]:
                ax.axis("off")
                continue
            df = importancedict[r_key][c_key].head(top_n)
            feature_col, value_col = df.columns[0], df.columns[1]
            sns.barplot(data=df, x=value_col, y=feature_col, ax=ax)
            ax.set_title(f"{r_key} - {c_key}", fontsize=16)
            ax.set_xlabel(f"Importance ({value_col})", fontsize=12)
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=11)
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class ShareFacet:
    """One heatmap panel of :func:`importance_share_heatmap`.

    ``columns`` are columns of the share matrix, drawn left to right under the
    x tick labels ``ticks``; ``group`` is the model name written once above
    all consecutive facets that share it (the tiers of an Activity model).
    """

    title: str
    columns: tuple[str, ...]
    ticks: tuple[str, ...]
    group: str = ""


def importance_share_heatmap(
    mat: pd.DataFrame, facets: Sequence[ShareFacet], *, footnote: str = ""
) -> Any:
    """Category shares of the two leading models in one heatmap (F20).

    Rows = categories (:data:`~strikecast.evaluation.importance.CATEGORY_ORDER`),
    one facet per model (or per tier of an Activity model), one column per
    importance metric; a single colour scale and colour bar for the whole
    figure so shares compare across models. Built on cell 52's per-tier facets.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    from strikecast.evaluation.importance import CATEGORY_ORDER  # noqa: PLC0415

    vmax = float(mat.to_numpy().max())
    rows = [c for c in CATEGORY_ORDER if c in mat.index][::-1]
    widths = [len(f.columns) for f in facets]
    width = max(9.0, 3.2 + 1.3 * sum(widths))
    left = 2.3 / width  # room for the category labels, fixed in inches
    grouped = any(len(list(m)) > 1 for g, m in itertools.groupby(facets, key=lambda f: f.group) if g)
    fig = plt.figure(figsize=(width, 6.8))
    grid = fig.add_gridspec(1, len(facets), width_ratios=widths, left=left, right=0.88,
                            top=0.78 if grouped else 0.84, bottom=0.14 if footnote else 0.08,
                            wspace=0.08)
    cbar_ax = fig.add_axes([0.9, 0.18, 0.015, 0.56])
    axes = []
    for i, facet in enumerate(facets):
        ax = fig.add_subplot(grid[0, i])
        sub = mat[list(facet.columns)].reindex(rows)
        sub.columns = list(facet.ticks)
        annot = sub.map(lambda v: f"{v:.0%}" if v > 0 else "")
        last = i == len(facets) - 1
        sns.heatmap(sub, ax=ax, cmap="Blues", vmin=0, vmax=vmax, annot=annot, fmt="",
                    linewidths=0.5, linecolor="white", cbar=last,
                    cbar_ax=cbar_ax if last else None, yticklabels=(i == 0),
                    cbar_kws={"label": "share within column (top 15)"} if last else None)
        ax.set(xlabel="", ylabel="")
        ax.set_title(textwrap.fill(facet.title, width=max(14, 12 * len(facet.columns))),
                     fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=0)
        axes.append(ax)
    fig.canvas.draw()  # positions are final only after layout
    for group, members in itertools.groupby(enumerate(facets), key=lambda t: t[1].group):
        idx = [i for i, _ in members]
        if not group or len(idx) < 2:  # a lone facet carries the name as its title
            continue
        left, right = axes[idx[0]].get_position().x0, axes[idx[-1]].get_position().x1
        fig.text((left + right) / 2, 0.875, group, ha="center", va="bottom", fontsize=13,
                 fontweight="bold")
    fig.suptitle("Feature-importance share by category  (top 15 features per column)",
                 fontsize=14, y=0.97)
    if footnote:
        fig.text(left, 0.02, textwrap.fill(footnote, width=int(13 * (0.88 - left) * width)),
                 ha="left", va="bottom", fontsize=9, style="italic")
    return fig


def cd_label(row: Mapping[str, Any]) -> str:
    """Cell 29 ``SIG_SPEC`` labels: ``CatBoost (Tw, Act)``, ``Chronos-2 (FT)``, ``ARIMA``."""
    name = str(row["model"]).removesuffix("_tuned")
    if name.startswith("chronos2"):
        return "Chronos-2 (FT)" if "fine" in name else "Chronos-2 (ZS)"
    if name == "arima":
        return "ARIMA"
    if name == "naive_weekly":
        return "Seasonal Naive"
    if name == "naive_last":
        return "Naive"
    if name in ("finalhurdle", "hurdle"):
        return "Hurdle"
    parts = name.split("_")
    from strikecast.reporting.compute import _FAMILY_WORDS  # noqa: PLC0415

    head = _FAMILY_WORDS.get(parts[0], parts[0])
    bits = [p[:2].capitalize() for p in parts[1:] if not (p.startswith("w") and p[1:].isdigit())]
    bits.append({"global": "Glb", "activity": "Act", "local": "Loc"}.get(
        str(row["paradigm"]), str(row["paradigm"])[:3].capitalize()))
    return f"{head} ({', '.join(bits)})"

