"""DATA figures and tables of the thesis, ported from ``eda.ipynb`` / ``eda_full.ipynb``.

Inputs come from :func:`strikecast.data.load_inputs` (the same parquet read the
notebooks were rewired onto). The two oblast maps need ``geopandas`` (the
``figures`` extra, audit C3); :func:`require_geopandas` raises
:class:`~strikecast.reporting.sources.MissingInput` so the builder skips them
with a clear message when it is absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strikecast.reporting.sources import MissingInput

__all__ = [
    "EdaData",
    "acf_pacf_figure",
    "load_eda",
    "marginal_statistics",
    "spatiotemporal_heatmap",
    "stl_side_by_side",
    "strike_activity_maps",
    "target_distribution",
]

TARGET = "act_drone_strike_on_ua"
TIER_COLORS = {1: "#f5c400", 2: "#1a6faf", 3: "#1a8c3a"}


@dataclass
class EdaData:
    master: pd.DataFrame  # sorted by event_date (eda_full cell 1)
    regions: list[str]
    tiers: dict[str, int]
    tlong: pd.DataFrame  # event_date, region, tier, y (modelled regions)
    active_days: dict[str, int]
    geojson: Path


def load_eda(data_dir: str | Path = "data") -> EdaData:
    """``eda_full.ipynb`` cells 1-2 (and ``eda.ipynb`` cell 3)."""
    from strikecast.data import load_inputs  # noqa: PLC0415

    data_dir = Path(data_dir)
    fixed, dataset = data_dir / "fixed", data_dir / "dataset"
    if not (dataset / "master_combined_timeseries.parquet").is_file():
        raise MissingInput(f"missing {dataset / 'master_combined_timeseries.parquet'}")
    inputs = load_inputs(fixed, dataset)
    df = inputs.master.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)
    prefix = f"{TARGET}_"
    regions_all = [c.replace(prefix, "") for c in df.columns if c.startswith(prefix)]
    active = {r: int((df[prefix + r] > 0).sum()) for r in regions_all}
    tiers = {r: int(inputs.activity_by_region.get(r, 0)) for r in regions_all}
    modelled = [r for r, t in tiers.items() if t > 0]
    tlong = df[["event_date"] + [prefix + r for r in modelled]].melt(
        "event_date", var_name="col", value_name="y"
    )
    tlong["region"] = tlong["col"].str.replace(prefix, "", regex=False)
    tlong["tier"] = tlong["region"].map(tiers)
    tlong = (
        tlong[["event_date", "region", "tier", "y"]]
        .sort_values(["region", "event_date"])
        .reset_index(drop=True)
    )
    return EdaData(
        master=df,
        regions=list(inputs.regions),
        tiers=tiers,
        tlong=tlong,
        active_days=active,
        geojson=data_dir / "for_eda" / "Ukraine_Admin_Regions.geojson",
    )


# --------------------------------------------------------------------------- #
# eda_full cell 4 / 5: marginal statistics + fig_eda1_target_distribution (F5)
# --------------------------------------------------------------------------- #
def marginal_statistics(tlong: pd.DataFrame) -> pd.DataFrame:
    """Zero rate and dispersion index ``var/mean`` pooled and per tier (the text's D)."""
    rows = []
    for name, sub in [("pooled", tlong), *[(f"Tier {t}", s) for t, s in tlong.groupby("tier")]]:
        y = sub["y"].to_numpy(dtype=float)
        mu, s2 = y.mean(), y.var(ddof=1)
        rows.append(
            {
                "group": name,
                "n": len(y),
                "zero_rate": float((y == 0).mean()),
                "mean": float(mu),
                "var": float(s2),
                "dispersion_index": float(s2 / max(mu, 1e-12)),
                "max": int(y.max()),
            }
        )
    return pd.DataFrame(rows)


def target_distribution(tlong: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    y = tlong["y"].values
    fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    bins = np.arange(0, min(y.max(), 40) + 2) - 0.5
    axes[0].hist(np.clip(y, 0, 40), bins=bins, edgecolor="white", linewidth=0.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Daily drone-strike events")
    axes[0].set_ylabel("Frequency (log scale)")
    tiers = sorted(tlong["tier"].unique())
    zr = [(tlong[tlong.tier == t].y == 0).mean() for t in tiers]
    dsp = [
        tlong[tlong.tier == t].y.var(ddof=1) / max(tlong[tlong.tier == t].y.mean(), 1e-12)
        for t in tiers
    ]
    x = np.arange(len(tiers))
    w = 0.4
    bars_zr = axes[1].bar(x - w / 2, zr, w, label="Zero rate")
    axes[1].bar_label(bars_zr, fmt="%.2f", padding=3)
    axes[1].set_ylabel("Zero rate")
    axes[1].set_ylim(0, 1.1)
    ax2 = axes[1].twinx()
    bars_dsp = ax2.bar(x + w / 2, dsp, w, color="tomato")
    ax2.bar_label(bars_dsp, fmt="%.2f", padding=3, color="tomato")
    ax2.set_ylabel("Dispersion index", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax2.set_ylim(0, max(dsp) * 1.15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Tier {t}" for t in tiers])
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(linestyle="-", axis="both")
    axes[0].set_title("(a) Daily event-count distribution (national)")
    axes[1].set_title("(b) Zero rate and dispersion index by tier")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# --------------------------------------------------------------------------- #
# eda_full cell 10: fig_eda2_onlypacfacf (F6)
# --------------------------------------------------------------------------- #
def acf_pacf_figure(tlong: pd.DataFrame, max_lag: int = 30) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: PLC0415

    nat = tlong.groupby("event_date")["y"].sum().sort_index().values.astype(float)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=False, sharex=True)
    plot_pacf(nat, lags=max_lag, method="ywm", ax=axes[0], title="(a) PACF — national daily total")
    plot_acf(nat, lags=max_lag, fft=True, ax=axes[1], title="(b) ACF — national daily total")
    axes[0].set_xlabel("Lags")
    axes[0].set_ylabel("PACF")
    axes[1].set_xlabel("Lags")
    axes[1].set_ylabel("ACF")
    axes[0].grid()
    axes[1].grid()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# eda_full cell 14: fig_eda2c_stl_side_by_side (F7)
# --------------------------------------------------------------------------- #
def stl_strengths(series: pd.Series, period: int = 7) -> tuple[Any, float, float]:
    from statsmodels.tsa.seasonal import STL  # noqa: PLC0415

    fit = STL(series, period=period, robust=True).fit()
    v_t, v_s, v_r = (np.var(x, ddof=1) for x in (fit.trend, fit.seasonal, fit.resid))
    return fit, max(0.0, 1 - v_r / (v_r + v_t)), max(0.0, 1 - v_r / (v_r + v_s))


def stl_side_by_side(tlong: pd.DataFrame) -> tuple[Any, dict[str, float]]:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    nat = tlong.groupby("event_date")["y"].sum().sort_index().asfreq("D")
    nat_diff = nat.diff().dropna()
    fit_raw, ft_raw, fs_raw = stl_strengths(nat)
    fit_diff, ft_diff, fs_diff = stl_strengths(nat_diff)
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True,
                             gridspec_kw={"hspace": 0.3, "wspace": 0.15})
    left = [
        (0, nat.values, "(a) Observed national total", "events"),
        (1, fit_raw.trend.values, f"(b) STL trend ($F_T$={ft_raw:.2f})", "trend"),
        (2, fit_raw.seasonal.values, f"(c) STL weekly seasonal ($F_S$={fs_raw:.2f})", "seasonal"),
        (3, fit_raw.resid.values, "(d) STL residual", "residual"),
    ]
    right = [
        (0, nat_diff.values, "(e) Differenced national total", "diff events"),
        (1, fit_diff.trend.values, f"(f) STL trend ($F_T$={ft_diff:.2f})", "trend"),
        (2, fit_diff.seasonal.values, f"(g) STL weekly seasonal ($F_S$={fs_diff:.2f})", "seasonal"),
        (3, fit_diff.resid.values, "(h) STL residual", "residual"),
    ]
    colors = {0: "#222", 1: "tomato", 2: "#4a6fa5", 3: "#777"}
    for col, (index, panels) in enumerate(((nat.index, left), (nat_diff.index, right))):
        for row, vals, title, ylab in panels:
            ax = axes[row, col]
            ax.plot(index, vals, lw=0.7 if row in (0, 2, 3) else 1.6, color=colors[row])
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            if row == 3:
                ax.set_xlabel("Date")
    fig.autofmt_xdate(rotation=45)
    fig.suptitle("STL Decomposition: Raw Target vs. Differenced Target", fontsize=14,
                 fontweight="bold", y=0.95)
    return fig, {"F_T_raw": ft_raw, "F_S_raw": fs_raw, "F_T_diff": ft_diff, "F_S_diff": fs_diff}


# --------------------------------------------------------------------------- #
# eda.ipynb cell 9: spatio-temporal heatmap (F4)
# --------------------------------------------------------------------------- #
def spatiotemporal_heatmap(master: pd.DataFrame, regions: list[str]) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    cols = [f"{TARGET}_{r}" for r in regions if f"{TARGET}_{r}" in master.columns]
    df_heat = master[["event_date", *cols]].copy()
    df_heat["event_date"] = pd.to_datetime(df_heat["event_date"])
    df_heat = df_heat.sort_values("event_date").set_index("event_date")
    df_heat.columns = [c.replace(f"{TARGET}_", "").title() for c in df_heat.columns]
    df_heat_t = df_heat.resample("W").sum().fillna(0).T
    df_heat_t = df_heat_t.loc[df_heat_t.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.cm.viridis.copy()
    cmap.set_under("#000000")
    sns.heatmap(df_heat_t, cmap=cmap, vmin=0.9, robust=True,
                cbar_kws={"label": "Intensity (Weekly Drone Strikes)"}, ax=ax, linewidths=0,
                linecolor=None)
    dates = df_heat_t.columns
    n_cols = len(dates)
    month_starts = [i for i in range(1, n_cols) if dates[i].month != dates[i - 1].month]
    for idx in month_starts:
        ax.axvline(x=idx, color="white", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_xticks(month_starts)
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in month_starts], rotation=45, ha="right")
    n_rows = len(df_heat_t)
    for x_pos, label, x_offset in [
        (int(n_cols * 0.70), "Start Validation Set", 25),
        (int(n_cols * 0.80), "Start Test Set", 2),
    ]:
        ax.axvline(x=x_pos, color="red", linestyle="--", linewidth=2.5, alpha=0.9)
        ax.text(x_pos - x_offset + 0.3, n_rows * -0.02, label, color="red", fontsize=10,
                fontweight="bold", va="top", ha="left",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 1, "pad": 2})
    ax.set_xlabel("Date (Weekly)", fontsize=12)
    ax.set_ylabel("Region", fontsize=12)
    ax.set_title(f"Spatio-Temporal Intensity of {TARGET} (Weekly)", fontsize=16,
                 fontweight="bold", pad=30)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# eda.ipynb cell 6: the two oblast maps (F2, F3) -- geopandas
# --------------------------------------------------------------------------- #
def require_geopandas() -> Any:
    try:
        import geopandas as gpd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise MissingInput(
            "geopandas is not installed (install the `figures` extra: "
            "`uv sync --extra figures`)"
        ) from exc
    return gpd


def strike_activity_maps(data: EdaData) -> tuple[Any, Any]:
    """``(strike_activity_per_region, strike_activity_per_region_activity_level)``."""
    import matplotlib.patches as mpatches  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    gpd = require_geopandas()
    if not data.geojson.is_file():
        raise MissingInput(f"missing {data.geojson}")
    ukr_admin = gpd.read_file(data.geojson).to_crs(epsg=32635)
    totals = pd.DataFrame(
        [
            {"region_lower": r, "Active_Days": data.active_days[r]}
            for r in data.regions
            if r in data.active_days
        ]
    )
    ukr_admin["region_lower"] = ukr_admin["ADM1_NAME"].str.lower()
    map_data = ukr_admin.merge(totals, on="region_lower", how="left")
    map_data["Active_Days"] = map_data["Active_Days"].fillna(0)
    activity = pd.DataFrame(list(data.tiers.items()), columns=["region_lower", "activity_level"])
    map_data = map_data.merge(activity, on="region_lower", how="left")
    map_data["activity_level"] = map_data["activity_level"].fillna(0).astype(int)
    edge = {0: "#222222", **TIER_COLORS}
    labels = {0: "Near-zero", 1: "Low", 2: "Medium", 3: "High"}
    alpha = 0.50
    max_days = map_data["Active_Days"].max()
    title = (
        "Drone Strike Activity per Ukrainian Region  |  28-09-2022 – 21-01-2025\n"
        "Day count with atleast one strike event"
    )

    def add_labels(ax: Any) -> None:
        for _, row in map_data.iterrows():
            pt = row.geometry.representative_point().coords[0]
            v = row["Active_Days"]
            ax.text(*pt, f"{row['ADM1_NAME']}\n{v:,.0f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white" if v >= max_days * 0.6 else "black",
                    clip_on=False)

    def add_margins(ax: Any, pct: float = 0.06) -> None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(xmin - (xmax - xmin) * pct, xmax + (xmax - xmin) * pct)
        ax.set_ylim(ymin - (ymax - ymin) * pct, ymax + (ymax - ymin) * pct)

    fig1, ax1 = plt.subplots(figsize=(15, 9))
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=8)
    map_data.plot(column="Active_Days", cmap="Reds", linewidth=0.6, ax=ax1, edgecolor="0.3",
                  legend=True, legend_kwds={"label": "Unique Days with Drone Strikes",
                                            "orientation": "vertical", "shrink": 0.75,
                                            "pad": 0.02})
    add_labels(ax1)
    add_margins(ax1)
    ax1.axis("off")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(15, 9))
    ax2.set_title(title.replace("2025\n", "2025 \n"), fontsize=13, fontweight="bold", pad=8)
    map_data.plot(column="Active_Days", cmap="Reds", linewidth=0, ax=ax2, edgecolor="none",
                  legend=False)
    for level, color in edge.items():
        subset = map_data[map_data["activity_level"] == level]
        if subset.empty:
            continue
        subset.plot(ax=ax2, facecolor=color, alpha=alpha, edgecolor="none", linewidth=0)
        subset.plot(ax=ax2, facecolor="none", edgecolor=color, linewidth=2.5)
    add_labels(ax2)
    add_margins(ax2)
    ax2.axis("off")
    patches = [
        mpatches.Patch(facecolor=edge[i], alpha=alpha + 0.3, edgecolor=edge[i], linewidth=2,
                       label=f"{i} – {labels[i]}")
        for i in range(4)
    ]
    ax2.legend(handles=patches, title="Activity tier", loc="lower left", fontsize=10,
               title_fontsize=11, framealpha=0.9)
    fig2.tight_layout()
    return fig1, fig2


def tiers_from_json(path: str | Path) -> dict[str, int]:
    """``analyse_results.ipynb`` cell 3 ``load_region_activity_dictionary``."""
    import json  # noqa: PLC0415

    mapping = {"little": 0, "low": 1, "medium": 2, "high": 3}
    payload: Mapping[str, list[str]] = json.loads(Path(path).read_text(encoding="utf-8"))
    return {region: mapping[k] for k, regions in payload.items() for region in regions}
