"""``results/analyse_results.ipynb`` computations as pure, tested functions.

Nothing here reads files: every function takes frames (from a
:class:`~strikecast.reporting.sources.ResultsSource`) and returns frames or
plain dicts. Cell numbers refer to ``analyse_results.ipynb`` (0-indexed).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "TABLE4_SPEC",
    "HorizonStats",
    "Table4Row",
    "activity_tier_table",
    "baseline_rows",
    "base_variables",
    "calibration_curves",
    "hurdle_bias_table",
    "horizon_statistics",
    "master_leaderboard",
    "model_key",
    "point_metrics",
    "prauc_vs_prevalence",
    "pretty_model",
    "reference_rmse",
    "region_matrix",
    "split_dimensions",
    "table4",
    "top_models_table",
]

ALPHA = 0.05


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
def model_key(row: Mapping[str, Any]) -> str:
    """``<model>_<paradigm>``: the notebook's display name (cells 12, 18, 26)."""
    return f"{row['model']}_{row['paradigm']}"


_FAMILY_WORDS = {
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU",
    "arima": "ARIMA",
    "linear": "Linear",
}


def pretty_model(model: str) -> str:
    """``catboost_tweedie_tuned`` -> ``CatBoost (Tweedie)``; ``lstm_poisson_w28`` ->
    ``LSTM (Poisson, $w=28$)``; Chronos -> ``Chronos-2 (Fine-tuned)``."""
    name = model.removesuffix("_tuned")
    if name.startswith("chronos2"):
        return "Chronos-2 (Fine-tuned)" if "fine" in name else "Chronos-2 (Zero-shot)"
    if name in ("naive_weekly",):
        return "Seasonal Naive"
    if name in ("naive_last",):
        return "Naive"
    if name in ("finalhurdle", "hurdle"):
        return "Hurdle"
    parts = name.split("_")
    head = _FAMILY_WORDS.get(parts[0], parts[0])
    details = []
    for p in parts[1:]:
        if p.startswith("w") and p[1:].isdigit():
            details.append(f"$w={p[1:]}$")
        else:
            details.append(p.capitalize())
    return f"{head} ({', '.join(details)})" if details else head


# --------------------------------------------------------------------------- #
# cells 1, 4: the master leaderboard and the cross-experiment SkillScore (B16)
# --------------------------------------------------------------------------- #
def reference_rmse(leaderboards: pd.DataFrame) -> float:
    """RMSE of ``diff`` ``naive_weekly`` (global, test): cell 1's ``naive_weekly_rmse_mean``."""
    ref = leaderboards[
        (leaderboards["Modelname"] == "diff") & (leaderboards["model"] == "naive_weekly")
    ]
    if ref.empty:
        from strikecast.reporting.sources import MissingInput  # noqa: PLC0415

        raise MissingInput("diff naive_weekly (test) is required for the SkillScore (B16)")
    return float(ref["RMSE"].iloc[0])


def master_leaderboard(leaderboards: pd.DataFrame, ref_rmse: float | None = None) -> pd.DataFrame:
    """Cell 4: ``SkillScore = 1 - RMSE / RMSE(diff naive_weekly)``, sorted descending.

    Columns ``Modelname, paradigm, model, SkillScore, mae, rmse``; the sort is
    the notebook's (pandas default quicksort on ``SkillScore``).
    """
    ref = reference_rmse(leaderboards) if ref_rmse is None else float(ref_rmse)
    df = leaderboards.copy()
    df["SkillScore"] = 1 - (df["RMSE"] / ref)
    df = df[["Modelname", "paradigm", "model", "SkillScore", "MAE", "RMSE"]].rename(
        columns={"MAE": "mae", "RMSE": "rmse"}
    )
    return df.sort_values(by="SkillScore", ascending=False).reset_index(drop=True)


def baseline_rows(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cells 5, 11, 12: diff ``naive_weekly`` (paradigm relabelled ``local``) and diff ARIMA."""
    naive = master[
        (master["Modelname"] == "diff")
        & master["model"].str.contains("naive_weekly", case=False, na=False)
    ].copy()
    naive["paradigm"] = "local"
    arima = master[
        (master["Modelname"] == "diff") & master["model"].str.contains("arima", case=False, na=False)
    ]
    return naive, arima


def top_with_baselines(master: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Cell 12's ``top_5_w_naive``: the top-k, then seasonal naive, then ARIMA."""
    naive, arima = baseline_rows(master)
    return pd.concat([master.head(k), naive, arima])


# --------------------------------------------------------------------------- #
# cell 7: Table 4 recomputed from the stored predictions (D6)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Table4Row:
    label: str
    paradigm_label: str
    configuration: str
    family: str
    paradigm: str
    model: str  # legacy spelling


#: Cell 7 ``TABLE4_SPEC`` (the hurdle is ``hurdle_cal``, D6).
TABLE4_SPEC: tuple[Table4Row, ...] = (
    Table4Row("CatBoost", "Activity", "Tweedie", "gbdt", "activity", "catboost_tweedie_tuned"),
    Table4Row("LightGBM", "Global", "Poisson", "gbdt", "global", "lightgbm_poisson_tuned"),
    Table4Row("XGBoost", "Global", "Tweedie", "gbdt", "global", "xgboost_tweedie_tuned"),
    Table4Row("LSTM", "Activity", "Poisson, $w=28$", "lstm", "activity", "lstm_poisson_w28_tuned"),
    Table4Row("GRU", "Activity", "Poisson, $w=28$", "lstm", "activity", "gru_poisson_w28_tuned"),
    Table4Row("Chronos-2-FT", "Local", "Fine-tuned", "chronos2", "local", "chronos2_fine_tuned"),
    Table4Row("Chronos-2-OS", "Local", "Zero-shot", "chronos2", "local", "chronos2_zero_shot"),
    Table4Row("ARIMA", "Local", "$p=7, d=1, q=1$", "diff", "global", "arima"),
    Table4Row("Hurdle", "Global", "SPE + CatBoost (Tweedie)", "finalhurdle", "global", "finalhurdle"),
    Table4Row("Seasonal Naive", "Local", "$m=7$", "diff", "global", "naive_weekly"),
    Table4Row("Naive", "Local", "-", "diff", "global", "naive_last"),
)
TABLE4_REFERENCE = "Seasonal Naive"


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Cell 7 ``point_metrics`` (MAPE over non-zero days, WAPE over all)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_pred - y_true
    nz = y_true != 0
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MAPE (%)": float(100 * np.mean(np.abs(e[nz]) / np.abs(y_true[nz]))),
        "WAPE (%)": float(100 * np.abs(e).sum() / np.abs(y_true).sum()),
    }


def table4(predictions: Mapping[str, pd.DataFrame], spec: Sequence[Table4Row] = TABLE4_SPEC) -> pd.DataFrame:
    """Cell 7 ``recompute_table``: one row per spec entry present in ``predictions``
    (keyed by ``label``), Skill Score against the Seasonal Naive row."""
    rows = []
    for entry in spec:
        if entry.label not in predictions:
            continue
        d = predictions[entry.label]
        rows.append(
            {
                "Model": entry.label,
                "Paradigm": entry.paradigm_label,
                "Configuration": entry.configuration,
                **point_metrics(d["y_true"].to_numpy(), d["y_pred"].to_numpy()),
            }
        )
    t = pd.DataFrame(rows)
    if t.empty or TABLE4_REFERENCE not in set(t["Model"]):
        from strikecast.reporting.sources import MissingInput  # noqa: PLC0415

        raise MissingInput("Table 4 needs the Seasonal Naive predictions (diff naive_weekly)")
    ref = t.loc[t["Model"] == TABLE4_REFERENCE, "RMSE"].item()
    t["Skill Score"] = 1 - t["RMSE"] / ref
    return t


def top_models_table(master: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """``tab:top_models``: ``master_df.head(k)`` with display names."""
    top = master.head(k).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Rank": np.arange(1, len(top) + 1),
            "Paradigm": top["paradigm"].str.capitalize(),
            "Model": top["model"].map(pretty_model),
            "SkillScore": top["SkillScore"],
            "MAE": top["mae"],
            "RMSE": top["rmse"],
        }
    )


# --------------------------------------------------------------------------- #
# cells 19, 21: per-region matrices
# --------------------------------------------------------------------------- #
def region_matrix(per_region: Mapping[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """Cell 19 ``build_region_matrix``: region x model (columns keep insertion order)."""
    return pd.DataFrame({name: df.set_index("region")[metric] for name, df in per_region.items()})


# --------------------------------------------------------------------------- #
# cell 27: horizon statistics (B17)
# --------------------------------------------------------------------------- #
@dataclass
class HorizonStats:
    metric: str
    n_models: int
    shapiro_p: dict[int, float]
    shapiro_reported_p: float
    normality_passed: bool
    levene_p: float
    variance_passed: bool
    test: str
    anova_F: float | None = None
    anova_p: float | None = None
    kruskal_H: float | None = None
    kruskal_p: float | None = None
    tukey: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k not in ("tukey", "shapiro_p")}
        out["shapiro_p"] = {str(k): v for k, v in self.shapiro_p.items()}
        return out


def horizon_statistics(per_horizon: Sequence[pd.DataFrame], metric: str) -> HorizonStats:
    """Cell 27 ``test_horizon_significance`` (without the plot).

    Shapiro-Wilk per horizon (the loop ``break``s at the first p < 0.05; the
    notebook prints that p, or the LAST horizon's when all pass -- kept as
    ``shapiro_reported_p``, the thesis' "p ~ 0.41"), Levene, then one-way ANOVA
    + Tukey HSD when both pass, else Kruskal-Wallis. Kruskal-Wallis and ANOVA
    are BOTH computed here (the thesis quotes the KW H for MAE and the ANOVA F
    for RMSE); ``test`` names the one the notebook's routing picks.
    """
    from scipy import stats  # noqa: PLC0415

    combined = pd.concat([df[["horizon", metric]] for df in per_horizon], ignore_index=True)
    horizons = sorted(combined["horizon"].unique())
    groups = [combined[combined["horizon"] == h][metric].dropna() for h in horizons]

    shapiro: dict[int, float] = {}
    normality_passed = True
    reported = float("nan")
    for h, group in zip(horizons, groups, strict=True):
        if len(group) >= 3:
            p = float(stats.shapiro(group)[1])
            shapiro[int(h)] = p
            reported = p
            if p < ALPHA:
                normality_passed = False
                break
        else:
            normality_passed = False
    levene_p = float(stats.levene(*groups)[1])
    variance_passed = levene_p >= ALPHA
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    result = HorizonStats(
        metric=metric,
        n_models=len(per_horizon),
        shapiro_p=shapiro,
        shapiro_reported_p=reported,
        normality_passed=normality_passed,
        levene_p=levene_p,
        variance_passed=variance_passed,
        test="anova" if (normality_passed and variance_passed) else "kruskal",
        anova_F=float(f_stat),
        anova_p=float(f_p),
        kruskal_H=float(h_stat),
        kruskal_p=float(h_p),
    )
    if result.test == "anova":
        from statsmodels.stats.multicomp import pairwise_tukeyhsd  # noqa: PLC0415

        tukey = pairwise_tukeyhsd(endog=combined[metric], groups=combined["horizon"], alpha=ALPHA)
        table = tukey.summary().data
        result.tukey = pd.DataFrame(table[1:], columns=table[0])
    return result


# --------------------------------------------------------------------------- #
# cells 34, 36, 38: hurdle diagnostics
# --------------------------------------------------------------------------- #
def calibration_curves(probs: pd.DataFrame, n_bins: int = 10) -> dict[str, Any]:
    """Cell 34: reliability curves, PR curves, PR-AUC and prevalence."""
    from sklearn.calibration import calibration_curve  # noqa: PLC0415
    from sklearn.metrics import auc, precision_recall_curve  # noqa: PLC0415

    y_true = probs["y_true"].astype(int)
    out: dict[str, Any] = {"prevalence": float(y_true.mean())}
    for tag, col in (("uncal", "y_prob_uncal"), ("cal", "y_prob_cal")):
        prob_true, prob_pred = calibration_curve(y_true, probs[col], n_bins=n_bins)
        prec, rec, _ = precision_recall_curve(y_true, probs[col])
        out[tag] = {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "precision": prec,
            "recall": rec,
            "pr_auc": float(auc(rec, prec)),
        }
    return out


def prauc_vs_prevalence(probs: pd.DataFrame, col: str = "y_prob_cal") -> pd.DataFrame:
    """Cell 36 ``analyze_skill_vs_prevalence`` on the calibrated test probabilities."""
    from sklearn.metrics import average_precision_score  # noqa: PLC0415

    records = []
    for region, group in probs.groupby("region"):
        y_true = group["y_true"]
        n_days, n_events = len(y_true), y_true.sum()
        if 0 < n_events < n_days:
            pr_auc = average_precision_score(y_true, group[col])
            prevalence = n_events / n_days
            records.append(
                {
                    "Region": region,
                    "Prevalence": float(prevalence),
                    "PR-AUC": float(pr_auc),
                    "Delta": float(pr_auc - prevalence),
                }
            )
    return pd.DataFrame(records)


def hurdle_bias_table(regressor_preds: pd.DataFrame) -> pd.DataFrame:
    """Cell 38 ``analyze_regressor_performance`` on the positive-event rows."""
    df = regressor_preds[regressor_preds["y_true"] > 0]
    records = []
    for region, group in df.groupby("region"):
        y_true, y_pred = group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float)
        e = y_pred - y_true
        records.append(
            {
                "Region": region,
                "Count (Days > 0)": len(y_true),
                "Max Actual Strikes": float(y_true.max()),
                "RMSE": float(np.sqrt(np.mean(e**2))),
                "MAE": float(np.mean(np.abs(e))),
                "Bias (Mean Error)": float(np.mean(e)),
            }
        )
    return pd.DataFrame(records).sort_values("RMSE", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# DATA tables: tiers, split, base variables
# --------------------------------------------------------------------------- #
TIER_THRESHOLDS: tuple[int, int, int] = (7, 84, 219)  # eda_full.ipynb cell 2 `tier_of`


def active_days(master: pd.DataFrame, regions: Sequence[str], target: str) -> dict[str, int]:
    """Unique days with at least one event per region (``eda.ipynb`` cell 6)."""
    return {
        r: int((master[f"{target}_{r}"] > 0).sum())
        for r in regions
        if f"{target}_{r}" in master.columns
    }


def activity_tier_table(days: Mapping[str, int], tiers: Mapping[str, int]) -> pd.DataFrame:
    """``tab:activitytiers``: per tier the thresholds and the observed range of n_days."""
    t1, t2, t3 = TIER_THRESHOLDS
    bounds = {0: (None, t1), 1: (t1, t2), 2: (t2, t3), 3: (t3, None)}
    rows = []
    for tier in (0, 1, 2, 3):
        members = sorted(r for r in days if tiers.get(r, 0) == tier)
        values = [days[r] for r in members]
        lo, hi = bounds[tier]
        rows.append(
            {
                "Tier": tier,
                "lower": lo,
                "upper": hi,
                "n_regions": len(members),
                "min_days": min(values) if values else None,
                "max_days": max(values) if values else None,
                "regions": ", ".join(members),
            }
        )
    return pd.DataFrame(rows)


def split_dimensions(n_days: int, n_regions: int) -> pd.DataFrame:
    """``tab:splitdimensions`` from the darts split every family uses (C11).

    ``split_series_list``: ``split_after(0.7)`` then ``split_after(1/3)`` of the rest.
    """
    from darts import TimeSeries  # noqa: PLC0415

    from strikecast.data.series import split_series_list  # noqa: PLC0415

    ts = TimeSeries.from_times_and_values(
        pd.date_range("2000-01-01", periods=n_days, freq="D"), np.zeros(n_days)
    )
    train, val, test = (x[0] for x in split_series_list([ts]))
    days = [len(train), len(val), len(test)]
    return pd.DataFrame(
        {
            "Split": ["Training", "Validation", "Test", "Total"],
            "Days / region": [*days, sum(days)],
            "Observations": [d * n_regions for d in days] + [sum(days) * n_regions],
        }
    )


BASE_EXCLUDE = ("region", "event_date", "Activity_Level")


def base_variables(panel_columns: Sequence[str], target: str) -> list[str]:
    """``tab:base_variables``: the panel columns minus the keys, target and tier (112)."""
    return [c for c in panel_columns if c not in (*BASE_EXCLUDE, target)]
