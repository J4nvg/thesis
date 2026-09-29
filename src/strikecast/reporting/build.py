"""The item registry, the runner and the manifest of ``strikecast figures``.

One :class:`Item` per figure/table of audit C's inventory (Part 1). Each
DATA/RESULTS item has a builder that writes its outputs into the output folder;
STATIC items (hand-drawn diagrams, typed tables) are listed in the manifest
only. A builder that lacks an input raises
:class:`~strikecast.reporting.sources.MissingInput`; the item is recorded as
``skipped`` with that reason and the run continues (a sparse store never
crashes the command).

Tables are written twice: ``<name>.csv`` (full precision) and ``<name>.tex``,
the ``tabular`` environment only (caption and label stay in ``main.tex``,
which ``\\input``s the fragment; D10).
"""

from __future__ import annotations

import functools
import json
import logging
import subprocess
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strikecast.reporting import compute as C
from strikecast.reporting import eda as E
from strikecast.reporting import plots as P
from strikecast.reporting.sources import MissingInput, ResultsSource
from strikecast.reporting.style import save_svg, thesis_style

__all__ = ["ITEMS", "BuildContext", "Item", "build_all", "write_manifest"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# context: lazily computed shared inputs
# --------------------------------------------------------------------------- #
class BuildContext:
    def __init__(
        self,
        source: ResultsSource,
        out_dir: str | Path,
        *,
        data_dir: str | Path = "data",
    ) -> None:
        self.source = source
        self.out = Path(out_dir)
        self.data_dir = Path(data_dir)
        self.numbers: dict[str, Any] = {}
        self.notes: list[str] = []
        self._friedman_cache: dict[str, Any] | None = None

    @functools.cached_property
    def master(self) -> pd.DataFrame:
        return C.master_leaderboard(self.source.leaderboards())

    @functools.cached_property
    def tiers(self) -> dict[str, int]:
        path = self.data_dir / "fixed" / "regions_activity_cat.json"
        if not path.is_file():
            raise MissingInput(f"missing {path}")
        return E.tiers_from_json(path)

    @functools.cached_property
    def eda(self) -> E.EdaData:
        return E.load_eda(self.data_dir)

    def top(self, k: int) -> pd.DataFrame:
        if len(self.master) < k:
            raise MissingInput(f"the master leaderboard has {len(self.master)} rows, need {k}")
        return self.master.head(k)

    @functools.cached_property
    def top5_w_naive(self) -> pd.DataFrame:
        naive, arima = C.baseline_rows(self.master)
        if naive.empty:
            raise MissingInput("diff naive_weekly is not in the leaderboard")
        if arima.empty:
            raise MissingInput("diff arima is not in the leaderboard")
        rows = pd.concat([self.top(5), naive, arima])
        return rows[~rows.apply(C.model_key, axis=1).duplicated()]

    def per(self, kind: str, rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
        get = self.source.per_region if kind == "region" else self.source.per_horizon
        return {C.model_key(r): get(r["Modelname"], r["paradigm"], r["model"]) for _, r in rows.iterrows()}

    # writers
    def figure(self, name: str, fig: Any, *, tight: bool = True) -> Path:
        """``tight=False`` where the notebook cell called a plain ``savefig(path)``."""
        return save_svg(fig, self.out / name, bbox_inches="tight" if tight else None)

    def table(self, name: str, frame: pd.DataFrame, tex: str) -> list[Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        csv = self.out / f"{name}.csv"
        frame.to_csv(csv, index=False, float_format="%.10g", lineterminator="\n")
        tex_path = self.out / f"{name}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        return [csv, tex_path]


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #
def _esc(text: Any) -> str:
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def _tabular(spec: str, header: str, rows: Sequence[str], footer: Sequence[str] = ()) -> str:
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [r + r" \\" for r in rows]
    if footer:
        lines.append(r"\midrule")
        lines += [r + r" \\" for r in footer]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# builders: tables
# --------------------------------------------------------------------------- #
TIER_DESCRIPTIONS = {
    0: "Regions in which drone-strike incidence is near-zero or exactly zero over the "
    r"observation window; \textit{excluded} from experimentation.",
    1: "Regions with a sparse but non-trivial strike history, typically distant from the "
    "front line.",
    2: "Regions with regular but moderate exposure, including major urban centres and "
    "southern coastal oblasts.",
    3: "Front-line and adjacent regions subjected to sustained, high-intensity strike activity.",
}


def build_activity_tiers(ctx: BuildContext) -> list[Path]:
    data = ctx.eda
    table = C.activity_tier_table(data.active_days, data.tiers)
    lows = {int(r.Tier): r.min_days for r in table.itertuples()}
    t3_max = int(table.loc[table["Tier"] == 3, "max_days"].iloc[0])
    n = r"n_{\text{days}}"
    ranges = {
        0: rf"${n} < {lows[1]}$",
        1: rf"${lows[1]} \leq {n} < {lows[2]}$",
        2: rf"${lows[2]} \leq {n} < {lows[3]}$",
        3: rf"${lows[3]} \leq {n} \leq {t3_max}$",
    }
    rows = [f"Tier {t} & {ranges[t]} & {TIER_DESCRIPTIONS[t]}" for t in range(4)]
    lines = [r"\begin{tabularx}{\textwidth}{l c Y}", r"\toprule",
             r"\textbf{Tier} & \boldmath$n_{\text{days}}$ & \textbf{Description} \\", r"\midrule"]
    lines += [r + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabularx}", ""]
    ctx.numbers.update({"TierOneMin": lows[1], "TierTwoMin": lows[2], "TierThreeMin": lows[3],
                        "TierThreeMax": t3_max})
    return ctx.table("tab_activitytiers", table, "\n".join(lines))


def build_split_dimensions(ctx: BuildContext) -> list[Path]:
    data = ctx.eda
    n_days = int(data.master["event_date"].nunique())
    n_regions = int(data.tlong["region"].nunique())
    t = C.split_dimensions(n_days, n_regions)
    rows = [f"{s} & {d} & {_thousands(o)}" for s, d, o in t.iloc[:3].itertuples(index=False)]
    tot = t.iloc[3]
    footer = [rf"\textbf{{Total}} & \textbf{{{tot['Days / region']}}} & "
              rf"\textbf{{{_thousands(tot['Observations'])}}}"]
    tex = _tabular("lrr", r"\textbf{Split} & \textbf{Days / region} & \textbf{Observations}",
                   rows, footer)
    ctx.numbers.update({"NDays": n_days, "NRegions": n_regions,
                        "NObservations": int(tot["Observations"])})
    return ctx.table("tab_splitdimensions", t, tex)


def build_base_variables(ctx: BuildContext) -> list[Path]:
    from strikecast.data import build_panel_legacy_regressor, load_inputs  # noqa: PLC0415

    ctx.eda  # noqa: B018 - raises MissingInput when data/ is absent
    inputs = load_inputs(ctx.data_dir / "fixed", ctx.data_dir / "dataset")
    panel = build_panel_legacy_regressor(inputs, target_col=E.TARGET).panel
    names = C.base_variables(list(panel.columns), E.TARGET)
    half = (len(names) + 1) // 2
    left, right = names[:half], names[half:] + [""] * (2 * half - len(names))
    rows = [f"{_esc(a)} & {_esc(b)} \\\\" for a, b in zip(left, right, strict=True)]
    ctx.numbers["NBaseVariables"] = len(names)
    # longtable rows only: the longtable environment, caption and label stay in main.tex
    return ctx.table("tab_base_variables_rows", pd.DataFrame({"Variable": names}),
                     "\n".join(rows) + "\n")


def _table4_predictions(ctx: BuildContext) -> tuple[dict[str, pd.DataFrame], list[str]]:
    preds, missing = {}, []
    for entry in C.TABLE4_SPEC:
        model = _native_model(ctx.source, entry.model)
        try:
            preds[entry.label] = ctx.source.predictions(entry.family, entry.paradigm, model)
        except MissingInput as exc:
            missing.append(f"{entry.label}: {exc}")
    return preds, missing


#: Store model names that genuinely end in ``_tuned`` (not the legacy "was tuned" suffix).
_STORE_NAMES_ENDING_TUNED = frozenset({"chronos2_fine_tuned"})


def _native_model(source: ResultsSource, legacy_model: str) -> str:
    if source.name == "store":
        if legacy_model == "finalhurdle":
            return "hurdle"
        if legacy_model in _STORE_NAMES_ENDING_TUNED:
            return legacy_model
        return legacy_model.removesuffix("_tuned")
    return legacy_model


def build_overall_performance(ctx: BuildContext) -> list[Path]:
    preds, missing = _table4_predictions(ctx)
    t = C.table4(preds)
    rows = []
    for r in t.itertuples(index=False):
        skill = r[-1]
        skill_s = f"${skill:.2f}$" if skill < 0 else f"{skill:.2f}"
        rows.append(f"{r.Model} & {r.Paradigm} & {r.Configuration} & {r.MAE:.2f} & "
                    f"{r.RMSE:.2f} & {skill_s}")
    header = (r"\textbf{Model} & \textbf{Paradigm} & \textbf{Configuration} & \textbf{MAE} & "
              r"\textbf{RMSE} & \makecell{\textbf{Skill}\\\textbf{Score}}")
    ctx.notes.extend(f"row missing: {m}" for m in missing)
    return ctx.table("tab_overall_performance", t, _tabular("lllccc", header, rows))


def build_top_models(ctx: BuildContext) -> list[Path]:
    t = C.top_models_table(ctx.master, 5)
    if len(t) < 5:
        ctx.notes.append(f"only {len(t)} models in the leaderboard")
    rows = [f"{r.Rank} & {r.Paradigm} & {r.Model} & {r.SkillScore:.3f} & {r.MAE:.2f} & "
            f"{r.RMSE:.2f}" for r in t.itertuples(index=False)]
    header = (r"\textbf{Rank} & \textbf{Paradigm} & \textbf{Model} & "
              r"\makecell{\textbf{Skill}\\\textbf{Score}} & \textbf{MAE} & \textbf{RMSE}")
    return ctx.table("tab_top_models", t, _tabular("lllcrr", header, rows))


def build_top20(ctx: BuildContext) -> list[Path]:
    """Appendix Top-20 = ``master_df[:20]`` including ``diff/global/linear`` (D7, C9)."""
    t = ctx.master.head(20).copy()
    rows = [f"{_esc(r.Modelname)} & {_esc(r.paradigm)} & {_esc(r.model)} & {r.SkillScore:.6f} & "
            f"{r.mae:.6f} & {r.rmse:.6f}" for r in t.itertuples(index=False)]
    header = (r"\textbf{Modelname} & \textbf{paradigm} & \textbf{model} & \textbf{SkillScore} & "
              r"\textbf{mae} & \textbf{rmse}")
    return ctx.table("tab_top20", t, _tabular("lllccc", header, rows))


def build_master(ctx: BuildContext) -> list[Path]:
    """The full cross-experiment leaderboard (B16), CSV + a longtable body."""
    t = ctx.master
    ctx.numbers["ReferenceRMSE"] = C.reference_rmse(ctx.source.leaderboards())
    rows = [f"{_esc(r.Modelname)} & {_esc(r.paradigm)} & {_esc(r.model)} & {r.SkillScore:.4f} & "
            f"{r.mae:.4f} & {r.rmse:.4f}" for r in t.itertuples(index=False)]
    header = (r"\textbf{Modelname} & \textbf{paradigm} & \textbf{model} & \textbf{SkillScore} & "
              r"\textbf{mae} & \textbf{rmse}")
    return ctx.table("master_leaderboard", t, _tabular("lllccc", header, rows))


def build_hurdle_bias(ctx: BuildContext, n_rows: int = 8) -> list[Path]:
    full = C.hurdle_bias_table(ctx.source.hurdle_regressor_predictions())
    rows = [f"{str(r[0]).title()} & {r[1]} & {r[2]:.1f} & {r[3]:.2f} & {r[4]:.2f} & "
            f"{'$' + format(r[5], '.2f') + '$' if r[5] < 0 else format(r[5], '.2f')}"
            for r in full.head(n_rows).itertuples(index=False)]
    header = (r"\textbf{Region} & \makecell[r]{\textbf{Count}\\\textbf{(Days $>0$)}} & "
              r"\makecell[r]{\textbf{Max Actual}\\\textbf{Strikes}} & \textbf{RMSE} & "
              r"\textbf{MAE} & \makecell[r]{\textbf{Bias}\\\textbf{(Mean Error)}}")
    return ctx.table("tab_hurdle_bias", full, _tabular("lrrrrr", header, rows))


TUNING_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("catboost_tweedie", "CatBoost-Tweedie",
     ("depth", "learning_rate", "iterations", "l2_leaf_reg", "subsample", "tweedie_variance_power")),
    ("lightgbm_poisson", "LightGBM-Poisson",
     ("num_leaves", "max_depth", "min_child_samples", "learning_rate", "n_estimators",
      "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")),
    ("lstm_poisson_w28", r"LSTM-Poisson ($w{=}28$)",
     ("hidden_dim", "n_rnn_layers", "hidden_fc_sizes", "dropout", "batch_size", "lr",
      "weight_decay")),
    ("chronos2_fine_tuned", "Chronos-2 (fine-tuned)", ("fine_tune_lr", "fine_tune_steps")),
)


def _fmt_param(name: str, v: Any) -> str:
    """The appendix formats: ints plain, 4 significant digits, sci for < 1e-2."""
    if name == "hidden_fc_sizes":
        s = str(v).strip("[]() ").replace(" ", "")
        return f"$[{s}]$" if s and s.lower() != "none" else r"$\text{none}$"
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer() and
                                            name not in ("l2_leaf_reg",)):
        return str(int(v))
    v = float(v)
    if abs(v) < 1e-2:
        mant, exp = f"{v:.3e}".split("e")
        if name == "fine_tune_lr":
            mant = f"{float(mant):.2f}"
        return rf"${mant} \times 10^{{{int(exp)}}}$"
    if name == "l2_leaf_reg":
        return f"{v:.1f}"
    # appendix: 4 significant digits below 0.1 (0.01207), 3 decimals above (0.998, 1.395)
    return f"{float(f'{v:.4g}'):g}" if abs(v) < 0.1 else f"{v:.3f}"


def build_tuning_best(ctx: BuildContext) -> list[Path]:
    records, rows = [], []
    for key, label, params in TUNING_ROWS:
        try:
            best = ctx.source.best_params(key)
        except MissingInput as exc:
            ctx.notes.append(f"{label}: {exc}")
            continue
        present = [p for p in params if p in best]
        for i, p in enumerate(present):
            first = rf"\multirow{{{len(present)}}}{{*}}{{{label}}}" if i == 0 else ""
            rows.append(rf"{first}  & \texttt{{{_esc(p)}}} & {_fmt_param(p, best[p])}")
            records.append({"model": key, "hyperparameter": p, "value": best[p]})
        rows.append(r"\midrule")
    if not records:
        raise MissingInput("no best_params for any T8 model")
    rows = rows[:-1]
    body = [r"\begin{tabular}{@{}lll@{}}", r"\toprule",
            r"\textbf{Model} & \textbf{Hyperparameter} & \textbf{Value} \\", r"\midrule"]
    body += [r if r == r"\midrule" else r + r" \\" for r in rows]
    body += [r"\bottomrule", r"\end{tabular}", ""]
    return ctx.table("tab_tuning_best_all", pd.DataFrame(records), "\n".join(body))


def build_friedman(ctx: BuildContext) -> list[Path]:
    res = _friedman(ctx)
    table = pd.DataFrame({f"{m} avg rank": r.avg_rank for m, r in res.items()}).sort_values(
        "RMSE avg rank"
    )
    rows = [f"{_esc(i)} & {v.iloc[0]:.2f} & {v.iloc[1]:.2f}" for i, v in table.iterrows()]
    ctx.numbers.update({
        "FriedmanCD": res["RMSE"].CD, "FriedmanRMSEp": res["RMSE"].p, "FriedmanMAEp": res["MAE"].p,
    })
    out = table.reset_index(names="model")
    for m, r in res.items():
        out[f"{m} p"] = r.p
        out[f"{m} CD"] = r.CD
    return ctx.table("tab_friedman_top5", out,
                     _tabular("lcc", r"\textbf{Model} & \textbf{RMSE avg rank} & "
                              r"\textbf{MAE avg rank}", rows))


def build_horizon_stats(ctx: BuildContext) -> list[Path]:
    top20 = ctx.top(20)
    per = list(ctx.per("horizon", top20).values())
    records, tukeys = [], []
    for metric in ("RMSE", "MAE"):
        h = C.horizon_statistics(per, metric)
        records.append(h.summary())
        if not h.tukey.empty:
            tk = h.tukey.copy()
            tk.insert(0, "metric", metric)
            tukeys.append(tk)
        key = metric.capitalize()
        ctx.numbers.update({
            f"Horizon{key}ShapiroP": h.shapiro_reported_p, f"Horizon{key}LeveneP": h.levene_p,
            f"Horizon{key}AnovaF": h.anova_F, f"Horizon{key}AnovaP": h.anova_p,
            f"Horizon{key}KruskalH": h.kruskal_H, f"Horizon{key}KruskalP": h.kruskal_p,
        })
    summary = pd.DataFrame(records)
    paths = []
    rows = [f"{r['metric']} & {r['shapiro_reported_p']:.3g} & {r['levene_p']:.3g} & "
            f"{r['anova_F']:.2f} & {r['anova_p']:.3g} & {r['kruskal_H']:.2f} & "
            f"{r['kruskal_p']:.3g} & {r['test']}" for r in records]
    header = (r"\textbf{Metric} & \textbf{Shapiro $p$} & \textbf{Levene $p$} & \textbf{$F$} & "
              r"\textbf{ANOVA $p$} & \textbf{$H$} & \textbf{KW $p$} & \textbf{Test}")
    paths += ctx.table("tab_horizon_stats", summary, _tabular("lrrrrrrl", header, rows))
    if tukeys:
        tk = pd.concat(tukeys, ignore_index=True)
        tk_rows = [f"{r['group1']} & {r['group2']} & {float(r['meandiff']):.4f} & "
                   f"{float(r['p-adj']):.4f}" for _, r in tk.iterrows()]
        paths += ctx.table("tab_horizon_tukey", tk,
                           _tabular("rrrr", r"\textbf{Day} & \textbf{Day} & "
                                    r"\textbf{Mean diff.} & \textbf{$p_\text{adj}$}", tk_rows))
    return paths


def build_prauc_numbers(ctx: BuildContext) -> list[Path]:
    probs = ctx.source.hurdle_classifier_probs()
    cc = C.calibration_curves(probs)
    reg = C.prauc_vs_prevalence(probs)
    ctx.numbers.update({"PrevalenceTest": cc["prevalence"], "PRAUCUncal": cc["uncal"]["pr_auc"],
                        "PRAUCCal": cc["cal"]["pr_auc"]})
    rows = [f"{_esc(r['Region'])} & {r['Prevalence']:.3f} & {r['PR-AUC']:.3f} & {r['Delta']:.3f}"
            for _, r in reg.sort_values("Delta", ascending=False).iterrows()]
    return ctx.table("tab_prauc_per_region", reg,
                     _tabular("lrrr", r"\textbf{Region} & \textbf{Prevalence} & \textbf{PR-AUC} & "
                              r"\textbf{$\Delta$}", rows))


def build_eda_stats(ctx: BuildContext) -> list[Path]:
    st = E.marginal_statistics(ctx.eda.tlong)
    for r in st.itertuples(index=False):
        tag = r.group.replace(" ", "").replace("Tier", "Tier").replace("pooled", "Pooled")
        ctx.numbers[f"Dispersion{tag}"] = r.dispersion_index
        ctx.numbers[f"ZeroRate{tag}"] = r.zero_rate
    rows = [f"{r.group} & {r.n} & {r.zero_rate:.3f} & {r.mean:.3f} & {r.dispersion_index:.3f} & "
            f"{r.max}" for r in st.itertuples(index=False)]
    return ctx.table("tab_eda_marginals", st,
                     _tabular("lrrrrr", r"\textbf{Group} & $n$ & \textbf{Zero rate} & "
                              r"\textbf{Mean} & \textbf{$D$} & \textbf{Max}", rows))


# --------------------------------------------------------------------------- #
# builders: figures
# --------------------------------------------------------------------------- #
def build_region_grids(ctx: BuildContext) -> list[Path]:
    rows = ctx.top5_w_naive
    colors = P.model_colors(rows)
    per = ctx.per("region", rows)
    paths = []
    for metric in ("RMSE", "MAE"):
        mat = C.region_matrix(per, metric)
        with thesis_style():
            fig = P.per_region_grid(mat, metric, ctx.tiers, colors)
            paths.append(ctx.figure(f"per_region_grid_by_tier_{metric}.svg", fig))
        mat.reset_index(names="region").to_csv(ctx.out / f"per_region_grid_by_tier_{metric}.csv",
                                               index=False, lineterminator="\n")
        paths.append(ctx.out / f"per_region_grid_by_tier_{metric}.csv")
    return paths


def _friedman(ctx: BuildContext) -> dict[str, Any]:
    if ctx._friedman_cache is not None:
        return ctx._friedman_cache
    from strikecast.evaluation.stats import build_block_matrices, friedman_nemenyi  # noqa: PLC0415

    _, arima = C.baseline_rows(ctx.master)
    if arima.empty:
        raise MissingInput("diff arima is not in the leaderboard")
    rows = pd.concat([ctx.top(5), arima]).drop_duplicates(["Modelname", "paradigm", "model"])
    frames = {P.cd_label(r): ctx.source.predictions(r["Modelname"], r["paradigm"], r["model"])
              for _, r in rows.iterrows()}
    m_rmse, m_mae = build_block_matrices(frames)
    res = {"RMSE": friedman_nemenyi(m_rmse), "MAE": friedman_nemenyi(m_mae)}
    ctx._friedman_cache = res
    return res


def build_cd(ctx: BuildContext) -> list[Path]:
    from strikecast.evaluation.stats import critical_difference_diagram  # noqa: PLC0415

    res = _friedman(ctx)
    paths = []
    for metric, name in (("RMSE", "cd_rmse_top5.svg"), ("MAE", "cd_mae_top5.svg")):
        with thesis_style(None):
            ax = critical_difference_diagram(res[metric],
                                             title=f"Critical-difference diagram — {metric}")
            fig = ax.get_figure()
            fig.tight_layout()
            paths.append(ctx.figure(name, fig))
    return paths


def build_calibration(ctx: BuildContext) -> list[Path]:
    curves = C.calibration_curves(ctx.source.hurdle_classifier_probs())
    with thesis_style():
        return [ctx.figure("calibration_prcurve.svg", P.calibration_prcurve(curves), tight=False)]


def build_prauc_prevalence(ctx: BuildContext) -> list[Path]:
    reg = C.prauc_vs_prevalence(ctx.source.hurdle_classifier_probs())
    with thesis_style():
        return [ctx.figure("prauc_vs_prevalence.svg", P.prauc_prevalence_scatter(reg), tight=False)]


def build_top5_horizon(ctx: BuildContext) -> list[Path]:
    rows = ctx.top(5)
    colors = P.model_colors(ctx.top5_w_naive)
    with thesis_style():
        fig = P.top5_horizon_lines(ctx.per("horizon", rows), colors)
        return [ctx.figure("top_5_combined.svg", fig)]


def build_top20_boxplot(ctx: BuildContext) -> list[Path]:
    per = list(ctx.per("horizon", ctx.top(20)).values())
    with thesis_style():
        return [ctx.figure("top20_rmse_horizon.svg", P.top20_horizon_boxplot(per, "RMSE"), tight=False)]


# --------------------------------------------------------------------------- #
# feature importance (F20, F21, F23): the two leading models of the leaderboard
# --------------------------------------------------------------------------- #
#: The tabular pool of the importance figures (Jan, 2026-09-29): the count GBDTs
#: under the Global or Activity paradigm. Local (20 regional models) is left
#: out: its importance is costly to compute and has no single summary.
FI_GBDT_FAMILIES: tuple[str, ...] = ("lightgbm", "xgboost", "catboost")
FI_GBDT_PARADIGMS: tuple[str, ...] = ("global", "activity")
_FI_TIERS: tuple[int, ...] = (1, 2, 3)
_FI_FOOTNOTE = ("Chronos-2: permutation importance of its covariates only; its own strike "
                "history is the model's context window and is not ranked.")


@dataclass(frozen=True)
class FIModel:
    """One model of the importance figures, as it appears in the master leaderboard."""

    family: str  # leaderboard ``Modelname`` (``gbdt`` / ``chronos2``)
    paradigm: str  # leaderboard paradigm (Chronos-2 rows say ``local``)
    model: str  # store name, without the legacy ``_tuned`` suffix
    rank: int  # 1-based master-leaderboard rank

    @property
    def key(self) -> str:
        return C.model_key({"model": self.model, "paradigm": self.paradigm})

    @property
    def title(self) -> str:
        name = C.pretty_model(self.model)
        return name if self.family == "chronos2" else f"{name}, {self.paradigm.capitalize()}"


def _plain_model(model: str) -> str:
    return model if model in _STORE_NAMES_ENDING_TUNED else model.removesuffix("_tuned")


def _fi_models(ctx: BuildContext) -> tuple[FIModel, FIModel]:
    """The best count GBDT (Global/Activity) and the best Chronos-2 variant, by SkillScore."""
    master = ctx.master.reset_index(drop=True)
    plain = master["model"].map(_plain_model)
    gbdt = master[(master["Modelname"] == "gbdt") & master["paradigm"].isin(FI_GBDT_PARADIGMS)
                  & plain.str.split("_").str[0].isin(FI_GBDT_FAMILIES)]
    chronos = master[(master["Modelname"] == "chronos2") & plain.str.startswith("chronos2")]
    if gbdt.empty:
        raise MissingInput("no Global/Activity count GBDT in the master leaderboard")
    if chronos.empty:
        raise MissingInput("no Chronos-2 row in the master leaderboard")
    picked = []
    for rows in (gbdt, chronos):
        i = int(rows.index[0])
        r = master.loc[i]
        picked.append(FIModel(str(r["Modelname"]), str(r["paradigm"]), plain[i], i + 1))
    g, c = picked
    ctx.notes.append(f"models: {g.key} (rank {g.rank}), {c.key} (rank {c.rank})")
    return g, c


def _missing_importance(ctx: BuildContext, experiment: str, model: str, paradigm: str,
                        what: str) -> MissingInput:
    msg = f"no feature importance for {what}"
    if ctx.source.name == "store":
        seed = getattr(ctx.source, "seed", 42)
        root = getattr(ctx.source, "root", "<store-root>")
        msg += (f"; compute it as a cluster job: `strikecast importance experiment={experiment} "
                f"model={model} paradigm={paradigm} seed={seed} --store-root {root}` "
                f"(or add it to strikecast.pipeline.importance_stage.DEFAULT_JOBS so "
                f"`submit_all.py --stages importance` schedules it)")
    return MissingInput(msg)


def _gbdt_importancedict(ctx: BuildContext, g: FIModel) -> dict[str, dict[str, pd.DataFrame]]:
    """``{group: {metric: top features}}`` of the chosen GBDT: one group per tier for an
    Activity model (``activity_<k>``), a single ``global`` group otherwise."""
    from strikecast.evaluation.importance import top_features  # noqa: PLC0415

    groups = ([(f"activity_{t}", f"activity_{t}_{g.model}") for t in _FI_TIERS]
              if g.paradigm == "activity" else [("global", f"global_{g.model}")])
    try:
        imp = ctx.source.gbdt_importance()
    except MissingInput as exc:
        raise _missing_importance(ctx, "count", g.model, g.paradigm, g.key) from exc
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for group, label in groups:
        sub = imp[imp["model"] == label]
        if sub.empty:
            raise _missing_importance(ctx, "count", g.model, g.paradigm, label)
        out[group] = {m: top_features(sub, f"agg_{m}") for m in ("gain", "perm")
                      if f"agg_{m}" in sub and not sub[f"agg_{m}"].isna().all()}
    return out


def _chronos_importance(ctx: BuildContext, c: FIModel) -> pd.DataFrame:
    try:
        return ctx.source.chronos_importance(c.model)
    except MissingInput as exc:
        raise _missing_importance(ctx, "chronos2", c.model, "global", c.key) from exc


def build_importance_share(ctx: BuildContext) -> list[Path]:
    """F20: category shares of the best GBDT (per tier if Activity) and the best Chronos-2."""
    from strikecast.evaluation.importance import (  # noqa: PLC0415
        CATEGORY_ORDER,
        category_importance_matrix,
        category_shares,
        top_features,
    )

    g, c = _fi_models(ctx)
    gbdt = category_importance_matrix(_gbdt_importancedict(ctx, g))
    chronos = category_shares(top_features(_chronos_importance(ctx, c), "importance"))
    facets, cols = [], {}
    for group in (f"Tier {t}" for t in _FI_TIERS) if g.paradigm == "activity" else ["global"]:
        metrics = [str(col).removeprefix(f"{group} ") for col in gbdt.columns
                   if str(col).startswith(f"{group} ")]
        names = tuple(f"{g.key} {group} {m}" if g.paradigm == "activity" else f"{g.key} {m}"
                      for m in metrics)
        cols.update(zip(names, (gbdt[f"{group} {m}"] for m in metrics), strict=True))
        if g.paradigm == "activity":
            facets.append(P.ShareFacet(group, names, tuple(metrics), g.title))
        else:
            facets.append(P.ShareFacet(g.title, names, tuple(metrics)))
    cols[f"{c.key} perm"] = chronos
    facets.append(P.ShareFacet(c.title, (f"{c.key} perm",), ("perm",)))
    mat = pd.DataFrame(cols).reindex(list(CATEGORY_ORDER)).fillna(0.0)
    mat = mat.loc[(mat != 0).any(axis=1)]
    mat.reset_index(names="category").to_csv(
        ctx.out / "Feature-importancesharebycategory_grouped.csv", index=False, lineterminator="\n"
    )
    with thesis_style():
        fig = P.importance_share_heatmap(mat, facets, footnote=_FI_FOOTNOTE)
        return [ctx.figure("Feature-importancesharebycategory_grouped.svg", fig),
                ctx.out / "Feature-importancesharebycategory_grouped.csv"]


def build_importance_per_activity(ctx: BuildContext) -> list[Path]:
    """F23: top-15 gain/permutation features of the best GBDT, one row per tier if Activity.

    The file name stays the thesis' (``main.tex`` includes it) even when the
    chosen model is Global and the figure has a single row.
    """
    g, _ = _fi_models(ctx)
    d = _gbdt_importancedict(ctx, g)
    title = ("Top 15 Feature Importances per Activity Level" if g.paradigm == "activity"
             else f"Top 15 Feature Importances, {g.title}")
    with thesis_style():
        fig = P.feature_importance_grid(d, title)
        return [ctx.figure("Top15FeatureImportancesperActivityLevel.svg", fig, tight=False)]


def build_chronos_importance(ctx: BuildContext) -> list[Path]:
    """F21: top-15 permutation importance of the best Chronos-2 variant."""
    _, c = _fi_models(ctx)
    d = {"chronos2": {"Permutation importance": _chronos_importance(ctx, c)}}
    title = ("Chronos2 Local Feature Importance" if c.model == "chronos2_fine_tuned"
             else f"Chronos2 Local Feature Importance ({C.pretty_model(c.model)})")
    with thesis_style():
        fig = P.feature_importance_grid(d, title)
        return [ctx.figure("Chronos2LocalFeatureImportance.svg", fig, tight=False)]


def build_eda_maps(ctx: BuildContext) -> list[Path]:
    E.require_geopandas()
    with thesis_style(None):
        f1, f2 = E.strike_activity_maps(ctx.eda)
        return [ctx.figure("strike_activity_per_region.svg", f1, tight=False),
                ctx.figure("strike_activity_per_region_activity_level.svg", f2, tight=False)]


def build_eda_heatmap(ctx: BuildContext) -> list[Path]:
    with thesis_style(None):
        fig = E.spatiotemporal_heatmap(ctx.eda.master, ctx.eda.regions)
        return [ctx.figure("spatiotemporalintensityheatmap.svg", fig, tight=False)]


def build_eda_distribution(ctx: BuildContext) -> list[Path]:
    with thesis_style(None):
        return [ctx.figure("fig_eda1_target_distribution.svg", E.target_distribution(ctx.eda.tlong))]


def build_eda_acf(ctx: BuildContext) -> list[Path]:
    with thesis_style(None):
        return [ctx.figure("fig_eda2_onlypacfacf.svg", E.acf_pacf_figure(ctx.eda.tlong))]


def build_eda_stl(ctx: BuildContext) -> list[Path]:
    with thesis_style(None):
        fig, strengths = E.stl_side_by_side(ctx.eda.tlong)
        ctx.numbers.update({f"STL{k.replace('_', '')}": v for k, v in strengths.items()})
        return [ctx.figure("fig_eda2c_stl_side_by_side.svg", fig)]


# --------------------------------------------------------------------------- #
# the registry (audit C Part 1)
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    id: str
    label: str
    line: int | None
    klass: str  # DATA | RESULTS | STATIC | EXTRA
    kind: str  # figure | table
    outputs: tuple[str, ...]
    inputs: str
    builder: Callable[[BuildContext], list[Path]] | None = None
    note: str = ""
    status: str = "pending"
    reason: str = ""
    written: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _items() -> list[Item]:
    S = "STATIC"
    return [
        Item("F1", "fig:timelines", 285, S, "figure", ("timeline.jpg",), "hand-made JPEG"),
        Item("F2", "fig:strikeactivity_perregion", 321, "DATA", "figure",
             ("strike_activity_per_region.svg",),
             "master parquet, regions_activity_cat.json, Ukraine_Admin_Regions.geojson (geopandas)",
             build_eda_maps, "eda.ipynb cell 6; built together with F3"),
        Item("F3", "fig:strikeactivity_perregion_activity_levels", 328, "DATA", "figure",
             ("strike_activity_per_region_activity_level.svg",), "as F2", None,
             "written by the F2 builder"),
        Item("F4", "fig:spatiotemporalintensityheatmap", 417, "DATA", "figure",
             ("spatiotemporalintensityheatmap.svg",), "master parquet", build_eda_heatmap,
             "eda.ipynb cell 9"),
        Item("F5", "fig:Targetdistribution", 426, "DATA", "figure",
             ("fig_eda1_target_distribution.svg",), "master parquet", build_eda_distribution,
             "eda_full.ipynb cell 5"),
        Item("F6", "fig:pacfacf", 437, "DATA", "figure", ("fig_eda2_onlypacfacf.svg",),
             "master parquet", build_eda_acf, "eda_full.ipynb cell 10"),
        Item("F7", "fig:STLDECOMP", 447, "DATA", "figure", ("fig_eda2c_stl_side_by_side.svg",),
             "master parquet", build_eda_stl, "eda_full.ipynb cell 14"),
        Item("F8", "fig:feature_selection", 532, S, "figure", ("feature_selection.svg",), "Lucidchart"),
        Item("F9", "fig:hurdle_approach", 706, S, "figure", ("HurdleCountModel.svg",), "Lucidchart"),
        Item("F10", "fig:Experimentalsetup", 747, S, "figure", ("Experimentaldesign.svg",),
             "Lucidchart"),
        Item("F11", "fig:expanding_window_validation", 772, S, "figure",
             ("expandingwindowvalidation.svg",), "Lucidchart"),
        Item("F12", "fig:rmse_per_region", 954, "RESULTS", "figure",
             ("per_region_grid_by_tier_RMSE.svg", "per_region_grid_by_tier_RMSE.csv"),
             "per-region test metrics of top-5 + seasonal naive (+ARIMA, not drawn); tier map",
             build_region_grids, "AR cell 21; built together with F13"),
        Item("F13", "fig:mae_per_region", 962, "RESULTS", "figure",
             ("per_region_grid_by_tier_MAE.svg", "per_region_grid_by_tier_MAE.csv"), "as F12", None,
             "written by the F12 builder"),
        Item("F14", "fig:cd_rmse_top5.svg", 982, "RESULTS", "figure", ("cd_rmse_top5.svg",),
             "test predictions of top-5 + ARIMA (140 region x horizon blocks)", build_cd,
             "AR cells 29-32; built together with F15. Ranks/CD reproduce; the thesis SVG is 473 pt tall (older code), this one 310 pt (audit C F14)"),
        Item("F15", "fig:cd_mae_top5.svg", 993, "RESULTS", "figure", ("cd_mae_top5.svg",), "as F14",
             None, "written by the F14 builder"),
        Item("F16", "fig:calibration_prcurve", 1011, "RESULTS", "figure",
             ("calibration_prcurve.svg",), "hurdle test classifier probabilities, raw + calibrated",
             build_calibration, "AR cell 34"),
        Item("F17", "fig:prauc_prevelance", 1020, "RESULTS", "figure", ("prauc_vs_prevalence.svg",),
             "hurdle test calibrated probabilities", build_prauc_prevalence, "AR cell 36"),
        Item("F18", "fig:top_5_horizon", 1067, "RESULTS", "figure", ("top_5_combined.svg",),
             "per-horizon test metrics of the top-5", build_top5_horizon, "AR cell 26"),
        Item("F19", "fig:rmse_horizon_boxplot", 1074, "RESULTS", "figure",
             ("top20_rmse_horizon.svg",), "per-horizon test metrics of master_df[:20] (D7)",
             build_top20_boxplot, "AR cell 27"),
        Item("F20", "fig:importancesharebycategory_grouped", 1115, "RESULTS", "figure",
             ("Feature-importancesharebycategory_grouped.svg",
              "Feature-importancesharebycategory_grouped.csv"),
             "importance of the best Global/Activity count GBDT (per tier if Activity) and the "
             "best Chronos-2 variant, picked from the master leaderboard", build_importance_share,
             "AR cells 47, 52; one heatmap for both models (2026-09-29)"),
        Item("F21", "fig:top_fi_chronos2", 1125, "RESULTS", "figure",
             ("Chronos2LocalFeatureImportance.svg",),
             "permutation importance of the best Chronos-2 variant (master leaderboard)",
             build_chronos_importance, "AR cells 45, 51"),
        Item("F22", "fig:judging_system", 1260, S, "figure", ("LLM_as_judge.svg",), "Lucidchart"),
        Item("F23", "fig:top_fi_per_activity", 1611, "RESULTS", "figure",
             ("Top15FeatureImportancesperActivityLevel.svg",),
             "importance of the best Global/Activity count GBDT (master leaderboard)",
             build_importance_per_activity,
             "AR cells 47, 50; one row per tier if the model is Activity, else one row"),
        Item("T1", "tab:acledData", 184, S, "table", (), "ACLED codebook (typed)"),
        Item("T2", "tab:activitytiers", 297, "DATA", "table",
             ("tab_activitytiers.csv", "tab_activitytiers.tex"),
             "master parquet + regions_activity_cat.json", build_activity_tiers,
             "tabularx {l c Y}; needs the Y column type main.tex defines"),
        Item("T3", "tab:splitdimensions", 352, "DATA", "table",
             ("tab_splitdimensions.csv", "tab_splitdimensions.tex"),
             "master parquet, darts split_series_list", build_split_dimensions, "C11: thesis prints 592/85/169 = 846 days, 16,920 obs; data give 593/85/169 = 847, 16,940"),
        Item("T4", "tab:overall_performance", 902, "RESULTS", "table",
             ("tab_overall_performance.csv", "tab_overall_performance.tex"),
             "test predictions of the 11 Table-4 models (hurdle = hurdle_cal, D6)",
             build_overall_performance,
             "AR cell 7. D6/C7: Hurdle row 0.80/2.02/0.09 (thesis 0.79/2.00/0.09); other 10 rows match"),
        Item("T5", "tab:top_models", 923, "RESULTS", "table",
             ("tab_top_models.csv", "tab_top_models.tex"), "master leaderboard", build_top_models,
             "master_df.head(5). C8: rank-1 MAE 0.77 (thesis prints 0.78)"),
        Item("T6", "tab:hurdle_bias", 1036, "RESULTS", "table",
             ("tab_hurdle_bias.csv", "tab_hurdle_bias.tex"),
             "hurdle count-head test predictions, y_true > 0", build_hurdle_bias,
             "AR cell 38; .tex top 8 by RMSE, CSV all regions. C6/D6: counts and maxima match, RMSE/MAE/bias are the stored-prediction values (Sumy 6.79/4.48/-3.13, thesis 7.35/4.83/-4.17)"),
        Item("T7", "tab:tuning-search-space", 1354, S, "table", (), "search spaces in code (typed)"),
        Item("T8", "tab:tuning-best-all", 1409, "RESULTS", "table",
             ("tab_tuning_best_all.csv", "tab_tuning_best_all.tex"), "best_params.json",
             build_tuning_best),
        Item("T9", "tab:base_variables", 1463, "DATA", "table",
             ("tab_base_variables_rows.csv", "tab_base_variables_rows.tex"), "panel columns",
             build_base_variables, "longtable ROWS only (the longtable header stays in main.tex)"),
        Item("T10", "(appendix Top 20, no label)", 1543, "RESULTS", "table",
             ("tab_top20.csv", "tab_top20.tex"), "master leaderboard", build_top20,
             "master_df[:20] incl. diff/global/linear (D7)"),
        Item("T11", "tab:semantic_categories", 1577, S, "table", (),
             "mirrors evaluation.importance.CATEGORY_RULES (typed)"),
        Item("X1", "(cross-experiment leaderboard, B16)", None, "EXTRA", "table",
             ("master_leaderboard.csv", "master_leaderboard.tex"), "every test leaderboard",
             build_master, "SkillScore = 1 - RMSE/RMSE(diff naive_weekly, global, test)"),
        Item("X2", "(inline: Friedman ranks, text 972-980)", None, "EXTRA", "table",
             ("tab_friedman_top5.csv", "tab_friedman_top5.tex"), "as F14", build_friedman),
        Item("X3", "(inline: horizon statistics, text 1079-1098, B17)", None, "EXTRA", "table",
             ("tab_horizon_stats.csv", "tab_horizon_stats.tex", "tab_horizon_tukey.csv",
              "tab_horizon_tukey.tex"), "as F19", build_horizon_stats,
             "F 6.7041, Shapiro 0.41, Levene 0.546, Tukey p reproduce; C10: KW p = 0.483 (thesis ~0.05)"),
        Item("X4", "(inline: PR-AUC / prevalence, text 1014-1031)", None, "EXTRA", "table",
             ("tab_prauc_per_region.csv", "tab_prauc_per_region.tex"), "as F16/F17",
             build_prauc_numbers),
        Item("X5", "(inline: EDA zero rate / dispersion, text 426ff)", None, "EXTRA", "table",
             ("tab_eda_marginals.csv", "tab_eda_marginals.tex"), "master parquet", build_eda_stats),
    ]


ITEMS: tuple[Item, ...] = tuple(_items())

#: Items written by another item's builder: status follows the builder's item.
_COMPANIONS = {"F3": "F2", "F13": "F12", "F15": "F14"}


def build_all(
    source: ResultsSource,
    out_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    only: Sequence[str] | None = None,
) -> list[Item]:
    """Run every builder; returns fresh :class:`Item` copies with status and outputs."""
    import dataclasses  # noqa: PLC0415

    ctx = BuildContext(source, out_dir, data_dir=data_dir)
    ctx.out.mkdir(parents=True, exist_ok=True)
    items = [dataclasses.replace(i, written=[], notes=[]) for i in _items()]
    by_id = {i.id: i for i in items}
    for item in items:
        if item.klass == "STATIC":
            item.status, item.reason = "static", "hand-drawn / typed; no generating code"
            continue
        if item.builder is None:
            continue
        if only is not None and item.id not in only:
            item.status, item.reason = "not-requested", ""
            continue
        ctx.notes = item.notes
        try:
            paths = item.builder(ctx)
        except MissingInput as exc:
            item.status, item.reason = "skipped", str(exc)
            logger.warning("%s %s skipped: %s", item.id, item.label, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - one broken item must not stop the rest
            item.status, item.reason = "failed", f"{type(exc).__name__}: {exc}"
            logger.error("%s %s FAILED:\n%s", item.id, item.label, traceback.format_exc())
            continue
        item.status = "generated"
        item.written = sorted({Path(p).name for p in paths})
    for comp, owner in _COMPANIONS.items():
        c, o = by_id[comp], by_id[owner]
        if only is not None and owner not in only:
            c.status = "not-requested"
            continue
        c.status, c.reason = o.status, o.reason
        c.written = [n for n in o.written if n in c.outputs]
        o.written = [n for n in o.written if n not in c.outputs]
    write_numbers(ctx)
    write_manifest(items, ctx)
    return items


_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def _macro_name(key: str) -> str:
    """TeX control words are letters only: ``Tier1`` -> ``TierOne``."""
    return "".join(_DIGITS[int(ch)] if ch.isdigit() else ch for ch in key if ch.isalnum())


def write_numbers(ctx: BuildContext) -> Path:
    """``numbers.tex``: ``\\newcommand`` macros for inline numbers (+ ``numbers.json``)."""
    lines = ["% generated by `strikecast figures`; do not edit"]
    for key in sorted(ctx.numbers):
        v = ctx.numbers[key]
        text = f"{v:.4g}" if isinstance(v, float) else str(v)
        lines.append(rf"\newcommand{{\sc{_macro_name(key)}}}{{{text}}}")
    path = ctx.out / "numbers.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ctx.out / "numbers.json").write_text(
        json.dumps({k: ctx.numbers[k] for k in sorted(ctx.numbers)}, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return path


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_manifest(items: Sequence[Item], ctx: BuildContext) -> tuple[Path, Path]:
    payload = {
        "source": ctx.source.describe(),
        "git_commit": _git_commit(),
        "items": [
            {
                "id": i.id, "label": i.label, "main_tex_line": i.line, "class": i.klass,
                "kind": i.kind, "status": i.status, "reason": i.reason,
                "outputs": list(i.written) if i.status == "generated" else list(i.outputs),
                "source_data": i.inputs, "note": i.note, "notes": list(i.notes),
            }
            for i in items
        ],
    }
    jpath = ctx.out / "MANIFEST.json"
    jpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for i in items:
        counts[i.status] = counts.get(i.status, 0) + 1
    md = [
        "# `strikecast figures` manifest",
        "",
        f"Source: {payload['source']}  ",
        f"Git commit: `{payload['git_commit']}`  ",
        "Status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| id | thesis label | main.tex line | class | status | outputs | source data | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for it in payload["items"]:
        notes = "; ".join([n for n in [it["note"], it["reason"], *it["notes"]] if n])
        md.append(
            f"| {it['id']} | `{it['label']}` | {it['main_tex_line'] or '–'} | {it['class']} | "
            f"{it['status']} | {', '.join(f'`{o}`' for o in it['outputs']) or '–'} | "
            f"{it['source_data']} | {notes.replace('|', '/')} |"
        )
    mpath = ctx.out / "MANIFEST.md"
    mpath.write_text("\n".join(md) + "\n", encoding="utf-8")
    return mpath, jpath
