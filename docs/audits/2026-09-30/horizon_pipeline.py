"""Horizon test pipeline without new citations (proposal, 2026-09-30).

Read-only: loads the saved seed-42 test metrics/predictions of runs_publication_20260929 and runs
the proposed repeated-measures horizon tests plus the repeated-measures ANOVA alternative and
robustness checks. Nothing is trained or written.

    uv run python docs/audits/2026-09-30/horizon_pipeline.py | tee docs/audits/2026-09-30/horizon_pipeline_output.txt

Design: blocks = configurations (rows), treatments = horizons 1..7 (columns), one score per cell.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs_publication_20260929"
FAMILY_DIR = {"gbdt": "count", "lstm": "count", "diff": "diff", "chronos2": "chronos2", "finalhurdle": "hurdle"}
H = list(range(1, 8))
ALPHA = 0.05
# Pre-specified post-hoc family: every horizon vs Day 1 (6) + every day vs the previous day (5 new).
COMPARISONS = [(h, 1) for h in H[1:]] + [(h, h - 1) for h in H[2:]]
METRICS = ("RMSE", "MAE")


def run_dir(r: pd.Series) -> Path:
    par = "global" if r.Modelname == "chronos2" else r.paradigm
    return RUNS / FAMILY_DIR[r.Modelname] / r.model / par / "seed=42/test"


def label(r: pd.Series) -> str:
    # family prefix: count-LSTM and diff-LSTM runs share model names (e.g. lstm_w14@global)
    return f"{FAMILY_DIR[r.Modelname]}/{r.model}@{r.paradigm}"


def matrices_from_csv(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per = {label(r): pd.read_csv(run_dir(r) / "metrics/per_horizon.csv").set_index("horizon") for _, r in rows.iterrows()}
    return {m: pd.DataFrame({k: v[m] for k, v in per.items()}).T[H] for m in METRICS}


def load_preds(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for _, r in rows.iterrows():
        p = pd.read_parquet(run_dir(r) / "predictions")
        ch = "hurdle" if r.Modelname == "finalhurdle" else "y_pred"  # hurdle stores prob/count/hurdle
        out[label(r)] = p[p.channel == ch]
    return out


def matrices_from_preds(preds: dict[str, pd.DataFrame], dates=None) -> dict[str, pd.DataFrame]:
    res = {m: {} for m in METRICS}
    for k, p in preds.items():
        if dates is not None:
            p = p[p.date.isin(dates)]
        e = p.y_true - p.y_pred.clip(lower=0)  # metrics.py:249 clips predictions at zero
        res["RMSE"][k] = np.sqrt((e ** 2).groupby(p.horizon).mean())
        res["MAE"][k] = e.abs().groupby(p.horizon).mean()
    return {m: pd.DataFrame(v).T[H] for m, v in res.items()}


def section(tag: str, title: str) -> None:
    print(f"\n{'=' * 110}\n[{tag}] {title}\n{'=' * 110}")


# --------------------------------------------------------------------------- tests
def friedman(M: pd.DataFrame) -> dict:
    N, k = M.shape
    fr = stats.friedmanchisquare(*[M[h].values for h in H])
    ff = (N - 1) * fr.statistic / (N * (k - 1) - fr.statistic)
    return dict(N=N, k=k, chi2=fr.statistic, p=fr.pvalue, FF=ff, FF_p=stats.f.sf(ff, k - 1, (k - 1) * (N - 1)),
                W=fr.statistic / (N * (k - 1)), ranks=M.rank(axis=1).mean())


def pairwise(M: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    for a, b in COMPARISONS:
        d = M[a] - M[b]
        pct = 100 * d / M[b]
        if kind == "wilcoxon":
            t = stats.wilcoxon(M[a], M[b])
        else:
            t = stats.ttest_rel(M[a], M[b])
        rows.append(dict(comparison=f"h{a} vs h{b}", worse=f"{int((d > 0).sum())}/{len(d)}",
                         mean_diff=d.mean(), median_pct=pct.median(), stat=t.statistic, p_raw=t.pvalue))
    df = pd.DataFrame(rows)
    rej, p_holm, _, _ = multipletests(df.p_raw, alpha=ALPHA, method="holm")
    return df.assign(p_holm=p_holm, sig=rej)


def rm_anova(M: pd.DataFrame) -> dict:
    X = M.values
    N, k = X.shape
    g = X.mean()
    ss_h = N * ((X.mean(0) - g) ** 2).sum()
    resid = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + g
    ss_res = (resid ** 2).sum()
    df1, df2 = k - 1, (k - 1) * (N - 1)
    F = (ss_h / df1) / (ss_res / df2)
    # Sphericity: orthonormal contrasts (normalised Helmert), sample covariance S (p x p, p = k-1)
    C = np.zeros((k, k - 1))
    for j in range(1, k):
        C[:j, j - 1] = 1
        C[j, j - 1] = -j
        C[:, j - 1] /= np.linalg.norm(C[:, j - 1])
    S = np.cov(X @ C, rowvar=False)
    p = k - 1
    mW = np.linalg.det(S) / (np.trace(S) / p) ** p
    mchi = -(N - 1 - (2 * p * p + p + 2) / (6 * p)) * np.log(mW)
    mdf = p * (p + 1) / 2 - 1
    gg = np.trace(S) ** 2 / (p * np.trace(S @ S))
    hf = min(1.0, (N * p * gg - 2) / (p * (N - 1 - p * gg)))
    long = M.rename_axis("cfg").reset_index().melt(id_vars="cfg", var_name="h", value_name="y")
    sm = AnovaRM(long, "y", "cfg", within=["h"]).fit().anova_table
    return dict(F=F, df1=df1, df2=df2, p=stats.f.sf(F, df1, df2), eta_p=ss_h / (ss_h + ss_res),
                F_statsmodels=float(sm["F Value"].iloc[0]),
                shapiro_resid_p=stats.shapiro(resid.ravel()).pvalue,
                mauchly_W=mW, mauchly_chi2=mchi, mauchly_df=mdf, mauchly_p=stats.chi2.sf(mchi, mdf),
                gg_eps=gg, gg_p=stats.f.sf(F, gg * df1, gg * df2), hf_eps=hf, hf_p=stats.f.sf(F, hf * df1, hf * df2))


def nemenyi(M: pd.DataFrame) -> tuple[pd.DataFrame, float, list[str]]:
    N, k = M.shape
    P = sp.posthoc_nemenyi_friedman(M.values)
    P.index = P.columns = [f"h{h}" for h in H]
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    sig = [f"{a}-{b}" for i, a in enumerate(P.index) for b in P.columns[i + 1:] if P.loc[a, b] < ALPHA]
    return P, cd, sig


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def summary_row(name: str, M: pd.DataFrame) -> dict:
    fr = friedman(M)
    wx = pairwise(M, "wilcoxon")
    ra = rm_anova(M)
    means = M.mean()
    return dict(variant=name, N=fr["N"], friedman=f"chi2={fr['chi2']:.1f} p={fmt_p(fr['p'])}", W=round(fr["W"], 2),
                best_h=f"h{means.idxmin()}", worst_h=f"h{means.idxmax()}",
                spread_pct=round(100 * (means.max() - means.min()) / means.min(), 2),
                holm_sig=", ".join(wx.loc[wx.sig, "comparison"]) or "none",
                rmANOVA_GG=f"F={ra['F']:.1f} p={fmt_p(ra['gg_p'])}")


# --------------------------------------------------------------------------- main
def report(tag: str, M: pd.DataFrame, m: str) -> None:
    N, k = M.shape
    section(f"P1{tag}-{m}", "Descriptives: column means, change relative to Day 1 within each configuration")
    rel = 100 * M.div(M[1], axis=0) - 100
    d = pd.DataFrame({"mean": M.mean(), "median": M.median(), "mean_rank": M.rank(axis=1).mean(),
                      "median_pct_vs_h1": rel.median(), "configs_worse_than_h1": (rel > 0).sum()})
    print(d.round(4).to_string())
    print(f"spread of column means: {100 * (M.mean().max() - M.mean().min()) / M.mean().min():.2f}% "
          f"(h{M.mean().idxmin()} lowest, h{M.mean().idxmax()} highest)")
    print("worst horizon per configuration:", M.idxmax(axis=1).value_counts().sort_index().to_dict())

    section(f"P2{tag}-{m}", "STEP 1 omnibus: Friedman test (Demsar 2006, sec. 3.2.2)")
    fr = friedman(M)
    print(f"Friedman chi2({k - 1}) = {fr['chi2']:.3f}, p = {fr['p']:.3g}   [N = {N} > 10 and k = {k} > 5: chi2 approximation OK]")
    print(f"Iman-Davenport F_F({k - 1},{(k - 1) * (N - 1)}) = {fr['FF']:.3f}, p = {fr['FF_p']:.3g}")
    print(f"Kendall's W = chi2 / (N(k-1)) = {fr['W']:.3f}")
    print("mean rank (1 = lowest error):", "  ".join(f"h{h}={r:.2f}" for h, r in fr["ranks"].items()))

    section(f"P3{tag}-{m}", "STEP 2 post-hoc: Wilcoxon signed-rank, 11 pre-specified pairs, Holm (Demsar 3.1.3 + p. 12-13)")
    print(pairwise(M, "wilcoxon").to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    section(f"P4{tag}-{m}", "STEP 3 (figure) Nemenyi on Friedman ranks, CD diagram input (Demsar p. 11)")
    _, cd, sig = nemenyi(M)
    print(f"CD = {cd:.3f} ranks; significant pairs: {', '.join(sig) or 'none'}")

    section(f"P5{tag}-{m}", "ALTERNATIVE: repeated-measures ANOVA (Demsar 3.2.1) with its assumption checks")
    ra = rm_anova(M)
    print(f"RM-ANOVA F({ra['df1']},{ra['df2']}) = {ra['F']:.3f}, p = {ra['p']:.3g}, partial eta^2 = {ra['eta_p']:.3f}"
          f"   (statsmodels AnovaRM F = {ra['F_statsmodels']:.3f})")
    print(f"normality of residuals: Shapiro-Wilk p = {ra['shapiro_resid_p']:.3g}")
    print(f"sphericity: Mauchly W = {ra['mauchly_W']:.4f}, chi2({ra['mauchly_df']:.0f}) = {ra['mauchly_chi2']:.2f}, "
          f"p = {ra['mauchly_p']:.3g}")
    print(f"Greenhouse-Geisser eps = {ra['gg_eps']:.3f} -> corrected p = {ra['gg_p']:.3g};  "
          f"Huynh-Feldt eps = {ra['hf_eps']:.3f} -> p = {ra['hf_p']:.3g}")
    print("parametric post-hoc (paired t, same 11 pairs, Holm):")
    print(pairwise(M, "t").to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def main() -> None:
    lb = pd.read_csv(RUNS / "_figures/master_leaderboard.csv")
    top20 = lb.head(20)
    Mcsv = matrices_from_csv(top20)
    preds_all = load_preds(lb)
    preds = {label(r): preds_all[label(r)] for _, r in top20.iterrows()}

    section("P0", "Data and sanity check")
    Mfull = matrices_from_preds(preds)
    Mall_csv, Mall_pred = matrices_from_csv(lb), matrices_from_preds(preds_all)
    for m in METRICS:
        print(f"{m}: per_horizon.csv == recomputed from predictions (all dates): top-20 "
              f"{np.allclose(Mcsv[m].values, Mfull[m].values, rtol=1e-9, atol=1e-12)}, all {len(lb)} "
              f"{np.allclose(Mall_csv[m].values, Mall_pred[m].loc[Mall_csv[m].index].values, rtol=1e-9, atol=1e-12)}")
    p0 = next(iter(preds.values()))
    per_h = p0.groupby("horizon").date.agg(["min", "max", "nunique"])
    print("target dates per horizon (first config; identical design for all):\n" + per_h.to_string())
    common = sorted(set.intersection(*[set(p0.loc[p0.horizon == h, "date"]) for h in H]))
    print(f"target dates shared by all 7 horizons: {len(common)} ({common[0].date()} .. {common[-1].date()})")
    print(f"prediction files loaded: {len(preds_all)} of {len(lb)} leaderboard rows")
    Mcommon = matrices_from_preds(preds, dates=set(common))

    designs = {"A": ("DESIGN A = current: each horizon scored on its own 164 target dates", Mcsv),
               "B": (f"DESIGN B = common target dates: every horizon scored on the same {len(common)} dates", Mcommon)}
    for tag, (title, Ms) in designs.items():
        print(f"\n\n{'#' * 110}\n# {title}\n{'#' * 110}")
        for m in METRICS:
            report(tag, Ms[m], m)

    section("P6", "ROBUSTNESS: same pipeline on alternative sets of configurations, both designs")
    sets = {
        "top20 (baseline)": top20,
        "top10": lb.head(10),
        "top30": lb.head(30),
        f"all {len(lb)} configurations": lb,
        f"top20 minus repeated models": top20.drop_duplicates("model"),
        "best paradigm of each model, top 20 models": lb.drop_duplicates(["Modelname", "model"]).head(20),
    }
    for m in METRICS:
        rows = []
        for n, rs in sets.items():
            sub = {label(r): preds_all[label(r)] for _, r in rs.iterrows()}
            rows.append(summary_row(f"A {n}", matrices_from_preds(sub)[m]))
            rows.append(summary_row(f"B {n}", matrices_from_preds(sub, dates=set(common))[m]))
        print(f"\n--- {m} ---")
        print(pd.DataFrame(rows).to_string(index=False))

    section("P7", "Why target dates matter: edge dates that only some horizons score")
    y = p0.drop_duplicates(["region", "date"]).set_index(["date"])["y_true"]
    daily = y.groupby(level=0).sum()
    edge_lo = [d for d in daily.index if d < common[0]]
    edge_hi = [d for d in daily.index if d > common[-1]]
    print(f"strikes/day (sum over regions): common dates mean {daily.loc[common].mean():.1f}, "
          f"early-only dates {daily.loc[edge_lo].mean():.1f}, late-only dates {daily.loc[edge_hi].mean():.1f}")
    print("which horizons score the edge dates: early-only date d is scored by h <= (d - first origin + 1); "
          "late-only date is scored by h >= (d - last origin + 1)")
    print("daily totals, early-only:", {str(d.date()): int(daily[d]) for d in edge_lo})
    print("daily totals, late-only: ", {str(d.date()): int(daily[d]) for d in edge_hi})
    for m in METRICS:
        a, b = Mcsv[m].mean(), Mcommon[m].mean()
        print(f"{m} column means  A all dates: " + "  ".join(f"h{h}={a[h]:.4f}" for h in H))
        print(f"{m} column means  B common:    " + "  ".join(f"h{h}={b[h]:.4f}" for h in H))

    section("P8", "Design B: which target dates make Day 1 worse than Day 2 (squared error, summed over top-20)")
    se = {}
    for k_, p in preds.items():
        q = p[p.date.isin(common) & p.horizon.isin([1, 2])]
        q = q.assign(se=(q.y_true - q.y_pred.clip(lower=0)) ** 2)
        se[k_] = q.pivot_table(index="date", columns="horizon", values="se", aggfunc="sum")
    tot = sum(se.values())
    diff = (tot[1] - tot[2]).sort_values(ascending=False)
    share = diff.clip(lower=0)
    print(f"sum over dates of SE(h1) - SE(h2): {diff.sum():.1f}; top-5 dates carry "
          f"{100 * diff.head(5).sum() / diff.sum():.0f}% of the net difference")
    prev = daily.shift(1)
    print(pd.DataFrame({"SE_h1_minus_h2": diff.head(8).round(1), "strikes_that_day": daily.reindex(diff.head(8).index),
                        "strikes_day_before": prev.reindex(diff.head(8).index)}).to_string())

    section("P9", "Design B split by model family: gradient-boosted trees vs all other top-20 configurations")
    fam = {label(r): ("GBDT" if r.Modelname == "gbdt" else "other") for _, r in top20.iterrows()}
    for m in METRICS:
        for g in ("GBDT", "other"):
            S = Mcommon[m][[fam[i] == g for i in Mcommon[m].index]]
            rel = 100 * S.div(S[1], axis=0) - 100
            fr = stats.friedmanchisquare(*[S[h] for h in H])
            w21, w71 = stats.wilcoxon(S[2], S[1]), stats.wilcoxon(S[7], S[1])
            print(f"{m} {g:5s} N={len(S):2d}  Friedman p={fr.pvalue:.3f}  "
                  f"Day2<Day1 {(S[2] < S[1]).sum()}/{len(S)} (raw p={w21.pvalue:.3f})  "
                  f"Day7<Day1 {(S[7] < S[1]).sum()}/{len(S)} (raw p={w71.pvalue:.3f})  "
                  "median % vs Day 1, Days 2-7: " + " ".join(f"{rel[h].median():+.1f}" for h in H[1:]))

    section("P10", "Design B per family over ALL 84 configurations (selection robustness)")

    def family(r: pd.Series) -> str:
        if r.Modelname in ("gbdt", "lstm", "chronos2", "finalhurdle"):
            return {"gbdt": "GBDT (count)", "lstm": "RNN (count)", "chronos2": "Chronos-2",
                    "finalhurdle": "Hurdle"}[r.Modelname]
        if r.model in ("arima", "linear", "naive_weekly", "naive_last"):
            return "Statistical/naive"
        return "GBDT (diff target)" if r.model in ("catboost", "xgboost", "lightgbm") else "RNN (diff target)"

    lbf = lb.assign(family=lb.apply(family, axis=1), rank=np.arange(1, len(lb) + 1))
    Ball = matrices_from_preds(preds_all, dates=set(common))

    def fam_row(name: str, rs: pd.DataFrame, m: str) -> dict:
        S = Ball[m].loc[[label(r) for _, r in rs.iterrows()]]
        rel = 100 * S.div(S[1], axis=0) - 100
        fr = stats.friedmanchisquare(*[S[h] for h in H]) if len(S) >= 3 else None
        return dict(set=name, n=len(S), ranks=f"{rs['rank'].min()}-{rs['rank'].max()}",
                    friedman_p=f"{fr.pvalue:.3f}" if fr else "-",
                    d2_better=f"{(S[2] < S[1]).sum()}/{len(S)}", d7_better=f"{(S[7] < S[1]).sum()}/{len(S)}",
                    median_pct_days2to7=" ".join(f"{rel[h].median():+.1f}" for h in H[1:]),
                    max_abs_median=round(float(rel[H[1:]].median().abs().max()), 2))

    for m in METRICS:
        rows = []
        for fname, rs in lbf.groupby("family", sort=False):
            rows.append(fam_row(f"{fname} (all)", rs, m))
            if len(rs) > 10:
                rows.append(fam_row(f"{fname} (top 10)", rs.head(10), m))
        for par in ("global", "activity", "local"):
            rows.append(fam_row(f"  GBDT (count) {par}", lbf[(lbf.family == "GBDT (count)") & (lbf.paradigm == par)], m))
        print(f"\n--- {m} ---")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
