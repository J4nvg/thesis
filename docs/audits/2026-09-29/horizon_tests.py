"""Evidence script for IMPROVED_HORIZON_TEXT.md (every number in that file comes from here).

Read-only: it loads the saved per-horizon test metrics of the seed-42 publication run and runs
the current (unpaired) and the proposed (paired) horizon tests. Nothing is trained or written.

    uv run python docs/audits/2026-09-29/horizon_tests.py

Output sections are labelled [E1]..[E10]; the proposal cites them by label.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy
import statsmodels
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs_publication_20260929"
# master-leaderboard family -> run-store directory. Chronos-2 is labelled "local" in the
# leaderboard but its run store lives under global/ (one directory per model).
FAMILY_DIR = {"gbdt": "count", "lstm": "count", "diff": "diff", "chronos2": "chronos2"}
H = list(range(1, 8))
ALPHA = 0.05


def load_top20() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    # Same selection as build.py::build_horizon_stats: ctx.top(20) = master_leaderboard.head(20)
    top = pd.read_csv(RUNS / "_figures/master_leaderboard.csv").head(20)
    per: dict[str, pd.DataFrame] = {}
    paths = []
    for _, r in top.iterrows():
        par = "global" if r.Modelname == "chronos2" else r.paradigm
        f = RUNS / FAMILY_DIR[r.Modelname] / r.model / par / "seed=42/test/metrics/per_horizon.csv"
        per[f"{r.model}@{r.paradigm}"] = pd.read_csv(f).set_index("horizon")
        paths.append(str(f.relative_to(ROOT)))
    top = top.assign(file=paths)
    return top, per


def section(tag: str, title: str) -> None:
    print(f"\n{'=' * 100}\n[{tag}] {title}\n{'=' * 100}")


def main() -> None:
    section("E0", "Environment")
    print(f"scipy {scipy.__version__}, statsmodels {statsmodels.__version__}, "
          f"scikit-posthocs {sp.__version__ if hasattr(sp, '__version__') else '?'}, "
          f"pandas {pd.__version__}, numpy {np.__version__}")

    top, per = load_top20()
    section("E1", "Input: the top-20 configurations (master_leaderboard.csv rows 1-20) and their files")
    print(top[["Modelname", "paradigm", "model", "rmse", "mae", "file"]].to_string())

    stored = pd.read_csv(RUNS / "_figures/tab_horizon_stats.csv")
    for metric in ("RMSE", "MAE"):
        M = pd.DataFrame({k: v[metric] for k, v in per.items()}).T[H]  # 20 configs x 7 horizons
        groups = [M[h].values for h in H]
        N, k = M.shape

        section(f"E2-{metric}", f"Input matrix: {metric} per configuration (rows) and horizon (cols)")
        print(M.round(4).to_string())
        print("\ncolumn mean:", "  ".join(f"h{h}={M[h].mean():.4f}" for h in H))
        print(f"range of column means: {M.mean().min():.4f} .. {M.mean().max():.4f} "
              f"(spread {100 * (M.mean().max() - M.mean().min()) / M.mean().min():.2f}% of the minimum)")
        print(f"mean(h1-h4) = {M[[1, 2, 3, 4]].values.mean():.4f}, mean(h5-h7) = {M[[5, 6, 7]].values.mean():.4f}")

        section(f"E3-{metric}", "CURRENT test (compute.py::horizon_statistics), recomputed; vs tab_horizon_stats.csv")
        sh = {h: stats.shapiro(M[h]).pvalue for h in H}
        print("Shapiro-Wilk p per horizon:", "  ".join(f"h{h}={p:.4f}" for h, p in sh.items()))
        lev = stats.levene(*groups).pvalue
        f = stats.f_oneway(*groups)
        kw = stats.kruskal(*groups)
        print(f"Levene p = {lev:.4f}  -> variance_passed = {lev >= ALPHA}")
        print(f"one-way ANOVA F({k - 1},{N * k - k}) = {f.statistic:.3f}, p = {f.pvalue:.3g}")
        print(f"Kruskal-Wallis H({k - 1}) = {kw.statistic:.3f}, p = {kw.pvalue:.3g}")
        row = stored[stored.metric == metric].iloc[0]
        print(f"stored in tab_horizon_stats.csv: levene_p={row.levene_p:.4f}, anova_F={row.anova_F:.3f}, "
              f"kruskal_H={row.kruskal_H:.3f}, kruskal_p={row.kruskal_p:.3g}, test={row.test}")

        section(f"E4-{metric}", "Why pairing matters: two-way sum-of-squares decomposition of the 20 x 7 matrix")
        g = M.values.mean()
        ss_tot = ((M.values - g) ** 2).sum()
        ss_mod = k * ((M.mean(axis=1) - g) ** 2).sum()
        ss_hor = N * ((M.mean(axis=0) - g) ** 2).sum()
        ss_res = ss_tot - ss_mod - ss_hor
        print(f"SS_total={ss_tot:.6f}  SS_between_configurations={ss_mod:.6f} ({ss_mod / ss_tot:.1%})  "
              f"SS_between_horizons={ss_hor:.6f} ({ss_hor / ss_tot:.1%})  SS_residual={ss_res:.6f} ({ss_res / ss_tot:.1%})")
        print("An unpaired test compares SS_between_horizons against SS_between_configurations + SS_residual;")
        print("a paired (repeated-measures) test compares it against SS_residual only.")

        section(f"E5-{metric}", "PROPOSED omnibus tests (paired; blocks = 20 configurations, treatments = 7 horizons)")
        fr = stats.friedmanchisquare(*groups)
        W = fr.statistic / (N * (k - 1))
        ff = (N - 1) * fr.statistic / (N * (k - 1) - fr.statistic)
        ff_p = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
        pg = stats.page_trend_test(M.values)  # columns ordered h1..h7 = predicted increasing error
        ranks = M.rank(axis=1)
        print(f"Friedman chi2({k - 1}) = {fr.statistic:.3f}, p = {fr.pvalue:.3g}")
        print(f"Kendall's W = chi2 / (N (k-1)) = {fr.statistic:.3f} / {N * (k - 1)} = {W:.3f}")
        # Check of that identity from W's own definition (Kendall & Babington Smith 1939):
        # W = 12 S / (N^2 (k^3 - k)), S = sum_j (R_j - mean R)^2, R_j = rank sum of horizon j.
        R = M.rank(axis=1).sum(axis=0)
        S = ((R - R.mean()) ** 2).sum()
        W_def = 12 * S / (N ** 2 * (k ** 3 - k))
        print(f"Kendall's W from its definition 12S/(N^2(k^3-k)) = {W_def:.3f}  (identical: {np.isclose(W, W_def)}; "
              "no ties within a configuration, so no tie correction applies)")
        print(f"Iman-Davenport F({k - 1},{(k - 1) * (N - 1)}) = {ff:.3f}, p = {ff_p:.3g}  (Demsar 2006, p. 11)")
        print(f"Page's L = {pg.statistic:.1f}, p = {pg.pvalue:.3g}, method = {pg.method}  (H1: h1 <= h2 <= ... <= h7)")
        print("mean Friedman rank per horizon (1 = lowest error):",
              "  ".join(f"h{h}={ranks[h].mean():.2f}" for h in H))

        section(f"E6-{metric}", "PROPOSED post-hoc: Wilcoxon signed-rank, each horizon vs Day 1 and vs previous day, Holm")
        tests = [(1, h) for h in H[1:]] + [(h, h + 1) for h in H[1:-1]]  # 6 + 5 = 11 distinct pairs
        rows = []
        for a, b in tests:
            w = stats.wilcoxon(M[b], M[a])
            pct = 100 * (M[b] - M[a]) / M[a]
            rows.append(dict(comparison=f"h{b} vs h{a}", configs_worse=f"{int((M[b] > M[a]).sum())}/20",
                             median_pct_change=round(pct.median(), 2), W=w.statistic, p_raw=w.pvalue))
        post = pd.DataFrame(rows)
        post["p_holm"] = multipletests(post.p_raw, method="holm")[1]
        post["reject_0.05"] = post.p_holm < ALPHA
        print(post.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

        section(f"E7-{metric}", "PROPOSED optional: Nemenyi post-hoc on the Friedman ranks (p-value matrix)")
        q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
        cd = q * np.sqrt(k * (k + 1) / (6 * N))
        nem = sp.posthoc_nemenyi_friedman(M.values)
        nem.index = nem.columns = [f"h{h}" for h in H]
        print(f"q_0.05 = {q:.3f}, CD = {cd:.3f} (mean ranks further apart than CD differ)")
        print(nem.round(3).to_string())
        sig = [f"h{a}-h{b}" for a, b in itertools.combinations(H, 2) if nem.loc[f"h{a}", f"h{b}"] < ALPHA]
        print("significant pairs:", ", ".join(sig) or "none")

        section(f"E8-{metric}", "For comparison only: unpaired Tukey HSD on the same data")
        long = M.stack().rename(metric).reset_index().rename(columns={"level_1": "horizon"})
        tk = pairwise_tukeyhsd(long[metric], long["horizon"], alpha=ALPHA)
        t = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
        print(t.to_string(index=False))
        rej = [f"h{r.group1}-h{r.group2}" for _, r in t.iterrows() if r.reject]
        print("significant pairs:", ", ".join(rej) or "none")
        print("same significant pairs as Nemenyi (E7):", sorted(rej) == sorted(sig))

    section("E9", "Top-5 per-horizon MAE (for the Chronos-2 sentence in the results text)")
    names = {"chronos2_fine_tuned@local": "FT", "chronos2_zero_shot@local": "ZS",
             "catboost_tweedie@global": "CatBoost", "lightgbm_poisson@global": "LightGBM",
             "arima@global": "ARIMA"}
    T = pd.DataFrame({v: per[k]["MAE"] for k, v in names.items()})
    print(T.round(3).to_string())
    others = T.drop(columns="FT")
    print("FT lowest at every horizon:", bool((T.idxmin(axis=1) == "FT").all()))
    print("gap to next-best model:", (others.min(axis=1) - T.FT).round(3).tolist())
    print("gap to CatBoost-Tweedie:", (T.CatBoost - T.FT).round(3).tolist())
    print("max-min over h1..h7 per model:", (T.max() - T.min()).round(3).to_dict())


if __name__ == "__main__":
    main()
