# Improved horizon test and text (proposal)

Date: 2026-09-29 · Repo commit: `91de6ca` · Status: **proposal only**. Nothing in `main.tex` or the
pipeline has been changed.

## How to verify this document

Every factual statement carries one or more evidence tags. They resolve as follows:

| Tag | Where to check |
|---|---|
| **[E0]…[E9]** | Sections of `horizon_tests_output.txt` in this folder, the saved output of `horizon_tests.py`. Re-run with `uv run python docs/audits/2026-09-29/horizon_tests.py`. The script only **reads** `runs_publication_20260929/` (seed-42 publication run); nothing is trained or written. |
| **[C file:line]** | A line in the code repo (`/Users/jan/projects/bsc_thesis_code/thesis`). |
| **[T line]** | A line in `/Users/jan/projects/writing/thesis_writing_folder/main.tex` (as of 2026-09-29). |
| **[F path]** | A result file in the run store. |
| **[L1]…[L5]** | Literature, listed with quotes and page numbers in §10. |
| **[X]** | Citation counts from Crossref or OpenAlex, retrieved 2026-09-29, with query URLs in §11. |

Numbers are for seed 42 only. They are placeholders until the seed sweep (§9).

---

## 0. Short answer: how much does this change?

| | Old test (unpaired) | New test (paired) | Does the conclusion change? |
|---|---|---|---|
| **RMSE** | Kruskal-Wallis H(6) = 37.24, p = 1.6e-6 [E3-RMSE] | Friedman χ²(6) = 48.04, p = 1.2e-8 [E5-RMSE] | **No.** The unpaired Tukey HSD and the paired Nemenyi flag *exactly the same 8 horizon pairs*: h1–h6, h2–h5, h2–h6, h2–h7, h3–h5, h3–h6, h3–h7, h4–h6 [E7-RMSE, E8-RMSE: "same significant pairs as Nemenyi (E7): True"]. |
| **MAE** | Kruskal-Wallis H(6) = 3.20, p = 0.784 [E3-MAE] | Friedman χ²(6) = 33.09, p = 1.0e-5 [E5-MAE] | **The statistical verdict flips; the practical meaning barely changes.** Mean MAE per horizon spans 0.7773–0.7834, a 0.79% spread [E2-MAE]. |

- **More valid:** the test now matches the design (§2).
- **More powerful:** it finds a consistent within-model change that the unpaired test cannot see. For MAE, Tukey finds no significant pair, while Nemenyi finds six [E8-MAE, E7-MAE].
- **Same substance:** the RMSE conclusion is unchanged, and for MAE the message "typical-day accuracy is practically stable" survives.
- **What becomes wrong:** the sentence "MAE shows no significant variation" [T 1143].
- **What the change restores:** post-hoc evidence for RMSE, which the current pipeline no longer produces (§2, reason 3).

---

## 1. What is done now

1. The top-20 configurations of the master leaderboard are taken [C src/strikecast/reporting/build.py:357 `top20 = ctx.top(20)`], giving the 20 rows listed in [E1].
2. Their per-horizon RMSE and MAE form 7 groups of 20 values [C src/strikecast/reporting/compute.py:251–268], treated as independent groups.
3. Shapiro-Wilk is run per horizon; the loop stops at the first failing horizon [C compute.py:270–277]. Levene follows [C compute.py:280].
4. ANOVA plus Tukey HSD runs if both checks pass, otherwise Kruskal-Wallis [C compute.py:292 `test="anova" if (normality_passed and variance_passed) else "kruskal"`]. Tukey runs **only** on the ANOVA route [C compute.py:298–301].
5. The thesis describes this procedure in [T 826] and reports it in [T 1079–1097] and [T 1143].

## 2. Why it needs to change

### Reason 1: the 7 groups are the same 20 configurations (a repeated-measures design)

- **Fact:** the same 20 configuration files feed every horizon column [E1, E2-RMSE, E2-MAE]. Each row of the matrix is one configuration measured at all 7 horizons.
- **Standard practice:** for "more than two related sample means", meaning the same units measured under every condition, the appropriate method is **repeated-measures** ANOVA, or its rank-based counterpart, the Friedman test, not a one-way test [L1 §3.2.1, p. 10; §3.2.2, p. 11].
- **Consequence, shown numerically:** a one-way (unpaired) test compares the horizon variation against the between-configuration variation plus the residual. A paired test compares it against the residual only [E4].

  | | Between configurations | Between horizons | Residual | Source |
  |---|---|---|---|---|
  | RMSE | 34.8% | 24.3% | 40.9% | [E4-RMSE] |
  | MAE | **97.3%** | 0.6% | 2.0% | [E4-MAE] |

  For MAE, 97.3% of all variation is *between configurations*. In the unpaired test this sits in the error term and drowns the horizon effect. That is why Kruskal-Wallis gives p = 0.784 [E3-MAE] while Friedman gives p = 1.0e-5 [E5-MAE] on identical data.

### Reason 2: the research question has a direction

- Sub-Q2 is about error *growing* with lead time [T 1143, "degrades significantly beyond the four-day mark"].
- One-way ANOVA, Kruskal-Wallis and Friedman all test only whether the groups are equal, with a non-directional alternative.
- Page's L test is built for the ordered alternative. Its null is m₁ = m₂ = … = mₙ, against the alternative m₁ ≤ m₂ ≤ … ≤ mₙ with at least one inequality strict, for data arranged as "subject i × treatment j" [L2, SciPy documentation quote in §10].

### Reason 3: on the rerun data, the current pipeline produces no post-hoc test for RMSE

- For RMSE, Levene p = 0.0106 < 0.05, so `variance_passed = False` [E3-RMSE]. The stored result says `test=kruskal` [F runs_publication_20260929/_figures/tab_horizon_stats.csv, row RMSE; reproduced in E3-RMSE].
- Tukey runs only on the ANOVA route [C compute.py:298]. There is therefore **no post-hoc result** behind the claim "Day 5 and later significantly worse than Day 3" [T 1090–1096, T 1143] in the publication run.
- The thesis numbers (Levene p ≈ 0.54, ANOVA F = 6.70) [T 1088–1090] came from the thesis run. The rerun differs: Levene p = 0.0106, F = 7.124 [E3-RMSE].

### Side issue: pretesting

- The Shapiro/Levene routing is a *two-stage procedure*: a pretest decides which test to use. Zimmerman (2004) found, for a Levene pretest followed by a t test or Welch test, that "the two-stage procedure fails to protect the significance level and usually makes the situation worse" [L3]. That study covers the two-sample case, not ours, so treat it as supporting context only.
- The proposed tests are rank-based and need neither a normality nor a variance pretest [L1 abstract: "simple, yet safe and robust non-parametric tests"].

## 3. Proposed procedure

Blocks = 20 configurations, treatments = 7 horizons, α = 0.05.

| Step | Test | What it answers | Evidence it is appropriate |
|---|---|---|---|
| 1 | **Friedman test**, χ²(k−1) | Do the horizons differ, with each configuration compared only with itself? | Recommended for comparing several methods over multiple data sets [L1 abstract; §3.2.2]. The χ² approximation needs "N > 10 and k > 5" [L1 p. 11]; here N = 20 and k = 7. |
| 1b | **Kendall's W** = χ² / (N(k−1)) | Effect size, from 0 (no agreement) to 1 (complete agreement) | W is "a normalization of the statistic of the Friedman test" and "ranges from 0 (no agreement) to 1 (complete agreement)" [L5]. The identity is checked numerically against W's own definition [E5: "identical: True"]. |
| 2 | **Page's L test** | Does error *increase* with lead time? | Test for ordered alternatives in the subject × treatment layout [L2]. |
| 3 | **Wilcoxon signed-rank** tests: each horizon vs Day 1 (6) and each day vs the previous day (5 more; h2 vs h1 is already among the first six), 11 in total | Which horizons differ? | Recommended for comparing two methods over multiple data sets [L1 abstract; §3.1.3, p. 7]. |
| 3b | **Holm correction** over those 11 p-values | Controls the family-wise error across the 11 tests | Described in [L1 pp. 12–13]: "more powerful than the Bonferroni-Dunn's and makes no additional assumptions about the hypotheses tested"; the procedures "can be used generally for controlling the family-wise error when multiple hypotheses of possibly various types are tested". |
| 4 (optional) | **Nemenyi** post-hoc with CD | Same CD logic as your model CD diagrams | [L1 p. 11]; you already use it [T 828–832]. |
| – | **Drop** Shapiro, Levene, one-way ANOVA, Kruskal-Wallis and Tukey for this analysis | | Reasons 1–3 above. |

**How our setting maps onto Demšar's:** his "classifiers" are our horizons, and his "data sets" (the blocks) are our configurations. Demšar assumes the data sets are independent: "running the algorithms on multiple data sets naturally gives a sample of independent measurements" [L1 p. 5]. Our configurations are not fully independent; see §5.

## 4. Effect on the results (seed 42, top 20)

| | Thesis text | Pipeline rerun, current test | **Proposed** |
|---|---|---|---|
| RMSE omnibus | ANOVA F = 6.70, p < 0.001 [T 1090] | KW H(6) = 37.24, p = 1.58e-6; Levene p = 0.0106, so no post-hoc [E3-RMSE] | Friedman χ²(6) = 48.04, p = 1.16e-8, W = 0.400; Page L = 2535, p = 3.95e-9 [E5-RMSE] |
| RMSE post-hoc | Tukey: Day 5 > Day 3; Days 6 and 7 > Days 1–4 [T 1090–1096] | none [C compute.py:298] | Day 5 > Day 4 in 19/20 (Holm p = 0.0053); Day 6 > Day 5 in 17/20 (Holm p = 0.0053); Day 6 > Day 1 in 17/20 (Holm p = 0.033); Day 7 vs Day 6 not significant (Holm p = 0.54) [E6-RMSE] |
| MAE omnibus | KW H = 5.49, p ≈ 0.05 [T 1080–1081] | KW H(6) = 3.20, p = 0.784 [E3-MAE] | Friedman χ²(6) = 33.09, p = 1.01e-5, W = 0.276; Page L = 2432, p = 8.64e-5 [E5-MAE] |
| MAE post-hoc | – | – | Only Day 5 > Day 4 is significant: 17/20 configurations, Holm p = 0.0094 [E6-MAE] |
| Size of change | – | – | Mean RMSE is 1.8527 on Day 3 and 1.8858 on Day 6, a 1.79% spread [E2-RMSE]. Mean MAE spans 0.7773–0.7834, a 0.79% spread [E2-MAE]. |
| Easiest and hardest day | "Days 1 and 7 are neither the easiest nor the hardest" [T 1096] | – | Still true. The lowest mean Friedman rank is h3 for both metrics (RMSE 2.45, MAE 2.60); the highest is h6 for RMSE (6.00) and h5 for MAE (5.30) [E5]. |

## 5. How reliable is the new outcome?

- **Validity improves.** The test matches the repeated-measures design [Reason 1; L1 §3.2.1].
- **Power improves, but it is not a different answer for RMSE.** RMSE significant pairs are identical under both approaches [E7-RMSE, E8-RMSE]. For MAE, the paired test detects the day-4-to-day-5 step, which is consistent across configurations (17/20) but small (median +0.59%) [E6-MAE].
- **Not fixed by this change**; put these in one limitation sentence:
  1. **Single seed:** every input file is from `seed=42` [E1, file paths].
  2. **Configurations are not independent units:** several share an architecture across paradigms. For example, `lightgbm_poisson`, `xgboost_poisson`, `catboost_tweedie` and `gru_poisson_w28` each appear twice in the top 20 [E1]. Demšar's framework assumes independent blocks [L1 p. 5]. The current unpaired test makes the same assumption, so this is not a new weakness.
  3. **One test period:** all configurations are scored on the same test window.

## 6. Standard or niche? How well known is each test?

Citation counts are a rough proxy only; Crossref and OpenAlex usually show fewer citations than Google Scholar, and old tests are often cited through textbooks instead of the original paper. Page numbers are given as far as Crossref confirms them ("p. 99 ff." = only the start page is confirmed).

| Test | Original source | Citations [X] | Verdict | Why |
|---|---|---|---|---|
| Friedman test | Friedman (1937), *JASA* 32, 675–701 | Crossref 7,565 | **Standard** | Highly cited; recommended for comparing multiple methods in ML [L1 abstract]; you already use it [T 828–832]. |
| Wilcoxon signed-rank | Wilcoxon (1945), *Biometrics Bulletin* 1, p. 80 ff. | Crossref 13,100 | **Standard** | Highly cited; recommended for pairwise comparisons [L1 abstract, §3.1.3]. |
| Holm correction | Holm (1979), *Scandinavian Journal of Statistics* | OpenAlex 21,862 | **Standard** | Highly cited; described in [L1 pp. 12–13] as more powerful than Bonferroni-Dunn with no extra assumptions. |
| Kendall's W | Kendall & Babington Smith (1939), *Ann. Math. Stat.* 10, 275–287 | Crossref 771 | **Standard** (as the effect size for Friedman) | A normalisation of the Friedman statistic [L5]; common in reporting practice. |
| Page's L trend test | Page (1963), *JASA* 58, 216–230 | Crossref 548 | **Established but specialist** | About 10–25× fewer citations than Friedman or Wilcoxon. It is in SciPy (`scipy.stats.page_trend_test`) and textbooks (the SciPy docs cite Neuhäuser 2012, pp. 150–152) [L2]. Statisticians know it; many ML readers will not, so **cite it if you use it**. |
| Nemenyi post-hoc | Nemenyi (1963), PhD thesis, Princeton | – (unpublished thesis) | **Standard in ML benchmarking** | Popularised by Demšar's CD diagrams [L1 p. 11]. Criticised by Benavoli et al. (2016) [L4], hence optional. |
| *Current:* Kruskal-Wallis | Kruskal & Wallis (1952), *JASA* 47, 583–621 | Crossref 14,076 | Standard | Correct test, wrong design (unpaired) for this data. |
| *Current:* Tukey HSD | Tukey (1949), *Biometrics* 5, p. 99 ff. | Crossref 3,594 | Standard | As above. |
| *Current:* Shapiro-Wilk | Shapiro & Wilk (1965), *Biometrika* 52, 591–611 | Crossref 14,882 | Standard | Not needed once the tests are rank-based. |

**Bottom line:** everything in the core proposal (Friedman, Wilcoxon, Holm, W) is mainstream. The only
less common element is Page's test, which is optional (§7, option B).

## 7. Citations: what to cite with a limited budget

**Fact:** `DemsarNemenyiFriedman` is already in `references.bib` and cited at [T 828] and [T 832].
The paper recommends Friedman and Wilcoxon [L1 abstract] and describes Holm [L1 pp. 12–13].
**One existing citation covers steps 1, 3, 3b and 4.**

**Fact:** `navarro2015learning` is cited only at [T 826], the Shapiro/Levene paragraph that this
proposal replaces. Check it yourself with `grep -n navarro2015learning main.tex`, which returns only line 826.

| Option | New citations | Net change | What to do |
|---|---|---|---|
| **A (recommended)** | Page (1963) | **0** | Cite Demšar for Friedman, Wilcoxon and Holm, and Page for the trend test. Remove Navarro. |
| B (tightest) | none | **−1** | Leave out Page's test. The Friedman test plus the adjacent-day Wilcoxon tests carry "rises after Day 4" on their own [E6]. Remove Navarro. |
| C | none | 0 | Keep Page's test without a citation. **Not advised:** it is a specialist test (§6). |

Do not spend citations on Friedman (1937), Wilcoxon (1945), Holm (1979) or Kendall (1939) separately;
Demšar covers the procedures. Your writing `CLAUDE.md` says only entries in `references.bib` may be
cited, so any new entry is added by you.

BibTeX for option A. The metadata was checked against Crossref [X, DOI 10.1080/01621459.1963.10500843]:
author Ellis Batten Page; *JASA* 58(301), 216–230, 1963.

```bibtex
@article{PageTrendTest1963,
  author  = {Page, Ellis Batten},
  title   = {Ordered Hypotheses for Multiple Treatments: A Significance Test for Linear Ranks},
  journal = {Journal of the American Statistical Association},
  volume  = {58},
  number  = {301},
  pages   = {216--230},
  year    = {1963},
  doi     = {10.1080/01621459.1963.10500843}
}
```

## 8. How to report

These are recommendations for consistency and verifiability, not facts to prove. They follow the
statistic/df/p/effect-size convention and keep the thesis's current number style (`$p < 0.001$`,
leading zero; see [T 970–979]).

- **Name the unit once:** "N = 20 configurations, each evaluated at all seven horizons (repeated measures)".
- **Omnibus:** statistic, df, p and effect size, e.g. `Friedman $\chi^2(6) = 48.0$, $p < 0.001$, Kendall's $W = 0.40$` [E5-RMSE].
- **Trend (option A):** `Page's $L = 2535$, $p < 0.001$` [E5-RMSE].
- **Post-hoc:** say once "Wilcoxon signed-rank, Holm-adjusted", then give the adjusted p together with the count and median change, e.g. "Day 5 exceeds Day 4 in 19 of 20 configurations (median +0.73%, $p = 0.005$)" [E6-RMSE].
- **Keep statistical and practical significance apart:** "significant but below 1%" [E2-MAE, E6-MAE] is a finding; "not significant" does not mean "stable".
- **Report one test family only.** Don't report ANOVA and Friedman side by side.
- **Optional:** Demšar notes that Friedman's χ² is "undesirably conservative" and gives the Iman-Davenport F [L1 p. 11]. Values are in [E5] (RMSE F(6,114) = 12.69; MAE F(6,114) = 7.23). Reporting χ² is enough.
- **Optional appendix table:** the 11 post-hoc rows of [E6] (comparison, configurations worse / 20, median % change, Holm p).

## 9. Proposed text for main.tex

### 9a. Methods: replace [T 826]

> Differences across forecast horizons are assessed on the twenty best configurations. Because every configuration is evaluated at all seven lead times, horizons are treated as repeated measures: the Friedman test ranks the seven horizons within each configuration and tests the equality of mean ranks, and Page's $L$ test evaluates the ordered alternative that error increases with lead time \citep{PageTrendTest1963}. Where the Friedman test rejects, Wilcoxon signed-rank tests compare each horizon with Day~1 and with the preceding day, with Holm's step-down correction over these eleven comparisons ($\alpha = 0.05$) \citep{DemsarNemenyiFriedman}. Being rank-based, these tests require neither normality nor equal variances.

(Option B: delete the Page clause and its citation.)

### 9b. Methods: [T 829–830]

"Following the same assumption check used for the horizon analysis, the rank-based Friedman test is
applied where the blocked-ANOVA normality assumption is not met." After 9a, "the same assumption
check" no longer exists. Replace with: "As in the horizon analysis, the rank-based Friedman test is used."

### 9c. Results: replace [T 1079–1097] ("The horizon-wise MAE distribution fails the" … "the ranking.")

> Across the twenty best configurations, RMSE differs between horizons (Friedman $\chi^2(6) = 48.0$, $p < 0.001$, Kendall's $W = 0.40$) and increases with lead time (Page's $L = 2535$, $p < 0.001$). Mean RMSE is lowest on Day~3 ($1.853$) and highest on Day~6 ($1.886$). The increase sets in after Day~4: Day~5 exceeds Day~4 in 19 of 20 configurations (Wilcoxon signed-rank, Holm-adjusted $p = 0.005$) and Day~6 exceeds Day~5 in 17 of 20 ($p = 0.005$); Day~6 is also worse than Day~1 ($p = 0.033$), whereas Day~7 does not differ from Day~6 (Figure~\ref{fig:rmse_horizon_boxplot}).
>
> MAE also varies across horizons (Friedman $\chi^2(6) = 33.1$, $p < 0.001$, $W = 0.28$; Page's $L = 2432$, $p < 0.001$), but the variation is small: mean MAE stays between $0.777$ and $0.783$ over all seven days, and the only significant step between consecutive days is from Day~4 to Day~5 (17 of 20 configurations, $p = 0.009$). Among the top five configurations, MAE varies by no more than $0.02$ between Day~1 and Day~7 (Figure~\ref{fig:top_5_horizon}, bottom panel); the fine-tuned Chronos-2 attains the lowest MAE at every horizon, $0.06$--$0.08$ below the best non-foundation model. For both metrics, Days~1 and~7 are neither the easiest nor the hardest horizons: error is lowest on Days~2--3.

Evidence for each number:

| Number in 9c | Source |
|---|---|
| 48.0, p < 0.001, W = 0.40, L = 2535 | [E5-RMSE] |
| 1.853 (Day 3), 1.886 (Day 6) | column means in [E2-RMSE] |
| 19 of 20 and p = 0.005 (h5 vs h4); 17 of 20 and p = 0.005 (h6 vs h5); p = 0.033 (h6 vs h1); h7 vs h6 not significant | [E6-RMSE] |
| 33.1, W = 0.28, L = 2432 | [E5-MAE] |
| 0.777–0.783 | column means in [E2-MAE] |
| 17 of 20, p = 0.009 | [E6-MAE] |
| Top-5 MAE varies ≤ 0.02 | max − min per model is at most 0.016 [E9] |
| FT lowest at every horizon; 0.06–0.08 below CatBoost-Tweedie | [E9]: "FT lowest at every horizon: True"; "gap to CatBoost-Tweedie: 0.060–0.079" |
| "lowest on Days 2–3" | mean ranks, lowest at h3 for both metrics, h2 second [E5] |

**Correction this exposes:** the thesis says Chronos-2 is "between 0.06 and 0.08 below the *next-best*
model" [T 1084–1085]. In the rerun, the next-best model at every horizon is zero-shot Chronos-2, only
0.001–0.010 behind [E9, "gap to next-best model"]. Hence the new wording "best non-foundation model".

### 9d. Discussion: replace the first three sentences of [T 1143] (up to "…significantly worse than Day~3).")

> For Sub-Q2, forecast error increases with the horizon, but by very different amounts for the two metrics. Both show a consistent step after Day~4 (repeated-measures Friedman and Page tests, $p < 0.001$), yet mean MAE varies by less than 1\% across the week, whereas RMSE rises by about 1.8\% from Day~3 to Day~6 and continues to deteriorate after the first step.

Evidence: Friedman and Page p < 0.001 for both metrics [E5]; the step after Day 4 for both metrics
[E6-RMSE, E6-MAE]; below 1% for MAE (0.79%) [E2-MAE]; about 1.8% for RMSE (1.79%) [E2-RMSE];
continues after the first step (h6 vs h5 significant for RMSE only) [E6].

Keep "Because RMSE penalises large errors most, …" and the rest of the paragraph. Add one limitation
sentence (§5):

> These horizon tests treat the twenty configurations as independent units, although several share an architecture, and rest on a single training seed.

### 9e. Abstract [T 84]

"RMSE degrades significantly beyond a four-day horizon while mean absolute error remains stable"
→ "forecast error rises significantly beyond a four-day horizon, markedly for RMSE and negligibly (below 1\%) for mean absolute error"

### 9f. Summary [T 134]

"Typical-day forecast error remains stable across the horizon while RMSE degrades significantly beyond four days."
→ "Typical-day forecast error remains practically stable across the horizon (MAE varies by less than 1\%), while RMSE degrades significantly beyond four days."

### 9g. Conclusion [T 1163]

"While forecast quality for typical days remains stable across a seven-day horizon, …"
→ "While forecast quality for typical days remains practically stable across a seven-day horizon, …"

## 10. Literature evidence (quotes you can check)

**[L1] Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR* 7, 1–30.**
Already in your bib as `DemsarNemenyiFriedman`. PDF: https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
- *Abstract (p. 1):* "we recommend a set of simple, yet safe and robust non-parametric tests for statistical comparisons of classifiers: the Wilcoxon signed ranks test for comparison of two classifiers and the Friedman test with the corresponding post-hoc tests for comparison of more classifiers over multiple data sets."
- *§3.2.1, p. 10:* "The common statistical method for testing the differences between more than two related sample means is the repeated-measures ANOVA (or within-subjects ANOVA) … The 'related samples' are again the performances of the classifiers measured across the same data sets".
- *§3.2.2, p. 11:* the Friedman statistic "is distributed according to χ²_F with k − 1 degrees of freedom, when N and k are big enough (as a rule of a thumb, N > 10 and k > 5)"; "Iman and Davenport (1980) showed that Friedman's χ²_F is undesirably conservative and derived a better statistic F_F"; "The Nemenyi test (Nemenyi, 1963) is similar to the Tukey test for ANOVA and is used when all classifiers are compared to each other."
- *§3.2.2, pp. 12–13:* "Holm's step-down procedure starts with the most significant p value …"; "Holm's procedure is more powerful than the Bonferroni-Dunn's and makes no additional assumptions about the hypotheses tested"; "they can be used generally for controlling the family-wise error when multiple hypotheses of possibly various types are tested."
- *§3.1.3, p. 7:* "The Wilcoxon signed-ranks test (Wilcoxon, 1945) is a non-parametric alternative to the paired t-test".
- *p. 5 (independence assumption):* "running the algorithms on multiple data sets naturally gives a sample of independent measurements".

**[L2] Page, E. B. (1963).** Ordered hypotheses for multiple treatments: a significance test for linear ranks. *JASA* 58(301), 216–230. doi:10.1080/01621459.1963.10500843 (metadata checked against Crossref).
SciPy documentation for `scipy.stats.page_trend_test` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.page_trend_test.html):
the null is "m₁ = m₂ = m₃ ⋯ = mₙ", the alternative "m₁ ≤ m₂ ≤ m₃ ≤ ⋯ ≤ mₙ, where at least one inequality is strict", and the input is "a m × n array; the element in row i and column j is the observation corresponding with subject i and treatment j". Its reference list includes Page (1963) and Neuhäuser, *Nonparametric Statistical Tests: A Computational Approach*, CRC Press, 2012, pp. 150–152.

**[L3] Zimmerman, D. W. (2004).** A note on preliminary tests of equality of variances. *British Journal of Mathematical and Statistical Psychology* 57, 173–181. doi:10.1348/000711004849222.
Abstract: "Preliminary tests of equality of variances used before a test of location are no longer widely recommended by statisticians … Simulations disclosed that the two-stage procedure fails to protect the significance level and usually makes the situation worse." (Two-sample t/Welch setting.)

**[L4] Benavoli, A., Corani, G., & Mangili, F. (2016).** Should we really use post-hoc tests based on mean-ranks? *JMLR* 17. Listed in OpenAlex; the arXiv version has 294 citations [X]. This is the source of the caveat on Nemenyi.

**[L5] Wikipedia, "Kendall's W"** (https://en.wikipedia.org/wiki/Kendall%27s_W): "It is a normalization of the statistic of the Friedman test"; "Kendall's W ranges from 0 (no agreement) to 1 (complete agreement)." The exact identity W = χ²/(N(k−1)) is verified numerically against W's definition in [E5].

## 11. Citation-count sources [X]

Retrieved 2026-09-29. Re-run a URL to check it.

- Crossref, `https://api.crossref.org/works/<DOI>`, field `is-referenced-by-count`:
  - Friedman 1937: 10.1080/01621459.1937.10503522 → 7,565
  - Wilcoxon 1945: 10.2307/3001968 → 13,100
  - Kruskal & Wallis 1952: 10.1080/01621459.1952.10483441 → 14,076
  - Tukey 1949: 10.2307/3001913 → 3,594
  - Kendall & Babington Smith 1939: 10.1214/aoms/1177732186 → 771
  - Page 1963: 10.1080/01621459.1963.10500843 → 548
  - Shapiro & Wilk 1965: 10.1093/biomet/52.3-4.591 → 14,882
  - Zimmerman 2004: 10.1348/000711004849222 → 248
- OpenAlex, `https://api.openalex.org/works?search=<title>`, field `cited_by_count`:
  - Holm 1979, "A Simple Sequentially Rejective Multiple Test Procedure" → 21,862
  - Demšar 2006 → 11,157 (the main record; a second JMLR record has 1,052)
  - Benavoli et al., arXiv 2015 → 294; JMLR 2016 record → 14
  - Page 1963 (OpenAlex; for comparison with Crossref) → 685
- Nemenyi (1963) is an unpublished Princeton PhD thesis and has no reliable count.

## 12. Actions for Jan

- [ ] **Verify.** Run `uv run python docs/audits/2026-09-29/horizon_tests.py` and compare with `horizon_tests_output.txt`. Open the Demšar PDF at the pages in §10.
- [ ] **Decide on citations:** option A (swap Navarro for Page, net 0) or B (no Page, net −1). See §7.
- [ ] If A: add the Page BibTeX (§7) to `references.bib`.
- [ ] Replace the methods paragraph (§9a) and fix [T 829–830] (§9b).
- [ ] Replace the horizon results [T 1079–1097] (§9c). This removes the ANOVA, Shapiro, Levene and Tukey numbers and fixes the "next-best model" sentence.
- [ ] Update the Discussion (§9d), abstract (§9e), summary (§9f) and conclusion (§9g). Search for any other "MAE … stable / no significant variation" phrasing.
- [ ] Decide on the optional appendix table of the 11 post-hoc comparisons (§8).
- [ ] Ask Claude to implement the paired test in the pipeline behind a `legacy` switch (`compute.py::horizon_statistics` and `build.py::build_horizon_stats`, writing Friedman, Page, Kendall's W and the Holm table to `tab_horizon_stats` and `numbers.tex`), so that §9's numbers regenerate after the seed sweep instead of being copied by hand.
- [ ] After the seed sweep, replace every number in §9. The seed-42 values are placeholders.
