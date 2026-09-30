# Recommendations section for the chapter draft (proposal)

Date: 2026-09-30 · Target: `\section{Recommendations}` in `/Users/jan/projects/writing/chapter?/main.tex`
(currently a stub: "start with basic autoregressive models first; chronos-2 has potential if goal is min.
MAE; signals"). **Proposal only; nothing in the writing folder was changed.**

All numbers: seed-42 publication run `runs_publication_20260929`, test window 2024-08-05 to 2025-01-21,
from stored predictions only (nothing retrained). Tags [A]-[I] point to §5. Numbers are placeholders
until the seed sweep; the recommendations that matter most (R4, R5) rest on large, consistent gaps,
not on leaderboard order.

---

## 0. Short answer

Audience from the chapter itself: "defense data science and developers of early warning systems";
decisions named in the intro: counter-UAS allocation, civil protection, personnel positioning;
contribution 4 promises "recommendations for analysts and tool developers".

| # | Recommendation | Core evidence |
|---|---|---|
| R1 | Benchmark every system against ARIMA/seasonal naive and zero-shot Chronos-2 first | ARIMA = 95% of the best Skill Score (0.159 vs 0.167), #5 overall; Chronos-2 zero-shot #3 without training [A] |
| R2 | One pooled model, count likelihood, single stage | Global > Local (best 1.845 vs 1.890 RMSE); hurdle Skill 0.09, differenced-MSE ≤ 0.07, differenced RNNs below seasonal naive [A] |
| R3 | Choose model by decision and region; report per tier | Tier 3 = 94% of pooled squared error, Sumy alone 56% [B]; Chronos-2 wins MAE and Tier 3, trees win Tier-2 RMSE [B] |
| R4 | Use forecasts as a rolling intensity level per oblast, not as day-specific surge warnings | Surge days forecast at 41-52% of the realised count from Day 1 on [C]; within-region surge AUC 0.62-0.69 [D]; cross-region ranking barely beats a static map [F] |
| R5 | Invest in low-latency strike data, weather forecasts and structured platform labels before broad OSINT signals | AR + static dominate trees, weather leads Chronos-2, political/economic signals ≤ 8% [H]; horizon profile flat, so it doubles as a latency budget [E] |
| R6 | Evaluate as deployed: forecast weather, frequent refits, several seeds, per-tier reporting, independent checks of temporal patterns | Observed-weather lookahead; top 5 within 0.018 RMSE; ACLED weekend dip absent in Air Force launch counts [I] |

---

## 1. What the analysis found that the chapter does not say yet

1. **Surges are under-forecast at every lead time, not only beyond Day 4/5.** On region-days with ≥ 8
   strikes (the chapter's own "> 7" threshold; 104 region-dates in 5 oblasts, mean 12.1 strikes), the
   leading models forecast 41-52% of the realised count at **every** horizon, Day 1 included [C]. Their
   surge-day RMSE barely moves with the horizon (Chronos-2 FT 7.89 → 8.05, CatBoost 8.48 → 8.33,
   ARIMA 7.42 → 7.40) [E]. The pooled top-20 RMSE rises only 1.2% from Days 1-4 to Days 5-7 [E].
   The significant Day-5+ increase is real but small next to the Day-1 gap.
2. **The thesis split "Chronos for typical days, trees for high-intensity days" does not hold in the new run.**
   On surge days ARIMA has the lowest RMSE (7.4-7.7), then Chronos-2 FT (7.8-8.1), LightGBM
   (8.1-8.5), CatBoost (8.2-8.5). Trees win the *typical* days (non-surge RMSE: CatBoost 1.14-1.15,
   Chronos-2 1.21-1.24, ARIMA 1.31-1.35) [E]. Their block-wise RMSE lead comes from the many
   low/medium-activity cells, not from surges.
3. **"Where" is mostly a static map.** Across the 20 oblasts on a given day, the models' ranking
   correlates with the realised ranking at Spearman 0.65-0.67, against 0.64-0.65 for the pre-test mean
   per oblast; top-3 overlap 0.59-0.63 vs 0.58 [F]. The models add level tracking (non-surge RMSE 1.15
   vs 1.39 for the static map) but little day-specific targeting information.
4. **Weekly rhythm may be partly a reporting rhythm.** In the test window, ACLED strike counts fall on
   Saturday (0.59 of the weekly mean) and Sunday (0.85), peaking Wednesday/Friday (1.23). Ukrainian Air
   Force launch counts (`act_confirmed_launched`) show no weekend dip (Sat 1.02, Sun 1.17); full window:
   ACLED Sun 0.75, launches 0.94-1.07 on all days [I]. Caveat: launches are national, long-range drones
   and may be dated to the morning report, while the ACLED target also contains tactical non-FPV drones.
   Enough to soften "operations follow a weekly rhythm" in the Conclusion stub, not enough to reject it.

---

## 2. Proposed LaTeX

```latex
\section{Recommendations}
The results translate into six recommendations for analysts who use strike forecasts and for developers of early-warning tools.

\textbf{Benchmark against simple and pre-trained baselines first.} A univariate ARIMA reaches 95\% of the best model's Skill Score (0.159 versus 0.167) and ranks fifth overall; the zero-shot Chronos-2, which requires no task-specific training, ranks third. An added covariate pipeline or a trained deep model is justified only by a demonstrated, block-wise improvement over these baselines, per region and horizon, not by a pooled score alone.

\textbf{Pool the regions and model the counts directly.} Global models trained across all oblasts outperform per-region models (best RMSE 1.845 versus 1.890), and single-stage Poisson or Tweedie objectives outperform both the two-stage hurdle (Skill Score 0.09) and MSE on the differenced target (at most 0.07). The simpler architecture is also the more accurate one.

\textbf{Match the model to the decision and the region.} A pooled error is effectively a front-line error: the six Tier-3 oblasts carry 94\% of the squared error, Sumy alone 56\%. The fine-tuned Chronos-2 gives the most accurate typical-day forecasts (MAE 0.71, against 0.76 for the best tree model; in Tier~1 about half the tree models' MAE) and is marginally best in Tier~3, whereas gradient-boosted trees are more accurate in the medium-activity Tier-2 oblasts (RMSE 0.88 versus 0.92). Accuracy should therefore be reported per activity tier and weighted by the operational relevance of each region rather than by its share of the pooled error.

\textbf{Treat forecasts as intensity levels, not surge warnings.} On region-days with more than seven strikes, the leading models forecast only 41--52\% of the realised count, already at a one-day lead, and rank a region's surge days only modestly above its other days (within-region ROC-AUC 0.62--0.69). Across oblasts, their daily ranking barely improves on a static map of past intensity. The forecasts are thus best used as a rolling estimate of each oblast's strike pressure over the coming week, which supports the allocation of counter-UAS assets, but not as a trigger for civil-protection alerts on a specific day. Anticipating surges requires a dedicated formulation: probabilistic forecasts, such as Chronos-2 quantiles or the full Tweedie distribution, evaluated on exceedance probabilities rather than on the conditional mean.

\textbf{Invest in timely event data before broad signal collection.} Recent strike history and static regional context carry most of the tree models' importance, and weather leads the covariates of Chronos-2, whereas diplomatic, cyber, nuclear, aid and macroeconomic signals contribute little at daily oblast resolution; this does not rule out a causal role at other scales. The binding constraint is latency: the forecasts depend on the previous days' counts, whereas ACLED releases its full archive with a one-year delay. Because each forecast is issued from the last observed day, the horizon profile doubles as a latency budget: a feed that arrives $k$ days late turns a Day-$h$ forecast into a Day-$(h{+}k)$ forecast, which the nearly flat horizon profile tolerates for a few days. Event data that code the weapon platform in a structured field, rather than in free-text notes, would remove the FPV label noise of this study.

\textbf{Evaluate as the system will be deployed.} The reported accuracy is an upper bound. The models saw observed rather than forecast weather over the seven-day lookahead, which favours the weather-driven Chronos-2 most; the process is non-stationary, with the most intense period in the test window, which calls for frequent refits; and the leading configurations lie within 0.018 RMSE of each other, a margin that should be tested over repeated training runs [seed sweep: add SD] before one model family is preferred. Apparent temporal regularities deserve an independent check as well: the weekend dip in ACLED strike counts is absent from the Ukrainian Air Force's daily launch reports, so part of the weekly rhythm may reflect reporting rather than operations.
```

Length: ~560 words. To shorten, merge R1 + R2 ("Start simple") and drop the last sentence of R6.

Style notes: follows the run-in `\textbf{}` headings of the chapter's methodology stub; adds no new
citations. Optional citations already in `references.bib`: `\citet{Bazzi2022}` after "not as a trigger ... on
a specific day" ("where" more tractable than "when"), `\citep{counterUASWeather}` after "weather leads
the covariates of Chronos-2".

---

## 3. Knock-on edits elsewhere in the chapter (not made)

| Where | Current text | Issue | Suggestion |
|---|---|---|---|
| Intro ¶2 | "the ability to anticipate high-intensity events degrades significantly beyond five days" | §1.1: surges are under-forecast from Day 1; the Day-5+ increase is small (+1.2%). Also says five, contribution 3 says four. | "typical daily-count accuracy remains stable across a seven-day window, while high-intensity days are under-forecast by about half at every lead time" |
| Contribution 3 | "the loss of skill on high-intensity days beyond a four-day horizon" | same | "the systematic under-forecasting of high-intensity days" |
| Intro ¶2 | "weather conditions, static and spatial covariates as leading external factor" | Holds for Chronos-2 (weather 83% of its top-15 importance); for the tree models weather is 12% of total gain, no feature in the top 15 [H] | "...with static regional context and, for the foundation model, weather as leading external factors" |
| Conclusion stub | "operations follow weekly rythm: seasonal strength 0.38" | §1.4 | "strike *reporting* follows a weekly rhythm ($F_S = 0.38$) that launch counts do not show" or keep with a caveat |
| Conclusion stub | "no predictive value is not saying there is no casual relation" | typo: casual → causal | — |
| Thesis conclusion (if reused) | "gradient-boosted trees better capture high-intensity days" | §1.2: reversed in the new run | "trees are most accurate on typical days in low- and medium-activity oblasts; no model forecasts surges well" |
| Recommendations stub | "chronos-2 has potential if goal is min. MAE" | Understated: Chronos-2 FT is now best on pooled RMSE and MAE, and zero-shot is #3 | covered by R1 and R3 |

---

## 4. Caveats

- Single seed (42). R1-R3 depend on small gaps (ARIMA vs best: 0.018 RMSE) and may shift; R4-R6 rest on
  large, consistent gaps across all leading models.
- Surge analysis: 104 region-dates (703 region-date-horizon rows), dominated by Sumy.
- Within-region surge AUC uses each Tier-3 oblast's top 10% test days as "surge".
- Hurdle surge figures omitted: the stored `hurdle` channel is the uncalibrated product (RMSE 2.057,
  equals `prob*count`); the leaderboard's 2.015 is the calibrated `hurdle_cal`.
- Weekday check: launches and ACLED strikes measure different drone populations (see §1.4).

## 5. Evidence and reproduction

Scripts (read-only, run with the repo `.venv`): `reco_analysis.py` → `reco_analysis_out.txt`,
`reco_analysis2.py` → `reco_analysis2_out.txt`, in this folder.

- [A] `runs_publication_20260929/_figures/master_leaderboard.csv`
- [B] `reco_analysis_out.txt` §A (per-tier RMSE/MAE, MSE shares)
- [C] `reco_analysis_out.txt` §B (surge pred/true ratio per horizon)
- [D] `reco_analysis_out.txt` §C (within-region surge AUC), §D (occurrence PR-AUC lift ≤ 0.09 for every model, hurdle classifier included)
- [E] `reco_analysis2_out.txt` (surge vs non-surge RMSE per horizon); `docs/audits/2026-09-29/horizon_tests_output.txt` (top-20 column means)
- [F] `reco_analysis2_out.txt` (cross-region Spearman / top-3 vs static pre-test mean)
- [H] `runs_publication_20260929/_figures/Feature-importancesharebycategory_grouped.csv`; full-feature shares of `count/catboost_tweedie/global/seed=42/importance/importance.csv` (weather 12% gain / 8% perm; activity tier 23% / 21%)
- [I] weekday means of `act_confirmed_launched` and the summed `act_drone_strike_on_ua_*` in `data/dataset/master_combined_timeseries.parquet`, divided by their weekly mean
