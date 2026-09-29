# Publication run (runs_publication_20260929, seed 42) vs thesis results

Status: analysis from stored outputs only (no reruns). VERIFIED = checked in files/code; HYPOTHESIS = inference.
Thesis side regenerated with `strikecast figures --source legacy` (same code as the new `_figures/`).

## 0. What the new run is
All 84 runs: git ce82a6f (clean), cluster A40 nodes. Differences from the thesis, all intended:
1. `leaky7` (true 7-day half-life) replaces `expdecay7` (which was ~ the raw series).
2. Five-branch "figure" feature selection: matched objectives, exactly 100 (feature, lag) pairs, ALL 27 future covariates kept (thesis: 8-12 selected weather variables).
3. Everything re-tuned (50 trials, TPE seed 42, Global); Activity/Local reuse Global params (as in the thesis).
4. XGBoost one model per horizon (D5); hurdle_cal primary (D6).

## 1. Sanity: what did NOT change (VERIFIED)
ARIMA 1.8535, seasonal naive 2.2045, naive 2.5790, Chronos-2 zero-shot 1.8493: identical. So data, splits, backtest schedule and metrics reproduce the thesis; every change below comes from models/features/tuning.

## 2. Leaderboard
| # | Thesis | RMSE | New | RMSE |
|---|---|---|---|---|
| 1 | Activity CatBoost-Tw | 1.8305 | Chronos-2 FT | 1.8359 |
| 2 | Chronos-2 FT | 1.8309 | Global CatBoost-Tw | 1.8446 |
| 3 | Global CatBoost-Tw | 1.8310 | Chronos-2 ZS | 1.8493 |
| 4 | Global LightGBM-Po | 1.8394 | Global LightGBM-Po | 1.8515 |
| 5 | Activity LSTM-Po w28 | 1.8405 | ARIMA | 1.8535 |
Activity CatBoost-Tw -> #11 (1.8679); Activity LSTM-Po w28 -> #23 (1.8958).
Context: in the thesis, places 1-3 were separated by 0.0005 RMSE; the new top 5 span 0.018. Rankings at one seed are fragile.

## 3. Activity CatBoost-Tweedie (why it left the top 5)
- VERIFIED: the whole loss is in the high-activity tier (6 regions): tier RMSE 3.2520 -> 3.3211; low 0.2170 -> 0.2172, medium 0.8797 -> 0.8849.
- VERIFIED: Sumy alone is 88% of the pooled MSE increase (+0.1217 of +0.1383). Sumy RMSE 6.1433 -> 6.3384. (Global CatBoost-Tw: Sumy is 129% of its increase; it improved elsewhere.)
- VERIFIED mechanism: the new model predicts ~10% lower in Sumy at every level (mean pred on days with y>15: 7.22 -> 6.56, true mean 21.3). That helps on quiet days and hurts on the 129 extreme days (y>15), which carry 87% of the extra squared error. Result: RMSE worse, MAE BETTER (0.7679 -> 0.7631). Pooled bias -0.2076 -> -0.2662.
- VERIFIED: the tuning plateau is flat. Top-5 CatBoost-Tw trials lie within 0.0008 CV RMSSE (0.9297-0.9305) with variance power 1.27-1.47 and lr 0.013-0.022. New winner: depth 5, lr 0.0182, 700 it, l2 3.5, subsample 0.976, power 1.472 (thesis: depth 4, lr 0.0121, 900 it, l2 5.0, subsample 0.998, power 1.395). Which point wins is close to arbitrary, yet it decides the Sumy spike level.
- VERIFIED: the high-tier model now has 442 inputs (thesis 291) and puts 22.6% of gain / 19.2% of permutation importance on future covariates (thesis 10.7% / 12.0%), taken from strike-history features. The 6-region tier has the least data to absorb 27 perfect-foresight weather/holiday covariates.
- HYPOTHESES (ranked): (a) new tuned params (higher Tweedie power + depth 5 shrink spike predictions); (b) dilution by the 27 unfiltered future covariates in the smallest tier; (c) leaky7 replacing a fast-reacting ~raw-lag feature with a smoothed one. Cannot be separated from files; see §8.
- Not a bug (VERIFIED indirectly): baselines reproduce exactly; the legacy golden verification reproduces the thesis pipeline.

## 4. RNNs (from the RNN agent)
- VERIFIED: movements go both ways (30 count RNN runs: mean dRMSE +0.0125, median +0.0036, SD 0.065, 17/30 worse); RMSSE barely moves (median -0.27%); rank correlation old vs new 0.29; the best old runs regressed most (regression to the mean). Sumy carries 60-130% of the net change in most runs (Activity LSTM-Po w28: +0.0553 overall, -0.0110 without Sumy).
- VERIFIED: re-tuning picked different configs in all 21 studies (same first 10-14 TPE trials, then divergence). Thesis LSTM-Po w28 config scores 0.9414 CV in the new study vs 0.9193 in the thesis; best CV worse in 13/15 count studies.
- VERIFIED: RNNs now get all 27 future covariates (input 107 -> 125 per step); with the same params CV slightly worse (median +0.010).
- VERIFIED: leaky7 does NOT reach the RNNs (they use raw past covariates). GPU training is nondeterministic; the thesis's own rerun of a best config moved CV by -0.013..+0.027.
- Conclusion: mostly noise + tuning draw; real degradations only GRU-Tw w7 Global (+0.166) and LSTM-MSE w28 Activity (+0.155).

## 5. Chronos-2 (from the Chronos/diff agent)
- VERIFIED: inputs identical (no selection, no window features in chronos_stage). New tuning follows the same 12 TPE trials; trial 10 (8.72e-5, 1500 = thesis) scored 1.1009 old / 1.1062 new, trial 11 (8.69e-5, 1600) 1.1105 / 1.1056, so noise flipped the winner between near-identical configs.
- RMSE +0.005 (Sumy +0.026, Zaporizhia +0.015; 15 low regions within 0.0003); prediction corr 0.9995. HYPOTHESIS (strong): fine-tuning nondeterminism.

## 6. Diff family
- VERIFIED: features changed a lot (old 52 past x 3 lags + 8 weather; new 100 pairs from 61 components + 27 future; 21/52 old components survive). Params moved (XGBoost depth 9 -> 4).
- XGBoost improved -0.11 (all at horizons 3-7); CatBoost/LightGBM only ~0.012-0.015. D5 per-horizon vs new params vs features: HYPOTHESIS, not separable from files.
- VERIFIED: linear (untuned, deterministic) 1.8885 -> 1.8790 is a pure feature effect.
- Best diff ML config: Activity CatBoost, Skill 0.074 (thesis: Global CatBoost 0.06; Global now 0.066).

## 7. Hurdle (from the hurdle agent)
- VERIFIED: the thesis Table 4 row (2.0015) comes from notebook session A; the thesis's own stored predictions (session B) give 2.0153. New = 2.0152. The global change was already present in the thesis files, not caused by the rerun.
- VERIFIED: new classifier lowers MSE by 0.0229, new count head raises it by 0.0224; they cancel. PR-AUC 0.8167 -> 0.8145.
- VERIFIED: hurdle was never tuned (notebook defaults), calibration identical -> not causes.
- VERIFIED new finding: thesis `tab:hurdle_bias` = session-B UNcalibrated hurdle (prob x count) on strike days, not the count head the text names; reporting/build.py:268 builds the count-head version, so neither regenerated table matches. Decide which quantity the paper reports.
- Existing wording issue l.1030: raw SPE probabilities are under-confident at the high end (0.74 predicted vs 0.88 observed).

## 8. Secondary tables / text claims that break (from the secondary agent)
- Friedman: CD 0.64 -> 0.52; set is the top five (ARIMA is in it). RMSE ranks: Global CatBoost-Tw 2.08, LightGBM 2.38 (tied), ARIMA 3.44, Chronos ZS 3.54, Chronos FT 3.56. MAE: Chronos FT 1.46 significantly best. Chronos FT = best pooled RMSE but worst block-wise RMSE rank.
- Horizon: RMSE Levene now fails (p=0.011) -> Kruskal H=37.2; ANOVA F 6.70 -> 7.12; Tukey: Day 5 vs 2 now significant; Day 7 differs only from 2 and 3. MAE Kruskal H 3.20, p 0.78 (thesis "5.49, p~0.05" was already wrong: p was 0.48). Chronos FT MAE margin over next-best (now Chronos ZS) 0.001-0.011, vs 0.06-0.08 over best non-Chronos.
- Per-region: RMSE claim holds (top 5 beat seasonal naive in all 20). MAE: naive beats the three non-Chronos leaders in 6 regions (Kirovohrad, Luhansk, Lviv, Odesa, Vinnytsia, Zaporizhia), not 3.
- Feature importance: still built from Activity CatBoost-Tw (now #11). T1 conflict 62/48% -> 51/52%; T1 weather perm 30 -> 25%; comms perm 15 -> 6%; T2 AR 45 -> 52%, macro 22 -> 14%, new spatial 9%; T3 82/66 -> 82/69%. All expdecay7 names -> leaky7. Top-15 overlap 2/15 (T1 gain) to 12/15 (T3 perm).
- Tuning appendix: every value changed.
- main.tex lines to update: 915, 940-949, 972-979 (+captions 984-997), 1006, 1031-1034, 1049-1056, 1080-1100, 1098-1123, 1135-1149, 1409-1450.

## 9. Proposed cluster jobs (not run)
1. Seeds 1-4 for the headline runs (count catboost_tweedie global/activity, lightgbm_poisson global, lstm_poisson_w28 activity, chronos2_fine_tuned, hurdle x3): measures run-to-run noise; the paper needs this before ranking claims.
2. CatBoost-Tw Activity 2x2 ablation {thesis params, new params} x {legacy=count features, figure features} (4 test runs, ~30 min each on 12 threads): separates tuning from features for the headline question. Optionally a third factor: figure selection with the thesis's 9 future covariates.
3. Hurdle legacy=hurdle featsel + run (all paradigms): separates new features from the session A/B difference.
