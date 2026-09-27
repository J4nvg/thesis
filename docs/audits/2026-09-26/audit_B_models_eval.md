# Audit B: models, tuning, backtest, evaluation

Date 2026-09-26. Repo `/Users/jan/projects/bsc_thesis_code/thesis`, branch `refactor` (dirty tree). Read-only audit; nothing in the repo or in `/Users/jan/projects/writing` was changed. `main.tex` = `/Users/jan/projects/writing/thesis_writing_folder/main.tex`. `C<n>` = findings in `audit_C_outputs_cluster.md` (same scratchpad); not repeated here. Scratch scripts: `threads_b.py` (output `threads_b.txt`).

## 0. Headline findings (read first)

1. **C13 is NOT data loss. The count-GBDT tuned params survive at full precision.** `_regression_GBDT.ipynb` cell 56 (source cleared, `execution_count: None`, output kept) prints `best_params_by_variant.items()` for all six count GBDTs as Python float reprs (round-trip exact float64). Its `lightgbm_poisson` entry equals `golden/converted/tuning/checkpoints_tune/lightgbm_poisson/best_params.json` digit for digit. **Proof it is the thesis's own set:** fitting count `catboost_tweedie` with the cell-56 params on test fold 0 through the refactor (`threads_b.py`) reproduces `golden/results/gbdt/predictions_long_test_global_catboost_tweedie_tuned.parquet` fold 0 to **max |Δ| = 1.8e-15** (140 rows), at 12 and at 15 threads. The coordinator's "four were tuned on Colab and are permanently lost" is contradicted by this: cell 56 is the Colab session's output.
   Cell-56 values (verbatim):
   * catboost_poisson: depth 4, learning_rate 0.022974267979887424, iterations 700, l2_leaf_reg 4.0, subsample 0.9994041367178963
   * catboost_tweedie: depth 4, learning_rate 0.012071584113760261, iterations 900, l2_leaf_reg 5.0, subsample 0.9977756396798295, tweedie_variance_power 1.3947466852349566
   * xgboost_poisson: max_depth 3, min_child_weight 9, learning_rate 0.03654613902394278, n_estimators 700, subsample 0.6453122838387236, colsample_bytree 0.6710559300757173, reg_alpha 0.002432579149558049, reg_lambda 0.0036107454705973092
   * xgboost_tweedie: max_depth 3, min_child_weight 2, learning_rate 0.01731752272762223, n_estimators 400, subsample 0.8416634198004798, colsample_bytree 0.8366070683982904, reg_alpha 0.66367867686419, reg_lambda 0.008253199534752557, tweedie_variance_power 1.2146452669370331
   * lightgbm_tweedie: num_leaves 17, max_depth 10, min_child_samples 168, learning_rate 0.018891200276189388, n_estimators 300, subsample 0.6733618039413735, colsample_bytree 0.7216968971838151, reg_alpha 0.12561043700013558, reg_lambda 0.05342937261279776, tweedie_variance_power 1.3329833121584336 (cross-check against the `lightgbm_tweedie_best.pkl` found on the cluster)
2. **`tab:tuning-best-all` alone is NOT usable as fixed params.** It lists only 4 models (CatBoost-T, LightGBM-P, LSTM-P-w28, Chronos-FT). **CatBoost-Poisson, XGBoost-Poisson, XGBoost-Tweedie and LightGBM-Tweedie are not in it at all.** CatBoost-T is complete (all 6 searched params) but rounded (lr 0.01207, subsample 0.998, tvp 1.395 = 3-4 significant digits), which would not reproduce the golden predictions. Use cell 56.
3. **Thread count does not change LightGBM or CatBoost output here.** Count `lightgbm_poisson`, test fold 0: predictions at num_threads 4, 12, 15 and 64 are **bit-identical** (`force_col_wise=True` makes histograms column-parallel). CatBoost-T at 12 vs 15 threads is identical to golden at 1e-15. So `threads: auto` → 64 on the cluster is harmless for these two; XGBoost could not be tested because it does not fit at all (§2). The residual LightGBM-P miss vs golden (max 0.17, mean 0.013 at fold 0, identical at every thread count) is therefore NOT a thread effect; F124's platform/library explanation stands (CatBoost is portable, LightGBM is not).
4. **The thesis GBDT numbers did not run on a 64-CPU A40 node.** Saved notebook outputs: `_regression_GBDT.ipynb` cell 9 "CPU count: 12" on `/content/drive/...` (Colab); thesis `main.tex:1351-1352` says "a T4 (Google Colab) for the GBDTs, ... i7-11800H for the hurdle model and statistical baselines". `final_hurdle.ipynb` cell 1 "CPU count: 4"; `damage_classifier.ipynb` "16"; `_diff_regression.ipynb` cell 7 source `available_threads = 4` with a stale output "16". Only the RNNs (and the diff script via `tune_diff_reg.sh`) are plausibly GPU4EDU jobs. Given item 3 this matters little for LightGBM/CatBoost.
5. **Latent PORT-BUG in RNN re-tuning** (B7): `tune_stage.run_trial` uses preset `for_tuning` (no `exp`) and a per-fold `PruningHook` for every model. The legacy RNN objectives (`_regression_LSTM.py:1090-1106`, `_diff_regression.py:1163-1173`) call `run_expanding_cv` (= `for_cv`: `exp` log-link applied) and never report per fold; pruning is only the PL callback on `train_loss` per epoch. Re-tuning a Poisson/Tweedie RNN in the refactor would optimise RMSSE of **log-space** predictions and mix two different quantities on the Optuna step axis. Harmless today only because all 21 RNN studies are imported from golden.
6. **Hurdle: Table 4 and the stored predictions are two different model fits** (deepens C6/C7). `final_hurdle.ipynb` cells 22-36 (exec 18-40) wrote `overall_metrics.csv` and `test_calibrated_overall.csv` (→ `leaderboard.csv`, Table 4: 0.7948/2.0015). Cells 48-49 (`execution_count: None`, another session) wrote all `global_*_preds.parquet` / `global_*_global_*.json`. They already differ on the **raw** hurdle (0.982586/2.039151 vs 0.984020/2.047963), so the whole model differs, not the calibration. The hurdle re-runs both LightGBM feature selections on every execution with an unpinned hash seed (F16) — the likely cause. Table 4's hurdle row and `tab:hurdle_bias` (session A) are unreproducible by any code; PR-AUC 0.817 and Figs. 16-17 (session B) are.
7. **The refactor's hurdle leaderboard row is the wrong component for the thesis.** `composite_stage.py:276` makes `hurdle_raw` primary (plain `global.json`); the thesis reports the Beta-calibrated hurdle (`main.tex:1004-1006`, Table 4). A store-walking leaderboard would print ~0.98/2.05 for "Hurdle".
8. **Scope cuts vs the thesis**: the `log` family is never reported (drop OK). The **diff MSE GBDT/RNN variants ARE reported** (`main.tex:879`, and `main.tex:937` "the best (Global CatBoost) attains a Skill Score of only 0.06") and the Table-4 naive rows plus the SkillScore denominator come from the diff run. `configs/experiment/diff.yaml` still contains all of them (good); the memory note "MSE-on-differenced variants are out" is stale and must not be acted on. The **damage classifiers are not reported anywhere in the thesis** (the "damage" appendix `main.tex:1256-1341` is the LLM-as-judge annotation), so `damage.yaml` is out of scope for "exactly as the thesis".

## 1. Run inventory (every number the thesis reports → source → refactor)

Status key: **VER** = ported and verified by golden (level E/F/H run on real data); **UNV** = ported, not verified on real data; **NP** = not ported; **OOS-R** = out of scope but reported.

| Thesis item (main.tex) | Legacy source → stored result | Refactor (config / registry / stage) | Status |
|---|---|---|---|
| Table 4 CatBoost Activity Tweedie 0.77/1.83/0.17; Top-20 #1, #3 (global) | `_regression_GBDT.ipynb` (Colab) → `results/gbdt/*catboost_tweedie_tuned*` | `count.yaml` / `catboost_tweedie` / activity+global test | UNV; params missing in store (C13) but recoverable (§0.1); fold 0 reproduces golden to 1e-15 |
| Table 4 LightGBM Global Poisson; Top-20 #4, #8, #16 | same → `lightgbm_poisson_tuned` | `lightgbm_poisson` | VER at level F (global cv/test, F124: max Δ 0.17 at fold 0) |
| Table 4 XGBoost Global Tweedie; Top-20 #10, #12, #14, #15 | same | `xgboost_tweedie`, `xgboost_poisson` | UNV; params recoverable (§0.1) |
| Top-20 #9, #17 CatBoost-Poisson | same | `catboost_poisson` | UNV; params recoverable |
| (no Top-20 row) LightGBM-Tweedie | same | `lightgbm_tweedie` | UNV; params recoverable / cluster pkl |
| Table 4 LSTM Activity Poisson w28; Top-20 #5, #18, #20; GRU rows #11, #13, #19 | `_regression_LSTM.py` (GPU4EDU) → `results/lstm/` | `count.yaml` / 15 RNNs / global+activity | UNV (params imported, no RNN run in store; GPU nondeterminism F9) |
| "baseline LSTM with MSE" (main.tex:640) | `lstm_w7/14/28` count | `count.yaml` `lstm_w*` | UNV |
| Table 4 Chronos-2 FT/OS (Local), Top-20 #2, #6 | `_chronos2.py` (A40) → `results/chronos2/` | `chronos2.yaml` | NP end to end (C15); adapter replay level F (F134) |
| Table 4 ARIMA Local p=7 d=1 q=1; Top-20 #7 (`diff global arima`) | `_diff_regression.py` → `results/diff/*arima*` | `diff.yaml` / `arima` (local kind) / global paradigm key | VER level F (cv+test) |
| Table 4 Seasonal Naive / Naive (Local) 0.89/2.20, 1.03/2.58; SkillScore denominator | `results/diff` naive_weekly / naive_last (global rows, relabelled local in AR cell 9) | `diff.yaml` naive_* | VER level E (exact) |
| Diff MSE best "Global CatBoost SS 0.06" (main.tex:937); diff `linear` = Top-20 rank-20 rule input (C9) | `_diff_regression.py` → `results/diff` | `diff.yaml` lightgbm/xgboost/catboost/6 RNNs/linear | lightgbm VER F, linear VER E; xgboost/catboost (GPU) and RNNs UNV |
| Table 4 Hurdle Global 0.79/2.00/0.09 (calibrated) | `final_hurdle.ipynb` session A → `leaderboard.csv` | `hurdle.yaml` / `hurdle` composite | calibration/scoring VER (level H on stored frames); predictions UNV and Table-4 row unreproducible (§0.6); wrong primary component (§0.7) |
| `tab:hurdle_bias` | session A (not stored) | `regressor` component | unreproducible (C6, §0.6) |
| PR-AUC 0.817 raw/cal, prevalence 0.294, Figs 16-17 | session B parquets | `classifier_raw/_cal` components | VER (level H on stored frames) |
| Beta[a=b] calibration | cell 20: `LogisticRegression(C=1e6)` on `logit(clip(p,1e-6))`, per horizon, fit on CV rows | `evaluation/calibration.py:165-175` | OK-VERIFIED (same algorithm; exact in level H) |
| Venn-Abers | cells 39-44 → `va_*.csv` | functions in `calibration.py`, not run by the pipeline | **not reported in thesis** (grep: 0 hits) → no action |
| SPE (DT depth 5, 100 est.), CatBoost Tweedie 1.5 count head, positive-only weights | cell 9-10 | `models/classifiers.py`, `models/hurdle.py` | UNV (fixed params, F7) |
| Damage classifiers | `damage_classifier.ipynb` (printed only) | `damage.yaml` | **not reported in thesis** → out of scope |
| `tab:tuning-search-space` | `_regression_GBDT.py:880-924`, LSTM suggester, `_chronos2.py:556-557` | `gbm.py:220-265`, `rnn.py` suggesters, `chronos.py:237-238` | OK-VERIFIED (GBDT, RNN identical ranges/steps/order); Chronos steps text error (C12) |
| `tab:tuning-best-all` | pickles + cell 56 | `runs/count/tuning/*` | LightGBM-P, LSTM-P-w28, Chronos OK; CatBoost-T = cell 56 rounded |
| Friedman/Nemenyi, Tukey, KW, Shapiro, Levene | `results/analyse_results.ipynb` | `evaluation/stats.py` (Friedman/Nemenyi only) | ranks VER (C32); horizon ANOVA/Tukey/KW/Levene NOT ported to `evaluation/` (only in AR notebook) |
| Feature importance (Figs 20, 21, 23) | GBDT cells 57-58, `_chronos2.py:686-688` | none | NP (C14) |

### Thesis-faithful job matrix (compare C16)

Derived from `main.tex:875-879` and the stored leaderboards (`results/gbdt` 36 rows = 6×3×2; `results/lstm` 60 = 15×2×2; `results/diff` 62 = (9×3 + 4)×2; `results/finalhurdle` 3 paradigms; `results/chronos2` 4 rows):

| Experiment | Models | Paradigms | Stages (thesis) | Seeds for "exactly as thesis" |
|---|---|---|---|---|
| count | 6 GBDT | global, activity, local | cv (not reported), **test** | 42 |
| count | 15 RNN (incl. 3 MSE) | global, activity (**no local**) | cv, **test** | 42 |
| diff | 3 GBDT + 6 RNN (MSE) | global, activity, **local (RNNs too)** | cv, **test** | 42 |
| diff | linear, arima, naive_last, naive_weekly | global only | cv, **test** | 42 (deterministic) |
| chronos2 | zero_shot, fine_tuned (+ tune 12 trials) | one predictor (thesis "Local", F133) | **test** only | 42 |
| hurdle | `hurdle` composite only (not the two component entries) | global, activity, local | cv (fits calibrators), **test** | 42 |
| damage, log | — | — | not reported | — |

Only the **test** stage feeds any thesis table/figure; CV is needed only for the hurdle's calibrators (and tuning, already imported). Minimal thesis matrix at seed 42: test stage = count GBDT 18 + count RNN 30 + diff tuned 27 + diff baselines 4 + Chronos 2 + hurdle 3 = **84 test runs**, plus 3 hurdle CV runs (calibrators) and 1 Chronos tuning study = **88 jobs**. Adding every CV stage (thesis ran them but reports none) gives 84 + 81 = 165. C16's defects (global-only default; RNN×local over-generation; standalone hurdle components) are confirmed; note C16 says "diff baselines global only" — correct — but **diff RNNs DO need local** (the thesis ran them, `results/diff` has `local,gru_w7_tuned`).

## 2. NEW BLOCKER: count XGBoost cannot be fitted in the pinned environment

`threads_b.py` fitting count `xgboost_tweedie` (cell-56 params, `reg:tweedie`, CPU, test fold 0) raises
`XGBoostError: objective.h:107: multioutput is not supported by the current objective function` (`threads_b.txt`). Mechanism: darts 0.43.0 `sklearn_model.py:1031-1043` wraps a model in `MultiOutputRegressor` only when `_supports_native_multioutput` is False (`sklearn_model.py:1590-1599`, reads `__sklearn_tags__().target_tags.multi_output`); with xgboost 3.1.3 + scikit-learn 1.6.0 XGBRegressor reports native multi-output, so darts hands it a 7-column target, which `reg:tweedie` (and `count:poisson`, same code path, not run) rejects. `gbm.py:130-157` passes no `multi_models` for the count family (darts default True). The thesis produced these rows (Table 4 XGBoost, Top-20 #10/#12/#14/#15) in a Colab environment whose darts/xgboost/sklearn versions are unrecorded, where darts evidently wrapped per horizon. No test fits a count XGBoost on real or synthetic data (equivalence tests use LightGBM/linear). **Also a silent behaviour question for the diff family:** diff XGBoost (`reg:squarederror`, GPU) does support native multi-output, so in this env it trains ONE multi-output booster instead of seven per-horizon boosters (the thesis's "separate model f_h per horizon", `main.tex:765-767`), silently. Needs a decision and a builder change (e.g. force `MultiOutputRegressor`), then re-verification against golden.

## 3. Thread-count and seed-42 reproducibility (item 7)

| Model (count, test fold 0, cell-56/golden params) | 4 | 12 | 15 | 64 threads | vs golden fold 0 |
|---|---|---|---|---|---|
| lightgbm_poisson | identical | identical | identical | identical | max 0.174, mean 0.0126 (F124; not threads) |
| catboost_tweedie | identical | identical | identical | identical | **max 1.8e-15** (bit-level reproduction) |
| xgboost_tweedie | fails to fit (§2) | | | | — |

* The refactor's `threads: auto` = `min(os.cpu_count(), 64)` (`schema.py:75-89`, C26) equals the legacy `get_available_threads()` (`src/prevalent_functions.py:245-249`). On the thesis machines that function returned 12 (count GBDT, Colab), 4 (hurdle), 16 (damage). For LightGBM (`force_col_wise=True`) and CatBoost CPU this has no effect on output (measured). The hurdle's SPE (`n_jobs`) and CatBoost count head were not measured; sklearn `n_jobs` in SPE parallelises estimators with per-estimator seeds, so it is expected to be thread-invariant (unverified).
* Seed-42 status in `runs/`: only `count/lightgbm_poisson` (global cv+test) and `diff/{naive_*,linear,arima,lightgbm}` exist. Tolerances in `tests/golden/test_pipeline_equality.py:172-206`: naives exact, linear 1e-6, ARIMA pred 0.5 / metric 1e-2 (TweedieDev 1e-1, F125), LightGBM pred 5.0 / metric 1-2 relative. The LightGBM ceilings (5.0 absolute on predictions, 100-200% relative on metrics) are so loose that they assert almost nothing; the real port assertion for LightGBM is the in-process legacy-runner test.
* **CatBoost gives a strictly better golden test than LightGBM**: with cell-56 params the count CatBoost family can be held at level E (≤1e-9), which is where Table 4's #1 row lives. Recommend adding `catboost_tweedie`/`catboost_poisson` (activity+global test) as level-E cases.
* RNN GPU determinism: neither legacy nor refactor sets `torch.use_deterministic_algorithms`, `cudnn.deterministic` or `CUBLAS_WORKSPACE_CONFIG` (`seeds.py` seeds random/numpy/torch/cuda only; darts `random_state` per model). So RNN seed-42 re-runs are level F at best (F9). Consistent with legacy; not a port bug.
* Versions: `pyproject.toml` pins darts 0.43.0, lightgbm 4.6.0, xgboost 3.1.3, catboost 1.2.10, torch 2.9.1, sklearn 1.6.0, pandas 3.0.2 (`requirements.txt` is the refactor's own export, not the thesis env). The only recorded thesis-era versions are the AutoGluon log (C25). The Colab GBDT env is unrecorded; CatBoost 1.2.10 evidently matches it bit for bit, xgboost/darts evidently do not (§2).

## 4. Findings

Severity: Crit = a thesis number cannot be produced; High = wrong/missing reported output or silent divergence; Med = latent or robustness; Low = text/cosmetic.

| ID | Area | Thesis (main.tex) | Legacy | Refactor | Category | Sev | Evidence | Decision for Jan |
|---|---|---|---|---|---|---|---|---|
| B1 | Tuned params, count GBDT (resolves C13) | 1409-1449 (tab:tuning-best-all) | `_regression_GBDT.ipynb` cell 56 output (all 6 variants, full precision) | `runs/count/tuning/` has only lightgbm_poisson; `import_golden_params.py` reads only `golden/converted/tuning` | NOT-PORTED (recoverable) | Crit | cell 56 lightgbm_poisson == golden JSON; catboost_tweedie fold 0 reproduces golden to 1.8e-15 | Import cell 56 as `best_params.json` for the 5 variants (not the rounded appendix values); cross-check lightgbm_tweedie with the cluster pkl |
| B2 | Appendix table completeness | 1409-1449 | — | — | TEXT≠CODE | Med | CatBoost-P, XGBoost-P/T, LightGBM-T absent; CatBoost-T rounded to 3-4 s.f. | Paper: give full-precision params (or a supplementary JSON) |
| B3 | Count XGBoost unrunnable | Table 4 row 3; Top-20 #10,#12,#14,#15 | Colab env (unrecorded versions) | `gbm.py:130-157` + darts 0.43 `sklearn_model.py:1031-1043,1590-1599` + xgboost 3.1.3 | PORT-BUG (env) | **Crit** | `threads_b.txt`: "multioutput is not supported by the current objective function" | Force per-horizon `MultiOutputRegressor` for count XGBoost (and decide the same for diff XGBoost, B4) |
| B4 | Diff XGBoost direct strategy | 765-767 ("separate model f_h per horizon") | per-horizon wrapper (inferred from B3) | native multi-output booster in this env | PORT-BUG (silent) | High | same darts branch; `reg:squarederror` supports multi-output so it runs silently | Same fix as B3; re-verify diff xgboost at level F |
| B5 | Threads | 1351-1352 (Colab/i7) | `get_available_threads()` = 12 (GBDT), 4 (hurdle), 16 (damage) per saved outputs; diff hard-codes 4 | `threads: auto` = min(cpu,64) | OK-VERIFIED for LightGBM/CatBoost | Low | bit-identical at 4/12/15/64 threads | None for LightGBM/CatBoost; C26 stays a cost issue only |
| B6 | LightGBM portability | — | Colab x86_64 | macOS arm64 run | TEXT≠CODE (F124) | Med | max 0.174 at fold 0, independent of threads | Accept level F for LightGBM; report the thesis numbers from stored predictions |
| B7 | RNN re-tuning objective | 881-893 | `_regression_LSTM.py:1090-1106`, `_diff_regression.py:1163-1173`: `run_expanding_cv` (exp log-link), no per-fold report, PL-callback pruning only | `tune_stage.py` `run_trial`: preset `for_tuning` (no exp) + per-fold `PruningHook`, on top of the PL callback in `rnn.py:_make_count_search_space` | PORT-BUG (latent) | Med | code reading; untested (tests tune a GBDT only) | Make the tuning preset/pruning per spec (`for_cv`, no per-fold report for neural) before any RNN re-tune |
| B8 | Hurdle primary component | 1004-1006, Table 4 (calibrated) | `leaderboard.csv` = calibrated | `composite_stage.py:276` primary `hurdle_raw` | PORT-BUG vs thesis | High | Table 4 quotes the Beta-calibrated hurdle | Make `hurdle_cal` primary (progress-log Q2) |
| B9 | Hurdle provenance | Table 4, tab:hurdle_bias | two notebook sessions (§0.6) | — | TEXT≠CODE | High | raw hurdle differs between sessions (0.9826/2.0392 vs 0.9840/2.0480) | Which session is canonical? Recommend session B (stored predictions) and updating Table 4 to 0.80/2.02/0.09 and the bias table |
| B10 | Calibration algorithm | 727-733, 753 | cell 20: `LogisticRegression(C=1e6)` on logit(clip 1e-6), per horizon, fit on all CV rows | `calibration.py:159-175` | OK-VERIFIED | — | level H exact | Text: line ~731 says the map is "fit on training data"; it is fit on validation-CV predictions (in sample for CV, F69) |
| B11 | Composite calibrators × seeds | — | one run | `composite_stage.py:302-362` | = C18 | High | — | (C18) |
| B12 | Tuning protocol text | 881-887 ("three-fold expanding-window backtest", MedianPruner everywhere) | GBDT/RNN: ~79 daily folds, 12 retrains (F12); Chronos: AutoGluon 3 internal windows, MASE, **no pruner** (`_chronos2.py:604-612`, F129) | same | TEXT≠CODE | Low | code | Fix text |
| B13 | ARIMA spec | 545-546 (AICc/unit roots), Table 4 (7,1,1) | `ARIMA(7,0,1)` on differences, darts `trend=None` → statsmodels constant for d=0 | `classical.py:_build_arima` | TEXT≠CODE (adds to F13) | Low | darts `arima.py:45,157` | Text: "ARIMA(7,1,1) with drift, fixed orders" |
| B14 | Backtest protocol | 752-789 | `run_final_test`: expanding, fit on all data before t0 (train+val+test-so-far), predict n=7 daily, retrain every 7 folds | `engine.py`, `run_stage.py:223-236` | OK-VERIFIED | — | level C exact (79/12 CV, 164/24 test), level D 1e-9 | none |
| B15 | Direct strategy for GBDT | 765-767 | darts `multi_models=True` default | same, except B3/B4 | OK except B3/B4 | — | — | — |
| B16 | SkillScore | 818-822, Table 4 | `1 − RMSE/RMSE(diff naive_weekly, global, test)` (AR cell 4) | leaderboard `SkillRMSE` per experiment vs its own `naive_weekly`; count has none (C1) | NOT-PORTED (cross-experiment) | High | AR cell 4 | Cross-experiment skill vs diff `naive_weekly` (=C1) |
| B17 | Horizon statistics | 1071-1098 (Shapiro, Levene, ANOVA, Tukey, KW) | AR cell 27 | only Friedman/Nemenyi/normality in `stats.py` | NOT-PORTED | Med | grep `stats.py` | Port with the figures (C1) |
| B18 | Diff MSE variants and log family | 879, 937 | `_diff_regression.py` | `diff.yaml` keeps all 9 tuned + 4 baselines | OK (memory note stale) | — | thesis reports diff MSE (SS 0.06) | Keep; drop only `log` |
| B19 | Damage classifiers | not reported | `damage_classifier.ipynb` | `damage.yaml` | OUT-OF-SCOPE | — | grep main.tex | Keep as extra or drop from the thesis matrix |
| B20 | Venn-Abers | not reported | cells 39-44 | not run by pipeline | OK (not needed) | — | grep 0 hits | none |
| B21 | Chronos | 666-668, Table 4 "Local" | `_chronos2.py` | `chronos.py` flags F127-F134; not runnable (C15) | = C15 | Crit | — | (C15) |
| B22 | Hurdle count head wording | 689-697 ("zero-truncated count model", "Tweedie") | CatBoost Tweedie 1.5 with library-default iterations/depth/lr, trained with positive-only sample weights (cell 9, 15) | `classifiers.py` | TEXT≠CODE | Low | cell 9 | Text: "CatBoost-Tweedie trained on positive days via sample weights, default hyperparameters" |

## 5. What the tests prove and do not prove

Known status: `tests/unit tests/equivalence` 1016 passed / 1 skipped / 1 xfailed; `tests/golden` 245 passed / 4 skipped (not re-run by this audit).
* **Prove:** the engine reproduces every legacy loop on a synthetic panel to 1e-9 (level D); fold schedules are identical on real data (level C); naive baselines exact and linear 1e-6 on real data; ARIMA and diff/count LightGBM-P within loose level-F ceilings; calibration/scoring of the hurdle exact given the stored frames (level H half); stats ranks match.
* **Do not prove:** that any count CatBoost/XGBoost, any RNN, any diff GPU GBDT, the hurdle predictions, the damage family or Chronos-2 run end to end on the real data; that count XGBoost can be fitted at all (it cannot, B3); that RNN re-tuning matches legacy (B7); any Table 4 / Top-20 number other than LightGBM-P global, ARIMA and the naives; seed ≠ 42 behaviour of composites (C18). The LightGBM level-F ceilings (5.0 absolute on predictions) are too loose to catch a real regression.

## 6. Questions for Jan

1. May I (or the next agent) turn `_regression_GBDT.ipynb` cell 56 into `runs/count/tuning/<variant>/best_params.json` for the five missing count GBDTs? It reproduces CatBoost-Tweedie to 1e-15. Does the cluster's `lightgbm_tweedie_best.pkl` equal cell 56's `lightgbm_tweedie`?
2. Count XGBoost (B3) and diff XGBoost (B4): force one model per horizon (the thesis's direct strategy)? This changes nothing else.
3. Hurdle: which session is canonical (B9), and should `hurdle_cal` be the leaderboard row (B8)?
4. Thesis-faithful matrix (§1): confirm RNN count = global+activity only, diff RNNs = all three paradigms, damage out of scope, only the test stage (plus hurdle CV) at seed 42 for "exactly as the thesis".
5. Text fixes for the paper: B2, B10, B12, B13, B22 (plus C8-C12).
