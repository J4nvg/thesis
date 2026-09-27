# Audit C: outputs (figures and tables) and orchestration (SLURM)

Date: 2026-09-26. Repo: `/Users/jan/projects/bsc_thesis_code/thesis`, branch `refactor` (dirty working tree, as expected). This audit was read-only. Scratch work (copies, executed notebooks, job matrices) is in this scratchpad directory.

The thesis source is `/Users/jan/projects/writing/thesis_writing_folder/main.tex`, cited as `main.tex:L`. The paper draft is `/Users/jan/projects/writing/chapter?/main.tex`, cited as `paper:L`. Notebook cells are 0-indexed, as `nbformat` numbers them.

## What was run

* Six unit-test files (Part 2 item 7): `test_make_jobs.py`, `test_cli.py`, `test_report_stage.py`, `test_sync_runs.py`, `test_run_store.py`, `test_tracking_wiring.py`. **117 passed in 10 s.**
* `scripts/slurm/make_jobs.py … --out -` for every experiment, three ways: with default selectors, with `--force`, and with `paradigm=global,activity,local`. The matrices are in `jobs_force_*.txt` and `jobs_full_*.txt`.
* `results/analyse_results.ipynb`, executed headless (`jupyter nbconvert --execute`) on a scratch copy of `results/`. It **ran top to bottom without an error** and wrote 23 SVGs to `nbrun/results/figs/`.
* `eda.ipynb` and `eda_full.ipynb`, executed headless on a scratch copy of `data/`. **Both fail on import**: `geopandas` and `ruptures` respectively.
* Numeric spot-checks (`spot_tables.py`, `spot2.py`). They recompute `master_df`, the Top-20, the hurdle bias table, the Friedman/Nemenyi ranks, the calibration numbers and the split geometry.
* An SVG comparator (`svgnorm.py`) that ignores the `<dc:date>` metadata and matplotlib's hash-salted ids. Every thesis figure was compared with the repo copy and with the headless output.

> **One side effect to report.** A probe that imported `strikecast` with the `envs/autogluon` Python 3.12 interpreter wrote bytecode caches into `src/strikecast/**/__pycache__/*.cpython-312.pyc`. These are git-ignored (`.gitignore:2 __pycache__/`) and harmless. Nothing else in the repo changed (`find -newer` check). The later runs used `PYTHONDONTWRITEBYTECODE=1`.

---

## Part 1: inventory of every figure and table in the report

Class key: **STATIC** means hand-drawn or hand-written with no generating code. **DATA** means computed from the raw dataset. **RESULTS** means computed from experiment outputs.

The "Auto today?" column answers whether the item can be regenerated automatically **from the refactored run store**. The **Partial** entries in it mean something narrower: the legacy notebook regenerates the item from the legacy `results/` headless, but nothing regenerates it from `runs/`.

`AR` below means `results/analyse_results.ipynb`.

### Figures (23 `\includesvg`/`\includegraphics` in main.tex, plus the logo in `frontmatter.tex:12`)

| # | Label | main.tex line | File | Class | Producing code | Inputs | Auto today? | What is missing | Spot-check |
|---|---|---|---|---|---|---|---|---|---|
| F1 | fig:timelines | 285 | timeline.jpg | STATIC | none (a 3903×1819 JPEG; no code references it) | – | n/a | – | – |
| F2 | fig:strikeactivity_perregion | 321 | strike_activity_per_region.svg | DATA | `eda.ipynb` cell 6 (Plot 1), `savefig` → `data/for_eda/plots/` | master parquet, `data/fixed/regions_activity_cat.json` (via `load_inputs`, cell 3), `data/for_eda/Ukraine_Admin_Regions.geojson` | **No**: cell 2 `import geopandas` fails because the package is not in `pyproject.toml` | port to a function; add geopandas | Same as the repo copy; only the embedded PNG colourbar bytes differ. The n_days labels match the data (Sumy 411). |
| F3 | fig:strikeactivity_perregion_activity_levels (**also paper:170**) | 328 | strike_activity_per_region_activity_level.svg | DATA | `eda.ipynb` cell 6 (Plot 2) | same as F2 | **No** (geopandas) | same as F2 | Identical after normalising metadata and ids |
| F4 | fig:spatiotemporalintensityheatmap (**also paper:204**) | 417 | spatiotemporalintensityheatmap.svg | DATA | `eda.ipynb` cell 9 | master parquet | **No**: the notebook dies at cell 2 | port | Byte-identical to `data/for_eda/plots/` |
| F5 | fig:Targetdistribution | 426 | fig_eda1_target_distribution.svg | DATA | `eda_full.ipynb` cell 5 (uses cells 1, 2, 4) | master parquet via `load_inputs` | **No**: cell 1 `import ruptures` fails | port; drop `ruptures`, which only serves `fig_eda3_changepoints`, not in the thesis | Byte-identical. D = 1.147/2.032/5.155 and zero rate 0.977 (cell 4) match the text's 1.15/2.03/5.16/0.98. |
| F6 | fig:pacfacf | 437 | fig_eda2_onlypacfacf.svg | DATA | `eda_full.ipynb` cell 10. It needs `P, A, tiers_arr, MAX_LAG` from cells 7–9. Its saved output is stale (a 1600×500 figure against the code's 8×10). | master parquet | **No** | port | Byte-identical (the file matches the current code, so only the saved output is stale) |
| F7 | fig:STLDECOMP (**also paper:215**) | 447 | fig_eda2c_stl_side_by_side.svg | DATA | `eda_full.ipynb` cell 14 | master parquet | **No** | port | Identical after normalising. F_T 0.73→0.09 and F_S 0.38→0.36 appear in the panel titles. |
| F8 | fig:feature_selection (**also paper:281**) | 532 | feature_selection.svg | STATIC | Lucidchart export (`lucid` signature in the SVG) | – | n/a | – | – |
| F9 | fig:hurdle_approach | 706 | HurdleCountModel.svg | STATIC | Lucidchart | – | n/a | – | – |
| F10 | fig:Experimentalsetup | 747 | Experimentaldesign.svg | STATIC | Lucidchart | – | n/a | – | – |
| F11 | fig:expanding_window_validation | 772 | expandingwindowvalidation.svg | STATIC | Lucidchart | – | n/a | – | – |
| F12 | fig:rmse_per_region | 954 | per_region_grid_by_tier_RMSE.svg | RESULTS | AR cell 21. The selection comes from cells 4, 5, 11 and 12, the loader from cell 19. | `top_5_w_naive` per-region CSVs: `gbdt/per_region_test_{activity,global}_*_tuned.csv`, `lstm/per_region_test_activity_lstm_poisson_w28_tuned.csv`, `chronos2/per_region_chronos2_fine_tuned.csv`, `diff/per_region_test_global_naive_weekly.csv`; tier map | **Partial** | a store reader, the name map (the store drops `_tuned`), seed aggregation | Byte-identical to `results/figs`. The macOS headless copy differs only in font (ArialMT) and width (1071 pt against 1163 pt). |
| F13 | fig:mae_per_region | 962 | per_region_grid_by_tier_MAE.svg | RESULTS | AR cell 21 | same as F12 | **Partial** | same | Byte-identical to `results/figs` |
| F14 | fig:cd_rmse_top5.svg | 982 | cd_rmse_top5.svg | RESULTS | AR cells 29 (block matrices), 31 (`friedman_nemenyi`) and 32 (diagram) | `predictions_long_test_*` parquets for top-5 + ARIMA (`SIG_SPEC`, cell 29) | **Partial**: `strikecast report` draws `cd_rmse.svg` per experiment over all runs, not the cross-experiment top-5 + ARIMA | cross-experiment selection | **Ranks reproduce exactly** (RMSE 2.65/2.75/3.14/3.34/4.51/4.62, CD = 0.64). The thesis SVG is 472.6 pt tall; the current code draws 307 pt (`figsize=(12, 0.3k+2.6)`), so it came from an older code version. |
| F15 | fig:cd_mae_top5.svg | 993 | cd_mae_top5.svg | RESULTS | AR cells 29, 31, 32 | same as F14 | **Partial** | same | MAE ranks 1.59/2.89/2.91/4.22/4.59/4.81 reproduce exactly. Same height difference as F14. |
| F16 | fig:calibration_prcurve | 1011 | calibration_prcurve.svg | RESULTS | AR cell 34 | `finalhurdle/global_test_classifier_probs.parquet` and `global_test_classifier_cal_probs.parquet` | **Partial** | map the hurdle store's probability channels | Identical. PR-AUC 0.817/0.817 and prevalence 0.294 match the text. |
| F17 | fig:prauc_prevelance | 1020 | prauc_vs_prevalence.svg | RESULTS | AR cell 36 | `global_test_classifier_cal_probs.parquet` | **Partial** | same as F16 | Identical except one marker id. The text says Δ "0.06–0.10"; the data give 0.057–0.108. |
| F18 | fig:top_5_horizon | 1067 | top_5_combined.svg | RESULTS | AR cell 26 | `per_horizon_test_*` CSVs of the top-5 | **Partial** | store `per_horizon.csv` plus the name map | Identical after normalising |
| F19 | fig:rmse_horizon_boxplot | 1074 | top20_rmse_horizon.svg | RESULTS | AR cell 27, `test_horizon_significance(master_df[:20])` | per-horizon CSVs of `master_df[:20]`, **which includes `diff/global/linear`** | **Partial** | a single, explicit top-20 rule | Identical. F = 6.7041 matches the text. The Kruskal–Wallis p does not (C10). |
| F20 | fig:importancesharebycategory_grouped | 1115 | Feature-importancesharebycategory_grouped.svg | RESULTS | AR cells 44, 47, 52 | `gbdt/importance_all.csv`, written by `_regression_GBDT.ipynb` cells 57–58 (gain plus permutation, `n_repeats=5`) | **No**: GBDT importance is not ported (C14) | a feature-importance job | Byte-identical to `results/figs` |
| F21 | fig:top_fi_chronos2 | 1125 | Chronos2LocalFeatureImportance.svg | RESULTS | AR cells 45, 49, 51 | `chronos2/feature_importance_chronos2_ft.csv`, written by `_chronos2.py:686-688` | **No**: not ported (C14, C15) | a Chronos importance job in the AutoGluon env | Byte-identical |
| F22 | fig:judging_system | 1260 | LLM_as_judge.svg | STATIC | Lucidchart | – | n/a | – | – |
| F23 | fig:top_fi_per_activity | 1611 | Top15FeatureImportancesperActivityLevel.svg | RESULTS | AR cells 47, 49, 50 | `gbdt/importance_all.csv` | **No** (C14) | a feature-importance job | Byte-identical |

Some SVGs in the thesis `fig/` folder are never included: `MAE_per_region`, `RMSE_per_region`, `top_5_MAE/RMSE`, `top_5_w_naive_*`, `top20_mae_horizon`, `prauc_f1_perregion` and `Global_GBDT_Permutation_Importance_Across_Horizons`. The 4 figures the paper draft uses are byte-identical to the thesis copies.

### Tables (11 in main.tex plus one paper-only table)

| # | Label | main.tex line | Class | Producing code | Inputs | Typed or generated? | Auto today? | What is missing | Spot-check |
|---|---|---|---|---|---|---|---|---|---|
| T1 | tab:acledData (**also paper:88**) | 184 | STATIC | – | ACLED codebook | typed | n/a | – | – |
| T2 | tab:activitytiers | 297 | DATA | thresholds only; `eda_full.ipynb` cell 2 `tier_of` and `data/fixed/regions_activity_cat.json` | master parquet | typed | No | generate from `load_inputs` | **OK.** 7/84/219/411 are the minimum n_days of each tier (rivne 7, kyiv 84, kharkiv 219) and the Sumy maximum. The `\label` sits before `\caption` (the label is unreferenced, so harmless). |
| T3 | tab:splitdimensions | 352 | DATA | none | – | typed | No | generate from the `SeriesBundle` split | **MISMATCH** (C11) |
| T4 | tab:overall_performance | 902 | RESULTS | typed from `master_df` (AR cells 4–5, the leaderboard CSVs). AR cell 7 now prints a *different* LaTeX table (`tab:best_per_family`, with MAPE, taking the hurdle from `hurdle_cal_preds`). | 6 `leaderboard.csv` files | typed | Partial | a family-representative rule; seed aggregation | **OK to 2 dp for 10 of 11 rows.** The hurdle row agrees only with `finalhurdle/leaderboard.csv` (C7). |
| T5 | tab:top_models | 923 | RESULTS | typed, `master_df.head(5)` | same as T4 | typed | No | – | **MISMATCH**: rank 1 MAE is printed 0.78; stored 0.767909 (T4 prints 0.77) (C8) |
| T6 | tab:hurdle_bias | 1036 | RESULTS | AR cell 38 (`display(head(10))`; 8 rows typed). The saved output is a `NameError`. | `finalhurdle/global_test_regressor_preds.parquet` (y_true > 0) | typed | Partial | – | **MISMATCH in every row's RMSE/MAE/bias** (C6) |
| T7 | tab:tuning-search-space | 1354 | STATIC (from code) | `_regression_GBDT.py:880-968`, `_chronos2.py:554-557` | – | typed | could come from `ModelSpec.search_space` | – | GBDT and RNN rows OK. Chronos `fine_tune_steps` is {600..3000} in the table but [200, 3000] in the code (C12). |
| T8 | tab:tuning-best-all | 1409 | RESULTS (tuning) | `best_params` pickles, converted to `golden/converted/tuning/*/best_params.json`; `golden/checkpoints/chronos2_best/best_params.json` | – | typed | Partial (`runs/<exp>/tuning/<m>/best_params.json`) | a CatBoost-Tweedie artefact | LightGBM-P OK, LSTM-P-w28 OK (25/50 pruned), Chronos OK (8.72e-5, 1500). **CatBoost-Tweedie cannot be verified** (C13). |
| T9 | tab:base_variables (longtable) | 1463 | DATA | typed from the panel columns (`eda.ipynb` cell 4 output) | panel | typed | No | generate from the panel columns | **OK**: all 112 names are in the panel |
| T10 | Top-20 (appendix, **no `\label`**) | 1543 | RESULTS | hand-edited `master_df` (AR cell 5 display) | leaderboard CSVs | typed and edited | Partial | an explicit rule | Values match `master_df` to 6 dp. **Row 20 was hand-substituted** (C9). |
| T11 | tab:semantic_categories | 1577 | STATIC | mirrors `classify_feature` (AR cell 52) | – | typed | could be generated | – | Consistent with the code |
| P1 | tab:covariates (**paper only**, paper:121) | – | DATA | none | panel columns plus a family map | typed | No | generate | The total of 112 matches T9. The per-family n were not checked. |

**Inline numbers in the text** are also report outputs, and none are generated: the EDA statistics, the Friedman ranks, the Tukey and Kruskal–Wallis p-values, PR-AUC, and the paper's 86% zeros, 143 region-days and max 37 (Sumy, 2024-11-13). All re-verified from data except the Kruskal–Wallis p (C10).

### The `USE_RUN_STORE` gap (74 against 117 rows)

`master_df` from the CSVs has 117 rows: diff 31 + log 31 + lstm 30 + chronos2 4 + gbdt 18 + finalhurdle 3. The store today holds 5 of 31 diff runs and 1 of 18 gbdt runs, which leaves 74 rows.

Closing the gap needs more than re-running models, because the switch only rewires the leaderboard loader:

* `FAMILY_TO_EXPERIMENT` maps only `diff` and `gbdt` (AR cell 0). `lstm`, `chronos2` and `finalhurdle` always come from the CSVs. The count store also holds the RNNs, so they would be double-counted once `gbdt→count` is filled.
* The archived `log` family (31 rows) can never come from the store.
* Store model names drop `_tuned`: the store has `lightgbm_poisson`, the CSV `lightgbm_poisson_tuned`.
* Every figure cell builds hard-coded legacy paths: cells 7, 16, 18–19, 21, 26–27, 29, 33–36, 38, 44–47 and 56. Several catch `FileNotFoundError` and just print a warning (cells 16, 18, 27), so a flipped switch would **silently drop models** from the figures.
* The notebook has no notion of seeds.

### Figure styling determinism

The **SVGs are not deterministic, and the styling depends on the machine**:

* No style file.
* `rcParams` are mutated globally mid-notebook. AR cell 32 sets `font.family: serif`; `sns.set_theme` runs in cells 16, 18, 21, 35 and 36.
* The figure style therefore depends on the order cells ran in. The saved AR has cells 27–36 at execution counts 49–58, run *before* cells 1–26 at 118–137.
* Seaborn prefers Arial. The macOS headless run produced **ArialMT**; every thesis figure is **DejaVuSans**, i.e. Linux/WSL, which the `:Zone.Identifier` files also point to. The different font metrics change `bbox_inches='tight'` sizes: `top_5_combined` is 596 pt wide on macOS and 634 pt in the thesis.
* No `svg.hashsalt` and no `metadata={'Date': None}`, so every run byte-differs.
* `strikecast.evaluation.stats.CD_RCPARAMS` (`stats.py:92`) scopes only the CD diagram. Correctly scoped, but it is the only styling control.

---

## Findings

Severity: **Crit** blocks the "one command re-computes everything" goal or yields silently wrong results. **High** means a report output or job family is missing or wrong. **Med** is a robustness or cost issue. **Low** is cosmetic.

| ID | Area | Evidence | Category | Severity | Decision needed from Jan |
|---|---|---|---|---|---|
| C1 | Report generation | `pipeline/report_stage.py:766-895` writes, per experiment only, the leaderboard, CI, pairwise tables, `cd_*.svg` and `summary.md`. None of the 11 RESULTS figures or 6 RESULTS tables are produced. There is no cross-experiment leaderboard, although Table 4, the Top-20 and the CD figures mix count + diff + chronos2 + hurdle. Count has no naive baseline (`count.yaml:69-75`), so its SkillScore needs diff's `naive_weekly`. There is no `.tex` output. | NOT-PORTED | Crit | Should the LaTeX `\input` generated `.tex` fragments? Should the paper show seed means with CIs, or seed 42 only? |
| C2 | Notebook rewire | AR cell 0 (`USE_RUN_STORE`, `FAMILY_TO_EXPERIMENT={"diff","gbdt"}`). The figure cells use legacy paths (listed above). Store names lack `_tuned`. Missing files are skipped with a print. | BROKEN (partial port) | High | Flip after re-runs, or retire the notebook in favour of ported figure functions (recommended)? |
| C3 | EDA notebooks | `eda.ipynb` cell 2 `import geopandas` and `eda_full.ipynb` cell 1 `import ruptures` both raise `ModuleNotFoundError` in `.venv` (verified headless). | BROKEN | Med | Add a `figures` extra, or drop the changepoint cell? |
| C4 | AR headless | `nbconvert --execute` ran all 61 cells and wrote 23 SVGs. Content matches the thesis; only font and size differ (see styling). The saved notebook has out-of-order execution counts and stale error outputs (cell 38 `NameError`, cell 52 `'tab:20'` colormap). | OK-VERIFIED (with caveat) | – | – |
| C5 | Styling | No mplstyle, no hashsalt, no Date metadata, global `rcParams` mutation; Arial on macOS against DejaVu on Linux. | MISSING-FEATURE | Med | Pin DejaVu Sans (matches the thesis)? |
| C6 | tab:hurdle_bias | `main.tex:1046-1053`: Sumy 7.35 / 4.83 / −4.17. Recomputed from `results/finalhurdle/global_test_regressor_preds.parquet` (the AR cell 38 logic, also via the headless run): **6.79 / 4.48 / −3.13**. All 8 rows differ in RMSE, MAE and bias; the counts (1093…) and maxima match. No stored file contains the thesis numbers. | MISMATCH (provenance lost) | High | Accept the recomputed values, or re-run the hurdle? |
| C7 | Hurdle headline numbers | Three stored versions disagree. `finalhurdle/leaderboard.csv` (renamed from `test_calibrated_overall.csv` in commit 848df73) gives 0.7948/2.0015, which Table 4 uses. `global_test_hurdle_cal_preds.parquet` and `global_test_global_hurdle_cal.json` give 0.7983/2.0153; AR cell 7 prints 0.80/2.02/0.09 from them. `overall_metrics.csv` / `hurdle_raw` give 0.983–0.984/2.039–2.048. | MISMATCH | High | Which is canonical? This ties to open question 2 of the progress log (`hurdle_raw` against `hurdle_cal`). |
| C8 | tab:top_models | `main.tex:932`: rank-1 MAE printed 0.78; stored 0.767909 (Table 4 `main.tex:908` prints 0.77) | MISMATCH | Low | Fix the text |
| C9 | Top-20 definition | The appendix (`main.tex:1549-1568`) drops `master_df` rank 20 (diff/global/linear, SS 0.143345) and lists rank 21. AR cell 27 uses `master_df[:20]`, linear included. The thesis statistics (F = 6.7041, H = 5.4895) only reproduce *with* linear; without it F = 8.75 and H = 6.02. | MISMATCH | Med | Which top-20 rule? |
| C10 | Kruskal–Wallis text | `main.tex:1082` says "H = 5.49, p ≈ 0.05". The AR cell 27 output and the recomputation give **p = 0.483** (df 6). | MISMATCH (text) | Med | Fix the text |
| C11 | tab:splitdimensions | `main.tex:352-370` has days 592/85/169 (= 846) and observations 11,844/1,692/3,384, which are 0.7/0.1/0.2 × 16,920, not days × 20. The data have **847 days** (2022-09-28..2025-01-21 inclusive; the master parquet has 847 rows; `eda_full` n = 16,940) and a split of **593/85/169** (darts `split_after`; the Chronos t0 = 677). That gives observations **11,860/1,700/3,380 = 16,940**. The paper says 847 at `paper:162,165` but 846/16,920 at `paper:178`. | MISMATCH | Med | Fix, and generate the table |
| C12 | Search-space table | Chronos `fine_tune_steps` is 600 in the table but 200 in the code (`_chronos2.py:557`, `models/chronos.py:239`) | MISMATCH (text) | Low | Fix the text |
| C13 | Count GBDT tuned params | `golden/converted/tuning/checkpoints_tune` has only `lightgbm_poisson` plus the 15 RNNs (`conversion_log.json`: 25 sources). No params exist for catboost_* (including **Table 4 #1 CatBoost-Tweedie**), xgboost_* or lightgbm_tweedie; `golden/converted/cv_predictions/checkpoints/catboost_tweedie_tuned.json` holds predictions only. `make_jobs` therefore emits 5 CPU tune jobs, and re-tuning gives new numbers. | NOT-PORTED (data loss) | High | Do the Colab `checkpoints_tune/*_best.pkl` files exist somewhere? Otherwise, re-tune, or hard-code the Appendix values as `params:`? |
| C14 | Feature importance | GBDT gain + permutation (`_regression_GBDT.ipynb` cells 57–58 → `importance_all.csv`) and Chronos permutation importance (`_chronos2.py:686-688`) have no pipeline equivalent. `composite_stage.py:549-616` covers the hurdle's native importances only; permutation is deliberately skipped (`composite_stage.py:642-646`). This blocks F20, F21 and F23. | NOT-PORTED | High | Include FI jobs in `submit_all` (hours of compute)? |
| C15 | Chronos-2 path | The registry never imports `strikecast.models.chronos` (`context.py:56-61`), so `make_jobs experiment=chronos2` raises `KeyError 'chronos2_zero_shot'` (run). `run_stage.make_forecaster` raises for kind `chronos` (`run_stage.py:211`). `tune_stage` has no AutoGluon path. `envs/autogluon` has `package=false`, so `import strikecast` fails without `PYTHONPATH=src` (verified), and it has no `wandb` although the default is `wandb_online`. The README command (`README.md:36-37`) cannot work. The `golden/checkpoints/chronos2_best` params are not imported by `import_golden_params.py`. | NOT-PORTED | Crit | – |
| C16 | Job matrix: paradigms | Every experiment composes `/paradigm: global` (for example `count.yaml:8`), so `make_jobs` without `paradigm=` emits **global only**: count gets 129 jobs, activity and local are missing, and Table 4's #1 (Activity CatBoost) is not produced. With `paradigm=global,activity,local` it **over-generates**: RNN × local (90 GPU jobs the thesis never ran; `results/lstm/leaderboard.csv` has global and activity only), diff baselines × activity/local (+16), and the hurdle/damage *components* as standalone models (`spe_event_classifier`, `catboost_tweedie_count_head`, `spe_damage_classifier`: +36 and +30). No per-model paradigm list exists in the schema (`schema.py:797`). | BROKEN | High | Confirm the thesis-faithful matrix (below). Are CV runs for activity/local needed at all? |
| C17 | Ordering and dependencies | `make_jobs` puts `tune`, `cv` and `test` lines in **one file per resource class** (`make_jobs.py:241-251`). It prints one `sbatch --array`, using `tune.sbatch` whenever tune is in the stages (`make_jobs.py:303-311`), with no `--dependency`. A cv/test task that starts before its tune finishes runs on **spec defaults with only a WARNING** (`run_stage.py:143-153`), completes, and is then **skipped forever** because the skip check uses the stage name, not its identity (`make_jobs.py:31-37,208`). | BROKEN | **Crit** | Should a tunable model without `best_params` be a hard error? |
| C18 | Composite calibration × seeds | CV runs only at the tuning seed (`composite_stage.py:975`; `make_jobs.seeds_for`). Calibrators are read from the **same run key's** `artifacts/` (`composite_stage.py:302-313`). A test at seed 1–4 finds none and **silently uses raw probabilities** (`composite_stage.py:352-362`). A seed-42 test racing its CV does the same. Seed CIs for the calibrated hurdle and damage channels are wrong. | BROKEN | High | Read the calibrators from `seed=<tuning_seed>`? (This overlaps the models/evaluation audit.) |
| C19 | GPU request | Neither template has `--gres` (`run.sbatch:38-44`, `tune.sbatch:39-45`). The sbatch line `make_jobs` prints for the GPU file lacks `--gres=gpu:1` (`make_jobs.py:309-310`). RNNs would silently run on CPU (`accelerator=auto`); diff XGBoost (`cuda`) and CatBoost (`GPU`) fail. | BROKEN | High | – |
| C20 | Fold-level resume | `run_stage.py:442-447`: "engine cannot start mid-schedule; recomputing from fold 0". `run_store.py:43-49` agrees. Grouped paradigms (activity, local) persist **nothing** until every group has finished (`run_stage.py:607-638`; composite `_replay`, `composite_stage.py:281-297`, `:896`). A task killed at the wall clock restarts from zero, so a stage longer than the partition limit **never finishes**. `README.md` ("resumes from its last retrain boundary") is inaccurate. | MISSING-FEATURE | **Crit** | – |
| C21 | Signals and requeue | No `--signal`, `--requeue` or `trap` in either template. SIGTERM kills Python without `fail_stage`, so the state stays `running`. `state.started` persists across attempts (`run_store.py:464`), so `state.json` durations mislead: count/lightgbm_poisson cv shows 946 min, the part mtimes about 40 min. | MISSING-FEATURE | Med | – |
| C22 | Tuning resume | Optuna SQLite with `load_if_exists=True`, and a total-budget count of COMPLETE + PRUNED + FAIL (`optuna_runner.py:324-356`), resumes at trial granularity. Caveats: a killed RUNNING trial stays a zombie (listed in `trials.csv`); SQLite locking on NFS/Lustre is unreliable; the TPE RNG state is not restored (F11). | OK-VERIFIED (with caveats) | Med | – |
| C23 | `state.json` race | cv@42 and test@42 of one (model, paradigm) share a run directory and run as concurrent array tasks. `PartWriter.write` (`run_store.py:687-721`) and `start/complete/fail_stage` read, modify and rewrite the whole file with no lock, so updates can be lost; for example cv can flip back to `running` and be re-queued. | BROKEN | Med | – |
| C24 | Environment per task | Every array task runs `uv sync --frozen` (`run.sbatch:73`, `tune.sbatch:69`). That is an *exact* sync, so it uninstalls the `dev` extra (nbconvert, jupyter, pytest) from the shared `.venv`. The tasks contend on `.venv/.lock` (flock is unreliable on NFS/Lustre), and a cold cache needs the network for Python 3.13 and the git dependency `imbalanced-ensemble`. `srun uv run` without `--frozen` or `--no-sync` (`run.sbatch:78`) may re-lock or re-sync. | BROKEN | High | uv or the `thesis` conda env on the cluster? |
| C25 | CUDA wheels | `uv.lock` has torch 2.9.1 `manylinux_2_28_x86_64` with the `nvidia-*-cu12` 12.8 deps from PyPI (not CPU-only). The Chronos log (`golden/checkpoints/chronos2_best/logs/predictor_log.txt:11-14`) shows CUDA 12.8 on an A40 with 44 GB, so they are compatible. Caveats: the manylinux_2_28 wheels (torch, lightgbm, xgboost) need **glibc ≥ 2.28**; the core env is Python 3.13, while the legacy cluster used 3.12 in the conda `thesis` env. | OK-VERIFIED (glibc unknown) | Med | Run `ldd --version` on a node |
| C26 | Threads | `threads: auto` resolves to `os.cpu_count()` capped at 64 (`schema.py:75-89`). It ignores the SLURM allocation (cgroup affinity), so a model gets up to 64 threads on an 8-CPU allocation. GBDT results depend on the thread count (F124). | MISSING-FEATURE | Med | – |
| C27 | Log directory | `#SBATCH --output=logs/slurm/...` requires the directory at submit time; the `mkdir` in the script (`run.sbatch:52`) runs too late, so the first submission's tasks die without logs. | BROKEN | Low | – |
| C28 | Data on the cluster | `data/` is tracked, including the master parquet and the geojson. `results/` is tracked (1681 files) despite `.gitignore:233`. `golden/` is ignored (`.gitignore:226`) but **required**: `count.yaml:66` and `diff.yaml:61` set `cache_path: golden/converted/feature_sets/*.json`, and `import_golden_params` needs `golden/converted/tuning` (8.2 MB total). Chronos-2 needs HF weights (internet or a pre-seeded `HF_HOME`). | MISSING (a manual step) | Med | – |
| C29 | W&B mode | All experiments default to `tracking: wandb_online` with `strict: true` (`wandb_online.yaml:15`), and the templates never select offline. The brief says offline on SLURM; plan §1 says online. | Decision | Low | Online or offline? |
| C30 | Single submission | `make_jobs` never calls `sbatch`. It has no report or figures stage, no dependency wiring, no Chronos env routing, no per-class partition or time, and no array throttle (`%N`). | MISSING-FEATURE | **Crit** | Cluster facts (below) |
| C31 | Tests | 117/117 passed in the six orchestration test files. They do not cover the default paradigm, Chronos, `--gres`, dependencies or the composite seed/calibration issue. | OK-VERIFIED | – | – |
| C32 | Numbers that match | Table 4 (10 of 11 rows); Top-20 values; Friedman/Nemenyi ranks and CD; the Tukey p-values; PR-AUC and prevalence; the LightGBM-P, LSTM-P-w28 and Chronos best params; the tier thresholds; the 112 base variables; the EDA D and zero rates; the paper's 86%, 143 and 37. | OK-VERIFIED | – | – |

---

## Runtime estimates per job type

The evidence, in decreasing reliability:

* **(a) Legacy Optuna studies** (`golden/converted/tuning/*/trials.csv`). They ran on GPU4EDU on 2026-05-06/07; the thesis appendix gives an A40, and the legacy `tune_*.sh` asked for `-p GPU --gres=gpu:1 --cpus-per-task=4` with no `--time`.
* **(b) The AutoGluon Chronos log.**
* **(c) Prediction-part mtimes in `runs/`.** These are laptop runs from 2026-09-19 on an M-series with 15 cores, contended by concurrent test runs. The `state.json` "durations" are wrong because of C21.
* **(d) The code structure.** One tuning trial is exactly one CV-stage backtest: 79 folds and 12 retrains (`tune_stage.py:1-30`). A trial's duration is therefore a proxy for one CV stage.

| Job type | Evidence | Estimate on the cluster | Risk against a 4 h / 24 h / 48 h / 72 h limit |
|---|---|---|---|
| Tune, count LightGBM-Tweedie (CPU, 50 trials) | count LightGBM-P study **2.14 h**; trial median 2.4 min, max 5.6 min (a) | 2–3 h | over 4 h possible |
| Tune, count XGBoost-P/T (CPU) | diff XGBoost on **GPU 2.12 h** (a); CPU hist typically 3–5× slower | 6–12 h (unverified) | exceeds 4 h |
| Tune, count CatBoost-P/T (CPU) | diff CatBoost on **GPU 8.44 h**, trial max **50.9 min** (a); CPU slower | **17–40 h** (unverified) | **exceeds 24 h likely**; needs trial resume plus requeue |
| Tune, RNN (GPU; all 24 already in golden) | 0.28–1.49 h per study (a) | 0.3–1.5 h | OK |
| Tune, Chronos-2 FT (GPU, AutoGluon env, 12 trials) | one fit at 1500 steps = **636 s** on the A40 (b); steps range 200–3000 | 1.5–4 h | OK at 4 h only barely |
| CV, global GBDT (CPU) | trial at best params: count LGBM-P 2.1 min; diff LGBM 0.9, XGB 2.5, CatBoost 4.4 min (a). Laptop: count LGBM-P CV **~40 min** (c, contended) | 2–40 min | OK |
| Test, global GBDT (CPU) | laptop: count LGBM-P test **2 h 40 min** (parts 12:12→14:52), diff LightGBM **~4.5 min** (c). 164 folds and 24 retrains against 79/12 for CV, so about 2–3× CV. | 5 min – 3 h | OK at 24 h |
| Test, activity GBDT | 3 groups | ≈ 1–1.5× global | OK |
| Test, local GBDT/hurdle | 20 regions × 7 horizon models per retrain; persisted only at the end (C20) | 0.5–4 h (no evidence) | over 4 h possible; zero progress survives a kill |
| ARIMA (diff, local per region) | laptop test **47 min** (15:02→15:49), CV 14 min (c) | 0.5–1 h | OK |
| RNN cv/test (GPU) | trial durations 0.3–4 min (a) | 2–15 min × 5 seeds | OK |
| Hurdle (CPU; 2 heads; calibration; **per-job LightGBM feature selection**, because `hurdle.yaml` sets `cache: false`) | none; the thesis ran it on an i7-11800H laptop | 0.5–3 h (local: more) | over 4 h possible |
| Damage (CPU; 4 SPE heads) | none | < 2 h | OK |
| Chronos-2 test (GPU) | zero-shot fit 100 s; about 2 s of prediction per fold (b); ~30 s per fold on CPU (`models/chronos.py:633`) | 10–20 min GPU (~80 min CPU) | OK |
| Feature importance: GBDT permutation (5 models × 7 horizons × ~500 features × 5 repeats) and Chronos | "hours" (progress log L865) | 2–10 h | over 4 h |
| `strikecast report` per experiment | the diff report (5 runs) took ~4 s (file mtimes); pairwise work grows with runs² (count has about 51 runs, 1275 pairs) | minutes | OK |

**The thesis-faithful matrix is about 490 jobs**:

| Family | Jobs |
|---|---|
| Count GBDT: 6 × 3 paradigms × (1 cv + 5 test) | 108 CPU |
| Count GBDT tuning (missing params, C13) | 5 CPU |
| Count RNN: 15 × 2 paradigms × 6 | 180 GPU |
| Diff tuned models: 9 × 3 paradigms × 6 | 162 (18 CPU + 144 GPU) |
| Diff baselines: 4 × global × (cv + test) | 8 CPU, 6 already complete |
| Hurdle: 3 paradigms × 6 | 18 CPU |
| Damage: 6 | 6 CPU |
| Chronos-2 (tune + zero-shot + 5 fine-tuned) | about 7 GPU |
| Feature importance | about 4 |
| Report | 1 |

**Rough totals:** about 150 CPU jobs with 150–400 CPU-h, and about 340 GPU jobs with 40–80 GPU-h. **Before sizing `--time`, run one pilot per job class on the cluster** with `--progress-every 7`. The laptop numbers are contended, and the count/lightgbm_poisson test ran about 70× slower than its tuning-trial proxy.

---

## Proposed design (not implemented)

### (a) Automatic generation of every report figure and table from the run store (`strikecast figures` / `make figures`)

1. **A `strikecast.reporting` package** with one pure function per artefact. Each function takes a `ResultsView` and returns a matplotlib `Figure` or a `DataFrame`. Port the AR cells and the EDA cells rather than executing notebooks, because the notebooks carry hidden state and global style. The `ResultsView` has two back-ends behind one interface:
   * `StoreView(runs/, seeds=…)`: multi-experiment and multi-seed. It maps registry name → legacy name → display label, and aggregates seeds as mean, SD and CI, or at seed 42 only.
   * `LegacyView(results/)`: a thin adapter over the current CSVs and parquets. It lets every figure be **golden-tested against the thesis SVGs today** and switched to the store later with one flag.
2. **A config** `configs/report/thesis.yaml`. It lists each artefact with its id, function, selection rule, output name and caption source, and it replaces the notebook's hand selections:
   * the families shown, with `log` excluded;
   * the Table-4 representative per family;
   * the top-k by SkillScore across experiments, measured against diff `naive_weekly`;
   * the top-20 rule, one definition for both the appendix and the horizon statistics (C9);
   * the FI models.

   Output names should match the thesis file names (`fig/per_region_grid_by_tier_RMSE.svg`, …) so the folder can be dropped into `writing/.../fig/`.
3. **Tables as `.tex` fragments** (`build/tables/tab_overall_performance.tex`, …), rendered from a fixed template per table with pinned number formats. The LaTeX switches to `\input{}`. Inline numbers go to `build/numbers.tex` as macros (`\newcommand{\KWp}{0.48}`), so the text stops drifting (C8, C10, C11).
4. **Deterministic style**:
   * a bundled `thesis.mplstyle` with `font.family: DejaVu Sans`, which is what the thesis used and ships with matplotlib;
   * `sns.set_theme(font="DejaVu Sans")`;
   * `svg.hashsalt: strikecast`;
   * `savefig(metadata={"Date": None})`;
   * `MPLBACKEND=Agg` and `PYTHONHASHSEED=0`;
   * fixed `figsize` values;
   * no global `rcParams` mutation.

   Add a CI check that regenerates from `LegacyView` and compares normalised SVGs against the thesis copies.
5. **The EDA figures** move to functions over `load_inputs`/`build_panel`. Add `geopandas` to a `figures` extra and drop `ruptures`, which is not needed for any thesis figure.
6. **Artefacts missing from the store that must become pipeline outputs first:**
   * GBDT gain and permutation importance, and Chronos permutation importance (C14);
   * hurdle classifier probability channels, raw and calibrated (for F16 and F17);
   * the per-positive regressor predictions (for T6);
   * the tuning `best_params` for T8.
7. **Execution.** `strikecast figures --store-root runs --out build/` runs as the last job of the DAG (CPU, under 1 h) and on the laptop after `scripts/sync_runs.py`. It writes a `build/MANIFEST.json` recording the store hash, the seeds used and the git commit.

A shortcut is possible but not recommended: parameterise the notebooks for papermill (`STORE_ROOT`, `OUT_DIR`). It still needs the name map and path rewiring of C2 and inherits the style problems of C5.

### (b) `scripts/slurm/submit_all.py`: one command, the whole DAG

Run it on the login node: `python scripts/slurm/submit_all.py --profile configs/cluster/<site>.yaml [--dry-run] [--resume]`.

**Job graph** (one SLURM array per box, split by resource class):

```
prep (CPU, ≤1 h): uv sync ONCE (--frozen --extra dev/figures) + envs/autogluon sync;
      import_golden_params; build+cache panel/series/feature-selection per experiment
      (incl. hurdle FS, currently cache:false); mkdir logs; pre-fetch HF Chronos weights
  ├─> tune_cpu[5]      count GBDT (C13)                           afterok:prep
  ├─> tune_chronos[1]  GPU, envs/autogluon                        afterok:prep
  ├─> cv_cpu[…], cv_gpu[…]        (only if Jan wants CV)          afterok:tune_* (else prep)
  ├─> test_cpu[…], test_gpu[…]    seeds×paradigms per model       afterok:tune_*
  ├─> comp_cv[hurdle×3, damage×1] CPU                             afterok:prep
  ├─> comp_test[…]                CPU                             afterok:comp_cv   (needs C18 fix)
  ├─> chronos_test[≈6]            GPU, envs/autogluon             afterok:tune_chronos
  ├─> fi_cpu[≈5] / fi_gpu[1]      GBDT perm. FI / Chronos FI      afterok:tune_*
  └─> report (CPU, ≤2 h)  afterany:<all of the above>
        strikecast report experiment=… (each) + cross-experiment + strikecast figures
        → build/{fig,tables,numbers.tex,MANIFEST.json}; exits non-zero if any stage incomplete
      └─> wandb_sync (optional; login/transfer node if compute nodes are offline)
```

**Dependency wiring:**

* `sbatch --parsable` returns each array's id, and the next array gets `--dependency=afterok:<id>` (or `afterany` for the report).
* Per-model precision (cv/test of model *m* waiting only for tune of *m*) can use one small array per tuned model, or `aftercorr` with index-aligned arrays. The tune set is small (6 arrays), so depending on the whole tune array is also acceptable.
* A model already tuned in the store gets no tune dependency.
* **A tunable model without `best_params.json` must fail, not silently fall back** (fixes C17). The skip check should compare the stage identity, or at least a `params_source`/params hash recorded in `state.json`.

**Resources per class** (the profile maps each class to a partition, time, CPUs, memory and GRES):

| Class | Jobs | Request |
|---|---|---|
| `cpu-short` | naive, linear, ARIMA, global GBDT cv/test, report | 4–8 CPU, 16 GB, 4 h |
| `cpu-long` | local/activity GBDT, hurdle, damage, CPU tuning, GBDT FI | 8 CPU, 32 GB, the partition maximum, `--requeue` |
| `gpu` | RNN cv/test, diff XGB/CatBoost | `--gres=gpu:1` (C19), 8 CPU, 32 GB, 12 h |
| `gpu-ag` | Chronos tune/test/FI | `--gres=gpu:1` (≥24 GB VRAM), 8 CPU, 64 GB; runs `envs/autogluon/.venv/bin/python -m strikecast.cli.main` with strikecast installed (path dependency) and wandb added (C15) |

The class comes from the model's device (as today) plus a `(family, stage, paradigm) → class/time` table seeded from the estimates above and corrected by the pilot runs. Throttle arrays (`--array=1-N%K`) to MaxSubmit and fair-share limits. Set `threads` from `SLURM_CPUS_PER_TASK` or `os.sched_getaffinity` (C26).

**Resume after a timeout:**

1. Engine `start_fold`: restart at `RunStore.resume_point`, which is already a legal retrain boundary, keeping the existing parts (C20).
2. Grouped paradigms persist per group: parts under `group=<g>/`, finished groups skipped on restart, or each region/tier group made its own sub-stage and hence its own array task, which also parallelises the local paradigm (C20).
3. `#SBATCH --signal=B:USR1@600 --requeue`. A bash `trap` forwards USR1 to Python, which flushes the buffered folds, writes `state=interrupted` and exits 99. The script then runs `scontrol requeue $SLURM_JOB_ID`, bounded by `SLURM_RESTART_COUNT`. If the site forbids requeue, rerunning `submit_all --resume` re-emits only unfinished work.
4. Optuna: `JournalStorage(JournalFileBackend(...))`, which is NFS-safe, or SQLite on node-local disk with a copy-back. At start-up, mark stale RUNNING trials as FAIL so a killed trial does not linger (C22). Tuning is already trial-granular.
5. Protect `state.json` (per-stage state files or an `fcntl` lock), because cv@42 and test@42 run concurrently (C23).
6. Record per-attempt start and end times in `state.json` so future sizing uses real durations (C21).

**Environment:**

* Build the venvs once, in `prep` or on the login node. Tasks call `.venv/bin/strikecast` directly, or `uv run --frozen --no-sync`; there is no per-task `uv sync` (C24).
* Put `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` and `HF_HOME` on shared storage. On offline nodes set `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline`.
* Export `PYTHONHASHSEED=0`, `MPLBACKEND=Agg` and `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.
* rsync `golden/converted/` (8.2 MB) to the cluster once (C28).
* `mkdir -p logs/slurm` before the first `sbatch` (C27).

**What we need from Jan about the cluster:**

1. Partition names and **MaxTime** per partition. Command: `sinfo -o "%P %l %D %c %m %G %f"`.
2. GPU types and memory, and the GRES syntax: is it `gpu:1` or `gpu:a40:1`? Is there a max GPU count per user?
3. The account and QOS names, and the per-user limits. Commands: `sacctmgr show assoc user=$USER format=account,qos,maxjobs,maxsubmit`, `sacctmgr show qos`, and `scontrol show config | grep -E "MaxArraySize|MaxJobCount|KillWait|OverTimeLimit"`.
4. Whether jobs may requeue, and the preemption policy. Is `--signal` honoured?
5. Storage: the scratch path, its quota and purge policy, and the home quota. What filesystem is it (NFS, Lustre, BeeGFS)? This decides SQLite/flock versus a journal.
6. **Internet on compute nodes**: PyPI/GitHub for `uv`, huggingface.co for the Chronos-2 weights, api.wandb.ai.
7. Node OS and **glibc version** (`ldd --version`; the wheels need ≥ 2.28), the NVIDIA driver (`nvidia-smi`; CUDA 12.8 seen in April), and the module system (`module avail`). Is `uv` allowed in `$HOME`? Is the old conda `/usr/local/anaconda3` still there?
8. CPUs and RAM per node, and whether node-local `$TMPDIR` exists.
9. The W&B entity, and where the API key lives on the cluster.

---

## Questions for Jan

1. **Cluster facts**: the 9 items above. Nothing can be sized or submitted without partitions, MaxTime, GRES syntax and whether compute nodes have internet.
2. **Count GBDT tuned params (C13)**: only LightGBM-Poisson survives, so CatBoost-Tweedie (Table 4 #1), XGBoost and LightGBM-Tweedie would be re-tuned and change. Do you still have the Colab `checkpoints_tune/*_best.pkl`? If not: re-tune, or pin the Appendix values for CatBoost-Tweedie?
3. **The job matrix (C16)**: confirm this matrix. Also, do you need CV runs for the activity and local paradigms? The thesis report never uses them, and skipping them saves about 20% of the jobs.
   * count GBDT: global, activity and local;
   * RNNs: global and activity only;
   * diff tuned models: 3 paradigms;
   * diff baselines: global only;
   * hurdle composite: 3 paradigms, with no standalone component runs;
   * damage: global only;
   * Chronos: local.
4. **Report format (C1)**: should the LaTeX `\input` generated tables, figures and numbers? Should the paper show seed means with CIs, with seed 42 as the thesis reference?
5. **Hurdle canonical numbers (C6, C7)**: Table 4 uses `leaderboard.csv` (0.79/2.00), while the stored predictions give 0.80/2.02. `tab:hurdle_bias` cannot be reproduced; the recomputed Sumy row is 6.79/4.48/−3.13, not 7.35/4.83/−4.17. Which numbers stand?
6. **Text fixes**:
   * Kruskal–Wallis p is 0.48, not ≈0.05 (C10);
   * 847 days and a 593/85/169 split (C11);
   * the top-5 rank-1 MAE is 0.77 (C8);
   * choose one top-20 definition (C9);
   * the Chronos steps range (C12).

   Apply them in the paper?
7. **W&B**: online (plan §1, `strict: true`) or offline on SLURM (this brief)?
8. **Cluster environment**: uv, which is locked, Python 3.13 and needs glibc ≥ 2.28, or the old conda `thesis` env (Python 3.12, unpinned)?
9. **Feature importance (C14)**: should the GBDT permutation importance and the Chronos importance become pipeline jobs (hours of compute), or stay frozen legacy artefacts?
10. **The `log` family**: drop it from the cross-experiment ranking? It is out of scope but still sits in `master_df`, and it affects no top-20 row.
