# Audit A: data and features (thesis Methodology, Appendix variables, expdecay7)

Date 2026-09-26. Repo `/Users/jan/projects/bsc_thesis_code/thesis`, branch `refactor` (dirty tree). Read-only audit; nothing in the repo or in `/Users/jan/projects/writing` was changed.
Citations: thesis `main.tex:L` (`writing/thesis_writing_folder/main.tex`), paper `paper:L` (`writing/chapter?/main.tex`), reviewer notes `REVIEW:L` (`chapter?/REVIEW_features.tex`). Audit C items are cited by ID (e.g. C11) and not redone.

**Budget note.** The coordinator cut this audit short. Items marked *not re-verified* were not checked in this session; everything else was run or read here. The GBDT scale-invariance measurement finished after the first draft and is included in 2(c).

Environment for every run: `.venv/bin/python` 3.13.15, darts 0.43.0, pandas 3.0.2, lightgbm 4.6.0, numpy 2.4.4; `PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1` unless stated. Scratch scripts and their outputs are in this scratchpad: `impulse.py/.out`, `options.py/.out`, `fi_expdecay.py/.out`, `hurdle_fi_sets.py`, `zip_fs*.py/.out`, `zipw_seed*.json`, `zipw_seed2_rerun.json`.

---

## 1. Findings table

Categories: PORT-BUG, NOT-PORTED, TEXT≠CODE, OK-VERIFIED. Severity: Crit / High / Med / Low.

| ID | Area | Thesis (main.tex) | Legacy | Refactor | Cat. | Sev. | Evidence | Decision for Jan |
|---|---|---|---|---|---|---|---|---|
| A1 | expdecay7 | 512-516: leaky integrator `s(t)=x_t+α s(t−1)`, α≈0.905, 7-day half-life | `src/ts_specific_tools.py:46` `{"function":"mean","mode":"ewm","alpha":ed_alpha}` with `ed_alpha=halflife_to_alpha(7)` (`_regression_GBDT.py:187`; `src/feature_tools.py:34-35`) | `config/schema.py:63-106` (same dict); pinned by `tests/unit/test_series_split.py:93-98` | TEXT≠CODE (bug carried over on purpose) | **Crit** for the paper | `impulse.py`: legacy and refactor outputs are `DataFrame.equals`-identical. Impulse response 0.906 / 0.085 / 0.008 at d+0/1/2 and 6e-8 at d+7. The thesis filter gives 1 / 0.906 / 0.820 and 0.5 at d+7. pandas' `alpha` weights the current value. | Decided: add a switch. The fix is specified in §2. |
| A2 | ewma14 | 506-510: EMA span 14, α=2/15≈0.133 | `ts_specific_tools.py:45` `span:14` | `schema.py:103` | OK-VERIFIED (small start-up caveat) | Low | α=2/(14+1)=0.1333. pandas' default `adjust=True` (darts passes nothing else) normalises the weights, so values differ from the recursion early in each series. At day 20 the impulse gives 0.1403 against α=0.1333. The effect is gone after ~2 months (`impulse.out`). REVIEW [11] is right. | Optional footnote |
| A3 | Rolling windows | 495-504: rsum W∈{7,14}, rmean W∈{7,28}, `Σ_{i=0}^{W-1} x_{t-i}` | `ts_specific_tools.py:41-44` (`min_periods=1`) | `schema.py:95-102` | OK-VERIFIED | Low | Impulse: rsum7 stays 1 for d+0..d+6, rsum14 for d+0..d+13, and rmean28 settles at 1/28. The current day is included (`include_current=True`). `min_periods=1` means the first W−1 days of each series are partial means over fewer days, e.g. rmean28 = 1/21 on day 20. | none |
| A4 | Window kwargs | – | `ts_specific_tools.py:48-52` | `schema.py:115-119` | OK-VERIFIED | – | darts `timeseries.py:4229-4281`. `include_current=True` means shift 0. `forecasting_safe=True` only forbids `center=True` and back-fill, and ewm/rolling are safe anyway. `treat_na=0` fills only NaNs the transform itself adds. With `min_periods=1` (rolling) and 0 (ewm) it adds none, so `treat_na` is inert. No leakage, because past covariates enter at lags ≤ −1. | none |
| A5 | Calendar encoders | 480-490: calendar features are "future-known covariates" | GBDT: `src/prevalent_functions.py:237-241` (`cyclic.future`). Tuned RNNs: `_regression_LSTM.py:1012-1013, 1220-1221` (`cyclic.past`) | `schema.py:222-224`; `models/rnn.py:90-99`, F106 at `rnn.py:688` | TEXT≠CODE | Med | "Future-known" holds for the GBDTs and hurdle/damage only. The tuned RNNs encode the calendar as PAST covariates (F106, confirmed). | Say so where the RNNs are described |
| A6 | RNNs and future covariates | fig. 8 caption (532): DL models get raw MinMax-scaled past covariates | `src/evaluation_tools.py:267-272, 329-330` (`model.supports_future_covariates`) | `models/adapters.py:63-89, 198-199`; `run_stage.py:277` | TEXT≠CODE, and **REVIEW claim wrong** | Med | In darts 0.43.0 `BlockRNNModel` is a `MixedCovariatesTorchModel` (`block_rnn_model.py:256`), and `supports_future_covariates` returns **True** (checked). The RNNs therefore DO receive the feature-selected future covariates (about 9 weather columns), MinMax-scaled on full length (F1). REVIEW note [1] ("BlockRNNModel does not accept future covariates") is wrong for this darts version; F28 is right. | Correct REVIEW [1]. State in the paper that the RNNs get selected weather as future inputs. |
| A7 | Lags | 519-524: target t−1..t−7; past t−1, −7, −14; future lookback 2 + 7 ahead | `prevalent_functions.py:230-243` | `schema.py:305-383` (tuple kept, F26) | OK-VERIFIED | – | `lags=7`, `[-1,-7,-14]`, `(2,7)` is a span giving 9 future lags. The stored importance tables confirm the design: 291 lagged features = 37×3 past + 9×9 future + 7 target + 90 encoder + 2 static (`results/gbdt/importance_*.csv`). | – |
| A8 | RNN input window | 610: "fixed lookback window (the input chunk length)" | `_regression_LSTM.py:1063-1078, 1206-1210` (from the variant name w7/w14/w28) | `models/rnn.py:307-331, 484-494` | OK-VERIFIED (the text omits the values) | Low | The window comes from the variant, not the tuned params. `INPUT_LAGS=7` is only in the unused default branch. | Give 7/14/28 in the text (REVIEW [15] is right) |
| A9 | Feature selection: weather "exempt" | 528: "Cyclic calendar features and weather variables are exempt from this filter" | the FS blocks subset future covariates too (`_regression_GBDT.py:630-631`) | `data/feature_selection.py:256-278`; `data_stage.py:657` | **TEXT≠CODE** | High | Weather is filtered like everything else. Kept: countreg 9 of 25 weather and 0 of 2 holidays; diffreg 8; zipoisson 13 (`golden/converted/feature_sets/*.json`). Only the darts cyclic encoders are exempt, because the model generates them and `_keep` intersects with the input components. | Fix the text, or exempt future covariates in the publication runs (a methodology change) |
| A10 | Feature selection: "same objective as downstream" | 528 | count: tweedie selector for poisson/tweedie GBDTs and all RNNs; diff: `regression` on the LEVEL target; hurdle regressor: Poisson for a CatBoost-Tweedie head | `feature_selection.py:402-491` | TEXT≠CODE (already F8/F27/F29) | Med | as described | Reword |
| A11 | **zipoisson question** | fig. 8 "Tweedie-on-positives" branch | `final_hurdle.ipynb` cell 9-10 (fresh selection each run, not cached) | `hurdle.yaml:57-80` (`cache:false`, `zipoisson_regressor`, weighted) | OK-VERIFIED for the refactor; **progress-log conclusion is wrong** | High (for level H) | The thesis hurdle regressor's feature-importance file (`results/finalhurdle/feature_importance_regressor_per_horizon.csv`, 324 lagged features) contains **48 past + 9 future** base covariates. A fresh final_hurdle cell-9 selection (`zipoisson_regressor_config` plus positive-only weights) under `PYTHONHASHSEED=2` reproduces the **past set exactly (48/48, Jaccard 1.000; re-run twice, `zipw_seed2.json` == `zipw_seed2_rerun.json`)**; the future overlap is Jaccard 0.455. Against `zipoisson.json` the Jaccard is only 0.229 (past) and 0.375 (future). So `zipoisson.json` is **not** the hurdle count head's set. It sits close to `countreg` (J≈0.83 against the count FI set). An unweighted prehurdle LightGBM-Tweedie selection gives J=0.946 to countreg and 0.872 to zipoisson (`zip_fs3.out`), which fits a prehurdle **tweedie** selection. The classifier FI (28 past / 12 future) was *not* compared with a fresh classifier selection (budget). | Do not point `cache_path` at `zipoisson.json`. Keep the hurdle's fresh selection. The thesis regressor set can be rebuilt from the FI file (48 past names, 9 future names) if level H needs it. |
| A12 | FS determinism and re-runnability | – | FS depends on `PYTHONHASHSEED` (F16). countreg used `device_type="gpu"`. | `run.sbatch:61-62`, `tune.sbatch:61-62` export `PYTHONHASHSEED=0`; `seeds.py:100-108` only warns; `schema.py:431` `device: gpu`; `num_threads: null` (`count.yaml:59`) | NOT-PORTED (determinism controls) | High | The selection is re-runnable in the refactor: leave `cache_path` unset and `build_or_load_features` computes it (`data_stage.py:514-532`). It is not guaranteed reproducible: (i) the count selector on GPU uses a different histogram builder (Q9); (ii) `num_threads=None` falls back to the OpenMP default, which was 64 on the thesis cluster (`TaskPlugin=task/none`); (iii) `deterministic`/`force_row_wise` are not set (F124); (iv) the hurdle has `cache:false`, so **every hurdle job re-selects** (C-estimate "per-job LightGBM FS"). Churn is at the rank-100 boundary: the fresh hurdle selection under 3 hash seeds gives 49/51/48 past columns. Whether PyPI lightgbm 4.6.0 has OpenCL GPU support on the cluster is *not verified*. | Run the re-selection ONCE as its own job, on CPU with `deterministic=True`, a fixed `num_threads`, and `PYTHONHASHSEED=0`. Cache it (hurdle and damage too). Record the JSON in the run store. |
| A13 | Cached FS vs changed features | – | pickle load bypasses FS (`_regression_GBDT.py:595-601`) | `data_stage.py:491-506` returns the `cache_path` set **without checking** that it was selected on the same window config; `count.yaml:66`, `diff.yaml:61` point at golden sets | PORT-BUG risk (latent today) | **Crit** once expdecay7 changes | With the fix on and `cache_path` still set, the legacy names `ewm_expdecay7_*` would silently subset the NEW columns if the name is unchanged. The run would then carry a thesis selection over different features. | See §2(b)/(f): rename the feature and guard `cache_path` |
| A14 | Panel cache key | – | – | `data_stage.py:174-175` hashes `cfg.data` only (paths, not file content) | PORT-BUG (Low) | Low | A changed `master_combined_timeseries.parquet` reuses a stale `shared/panel.<hash>.parquet`. | Add the parquet's sha256 to `_panel_hash` |
| A15 | Split per family | 339-370 (846 days, 592/85/169) | `ts_specific_tools.py:5-19` (0.7, then 1/3); `:90-93` `TRAIN_VAL_END=0.7999…` | `series.py:146-161, 211-213`; `schema.py:133-158` | see **C11**; OK-VERIFIED port | Med (text) | Every darts family reports 593/85/169 and a CV view of 676 (`golden/converted/feature_sets/*.json` `region0`). Chronos: F3/F34. | Text fix is in C11 |
| A16 | Activity tiers, 20 regions | 297-311, 335 | `prevalent_functions.py:63-64` (`!= 0`); `data/fixed/regions_activity_cat.json` | `panel.py:212-216` (Q5); `load.py:69-73` | OK-VERIFIED (per T2/C32) | – | The golden sets have `n_regions=20`. The tier thresholds match (T2 in audit C). `inputs.regions` holds 25 regions, and 5 tier-0 regions are dropped. | – |
| A17 | Six base event categories | 468 "Cartesian product … six" | not in repo (parquet generation missing) | – | TEXT≠CODE (*partly verified*) | Low | REVIEW L… (HUNK D) is plausible: 2×2×2 would be 8, so "six" needs 3 party-country pairs × FPV. The panel contains `act_drone_strike_on_rus_in_ua_nofpv` and `…_rus_in_rus_nofpv` (appendix 1502, 1528), and no FPV series survives. *Not re-verified against the parquet in this session.* | Reword per REVIEW HUNK D |
| A18 | 115-dimensional covariate vector | 342-344 "dimension 115" | – | – | TEXT≠CODE (*not re-verified*) | Low | The appendix lists 112 base variables (T9 in audit C), and the paper's table has n=112 (P1). 115 is not explained; it may count the target, region and tier. | Reconcile 112 against 115 |
| A19 | Stale docstring | – | – | `data/series.py:350-387` says `to_autogluon_frame` is unimplementable, but `data/autogluon.py` exists | Low | Low | – | – |
| A20 | Dead legacy helper with the same bug | – | `src/feature_tools.py:80-83` `add_temporal_aggregates(ed=…)` uses `ewm(alpha=halflife_to_alpha(n), adjust=False)` on `x.shift(1)` | not ported | OK (not called on the thesis path, *not exhaustively verified*) | Low | Same misuse of α. | Delete or fix with the switch |

---

## 2. expdecay7 in depth

### 2(a) The bug, confirmed in legacy and refactor

* Legacy: `src/ts_specific_tools.py:46` builds `{"function":"mean","mode":"ewm","alpha":ed_alpha}`. Every script passes `ed_alpha=halflife_to_alpha(7)=2**(-1/7)=0.9057236642639067` (`_regression_GBDT.py:187`; the same call sits in the LSTM, diff and hurdle scripts).
* Refactor: `config/schema.py:104-105` builds the same dict from `DEFAULT_EWM_HALFLIFE=7.0` (`:63`). The only consumer is `data/series.py:277` (`WindowTransformer(**config.window.transformer_kwargs())`).
* darts hands the dict to `pd.DataFrame.ewm(alpha=…)` (`darts/timeseries.py:4005, 4226-4249`). In pandas, α is the weight on the CURRENT value, so the feature is `s_t = 0.906·x_t + 0.094·s_{t−1}` (with `adjust=True` normalisation).
* Measured (`impulse.out`; legacy and refactor columns bit-identical):

| filter | d+0 | d+1 | d+2 | d+7 | d+14 | d+21 |
|---|---|---|---|---|---|---|
| **code (expdecay7)** | 0.9057 | 0.0854 | 0.0081 | 6e-8 | 4e-15 | 0 |
| thesis eq. `exp_decay` (leaky integrator) | 1.0000 | 0.9057 | 0.8203 | 0.5000 | 0.2500 | 0.1250 |

  The code's feature is essentially the raw series. Consistently with that, **countreg keeps no raw past covariate at all**: all 37 selected past covariates are window features, 8 of them expdecay7 (`countreg.json`). So expdecay7 stood in for "the raw value at lag −1/−7/−14".

### 2(b) Every place the switch must touch

| Where | What | Action |
|---|---|---|
| `src/strikecast/config/schema.py:63-106` | `DEFAULT_EWM_HALFLIFE`, `halflife_to_alpha`, `default_window_transforms()` (the expdecay7 dict at `:104-105`) | Make the sixth transform depend on the new switch (2(f)) |
| `src/strikecast/config/schema.py:109-130` `WindowTransformConfig` | holds `transforms` | Add the switch field, and a validator that rejects a hand-written `transforms` list contradicting it |
| `src/strikecast/data/series.py:277` | consumer | No change if it is config-driven |
| `src/strikecast/pipeline/data_stage.py:178-189` | `_series_hash` covers `cfg.series` (so the transforms too), and `_features_hash` covers the series hash | Identity moves automatically (2(f)) |
| `src/strikecast/pipeline/data_stage.py:491-506` | the `cache_path` short-circuit | **Guard**: refuse a `cache_path` whose recorded window config differs from the run's. The converted JSONs record none, so treat a missing record as "legacy". (A13) |
| `configs/experiment/count.yaml:56-66`, `diff.yaml:53-61` | `cache_path: golden/converted/feature_sets/{countreg,diffreg}.json` | Publication runs: `cache_path: null` (fresh selection). Keep a golden-only override, e.g. `configs/series/thesis_legacy.yaml` or a `+golden=true` group, that sets the legacy mode **and** the cache path together |
| `configs/experiment/hurdle.yaml:57-80`, `damage.yaml:64-71` | fresh selections (`cache:false`) | Recompute automatically. Set `cache: true` so the selection is computed once, not per job (A12) |
| `configs/experiment/chronos2.yaml` | no window features (`_chronos2.py`; `results/chronos2/feature_importance_*.csv` has 0 `ewm`/`rolling` names) | Unaffected |
| RNN specs (`models/rnn.py:609, 636` `needs_raw_past_covs=True`) | RNN past inputs are the raw, un-windowed covariates | Directly unaffected. **Indirectly affected**: their future-covariate subset comes from the count/diff selection (A6), which changes after re-selection |
| `tests/unit/test_series_split.py:93-98` | asserts `transforms[-1]["alpha"] == halflife_to_alpha(7)` | Split into a legacy-mode test and a fixed-mode test (impulse 1, 0.906, 0.820, 0.5 at d+7) |
| `tests/golden/test_series_bundle_equality.py:122-134` | `SeriesConfig()` default against the legacy `ed_alpha` call | Pin `SeriesConfig(window=WindowTransformConfig(expdecay="legacy_alpha"))` explicitly, whatever the default |
| `tests/golden/test_feature_selection_equality.py:38, 93` | `EWM_HALFLIFE=7` via the legacy call | Legacy only; unchanged, but add a fixed-mode determinism test (same selection twice) |
| `tests/golden/test_pipeline_equality.py`, `test_hurdle_pipeline_equality.py`, `test_metrics_equality.py`, `test_schedule_equality.py` | build bundles with the default series config and compare with golden | Pin legacy mode explicitly (the E/F cases use the cached sets) |
| Golden fixtures | `golden/converted/feature_sets/{countreg,diffreg,zipoisson}.json` and their `past_covs.parquet`, `features/*_saved_sets.pkl` | Legacy-only; never used by fixed runs |
| Feature-name lists | 42 `ewm_expdecay7_*` names in the three cached sets (countreg 8, diffreg 23, zipoisson 11) | Kept for golden only. The rename in 2(f) makes the fixed names disjoint |
| Semantic categories (`tab:semantic_categories`, `main.tex:1577`; `results/analyse_results.ipynb` cell 52 `classify_feature`, per audit C T11) | substring rules on base-variable names | No change needed if the base name stays in the feature name (`<prefix>_<base>`). *Cell 52 was not re-read here*; confirm it has no `expdecay`-specific rule |
| Importance artefacts | `results/gbdt/importance_*.csv`, `results/finalhurdle/feature_importance_*.csv` | Legacy artefacts. F20/F23 must be regenerated after the re-run; that is blocked by C14 (FI not ported) |
| Docs | `docs/REFACTOR_PLAN.md:51` (§2.2 row 4), §4 (new flag), README | Add a methodology flag (e.g. F126) that records the bug and the switch |
| `src/feature_tools.py:80-83` | the same α misuse in a dead helper | Legacy oracle; leave as is |

### 2(c) Recommended "true 7-day half-life" implementation

Measured with the real darts `WindowTransformer` (`options.out`; an impulse on day 3 and a second one on day 40):

| variant (darts transform dict) | d+0 | d+1 | d+7 | relation to the thesis equation |
|---|---|---|---|---|
| A legacy `mean, ewm, alpha=0.906` | 0.906 | 0.085 | 0 | wrong filter |
| **C1 `sum, ewm, alpha=1−2^(−1/7)`** ≡ **C2 `sum, ewm, halflife=7`** | **1.000** | **0.906** | **0.500** | **identical to the literal loop `s_t = x_t + 2^(−1/7)·s_{t−1}`, `s_{−1}=0`: max abs diff 0.0** |
| B1 `mean, ewm, halflife=7` (pandas default `adjust=True`; REVIEW option B) | 0.288 (early) / 0.098 (day 40) | … | … | normalised EWM, α=1−2^(−1/7)=0.0943. The ratio to C1 varies: 3.47 at t=3, 7.04 at t=10, 9.28 at t=20, 10.52 at t=50, converging to 1/(1−a)=10.607 |
| B3 `mean, ewm, halflife=7, adjust=False` | 0.0943 | 0.0854 | 0.0471 | exactly α × C1 (constant ratio 10.6071 in the test), except that pandas sets `y_0=x_0`; with x_0≠0 the gap decays by a^(t+1) |

pandas `ewm(...).sum()` accepts only `adjust=True`; `adjust=False` raises `NotImplementedError` (verified). For `sum`, `adjust=True` IS the unnormalised recursion, so C1/C2 need no extra kwarg.

**Recommendation: C2**, i.e. `{"function": "sum", "mode": "ewm", "halflife": 7, "function_name": "leaky7"}`. It matches the thesis equation (`main.tex:512-516`) literally. The text then needs only α 0.905→0.906 (2^(−1/7)=0.9057), plus "s_{−1}=0 at the series start" and "about 10.6 times the EWM of the same half-life". It is expressible in darts with no custom function. Specifying `halflife` rather than a hand-computed α removes the slot ambiguity that caused the bug.

If Jan prefers the reviewer's unified EWM equation (`REVIEW:118-127`, VARIANT B), use **B3** (`mean`, `halflife=7`, `adjust=False`), not B1. B3 is the recursion `s_t=αx_t+(1−α)s_{t−1}` exactly as written, and a pure constant rescale of C2. B1's `adjust=True` warm-up distorts the first ~2 months of every series, and all of that falls in the training window.

**Does the scale matter to the models?**
* The GBDTs get no scaler: `_maybe_scale_covs` runs only for `is_neural` (`models/adapters.py:63-89, 198-199`).
* **Measured, not only argued** (`scale_inv.py`, `scale_inv2.py`, `scale_inv3.py`; outputs in `scale_inv*.out`). Setup: the count bundle, the countreg selection, the `countreg` LightGBM with 300 trees, fitted on train and predicting the CV view. Max |Δprediction| against C1:
  * C1×2 and C1×0.5 (exact powers of two): **0.0**, bit-identical;
  * C1×(1−a) (the B3-type rescale): **0.294**;
  * C1×1000: **0.452**;
  * B1: 0.630;
  * legacy expdecay7: 0.417.

  No expdecay value falls inside LightGBM's `kZeroThreshold` (min non-zero 1.2e-32; 0 values ≤1e-35 before or after scaling), so the cause is floating-point rounding of the bin boundaries. One changed split cascades, as in F124. **The GBDTs are therefore invariant to scale in principle but not bit-for-bit.** C2 and B3 are statistically equivalent choices, but they are not the same run. Whichever is chosen has to be fixed before the selection re-run and the tuning.
* The RNNs never see the window features (`needs_raw_past_covs=True`), so MinMax scaling is irrelevant to the choice.
* The diff `linear` baseline (unregularised OLS, `_diff_regression.py:360-365`; `models/classical.py:117-133`) uses the windowed covariates and is scale-equivariant.
* The hurdle's SPE decision trees and CatBoost head are order-based.

**Conclusion: C2 and B3 differ only by a constant, so they are equivalent in expectation for every model. They are not bit-identical: non-power-of-two rescaling perturbs LightGBM at the F124 level. Choose on text fidelity: C2, and freeze it before re-running the selection.**

### 2(d) How much the thesis results depended on expdecay7

Selected sets (`golden/converted/feature_sets/*.json`):

| selection (used by) | past kept | of which expdecay7 | raw past kept |
|---|---|---|---|
| countreg (count GBDTs; the RNNs' future subset) | 37 | **8** | 0 |
| diffreg (diff GBDTs, linear; the diff RNNs' future subset) | 52 | **23** | 3 |
| zipoisson.json (not used by the final hurdle; A11) | 38 | 11 | 0 |
| thesis hurdle regressor, from its FI file | 48 | 11 | – |
| thesis hurdle classifier, from its FI file | 28 | 9 | – |

Stored importances (`fi_expdecay.out`, from `results/gbdt/importance_all.csv`; 24 of the 291 lagged features are expdecay7):

| model | best gain rank | expdecay7 in gain top-15 | share of top-15 gain | share of all gain | best perm rank | in perm top-15 |
|---|---|---|---|---|---|---|
| **activity tier 1 CatBoost-Tweedie** (Fig. `importancesharebycategory_grouped`) | **1** | **6** | **41.6%** | 16.6% | **1** | 6 (47% of top-15 perm) |
| **activity tier 2 CatBoost-Tweedie** | **2** | 4 | **30.2%** | 14.3% | 14 | 1 |
| activity tier 3 CatBoost-Tweedie | 38 | 0 | 0% | 5.1% | 26 | 0 |
| global LightGBM-Poisson | 24 | 0 | 0% | 2.6% | 23 | 0 |
| tier 1/2/3 LightGBM-Poisson | 3 / 4 / 24 | 5 / 1 / 0 | 33% / 7% / 0% | 17% / 15% / 5% | – | – |

Tier-1 CatBoost top-15 by gain: rank 1 `ewm_expdecay7_acled_other_ua_disrupted_weapons_use_lag-1`, rank 3 the same feature at lag −14, rank 6 `…act_total_damage_events_lag-7`, rank 8 `…act_total_daily_strike_events_lag-14`, and ranks 13 and 14. Tier 2: ranks 2 and 3 are `ewm_expdecay7_act_total_daily_strike_events` at lags −7 and −1.

The hurdle count head has 0 expdecay7 features in its top 15 (best rank 20; 11.4% of the importance). Chronos-2 is unaffected (no window features).

So the thesis's **tier-1 "conflict-and-damage" and tier-2 "autoregressive" shares rest to a large part on expdecay7 columns**. Because those columns are ≈ raw values, the interpretation "recent raw conflict counts matter" survives; any reading as "lingering or decaying effects" does not. Tier 3 and global are barely affected. Headline accuracy impact: unknown until the re-run. The feature is nearly duplicated by the raw value at the same lag. After the fix, the leaky integrator will carry information that the rolling and EWMA features only partly carry, so a small shift in the GBDT, diff-linear and top-20 membership (C9) is plausible.

### 2(e) Sentences that need rewording

* `main.tex:506` "two exponential filters are applied", `:512` "An Exponential Decay (leaky integrator) filter, simulating the lingering effect of discrete events", `:516` "decay factor α ≈ 0.905 … half-life of 7 days". After C2 these are correct, apart from 0.905→0.906 and the `s_{−1}=0` note. With B3, use REVIEW VARIANT B (`REVIEW:125-127`).
* `paper:256-266` (the same text as the thesis) and `paper:287` "rolling and exponentially decayed signals": correct after the fix.
* `main.tex:1117-1119` (tier 1 "conflict-and-damage features carry both rankings (62% gain, 48% permutation)"; tier 2 "autoregressive history (45%)") and `main.tex:1147-1150` (Discussion, the same numbers): these shares are partly expdecay7 mass (2(d)). They will change after the re-run and must be regenerated (blocked by C14). Nowhere does the thesis read expdecay7 as a week-long memory explicitly. But "autoregressive strike history" in tier 2 is mostly `ewm_expdecay7_act_total_daily_strike_events`, which in the thesis run was effectively yesterday's national total.
* `main.tex:1130` "Chronos-2 is not supplied with the engineered lag, rolling-window and exponential-decay covariates": stays true.
* `main.tex:1152` "recent local history is nearly sufficient" and `main.tex:1163` (Conclusion, "local autoregressive history and meteorological conditions"): re-check after the re-run.
* Appendix figure `fig:top_fi_per_activity` (`main.tex:1611`) lists `ewm_expdecay7_*` names: regenerate.

### 2(f) Config key, default, and run-store identity

* Key: `series.window.expdecay: Literal["legacy_alpha", "halflife"]` (in `WindowTransformConfig`, `schema.py:109`), plus `series.window.expdecay_halflife: float = 7.0`. `default_window_transforms(mode)` emits:
  * `legacy_alpha`: `{"function":"mean","mode":"ewm","alpha":2**(-1/7),"function_name":"expdecay7"}` (bit-identical to the thesis);
  * `halflife`: `{"function":"sum","mode":"ewm","halflife":7.0,"function_name":"leaky7"}`.
* **Default `halflife`** (Jan's decision: publication runs use the fix). Every golden test and a `thesis_legacy` config group set `legacy_alpha` explicitly, together with the golden `cache_path`s. This breaks the plan's "default = thesis behaviour" rule (§1, `schema.py:202-212`), so record it as the one documented exception (new §4 flag).
* **Different `function_name`** (`leaky7`, not `expdecay7`) gives column names `ewm_leaky7_<base>`. They can never collide with legacy feature names, cached selections, W&B feature tables or importance CSVs, and a legacy `cache_path` then fails loudly: `subset_safe` finds none of the expdecay7 names, and the A13 guard refuses.
* **Identity.** The transform dict is part of `SeriesConfig`, so it changes:
  * `_series_hash`, and through it `_features_hash` (`data_stage.py:178-189`);
  * every `DataArtifacts.upstream` (`data_stage.py:110-112`);
  * `ExperimentConfig.resolved_hash` (`schema.py:893-908`).

  Legacy and fixed runs therefore get disjoint `shared/series.<hash>/` and `feature_selection.<hash>.json` entries and disjoint stage identities.

  **Caveat to check before relying on it:** audit C (C17) found that `make_jobs`' skip check keys on the stage name, not the identity (`make_jobs.py:31-37, 208`). A fixed run could therefore be skipped as "complete" because a legacy run exists under the same `runs/<exp>/<model>/<paradigm>/seed=…/<stage>` path. Either put the series hash (or a `features=legacy|v2` tag) into the run-directory path, or make the skip compare identities. Otherwise legacy and fixed runs collide in the store layout even though their hashes differ.

---

## 3. What the data-level tests actually prove

| Test | Asserts | Strength | Gaps |
|---|---|---|---|
| `tests/golden/test_panel_equality.py` (level A) | `assert_frame_equal(check_dtype=True)` of the regressor, binarised, damage (×4) and Chronos panels against the legacy functions **run today, in-process**; the global weather column order | Exact | Compares with legacy code on today's pandas 3.0.2, not with the thesis-time frames. The Chronos oracle is a **transcription** in the test (`:84-122`), not `_chronos2.py` itself. |
| `test_covariates_equality.py` (A) | exact list equality of the past/future/holiday splits, including the empty-exclude quirk | Exact | same in-process caveat |
| `test_series_bundle_equality.py` (B) | exact `np.testing.assert_array_equal` of every bundle list (values, index, components, statics) against `build_ts_and_apply_window_transformer` + `get_covs_and_encodings`, with the legacy `ed_alpha` | Exact, **count/regressor panel only** | No hurdle, damage or Chronos bundle. The `subset_components` test uses a **stand-in** selection (first 40/10 components, `:194-195`), not the real cached sets. Pins the buggy expdecay7 through `SeriesConfig()` defaults (`:132`), so it breaks as soon as the default flips unless it is pinned (2(b)). |
| `test_feature_selection_equality.py` (G) | **both equality tests are `xfail(strict=False)`** (`:190, 199`), so they can never fail. The only hard asserts are non-emptiness, subset-of-available, a monotone gain table and a JSON round trip. | **Weak**: overlap is only printed | Marked `slow`. In the earlier golden run (245 passed, 4 skipped, 52 s) the module seems not to have been collected or run (no xfail/xpass in the summary; `pyproject.toml:153-154` has no marker filter). *Not re-run here.* The count selector runs on CPU although the thesis used GPU. **No test at all for the hurdle/damage selectors** (zipoisson_*), and none checks that a fresh selection is deterministic across two processes. |
| `tests/unit/test_series_split.py` | the split fractions/lengths on a synthetic panel; `:93-98` pins the default transform names and the **legacy α** | Exact, synthetic | Pins the bug (2(b)) |
| `tests/unit/test_feature_names.py` | `parse_feature_names` equals legacy `clean_feature_names` (`:38`); the suffix quirks (`:73, :85`) | Exact | – |
| `tests/unit/test_cache.py` | `content_hash` stability and order sensitivity; `cached_parquet` builds once | Exact | Does not cover `data_stage` hashing or the A14 data-content gap |

**Bottom line.** Levels A and B prove that the refactor equals the legacy *code* on today's data and today's pandas. They do not prove equality with the *exact thesis inputs*. Those inputs were the pickled post-selection covariates (`features/*.pkl`, loaded at `_regression_GBDT.py:595-601`), computed with an older pandas; F124 measured ~1 ULP differences in `ewm_*` columns. Only the E/F pipeline tests (audit C scope) use the converted cached sets. The feature selection itself (G) is asserted by nothing.

---

## 4. Questions for Jan

1. **expdecay7 form**: literal leaky integrator (C2, `sum`/`halflife=7`; matches your equation) or normalised EWM (B3, `mean`/`halflife=7`/`adjust=False`; matches the reviewer's unified equation)? The models are indifferent in expectation (2(c)); a rescale is not bit-identical in LightGBM (Δpred up to 0.29 on one fit). Only the text differs. Also OK with renaming the feature to `leaky7`?
2. **Default flip**: make the fixed filter the schema default (with legacy pinned in the golden tests and a `thesis_legacy` group), as recommended? It is the first exception to "default = thesis behaviour".
3. **Weather exemption (A9)**: the thesis says weather is exempt from selection, but the code filters it (9 of 25 kept). For the re-run, fix the text, or actually exempt all future covariates?
4. **Re-selection protocol (A12)**: OK to run the new selection once per family/head on CPU with `deterministic=True`, fixed `num_threads`, `PYTHONHASHSEED=0`, cached, instead of on GPU as in the thesis? Should the hurdle and damage selections be cached too (they are recomputed per job today)?
5. **zipoisson (A11)**: the evidence says `zipoisson.json` is not the thesis hurdle's regressor set; the thesis hurdle re-selected fresh each run. Do you remember producing `zipoisson_saved_sets.pkl` in `regressor_part_prehurdle.ipynb` with LightGBM-Tweedie? For level H, rebuild the thesis hurdle sets from the FI CSVs instead?
6. **Store layout (2(f) caveat, C17)**: put a feature-version tag into the run path so legacy and fixed runs never share a directory?
7. **A18**: where does "covariate vector of dimension 115" (`main.tex:342-344`) come from, against the 112 listed variables?

(New fact from the coordinator, recorded for completeness: the count CatBoost/XGBoost tuning results from Colab are permanently lost, and `lightgbm_tweedie_best.pkl` exists on the cluster. This does not change any data/feature finding. It does mean that re-tuning is needed anyway, and the re-tune should use the fixed features.)
