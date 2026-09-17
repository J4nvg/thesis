# Refactor plan: from notebooks to a reproducible experiment package

Status: draft v3.2, 2026-09-17. Decisions in §1 were agreed with Jan; §10 records what was resolved and the one question still open.

---

## 1. Decisions and constraints

| Topic | Decision |
|---|---|
| Purpose | The code base becomes the basis for a **publication** derived from the thesis. Two analyses must be added: confidence intervals from multiple seeds, and a quantified comparison of how much better one model type performs than another. |
| Behaviour | **Behaviour-preserving refactor.** The methodology stays exactly as it is today. Anything that looks methodologically questionable is *flagged* (§4), never changed. Where a flag is worth acting on later, it gets a config switch whose default reproduces current behaviour. New analyses (§7) are additive and consume saved predictions; they do not alter any pipeline. |
| Scope | Everything in the thesis: count regressors (GBDT + RNN with Poisson/Tweedie objectives), Chronos-2 zero-shot and fine-tuned, the hurdle model, the damage classifiers, the naive/seasonal-naive/ARIMA baselines, EDA and results analysis. The differenced-target branch **stays in scope**, including its MSE-on-differenced GBDT and RNN variants, its own feature-selection cache `features/diffreg_saved_sets.pkl` and its tuning artefacts under `checkpoints_tune_diff/`. **Out of scope:** the log-transformed regression family (`_log_regression`), which the thesis does not mention. The ARIMA baseline reported in the thesis (Table 4, local, p=7, d=1, q=1) was produced by the diff run, so that exact ARIMA code path is retained as a baseline inside the diff experiment. |
| Seeds | **Tune once, evaluate over N seeds.** Optuna runs with the fixed tuning seed under the Global paradigm, as in the thesis. Test and paradigm evaluation repeat per seed and are aggregated. |
| Storage | **Local run store is the source of truth.** Weights & Biases mirrors config, per-fold and per-trial metrics, tables and artifacts. Cluster nodes have internet, so W&B runs **online by default**; offline mode remains a config option. |
| Device policy | **Exactly as the thesis ran, set per model in config.** Count-family GBDTs on CPU, since their tuned builders use `device_type="cpu"`, `device="cpu"` and `task_type="CPU"`. Diff-branch XGBoost and CatBoost on GPU with LightGBM on CPU. Hurdle CatBoost on CPU. |
| Seed list | `eval_seeds` is an explicit config list, default `[42, 1, 2, 3, 4]`. Seeds can be added or removed later and completed runs are reused, because the run store is keyed by seed. Seed sweeps cover the test stage for all three paradigms; validation CV stays single-seed. |
| Package name | `strikecast`. |
| Config tooling | Hydra in the CLI layer over pydantic schemas. |

Non-goals: new models, new features, new splits, changes to metrics already reported.

---

## 2. Inventory of the current code base

### 2.1 Experiment families

| Family | Entry point | Target | Models | Tuning | Paradigms | Persisted outputs | In scope |
|---|---|---|---|---|---|---|---|
| count-gbdt | `_regression_GBDT.ipynb` (`tune_gbdt.sh` calls `_regression_GBDT.py`, **not in the repo**) | level counts | lightgbm/xgboost/catboost × {poisson, tweedie}; naive_last, naive_weekly, linear | Optuna 50 trials, TPE seed 42, MedianPruner(5), objective RMSSE_mean on validation folds | global, activity, local | `results/gbdt/` (523 files), `checkpoints/`, `checkpoints_tune/`, `features/countreg_saved_sets.pkl` | yes |
| count-rnn | `_regression_LSTM.ipynb` (`tune_lstm.sh` calls `_regression_LSTM.py`, **not in repo**) | level counts | {lstm, gru} × {poisson, tweedie, mse} × window {7, 14, 28} | same | global, activity | `results/lstm/` (421), `checkpoints/`, `checkpoints_tune/` | yes |
| chronos2 | `_chronos2.py` (no notebook) | level counts, AutoGluon `TimeSeriesDataFrame` | Chronos-2 zero-shot, Chronos-2 fine-tuned, naives | Optuna 12 trials on AutoGluon's 3-window internal validation (`fine_tune_lr`, `fine_tune_steps`) | local (regions as multivariate) | `results/chronos2/` (25), `checkpoints/chronos2_*` | yes |
| hurdle | `final_hurdle.ipynb` | binary event target + count target with positive-only sample weights | SelfPacedEnsemble(DecisionTree, depth 5, 100 estimators) classifier + CatBoost Tweedie(1.5) regressor; hurdle = p × count | none (fixed hyper-parameters) | global, activity, local | `results/finalhurdle/` (71): raw, sigmoid-calibrated and Venn-Abers views, gain and permutation importance | yes |
| damage | `damage_classifier.ipynb` | 4 binary damage targets (health, education, residential, energy) | SelfPacedEnsemble(DecisionTree) | none | global | **nothing persisted** (printed only) | yes |
| arima baseline | produced inside `_diff_regression` | first-differenced target, anchored inverse | ARIMA(7, 0, 1) on differences with `NaiveMean` fallback, which is the thesis's ARIMA(7,1,1) | none | local | `results/diff/*arima*` | yes, produced inside the diff experiment |
| diff (MSE branch) | `_diff_regression.ipynb` + `.py` | first difference | lightgbm/xgboost/catboost (MSE), {lstm, gru} × {7,14,28} (MSE), linear | Optuna 50 | global, activity, local | `results/diff/` (435), `checkpoints_diff/`, `checkpoints_tune_diff/`, `features/diffreg_saved_sets.pkl` | yes |
| log | `_log_regression.ipynb` + `.py` | log1p | same lineup as diff, arima d=1 | Optuna 50 | global, activity, local | `results/log/` (435), `checkpoints_log/`; `checkpoints_tune_log/` missing | **no** (archived) |
| prehurdle-clf | `event_classifiers_prehurdle.ipynb` | binary event | spe_dt, spe_lgbm, logistic, lightgbm, xgboost, catboost, naive | optional Optuna on winner | global | `stage1_*.csv/parquet` in cwd, **not in repo** | model lineup enters the registry; see §10 |
| prehurdle-reg | `regressor_part_prehurdle.ipynb` | level counts | lightgbm/xgboost/catboost × {default, tweedie}, linear, naive_last, naive_mean | none | global | none | same |
| analysis | `results/analyse_results.ipynb` | reads leaderboards and per-region CSVs by hard-coded names | Friedman + Nemenyi, critical-difference plots, calibration plots | | | `results/figs/` | rewired to the run store |
| eda | `eda.ipynb`, `eda_full.ipynb` | | | | | `data/for_eda/plots/` | rewired to `build_panel` |

### 2.2 The shared pipeline as it exists today

| # | Stage | Current implementation |
|---|---|---|
| 1 | Load regions, activity tiers, master parquet | `src/prevalent_functions.py::load_data` |
| 2 | Feature engineering to long panel | `get_engineered_features`; near-duplicate `get_engineered_features_damageclassifiers`; a third inline copy in `_chronos2.py` §2 |
| 3 | Future vs past covariate split | `split_future_and_past_cov` |
| 4 | Darts `TimeSeries` + windowed transforms (rsum7/14, rmean7/28, ewma14, expdecay7) | `src/ts_specific_tools.py::build_ts_and_apply_window_transformer` (window list hard-coded) |
| 5 | Static-covariate encoding, 70/10/20 split, CV view | `get_covs_and_encodings` (returns a 9-tuple) |
| 6 | Feature selection: LightGBM on everything, top-100 by mean gain | copy-pasted block in every notebook; cached as `features/<family>_saved_sets.pkl` (pickle of the whole 9-tuple) |
| 7 | Raw (un-windowed) past covariates for RNNs | `get_covs_and_encodings` called a second time |
| 8 | Naive scales from the training portion | `compute_naive_scales` |
| 9 | Optuna tuning per variant | inline; pickles `(best_params, study)` only after all trials finish |
| 10 | Validation-CV re-run with best params + baselines | inline; pickles `(long_df, fold_preds)` per model |
| 11 | Leaderboard | inline |
| 12 | Test: fit on train+val, predict daily, retrain weekly | `run_final_test` and per-family redefinitions |
| 13 | Activity-level and per-region paradigms, CV and test | wrappers; **not checkpointed** |
| 14 | Persist ~400 flat files per family | final cell |

### 2.3 Per-family differences that must be preserved as configuration

| Aspect | count (gbdt / rnn) | diff | hurdle | arima baseline | chronos2 |
|---|---|---|---|---|---|
| Target transform | identity | first difference, anchor on last actual | identity; binary + count | first difference, anchor on last actual | identity |
| `lags_past_covariates` | `[-1, -7, -14]` via `get_common_kwargs` | `[-1, -7, -14]` via `get_common_kwargs` | `[-1, -7, -14]` | none | known covariates only |
| `lags_future_covariates` | `(2, 7)` | `(2, 7)` | `(2, 7)` | none | |
| Loss | poisson / tweedie(1.5, tuned 1.1–1.9); RNN also MSE | MSE or RMSE for the GBDTs, MSE for the RNNs | SPE-DT + CatBoost Tweedie(1.5), positive-only weights | MLE | MASE (AutoGluon) |
| RNN post-processing | median of 200 samples when a likelihood is set; `exp` when `_count_log_link` | none | — | — | median quantile column |
| Local-model fallback | — | `NaiveMean` on exception, for ARIMA with d=0 on the differences | — | `NaiveMean` on any exception | — |
| Early stopping | patience 10 in default builder, 5 in tuned builder, both on `train_loss` | same as count | — | — | — |
| Retrain cadence | every 7 days, predict daily | same | same | same | fixed predictor, daily rolling context |
| Split | `split_after(0.7)` then `split_after(1/3)` | same | same | same | `train_test_split(prediction_length=round(0.2·n))` |
| Naive scales for test | train only | train only | separate `*_scales_test` | train only | train+val |
| Calibration | — | — | sigmoid per horizon (5-fold OOF), Venn-Abers | — | — |
| Feature selection<br>Figure 8 of the thesis matches this except for F27 to F29. | one `build_regressor("lightgbm_tweedie")` on level counts, objective tweedie, variance power 1.5, `device_type="gpu"`, top-100, shared with RNNs | its own selection, cached as `diffreg_saved_sets.pkl`, LightGBM with objective `regression` on CPU with 4 threads, run on the LEVEL target while the models train on differences | two selections: binary LightGBM with `is_unbalance` for the classifier, Poisson LightGBM with positive-only sample weights for the regressor, library-default hyper-parameters | none | none |
| Device | GBDTs on CPU: `device_type="cpu"`, `device="cpu"`, `task_type="CPU"` | XGBoost `device="cuda"`, CatBoost `task_type="GPU"`, LightGBM on CPU | CatBoost on CPU | CPU | — |

---

## 3. What blocks reuse today

1. **Seven copies of the backtest loop.** `src/evaluation_tools.py` has `run_expanding_cv`, `run_expanding_cv_iter`, `run_final_test`; the log and diff scripts each redefine all three plus per-activity and per-region wrappers; the hurdle notebook has `run_hurdle_cv`, `run_final_test`, and per-activity/per-region versions; the damage notebook has `run_damage_cv` and `run_final_test`; Chronos has `chronos2_rolling_long`. All implement the same schedule.
2. **Duplicated metrics and feature engineering.** Chronos re-implements `base_metrics`, `_scaled_metrics`, `evaluate_long`, `_skill`, and the heatmap. Three copies of the panel construction.
3. **Configuration is module-level globals.** Model lineups are hard-coded lists with commented-out alternatives. `REGRESSORS_TO_RUN` in the log script is dead. Thread count and `cuda`/`cpu` are hard-coded inside each builder.
4. **Seeds cannot vary.** `RANDOM_STATE = 42` everywhere. The seed is not part of any file name or cache key, so a rerun with another seed silently loads seed-42 caches and overwrites seed-42 results.
5. **Checkpointing is coarse and partial.** Studies are saved only after all 50 trials. Test, activity and local stages are never checkpointed. All result files are written in the last cell. Caches are keyed by model name, not by config. Everything is pickled darts / Optuna objects, tied to `darts==0.43.0` and `optuna==4.8.0`.
6. **Tracking is broken.** Comet ML is initialised with a hard-coded API key committed in three commits and never logs anything afterwards.
7. **Environment.** `requirements.txt` was deleted; the local `.venv` is empty; the cluster uses a conda env; `imbalanced-ensemble` was installed from a local patched fork (`/home/jan/thesispatch/imbalanced-ensemble`) that is not in the repo. Notebooks and `.py` exports have drifted, and two exports are missing.
8. **Repository hygiene.** ~120 MB of pickles and results tracked in git, 261 `:Zone.Identifier` files, results as ~1900 flat files whose meaning lives in the file name.
9. **Side effects.** `get_top_100_from_lgbm` writes a CSV to the working directory; `print` is used for all logging.

---

## 4. Methodology flags (documented, not changed)

Default behaviour is always the current one. Flags marked "publication" are worth a sentence in the paper.

| # | Observation | Location | Treatment |
|---|---|---|---|
| F1 | Neural covariate `Scaler` is fit on full-length covariates, including validation and test periods. | `src/evaluation_tools.py:267` `_maybe_scale_covs` | Preserve. Config `covariate_scaling.fit_on: full`; future option `train_only`. Publication. |
| F2 | The CV leaderboard is computed on the same validation folds the Optuna objective minimised. | tuning + "re-run val CV" cells | Preserve. `strikecast report` labels these "selection folds". |
| F3 | Chronos splits by `round(0.2·n)` steps while Darts families split by `split_after` fractions; test segments can differ by a day. | `_chronos2.py` §3 vs `split_series_list` | Preserve per family. Golden test records each family's exact test start date. |
| F4 | Test predictions are issued daily with 7-day horizons, so each date is scored up to 7 times with different horizons. Horizon errors overlap in time. | all `run_final_test*` | Preserve. Matters for §7: bootstrap must resample by forecast origin, not by row. |
| F5 | ARIMA has a `NaiveMean` fallback on any exception. | diff `run_expanding_cv` | Preserve as `local_model.fallback: naive_mean`. It applies to the diff experiment, which is where ARIMA lives. |
| F6 | Naive scales for the hurdle test use a separate scale set; regression families reuse training scales. | `final_hurdle.ipynb` cell 19 | Preserve per family. |
| F7 | Hurdle and damage classifiers have no hyper-parameter tuning. | `final_hurdle.ipynb` cell 6 | Preserve. |
| F8 | Feature selection uses one LightGBM on the training series with full-length covariates; the selected set is shared with the RNNs. The selector differs per family: the count family uses `build_regressor("lightgbm_tweedie")` on GPU, the diff family uses objective `regression` on CPU with 4 threads and selects on the level target while its models train on differences, and the hurdle family runs two library-default selectors, binary with `is_unbalance` and Poisson with positive-only weights. | FS blocks | Preserve per family. Selected names stored as JSON. |
| F9 | GPU XGBoost, GPU CatBoost and GPU RNN training are not bit-reproducible. | builders | Golden comparisons use tolerances; CPU-deterministic models compare exactly. The count-family GBDTs run on CPU and are therefore bit-reproducible; only the diff-branch GPU GBDTs and the RNNs are not. This is also the *reason* seed CIs are needed. |
| F10 | Early-stopping patience differs between default (10) and tuned (5) builders. | `NN_TRAINER_KWARGS` vs `_nn_trainer_kwargs` | Preserve both as `trainer.default` and `trainer.tuned`. |
| F11 | Optuna resume from SQLite does not restore the TPE sampler RNG state. | new design | Accept, document. Golden uses stored best params, never re-tuning. |
| F12 | The thesis describes tuning as "a three-fold expanding-window backtest". The code rolls daily across the validation segment with weekly retrains (about 12 retrain points, ~85 daily folds). | thesis §5 vs `run_expanding_cv_iter` | Reconcile the text in the publication. Publication. |
| F13 | The thesis's baseline paragraph says ARIMA orders are chosen by AICc and unit-root tests; the code fixes p=7, d=1, q=1, which is what Table 4 reports. | thesis §5 vs `build_regressor("arima")` | Reconcile the text in the publication. Publication. |
| F14 | Activity and local paradigms reuse the Global-tuned configuration. | all families | Preserve. Already stated in the thesis. |
| F15 | **Resolved.** Earlier drafts claimed the past-covariate lags differ between families. They do not. Every in-scope family uses `[-1, -7, -14]` from `get_common_kwargs`, including the diff branch via `COMMON_KWARGS_TAB = get_common_kwargs()`. Only the archived log family used `[-1]`. | `_diff_regression.py:271`, `src/prevalent_functions.py::get_common_kwargs` | No action. One shared lag block in config. |
| F16 | Per-process hash ordering. `get_engineered_features` and `split_future_and_past_cov` build column lists with `list(set(...))`, so covariate column order depends on `PYTHONHASHSEED` and differs between interpreter runs. That changes LightGBM tie-breaking during feature selection. Repeated legacy runs give different top-100 sets, measured Jaccard 0.73 to 0.95 against the cached sets, with all churn at the rank-100 boundary. The seed behind `features/*_saved_sets.pkl` is unknowable. | `get_engineered_features`, `split_future_and_past_cov`, FS blocks | The cached feature sets are DATA to load, not output to reproduce. Golden level G becomes "pipeline inputs bit-identical, selection loaded from cache". New runs pin `PYTHONHASHSEED=0` and record it in `env.json`. Publication. |
| F17 | Train plus validation boundary. `TRAIN_VAL_END = 0.70 + 0.10` evaluates to `0.7999999999999999` and is passed to `split_before`, so the validation CV view is two days shorter than train plus val, 676 days against 678 on the real panel. | `get_covs_and_encodings` | Preserve and pin with a unit test. Publication. |
| F18 | Static-covariate encoders are fit four times independently on full-length series, for target, past, future and raw past, before any split. Label encoding is harmless, but it is formally fit on the test period. | `get_covs_and_encodings` | Preserve. |
| F19 | The realised 70/10/20 split is two relative `split_after` calls. The test fraction is a rounding artefact rather than an exact 20%. | `split_series_list` | Preserve. |
| F20 | `split_future_and_past_cov` with an empty `exclude` list falls back to the default exclusion set, because the guard is `if not exclude`. An empty list cannot mean "exclude nothing". | `split_future_and_past_cov` | Preserve. Config keeps `null` and `[]` distinct only if a later switch asks for it. |
| F21 | `future_covariates` may name columns that do not exist in the panel, the de-suffixed weather names. Nothing checks, and the missing names are dropped silently. | `split_future_and_past_cov` | Preserve. The new code logs the dropped names. |
| F22 | A stray `index` column from `master.reset_index()` survives into the Chronos panel. It is only excluded downstream. | `_chronos2.py` §2 | Preserve. |
| F23 | Binarisation position differs between panel variants. The regressor variant binarises before the interaction features, the damage variant after the GDELT step. Column order is preserved per variant. | `get_engineered_features` vs `get_engineered_features_damageclassifiers` | Preserve. `build_panel(binarize=..., binarize_at=...)` reproduces both. |
| F24 | Regions missing from the activity map get NaN and are kept by the `!= 0` filter. | activity-tier filter | Preserve. |
| F25 | `top_k=100` counts lagged names, not base features, so only about 52 past and 8 future base features survive. | `get_top_100_from_lgbm` | Preserve. |
| F26 | `lags_future_covariates=(2, 7)` is a tuple, which darts reads as a span, not two explicit lags. A JSON round trip to a list would silently change the meaning. | `get_common_kwargs` | Serialisation records tuple-ness. Config schema keeps the type. |
| F27 | Figure 8 labels the hurdle count-head selector "Tweedie objective LGBM, >0 only". The code fits `LightGBMModel(objective="poisson")` with positive-only sample weights, so the ">0 only" part is implemented via `sample_weight` and the objective is Poisson, not Tweedie. | `final_hurdle.ipynb` cells 6-7 | Preserve the Poisson objective. Correct the figure or the text. Publication. |
| F28 | Figure 8 notes "all models get all future covariate features". The RNN variants receive only the feature-selected future covariates (`full_fut_covs`, about 8 of 27 columns, produced after `future_covs_list = [subset_safe(...)]`), while their past covariates are the full raw un-windowed set, MinMax scaled. Chronos-2 does receive all future covariates as `known_covariates_names` plus all raw past covariates. | `_regression_LSTM.ipynb`, `_regression_GBDT.ipynb`, `_chronos2.py` | Preserve. Correct the figure note for the RNN branch. Publication. |
| F29 | Figure 8 labels the diff-branch selector "Differenced Regression LGBM". The selector uses `objective="regression"` but is fit on the LEVEL target, because feature selection happens before differencing; only the downstream models train on differences. | `_diff_regression.py` FS block | Preserve. Clarify in the text. Publication. |

---

## 5. Target architecture

Package name: `strikecast`.

### 5.1 Layout

```
pyproject.toml                 uv-managed, locked; extras: gpu, autogluon, dev
src/strikecast/
  config/schema.py             pydantic models: ExperimentConfig, DataConfig, SplitConfig,
                               TransformConfig, ModelConfig, BacktestConfig, TuningConfig,
                               SeedConfig, TrackingConfig. Pure python, no Hydra import.
  data/
    load.py                    regions, activity tiers, master parquet
    panel.py                   build_panel(): the ONE feature-engineering function
                               (replaces get_engineered_features, the damage variant,
                               and the Chronos inline copy; takes binarize=[...])
    covariates.py              future/past split
    series.py                  TimeSeries construction, window transforms from config,
                               static encoding, 70/10/20 split, CV view -> SeriesBundle
    feature_selection.py       LightGBM gain ranking, top-k, stored as JSON
    autogluon.py               SeriesBundle -> TimeSeriesDataFrame adapter
    cache.py                   content-hash cache (parquet + json, never pickle)
  transforms/
    base.py                    TargetTransform: forward(series) / inverse(pred, context)
    identity.py  diff.py       (log1p is not ported)
  models/
    registry.py                @register("lightgbm_poisson") -> ModelSpec
    gbm.py  rnn.py  classical.py  classifiers.py  hurdle.py  chronos.py
    adapters.py                ModelAdapter: fit/predict wrapper that owns covariate
                               scaling (F1), sampling + median, log-link inversion,
                               local-model fallback (F5), device and thread selection
  backtest/
    schedule.py                fold schedule: start_frac, horizon, predict_stride,
                               retrain_stride -> [(t0, retrain: bool)]
    grouping.py                global | activity | local partitioning
    engine.py                  ExpandingWindowBacktest.run(...) -> PredictionSet
    hooks.py                   FoldHook protocol: Persist, Pruning, Wandb, Progress
  evaluation/
    metrics.py                 base, scaled, skill, classification, hurdle metrics
    aggregate.py               evaluate_long views
    calibration.py             sigmoid OOF, per-horizon calibrators, Venn-Abers
    leaderboard.py
    seeds.py                   cross-seed aggregation and CIs            (§7)
    comparison.py              paired model comparisons, effect sizes    (§7)
    stats.py                   Friedman, Nemenyi, CD diagrams (from analyse_results)
  tuning/optuna_runner.py      SQLite storage, load_if_exists, per-fold pruning hook
  store/run_store.py           run directories, config snapshot, state.json, stage hashes
  tracking/wandb_tracker.py    Tracker protocol; NoopTracker
  cli/main.py                  Hydra entry point: run, tune, evaluate, report, verify
configs/                       Hydra YAML groups (experiment, model, paradigm, backtest,
                               tracking, seeds, hydra/launcher)
scripts/slurm/                 array-job templates
notebooks/                     analysis + EDA only; read from the run store
tests/unit  tests/equivalence  tests/golden
docs/
archive/                       log-family notebooks and scripts, kept read-only
```

### 5.2 Core abstractions

**`SeriesBundle`** replaces the 9-tuple. Fields: `region_names`, `target_full`, `target_train/val/test`, `target_cv_view`, `past_covs` (windowed), `raw_past_covs`, `future_covs`, `fractions`, `cv_start_frac`, `train_val_end`, `activity_by_region`. Serialised as parquet plus a JSON manifest.

**`TargetTransform`.** `forward(series_list)` and `inverse(pred, context)`. `Identity` and `Diff` (anchor is the last actual before the forecast window, taken from `context`). Transform is set per experiment, `identity` for the count family and `diff` for the diff branch, with per-model overrides allowed. The ARIMA baseline lives in the diff experiment, which reproduces the diff run's ARIMA exactly.

**`ModelSpec`** (one per registry entry): `family`, `kind` (`global` | `local` | `naive` | `composite`), `build(params, ctx)`, `defaults`, `search_space(trial)`, `from_best_params(best)`, `needs_raw_past_covs`, `is_neural`, `postprocess`, `stochastic: bool` (drives whether seeds are swept, §7). Adding a model is one file and one YAML. Removing a model is deleting the name from the experiment YAML.

**`RunContext`**: `seed`, `device`, `threads`. Builders take this instead of reading globals. `device` is set per model in config, exactly as the thesis ran it, rather than auto-detected: the count-family GBDTs on CPU, diff-branch XGBoost and CatBoost on GPU, LightGBM always on CPU.

**`ExpandingWindowBacktest`.** One loop with the exact semantics of today's runners:

```
for t0 in range(start_idx, n_total - horizon + 1, predict_stride):
    retrain = (t0 - start_idx) % retrain_stride == 0
    if retrain: fit on [ts.drop_after(time_index[t0]) for ts in targets]
    predict n=horizon from the growing context [ts.drop_after(time_index[t0])]
    yield fold
```

`predict_stride=1, retrain_stride=7` reproduces every CV and test call in every family. The generator form feeds the Optuna pruner exactly as `run_expanding_cv_iter` does. `grouping` partitions series lists and restores original order, replacing the per-activity and per-region wrappers. Composite models (hurdle) return channels `prob`, `count`, `hurdle` per fold.

**`PredictionSet`.** Long dataframe: `region, fold, origin_date, horizon, date, y_true, y_pred[, channel]`. `origin_date` is new but derivable from today's `fold` index; it is what the block bootstrap in §7 resamples on. This is the only thing persisted from a backtest and what all evaluation consumes. No darts objects are ever pickled.

### 5.3 Run store and resume

```
runs/
  <experiment>/
    shared/
      panel.<hash>.parquet
      series.<hash>/                       SeriesBundle (parquet + manifest.json)
      feature_selection.<hash>.json
    tuning/<model_variant>/
      optuna.sqlite3  best_params.json  trials.csv
    <model_variant>/<paradigm>/seed=<s>/
      config.yaml  env.json  state.json
      cv/predictions/part-<fold_from>-<fold_to>.parquet
      cv/metrics/{global.json, per_region.csv, per_horizon.csv, per_region_horizon.csv,
                  per_activity_level.csv, per_activity_horizon.csv}
      test/...
      artifacts/                            importances, calibrators.json, plots
    report/                                 cross-seed and cross-model aggregates (§7)
```

Stage identity is `hash(resolved stage config + upstream hashes + seed)`. A complete stage is skipped. A partial stage resumes from its last completed retrain boundary and recomputes at most six folds. Prediction parts are written atomically after every retrain window, so a crash loses at most one week of folds. Tuning uses `optuna.create_study(storage="sqlite:///...", load_if_exists=True)` (F11).

### 5.4 Seeds

`SeedConfig(tuning_seed=42, eval_seeds=[42, 1, 2, 3, 4])`. `eval_seeds` is an explicit config list. Seeds can be added or removed later and completed runs are reused, because the run store is keyed by seed. Seed sweeps cover the test stage for all three paradigms; validation CV stays single-seed. The seed sets `random`, `numpy`, `torch`, each model's `random_state` / `random_seed`, AutoGluon's `random_seed`, the SPE classifier's `random_state`, and the `random_state` of the OOF calibration splits. The Optuna sampler uses `tuning_seed` only. Seed 42 with the current configs must reproduce today's outputs within §6 tolerances. Deterministic models (`stochastic: false`: naives, linear, ARIMA) run once and are broadcast across seeds in reports.

### 5.5 Tracking

`Tracker` protocol with `WandbTracker` and `NoopTracker`. One W&B run per `(experiment, model_variant, paradigm, seed)`, `group=experiment`, `job_type=stage`, `tags=[family, kind]`, `config=` the resolved YAML. Per fold: running global metrics with `step=fold`. Per trial: Optuna value and params. At stage end: leaderboard and per-view tables as `wandb.Table`, predictions and metrics as artifacts. Cluster nodes have internet, so runs are online by default; `WANDB_MODE=offline` stays available as a config option, with `wandb sync` from the laptop or a SLURM epilogue. The tracker never holds state the run store does not also hold.

### 5.6 CLI and cluster

```
strikecast tune   experiment=count  model=catboost_tweedie
strikecast run    experiment=count  model=lightgbm_poisson paradigm=global stage=cv,test seed=42
strikecast run -m experiment=count  model=lightgbm_poisson,xgboost_tweedie \
           paradigm=global,activity,local stage=test seed=1,2,3,4,5
strikecast report experiment=count           # §7 tables and figures
strikecast verify --golden golden/           # §6 comparisons
```

Hydra lives only in `cli/`; the library takes plain pydantic configs. Multirun with the submitit launcher, or a plain SLURM array reading a `jobs.txt` matrix, replaces the four `tune_*.sh` scripts. Jobs declare their resource class (`cpu`, `gpu`) from the model spec.

### 5.7 Environment

`pyproject.toml` with the historical pins (`darts==0.43.0`, `torch==2.9.1`, `lightning==2.5.6`, `pytorch-lightning==2.5.2`, `optuna==4.8.0`, `autogluon.timeseries==1.5.0`, `lightgbm==4.6.0`, `xgboost==3.1.3`, `catboost==1.2.10`, `pandas==3.0.2`, `numpy==2.4.4`, `scikit-learn==1.6.0`, `statsmodels==0.14.6`, `venn-abers==1.5.3`) and a `uv.lock`. Extras: `gpu`, `autogluon`, `dev`; the diff branch needs the `gpu` extra. `imbalanced-ensemble` is pinned as a git dependency to the branch behind https://github.com/ZhiningLiu1998/imbalanced-ensemble/pull/41. `uv sync` works on the cluster too; a `requirements.txt` is exported from the lock for conda users. Chronos-2 lives in a separate environment under `envs/autogluon/`, with pandas 2.3.3 and AutoGluon 1.5.0, where darts still coexists; the main environment keeps the historical pins above.

---

## 6. Behaviour-preservation strategy

Golden material is frozen in phase 0. Comparisons run at increasing cost.

| Level | What | Method | Tolerance |
|---|---|---|---|
| A | Panel | `build_panel()` vs `get_engineered_features` (and the damage and Chronos variants) on the real parquet | exact `assert_frame_equal` |
| B | Series bundle | values, time index, static covariates of every list | exact |
| C | Schedule | fold `t0` indices and retrain flags for every family's CV and test config | exact |
| D | Loop equivalence | new engine vs each old runner on a small synthetic panel with deterministic models (naive, linear, LightGBM CPU single-thread) | 1e-9 |
| E | Predictions vs `results/` | naive_last, naive_weekly, linear, ARIMA from `results/diff`, and the full count-family GBDT lineup, which is CPU-deterministic, with stored best params; compare `predictions_long_*.parquet` | 1e-6 |
| F | Metrics vs `results/` | diff-branch GPU GBDTs and the RNNs, with stored best params; compare `global_*.json` and per-view CSVs | family-specific tolerance, recorded in the golden report (F9) |
| G | Feature selection | pipeline inputs bit-identical, selection loaded from cache. The names extracted from `features/*_saved_sets.pkl` are data, not a target to reproduce, because column order and therefore LightGBM tie-breaking depend on `PYTHONHASHSEED` (F16). New runs pin `PYTHONHASHSEED=0`. | exact on the loaded set and on the FS input matrix; re-selection compared by Jaccard only, reported not asserted |
| H | Hurdle and calibration | hurdle predictions, sigmoid and Venn-Abers views vs `results/finalhurdle/` | 1e-6 for CPU CatBoost; calibration exact given identical inputs |
| I | Thesis tables | Table 4 and Appendix E rows recomputed from the run store | exact to reported precision |

`strikecast verify` runs A–D and G in CI on every push (no GPU, minutes). E, F, H and I run on demand and produce `golden_report.md`.

---

## 7. Publication analyses (additive)

Both analyses consume `PredictionSet`s from the run store and never touch the training pipelines.

### 7.1 Confidence intervals from multiple seeds

Two distinct sources of uncertainty are reported separately, because they answer different questions.

**Training variance (seeds).** For each `(model_variant, paradigm)` and each evaluation seed, the test stage is rerun with the Global-tuned parameters. Per seed, compute the existing global metrics (MAE, RMSE, SkillScore vs seasonal naive, MASE_mean, RMSSE_mean) and the per-horizon and per-tier views. Across seeds, report mean, standard deviation, and a t-interval at 95%. With five to ten seeds the t-interval is the honest choice; the report also prints min and max. Deterministic models get a point value and an explicit "deterministic" marker rather than a zero-width interval.

Which components are stochastic: GBDTs (row and column subsampling), RNNs (initialisation, batch order), SPE classifier (undersampling), Chronos fine-tuning, OOF calibration folds. Cost: the local paradigm dominates (20 fits per retrain point). Default `eval_seeds` for a first pass is `[42, 1, 2, 3, 4]`; the seed list is a config field, so extending it later reuses completed runs.

**Evaluation-sample variance (bootstrap).** Independently of seeds, a block bootstrap over forecast origins gives a CI for each metric on the fixed test window: resample `origin_date`s with replacement (a moving-block variant with block length 7 covers the overlap in F4), keep all regions and horizons attached to each origin, recompute the metric. 1000 replicates, percentile intervals. This is what a reviewer asks for when there is only one test period.

Output: `runs/<experiment>/report/leaderboard_ci.csv` with both intervals, and the Table 4 layout extended with `± seed SD` and `[bootstrap 95% CI]` columns.

### 7.2 Quantifying how much better one model type is than another

"Model type" is defined at two levels and both are reported: the best configuration per family (as in Table 4) and the family as a whole (all its variants and paradigms).

1. **Paired differences on identical forecast instances.** For models A and B, on every `(region, origin_date, horizon)` compute the loss difference `d = L(A) − L(B)` for squared and absolute error. Report the mean difference, its relative size (percent RMSE and MAE improvement), and a block-bootstrap 95% CI over origins. This is the primary "how much better" number and it is in the metric's own units.
2. **Diebold–Mariano test** on the same loss differentials, per horizon and pooled, with a HAC variance (Newey–West, bandwidth `horizon − 1`) to account for overlapping forecasts (F4). Reported as a p-value matrix alongside the difference matrix. Harvey–Leybourne–Newbold small-sample correction applied.
3. **Effect size.** Cliff's delta on per-origin loss differences (probability that A beats B on a random forecast origin, rescaled to [−1, 1]) so improvements are comparable across metrics and tiers.
4. **Rank statistics.** The Friedman and Nemenyi machinery already in `analyse_results.ipynb` moves to `evaluation/stats.py` unchanged; critical-difference diagrams are regenerated from the run store. This keeps continuity with the thesis.
5. **Seed-aware family comparison.** With N seeds, the per-seed metric of family A minus family B (same seed) gives N paired differences; report mean and t-interval. This separates "A is better" from "A happened to get a lucky seed".
6. **Breakdowns.** Every comparison is produced globally, per horizon, and per activity tier, because the thesis's main finding (RMSE degrading after day four; ARIMA close to the leaders in high-activity regions) lives in those breakdowns.

Output: `report/pairwise_<metric>.csv` (difference, CI, DM p-value, Cliff's delta for every pair), `report/family_comparison.csv`, `report/cd_<metric>.svg`, and a `report/summary.md` that renders the tables the paper needs.

---

## 8. Phases

Each phase ends with a green `strikecast verify` at the levels it introduces. Estimates are rough working days.

### P0 Safety and baseline (1–2 days)
- Delete the `comet_ml` initialisation from every notebook and script. The committed key is already invalid, so no history rewrite is needed.
- Tag the current commit `thesis-final`. Copy `results/`, `checkpoints*/`, `features/` to `golden/` with a hash manifest.
- Build a `uv` environment from the historical pins and **convert every pickle to durable formats while the old environment can still load them**: best params to JSON, selected feature names to JSON, fold predictions to parquet where not already saved. This is the only chance to recover `checkpoints_tune*/` and `features/` content.
- `pyproject.toml` and `uv.lock`, with `imbalanced-ensemble` pinned to the PR #41 branch. Verify `import darts, torch` on the laptop (CPU/MPS) and the cluster.
- Move the log notebooks and scripts to `archive/`. The diff branch stays in scope and is ported like the count family. Restore or drop the missing `.py` exports referenced by `tune_gbdt.sh` and `tune_lstm.sh`.
- Remove `:Zone.Identifier` files. `runs/`, `golden/`, `checkpoints*` and `results/` are git-ignored going forward, with W&B artifacts as the mirror.

Acceptance: fresh clone, `uv sync`, one existing script runs its data stage.

### P1 Data pipeline extraction (2 days)
- `strikecast.data` with `build_panel(binarize=[...])` covering all three copies, `SeriesBundle`, feature selection, cache.
- Old scripts import from `strikecast.data` with identical outputs. Levels A, B, G.

**Phase 1 status: delivered on 2026-09-17.** `strikecast.data` is in place with `load`, `panel`, `covariates`, `series`, `feature_selection` and `cache`, plus `strikecast.config.schema`. 52 tests pass, with 2 expected failures on exact feature-selection equality, see F16. Levels A and B are exact on the real data. Chronos-2 keeps its own environment under `envs/autogluon/`, pandas 2.3.3 and AutoGluon 1.5.0 with darts coexisting; see §5.7.

### P2 Backtest engine and transforms (3–4 days)
- `schedule`, `grouping`, `engine`, `hooks`; `Identity`, `Diff`; `ModelAdapter` with scaling, sampling, log-link, fallback.
- Levels C, D against every old runner, including the hurdle loop shape.

### P3 Models, config, seeds, run store, CLI (4–5 days)
- Registry with every in-scope variant from §2.1; search spaces next to builders; per-family YAML reproducing §2.3 exactly.
- `RunStore` with resume; Optuna on SQLite; `RunContext`; Hydra CLI.
- Levels E, F for the count families, the diff branch and the ARIMA baseline.

### P4 Tracking (1 day)
- `WandbTracker`, offline and online configs, sync script. Smoke-tested on a two-fold run.

### P5 Remaining families (5–7 days)
- Hurdle: `HurdleForecaster` composite, positive-only weights, `calibration.py`, three paradigms, importance and permutation outputs. Level H.
- Damage classifiers: multi-target loop; persistence that does not exist today.
- Chronos-2: AutoGluon adapter, tuning via AutoGluon internal validation, rolling backtest on the shared schedule, predictor saved under the run store.
- Pre-hurdle lineups enter the registry, see §10. Level I.

### P6 Multi-seed runs on the cluster (2–3 days of engineering, plus compute)
- Seed sweeps over the test stage and paradigms; SLURM array templates; run-store sync between cluster scratch and laptop.
- `evaluation/seeds.py` and the first `leaderboard_ci.csv`.

### P7 Publication statistics (3–4 days)
- `evaluation/comparison.py` (paired differences, block bootstrap, Diebold–Mariano, Cliff's delta), `evaluation/stats.py` (moved Friedman/Nemenyi/CD), `strikecast report` with §7 outputs.
- `analyse_results.ipynb` and the EDA notebooks rewired to the run store and `build_panel`.

### P8 Cleanup (1–2 days)
- Delete the old scripts once golden passes; README and docs; CI with unit and equivalence tests on a synthetic panel.

Total: roughly 23–30 working days plus cluster compute for the seed sweeps. P0 first; P1–P3 sequential; P4 alongside P3; P5 after P3; P6 after P4 and P5; P7 after P6 has at least one multi-seed family.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| GPU nondeterminism makes exact golden impossible for the diff branch's GPU XGBoost and CatBoost and for the RNNs | Level F with tolerances; the CPU-deterministic count family and baselines at level E; stored best params instead of re-tuning |
| Old pickles become unloadable if the environment moves on | P0 converts them first, in an environment built from the historical pins |
| Patched `imbalanced-ensemble` fork | Pinned as a git dependency to the PR #41 branch; without it the hurdle and damage families cannot be reproduced |
| Seed sweeps on the local paradigm are expensive | Seeds default to the test stage; run on SLURM; resume makes crashes cheap; deterministic models are not swept |
| Overlapping horizons violate independence assumptions in naive CIs | Block bootstrap by origin and HAC variance in DM tests (§7) |
| Hydra working-directory and launcher quirks | Hydra isolated in `cli/`; `hydra.job.chdir=false`; run store paths absolute |
| Repo size and history rewrite | Artefacts leave git in P0; history rewrite only with explicit go-ahead |

---

## 10. Resolved, and what is still open

| Question | Resolution |
|---|---|
| Diff branch | Stays in scope as a full experiment, GBDT and RNN variants included. Only the log family is archived. |
| `imbalanced-ensemble` patch | Pinned as a git dependency to the branch behind https://github.com/ZhiningLiu1998/imbalanced-ensemble/pull/41. |
| Comet key | The committed key is already invalid. The `comet_ml` initialisation is deleted from the code; git history is not rewritten. |
| Cluster internet | SLURM nodes reach the internet, so W&B runs online by default and offline mode is a config option. |
| Package name | `strikecast`. |
| Config tooling | Hydra in the CLI layer over pydantic schemas. |
| Number of seeds | `eval_seeds` is an explicit list, default five: `[42, 1, 2, 3, 4]`. It can grow or shrink later, and completed runs are reused. |
| Devices | Exactly as the thesis ran, per model in config. See §1. |
| Seed sweep scope | The test stage, for all three paradigms. Validation CV stays single-seed. |
| Artefact storage | `runs/`, `golden/`, `checkpoints*` and `results/` are git-ignored going forward, with W&B artifacts as the mirror. |

Still open:

1. **Pre-hurdle exploration notebooks.** The working assumption is that their model lineups enter the registry and the notebooks themselves move to `archive/`, rather than being kept runnable end-to-end. Jan can veto this.

---

## Appendix A: old → new mapping

| Old | New |
|---|---|
| `src/prevalent_functions.py::load_data` | `strikecast.data.load.load_inputs` |
| `get_engineered_features`, `get_engineered_features_damageclassifiers`, Chronos §2 | `strikecast.data.panel.build_panel(binarize=...)` |
| `split_future_and_past_cov` | `strikecast.data.covariates.split_covariates` |
| `build_ts_and_apply_window_transformer`, `get_covs_and_encodings`, `split_series_list` | `strikecast.data.series.build_bundle(config) -> SeriesBundle` |
| diff helpers in `_diff_regression` | `strikecast.transforms.Diff` |
| feature-selection blocks, `get_top_100_from_lgbm`, `clean_feature_names`, `subset_safe` | `strikecast.data.feature_selection.select_top_k` |
| `build_regressor`, `build_gbm_from_params`, `build_lstm_from_params`, `build_lstm_count`, `_build_lstm_from_best`, `get_event_classifier`, `get_count_regressor`, `get_damage_classifier`, `build_classifier` | `strikecast.models.registry` entries |
| `_suggest_*_params`, `SUGGESTERS_BY_FAMILY` | `ModelSpec.search_space` |
| `run_expanding_cv`, `run_expanding_cv_iter`, `run_final_test`, `run_expanding_cv_per_activity`, `run_expanding_cv_per_region`, `run_hurdle_cv*`, `run_damage_cv`, `chronos2_rolling_long` | `strikecast.backtest.engine.ExpandingWindowBacktest` + `grouping` |
| `_maybe_scale_covs`, `num_samples=200` + median, `_count_log_link`, `NaiveMean` fallback | `strikecast.models.adapters.ModelAdapter` |
| `collect_predictions_long`, `naive_collect_long` | `PredictionSet.from_folds` |
| `base_metrics`, `_scaled_metrics`, `_skill`, `_hurdle_metrics`, `_classif_metrics_hurdle` | `strikecast.evaluation.metrics` |
| `evaluate_long`, `evaluate_hurdle_long`, `evaluate_classif_long` | `strikecast.evaluation.aggregate.evaluate` with a metric-set parameter |
| calibration cells (sigmoid OOF, per-horizon, Venn-Abers) | `strikecast.evaluation.calibration` |
| `LightningPruningCallback`, study loop | `strikecast.tuning.optuna_runner` |
| `_save_group`, "Persist all results" cells | `strikecast.store.run_store` |
| `comet_ml.start(...)` | `strikecast.tracking.WandbTracker` |
| `tune_*.sh` | `scripts/slurm/array.sh` + Hydra multirun |
| `results/analyse_results.ipynb` statistics | `strikecast.evaluation.stats`, `strikecast.evaluation.comparison`, `strikecast report` |

## Appendix B: example configs

`configs/experiment/count.yaml`
```yaml
defaults:
  - /backtest: weekly_retrain
  - /paradigm: global
  - /tracking: wandb_online
  - _self_
name: count
data:
  target: act_drone_strike_on_ua
  activity_min_level: 1
  low_prevalence_ratio: 0.1
split: {train: 0.70, val: 0.10, test: 0.20}
transform: {kind: identity}                 # per-model override allowed
common_kwargs:
  lags: 7
  lags_past_covariates: [-1, -7, -14]
  lags_future_covariates: [2, 7]
  output_chunk_length: 7
  add_encoders: {cyclic: {future: [month, week, dayofyear, dayofweek, day]}}
feature_selection: {method: lightgbm_gain, objective: tweedie, top_k: 100}
# selection ran on GPU: device: gpu, tweedie_variance_power 1.5
models:
  - lightgbm_poisson
  - lightgbm_tweedie
  - xgboost_poisson
  - xgboost_tweedie
  - catboost_poisson
  - catboost_tweedie
  - lstm_poisson_w7 # ... all rnn variants
  - linear
  - naive_last
  - naive_weekly
  # ARIMA lives in the diff experiment, not here
device: {lightgbm: cpu, xgboost: cpu, catboost: cpu}   # the GBDTs ran on CPU
tuning: {n_trials: 50, sampler: tpe, pruner: {kind: median, n_warmup_steps: 5},
         objective: RMSSE_mean}
seeds: {tuning_seed: 42, eval_seeds: [42, 1, 2, 3, 4]}
```

`configs/model/catboost_tweedie.yaml`
```yaml
spec: catboost
stochastic: true
loss: {kind: tweedie, variance_power: 1.5}
defaults: {depth: 5, learning_rate: 0.05, iterations: 500, l2_leaf_reg: 3,
           subsample: 0.8, bootstrap_type: Bernoulli, boost_from_average: false}
device: cpu           # the count family ran on CPU, as today
search_space:
  depth: {int: [4, 8]}
  learning_rate: {float: [0.01, 0.2], log: true}
  iterations: {int: [200, 1000], step: 100}
  l2_leaf_reg: {float: [1.0, 5.0], step: 0.5}
  subsample: {float: [0.6, 1.0]}
  tweedie_variance_power: {float: [1.1, 1.9]}
```

`configs/experiment/diff.yaml`
```yaml
defaults:
  - /backtest: weekly_retrain
  - /paradigm: global
  - /tracking: wandb_online
  - _self_
name: diff
data:
  target: act_drone_strike_on_ua
  activity_min_level: 1
  low_prevalence_ratio: 0.1
split: {train: 0.70, val: 0.10, test: 0.20}
transform: {kind: diff}                     # anchor on the last actual
common_kwargs:
  lags: 7
  lags_past_covariates: [-1, -7, -14]
  lags_future_covariates: [2, 7]
  output_chunk_length: 7
  add_encoders: {cyclic: {future: [month, week, dayofyear, dayofweek, day]}}
feature_selection: {method: lightgbm_gain, objective: regression, top_k: 100}
models:
  - lightgbm_mse
  - xgboost_mse
  - catboost_mse
  - lstm_mse_w7 # ... all rnn variants, lstm and gru x w7/w14/w28, all MSE
  - linear
  - naive_last
  - naive_weekly
  - {name: arima, fallback: naive_mean}     # ARIMA on differences, d=0
device: {lightgbm: cpu, xgboost: cuda, catboost: gpu}
tuning: {n_trials: 50, sampler: tpe, pruner: {kind: median, n_warmup_steps: 5},
         objective: RMSSE_mean}
seeds: {tuning_seed: 42, eval_seeds: [42, 1, 2, 3, 4]}
```

`configs/backtest/weekly_retrain.yaml`
```yaml
horizon: 7
cv:   {start: cv_start_frac,  predict_stride: 1, retrain_stride: 7}
test: {start: train_val_end,  predict_stride: 1, retrain_stride: 7}
```

## Appendix C: what the common tasks look like afterwards

Add a model: create `src/strikecast/models/<name>.py` with one `@register` function returning a `ModelSpec`, add `configs/model/<name>.yaml`, and list the name under `models:` in the experiment YAML.

Remove a model: delete the name from the experiment YAML.

Rerun with three new seeds on the cluster, then produce the paper tables:
```
strikecast run -m experiment=count paradigm=global,activity,local stage=test seed=7,8,9 \
        hydra/launcher=slurm
strikecast report experiment=count
```
Tuning is not repeated because `runs/count/tuning/<model>/best_params.json` exists. The report picks up every seed present in the run store.
