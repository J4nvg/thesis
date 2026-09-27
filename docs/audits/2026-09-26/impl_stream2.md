# Stream 2 (WP2) hand-off: Chronos-2 through the pipeline + feature importance

Status: DONE except the items under "Open issues" (2026-09-26). Nothing committed.

## 1. Chronos-2 end to end (audit C15)

### What changed
| file | change |
|---|---|
| `src/strikecast/pipeline/context.py` | `get_spec` always imports `strikecast.models.chronos` (`_EXTRA_SPEC_MODULES`), so `make_jobs experiment=chronos2` and the CLI resolve `chronos2_*`. In `envs/autogluon` the registry module itself cannot import (`classifiers` needs `imbalanced-ensemble`); the per-module fallback + chronos covers it. |
| `src/strikecast/pipeline/chronos_stage.py` (new) | `ChronosData` (a `DataArtifacts`), `prepare_chronos_data` (`_chronos2.py` §1-3: `build_panel_legacy_chronos` -> AutoGluon frames; 27 known + 84 past covariates incl. the stray `index`, F123; NO feature selection, NO window features -> the expdecay switch does not apply), `run_chronos_stage` (`_chronos2.py` §7/9/10), `tune_chronos_model` (§8), `load_stage_predictor`. |
| `src/strikecast/tuning/autogluon_runner.py` (new) | the fine-tune study on `optuna_runner.tune`: search space `models.chronos.search_space` (lr log-U[1e-6,1e-4], steps 200..3000 step 100), `TPESampler(seed=tuning_seed)`, 12 trials, objective = AutoGluon internal validation MASE (`-leaderboard().score_val`, F129/F130), `gc_after_trial=True`, pruner = Optuna's default `MedianPruner` never consulted (= legacy "no pruner", B12). Trial predictors are fit in `tuning/<model>/trials/trial_NNN/` and deleted after the trial. |
| `pipeline/data_stage.py` (3 lines) | `prepare_data` -> `prepare_chronos_data` when `data.panel_variant == "chronos"`. |
| `pipeline/run_stage.py` (2 blocks) | `run_stage` dispatches `kind == "chronos"` to `run_chronos_stage` (Stream 3 then wrapped it with `_accepted(...)` for `max_folds`/`allow_default_params`, both supported); `make_forecaster` raises a pointer to the chronos stage for that kind (like composites). |
| `pipeline/tune_stage.py` (1 block) | `tune_model` dispatches `kind == "chronos"` to `tune_chronos_model`. |
| `envs/autogluon/pyproject.toml`, `uv.lock`, `README.md` | the env is an (empty) editable package whose `.pth` puts `<repo>/src` on `sys.path` (hatchling `dev-mode-dirs`): `import strikecast` works with no `PYTHONPATH`. Runtime deps added with the main lock's pins: hydra-core 1.3.7, omegaconf 2.3.0, pydantic 2.13.0, pyyaml 6.0.3, wandb 0.30.0, venn-abers 1.5.3, scikit-posthocs 0.17.0 (seaborn 0.13.2 transitively). Re-locked (`uv lock --project envs/autogluon`). The `.pth` holds the ABSOLUTE checkout path: `uv sync --project envs/autogluon` must run on the checkout that runs the jobs. |
| `scripts/import_golden_chronos.py` (new) | `golden/checkpoints/chronos2_best/best_params.json` + `golden/results/chronos2/optuna_trials.csv` -> `<store>/chronos2/tuning/chronos2_fine_tuned/{best_params.json,trials.csv}` in the `BEST_PARAMS_KEYS` format (best_value 1.1009486 from the CSV, cross-checked against the sidecar). Idempotent; `--force`; `--dry-run`. Separate file so Stream 1's `import_golden_params.py` was not touched. |

### Run-store layout of a Chronos run
```
<store>/chronos2/tuning/chronos2_fine_tuned/{optuna.sqlite3,best_params.json,trials.csv}
<store>/chronos2/<model>/global/seed=<s>/
    state.json config.yaml env.json
    artifacts/predictor/            AutoGluon TimeSeriesPredictor (fit once; ~6-20 MB; base weights in HF_HOME)
    artifacts/predictor.json        fit identity (hyper-params, seed, upstream hashes); a mismatch refits
    artifacts/internal_leaderboard.json   AutoGluon's internal val score (-MASE)
    test/predictions/part-*.parquet test/metrics/{global.json,per_region,per_horizon,per_region_horizon}.csv
    importance/...                  see section 2
```
The metric views are `chronos_evaluate_long` (the `_chronos2.py` metric set: MAE, RMSE, MedAE, ME,
ZeroAcc, n, MASE_mean, MASE_median, RMSSE_mean; no deviances, no activity views), scales train-only.

### Commands (repo root; AutoGluon env)
```bash
uv sync --project envs/autogluon                       # setup job, once, on the cluster checkout
PY="envs/autogluon/.venv/bin/python -m strikecast.cli.main"
# publication (D4: re-tune; Chronos has no window features, but it is re-tuned like every model)
$PY tune experiment=chronos2 model=chronos2_fine_tuned --store-root <root>
$PY run  experiment=chronos2 model=chronos2_zero_shot  paradigm=global stage=test seed=42 --store-root <root>
$PY run  experiment=chronos2 model=chronos2_fine_tuned paradigm=global stage=test seed=42 --store-root <root>
$PY importance experiment=chronos2 model=chronos2_fine_tuned,chronos2_zero_shot seed=42 --store-root <root>
# thesis-mode reproduction (legacy store root): import the thesis winner instead of tuning
python scripts/import_golden_chronos.py --store-root <legacy_root>
```
- **Internet**: yes, once. The first fit downloads `autogluon/chronos-2` (~460 MB) from Hugging Face.
  Pre-fetch in the setup job with `HF_HOME=<shared>` exported for all Chronos jobs:
  `envs/autogluon/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('autogluon/chronos-2')"`.
- **No `cv` stage** (F131): `stage=cv` is skipped with a log line (outcome `skipped`), so the CLI
  default `stage=cv,test` is harmless, but `make_jobs` EMITS chronos `cv` jobs (a GPU job doing only
  data prep) -- Stream 3: follow `cfg.stages`.
- **Zero-shot is deterministic** (`stochastic=False`): `run_experiment` runs it under the first
  seed only; `chronos2_fine_tuned` is a full fine-tune per seed.
- `chronos2_fine_tuned` without `tuning/.../best_params.json` fails (C17 guard) unless
  `--allow-default-params` (then it uses the thesis winner, lr 8.72e-5 / 1500 steps).
- `--max-folds N` works (part of the stage identity); the predictor is reused across stage
  identities when its fit identity matches (a benchmark run does not force a second fit).
- W&B: the default `tracking: wandb_online` works in the env (wandb installed); `tracking=noop` offline.
- Interrupt (C21): `KeyboardInterrupt`/`StageInterrupted` marks the stage `interrupted`.

### Run-time estimates (for `--time`)
| job | evidence | estimate (A40 GPU) |
|---|---|---|
| tune chronos2_fine_tuned (12 trials) | thesis log: one fit at 1500 steps = 636 s; steps 200..3000 | ~2.3 h expected, 4.2 h worst case -> `--time 8:00:00`, `--gres=gpu:1` |
| test chronos2_zero_shot | thesis log fit 100 s; ~2 s/fold GPU; laptop CPU: fit 73 s, ~24 s/fold; 164 folds | ~7 min GPU (~70 min CPU) -> `--time 1:00:00` |
| test chronos2_fine_tuned (per seed) | fit 636 s at 1500 steps (<= ~21 min at 3000) + 164 x 2 s | ~20-30 min -> `--time 2:00:00` |
| importance chronos (per model) | 112 features x 5 iterations = 560 predictions of the full frame | ~20 min GPU (at ~2 s/prediction); ~4 h CPU -> `--time 2:00:00` GPU |

### Tests / results
- `tests/unit/test_chronos_stage.py` (new, 14 tests, both envs): dispatch, store layout, metrics =
  `chronos_evaluate_long`, predictions == `models.chronos.run_backtest`, predictor reuse by fit
  identity, skip/`--force`/`max_folds`, `cv` skipped, C17 guard, tuned params reach the predictor,
  F133 paradigm guard; the study with a fake fit gives the SAME 12 suggestions/best params as the
  legacy in-memory `create_study(direction="minimize", sampler=TPESampler(seed=42))`; a failing fit
  aborts the study and cleans up; importer round trip.
- `tests/golden/test_chronos_equality.py`: no more `PYTHONPATH`; new `slow` test
  `test_the_pipeline_fits_what_the_thesis_fit` (AutoGluon env): `run_stage` FITS a fresh zero-shot
  predictor from the pipeline's own data stage; internal MASE and fold 0 vs the thesis.
- Smoke on the laptop (CPU): pipeline-fit zero-shot: internal MASE 1.1458 (thesis log 1.1459);
  2 folds vs `golden/results/chronos2/predictions_long_chronos2_zero_shot.parquet`: max |d| 2.1e-3,
  mean 4.4e-5, corr 0.99999998 (F134 tolerance).

## 2. Feature importance as a stage (audit C14)

### What changed
| file | change |
|---|---|
| `src/strikecast/evaluation/importance.py` (new) | pure functions: `gbm_importances` (= `get_gbm_importances`, `_regression_GBDT.ipynb` cell 57: per-horizon gain + `permutation_importance(n_repeats=5, random_state=42, n_jobs=-1)` on the darts lagged design matrix of the fitting series), `chronos_importance` (= `_chronos2.py:686`), `classify_feature`/`CATEGORY_RULES`/`CATEGORY_ORDER` (= `results/analyse_results.ipynb` cell 52 = `tab:semantic_categories`), `top_features`, `category_shares`, `category_importance_matrix` (cell 52, the F20 matrix), `category_shares_long`, `window_transform`. |
| `src/strikecast/pipeline/importance_stage.py` (new) | `run_importance` (stage `importance` in `state.json`; identity = hash of params/settings + upstream (+ the Chronos test stage hash); skip when complete; `--force`; interrupt-aware), `collect_importance`, `DEFAULT_JOBS`. GBDT: build the TUNED model (`resolve_params`), fit on each paradigm group's FULL series (`target_full` + post-selection covariates, exactly cell 57), gain + permutation. Chronos: reload `artifacts/predictor/` of the completed test stage, `feature_importance(full frame, model=<its model>, relative_scores=True)`. |
| `src/strikecast/cli/main.py` (additive) | subcommand `importance` + flags `--no-permutation`, `--n-jobs`. |

### Commands
```bash
# thesis set (DEFAULT_JOBS): count lightgbm_poisson + catboost_tweedie x {global, activity}; seed = tuning seed
PYTHONHASHSEED=0 strikecast importance experiment=count --store-root <root> --n-jobs $SLURM_CPUS_PER_TASK
# thesis-mode reproduction: add legacy=count and a legacy store root with the golden params imported
# explicit:  strikecast importance experiment=count model=catboost_tweedie paradigm=activity seed=42
# Chronos (AutoGluon env, after the test stage of the same seed): see section 1
```
Dependencies: GBDT importance needs `featsel` + `tune` (best params) only -- it fits its own model
on the full series and does NOT need the test stage. Chronos importance needs the Chronos test
stage (same seed). CPU job for GBDT; GPU job (AutoGluon env) for Chronos.
Set `PYTHONWARNINGS="ignore:X does not have valid feature names"` in the job: sklearn warns once
per permutation call (legacy behaviour; ~1,400 log lines per model otherwise).

### Output files and schemas (for the figures stream)
```
<root>/count/<model>/<paradigm>/seed=<s>/importance/importance.csv
    Feature, h1_gain..h7_gain, h1_perm..h7_perm, agg_gain, agg_perm, model
    (= results/gbdt/importance_*.csv + the `model` label: global_<m> | activity_<k>_<m>;
     rows sorted by agg_perm desc per label; --no-permutation: *_perm NaN, sorted by agg_gain)
<root>/count/<model>/<paradigm>/seed=<s>/importance/category_shares.csv
    model, metric (gain|perm), category, share, top_n   (top-15 shares; each (model, metric) sums to 1)
<root>/count/<model>/<paradigm>/seed=<s>/importance/timings.json   (fit/gain/design/perm seconds per group)
<root>/count/report/importance/seed=<s>/importance_all.csv    (= results/gbdt/importance_all.csv
    schema; order: global labels, then per tier lightgbm_poisson, catboost_tweedie; it ALSO holds
    global_catboost_tweedie, which the thesis kept in importance_global_catboost_tweedie_2.csv)
<root>/count/report/importance/seed=<s>/category_shares.csv    (all runs, long format)
<root>/chronos2/<model>/global/seed=<s>/importance/importance.csv
    AutoGluon's frame: unnamed index = feature, importance, stdev, n, p99_low, p99_high
    (= results/chronos2/feature_importance_chronos2_{ft,zs}.csv; 112 rows = 27 known + 84 past + Activity_Level)
<root>/chronos2/report/importance/seed=<s>/feature_importance_<model>.csv   (same format)
```
Figure recipes (analysis notebook cells 44-52): F20 = `category_importance_matrix({f"activity_{k}":
{"gain": top_features(sub_k, "agg_gain"), "perm": top_features(sub_k, "agg_perm")}})` with
`sub_k` = rows `activity_<k>_catboost_tweedie` of importance_all.csv; F23 = the same top-15 frames as
bar charts; F21 = top 15 of `feature_importance_chronos2_fine_tuned.csv` (`importance` column).

### Validation (thesis mode, `legacy=count`, golden params, laptop)
- **CatBoost-Tweedie activity (the Fig 20/23 source) reproduces the thesis to machine precision**
  for tiers 2 and 3 (max |d| ~1e-16, every horizon, gain and permutation) and to <= 6.8e-5 of the
  column maximum for tier 1; the F20 matrix gives exactly the percentages `main.tex` 1117-1119
  quotes. Global CatBoost-Tweedie equals `results/gbdt/importance_global_catboost_tweedie_2.csv`
  to 1e-16.
- LightGBM-Poisson (global + 3 tiers): level F (F124, LightGBM trees not portable): max |d| <=
  7.7e-4 (gain) / 1.05e-3 (permutation) of the column total, Spearman >= 0.9957, top-15 overlap
  15/13/14/15 (gain) and 15/15/15/15 (permutation).
- The collected `importance_all.csv` has exactly the thesis file's columns and, apart from the
  extra `global_catboost_tweedie` label, its label order.
- `tests/unit/test_importance.py` (new): port == verbatim cell 57 (exact, LightGBM and CatBoost with
  darts encoders), `n_jobs` never changes a value, category rules == notebook == thesis table,
  expdecay7 -> leaky7 never moves a category, the thesis percentages from
  `results/gbdt/importance_all.csv`, stage outputs/labels/skip, collect order, Chronos branch.
- `tests/golden/test_importance_equality.py` (new): level-F comparison of a thesis-mode store's
  importance runs with `results/gbdt/importance_all.csv`, plus the tight CatBoost/F20 test; skips
  when no run exists (store root `runs/` or `STRIKECAST_LEGACY_STORE`).

### Run-time estimates (laptop M-series, 15 cores, `n_jobs=-1`, heavily contended by the other streams' test runs -- upper bounds)
| run | fit | permutation (rows of X) | total |
|---|---|---|---|
| lightgbm_poisson global | 34 s | 492 s (16,540 rows x 291 features x 5 repeats x 7 horizons) | 527 s |
| lightgbm_poisson activity (3 tiers) | 36/36/44 s | 336/152/181 s (8,270/3,308/4,962) | 785 s |
| catboost_tweedie global | 36 s | 98 s | 135 s |
| catboost_tweedie activity (3 tiers) | 19-26 s each | 25-61 s each | 188 s |
| **whole count thesis set** (`strikecast importance experiment=count`) | | | **~27 min** (+ ~1 min data stage) |
Recommendation: one CPU job, `-c 48`, `--time 2:00:00`, `--n-jobs $SLURM_CPUS_PER_TASK`
(`n_jobs=-1` = `os.cpu_count()` = 64 on these nodes regardless of `-c`, TaskPlugin=task/none; each
joblib worker additionally runs the booster's own threads, so do not oversubscribe further).
`--no-permutation` (gain only) takes ~1-2 min per model. Re-running the command when every run is
complete only skips and re-collects (seconds), so the report/figures job can call it to rebuild
`importance_all.csv` from whatever finished.

## 3. expdecay switch
The category rules match the BASE variable (`disrupted_weapons`, `total_daily_strike_events`, ...),
never the window prefix, so `ewm_expdecay7_*` and `ewm_leaky7_*` map to the same category (unit
test). `window_transform(name)` returns `ewm_expdecay7` / `ewm_leaky7` / `rolling_rsum7` / ... for
tables that want to split them. Chronos has no window features. The thesis's F20 numbers (62 %/48 %
etc.) are partly expdecay7 mass (audit A §2(d)); they will change after the leaky re-run.

## Text vs code (for Jan; methodology frozen, no code change)
- Thesis §"Feature importance" defines permutation importance on HELD-OUT data with a validation
  LOSS; the code permutes the TRAINING design matrix of a model fit on the FULL series (incl. the
  test period) and scores R^2 (`estimator.score`). Chronos-2: AutoGluon scores the last 7 days of
  the full frame (MASE, relative scores).
- The Chronos importance covers 112 inputs (27 known + 84 past incl. the stray `index` + the static
  `Activity_Level`).

## Open issues
- Stream 3: `make_jobs` emits `cv` jobs for chronos2 (should follow `cfg.stages`); Chronos jobs
  must run with `envs/autogluon/.venv/bin/python` and `HF_HOME`; add `importance` jobs (count: CPU
  after tune; chronos2: GPU after the chronos test stage).
- Lint errors (UP017) currently in Stream 3's `scripts/slurm/smoke_test.py` and `submit_all.py`.
- `strikecast evaluate experiment=chronos2` is not supported (it calls the darts `_evaluate`);
  `run` writes the metrics, so nothing needs it.
