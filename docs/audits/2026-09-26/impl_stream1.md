# Implementation Stream 1 — WP1 correctness fixes (hand-off)

Status: IN PROGRESS (updated after each item).

## Keys and commands other streams need (FINAL)

- **Config switch** `series.window.expdecay: legacy_alpha | leaky` (in `WindowTransformConfig`).
  - `leaky` (schema default and every publication experiment YAML): darts
    `{"function":"sum","mode":"ewm","halflife":7.0,"function_name":"leaky7"}` = the literal
    `s_t = x_t + 2^(-1/7) s_{t-1}`, `s_{-1}=0` (verified bit-exact on an impulse and random
    series). Columns are named `ewm_leaky7_<base>`.
  - `legacy_alpha`: the thesis' buggy `expdecay7` (`mean/ewm/alpha=2^(-1/7)`), bit-identical to legacy.
  - `transforms` is derived from `expdecay`; a contradicting hand-written list is rejected.
    Build legacy configs as `WindowTransformConfig(expdecay="legacy_alpha")` (not `model_copy`).
  - Resolved config hash differs between modes (tested).
- **Legacy (thesis-faithful) mode = Hydra group `legacy`**: `configs/legacy/{count,diff,hurdle,damage,chronos2}.yaml`,
  selected with **`legacy=<experiment>`** (NO `+`; every experiment's defaults list ends with
  `- /legacy: null`). count/diff: `expdecay: legacy_alpha` + the golden cached set
  (`golden/converted/feature_sets/{countreg,diffreg}.json`) + the thesis selector settings.
  hurdle/damage: `expdecay: legacy_alpha` only (no thesis cache exists, A11) -> they still need `featsel`.
  chronos2: no-op (no window features).
- **Feature selection CLI** (run ONCE per experiment before tune/cv/test; SLURM job, CPU only):
  ```
  PYTHONHASHSEED=0 strikecast featsel experiment=<count|diff|hurdle|damage> [legacy=<exp>] [tracking=noop] --store-root <root> [--force]
  ```
  Computes every head (hurdle: regressor + classifier; damage: one per key) and writes
  `<root>/<exp>/shared/feature_selection.<hash>.json` (with a `provenance` block). Hard-fails
  unless `PYTHONHASHSEED=0`. Idempotent (loads when present; `--force` recomputes). Must get the
  SAME overrides and `--store-root` as the later jobs (the hash covers the config).
  Later `tune`/`run`/`evaluate` jobs raise `FeatureSelectionMissing` ("run `strikecast featsel ...`
  first") when it is missing (publication YAMLs set `feature_selection.require_cached: true`).
  count/diff in `legacy=` mode need no featsel (the golden cache satisfies it).
  Publication selection settings: `device: cpu`, `num_threads: 8`, `deterministic: true`
  (LightGBM `deterministic=True`, `force_col_wise=True`), `cache: true`, `cache_path: null`.

## Items

### 1. expdecay7 switch (A1, A13, D1) — DONE
- `src/strikecast/config/schema.py`: `ExpdecayMode`, `DEFAULT_EXPDECAY="leaky"`,
  `EXPDECAY_FUNCTION_NAMES`, `expdecay_transform()`, `default_window_transforms(expdecay)`,
  `WindowTransformConfig.expdecay` + two validators.
- Publication YAMLs (`configs/experiment/{count,diff,hurdle,damage}.yaml`) set
  `series.window.expdecay: leaky`; all five experiment YAMLs got `- /legacy: null` appended to `defaults`.
- Golden tests pinned to `legacy_alpha` explicitly: `test_series_bundle_equality`, `test_naive_equality`,
  `test_metrics_equality`, `test_schedule_equality`, `test_hurdle_pipeline_equality`,
  `test_pipeline_equality` (via `legacy=<exp>`).
- Tests: `tests/unit/test_series_split.py` (both modes, impulse + random exactness, contradiction
  validator, hash differs), `tests/unit/test_config_loader.py` (publication = leaky, `legacy=` = expdecay7).

### 2. Feature-selection cache guard + deterministic featsel (A12, A13, D2) — DONE (see keys above)
- `pipeline/data_stage.py`: `check_selection_provenance` (golden JSON without provenance only under
  `legacy_alpha` and for its own selector/target; store/cache_path JSON with provenance must match
  window, selector identity, lag skeleton, target), `FeatureCacheMismatch`, `FeatureSelectionMissing`,
  `require_pythonhashseed_zero`, `select_features` (= `featsel`), `compute_features=` threaded through
  `prepare_data`/`prepare_composite_data`.
- `data/feature_selection.py`: `FeatureSelection.to_json(path, provenance=...)`, atomic write.
- `config/schema.py` `FeatureSelectionStageConfig`: `deterministic`, `require_cached` (+ validator).
- `cli/main.py`: `featsel` subcommand (additive).
- Tests: `tests/unit/test_feature_selection_cache.py` (19 tests incl. hurdle/damage selectors,
  per-head featsel, repeatability, hash-seed hard-fail, A13 refusals, A14).

### 3. Panel cache keyed on data content (A14) — DONE
- `_panel_hash` now also hashes the sha256 of the four input files (`input_digests`). All panel/series/
  feature hashes therefore changed once (expected).

### 4. XGBoost one model per horizon (B3, B4, D5) — DONE
- `models/gbm.py`: every XGBoost variant (count `xgboost_poisson`/`xgboost_tweedie`, diff `xgboost`)
  is built as `PerHorizonXGBModel` (darts `XGBModel` subclass whose `_supports_native_multioutput`
  is False), so darts always wraps it in its per-horizon `MultiOutputRegressor` (7 boosters).
  Constructor kwargs unchanged; created lazily (module `__getattr__`), picklable, `untrained_model()` keeps the class.
- Verified: count Poisson/Tweedie now FIT (they raised before), diff XGBoost trains 7 boosters.
- Tests (`tests/unit/test_gbm_specs.py`): `test_count_xgboost_fits_one_booster_per_horizon`
  (xgboost_poisson, xgboost_tweedie, diff xgboost on a synthetic panel), cell-56 build test;
  `fingerprint` treats the subclass as `XGBModel`.
- NOTE for the diff XGBoost golden/verification: it now trains 7 boosters instead of 1 (the thesis'
  direct strategy); compare to golden at level F.

### 5. RNN tuning objective = legacy (B7) — DONE
- `pipeline/tune_stage.py`: `trial_protocol(spec)` -> `("for_cv", False)` for neural specs
  (exp log-link applied, all folds scored once, NO `PruningHook`/`trial.report`; pruning only via the
  PL callback on `train_loss` already in the RNN search space), `("for_tuning", True)` for GBDTs (unchanged).
- Tests (`tests/unit/test_pipeline_stages.py`): neural trials use `for_cv` and report no intermediate
  values; GBDT trials keep per-fold reports; every registered RNN/GBDT spec maps to the right protocol.

### 6. Hurdle/damage calibration (B8, C18, D6) — DONE
- `pipeline/composite_stage.py`: hurdle primary component (plain `global.json`, the leaderboard row) is
  now `hurdle_cal` (`HURDLE_PRIMARY_COMPONENT`); `hurdle_raw@*` still written. Damage primary unchanged
  (`<first key>_raw`, not in the thesis).
- A calibrated non-CV stage at ANY seed reads `artifacts/calibrators.json` from
  `calibration_key(cfg, key)` = same experiment/model/paradigm, `seed=seeds.tuning_seed`.
  Missing file or missing channel -> `CalibratorsMissing`, raised BEFORE the stage starts (no
  half-written run, no silent raw fallback). The calibrators' sha256 is part of the test-stage identity
  (refitting them re-runs the test stages).
- ORDER CONSTRAINT for Stream 3: hurdle/damage `test` jobs (every seed) need
  `afterok:` the tuning-seed `cv` job of the same paradigm.
- Tests (`tests/unit/test_composite_stage.py`): primary = hurdle_cal; missing calibrators raise
  (hurdle, damage); seed-1 test uses seed-42 calibrators; identity moves when calibrators change;
  empty calibrator file raises.

### 7. Tuned params from `_regression_GBDT.ipynb` cell 56 (B1) + golden tightening — DONE
- `scripts/import_golden_params.py` (additive): `parse_cell56` (`ast.literal_eval` on the list inside
  `dict_items(...)`, type/finiteness/duplicate checks, never `eval`), anchored on
  `lightgbm_poisson == golden/converted/.../lightgbm_poisson/best_params.json` (exact), optional
  cross-check of `golden/from_cluster/checkpoints_tune/lightgbm_tweedie_best.pkl` (`(best_params, Study)`;
  currently ABSENT -> reported, not an error; mismatch -> exit 1). Writes the five variants
  (lightgbm_tweedie, xgboost_poisson, xgboost_tweedie, catboost_poisson, catboost_tweedie) to
  `<store-root>/count/tuning/<variant>/best_params.json` in the `BEST_PARAMS_KEYS` layout
  (`best_value`/trial counts `null`, `source_file` = the notebook cell, a `provenance` block marking
  them as legacy-feature params). Idempotent; never clobbers a differing file without `--force`;
  `--no-cell56` skips. golden/ is NOT written (frozen).
- RUN: `python scripts/import_golden_params.py` -> 5 imported into `runs/count/tuning/`; re-run: 30 unchanged.
- Golden: `test_count_catboost_tweedie_reproduces_golden_on_the_first_retrain_window`
  (`tests/golden/test_pipeline_equality.py`, legacy mode, cell-56 params, test folds 0-6 = one
  retrain window, asserted <= 1e-9: PASSES, 24.5 s).
- LightGBM ceilings tightened (B6), re-measured on the stored runs: count lightgbm_poisson cv max 0.50 /
  mean 0.011, test max 0.95 / mean 0.017; diff lightgbm cv max 2.90 / mean 0.128, test 3.69 / 0.204.
  New ceilings: count max 1.0 (cv) / 2.0 (test) + mean 0.03 / 0.05; diff max 5.0 (unchanged, observed 3.69)
  + mean 0.35 / 0.5 (new `Case.pred_mean_tol`).
- Tests: `tests/unit/test_import_golden_params.py` (real cell 56, literal-only parse incl. an
  `__import__` payload, idempotence/conflict, pickle cross-check match/mismatch).

### Also done (small, in my files)
- `tests/golden/test_feature_selection_equality.py`: the two `xfail(strict=False)` "equality" tests are
  replaced by real assertions (`..._agrees_with_golden_up_to_near_ties`): overlap floors measured under
  PYTHONHASHSEED=0 (diffreg past J 0.729 / future 0.778, countreg 0.946 / 0.800; floors 0.65/0.6 and
  0.85/0.6), size gap <= 5, and every golden-only component ranks < 250 in our gain table
  (observed max 168 = near-ties at the top-100 cut).
- `tune_stage`: a fresh study's `best_params.json` gets a `provenance` block (`expdecay`, window,
  panel/series/features hashes), so publication-tuned params (leaky7) and imported thesis params
  (expdecay7, `provenance.features` says so) are distinguishable.
- Real-data check: `featsel experiment=count` (publication, leaky) run twice in separate processes ->
  identical top-100 (30 past, 8 future; 8 `leaky7` columns kept; 50 s each on the laptop, 8 threads).

## For the other streams

**Stream 3 (SLURM / verification):**
1. DAG: `featsel` (one CPU job per experiment; hurdle/damage compute all heads in that one job) must
   precede every `tune`/`cv`/`test` of that experiment (they fail loudly otherwise). Command above.
   count/diff in `legacy=` mode need no featsel.
2. Hurdle/damage `test` at any seed needs `afterok` on the tuning-seed `cv` of the same paradigm
   (calibrators; `CalibratorsMissing` otherwise).
3. Legacy verification jobs: add `legacy=<experiment>` to every command (Hydra group, no `+`).
   Tuned params for the legacy count GBDTs are in `runs/count/tuning/` after
   `python scripts/import_golden_params.py` (5 new from cell 56).
4. **Store collision (not fixed, not mine):** run paths have no feature tag
   (`<root>/<exp>/<model>/<paradigm>/seed=N/<stage>` and `<root>/<exp>/tuning/<model>/`). Legacy and
   publication runs MUST use different `--store-root`s, otherwise publication tuning overwrites the
   imported thesis params (or vice versa) and the skip check may treat one as the other's complete stage.
   Suggest: legacy/golden verification = `runs/` (the golden tests read it), publication = e.g. `runs_pub/`.
   The `provenance.expdecay` in `best_params.json` can back a guard in `run_stage` (C17 code).
5. C17 (yours): cv/test must hard-fail when a tunable model has no best_params. D4: publication re-tunes
   every model on the leaky7 features.
6. Diff XGBoost now trains 7 per-horizon boosters (B4); count XGBoost can finally be fitted (B3).
   Expect diff xgboost vs golden at level F.
7. Thread settings: featsel pins `num_threads: 8` in the YAML (part of the selection identity);
   do not export `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` expecting it to change that.

**Stream 2 (Chronos / FI):** `configs/legacy/chronos2.yaml` is a no-op so `legacy=chronos2` composes;
chronos2.yaml got `- /legacy: null` appended to its defaults (nothing else touched). The schema default
window mode is now `leaky` (Chronos reads no window features, so no effect). Feature importance on
publication runs will see `ewm_leaky7_*` column names.

**Everyone:** every panel/series/feature-selection hash changed once (new schema fields + input-file
content in the panel hash); cached `shared/` entries are rebuilt on first use. Existing complete runs in
`runs/` are still readable by the golden tests.

## Open issues / not done
- The legacy hurdle/damage selection cannot reproduce the thesis (no cached set; A11) — `legacy=hurdle`
  selects deterministically on expdecay7 columns instead.
- `strikecast featsel` for hurdle/damage was not run on the real data here (only count; unit tests cover
  the per-head path with a patched panel builder).
- The cluster pickle `golden/from_cluster/checkpoints_tune/lightgbm_tweedie_best.pkl` is absent; re-run
  `python scripts/import_golden_params.py` after copying it to get the cross-check.
- No test proves `PerHorizonXGBModel` reproduces the thesis' Colab XGBoost numbers (unrecorded env);
  the verification job should compare count XGBoost fold 0 vs golden (level F).

(final test results below)
