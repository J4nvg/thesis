# Refactor progress log

Persistent hand-off notes so any new session can continue `docs/REFACTOR_PLAN.md`
without re-deriving state. **Every agent appends one entry when it finishes**
(newest at the bottom, never rewrite older entries). Keep entries short:
what was delivered, which files, test status, what is left, open questions.

## State at 2026-09-18 (start of the multi-agent Phase 3 push)

- Branch `refactor`. Phases 0-2 committed (`2e287a2`).
- Phase 3 is half-delivered and **uncommitted**: `models/{spec,gbm,rnn,classical,classifiers}.py`,
  `config/{schema,loader}.py`, `seeds.py`, `store/run_store.py`, `tuning/optuna_runner.py`,
  `evaluation/{metrics,aggregate,calibration,scoring}.py`, plus unit tests and
  `tests/legacy_ref/*_builders.py`.
- Missing for P3/P4: `configs/` YAML tree (loader expects it), registry wiring in
  `models/__init__.py`, `models/registry.py` or equivalent, pipeline orchestration
  (config -> data -> bundle -> spec -> forecaster -> engine -> store -> metrics),
  `cli/main.py`, `tracking/`, `evaluation/{leaderboard,seeds,comparison,stats}.py`,
  level E/F golden tests for count/diff.
- Unit test status at start: `tests/unit/test_rnn_specs.py` fails to collect
  (`test_spec_flags` uses no argument 'experiment'); `test_calibration.py::test_oof_matches_legacy_with_a_single_class_group`
  and `test_optuna_runner.py::test_study_reproduces_the_legacy_create_study_and_optimize_options` fail.
  Everything else: 349 passed.
- Run tests with `.venv/bin/python -m pytest tests/unit -q -p no:cacheprovider` (no `timeout` binary on macOS).

## Work streams launched 2026-09-18 (Opus agents, disjoint files)

| Stream | Owns | Status |
|---|---|---|
| A. P3 fixes + registry wiring | `models/__init__.py`, `models/registry.py`, the 3 failing tests, `calibration.py`, `optuna_runner.py` | done |
| B. `configs/` YAML tree + loader tests | `configs/**`, `config/loader.py`, `tests/unit/test_config_loader.py` | done |
| C. Tracking (P4) | `tracking/**`, `tests/unit/test_tracking.py`, `scripts/wandb_sync.sh` | done |
| D. Publication evaluation (P6/P7, additive) | `evaluation/{leaderboard,seeds,comparison,stats}.py` + tests | done |
| E. Pipeline orchestration + CLI | `pipeline/**`, `cli/**`, `tests/unit/test_pipeline*.py` | done |

## Entries

### Stream C — 2026-09-18

**Delivered: Phase 4 tracking (§5.5).** `strikecast.tracking` is in place and the
package imports without `wandb` installed (every `import wandb` is lazy, inside a
method). Comet ML stays dropped.

- `src/strikecast/tracking/protocol.py` — the `Tracker` Protocol
  (`start(run_key, config, tags, *, stage="run") -> str | None`, `log_fold(step, metrics)`,
  `log_trial(number, value, params)`, `log_tables(tables, *, stage=None)`,
  `log_artifact(path, kind, *, name=None, metadata=None)`, `finish(status="success")`),
  `ARTIFACT_KINDS`, and `run_name(run_key)` which returns exactly
  `str(RunKey.relative())` — a W&B run name and the run directory it mirrors are
  the same string. `RunKey` is a type hint only; the tracker duck-types the four
  coordinates, so this package does not import the store at runtime.
- `src/strikecast/tracking/noop.py` — `NoopTracker`, stateless (`__dict__ == {}`),
  never touches `wandb`.
- `src/strikecast/tracking/wandb_tracker.py` — `WandbTracker`. One W&B run per
  `(experiment, model_variant, paradigm, seed)` **and stage**: `name` = the run
  directory, `group` = experiment (`TrackingConfig.group` overrides), `job_type` =
  stage, `tags` = `[family, kind]` from the model spec, `config` = the resolved
  config plus a `run` block with the four coordinates and the stage, `id` =
  `wandb_run_id(name, stage)` (a sha1 digest, so re-running a stage resumes its
  mirror instead of duplicating it, `resume="allow"`). Online by default;
  `mode="offline"` is passed to `wandb.init` *and* exported as `WANDB_MODE=offline`
  so anything wandb spawns agrees. After a successful `start`, a failing wandb call
  is logged at WARNING and disables the tracker rather than aborting the run (the
  store already holds what was being mirrored); `strict=True` re-raises, which is
  what the P4 smoke test should use. A missing `wandb` package raises an
  `ImportError` naming `tracking.backend='noop'`.
- `src/strikecast/tracking/hook.py` — `TrackerFoldHook`, a `FoldHook`-compatible
  adapter: it scores the **cumulative** bundle the engine hands it (the same one
  `PruningHook` scores, so W&B and the pruner cannot tell different stories) and
  calls `log_fold(fold.index, metrics)`, i.e. `step=fold`. `prefix=` separates the
  cv and test curves, `every=` mirrors once per retrain window on long stages,
  non-numeric columns of an `evaluate_long` global row are dropped
  (`numeric_metrics`), and a failing `metrics_fn` is logged and swallowed —
  the inverse of `PruningHook`, whose exceptions are load bearing.
- `src/strikecast/tracking/factory.py` — `make_tracker(cfg, *, dir=None, strict=False)`.
  `cfg=None` or `backend="noop"` gives a `NoopTracker`; `TrackingConfig` is
  duck-typed, so `config/` stays Stream B's.
- `scripts/wandb_sync.sh` — syncs `offline-run-*` / `run-*` directories from cluster
  scratch (`-d`, `-p`, `-e`, `-n` dry run, `-f` re-sync, `-c` delete after sync).
  Skips directories already carrying `.synced`, never touches `runs/`, reports and
  skips a failed directory rather than aborting the batch, and exits with the
  number of failures. Portable bash (no `mapfile`), `bash -n` clean.

**Tests.** `tests/unit/test_tracking.py`: **46 passed**, no network and no real
`wandb` (a fake module injected with `monkeypatch.setitem(sys.modules, ...)`).
Covers run naming/grouping/job-type/tags/config, the 16-way uniqueness of run ids
over model × paradigm × seed × stage, `step=fold` logging, trial logging, tables
and artifacts (file and directory), online/offline and the `WANDB_MODE` export,
offline `dir` creation, `finish` idempotence and exit codes, "a broken mirror
cannot abort the run", `NoopTracker` being a genuine no-op, the fold hook, and a
subprocess check that `import strikecast.tracking` succeeds with `wandb` blocked
from `sys.meta_path`.

Full unit suite after this stream: **609 passed, 2 failed** — the two Stream A
failures that were already failing at the start of the push
(`test_calibration.py::test_oof_matches_legacy_with_a_single_class_group`,
`test_optuna_runner.py::test_study_reproduces_the_legacy_create_study_and_optimize_options`).
The `test_rnn_specs.py` collection error is gone. `ruff check` clean on the new
files. Nothing outside `tracking/**`, `tests/unit/test_tracking.py` and
`scripts/wandb_sync.sh` was touched; nothing committed.

**What remains for P4.**

- Wiring: nobody calls `make_tracker` yet. The pipeline/CLI stream (E) has to
  `tracker.start(run_key, resolved_cfg, tags=(spec.family, spec.kind), stage=...)`,
  add `TrackerFoldHook` next to `PersistHook` in the engine's hook list,
  `log_tables(metric_views, stage=...)` after `write_metrics`, `log_artifact` on
  the predictions/metrics directories, and `finish()` in a `try/finally`.
- The tuning stream should call `log_trial` from an Optuna callback on a
  `stage="tune"` run; `tuning/optuna_runner.py::tune` has no callback hook today,
  so either it grows a `callbacks=` argument or the caller passes one through
  `cfg`. Not added here — `tuning/` is not this stream's file.
- `RunStore.write_env` should record the W&B run id that `start` returns, so a run
  directory can be traced to its mirror from disk alone. One line in the pipeline.
- Not smoke-tested against real W&B on a two-fold run (P4's acceptance criterion);
  that needs the pipeline from Stream E.

**Open questions / gaps.**

1. `TrackingConfig` (`config/schema.py`) has `backend`, `mode`, `project`, `entity`,
   `group` and nothing else. Three fields would help and were NOT added, because
   `config/` belongs to another stream:
   - `dir` — where offline runs are written (cluster scratch, `$WANDB_DIR`).
     `make_tracker(cfg, dir=...)` takes it as a keyword for now.
   - `strict` — fail loudly on tracking errors (useful for the P4 smoke test).
   - `tags` — extra tags beyond `[family, kind]`.
2. The stage is not part of `RunKey`, but it is part of the W&B identity
   (`job_type` and the run id). `start(..., stage=...)` is therefore a
   keyword-only argument with the default `"run"`. If the pipeline prefers one
   W&B run per run key spanning both stages, that is a one-line change to
   `wandb_run_id`, but then the fold-step axes of cv and test collide, which is
   why it is split this way.
3. `log_trial` logs at `step=trial.number`, which is only safe because tuning is a
   separate W&B run from cv/test. If a future caller mixes trials and folds on one
   run, W&B's monotonic step requirement will bite.

### Stream B — 2026-09-18

**Delivered: the `configs/` Hydra tree and its loader tests.** `load_experiment`
now composes and validates all four in-scope experiments.

Files added:

- `configs/experiment/{count,diff,hurdle,damage}.yaml` — the four families.
  Every value carries a comment citing the legacy line it comes from
  (`_regression_GBDT.py`, `_regression_LSTM.py`, `_diff_regression.py`,
  `final_hurdle.ipynb`, `damage_classifier.ipynb`, `src/prevalent_functions.py`).
- `configs/backtest/weekly_retrain.yaml` — `stages.{cv,test}`: horizon 7,
  predict_stride 1, retrain_stride 7, the F81 adapter map and the train-only
  naive scales. Shared by all four families.
- `configs/paradigm/{global,activity,local}.yaml`, `configs/tracking/{wandb_online,
  wandb_offline,noop}.yaml`, `configs/seeds/default.yaml`. All group files carry
  `# @package _global_`; so do the experiment files, which the loader docstring
  did not say but Hydra requires for a primary config inside a group directory.
- `tests/unit/test_config_loader.py` — 82 tests: composition, the three name
  spellings, overrides, four flavours of loud failure, and the frozen §2.3
  settings per family.

Model lineups are taken from the registry, not from Appendix B, and the test
asserts set-equality with `registered_names(<experiment>)`: count 21
(6 GBDT + 15 RNN, **no** baselines — F84/F86: `REGRESSORS_TO_RUN` is dead code
in the count scripts), diff 13 (3 GBDT + 6 RNN + linear + arima + 2 naives),
hurdle 3, damage 2. Appendix B lists `linear`/`naive_*` under `count`; that is
the plan being looser than the evidence, and `models/classical.py` documents why.

**Schema extensions (additive, `config/schema.py` only):**

1. `available_threads()` + `threads: int | Literal["auto"] | None` +
   `ExperimentConfig.resolved_threads`. The count/hurdle/damage scripts call
   `get_available_threads()` at import time, so `"auto"` is the legacy value;
   writing an integer would pin one machine's core count into `resolved_hash()`.
   `_diff_regression.py:90` hard-codes 4, so `diff.yaml` writes `threads: 4`.
   `to_run_context` now passes `resolved_threads`.
2. `ExperimentConfig.feature_selections: dict[str, FeatureSelectionStageConfig]`
   + `feature_selection_for(head)`. The hurdle family ran **two** selections
   (F27, `final_hurdle.ipynb` cells 9-11) and the schema had room for one.
   Unlisted heads fall back to `feature_selection`.
3. `DataConfig.keep_index` / `DataConfig.validate_target` — derived properties
   mirroring the three `build_panel_legacy_*` wrappers, so that
   `panel_variant: damage` fully specifies `target=""` + `validate_target=False`
   (Q7) without the pipeline hard-coding it.

**Tests:** `tests/unit/test_config_loader.py` 82 passed. Full unit suite
`738 passed` (the three tests listed as failing at the start of the push now
pass, so Stream A appears to have landed its fixes).

**Remains / for other streams:**

- No `configs/model/<name>.yaml` group: model hyper-parameters live in the specs
  (`ModelSpec.defaults` / `search_space`) and tuned values in
  `best_params.json`, so a model group would duplicate them. If the CLI wants
  `model=<name>` selection it can override `models=[<name>]`, which is tested.
- No `configs/hydra/launcher/` group yet (plan §5.6, submitit/SLURM) — that
  belongs with Stream E's CLI.
- The pipeline still has to decide how `data.binarize` produces the hurdle's
  *two* panels (the count head wants the unbinarised one, the classifier the
  binarised one); the config expresses the binarisation, not the pairing.

**Open questions:**

- `damage.yaml` pins `paradigms: [global]` because `damage_classifier.ipynb` has
  no per-activity/per-region wrapper. Hurdle and the two regression families
  default to `global` and are swept with `paradigm=activity,local`, per
  Appendix B. Confirm that is the intended CLI shape.
- `ExperimentConfig.device_for` keys off the first `_`-separated token, so
  `spe_event_classifier`/`spe_damage_classifier` resolve under the family key
  `spe` and `catboost_tweedie_count_head` under `catboost`. The YAMLs are
  written to match, and a test asserts `device_for(m) == get_spec(m, exp).device`
  for every model of every family, but the token rule is fragile if a future
  model name starts with an existing family token.

### Stream D — 2026-09-18

**Delivered: the publication evaluation modules (§7.1, §7.2; phases P6/P7).**
All four are strictly additive (§1): they read `global.json` metric files and
saved `PredictionSet` frames out of the run store and never call a pipeline.
Nothing outside `evaluation/{leaderboard,seeds,comparison,stats}.py`, the four
new tests and the appended block in `evaluation/__init__.py` was touched;
nothing committed.

- `src/strikecast/evaluation/leaderboard.py` — walks the §5.3 layout
  (`runs/<experiment>/<model>/<paradigm>/seed=<s>/{cv,test}/metrics/global.json`),
  skipping the `shared/`, `tuning/` and `report/` siblings, and assembles the
  legacy `lb_rows` (`{"split", "paradigm", "model", **global_metrics}`) which it
  hands to the existing `aggregate.leaderboard` — so the sort (`["split",
  "MASE_mean"]`, quicksort, NaN last) and the column order stay the ported ones
  and are not re-implemented. `MetricRow`, `discover_metric_files`,
  `collect_metric_rows` (filters on seed/model/paradigm), `build_leaderboard`
  (one seed, default 42), `metric_frame` (the long frame Stream D's seed
  aggregation consumes) and `write_leaderboard`. `golden_metric_rows` is a
  compatibility reader for the thesis' flat
  `golden/results/<family>/global_<split>_<paradigm>_<model>.json` naming,
  splitting paradigm from model against the known paradigm set (both can be the
  literal `global`).
  **Verified against golden:** rebuilding `gbdt` (36 rows) and `diff` (62 rows)
  gives identical columns, identical row order and identical values — max
  absolute deviation 4.4e-16 and 3.6e-12 respectively, i.e. CSV float
  round-trip only.
- `src/strikecast/evaluation/seeds.py` — cross-seed aggregation per
  `(experiment, model, paradigm, stage, metric)`: `n`, `mean`, `std` (ddof=1),
  `min`, `max`, `t_lo`/`t_hi` (Student-t, df = n−1) and `boot_lo`/`boot_hi`
  (percentile bootstrap of the mean over seeds, 1000 replicates, deterministic
  given `boot_seed`), plus `deterministic`, `expected_seeds` and the `seeds`
  list. `broadcast_deterministic` replicates a `stochastic=False` model's single
  run across `eval_seeds` (§5.4) — additively, so a deterministic model that was
  run under several seeds keeps its real rows; a model absent from the
  stochastic map is never broadcast. A deterministic cell is reported as a point
  value with NaN spread and NaN intervals, not a zero-width interval (§7.1).
  `leaderboard_ci(...)` is the one-call path and writes
  `runs/<experiment>/report/leaderboard_ci.csv`.
- `src/strikecast/evaluation/comparison.py` — §7.2 items 1–3 on the long
  prediction frame. `paired_differences` inner-joins on
  `(region, origin_date, horizon)` — deliberately **not** on `fold`, which is an
  index into each run's own schedule (F33/F51) — with a one-to-one merge
  validation so a composite forecaster's multi-channel frame has to be narrowed
  with `channel=` rather than silently fanning out. `d = L(A) − L(B)` for the
  squared and absolute losses, no clipping (the metrics port clips because the
  legacy code does; a clip here would change who wins). `block_bootstrap_ci` is
  a non-circular moving-block bootstrap over the sorted unique `origin_date`s,
  default block length 7, which draws origins whole (all regions and horizons
  come with a drawn origin) and trims to exactly n origins. `diebold_mariano`
  runs on the per-origin mean differential with a HAC long-run variance
  (Bartlett/Newey-West by default, `weights="truncated"` for the original DM),
  default lag `horizon − 1` and `max(horizon) − 1` when pooled, HLN small-sample
  correction on by default against `t_{n−1}` (`hln=False` gives the plain
  statistic against the normal). `cliffs_delta` is the paired form
  `(#(d<0) − #(d>0))/n`, oriented so positive means A is better, with Romano
  magnitude labels. `compare_pair` returns a `PairSummary`, `pairwise_table`
  covers every (optionally ordered) pair × horizon scope with a per-pair
  bootstrap stream, `write_pairwise` writes `report/pairwise_<metric>.csv`.
- `src/strikecast/evaluation/stats.py` — §7.2 item 4, ported from
  `golden/results/analyse_results.ipynb` cells 27–30 (the `.py` next to it is a
  7-line stub holding only the `q` one-liner) with the arithmetic unchanged:
  `build_block_matrices` (the `(region, horizon)` block × model RMSE/MAE
  matrices, now taking `{label: frame}` instead of deriving parquet paths),
  `normality_report` (the two-way ANOVA residual Shapiro-Wilk of cell 28, now
  returning a dataclass instead of printing), `nemenyi_critical_distance`
  (`q = studentized_range.ppf(1−α, k, ∞)/√2`, `CD = q·√(k(k+1)/6N)`),
  `friedman_nemenyi` (returns a `FriedmanResult` that still supports the
  notebook's `r["avg_rank"]` mapping access), `tied_with_best`,
  `average_rank_table` (cell 29's `sig_top5_table`) and
  `critical_difference_diagram` / `save_critical_difference_diagram` (cell 30,
  including the y=0.5→1 crossbar nudge and the "CD =" label reposition). The one
  deliberate difference is cosmetic: cell 30's `mpl.rcParams.update` is applied
  through `rc_context` around the draw instead of globally at import, so
  importing the module does not put serif fonts on every other figure in the
  process. A test asserts the rcParams do not leak.
- `src/strikecast/evaluation/__init__.py` — 43 names appended via a second
  import block and `__all__ += [...]`; the existing calibration block and its
  `__all__` literal were left untouched.

**Tests.** `tests/unit/test_{leaderboard,seed_aggregation,comparison,stats}.py`:
**103 passed** (23 / 21 / 38 / 21). Full unit suite after this stream:
**797 passed, 0 failed**. `ruff check` clean on all new files.
Golden coverage: four `@pytest.mark.golden` tests rebuild both thesis
leaderboards, replay one seed-42 row, check the paradigm/model split when both
say `global`, and round-trip golden JSON through a synthetic §5.3 tree back into
`build_leaderboard(seed=42)`. The synthetic tests cover discovery (including the
reserved-directory and non-`seed=` decoys), the t- and bootstrap intervals, the
deterministic broadcast, block contiguity, DM lag/HLN/weights/HAC growth under
autocorrelation, Cliff's delta edge cases, and the Friedman/Nemenyi port against
the notebook expressions re-derived inline.

Smoke-checked on the real artefacts: five `golden/results/gbdt/predictions_long_test_*`
frames give a 140 × 5 block matrix (20 regions × 7 horizons), Friedman
χ²=133.8, p=6e-28, CD=0.516, and a `compare_pair` over 164 origins with lag 6.

**What remains.**

- Nobody calls these yet. `strikecast report` (P7) has to: build the
  leaderboard, call `leaderboard_ci` with the `stochastic` map from the model
  registry (`{name: spec.stochastic}` — Stream A owns `models/registry.py`, so
  the map is a caller argument here, not an import), run `pairwise_table` per
  metric and per breakdown, and write `report/cd_<metric>.svg` and
  `report/summary.md`. The `summary.md` renderer is **not** written.
- `family_comparison.csv` (§7.2 item 5, the seed-paired family difference) is
  not implemented: it needs a family→variants mapping that lives in the configs
  (Stream B) and at least one real multi-seed run (P6).
- The per-activity-tier breakdown (§7.2 item 6) is supported by `scope=` but has
  no helper: it needs `activity_by_region`, which comes from the `SeriesBundle`,
  not from the prediction frame.
- The legacy `golden/results/*/predictions_long_*.parquet` files carry only the
  six `LEGACY_COLUMNS` — no `origin_date`, no `channel`. Comparing against them
  requires backfilling `origin_date` as the per-fold minimum `date`. New runs
  through `PredictionSet` carry both columns, so this is only a golden-data
  concern; no equivalence test depends on it today.
- Not run against a real multi-seed store, because none exists yet (P6).

**Open questions.**

1. **Block length.** `DEFAULT_BLOCK_LENGTH = 7` is the plan's number (§7.1,
   "a moving-block variant with block length 7 covers the overlap in F4") and
   matches the 7-day horizon. It is not tuned. With 164 test origins that is
   ~23 effective blocks, which is thin; the usual n^(1/3) rule would say ~5.
   Worth a sensitivity row in the paper (block ∈ {1, 5, 7, 14}) rather than a
   silent default.
2. **Non-circular blocks under-sample the tail.** The last `L−1` origins can
   only appear inside a block that starts earlier. A circular variant would fix
   the coverage but wraps forecast time onto itself. Left non-circular; say so
   in the methods section.
3. **DM small-sample correction.** HLN is on by default, against `t_{n−1}`.
   The HLN factor uses the *nominal* horizon h, but the per-origin series was
   already averaged over regions (which is what makes the HAC correction
   one-dimensional). Whether h or an effective h is right here is a judgement
   call; `hln=False` and `lag=` are exposed so both can be reported.
4. **Cross-sectional dependence between regions is handled only by keeping an
   origin whole.** There is no correction for the fact that 20 regions at the
   same origin are correlated. That is the standard panel-DM caveat and should
   be stated rather than fixed.
5. **The seed bootstrap over 5 seeds is weak** — only n^n distinct resamples and
   it cannot leave `[min, max]`. It is emitted because §7.1 asks for it; the
   t-interval is the one to quote. Consider dropping the columns or raising
   `eval_seeds` to 10.
6. **A name collision to watch.** `evaluation.aggregate` exports a *function*
   called `leaderboard` and this stream adds a *module* called `leaderboard`.
   `evaluation/__init__.py` does not currently re-export the function; if
   whoever owns `aggregate.py` adds `from .aggregate import leaderboard` to the
   package `__init__`, the attribute `strikecast.evaluation.leaderboard` becomes
   the function and shadows the submodule attribute (the submodule stays
   importable by path). Renaming the function to `leaderboard_frame` would be
   the clean fix, but `aggregate.py` is not this stream's file.
7. **`model` vs `model_variant`.** The plan says `model_variant`; `RunKey` and
   the legacy `leaderboard.csv` both say `model`. These modules use `model`
   throughout for consistency with the store and the golden CSVs.
### Stream A — 2026-09-18

**Delivered**

1. *The three broken unit tests, fixed on the wrong side of each.*
   - `tests/unit/test_rnn_specs.py` did not collect: two `parametrize(("name", "experiment"))`
     decorators over functions taking only `name`. The TEST was wrong throughout — it
     also unpacked `DIFF_VARIANTS` as `(name, legacy)` pairs (it is a flat tuple of
     names), read `GOLDEN_TUNING_DIR` as `name -> (subdir, variant)` (it is
     `experiment -> subdir`, and the variant directory IS the spec name in both
     families), called `get_spec(name)` for `lstm_w7/w14/w28`, which are registered
     for BOTH `count` and `diff` and therefore need an experiment, compared the
     default params dicts with `==` although both sides hold fresh `EarlyStopping`
     objects (the file's own `_normalise` exists for that), and asserted
     `registered_names("count") == COUNT_VARIANTS` although the registry is
     process-global and `gbm` registers into the same family. `models/rnn.py` is
     unchanged. Nothing was relaxed: every spec is still checked for its own
     experiment, `payload["source_dir"]` is now checked too, and the lineup is
     compared in order over the RNN slice of the registry.
   - `test_calibration.py::test_oof_matches_legacy_with_a_single_class_group`: the CODE
     is right and the TEST expected the wrong thing. `_oof_one_group` falls back to an
     in-sample sigmoid below two minority rows (Q5/F115); with ZERO minority rows that
     is a `LogisticRegression` on one class, which sklearn refuses — so the legacy
     notebook cell raises `ValueError` there too, identically. The test now pins both
     sides raising the same error, and a new
     `test_oof_matches_legacy_with_a_single_minority_row` covers the fallback that does
     run (one minority row: in-sample sigmoid, arrays equal, other groups still split).
     `calibration.py`'s Q5 paragraph now records the crash; no behaviour changed.
   - `test_optuna_runner.py::test_study_reproduces_the_legacy_create_study_and_optimize_options`:
     the TEST read `sampler._rng.rng_seed`, which optuna 4.8 does not have (`_rng` is a
     `LazyRandomState`). The seeding is now pinned properly: the sampler's numpy RNG
     state is snapshotted inside the `create_study` spy (before `optimize` advances it)
     and compared with a fresh `TPESampler(seed=42)`, plus a negative check against
     `seed=43`. `optuna_runner.py` is unchanged.

2. *Registry wiring.* New `src/strikecast/models/registry.py` imports `gbm`, `rnn`,
   `classical` and `classifiers` once and exposes `get_spec`, `registered_names`,
   `all_specs`, `registered_experiments`, `specs_by_experiment` and `EXPECTED_COUNTS`.
   `spec.py` gained `all_specs(experiment=None)` and `registered_experiments()`
   (additive). `models/__init__.py` now resolves every export lazily through
   `__getattr__`, so `import strikecast.models` costs ~15 ms and imports no torch /
   darts / lightgbm / xgboost / catboost; asking the registry anything imports the lot.
   New `tests/unit/test_registry.py` (55 tests) asserts the plan §2.1 lineup literally:
   count = 6 GBDT + 15 RNN, diff = 3 GBDT + 6 RNN + linear + arima + 2 naives,
   hurdle = 3, damage = 2 (39 specs, 36 distinct names), that `lstm_w7/w14/w28` are the
   only shared names and resolve to different specs per experiment, that the tunable
   specs are exactly the GBDT + RNN lineup and the baselines are `stochastic=False`,
   that `is_neural == needs_raw_past_covs` is exactly the RNNs, and that the package
   import stays library-free (checked in a subprocess).

**Files touched**: `src/strikecast/models/registry.py` (new), `src/strikecast/models/__init__.py`,
`src/strikecast/models/spec.py` (two additive helpers + `__all__`),
`src/strikecast/evaluation/calibration.py` (docstring only),
`tests/unit/test_registry.py` (new), `tests/unit/test_rnn_specs.py`,
`tests/unit/test_calibration.py`, `tests/unit/test_optuna_runner.py`.
Nothing under `tests/legacy_ref/` was modified, and no file owned by streams B–E.

**Tests** (`.venv/bin/python -m pytest <path> -q -p no:cacheprovider`):
- `tests/unit`: **612 passed** (was 349 passed + 1 collection error + 2 failures).
- `tests/equivalence`: **33 passed, 1 xfailed** (F80).
- `tests/golden`: **221 passed, 2 xfailed** (F16), **4 failed** — see below.

**Pre-existing failure, diagnosed, NOT fixed (outside this stream)**:
`tests/golden/test_metrics_equality.py::test_hurdle_all[{cv,test}-classifier_{raw,cal}]`.
Only the `per_region` view of the `classification` metric set differs, and only in ROW
ORDER: aligned on `region`, every metric is exactly equal. Both the notebook cell
(`final_hurdle.ipynb` cell 14) and `evaluation/aggregate.py` do the identical
`long_df.groupby("region")` then `.sort_values("F1", ascending=False)`, and 10 of the 20
regions tie at `F1 == 0`, so the surviving order is whatever the (unstable) sort gave —
which differs between the pandas that wrote `golden/` and pandas 3.0.2 here. It is an
environment artifact, not a methodology or port difference. Suggested resolution for
whoever owns `evaluation/aggregate.py`: compare that view order-insensitively in the
golden test and record a flag, rather than adding a tie-break the thesis did not have.

**What remains for the model layer**: nothing in P3's registry itself. Chronos-2 and the
pre-hurdle lineups are P5 (plan §8, §10 open point 1). The registry is not yet consumed
by anything — the pipeline (stream E) and the experiment YAML (stream B) still have to
map `experiment + model name -> spec -> forecaster`.

**Open questions**
1. Flag numbers have collided between parallel P3 modules: `models/rnn.py::RNN_FLAGS`
   and `evaluation/calibration.py` both use F100–F117 for different observations
   (e.g. two different "F115"). One owner should renumber before §4 of the plan absorbs
   them.
2. `registered_names(experiment)` returns names, so a name shared by two families
   appears once; `all_specs(experiment)` returns specs. Callers that key runs by model
   name must therefore always carry the experiment too. Worth stating in the run-store
   key (§5.3) before the pipeline lands.
3. Should `EXPECTED_COUNTS` live in `registry.py` at all? It is documentation that the
   unit test checks; if the experiment YAML becomes the source of truth for lineups
   (stream B), the test should read it from there instead.

### Stream E — 2026-09-18

**Delivered: the orchestration layer and the CLI (P3's spine).** Config in, run
store out; nothing under `pipeline/` or `cli/` imports Hydra except
`cli/main.py`, and the library still takes plain pydantic configs.

- `src/strikecast/pipeline/context.py` — re-exports `RunContext` from
  `models/spec.py` (not redefined), `make_run_context`, `resolve_store_root`
  (absolute, §9), a lazy `get_spec` that prefers Stream A's
  `models/registry.py` and falls back to importing the spec modules, and
  `make_tracker`/`make_fold_hook`/`track` over Stream C's `strikecast.tracking`
  with a local `NoopTracker` stub of the same protocol for when it is absent.
  `track()` swallows tracker errors: the store is the source of truth (§5.5).
- `src/strikecast/pipeline/data_stage.py` — `prepare_data(cfg, store)` plus the
  three composable halves `build_or_load_panel` / `build_or_load_bundle` /
  `build_or_load_features`, all through `runs/<exp>/shared/` with content
  hashes over the config slice plus the upstream hashes. The panel caches a
  `panel.<hash>.meta.json` sidecar for `global_weather_columns`, whose order is
  process-dependent (panel Q2) and carries into the future covariates. Feature
  selection resolves `cache_path` (a converted legacy set, F16) first, then the
  store's JSON, then a fresh `select_top_k` on `(target_train, past_covs,
  future_covs)`; `subset_components` then re-runs the encode+split exactly as
  the notebooks do.
- `src/strikecast/pipeline/tune_stage.py` — `tune_model` / `tune_experiment`.
  Reassembles the legacy objective from parts: `spec.search_space` = the
  suggester, `engine.iter_folds` = `run_expanding_cv_iter`, `PruningHook` =
  report/prune/return, `tuning.optuna_runner.tune` = the study. Global
  paradigm, CV stage, `for_tuning` preset (F56), tuning seed only (F14), the
  objective metric explicit (F39), `n_trials` as a TOTAL so a study resumes
  (F11). `best_params.json` existing means "do not re-tune" (Appendix C).
- `src/strikecast/pipeline/run_stage.py` — `run_stage(cfg, model, paradigm,
  seed, stage, data)` and `run_experiment(...)`. Stage identity =
  `stage_hash(resolved stage config, (panel, series, features) hashes, seed)`;
  a complete stage under the same identity is skipped. Seeds everything, snaps
  `config.yaml`/`env.json` (with the tracker run id when there is one), builds
  the forecaster from spec + resolved params + `RunContext`, runs the engine
  under `grouping.partition`, streams parts through `PersistHook`, reloads,
  evaluates and writes the metric views, marks complete. `run_experiment` keeps
  CV single-seed and runs `stochastic=False` models once (§5.4).
- `src/strikecast/pipeline/report_stage.py` — thin: `evaluation.leaderboard`
  and `evaluation.seeds` (Stream D) do the work; a minimal seed-42 leaderboard
  is the fallback when they are absent.
- `src/strikecast/cli/main.py` + `[project.scripts] strikecast` in
  `pyproject.toml` — `run`, `tune`, `evaluate`, `report`, `verify`. Grammar per
  Appendix C: `experiment=<name>` names the primary config, `model=`,
  `paradigm=`, `stage=` and `seed=` are comma-list SELECTORS consumed by the
  CLI, everything else is a Hydra override passed through verbatim
  (`tracking=noop`, `seeds.eval_seeds=[1,2]`). Composition always adds
  `hydra.job.chdir=false`, `--store-root` is applied as `++store.root=` and the
  root is resolved absolute. A single `paradigm=` selector is ALSO composed as
  its Hydra group so the snapshot names the paradigm the run used, falling back
  to selector-only when the experiment's defaults list has no `paradigm` entry.
  `-m` is accepted and expands the product in process; the submitit/SLURM
  launcher is P6. `verify` is the promised stub: it lists the nine golden
  levels of §6, which of them run under pytest today and which have material on
  disk.

**Legacy behaviour preserved (and where it is written down).**

- `cv` runs on `target_cv_view` from `cv_start_frac`, `test` on `target_full`
  from `train_val_end` (F17, F19) — the two lists `run_expanding_cv` and
  `run_final_test` are called with.
- **F80**: the diff family's CV stage gets an explicit `model_targets`, built
  by differencing the bundle's un-encoded full target list and re-running the
  legacy encode+split, which is literally `_diff_regression.py`:234-239.
  Verified on the real panel: count CV 79 folds / 12 retrains from 2024-05-11,
  diff CV the same 79/12 from 2024-05-12 (the F33 one-day shift), both tests
  164/24, diff test on 846 steps (F51) — i.e. the golden level-C schedule.
- **F81**: the adapter preset comes from `StageConfig.adapter_for(paradigm)`,
  so Global CV is `for_cv` and Activity/Local CV are `for_test`.
- **F6/F67**: the MASE/RMSSE denominators come from the stage's `naive_scales`
  block.
- The legacy Global `evaluate_long` call passes NO activity map and the
  Activity/Local ones do (`_regression_GBDT.py`:988/1207/1308 vs :1478ff), so a
  Global stage writes four metric views and the other two write six. Pinned by
  a test.
- Naive baselines bypass the engine entirely, as `naive_collect_long` does.

**Tests.** `tests/unit/test_pipeline_stages.py` (20) and
`tests/unit/test_cli.py` (12), both on a synthetic 4-region / 121-step panel
with the real `linear` and naive specs: the shared bundle cache, the legacy
feature-set schema, the F80 override, cv+test with the part layout
(19 folds → 3 parts) and the four/six metric views, resume (second call skips,
`--force` does not), a changed stage config getting a new identity, the local
paradigm partition agreeing with global for a region-independent model, the
seed loop (CV single-seed, deterministic models broadcast, stochastic models
swept with distinct stage hashes), an Optuna study of 2 trials with
`best_params.json` not being re-tuned, and the CLI's argument grammar,
`verify`, `run`, `evaluate`, `report`, `tune` and skip-on-rerun. Every CLI test
runs in process with `tracking=noop` against a throw-away `configs/` tree in
`tmp_path`.

**Test counts.** `tests/unit`: **829 passed, 0 failed** (the three tests Stream
A owned are fixed; nothing is xfailed in `tests/unit`).
`tests/unit tests/equivalence`: **862 passed, 1 xfailed** (the strict F80
xfail). `ruff check` clean on every new file. Nothing committed; the only
shared file touched is `pyproject.toml`, additively (`[project.scripts]`).

**What remains.**

1. **Level E/F golden runs on real data** — the next step. Everything is wired:
   `strikecast run experiment=count model=<v> paradigm=global stage=cv,test
   seed=42` with the converted best params under
   `golden/converted/tuning/<v>/best_params.json` should reproduce
   `results/count/predictions_long_<v>_tuned.parquet` at 1e-6. Nobody has run
   it; the data stage and the schedule are verified on the real panel (above),
   the predictions are not.
2. **Best params are not imported yet.** `resolve_params` reads
   `runs/<exp>/tuning/<model>/best_params.json`; the converted goldens live
   under `golden/converted/tuning/`. Copying them in (or pointing the store at
   them) is one step and is what a level-E run needs first.
3. **Partial-stage resume recomputes.** A *complete* stage is skipped; a
   partial one restarts at fold 0 and overwrites its parts, because the engine
   cannot start mid-schedule (`RunStore.resume_point` says so). Honouring the
   resume point needs a `start_fold` on `BacktestConfig`/the schedule. Cheap
   with `retrain_stride=7`, expensive for a long local-paradigm test stage.
4. **P5 families.** `run_stage` raises `NotImplementedError` for
   `kind="composite"` and `data_stage` raises for a multi-panel (damage) build.
   The metric-set mapping for `hurdle`/`damage` is already wired.
5. **No SLURM templates.** `-m` expands in process; `scripts/slurm/` and the
   submitit launcher are P6.
6. **W&B not smoke-tested.** The pipeline calls `start` / `log_fold` (through
   Stream C's `TrackerFoldHook`) / `log_tables` / `log_artifact` / `finish` and
   records the returned run id in `env.json`, but only against `NoopTracker`.
   `tune_stage` logs trials AFTER the study, from `trials_dataframe()`, because
   `tuning/optuna_runner.tune` has no callback argument (Stream C flagged this
   too).

**Flagged, not fixed.**

- `evaluation/leaderboard.build_leaderboard` raises `KeyError: 'split'` on a
  run store with no matching metrics, because `aggregate.leaderboard` sorts an
  empty frame by `["split", "MASE_mean"]`. `report_stage` works around it by
  calling `collect_metric_rows` first and returning early; the fix belongs in
  `evaluation/`.
- `strikecast.cli.main` had to stay the MODULE, so `cli/__init__.py` imports
  nothing and `pipeline/__init__.py` does not re-export the `run_stage`
  FUNCTION. Importing `strikecast.pipeline.run_stage` gives the module.
- A naive baseline under `paradigm=activity`/`local` produces exactly the
  Global numbers (it is region-independent). The run store keys it by paradigm
  anyway, so a leaderboard can quote it per view; the legacy scripts only ever
  computed it once.

**Open questions.**

1. **Does the CV stage belong under a seed at all?** It is single-seed by
   §5.4, and it currently lands in `.../seed=42/cv/`, which reads as if seed 42
   were one of several. The alternative is `.../cv/` outside the seed level,
   which would change the store layout §5.3 draws. Left as drawn.
2. **`data.activity_by_region` on a cache hit.** The activity map is not in the
   panel parquet, so `prepare_data` re-reads `load_inputs` for it even when the
   panel is cached (fast: three small files). Putting it in the sidecar would
   remove the read but duplicate state.
3. **Which metric set for a future family** is keyed off the experiment NAME
   (`METRIC_SET_BY_EXPERIMENT`). A config field would be cleaner, but that is
   `config/schema.py`, another stream's file.

### Coordinator — 2026-09-18 (end of the five-stream push)

All five streams delivered; see their entries above. Combined suite
(`tests/unit tests/equivalence tests/golden`): **1083 passed, 4 failed, 3 xfailed**.
The 4 failures are pre-existing, `tests/golden/test_metrics_equality.py::test_hurdle_all[*-classifier_*]`:
only the `per_region` classification view differs, and only in row order among
regions tied at F1 == 0 (unstable sort, pandas version artefact; values equal when
aligned on region). Not fixed: needs an order-insensitive comparison plus a flag,
owner `evaluation/aggregate.py`. Nothing committed yet (40 changed/untracked paths).

**Next steps, in order**
1. Commit the Phase 3/4 work (`git add -A src configs tests docs pyproject.toml scripts`).
2. Copy converted best params from `golden/converted/tuning/` into
   `runs/<exp>/tuning/<model>/best_params.json` (Stream E item 2), then run the
   count and diff experiments at seed 42 and compare with `golden/results/` (levels E/F).
3. Resolve the hurdle per-region golden failure (order-insensitive compare + flag).
4. Wire the W&B tracker into `pipeline/run_stage.py` and do the P4 two-fold smoke test.
5. P5 families (composite hurdle, damage multi-panel, Chronos-2) still raise NotImplementedError in the pipeline.
6. Decisions for Jan: count has no baselines (registry follows legacy, not Appendix B);
   damage pinned to global paradigm; `summary.md` renderer for `strikecast report`;
   block length / DM correction choices in `evaluation/comparison.py`; F100-F117
   flag numbers collide between `models/rnn.py` and `evaluation/calibration.py`.

## Decisions by Jan, 2026-09-18 (after round 1)

- Round-1 work committed as `0f468e2`.
- Count experiment has **no baselines**: the registry follows the legacy scripts; Appendix B of the plan is to be corrected, not the code.
- Hurdle `per_region` tie-order golden failures: compare order-insensitively in the test and record a methodology flag. No tie-break added to the pipeline.
- Round 2 priorities: everything (E/F golden verification, P5 hurdle+damage, P5 Chronos-2, W&B wiring + SLURM), but full experiments take days, so golden verification runs on a cheap deterministic subset only.

## Work streams launched 2026-09-18, round 2 (Opus agents, disjoint files)

| Stream | Owns | Status |
|---|---|---|
| F. Level E/F golden verification on a cheap subset + hurdle tie fix + Appendix B fix | `scripts/import_golden_params.py`, `tests/golden/test_pipeline_equality.py`, `tests/golden/test_metrics_equality.py`, `docs/REFACTOR_PLAN.md` (§4 flag, Appendix B), `pipeline/run_stage.py` only for bugs it finds | in progress |
| G. P5 hurdle + damage through the pipeline | `pipeline/composite_stage.py` (new), `pipeline/data_stage.py` (two-panel build), `models/hurdle.py`, `tests/unit/test_composite_stage.py`, `tests/golden/test_hurdle_pipeline_equality.py` | in progress |
| H. P5 Chronos-2 adapter | `models/chronos.py`, `data/autogluon.py`, `configs/experiment/chronos2.yaml`, `envs/autogluon/**`, `tests/unit/test_chronos*.py` | in progress |
| I. W&B wiring + SLURM templates | `pipeline/context.py`, `pipeline/tune_stage.py` (tracker calls), `tuning/optuna_runner.py` (callbacks arg), `store/run_store.py` (wandb id in env), `scripts/slurm/**`, `configs/hydra/**`, `tests/unit/test_tracking_wiring.py` | in progress |
