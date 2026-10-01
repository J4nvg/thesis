# Feature ablation: selector value and feature-group contributions

Sensitivity experiment, pilot on **catboost_tweedie @ global** (leaderboard #2), test stage only.
Everything for it lives in this folder:

| file | what |
|---|---|
| `variants.tsv` | the variant table, single source of truth (read by the submit script and the analysis) |
| `analysis.py` | read-only analysis of the stores, sections R0 to R5, writes `output/` |
| `figures.py` | F1 to F5 from `output/*.csv`, writes `figures/` (SVG + PNG + the CSV behind each) |
| `PROMOTING_FIGURES.md` | how to move a figure into the thesis figure set (`strikecast figures`) later |
| `output/`, `figures/` | created by the two scripts |

## Why

The paper's GBDTs train on the top-100 `(feature, lag)` past-covariate pairs picked by a
matched-objective LightGBM gain selector. Nothing so far measures what that selector buys, or
which feature groups carry the signal. Two questions:

1. **Selector value.** Does the selected top-100 beat (a) every pair in the pool, no selection,
   and (b) 100 random pairs (5 draws)?
2. **Group contributions.** Which feature groups help on their own (core + one group vs core), and
   which ones the selected model needs (leave one group out of the selected 100)? Can the
   selector replace a dropped group with substitutes from the remaining pool?

## Design decisions (Jan, 2026-10-01)

- **Default hyperparameters everywhere** (`depth 5, lr 0.05, 500 it, l2 3, subsample 0.8,
  p 1.5`), the `selected` reference included, so only the feature space differs between runs.
  Every job gets `--allow-default-params`. The one exception is the control `selected_tuned`,
  which uses the publication `best_params.json` and must reproduce `runs_publication`
  bit-for-bit (proves the switch is inert under `mode=selected`).
- **Core** = the target's own 7 lags + the 2 static covariates (region, activity tier). Always
  on. Every other input, future covariates and calendar encoders included, belongs to exactly one
  switchable group.
- **Leave-one-out from the selected 100**, not from `all`.
- **Fixed and re-selected ablations.** Fixed edits the feature set directly (drop a group's pairs,
  or core + all lags of one group). Re-selected restricts the selector's *pool* and lets it pick
  100 again, so substitutes can fill the hole.
- **3 seeds (42, 1, 2: the first `eval_seeds`)** for the key contrasts; ablations on seed 42 only.
- **Global paradigm only** in the pilot.

### Groups

Past pool: 595 components x lags `[-1, -7, -14]` = **1785 pairs**; every component classifies
into one of 7 categories (`evaluation/importance.py::CATEGORY_RULES`), none in "Other".
Future inputs: 25 weather/geomagnetic components, 2 holidays and darts' cyclic calendar encoders
(5 x sin/cos), all over lags -2..6.

| slug | category (importance plot) | inputs | columns |
|---|---|---|---|
| `strikes` | Autoregressive strikes | 28 past comps / 84 pairs (other strike series; the target's own lags are core) | 84 |
| `spatial` | Spatial / static | 35 past / 105 pairs (the statics are core) | 105 |
| `conflict` | Conflict & damage | 112 past / 336 pairs | 336 |
| `comms` | Comms / diplo / aid | 252 past / 756 pairs | 756 |
| `macro` | Macroeconomic | 77 past / 231 pairs | 231 |
| `missile` | Missile / launch | 21 past / 63 pairs | 63 |
| `cyber` | Cyber | 70 past / 210 pairs | 210 |
| `weather` | Weather / geomag. | 25 future comps | 225 |
| `calendar` | Calendar | 2 holidays + 10 cyclic encoder columns (the encoders are "Other" in the importance plot; folded in here) | 108 |

The selected 100 (tweedie selection `a5c51cd6`) by group: conflict 32, spatial 22, comms 20,
strikes 17, macro 9; missile and cyber 0. Leave-one-out therefore covers the 5 past groups plus
weather and calendar.

### Modes (`sensitivity.feature_space.mode`)

| mode | feature space |
|---|---|
| `selected` | the cached publication selection (control and reference) |
| `all` | core + every group: all 1785 pairs + all future inputs + encoders (= `groups` with all 9) |
| `random` | k = 100 pairs from the 1785; future inputs and encoders as in publication. `draw` 1-3 uniform (`np.random.default_rng(draw)` over the canonically ordered pool); `draw` 4-5 **stratified**: the selected set's per-group counts (32/22/20/17/9), uniform within group. Separates "which groups" from "which features within a group" |
| `groups` | core + the listed groups, each past group with all 3 lags, no selector (`groups=[]` is core) |
| `selected_minus` | the selected 100 minus the listed groups' pairs (future groups: drop the weather comps / holidays + encoders). Leaves holes |
| `reselect_only` / `reselect_drop` | restrict the pool to the listed groups / to the pool minus them, then run the family's selector (`count_tweedie`, top-100). Future groups removed from the pool are also removed from the model's inputs. A pool with <= 100 pairs keeps everything (logged) |

## Runs (round 1)

All `catboost_tweedie@global`, test stage, CPU jobs (`-c 16`, 48 GB, partition `GPU`), 6 h for
`all` and `only_comms` (~2.1k / 756 columns), 3 h otherwise. **37 variants, 53 jobs.** One store
per variant, `runs_feature_ablation/<variant>/`, seeded with `count/shared` (19 MB; the
`reselect_*` variants add their own `feature_selection.<hash>.json` there). Only `selected_tuned`
also gets `count/tuning/catboost_tweedie`.

| variant | seeds | jobs | params | contrast |
|---|---|---|---|---|
| `selected_tuned` | 42 | 1 | tuned | reproduces `runs_publication` |
| `selected` | 42,1,2 | 3 | defaults | reference |
| `all` | 42,1,2 | 3 | defaults | vs selected |
| `core` | 42,1,2 | 3 | defaults | reference for `only_*` |
| `random_d1`..`random_d3` (uniform) | 42,1,2 | 9 | defaults | vs selected |
| `random_d4`, `random_d5` (stratified) | 42,1,2 | 6 | defaults | vs selected |
| `only_<g>` x 9 (fixed, all lags) | 42 | 9 | defaults | vs core |
| `drop_<g>` x 7 (selected minus g) | 42 | 7 | defaults | vs selected |
| `reselect_drop_<g>` x 7 (pool minus g -> top-100) | 42 | 7 | defaults | vs selected |
| `reselect_only_<g>` x 5 (comms, conflict, macro, cyber, spatial) | 42 | 5 | defaults | vs core and vs `only_<g>` |

`drop_*`/`reselect_drop_*` cover strikes, spatial, conflict, comms, macro, weather, calendar.

## Caveats

- **No tuning**: absolute numbers sit below the publication model (tuned). The contrasts are
  like-for-like (same defaults on both sides), the absolute values are not the paper's.
- **Seed 42 only for the ablations** (R3, R4): their CIs are over forecast origins, not over
  seeds. R2 shows how large the seed-to-seed spread is for comparison.
- **The selector is gain based** (LightGBM, matched objective): a group it does not pick may still
  carry signal that a different ranking would find. `reselect_*` and the stratified draws probe
  this, they do not settle it.
- **Weather is a perfect-foresight future covariate** (observed values over lags -2..6): its
  contribution is an upper bound on what a weather forecast would deliver.

## Submit (cluster login node, checkout `~/reruns/thesis`)

```bash
cd ~/reruns/thesis && git pull
bash scripts/slurm/feature_ablation.sh --dry-run   # 53 sbatch lines, nothing written
bash scripts/slurm/feature_ablation.sh             # seed the stores + submit
bash scripts/slurm/feature_ablation.sh --status    # queue, completed stages, failed logs
```

## Retrieve (Jan, own terminal: the SSH key has a passphrase)

```bash
rsync -a uvt:reruns/thesis/runs_feature_ablation/ /Users/jan/projects/bsc_thesis_code/thesis/runs_feature_ablation/
```

## Analyse and plot (laptop, reads only)

```bash
uv run python docs/feature_ablation/analysis.py      # also writes output/analysis_output.txt
uv run python docs/feature_ablation/figures.py
```

`analysis.py` works on a partial store (R0 lists what is missing; R5 marks the reading
provisional). Options: `--root`, `--pub` (default `runs_publication_20260929`), `--runs
catboost_tweedie@global ...`, `--variants-file`, `--out`, `--n-boot` (default 1000). Outputs:

- `results.csv`: one row per contrast x loss x seed scope: `run, section, variant, reference,
  seed` (`42`, `1`, `2` or `pooled`), `loss` (`squared`/`absolute`), `metric` (`rmse`/`mae`),
  `value_variant, value_reference, delta` (variant - reference, negative = variant better),
  `pct`, `ci_lo, ci_hi` (95% moving-block bootstrap over origins, block 7), `dm_p` (HLN-DM),
  `p_holm` (Holm within run x section x loss x seed scope), `cliffs_delta` (paired, positive =
  variant better), `n_pairs`, `note`; pooled rows add `n_seeds, delta_seed_mean,
  seeds_sig_better, seeds_sig_worse`.
- `metrics.csv`: `variant, seed, mae, rmse, skill` + pair counts from `feature_space.json`
  (`n_past_pairs, n_future, calendar_encoders, kept_groups, pairs_<group>`).
- `per_horizon.csv`: `variant, seed, horizon, rmse, mae`.

Paired tests use `evaluation/comparison.py::compare_pair` on clipped predictions (the
leaderboard clips at 0, `metrics.py:249`), squared loss (RMSE, primary) and absolute loss (MAE,
secondary). **Seeds:** per seed, variant seed s is paired with reference seed s. The pooled row
averages each `(region, origin_date, horizon)` instance's loss over the seeds both sides
completed and runs the same bootstrap/DM/Cliff's delta on that seed-averaged differential
(`compare_pair` cannot stack seeds itself: its join keys must be unique per frame). The RMSE CI
is the same origin resamples as compare_pair's MSE CI, mapped to RMSE.
**SkillScore** = 1 - RMSE / RMSE(diff `naive_weekly`, global, test) as in the master
leaderboard (`reporting/compute.py`); R1 checks the recomputation against
`master_leaderboard.csv` (catboost_tweedie@global 0.16323).

Self-test without cluster results (every variant's store is replaced by the publication store,
only seed 42 exists, every delta is 0, R1 must PASS; output goes to the temp dir):

```bash
uv run python docs/feature_ablation/analysis.py --selftest --n-boot 200
uv run python docs/feature_ablation/figures.py --selftest
```

Figures: **F1** selector value (selected / all / 5 draws, seed points + seed mean with the pooled
CI of the difference to selected, RMSE and MAE); **F2** single groups, delta vs core, fixed and
re-selected side by side, with pair counts; **F3** leave-one-out, delta vs selected, fixed vs
re-selected; **F4** per-horizon RMSE/MAE of selected / all / core (seed mean, seed-range band);
**F5** cumulative path (round 2, only once `path_*` stores exist).

## Pre-registered reading

Primary metric RMSE, secondary MAE, SkillScore reported. Holm correction within each section.
95% CIs are block-bootstrap CIs of the difference.

- **R1 gate**: `selected_tuned` reproduces the publication predictions exactly (max |delta
  y_pred| = 0). If R1 fails, stop: nothing else is interpreted.
- **Selector adds value**: `selected` beats `all` on RMSE and MAE, 95% block-bootstrap CI
  excluding 0, across the 3 seeds (pooled CI excludes 0 and every seed points the same way).
- **Selector beats chance**: `selected` beats all 5 draws on RMSE (descriptive) and beats the
  median draw by DM at p < 0.05. Stratified ~ uniform -> groups don't matter, only the
  within-group picks; stratified much better than uniform -> group composition is the value
  (R5 prints the share of the uniform-to-selected gap the stratified draws close; descriptive).
- **A group carries signal**: `only_g` beats `core` *and* `drop_g` loses to `selected`, CIs
  excluding 0. **It is replaceable** if `reselect_drop_g` ~ `selected` (CI includes 0) while
  `drop_g` < `selected`.

## Round 2: cumulative path

After round 1, R5 orders the groups by single-group RMSE gain over core (`only_g` vs `core`)
and prints the exact commands for a cumulative path k = 2..8 (k = 1 is the best `only_g`,
k = 9 is `all`), e.g.

```bash
bash scripts/slurm/feature_ablation.sh --combo k2 <g1>,<g2>
...
bash scripts/slurm/feature_ablation.sh --combo k8 <g1>,...,<g8>
```

Each submits one `path_k<k>` variant (`mode=groups`, seed 42, default params). The analysis
picks the `path_*` stores up automatically and `figures.py` draws F5.

## Extending to the top-20

The other count GBDTs (LightGBM / XGBoost / CatBoost x Poisson / Tweedie) and the activity
paradigm work as is through `RUNS=` (and `SEEDS=` to override the seed column):

```bash
RUNS="catboost_tweedie@global lightgbm_tweedie@global catboost_tweedie@activity" \
  bash scripts/slurm/feature_ablation.sh --dry-run
uv run python docs/feature_ablation/analysis.py --runs catboost_tweedie@global lightgbm_tweedie@global
uv run python docs/feature_ablation/figures.py --run lightgbm_tweedie@global --out docs/feature_ablation/figures/lightgbm_tweedie
```

Out of scope for the pilot: RNNs take components, not `(feature, lag)` pairs (they would need a
component-level variant of every mode); Chronos-2 and ARIMA have no feature selection to ablate;
`linear` lives in the diff store with its own selection.

## Code map

- Schema: `SensitivityConfig.feature_space` / `FeatureSpaceConfig` in
  `src/strikecast/config/schema.py` (off by default; publication runs bit-identical).
- Pure feature-space logic (groups, pools, random/stratified draws, restricted pools,
  `feature_space.json` payload): `src/strikecast/data/feature_space.py`.
- Data stage: `src/strikecast/pipeline/data_stage.py::_prepare_one` (re-selection on a restricted
  pool before `build_or_load_features`; the other modes edit the feature set after it;
  `_features_hash` changes only for `reselect_*`).
- Model kwargs: `src/strikecast/models/spec.py::darts_common_kwargs` (no past lags, no encoders,
  no future lags per `RunContext`).
- Run stage: `src/strikecast/pipeline/run_stage.py::_covariates` (`None` for an empty keep list)
  and `_write_feature_space` (`<run>/feature_space.json`).
- Tests: `tests/unit/test_feature_space.py`.
- Submit: `scripts/slurm/feature_ablation.sh` (reads `variants.tsv`).
- Hydra override syntax (one job by hand):

  ```bash
  +sensitivity.feature_space.mode=groups '+sensitivity.feature_space.groups=[weather,cyber]'
  +sensitivity.feature_space.mode=random +sensitivity.feature_space.draw=4 +sensitivity.feature_space.stratified=true
  ```
