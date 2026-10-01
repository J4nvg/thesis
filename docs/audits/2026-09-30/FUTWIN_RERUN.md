# Sensitivity rerun: future-covariate window (count GBDT, seed 42)

Written 2026-09-30. **Status: done 2026-10-01 (jobs 65021-65039 from `~/reruns/thesis`), 18/18 + control
complete. Verdict: REJECTED** (`futwin_rerun_output.txt`):

- R1 control PASS (22960 rows, max |pred diff| = 0).
- R5: the median Day-1 penalty shrinks only 12 % on MAE (+1.24 % to +1.09 %) and 16 % on RMSE
  (+2.32 % to +1.95 %); Day 2 still beats Day 1 in 15/18 (RMSE) and 16/18 (MAE) configurations.
- R4: the wider window is slightly worse overall (median RMSE +0.58 %, MAE +0.09 %), so the
  publication default `(2, 7)` stays.

So the missing pre-target weather is not the cause of the GBDT Day-1 penalty. What the rerun did *not*
change: the Day-1 sub-model still sees 6 post-target weather days (13.8 % of its importance in
`day1_trace.py`); darts' shared window cannot remove them for Day 1 only. The first "next suspect"
below cannot single out Day 1: every sub-model's freshest training label is 7 days before its target.

## Why

On the same 158 target dates, every count GBDT (18/18) and every hurdle model (3/3) forecasts Day 1
1.5–3% worse than Days 2–7; the RNNs, Chronos-2 and ARIMA don't. `day1_trace.py` (same folder)
traces it to the future-covariate window: darts with `multi_models=True` gives every horizon's
sub-model the same window `lags_future_covariates=(2, 7)`, anchored at the origin (origin−2 …
origin+6), and weather is only a future covariate. Relative to its own target, the Day-1 model sees
2 days of weather before the target and 6 after; the Day-7 model sees 8 before and none after.
Evidence: darts `sklearn_model.py` 560–567 (no per-step shift), per-horizon importance (Day 1 puts
13.8% on post-target weather, pre-target weather ≥3 days rises 0% → 11.5% from Day 1 to Day 7), and
only the model classes with this window show the penalty.

**Hypothesis tested here:** with `(8, 7)` every sub-model sees at least 8 pre-target weather days,
so the Day-1 penalty should disappear.

## What runs

| Store (cluster: `~/reruns/thesis/…`) | Jobs | Window |
|---|---|---|
| `runs_sensitivity_futwin/lags_8_7/` | 18: `{lightgbm,xgboost,catboost}_{poisson,tweedie}` × global/activity/local | `(8, 7)` |
| `runs_sensitivity_futwin/control_2_7/` | 1: `catboost_tweedie@global` | `(2, 7)` through the same switch; must equal `runs_publication` exactly |

Everything else is the publication run. The same seed (42), the same tuned `best_params.json` and
the same feature selections are copied from `runs_publication/count/`. The switch is outside the
feature-selection hash, so both selections load from cache: Poisson `d41d56d9…`, Tweedie
`a5c51cd6…`, checked in `tests/unit/test_sensitivity.py`. Only the test stage runs, with no
tuning and no W&B (`tracking=noop`). There are 19 CPU jobs (`-c 16`, 48 GB, partition `GPU`,
2–6 h each), and they run in parallel.

**Caveat:** the parameters were tuned under `(2, 7)` and are not retuned, so this is a sensitivity
check, not a replacement for the publication numbers.

## Code (switch, default off; publication runs unchanged)

- `src/strikecast/config/schema.py`: `SensitivityConfig.future_covariate_lags` and
  `ExperimentConfig.sensitivity`. The default is `None`, and it is left out of the config dump, so
  existing `config.yaml` snapshots are unchanged.
- `src/strikecast/models/spec.py`: `RunContext.future_lags`; `darts_common_kwargs` replaces only
  `lags_future_covariates`.
- `src/strikecast/pipeline/run_stage.py`: the block joins the stage identity only when it is set.
- `tests/unit/test_sensitivity.py` (8 tests). The config, spec, pipeline, CLI and store unit tests
  pass (383 in total).
- `scripts/slurm/sensitivity_futwin.sh`: seeds the stores, submits, and has `--dry-run` and `--status`.
- `docs/audits/2026-09-30/futwin_rerun_analysis.py`: the analysis. It can be self-tested on the
  publication store.

## Submit

On the laptop: commit and push these files to `refactor`. On the cluster (login node):

```sh
cd ~/reruns/thesis && git pull
bash scripts/slurm/sensitivity_futwin.sh --dry-run    # 19 sbatch lines, nothing written
bash scripts/slurm/sensitivity_futwin.sh              # writes runs_sensitivity_futwin/SUBMISSION.md
bash scripts/slurm/sensitivity_futwin.sh --status     # queue, complete count, failed log tails
```

Logs go to `logs/slurm/sensitivity_futwin/sens-futwin-<window>-<model>-<paradigm>-<jobid>.out`.
The last line of each is `STRIKECAST_EXIT <code>`.

## Retrieve (laptop, repo root)

```sh
rsync -a uvt:reruns/thesis/runs_sensitivity_futwin/ runs_sensitivity_futwin/   # uvt = ~/.ssh/config alias
```

This is a fresh store that only the cluster writes to, so a plain rsync mirror is enough;
`sync_runs.py` is not needed. The size is about 20 MB of inputs plus 19 × ~1 MB of predictions.

## Analyse

```sh
uv run python docs/audits/2026-09-30/futwin_rerun_analysis.py \
    | tee docs/audits/2026-09-30/futwin_rerun_output.txt
```

The script reports:
- **R0:** inventory (18/18 complete?).
- **R1:** the control. It must print `PASS`; otherwise stop and find what else differs.
- **R2:** the Day-1 penalty per configuration, `(2,7)` vs `(8,7)`, on the same target dates.
- **R3:** family-level Friedman and Day 2 vs Day 1.
- **R4:** overall RMSE/MAE change.
- **R5:** the verdict.

## Pre-registered decision rule (R5)

Median Day-1 penalty over the 18 configurations, on both MAE and RMSE. Under `(2,7)` it is
+1.24% for MAE and +2.32% for RMSE.

- **Confirmed:** it shrinks by ≥ 2/3 on both metrics. The penalty is the window artefact.
  - Paper: add one methods sentence (the direct GBDTs share one origin-anchored future window, so
    the Day-1 model saw the least pre-target weather; widening it to 8 days removes the Day-1
    step), and report the horizon conclusion as flat for every family.
  - The family-split figure can go; use the pooled figure.
  - Decide whether R4 (overall accuracy under `(8,7)`) is worth a footnote.
- **Partial:** it shrinks by more than 1/3 but less than 2/3. The window explains part of it. Keep
  the family figure, mention the mechanism as the main cause, and look at the per-paradigm pattern
  in R2.
- **Rejected:** it shrinks by ≤ 1/3. The window is not the cause. Next suspects, from
  `day1_trace.py`:
  - Day-1 sub-model training labels end 6 days earlier (the `MultiOutputRegressor` needs all 7
    labels).
  - The selection's past-lag set.
  - A test that puts weather in the past covariates as well.

Horizon text, figures and conclusions: `docs/audits/2026-09-30/horizon_pipeline.py`,
`horizon_figure.py` and the chat of 2026-09-30. Memory: `horizon-analysis-findings`, `futwin-rerun`.
