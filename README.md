# Machine and Deep Learning for Conflict Event Forecasting

**Multi-Horizon Spatiotemporal Prediction of Drone Strikes on Ukrainian Regions**

Code for Jan van Gestel's Bachelor's thesis in Cognitive Science & Artificial
Intelligence at Tilburg University, refactored into the reproducible
`strikecast` package and extended with the publication analyses of
[`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md) §7.

The refactor is **behaviour-preserving**: the methodology is exactly the
thesis's. Anything methodologically questionable is *flagged* in §4 of the plan
(F1-F130), never silently changed. Where a flag is worth acting on, it gets a
config switch whose default reproduces the thesis.

---

## Install

The project is `uv`-managed with a frozen lock.

```sh
uv sync --frozen --extra dev     # dev brings pytest + ruff
uv run strikecast --help
```

Python 3.13. The examples below use `.venv/bin/python -m strikecast.cli.main`;
`uv run strikecast` is the same entry point.

### Chronos-2 needs a second environment

AutoGluon 1.5.0 caps `pandas<2.4` and cannot coexist with the core's pandas
3.0.2 pin (plan §5.7), so that one family lives in its own uv project:

```sh
uv sync --project envs/autogluon
uv run --project envs/autogluon python -m strikecast.cli.main \
    run experiment=chronos2 stage=test seed=42
```

Everything else in `strikecast` imports without AutoGluon: only
`strikecast.data.autogluon` and `strikecast.models.chronos` need it, and both
keep their imports inside functions. See `envs/autogluon/README.md`.

---

## The CLI

Five subcommands. Everything after the subcommand is `key=value`; `model`,
`paradigm`, `stage` and `seed` are **selectors** that expand to the full
product, and every other key is a Hydra override.

```sh
# 1. run a backtest stage
PYTHONHASHSEED=0 strikecast run experiment=diff \
    model=naive_last,linear,arima paradigm=global stage=cv,test seed=42 \
    tracking=noop --store-root runs

# 2. tune hyper-parameters (Optuna on SQLite, resumable)
strikecast tune experiment=count model=lightgbm_poisson --n-trials 50

# 3. recompute metrics from stored predictions, without refitting
strikecast evaluate experiment=diff model=arima stage=test seed=42

# 4. build the publication report (plan §7)
strikecast report experiment=diff --store-root runs

# 5. list the golden levels and what material exists for them
strikecast verify
```

`PYTHONHASHSEED=0` matters: the legacy covariate column order depends on it
(flag F16).

### What `report` writes

Into `runs/<experiment>/report/`:

| File | Plan | Contents |
|---|---|---|
| `leaderboard.csv` | §7.1 | one row per `(split, paradigm, model)` at the leaderboard seed |
| `leaderboard_ci.csv` | §7.1 | seed mean, SD, Student-t 95% interval, min/max; deterministic models marked, never given a zero-width interval |
| `pairwise_<metric>.csv` | §7.2 1-3 | per pair and per breakdown (pooled, per horizon, per activity tier): mean loss difference, % improvement, moving-block bootstrap CI over forecast origins, Diebold-Mariano p (HAC/Newey-West, HLN-corrected), Cliff's delta |
| `family_comparison.csv` | §7.2 5 | seed-paired family-vs-family differences with a t-interval |
| `cd_<metric>.svg` | §7.2 4 | Friedman/Nemenyi critical-difference diagram |
| `summary.md` | §7.2 | the tables the paper needs, rendered |

`--n-boot` sets the bootstrap replicates (default 1000); `--no-comparisons`
writes the leaderboards only.

---

## Run store

`runs/` is local truth, mirrored to W&B. Layout (plan §5.3):

```
runs/<experiment>/
  shared/        panel.<hash>.parquet, series.<hash>/, feature_selection.<hash>.json
  tuning/<model>/    optuna.sqlite3, best_params.json, trials.csv
  <model>/<paradigm>/seed=<s>/
    config.yaml  env.json  state.json
    {cv,test}/predictions/part-<from>-<to>.parquet
    {cv,test}/metrics/{global.json, per_region.csv, per_horizon.csv, ...}
    artifacts/     importances, calibrators.json
  report/        the §7 outputs above
```

Stage identity is `hash(resolved stage config + upstream hashes + seed)`. A
complete stage is skipped; a partial one is recomputed from fold 0 (the engine
cannot start mid-schedule yet, audit C20). A tunable model without
`tuning/<model>/best_params.json` is an error in `cv`/`test` unless
`--allow-default-params` is given. `state.json` records every attempt (host,
SLURM job, start/end) and is updated under a per-run lock, so the `cv` and
`test` jobs of one run can run concurrently. Move a store between cluster and
laptop with `scripts/sync_runs.py` (`--dry-run` first).

To seed the store with the thesis's own tuned parameters:

```sh
python scripts/import_golden_params.py --dry-run
python scripts/import_golden_params.py
```

---

## Tests

```sh
.venv/bin/python -m pytest tests/unit tests/equivalence -q -p no:cacheprovider
```

| Suite | What it pins |
|---|---|
| `tests/unit` | the package, on synthetic frames and fakes — no real data |
| `tests/equivalence` | **level D**: the ported engine against the verbatim legacy runners in `tests/legacy_ref/`, on a synthetic 4-region panel |
| `tests/golden` | **levels A-I**: against `golden/`, the frozen thesis output. Needs real data; skips cleanly without it |

Levels A-D and G run in CI (`.github/workflows/ci.yml`) because they need
neither real data nor `golden/`. Levels E, F, H and I compare against the
thesis's own stored results, which live only on Jan's machines.

`-m "not slow"` deselects the tests that fit models. Some golden tests need a
run store first; they skip with the exact command that produces it.

### Reproducibility caveats worth knowing

- **GBDT results are not portable across machines** (F124). The *verbatim
  legacy runner* does not reproduce its own stored predictions here either,
  while the ported engine and that legacy runner agree to 0.0 in-process. The
  GBDTs are therefore pinned by in-process comparison, not against `golden/`.
- **`TweedieDev` is discontinuous at `y_pred = 0`** (F125), and the differenced
  branch predicts across zero. On `diff/arima/test`, 8 of 22,960 rows explain
  88.8% of its deviation while every other metric agrees to ≤3.1e-3. Do not
  report it for the diff family.

---

## Cluster

One command submits the whole thesis re-run as a DAG of SLURM jobs
(`scripts/slurm/submit_all.py`; design and evidence in
`docs/audits/2026-09-26/impl_stream3.md`). It runs on the **login node** with the
system `python3` and only calls `git`, `sbatch` and `squeue`: environments are
built and everything is computed by SLURM jobs.

**1. Clone and copy the git-ignored inputs.** `data/` and `results/` are tracked;
`golden/` is not, and three parts of it are required (8 MB + 63 MB + one JSON):
`golden/converted/` (the thesis' feature-selection caches and converted Optuna
studies, read by `legacy=<experiment>` and `scripts/import_golden_params.py`),
`golden/results/` (the stored thesis predictions the legacy verification compares
against) and `golden/checkpoints/chronos2_best/best_params.json` (the thesis'
Chronos-2 fine-tune parameters). From the laptop's repo root:

```sh
ssh <you>@aurometalsaurus 'git clone -b refactor https://github.com/J4nvg/thesis.git ~/thesis-refactor'
rsync -avR golden/converted golden/results golden/checkpoints/chronos2_best/best_params.json \
    <you>@aurometalsaurus:thesis-refactor/
```

**2. Build the environments once** (a GPU job installs uv into `~/.local/bin`,
runs `uv sync --frozen` for `.venv` and `envs/autogluon`, smoke-tests CUDA,
LightGBM/XGBoost/CatBoost and downloads the Chronos-2 weights into
`~/.cache/huggingface`). Optionally log in to W&B afterwards (runs are online,
`strict: false`: a W&B problem never fails a job):

```sh
cd ~/thesis-refactor
python3 scripts/slurm/submit_all.py --setup-only
python3 scripts/slurm/submit_all.py status          # wait until both setup jobs are done
.venv/bin/wandb login                               # once; writes ~/.netrc
```

**3. Look before submitting.** `--dry-run` prints the job counts per stage and
class and every `sbatch` line (dependencies shown by name):

```sh
python3 scripts/slurm/submit_all.py --dry-run
```

**4. Pilot, then size the wall times.** `--benchmark` runs the same matrix with
2 retrain windows per stage and 2 trials per study into `runs_benchmark/`;
`timings` turns the measured seconds per fold/trial into
`configs/cluster/time_table.json` (commit it; `submit_all.py` reads it):

```sh
python3 scripts/slurm/submit_all.py --benchmark
python3 scripts/slurm/submit_all.py timings          # after the pilot has finished
git add configs/cluster/time_table.json && git commit -m "cluster time table" && git push
```

**5. Submit the thesis re-run (seed 42) and watch it.** Refuses a dirty or
unpushed checkout (`--allow-dirty` overrides) and writes a manifest to
`logs/submissions/`:

```sh
python3 scripts/slurm/submit_all.py
python3 scripts/slurm/submit_all.py status           # pending/running/done/failed + failed log tails
```

The DAG: `setup` → `featsel` (per family, CPU) → `tune` (per model, requeued
before the 36 h limit and resumed from the Optuna study) → `cv` (only where a
later stage needs it: the hurdle's calibrators) → `test` → `importance` →
`report` → `figures` (a no-op until `strikecast figures` exists), plus the
independent legacy verification (`verify_prep` → `verify` → `verify_report`,
into `runs_verify/`; the pass/fail table is
`runs_verify/_verification/verification.{csv,md}`). Each model's cv/test waits
only for its own tune. GPU jobs ask for `--gres=gpu:1 --gres-flags=disable-binding
-c 8`, CPU jobs for `-c 48`; a job class estimated above 30 h goes to
`GPUExtended`. Rerunning the same command after a failure resubmits only what is
not complete.

**6. More seeds later**, without redoing tune/cv (jobs still queued from the
first submission become dependencies automatically):

```sh
python3 scripts/slurm/submit_all.py --seeds 1,2,3,4 --stages test,report
```

**7. Bring the results home.** Copy the cluster store next to the laptop's and
merge it stage by stage (a complete stage is never replaced by a partial one):

```sh
rsync -a <you>@aurometalsaurus:thesis-refactor/runs_publication/ ~/cluster_pull/runs_publication/
python scripts/sync_runs.py ~/cluster_pull/runs_publication runs_publication --dry-run
python scripts/sync_runs.py ~/cluster_pull/runs_publication runs_publication
```

Stores: `runs_publication/` (the re-run, fixed `leaky7` features, re-tuned),
`runs_benchmark/` (pilot), `runs_verify/` (legacy mode) and `runs/` (the laptop's
legacy-mode runs the golden tests read) must stay separate: tuned parameters and
feature-selection caches are per store. Single jobs can still be generated with
`scripts/slurm/make_jobs.py` (the same thesis matrix, one line per job).

---

## Repository layout

| Path | What |
|---|---|
| `src/strikecast/` | the package: `data`, `backtest`, `transforms`, `models`, `pipeline`, `evaluation`, `store`, `tuning`, `tracking`, `config`, `cli` |
| `configs/` | the Hydra tree: `experiment/`, `paradigm/`, `backtest/`, `seeds/`, `tracking/` |
| `tests/` | `unit/`, `equivalence/`, `golden/`, and `legacy_ref/` (the oracles) |
| `docs/` | `REFACTOR_PLAN.md` (scope, flags, phases), `REFACTOR_PROGRESS.md` (hand-off log), `RETIREMENT.md` |
| `scripts/` | `import_golden_params.py`, `sync_runs.py`, `slurm/` |
| `src/*.py`, `*.ipynb` | **the thesis's original code, frozen.** It is the oracle the golden tests compare against — see `docs/RETIREMENT.md` before moving anything |
| `archive/` | out-of-scope experiments (the log-transformed regression family) |
| `golden/`, `runs/` | git-ignored; frozen thesis output and the run store |

## Data

`data/fixed/` and `data/dataset/` hold the region list, actor mappings and the
master parquet. The panel is built by `strikecast.data.build_panel`; the EDA
notebooks (`eda.ipynb`, `eda_full.ipynb`) and `results/analyse_results.ipynb`
read through the package rather than rebuilding it inline.
