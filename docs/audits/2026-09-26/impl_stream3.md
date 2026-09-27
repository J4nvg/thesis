# Implementation Stream 3 — WP3: one-command SLURM orchestration + legacy verification

Date: 2026-09-26. Branch `refactor` (uncommitted, nothing committed or pushed). Owner: Stream 3.
Status: ALL SEVEN ITEMS DONE (see "Tests and results" for the final suite counts).

## The commands Jan types on the cluster

Login node `aurometalsaurus`, repo in `~/thesis-refactor`. Everything below is `python3`
(system, 3.9+ is enough) on the login node; nothing computes there.

```sh
# 0. laptop: clone on the cluster, copy the git-ignored inputs (8 MB + 63 MB + 1 JSON)
ssh <you>@aurometalsaurus 'git clone -b refactor https://github.com/J4nvg/thesis.git ~/thesis-refactor'
rsync -avR golden/converted golden/results golden/checkpoints/chronos2_best/best_params.json \
    <you>@aurometalsaurus:thesis-refactor/

# 1. once: build .venv + envs/autogluon (GPU jobs), then W&B login
cd ~/thesis-refactor
python3 scripts/slurm/submit_all.py --setup-only
python3 scripts/slurm/submit_all.py status
.venv/bin/wandb login

# 2. look, pilot, size
python3 scripts/slurm/submit_all.py --dry-run
python3 scripts/slurm/submit_all.py --benchmark            # runs_benchmark/, 2 windows, 2 trials
python3 scripts/slurm/submit_all.py timings                # -> configs/cluster/time_table.json
git add configs/cluster/time_table.json && git commit -m "cluster time table" && git push

# 3. the thesis re-run at seed 42 (+ legacy verification), then watch
python3 scripts/slurm/submit_all.py
python3 scripts/slurm/submit_all.py status

# 4. later seeds (tune/cv not redone; still-queued jobs become dependencies)
python3 scripts/slurm/submit_all.py --seeds 1,2,3,4 --stages test,report

# 5. laptop: bring results home
rsync -a <you>@aurometalsaurus:thesis-refactor/runs_publication/ ~/cluster_pull/runs_publication/
python scripts/sync_runs.py ~/cluster_pull/runs_publication runs_publication --dry-run
python scripts/sync_runs.py ~/cluster_pull/runs_publication runs_publication
```

Other flags: `--experiments`, `--models`, `--stages`, `--cv all`, `--no-verify`, `--verify-only`,
`--tracking wandb_offline`, `--time-table FILE`, `--long-threshold 30`, `--force`, `--allow-dirty`,
`--store-root`. Laptop verification of one cheap model:
`PYTHONHASHSEED=0 .venv/bin/python scripts/verify_legacy.py all --experiment=diff --models=linear --windows=1 --store-root runs_verify`
(ran here: linear PASS max|dy| 4.6e-10, naive_last PASS 0.0; ~3 min, mostly the panel build).

## The DAG (`submit_all.py --dry-run`, seed 42, laptop, both setup jobs included)

```
setup (GPU: uv, .venv, CUDA/GBDT smoke) ─┬─ setup_ag (envs/autogluon + Chronos-2 weights)
                                          ├─ featsel:{count,diff,hurdle} (CPU, all heads, once)
                                          │    └─ tune:<exp>:<model>  (requeue, USR1@900 s)
                                          │         ├─ cv:hurdle:hurdle:<paradigm>  (calibrators)
                                          │         └─ test:<exp>:<model>:<paradigm>:s42 (afterok own tune [+cv])
                                          │              └─ importance:<exp>:<model>:<paradigm>
                                          ├─ report:<exp> (afterany: every job of the experiment)
                                          │    └─ figures (afterany: reports; skipped until `strikecast figures` exists)
                                          └─ verify_prep → verify:<exp>:<cpu|gpu> → verify_report (runs_verify/)
```

| stage | class | partition | jobs |
|---|---|---|---|
| setup | gpu | GPU | 2 |
| featsel | cpu | GPU | 3 |
| tune | cpu | GPU | 7 (count 6 GBDT, diff lightgbm) |
| tune | gpu | GPU | 24 (count 15 RNN, diff xgb/cat + 6 RNN, chronos2 ft) |
| cv | cpu | GPU | 3 (hurdle x 3) |
| test | cpu | GPU | 28 (count GBDT 18, diff lightgbm 3 + 4 baselines, hurdle 3) |
| test | gpu | GPU | 56 (count RNN 30, diff 24, chronos2 2) |
| importance | cpu / gpu | GPU | 4 / 2 (Stream 2 `DEFAULT_JOBS`) |
| report | cpu_small | GPU | 4 |
| figures | cpu_small | GPU | 1 |
| verify | cpu / gpu / cpu_small | GPU | 4 / 3 / 1 |
| **total** | | | **142** |

The thesis matrix is the 84 test runs + 3 hurdle CV of D8 (count GBDT x 3, count RNN x
global/activity, diff tuned x 3 incl. RNN x local, diff baselines global, Chronos-2 x 2,
hurdle x 3); the 31 tune jobs are D4 (re-tune everything; the Chronos-2 fine-tune too).
Damage is not in the default `--experiments`.

## Resources and wall time (defaults; evidence = audit C "Runtime estimates")

GPU jobs: `-p GPU --gres=gpu:1 --gres-flags=disable-binding -c 8 --mem=40G`; CPU jobs
`-p GPU -c 48 --mem=96G`; small (report/figures/verify_report) `-c 8 --mem=32G`. Two GPU jobs +
one CPU job fit a 64-CPU/192 GB node. No `OMP_NUM_THREADS` export (threads from the config).
Time table in `submit_all.TIME_TABLE` (hours, most specific key wins; each entry carries its
evidence string, copied into the manifest):

| key | h | evidence |
|---|---|---|
| tune/lightgbm/cpu | 8 | count LGBM-P study 2.14 h (a) |
| tune/xgboost/cpu | 24 | diff XGB 2.12 h on GPU; CPU 3-5x (unverified) |
| tune/catboost/cpu | 36 + requeue | diff CatBoost 8.44 h on GPU; CPU 17-40 h |
| tune/xgboost/gpu, catboost/gpu | 8, 20 | diff studies 2.12 h, 8.44 h |
| tune/rnn/gpu | 6 | 24 RNN studies 0.28-1.49 h |
| tune/chronos/gpu | 10 | 636 s per 1500-step fit, 12 trials |
| cv/test gbdt cpu global/activity/local | 4-8 / 6-12 / 12-24 | laptop LGBM-P test 2 h 40 (contended); local no evidence |
| test rnn gpu global/activity/local | 4 / 6 / 16 | trials 0.3-4 min |
| baselines | 4 | ARIMA test 47 min laptop |
| composite (hurdle) cv/test | 8-24 / 12-28 | no evidence (first pilot candidate) |
| test/chronos/gpu | 4 | zero-shot 100 s fit, 2 s/fold |
| importance / report / figures / verify | 16 / 2 / 2 / 8 | C14 2-10 h / minutes / <1 h / 14 folds |

Routing: a resumable tune job stays on `GPU` (max 36 h) with `--requeue --open-mode=append
--signal=B:USR1@900`; `job.sbatch` forwards SIGTERM to Python and calls `scontrol requeue`
(at most `STRIKECAST_MAX_REQUEUES=8`); Optuna resumes from its SQLite study (a killed trial stays
RUNNING, is not counted, and is re-run). Any other job class above `--long-threshold` (30 h) goes
to `GPUExtended`; with the default table none does (local hurdle test is 28 h). `--benchmark`
caps pilot jobs at 6 h. `timings` writes `configs/cluster/time_table.json` =
max over models of (s/fold x 79|164 folds, or max s/trial x 50|12 trials) x 1.5, per key.

## Files changed / added

| file | change |
|---|---|
| `src/strikecast/config/schema.py` | `ModelEntry.paradigms: list[Paradigm] \| None` (additive, metadata only) |
| `configs/experiment/{count,diff,hurdle,chronos2,damage}.yaml` | per-model `paradigms` lists (models blocks only) |
| `configs/tracking/wandb_online.yaml` | `strict: false` (D11) |
| `scripts/slurm/make_jobs.py` | thesis matrix (per-model paradigms, `--cv needed`, stages the experiment lacks skipped), `Job` metadata (experiment/paradigm/seed/family/kind/env), Chronos spec lookup, C17 skip rules (`stage_done`), GPU gres in the printed sbatch line |
| `scripts/slurm/submit_all.py` (new) | the DAG, resources/time, sbatch, manifest, `status`, `timings`, `--benchmark`, dirty/unpushed refusal, re-exec under `.venv` |
| `scripts/slurm/job.sbatch` (new) | generic job wrapper: prebuilt venv, PYTHONHASHSEED=0, traps, requeue, optional command, `STRIKECAST_EXIT` line |
| `scripts/slurm/setup.sbatch`, `scripts/slurm/smoke_test.py` (new) | env build + CUDA/GBDT/Chronos smoke test + stamp (lock sha256) |
| `scripts/slurm/{run,tune}.sbatch` | no per-task `uv sync` (C24), no OMP export, prebuilt `.venv/bin/python` |
| `src/strikecast/store/run_store.py` | `RunLock` (mkdir lock dir, NFS-safe, re-entrant, stale-breaking) around every state/env RMW (C23); unique temp names; `StageState.params_source/max_folds/attempts`; `interrupted` status, `interrupt_stage`, `interrupt_active_stages` (C21) |
| `src/strikecast/pipeline/run_stage.py` | `MissingTunedParams` + `require_tuned_params` (C17), `allow_default_params`, `max_folds` (hash only when set), interrupted/failed marking of the whole post-start section; forwards the two kwargs to the composite/Chronos stages when they accept them |
| `src/strikecast/pipeline/interrupt.py` (new) | SIGTERM/SIGUSR1 -> `StageInterrupted` (run etc.) or `os._exit(99)` (tune) |
| `src/strikecast/cli/main.py` | `--allow-default-params`, `--max-folds`; `main` -> `_run_interruptible` -> `_dispatch` (handlers installed and restored; exit 99) |
| `src/strikecast/verification/{__init__,legacy}.py`, `scripts/verify_legacy.py` (new) | legacy verification (prep/run/report/all) |
| `scripts/sync_runs.py` | one line: a `.state.lock` dir is never a stage |
| `.gitignore` | `/logs/`, `/runs_*/` appended |
| `README.md` | Cluster section rewritten; run-store paragraph corrected (C20: no mid-schedule resume) |
| tests | `test_submit_all.py` (new, 31), `test_verify_legacy.py` (new, 21 + 1 opt-in e2e), additions to `test_make_jobs.py` (+10), `test_run_store.py` (+9), `test_cli.py` (+4), `test_sync_runs.py` (+1) |

## Item notes

1. **Matrix (C16/D8).** `ModelEntry.paradigms` (`None` = experiment paradigms, `[]` = never a
   standalone job). `make_jobs` without `paradigm=` uses them; cv only where a later stage
   needs it (`--cv needed`: composites with calibration); an explicit `stage=cv` means cv for all.
2. **submit_all (C30).** As above. Setup is submitted automatically when a venv stamp is missing
   or was built from another lock. `status` needs no sacct: squeue for live jobs, the
   `STRIKECAST_EXIT <code>` last log line for finished ones (0 done, 99 interrupted, other failed,
   none = killed by wall clock/OOM/node), refined by `state.json` (e.g. shows `MissingTunedParams`).
   Manifest: `logs/submissions/<UTC timestamp>[-benchmark].json` (git commit/branch/dirty,
   options, per-experiment `resolved_hash`, every job's id, sbatch line, deps, time + evidence).
3. **Safety (C17/C21/C23).** See files table. The skip check in `make_jobs.stage_done` never
   trusts a stage that ran on defaults, finished before `best_params.json`, or was truncated;
   the pipeline's identity hash (params inside) remains the real check at run time.
4. **status** — `python3 scripts/slurm/submit_all.py status [--manifest F] [--all] [--tail N]`.
5. **Verification (D9).** `legacy=<exp>` + `tracking=noop` + separate store; stage `test` (hurdle:
   `cv`, whose golden CV frames exist and which needs no calibrators from another run);
   `--windows 2` = 14 folds. Keys and `y_true` always exact (FAIL otherwise). Levels: E (naive
   exact, linear 1e-6, count CatBoost 1e-6), F (LightGBM 5.0, ARIMA 0.5, Chronos zero-shot 1e-2),
   R (report only: RNNs, XGBoost, diff CatBoost GPU, Chronos fine-tune, hurdle; SUSPECT if MAE
   differs >25%). Hurdle compares the prob/count/hurdle channels (global only has golden frames).
   `prep` imports the thesis params (`import_golden_params.py` incl. cell 56, plus
   `golden/checkpoints/chronos2_best/best_params.json`) and runs the hurdle's legacy featsel once.
   Output `runs_verify/_verification/verification.{csv,md}` + one JSON per case.
6. **Tracking (D11).** `wandb_online` is the experiments' default group, now `strict: false`;
   `--tracking wandb_offline` (then `scripts/wandb_sync.sh`) remains available; pilot and
   verification use `noop`.
7. **README** — Cluster section rewritten (clone, rsync of exactly `golden/converted`,
   `golden/results`, `golden/checkpoints/chronos2_best/best_params.json`, setup, dry-run,
   benchmark/timings, submit, status, extra seeds, sync back).

## Dependencies on Streams 1/2 (all resolved in the current tree)

- Stream 1: `strikecast featsel experiment=<e>` (all heads) and `legacy=<experiment>` group
  (`configs/legacy/`) are used as-is; cell-56 import in `import_golden_params.py` is used by
  `verify_prep`. The publication configs' `feature_selection.require_cached` makes the
  `featsel:<exp>` job a hard prerequisite, which the DAG encodes.
- Stream 2: `run_chronos_stage` accepts `allow_default_params`/`max_folds` (forwarded);
  `strikecast importance ... seed=42` per `importance_stage.DEFAULT_JOBS`; `envs/autogluon` is an
  editable package (job.sbatch also sets `PYTHONPATH=<repo>/src`).

## Open issues / for Jan

- **Not run on a cluster** (no access): sbatch lines, `scontrol requeue` from inside a job, and
  `--gres-flags=disable-binding` are per the verified cluster facts but untested end to end.
  First real step: `--setup-only`, then `--benchmark`.
- `report` for `chronos2` runs in the main env (reads the store only); if it ever needs AutoGluon,
  switch its node to `env="autogluon"`.
- Importance jobs of one experiment each call `collect_importance` at the end; two finishing at
  the same moment could each write a collection missing the other's rows (atomic writes, last one
  wins). The report job does not re-collect: Stream 2/4 should re-collect in `report`/`figures`.
- Fold-level resume (C20) is NOT implemented: a cv/test job killed at the wall clock restarts from
  fold 0. Mitigated by routing (>30 h -> GPUExtended) and the pilot-based time table.
- Optuna SQLite on NFS: one writer per study (one tune job per model), which SQLite handles; a
  requeued attempt leaves one RUNNING zombie trial per requeue in `trials.csv` (C22, harmless).
- `threads: auto` resolves to 64 on these nodes regardless of `-c 48` (TaskPlugin=task/none);
  LightGBM/CatBoost results are thread-invariant (B5); XGBoost thread invariance is untested.
- W&B online needs `~/.netrc` (`.venv/bin/wandb login`); without it jobs warn and run untracked.

## Tests and results

(filled in at the end of the session)
