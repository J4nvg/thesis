# Weights & Biases: laptop and cluster

How the `strikecast` pipeline mirrors runs to W&B (online), how to set it up once,
and how to check that it works. Everything below is taken from the code; `file:line`
references are to the `refactor` branch as of 2026-09-27.

**The one rule:** the local run store (`runs*/`) is the source of truth. W&B is a
mirror of it (`src/strikecast/tracking/protocol.py:1-20`). With `strict: false` (the
default, decision D11) a W&B problem is logged as a WARNING and the job carries on
untracked; results on disk are never affected
(`src/strikecast/tracking/wandb_tracker.py:149-157`, `:299-310`;
`src/strikecast/pipeline/context.py:269-284`).

## 1. What ends up in W&B

One W&B run per **(experiment, model, paradigm, seed) and stage** — the same unit as a
run-store directory (`wandb_tracker.py:3-16`):

| W&B field | Value | Where |
|---|---|---|
| project | `tracking.project`, default `strikecast` | `configs/tracking/wandb_online.yaml:6` |
| entity | `tracking.entity`, default `null` = your W&B default entity | `wandb_online.yaml:7` |
| name | `<experiment>/<model>/<paradigm>/seed=<s>`, identical to the run directory | `protocol.py:50-60` |
| group | the experiment (`tracking.group` overrides it) | `wandb_tracker.py:138` |
| job_type | the stage: `tune`, `cv` or `test` | `wandb_tracker.py:139` |
| tags | `[family, kind]` from the model spec, plus `tracking.tags` | `pipeline/context.py:224-237` |
| config | the full resolved experiment config, plus `run.{experiment,model,paradigm,seed,stage,dir}` | `wandb_tracker.py:268-288` |
| id | first 16 hex chars of `sha1("<name>#<stage>")` | `wandb_tracker.py:55-64` |

Because the id is deterministic and `resume="allow"` (`wandb_tracker.py:86`),
re-running a stage (a requeued tune job, a resubmission after a failure, `--force`)
**continues the same W&B run** instead of creating a duplicate.

What each stage logs:

- **cv / test** (`pipeline/run_stage.py:554-568`, `:631-633`, `:707-713`):
  running global metrics after each retrain window as `cv/<metric>` / `test/<metric>`
  with `step = fold index` (the curve is the cumulative score, the same one the Optuna
  pruner sees, `tracking/hook.py:10-13`); at the end the metric views as W&B Tables
  (`<stage>/<view>`) and two artifacts, the `predictions/` and `metrics/` directories
  (type `predictions` / `metrics`, name `<exp>__<model>__<paradigm>__seed-<s>__<stage>__<kind>`,
  `wandb_tracker.py:290-294`).
- **tune** (`pipeline/tune_stage.py:303-320`, `:183-213`): one W&B run per study,
  named `<exp>/<model>/global/seed=42`, with one point per finished trial
  (`trial/number`, `trial/value`, `trial/params/<name>`, step = trial number). Pruned
  and failed trials are skipped.
- **Chronos-2 and composite (hurdle) stages**: start/finish, tables and the two
  artifacts, but no per-fold curve (`pipeline/chronos_stage.py:502-574`,
  `pipeline/composite_stage.py:770-859`).
- W&B itself adds system metrics (CPU/GPU), console output and the git commit.

**Not logged:** `featsel`, `importance`, `evaluate`, `report`, `verify`; model
weights; the tune stage's `best_params.json` / `trials.csv` (they stay in
`tuning/<model>/` of the store). Runs finish as W&B state *finished* or *failed*
(`wandb_tracker.py:52`, `:236-247`); a job that is SIGKILLed shows as *crashed*.

## 2. One-time setup

1. **Account, entity, project.** Sign in at <https://wandb.ai>. Your personal entity
   is your username; a team (e.g. a university team) is a separate entity. The
   project `strikecast` is created automatically on the first run. If you want runs in
   a team, either make it your *default entity* (W&B settings page) or set it
   explicitly (below).
2. **API key.** Copy it from <https://wandb.ai/authorize>. Never paste it into a
   config file, notebook or script in this repo.
3. **Laptop login** (writes `~/.netrc`, once):
   ```sh
   uv run wandb login          # or: .venv/bin/wandb login
   ```
4. **Cluster login**, on the login node after the setup job has built `.venv`
   (README "Cluster", step 2). `/home` is shared NFS, so the `~/.netrc` written here
   is what every compute job reads:
   ```sh
   cd ~/thesis-refactor
   .venv/bin/wandb login
   chmod 600 ~/.netrc
   ```
   The Chronos-2 jobs use `envs/autogluon/.venv`, which has its own `wandb==0.30.0`
   (`envs/autogluon/pyproject.toml:53`) but reads the same `~/.netrc`.
5. **Alternative: `WANDB_API_KEY`.** `submit_all.py` submits every job with
   `--export=ALL,...` (`scripts/slurm/submit_all.py:551-554`), so a variable exported
   in the login shell *at submit time* reaches all jobs, including requeued ones:
   ```sh
   read -rs WANDB_API_KEY && export WANDB_API_KEY   # typed, not in shell history
   python3 scripts/slurm/submit_all.py
   ```
   `~/.netrc` is simpler and survives new shells; use one or the other.
6. **Old Comet key.** `docs/REFACTOR_PLAN.md:90`, `:380`, `:450` record that a Comet
   API key was committed in the old notebooks and is still in git history (commits
   `ed0ebe1`, `5bfe32d`, `848df73`, `b1bfd79`); the plan says it is already invalid and
   history is not rewritten. Confirm in the Comet dashboard that it is revoked,
   especially if the GitHub repo is (or becomes) public.

**Choosing entity / project** — what actually works in this code:

| Setting | Laptop / single run | Cluster via `submit_all.py` |
|---|---|---|
| entity | `tracking.entity=<team>` (Hydra override) or `export WANDB_ENTITY=<team>` (used because the code passes `entity=None`) | `--wandb-entity <team>` (or `export WANDB_ENTITY=<team>`, or your W&B default entity) |
| project | `tracking.project=<name>` | `--wandb-project <name>` |
| group, tags | `tracking.group=<g>`, `tracking.tags=[a,b]` | tags: `--wandb-tag a --wandb-tag b`; group: not exposed |

`WANDB_PROJECT` has **no** effect on pipeline runs: the tracker always passes
`project=` explicitly (`wandb_tracker.py:134`). It is only read by `scripts/wandb_sync.sh`.

## 3. Commands

All experiment configs compose `/tracking: wandb_online` by default
(`configs/experiment/diff.yaml:11`, same in `count`, `hurdle`, `damage`, `chronos2`),
so online tracking needs no flag.

**Laptop, one run (online):**
```sh
PYTHONHASHSEED=0 uv run strikecast run experiment=diff model=linear paradigm=global \
    stage=cv,test seed=42 --store-root runs
# add tracking.entity=<team> / tracking.project=<name> as needed
```

**Cluster, the whole DAG (online is the default):**
```sh
python3 scripts/slurm/submit_all.py --dry-run     # check the lines: no tracking= means wandb_online
python3 scripts/slurm/submit_all.py
```
`--tracking <group>` replaces the group for every job (`submit_all.py:1113-1115`,
`:307`, `:681`); `--benchmark` defaults to `noop` (`submit_all.py:1160`), and the legacy
verification jobs always use `noop`.

**Offline, synced later** (for a node without network, or to avoid rate limits):
```sh
python3 scripts/slurm/submit_all.py --tracking wandb_offline
# jobs write wandb/offline-run-* in the repo dir (the job's cwd, job.sbatch:39)
# afterwards, on the login node:
cd ~/thesis-refactor
PATH="$PWD/.venv/bin:$PATH" scripts/wandb_sync.sh -n     # dry run: what would sync
PATH="$PWD/.venv/bin:$PATH" scripts/wandb_sync.sh -c     # sync, delete synced dirs
```
`wandb_sync.sh` defaults to `$WANDB_DIR/wandb` or `./wandb`, project `strikecast`
(`-p`), entity `$WANDB_ENTITY` (`-e`), skips dirs already marked `.synced` and exits
with the number of failures (`scripts/wandb_sync.sh:15-33`, `:86-114`). On the laptop
the same works as `tracking=wandb_offline` then `uv run scripts/wandb_sync.sh`.

**No tracking at all:** add `tracking=noop` (laptop) or `--tracking noop`
(`submit_all.py`). This also skips the per-fold re-scoring
(`pipeline/context.py:240-266`).

## 4. Check that it works

**a. Connectivity only (seconds, no data).** Run on the laptop, and on the cluster
once on the login node and once inside a job (`srun -p GPU --pty bash`):
```sh
.venv/bin/python - <<'EOF'
from types import SimpleNamespace as Key
from strikecast.tracking import WandbTracker
t = WandbTracker(project="strikecast-smoke", strict=True)   # strict: fail loudly
print(t.start(Key(experiment="smoke", model="hello", paradigm="global", seed=0),
              {"note": "connectivity"}, stage="cv"))
t.log_fold(0, {"cv/MAE": 1.0}); t.finish()
EOF
```
It prints the run id; the run appears in project `strikecast-smoke`. Delete the
project afterwards.

**b. Real pipeline smoke (minutes).** The cheapest real run is a naive model on two
retrain windows (14 folds). `diff` requires a cached feature selection
(`configs/experiment/diff.yaml:76`), so build it once first:
```sh
PYTHONHASHSEED=0 uv run strikecast featsel experiment=diff --store-root runs_smoke
PYTHONHASHSEED=0 uv run strikecast run experiment=diff model=naive_last paradigm=global \
    stage=cv seed=42 --max-folds 14 --store-root runs_smoke \
    tracking.project=strikecast-smoke tracking.strict=true
```
In the W&B UI (project `strikecast-smoke`, group `diff`) you should see one run
`diff/naive_last/global/seed=42`, job type `cv`, tags `[family, kind]`, charts
`cv/<metric>` with points at fold 0 and 7, the metric Tables, and two artifacts
(`...__cv__predictions`, `...__cv__metrics`). Use `strict=true` only for smoke tests.

**Run directory -> W&B run.** The id is recorded on disk
(`store/run_store.py:609-651`):
```sh
jq '.tracker_run_ids' runs_smoke/diff/naive_last/global/seed=42/env.json
# {"cv": "<id>"}  ->  https://wandb.ai/<entity>/<project>/runs/<id>
```
`state.json` also has `tracker_run_id` per stage. Tune runs are not recorded on disk;
compute their id instead (works for any run, since it is deterministic):
```sh
uv run python -c "from strikecast.tracking import wandb_run_id; \
print(wandb_run_id('diff/lightgbm/global/seed=42', 'tune'))"
```
**W&B run -> run directory.** The run name *is* the path under the store root; the
config also carries `run.dir` and `run.stage`, so you can filter the runs table on them.

## 5. Troubleshooting

- **Jobs run but nothing appears in W&B.** Look in the job log
  (`logs/slurm/strikecast-<jobid>.out`) for `wandb.init failed ... tracking disabled`
  or `tracker construction failed`. Usually: not logged in / HTTP 401 (no or wrong
  `~/.netrc` on the cluster, expired key). Fix with `.venv/bin/wandb login --relogin`;
  already finished jobs stay untracked (rerunning them with `--force` recomputes, so
  usually just accept the gap: the store has everything).
- **Wrong entity / "project not found" / 403.** The key's user has no access to the
  team, or `WANDB_ENTITY` was not exported when `submit_all.py` ran.
- **Proxy / network.** Compute nodes have internet, so nothing is needed normally. If
  a node cannot reach `api.wandb.ai`, use `--tracking wandb_offline` and sync later. A
  slow `wandb.init` can be given more time with `export WANDB_INIT_TIMEOUT=300`
  before submitting (propagated by `--export=ALL`).
- **Rate limits with ~140 concurrent jobs.** Per-fold logging is light (one point per
  retrain window), but every cv/test run uploads its predictions and metrics
  directories as artifacts, and many runs start at once. W&B retries HTTP 429s itself;
  a call that still fails disables the mirror for that run only. If many runs end up
  half-logged, switch to offline and sync in batches with `wandb_sync.sh`.
- **Duplicate or "resumed" runs.** Same key + stage = same id, so a rerun resumes the
  existing W&B run; that is intended. W&B ignores logs with a step lower than one
  already logged, so a rerun's per-fold curve is not redrawn (see Known gaps). A run
  from a different project or entity is a genuinely separate run.
- **`wandb/` dirs piling up.** Online *and* offline runs leave `wandb/run-*` /
  `wandb/offline-run-*` in the repo dir on NFS (git-ignored, `.gitignore:227`). After
  checking the runs in the UI they can be deleted (`rm -rf wandb/run-*`); offline ones
  with `wandb_sync.sh -c`. Note `wandb_sync.sh` also picks up online `run-*` dirs and
  would re-upload them into the same runs: use `-n` first.
- **`strict: true` vs `false`.** `false` (default, D11, `wandb_online.yaml:14-19`): a
  W&B error warns and disables the mirror, the job succeeds. `true`
  (`tracking.strict=true`): the error fails the job. Use `true` only for smoke tests,
  never for 30-hour jobs.

## Known gaps

Found while writing this guide; not fixed.

1. **Per-fold curves only for single-group paradigms.** The fold hook is attached only
   when the paradigm has one region group (`pipeline/run_stage.py:707-716`); `local` and
   `activity` runs, Chronos-2 and composite stages log only the final tables/artifacts.
2. **Tune run id is not recorded in the store.** `tracker.start` goes through
   `track(...)`, which discards the id (`pipeline/tune_stage.py:303-310`), so no
   `env.json`/`state.json` names the tune run (compute it with `wandb_run_id`). The
   `tuning` and `config` artifact kinds (`tracking/protocol.py:41-47`) are never logged.
3. **Rerun after a partial run loses the new curve.** A cv/test job killed at the wall
   clock restarts from fold 0 but resumes the same W&B id (`wandb_tracker.py:55-64`,
   `:86`); W&B drops the lower-step `log_fold` points (`wandb_tracker.py:165`).
4. ~~Project/entity/tags not exposed by `submit_all.py`~~ — **fixed 2026-09-27**: `--wandb-project`,
   `--wandb-entity`, `--wandb-tag` (repeatable) append `tracking.*` overrides to every job line.
   `WANDB_PROJECT` is still ignored (`wandb_tracker.py:134`); use the flag.
5. **No switch to turn off artifact uploads** (`pipeline/run_stage.py:632-633`); each
   cv/test run uploads its prediction parquet, which drives storage use and rate limits.
6. ~~No `WANDB_DIR` on the cluster~~ — **fixed 2026-09-27**: `scripts/slurm/job.sbatch` defaults
   `WANDB_DIR` to `<repo>/logs/wandb` (an exported `WANDB_DIR` wins).
