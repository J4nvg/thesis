#!/bin/bash
# SLURM array template for `strikecast tune` (plan §5.6, §8 P6).
#
# Replaces `tune_gbdt.sh`, `tune_lstm.sh` and `tune_diff_reg.sh`, which ran the
# variants of one family sequentially in one process. ONE array task = ONE line
# of a jobs file = ONE Optuna study:
#
#     tune experiment=count model=lightgbm_poisson
#
# Generate the jobs file (models that already have `best_params.json` are left
# out -- Appendix C: "Tuning is not repeated"), then submit:
#
#     uv run python scripts/slurm/make_jobs.py experiment=count stage=tune --out tune_jobs
#     # -> tune_jobs.cpu.txt: 6 CPU job(s)
#     sbatch --array=1-6 --partition=<cpu-partition> scripts/slurm/tune.sbatch tune_jobs.cpu.txt
#
# The generator splits by resource class (§5.6, the class comes from the model
# spec's device); submit the GPU file with the GPU options on the command line,
# which override the #SBATCH directives below:
#
#     sbatch --array=1-15 --partition=<gpu-partition> --gres=gpu:1 \
#            scripts/slurm/tune.sbatch tune_jobs.gpu.txt
#
# Tuning runs ONCE, under `seeds.tuning_seed` and the Global paradigm (§1
# "Seeds", F14) -- there is no paradigm or seed axis in a tune job. The study
# lives in `runs/<experiment>/tuning/<model>/optuna.sqlite3` with
# `load_if_exists=True`, so a task that hits the wall clock is continued by
# resubmitting the same line; it does not start over (§5.3). F11 records what
# resuming does not restore: the TPE sampler's RNG state.
#
# Because every task writes to its OWN study file, several tasks of this array
# never contend for one SQLite database. Do NOT parallelise a single study
# across tasks: that would need a shared storage and a different F11 story.
#
# There is no submitit route and no `configs/hydra/launcher/` group: the CLI
# expands its selectors in process, so a Hydra launcher would run the sweep on
# the login node without saying so. See the header of `run.sbatch`.
#
#SBATCH --job-name=strikecast-tune
#SBATCH --output=logs/slurm/tune-%A_%a.out
#SBATCH --error=logs/slurm/tune-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-1

set -euo pipefail

JOBS="${1:-${STRIKECAST_JOBS:-tune_jobs.cpu.txt}}"
TASK="${SLURM_ARRAY_TASK_ID:-1}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

LINE="$(sed -n "${TASK}p" "$JOBS")"
if [ -z "$LINE" ]; then
  echo "make_jobs: no line ${TASK} in ${JOBS}" >&2
  exit 2
fi

# F16: CPython reads PYTHONHASHSEED before any user code runs.
export PYTHONHASHSEED=0
# No OMP_NUM_THREADS export: thread counts come from the config (`threads:`),
# never from the allocation (audit 2026-09-26 B5/WP1).
export TOKENIZERS_PARALLELISM=false
STORE_ROOT="${STRIKECAST_STORE_ROOT:-$PWD/runs}"
export WANDB_DIR="${WANDB_DIR:-$STORE_ROOT/../wandb}"

# NO per-task `uv sync` (audit 2026-09-26 C24): an exact sync per task strips
# extras from the shared .venv and races on its lock over NFS. The env is built
# ONCE by `python3 scripts/slurm/submit_all.py --setup-only`; this task only
# checks that it exists. (submit_all.py is the recommended route; it submits
# job.sbatch per job with dependencies. These array templates remain for
# hand-made job lists.)
PY="$PWD/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "no $PY: run 'python3 scripts/slurm/submit_all.py --setup-only' first" >&2
  exit 3
fi

echo "[$(date -Is)] task ${TASK} on $(hostname): strikecast ${LINE}"
# Word splitting is intended: every token of LINE is one `key=value` argument.
# shellcheck disable=SC2086
srun "$PY" -m strikecast.cli.main $LINE --store-root "$STORE_ROOT" -v
echo "[$(date -Is)] task ${TASK} done"
