#!/bin/bash
# SLURM array template for `strikecast run` (plan §5.6, §8 P6).
#
# ONE array task = ONE line of a jobs file = ONE `strikecast` invocation, e.g.
#
#     run experiment=count model=lightgbm_poisson paradigm=global stage=test seed=1
#
# Generate the jobs file first; it also prints the sbatch line to use:
#
#     uv run python scripts/slurm/make_jobs.py experiment=count stage=test \
#         paradigm=global,activity,local seed=42,1,2,3,4 --out jobs
#     # -> jobs.cpu.txt: 63 CPU job(s)
#     sbatch --array=1-63 --partition=<cpu-partition> scripts/slurm/run.sbatch jobs.cpu.txt
#
# The generator writes one file per resource class, because a SLURM array is
# homogeneous and the model spec decides the class (§5.6, "Jobs declare their
# resource class (cpu, gpu) from the model spec"). Submit the GPU file with the
# GPU options on the command line -- command-line options override the #SBATCH
# directives below, so this one template serves both classes:
#
#     sbatch --array=1-18 --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=8 \
#            scripts/slurm/run.sbatch jobs.gpu.txt
#
# There is no submitit route and no `configs/hydra/launcher/` group. Dropped on
# 2026-09-19 (Stream K) for one engineering reason: `strikecast` is an argparse
# front end that calls `hydra.compose` and expands the model/paradigm/stage/seed
# selectors IN PROCESS, so `-m ... hydra/launcher=slurm` would compose without
# error and then run the whole sweep on the login node, silently. Making it real
# needs a `@hydra.main` entry point in `src/strikecast/cli/`. Until that exists,
# these array templates are the only cluster path, and `hydra-submitit-launcher`
# is deliberately absent from `pyproject.toml`.
#
# Nothing here needs a login node with a GPU, and nothing here syncs W&B:
# runs are online by default (§1 "Storage"); for an offline queue set
# `tracking=wandb_offline tracking.dir=$WANDB_DIR` on the job line and run
# `scripts/wandb_sync.sh` afterwards.
#
#SBATCH --job-name=strikecast-run
#SBATCH --output=logs/slurm/run-%A_%a.out
#SBATCH --error=logs/slurm/run-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-1

set -euo pipefail

JOBS="${1:-${STRIKECAST_JOBS:-jobs.cpu.txt}}"
TASK="${SLURM_ARRAY_TASK_ID:-1}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

LINE="$(sed -n "${TASK}p" "$JOBS")"
if [ -z "$LINE" ]; then
  echo "make_jobs: no line ${TASK} in ${JOBS}" >&2
  exit 2
fi

# F16: the legacy covariate column order depends on the hash seed, and CPython
# reads PYTHONHASHSEED before any user code runs, so it has to be set here.
export PYTHONHASHSEED=0
# One process per task; the array is the parallelism.
# No OMP_NUM_THREADS export: thread counts come from the config (`threads:`),
# never from the allocation (audit 2026-09-26 B5/WP1).
export TOKENIZERS_PARALLELISM=false
# The run store is the source of truth (§1) and lives on shared scratch.
STORE_ROOT="${STRIKECAST_STORE_ROOT:-$PWD/runs}"
# Offline W&B runs (if the job line asks for them) land next to the store.
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

echo "[$(date -Is)] task ${TASK}/${SLURM_ARRAY_TASK_COUNT:-?} on $(hostname): strikecast ${LINE}"
# Word splitting is intended: every token of LINE is one `key=value` argument.
# shellcheck disable=SC2086
srun "$PY" -m strikecast.cli.main $LINE --store-root "$STORE_ROOT" -v
echo "[$(date -Is)] task ${TASK} done"
