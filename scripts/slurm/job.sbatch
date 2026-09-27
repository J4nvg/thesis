#!/bin/bash
# One strikecast job of the submit_all.py DAG (audit 2026-09-26 C30).
#
#     sbatch [resources, dependencies...] scripts/slurm/job.sbatch <env> <python args...>
#
# <env> is `main` (the repo's .venv) or `autogluon` (envs/autogluon/.venv, the
# Chronos-2 family). <python args> are handed to that interpreter verbatim, e.g.
#
#     job.sbatch main -m strikecast.cli.main run experiment=count model=lightgbm_poisson \
#                     paradigm=global stage=test seed=42 --store-root runs_publication
#
# submit_all.py writes every sbatch line (see --dry-run); nothing here is meant
# to be edited per job. What this wrapper guarantees:
#
# * NO environment build: the prebuilt venv is used directly (C24). A missing
#   venv is an error that names the fix (`submit_all.py --setup-only`).
# * PYTHONHASHSEED=0 for every job (F16), MPLBACKEND=Agg, no OMP_NUM_THREADS
#   export: thread counts come from the config (`threads:`), never from the
#   allocation (audit B5: LightGBM/CatBoost are thread-count invariant anyway).
# * Signals (C21/C22): SLURM's SIGTERM (wall clock, scancel, requeue) reaches the
#   Python process, which marks its stage `interrupted` and exits 99. A job
#   submitted with `--requeue --signal=B:USR1@<s>` (the resumable tune jobs)
#   gets USR1 <s> seconds before its limit: the trap forwards SIGTERM to Python,
#   waits for it, and requeues this job (`scontrol requeue`), at most
#   STRIKECAST_MAX_REQUEUES times; the Optuna study continues where it stopped.
# * STRIKECAST_OPTIONAL=1 (the `figures` placeholder): if the CLI does not know
#   the subcommand yet, say so and exit 0 instead of failing the DAG.
# * The last line of the log is `STRIKECAST_EXIT <code>`; `submit_all.py status`
#   reads it (there is no sacct on this cluster).
#
#SBATCH --job-name=strikecast
#SBATCH --output=logs/slurm/strikecast-%j.out

set -uo pipefail

ENV_NAME="${1:?usage: job.sbatch <main|autogluon> <python args...>}"
shift

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2
REPO="$PWD"

export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
# W&B writes a local run folder even online; keep them out of the repo root.
export WANDB_DIR="${WANDB_DIR:-$REPO/logs/wandb}"
mkdir -p "$WANDB_DIR"

case "$ENV_NAME" in
  main) PY="$REPO/.venv/bin/python" ;;
  autogluon)
    PY="$REPO/envs/autogluon/.venv/bin/python"
    # envs/autogluon installs a .pth that points at <repo>/src; PYTHONPATH is
    # the belt to those braces for a checkout that moved after the sync.
    export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
    ;;
  *) echo "job.sbatch: unknown env '$ENV_NAME' (main|autogluon)" >&2; echo "STRIKECAST_EXIT 2"; exit 2 ;;
esac

if [ ! -x "$PY" ]; then
  echo "job.sbatch: $PY does not exist; build the environments first:" >&2
  echo "    python3 scripts/slurm/submit_all.py --setup-only" >&2
  echo "STRIKECAST_EXIT 3"
  exit 3
fi

if [ "${STRIKECAST_OPTIONAL:-0}" = "1" ] && [ "${1:-}" = "-m" ] && [ "${2:-}" = "strikecast.cli.main" ]; then
  if ! "$PY" -m strikecast.cli.main --help 2>/dev/null | grep -qw -- "${3:-}"; then
    echo "job.sbatch: optional command '${3:-}' is not available in this checkout; skipping"
    echo "STRIKECAST_EXIT 0"
    exit 0
  fi
fi

MAX_REQUEUES="${STRIKECAST_MAX_REQUEUES:-8}"
RESTARTS="${SLURM_RESTART_COUNT:-0}"
CHILD=""

forward_and_wait() {
  # $1: seconds to wait for the child after forwarding SIGTERM
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM "$CHILD" 2>/dev/null
    local waited=0
    while kill -0 "$CHILD" 2>/dev/null && [ "$waited" -lt "$1" ]; do
      sleep 1
      waited=$((waited + 1))
    done
  fi
}

on_usr1() {
  echo "[$(date -Is)] USR1: the wall clock is near; stopping the job"
  forward_and_wait 120
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$RESTARTS" -lt "$MAX_REQUEUES" ]; then
    if scontrol requeue "$SLURM_JOB_ID"; then
      echo "[$(date -Is)] requeued job $SLURM_JOB_ID (restart $((RESTARTS + 1))/$MAX_REQUEUES)"
      echo "STRIKECAST_EXIT requeued"
      exit 0
    fi
    echo "job.sbatch: scontrol requeue failed; rerun submit_all.py to resubmit the unfinished work" >&2
  else
    echo "job.sbatch: requeue limit ($MAX_REQUEUES) reached or not under SLURM" >&2
  fi
  echo "STRIKECAST_EXIT 99"
  exit 99
}

on_term() {
  echo "[$(date -Is)] SIGTERM: forwarding to the job"
  forward_and_wait 25
  echo "STRIKECAST_EXIT 99"
  exit 99
}

trap on_usr1 USR1
trap on_term TERM

echo "[$(date -Is)] job ${SLURM_JOB_ID:-local} (attempt $((RESTARTS + 1))) on $(hostname) [$ENV_NAME]: $*"
"$PY" "$@" &
CHILD=$!
# `wait` returns early (>128) when a trapped signal arrives; loop until the
# child is really gone so its own exit status is the job's.
while true; do
  wait "$CHILD"
  STATUS=$?
  if ! kill -0 "$CHILD" 2>/dev/null; then
    break
  fi
done

echo "[$(date -Is)] job ${SLURM_JOB_ID:-local} finished with status $STATUS"
echo "STRIKECAST_EXIT $STATUS"
exit "$STATUS"
