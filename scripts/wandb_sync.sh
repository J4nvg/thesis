#!/usr/bin/env bash
# Sync offline W&B runs from cluster scratch to wandb.ai.
#
# Plan §5.5: runs are ONLINE by default because the cluster nodes reach the
# internet. `tracking.mode=offline` is the fallback for a node or a queue that
# does not, and this script is the second half of it -- run it from the laptop
# after pulling scratch down, or from a SLURM epilogue on a node that has
# network.
#
# The local run store under runs/ is the source of truth; nothing here touches
# it. Syncing only replays event files W&B already buffered, so it is safe to
# re-run, and a failed directory is reported and skipped rather than aborting
# the batch.
#
# Usage:
#   scripts/wandb_sync.sh [-d DIR] [-p PROJECT] [-e ENTITY] [-n] [-f] [-c]
#
#   -d DIR      directory holding offline-run-* dirs
#               (default: $WANDB_DIR/wandb, else ./wandb)
#   -p PROJECT  W&B project      (default: $WANDB_PROJECT, else strikecast)
#   -e ENTITY   W&B entity       (default: $WANDB_ENTITY, else the wandb default)
#   -n          dry run: list what would be synced, sync nothing
#   -f          re-sync directories already marked as synced
#   -c          delete each directory after it syncs successfully
#   -h          this help
#
# Exit status is the number of directories that failed to sync (capped at 125).

set -euo pipefail

dir="${WANDB_DIR:-.}/wandb"
project="${WANDB_PROJECT:-strikecast}"
entity="${WANDB_ENTITY:-}"
dry_run=0
force=0
clean=0

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }

while getopts ":d:p:e:nfch" opt; do
  case "$opt" in
    d) dir="$OPTARG" ;;
    p) project="$OPTARG" ;;
    e) entity="$OPTARG" ;;
    n) dry_run=1 ;;
    f) force=1 ;;
    c) clean=1 ;;
    h) usage; exit 0 ;;
    :) echo "wandb_sync: -$OPTARG needs an argument" >&2; exit 2 ;;
    \?) echo "wandb_sync: unknown option -$OPTARG" >&2; exit 2 ;;
  esac
done

if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb_sync: the wandb CLI is not on PATH (try: uv run wandb --version)" >&2
  exit 2
fi

if [ ! -d "$dir" ]; then
  echo "wandb_sync: no such directory: $dir" >&2
  exit 2
fi

# Offline runs are `offline-run-<timestamp>-<id>`; a crashed online run leaves
# `run-<timestamp>-<id>` behind, which `wandb sync` can replay just as well.
candidates=()
while IFS= read -r run_dir; do
  candidates+=("$run_dir")
done < <(find "$dir" -maxdepth 1 -mindepth 1 -type d \
  \( -name 'offline-run-*' -o -name 'run-*' \) | sort)

if [ "${#candidates[@]}" -eq 0 ]; then
  echo "wandb_sync: nothing to sync in $dir"
  exit 0
fi

sync_args=(--project "$project")
if [ -n "$entity" ]; then
  sync_args+=(--entity "$entity")
fi

synced=0
skipped=0
failed=0

for run_dir in "${candidates[@]}"; do
  # `wandb sync` drops a `.synced` marker in the run directory when it finishes.
  if [ "$force" -eq 0 ] && [ -e "$run_dir/.synced" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  if [ "$dry_run" -eq 1 ]; then
    echo "would sync: $run_dir"
    synced=$((synced + 1))
    continue
  fi
  echo "syncing: $run_dir"
  if wandb sync "${sync_args[@]}" "$run_dir"; then
    synced=$((synced + 1))
    if [ "$clean" -eq 1 ]; then
      rm -rf -- "$run_dir"
      echo "removed: $run_dir"
    fi
  else
    echo "wandb_sync: FAILED $run_dir" >&2
    failed=$((failed + 1))
  fi
done

echo "wandb_sync: $synced synced, $skipped already synced, $failed failed (dir=$dir, project=$project)"
if [ "$failed" -gt 125 ]; then
  exit 125
fi
exit "$failed"
