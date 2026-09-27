#!/bin/bash
# Build an environment ONCE, on a compute node (audit 2026-09-26 C24/C25/D2-4).
#
#     sbatch [resources] scripts/slurm/setup.sbatch main        # .venv (+ smoke test)
#     sbatch [resources] scripts/slurm/setup.sbatch autogluon   # envs/autogluon/.venv (+ Chronos-2 weights)
#
# submit_all.py submits both (`--setup-only` submits nothing else); the login
# node never builds or computes anything. The cluster has no environment
# modules and no uv, but compute nodes reach the internet, so:
#
# 1. uv is installed into ~/.local/bin with the official installer if missing
#    (shared /home, so once per user);
# 2. `uv sync --frozen` builds the env from the committed lock -- exactly once,
#    here; array/job tasks never sync (C24);
# 3. scripts/slurm/smoke_test.py checks CUDA with a tiny op, imports and fits
#    LightGBM/XGBoost/CatBoost (XGBoost/CatBoost also on the GPU, as the diff
#    family runs them), and for `autogluon` downloads the Chronos-2 weights into
#    $HF_HOME with a tiny zero-shot fit; it writes the stamp
#    <venv>/.strikecast-setup.json (lock hash) that submit_all.py checks.
#
# The job fails loudly (and every dependent job is cancelled) if any step fails.
#
#SBATCH --job-name=strikecast-setup
#SBATCH --output=logs/slurm/setup-%j.out

set -euo pipefail

WHAT="${1:-main}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
REPO="$PWD"

export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# uv's managed CPython (the core env needs 3.13) lives on the shared /home too.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"

echo "[$(date -Is)] setup '$WHAT' on $(hostname), repo $REPO"
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv into ~/.local/bin (official installer)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
ldd --version | head -1 || true

case "$WHAT" in
  main)
    EXTRAS=(--extra gpu)
    # The figures stream adds a `figures` extra (geopandas, ...); include it when present.
    if grep -qE '^figures[[:space:]]*=' pyproject.toml; then EXTRAS+=(--extra figures); fi
    uv sync --frozen "${EXTRAS[@]}"
    .venv/bin/python scripts/slurm/smoke_test.py main
    ;;
  autogluon)
    uv sync --frozen --project envs/autogluon
    PYTHONPATH="$REPO/src" envs/autogluon/.venv/bin/python scripts/slurm/smoke_test.py autogluon
    ;;
  *)
    echo "setup.sbatch: unknown target '$WHAT' (main|autogluon)" >&2
    exit 2
    ;;
esac
echo "[$(date -Is)] setup '$WHAT' OK"
echo "STRIKECAST_EXIT 0"
