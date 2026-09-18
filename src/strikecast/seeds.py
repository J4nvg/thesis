"""Seeding and environment capture.

Two functions, both consumed by the run store (refactor plan sec. 5.3): every
stage directory holds a ``config.yaml`` and an ``env.json``, and the latter is
what :func:`record_env` produces.

Why the ``PYTHONHASHSEED`` check exists (flag F16)
--------------------------------------------------
``get_engineered_features`` and ``split_future_and_past_cov`` build their column
lists with ``list(set(...))``. Python randomises string hashing per interpreter
process, so the *order* of the covariate columns -- and therefore the column
order of LightGBM's design matrix, and therefore how it breaks ties between
equally good splits during feature selection -- differs between interpreter runs
unless ``PYTHONHASHSEED`` is pinned. Measured churn against the cached sets is
Jaccard 0.73 to 0.95, all of it at the rank-100 boundary.

``PYTHONHASHSEED`` cannot be set from inside a running interpreter: CPython
reads it before any user code executes. :func:`seed_everything` therefore
*warns* rather than sets, and :func:`record_env` records the value that was
actually in force, so a run whose feature selection is not reproducible says so
in its own ``env.json``. New runs pin ``PYTHONHASHSEED=0`` in the launcher; the
legacy runs never pinned it, which is why ``features/*_saved_sets.pkl`` is data
to load rather than output to reproduce.

What the seed covers (sec. 5.4)
-------------------------------
``random``, ``numpy``, ``torch`` (CPU and, when present, every CUDA device).
Per-model seeds (``random_state`` / ``random_seed`` / AutoGluon's
``random_seed`` / the SPE classifier's ``random_state`` / the OOF calibration
splits) are set from ``RunContext.seed`` by the model specs, not here, because
the legacy builders passed them as constructor kwargs.

Seeding does **not** make every model deterministic: GPU XGBoost, GPU CatBoost
and GPU RNN training are not bit-reproducible (flag F9). That is the reason the
publication reports seed confidence intervals rather than a single number.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["REPRODUCIBLE_HASH_SEED", "TRACKED_DISTRIBUTIONS", "record_env", "seed_everything"]

#: The value ``PYTHONHASHSEED`` must hold for feature selection to be
#: reproducible across processes (flag F16).
REPRODUCIBLE_HASH_SEED = "0"

#: Distributions whose versions go into ``env.json``. The first block is what
#: golden comparisons are pinned to (sec. 5.7); the second is everything that
#: can change a number.
TRACKED_DISTRIBUTIONS = (
    "darts",
    "torch",
    "lightning",
    "pytorch-lightning",
    "lightgbm",
    "xgboost",
    "catboost",
    "optuna",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "pydantic",
    "hydra-core",
    "omegaconf",
    "imbalanced-ensemble",
    "venn-abers",
    "wandb",
)


def seed_everything(seed: int) -> int:
    """Seed ``random``, ``numpy`` and ``torch``; warn if hashing is unpinned.

    Args:
        seed: the seed to set. This is ``SeedConfig.tuning_seed`` for the tuning
            stage and one of ``SeedConfig.eval_seeds`` for an evaluation run.

    Returns:
        ``seed``, so that call sites can write ``seed = seed_everything(seed)``.

    ``torch`` is imported lazily: seeding must work in a process that has not
    paid for the torch import (the data stage, the report stage), and the import
    costs seconds.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != REPRODUCIBLE_HASH_SEED:
        logger.warning(
            "PYTHONHASHSEED is %s, not %r. CPython reads it before any user code runs, so it "
            "cannot be set from here. Feature selection is order-dependent and therefore not "
            "reproducible across processes in this run (flag F16); the value is recorded in "
            "env.json. Launch with PYTHONHASHSEED=%s to pin it.",
            "unset" if hash_seed is None else repr(hash_seed),
            REPRODUCIBLE_HASH_SEED,
            REPRODUCIBLE_HASH_SEED,
        )

    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - torch is a hard dependency
        logger.warning("torch is not importable; only random and numpy were seeded")
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)
    return seed


def _version(distribution: str) -> str | None:
    """Installed version of ``distribution``, or ``None`` when absent."""
    try:
        return _dist_version(distribution)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """``git rev-parse HEAD`` of the repository this package lives in.

    ``None`` when git is unavailable, the package is installed outside a
    checkout, or the command fails for any other reason -- an environment
    record must never be the thing that fails a run.
    """
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes; ``None`` when unknown."""
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def record_env() -> dict:
    """The payload of a stage's ``env.json``.

    Everything that can move a number between two otherwise identical runs:
    interpreter, platform, library versions, the hash seed (flag F16), the CPU
    count (thread counts change GBDT reduction order) and the git commit.

    CUDA fields are ``None`` when torch is absent or CPU-only, which is itself
    worth recording: the diff branch's XGBoost and CatBoost run on GPU and are
    not bit-reproducible there (flag F9).
    """
    hash_seed = os.environ.get("PYTHONHASHSEED")
    env: dict = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "PYTHONHASHSEED": hash_seed,
        "pythonhashseed_pinned": hash_seed == REPRODUCIBLE_HASH_SEED,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "packages": {name: _version(name) for name in TRACKED_DISTRIBUTIONS},
        "cuda_available": None,
        "cuda_version": None,
        "cuda_devices": None,
    }

    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return env

    available = bool(torch.cuda.is_available())
    env["cuda_available"] = available
    env["cuda_version"] = torch.version.cuda
    env["cuda_devices"] = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if available
        else []
    )
    return env
