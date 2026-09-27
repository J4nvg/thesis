"""Shared fixtures for the strikecast test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_DIR = REPO_ROOT / "data" / "fixed"
DATASET_DIR = REPO_ROOT / "data" / "dataset"
MASTER_PARQUET = DATASET_DIR / "master_combined_timeseries.parquet"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "golden: behaviour-preservation comparison against the legacy `src` package",
    )
    config.addinivalue_line(
        "markers",
        "slow: minutes, not seconds -- fits a model or runs a real backtest. "
        "Deselect with `-m 'not slow'` (what CI does).",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def inputs():
    """Real inputs from ``data/fixed`` and ``data/dataset``; skips if absent."""
    if not (FIXED_DIR / "regions.txt").exists() or not MASTER_PARQUET.exists():
        pytest.skip(f"real data not available under {FIXED_DIR} / {DATASET_DIR}")
    from strikecast.data import load_inputs

    return load_inputs(FIXED_DIR, DATASET_DIR)


@pytest.fixture(scope="session")
def legacy_src():
    """The legacy top-level ``src`` package, importable from the repo root."""
    if not (REPO_ROOT / "src" / "prevalent_functions.py").exists():
        pytest.skip("legacy `src` package not present")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import src  # noqa: PLC0415

    return src


@pytest.fixture(scope="session")
def fixed_dir() -> Path:
    return FIXED_DIR
