"""Golden level F for feature importance (audit C14): the pipeline vs ``results/gbdt``.

``results/gbdt/importance_all.csv`` is what Figs 20 and 23 of the thesis are
drawn from (``_regression_GBDT.ipynb`` cells 57-58). This module compares it
with what ``strikecast importance`` writes into a THESIS-MODE run store
(``legacy=count``: the legacy ``expdecay7`` filter and the thesis's cached
feature set, plus the thesis's tuned params). Nothing is trained here; the
runs are made once, out of band, and the tests skip when they are absent::

    python scripts/import_golden_params.py --store-root runs --experiment count
    PYTHONHASHSEED=0 strikecast importance experiment=count legacy=count \\
        tracking=noop --store-root runs            # the thesis's DEFAULT_JOBS

``STRIKECAST_LEGACY_STORE`` overrides the store root (default ``runs/``, the
root ``test_pipeline_equality.py`` reads).

Why level F and not E
---------------------
LightGBM trees are not portable across platforms (F124, see
``test_pipeline_equality.py``): the thesis fit on Colab, and a refit here grows
slightly different trees. Measured on an Apple-silicon laptop (2026-09-26):
max |d| <= 7.7e-4 (gain) and <= 1.05e-3 (permutation) of the column total for
``global_lightgbm_poisson`` and the three ``activity_<k>_lightgbm_poisson``
runs, Spearman >= 0.9957, top-15 overlap >= 13/15 (gain) and 15/15
(permutation); category shares within 0.06. The exact port check (port ==
cell 57 on one fitted model) is ``tests/unit/test_importance.py``.

CatBoost on CPU IS portable (audit B1: count CatBoost-Tweedie reproduces golden
fold 0 to 1.8e-15), and so are its importances: the three
``activity_<k>_catboost_tweedie`` runs -- the source of Figs 20 and 23 --
agree with the thesis to |d| ~ 1e-16 for tiers 2 and 3 and to 7e-5 of the
column maximum for tier 1, every horizon, gain and permutation (and
``global_catboost_tweedie`` equals ``importance_global_catboost_tweedie_2.csv``
to 1e-16). They get the
tight :func:`test_catboost_activity_reproduces_figs_20_and_23`, which also
recomputes the category-share percentages the thesis text quotes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STORED = REPO_ROOT / "results" / "gbdt" / "importance_all.csv"
STORE_ROOT = Path(os.environ.get("STRIKECAST_LEGACY_STORE", REPO_ROOT / "runs"))
SEED = 42

#: Level-F tolerances (see the module docstring for the measured values).
MAX_GAIN_DIFF_OF_TOTAL = 2e-3
MAX_PERM_DIFF_OF_TOTAL = 3e-3
MIN_SPEARMAN = 0.99
MIN_TOP15_OVERLAP = 12
MAX_SHARE_DIFF = 0.10

pytestmark = pytest.mark.golden

RUNS = [
    ("lightgbm_poisson", "global"),
    ("lightgbm_poisson", "activity"),
    ("catboost_tweedie", "global"),
    ("catboost_tweedie", "activity"),
]


def _run_frame(model: str, paradigm: str) -> pd.DataFrame:
    path = STORE_ROOT / "count" / model / paradigm / f"seed={SEED}" / "importance" / "importance.csv"
    if not path.exists():
        pytest.skip(f"no importance run at {path}; see this module's docstring")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def stored() -> pd.DataFrame:
    if not STORED.exists():
        pytest.skip(f"{STORED} not present")
    return pd.read_csv(STORED)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).correlation)


@pytest.mark.parametrize(("model", "paradigm"), RUNS, ids=lambda v: str(v))
def test_the_pipeline_reproduces_the_thesis_importances(
    stored: pd.DataFrame, model: str, paradigm: str
) -> None:
    from strikecast.evaluation.importance import category_shares, top_features

    got_all = _run_frame(model, paradigm)
    compared = 0
    for label in dict.fromkeys(got_all["model"]):
        expected = stored[stored["model"] == label]
        if expected.empty:  # e.g. global_catboost_tweedie is not in importance_all.csv
            continue
        compared += 1
        got = got_all[got_all["model"] == label].set_index("Feature")
        exp = expected.set_index("Feature")
        assert set(got.index) == set(exp.index), label
        exp = exp.loc[got.index]

        for col in ("agg_gain", "agg_perm"):
            if got[col].isna().all():
                continue  # a gain-only run
            diff = np.abs(got[col] - exp[col]).max()
            limit = MAX_GAIN_DIFF_OF_TOTAL if col == "agg_gain" else MAX_PERM_DIFF_OF_TOTAL
            assert diff <= limit * exp[col].abs().sum(), (label, col, diff)
            assert _spearman(got[col], exp[col]) >= MIN_SPEARMAN, (label, col)
            overlap = set(got[col].nlargest(15).index) & set(exp[col].nlargest(15).index)
            assert len(overlap) >= MIN_TOP15_OVERLAP, (label, col, len(overlap))
            shares = pd.concat(
                [
                    category_shares(top_features(got.reset_index(), col)),
                    category_shares(top_features(exp.reset_index(), col)),
                ],
                axis=1,
            ).fillna(0.0)
            assert (shares.iloc[:, 0] - shares.iloc[:, 1]).abs().max() <= MAX_SHARE_DIFF, label
    if compared == 0:
        pytest.skip(f"{model}/{paradigm}: no label of this run is in {STORED.name}")


#: CatBoost is portable (see the module docstring); tolerance relative to the
#: column maximum, measured worst case 6.8e-5 (tier 1, h7 gain).
CATBOOST_REL = 1e-3


def test_catboost_activity_reproduces_figs_20_and_23(stored: pd.DataFrame) -> None:
    from strikecast.evaluation.importance import (
        category_importance_matrix,
        importance_columns,
        top_features,
    )

    got_all = _run_frame("catboost_tweedie", "activity")
    importancedict = {}
    for tier in (1, 2, 3):
        label = f"activity_{tier}_catboost_tweedie"
        got = got_all[got_all["model"] == label].set_index("Feature")
        exp = stored[stored["model"] == label].set_index("Feature").loc[got.index]
        for col in importance_columns(7)[1:]:
            if got[col].isna().all():
                continue
            scale = exp[col].abs().max()
            assert np.abs(got[col] - exp[col]).max() <= CATBOOST_REL * scale, (label, col)
        importancedict[f"activity_{tier}"] = {
            "gain": top_features(got.reset_index(), "agg_gain"),
            "perm": top_features(got.reset_index(), "agg_perm"),
        }
    matrix = category_importance_matrix(importancedict)
    quoted = {  # main.tex 1117-1119
        ("Conflict & damage", "Tier 1 gain"): 62,
        ("Conflict & damage", "Tier 1 perm"): 48,
        ("Weather / geomag.", "Tier 1 perm"): 30,
        ("Comms / diplo / aid", "Tier 1 perm"): 15,
        ("Autoregressive strikes", "Tier 2 gain"): 45,
        ("Conflict & damage", "Tier 2 gain"): 32,
        ("Macroeconomic", "Tier 2 gain"): 22,
        ("Weather / geomag.", "Tier 2 perm"): 56,
        ("Autoregressive strikes", "Tier 2 perm"): 28,
        ("Conflict & damage", "Tier 2 perm"): 16,
        ("Autoregressive strikes", "Tier 3 gain"): 82,
        ("Autoregressive strikes", "Tier 3 perm"): 66,
        ("Spatial / static", "Tier 3 gain"): 18,
        ("Spatial / static", "Tier 3 perm"): 29,
    }
    for (category, column), percent in quoted.items():
        assert round(100 * matrix.loc[category, column]) == percent, (category, column)


def test_global_catboost_matches_its_own_thesis_file() -> None:
    """``importance_global_catboost_tweedie_2.csv``: cell 57 as saved (global CatBoost-T)."""
    path = REPO_ROOT / "results" / "gbdt" / "importance_global_catboost_tweedie_2.csv"
    if not path.exists():
        pytest.skip(f"{path} not present")
    got_all = _run_frame("catboost_tweedie", "global")
    got = got_all[got_all["model"] == "global_catboost_tweedie"].set_index("Feature")
    exp = pd.read_csv(path).set_index("Feature")
    assert set(got.index) == set(exp.index)
    exp = exp.loc[got.index]
    for col in exp.columns:
        if got[col].isna().all():
            continue
        assert np.abs(got[col] - exp[col]).max() <= CATBOOST_REL * exp[col].abs().max(), col
