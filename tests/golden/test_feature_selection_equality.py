"""Golden level G: the ported feature selection must reproduce the cached sets.

The legacy selections were cached as ``features/<family>_saved_sets.pkl`` and
converted to ``golden/converted/feature_sets/<family>.json``.  This module
re-runs the selection through ``strikecast.data.feature_selection`` on the real
data and compares the surviving past/future covariate components.

Both families run the *same* data pipeline; they differ only in the LightGBM
that ranks the features (see ``feature_selection.countreg_config`` /
``diffreg_config``).  Neither reproduces the cached set exactly, and
neither can: the selection is PYTHONHASHSEED-dependent (see
``_NONDETERMINISM_REASON``).  Until 2026-09-26 the equality tests were
``xfail(strict=False)`` and could never fail (audit A, section 3). They now
assert what IS reproducible: the disagreement is confined to near-ties at the
top-100 cut (every golden-only component sits just below our cut) and the
overlap stays above a floor measured under ``PYTHONHASHSEED=0``. A selection on
the wrong features, the wrong target or the wrong objective fails both.

Measured 2026-09-26 (``PYTHONHASHSEED=0``, CPU, 4 threads):

========  =================  ===================  ===========================
family    past (ours/golden)  Jaccard past/future  golden-only past, our ranks
========  =================  ===================  ===========================
diffreg   50 / 52            0.729 / 0.778        100 ... 168 (9 components)
countreg  35 / 37            0.946 / 0.800        107, 141
========  =================  ===================  ===========================

Marked ``slow``: one fit is 7 LightGBM boosters over ~1.4k lagged features.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

pytestmark = [pytest.mark.golden, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
GOLDEN_DIR = REPO_ROOT / "golden" / "converted" / "feature_sets"

# The legacy pipeline constants, copied from `_diff_regression.py` /
# `_regression_GBDT.ipynb` (both families use identical values).
TARGET = "act_drone_strike_on_ua"
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
EWM_HALFLIFE = 7
RANDOM_STATE = 42
# `_diff_regression.py` hard-codes `available_threads = 4`.
LEGACY_THREADS = 4


def _require_inputs():
    if not (DATA_DIR / "fixed").is_dir() or not (DATA_DIR / "dataset").is_dir():
        pytest.skip("real data/ directory not present")
    if not GOLDEN_DIR.is_dir():
        pytest.skip("golden/converted/feature_sets not present")


@pytest.fixture(scope="module")
def legacy_inputs():
    """Rebuild the un-selected pipeline inputs with the legacy `src` package.

    Reproduces `_diff_regression.py` cells 8-15 / `_regression_GBDT.ipynb`
    cells 12-22 exactly; the two are identical up to the feature-selection
    step.
    """
    _require_inputs()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    legacy = pytest.importorskip("src", reason="legacy `src` package not present")

    import numpy as np

    np.random.seed(RANDOM_STATE)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    fixed = legacy.construct_path(str(DATA_DIR), "fixed")
    dataset = legacy.construct_path(str(DATA_DIR), "dataset")

    regions, master_timeseries, regions_activity = legacy.load_data(
        data_path=fixed, dataset_path=dataset
    )
    for_global_reset, global_weather_columns = legacy.get_engineered_features(
        master_timeseries=master_timeseries,
        data_path=fixed,
        target_col=TARGET,
        regions=regions,
        regions_activity=regions_activity,
        binarize_target=False,
    )
    _, future_covariates, _, past_covariates = legacy.split_future_and_past_cov(
        for_global_reset, global_weather_columns, TARGET
    )
    target_series_list, past_covs_list, future_covs_list = (
        legacy.build_ts_and_apply_window_transformer(
            for_global_reset,
            TARGET,
            past_covariates,
            future_covariates,
            ed_alpha=legacy.halflife_to_alpha(EWM_HALFLIFE),
        )
    )
    (_, train_target, _, _, full_past_covs, full_fut_covs, _, _, _) = (
        legacy.get_covs_and_encodings(
            target_series_list, past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC
        )
    )
    return train_target, full_past_covs, full_fut_covs


@pytest.fixture(scope="module")
def selections(legacy_inputs):
    """Run both family selections once; the fits are the expensive part."""
    from strikecast.data.feature_selection import (
        countreg_config,
        diffreg_config,
        select_top_k,
    )

    train_target, full_past_covs, full_fut_covs = legacy_inputs
    out = {}
    for name, config in (
        ("diffreg", diffreg_config(device_type="cpu", num_threads=LEGACY_THREADS)),
        # legacy ran this one on the GPU; forced to CPU here (see Q9)
        ("countreg", countreg_config(device_type="cpu", num_threads=LEGACY_THREADS)),
    ):
        out[name] = select_top_k(train_target, full_past_covs, full_fut_covs, config)
    return out


def _golden(family: str) -> dict:
    payload = json.loads((GOLDEN_DIR / f"{family}.json").read_text(encoding="utf-8"))
    return {
        "past": set(payload["past_covariate_components"]),
        "future": set(payload["future_covariate_components"]),
    }


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _missed_ranks(selection, missing: set) -> list[int]:
    """Where the golden-only components sit in *our* gain ranking.

    The cached golden sets store component *sets*, not the ranking that
    produced them, so a Spearman correlation against the golden top-100 is not
    computable.  The informative substitute is how far outside our own top-k
    each golden-only component fell: rank 100-150 means a near-tie, rank 1500
    means a genuine disagreement.
    """
    if not missing:
        return []
    ranks = []
    features = selection.gain_table["Feature"].to_list()
    for component in missing:
        hits = [i for i, f in enumerate(features) if f.startswith(component + "_")]
        if hits:
            ranks.append(min(hits))
    return sorted(ranks)


def _report(family: str, selection, golden: dict) -> str:
    return (
        f"{family}: past {len(selection.past_keep)} vs {len(golden['past'])} golden, "
        f"Jaccard={_jaccard(set(selection.past_keep), golden['past']):.3f}; "
        f"future {len(selection.future_keep)} vs {len(golden['future'])} golden, "
        f"Jaccard={_jaccard(set(selection.future_keep), golden['future']):.3f}; "
        f"golden-only past components sit at our gain ranks "
        f"{_missed_ranks(selection, golden['past'] - set(selection.past_keep))} "
        f"(top_k={selection.config.top_k}; a Spearman rank correlation against "
        f"the cache is not computable -- it stores sets, not ranks)"
    )


_NONDETERMINISM_REASON = (
    "The cached feature sets cannot be reproduced by any re-run, on this or any "
    "other machine. `get_engineered_features` and `split_future_and_past_cov` "
    "build their column lists with `list(set(...))`, so the ORDER of the past / "
    "future covariate columns depends on PYTHONHASHSEED. That order becomes the "
    "column order of LightGBM's design matrix, which decides how it breaks ties "
    "between equally-good splits, which perturbs the gains, and a hard top-100 "
    "cut over ~2.1k lagged features then swaps whole features in and out. "
    "Verified here: two runs of the identical script in separate processes "
    "disagree (max |delta mean_gain| ~48, 54 vs 55 past covariates), while two "
    "runs under PYTHONHASHSEED=0 are bit-identical. The legacy runs did not pin "
    "PYTHONHASHSEED, so the seed behind the cache is unknowable. The data "
    "pipeline feeding this step does reproduce the golden parquet inputs "
    "bit-for-bit, so the drift is confined to the ranking. `countreg` carries a "
    "second, independent source: it was fitted with `device_type=\"gpu\"`, whose "
    "histogram builder differs from the CPU one. Overlap is therefore reported "
    "rather than the assertion being loosened."
)


#: A golden-only component must rank within our top NEAR_TIE_RANK lagged features:
#: it was cut by a near-tie, not by a different model (observed max: 168).
NEAR_TIE_RANK = 250

#: ``family -> (min past Jaccard, min future Jaccard, max |size difference|)``,
#: the measured values (module docstring) minus a margin for hash-seed churn.
OVERLAP_FLOORS = {
    "diffreg": (0.65, 0.6, 5),
    "countreg": (0.85, 0.6, 5),
}


def _assert_agrees_up_to_near_ties(family: str, sel) -> None:
    golden = _golden(family)
    report = _report(family, sel, golden) + f" -- {_NONDETERMINISM_REASON[:80]}..."
    min_past, min_future, max_size_gap = OVERLAP_FLOORS[family]
    assert _jaccard(set(sel.past_keep), golden["past"]) >= min_past, report
    assert _jaccard(set(sel.future_keep), golden["future"]) >= min_future, report
    assert abs(len(sel.past_keep) - len(golden["past"])) <= max_size_gap, report
    missing = golden["past"] - set(sel.past_keep)
    ranks = _missed_ranks(sel, missing)
    # every golden-only component is a real component we ranked, near the cut
    assert len(ranks) == len(missing), report
    assert all(rank < NEAR_TIE_RANK for rank in ranks), report
    missing_future = golden["future"] - set(sel.future_keep)
    assert missing_future <= set(sel.available_future), report


def test_diffreg_selection_agrees_with_golden_up_to_near_ties(selections):
    """The diff family was fitted on CPU (`device_type="cpu"`, 4 threads).

    Exact equality is impossible (``_NONDETERMINISM_REASON``); what is asserted
    is that every disagreement is a near-tie at the top-100 cut."""
    _assert_agrees_up_to_near_ties("diffreg", selections["diffreg"])


def test_countreg_selection_agrees_with_golden_up_to_near_ties(selections):
    """The count family was fitted with `device_type="gpu"` (quirk Q9); CPU here."""
    _assert_agrees_up_to_near_ties("countreg", selections["countreg"])


@pytest.mark.parametrize("family", ["diffreg", "countreg"])
def test_overlap_is_reported(selections, capsys, family):
    """Always print the agreement numbers, whether or not the sets match."""
    sel = selections[family]
    golden = _golden(family)
    with capsys.disabled():
        print("\n" + _report(family, sel, golden))
    assert sel.past_keep, "selection produced no past covariates at all"
    assert sel.future_keep, "selection produced no future covariates at all"


@pytest.mark.parametrize("family", ["diffreg", "countreg"])
def test_selection_is_a_plausible_superset_of_the_pipeline(selections, family):
    """Sanity floor: whatever we select must be real covariate components."""
    sel = selections[family]
    assert set(sel.past_keep) <= set(sel.available_past)
    assert set(sel.future_keep) <= set(sel.available_future)
    assert len(sel.gain_table) == len(sel.gain_table["Feature"].unique())
    assert list(sel.gain_table.columns[:1]) == ["Feature"]
    assert {"mean_gain", "agg_gain"} <= set(sel.gain_table.columns)
    assert sel.gain_table["mean_gain"].is_monotonic_decreasing


def test_selection_roundtrips_through_json(selections, tmp_path):
    from strikecast.data.feature_selection import FeatureSelection

    sel = selections["diffreg"]
    path = sel.to_json(tmp_path / "diffreg.json")
    back = FeatureSelection.from_json(path)
    assert back.past_keep == sel.past_keep
    assert back.future_keep == sel.future_keep
    assert back.top_features == sel.top_features
    assert back.config == sel.config
