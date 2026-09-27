"""Level-I golden test: recompute every stored metric view from its predictions.

For every family directory under ``golden/results/`` that stores a
``predictions_long_<key>.parquet`` next to a ``global_<key>.json`` and the
per-view CSVs, this module reloads the predictions, re-runs
:func:`strikecast.evaluation.aggregate.evaluate` with the scales the legacy run
used, and compares value for value against the stored files. Nothing is
re-trained; the predictions are the frozen input.

Families and how each one is wired
----------------------------------

``gbdt`` / ``lstm`` / ``diff``
    ``metric_set="count"``, train-only scales, activity map supplied. All three
    call ``compute_naive_scales(train_target, region_names, 7)`` on the LEVEL
    target (``_regression_GBDT.py:719``, ``_regression_LSTM.py:723``,
    ``_diff_regression.py:603`` -- the diff family scales on the level series
    too, because ``train_target`` there comes from the level
    ``get_covs_and_encodings`` call at ``:523``, not from the differenced one),
    so ONE scale set serves all three families and both stages (plan F6).
    Six views each.

``finalhurdle``
    Five components per stage, from ``final_hurdle.ipynb`` cells 48-49. The
    classifier components use ``metric_set="classification"`` on the ``y_prob``
    frames; ``regressor`` uses ``metric_set="hurdle"`` on the count-head frame
    **filtered to ``y_true > 0``** (plan F65: the same rows are also scored
    unfiltered into ``regressor_per_horizon.csv``, which is a different file
    and a different number); ``hurdle_raw`` / ``hurdle_cal`` use
    ``metric_set="hurdle"`` on the product frames. Scales are train-only for
    the CV stage and **train+val** for the test stage (plan F67). No activity
    map: neither hurdle aggregator had that parameter, so the family stores
    four views, not six.

``chronos2``
    NOT covered here. Its stored views come from
    ``_chronos2.py::evaluate_long``, which is a different function with
    different columns and no activity views; it is ported separately as
    :func:`strikecast.evaluation.metrics.chronos_evaluate_long` and its
    differences are listed in
    :data:`strikecast.evaluation.metrics.CHRONOS_DIFFERENCES`.

Comparison rules (as specified in the task)
-------------------------------------------

* ``global_<key>.json`` with ``math.isclose(rel_tol=1e-9)``, NaN-tolerant.
* CSV views with ``assert_frame_equal(check_exact=False, rtol=1e-9)``.
* One exception, :data:`ORDER_INSENSITIVE_VIEWS`: the ``classification``
  metric set's ``per_region`` view is aligned on ``region`` first, because its
  stored row order is an unstable-sort artefact of the pandas build that wrote
  it (plan flag F123). Both sides must still carry the same regions, once
  each, and every value is compared at the same 1e-9.
* Dtypes: the CSVs are written with ``to_csv(index=False)`` and read back with
  a plain ``read_csv``. On the recorded corpus the round trip is LOSSLESS for
  every column of every file -- ``n``/``TP``/``FP``/``TN``/``FN`` and
  ``activity_level`` come back ``int64`` and everything else ``float64`` /
  ``str``, exactly the dtypes ``evaluate`` produces -- so the comparison is run
  with ``check_dtype=True``. :func:`_assert_view_equal` still records any
  coercion it has to make in :data:`DTYPE_COERCIONS` and re-checks the values
  afterwards, so a future file whose round trip IS lossy fails loudly on the
  values while telling us which column widened. The two round trips that would
  be lossy if they ever occurred are documented there.

Cost: ``-m "not slow"`` runs one representative per family (4 cases). The slow
set runs all 158 count keys and all 10 hurdle components.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.golden

pytest.importorskip("darts")

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "golden" / "results"

TARGET = "act_drone_strike_on_ua"
SEASONALITY = 7

COUNT_FAMILIES = ("gbdt", "lstm", "diff")
COUNT_VIEWS = (
    "per_region",
    "per_horizon",
    "per_region_horizon",
    "per_activity_level",
    "per_activity_horizon",
)
HURDLE_VIEWS = ("per_region", "per_horizon", "per_region_horizon")

#: One cheap, representative key per family for the ``not slow`` run.
FAST_COUNT_KEYS = {
    "gbdt": "cv_global_lightgbm_poisson_tuned",
    "lstm": "cv_global_lstm_poisson_w7_tuned",
    "diff": "cv_global_lightgbm_tuned",
}
FAST_HURDLE_CASE = ("cv", "hurdle_raw")

#: Views the recorded corpus simply does not contain, with the reason.
#:
#: ``results/diff`` (and therefore ``golden/results/diff``) is missing exactly
#: three ``*_global_naive_weekly`` views. Exactly the same three views exist
#: under a ``*_local_naive_weekly`` name instead, with byte-equal content --
#: ``test_naive_weekly_local_files_hold_the_global_views`` below proves it --
#: and the ``*_global_naive_weekly`` names were never committed
#: (``git log --diff-filter=A`` finds only the ``local`` ones). The predictions
#: and the other three views are present and DO match, so this is a corpus
#: completeness gap, not a metric disagreement. See report flag F122.
MISSING_GOLDEN_VIEWS: dict[tuple[str, str], tuple[str, ...]] = {
    ("diff", "cv_global_naive_weekly"): ("per_activity_level",),
    ("diff", "test_global_naive_weekly"): ("per_region", "per_horizon"),
}

#: ``(metric_set, view) -> key column`` for the ONE view whose stored row order
#: cannot be reproduced and is not a number (plan flag **F123**).
#:
#: ``evaluate``'s ``classification`` aggregator does what
#: ``final_hurdle.ipynb`` cell 14 does: ``long_df.groupby("region")`` and then
#: ``.sort_values("F1", ascending=False)``. Ten of the twenty regions tie at
#: ``F1 == 0`` (they have no predicted positives at threshold 0.5, F70), and
#: ``sort_values`` defaults to quicksort, which is **not stable**, so which of
#: the tied regions comes out first depends on the pandas build. The pandas
#: that wrote ``golden/results/finalhurdle`` and pandas 3.0.2 here disagree.
#: Aligned on ``region`` every metric of every one of those rows is equal to
#: 1e-9, so the fix is to compare this view order-insensitively rather than to
#: add a tie-break the thesis never had (Jan's decision, 2026-09-18). Every
#: other view of every other family stays strictly ordered.
ORDER_INSENSITIVE_VIEWS: dict[tuple[str, str], str] = {
    ("classification", "per_region"): "region",
}

#: Filled by :func:`_assert_view_equal` whenever a CSV column comes back with a
#: different dtype from the recomputed one. Expected to stay empty on the
#: recorded corpus; the two coercions that *could* happen are an all-NaN
#: integer column (``n`` would read back as ``float64``) and an
#: ``activity_level`` that contains a NaN tier (``int64`` -> ``float64``,
#: plan F24 / aggregate Q4).
DTYPE_COERCIONS: dict[tuple[str, str], tuple[str, str]] = {}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _count_keys(family: str) -> list[str]:
    """Every ``<key>`` with both a predictions parquet and a global JSON."""
    directory = GOLDEN / family
    if not directory.is_dir():
        return []
    keys = []
    for path in sorted(directory.glob("predictions_long_*.parquet")):
        key = path.name[len("predictions_long_") : -len(".parquet")]
        if (directory / f"global_{key}.json").exists():
            keys.append(key)
    return keys


COUNT_CASES = [(family, key) for family in COUNT_FAMILIES for key in _count_keys(family)]

#: ``(stage, component) -> (predictions file, metric_set, positive_only)``.
#: From ``final_hurdle.ipynb`` cells 48 (CV) and 49 (test).
HURDLE_COMPONENTS = {
    "classifier_raw": ("classifier_probs", "classification", False),
    "classifier_cal": ("classifier_cal_probs", "classification", False),
    "regressor": ("regressor_preds", "hurdle", True),
    "hurdle_raw": ("hurdle_preds", "hurdle", False),
    "hurdle_cal": ("hurdle_cal_preds", "hurdle", False),
}
HURDLE_CASES = [(stage, component) for stage in ("cv", "test") for component in HURDLE_COMPONENTS]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle(inputs):
    """The real count-family `SeriesBundle`; the hurdle regressor head shares it.

    ``final_hurdle.ipynb`` cell 3 builds its regressor panel with
    ``get_engineered_features(..., binarize_target=False)`` on the same target
    and then runs the same ``split_future_and_past_cov`` /
    ``build_ts_and_apply_window_transformer`` / ``get_covs_and_encodings``
    chain as the count notebooks, so ``train_target_r`` / ``val_target_r`` are
    this bundle's ``target_train`` / ``target_val``. (The classifier head's
    binarised panel is NOT needed: the stored classifier parquets already carry
    the binary ``y_true``, and the classification metric set uses no scales.)
    """
    from strikecast.config.schema import SeriesConfig, WindowTransformConfig  # noqa: PLC0415
    from strikecast.data import (  # noqa: PLC0415
        build_bundle,
        build_panel_legacy_regressor,
        split_covariates,
    )

    panel_result = build_panel_legacy_regressor(inputs, TARGET)
    split = split_covariates(panel_result.panel, panel_result.global_weather_columns, TARGET)
    return build_bundle(
        panel=panel_result.panel,
        target=TARGET,
        past_covariates=split.past_covariates,
        future_covariates=split.future_covariates,
        config=SeriesConfig(window=WindowTransformConfig(expdecay="legacy_alpha")),  # thesis: legacy expdecay7 (A1/D1)
        activity_by_region=inputs.activity_by_region,
    )


@pytest.fixture(scope="module")
def train_scales(bundle):
    """``compute_naive_scales(train_target, region_names, 7)`` -- every count family."""
    from strikecast.evaluation.metrics import compute_naive_scales  # noqa: PLC0415

    return compute_naive_scales(bundle.target_train, bundle.region_names, SEASONALITY)


@pytest.fixture(scope="module")
def train_val_scales(bundle):
    """The hurdle's test-stage scales (plan F67): train+val, not train."""
    from strikecast.evaluation.metrics import compute_naive_scales  # noqa: PLC0415

    appended = [
        tr.append(vl) for tr, vl in zip(bundle.target_train, bundle.target_val, strict=True)
    ]
    return compute_naive_scales(appended, bundle.region_names, SEASONALITY)


@pytest.fixture(scope="module")
def activity_map(inputs):
    return inputs.activity_by_region


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _assert_global_equal(got: dict, path: Path) -> None:
    stored = json.loads(path.read_text())
    assert set(stored) == set(got), (
        f"{path.name}: key mismatch, stored-only={set(stored) - set(got)}, "
        f"computed-only={set(got) - set(stored)}"
    )
    for key, want in stored.items():
        have = got[key]
        if isinstance(want, float) and math.isnan(want):
            assert isinstance(have, float) and math.isnan(have), f"{path.name}: {key} {have} != NaN"
            continue
        assert math.isclose(want, have, rel_tol=1e-9), (
            f"{path.name}: {key} stored={want!r} recomputed={have!r} "
            f"(rel diff {abs(want - have) / abs(want) if want else float('inf'):.3e})"
        )


def _assert_view_equal(got: pd.DataFrame, path: Path, *, align_on: str | None = None) -> None:
    want = pd.read_csv(path)
    assert list(want.columns) == list(got.columns), (
        f"{path.name}: column mismatch\n  stored:     {list(want.columns)}\n"
        f"  recomputed: {list(got.columns)}"
    )
    for column in got.columns:
        if str(want[column].dtype) != str(got[column].dtype):
            DTYPE_COERCIONS[path.name, column] = (
                str(got[column].dtype),
                str(want[column].dtype),
            )
            want[column] = want[column].astype(got[column].dtype)
    if align_on is not None:
        # F123: the row ORDER of this view is an unstable-sort artefact, not a
        # number. Both sides must still hold the same keys, exactly once each.
        assert sorted(got[align_on]) == sorted(want[align_on]), (
            f"{path.name}: {align_on} keys differ\n"
            f"  stored-only:     {sorted(set(want[align_on]) - set(got[align_on]))}\n"
            f"  recomputed-only: {sorted(set(got[align_on]) - set(want[align_on]))}"
        )
        got = got.sort_values(align_on, kind="stable").reset_index(drop=True)
        want = want.sort_values(align_on, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9, obj=path.name)


def _views_for(family: str, key: str, views: tuple[str, ...]) -> tuple[str, ...]:
    """The views this key is expected to have, minus the documented gaps."""
    missing = MISSING_GOLDEN_VIEWS.get((family, key), ())
    directory = GOLDEN / family
    absent = tuple(v for v in views if not (directory / f"{v}_{key}.csv").exists())
    assert absent == missing, (
        f"{family}/{key}: golden views absent from disk = {absent}, "
        f"documented as absent = {missing}. Update MISSING_GOLDEN_VIEWS."
    )
    return tuple(v for v in views if v not in missing)


# ---------------------------------------------------------------------------
# Count families: gbdt, lstm, diff
# ---------------------------------------------------------------------------


def _check_count_key(family: str, key: str, train_scales, activity_map) -> None:
    from strikecast.evaluation.aggregate import evaluate  # noqa: PLC0415

    directory = GOLDEN / family
    mae_scales, rmse_scales = train_scales
    long_df = pd.read_parquet(directory / f"predictions_long_{key}.parquet")
    result = evaluate(long_df, mae_scales, rmse_scales, activity_map, metric_set="count")

    _assert_global_equal(result["global"], directory / f"global_{key}.json")
    for view in _views_for(family, key, COUNT_VIEWS):
        _assert_view_equal(result[view], directory / f"{view}_{key}.csv")


@pytest.mark.parametrize("family", COUNT_FAMILIES)
def test_count_family_fast(family, train_scales, activity_map) -> None:
    """One representative key per count family; runs in the default suite."""
    key = FAST_COUNT_KEYS[family]
    if not (GOLDEN / family / f"predictions_long_{key}.parquet").exists():
        pytest.skip(f"golden predictions missing: {family}/{key}")
    _check_count_key(family, key, train_scales, activity_map)


@pytest.mark.slow
@pytest.mark.parametrize(("family", "key"), COUNT_CASES, ids=lambda v: v)
def test_count_family_all(family, key, train_scales, activity_map) -> None:
    """Every stored count key: 36 gbdt + 60 lstm + 62 diff."""
    _check_count_key(family, key, train_scales, activity_map)


def test_count_cases_were_discovered() -> None:
    """Guards against a silently empty parametrisation."""
    if not GOLDEN.is_dir():
        pytest.skip(f"golden corpus not available at {GOLDEN}")
    per_family = {f: len(_count_keys(f)) for f in COUNT_FAMILIES}
    assert per_family == {"gbdt": 36, "lstm": 60, "diff": 62}, per_family


# ---------------------------------------------------------------------------
# finalhurdle
# ---------------------------------------------------------------------------


def _check_hurdle_case(stage: str, component: str, train_scales, train_val_scales) -> None:
    from strikecast.evaluation.aggregate import evaluate  # noqa: PLC0415

    directory = GOLDEN / "finalhurdle"
    source, metric_set, positive_only = HURDLE_COMPONENTS[component]
    predictions = directory / f"global_{stage}_{source}.parquet"
    if not predictions.exists():
        pytest.skip(f"golden predictions missing: {predictions.name}")

    long_df = pd.read_parquet(predictions)
    if positive_only:
        # F65: the count head is scored on positive-event days only in this
        # file set. `.reset_index(drop=True)` is cell 48/49's own call.
        long_df = long_df[long_df["y_true"] > 0].copy().reset_index(drop=True)

    # F67: train-only scales for CV, train+val for test.
    mae_scales, rmse_scales = train_scales if stage == "cv" else train_val_scales
    result = evaluate(long_df, mae_scales, rmse_scales, None, metric_set=metric_set)

    key = f"{stage}_global_{component}"
    _assert_global_equal(result["global"], directory / f"global_{key}.json")
    for view in _views_for("finalhurdle", key, HURDLE_VIEWS):
        _assert_view_equal(
            result[view],
            directory / f"{view}_{key}.csv",
            align_on=ORDER_INSENSITIVE_VIEWS.get((metric_set, view)),
        )


def test_hurdle_fast(train_scales, train_val_scales) -> None:
    """One representative hurdle component; runs in the default suite."""
    _check_hurdle_case(*FAST_HURDLE_CASE, train_scales, train_val_scales)


@pytest.mark.slow
@pytest.mark.parametrize(("stage", "component"), HURDLE_CASES, ids=lambda v: v)
def test_hurdle_all(stage, component, train_scales, train_val_scales) -> None:
    """All ten stored hurdle components (5 per stage)."""
    _check_hurdle_case(stage, component, train_scales, train_val_scales)


# ---------------------------------------------------------------------------
# Corpus quirks
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_naive_weekly_local_files_hold_the_global_views(train_scales, activity_map) -> None:
    """The three ``*_local_naive_weekly`` files equal the missing global views.

    See :data:`MISSING_GOLDEN_VIEWS`. The naive baselines are computed on the
    level series and do not depend on the training paradigm, so a ``local``
    naive_weekly frame is the same frame as the ``global`` one; what this test
    pins is that the recomputed GLOBAL view reproduces those three files
    exactly, i.e. no metric is missing from the corpus, only a file name.
    """
    from strikecast.evaluation.aggregate import evaluate  # noqa: PLC0415

    directory = GOLDEN / "diff"
    mae_scales, rmse_scales = train_scales
    substitutes = {
        ("cv_global_naive_weekly", "per_activity_level"): "per_activity_level_cv_local_naive_weekly.csv",
        ("test_global_naive_weekly", "per_region"): "per_region_test_local_naive_weekly.csv",
        ("test_global_naive_weekly", "per_horizon"): "per_horizon_test_local_naive_weekly.csv",
    }
    cache: dict[str, dict] = {}
    for (key, view), filename in substitutes.items():
        path = directory / filename
        if not path.exists():
            pytest.skip(f"golden file missing: {filename}")
        if key not in cache:
            long_df = pd.read_parquet(directory / f"predictions_long_{key}.parquet")
            cache[key] = evaluate(long_df, mae_scales, rmse_scales, activity_map)
        _assert_view_equal(cache[key][view], path)
