"""Level D: ``run_grouped`` + the engine vs the four per-activity / per-region wrappers.

Oracles:

* ``src/evaluation_tools.py::run_expanding_cv_per_activity`` and
  ``::run_expanding_cv_per_region`` -- both call ``run_final_test`` internally,
  so the engine side uses ``GlobalDartsForecaster.for_test``;
* ``tests/legacy_ref/diff_runners.py::run_final_test_diff_per_activity`` and
  ``::run_final_test_diff_per_region``.

The activity map is deliberately interleaved (``r0``/``r2`` -> tier 2,
``r1``/``r3`` -> tier 1), so ``for level in sorted(groups)`` visits ``[1, 3]``
before ``[0, 2]`` and a wrapper that returned group-ordered results instead of
scattering them back would produce a different frame. Region order restoration
is therefore asserted twice: implicitly against the legacy wrappers, and
explicitly against a single group run.
"""

from __future__ import annotations

import pytest
from conftest import (
    HORIZON,
    PREDICT_STRIDE,
    RETRAIN_STRIDE,
    TRAIN_VAL_END,
    assert_fold_preds_equal,
    assert_long_frames_equal,
    linear_builder,
)

from strikecast.backtest import ExpandingWindowBacktest
from strikecast.backtest.grouping import partition, run_grouped
from strikecast.backtest.predictions import PredictionSet
from strikecast.backtest.protocols import SINGLE_CHANNEL
from strikecast.config.schema import BacktestConfig
from strikecast.models.adapters import GlobalDartsForecaster
from strikecast.transforms import Diff, Identity

CFG = BacktestConfig(
    start_frac=TRAIN_VAL_END,
    horizon=HORIZON,
    predict_stride=PREDICT_STRIDE,
    retrain_stride=RETRAIN_STRIDE,
)


@pytest.fixture(scope="module")
def legacy():
    from src import evaluation_tools  # noqa: PLC0415

    return evaluation_tools


@pytest.fixture(scope="module")
def diff_runners():
    from tests.legacy_ref import diff_runners as module  # noqa: PLC0415

    return module


def _run_one(transform):
    engine = ExpandingWindowBacktest(CFG, transform)

    def run_one(group, targets, past, future):
        del group
        return engine.run(GlobalDartsForecaster.for_test(linear_builder), targets, past, future)

    return run_one


# --------------------------------------------------------------------------- #
# the partition itself
# --------------------------------------------------------------------------- #
def test_activity_groups_are_sorted_by_level_not_by_region_order(panel) -> None:
    groups = partition(panel.region_names, "activity", panel.activity_by_region)
    assert [(g.label, g.indices) for g in groups] == [("1", [1, 3]), ("2", [0, 2])]


# --------------------------------------------------------------------------- #
# count family
# --------------------------------------------------------------------------- #
def test_per_activity_matches_legacy(panel, legacy) -> None:
    got = run_grouped(
        _run_one(Identity()),
        region_names=panel.region_names,
        paradigm="activity",
        targets=panel.target_full,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        activity_by_region=panel.activity_by_region,
    )
    old = legacy.run_expanding_cv_per_activity(
        linear_builder,
        panel.target_full,
        panel.region_names,
        panel.activity_by_region,
        TRAIN_VAL_END,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(got[SINGLE_CHANNEL], old, "count/activity ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_full, got, panel.region_names),
        panel.target_full,
        old,
        panel.region_names,
    )


def test_per_activity_restores_region_order(panel) -> None:
    """Region 1 and 3 come out of the FIRST group, at positions 1 and 3."""
    got = run_grouped(
        _run_one(Identity()),
        region_names=panel.region_names,
        paradigm="activity",
        targets=panel.target_full,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        activity_by_region=panel.activity_by_region,
    )

    group_indices = [1, 3]
    group_only = ExpandingWindowBacktest(CFG, Identity()).run(
        GlobalDartsForecaster.for_test(linear_builder),
        [panel.target_full[i] for i in group_indices],
        [panel.past_covs[i] for i in group_indices],
        [panel.future_covs[i] for i in group_indices],
    )

    assert_fold_preds_equal(
        [got[SINGLE_CHANNEL][i] for i in group_indices],
        group_only[SINGLE_CHANNEL],
        "scatter ",
    )
    # and the tier-2 group's regions did NOT end up in those slots
    with pytest.raises(AssertionError):
        assert_fold_preds_equal(
            [got[SINGLE_CHANNEL][0]], [group_only[SINGLE_CHANNEL][0]], "wrong slot "
        )


def test_per_region_matches_legacy(panel, legacy) -> None:
    got = run_grouped(
        _run_one(Identity()),
        region_names=panel.region_names,
        paradigm="local",
        targets=panel.target_full,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
    )
    old = legacy.run_expanding_cv_per_region(
        linear_builder,
        panel.target_full,
        TRAIN_VAL_END,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(got[SINGLE_CHANNEL], old, "count/local ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_full, got, panel.region_names),
        panel.target_full,
        old,
        panel.region_names,
    )


def test_global_paradigm_is_the_ungrouped_run(panel) -> None:
    got = run_grouped(
        _run_one(Identity()),
        region_names=panel.region_names,
        paradigm="global",
        targets=panel.target_full,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
    )
    direct = ExpandingWindowBacktest(CFG, Identity()).run(
        GlobalDartsForecaster.for_test(linear_builder),
        panel.target_full,
        panel.past_covs,
        panel.future_covs,
    )
    assert_fold_preds_equal(got[SINGLE_CHANNEL], direct[SINGLE_CHANNEL], "global ")


# --------------------------------------------------------------------------- #
# diff family
# --------------------------------------------------------------------------- #
def test_diff_per_activity_matches_legacy(panel, diff_runners) -> None:
    got = run_grouped(
        _run_one(Diff()),
        region_names=panel.region_names,
        paradigm="activity",
        targets=panel.target_full,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        activity_by_region=panel.activity_by_region,
    )
    old = diff_runners.run_final_test_diff_per_activity(
        linear_builder,
        panel.target_full_diff,
        panel.target_full,
        panel.region_names,
        panel.activity_by_region,
        TRAIN_VAL_END,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(got[SINGLE_CHANNEL], old, "diff/activity ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_full, got, panel.region_names),
        panel.target_full,
        old,
        panel.region_names,
    )


def test_diff_per_region_matches_legacy(panel, diff_runners) -> None:
    got = run_grouped(
        _run_one(Diff()),
        region_names=panel.region_names,
        paradigm="local",
        targets=panel.target_full,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
    )
    old = diff_runners.run_final_test_diff_per_region(
        linear_builder,
        panel.target_full_diff,
        panel.target_full,
        TRAIN_VAL_END,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(got[SINGLE_CHANNEL], old, "diff/local ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_full, got, panel.region_names),
        panel.target_full,
        old,
        panel.region_names,
    )
