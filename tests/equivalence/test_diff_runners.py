"""Level D: the engine with the ``Diff`` transform vs the diff-family runners.

Oracles: ``tests/legacy_ref/diff_runners.py``, the verbatim copy of
``_diff_regression.py`` lines 660-1005 -- ``run_expanding_cv``,
``run_expanding_cv_iter`` and ``run_final_test_diff``, global and local, with
the ``NaiveMean`` fallback (F5) exercised by a builder that raises for exactly
one region.

The legacy runners take TWO lists: ``target_diff_list`` (what the model sees)
and ``target_level_list`` (the un-diff anchor and the scoring actuals). The
engine takes one level list and derives the model-space list with
``Diff.forward``. Whether that reproduces the legacy pair depends on the stage:

* TEST stage -- ``target_full_diff`` is ``Diff(...).fit_transform(target_full)``,
  which is exactly ``Diff.forward(target_full)``. The engine reproduces it.
* CV stage -- ``target_for_cv_diff`` is the full diffed list SPLIT at
  ``TRAIN_VAL_END``, which is one step LONGER than differencing the level CV
  view. No level list handed to the engine produces it, so the natural wiring
  does not reproduce the legacy CV. That is asserted as a strict xfail below,
  with the exact lengths, plus a green test showing the one wiring that does
  reproduce it.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import (
    CV_START_EARLY,
    CV_START_VAL,
    HORIZON,
    PREDICT_STRIDE,
    RETRAIN_STRIDE,
    TRAIN_VAL_END,
    FlakyLocalModel,
    assert_fold_preds_equal,
    assert_long_frames_equal,
    linear_builder,
)
from darts.models import NaiveMean

from strikecast.backtest import ExpandingWindowBacktest
from strikecast.backtest.predictions import PredictionSet
from strikecast.backtest.protocols import SINGLE_CHANNEL
from strikecast.config.schema import BacktestConfig
from strikecast.models.adapters import GlobalDartsForecaster, LocalDartsForecaster
from strikecast.transforms import Diff


@pytest.fixture(scope="module")
def diff_runners():
    from tests.legacy_ref import diff_runners as module  # noqa: PLC0415

    return module


def _cfg(start_frac: float) -> BacktestConfig:
    return BacktestConfig(
        start_frac=start_frac,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )


def _cv_level_plus_one(panel) -> list:
    """The level list whose ``Diff.forward`` IS the legacy ``target_for_cv_diff``.

    The legacy CV diff view covers level days ``1 .. len(cv_view)``; differencing
    a level list of ``len(cv_view) + 1`` days produces exactly that.
    """
    n = len(panel.target_cv[0]) + 1
    return [ts[:n] for ts in panel.target_full]


# --------------------------------------------------------------------------- #
# the model-space lists
# --------------------------------------------------------------------------- #
def test_forward_reproduces_the_legacy_test_stage_diff_list(panel) -> None:
    """``target_full_diff`` == ``Diff.forward(target_full)``, value for value."""
    assert_fold_preds_equal(
        [[ts] for ts in Diff().forward(panel.target_full)],
        [[ts] for ts in panel.target_full_diff],
        "forward(full) ",
    )


def test_cv_diff_view_is_one_step_longer_than_forward_on_the_cv_view(panel) -> None:
    """F51 / F52, reproduced on the synthetic panel with the real panel's pattern.

    Real panel: level 847 -> CV view 676, diffed 846 -> diffed CV view 676,
    ``Diff(level CV view)`` 675. Synthetic panel: 121 / 95 / 120 / 95 / 94.
    """
    level_cv = panel.target_cv[0]
    diff_cv = panel.target_cv_diff[0]
    forward_of_cv = Diff().forward(panel.target_cv)[0]

    assert (len(panel.target_full[0]), len(panel.target_full_diff[0])) == (121, 120)
    assert len(level_cv) == 95
    assert len(diff_cv) == 95
    assert len(forward_of_cv) == 94
    assert diff_cv.time_index[-1] == level_cv.time_index[-1] + pd.Timedelta(days=1)
    assert forward_of_cv.time_index[-1] == level_cv.time_index[-1]

    # ... and therefore the schedules differ: int(0.7 * 95) = 66 vs int(0.7 * 94) = 65
    assert int(CV_START_EARLY * len(diff_cv)) == 66
    assert int(CV_START_EARLY * len(forward_of_cv)) == 65


# --------------------------------------------------------------------------- #
# TEST stage: the natural wiring reproduces the legacy runner exactly
# --------------------------------------------------------------------------- #
def test_test_stage_global_matches_legacy(panel, diff_runners) -> None:
    new = ExpandingWindowBacktest(_cfg(TRAIN_VAL_END), Diff()).run(
        GlobalDartsForecaster.for_test(linear_builder),
        panel.target_full,
        panel.past_covs,
        panel.future_covs,
    )
    old = diff_runners.run_final_test_diff(
        linear_builder,
        panel.target_full_diff,
        panel.target_full,
        TRAIN_VAL_END,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        horizon=HORIZON,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "diff/test/global ")
    assert len(old[0]) == 19  # int(0.7999999999999999 * 120) = 95 .. 113 inclusive
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_full, new, panel.region_names),
        panel.target_full,
        old,
        panel.region_names,
    )


def test_test_stage_local_with_naive_mean_fallback_matches_legacy(panel, diff_runners) -> None:
    """F5: the local builder raises for region ``r1`` and both sides fall back."""
    new = ExpandingWindowBacktest(_cfg(TRAIN_VAL_END), Diff()).run(
        LocalDartsForecaster(FlakyLocalModel, fallback=NaiveMean), panel.target_full
    )
    old = diff_runners.run_final_test_diff(
        FlakyLocalModel,
        panel.target_full_diff,
        panel.target_full,
        TRAIN_VAL_END,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        horizon=HORIZON,
        is_local=True,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "diff/test/local ")
    # the fallback really fired: region 1 differs from the others' +0.25 offset
    pure_naive = ExpandingWindowBacktest(_cfg(TRAIN_VAL_END), Diff()).run(
        LocalDartsForecaster(NaiveMean), panel.target_full
    )
    assert_fold_preds_equal(
        [new[SINGLE_CHANNEL][1]], [pure_naive[SINGLE_CHANNEL][1]], "fallback region "
    )
    with pytest.raises(AssertionError):
        assert_fold_preds_equal(
            [new[SINGLE_CHANNEL][0]], [pure_naive[SINGLE_CHANNEL][0]], "non-fallback region "
        )
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_full, new, panel.region_names),
        panel.target_full,
        old,
        panel.region_names,
    )


# --------------------------------------------------------------------------- #
# CV stage
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DISCREPANCY (F80): the legacy diff CV runs on `target_for_cv_diff` = "
        "Diff(full).split_before(TRAIN_VAL_END), 95 steps on this panel (676 on "
        "the real one). The engine derives its model-space list with "
        "Diff.forward(level_targets); handed the level CV view it gets 94 steps "
        "(675 on the real panel), so start_idx is 65 instead of 66 and every "
        "fold is shifted one day earlier. Legacy is right; src/ is not changed."
    ),
)
def test_cv_global_natural_wiring_matches_legacy(panel, diff_runners) -> None:
    new = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Diff()).run(
        GlobalDartsForecaster.for_cv(linear_builder),
        panel.target_cv,
        panel.past_covs,
        panel.future_covs,
    )
    old = diff_runners.run_expanding_cv(
        linear_builder,
        panel.target_cv_diff,
        panel.target_cv,
        CV_START_EARLY,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "diff/cv/global ")


@pytest.mark.parametrize("start_frac", [CV_START_EARLY, CV_START_VAL])
def test_cv_global_matches_legacy_with_explicit_model_targets(
    panel, diff_runners, start_frac
) -> None:
    """The resolution of F80: hand the engine the legacy pair explicitly.

    ``level_targets=target_for_cv`` (the anchor list the legacy runner uses)
    and ``model_targets=target_for_cv_diff`` (the list it schedules and slices
    on). No extended level list, no re-differencing: this is exactly what
    ``_diff_regression.py`` passes to ``run_expanding_cv``.
    """
    new = ExpandingWindowBacktest(_cfg(start_frac), Diff()).run(
        GlobalDartsForecaster.for_cv(linear_builder),
        panel.target_cv,
        panel.past_covs,
        panel.future_covs,
        model_targets=panel.target_cv_diff,
    )
    old = diff_runners.run_expanding_cv(
        linear_builder,
        panel.target_cv_diff,
        panel.target_cv,
        start_frac,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, f"diff/cv/explicit/{start_frac} ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_cv, new, panel.region_names),
        panel.target_cv,
        old,
        panel.region_names,
    )


@pytest.mark.parametrize("start_frac", [CV_START_EARLY, CV_START_VAL])
def test_cv_global_matches_legacy_when_forward_reproduces_the_cv_diff_view(
    panel, diff_runners, start_frac
) -> None:
    """The wiring that DOES reproduce the legacy CV: one extra level day.

    ``Diff.forward(level[: len(cv) + 1])`` is ``target_for_cv_diff`` exactly, so
    the schedule, the model inputs and the anchors all line up and the fold
    bundles are equal to 1e-9. Recorded here so the fix for the xfail above is
    unambiguous; nothing in ``src/`` was changed to make it pass.
    """
    level_for_engine = _cv_level_plus_one(panel)
    assert_fold_preds_equal(
        [[ts] for ts in Diff().forward(level_for_engine)],
        [[ts] for ts in panel.target_cv_diff],
        "forward(cv+1) ",
    )

    new = ExpandingWindowBacktest(_cfg(start_frac), Diff()).run(
        GlobalDartsForecaster.for_cv(linear_builder),
        level_for_engine,
        panel.past_covs,
        panel.future_covs,
    )
    old = diff_runners.run_expanding_cv(
        linear_builder,
        panel.target_cv_diff,
        panel.target_cv,
        start_frac,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, f"diff/cv/global/{start_frac} ")
    # the long frame is built against the LEGACY actuals, `target_for_cv`
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(panel.target_cv, new, panel.region_names),
        panel.target_cv,
        old,
        panel.region_names,
    )


def test_cv_local_with_fallback_matches_legacy(panel, diff_runners) -> None:
    level_for_engine = _cv_level_plus_one(panel)

    new = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Diff()).run(
        LocalDartsForecaster(FlakyLocalModel, fallback=NaiveMean), level_for_engine
    )
    old = diff_runners.run_expanding_cv(
        FlakyLocalModel,
        panel.target_cv_diff,
        panel.target_cv,
        CV_START_EARLY,
        is_local=True,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "diff/cv/local ")


def test_cv_iter_matches_every_yield(panel, diff_runners) -> None:
    """``run_expanding_cv_iter`` (diff twin): compare the bundle at every yield."""
    level_for_engine = _cv_level_plus_one(panel)

    engine = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Diff())
    new_iter = engine.iter_folds(
        GlobalDartsForecaster.for_tuning(linear_builder),
        level_for_engine,
        panel.past_covs,
        panel.future_covs,
    )
    old_iter = diff_runners.run_expanding_cv_iter(
        linear_builder,
        panel.target_cv_diff,
        panel.target_cv,
        CV_START_EARLY,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    cumulative: list[list] = [[] for _ in level_for_engine]
    n_folds = 0
    for fold_result, legacy_bundle in zip(new_iter, old_iter, strict=True):
        for r_idx, pred in enumerate(fold_result.predictions[SINGLE_CHANNEL]):
            cumulative[r_idx].append(pred)
        assert_fold_preds_equal(cumulative, legacy_bundle, f"diff/iter yield {n_folds} ")
        n_folds += 1

    assert n_folds == 23
