"""Level D: the engine vs the three count-family runners in ``src/evaluation_tools``.

``ExpandingWindowBacktest(Identity)`` plus ``GlobalDartsForecaster`` /
``LocalDartsForecaster``, driven end to end, against

* ``run_expanding_cv``      -> ``GlobalDartsForecaster.for_cv``
* ``run_final_test``        -> ``GlobalDartsForecaster.for_test``
* ``run_expanding_cv_iter`` -> ``GlobalDartsForecaster.for_tuning``

with and without covariates, global and local, comparing fold by fold AND
through ``PredictionSet.legacy_frame()`` vs ``collect_predictions_long``.

These go past ``tests/unit/test_engine.py`` and ``tests/unit/test_adapters.py``
in three ways: the real ``Identity`` transform and the real adapters are used
(the unit tests use local fakes), the runs are engine-driven rather than hand
driven, and the long frame is compared as well as the fold bundles.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    CV_START_EARLY,
    CV_START_VAL,
    HORIZON,
    PREDICT_STRIDE,
    RETRAIN_STRIDE,
    TRAIN_VAL_END,
    assert_fold_preds_equal,
    assert_long_frames_equal,
    linear_builder,
    linear_builder_no_covs,
    log_link_linear_builder,
)
from darts.models import NaiveMean

from strikecast.backtest import ExpandingWindowBacktest
from strikecast.backtest.hooks import CallbackHook
from strikecast.backtest.predictions import PredictionSet
from strikecast.backtest.protocols import SINGLE_CHANNEL
from strikecast.config.schema import BacktestConfig
from strikecast.models.adapters import GlobalDartsForecaster, LocalDartsForecaster
from strikecast.transforms import Identity


@pytest.fixture(scope="module")
def legacy():
    from src import evaluation_tools  # noqa: PLC0415

    return evaluation_tools


def _cfg(start_frac: float) -> BacktestConfig:
    return BacktestConfig(
        start_frac=start_frac,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )


def _snapshot(bundle) -> list[list]:
    """The count-family iterator yields the LIVE list; copy it before advancing."""
    return [list(region) for region in bundle]


# --------------------------------------------------------------------------- #
# run_expanding_cv  <->  GlobalDartsForecaster.for_cv
# --------------------------------------------------------------------------- #
def test_cv_global_linear_with_covariates(panel, legacy) -> None:
    targets = panel.target_cv

    new = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity()).run(
        GlobalDartsForecaster.for_cv(linear_builder),
        targets,
        panel.past_covs,
        panel.future_covs,
    )
    old = legacy.run_expanding_cv(
        linear_builder,
        targets,
        CV_START_EARLY,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "cv/global/covs ")
    assert len(old[0]) == 23  # int(0.7 * 95) = 66 .. 88 inclusive
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(targets, new, panel.region_names),
        targets,
        old,
        panel.region_names,
    )


def test_cv_global_linear_without_covariates(panel, legacy) -> None:
    targets = panel.target_cv

    new = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity()).run(
        GlobalDartsForecaster.for_cv(linear_builder_no_covs), targets
    )
    old = legacy.run_expanding_cv(
        linear_builder_no_covs,
        targets,
        CV_START_EARLY,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "cv/global/nocovs ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(targets, new, panel.region_names),
        targets,
        old,
        panel.region_names,
    )


def test_cv_global_at_the_real_cv_start_frac(panel, legacy) -> None:
    """``CV_START_VAL = 0.875``: a single retrain window, the legacy CV shape."""
    targets = panel.target_cv

    new = ExpandingWindowBacktest(_cfg(CV_START_VAL), Identity()).run(
        GlobalDartsForecaster.for_cv(linear_builder),
        targets,
        panel.past_covs,
        panel.future_covs,
    )
    old = legacy.run_expanding_cv(
        linear_builder,
        targets,
        CV_START_VAL,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "cv/global/0.875 ")


def test_cv_local_naive_mean(panel, legacy) -> None:
    targets = panel.target_cv

    new = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity()).run(
        LocalDartsForecaster(NaiveMean), targets, panel.past_covs, panel.future_covs
    )
    old = legacy.run_expanding_cv(
        NaiveMean,
        targets,
        CV_START_EARLY,
        is_local=True,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "cv/local ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(targets, new, panel.region_names),
        targets,
        old,
        panel.region_names,
    )


# --------------------------------------------------------------------------- #
# run_final_test  <->  GlobalDartsForecaster.for_test
# --------------------------------------------------------------------------- #
def test_final_test_global_linear(panel, legacy) -> None:
    targets = panel.target_full

    new = ExpandingWindowBacktest(_cfg(TRAIN_VAL_END), Identity()).run(
        GlobalDartsForecaster.for_test(linear_builder),
        targets,
        panel.past_covs,
        panel.future_covs,
    )
    old = legacy.run_final_test(
        linear_builder,
        targets,
        TRAIN_VAL_END,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        horizon=HORIZON,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "test/global ")
    assert len(old[0]) == 19  # int(0.7999999999999999 * 121) = 96 .. 114 inclusive
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(targets, new, panel.region_names),
        targets,
        old,
        panel.region_names,
    )


def test_final_test_local_naive_mean(panel, legacy) -> None:
    targets = panel.target_full

    new = ExpandingWindowBacktest(_cfg(TRAIN_VAL_END), Identity()).run(
        LocalDartsForecaster(NaiveMean), targets
    )
    old = legacy.run_final_test(
        NaiveMean,
        targets,
        TRAIN_VAL_END,
        is_local=True,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        horizon=HORIZON,
        verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], old, "test/local ")
    assert_long_frames_equal(
        PredictionSet.from_fold_preds(targets, new, panel.region_names),
        targets,
        old,
        panel.region_names,
    )


# --------------------------------------------------------------------------- #
# run_expanding_cv_iter  <->  GlobalDartsForecaster.for_tuning
# --------------------------------------------------------------------------- #
def test_tuning_iter_global_matches_every_yield(panel, legacy) -> None:
    """The Optuna shape: compare the cumulative bundle at EVERY yield."""
    targets = panel.target_cv

    hook_snapshots: list[list[list]] = []
    engine = ExpandingWindowBacktest(
        _cfg(CV_START_EARLY),
        Identity(),
        hooks=[CallbackHook(lambda r, cum: hook_snapshots.append(_snapshot(cum[SINGLE_CHANNEL])))],
    )
    new_iter = engine.iter_folds(
        GlobalDartsForecaster.for_tuning(linear_builder),
        targets,
        panel.past_covs,
        panel.future_covs,
    )
    old_iter = legacy.run_expanding_cv_iter(
        linear_builder,
        targets,
        CV_START_EARLY,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=panel.past_covs,
        future_covs=panel.future_covs,
        verbose=False,
    )

    cumulative: list[list] = [[] for _ in targets]
    n_folds = 0
    for fold_result, legacy_bundle in zip(new_iter, old_iter, strict=True):
        for r_idx, pred in enumerate(fold_result.predictions[SINGLE_CHANNEL]):
            cumulative[r_idx].append(pred)
        assert_fold_preds_equal(cumulative, legacy_bundle, f"tuning yield {n_folds} ")
        # the hook saw the same bundle, at the same point in the loop
        assert_fold_preds_equal(
            hook_snapshots[n_folds], legacy_bundle, f"tuning hook {n_folds} "
        )
        n_folds += 1

    assert n_folds == 23
    assert len(hook_snapshots) == n_folds


def test_tuning_iter_local_matches_every_yield(panel, legacy) -> None:
    targets = panel.target_cv

    engine = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity())
    new_iter = engine.iter_folds(LocalDartsForecaster(NaiveMean), targets)
    old_iter = legacy.run_expanding_cv_iter(
        NaiveMean,
        targets,
        CV_START_EARLY,
        is_local=True,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        verbose=False,
    )

    cumulative: list[list] = [[] for _ in targets]
    n_folds = 0
    for fold_result, legacy_bundle in zip(new_iter, old_iter, strict=True):
        for r_idx, pred in enumerate(fold_result.predictions[SINGLE_CHANNEL]):
            cumulative[r_idx].append(pred)
        assert_fold_preds_equal(cumulative, legacy_bundle, f"tuning/local yield {n_folds} ")
        n_folds += 1

    assert n_folds == 23


# --------------------------------------------------------------------------- #
# the one post-processing difference between the three runners (F55 / F56)
# --------------------------------------------------------------------------- #
def test_log_link_exp_divides_the_three_presets_exactly_as_legacy_does(panel, legacy) -> None:
    """``_count_log_link``: ``exp`` in CV and test, NOT in the tuning iterator.

    A non-neural model makes every other difference between the three runners
    invisible, so this is the end-to-end proof that the preset wiring matches.
    """
    targets = panel.target_cv
    engine = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity())

    cv_new = engine.run(GlobalDartsForecaster.for_cv(log_link_linear_builder), targets)
    cv_old = legacy.run_expanding_cv(
        log_link_linear_builder,
        targets,
        CV_START_EARLY,
        horizon=HORIZON,
        stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        verbose=False,
    )
    assert_fold_preds_equal(cv_new[SINGLE_CHANNEL], cv_old, "log-link cv ")

    tune_new = [
        result.predictions[SINGLE_CHANNEL]
        for result in engine.iter_folds(
            GlobalDartsForecaster.for_tuning(log_link_linear_builder), targets
        )
    ]
    tune_old = list(
        legacy.run_expanding_cv_iter(
            log_link_linear_builder,
            targets,
            CV_START_EARLY,
            horizon=HORIZON,
            stride=PREDICT_STRIDE,
            retrain_stride=RETRAIN_STRIDE,
            verbose=False,
        )
    )[-1]
    per_region = [[fold[r_idx] for fold in tune_new] for r_idx in range(len(targets))]
    assert_fold_preds_equal(per_region, tune_old, "log-link tuning ")

    # and the two really do differ: exp() was applied in CV and not in tuning
    np.testing.assert_allclose(
        cv_old[0][0].values(), np.exp(tune_old[0][0].values()), rtol=0, atol=1e-12
    )

    test_new = ExpandingWindowBacktest(_cfg(TRAIN_VAL_END), Identity()).run(
        GlobalDartsForecaster.for_test(log_link_linear_builder), panel.target_full
    )
    test_old = legacy.run_final_test(
        log_link_linear_builder,
        panel.target_full,
        TRAIN_VAL_END,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        horizon=HORIZON,
        verbose=False,
    )
    assert_fold_preds_equal(test_new[SINGLE_CHANNEL], test_old, "log-link test ")
