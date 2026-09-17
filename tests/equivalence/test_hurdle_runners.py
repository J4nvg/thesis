"""Level D: the engine + ``HurdleForecaster`` vs the four legacy hurdle loops.

Oracles: ``tests/legacy_ref/hurdle_runners.py`` (``final_hurdle.ipynb`` cells
15, 24, 29 and 34). All three channels -- ``prob``, ``count`` and ``hurdle`` --
are compared fold by fold and through ``PredictionSet.legacy_frame()`` against
``collect_predictions_long``: ``prob`` against the binary actuals, ``count`` and
``hurdle`` against the count actuals.

``tests/unit/test_composite.py`` drives the forecaster BY HAND with stub models.
Here it is driven by the real ``ExpandingWindowBacktest`` with real darts models
(``SKLearnClassifierModel`` -> ``LogisticRegression`` with the
``classprobability`` likelihood, so ``predict_likelihood_parameters=True``
returns ``0_p0`` / ``0_p1`` and the legacy "last component" rule picks
``P(Y > 0)``; ``LinearRegressionModel`` and a Poisson ``LightGBMModel`` as the
count head, both fit with ``sample_weight=list[TimeSeries]``).
"""

from __future__ import annotations

import pytest
from conftest import (
    CV_START_EARLY,
    HORIZON,
    PREDICT_STRIDE,
    RETRAIN_STRIDE,
    TRAIN_VAL_END,
    assert_fold_preds_equal,
    assert_long_frames_equal,
    classifier_builder,
    count_head_builder,
    lgbm_count_head_builder,
)

from strikecast.backtest import ExpandingWindowBacktest
from strikecast.backtest.grouping import run_grouped
from strikecast.backtest.predictions import PredictionSet
from strikecast.config.schema import BacktestConfig
from strikecast.models.composite import LEGACY_MIN_POSITIVE_SAMPLES, HurdleForecaster
from strikecast.transforms import Identity

CHANNELS = ("prob", "count", "hurdle")


@pytest.fixture(scope="module")
def hurdle_runners():
    from tests.legacy_ref import hurdle_runners as module  # noqa: PLC0415

    return module


def _cfg(start_frac: float) -> BacktestConfig:
    return BacktestConfig(
        start_frac=start_frac,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )


class _CountingBuilders:
    """Wraps the two builders so the number of fitted heads can be compared."""

    def __init__(self, clf, reg) -> None:
        self._clf, self._reg = clf, reg
        self.n_clf = 0
        self.n_reg = 0

    def classifier(self):
        self.n_clf += 1
        return self._clf()

    def regressor(self):
        self.n_reg += 1
        return self._reg()


def _assert_all_channels(got, expected, label: str) -> None:
    exp_c, exp_r, exp_h = expected
    assert_fold_preds_equal(got["prob"], exp_c, f"{label}prob ")
    assert_fold_preds_equal(got["count"], exp_r, f"{label}count ")
    assert_fold_preds_equal(got["hurdle"], exp_h, f"{label}hurdle ")


def _assert_long_frames(got, expected, binary_actuals, count_actuals, region_names) -> None:
    exp_c, exp_r, exp_h = expected
    prob_set = PredictionSet.from_fold_preds(binary_actuals, {"prob": got["prob"]}, region_names)
    assert_long_frames_equal(prob_set, binary_actuals, exp_c, region_names, "prob")

    count_set = PredictionSet.from_fold_preds(
        count_actuals, {"count": got["count"], "hurdle": got["hurdle"]}, region_names
    )
    assert_long_frames_equal(count_set, count_actuals, exp_r, region_names, "count")
    assert_long_frames_equal(count_set, count_actuals, exp_h, region_names, "hurdle")


# --------------------------------------------------------------------------- #
# cell 15: run_hurdle_cv
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "reg_builder", [count_head_builder, lgbm_count_head_builder], ids=["linear", "lgbm_poisson"]
)
def test_hurdle_cv_matches_legacy(hurdle_panel, hurdle_runners, reg_builder) -> None:
    hp = hurdle_panel
    legacy_b = _CountingBuilders(classifier_builder, reg_builder)
    expected = hurdle_runners.run_hurdle_cv(
        PREDICT_STRIDE,
        RETRAIN_STRIDE,
        target_for_cv_c=hp.binary_cv,
        target_for_cv_r=hp.count_cv,
        full_weights=hp.weights_full,
        full_past_covs_c=hp.clf_past,
        full_fut_covs_c=hp.clf_future,
        full_past_covs_r=hp.reg_past,
        full_fut_covs_r=hp.reg_future,
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        CV_START_VAL=CV_START_EARLY,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_b = _CountingBuilders(classifier_builder, reg_builder)
    forecaster = HurdleForecaster(
        new_b.classifier,
        new_b.regressor,
        binary_targets=hp.binary_cv,
        weights=hp.weights_full,
        clf_past=hp.clf_past,
        clf_future=hp.clf_future,
        reg_past=hp.reg_past,
        reg_future=hp.reg_future,
    )
    got = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity()).run(forecaster, hp.count_cv)

    assert tuple(got) == CHANNELS
    _assert_all_channels(got, expected, "hurdle/cv/")
    assert (new_b.n_clf, new_b.n_reg) == (legacy_b.n_clf, legacy_b.n_reg) == (4, 4)
    _assert_long_frames(got, expected, hp.binary_cv, hp.count_cv, hp.region_names)


# --------------------------------------------------------------------------- #
# cell 24: run_final_test (hurdle)
# --------------------------------------------------------------------------- #
def test_hurdle_test_matches_legacy(hurdle_panel, hurdle_runners) -> None:
    hp = hurdle_panel
    legacy_b = _CountingBuilders(classifier_builder, count_head_builder)
    exp_c, exp_r, exp_h, full_c, full_r = hurdle_runners.run_hurdle_test(
        PREDICT_STRIDE,
        RETRAIN_STRIDE,
        train_target_c=hp.train_c,
        val_target_c=hp.val_c,
        test_target_c=hp.test_c,
        train_target_r=hp.train_r,
        val_target_r=hp.val_r,
        test_target_r=hp.test_r,
        full_past_covs_c=hp.clf_past,
        full_fut_covs_c=hp.clf_future,
        full_past_covs_r=hp.reg_past,
        full_fut_covs_r=hp.reg_future,
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        TRAIN_VAL_END=TRAIN_VAL_END,
        OUTPUT_CHUNK_LEN=HORIZON,
    )
    expected = (exp_c, exp_r, exp_h)

    new_b = _CountingBuilders(classifier_builder, count_head_builder)
    forecaster = HurdleForecaster(
        new_b.classifier,
        new_b.regressor,
        binary_targets=full_c,
        count_targets=full_r,
        weights=hurdle_runners.make_positive_only_weights(full_r),
        clf_past=hp.clf_past,
        clf_future=hp.clf_future,
        reg_past=hp.reg_past,
        reg_future=hp.reg_future,
    )
    got = ExpandingWindowBacktest(_cfg(TRAIN_VAL_END), Identity()).run(forecaster, full_r)

    _assert_all_channels(got, expected, "hurdle/test/")
    assert len(exp_c[0]) == 19
    _assert_long_frames(got, expected, full_c, full_r, hp.region_names)


# --------------------------------------------------------------------------- #
# cell 29: run_hurdle_cv_per_activity
# --------------------------------------------------------------------------- #
def test_hurdle_cv_per_activity_matches_legacy(hurdle_panel, hurdle_runners) -> None:
    hp = hurdle_panel
    legacy_b = _CountingBuilders(classifier_builder, count_head_builder)
    expected = hurdle_runners.run_hurdle_cv_per_activity(
        PREDICT_STRIDE,
        RETRAIN_STRIDE,
        target_for_cv_c=hp.binary_cv,
        target_for_cv_r=hp.count_cv,
        full_weights=hp.weights_full,
        full_past_covs_c=hp.clf_past,
        full_fut_covs_c=hp.clf_future,
        full_past_covs_r=hp.reg_past,
        full_fut_covs_r=hp.reg_future,
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        region_names=hp.region_names,
        regions_activity=hp.activity_by_region,
        CV_START_VAL=CV_START_EARLY,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_b = _CountingBuilders(classifier_builder, count_head_builder)
    engine = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity())

    def run_one(group, targets, past, future):
        del past, future
        forecaster = HurdleForecaster(
            new_b.classifier,
            new_b.regressor,
            binary_targets=[hp.binary_cv[i] for i in group.indices],
            weights=[hp.weights_full[i] for i in group.indices],
            clf_past=[hp.clf_past[i] for i in group.indices],
            clf_future=[hp.clf_future[i] for i in group.indices],
            reg_past=[hp.reg_past[i] for i in group.indices],
            reg_future=[hp.reg_future[i] for i in group.indices],
        )
        return engine.run(forecaster, targets)

    got = run_grouped(
        run_one,
        region_names=hp.region_names,
        paradigm="activity",
        targets=hp.count_cv,
        activity_by_region=hp.activity_by_region,
    )

    _assert_all_channels(got, expected, "hurdle/activity/")
    assert (new_b.n_clf, new_b.n_reg) == (legacy_b.n_clf, legacy_b.n_reg)
    _assert_long_frames(got, expected, hp.binary_cv, hp.count_cv, hp.region_names)


# --------------------------------------------------------------------------- #
# cell 34: run_hurdle_cv_per_region, with the MIN_POSITIVE_SAMPLES dummy branch
# --------------------------------------------------------------------------- #
def test_hurdle_cv_per_region_matches_legacy_including_the_dummy_branch(
    hurdle_panel, hurdle_runners
) -> None:
    hp = hurdle_panel
    legacy_b = _CountingBuilders(classifier_builder, count_head_builder)
    expected = hurdle_runners.run_hurdle_cv_per_region(
        PREDICT_STRIDE,
        RETRAIN_STRIDE,
        target_for_cv_c=hp.binary_cv,
        target_for_cv_r=hp.count_cv,
        full_weights=hp.weights_full,
        full_past_covs_c=hp.clf_past,
        full_fut_covs_c=hp.clf_future,
        full_past_covs_r=hp.reg_past,
        full_fut_covs_r=hp.reg_future,
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        region_names=hp.region_names,
        CV_START_VAL=CV_START_EARLY,
        OUTPUT_CHUNK_LEN=HORIZON,
    )
    # both branches must have fired, or the comparison proves nothing
    assert 0 < legacy_b.n_reg < legacy_b.n_clf

    new_b = _CountingBuilders(classifier_builder, count_head_builder)
    engine = ExpandingWindowBacktest(_cfg(CV_START_EARLY), Identity())

    def run_one(group, targets, past, future):
        del past, future
        (i,) = group.indices
        forecaster = HurdleForecaster(
            new_b.classifier,
            new_b.regressor,
            binary_targets=[hp.binary_cv[i]],
            count_targets=[hp.count_cv[i]],
            weights=[hp.weights_full[i]],
            clf_past=[hp.clf_past[i]],
            clf_future=[hp.clf_future[i]],
            reg_past=[hp.reg_past[i]],
            reg_future=[hp.reg_future[i]],
            min_positive_samples=LEGACY_MIN_POSITIVE_SAMPLES,
        )
        return engine.run(forecaster, targets)

    got = run_grouped(
        run_one,
        region_names=hp.region_names,
        paradigm="local",
        targets=hp.count_cv,
    )

    _assert_all_channels(got, expected, "hurdle/local/")
    assert (new_b.n_clf, new_b.n_reg) == (legacy_b.n_clf, legacy_b.n_reg)
    _assert_long_frames(got, expected, hp.binary_cv, hp.count_cv, hp.region_names)
