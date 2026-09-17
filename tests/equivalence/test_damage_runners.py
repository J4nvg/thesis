"""Level D: the engine + ``MultiTargetClassifierForecaster`` vs the damage loops.

Oracles: ``tests/legacy_ref/damage_runners.py`` (``damage_classifier.ipynb``
cells 17 and 26), run with TWO damage keys so that the per-key covariate lists,
the per-key model set and the channel order are all exercised.

Both legacy loops take the schedule from the FIRST key alone and apply it to
every key (flag F68), which is why the engine is handed that first key's target
list as ``level_targets`` while the forecaster slices every key's own list.
"""

from __future__ import annotations

import pytest
from conftest import (
    HORIZON,
    PREDICT_STRIDE,
    RETRAIN_STRIDE,
    assert_fold_preds_equal,
    assert_long_frames_equal,
    classifier_builder,
)

from strikecast.backtest import ExpandingWindowBacktest
from strikecast.backtest.predictions import PredictionSet
from strikecast.config.schema import BacktestConfig
from strikecast.models.composite import MultiTargetClassifierForecaster
from strikecast.transforms import Identity


@pytest.fixture(scope="module")
def damage_runners():
    from tests.legacy_ref import damage_runners as module  # noqa: PLC0415

    return module


def _cfg(start_frac: float) -> BacktestConfig:
    return BacktestConfig(
        start_frac=start_frac,
        horizon=HORIZON,
        predict_stride=PREDICT_STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )


class _CountingBuilder:
    def __init__(self, builder) -> None:
        self._builder = builder
        self.n = 0

    def __call__(self):
        self.n += 1
        return self._builder()


def test_damage_cv_matches_legacy(damage_panel, damage_runners) -> None:
    dp = damage_panel
    keys = dp.keys
    start_frac = dp.block(keys[0])["csv"]

    legacy_builder = _CountingBuilder(classifier_builder)
    expected = damage_runners.run_damage_cv(
        PREDICT_STRIDE,
        RETRAIN_STRIDE,
        damage_classes=dp.classes,
        get_damage_classifier=legacy_builder,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_builder = _CountingBuilder(classifier_builder)
    forecaster = MultiTargetClassifierForecaster(
        new_builder,
        {k: dp.block(k)["TCV"] for k in keys},
        {k: dp.block(k)["FPC"] for k in keys},
        {k: dp.block(k)["FFC"] for k in keys},
    )
    got = ExpandingWindowBacktest(_cfg(start_frac), Identity()).run(
        forecaster, dp.block(keys[0])["TCV"]
    )

    assert tuple(got) == tuple(keys)
    assert new_builder.n == legacy_builder.n == len(keys) * 4  # 4 retrain points
    for key in keys:
        assert_fold_preds_equal(got[key], expected[key], f"damage/cv/{key} ")
        # the kept component is the LAST likelihood component, P(damage)
        assert got[key][0][0].n_components == 1
        actuals = dp.block(key)["TCV"]
        assert_long_frames_equal(
            PredictionSet.from_fold_preds(actuals, {key: got[key]}, dp.region_names),
            actuals,
            expected[key],
            dp.region_names,
            key,
        )


def test_damage_test_matches_legacy(damage_panel, damage_runners) -> None:
    dp = damage_panel
    keys = dp.keys
    start_frac = dp.block(keys[0])["tve"]

    legacy_builder = _CountingBuilder(classifier_builder)
    expected, full_target_d = damage_runners.run_damage_test(
        PREDICT_STRIDE,
        RETRAIN_STRIDE,
        damage_classes=dp.classes,
        get_damage_classifier=legacy_builder,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_builder = _CountingBuilder(classifier_builder)
    forecaster = MultiTargetClassifierForecaster(
        new_builder,
        full_target_d,
        {k: dp.block(k)["FPC"] for k in keys},
        {k: dp.block(k)["FFC"] for k in keys},
    )
    got = ExpandingWindowBacktest(_cfg(start_frac), Identity()).run(
        forecaster, full_target_d[keys[0]]
    )

    assert len(expected[keys[0]][0]) == 19
    for key in keys:
        assert_fold_preds_equal(got[key], expected[key], f"damage/test/{key} ")
        actuals = full_target_d[key]
        assert_long_frames_equal(
            PredictionSet.from_fold_preds(actuals, {key: got[key]}, dp.region_names),
            actuals,
            expected[key],
            dp.region_names,
            key,
        )
