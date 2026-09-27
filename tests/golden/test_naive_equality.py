"""Level D/E golden test: the ported naive baselines on the REAL bundle.

Two comparisons, both exact:

* **D** -- `strikecast.backtest.naive.naive_prediction_set(...).legacy_frame()`
  against `src.evaluation_tools.naive_collect_long`, run side by side on the
  same series in the same process.
* **E** -- the same frame against the stored
  `golden/results/diff/predictions_long_{cv,test}_global_naive_{last,weekly}.parquet`.
  Those are the only stored naive prediction frames in `golden/`: the count
  family (`golden/results/gbdt`, `golden/results/lstm`) persisted naive
  *metrics* but not naive *predictions*. They are nonetheless the right level-E
  target for the count family, because `_diff_regression.py` computes its
  naives on the **level** series (`target_for_cv` / `target_full` with
  `CV_START_VAL` / `TRAIN_VAL_END`, horizon 7, stride 1) through the very same
  `naive_collect_long` the count notebooks call -- the diff transform never
  touches them.

Fixtures: `inputs` and `legacy_src` come from `tests/conftest.py`. The bundle is
built through the new API rather than the legacy 9-tuple because
`tests/golden/test_series_bundle_equality.py` already proves the two are
identical, so re-running the legacy pipeline here would only duplicate loading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.golden

pytest.importorskip("darts")

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIFF = REPO_ROOT / "golden" / "results" / "diff"

TARGET = "act_drone_strike_on_ua"
HORIZON = 7
STRIDE = 1
METHODS = ("naive_last", "naive_weekly")
VIEWS = ("cv", "test")


@pytest.fixture(scope="module")
def bundle(inputs):
    """The real `SeriesBundle`, count-family variant."""
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


def _view(bundle, name: str):
    """`(series_list, start_frac)` for the validation-CV or the test stage.

    The legacy calls are
    ``naive_collect_long(target_for_cv, ..., CV_START_VAL, 7, CV_STRIDE)`` and
    ``naive_collect_long(target_full, ..., TEST_START_FRAC=TRAIN_VAL_END, 7, 1)``.
    """
    if name == "cv":
        return bundle.target_cv_view, bundle.cv_start_frac
    return bundle.target_full, bundle.train_val_end


@pytest.fixture(scope="module")
def prediction_sets(bundle):
    from strikecast.backtest.naive import naive_prediction_set  # noqa: PLC0415

    out = {}
    for view in VIEWS:
        series, start_frac = _view(bundle, view)
        for method in METHODS:
            out[view, method] = naive_prediction_set(
                series, bundle.region_names, method, start_frac, HORIZON, STRIDE
            )
    return out


# --------------------------------------------------------------------------
# Level D: against the legacy function, same process
# --------------------------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("method", METHODS)
def test_matches_naive_collect_long(legacy_src, bundle, prediction_sets, view, method) -> None:
    series, start_frac = _view(bundle, view)
    legacy_long, _legacy_fold_preds = legacy_src.naive_collect_long(
        series, bundle.region_names, method, start_frac, HORIZON, STRIDE
    )
    prediction_sets[view, method].assert_equal_legacy(legacy_long)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("method", METHODS)
def test_fold_preds_match_legacy_forecasts(legacy_src, bundle, view, method) -> None:
    """The TimeSeries themselves, not just the flattened frame."""
    import numpy as np  # noqa: PLC0415

    from strikecast.backtest.naive import NAIVE_METHODS  # noqa: PLC0415

    legacy_fn = {
        "naive_last": legacy_src.naive_last_historical_forecasts,
        "naive_weekly": legacy_src.naive_weekly_historical_forecasts,
    }[method]
    series, start_frac = _view(bundle, view)
    for ts in series:
        new = NAIVE_METHODS[method](ts, start_frac, HORIZON, STRIDE)
        old = legacy_fn(ts, start_frac, HORIZON, STRIDE)
        assert len(new) == len(old)
        for a, b in zip(new, old, strict=True):
            np.testing.assert_array_equal(a.values(), b.values())
            assert a.time_index.equals(b.time_index)


def test_origin_date_equals_first_predicted_date(bundle, prediction_sets) -> None:
    """The one new column: it is `pred.time_index[0]`, i.e. horizon 1's date."""
    for (_view_name, _method), ps in prediction_sets.items():
        frame = ps.frame
        h1 = frame[frame["horizon"] == 1]
        assert (h1["origin_date"] == h1["date"]).all()
        # and it is constant within a (region, fold)
        assert (frame.groupby(["region", "fold"])["origin_date"].nunique() == 1).all()


def test_fold_counts_match_the_legacy_schedule(bundle, prediction_sets) -> None:
    """Pins the realised schedule: 79 CV folds, 164 test folds, 20 regions."""
    expected = {"cv": 79, "test": 164}
    for (view, _method), ps in prediction_sets.items():
        frame = ps.frame
        assert frame["fold"].nunique() == expected[view]
        assert frame["region"].nunique() == len(bundle.region_names)
        assert len(frame) == expected[view] * HORIZON * len(bundle.region_names)


# --------------------------------------------------------------------------
# Level E: against the stored prediction frames
# --------------------------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("method", METHODS)
def test_matches_stored_golden_predictions(prediction_sets, view, method) -> None:
    path = GOLDEN_DIFF / f"predictions_long_{view}_global_{method}.parquet"
    if not path.exists():
        pytest.skip(f"stored golden predictions not available: {path}")
    stored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(
        prediction_sets[view, method].legacy_frame(), stored, check_dtype=True, obj=str(path.name)
    )
