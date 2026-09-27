"""Level B golden test: `SeriesBundle` == the legacy 9-tuple, exactly.

Builds the panel with the LEGACY functions on the real data under `data/fixed`
and `data/dataset`, then asserts that every list produced by
`strikecast.data.series.build_bundle` is identical -- values, time index,
components and static covariates -- to what
`build_ts_and_apply_window_transformer` + `get_covs_and_encodings` produce.

Fixtures are local on purpose: `tests/conftest.py` belongs to another work
stream and this file must not depend on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.golden

TARGET = "act_drone_strike_on_ua"
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
FIXED_DATA_PATH = REPO_ROOT / "data" / "fixed"
DATASET_PATH = REPO_ROOT / "data" / "dataset"

_REQUIRED = [
    FIXED_DATA_PATH / "regions.txt",
    FIXED_DATA_PATH / "regions_activity_cat.json",
    DATASET_PATH / "master_combined_timeseries.parquet",
]

pytest.importorskip("darts")
if not all(p.exists() for p in _REQUIRED):
    pytest.skip("real dataset not available", allow_module_level=True)


@pytest.fixture(scope="module")
def legacy():
    """Run the legacy pipeline end to end and return everything it produced."""
    from darts import TimeSeries  # noqa: PLC0415
    from src import (  # noqa: PLC0415
        build_ts_and_apply_window_transformer,
        get_covs_and_encodings,
        get_engineered_features,
        halflife_to_alpha,
        load_data,
        split_future_and_past_cov,
        subset_safe,
    )

    regions, master_timeseries, regions_activity = load_data(
        data_path=str(FIXED_DATA_PATH), dataset_path=str(DATASET_PATH)
    )
    panel, global_weather_columns = get_engineered_features(
        master_timeseries=master_timeseries,
        data_path=str(FIXED_DATA_PATH),
        target_col=TARGET,
        regions=regions,
        regions_activity=regions_activity,
        binarize_target=False,
    )
    _, future_covariates, _, past_covariates = split_future_and_past_cov(
        panel, global_weather_columns, TARGET
    )

    target_series_list, past_covs_list, future_covs_list = (
        build_ts_and_apply_window_transformer(
            panel, TARGET, past_covariates, future_covariates, ed_alpha=halflife_to_alpha(7)
        )
    )
    raw_past_covs_list = TimeSeries.from_group_dataframe(
        panel, group_cols="region", time_col="event_date", value_cols=past_covariates
    )
    nine = get_covs_and_encodings(
        target_series_list, past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC
    )
    (
        region_names, train_target, val_target, test_target,
        full_past_covs, full_fut_covs, target_for_cv, train_val_end, cv_start_val,
    ) = nine
    # the notebooks' second call, for the RNN raw past covariates
    full_raw_past_covs = get_covs_and_encodings(
        target_series_list, raw_past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC
    )[4]
    # the notebooks' `target_full`
    target_full = [
        tr.append(vl).append(te)
        for tr, vl, te in zip(train_target, val_target, test_target, strict=True)
    ]

    return {
        "panel": panel,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
        "regions_activity": regions_activity,
        "target_series_list": target_series_list,
        "past_covs_list": past_covs_list,
        "future_covs_list": future_covs_list,
        "region_names": region_names,
        "target_full": target_full,
        "target_train": train_target,
        "target_val": val_target,
        "target_test": test_target,
        "target_cv_view": target_for_cv,
        "past_covs": full_past_covs,
        "future_covs": full_fut_covs,
        "raw_past_covs": full_raw_past_covs,
        "train_val_end": train_val_end,
        "cv_start_frac": cv_start_val,
        "subset_safe": subset_safe,
        "get_covs_and_encodings": get_covs_and_encodings,
    }


@pytest.fixture(scope="module")
def bundle(legacy):
    from strikecast.config.schema import SeriesConfig, WindowTransformConfig  # noqa: PLC0415
    from strikecast.data.series import build_bundle  # noqa: PLC0415

    return build_bundle(
        panel=legacy["panel"],
        target=TARGET,
        past_covariates=legacy["past_covariates"],
        future_covariates=legacy["future_covariates"],
        config=SeriesConfig(window=WindowTransformConfig(expdecay="legacy_alpha")),  # thesis: legacy expdecay7 (A1/D1)
        activity_by_region=legacy["regions_activity"],
    )


def assert_series_lists_equal(new, old, label: str) -> None:
    assert len(new) == len(old), f"{label}: list length"
    for i, (a, b) in enumerate(zip(new, old, strict=True)):
        np.testing.assert_array_equal(a.values(), b.values(), err_msg=f"{label}[{i}] values")
        assert a.time_index.equals(b.time_index), f"{label}[{i}] time index"
        assert list(a.components) == list(b.components), f"{label}[{i}] components"
        pd.testing.assert_frame_equal(
            a.static_covariates, b.static_covariates, obj=f"{label}[{i}] static covariates"
        )


@pytest.mark.parametrize(
    "attr",
    [
        "target_full",
        "target_train",
        "target_val",
        "target_test",
        "target_cv_view",
        "past_covs",
        "raw_past_covs",
        "future_covs",
    ],
)
def test_series_lists_match_legacy(bundle, legacy, attr) -> None:
    assert_series_lists_equal(getattr(bundle, attr), legacy[attr], attr)


def test_region_names_match_legacy(bundle, legacy) -> None:
    assert bundle.region_names == list(legacy["region_names"])


def test_fractions_match_legacy(bundle, legacy) -> None:
    assert bundle.train_val_end == legacy["train_val_end"]
    assert bundle.cv_start_frac == legacy["cv_start_frac"]
    assert bundle.fractions.train == TRAIN_FRAC
    assert bundle.fractions.val == VAL_FRAC


def test_activity_by_region_carried(bundle, legacy) -> None:
    assert bundle.activity_by_region == dict(legacy["regions_activity"])


def test_raw_lists_are_unencoded(bundle, legacy) -> None:
    """The bundle keeps the pre-encoding lists that feature selection re-uses."""
    assert bundle.raw is not None
    assert_series_lists_equal(bundle.raw.target, legacy["target_series_list"], "raw.target")
    assert_series_lists_equal(bundle.raw.past, legacy["past_covs_list"], "raw.past")
    assert_series_lists_equal(bundle.raw.future, legacy["future_covs_list"], "raw.future")


def test_subset_components_matches_legacy(bundle, legacy) -> None:
    """The post-feature-selection re-encode, as the notebooks perform it."""
    subset_safe = legacy["subset_safe"]
    get_covs_and_encodings = legacy["get_covs_and_encodings"]

    # stand-in for `clean_feature_names(top_100)`: sets, as the legacy call uses
    past_keep = set(list(legacy["past_covs_list"][0].components)[:40])
    future_keep = set(list(legacy["future_covs_list"][0].components)[:10]) | {"not_a_feature"}

    past_subset = [subset_safe(ts, past_keep) for ts in legacy["past_covs_list"]]
    future_subset = [subset_safe(ts, future_keep) for ts in legacy["future_covs_list"]]
    (
        region_names, train_target, val_target, test_target,
        full_past_covs, full_fut_covs, target_for_cv, train_val_end, cv_start_val,
    ) = get_covs_and_encodings(
        legacy["target_series_list"], past_subset, future_subset, TRAIN_FRAC, VAL_FRAC
    )

    from strikecast.data.series import subset_components  # noqa: PLC0415

    new = subset_components(bundle, past_keep, future_keep)

    assert new.region_names == list(region_names)
    assert new.train_val_end == train_val_end
    assert new.cv_start_frac == cv_start_val
    assert_series_lists_equal(new.target_train, train_target, "subset target_train")
    assert_series_lists_equal(new.target_val, val_target, "subset target_val")
    assert_series_lists_equal(new.target_test, test_target, "subset target_test")
    assert_series_lists_equal(new.target_cv_view, target_for_cv, "subset target_cv_view")
    assert_series_lists_equal(new.past_covs, full_past_covs, "subset past_covs")
    assert_series_lists_equal(new.future_covs, full_fut_covs, "subset future_covs")
    # the RNN raw past covariates are untouched by feature selection
    assert_series_lists_equal(new.raw_past_covs, legacy["raw_past_covs"], "subset raw_past_covs")
