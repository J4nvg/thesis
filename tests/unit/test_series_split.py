"""Unit tests for `strikecast.data.series` on a synthetic panel.

Fixtures are defined locally on purpose: `tests/conftest.py` is owned by another
work stream and this file must not depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strikecast.config.schema import SeriesConfig, SplitConfig, halflife_to_alpha
from strikecast.data import series as S

N_DAYS = 100
REGIONS = ["alpha", "beta", "gamma"]
TARGET = "y"
PAST_COVS = ["p1", "p2"]
FUTURE_COVS = ["f1"]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=N_DAYS, freq="D")
    frames = []
    for i, region in enumerate(REGIONS):
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "event_date": dates,
                    "Activity_Level": f"level_{i % 2}",
                    TARGET: rng.poisson(1.0, N_DAYS).astype(float),
                    "p1": rng.normal(size=N_DAYS),
                    "p2": rng.normal(size=N_DAYS),
                    "f1": np.arange(N_DAYS, dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def bundle(panel: pd.DataFrame) -> S.SeriesBundle:
    return S.build_bundle(
        panel=panel,
        target=TARGET,
        past_covariates=PAST_COVS,
        future_covariates=FUTURE_COVS,
        config=SeriesConfig(),
        activity_by_region={r: i % 2 for i, r in enumerate(REGIONS)},
    )


def test_region_names_are_pre_encoding(bundle: S.SeriesBundle) -> None:
    assert bundle.region_names == REGIONS
    # after encoding, `region` is a numeric code, not the name
    statics = bundle.target_full[0].static_covariates
    assert statics is not None
    assert statics["region"].iloc[0] == 0.0


def test_split_lengths(bundle: S.SeriesBundle) -> None:
    # split_after(0.7) -> 70 / 30, then split_after(1/3) -> 10 / 20
    for i in range(len(REGIONS)):
        assert len(bundle.target_full[i]) == N_DAYS
        assert len(bundle.target_train[i]) == 70
        assert len(bundle.target_val[i]) == 10
        assert len(bundle.target_test[i]) == 20


def test_cv_view_length_and_fractions(bundle: S.SeriesBundle) -> None:
    assert bundle.train_val_end == 0.70 + 0.10  # 0.7999999999999999 (Q1)
    assert bundle.cv_start_frac == pytest.approx(0.875)
    for i in range(len(REGIONS)):
        # Q1: split_before(0.7999999999999999) truncates to 79, NOT to the 80
        # points that train (70) + val (10) cover. The CV view therefore ends
        # one day BEFORE the end of val. Preserved, not fixed.
        assert len(bundle.target_cv_view[i]) == 79
        assert len(bundle.target_train[i]) + len(bundle.target_val[i]) == 80
        assert bundle.target_cv_view[i].end_time() < bundle.target_val[i].end_time()  # type: ignore[operator]


def test_window_transform_component_counts(bundle: S.SeriesBundle) -> None:
    # keep_non_transformed=True -> raw + 6 transforms, all per component
    assert bundle.raw_past_covs[0].n_components == len(PAST_COVS)
    assert bundle.past_covs[0].n_components == len(PAST_COVS) * 7
    assert bundle.future_covs[0].n_components == len(FUTURE_COVS)


def test_default_window_transform_alpha() -> None:
    transforms = SeriesConfig().window.transforms
    assert [t["function_name"] for t in transforms] == [
        "rsum14", "rsum7", "rmean7", "rmean28", "ewma14", "expdecay7",
    ]
    assert transforms[-1]["alpha"] == halflife_to_alpha(7)


def test_split_config_derived_fractions() -> None:
    split = SplitConfig()
    assert split.train_val_end == 0.7999999999999999
    assert split.cv_start_frac == 0.7 / 0.7999999999999999


def test_positive_only_weights(bundle: S.SeriesBundle) -> None:
    weights = S.positive_only_weights(bundle.target_train)
    for w, t in zip(weights, bundle.target_train, strict=True):
        expected = (t.values().ravel() > 0).astype(float)
        np.testing.assert_array_equal(w.values().ravel(), expected)
        assert w.time_index.equals(t.time_index)


def test_subset_components(bundle: S.SeriesBundle) -> None:
    assert bundle.raw is not None
    past_keep = [c for c in bundle.raw.past[0].components if c.endswith("p1")]
    future_keep = ["f1", "does_not_exist"]
    sub = S.subset_components(bundle, past_keep, future_keep)

    assert list(sub.past_covs[0].components) == list(past_keep)
    assert list(sub.future_covs[0].components) == ["f1"]
    # the target side and the un-windowed raw past covariates are untouched
    np.testing.assert_array_equal(sub.target_full[0].values(), bundle.target_full[0].values())
    np.testing.assert_array_equal(
        sub.raw_past_covs[0].values(), bundle.raw_past_covs[0].values()
    )
    assert sub.region_names == bundle.region_names


def test_autogluon_stub(bundle: S.SeriesBundle) -> None:
    with pytest.raises(NotImplementedError):
        S.to_autogluon_frame(bundle)


def test_serialise_roundtrip(bundle: S.SeriesBundle, tmp_path) -> None:
    S.serialise(bundle, tmp_path / "bundle")
    loaded = S.load(tmp_path / "bundle")

    assert loaded.region_names == bundle.region_names
    assert loaded.activity_by_region == bundle.activity_by_region
    assert loaded.train_val_end == bundle.train_val_end
    assert loaded.cv_start_frac == bundle.cv_start_frac
    assert loaded.fractions == bundle.fractions

    for attr, _ in S._BUNDLE_LISTS:
        new, old = getattr(loaded, attr), getattr(bundle, attr)
        assert len(new) == len(old)
        for a, b in zip(new, old, strict=True):
            np.testing.assert_array_equal(a.values(), b.values())
            assert a.time_index.equals(b.time_index)
            assert list(a.components) == list(b.components)
            pd.testing.assert_frame_equal(a.static_covariates, b.static_covariates)

    assert loaded.raw is not None
    assert bundle.raw is not None
    for name in ("target", "past", "future", "raw_past"):
        for a, b in zip(getattr(loaded.raw, name), getattr(bundle.raw, name), strict=True):
            np.testing.assert_array_equal(a.values(), b.values())
            assert list(a.components) == list(b.components)

    # a loaded bundle still supports the post-feature-selection step
    sub = S.subset_components(loaded, ["p1"], ["f1"])
    assert list(sub.future_covs[0].components) == ["f1"]
