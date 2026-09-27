"""Unit tests for `strikecast.data.series` on a synthetic panel.

Fixtures are defined locally on purpose: `tests/conftest.py` is owned by another
work stream and this file must not depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strikecast.config.schema import (
    SeriesConfig,
    SplitConfig,
    WindowTransformConfig,
    halflife_to_alpha,
)
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


def test_legacy_window_transform_alpha() -> None:
    """``legacy_alpha`` is the thesis' list bit for bit (A1: the buggy filter)."""
    transforms = SeriesConfig(window=WindowTransformConfig(expdecay="legacy_alpha")).window.transforms
    assert [t["function_name"] for t in transforms] == [
        "rsum14", "rsum7", "rmean7", "rmean28", "ewma14", "expdecay7",
    ]
    assert transforms[-1] == {
        "function": "mean", "mode": "ewm", "alpha": halflife_to_alpha(7),
        "function_name": "expdecay7",
    }


def test_default_window_transform_is_the_leaky_integrator() -> None:
    """Decision D1: the schema default is the FIXED filter, named ``leaky7``."""
    window = SeriesConfig().window
    assert window.expdecay == "leaky"
    assert [t["function_name"] for t in window.transforms] == [
        "rsum14", "rsum7", "rmean7", "rmean28", "ewma14", "leaky7",
    ]
    assert window.transforms[-1] == {
        "function": "sum", "mode": "ewm", "halflife": 7.0, "function_name": "leaky7",
    }
    # the first five transforms do not depend on the mode
    legacy = WindowTransformConfig(expdecay="legacy_alpha")
    assert window.transforms[:5] == legacy.transforms[:5]


def _apply_expdecay(mode: str, values: np.ndarray) -> np.ndarray:
    from darts import TimeSeries  # noqa: PLC0415
    from darts.dataprocessing.transformers import WindowTransformer  # noqa: PLC0415

    window = WindowTransformConfig(expdecay=mode)  # type: ignore[arg-type]
    kwargs = window.transformer_kwargs()
    kwargs["transforms"] = [window.transforms[-1]]
    index = pd.date_range("2022-01-01", periods=len(values), freq="D")
    ts = TimeSeries.from_times_and_values(index, values, columns=["x"])
    out = WindowTransformer(**kwargs).transform(ts)
    name = window.transforms[-1]["function_name"]
    (column,) = [c for c in out.components if name in c]
    return out[column].values().ravel()


def _leaky_recursion(values: np.ndarray) -> np.ndarray:
    decay = 2 ** (-1 / 7)
    out, state = np.zeros_like(values), 0.0
    for t, x in enumerate(values):
        state = x + decay * state
        out[t] = state
    return out


def test_leaky7_is_the_literal_leaky_integrator() -> None:
    """``s_t = x_t + 2^(-1/7) s_{t-1}``, ``s_{-1} = 0``, EXACTLY (not approx)."""
    impulse = np.zeros(60)
    impulse[3] = 1.0
    got = _apply_expdecay("leaky", impulse)
    np.testing.assert_array_equal(got, _leaky_recursion(impulse))
    # the thesis' impulse response: 1 on the day, 0.906 a day later, 0.5 after 7 days
    assert got[3] == 1.0
    assert got[4] == pytest.approx(0.9057236642639067, abs=1e-15)
    assert got[10] == pytest.approx(0.5, abs=1e-12)

    rng = np.random.default_rng(7)
    noise = rng.poisson(2.0, 200).astype(float) * rng.random(200) + 3.0
    np.testing.assert_array_equal(_apply_expdecay("leaky", noise), _leaky_recursion(noise))


def test_legacy_expdecay7_is_almost_the_raw_series() -> None:
    """The legacy filter weights TODAY with alpha=0.906 (A1): 0.906, 0.085, ~0."""
    impulse = np.zeros(30)
    impulse[3] = 1.0
    got = _apply_expdecay("legacy_alpha", impulse)
    assert got[3] == pytest.approx(0.906, abs=1e-3)  # adjust=True start-up
    assert got[4] == pytest.approx(0.0854, abs=1e-3)
    assert got[10] < 1e-6


def test_window_transforms_must_agree_with_expdecay() -> None:
    legacy = WindowTransformConfig(expdecay="legacy_alpha").transforms
    with pytest.raises(ValueError, match="legacy_alpha"):
        WindowTransformConfig(expdecay="leaky", transforms=legacy)
    tampered = [*legacy[:5], {**legacy[5], "alpha": 0.5}]
    with pytest.raises(ValueError, match="differs"):
        WindowTransformConfig(expdecay="legacy_alpha", transforms=tampered)
    # an explicit, consistent list is accepted and a JSON round trip is stable
    window = WindowTransformConfig(expdecay="legacy_alpha", transforms=legacy)
    assert WindowTransformConfig(**window.model_dump(mode="json")) == window


def test_expdecay_mode_changes_every_identity() -> None:
    """The resolved config hash (stage identity) differs between the modes."""
    from strikecast.config.schema import ExperimentConfig  # noqa: PLC0415

    leaky = ExperimentConfig(name="count")
    legacy = ExperimentConfig(
        name="count", series=SeriesConfig(window=WindowTransformConfig(expdecay="legacy_alpha"))
    )
    assert leaky.resolved_hash() != legacy.resolved_hash()
    for cfg in (leaky, legacy):
        again = ExperimentConfig(**cfg.model_dump(mode="json"))
        assert again.resolved_hash() == cfg.resolved_hash()
        assert again.series.window.expdecay == cfg.series.window.expdecay


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
