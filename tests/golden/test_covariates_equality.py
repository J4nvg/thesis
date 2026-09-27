"""Level A: `split_covariates` must reproduce `split_future_and_past_cov` exactly."""

from __future__ import annotations

import pytest

from strikecast.data import (
    build_panel_legacy_chronos,
    build_panel_legacy_regressor,
    split_covariates,
)

pytestmark = pytest.mark.golden

TARGET = "act_drone_strike_on_ua"
TARGET_BINARY = "act_drone_strike_on_ua_binary"


def _assert_same(legacy, new):
    l_hol, l_fut, l_excl, l_past = legacy
    assert new.holiday_cols == l_hol
    assert new.future_covariates == l_fut
    assert set(new.exclude_cols) == set(l_excl)
    assert new.past_covariates == l_past
    # tuple-unpacking parity with the legacy return value
    a, b, c, d = new
    assert (a, b, set(c), d) == (l_hol, l_fut, set(l_excl), l_past)


def test_split_matches_legacy_reset_panel(inputs, legacy_src):
    res = build_panel_legacy_regressor(inputs, target_col=TARGET)
    legacy = legacy_src.split_future_and_past_cov(
        res.panel, res.global_weather_columns, TARGET
    )
    _assert_same(legacy, split_covariates(res.panel, res.global_weather_columns, TARGET))


def test_split_matches_legacy_binary_panel(inputs, legacy_src):
    res = build_panel_legacy_regressor(
        inputs, target_col=TARGET_BINARY, binarize_target=True, target_raw_col=TARGET
    )
    legacy = legacy_src.split_future_and_past_cov(
        res.panel, res.global_weather_columns, TARGET_BINARY
    )
    _assert_same(
        legacy, split_covariates(res.panel, res.global_weather_columns, TARGET_BINARY)
    )


def test_split_matches_legacy_multiindex_panel(inputs, legacy_src):
    """`_chronos2.py` passes the MultiIndexed frame, not the reset one."""
    res = build_panel_legacy_chronos(inputs, target=TARGET)
    legacy = legacy_src.split_future_and_past_cov(
        res.panel, res.global_weather_columns, TARGET
    )
    _assert_same(legacy, split_covariates(res.panel, res.global_weather_columns, TARGET))


def test_split_with_explicit_exclude_matches_legacy(inputs, legacy_src):
    res = build_panel_legacy_regressor(inputs, target_col=TARGET)
    exclude = {TARGET, "region", "event_date"}
    legacy = legacy_src.split_future_and_past_cov(
        res.panel, res.global_weather_columns, TARGET, exclude=exclude
    )
    _assert_same(
        legacy,
        split_covariates(res.panel, res.global_weather_columns, TARGET, exclude=exclude),
    )


def test_empty_exclude_falls_back_to_default(inputs, legacy_src):
    """Legacy quirk: an empty `exclude` is falsy, so the default set is used."""
    res = build_panel_legacy_regressor(inputs, target_col=TARGET)
    legacy = legacy_src.split_future_and_past_cov(
        res.panel, res.global_weather_columns, TARGET, exclude=set()
    )
    new = split_covariates(res.panel, res.global_weather_columns, TARGET, exclude=set())
    _assert_same(legacy, new)
    assert "Activity_Level" in new.exclude_cols
