"""`parse_feature_names` must match the legacy `clean_feature_names` exactly."""

from __future__ import annotations

import pytest

from strikecast.data.feature_selection import parse_feature_names

# A handful of synthetic darts lagged-feature names, plus one that matches no
# suffix at all (darts' cyclic encoder columns look like this in practice).
SYNTHETIC_NAMES = [
    "foo_pastcov_lag-1",
    "bar_futcov_lag2",
    "x_target_lag-7",
    "datetime_attribute_month_sin",  # unmatched
]

# More adversarial names, to pin the quirks down.
QUIRKY_NAMES = [
    "region_statcov",  # static covariate, no lag suffix
    "ewm_ewma14_act_target_count_pastcov_lag-14",  # '_target' occurs mid-name
    "rolling_rsum7_7_com_x_futcov_lag5",
    "",  # empty string: matches nothing
]


def _legacy_clean_feature_names():
    """Import the legacy implementation, skipping if the old package is gone."""
    legacy = pytest.importorskip("src.feature_tools", reason="legacy `src` package not present")
    return legacy.clean_feature_names


@pytest.mark.parametrize(
    "names",
    [SYNTHETIC_NAMES, QUIRKY_NAMES, SYNTHETIC_NAMES + QUIRKY_NAMES, []],
    ids=["synthetic", "quirky", "combined", "empty"],
)
def test_matches_legacy(names):
    legacy = _legacy_clean_feature_names()
    assert parse_feature_names(names) == legacy(list(names))


def test_expected_buckets():
    got = parse_feature_names(SYNTHETIC_NAMES)
    assert got["pastcov_features_base"] == {"foo"}
    assert got["pastcov_features_actual"] == {"foo_pastcov_lag-1"}
    assert got["futcov_features_base"] == {"bar"}
    assert got["futcov_features_actual"] == {"bar_futcov_lag2"}
    assert got["target_features_base"] == {"x"}
    assert got["statcov_features_base"] == set()
    assert got["features_base"] == {"foo", "bar", "x"}
    # the unmatched name is silently dropped from every bucket
    assert "datetime_attribute_month_sin" not in got["actual_features"]


def test_empty_input_returns_all_empty_buckets():
    got = parse_feature_names([])
    assert set(got) == {
        "futcov_features_base",
        "futcov_features_actual",
        "pastcov_features_base",
        "pastcov_features_actual",
        "statcov_features_base",
        "statcov_features_actual",
        "target_features_base",
        "target_features_actual",
        "features_base",
        "actual_features",
    }
    assert all(v == set() for v in got.values())


def test_quirk_suffix_matches_mid_name():
    """Q2: '_target' is found before '_pastcov' is ever considered.

    The suffixes are tried in the order pastcov, futcov, statcov, target, so a
    name containing both is bucketed by whichever comes first in that list --
    here '_pastcov' wins and the '_target' inside the base name survives.
    """
    got = parse_feature_names(["ewm_ewma14_act_target_count_pastcov_lag-14"])
    assert got["pastcov_features_base"] == {"ewm_ewma14_act_target_count"}
    assert got["target_features_base"] == set()


def test_quirk_statcov_without_lag():
    got = parse_feature_names(["region_statcov"])
    assert got["statcov_features_base"] == {"region"}
    assert got["statcov_features_actual"] == {"region_statcov"}
