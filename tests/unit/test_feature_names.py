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


# --------------------------------------------------------------------------- #
# the figure protocol's strict parser (plan 2026-09-28; resolves Q2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("foo_pastcov_lag-1", ("foo", -1)),
        ("ewm_ewma14_act_target_count_pastcov_lag-14", ("ewm_ewma14_act_target_count", -14)),
        # a base name that itself contains `_pastcov`: only the TRAILING one splits
        ("x_pastcov_y_pastcov_lag-7", ("x_pastcov_y", -7)),
        ("a_futcov_b_pastcov_lag-7", ("a_futcov_b", -7)),
        # never in the pool: target lags, future covs, statics, encoders, junk
        ("x_target_lag-7", None),
        ("bar_futcov_lag2", None),
        ("darts_enc_fc_cyc_month_sin_futcov_lag0", None),
        ("region_statcov_target_x", None),
        ("foo_pastcov_lag3", None),  # past lags are negative
        ("foo_pastcov_lag-1_extra", None),
        ("_pastcov_lag-1", None),
        ("", None),
    ],
)
def test_parse_pastcov_lag_is_strict_and_trailing(name, expected):
    from strikecast.data.feature_selection import parse_pastcov_lag

    assert parse_pastcov_lag(name) == expected


def test_the_strict_parser_differs_from_the_legacy_one_only_where_q2_bites():
    """The legacy parser truncates at the FIRST suffix; the strict one does not."""
    from strikecast.data.feature_selection import parse_pastcov_lag

    name = "x_pastcov_y_pastcov_lag-7"
    assert parse_feature_names([name])["pastcov_features_base"] == {"x"}  # Q2
    assert parse_pastcov_lag(name) == ("x_pastcov_y", -7)
    # a mid-name `_target` is harmless to both (`_pastcov` is tried first)
    name = "ewm_act_target_count_pastcov_lag-1"
    assert parse_feature_names([name])["pastcov_features_base"] == {"ewm_act_target_count"}
    assert parse_pastcov_lag(name) == ("ewm_act_target_count", -1)
