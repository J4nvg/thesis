"""Level A: `build_panel` must reproduce every legacy panel variant exactly."""

from __future__ import annotations

import pandas as pd
import pytest

from strikecast.data import (
    build_panel_legacy_chronos,
    build_panel_legacy_damage,
    build_panel_legacy_regressor,
)

pytestmark = pytest.mark.golden

TARGET = "act_drone_strike_on_ua"
TARGET_BINARY = "act_drone_strike_on_ua_binary"
DAMAGE_TARGETS = [
    f"act_drone_infra_ua_{x}_intent"
    for x in ("health", "education", "residential", "energy")
]


def _fresh_master(inputs):
    """A pristine copy: the legacy functions mutate their argument (panel quirk Q1)."""
    return inputs.master.copy()


def test_regressor_panel_matches_legacy(inputs, legacy_src, fixed_dir):
    legacy_df, legacy_gwc = legacy_src.get_engineered_features(
        master_timeseries=_fresh_master(inputs),
        data_path=str(fixed_dir),
        target_col=TARGET,
        regions=inputs.regions,
        regions_activity=inputs.activity_by_region,
        binarize_target=False,
    )
    new = build_panel_legacy_regressor(inputs, target_col=TARGET)

    pd.testing.assert_frame_equal(new.panel, legacy_df, check_dtype=True, check_like=False)
    assert new.global_weather_columns == legacy_gwc


def test_regressor_panel_binarized_matches_legacy(inputs, legacy_src, fixed_dir):
    legacy_df, legacy_gwc = legacy_src.get_engineered_features(
        master_timeseries=_fresh_master(inputs),
        data_path=str(fixed_dir),
        target_col=TARGET_BINARY,
        regions=inputs.regions,
        regions_activity=inputs.activity_by_region,
        binarize_target=True,
        target_raw_col=TARGET,
    )
    new = build_panel_legacy_regressor(
        inputs, target_col=TARGET_BINARY, binarize_target=True, target_raw_col=TARGET
    )

    pd.testing.assert_frame_equal(new.panel, legacy_df, check_dtype=True, check_like=False)
    assert new.global_weather_columns == legacy_gwc
    # Quirk Q4: the binary column lands BEFORE the interaction features here.
    cols = list(new.panel.columns)
    assert cols.index(TARGET_BINARY) < cols.index("ratio_ua_rus_armed_clash")
    assert TARGET not in cols


def test_damage_panels_match_legacy(inputs, legacy_src, fixed_dir):
    legacy_dfs, legacy_gwc = legacy_src.get_engineered_features_damageclassifiers(
        master_timeseries=_fresh_master(inputs),
        data_path=str(fixed_dir),
        targets_cols=DAMAGE_TARGETS,
        regions=inputs.regions,
        regions_activity=inputs.activity_by_region,
    )
    new = build_panel_legacy_damage(inputs, targets_cols=DAMAGE_TARGETS)

    assert len(new.panels) == len(legacy_dfs) == len(DAMAGE_TARGETS)
    for target, new_df, legacy_df in zip(DAMAGE_TARGETS, new.panels, legacy_dfs, strict=True):
        pd.testing.assert_frame_equal(new_df, legacy_df, check_dtype=True, check_like=False)
        # Quirk Q4: here the binary column is LAST.
        assert list(new_df.columns)[-1] == f"{target}_binary"
    assert new.global_weather_columns == legacy_gwc


def _legacy_chronos_panel(master, regions, regions_activity, actors, legacy_src):
    """The inline copy in `_chronos2.py` lines 125-190, transcribed verbatim."""
    locale_env_weather_columns = [
        c for c in master.columns if "env" in c and "holiday" not in c
    ]
    global_weather_columns = list(
        set(
            "_".join(c.split("_")[:-1]) if c != "env_k_max" else c
            for c in locale_env_weather_columns
        )
    )
    master["act_total_daily_strike_events"] = master[
        [x for x in master.columns.tolist() if "act_drone_strike_on_ua" in x]
    ].sum(axis=1)
    master["act_total_damage_events"] = master[
        [x for x in master.columns.tolist() if "act_drone_infra_ua" in x]
    ].sum(axis=1)

    for_global = legacy_src.get_pivoted_table(df=master.reset_index(), regions=regions)
    for_global["Activity_Level"] = for_global.index.get_level_values("region").map(
        regions_activity
    )
    for_global = for_global[for_global["Activity_Level"] != 0]
    for_global, _removed = legacy_src.remove_low_prevalence(
        df=for_global, ratio=0.1, specific="acled_other_"
    )

    gdelt_cols = (
        set(legacy_src.get_all_X("com_", for_global))
        - set(legacy_src.get_all_X("com_ners", for_global))
        - set(legacy_src.get_all_X("com_aid", for_global))
    )
    for_global = legacy_src.aggregate_gdelt_primary_secondary(
        df=for_global, columns=gdelt_cols, actors=actors
    )

    infra_cols = legacy_src.get_all_X("act_drone_infra_ua_", for_global)
    for_global[infra_cols] = for_global[infra_cols].fillna(0)
    return for_global, global_weather_columns


def test_chronos_panel_matches_legacy(inputs, legacy_src):
    legacy_df, legacy_gwc = _legacy_chronos_panel(
        _fresh_master(inputs),
        inputs.regions,
        inputs.activity_by_region,
        inputs.actors,
        legacy_src,
    )
    new = build_panel_legacy_chronos(inputs, target="act_drone_strike_on_ua")

    pd.testing.assert_frame_equal(new.panel, legacy_df, check_dtype=True, check_like=False)
    assert new.global_weather_columns == legacy_gwc
    # Quirk Q3: the MultiIndex is kept and the stray "index" column survives.
    assert new.panel.index.names == ["region", "event_date"]
    assert "index" in new.panel.columns
    # And the interaction features are absent for this variant.
    assert "ratio_ua_rus_armed_clash" not in new.panel.columns
    assert "dist_x_clash" not in new.panel.columns


def test_build_panel_does_not_mutate_inputs(inputs):
    """Quirk Q1: unlike the legacy functions, build_panel leaves `inputs.master` alone."""
    before = list(inputs.master.columns)
    build_panel_legacy_regressor(inputs, target_col=TARGET)
    assert list(inputs.master.columns) == before
    assert "act_total_daily_strike_events" not in inputs.master.columns


def test_activity_level_zero_regions_dropped(inputs):
    """Quirk Q5: tier-0 regions are removed; Activity_Level stays as a column."""
    panel = build_panel_legacy_regressor(inputs, target_col=TARGET).panel
    assert "Activity_Level" in panel.columns
    assert (panel["Activity_Level"] != 0).all()
    kept = set(panel["region"].unique())
    zero_regions = {r for r, t in inputs.activity_by_region.items() if t == 0}
    assert not (kept & zero_regions)
