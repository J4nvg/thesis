"""The ONE feature-engineering function.

This module replaces three near-duplicate copies of the same pipeline:

=========================================  =========================================
Legacy                                     Wrapper here
=========================================  =========================================
``prevalent_functions.get_engineered_features``            :func:`build_panel_legacy_regressor`
(``binarize_target=False`` and ``=True``)
``prevalent_functions.get_engineered_features_damageclassifiers``  :func:`build_panel_legacy_damage`
``_chronos2.py`` lines 125-190 (inline copy)               :func:`build_panel_legacy_chronos`
=========================================  =========================================

All three are thin calls into :func:`build_panel`, which has three explicit
switches: ``add_interactions``, ``binarize`` (+ ``binarize_stage``) and
``keep_index``.

Pipeline (the numbering follows the legacy comment block)
---------------------------------------------------------
1. Identify the weather columns -> ``global_weather_columns``.
2. Add the two national aggregates ``act_total_daily_strike_events`` and
   ``act_total_damage_events``.
3. Pivot wide -> long-hierarchical on ``(region, event_date)``.
4. Map ``Activity_Level`` from the region and drop tier-0 ("little") regions.
5. Drop low-prevalence ``acled_other_*`` columns.
6. *(regressor with ``binarize_target=True`` only)* binarise the target and
   drop the raw column -- this happens **before** step 7.
7. *(not Chronos)* ratio / interaction features.
8. Aggregate the GDELT primary/secondary actor columns.
9. Fill ``act_drone_infra_ua_*`` NaNs with 0.
10. *(damage only)* one frame per target, each binarised **after** steps 7-9.
11. ``reset_index()`` and drop the stray ``"index"`` column, unless
    ``keep_index=True`` (Chronos keeps the MultiIndex).

Legacy quirks preserved on purpose
----------------------------------
Q1. **In-place mutation of the input.** The legacy functions add
    ``act_total_daily_strike_events`` and ``act_total_damage_events`` directly
    to the caller's ``master_timeseries``. :func:`build_panel` copies first and
    never mutates ``inputs.master``, but produces the identical result. (The
    legacy mutation happens to be idempotent: neither new column name contains
    ``act_drone_strike_on_ua`` or ``act_drone_infra_ua``, so re-running does
    not double-count.)
Q2. **``list(set(...))`` ordering.** ``global_weather_columns`` is built from a
    ``set`` of strings, and ``gdelt_cols`` likewise. Python randomises string
    hashing per interpreter process, so the ORDER of ``global_weather_columns``
    -- and therefore the order of ``future_covariates`` -- and the order in
    which GDELT columns are aggregated (hence the column order of the
    aggregated frame) is stable within one process but differs between
    processes. This is reproduced exactly, not fixed. Golden tests compare
    legacy and new inside the same process, where the orderings agree.
Q3. **The stray ``"index"`` column.** ``master`` has an unnamed, non-default
    integer index, so ``master.reset_index()`` creates a column called
    ``"index"``. It survives the pivot as a "global" column. The reset-index
    variants drop it at the very end; the Chronos variant does **not** -- it
    stays in the frame and is only excluded later by
    :func:`strikecast.data.covariates.split_covariates` via ``exclude_cols``.
Q4. **Binarisation position differs between variants.** The regressor variant
    binarises at step 6 (before the interaction features), so
    ``<target>_binary`` sits *before* ``ratio_ua_rus_armed_clash`` in the
    column order. The damage variant binarises at step 10 (after everything),
    so ``<target>_binary`` is the *last* column. Contrary to appearances the
    damage variant DOES add the ratio/interaction features -- only its
    binarisation is moved.
Q5. **``Activity_Level`` handling.** Regions are mapped through
    ``activity_by_region`` and then filtered with ``!= 0``, i.e. tier-0
    ("little" activity) regions are dropped. A region missing from the mapping
    becomes NaN, and ``NaN != 0`` is True, so it is *kept*. Preserved as-is.
    ``Activity_Level`` stays in the frame as a column and is later excluded
    from the covariates.
Q6. **Hard-coded ``low_prevalence_ratio=0.1`` and ``specific="acled_other_"``.**
    The ratio is exposed as a parameter (default 0.1, the legacy value); the
    prefix is not.
Q7. **No validation in the damage variant.** ``get_engineered_features`` raises
    if ``target_col`` is missing after engineering; the damage and Chronos
    copies do not check anything. Reproduced: ``validate_target`` defaults to
    True and is False for the damage/Chronos wrappers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from ._legacy_helpers import (
    aggregate_gdelt_primary_secondary,
    get_all_X,
    get_pivoted_table,
    remove_low_prevalence,
    to_binary_classification,
)
from .load import Inputs

logger = logging.getLogger(__name__)

__all__ = [
    "PanelResult",
    "build_panel",
    "build_panel_legacy_chronos",
    "build_panel_legacy_damage",
    "build_panel_legacy_regressor",
]

#: ``specific=`` argument the legacy code hard-codes for ``remove_low_prevalence``.
LOW_PREVALENCE_PREFIX = "acled_other_"


@dataclass(frozen=True)
class PanelResult:
    """Output of :func:`build_panel`.

    ``panels`` holds one dataframe per binarisation target for the damage
    variant, and exactly one otherwise. ``panel`` is the convenience accessor
    for the single-frame case -- the frame the legacy code calls
    ``for_global_reset`` (or ``for_global`` when ``keep_index=True``).
    """

    panels: list[pd.DataFrame] = field(repr=False)
    global_weather_columns: list[str]

    @property
    def panel(self) -> pd.DataFrame:
        if len(self.panels) != 1:
            raise ValueError(
                f"PanelResult holds {len(self.panels)} panels; use .panels instead of .panel"
            )
        return self.panels[0]


def _global_weather_columns(master: pd.DataFrame) -> list[str]:
    """Step 1, verbatim from the legacy code (see quirk Q2 about ordering)."""
    locale_env_weather_columns = [
        c for c in master.columns if "env" in c and "holiday" not in c
    ]
    return list(
        set(
            "_".join(c.split("_")[:-1]) if c != "env_k_max" else c
            for c in locale_env_weather_columns
        )
    )


def build_panel(
    inputs: Inputs,
    target: str,
    binarize: Sequence[str] = (),
    low_prevalence_ratio: float = 0.1,
    *,
    add_interactions: bool = True,
    binarize_stage: str = "early",
    keep_index: bool = False,
    validate_target: bool = True,
) -> PanelResult:
    """Build the long panel.

    Parameters
    ----------
    inputs:
        Loaded by :func:`strikecast.data.load.load_inputs`. Never mutated (Q1).
    target:
        The target column name, used only for the post-engineering existence
        check and the summary log line (legacy ``target_col``).
    binarize:
        Raw count columns to turn into ``<col>_binary``. The raw column is
        dropped afterwards, exactly as the legacy code does. Empty for the
        plain regressor and Chronos variants; one entry for the event
        classifier; four entries for the damage classifiers.
    low_prevalence_ratio:
        ``ratio`` for ``remove_low_prevalence`` on ``acled_other_*`` (Q6).
    add_interactions:
        Add ``ratio_ua_rus_armed_clash`` and ``dist_x_clash`` (step 7). True
        for the regressor and damage variants, False for Chronos.
    binarize_stage:
        ``"early"`` binarises at step 6, before the interaction features, and
        produces a single frame (regressor variant). ``"late"`` binarises at
        step 10 on a copy per target and produces one frame per target (damage
        variant). See Q4. Ignored when ``binarize`` is empty.
    keep_index:
        Keep the ``(region, event_date)`` MultiIndex instead of resetting it
        (Chronos variant). See Q3.
    validate_target:
        Raise if ``target`` is absent after engineering (Q7).
    """
    if binarize_stage not in ("early", "late"):
        raise ValueError(f"binarize_stage must be 'early' or 'late', got {binarize_stage!r}")
    binarize = list(binarize)
    if binarize_stage == "early" and len(binarize) > 1:
        raise ValueError(
            "binarize_stage='early' reproduces get_engineered_features, which binarises a "
            "single column; pass binarize_stage='late' for multiple targets"
        )

    # --- 1. weather columns ------------------------------------------------ #
    master = inputs.master.copy()  # Q1: the legacy code mutates its argument here.
    global_weather_columns = _global_weather_columns(master)

    # --- 2. national aggregates ------------------------------------------- #
    master["act_total_daily_strike_events"] = master[
        [x for x in master.columns.tolist() if "act_drone_strike_on_ua" in x]
    ].sum(axis=1)
    master["act_total_damage_events"] = master[
        [x for x in master.columns.tolist() if "act_drone_infra_ua" in x]
    ].sum(axis=1)

    # --- 3. long hierarchical ---------------------------------------------- #
    for_global = get_pivoted_table(df=master.reset_index(), regions=inputs.regions)

    # --- 4. activity tiers, drop tier 0 (Q5) ------------------------------- #
    for_global["Activity_Level"] = for_global.index.get_level_values("region").map(
        inputs.activity_by_region
    )
    for_global = cast(pd.DataFrame, for_global[for_global["Activity_Level"] != 0])

    # --- 5. low-prevalence ACLED "other" categories ------------------------ #
    for_global, _removed = remove_low_prevalence(
        df=for_global, ratio=low_prevalence_ratio, specific=LOW_PREVALENCE_PREFIX
    )

    # --- 6. early binarisation (regressor variant, Q4) --------------------- #
    if binarize and binarize_stage == "early":
        raw_col = binarize[0]
        for_global = to_binary_classification(for_global, raw_col)
        for_global = for_global.drop(columns=[raw_col])

    # --- 7. ratio / interaction features ----------------------------------- #
    if add_interactions:
        if (
            "acled_other_ua_armed_clash" in for_global.columns
            and "acled_other_rus_armed_clash" in for_global.columns
        ):
            for_global["ratio_ua_rus_armed_clash"] = for_global[
                "acled_other_ua_armed_clash"
            ] / (for_global["acled_other_rus_armed_clash"] + 1)

        dist_cols = get_all_X("dist_to_nearest_ru_km", for_global)
        if dist_cols and "acled_other_rus_armed_clash" in for_global.columns:
            for_global["dist_x_clash"] = (
                for_global[dist_cols[0]] * for_global["acled_other_rus_armed_clash"]
            )

    # --- 8. aggregate GDELT primary/secondary (Q2 ordering) ---------------- #
    gdelt_cols = (
        set(get_all_X("com_", for_global))
        - set(get_all_X("com_ners", for_global))
        - set(get_all_X("com_aid", for_global))
    )
    for_global = aggregate_gdelt_primary_secondary(
        df=for_global, columns=cast(list, gdelt_cols), actors=inputs.actors
    )

    # --- 9. fix NaN in infra ----------------------------------------------- #
    infra_cols = get_all_X("act_drone_infra_ua_", for_global)
    for_global[infra_cols] = for_global[infra_cols].fillna(0)

    # NOTE: the legacy code has a commented-out drop of the `fin_ch` columns
    # here. It is commented out there, so it is absent here.

    logger.info("Shape after feature engineering: %s", for_global.shape)

    # --- 10. late binarisation (damage variant, Q4) ------------------------ #
    if binarize and binarize_stage == "late":
        frames = []
        for raw_col in binarize:
            copied = for_global.copy()
            copied = to_binary_classification(copied, raw_col)
            copied = copied.drop(columns=[raw_col])
            logger.info("%s", raw_col)
            logger.info("Amount of target =1: %.3f", copied[f"{raw_col}_binary"].sum())
            logger.info("Total rows: %.3f", copied[f"{raw_col}_binary"].count())
            logger.info("Target positive rate: %.3f", copied[f"{raw_col}_binary"].mean())
            frames.append(copied)
    else:
        frames = [for_global]

    # --- target validation + summary log (Q7) ------------------------------ #
    if validate_target:
        if target not in frames[0].columns:
            raise ValueError(
                f"target_col={target!r} not found after feature engineering. "
                f"Available columns (first 20): {sorted(frames[0].columns.tolist())[:20]}"
            )
        if binarize:
            logger.info("Target positive rate: %.3f", frames[0][target].mean())
        else:
            logger.info("Target mean: %.3f", frames[0][target].mean())

    # --- 11. reset index (Q3) ---------------------------------------------- #
    if not keep_index:
        reset = []
        for frame in frames:
            out = frame.reset_index()
            if "index" in out.columns:
                out = out.drop(columns=["index"])
            reset.append(out)
        frames = reset

    return PanelResult(panels=frames, global_weather_columns=global_weather_columns)


# --------------------------------------------------------------------------- #
# Thin legacy wrappers
# --------------------------------------------------------------------------- #
def build_panel_legacy_regressor(
    inputs: Inputs,
    target_col: str,
    binarize_target: bool = False,
    target_raw_col: str | None = None,
    low_prevalence_ratio: float = 0.1,
) -> PanelResult:
    """Reproduce ``prevalent_functions.get_engineered_features``.

    Covers both ``binarize_target=False`` (count regressors, diff branch,
    Chronos-comparable GBDT panel, EDA) and ``binarize_target=True`` (the
    hurdle event classifier), where ``target_raw_col`` is binarised at step 6
    and dropped.
    """
    if binarize_target and target_raw_col is None:
        raise ValueError("You must provide 'target_raw_col' if 'binarize_target' is True.")
    binarize = [target_raw_col] if binarize_target and target_raw_col is not None else []
    return build_panel(
        inputs,
        target=target_col,
        binarize=binarize,
        low_prevalence_ratio=low_prevalence_ratio,
        add_interactions=True,
        binarize_stage="early",
        keep_index=False,
        validate_target=True,
    )


def build_panel_legacy_damage(
    inputs: Inputs,
    targets_cols: Sequence[str],
    low_prevalence_ratio: float = 0.1,
) -> PanelResult:
    """Reproduce ``prevalent_functions.get_engineered_features_damageclassifiers``.

    Returns one reset-index frame per entry of ``targets_cols``, in order, on
    ``PanelResult.panels``. Unlike the regressor variant it binarises *after*
    the interaction and GDELT steps (Q4) and performs no target validation
    (Q7).
    """
    return build_panel(
        inputs,
        target="",
        binarize=list(targets_cols),
        low_prevalence_ratio=low_prevalence_ratio,
        add_interactions=True,
        binarize_stage="late",
        keep_index=False,
        validate_target=False,
    )


def build_panel_legacy_chronos(
    inputs: Inputs,
    target: str,
    low_prevalence_ratio: float = 0.1,
) -> PanelResult:
    """Reproduce the inline copy in ``_chronos2.py`` (lines 125-190).

    Two differences from the regressor variant: the ratio/interaction features
    of step 7 are skipped, and the ``(region, event_date)`` MultiIndex is kept
    (so the stray ``"index"`` column also survives -- see Q3).
    """
    return build_panel(
        inputs,
        target=target,
        binarize=[],
        low_prevalence_ratio=low_prevalence_ratio,
        add_interactions=False,
        keep_index=True,
        validate_target=False,
    )
