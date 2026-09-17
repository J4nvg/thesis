"""Verbatim copies of the legacy helper functions from the top-level ``src`` package.

These are duplicated -- deliberately, character for character -- so that
``strikecast.data`` never imports the legacy ``src`` package while still
reproducing its behaviour bit for bit.

Provenance:
    ``get_pivoted_table``               src/dataset_specific_tools.py
    ``aggregate_gdelt_primary_secondary`` src/dataset_specific_tools.py
    ``remove_low_prevalence``           src/dataset_specific_tools.py
    ``load_region_activity_dictionary`` src/dataset_specific_tools.py
    ``load_actors_dict``                src/dataset_specific_tools.py
    ``to_binary_classification``        src/feature_tools.py
    ``get_all_X``                       src/general_tools.py
    ``get_list_from_file``              src/general_tools.py
    ``construct_path``                  src/general_tools.py

Do not "fix" anything here. Quirks that matter are documented in
``strikecast.data.panel``.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from pandas import DataFrame

__all__ = [
    "aggregate_gdelt_primary_secondary",
    "construct_path",
    "get_all_X",
    "get_list_from_file",
    "get_pivoted_table",
    "load_actors_dict",
    "load_region_activity_dictionary",
    "remove_low_prevalence",
    "to_binary_classification",
]


# --------------------------------------------------------------------------- #
# src/general_tools.py
# --------------------------------------------------------------------------- #
def get_list_from_file(filename):
    l = []  # noqa: E741
    with open(filename) as file:
        for line in file:
            l.append(line.strip().replace(",", "").lower())
    return l.copy()


def construct_path(*args):
    return os.path.join(*args)


def get_all_X(columtype, df):
    return [x for x in df.columns.tolist() if columtype in x.lower()]


# --------------------------------------------------------------------------- #
# src/feature_tools.py
# --------------------------------------------------------------------------- #
def to_binary_classification(df: pd.DataFrame, column: str) -> pd.DataFrame:
    to_apply = df.copy()

    if column not in to_apply.columns:
        raise ValueError(f"Column not found in DataFrame: {column}")

    to_apply[f"{column}_binary"] = (to_apply[column] > 0).astype(int)
    return to_apply


# --------------------------------------------------------------------------- #
# src/dataset_specific_tools.py
# --------------------------------------------------------------------------- #
def load_actors_dict(filepath: str = "actors.json") -> dict:
    """
    Loads the actors configuration from a JSON file.
    """
    with open(filepath) as file:
        actors_dict = json.load(file)
    return actors_dict


def load_region_activity_dictionary(filepath: str = "regions_activity_cat.json") -> dict:
    """
    Loads the region activity configuration from a JSON file.
    """
    r_dict = {}
    region_activity_dictionary = {}
    tier_mapping = {"little": 0, "low": 1, "medium": 2, "high": 3}
    with open(filepath) as file:
        region_activity_dictionary = json.load(file)

    for key, value in region_activity_dictionary.items():
        for values in value:
            r_dict[values] = tier_mapping[key]
    return r_dict


def aggregate_gdelt_primary_secondary(df: DataFrame, columns: list, actors: dict) -> DataFrame:
    to_apply = df.copy()

    def get_actor_group(actor):
        if actor in actors.get("primary_actors", []):
            return actor  # 'rus' or 'ua'
        elif actor in actors.get("secondary_actors_red", []):
            return "redsecondary"
        elif actor in actors.get("secondary_actors_blue", []):
            return "bluesecondary"
        return actor

    cols_to_drop = []

    for col in columns:
        try:
            prefix, actor1, actor2 = col.rsplit("_", 2)

            group1 = get_actor_group(actor1)
            group2 = get_actor_group(actor2)

            groups = set([group1, group2])  # noqa: C405
            suffix = None

            if groups == {"redsecondary", "rus"}:
                suffix = "redsecondary_rus"

            elif groups == {"redsecondary", "ua"}:
                suffix = "redsecondary_ua"

            elif groups == {"bluesecondary", "rus"}:
                suffix = "bluesecondary_rus"

            elif groups == {"bluesecondary", "ua"}:
                suffix = "bluesecondary_ua"

            elif groups == {"redsecondary", "bluesecondary"}:
                suffix = "redsecondary_bluesecondary"

            elif groups == {"rus", "ua"}:
                suffix = "rus_ua"  # Handles primary-primary interactions implicitly
            else:
                # Fallback for any unknown actors/categories (alphabetical)
                sorted_groups = sorted(list(groups))
                suffix = f"{sorted_groups[0]}_{sorted_groups[1]}"

            new_col = f"{prefix}_{suffix}"

            # Aggregate the values
            if new_col != col:
                if new_col not in to_apply.columns:
                    to_apply[new_col] = to_apply[col]
                else:
                    to_apply[new_col] = to_apply[new_col] + to_apply[col]

                cols_to_drop.append(col)

        except ValueError:
            continue

    # Drop the original columns so we're only left with the aggregated ones
    to_apply = to_apply.drop(columns=list(set(cols_to_drop)))

    return to_apply


def remove_low_prevalence(
    df: DataFrame, ratio: float = 0.01, specific: str = "acled_other_", verbose=False
) -> tuple[DataFrame, list]:
    to_check = df.copy()
    low_ratio = []
    for c in to_check.columns.tolist():
        if c.startswith(specific):
            sum = to_check.loc[to_check[c] > 0, c].sum()  # noqa: A001
            count = to_check[c].count()
            if sum / count < ratio:
                low_ratio.append(c)
                if verbose:
                    print(f"removing {c}, ratio: {sum / count}")
    to_check = to_check.drop(low_ratio, axis=1)
    return to_check, low_ratio


def get_pivoted_table(df: DataFrame, regions: list):
    to_apply = df.copy()
    regional_cols = [c for c in to_apply.columns if any(c.lower().endswith(f"_{r}") for r in regions)]
    global_cols = [c for c in to_apply.columns if c not in regional_cols and c != "event_date"]

    df_regional = to_apply[["event_date"] + regional_cols].copy()

    df_melted = df_regional.melt(
        id_vars=["event_date"], var_name="raw_col_name", value_name="value"
    )

    df_melted[["metric", "region"]] = df_melted["raw_col_name"].str.rsplit("_", n=1, expand=True)

    df_long_regional = df_melted.pivot_table(
        index=["event_date", "region"], columns="metric", values="value", aggfunc="first"
    ).reset_index()

    df_global = to_apply[["event_date"] + global_cols].copy()
    df_final = pd.merge(df_long_regional, df_global, on="event_date", how="left")

    df_final["event_date"] = pd.to_datetime(df_final["event_date"])
    df_final = df_final.set_index(["region", "event_date"]).sort_index()

    return df_final
