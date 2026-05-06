from __future__ import annotations
import pandas as pd
from pandas import DataFrame
import numpy as np


def get_calendar_features(df: DataFrame):
    to_apply = df.copy()
    dates = to_apply.index.get_level_values('event_date')
    to_apply['cal_month_of_year'] = dates.month
    to_apply['cal_week_of_year'] = dates.isocalendar().week.values
    to_apply['cal_day_of_year'] = dates.dayofyear
    to_apply['cal_is_weekend'] = (dates.dayofweek >= 5).astype(int)
    to_apply['cal_day_of_week'] = dates.dayofweek
    to_apply['cal_day_of_month'] = dates.day
    return to_apply

def get_daily_national_aggregates_damage(dataframe:DataFrame,regions:list):
    """
    get infratype split daily aggregates of intentional damage
    """
    df = dataframe.copy()
    infra_types = ['education','health','energy','residential']
    # intentions = ['unintend','intent']
    # act_drone_infra_ua_education_intent_region
    all_colls = df.columns.tolist()
    for infra_type in infra_types:
        df[f'act_total_daily_national_intentional_{infra_type}_damage'] = df[[x for x in all_colls if ('intent' in x and infra_type in x and not 'unintent')]].sum(axis=1)
    return df

def alpha_to_halflife(alpha):
    return -1 / np.log2(alpha)

def halflife_to_alpha(halflife_days):
    return 2 ** (-1 / halflife_days)

def add_temporal_aggregates(
    df: pd.DataFrame,
    cols: list,
    group_col: str = 'region',
    **kwargs
) -> pd.DataFrame:
    """
    Add rolling/EWMA features per group (respecting the hierarchical index).
    df must have a MultiIndex containing the group_col and the date.

    Usage Example:
        df_new = add_temporal_aggregates(
            df, cols=['sales'],
            rsum=[7, 14], ma=[7], ewma=[14], ed=[7]
        )
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    idx_names = df.index.names
    result = df.copy().reset_index()

    new_cols = {}  # accumulate here, concat once at the end

    for col in cols:
        grp = result.groupby(group_col)[col]

        for n in kwargs.get('rsum', []):
            new_cols[f'{col}_roll{n}_sum'] = grp.transform(
                lambda x, n=n: x.shift(1).rolling(n, min_periods=1).sum()
            )

        for n in kwargs.get('ma', []):
            new_cols[f'{col}_roll{n}_mean'] = grp.transform(
                lambda x, n=n: x.shift(1).rolling(n, min_periods=1).mean()
            )

        for n in kwargs.get('ewma', []):
            new_cols[f'{col}_ewma{n}'] = grp.transform(
                lambda x, n=n: x.shift(1).ewm(span=n, adjust=False).mean()
            )

        for n in kwargs.get('ed', []):
            new_cols[f'{col}_expdecay{n}'] = grp.transform(
                lambda x, n=n: x.shift(1).ewm(alpha=halflife_to_alpha(n), adjust=False).mean()
            )

    result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)

    if None not in idx_names and len(idx_names) > 0:
        return result.set_index(idx_names)
    return result.set_index([group_col, 'event_date'])


def to_binary_classification(df: pd.DataFrame, column: str) -> pd.DataFrame:
    to_apply = df.copy()

    if column not in to_apply.columns:
        raise ValueError(f"Column not found in DataFrame: {column}")

    to_apply[f"{column}_binary"] = (to_apply[column] > 0).astype(int)
    return to_apply

def clean_feature_names(input_list: list) -> dict:
    r_dict = {
        "futcov_features_base": set(),
        "futcov_features_actual": set(),
        "pastcov_features_base": set(),
        "pastcov_features_actual": set(),
        "statcov_features_base": set(),
        "statcov_features_actual": set(),
        "target_features_base": set(),
        "target_features_actual": set(),
        "features_base": set(),
        "actual_features": set(),
    }

    if not input_list: 
        return r_dict
    
    # Define the suffixes to look for
    suffixes = ['_pastcov', '_futcov', '_statcov', '_target']
    
    for variable in input_list:
        matched = False
        for suffix in suffixes:
            if suffix in variable:
                var = variable.split(suffix)[0]
                prefix = suffix.strip('_') # turns '_pastcov' into 'pastcov'
                
                r_dict['features_base'].add(var)
                r_dict['actual_features'].add(variable)
                r_dict[f'{prefix}_features_base'].add(var)
                r_dict[f'{prefix}_features_actual'].add(variable)
                
                matched = True
                break # Stop searching suffixes once a match is found
                
        if not matched:
            print(f"'{variable}': none of pastcov, futcov, statcov, target ..")
            
    return r_dict