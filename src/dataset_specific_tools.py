from __future__ import annotations
import pandas as pd
from pandas import DataFrame
import json
def load_actors_dict(filepath: str = 'actors.json') -> dict:
    """
    Loads the actors configuration from a JSON file.
    """
    with open(filepath, 'r') as file:
        actors_dict = json.load(file)
    return actors_dict

def load_region_activity_dictionary(filepath: str = 'regions_activity_cat.json') -> dict:
    """
    Loads the region activity configuration from a JSON file.
    """
    r_dict = {}
    region_activity_dictionary = {}
    tier_mapping = {'little': 0, 'low': 1, 'medium': 2, 'high': 3}
    with open(filepath, 'r') as file:
        region_activity_dictionary = json.load(file)
    
    for key,value in region_activity_dictionary.items():
        for values in value:
            r_dict[values] = tier_mapping[key]
    return r_dict


def aggregate_gdelt_primary_secondary(df: DataFrame, columns: list, actors: dict) -> DataFrame:
    to_apply = df.copy()
    
    def get_actor_group(actor):
        if actor in actors.get('primary_actors', []):
            return actor  # 'rus' or 'ua'
        elif actor in actors.get('secondary_actors_red', []):
            return 'redsecondary'
        elif actor in actors.get('secondary_actors_blue', []):
            return 'bluesecondary'
        return actor

    cols_to_drop = []
    
    for col in columns:
        try:
            prefix, actor1, actor2 = col.rsplit('_', 2)
            
            group1 = get_actor_group(actor1)
            group2 = get_actor_group(actor2)
            
            groups = set([group1, group2])
            suffix = None
            
            if groups == {'redsecondary', 'rus'}:
                suffix = 'redsecondary_rus'

            elif groups == {'redsecondary', 'ua'}:
                suffix = 'redsecondary_ua'

            elif groups == {'bluesecondary', 'rus'}:
                suffix = 'bluesecondary_rus'

            elif groups == {'bluesecondary', 'ua'}:
                suffix = 'bluesecondary_ua'

            elif groups == {'redsecondary', 'bluesecondary'}:
                suffix = 'redsecondary_bluesecondary'
                
            elif groups == {'rus', 'ua'}:
                suffix = 'rus_ua' # Handles primary-primary interactions implicitly
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


def remove_low_prevalence(df:DataFrame,ratio:float=0.01,specific:str='acled_other_',verbose=False) -> tuple[DataFrame,list] :
    to_check = df.copy()
    low_ratio = []
    for c in to_check.columns.tolist():
        if c.startswith(specific):
            sum = to_check.loc[to_check[c] > 0, c].sum()
            count = to_check[c].count()
            if sum/count < ratio:
                low_ratio.append(c)
                if verbose:
                    print(f"removing {c}, ratio: {sum/count}")
    to_check = to_check.drop(low_ratio, axis=1)
    return to_check, low_ratio


def get_pivoted_table(df:DataFrame,regions:list):

    to_apply = df.copy()
    regional_cols = [c for c in to_apply.columns if any(c.lower().endswith(f"_{r}") for r in regions)]
    global_cols = [c for c in to_apply.columns if c not in regional_cols and c != 'event_date']

    df_regional = to_apply[['event_date'] + regional_cols].copy()

    df_melted = df_regional.melt(
        id_vars=['event_date'],
        var_name='raw_col_name',
        value_name='value'
    )

    df_melted[['metric', 'region']] = df_melted['raw_col_name'].str.rsplit('_', n=1, expand=True)

    df_long_regional = df_melted.pivot_table(
        index=['event_date', 'region'],
        columns='metric',
        values='value',
        aggfunc='first'
    ).reset_index()

    df_global = to_apply[['event_date'] + global_cols].copy()
    df_final = pd.merge(df_long_regional, df_global, on='event_date', how='left')

    df_final['event_date'] = pd.to_datetime(df_final['event_date'])
    df_final = df_final.set_index(['region', 'event_date']).sort_index()

    return df_final

def get_diff_and_normal_stationarity(df:DataFrame):
    import warnings
    import numpy as np
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from darts import TimeSeries
    from darts.utils.statistics import stationarity_tests
    from src import get_all_X
    
    warnings.filterwarnings("ignore", category=InterpolationWarning)
    master_timeseries = df.copy()
    # Assuming get_all_X is defined elsewhere
    targets = get_all_X('act_drone_strike_on_ua', master_timeseries)

    stationarity_results = {}

    # --- Phase 1: Original Series ---
    df_working = master_timeseries.reset_index()

    for target in targets:
        if df_working[target].std() == 0:
            stationarity_results[target] = True
            continue
            
        targetts = TimeSeries.from_dataframe(
            df_working, 
            time_col='event_date', 
            value_cols=target
        )
        
        # Safely try the test
        try:
            stationarity_results[target] = stationarity_tests(targetts)
        except Exception as e:
            print(f"Phase 1: Statsmodels math error on {target} ({type(e).__name__}). Defaulting to False.")
            stationarity_results[target] = False

    df_working_2 = master_timeseries.reset_index()

    for target in targets:
        df_working_2[target] = df_working_2[target].diff()

    df_working_2 = df_working_2.dropna(subset=targets).reset_index(drop=True)

    for target in targets:
        if df_working_2[target].nunique() <= 1:
            stationarity_results[f"{target}_afterdiff"] = True 
            continue
            
        if np.isinf(df_working_2[target]).any():
            stationarity_results[f"{target}_afterdiff"] = False
            continue

        targetts = TimeSeries.from_dataframe(
            df_working_2, 
            time_col='event_date', 
            value_cols=target
        )
        try:
            is_stationary = stationarity_tests(targetts)
            stationarity_results[f"{target}_afterdiff"] = is_stationary
        except Exception as e:
            # If it crashes here, the series is usually highly degenerate/sparse.
            print(f"Phase 2: Statsmodels math error on {target}_afterdiff ({type(e).__name__}). Defaulting to False.")
            stationarity_results[f"{target}_afterdiff"] = False

    print("=== Final Stationarity Summary ===")
    for target_name, stat_result in stationarity_results.items():
        print(f"{target_name}: {'Stationary' if stat_result else 'Non-Stationary'}")



