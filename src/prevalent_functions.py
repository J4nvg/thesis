from . import to_binary_classification,get_all_X,aggregate_gdelt_primary_secondary,load_actors_dict,get_all_X,remove_low_prevalence,get_pivoted_table,get_list_from_file,construct_path,load_region_activity_dictionary
import pandas as pd
import os


def load_data(
    data_path ,
    dataset_path,
):
    """
    returns regions, master_timeseries, regions_activity
    """
    regions = get_list_from_file(construct_path(data_path, "regions.txt"))
    master_timeseries = pd.read_parquet(construct_path(dataset_path, "master_combined_timeseries.parquet"))
    regions_activity  = load_region_activity_dictionary(construct_path(data_path, "regions_activity_cat.json"))

    print(f"master_timeseries shape: {master_timeseries.shape}")
    print(f"regions: {len(regions)}  -  {regions[:5]} ...")
    print(f"start: {master_timeseries['event_date'].min()} end: {master_timeseries['event_date'].max()}")
    return regions,master_timeseries,regions_activity


## Feature engineering

# 1. Identify weather / holiday columns (future covariates) and everything else
#    (past covariates).
# 2. Pivot from wide to long-hierarchical on region.
# 3. Assign activity label per regions.
# 4. Drop no activity regions
# 5. Drop low-prevalence ACLED categories.
# 6. Binarise the target.
# 7. Aggregate GDELT primary/secondary actors.
# 8. Drop financial export columns.
def get_engineered_features(
        master_timeseries,
        data_path,
        target_col,
        regions, 
        regions_activity,
        binarize_target=False,   # Added optional flag
        target_raw_col=None      # Needed if we are binarizing
):
    # Identify weather columns
    locale_env_weather_columns = [
        c for c in master_timeseries.columns
        if "env" in c and "holiday" not in c
    ]

    global_weather_columns = list(set(
        "_".join(c.split("_")[:-1]) if c != "env_k_max" else c
        for c in locale_env_weather_columns
    ))

    # add total strikes on day x
    master_timeseries['act_total_daily_strike_events'] = master_timeseries[[x for x in master_timeseries.columns.tolist() if 'act_drone_strike_on_ua' in x]].sum(axis=1)
    # add total damage events regardless of intention and target
    master_timeseries['act_total_damage_events'] = master_timeseries[[x for x in master_timeseries.columns.tolist() if 'act_drone_infra_ua' in x]].sum(axis=1)

    # Long hierarchical
    for_global = get_pivoted_table(df=master_timeseries.reset_index(), regions=regions)

    # Drop low-activity regions
    for_global["Activity_Level"] = for_global.index.get_level_values("region").map(regions_activity)
    for_global = for_global[for_global["Activity_Level"] != 0]

    # Drop low-prevalence ACLED "other" categories
    for_global, _removed = remove_low_prevalence(df=for_global, ratio=0.1, specific="acled_other_")

    # --- FOR TARGET CLASSIFICATION ONLY ---
    if binarize_target:
        if target_raw_col is None:
            raise ValueError("You must provide 'target_raw_col' if 'binarize_target' is True.")
        
        for_global = to_binary_classification(for_global, target_raw_col)
        for_global.drop(columns=[target_raw_col], inplace=True)
    # ------------------------------

    
    # Ratio / Interaction features
    if "acled_other_ua_armed_clash" in for_global.columns and "acled_other_rus_armed_clash" in for_global.columns:
        for_global["ratio_ua_rus_armed_clash"] = (
            for_global["acled_other_ua_armed_clash"] / (for_global["acled_other_rus_armed_clash"] + 1)
        )

    dist_cols = get_all_X("dist_to_nearest_ru_km", for_global)
    if dist_cols and "acled_other_rus_armed_clash" in for_global.columns:
        for_global["dist_x_clash"] = (
            for_global[dist_cols[0]] * for_global["acled_other_rus_armed_clash"]
        )





    # Aggregate GDELT primary/secondary
    gdelt_cols = (
        set(get_all_X("com_", for_global))
        - set(get_all_X("com_ners", for_global))
        - set(get_all_X("com_aid",  for_global))
    )
    actors = load_actors_dict(construct_path(data_path, "actors.json"))
    for_global = aggregate_gdelt_primary_secondary(df=for_global, columns=gdelt_cols, actors=actors)

    # Fix na in infra
    infra_cols = get_all_X("act_drone_infra_ua_",for_global)
    for_global[infra_cols] = for_global[infra_cols].fillna(0)

    # Drop financial export columns
    # for_global = for_global.drop(columns=get_all_X("fin_ch", for_global))

    print(f"Shape after feature engineering: {for_global.shape}")
    
    # Label prints differently depending on the task
    if binarize_target:
        print(f"Target positive rate: {for_global[target_col].mean():.3f}")
    else:
        print(f"Target mean: {for_global[target_col].mean():.3f}")

    for_global_reset = for_global.reset_index()
    if "index" in for_global_reset.columns:
        for_global_reset = for_global_reset.drop(columns=["index"])
        
    return for_global_reset, global_weather_columns

def get_engineered_features_damageclassifiers(
        master_timeseries,
        data_path,
        targets_cols,
        regions, 
        regions_activity,
):
    # Identify weather columns
    locale_env_weather_columns = [
        c for c in master_timeseries.columns
        if "env" in c and "holiday" not in c
    ]

    global_weather_columns = list(set(
        "_".join(c.split("_")[:-1]) if c != "env_k_max" else c
        for c in locale_env_weather_columns
    ))

    # add total strikes on day x
    master_timeseries['act_total_daily_strike_events'] = master_timeseries[[x for x in master_timeseries.columns.tolist() if 'act_drone_strike_on_ua' in x]].sum(axis=1)
    # add total damage events regardless of intention and target
    master_timeseries['act_total_damage_events'] = master_timeseries[[x for x in master_timeseries.columns.tolist() if 'act_drone_infra_ua' in x]].sum(axis=1)

    # Long hierarchical
    for_global = get_pivoted_table(df=master_timeseries.reset_index(), regions=regions)

    # Drop low-activity regions
    for_global["Activity_Level"] = for_global.index.get_level_values("region").map(regions_activity)
    for_global = for_global[for_global["Activity_Level"] != 0]

    # Drop low-prevalence ACLED "other" categories
    for_global, _removed = remove_low_prevalence(df=for_global, ratio=0.1, specific="acled_other_")


    
    # Ratio / Interaction features
    if "acled_other_ua_armed_clash" in for_global.columns and "acled_other_rus_armed_clash" in for_global.columns:
        for_global["ratio_ua_rus_armed_clash"] = (
            for_global["acled_other_ua_armed_clash"] / (for_global["acled_other_rus_armed_clash"] + 1)
        )

    dist_cols = get_all_X("dist_to_nearest_ru_km", for_global)
    if dist_cols and "acled_other_rus_armed_clash" in for_global.columns:
        for_global["dist_x_clash"] = (
            for_global[dist_cols[0]] * for_global["acled_other_rus_armed_clash"]
        )

    # Aggregate GDELT primary/secondary
    gdelt_cols = (
        set(get_all_X("com_", for_global))
        - set(get_all_X("com_ners", for_global))
        - set(get_all_X("com_aid",  for_global))
    )
    actors = load_actors_dict(construct_path(data_path, "actors.json"))
    for_global = aggregate_gdelt_primary_secondary(df=for_global, columns=gdelt_cols, actors=actors)

    # Fix na in infra
    infra_cols = get_all_X("act_drone_infra_ua_",for_global)
    for_global[infra_cols] = for_global[infra_cols].fillna(0)

    to_return_df = []

    #------------------
    for target in targets_cols:
        copied_df = for_global.copy()
        print(target)
        # print(get_all_X("act_drone_infra_ua_",copied_df))
        copied_df = to_binary_classification(copied_df, target)
        copied_df.drop(columns=[target], inplace=True)
        print(f"Amount of target =1: {copied_df[f"{target}_binary"].sum():.3f}")
        print(f"Total rows: {copied_df[f"{target}_binary"].count():.3f}")
        print(f"Target positive rate: {copied_df[f"{target}_binary"].mean():.3f}")
        # Label prints differently depending on the task
        copied_df_reset = copied_df.reset_index()
        if "index" in copied_df_reset.columns:
            copied_df_reset = copied_df_reset.drop(columns=["index"])
        to_return_df.append(copied_df_reset)

    return to_return_df, global_weather_columns


def split_future_and_past_cov(for_global_reset,global_weather_columns,target, exclude=None):
    holiday_cols      = get_all_X("holiday", for_global_reset)
    future_covariates = holiday_cols + global_weather_columns

    if not exclude:
        exclude_cols = {target, "Activity_Level", "index", "level_0", "region", "event_date"}
    else:
        exclude_cols = exclude

    past_covariates   = [
        c for c in for_global_reset.columns
        if c not in future_covariates and c not in exclude_cols
    ]

    print(f"# future covariates : {len(future_covariates)}")
    print(f"# past covariates   : {len(past_covariates)}")
    return holiday_cols,future_covariates, exclude_cols, past_covariates

def get_common_kwargs(input_lags=7,output_chunk_len=7):
    common_kwargs = dict(
    lags                    = input_lags,
    lags_past_covariates    = [-1,-7,-14],
    lags_future_covariates  = (2, output_chunk_len),
    output_chunk_length     = output_chunk_len,
    multi_models            = True,
    output_chunk_shift      = 0,
    add_encoders            = {
        "cyclic": {
            "future": ["month", "week", "dayofyear", "dayofweek", "day"]
            },
    },
    )
    return common_kwargs

def get_available_threads():
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return min(cpu_count, 64)


def get_top_100_from_lgbm(fitted_model):
    gain_imps = {"Feature": fitted_model.lagged_feature_names}
    underlying_model = fitted_model.model
    estimators = underlying_model.estimators_ if hasattr(underlying_model, "estimators_") else [underlying_model]
    for h, est in enumerate(estimators, start=1):
        gain_imps[f"h{h}_gain"] = est.booster_.feature_importance(importance_type="gain")

    df_gain = pd.DataFrame(gain_imps)
    df_gain["mean_gain"] = df_gain.filter(like="_gain").mean(axis=1)
    df_gain = df_gain.sort_values("mean_gain", ascending=False).reset_index(drop=True)
    df_gain.to_csv("prelimFeatureImportanceRegressor.csv", index=False)
    print(df_gain.head(20))
    top_100_features      = df_gain.head(100)["Feature"].to_list()
    return top_100_features