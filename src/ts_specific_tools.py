from __future__ import annotations
from darts import TimeSeries
from darts.dataprocessing.transformers import WindowTransformer, StaticCovariatesTransformer

def split_series_list(series_list):
    train_list, val_list, test_list = [], [], []
    
    for ts in series_list:
        # Step 1: Split at 70% (Train gets 70%, temp gets the remaining 30%)
        train, temp = ts.split_after(0.7)
        
        # Step 2: Split the remaining 30% (1/3 of 30% is 10% for Val, the rest 20% is Test)
        val, test = temp.split_after(1/3)
        
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
        
    return train_list, val_list, test_list

def build_ts_and_apply_window_transformer(for_global_reset, target,past_covariates,future_covariates,ed_alpha):

    target_series_list = TimeSeries.from_group_dataframe(
        for_global_reset,
        group_cols="region", time_col="event_date",
        value_cols=target, static_cols=["Activity_Level"],
    )
    past_covs_raw = TimeSeries.from_group_dataframe(
        for_global_reset,
        group_cols="region", time_col="event_date",
        value_cols=past_covariates,
    )
    future_covs_list = TimeSeries.from_group_dataframe(
        for_global_reset,
        group_cols="region", time_col="event_date",
        value_cols=future_covariates,
    )

    # Windowed transforms on past covariates (rolling sums/means/EWMA)
    window_transforms = [
        {"function": "sum",  "mode": "rolling", "window": 14, "min_periods": 1, "function_name": "rsum14"},
        {"function": "sum",  "mode": "rolling", "window": 7, "min_periods": 1, "function_name": "rsum7"},
        {"function": "mean", "mode": "rolling", "window":  7, "min_periods": 1, "function_name": "rmean7"},
        {"function": "mean", "mode": "rolling", "window":  28, "min_periods": 1, "function_name": "rmean28"},
        {"function": "mean", "mode": "ewm", "span": 14,                              "function_name": "ewma14"},
        {"function": "mean", "mode": "ewm", "alpha": ed_alpha,           "function_name": "expdecay7"},
    ]
    window_transformer = WindowTransformer(
        transforms=window_transforms,
        treat_na=0, forecasting_safe=True,
        keep_non_transformed=True, include_current=True, keep_names=False,
    )
    past_covs_list = window_transformer.transform(past_covs_raw)

    print(f"Past-cov components: raw={past_covs_raw[0].n_components}  transformed={past_covs_list[0].n_components}")
    
    return target_series_list, past_covs_list,future_covs_list


def get_covs_and_encodings(target_series_list,past_covs_list,future_covs_list,TRAIN_FRAC,VAL_FRAC):
    """
    input: Target_series_list, past_covs_list, future_covs_list,train_frac,val_frac
    output: region_names
            train_target, val_target, test_target
            full_past_covs, full_fut_covs, 
            target_for_cv
    """
    # Capture names (pre-encoding)

    region_names = [ts.static_covariates["region"].iloc[0] for ts in target_series_list]
    # Encode
    statcov_t = StaticCovariatesTransformer()
    target_series_list = statcov_t.fit_transform(target_series_list)
    statcov_p = StaticCovariatesTransformer()
    past_covs_list     = statcov_p.fit_transform(past_covs_list)
    statcov_f = StaticCovariatesTransformer()
    future_covs_list   = statcov_f.fit_transform(future_covs_list)

    # Split
    train_target, val_target, test_target = split_series_list(target_series_list)

    full_past_covs = past_covs_list
    full_fut_covs  = future_covs_list

    # Validation-only series for CV backtests
    # Model comparison MUST NOT touch the test set. We build a truncated view of
    # each target that ends at the end of val (= start of test), then roll CV
    # across the val portion only. Covariates stay full-length (the predictor
    # looks ahead inside the prediction window).
    TRAIN_VAL_END = TRAIN_FRAC + VAL_FRAC              # 0.80 of the full series
    CV_START_VAL  = TRAIN_FRAC / TRAIN_VAL_END         # 0.875 of the truncated series

    target_for_cv = [ts.split_before(TRAIN_VAL_END)[0] for ts in target_series_list]

    print(f"# regions: {len(target_series_list)}")
    print(f"train len (region 0): {len(train_target[0])} days")
    print(f"val   len (region 0): {len(val_target[0])} days")
    print(f"test  len (region 0): {len(test_target[0])} days  <- untouched until the final evaluation")
    print(f"CV view length       : {len(target_for_cv[0])} days (train+val)")
    print(f"CV start on that view: {CV_START_VAL:.3f}  -> rolls across the val segment only")

    return region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END,CV_START_VAL


def make_positive_only_weights(target_list):
    """Weight 1.0 where y > 0, 0.0 where y == 0. Keeps the time index intact."""
    weights = []
    for ts in target_list:
        vals = ts.values().ravel()
        w = (vals > 0).astype(float)
        weights.append(
            TimeSeries.from_times_and_values(
                ts.time_index, w, static_covariates=ts.static_covariates
            )
        )
    return weights