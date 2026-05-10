from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.metrics import mean_poisson_deviance, mean_tweedie_deviance

EPS = 1e-9


# ---------------------------------------------------------------------------
# Naive scales
# ---------------------------------------------------------------------------

def compute_naive_scales(target_list, region_names, seasonality: int = 7):
    """In-sample mean |Δ_m| and sqrt(mean Δ_m²) for each region (m = seasonality).

    Returns two ``{region_name: scale}`` dicts consumed by ``evaluate_long``.
    Uses the **training** portion of each series so the scale never leaks
    information from the CV folds it's going to be evaluating.
    """
    mae_s, rmse_s = {}, {}
    for name, ts in zip(region_names, target_list):
        y = np.asarray(ts.values(), dtype=float).ravel()
        if len(y) <= seasonality:
            mae_s[name] = rmse_s[name] = np.nan
            continue
        err = y[seasonality:] - y[:-seasonality]
        mae_s[name]  = float(np.mean(np.abs(err)))
        rmse_s[name] = float(np.sqrt(np.mean(err ** 2)))
    return mae_s, rmse_s


# ---------------------------------------------------------------------------
# Point-forecast metric primitives
# ---------------------------------------------------------------------------

def base_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0.0, None)
    err    = y_pred - y_true
    return {
        "MAE":        float(np.mean(np.abs(err))),
        "RMSE":       float(np.sqrt(np.mean(err ** 2))),
        "MedAE":      float(np.median(np.abs(err))),
        "ME":         float(np.mean(err)),
        "PoissonDev": float(mean_poisson_deviance(y_true, np.maximum(y_pred, EPS))),
        "TweedieDev": float(mean_tweedie_deviance(y_true, np.maximum(y_pred, EPS), power=1.5)),
        "ZeroAcc":    float(np.mean((y_true == 0) == (y_pred < 0.5))),
        "n":          int(len(y_true)),
    }


def _scaled_metrics(sub_df, mae_scale, rmse_scale):
    """MASE and RMSSE for a region-scoped subset. NaN if the scale is degenerate."""
    y_true = sub_df["y_true"].to_numpy(dtype=float)
    y_pred = np.clip(sub_df["y_pred"].to_numpy(dtype=float), 0.0, None)
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {
        "MASE":  float(mae  / mae_scale)  if mae_scale  and mae_scale  > 0 else np.nan,
        "RMSSE": float(rmse / rmse_scale) if rmse_scale and rmse_scale > 0 else np.nan,
    }


def _skill(model_val, ref_val):
    if ref_val is None or ref_val == 0 or np.isnan(ref_val):
        return np.nan
    return float(1.0 - model_val / ref_val)


# ---------------------------------------------------------------------------
# Evaluation aggregator (regression notebooks — scales required, no globals)
# ---------------------------------------------------------------------------

def evaluate_long(long_df, mae_scales, rmse_scales, regions_activity=None):
    """Aggregate metrics over (region), (horizon), (region, horizon), and globally.

    mae_scales / rmse_scales: ``{region_name: float}`` dicts from
    ``compute_naive_scales``. Pass the training-set scales so the function
    never touches module-level state.

    regions_activity: optional ``{region_name: int}`` dict (levels 1/2/3).
    When provided, two extra views are added to the result:
      ``per_activity_level``  — one row per activity level
      ``per_activity_horizon`` — activity_level × horizon
    """
    def _rows_from_groupby(group_cols):
        rows = []
        for keys, sub in long_df.groupby(group_cols):
            row = base_metrics(sub["y_true"], sub["y_pred"])
            if isinstance(group_cols, str):
                key_map = {group_cols: keys}
            else:
                key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            region_key = key_map.get("region")
            if region_key is not None:
                row.update(_scaled_metrics(
                    sub, mae_scales.get(region_key), rmse_scales.get(region_key),
                ))
            row.update(key_map)
            rows.append(row)
        return pd.DataFrame(rows)

    per_region         = _rows_from_groupby("region").sort_values("MASE", ascending=True).reset_index(drop=True)
    per_horizon        = _rows_from_groupby("horizon").sort_values("horizon").reset_index(drop=True)
    per_region_horizon = _rows_from_groupby(["region", "horizon"]).sort_values(["region", "horizon"]).reset_index(drop=True)

    global_row = base_metrics(long_df["y_true"], long_df["y_pred"])
    global_row["MASE_mean"]   = float(per_region["MASE"].mean())
    global_row["MASE_median"] = float(per_region["MASE"].median())
    global_row["RMSSE_mean"]  = float(per_region["RMSSE"].mean())

    result = {
        "per_region":         per_region,
        "per_horizon":        per_horizon,
        "per_region_horizon": per_region_horizon,
        "global":             global_row,
    }

    if regions_activity is not None:
        df2 = long_df.copy()
        df2["activity_level"] = df2["region"].map(regions_activity)

        act_rows = []
        for level, sub in df2.groupby("activity_level"):
            row = base_metrics(sub["y_true"], sub["y_pred"])
            pr_sub = per_region[per_region["region"].isin(sub["region"].unique())]
            row["MASE_mean"]      = float(pr_sub["MASE"].mean())
            row["RMSSE_mean"]     = float(pr_sub["RMSSE"].mean())
            row["activity_level"] = level
            act_rows.append(row)
        result["per_activity_level"] = (
            pd.DataFrame(act_rows).sort_values("activity_level").reset_index(drop=True)
        )

        ah_rows = []
        for (level, h), sub in df2.groupby(["activity_level", "horizon"]):
            row = base_metrics(sub["y_true"], sub["y_pred"])
            row["activity_level"] = level
            row["horizon"]        = int(h)
            ah_rows.append(row)
        result["per_activity_horizon"] = (
            pd.DataFrame(ah_rows)
              .sort_values(["activity_level", "horizon"])
              .reset_index(drop=True)
        )

    return result


# ---------------------------------------------------------------------------
# Naive forecast baselines
# ---------------------------------------------------------------------------

def naive_last_historical_forecasts(actual_ts, start_frac, horizon, stride):
    """y_hat(t+h) = y(t-1), repeated across the horizon."""
    n = len(actual_ts)
    vals = actual_ts.values().ravel()
    start_idx = max(int(start_frac * n), 1)
    preds = []
    for t0 in range(start_idx, n - horizon + 1, stride):
        last_val = float(vals[t0 - 1])
        time_index = actual_ts.time_index[t0 : t0 + horizon]
        values = np.full((horizon, 1), last_val, dtype=float)
        preds.append(TimeSeries.from_times_and_values(time_index, values))
    return preds


def naive_weekly_historical_forecasts(actual_ts, start_frac, horizon, stride):
    """y_hat(t+h) = y(t+h-7): same-day-last-week seasonal naive."""
    n = len(actual_ts)
    vals = actual_ts.values().ravel()
    start_idx = max(int(start_frac * n), 7)
    preds = []
    for t0 in range(start_idx, n - horizon + 1, stride):
        lagged = vals[t0 - 7 : t0 - 7 + horizon].reshape(-1, 1).astype(float)
        time_index = actual_ts.time_index[t0 : t0 + horizon]
        preds.append(TimeSeries.from_times_and_values(time_index, lagged))
    return preds


def naive_historical_forecasts(actual_ts, start_frac, horizon, stride,
                                method: str = "last"):
    """Persistence forecast baseline.

    method="last"
        Predict the most recent observed value regardless of sign.
        Matches the original ``event_classifiers_prehurdle`` behaviour.
    method="last_non_negative"
        Predict the most recent non-negative historical observation.
        Matches the original ``regressor_part_prehurdle`` behaviour.
    method="non_negative_mean"
        Predict the mean of all non-negative historical observations.
    """
    n_total   = len(actual_ts)
    start_idx = int(start_frac * n_total)
    preds     = []

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        pred_val = 0.0
        if t0 > 0:
            if method == "last":
                pred_val = float(actual_ts.values()[t0 - 1, 0])
            elif method in ("last_non_negative", "non_negative_mean"):
                past_vals = actual_ts.values()[:t0, 0]
                non_neg   = past_vals[past_vals >= 0]
                if len(non_neg) > 0:
                    pred_val = (float(non_neg[-1]) if method == "last_non_negative"
                                else float(np.mean(non_neg)))
            else:
                raise ValueError(f"Unknown naive_historical_forecasts method: {method!r}")

        time_index = actual_ts.time_index[t0 : t0 + horizon]
        values     = np.full((horizon, 1), pred_val, dtype=float)
        preds.append(TimeSeries.from_times_and_values(time_index, values))

    return preds


def naive_collect_long(target_list, region_names, method, start_frac, horizon, stride):
    """Produce naive fold predictions in the same long-DataFrame format as CV models.

    method   : "naive_last" | "naive_weekly"
    horizon  : forecast horizon (required — avoids implicit coupling to notebook constants)
    stride   : CV stride (required — same reason)
    """
    fn = {
        "naive_last":   naive_last_historical_forecasts,
        "naive_weekly": naive_weekly_historical_forecasts,
    }[method]
    fold_preds_list = [fn(ts, start_frac, horizon, stride) for ts in target_list]
    return collect_predictions_long(target_list, fold_preds_list, region_names), fold_preds_list


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def collect_predictions_long(actuals, fold_preds_list, region_names):
    """Flatten historical_forecasts output into a long DataFrame (y_pred column)."""
    rows = []
    for r_idx, (actual_ts, fold_preds) in enumerate(zip(actuals, fold_preds_list)):
        region     = region_names[r_idx]
        actual_map = dict(zip(actual_ts.time_index, actual_ts.values().ravel()))
        for f_idx, pred in enumerate(fold_preds):
            pred_values = pred.values().ravel()
            for h, (t, y_pred) in enumerate(zip(pred.time_index, pred_values), start=1):
                if t in actual_map:
                    rows.append({
                        "region":  region,
                        "fold":    f_idx,
                        "horizon": h,
                        "date":    t,
                        "y_true":  float(actual_map[t]),
                        "y_pred":  float(y_pred),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CV infrastructure
# ---------------------------------------------------------------------------

def _maybe_scale_covs(past_covs, future_covs, do_scale):
    """Scale covariates globally for neural nets; identity otherwise."""
    if not do_scale:
        return past_covs, future_covs
    ps = Scaler(); fs = Scaler()
    return ps.fit_transform(past_covs), fs.fit_transform(future_covs)


def run_expanding_cv(
    builder_fn,
    target_list,
    start_frac,
    *,
    is_local: bool = False,
    is_neural: bool = False,
    horizon: int,
    stride: int,
    retrain_stride: int = None,
    past_covs=None,
    future_covs=None,
    verbose: bool = True,
):
    """Expanding-window CV returning level-space predictions.

    horizon / stride are required keyword-only args so the caller always
    passes them explicitly rather than relying on notebook-level constants.

    retrain_stride -- how often to retrain (default: every fold, i.e. every stride steps).
    When retrain_stride > stride the frozen model predicts from a growing context
    between retrains, matching the run_final_test protocol.
    """
    if retrain_stride is None:
        retrain_stride = stride

    ref_ts    = target_list[0]
    n_total   = len(ref_ts)
    start_idx = int(start_frac * n_total)

    n_regions      = len(target_list)
    all_fold_preds = [[] for _ in range(n_regions)]
    n_preds    = 0
    n_retrains = 0
    model      = None
    _local_builder = None

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        steps_since_start = t0 - start_idx
        split_time        = ref_ts.time_index[t0]

        if steps_since_start % retrain_stride == 0:
            train_series = [ts.drop_after(split_time) for ts in target_list]
            if is_local:
                _local_builder = builder_fn
            else:
                model = builder_fn()
                fit_kwargs = {"series": train_series}
                if past_for_fit is not None and model.supports_past_covariates:
                    fit_kwargs["past_covariates"] = past_for_fit
                if fut_for_fit is not None and model.supports_future_covariates:
                    fit_kwargs["future_covariates"] = fut_for_fit
                model.fit(**fit_kwargs)
            n_retrains += 1
            if verbose:
                print(f"   retrain {n_retrains}  (data up to {split_time.date()})")

        pred_series = [ts.drop_after(split_time) for ts in target_list]

        if is_local:
            preds = []
            for ts in pred_series:
                m = _local_builder()
                m.fit(ts)
                preds.append(m.predict(n=horizon))
        else:
            pred_kwargs = {"n": horizon, "series": pred_series}
            if past_for_fit is not None and model.supports_past_covariates:
                pred_kwargs["past_covariates"] = past_for_fit
            if fut_for_fit is not None and model.supports_future_covariates:
                pred_kwargs["future_covariates"] = fut_for_fit

            is_probabilistic = is_neural and getattr(model, "likelihood", None) is not None
            if is_probabilistic:
                pred_kwargs["num_samples"] = 200
            preds = model.predict(show_warnings=False, **pred_kwargs)
            if is_probabilistic:
                preds = [p.quantile(0.5) for p in preds]
            if getattr(model, "_count_log_link", False):
                preds = [p.map(np.exp) for p in preds]

        for r_idx, pred in enumerate(preds):
            all_fold_preds[r_idx].append(pred)
        n_preds += 1

    if verbose:
        print(f"   {n_preds} predictions, {n_retrains} retrains complete")
    return all_fold_preds


def run_final_test(
    builder_fn,
    target_list,
    start_frac,
    *,
    predict_stride: int = 1,
    retrain_stride: int,
    horizon: int,
    past_covs=None,
    future_covs=None,
    is_local: bool = False,
    is_neural: bool = False,
    verbose: bool = True,
):
    """Expanding-window test evaluation with decoupled predict / retrain strides.

    predict_stride -- how often a new horizon-day forecast is issued (default: every day)
    retrain_stride -- how often the model is retrained (e.g. OUTPUT_CHUNK_LEN = 7)
    Between retrains the frozen model predicts from a growing context window.
    Output format is identical to run_expanding_cv.
    """
    ref_ts    = target_list[0]
    n_total   = len(ref_ts)
    start_idx = int(start_frac * n_total)
    n_regions = len(target_list)

    all_fold_preds = [[] for _ in range(n_regions)]
    n_preds    = 0
    n_retrains = 0
    model      = None
    _local_builder = None

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, predict_stride):
        steps_since_start = t0 - start_idx

        if steps_since_start % retrain_stride == 0:
            retrain_time = ref_ts.time_index[t0]
            train_series = [ts.drop_after(retrain_time) for ts in target_list]

            if is_local:
                _local_builder = builder_fn
            else:
                model = builder_fn()
                fit_kwargs = {"series": train_series}
                if past_for_fit is not None and model.supports_past_covariates:
                    fit_kwargs["past_covariates"] = past_for_fit
                if fut_for_fit is not None and model.supports_future_covariates:
                    fit_kwargs["future_covariates"] = fut_for_fit
                model.fit(**fit_kwargs)

            n_retrains += 1
            if verbose:
                print(f"   retrain {n_retrains}  (data up to {retrain_time.date()})")

        split_time  = ref_ts.time_index[t0]
        pred_series = [ts.drop_after(split_time) for ts in target_list]

        if is_local:
            preds = []
            for ts in pred_series:
                m = _local_builder()
                m.fit(ts)
                preds.append(m.predict(n=horizon))
        else:
            pred_kwargs = {"n": horizon, "series": pred_series, "show_warnings": False}
            if past_for_fit is not None and model.supports_past_covariates:
                pred_kwargs["past_covariates"] = past_for_fit
            if fut_for_fit is not None and model.supports_future_covariates:
                pred_kwargs["future_covariates"] = fut_for_fit
            preds = model.predict(**pred_kwargs)

        for r_idx, pred in enumerate(preds):
            all_fold_preds[r_idx].append(pred)
        n_preds += 1

    if verbose:
        print(f"   {n_preds} daily predictions, {n_retrains} retrains complete")
    return all_fold_preds


def run_expanding_cv_iter(
    builder_fn,
    target_list,
    start_frac,
    *,
    is_local: bool = False,
    is_neural: bool = False,
    horizon: int,
    stride: int,
    retrain_stride: int = None,
    past_covs=None,
    future_covs=None,
    verbose: bool = False,
):
    """Generator twin of run_expanding_cv.

    After each prediction step yields the cumulative fold_preds bundle so the
    caller can score-so-far and decide whether to prune.
    horizon / stride are required keyword-only args.

    retrain_stride -- how often to retrain (default: every stride steps).
    """
    if retrain_stride is None:
        retrain_stride = stride

    ref_ts    = target_list[0]
    n_total   = len(ref_ts)
    start_idx = int(start_frac * n_total)
    n_regions = len(target_list)
    all_fold_preds = [[] for _ in range(n_regions)]
    model      = None
    _local_builder = None

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        steps_since_start = t0 - start_idx
        split_time        = ref_ts.time_index[t0]

        if steps_since_start % retrain_stride == 0:
            train_series = [ts.drop_after(split_time) for ts in target_list]
            if is_local:
                _local_builder = builder_fn
            else:
                model = builder_fn()
                fit_kwargs  = {"series": train_series}
                pred_kwargs = {"n": horizon, "series": train_series}
                if past_for_fit is not None and model.supports_past_covariates:
                    fit_kwargs["past_covariates"]  = past_for_fit
                    pred_kwargs["past_covariates"] = past_for_fit
                if fut_for_fit is not None and model.supports_future_covariates:
                    fit_kwargs["future_covariates"]  = fut_for_fit
                    pred_kwargs["future_covariates"] = fut_for_fit
                model.fit(**fit_kwargs)

        pred_series = [ts.drop_after(split_time) for ts in target_list]

        if is_local:
            preds = []
            for ts in pred_series:
                m = _local_builder()
                m.fit(ts)
                preds.append(m.predict(n=horizon))
        else:
            pred_kwargs = {"n": horizon, "series": pred_series}
            if past_for_fit is not None and model.supports_past_covariates:
                pred_kwargs["past_covariates"] = past_for_fit
            if fut_for_fit is not None and model.supports_future_covariates:
                pred_kwargs["future_covariates"] = fut_for_fit

            is_probabilistic = is_neural and getattr(model, "likelihood", None) is not None
            if is_probabilistic:
                pred_kwargs["num_samples"] = 200
            preds = model.predict(show_warnings=False, **pred_kwargs)
            if is_probabilistic:
                preds = [p.quantile(0.5) for p in preds]

        for r_idx, pred in enumerate(preds):
            all_fold_preds[r_idx].append(pred)

        yield all_fold_preds


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_region_horizon_heatmap(
    per_region_horizon,
    metric: str = "MASE",
    title=None,
    ax=None,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
):
    """Heatmap of a metric across regions * horizon days.

    vmin / vmax
        When both are None (default) the colour scale is derived from the
        data, giving full contrast for any metric range (RMSE, MAE, etc.).
        Pass vmin=0, vmax=1 for bounded metrics such as F1 or AUC.
    """
    pivot = per_region_horizon.pivot(index="region", columns="horizon", values=metric)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(1 + 0.8 * pivot.shape[1], 0.4 * pivot.shape[0] + 1))
    _vmin = np.nanmin(pivot.values) if vmin is None else vmin
    _vmax = np.nanmax(pivot.values) if vmax is None else vmax
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=_vmin, vmax=_vmax)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Horizon (days ahead)"); ax.set_ylabel("Region")
    ax.set_title(title or f"{metric} by region x horizon")
    mid = 0.5 * (_vmin + _vmax)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < mid else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-group training helpers
# ---------------------------------------------------------------------------

def run_expanding_cv_per_activity(
    builder_fn,
    target_list,
    region_names,
    regions_activity,
    start_frac,
    *,
    horizon: int,
    predict_stride: int = 1,
    retrain_stride: int,
    past_covs=None,
    future_covs=None,
    is_neural: bool = False,
    verbose: bool = True,
):
    """Train one model per activity-level group (levels 1, 2, 3).

    Internally calls ``run_final_test`` with decoupled predict_stride / retrain_stride,
    so daily predictions are issued even when retraining is weekly.  Pass
    ``target_for_cv`` + ``CV_START_VAL`` for the validation phase, or the full
    ``target_list`` + ``TRAIN_VAL_END`` for the test phase.

    Returns fold_preds assembled in the original region order so
    ``collect_predictions_long`` works unchanged.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, region in enumerate(region_names):
        groups[regions_activity[region]].append(i)

    all_fold_preds = [None] * len(target_list)

    for level in sorted(groups):
        indices = groups[level]
        if verbose:
            print(f"\n--- Activity level {level} ({len(indices)} regions) ---")
        group_targets = [target_list[i] for i in indices]
        group_past    = [past_covs[i]   for i in indices] if past_covs   is not None else None
        group_future  = [future_covs[i] for i in indices] if future_covs is not None else None

        group_preds = run_final_test(
            builder_fn, group_targets, start_frac,
            horizon=horizon,
            predict_stride=predict_stride,
            retrain_stride=retrain_stride,
            past_covs=group_past,
            future_covs=group_future,
            is_neural=is_neural,
            verbose=verbose,
        )
        for group_idx, orig_idx in enumerate(indices):
            all_fold_preds[orig_idx] = group_preds[group_idx]

    return all_fold_preds


def run_expanding_cv_per_region(
    builder_fn,
    target_list,
    start_frac,
    *,
    horizon: int,
    predict_stride: int = 1,
    retrain_stride: int,
    past_covs=None,
    future_covs=None,
    is_neural: bool = False,
    verbose: bool = True,
):
    """Train one independent model per region (local model with its own covariates).

    Internally calls ``run_final_test`` per region with a single-element list so
    each model never sees other regions' data.  Decoupled predict_stride /
    retrain_stride so daily predictions are issued between weekly retrains.
    Works for both CV and test phases via ``start_frac``.

    Returns fold_preds in original region order.
    """
    all_fold_preds = []
    for i in range(len(target_list)):
        if verbose:
            print(f"\n--- Region {i + 1}/{len(target_list)} ---")
        region_preds = run_final_test(
            builder_fn,
            [target_list[i]],
            start_frac,
            horizon=horizon,
            predict_stride=predict_stride,
            retrain_stride=retrain_stride,
            past_covs=[past_covs[i]]   if past_covs   is not None else None,
            future_covs=[future_covs[i]] if future_covs is not None else None,
            is_neural=is_neural,
            verbose=verbose,
        )
        all_fold_preds.append(region_preds[0])
    return all_fold_preds


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importances_per_horizon(darts_model, feature_names):
    """Works for LightGBM/XGBoost/CatBoost native Darts models."""
    try:
        estimators = darts_model.model.estimators_
    except AttributeError:
        return None
    imps = {"Feature": feature_names}
    for h, est in enumerate(estimators, start=1):
        try:
            imps[f"h{h}_importance"] = est.feature_importances_
        except AttributeError:
            try:
                imps[f"h{h}_importance"] = est.get_feature_importance()
            except Exception:
                return None
    return pd.DataFrame(imps)
