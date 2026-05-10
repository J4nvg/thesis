#!/usr/bin/env python
# coding: utf-8

# # Chronos-2 — fine-tuned forecasting of zero-inflated drone-strike counts
# 
# Single-model notebook: fine-tunes **AutoGluon's Chronos-2** on the per-region
# drone-strike series, hyperparameter-tunes via Optuna, and evaluates on the
# held-out test set with the same metric suite used by the GBDT notebook.
# 
# Pipeline:
# 1. Load + feature-engineer (identical to the GBDT notebook so leaderboards line up).
# 2. Build a `TimeSeriesDataFrame` with future covariates (holidays + weather),
#    past covariates (everything else), and one static covariate (`Activity_Level`).
# 3. Optuna search over `fine_tune_lr` * `fine_tune_steps` using AG-TS's internal
#    3-window backtest as the validation score.
# 4. Refit the winning configuration on `train_data`, persist the predictor.
# 5. Rolling 7-day backtest on the test segment for: zero-shot Chronos-2,
#    fine-tuned Chronos-2 (best Optuna trial), and the two naive floors.
# 6. Save per-region / per-horizon / per-region-horizon / global metrics for every
#    model so the comparison can be merged into the master leaderboard.
# 

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')

# PROJECT_DIR = '/content/drive/MyDrive/thesis/CODEBASE'
PROJECT_DIR = './'
# %cd $PROJECT_DIR


# In[2]:


# !nvidia-smi


# In[4]:


# !pip install -U autogluon.timeseries optuna


# In[5]:


import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


# ## Imports & config

# In[ ]:


import json
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import *
from src import _scaled_metrics, _skill  # not re-exported by import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)


# In[ ]:


# -------------------- CONFIG --------------------
DATA_FOLDER       = "./data"
FIXED_DATA_PATH   = construct_path(DATA_FOLDER, "fixed")
DATASET_PATH      = construct_path(DATA_FOLDER, "dataset")

TARGET            = "act_drone_strike_on_ua"
OUTPUT_CHUNK_LEN  = 7      # daily 7-step forecast horizon
CV_STRIDE         = 1      # daily rolling folds for the test backtest
NAIVE_SEASONALITY = 7      # weekly seasonality for MASE/RMSSE scales

# Train / val / test split: AG-TS gets train+val (80%) and decides its own
# internal validation; the last 20% is the untouched test segment.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20

# AG-TS internal backtesting during fit
NUM_VAL_WINDOWS   = 3      # see explanation cell below

# Optuna config (each trial fine-tunes Chronos-2 from scratch)
OPTUNA_N_TRIALS   = 12
OPTUNA_TIMEOUT_S  = None

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Output paths
RESULTS_DIR        = Path(PROJECT_DIR) / "results" / "chronos2"
CKPT_DIR           = Path(PROJECT_DIR) / "checkpoints"
TUNE_CKPT_DIR      = Path(PROJECT_DIR) / "checkpoints_tune"
BEST_PREDICTOR_DIR = CKPT_DIR / "chronos2_best"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TUNE_CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ## 1. Load data

# In[ ]:


regions,master_timeseries,regions_activity = load_data(data_path=FIXED_DATA_PATH,dataset_path=DATASET_PATH)


# ## 2. Feature engineering
# 
# Identical to the GBDT notebook so the comparison is apples-to-apples:
# 1. Identify weather / holiday columns (future covariates) and everything else (past covariates).
# 2. Pivot from wide to long-hierarchical on region.
# 3. Assign activity label per region; drop no-activity regions.
# 4. Drop low-prevalence ACLED categories.
# 5. Aggregate GDELT primary/secondary actors.
# 6. Fill infrastructure-damage NaNs with 0.

# In[9]:


# Identify weather columns
locale_env_weather_columns = [
    c for c in master_timeseries.columns
    if "env" in c and "holiday" not in c
]
global_weather_columns = list(set(
    "_".join(c.split("_")[:-1]) if c != "env_k_max" else c
    for c in locale_env_weather_columns
))

# Add total daily strike events (national aggregate)
master_timeseries['act_total_daily_strike_events'] = master_timeseries[
    [x for x in master_timeseries.columns.tolist() if 'act_drone_strike_on_ua' in x]
].sum(axis=1)
# Add total damage events
master_timeseries['act_total_damage_events'] = master_timeseries[
    [x for x in master_timeseries.columns.tolist() if 'act_drone_infra_ua' in x]
].sum(axis=1)

# Long hierarchical
for_global = get_pivoted_table(df=master_timeseries.reset_index(), regions=regions)

# Drop low-activity regions
for_global["Activity_Level"] = for_global.index.get_level_values("region").map(regions_activity)
for_global = for_global[for_global["Activity_Level"] != 0]

# Drop low-prevalence ACLED "other" categories
for_global, _removed = remove_low_prevalence(df=for_global, ratio=0.1, specific="acled_other_")

# Aggregate GDELT primary/secondary
gdelt_cols = (
    set(get_all_X("com_", for_global))
    - set(get_all_X("com_ners", for_global))
    - set(get_all_X("com_aid",  for_global))
)
actors = load_actors_dict(construct_path(FIXED_DATA_PATH, "actors.json"))
for_global = aggregate_gdelt_primary_secondary(df=for_global, columns=gdelt_cols, actors=actors)

# Fix NaN in infra
infra_cols = get_all_X("act_drone_infra_ua_", for_global)
for_global[infra_cols] = for_global[infra_cols].fillna(0)

print(f"Shape after feature engineering: {for_global.shape}")
print(f"Target positive rate: {for_global[TARGET].mean():.3f}")
print(f"Target mean: {for_global[TARGET].mean():.3f},  max: {for_global[TARGET].max()}")


# In[ ]:


# Future vs past covariate split
holiday_cols, future_covariates, exclude_cols, past_covariates = split_future_and_past_cov(for_global,global_weather_columns,TARGET)


# ## 3. Build `TimeSeriesDataFrame`
# 
# AutoGluon-TS expects a long-format dataframe indexed by `(item_id, timestamp)`.
# Our `(region, event_date)` MultiIndex maps directly. Static features
# (`Activity_Level`) live in a separate small dataframe attached to the
# `TimeSeriesDataFrame`.

# In[11]:


from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Long-format with item_id / timestamp index
ts_df = (
    for_global
    .reset_index()
    .rename(columns={"region": "item_id", "event_date": "timestamp"})
    .sort_values(["item_id", "timestamp"])
    .set_index(["item_id", "timestamp"])
)
# Pull static feature off the panel
static_df = (
    ts_df.reset_index()[["item_id", "Activity_Level"]]
         .drop_duplicates()
         .set_index("item_id")
)
ts_df = ts_df.drop(columns=["Activity_Level"])

tsdf = TimeSeriesDataFrame(ts_df, static_features=static_df)
print(f"Items (regions): {tsdf.num_items}")
print(f"Length per item: {tsdf.num_timesteps_per_item().min()} (min) "
      f"/ {tsdf.num_timesteps_per_item().max()} (max)")
print(f"Frequency      : {tsdf.freq}")
print(f"Total columns  : {tsdf.shape[1]}  "
      f"(target=1, future_cov={len(future_covariates)}, past_cov={len(past_covariates)})")


# In[12]:


# Train+val (80%) for fitting; test (20%) held back for the final backtest.
n = int(tsdf.num_timesteps_per_item().min())
test_size = int(round(TEST_FRAC * n))
train_data, test_data_ag = tsdf.train_test_split(prediction_length=test_size)

print(f"Total timesteps per item: {n}")
print(f"train_data length:        {int(train_data.num_timesteps_per_item().min())} "
      f"(~{(1 - TEST_FRAC) * 100:.0f}%)")
print(f"test segment length:      {test_size} "
      f"(~{TEST_FRAC * 100:.0f}%) — kept fully separate, never touched until §6")


# ## 4. Evaluation utilities
# 
# Same metric philosophy as the GBDT notebook, minus the count-likelihood
# deviance metrics — Chronos-2 produces quantile regression outputs, not Poisson
# or Tweedie parameters, so `mean_poisson_deviance` on the median quantile
# isn't a meaningful proper scoring rule for this model. We keep:
# 
# | Metric | Answers |
# |---|---|
# | **MAE**    | Average miss size (L1, robust to zero inflation) |
# | **RMSE**   | Worst-miss sensitivity |
# | **MedAE**  | Typical-day miss (ignores the right tail) |
# | **ME**     | Bias direction (`+` over-forecasts, `-` under-forecasts) |
# | **ZeroAcc**| Fraction of zeros classified correctly (`pred < 0.5` vs `y_true == 0`) |
# | **MASE**   | Beats seasonal naive? (`< 1` = yes; primary ranking metric) |
# | **RMSSE**  | Squared-error twin of MASE (M5 standard) |
# 
# Four views of quality, identical in shape to the GBDT notebook so leaderboards
# can be concatenated row-wise:
# 
# | scope                | granularity                        |
# | -------------------- | ---------------------------------- |
# | `per_region`         | one row per region (folds * horizons pooled) |
# | `per_horizon`        | one row per horizon step 1..7      |
# | `per_region_horizon` | region * horizon (feeds the heatmap) |
# | `global`             | pooled metrics + per-region MASE/RMSSE averaged |
# 

# In[13]:


# ---------------------------------------------------------------
# Per-region in-sample MAE/RMSE scales for MASE / RMSSE.
# Computed on TRAINING data only — never the validation/test segment.
# ---------------------------------------------------------------
def compute_naive_scales_from_tsdf(train_tsdf, target_col, seasonality=7):
    """Per-region in-sample mean |y_t - y_{t-s}| and sqrt(mean (y_t - y_{t-s})^2)."""
    mae_s, rmse_s = {}, {}
    for item_id in train_tsdf.item_ids:
        y = train_tsdf.loc[item_id][target_col].to_numpy(dtype=float).ravel()
        if len(y) <= seasonality:
            mae_s[item_id] = rmse_s[item_id] = np.nan
            continue
        err = y[seasonality:] - y[:-seasonality]
        mae_s[item_id]  = float(np.mean(np.abs(err)))
        rmse_s[item_id] = float(np.sqrt(np.mean(err ** 2)))
    return mae_s, rmse_s


MAE_SCALES, RMSE_SCALES = compute_naive_scales_from_tsdf(
    train_data, TARGET, seasonality=NAIVE_SEASONALITY,
)
print(f"Computed naive (lag={NAIVE_SEASONALITY}) scales for {len(MAE_SCALES)} regions")


# In[14]:


# ---------------------------------------------------------------
# Metric primitives: clipped to >= 0 because count predictions can't be
# negative; Chronos-2 quantiles can occasionally drop below zero on sparse
# series.
# ---------------------------------------------------------------
def base_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0.0, None)
    err    = y_pred - y_true
    return {
        "MAE":     float(np.mean(np.abs(err))),
        "RMSE":    float(np.sqrt(np.mean(err ** 2))),
        "MedAE":   float(np.median(np.abs(err))),
        "ME":      float(np.mean(err)),
        "ZeroAcc": float(np.mean((y_true == 0) == (y_pred < 0.5))),
        "n":       int(len(y_true)),
    }


def _scaled_metrics(sub_df, mae_scale, rmse_scale):
    """MASE and RMSSE for a region-scoped subset."""
    y_true = sub_df["y_true"].to_numpy(dtype=float)
    y_pred = np.clip(sub_df["y_pred"].to_numpy(dtype=float), 0.0, None)
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {
        "MASE":  float(mae  / mae_scale)  if mae_scale  and mae_scale  > 0 else np.nan,
        "RMSSE": float(rmse / rmse_scale) if rmse_scale and rmse_scale > 0 else np.nan,
    }


# ---------------------------------------------------------------
# evaluate_long: takes a long-format dataframe with columns
# [region, fold, horizon, date, y_true, y_pred] and returns the four views.
# ---------------------------------------------------------------
def evaluate_long(long_df):
    """Returns dict with keys: per_region, per_horizon, per_region_horizon, global."""
    df = long_df.copy()

    # ---- per_region (pooled folds * horizons) ----
    rows = []
    for region, sub in df.groupby("region", sort=False):
        m = base_metrics(sub["y_true"], sub["y_pred"])
        s = _scaled_metrics(sub, MAE_SCALES.get(region), RMSE_SCALES.get(region))
        rows.append({"region": region, **m, **s})
    per_region = pd.DataFrame(rows)

    # ---- per_horizon (pooled folds * regions) ----
    rows = []
    for h, sub in df.groupby("horizon", sort=True):
        m = base_metrics(sub["y_true"], sub["y_pred"])
        rows.append({"horizon": int(h), **m})
    per_horizon = pd.DataFrame(rows)

    # ---- per_region_horizon (cell granularity, drives the heatmap) ----
    rows = []
    for (region, h), sub in df.groupby(["region", "horizon"], sort=True):
        m = base_metrics(sub["y_true"], sub["y_pred"])
        s = _scaled_metrics(sub, MAE_SCALES.get(region), RMSE_SCALES.get(region))
        rows.append({"region": region, "horizon": int(h), **m, **s})
    per_region_horizon = pd.DataFrame(rows)

    # ---- global (pooled everything + per-region MASE/RMSSE averaged) ----
    global_metrics = base_metrics(df["y_true"], df["y_pred"])
    global_metrics["MASE_mean"]   = float(per_region["MASE"].mean(skipna=True))
    global_metrics["MASE_median"] = float(per_region["MASE"].median(skipna=True))
    global_metrics["RMSSE_mean"]  = float(per_region["RMSSE"].mean(skipna=True))

    return {
        "per_region":         per_region,
        "per_horizon":        per_horizon,
        "per_region_horizon": per_region_horizon,
        "global":             global_metrics,
    }


# ## 5. Rolling 7-day backtest helper
# 
# Replicates the `run_expanding_cv` semantics from the GBDT notebook for an
# AG-TS predictor:
# - start at the first day of the test segment,
# - step forward by `CV_STRIDE = 1` (daily rolling folds, matching all other notebooks),
# - at each fold, condition on all data up to `t0` and forecast 7 days ahead,
# - collect predictions in long format keyed by `(region, fold, horizon, date)`.
# 
# Naive baselines (`naive_last`, `naive_weekly`) emit predictions in the same
# shape so they all flow into `evaluate_long` unchanged.
# 

# In[15]:


# AG-TS prediction columns can carry numeric names like 0.5 (float) or
# '0.5' (str) or 'mean' depending on quantile_levels and version.
def _pick_median_col(pred_df):
    for cand in [0.5, "0.5", "mean"]:
        if cand in pred_df.columns:
            return cand
    raise KeyError(f"Cannot find median column in {pred_df.columns.tolist()}")


def chronos2_rolling_long(predictor, full_tsdf, test_start_idx,
                          target_col=TARGET, future_cov_cols=None,
                          horizon=OUTPUT_CHUNK_LEN, stride=CV_STRIDE):
    """Expanding-window 7-day backtest for an AG-TS predictor.

    For each fold t0 in [test_start_idx, n - horizon + 1, stride):
      context = full_tsdf[: t0]
      known   = full_tsdf[t0 : t0+horizon][future_cov_cols]   (= 'future covariates')
      pred    = predictor.predict(context, known_covariates=known)

    Returns long-format DF with columns: region, fold, horizon, date, y_true, y_pred.
    """
    n_per_item = int(full_tsdf.num_timesteps_per_item().iloc[0])
    long_rows = []
    fold_idx = 0
    for t0 in range(test_start_idx, n_per_item - horizon + 1, stride):
        context = full_tsdf.slice_by_timestep(None, t0)
        future_slice = full_tsdf.slice_by_timestep(t0, t0 + horizon)
        known = future_slice[future_cov_cols] if future_cov_cols else None

        pred = predictor.predict(context, known_covariates=known)
        median_col = _pick_median_col(pred)
        pred_df = pred[[median_col]].rename(columns={median_col: "y_pred"}).reset_index()

        truth_df = (future_slice[[target_col]]
                    .rename(columns={target_col: "y_true"})
                    .reset_index())

        merged = pred_df.merge(truth_df, on=["item_id", "timestamp"])
        # Per-region horizon index (1..H), assumes timestamps are sorted within group
        merged = merged.sort_values(["item_id", "timestamp"])
        merged["horizon"] = merged.groupby("item_id").cumcount() + 1
        merged["fold"]    = fold_idx
        merged = merged.rename(columns={"item_id": "region", "timestamp": "date"})
        long_rows.append(merged[["region", "fold", "horizon", "date", "y_true", "y_pred"]])

        fold_idx += 1
        if fold_idx == 1 or fold_idx % 4 == 0:
            last_date = future_slice.index.get_level_values("timestamp")[-1].date()
            print(f"    fold {fold_idx}: forecasted up to {last_date}")

    print(f"    {fold_idx} folds complete")
    return pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()


# ## 6. Naive baselines (floor)
# 
# Two zero-parameter floors any model must clear, computed directly from the
# `TimeSeriesDataFrame` so the long-format output flows into `evaluate_long`
# unchanged:
# 
# - **`naive_last`** — `y_hat(t+h) = y(t-1)` for every horizon step.
# - **`naive_weekly`** — `y_hat(t+h) = y(t+h-7)`.

# In[16]:


def naive_rolling_long(full_tsdf, test_start_idx, method="weekly",
                       target_col=TARGET, horizon=OUTPUT_CHUNK_LEN, stride=CV_STRIDE):
    """Closed-form expanding-window predictions in the same long-format shape."""
    n_per_item = int(full_tsdf.num_timesteps_per_item().iloc[0])
    rows = []
    for region in full_tsdf.item_ids:
        y = full_tsdf.loc[region][target_col].to_numpy(dtype=float).ravel()
        ts_index = full_tsdf.loc[region].index
        fold_idx = 0
        for t0 in range(test_start_idx, n_per_item - horizon + 1, stride):
            for h in range(1, horizon + 1):
                date = ts_index[t0 + h - 1]
                if method == "last":
                    y_pred = float(y[t0 - 1])
                elif method == "weekly":
                    if t0 + h - 1 - 7 < 0:
                        continue
                    y_pred = float(y[t0 + h - 1 - 7])
                else:
                    raise ValueError(method)
                rows.append({
                    "region":  region,
                    "fold":    fold_idx,
                    "horizon": h,
                    "date":    date,
                    "y_true":  float(y[t0 + h - 1]),
                    "y_pred":  y_pred,
                })
            fold_idx += 1
    return pd.DataFrame(rows)


# ## 7. Reference run: zero-shot Chronos-2
# 
# Before tuning, fit a *zero-shot* Chronos-2 (no fine-tuning) so we have a
# baseline that isolates the value of fine-tuning vs the value of the
# foundation model itself.

# In[22]:


ZEROSHOT_DIR = CKPT_DIR / "chronos2_zeroshot"
if ZEROSHOT_DIR.exists():
    print(f"Loading cached zero-shot predictor from {ZEROSHOT_DIR}")
    predictor_zs = TimeSeriesPredictor.load(str(ZEROSHOT_DIR))
else:
    predictor_zs = TimeSeriesPredictor(
        prediction_length=OUTPUT_CHUNK_LEN,
        target=TARGET,
        known_covariates_names=future_covariates,
        eval_metric="MASE",
        freq="D",
        path=str(ZEROSHOT_DIR),
    ).fit(
        train_data=train_data,
        hyperparameters={"Chronos2": [{"ag_args": {"name_suffix": "ZeroShot"}}]},
        enable_ensemble=False,
        num_val_windows=NUM_VAL_WINDOWS,
        random_seed=RANDOM_STATE,
        # skip_model_selection=True,
    )

print("\n--- Zero-shot internal leaderboard (val MASE = -score_val) ---")
print(predictor_zs.leaderboard())


# ## 8. Hyperparameter tuning (Optuna)
# 
# We tune the two knobs that actually move the needle for Chronos-2 fine-tuning:
# 
# | Param | Range | Why |
# |---|---|---|
# | `fine_tune_lr` | log-uniform `[1e-6, 1e-4]` | Below 1e-6 the foundation weights barely move; above 1e-4 we risk catastrophic forgetting on a dataset this small. |
# | `fine_tune_steps` | `[200, 3000]` | Below 200 fine-tuning hasn't converged; above 3000 we overfit on ~1200 train days. |
# 
# The objective is AG-TS's internal validation score (mean MASE across the 3
# backtest windows). Each trial fits a fresh predictor in a tempdir; the best
# trial's params are persisted, and the final predictor is refit and saved
# afterwards in §9.

# In[18]:


import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_CKPT = TUNE_CKPT_DIR / "chronos2_finetune_best.pkl"


def chronos2_objective(trial):
    fine_tune_lr    = trial.suggest_float("fine_tune_lr", 1e-6, 1e-4, log=True)
    fine_tune_steps = trial.suggest_int("fine_tune_steps", 200, 3000, step=100)

    trial_dir = tempfile.mkdtemp(prefix="ag_chronos2_trial_")
    try:
        pred = TimeSeriesPredictor(
            prediction_length=OUTPUT_CHUNK_LEN,
            target=TARGET,
            known_covariates_names=future_covariates,
            eval_metric="MASE",
            freq="D",
            path=trial_dir,
            verbosity=0,
        ).fit(
            train_data=train_data,
            hyperparameters={
                "Chronos2": {
                    "fine_tune":       True,
                    "fine_tune_lr":    fine_tune_lr,
                    "fine_tune_steps": fine_tune_steps,
                    "ag_args": {"name_suffix": "FT"},
                }
            },
            enable_ensemble=False,
            num_val_windows=NUM_VAL_WINDOWS,
            random_seed=RANDOM_STATE,
            verbosity=0,
        )
        # AG-TS reports MASE as -score (higher is better convention) — flip back.
        score_val = pred.leaderboard().iloc[0]["score_val"]
        mase = -float(score_val)
        trial.set_user_attr("predictor_path", trial_dir)
        return mase
    except Exception as e:
        # Cleanup on failure; let Optuna treat it as a bad trial.
        shutil.rmtree(trial_dir, ignore_errors=True)
        print(f"[trial {trial.number}] FAILED: {type(e).__name__}: {e}")
        raise


# In[19]:


if OPTUNA_CKPT.exists():
    with open(OPTUNA_CKPT, "rb") as f:
        best_params, study = pickle.load(f)
    print(f"Loaded cached Optuna study (best MASE = {study.best_value:.4f})")
    print(f"Best params: {best_params}")
else:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        chronos2_objective,
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT_S,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    best_params = study.best_params
    with open(OPTUNA_CKPT, "wb") as f:
        pickle.dump((best_params, study), f)

    # Cleanup non-best trial dirs
    best_path = study.best_trial.user_attrs.get("predictor_path")
    for t in study.trials:
        p = t.user_attrs.get("predictor_path")
        if p and p != best_path and Path(p).exists():
            shutil.rmtree(p, ignore_errors=True)

    print(f"\nBest val MASE: {study.best_value:.4f}")
    print(f"Best params:   {best_params}")

# Quick optimization-history dump
trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
trials_df.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
print(trials_df.head(20))


# ## 9. Refit & persist the best Chronos-2
# 
# We refit the best trial's configuration into a permanent location
# (`./checkpoints/chronos2_best/`). This is the artefact you'd ship to inference
# and the one §10 evaluates.

# In[ ]:


# Wipe any previous best (safe — only this directory)
if BEST_PREDICTOR_DIR.exists():
    shutil.rmtree(BEST_PREDICTOR_DIR)

predictor_ft = TimeSeriesPredictor(
    prediction_length=OUTPUT_CHUNK_LEN,
    target=TARGET,
    known_covariates_names=future_covariates,
    eval_metric="MASE",
    freq="D",
    path=str(BEST_PREDICTOR_DIR),
).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos2": {
            "fine_tune":       True,
            "fine_tune_lr":    best_params["fine_tune_lr"],
            "fine_tune_steps": best_params["fine_tune_steps"],
            "ag_args": {"name_suffix": "FT_best"},
        }
    },
    enable_ensemble=False,
    num_val_windows=NUM_VAL_WINDOWS,
    random_seed=RANDOM_STATE,
)

# Persist a json sidecar with the winning config + provenance
with open(BEST_PREDICTOR_DIR / "best_params.json", "w") as f:
    json.dump({
        "best_params":     best_params,
        "best_val_mase":   float(study.best_value) if not OPTUNA_CKPT.exists() else None,
        "n_trials":        len(study.trials),
        "num_val_windows": NUM_VAL_WINDOWS,
        "prediction_length": OUTPUT_CHUNK_LEN,
        "target":          TARGET,
        "future_covariates": future_covariates,
        "n_past_covariates": len(past_covariates),
    }, f, indent=2, default=str)

print(f"\nBest predictor saved to: {BEST_PREDICTOR_DIR.resolve()}")
print(predictor_ft.leaderboard())


# ## 10. Test-set rolling backtest
# 
# Now we hit the held-out 20% segment. For each model we run an expanding-window
# daily backtest with `stride=1` (matching all other notebooks), then push the
# long-format predictions through `evaluate_long` to get the four metric views.
# 

# In[ ]:


# Index into the full series where the test segment begins
n_full        = int(tsdf.num_timesteps_per_item().min())
test_start_ix = n_full - test_size   # first index in the test segment

print(f"Test starts at timestep {test_start_ix} / {n_full}")
print(f"Number of test folds (stride={CV_STRIDE}): "
      f"{(n_full - test_start_ix - OUTPUT_CHUNK_LEN) // CV_STRIDE + 1}")

long_by_model = {}

# ---- chronos2_zero_shot ----
print("\n=== chronos2_zero_shot rolling backtest ===")
long_by_model["chronos2_zero_shot"] = chronos2_rolling_long(
    predictor_zs, tsdf, test_start_ix,
    target_col=TARGET, future_cov_cols=future_covariates,
)

# ---- chronos2_fine_tuned (best) ----
print("\n=== chronos2_fine_tuned (best Optuna) rolling backtest ===")
long_by_model["chronos2_fine_tuned"] = chronos2_rolling_long(
    predictor_ft, tsdf, test_start_ix,
    target_col=TARGET, future_cov_cols=future_covariates,
)

# ---- naive_last ----
print("\n=== naive_last rolling backtest ===")
long_by_model["naive_last"] = naive_rolling_long(
    tsdf, test_start_ix, method="last", target_col=TARGET,
)

# ---- naive_weekly ----
print("\n=== naive_weekly rolling backtest ===")
long_by_model["naive_weekly"] = naive_rolling_long(
    tsdf, test_start_ix, method="weekly", target_col=TARGET,
)

for name, df in long_by_model.items():
    print(f"  {name:25s}  rows={len(df):6d}  regions={df['region'].nunique()}  "
          f"horizons={sorted(df['horizon'].unique())}")


# In[ ]:


# ---------------- Compute all four metric views per model ----------------
results_by_model = {name: evaluate_long(df) for name, df in long_by_model.items()}

# ---------------- Cross-model leaderboard (sorted by MASE_mean) ----------------
ref = results_by_model.get("naive_weekly", {}).get("global")

def _skill(model_val, ref_val):
    if ref_val is None or ref_val == 0 or np.isnan(ref_val):
        return np.nan
    return float(1.0 - model_val / ref_val)

leaderboard_rows = []
for name, res in results_by_model.items():
    g = res["global"]
    leaderboard_rows.append({
        "model":       name,
        "MAE":         g["MAE"],
        "RMSE":        g["RMSE"],
        "MedAE":       g["MedAE"],
        "ME":          g["ME"],
        "ZeroAcc":     g["ZeroAcc"],
        "MASE_mean":   g["MASE_mean"],
        "MASE_median": g["MASE_median"],
        "RMSSE_mean":  g["RMSSE_mean"],
        "SkillRMSE":   _skill(g["RMSE"], ref["RMSE"]) if ref else np.nan,
        "SkillMAE":    _skill(g["MAE"],  ref["MAE"])  if ref else np.nan,
    })

leaderboard = (
    pd.DataFrame(leaderboard_rows)
      .sort_values("MASE_mean", ascending=True)
      .reset_index(drop=True)
)
leaderboard


# ## 11. Persist all results
# 
# For each model, we save:
# - `predictions_long_<model>.parquet`  — every (region, fold, horizon, date) prediction
# - `per_region_<model>.csv`            — one row per region (folds * horizons pooled)
# - `per_horizon_<model>.csv`           — one row per horizon step
# - `per_region_horizon_<model>.csv`    — region * horizon (heatmap source)
# - `global_<model>.json`               — pooled metrics + scale-free averages
# 
# Plus the cross-model artefacts:
# - `leaderboard.csv`                   — comparison table sorted by MASE_mean
# - `optuna_trials.csv`                 — full Optuna trial history (saved earlier)

# In[ ]:


def _safe_name(name):
    return name.replace("/", "_").replace(" ", "_")

for name, res in results_by_model.items():
    safe = _safe_name(name)
    long_by_model[name].to_parquet(RESULTS_DIR / f"predictions_long_{safe}.parquet", index=False)
    res["per_region"].to_csv(           RESULTS_DIR / f"per_region_{safe}.csv",         index=False)
    res["per_horizon"].to_csv(          RESULTS_DIR / f"per_horizon_{safe}.csv",        index=False)
    res["per_region_horizon"].to_csv(   RESULTS_DIR / f"per_region_horizon_{safe}.csv", index=False)
    with open(RESULTS_DIR / f"global_{safe}.json", "w") as f:
        json.dump(res["global"], f, indent=2, default=float)
    print(f"  saved: {name}")

leaderboard.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
print(f"\nAll results in: {RESULTS_DIR.resolve()}")
print(sorted(p.name for p in RESULTS_DIR.iterdir()))


# ## 12. Per-region detail for the winning model

# In[ ]:


winner = leaderboard.iloc[0]["model"]
print(f"Winning model on MASE_mean: {winner}\n")
print("--- per_region ---")
print(results_by_model[winner]["per_region"].sort_values("MASE"))
print("\n--- per_horizon ---")
print(results_by_model[winner]["per_horizon"])


# In[ ]:


def plot_region_horizon_heatmap(per_region_horizon, metric="MASE",
                                title=None, ax=None, cmap="viridis"):
    pivot = per_region_horizon.pivot(index="region", columns="horizon", values=metric)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(1 + 0.8 * pivot.shape[1], 0.4 * pivot.shape[0] + 1))
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Horizon (days ahead)"); ax.set_ylabel("Region")
    ax.set_title(title or f"{metric} by region * horizon — {winner}")
    mid = 0.5 * (vmin + vmax)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < mid else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    return ax


plot_region_horizon_heatmap(
    results_by_model[winner]["per_region_horizon"], metric="MASE",
)
plt.savefig(RESULTS_DIR / f"heatmap_{_safe_name(winner)}_MASE.png", dpi=120, bbox_inches="tight")
plt.show()


# In[ ]:


# Summary
print("=" * 70)
print(f"Best Chronos-2 fine-tune params: {best_params}")
print(f"Saved predictor:                 {BEST_PREDICTOR_DIR.resolve()}")
print(f"Results directory:               {RESULTS_DIR.resolve()}")
print(f"Leaderboard winner (MASE_mean):  {winner}")
print("=" * 70)

