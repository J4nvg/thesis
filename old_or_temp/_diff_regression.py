#!/usr/bin/env python
# coding: utf-8

# # Differenced regression — basic regressors only
# 
# Companion notebook to `_regression_GBDT.ipynb` and `_regression_LSTM.ipynb`.
# 
# **What's different here**
# * The target is **differenced once** (`y'_t = y_t - y_{t-1}`) before any model
#   sees it. The diffed series is much closer to stationary, which lets the
#   basic regressors compete on a level playing field with their plain MSE
#   loss instead of needing count-aware likelihoods.
# * Every model is trained on the diffed target. Predictions in diff-space
#   are un-diffed back to levels (anchor = last actual value before the
#   forecast window) so the leaderboard / MASE / RMSSE / etc. stay
#   comparable to the count-based notebooks.
# 
# **Lineup** (all hyper-parameter tuned on the validation CV folds, except
# linear / arima / naives which run with default configs):
# * `lightgbm` — `objective="regression"` (MSE)
# * `xgboost`  — `objective="reg:squarederror"`
# * `catboost` — `loss_function="RMSE"`
# * `lstm`     — Darts `BlockRNNModel` with default MSE loss
# * `arima`    — fed the diffed series with `d=0` (per region, no covariates)
# * `linear`   — Darts `LinearRegressionModel` — no tunable knobs, included as a floor
# * `naive_last`, `naive_weekly` — same persistence baselines as the other notebooks
# 
# Evaluation reuses the same expanding-window CV, naive scales (computed on
# the **level** training series), and metric pipeline.

# In[14]:


# from google.colab import drive
# drive.mount('/content/drive')

# PROJECT_DIR = '/content/drive/MyDrive/thesis/CODEBASE'
PROJECT_DIR = './'
# %cd $PROJECT_DIR


# In[15]:


# !nvidia-smi


# In[ ]:




# In[17]:


# !pip install "darts[all]" statsmodels optuna comet_ml


# In[18]:


import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


# Imports and config

# In[19]:


import comet_ml
from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="vxoDZPOZIECxzRsS9P8X15IrV",
  project_name="thesis",
  workspace="jan-van-gestel"
)


# In[ ]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import *
from src import _maybe_scale_covs, _skill  # not re-exported by import *
from darts import TimeSeries
from darts.dataprocessing.transformers import (
    WindowTransformer, StaticCovariatesTransformer, Scaler, Diff,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)


available_threads = get_available_threads()
print(f'CPU count: {available_threads}')

# 

# In[21]:


# -------------------- CONFIG --------------------
DATA_FOLDER       = "./data"
FIXED_DATA_PATH   = construct_path(DATA_FOLDER, "fixed")
DATASET_PATH      = construct_path(DATA_FOLDER, "dataset")

TARGET            = "act_drone_strike_on_ua"

OUTPUT_CHUNK_LEN  = 7
INPUT_LAGS        = 7
MULTI_MODELS      = True

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20
CV_STRIDE                       = 7

# ---- Model groups ---------------------------------------------------------
# Basic regressors only — all use plain regression losses (MSE / RMSE).
# Differencing is applied to the target outside the model, so every learner
# sees a (near-)stationary signal and can be trained with its default loss.
NAIVE_MODELS    = {"naive_last", "naive_weekly"}
NEURAL_MODELS   = {"lstm"}
LOCAL_MODELS    = {"arima"}
GBM_VARIANTS    = ["lightgbm", "xgboost", "catboost"]
LINEAR_VARIANTS = ["linear"]    # no tuning — included as a floor

REGRESSORS_TO_RUN = [
    "naive_last", "naive_weekly",
    "arima",
    "linear",
]

# Optuna config
OPTUNA_N_TRIALS  = 50    # per study
OPTUNA_TIMEOUT_S = None  # optional wall-clock cap per variant

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import torch
torch.manual_seed(RANDOM_STATE)


# ## 1. Load data

# In[ ]:


regions,master_timeseries,regions_activity = load_data(data_path=FIXED_DATA_PATH,dataset_path=DATASET_PATH)


# In[ ]:


for_global_reset, global_weather_columns = get_engineered_features(
    master_timeseries=master_timeseries,
    data_path=FIXED_DATA_PATH,
    target_col=TARGET,
    regions=regions,
    regions_activity=regions_activity,
    binarize_target=False
)


# In[24]:


print(for_global_reset.head())


# In[25]:


print(for_global_reset.isna().any()[lambda x: x])
# print(for_global_reset[for_global_reset["Activity_Level"].isna() == True][['Activity_Level','event_date','region']])


# In[ ]:


# Future vs past covariate split
holiday_cols, future_covariates, exclude_cols, past_covariates = split_future_and_past_cov(for_global_reset,global_weather_columns,TARGET)


# ## Build Darts TimeSeries and apply windowed transforms
# Getting lag
#  variabels

# In[27]:


target_series_list, past_covs_list,future_covs_list = build_ts_and_apply_window_transformer(for_global_reset,TARGET,past_covariates,future_covariates,ed_alpha=halflife_to_alpha(7))


# In[28]:


raw_past_covs_list = TimeSeries.from_group_dataframe(
    for_global_reset,
    group_cols="region", time_col="event_date",
    value_cols=past_covariates,
)


# ## Encode static covariates and split 70/10/20
# 

# In[29]:


region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END,CV_START_VAL =\
      get_covs_and_encodings(target_series_list,past_covs_list,future_covs_list,TRAIN_FRAC,VAL_FRAC)


# ## Difference the target series
# 
# The drone-strike count series are non-stationary (slow drifts, level shifts).
# Apply a single first-order difference per region: `y'_t = y_t - y_{t-1}`.
# 
# * Every learner downstream is trained on the **diffed** target — its default
#   regression loss (MSE / RMSE) becomes appropriate, no count-aware likelihood
#   needed.
# * Past and future covariates stay raw (the learners can lag-shift them
#   themselves; no need to diff features as well).
# * At prediction time we **un-diff** by anchoring on the last actual level
#   before the forecast window: `y_hat(t+h) = y(t-1) + cumsum(y'_hat)`. The
#   un-diffing happens inside `run_expanding_cv` so callers always get
#   predictions in the original level space and the existing eval pipeline
#   works unchanged.
# 

# In[30]:


# Per-region first-order difference of the target. We hold on to BOTH lists:
# the diffed list trains the model, the level list (target_for_cv from above)
# is what predictions get un-diffed onto + scored against.

diff_transformer = Diff(lags=1, dropna=True)
target_series_diff_list = diff_transformer.fit_transform(target_series_list)

_, train_target_diff, val_target_diff, test_target_diff, _, _, target_for_cv_diff, _, _ = \
    get_covs_and_encodings(target_series_diff_list, past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC)

print(f"original len (region 0): {len(target_series_list[0])}")
print(f"diffed   len (region 0): {len(target_series_diff_list[0])}  (loses 1 timestep)")
print(f"diffed train/val/test:   {len(train_target_diff[0])} / {len(val_target_diff[0])} / {len(test_target_diff[0])}")
print(f"target_for_cv_diff len:  {len(target_for_cv_diff[0])}")


# ## Regressors
# 
# Each model gets the **same** `INPUT_LAGS`, `OUTPUT_CHUNK_LEN`, encoders and
# covariates so only the learner changes. All GBDTs use a plain regression
# objective (MSE / RMSE) because differenced counts can be negative — Poisson /
# Tweedie likelihoods don't apply here.
# 
# **What's worth tuning** (★ = high impact, · = secondary):
# * GBDTs — `★ n_estimators + learning_rate`, `★ max_depth / num_leaves / depth`,
#   `★ min_child_samples / min_child_weight / l2_leaf_reg`,
#   `· subsample`, `· colsample_bytree`, `· reg_alpha / reg_lambda`.
# * `lstm` — `★ hidden_dim`, `★ n_rnn_layers`, `★ dropout`,
#   `· batch_size`, `· lr`, `· weight_decay`. Default MSE loss; covariates
#   scaled, target left as-is (it's already centered around zero post-diff).
# * `arima` — `★ p, q`, `★ d` left at 0 since the input is already diffed.
# * `linear` — included as a floor (no real tuning surface beyond the shared lags).
# 

# In[ ]:


from pytorch_lightning.callbacks import EarlyStopping
# --- Shared forecasting skeleton: every tabular model gets the same inputs ---
COMMON_KWARGS_TAB = get_common_kwargs()

ES_NN = EarlyStopping(monitor="train_loss", patience=10, min_delta=1e-4, mode="min")

NN_TRAINER_KWARGS = dict(
    accelerator          = "auto",
    enable_progress_bar  = True,
    enable_model_summary = True,
    log_every_n_steps    = 10,
    callbacks            = [ES_NN],
    gradient_clip_val    = 1.0,
)


def build_regressor(name: str):
    """Return a Darts forecasting model ready to ``.fit()`` on the DIFFED target.

    All GBDTs use plain regression losses; the LSTM uses the default MSE; ARIMA
    runs on the already-diffed series with d=0; the linear baseline has no
    tuning surface beyond the shared `INPUT_LAGS`.
    """
    name = name.lower()

    # ---------------- Gradient boosters (defaults / fallback configs) ------
    if name == "lightgbm":
        from darts.models import LightGBMModel
        return LightGBMModel(
            **COMMON_KWARGS_TAB,
            objective         = "regression",         # MSE
            num_leaves        = 31,
            max_depth         = 5,
            min_child_samples = 30,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            learning_rate     = 0.05,
            n_estimators      = 500,
            reg_alpha         = 0.0,
            reg_lambda        = 0.0,
            random_state      = RANDOM_STATE,
            verbose           = -1,
            device_type       = "cpu",
            num_threads       = available_threads,
            force_col_wise    = True,
        )

    if name == "xgboost":
        from darts.models import XGBModel
        return XGBModel(
            **COMMON_KWARGS_TAB,
            objective         = "reg:squarederror",   # MSE
            max_depth         = 5,
            min_child_weight  = 3,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            learning_rate     = 0.05,
            n_estimators      = 500,
            reg_alpha         = 0.0,
            reg_lambda        = 1.0,
            tree_method       = "hist",
            device            = "cpu",
            n_jobs            = available_threads,
            random_state      = RANDOM_STATE,
            verbosity         = 0,
        )

    if name == "catboost":
        from darts.models import CatBoostModel
        return CatBoostModel(
            **COMMON_KWARGS_TAB,
            loss_function     = "RMSE",
            depth             = 5,
            learning_rate     = 0.05,
            iterations        = 500,
            l2_leaf_reg       = 3,
            subsample         = 0.8,
            bootstrap_type    = "Bernoulli",
            task_type         = "CPU",
            thread_count      = available_threads,
            random_seed       = RANDOM_STATE,
            verbose           = False,
        )

    # ---------------- Linear baseline (no real tuning surface) -------------
    if name == "linear":
        from darts.models import LinearRegressionModel
        return LinearRegressionModel(
            **COMMON_KWARGS_TAB,
        )

    # ---------------- LSTM with default MSE loss ---------------------------
    if name == "lstm":
        from darts.models import BlockRNNModel
        return BlockRNNModel(
            model               = "LSTM",
            input_chunk_length  = INPUT_LAGS,
            output_chunk_length = OUTPUT_CHUNK_LEN,
            hidden_dim          = 32,
            n_rnn_layers        = 1,
            dropout             = 0.1,
            batch_size          = 64,
            n_epochs            = 30,
            random_state        = RANDOM_STATE,
            add_encoders = {
                "cyclic": {
                    "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                           },
                },
                pl_trainer_kwargs   = NN_TRAINER_KWARGS,
        )

    # ---------------- ARIMA (per region, no covariates here) ---------------
    # The input is already diffed, so d=0.
    if name == "arima":
        from darts.models import ARIMA
        return ARIMA(
            p = 7,
            d = 0,
            q = 1,
            random_state = RANDOM_STATE,
        )

    if name in NAIVE_MODELS:
        return None

    raise ValueError(f"Unknown regressor name: {name!r}")


# ---------------------------------------------------------------------------
# Param-driven builders used by Optuna trials.
# Each variant is a single family name (no objective sub-string), so the
# builder only has to inject the tuned hyperparameters on top of the shared
# COMMON_KWARGS_TAB.
# ---------------------------------------------------------------------------
def build_gbm_from_params(variant: str, params: dict):
    variant = variant.lower()
    p = dict(params)

    if variant == "lightgbm":
        from darts.models import LightGBMModel
        return LightGBMModel(
            **COMMON_KWARGS_TAB,
            objective      = "regression",
            random_state   = RANDOM_STATE,
            verbose        = -1,
            # device_type  = "gpu",
            # device_type    = "cpu",
            # num_threads    = available_threads,
            force_col_wise = True,
            **p,
        )

    if variant == "xgboost":
        from darts.models import XGBModel
        return XGBModel(
            **COMMON_KWARGS_TAB,
            objective    = "reg:squarederror",
            tree_method  = "hist",
            device       = "cuda",
            # device       = "cpu",
            # n_jobs       = available_threads,
            random_state = RANDOM_STATE,
            verbosity    = 0,
            **p,
        )

    if variant == "catboost":
        from darts.models import CatBoostModel
        return CatBoostModel(
            **COMMON_KWARGS_TAB,
            loss_function      = "RMSE",
            boost_from_average = False,
            bootstrap_type     = "Bernoulli",
            task_type          = "GPU",
            # task_type          = "CPU",
            # thread_count       = available_threads,
            random_seed        = RANDOM_STATE,
            verbose            = False,
            **p,
        )

    raise ValueError(f"Unknown GBM variant: {variant!r}")


def build_lstm_from_params(params: dict):
    """LSTM with default MSE loss — no log-link, no count likelihood."""
    from darts.models import BlockRNNModel
    return BlockRNNModel(**params)


# ## Feature selection
# 
# Same approach as the other notebooks: train one LightGBM (regression objective
# this time, since we are predicting differenced counts) on everything, rank
# features by gain importance, keep the top 100. Reuses the saved feature set
# from `_regression_GBDT.ipynb`/`_regression_LSTM.ipynb` if it already exists so
# all four notebooks compete on the same feature set.
# 

# In[32]:


# A single LightGBM-regression model, trained on everything, purely to rank features.
# We rank in level space (matching the GBDT/LSTM notebooks) so the saved feature
# set is shared across all four comparisons.
import pickle
from pathlib import Path
path = Path('./features/diffreg_saved_sets.pkl')
if path.exists():
    with open(path, "rb") as f:
        region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END, CV_START_VAL = pickle.load(f)
    print(f"Found saved sets in {path}")
else:
    model_feature_selection = build_regressor("lightgbm")
    model_feature_selection.fit(
        series            = train_target,
        past_covariates   = full_past_covs,
        future_covariates = full_fut_covs,
    )
    gain_imps = {"Feature": model_feature_selection.lagged_feature_names}

    underlying_model = model_feature_selection.model

    if hasattr(underlying_model, "estimators_"):
        estimators = underlying_model.estimators_
    else:
        estimators = [underlying_model]

    for h, est in enumerate(estimators, start=1):
        gain_imps[f"h{h}_gain"] = est.booster_.feature_importance(importance_type="gain")

    df_gain = pd.DataFrame(gain_imps)
    df_gain["agg_gain"] = df_gain.iloc[:, 1:].mean(axis=1)
    df_gain = df_gain.sort_values(by="agg_gain", ascending=False).reset_index(drop=True)
    df_gain.to_csv("prelimFeatureImportanceRegressor.csv", index=False)

    print(df_gain.head(20))
    top_100_features      = df_gain.head(100)["Feature"].to_list()
    top_100_features_dict = clean_feature_names(top_100_features)

    past_covs_list   = [subset_safe(ts, top_100_features_dict["pastcov_features_base"]) for ts in past_covs_list]
    future_covs_list = [subset_safe(ts, top_100_features_dict["futcov_features_base"])  for ts in future_covs_list]

    region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END, CV_START_VAL = \
        get_covs_and_encodings(target_series_list, past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC)

    sample   = past_covs_list[0]
    base_set = top_100_features_dict["pastcov_features_base"]
    present  = base_set & set(sample.components)
    missing  = base_set - set(sample.components)
    print(f"{len(present)}/{len(base_set)} base features present in past_covs")
    if missing:
        print("missing (first 10):", list(missing)[:10])
    sample   = future_covs_list[0]
    base_set = top_100_features_dict["futcov_features_base"]
    present  = base_set & set(sample.components)
    missing  = base_set - set(sample.components)
    print(f"{len(present)}/{len(base_set)} base features present in future_covs")
    if missing:
        print("missing (first 10):", list(missing)[:10])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((region_names, train_target, val_target, test_target,
                     full_past_covs, full_fut_covs, target_for_cv,
                     TRAIN_VAL_END, CV_START_VAL), f)
    print(f"Saved computed sets to {path}")


# In[ ]:


_, _, _, _, full_raw_past_covs_LSTM, _, _, _, _ = \
    get_covs_and_encodings(target_series_list, raw_past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC)


# ### Re-derive diffed targets to match the selected feature set
# 
# The feature-selection step may rebuild `target_series_list` / `target_for_cv`
# on a pruned covariate set. Re-derive the diffed target lists from those so
# the CV indices are aligned with the encoded targets.
# 

# In[ ]:


# Re-derive the diffed targets after feature selection. `target_series_list`
# itself is unchanged by the FS step (only past/future covs get pruned), so
# the diffed series is identical — but routing through `get_covs_and_encodings`
# again gives us the encoded statics that the boosters need.
diff_transformer = Diff(lags=1, dropna=True)
target_series_diff_list = diff_transformer.fit_transform(target_series_list)

_, train_target_diff, val_target_diff, test_target_diff, _, _, target_for_cv_diff, _, _ = \
    get_covs_and_encodings(target_series_diff_list, past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC)

print(f"diffed target_for_cv length: {len(target_for_cv_diff[0])}")
print(f"CV start (reusing level CV_START_VAL): {CV_START_VAL:.3f}")


# ##  Evaluation utilities
# 
# Same metric suite as the other notebooks — every fold-pred is compared to the
# **level-space** target via `evaluate_long`, with naive scales computed on the
# original training series. This keeps MASE / RMSSE comparable across notebooks.
# 

# In[ ]:


from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_poisson_deviance, mean_tweedie_deviance,
)

EPS               = 1e-9   # guards log / ratio when preds land at zero
NAIVE_SEASONALITY = 7      # weekly seasonality for MASE / RMSSE scales


# compute_naive_scales imported from src

# base_metrics imported from src

MAE_SCALES, RMSE_SCALES = compute_naive_scales(
    train_target, region_names, seasonality=NAIVE_SEASONALITY,
)


# In[ ]:


# plot_region_horizon_heatmap imported from src



# ## Naive baselines
# 
# Identical to the other notebooks — applied to the **level** target (not the
# diffed one) so they remain a meaningful floor in the original units. Any
# serious model has to beat both.
# 

# In[ ]:


# naive_last_historical_forecasts imported from src



# In[ ]:


# naive_collect_long imported from src



# ## Train + cross-validate every model
# 
# Same expanding-window CV as the other notebooks, with one twist: every
# non-naive learner is trained on the **diffed** target, predicts diffs for the
# next 7 days, and the runner converts predictions back to the **level** scale
# using the last actual value as anchor:
# 
# ```
# y_hat_level(t+h) = y(t-1) + cumsum( y_hat_diff(t+1 .. t+h) )
# ```
# 
# The level predictions are what flows into `evaluate_long`, so the leaderboard
# is in the same units as the GBDT / LSTM / Chronos2 notebooks.
# 
# ARIMA is local (one model per region, no covariates here, `d=0` because the
# input is already diffed). Naive baselines run directly on the level series.
# Linear, GBDTs and LSTM are global — one model fit across all regions.
# 

# In[ ]:


# _maybe_scale_covs imported from src

def _diff_to_level(diff_pred_ts, level_anchor_ts):
    """Un-diff a single fold-prediction: anchor = last actual level before the
    forecast window. Returns a TimeSeries on the same time index as the input.
    """
    first_pred_time = diff_pred_ts.time_index[0]
    # find the timestamp immediately before the first predicted timestamp
    anchor_idx = level_anchor_ts.time_index.get_loc(first_pred_time) - 1
    anchor_value = float(level_anchor_ts.values()[anchor_idx, 0])

    diff_vals  = diff_pred_ts.values().ravel()
    level_vals = anchor_value + np.cumsum(diff_vals)
    return TimeSeries.from_times_and_values(
        diff_pred_ts.time_index,
        level_vals.reshape(-1, 1),
    )


# In[ ]:


def run_expanding_cv(
    builder_fn,
    target_diff_list,
    target_level_list,
    start_frac,
    *,
    is_local=False,
    is_neural=False,
    horizon=OUTPUT_CHUNK_LEN,
    stride=CV_STRIDE,
    past_covs=None,
    future_covs=None,
    verbose=True,
):
    """Expanding-window CV on diffed targets. Predictions returned in LEVEL space.

    target_diff_list  -- what the model is trained / predicted on
    target_level_list -- used for the un-diff anchor (and is also what the
                         caller will compare against via evaluate_long)
    start_frac        -- fraction of the diffed series at which CV starts
    """
    ref_diff  = target_diff_list[0]
    n_total   = len(ref_diff)
    start_idx = int(start_frac * n_total)

    n_regions      = len(target_diff_list)
    all_fold_preds = [[] for _ in range(n_regions)]
    n_folds        = 0

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        split_time   = ref_diff.time_index[t0]
        train_series = [ts.drop_after(split_time) for ts in target_diff_list]

        if is_local:
            # ARIMA per region, fit on diffed series with d=0.
            diff_preds = []
            for ts in train_series:
                m = builder_fn()
                m.fit(ts)
                diff_preds.append(m.predict(n=horizon))
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
            diff_preds = model.predict(show_warnings=False, **pred_kwargs)

        # Un-diff each region's prediction back to level space.
        level_preds = [
            _diff_to_level(dp, target_level_list[r_idx])
            for r_idx, dp in enumerate(diff_preds)
        ]

        for r_idx, p in enumerate(level_preds):
            all_fold_preds[r_idx].append(p)
        n_folds += 1
        if verbose and (n_folds == 1 or n_folds % 4 == 0):
            print(f"   fold {n_folds} done  (trained up to {split_time.date()})")

    if verbose:
        print(f"   {n_folds} folds complete")
    return all_fold_preds


def run_final_test_diff(
    builder_fn,
    target_diff_list,
    target_level_list,
    start_frac,
    *,
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    horizon=OUTPUT_CHUNK_LEN,
    past_covs=None,
    future_covs=None,
    is_local=False,
    is_neural=False,
    verbose=True,
):
    """Like run_expanding_cv but predict_stride and retrain_stride are decoupled.

    Trains in diff space; returns predictions in level space (via _diff_to_level).
    """
    ref_diff  = target_diff_list[0]
    n_total   = len(ref_diff)
    start_idx = int(start_frac * n_total)
    n_regions = len(target_diff_list)

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
            retrain_time = ref_diff.time_index[t0]
            train_series = [ts.drop_after(retrain_time) for ts in target_diff_list]

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

        split_time   = ref_diff.time_index[t0]
        pred_series  = [ts.drop_after(split_time) for ts in target_diff_list]

        if is_local:
            diff_preds = []
            for ts in pred_series:
                m = _local_builder()
                m.fit(ts)
                diff_preds.append(m.predict(n=horizon))
        else:
            pred_kwargs = {"n": horizon, "series": pred_series, "show_warnings": False}
            if past_for_fit is not None and model.supports_past_covariates:
                pred_kwargs["past_covariates"] = past_for_fit
            if fut_for_fit is not None and model.supports_future_covariates:
                pred_kwargs["future_covariates"] = fut_for_fit
            diff_preds = model.predict(**pred_kwargs)

        level_preds = [
            _diff_to_level(dp, target_level_list[r_idx])
            for r_idx, dp in enumerate(diff_preds)
        ]
        for r_idx, p in enumerate(level_preds):
            all_fold_preds[r_idx].append(p)
        n_preds += 1

    if verbose:
        print(f"   {n_preds} daily predictions, {n_retrains} retrains complete")
    return all_fold_preds


def run_expanding_cv_iter(
    builder_fn,
    target_diff_list,
    target_level_list,
    start_frac,
    *,
    is_local=False,
    is_neural=False,
    horizon=OUTPUT_CHUNK_LEN,
    stride=CV_STRIDE,
    past_covs=None,
    future_covs=None,
    verbose=False,
):
    """Generator twin of run_expanding_cv — yields cumulative fold preds after
    each fold so Optuna's MedianPruner can fire."""
    ref_diff  = target_diff_list[0]
    n_total   = len(ref_diff)
    start_idx = int(start_frac * n_total)
    n_regions = len(target_diff_list)
    all_fold_preds = [[] for _ in range(n_regions)]

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        split_time   = ref_diff.time_index[t0]
        train_series = [ts.drop_after(split_time) for ts in target_diff_list]

        if is_local:
            diff_preds = []
            for ts in train_series:
                m = builder_fn()
                m.fit(ts)
                diff_preds.append(m.predict(n=horizon))
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
            diff_preds = model.predict(show_warnings=False, **pred_kwargs)

        level_preds = [
            _diff_to_level(dp, target_level_list[r_idx])
            for r_idx, dp in enumerate(diff_preds)
        ]
        for r_idx, p in enumerate(level_preds):
            all_fold_preds[r_idx].append(p)

        yield [list(rp) for rp in all_fold_preds]


# In[ ]:


from pytorch_lightning.callbacks import EarlyStopping, Callback as PLCallback
import optuna


def _suggest_lightgbm_params(trial):
    return {
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _suggest_xgboost_params(trial):
    return {
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=100),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _suggest_catboost_params(trial):
    return {
        "depth":         trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "iterations":    trial.suggest_int("iterations", 200, 1000, step=100),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 5.0, step=0.5),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
    }


SUGGESTERS_BY_FAMILY = {
    "lightgbm": _suggest_lightgbm_params,
    "xgboost":  _suggest_xgboost_params,
    "catboost": _suggest_catboost_params,
}


class LightningPruningCallback(PLCallback):
    """Optuna pruning callback that plays nicely with PL >= 2.x."""
    def __init__(self, trial: optuna.Trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def _maybe_prune(self, trainer):
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        self.trial.report(float(score.detach().cpu()), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"pruned at epoch {trainer.current_epoch} ({self.monitor}={float(score):.4f})"
            )

    def on_train_epoch_end(self, trainer, pl_module):
        self._maybe_prune(trainer)


def _nn_trainer_kwargs(trial=None):
    callbacks = [EarlyStopping(monitor="train_loss", patience=5,
                               min_delta=1e-4, mode="min")]
    if trial is not None:
        callbacks.append(LightningPruningCallback(trial, monitor="train_loss"))
    return dict(
        accelerator          = "auto",
        enable_progress_bar  = False,
        enable_model_summary = False,
        log_every_n_steps    = 10,
        gradient_clip_val    = 1.0,
        callbacks            = callbacks,
    )



def _suggest_lstm_params(trial, input_chunk_length: int, model_type: str = "LSTM"):
    """RNN (LSTM or GRU) with default MSE loss — model type fixed per variant."""
    fc_choice = trial.suggest_categorical("hidden_fc_sizes", ["none", "32", "64", "64_32"])
    fc_map    = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}
    return dict(
        model               = model_type,
        input_chunk_length  = input_chunk_length,
        output_chunk_length = OUTPUT_CHUNK_LEN,
        hidden_dim          = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
        n_rnn_layers        = trial.suggest_int("n_rnn_layers", 1, 3),
        hidden_fc_sizes     = fc_map[fc_choice],
        dropout             = trial.suggest_float("dropout", 0.0, 0.4),
        batch_size          = trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        n_epochs            = 100,
        optimizer_kwargs    = {
            "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        },
        random_state        = RANDOM_STATE,
        add_encoders = {
                "cyclic": {
                    "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                           },
                },
        pl_trainer_kwargs   = _nn_trainer_kwargs(trial),
    )


def _score_fold_preds(fold_preds, target_level_list, region_names_, metric="RMSSE_mean"):
    long_df = collect_predictions_long(target_level_list, fold_preds, region_names_)
    res     = evaluate_long(long_df, MAE_SCALES, RMSE_SCALES)
    return float(res["global"][metric])


def make_gbm_objective(variant: str):
    """Optuna objective with per-fold pruning."""
    suggester = SUGGESTERS_BY_FAMILY[variant.lower()]

    def _objective(trial):
        params  = suggester(trial)
        builder = lambda: build_gbm_from_params(variant, params)

        last_score = None
        for step, cumulative_fold_preds in enumerate(run_expanding_cv_iter(
            builder,
            target_diff_list  = target_for_cv_diff,
            target_level_list = target_for_cv,
            start_frac        = CV_START_VAL,
            past_covs         = full_past_covs,
            future_covs       = full_fut_covs,
            verbose           = False,
        )):
            last_score = _score_fold_preds(
                cumulative_fold_preds, target_for_cv, region_names,
                metric="RMSSE_mean",
            )
            trial.report(last_score, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(last_score)

    return _objective


def make_nn_objective(variant: str):
    """Optuna objective for RNN variants (LSTM and GRU) — default MSE loss."""
    model_type, icl = parse_lstm_variant(variant)
    def _objective(trial):
        params  = _suggest_lstm_params(trial, input_chunk_length=icl, model_type=model_type)
        builder = lambda: build_lstm_from_params(params)

        fold_preds = run_expanding_cv(
            builder,
            target_diff_list  = target_for_cv_diff,
            target_level_list = target_for_cv,
            start_frac        = CV_START_VAL,
            past_covs         = full_raw_past_covs_LSTM,
            future_covs       = full_fut_covs,
            is_neural         = True,
            verbose           = False,
        )
        return _score_fold_preds(
            fold_preds, target_for_cv, region_names, metric="RMSSE_mean"
        )

    return _objective


# ### Tune every variant
# 
# One Optuna study per family. Each study minimises `RMSSE_mean` on the val
# folds. GBDTs and LSTM run their own loops; ARIMA and the linear baseline have
# no real tuning surface so they are evaluated with their default configs in the
# next section.
# 

# In[ ]:


from pathlib import Path
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNE_CKPT = Path(PROJECT_DIR) / "checkpoints_tune_diff"
TUNE_CKPT.mkdir(exist_ok=True)

best_params_by_variant = {}
studies                = {}
NN_VARIANTS = ["lstm_w7", "lstm_w14", "lstm_w28", "gru_w7", "gru_w14", "gru_w28"]

# Tune every GBM family + the LSTM. Linear and ARIMA skip tuning (see next cell).
all_variants = list(GBM_VARIANTS) + list(NN_VARIANTS)

for variant in all_variants:
    ckpt = TUNE_CKPT / f"{variant}_best.pkl"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            best_params_by_variant[variant], studies[variant] = pickle.load(f)
        print(f"[{variant}] loaded cached best_value={studies[variant].best_value:.4f}")
        continue

    is_nn    = variant in NN_VARIANTS
    n_trials = OPTUNA_N_TRIALS
    print(f"\n=== Tuning {variant} ({n_trials} trials) ===")

    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner    = optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    if is_nn:
        study.optimize(
            make_nn_objective(variant),
            n_trials = n_trials,
            timeout  = OPTUNA_TIMEOUT_S,
        )
    else:
        study.optimize(
            make_gbm_objective(variant),
            n_trials          = n_trials,
            timeout           = OPTUNA_TIMEOUT_S,
            show_progress_bar = True,
            n_jobs            = 1,
        )

    best_params_by_variant[variant] = study.best_params
    studies[variant]                = study

    with open(ckpt, "wb") as f:
        pickle.dump((study.best_params, study), f)

    print(f"[{variant}] best score (val CV) = {study.best_value:.4f}")
    print(f"[{variant}] best params: {study.best_params}")


# ### Re-run val-CV with best params (and run the un-tuned baselines)
# 
# Stores `long_df` (region * fold * horizon) per model in `long_by_model` so the
# leaderboard can score everything in one place. Naive baselines and the ARIMA /
# linear-regression baselines run here too (they don't go through Optuna).
# 

# In[ ]:


import pickle
from pathlib import Path

CKPT = Path(PROJECT_DIR) / "checkpoints_diff"
CKPT.mkdir(exist_ok=True)

long_by_model       = {}
fold_preds_by_model = {}


def _build_lstm_from_best(variant: str, best: dict):
    """Reconstruct the RNN build kwargs from study.best_params (flat dict)."""
    fc_map = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}
    b      = dict(best)
    model_type, icl = parse_lstm_variant(variant)
    params = dict(
        model               = model_type,
        input_chunk_length  = icl,
        output_chunk_length = OUTPUT_CHUNK_LEN,
        hidden_dim          = b["hidden_dim"],
        n_rnn_layers        = b["n_rnn_layers"],
        hidden_fc_sizes     = fc_map[b["hidden_fc_sizes"]],
        dropout             = b["dropout"],
        batch_size          = b["batch_size"],
        n_epochs            = 100,
        optimizer_kwargs    = {"lr": b["lr"], "weight_decay": b["weight_decay"]},
        random_state        = RANDOM_STATE,
        add_encoders = {
                "cyclic": {
                    "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                           },
                },
        pl_trainer_kwargs   = _nn_trainer_kwargs(),
    )
    return lambda: build_lstm_from_params(params)


# --- 1) Tuned variants -----------------------------------------------------
for variant, best_params in best_params_by_variant.items():
    name = f"{variant}_tuned"
    if name in long_by_model:
        print(f"skip {name} — already cached in long_by_model")
        continue
    print(f"\n=== Re-running val CV with best {variant} params ===")

    if variant in NN_VARIANTS:
        builder = _build_lstm_from_best(variant, best_params)
        is_neural       = True
        chosen_past     = full_raw_past_covs_LSTM
    else:
        builder         = lambda p=best_params, v=variant: build_gbm_from_params(v, p)
        is_neural       = False
        chosen_past     = full_past_covs

    fold_preds = run_expanding_cv(
        builder,
        target_diff_list  = target_for_cv_diff,
        target_level_list = target_for_cv,
        start_frac        = CV_START_VAL,
        past_covs         = chosen_past,
        future_covs       = full_fut_covs,
        is_neural         = is_neural,
    )
    long_df = collect_predictions_long(target_for_cv, fold_preds, region_names)
    long_by_model[name]       = long_df
    fold_preds_by_model[name] = fold_preds
    with open(CKPT / f"{name}.pkl", "wb") as f:
        pickle.dump((long_df, fold_preds), f)


# --- 2) ARIMA (per region, no covariates, fed the diffed series) -----------
if "arima" not in long_by_model:
    print("\n=== ARIMA (local, per region) ===")
    arima_fold_preds = run_expanding_cv(
        lambda: build_regressor("arima"),
        target_diff_list  = target_for_cv_diff,
        target_level_list = target_for_cv,
        start_frac        = CV_START_VAL,
        is_local          = True,
    )
    long_by_model["arima"]       = collect_predictions_long(target_for_cv, arima_fold_preds, region_names)
    fold_preds_by_model["arima"] = arima_fold_preds


# --- 3) Linear baseline (no tuning, global) --------------------------------
if "linear" not in long_by_model:
    print("\n=== Linear regression (global, default config) ===")
    lin_fold_preds = run_expanding_cv(
        lambda: build_regressor("linear"),
        target_diff_list  = target_for_cv_diff,
        target_level_list = target_for_cv,
        start_frac        = CV_START_VAL,
        past_covs         = full_past_covs,
        future_covs       = full_fut_covs,
    )
    long_by_model["linear"]       = collect_predictions_long(target_for_cv, lin_fold_preds, region_names)
    fold_preds_by_model["linear"] = lin_fold_preds


# --- 4) Naive baselines (level series, no diff) ----------------------------
for naive in ["naive_last", "naive_weekly"]:
    if naive in long_by_model:
        continue
    print(f"\n=== {naive} ===")
    long_df, fold_preds = naive_collect_long(target_for_cv, region_names, naive, CV_START_VAL, OUTPUT_CHUNK_LEN, CV_STRIDE)
    long_by_model[naive]       = long_df
    fold_preds_by_model[naive] = fold_preds


# ## Leaderboard — which (basic) regressor wins on the diffed target?

# In[ ]:


results_by_model = {}
for name, long_df in long_by_model.items():
    res = evaluate_long(long_df, MAE_SCALES, RMSE_SCALES)
    results_by_model[name] = res


# In[ ]:


# Sorted by RMSSE_mean (scale-free, comparable across regions and notebooks).
ref_name = "naive_weekly"
ref_g    = results_by_model.get(ref_name, {}).get("global", None)

# _skill imported from src

leaderboard_rows = []
for name, res in results_by_model.items():
    g = res["global"]
    leaderboard_rows.append({
        "model":       name,
        "MAE":         g["MAE"],
        "RMSE":        g["RMSE"],
        "MedAE":       g["MedAE"],
        "ME":          g["ME"],
        "PoissonDev":  g["PoissonDev"],
        "TweedieDev":  g["TweedieDev"],
        "ZeroAcc":     g["ZeroAcc"],
        "MASE_mean":   g["MASE_mean"],
        "MASE_median": g["MASE_median"],
        "RMSSE_mean":  g["RMSSE_mean"],
        "SkillMAE":    _skill(g["MAE"],        ref_g["MAE"])        if ref_g else float("nan"),
        "SkillRMSE":   _skill(g["RMSE"],       ref_g["RMSE"])       if ref_g else float("nan"),
        "SkillMASE":   _skill(g["MASE_mean"],  ref_g["MASE_mean"])  if ref_g else float("nan"),
        "SkillRMSSE":  _skill(g["RMSSE_mean"], ref_g["RMSSE_mean"]) if ref_g else float("nan"),
    })
leaderboard = (
    pd.DataFrame(leaderboard_rows)
      .sort_values("RMSSE_mean", ascending=True)
      .reset_index(drop=True)
)
leaderboard


# In[ ]:


winner = leaderboard.iloc[0]["model"]
print(f"Winning model on RMSSE_mean: {winner}")
results_by_model[winner]["per_region"]


# In[ ]:


winner = leaderboard.iloc[0]["model"]
print(f"Winning model on RMSSE_mean: {winner}")
results_by_model[winner]["per_horizon"]


# ## Test-set evaluation
# 
# Final hold-out CV: train on `train + val` (both in diff space), predict on the
# test segment, un-diff to levels, evaluate against the level test target. Run
# once per tuned model + ARIMA + linear + naives.
# 

# In[ ]:


target_full      = [tr.append(vl).append(te)
                    for tr, vl, te in zip(train_target, val_target, test_target)]
target_full_diff = [tr.append(vl).append(te)
                    for tr, vl, te in zip(train_target_diff, val_target_diff, test_target_diff)]
TEST_START_FRAC  = TRAIN_VAL_END

test_long_by_model       = {}
test_fold_preds_by_model = {}

# 1) Tuned variants
for variant, best_params in best_params_by_variant.items():
    name = f"{variant}_tuned"
    print(f"\n=== Test-set CV: {name} ===")
    if variant in NN_VARIANTS:
        builder     = _build_lstm_from_best(variant, best_params)
        is_neural   = True
        chosen_past = full_raw_past_covs_LSTM
    else:
        builder     = lambda p=best_params, v=variant: build_gbm_from_params(v, p)
        is_neural   = False
        chosen_past = full_past_covs

    fold_preds = run_final_test_diff(
        builder,
        target_diff_list  = target_full_diff,
        target_level_list = target_full,
        start_frac        = TEST_START_FRAC,
        predict_stride    = 1,
        retrain_stride    = OUTPUT_CHUNK_LEN,
        past_covs         = chosen_past,
        future_covs       = full_fut_covs,
        is_neural         = is_neural,
    )
    test_long_by_model[name]       = collect_predictions_long(target_full, fold_preds, region_names)
    test_fold_preds_by_model[name] = fold_preds

# 2) ARIMA
print("\n=== Test-set CV: arima ===")
arima_fold_preds = run_final_test_diff(
    lambda: build_regressor("arima"),
    target_diff_list  = target_full_diff,
    target_level_list = target_full,
    start_frac        = TEST_START_FRAC,
    predict_stride    = 1,
    retrain_stride    = OUTPUT_CHUNK_LEN,
    is_local          = True,
)
test_long_by_model["arima"]       = collect_predictions_long(target_full, arima_fold_preds, region_names)
test_fold_preds_by_model["arima"] = arima_fold_preds

# 3) Linear baseline
print("\n=== Test-set CV: linear ===")
lin_fold_preds = run_final_test_diff(
    lambda: build_regressor("linear"),
    target_diff_list  = target_full_diff,
    target_level_list = target_full,
    start_frac        = TEST_START_FRAC,
    predict_stride    = 1,
    retrain_stride    = OUTPUT_CHUNK_LEN,
    past_covs         = full_past_covs,
    future_covs       = full_fut_covs,
)
test_long_by_model["linear"]       = collect_predictions_long(target_full, lin_fold_preds, region_names)
test_fold_preds_by_model["linear"] = lin_fold_preds

# 4) Naives on the level series
for naive in ["naive_last", "naive_weekly"]:
    print(f"\n=== Test-set CV: {naive} ===")
    long_df, fold_preds = naive_collect_long(target_full, region_names, naive, TEST_START_FRAC, OUTPUT_CHUNK_LEN, 1)
    test_long_by_model[naive]       = long_df
    test_fold_preds_by_model[naive] = fold_preds

test_results_by_model = {n: evaluate_long(df, MAE_SCALES, RMSE_SCALES) for n, df in test_long_by_model.items()}

test_rows = []
for name, res in test_results_by_model.items():
    g = res["global"]
    test_rows.append({
        "model":      name,
        "MAE":        g["MAE"],
        "RMSE":       g["RMSE"],
        "MedAE":      g["MedAE"],
        "ME":         g["ME"],
        "PoissonDev": g["PoissonDev"],
        "TweedieDev": g["TweedieDev"],
        "ZeroAcc":    g["ZeroAcc"],
        "MASE_mean":  g["MASE_mean"],
        "RMSSE_mean": g["RMSSE_mean"],
    })

test_leaderboard = (
    pd.DataFrame(test_rows)
      .sort_values("RMSSE_mean", ascending=True)
      .reset_index(drop=True)
)
print(test_leaderboard)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




