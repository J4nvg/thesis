#!/usr/bin/env python
# coding: utf-8

# # Direct regression — zero-inflated drone strike count forecasting
# 
# Fair comparison across:
#  Gradient boosters (LightGBM / XGBoost / CatBoost) — Poisson and Tweedie tuned **separately**
#  Persistence baselines (yesterday, last week same day)
#  ARIMA (per-region classical baseline)
#  Darts LSTM (BlockRNN)
# 
# Every model is fit & evaluated on the **same expanding-window folds** over
# the validation segment. Test set is untouched until the final run.
# 

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')

# PROJECT_DIR = '/content/drive/MyDrive/thesis/CODEBASE'
PROJECT_DIR = './'
# %cd $PROJECT_DIR


# In[2]:


# !nvidia-smi


# In[ ]:





# In[ ]:





# In[4]:


# !pip install "darts[all]" statsmodels optuna comet_ml


# In[5]:


import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


# Imports and config

# In[6]:


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
    WindowTransformer, StaticCovariatesTransformer, Scaler,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)

available_threads = get_available_threads()
print(f'CPU count: {available_threads}')

# 

# In[ ]:


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
NAIVE_MODELS  = {"naive_last", "naive_weekly"}
NEURAL_MODELS  = {"lstm_poisson", "lstm_tweedie"}            
LOCAL_MODELS  = {"arima"}

# Each GBM family is split by objective so Optuna tunes Poisson and Tweedie
# in their own studies (see _suggest_*_params + the tuning loop below).
GBM_FAMILIES  = ["lightgbm", "xgboost", "catboost"]
GBM_OBJECTIVES = ["poisson", "tweedie"]
GBM_VARIANTS  = [f"{fam}_{obj}" for fam in GBM_FAMILIES for obj in GBM_OBJECTIVES]
# -> ["lightgbm_poisson", "lightgbm_tweedie",
#     "xgboost_poisson",  "xgboost_tweedie",
#     "catboost_poisson", "catboost_tweedie"]

# Tuned GBMs are added to REGRESSORS_TO_RUN dynamically after Optuna finishes.
REGRESSORS_TO_RUN = [
    "naive_last", "naive_weekly",
    "arima",
]

# Optuna config
OPTUNA_N_TRIALS    = 50   # per study; each Poisson/Tweedie variant gets its own
OPTUNA_TIMEOUT_S   = None  # 1 h cap per variant study

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


# In[ ]:


print(for_global_reset.head())


# In[ ]:


print(for_global_reset.isna().any()[lambda x: x])
# printfor_global_reset[for_global_reset["Activity_Level"].isna() == True][['Activity_Level','event_date','region']])


# In[ ]:


# Future vs past covariate split
holiday_cols, future_covariates, exclude_cols, past_covariates = split_future_and_past_cov(for_global_reset,global_weather_columns,TARGET)


# ## Build Darts TimeSeries and apply windowed transforms
# Getting lag
#  variabels

# In[ ]:


target_series_list, past_covs_list,future_covs_list = build_ts_and_apply_window_transformer(for_global_reset,TARGET,past_covariates,future_covariates,ed_alpha=halflife_to_alpha(7))


# In[ ]:


raw_past_covs_list = TimeSeries.from_group_dataframe(
    for_global_reset,
    group_cols="region", time_col="event_date",
    value_cols=past_covariates,
)


# ## Encode static covariates and split 70/10/20
# 

# In[ ]:


region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END,CV_START_VAL =\
      get_covs_and_encodings(target_series_list,past_covs_list,future_covs_list,TRAIN_FRAC,VAL_FRAC)


# ## Hyper-parameter cheat sheet
# 
# A compact reference for what to sweep first, second, third.
# 
# | Model | First knob (★★★) | Second knob (★★) | Third knob (★) | Notes |
# |---|---|---|---|---|
# | `lightgbm_*` / `xgboost_*` / `catboost_*` | `n_estimators` + `learning_rate` | `max_depth` / `num_leaves` / `depth` | `min_child_samples` / `min_child_weight` / `l2_leaf_reg` | Poisson and Tweedie are **tuned in separate Optuna studies**. For Tweedie, also sweep `tweedie_variance_power ∈ (1.1, 1.9)`. Poisson vs Tweedie matters most when there is little overdispersion. |
# | `poisson_glm` | `alpha` (L2) | `max_iter` | — | Assumes mean == variance. Bad when data is overdispersed. |
# | `tweedie_glm` | `power ∈ (1, 2)` | `alpha` (L2) | `max_iter` | `power = 1.5` is a safe starting point for zero-inflated continuous-ish counts. |
# | `negbin_glm` | `alpha` (dispersion) | `maxiter` | — | Preferred over Poisson when `Var(Y) > E[Y]`. |
# | `lstm` (BlockRNN) | `hidden_dim`, `n_rnn_layers` | `dropout`, `n_epochs` | `batch_size`, `learning_rate` | No count likelihood in BlockRNN — consider scaling the target. |
# | `arima` | `p`, `q` | `d` | — | Local per region; consider `AutoARIMA` once for each region to find starting orders. |
# | `naive_*` | — | — | — | No parameters. These are your floor. |
# 
# **Sanity rule**: whichever model you pick, its global `MASE_mean` must be
# below 1 (i.e. `SkillRMSE` > 0 vs `naive_weekly`). If it isn't, the
# covariates aren't carrying signal — inspect per-horizon metrics (it's
# common for h=1 to be easy and h=7 to collapse).
# 

# ## Regressors
# 
# Every tabular model shares the same `lags`, `output_chunk_length`,
# `multi_models` and `add_encoders` setup so the comparison is fair —
# **only the learner and its loss/likelihood change**. The LSTM uses the
# same input / output chunk lengths; ARIMA is local by construction.
# 
# **What's worth tuning** (marked `★` = high impact, `·` = secondary):
# - **Gradient boosters**: `★ n_estimators + learning_rate` (trade-off), `★ max_depth / num_leaves`,
#   `★ min_child_samples` (zero-inflation: too-small splits overfit zeros), `· subsample`,
#   `· colsample_bytree`, `· reg_alpha / reg_lambda`. For Tweedie, also `★ tweedie_variance_power` ∈ (1, 2).
#   Poisson and Tweedie are tuned in **separate studies** so each objective gets a
#   fair search budget.
# - **GLMs**: `★ alpha` (L2), `★ power` for Tweedie, `★ dispersion (alpha)` for NegBin,
#   `· max_iter`. Standardise inputs — these are linear in the features.
# - **LSTM (BlockRNN)**: `★ hidden_dim`, `★ n_rnn_layers`, `★ dropout`, `★ n_epochs`,
#   `· batch_size`, `· learning_rate`. Consider target scaling since there's no count likelihood.
# - **ARIMA**: `★ p, q` (grid `p ∈ {1,3,7,14}`, `q ∈ {0,1,2}`), `★ d ∈ {0, 1}`.
#   Use `AutoARIMA` if you want automatic order selection (slower).
# - **Naive**: no parameters — they are your floor. Any model must beat both.
# 

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/metrics/point.py
class PoissonNLLLogLink(nn.Module):
    def forward(self, y_pred, y_true):
        return F.poisson_nll_loss(
            y_pred, y_true, log_input=True, full=False, reduction="mean" # Darts expects a scalar loss back so reduction = mean
        )


class TweedieNLLLogLink(nn.Module):
    def __init__(self, power: float = 1.5):
        super().__init__()
        if not (1.0 < power < 2.0):
            raise ValueError("power must lie strictly in (1, 2)")
        self.power = power

    def forward(self, y_pred, y_true):
        p  = self.power
        mu = torch.exp(y_pred)                              
        a  = -y_true * torch.pow(mu, 1.0 - p) / (1.0 - p)
        b  =           torch.pow(mu, 2.0 - p) / (2.0 - p)
        return torch.mean(a + b)

def build_lstm_count(objective_kind: str, params: dict, *, tweedie_power: float = 1.5):
    """LSTM in deterministic point-forecast mode, trained with Poisson or
    Tweedie NLL on a log-link. Mirrors GBDT objectives for fair comparison.

    The network parameterizes log(μ); predictions come out in log-space and
    are inverted via exp() inside run_expanding_cv (gated on _count_log_link).
    """
    if objective_kind == "poisson":
        loss_fn = PoissonNLLLogLink()
    elif objective_kind == "tweedie":
        loss_fn = TweedieNLLLogLink(power=tweedie_power)
    else:
        raise ValueError(f"unknown objective_kind {objective_kind!r}")

    p = dict(params)
    p.pop("likelihood", None)        # force deterministic — no sampling
    p["loss_fn"]    = loss_fn
    p["likelihood"] = None

    from darts.models import BlockRNNModel
    m = BlockRNNModel(**p)
    m._count_log_link = True         # signal to run_expanding_cv to invert the link
    return m


# In[ ]:


from pytorch_lightning.callbacks import EarlyStopping
# --- Shared forecasting skeleton: every tabular model gets the same inputs ---
COMMON_KWARGS_TAB = get_common_kwargs()

ES_NN  = EarlyStopping(monitor="val_loss",   patience=10, min_delta=1e-4, mode="min")
# ES = EarlyStopping(
#     monitor="train_loss",   # use "val_loss" if you pass val_series to .fit()
#     patience=5,
#     min_delta=1e-4,
#     mode="min",
# )

NN_TRAINER_KWARGS = dict(
    accelerator          = "gpu",
    devices = 1,
    precision = "32-true",
    enable_progress_bar  = True,
    enable_model_summary = True,
    log_every_n_steps    = 10,
    callbacks            = [ES_NN],
    gradient_clip_val    = 1.0,
)


# ---------------------- build_regressor ----------------------
def build_regressor(name: str):
    """Return a Darts forecasting model ready to ``.fit()``.

    Every branch uses the same ``INPUT_LAGS`` / ``OUTPUT_CHUNK_LEN`` so that
    a 7-day forecast from yesterday is directly comparable across models.
    These default-config branches are used for the feature-selection pass and
    as a fallback; the Optuna-tuned versions go through ``build_gbm_from_params``.
    """
    name = name.lower()

    # ======================================================================
    # 1) Gradient boosters — Darts native wrappers (default configs)
    # ======================================================================
    if name.startswith("lightgbm_poisson"):
        from darts.models import LightGBMModel
        return LightGBMModel(
            **COMMON_KWARGS_TAB,
            objective         = "poisson",
            # --- tuning surface ---
            num_leaves        = 31,      # ★ 15–127 (capacity)
            max_depth         = 5,       # ★ -1 or 3–10
            min_child_samples = 30,      # ★ 10–200 — zero-heavy targets need higher
            subsample         = 0.8,     # · 0.6–1.0
            colsample_bytree  = 0.8,     # · 0.6–1.0
            learning_rate     = 0.05,    # ★ pair with n_estimators
            n_estimators      = 500,     # ★ use early stopping on a held-out val
            reg_alpha         = 0.0,     # · L1
            reg_lambda        = 0.0,     # · L2
            random_state      = RANDOM_STATE,
            verbose           = -1,
            device_type       = "cpu",
            num_threads = available_threads,
            force_col_wise = True,
        )
    if name.startswith("lightgbm_tweedie"):
        from darts.models import LightGBMModel
        return LightGBMModel(
            **COMMON_KWARGS_TAB,
            objective              = "tweedie",
            # --- tuning surface ---
            num_leaves             = 31,      # ★ 15–127 (capacity)
            max_depth              = 5,       # ★ -1 or 3–10
            min_child_samples      = 30,      # ★ 10–200 — zero-heavy targets need higher
            subsample              = 0.8,     # · 0.6–1.0
            colsample_bytree       = 0.8,     # · 0.6–1.0
            learning_rate          = 0.05,    # ★ pair with n_estimators
            n_estimators           = 500,     # ★ use early stopping on a held-out val
            reg_alpha              = 0.0,     # · L1
            reg_lambda             = 0.0,     # · L2
            tweedie_variance_power = 1.5,
            random_state           = RANDOM_STATE,
            verbose                = -1,
            device_type            = "cpu",
            num_threads = available_threads,
            force_col_wise = True,
        )

    if name.startswith("xgboost_poisson"):
        from darts.models import XGBModel
        return XGBModel(
            **COMMON_KWARGS_TAB,
            objective         = "count:poisson",
            # --- tuning surface ---
            max_depth         = 5,       # ★ 3–10
            min_child_weight  = 3,       # ★ 1–10
            subsample         = 0.8,     # · 0.6–1.0
            colsample_bytree  = 0.8,     # · 0.6–1.0
            learning_rate     = 0.05,    # ★
            n_estimators      = 500,     # ★
            reg_alpha         = 0.0,     # · L1
            reg_lambda        = 1.0,     # · L2
            tree_method       = "hist",
            device            = "cpu",
            n_jobs = available_threads,
            random_state      = RANDOM_STATE,
            verbosity         = 0,
        )
    if name.startswith("xgboost_tweedie"):
        from darts.models import XGBModel
        return XGBModel(
            **COMMON_KWARGS_TAB,
            objective              = "reg:tweedie",
            # --- tuning surface ---
            max_depth              = 5,       # ★ 3–10
            min_child_weight       = 3,       # ★ 1–10
            subsample              = 0.8,     # · 0.6–1.0
            colsample_bytree       = 0.8,     # · 0.6–1.0
            learning_rate          = 0.05,    # ★
            n_estimators           = 500,     # ★
            reg_alpha              = 0.0,     # · L1
            reg_lambda             = 1.0,     # · L2
            tweedie_variance_power = 1.5,
            tree_method            = "hist",
            device                 = "cpu",
            n_jobs = available_threads,
            random_state           = RANDOM_STATE,
            verbosity              = 0,
        )

    if name.startswith("catboost_poisson"):
        from darts.models import CatBoostModel
        return CatBoostModel(
            **COMMON_KWARGS_TAB,
            loss_function     = "Poisson",
            boost_from_average= False,
            # --- tuning surface ---
            depth             = 5,       # ★ 4–10
            learning_rate     = 0.05,    # ★
            iterations        = 500,     # ★ equivalent of n_estimators
            l2_leaf_reg       = 3,       # ★ 1–10
            subsample         = 0.8,     # · (bootstrap_type=Bernoulli)
            bootstrap_type    = "Bernoulli",
            # task_type         = "GPU",
            task_type         = "CPU",
            thread_count = available_threads,
            random_seed       = RANDOM_STATE,
            verbose           = False,
        )
    if name.startswith("catboost_tweedie"):
        from darts.models import CatBoostModel
        return CatBoostModel(
            **COMMON_KWARGS_TAB,
            loss_function     = "Tweedie:variance_power=1.5",
            boost_from_average= False,
            # --- tuning surface ---
            depth             = 5,       # ★ 4–10
            learning_rate     = 0.05,    # ★
            iterations        = 500,     # ★ equivalent of n_estimators
            l2_leaf_reg       = 3,       # ★ 1–10
            subsample         = 0.8,     # · (bootstrap_type=Bernoulli)
            bootstrap_type    = "Bernoulli",
            # task_type         = "GPU",
            task_type         = "CPU",
            thread_count = available_threads,
            random_seed       = RANDOM_STATE,
            verbose           = False,
        )

    if name == "lstm":
        from darts.models import BlockRNNModel
        return BlockRNNModel(
            model                = "LSTM",
            input_chunk_length   = INPUT_LAGS,
            output_chunk_length  = OUTPUT_CHUNK_LEN,
            hidden_dim           = 32,       # ★ 16–128
            n_rnn_layers         = 1,        # ★ 1–3
            dropout              = 0.1,      # ★ 0.0–0.3
            batch_size           = 64,       # · 32–256
            n_epochs             = 30,       # ★ add EarlyStopping
            random_state         = RANDOM_STATE,
            add_encoders = {"cyclic": {"future": ["month", "week", "dayofweek"]}},
            pl_trainer_kwargs    = NN_TRAINER_KWARGS,
        )

    # LOCAL model (one fit per region, no covariates here)
    if name == "arima":
        from darts.models import ARIMA
        return ARIMA(
            p = 7,              # ★ AR order: try {1, 3, 7, 14}
            d = 0,              # ★ diff order: 0 (stationary-ish) or 1
            q = 1,              # ★ MA order: try {0, 1, 2}
            random_state = RANDOM_STATE,
        )

    if name in NAIVE_MODELS:
        return None

    raise ValueError(f"Unknown regressor name: {name!r}")


# ---------------------------------------------------------------------------
# Param-driven GBM builder used by Optuna trials.
# ``variant`` is a string like "lightgbm_poisson" / "xgboost_tweedie" — the
# objective is fixed by the variant, NOT a tunable parameter.
# Same COMMON_KWARGS_TAB so models stay comparable across the lineup.
# ---------------------------------------------------------------------------
def build_gbm_from_params(variant: str, params: dict):
    variant = variant.lower()
    family, objective_kind = variant.split("_", 1)
    if objective_kind not in {"poisson", "tweedie"}:
        raise ValueError(f"Unknown objective kind in variant {variant!r}")
    p = dict(params)  # copy so we can pop

    if family == "lightgbm":
        from darts.models import LightGBMModel
        if objective_kind == "tweedie":
            extra = {"objective": "tweedie",
                     "tweedie_variance_power": p.pop("tweedie_variance_power")}
        else:
            extra = {"objective": "poisson"}
        return LightGBMModel(
            **COMMON_KWARGS_TAB,
            random_state = RANDOM_STATE,
            verbose      = -1,
            # device_type  = "gpu",
            device_type            = "cpu",
            num_threads = available_threads,
            force_col_wise = True,
            **p, **extra,
        )

    if family == "xgboost":
        from darts.models import XGBModel
        if objective_kind == "tweedie":
            extra = {"objective": "reg:tweedie",
                     "tweedie_variance_power": p.pop("tweedie_variance_power")}
        else:
            extra = {"objective": "count:poisson"}
        return XGBModel(
            **COMMON_KWARGS_TAB,
            tree_method  = "hist",
            # device       = "cuda",
            device                 = "cpu",
            n_jobs = available_threads,
            random_state = RANDOM_STATE,
            verbosity    = 0,
            **p, **extra,
        )

    if family == "catboost":
        from darts.models import CatBoostModel
        if objective_kind == "tweedie":
            vp = p.pop("tweedie_variance_power")
            loss = f"Tweedie:variance_power={vp}"
        else:
            loss = "Poisson"
        return CatBoostModel(
            **COMMON_KWARGS_TAB,
            loss_function      = loss,
            boost_from_average = False,
            bootstrap_type     = "Bernoulli",
            # task_type          = "GPU",
            task_type         = "CPU",
            thread_count = available_threads,
            random_seed        = RANDOM_STATE,
            verbose            = False,
            **p,
        )

    raise ValueError(f"Unknown GBM family: {family!r}")

def build_nn_from_params(family: str, params: dict):
    """LSTM only """
    p = dict(params)
    if family == "lstm":
        from darts.models import BlockRNNModel
        return BlockRNNModel(**p)
    raise ValueError(f"Unknown NN family: {family!r}")


# # Build simple LGBM For feature selection

# In[ ]:


# A single LightGBM-Tweedie model, trained on everything, purely to rank features
import pickle
from pathlib import Path
path = Path('./features/countreg_saved_sets.pkl')
if path.exists():
    with open(path, "rb") as f:
        region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END, CV_START_VAL = pickle.load(f)
    print(f"Found saved sets in {path}")    
else:
    model_feature_selection = build_regressor("lightgbm_tweedie")
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
        gain_imps[f"h{h}_gain"] = est.booster_.feature_importance(importance_type='gain')

    df_gain = pd.DataFrame(gain_imps)
    df_gain["agg_gain"] = df_gain.iloc[:, 1:].mean(axis=1)

    df_gain = df_gain.sort_values(by="agg_gain", ascending=False).reset_index(drop=True)

    df_gain.to_csv('prelimFeatureImportanceRegressor.csv', index=False)
    threshold = .99
    print(df_gain.head(20))
    top_100_features = df_gain.head(100)['Feature'].to_list()
    top_100_features_dict = clean_feature_names(top_100_features)

    past_covs_list   = [subset_safe(ts, top_100_features_dict['pastcov_features_base']) for ts in past_covs_list]
    future_covs_list = [subset_safe(ts, top_100_features_dict['futcov_features_base'])  for ts in future_covs_list]

    region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END, CV_START_VAL = \
        get_covs_and_encodings(target_series_list, past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC)

    sample = past_covs_list[0]
    base_set = top_100_features_dict['pastcov_features_base']
    present  = base_set & set(sample.components)
    missing  = base_set - set(sample.components)
    print(f"{len(present)}/{len(base_set)} base features present in past_covs")
    if missing:
        print("missing (first 10):", list(missing)[:10])
    sample = future_covs_list[0]
    base_set = top_100_features_dict['futcov_features_base']
    present  = base_set & set(sample.components)
    missing  = base_set - set(sample.components)
    print(f"{len(present)}/{len(base_set)} base features present in future_covs")
    if missing:
        print("missing (first 10):", list(missing)[:10])


    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump((region_names, train_target, val_target, test_target, full_past_covs, full_fut_covs, target_for_cv, TRAIN_VAL_END, CV_START_VAL), f)
    print(f"Saved computed sets to {path}")


# In[ ]:


_, _, _, _, full_raw_past_covs_LSTM, _, _, _, _ = \
    get_covs_and_encodings(target_series_list, raw_past_covs_list, future_covs_list, TRAIN_FRAC, VAL_FRAC)


# ##  Evaluation utilities
# 
# Metric suite chosen for **zero-inflated count-series** forecasting. Each
# metric earns its place by answering a question the others can't:
# 
# | Metric | Answers | Why it fits zero-inflated counts |
# |---|---|---|
# | **MAE** | Average miss size | Robust to the many zeros (L1 loss, no squared explosion) |
# | **RMSE** | Worst-miss sensitivity | Penalises the few days where we badly under-forecast a big strike wave |
# | **MedAE** | Typical-day miss | Ignores the tail; useful when the long right-tail dominates RMSE |
# | **ME** | Systematic bias | `+` = over-forecasting strikes, `-` = under-forecasting. Naive-model ME should be near 0 |
# | **PoissonDev** | Count-likelihood fit | Proper scoring rule for counts; natural loss for the Poisson regressors |
# | **TweedieDev** (power=1.5) | Zero-inflated fit | Proper scoring rule that rewards predicting exact zeros correctly |
# | **ZeroAcc** | Zero-prediction sanity | Fraction of days where `pred < 0.5` matches `y_true == 0` |
# | **MASE** | Did we beat seasonal-naive? | Scale-free; divides MAE by in-sample MAE of a weekly naive. `< 1` = better than naive |
# | **RMSSE** | Squared-error variant of MASE | The M5-competition standard |
# 
# 
# Two views of quality:
# 
# | scope              | granularity                           |
# | ------------------ | ------------------------------------- |
# | `per_region`       | one row per region (pooled folds + horizons) |
# | `per_horizon`      | one row per horizon step 1..7 |
# | `per_region_horizon` | region * horizon — feeds the heatmap  |
# | `global`           | pooled metrics + MASE/RMSSE averaged across regions |
# 

# In[ ]:


from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_poisson_deviance, mean_tweedie_deviance,
)

EPS               = 1e-9   # guards log / ratio when preds land at zero
NAIVE_SEASONALITY = 7      # weekly seasonality for MASE / RMSSE scales

# ----------------------------------------------------------------------
# Per-region in-sample scales for MASE / RMSSE
# ----------------------------------------------------------------------
# MASE (Hyndman-Koehler 2006) is the gold standard scale-free metric for
# forecast evaluation. It divides the out-of-sample MAE by the MAE of a
# seasonal-naive forecast on the *training* data. Result < 1 => beats the
# in-sample seasonal-naive benchmark. Because it normalises per series it
# is the only sensible way to average errors across regions of very
# different activity levels (donetsk vs chernivtsi etc.).
# RMSSE (from the M5 competition) is its squared-error twin.
# compute_naive_scales imported from src

# evaluate_long imported from src

MAE_SCALES, RMSE_SCALES = compute_naive_scales(
    train_target, region_names, seasonality=NAIVE_SEASONALITY,
)


# In[ ]:


# plot_region_horizon_heatmap imported from src



# ## Naive baselines
# 
# Two rock-bottom floors, both produced as Darts `TimeSeries` so they feed the
# exact same evaluator as every other model.
# 
# - **`naive_last`** — persistence: `y_hat(t+h) = y(t-1)` for every horizon step.
#   (*'today's prediction = yesterday's value, repeated'*)
# - **`naive_weekly`** — seasonal naive: `y_hat(t+h) = y(t+h-7)`.
#   (*'forecast this week = last week, shifted forward'*)
# 
# Any serious model must beat both. Weekly naive is especially hard to beat on
# calendar-driven targets.
# 

# In[ ]:


# naive_last_historical_forecasts imported from src



# In[ ]:


# naive_collect_long imported from src



# ## Train + cross-validate every model
# 
# Each non-naive model is trained with an **expanding-window backtest on the
# validation segment only**. Folds are weekly (stride = `CV_STRIDE`) so each
# 7-day prediction window is non-overlapping. The test set (last 20 %) is
# never touched here — it only sees the winning configuration once, at the end.
# 
# The loop dispatches by model family:
# - **Global tabular / neural** (`GBM_MODELS`, `GLM_MODELS`, `NEURAL_MODELS`) —
#   one Darts model fit across all regions, using past + future covariates.
#   Neural models get their covariates scaled (targets are left on the original
#   scale to keep count-style losses meaningful).
# - **Local** (`LOCAL_MODELS`, currently just ARIMA) — a fresh model per region,
#   no covariates.
# - **Naive** — closed-form shift operations, no fitting at all.
# 

# In[ ]:


# _maybe_scale_covs imported from src

def run_manual_expanding_cv(name):
    """Expanding-window CV with a fresh model per fold.

    Returns ``list[list[TimeSeries]]`` — outer by region, inner by fold,
    matching the shape produced by Darts' ``historical_forecasts``.
    """
    ref_ts    = target_for_cv[0]                 # shared time axis
    n_total   = len(ref_ts)
    start_idx = int(CV_START_VAL * n_total)

    n_regions      = len(target_for_cv)
    all_fold_preds = [[] for _ in range(n_regions)]
    n_folds        = 0

    is_local   = name in LOCAL_MODELS
    is_neural  = name in NEURAL_MODELS
    # Scale covariates for neural nets
    needs_scaling = is_neural

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        full_past_covs, full_fut_covs, do_scale=needs_scaling,
    )

    for t0 in range(start_idx, n_total - OUTPUT_CHUNK_LEN + 1, CV_STRIDE):
        split_time   = ref_ts.time_index[t0]
        train_series = [ts.drop_after(split_time) for ts in target_for_cv]

        if is_local:
            # One model per region; ARIMA doesn't use covariates here.
            preds = []
            for ts in train_series:
                m = build_regressor(name)
                m.fit(ts)
                preds.append(m.predict(n=OUTPUT_CHUNK_LEN))
        else:
            model      = build_regressor(name)
            fit_kwargs = {"series": train_series}
            pred_kwargs = {"n": OUTPUT_CHUNK_LEN, "series": train_series}
            if model.supports_past_covariates:
                fit_kwargs["past_covariates"]  = past_for_fit
                pred_kwargs["past_covariates"] = past_for_fit
            if model.supports_future_covariates:
                fit_kwargs["future_covariates"]  = fut_for_fit
                pred_kwargs["future_covariates"] = fut_for_fit
            model.fit(**fit_kwargs)

            # Probabilistic models (e.g. TFT + QuantileRegression) need
            # multiple MC samples before we can take a point estimate.
            is_probabilistic = is_neural and getattr(model, "likelihood", None) is not None
            if is_probabilistic:
              pred_kwargs["num_samples"] = 200
            preds = model.predict(show_warnings=False, **pred_kwargs)
            if is_probabilistic:
              preds = [p.quantile(0.5) for p in preds]  # median = point forecast

        for r_idx, pred in enumerate(preds):
            all_fold_preds[r_idx].append(pred)

        n_folds += 1
        if n_folds == 1 or n_folds % 4 == 0:
            print(f"   fold {n_folds} done  (trained up to {split_time.date()})")

    print(f"   {n_folds} folds complete")
    return all_fold_preds


# In[ ]:


# run_expanding_cv_iter imported from src



# In[ ]:


from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CometLogger
import optuna

# ---------------------------------------------------------------------------
# Generalized expanding-window CV: works for the val phase (target_for_cv,
# CV_START_VAL) and for the test phase (full series, TRAIN_VAL_END).
# Returns fold predictions in the same shape as run_manual_expanding_cv.
# ---------------------------------------------------------------------------
# run_expanding_cv imported from src

def _suggest_lightgbm_params(trial, objective_kind: str):
    p = {
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
    if objective_kind == "tweedie":
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return p


def _suggest_xgboost_params(trial, objective_kind: str):
    p = {
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=100),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if objective_kind == "tweedie":
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return p


def _suggest_catboost_params(trial, objective_kind: str):
    p = {
        "depth":         trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "iterations":    trial.suggest_int("iterations", 200, 1000, step=100),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 5.0, step=0.5),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
    }
    if objective_kind == "tweedie":
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return p


SUGGESTERS_BY_FAMILY = {
    "lightgbm": _suggest_lightgbm_params,
    "xgboost":  _suggest_xgboost_params,
    "catboost": _suggest_catboost_params,
}


def _nn_trainer_kwargs(trial=None):
    """Trainer kwargs with early stopping. Optuna trial enables LightningPruning."""
    callbacks = [EarlyStopping(monitor="train_loss", patience=5, min_delta=1e-4, mode="min")]

    if trial is not None:
        try:
            from optuna.integration import PyTorchLightningPruningCallback
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="train_loss"))
        except ImportError:
            pass

    comet_logger = CometLogger(
        api_key="X",
        project_name="thesis",
        experiment_name=f"lstm-trial-{trial.number}" if trial else "lstm-final"
    )

    return dict(
        accelerator          = "auto",
        enable_progress_bar  = False,
        enable_model_summary = False,
        log_every_n_steps    = 10,
        gradient_clip_val    = 1.0,
        callbacks            = callbacks,
        logger               = comet_logger, 
    )


# TFT removed from comparison — only LSTM remains in the NN search space.
def _suggest_lstm_params(trial, objective_kind: str):
    fc_choice = trial.suggest_categorical("hidden_fc_sizes", ["none", "32", "64", "64_32"])
    fc_map = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}

    params = dict(
        model               = trial.suggest_categorical("model", ["LSTM", "GRU"]),
        input_chunk_length  = trial.suggest_categorical("input_chunk_length", [7, 14, 21, 28]),
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
    extras = {}
    if objective_kind == "tweedie":
        extras["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return params, extras

def _score_fold_preds(fold_preds, target_list, region_names_, metric="MASE_mean"):
    """Run our existing metric pipeline on a fold-pred bundle. Returns the chosen global metric."""
    long_df = collect_predictions_long(target_list, fold_preds, region_names_)
    res     = evaluate_long(long_df, MAE_SCALES, RMSE_SCALES)
    return float(res["global"][metric])


def make_gbm_objective(variant: str):
    """Optuna objective that reports per-fold cumulative score so
    MedianPruner can actually fire."""
    family, objective_kind = variant.lower().split("_", 1)
    suggester = SUGGESTERS_BY_FAMILY[family]

    def _objective(trial):
        params  = suggester(trial, objective_kind)
        builder = lambda: build_gbm_from_params(variant, params)

        last_score = None
        for step, cumulative_fold_preds in enumerate(run_expanding_cv_iter(
            builder,
            target_list = target_for_cv,
            start_frac  = CV_START_VAL,
            horizon     = OUTPUT_CHUNK_LEN,
            stride      = CV_STRIDE,
            past_covs   = full_past_covs,
            future_covs = full_fut_covs,
            verbose     = False,
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
    """One Optuna study per LSTM variant: 'lstm_poisson' or 'lstm_tweedie'."""
    family, objective_kind = variant.lower().split("_", 1)
    if family != "lstm" or objective_kind not in {"poisson", "tweedie"}:
        raise ValueError(f"unsupported NN variant {variant!r}")

    def _objective(trial):
        params, extras = _suggest_lstm_params(trial, objective_kind)
        tw_p = extras.get("tweedie_variance_power", 1.5)
        builder = lambda: build_lstm_count(objective_kind, params, tweedie_power=tw_p)

        fold_preds = run_expanding_cv(
            builder,
            target_list = target_for_cv,
            start_frac  = CV_START_VAL,
            horizon     = OUTPUT_CHUNK_LEN,
            stride      = CV_STRIDE,
            past_covs   = full_raw_past_covs_LSTM,
            future_covs = full_fut_covs,
            is_neural   = True,
            verbose     = False,
        )
        return _score_fold_preds(
            fold_preds, target_for_cv, region_names, metric="RMSSE_mean"
        )

    return _objective


# 

# In[ ]:


from pathlib import Path
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNE_CKPT = Path(PROJECT_DIR) / "checkpoints_tune"
TUNE_CKPT.mkdir(exist_ok=True)

best_params_by_variant = {}
studies                = {}
NN_VARIANTS = ["lstm_poisson", "lstm_tweedie"]
all_variants = GBM_VARIANTS    # 6 GBM + 2 LSTM
# all_variants = NN_VARIANTS    # 6 GBM + 2 LSTM

for variant in all_variants:
    ckpt = TUNE_CKPT / f"{variant}_best.pkl"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            best_params_by_variant[variant], studies[variant] = pickle.load(f)
        print(f"[{variant}] loaded cached best_value={studies[variant].best_value:.4f}")
        continue

    is_nn = variant in NN_VARIANTS
    n_trials = 25 if is_nn else OPTUNA_N_TRIALS
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
            timeout  = OPTUNA_TIMEOUT_S if "OPTUNA_TIMEOUT_S" in dir() else None,
        )
    else:
        study.optimize(
            make_gbm_objective(variant),
            n_trials = n_trials,
            timeout  = OPTUNA_TIMEOUT_S if "OPTUNA_TIMEOUT_S" in dir() else None,
            show_progress_bar = True,
            n_jobs = 1
        )

    best_params_by_variant[variant] = study.best_params
    studies[variant]                = study

    with open(ckpt, "wb") as f:
        pickle.dump((study.best_params, study), f)

    print(f"[{variant}] best score (val CV) = {study.best_value:.4f}")
    print(f"[{variant}] best params: {study.best_params}")


# In[ ]:


import pickle
from pathlib import Path

CKPT = Path(PROJECT_DIR) / "checkpoints"
CKPT.mkdir(exist_ok=True)

long_by_model       = {}
fold_preds_by_model = {}


def _build_lstm_from_best(variant: str, best: dict):
    """Reconstruct the LSTM trial params from study.best_params for a re-run.
    `study.best_params` is flat; rebuild the same dict shape _suggest_lstm_params
    produces, then strip tweedie_variance_power into a separate kwarg."""
    fc_map = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}
    b      = dict(best)
    tw_p   = b.pop("tweedie_variance_power", 1.5)

    params = dict(
        model               = b["model"],
        input_chunk_length  = b["input_chunk_length"],
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
        pl_trainer_kwargs   = _nn_trainer_kwargs(),    # no trial -> no pruning callback
    )
    objective_kind = variant.split("_", 1)[1]          # "poisson" / "tweedie"
    return lambda: build_lstm_count(objective_kind, params, tweedie_power=tw_p)


for variant, best_params in best_params_by_variant.items():
    name = f"{variant}_tuned"
    if name in long_by_model:
        print(f"skip {name} — already cached in long_by_model")
        continue

    print(f"\n=== Re-running val CV with best {variant} params ===")

    # --- ADD CONDITIONAL COVARIATE ROUTING ---
    if variant in NN_VARIANTS:
        builder   = _build_lstm_from_best(variant, best_params)
        is_neural = True
        chosen_past_covs = full_raw_past_covs_LSTM
    else:
        builder   = lambda p=best_params, v=variant: build_gbm_from_params(v, p)
        is_neural = False
        chosen_past_covs = full_past_covs     

    fold_preds = run_expanding_cv(
        builder,
        target_list = target_for_cv,
        start_frac  = CV_START_VAL,
        horizon      = OUTPUT_CHUNK_LEN,
        stride       = CV_STRIDE,
        past_covs   = chosen_past_covs,      
        future_covs = full_fut_covs,
        is_neural   = is_neural,
    )

    long_df = collect_predictions_long(target_for_cv, fold_preds, region_names)

    long_by_model[name]       = long_df
    fold_preds_by_model[name] = fold_preds

    with open(CKPT / f"{name}.pkl", "wb") as f:
        pickle.dump((long_df, fold_preds), f)


# ## 10. Leaderboard - which Regressor wins overall?

# In[ ]:


results_by_model = {}
for name, long_df in long_by_model.items():
    res = evaluate_long(long_df, MAE_SCALES, RMSE_SCALES)
    results_by_model[name] = res


# In[ ]:


# Leaderboard — sorted by MASE_mean (scale-free, comparable across regions).
# Skill scores give a direct sanity check:
#   Skill = 1 - metric(model) / metric(naive_weekly_OOS)
#     > 0  model is better than the out-of-sample seasonal naive
#     = 0  same
#     < 0  worse than doing nothing smart
# If `naive_weekly` wasn't run, skill columns stay NaN.
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
      .sort_values("MASE_mean", ascending=True)
      .reset_index(drop=True)
)
leaderboard


# ## 13. Per-region detail for the winning model

# In[ ]:


winner = leaderboard.iloc[0]["model"]
print(f"Winning model on MASE_mean: {winner}")
results_by_model[winner]["per_region"]


# In[ ]:


winner = leaderboard.iloc[0]["model"]
print(f"Winning model on MASE_mean: {winner}")
results_by_model[winner]["per_horizon"]


# In[ ]:


target_full = [tr.append(vl).append(te)
               for tr, vl, te in zip(train_target, val_target, test_target)]
TEST_START_FRAC = TRAIN_VAL_END

test_long_by_model       = {}
test_fold_preds_by_model = {}

for variant, best_params in best_params_by_variant.items():
    name = f"{variant}_tuned"
    print(f"\n=== Test-set CV: {name} ===")

    if variant in NN_VARIANTS:
        builder   = _build_lstm_from_best(variant, best_params)
        is_neural = True
        chosen_past_covs = full_raw_past_covs_LSTM

    else:
        builder   = lambda p=best_params, v=variant: build_gbm_from_params(v, p)
        is_neural = False
        chosen_past_covs = full_past_covs  


    fold_preds = run_final_test(
        builder,
        target_list    = target_full,
        start_frac     = TEST_START_FRAC,
        predict_stride = 1,
        retrain_stride = OUTPUT_CHUNK_LEN,
        horizon        = OUTPUT_CHUNK_LEN,
        past_covs      = chosen_past_covs,
        future_covs    = full_fut_covs,
        is_neural      = is_neural,
    )
    long_df = collect_predictions_long(target_full, fold_preds, region_names)
    test_long_by_model[name]       = long_df
    test_fold_preds_by_model[name] = fold_preds

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
      .sort_values("MASE_mean", ascending=True)
      .reset_index(drop=True)
)
print(test_leaderboard)


# In[ ]:





# In[ ]:





# In[ ]:




