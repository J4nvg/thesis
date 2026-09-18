"""Verbatim extraction of the diff-family model builders and Optuna suggesters.

Source
------
``_diff_regression.py`` at git commit
``2e287a229215eb7e68c80c0291f5075ac8baa7bd`` (branch ``refactor``).

======================================  ======================================
copied here                             source lines
======================================  ======================================
``COMMON_KWARGS_TAB``                   271
``build_regressor``                     285-402
``build_gbm_from_params``               411-461
``_suggest_lightgbm_params``            1015-1026
``_suggest_xgboost_params``             1028-1038
``_suggest_catboost_params``            1040-1047
``SUGGESTERS_BY_FAMILY``                1050-1054
======================================  ======================================

Deviations from the source (all mechanical)
-------------------------------------------
1. Module globals used by the bodies are keyword-only parameters *with the
   same names*: ``COMMON_KWARGS_TAB``, ``RANDOM_STATE``, ``NAIVE_MODELS``,
   ``INPUT_LAGS``, ``OUTPUT_CHUNK_LEN``, ``NN_TRAINER_KWARGS``,
   ``available_threads``.  Their defaults are the script's own values, which
   for this script are all literals: ``RANDOM_STATE = 42`` (line 133),
   ``available_threads = 4`` (line 90, hard-coded, NOT
   ``get_available_threads()`` as in the count scripts).
2. ``get_common_kwargs`` arrives through ``from src import *`` in the script;
   it is imported explicitly here from ``src.prevalent_functions``.
3. ``NN_TRAINER_KWARGS`` (lines 274-281) is a keyword with default ``None`` so
   importing this module does not import pytorch-lightning.  Only the ``lstm``
   branch reads it, and that branch belongs to
   ``src/strikecast/models/rnn.py``, not to the GBM/classical specs.

Every function body below is byte-identical to the source.

This module is TEST-ONLY.  Nothing under ``src/`` may import it.
"""
# ruff: noqa: E501, E251, E252, E262, E301, E302, E303, W291, W293, N803, ARG001, D
# (alignment whitespace, keyword spacing, a trailing space on line 395 of the
#  source and shadowed names are evidence; see tests/legacy_ref/README.md.)

from __future__ import annotations

from src.prevalent_functions import get_common_kwargs

# --- module globals of _diff_regression.py, as module constants -------------
COMMON_KWARGS_TAB = get_common_kwargs()          # _diff_regression.py:271
RANDOM_STATE = 42                                # _diff_regression.py:133
NAIVE_MODELS = {"naive_last", "naive_weekly"}    # _diff_regression.py:115
INPUT_LAGS = 7                                   # _diff_regression.py:106
OUTPUT_CHUNK_LEN = 7                             # _diff_regression.py:105
available_threads = 4                            # _diff_regression.py:90


# --------------------------------------------------------------------------
# _diff_regression.py 285-402
# --------------------------------------------------------------------------
def build_regressor(
    name: str,
    *,
    COMMON_KWARGS_TAB=COMMON_KWARGS_TAB,
    RANDOM_STATE=RANDOM_STATE,
    NAIVE_MODELS=NAIVE_MODELS,
    INPUT_LAGS=INPUT_LAGS,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
    NN_TRAINER_KWARGS=None,
):
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
            multi_models      = True,
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
            # device_type  = "gpu",
            num_threads       = available_threads,
            force_col_wise    = True,
        )

    if name == "xgboost":
        from darts.models import XGBModel
        return XGBModel(
            **COMMON_KWARGS_TAB,
            multi_models      = True,
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
            # device            = "cpu",
            device       = "cuda",
            # n_jobs            = available_threads,
            random_state      = RANDOM_STATE,
            verbosity         = 0,
        )

    if name == "catboost":
        from darts.models import CatBoostModel
        return CatBoostModel(
            **COMMON_KWARGS_TAB,
            multi_models      = True,
            loss_function     = "RMSE",
            depth             = 5,
            learning_rate     = 0.05,
            iterations        = 500,
            l2_leaf_reg       = 3,
            subsample         = 0.8,
            bootstrap_type    = "Bernoulli",
            # task_type         = "CPU",
            task_type          = "GPU",
            # thread_count      = available_threads,
            random_seed       = RANDOM_STATE,
            verbose           = False,
        )

    # ---------------- Linear baseline (no real tuning surface) -------------
    if name == "linear":
        from darts.models import LinearRegressionModel
        return LinearRegressionModel(
            **COMMON_KWARGS_TAB,
            multi_models = True,
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
# _diff_regression.py 411-461
def build_gbm_from_params(
    variant: str,
    params: dict,
    *,
    COMMON_KWARGS_TAB=COMMON_KWARGS_TAB,
    RANDOM_STATE=RANDOM_STATE,
    available_threads=available_threads,
):
    variant = variant.lower()
    p = dict(params)

    if variant == "lightgbm":
        from darts.models import LightGBMModel
        return LightGBMModel(
            **COMMON_KWARGS_TAB,
            multi_models   = True,
            objective      = "regression",
            random_state   = RANDOM_STATE,
            verbose        = -1,
            # device_type  = "gpu",
            device_type    = "cpu",
            num_threads    = available_threads,
            force_col_wise = True,
            **p,
        )

    if variant == "xgboost":
        from darts.models import XGBModel
        return XGBModel(
            **COMMON_KWARGS_TAB,
            multi_models = True,
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
            multi_models       = True,
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


# --------------------------------------------------------------------------
# _diff_regression.py 1015-1054
# --------------------------------------------------------------------------
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
