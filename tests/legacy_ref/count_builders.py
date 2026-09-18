"""Verbatim extraction of the count-family model builders and Optuna suggesters.

Source
------
``_regression_GBDT.py`` and ``_regression_LSTM.py`` at git commit
``2e287a229215eb7e68c80c0291f5075ac8baa7bd`` (branch ``refactor``); both are
byte-identical to their notebooks for these cells.

======================================  ======================================
copied here                             source lines
======================================  ======================================
``COMMON_KWARGS_TAB``                   ``_regression_GBDT.py`` 310
``build_regressor_gbdt``                ``_regression_GBDT.py`` 333-504
``build_regressor_lstm``                ``_regression_LSTM.py`` 337-508
``build_gbm_from_params``               ``_regression_GBDT.py`` 513-576
``_suggest_lightgbm_params``            ``_regression_GBDT.py`` 880-895
``_suggest_xgboost_params``             ``_regression_GBDT.py`` 897-911
``_suggest_catboost_params``            ``_regression_GBDT.py`` 913-924
``SUGGESTERS_BY_FAMILY``                ``_regression_GBDT.py`` 926-930
======================================  ======================================

The two count scripts define the *same* ``build_regressor`` name with drifted
bodies, so both are kept (see "Deviations" 2).  ``build_gbm_from_params`` and
all three suggesters are byte-identical between the two scripts
(``diff`` of ``_regression_GBDT.py`` 513-576 against ``_regression_LSTM.py``
517-580, and of 880-931 against 884-935, is empty), so a single copy stands
for both.

Deviations from the source (all mechanical)
-------------------------------------------
1. Module globals used by the bodies are keyword-only parameters *with the
   same names*, so every body stays byte-identical:
   ``COMMON_KWARGS_TAB``, ``RANDOM_STATE``, ``NAIVE_MODELS``, ``INPUT_LAGS``,
   ``OUTPUT_CHUNK_LEN``, ``NN_TRAINER_KWARGS``, ``available_threads``.
   The defaults are the values the scripts held (``RANDOM_STATE = 42``;
   ``COMMON_KWARGS_TAB = get_common_kwargs()``), except ``available_threads``
   and ``NN_TRAINER_KWARGS``, which have no script-independent value and must
   be passed by the caller.
2. ``build_regressor`` is renamed per source script: ``build_regressor_gbdt``
   (``_regression_GBDT.py``, GBDT default branches on GPU, threads commented
   out) and ``build_regressor_lstm`` (``_regression_LSTM.py``, the same
   branches on CPU with ``num_threads``/``n_jobs``/``thread_count`` live).
   One module cannot hold two functions of the same name.
3. ``COMMON_KWARGS_TAB`` needs ``get_common_kwargs``, which
   ``_regression_GBDT.py`` pulls in through ``from src import *``; it is
   imported explicitly here from its origin,
   ``src.prevalent_functions``.
4. The ``lstm`` branch of both ``build_regressor`` copies reads
   ``NN_TRAINER_KWARGS``, whose script value is an ``EarlyStopping`` instance
   inside a dict.  It is a required-in-practice keyword here (default
   ``None``) so importing this module does not import pytorch-lightning.  The
   neural branches are NOT exercised by the GBM/classical spec tests; they
   belong to ``src/strikecast/models/rnn.py``.  Note the two scripts also
   drift in ``ES_NN``: ``monitor="val_loss"`` in ``_regression_GBDT.py``:312
   against ``monitor="train_loss"`` in ``_regression_LSTM.py``:316.

This module is TEST-ONLY.  Nothing under ``src/`` may import it.
"""
# ruff: noqa: E501, E251, E252, E262, E301, E302, E303, RUF002, RUF003, N803, ARG001, D
# (alignment whitespace, keyword spacing, unicode bullets and shadowed names in
#  the copied bodies are evidence; see tests/legacy_ref/README.md rule 2.)

from __future__ import annotations

from src.prevalent_functions import get_common_kwargs

# --- module globals of the count scripts, as module constants ---------------
COMMON_KWARGS_TAB = get_common_kwargs()          # _regression_GBDT.py:310
RANDOM_STATE = 42                                # _regression_GBDT.py:134
NAIVE_MODELS = {"naive_last", "naive_weekly"}    # _regression_GBDT.py:110
INPUT_LAGS = 7                                   # _regression_GBDT.py:104
OUTPUT_CHUNK_LEN = 7                             # _regression_GBDT.py:103


# --------------------------------------------------------------------------
# _regression_GBDT.py 333-504   (default branches: GBDTs on GPU)
# --------------------------------------------------------------------------
def build_regressor_gbdt(
    name: str,
    *,
    COMMON_KWARGS_TAB=COMMON_KWARGS_TAB,
    RANDOM_STATE=RANDOM_STATE,
    NAIVE_MODELS=NAIVE_MODELS,
    INPUT_LAGS=INPUT_LAGS,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
    NN_TRAINER_KWARGS=None,
):
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
            device_type  = "gpu",
            # device_type       = "cpu",
            # num_threads = available_threads,
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
            device_type  = "gpu",
            # device_type            = "cpu",
            # num_threads = available_threads,
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
            device       = "cuda",
            # device            = "cpu",
            # n_jobs = available_threads,
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
            device       = "cuda",
            # device                 = "cpu",
            # n_jobs = available_threads,
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
            task_type         = "GPU",
            # task_type         = "CPU",
            # thread_count = available_threads,
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
            task_type         = "GPU",
            # task_type         = "CPU",
            # thread_count = available_threads,
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


# --------------------------------------------------------------------------
# _regression_LSTM.py 337-508   (same branches, GBDTs on CPU with threads)
# --------------------------------------------------------------------------
def build_regressor_lstm(
    name: str,
    *,
    COMMON_KWARGS_TAB=COMMON_KWARGS_TAB,
    RANDOM_STATE=RANDOM_STATE,
    NAIVE_MODELS=NAIVE_MODELS,
    INPUT_LAGS=INPUT_LAGS,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
    NN_TRAINER_KWARGS=None,
    available_threads=None,
):
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
            add_encoders = {
                "cyclic": {
                    "future": ["month", "week", "dayofyear", "dayofweek", "day"]
                           },
                },
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
# _regression_GBDT.py 513-576 == _regression_LSTM.py 517-580 (byte-identical)
def build_gbm_from_params(
    variant: str,
    params: dict,
    *,
    COMMON_KWARGS_TAB=COMMON_KWARGS_TAB,
    RANDOM_STATE=RANDOM_STATE,
    available_threads=None,
):
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


# --------------------------------------------------------------------------
# _regression_GBDT.py 880-930 == _regression_LSTM.py 884-934 (byte-identical)
# --------------------------------------------------------------------------
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
