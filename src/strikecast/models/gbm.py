"""Gradient-boosting model specs: LightGBM, XGBoost and CatBoost.

One :class:`~strikecast.models.spec.ModelSpec` per GBDT variant the thesis ran,
for two experiments:

``count``
    ``_regression_GBDT.py`` / ``_regression_LSTM.py``.  Six variants, the three
    families crossed with the two count objectives (Poisson, Tweedie).  Legacy
    variant strings ``GBM_VARIANTS`` (``_regression_GBDT.py``:118-121); result
    files are named ``<variant>_tuned`` (``_regression_GBDT.py``:1162).
``diff``
    ``_diff_regression.py``.  Three variants, one per family, all on a plain
    squared-error objective (``regression`` / ``reg:squarederror`` / ``RMSE``).
    Legacy variant strings ``GBM_VARIANTS = ["lightgbm", "xgboost",
    "catboost"]`` (``_diff_regression.py``:119), the names of the tuning
    directories under ``golden/converted/tuning/checkpoints_tune_diff/``;
    result files are again ``<variant>_tuned``.

What ``build`` reproduces
-------------------------
``build(params, ctx)`` is the legacy ``build_gbm_from_params(variant, params)``
of the owning script, with three module globals lifted into ``ctx``:

======================  ==================================================
legacy global           ``RunContext`` field
======================  ==================================================
``RANDOM_STATE`` (42)   ``ctx.seed`` -> ``random_state`` / ``random_seed``
device literal          ``ctx.device`` -> ``device_type`` (LightGBM),
                        ``device`` (XGBoost), ``task_type`` (CatBoost)
``available_threads``   ``ctx.threads`` -> ``num_threads`` / ``n_jobs`` /
                        ``thread_count``, and ONLY for the variants whose
                        legacy builder actually passed it
======================  ==================================================

``spec.device`` defaults to the device the *tuned* builder used, because the
tuned builders produced every reported number: CPU everywhere in the count
family (``device_type="cpu"``, ``device="cpu"``, ``task_type="CPU"``;
``_regression_GBDT.py``:513-576) and, in the diff family, LightGBM on CPU,
XGBoost ``device="cuda"``, CatBoost ``task_type="GPU"``
(``_diff_regression.py``:411-461).  See plan sec. 2.3 row "Device" and F9.

``spec.threads_from_context`` is True exactly where the legacy builder passed a
thread keyword.  Where it was commented out, the keyword is never emitted and
the library default applies, exactly as before.  ``ctx.threads is None`` also
suppresses the keyword, which is what reproduces the GPU default branch.

The darts skeleton is :func:`~strikecast.models.spec.darts_common_kwargs`:
``legacy_common_kwargs()`` bit-for-bit when ``ctx.past_lags is None``, the
selected per-component past lags otherwise (figure protocol).

``defaults`` are the untuned parameters of the legacy ``build_regressor``
branch (minus the shared ``COMMON_KWARGS_TAB``, the device literal, the seed
and the fixed objective/verbosity kwargs).  ``build(spec.defaults, ctx)``
reproduces ``build_regressor(<variant>)`` exactly, given the ctx that branch
ran under.  The single documented exception is the diff CatBoost variant: its
default branch omits ``boost_from_average`` while its tuned builder passes
``boost_from_average=False`` (flag F83, see below); ``build`` always follows
the tuned builder.

XGBoost: one booster per horizon (audit 2026-09-26 B3/B4, decision D5)
----------------------------------------------------------------------
darts 0.43 wraps a model in its per-horizon ``MultiOutputRegressor`` only when
the estimator does NOT report native multi-output support
(``sklearn_model.py``: ``_supports_native_multioutput``). With xgboost 3.1.3 +
scikit-learn 1.6 ``XGBRegressor`` reports it, so darts hands it a 7-column
target: ``count:poisson`` / ``reg:tweedie`` then fail ("multioutput is not
supported by the current objective function", B3) and the diff family's
``reg:squarederror`` silently trains ONE multi-output booster (B4). The thesis
text (a separate model ``f_h`` per horizon) and the Colab environment that
produced the count XGBoost numbers wrapped per horizon, so every XGBoost variant
is built as :class:`PerHorizonXGBModel`, which reports no native multi-output
support and therefore always gets darts' per-horizon wrapper, exactly like
LightGBM and CatBoost here.

Flags raised by this module (see ``docs/REFACTOR_PLAN.md`` sec. 4)
-----------------------------------------------------------------
F83, F84, F85 -- recorded in the phase report, not acted upon here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


from .spec import ModelSpec, RunContext, darts_common_kwargs, register

__all__ = [
    "COUNT_GBM_VARIANTS",
    "DIFF_GBM_VARIANTS",
    "RESULT_FILE_SUFFIX",
]

#: Legacy ``GBM_VARIANTS`` of ``_regression_GBDT.py``:118-121, in lineup order.
COUNT_GBM_VARIANTS = (
    "lightgbm_poisson",
    "lightgbm_tweedie",
    "xgboost_poisson",
    "xgboost_tweedie",
    "catboost_poisson",
    "catboost_tweedie",
)

#: Legacy ``GBM_VARIANTS`` of ``_diff_regression.py``:119, in lineup order.
DIFF_GBM_VARIANTS = ("lightgbm", "xgboost", "catboost")

#: The legacy runners store results under ``f"{variant}_tuned"``
#: (``_regression_GBDT.py``:1162, ``_diff_regression.py``:1268).
RESULT_FILE_SUFFIX = "_tuned"

#: ``OPTUNA_N_TRIALS`` in both scripts.
_N_TRIALS = 50


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #
def _threads(ctx: RunContext, keyword: str, *, enabled: bool) -> dict[str, Any]:
    """``{keyword: ctx.threads}`` where the legacy builder passed it, else ``{}``."""
    if not enabled or ctx.threads is None:
        return {}
    return {keyword: ctx.threads}


def _build_lightgbm(
    params: Mapping[str, Any],
    ctx: RunContext,
    *,
    objective: str,
    tweedie: bool,
    multi_models: bool,
    pass_threads: bool,
) -> Any:
    """``build_gbm_from_params`` LightGBM branch (count 519-535 / diff 416-428)."""
    from darts.models import LightGBMModel

    p = dict(params)
    extra: dict[str, Any] = {"objective": objective}
    if tweedie:
        # legacy pops, so a params dict without the key raises, exactly as before
        extra["tweedie_variance_power"] = p.pop("tweedie_variance_power")
    head: dict[str, Any] = {"multi_models": True} if multi_models else {}
    return LightGBMModel(
        **darts_common_kwargs(ctx),
        **head,
        random_state=ctx.seed,
        verbose=-1,
        device_type=ctx.device,
        **_threads(ctx, "num_threads", enabled=pass_threads),
        force_col_wise=True,
        **p,
        **extra,
    )


def _per_horizon_xgb_class() -> type:
    """:class:`PerHorizonXGBModel`, created lazily so importing this module
    does not import darts/xgboost."""
    global _PER_HORIZON_XGB
    if _PER_HORIZON_XGB is None:
        from darts.models import XGBModel

        class PerHorizonXGBModel(XGBModel):
            """darts ``XGBModel`` that always trains one booster per horizon (B3/B4, D5).

            Reporting no native multi-output support makes darts wrap the
            ``XGBRegressor`` in its ``MultiOutputRegressor`` whenever
            ``output_chunk_length > 1`` and ``multi_models`` -- the direct
            strategy the thesis describes, and the only one ``count:poisson``
            and ``reg:tweedie`` can fit.
            """

            @property
            def _supports_native_multioutput(self) -> bool:
                return False

        PerHorizonXGBModel.__module__ = __name__
        PerHorizonXGBModel.__qualname__ = "PerHorizonXGBModel"
        _PER_HORIZON_XGB = PerHorizonXGBModel
    return _PER_HORIZON_XGB


_PER_HORIZON_XGB: type | None = None


def __getattr__(name: str) -> Any:
    # `from strikecast.models.gbm import PerHorizonXGBModel` (and unpickling)
    # resolve the lazily created class through the module.
    if name == "PerHorizonXGBModel":
        return _per_horizon_xgb_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_xgboost(
    params: Mapping[str, Any],
    ctx: RunContext,
    *,
    objective: str,
    tweedie: bool,
    multi_models: bool,
    pass_threads: bool,
) -> Any:
    """``build_gbm_from_params`` XGBoost branch (count 537-552 / diff 430-442).

    Built as :class:`PerHorizonXGBModel` (one booster per horizon, B3/B4, D5);
    every constructor kwarg is the legacy builder's.
    """
    XGBModel = _per_horizon_xgb_class()  # noqa: N806

    p = dict(params)
    extra: dict[str, Any] = {"objective": objective}
    if tweedie:
        extra["tweedie_variance_power"] = p.pop("tweedie_variance_power")
    head: dict[str, Any] = {"multi_models": True} if multi_models else {}
    return XGBModel(
        **darts_common_kwargs(ctx),
        **head,
        tree_method="hist",
        device=ctx.device,
        **_threads(ctx, "n_jobs", enabled=pass_threads),
        random_state=ctx.seed,
        verbosity=0,
        **p,
        **extra,
    )


def _build_catboost(
    params: Mapping[str, Any],
    ctx: RunContext,
    *,
    loss_function: str,
    tweedie: bool,
    multi_models: bool,
    pass_threads: bool,
) -> Any:
    """``build_gbm_from_params`` CatBoost branch (count 554-574 / diff 444-459).

    The Tweedie loss is a *string*, interpolated from the tuned variance power:
    ``f"Tweedie:variance_power={vp}"``.  ``boost_from_average=False`` is always
    passed, because both tuned builders pass it (F83).
    """
    from darts.models import CatBoostModel

    p = dict(params)
    if tweedie:
        vp = p.pop("tweedie_variance_power")
        loss = f"Tweedie:variance_power={vp}"
    else:
        loss = loss_function
    head: dict[str, Any] = {"multi_models": True} if multi_models else {}
    return CatBoostModel(
        **darts_common_kwargs(ctx),
        **head,
        loss_function=loss,
        boost_from_average=False,
        bootstrap_type="Bernoulli",
        task_type=ctx.device,
        **_threads(ctx, "thread_count", enabled=pass_threads),
        random_seed=ctx.seed,
        verbose=False,
        **p,
    )


_BUILDERS = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "catboost": _build_catboost,
}
_OBJECTIVE_KEY = {
    "lightgbm": "objective",
    "xgboost": "objective",
    "catboost": "loss_function",
}


# --------------------------------------------------------------------------- #
# search spaces                                                                #
# --------------------------------------------------------------------------- #
def _suggest_lightgbm(trial: Any, tweedie: bool) -> dict[str, Any]:
    """``_suggest_lightgbm_params`` (count 880-895, diff 1015-1026)."""
    p = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if tweedie:
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return p


def _suggest_xgboost(trial: Any, tweedie: bool) -> dict[str, Any]:
    """``_suggest_xgboost_params`` (count 897-911, diff 1028-1038)."""
    p = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if tweedie:
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return p


def _suggest_catboost(trial: Any, tweedie: bool) -> dict[str, Any]:
    """``_suggest_catboost_params`` (count 913-924, diff 1040-1047)."""
    p = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0, step=0.5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    if tweedie:
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return p


_SUGGESTERS = {
    "lightgbm": _suggest_lightgbm,
    "xgboost": _suggest_xgboost,
    "catboost": _suggest_catboost,
}


# --------------------------------------------------------------------------- #
# default-branch parameters of build_regressor                                 #
# --------------------------------------------------------------------------- #
#: ``build_regressor`` LightGBM defaults, count 340-366 and diff 296-313.
#: Identical in both scripts.
_LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}

#: ``build_regressor`` XGBoost defaults, count 390-411 and diff 317-336.
_XGBOOST_DEFAULTS: dict[str, Any] = {
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

#: ``build_regressor`` CatBoost defaults, count 434-453 and diff 340-355.
_CATBOOST_DEFAULTS: dict[str, Any] = {
    "depth": 5,
    "learning_rate": 0.05,
    "iterations": 500,
    "l2_leaf_reg": 3,
    "subsample": 0.8,
}

_FAMILY_DEFAULTS = {
    "lightgbm": _LIGHTGBM_DEFAULTS,
    "xgboost": _XGBOOST_DEFAULTS,
    "catboost": _CATBOOST_DEFAULTS,
}

#: The Tweedie variance power the count default branches hard-code
#: (``tweedie_variance_power=1.5``, ``"Tweedie:variance_power=1.5"``).
_DEFAULT_TWEEDIE_POWER = 1.5


# --------------------------------------------------------------------------- #
# registration                                                                 #
# --------------------------------------------------------------------------- #
def _make_spec(
    *,
    name: str,
    family: str,
    experiment: str,
    objective: str,
    tweedie: bool,
    device: str,
    multi_models: bool,
    pass_threads: bool,
) -> ModelSpec:
    builder = _BUILDERS[family]
    objective_key = _OBJECTIVE_KEY[family]
    suggester = _SUGGESTERS[family]

    defaults = dict(_FAMILY_DEFAULTS[family])
    if tweedie:
        defaults["tweedie_variance_power"] = _DEFAULT_TWEEDIE_POWER

    def build(params: Mapping[str, Any], ctx: RunContext) -> Any:
        return builder(
            params,
            ctx,
            tweedie=tweedie,
            multi_models=multi_models,
            pass_threads=pass_threads,
            **{objective_key: objective},
        )

    def search_space(trial: Any) -> dict[str, Any]:
        return suggester(trial, tweedie)

    return register(
        ModelSpec(
            name=name,
            family=family,
            kind="global",
            build=build,
            experiments=(experiment,),
            defaults=MappingProxyType(defaults),
            search_space=search_space,
            # ``from_best_params`` stays the identity: the legacy runner passes
            # ``study.best_params`` straight into ``build_gbm_from_params``
            # (_regression_GBDT.py:1175, _diff_regression.py:1281), and the
            # tweedie_variance_power pop happens inside the builder.
            is_neural=False,
            stochastic=True,  # subsample / colsample bagging
            needs_raw_past_covs=False,
            n_trials=_N_TRIALS,
            device=device,
            threads_from_context=pass_threads,
            tags=("gbdt", experiment),
        )
    )


# ---- count experiment -----------------------------------------------------
# Tuned builders: LightGBM device_type="cpu" + num_threads, XGBoost
# device="cpu" + n_jobs, CatBoost task_type="CPU" + thread_count.
#: CatBoost's Tweedie loss string is interpolated from the tuned variance power
#: inside :func:`_build_catboost`, so the spec's ``loss_function`` is unused.
_CATBOOST_TWEEDIE_LOSS_IS_BUILT = "unused: see _build_catboost"

_COUNT_OBJECTIVES = {
    ("lightgbm", "poisson"): "poisson",
    ("lightgbm", "tweedie"): "tweedie",
    ("xgboost", "poisson"): "count:poisson",
    ("xgboost", "tweedie"): "reg:tweedie",
    ("catboost", "poisson"): "Poisson",
    ("catboost", "tweedie"): _CATBOOST_TWEEDIE_LOSS_IS_BUILT,
}
_COUNT_DEVICE = {"lightgbm": "cpu", "xgboost": "cpu", "catboost": "CPU"}

for _variant in COUNT_GBM_VARIANTS:
    _family, _objective_kind = _variant.split("_", 1)
    _make_spec(
        name=_variant,
        family=_family,
        experiment="count",
        objective=_COUNT_OBJECTIVES[(_family, _objective_kind)],
        tweedie=_objective_kind == "tweedie",
        device=_COUNT_DEVICE[_family],
        multi_models=False,  # count builders never pass multi_models
        pass_threads=True,  # all three count tuned builders pass threads
    )

# ---- diff experiment ------------------------------------------------------
# Tuned builders: LightGBM device_type="cpu" + num_threads, XGBoost
# device="cuda" with n_jobs COMMENTED OUT, CatBoost task_type="GPU" with
# thread_count COMMENTED OUT.
_DIFF_OBJECTIVE = {
    "lightgbm": "regression",
    "xgboost": "reg:squarederror",
    "catboost": "RMSE",
}
_DIFF_DEVICE = {"lightgbm": "cpu", "xgboost": "cuda", "catboost": "GPU"}
_DIFF_PASS_THREADS = {"lightgbm": True, "xgboost": False, "catboost": False}

for _variant in DIFF_GBM_VARIANTS:
    _make_spec(
        name=_variant,
        family=_variant,
        experiment="diff",
        objective=_DIFF_OBJECTIVE[_variant],
        tweedie=False,
        device=_DIFF_DEVICE[_variant],
        multi_models=True,  # diff builders pass multi_models=True explicitly
        pass_threads=_DIFF_PASS_THREADS[_variant],
    )
