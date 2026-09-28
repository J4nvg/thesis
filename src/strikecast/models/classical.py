"""Untuned baseline specs: linear, ARIMA and the two naives.

These are the ``REGRESSORS_TO_RUN`` entries that never went through Optuna
(``_diff_regression.py``:124-129, ``_regression_GBDT.py``:124-127).  None of
them is stochastic, so the seed sweep of plan sec. 5.4 runs them once and
broadcasts (``stochastic=False``).

All four are registered for the **diff experiment only**
----------------------------------------------------------
The count family produced no baselines at all.  ``REGRESSORS_TO_RUN``
(``_regression_GBDT.py``:124-127, ``_regression_LSTM.py``) is dead code: nothing
iterates it, the CV and test loops iterate ``best_params_by_variant`` instead
(``_regression_GBDT.py``:1161, 1281), and ``golden/results/gbdt/leaderboard.csv``
holds 36 rows -- 6 GBDT variants x 3 paradigms x 2 splits -- with no baseline
row at all.  Per name:

``linear``
    ``LinearRegressionModel`` appears in exactly one script,
    ``_diff_regression.py``:360-365 (``LINEAR_VARIANTS = ["linear"]``, line
    121).  ``grep -n LinearRegressionModel _regression_GBDT.py
    _regression_LSTM.py`` is empty: the count scripts have no ``linear`` branch
    in ``build_regressor`` at all.
``arima``
    Both count scripts *define* an ``arima`` branch (identical to the diff one)
    and list ``"arima"`` in their dead ``REGRESSORS_TO_RUN``, but it never ran
    there (F86).  The ARIMA numbers the thesis reports come from the diff run
    (plan sec. 2.1, F13), as Appendix B states.
``naive_last`` / ``naive_weekly``
    In ``NAIVE_MODELS`` and ``REGRESSORS_TO_RUN`` of every script and built
    identically everywhere (``build_regressor`` returns ``None``; the runner
    dispatches to the naive historical-forecast helpers), but only the diff run
    executed and persisted them (F84).  Registering them for ``count`` would
    claim a count-family baseline the thesis does not have; the count
    leaderboard's skill columns, which divide by ``naive_weekly``, are NaN for
    exactly that reason (``_regression_GBDT.py``:1220).

ARIMA specifics
---------------
``ARIMA(p=7, d=0, q=1, random_state=RANDOM_STATE)`` on the **differenced**
target, which is the thesis's ARIMA(7, 1, 1) on levels (plan sec. 2.1, F13).
``kind="local"`` -- one model per region, rebuilt and refit inside every
prediction step -- with the ``NaiveMean`` fallback of F5
(``tests/legacy_ref/diff_runners.py``:138, 221, 309).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


from .spec import ModelSpec, RunContext, darts_common_kwargs, register

__all__ = ["NAIVE_NAMES"]

#: ``NAIVE_MODELS`` of all three scripts, in lineup order.
NAIVE_NAMES = ("naive_last", "naive_weekly")


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #
def _build_linear(params: Mapping[str, Any], ctx: RunContext) -> Any:
    """``build_regressor("linear")`` -- ``_diff_regression.py``:360-365.

    ``LinearRegressionModel(**COMMON_KWARGS_TAB, multi_models=True)``: no
    hyper-parameters, no ``random_state``, no device.  ``params`` and ``ctx``
    are accepted for the registry signature; ``ctx`` is read only for
    ``past_lags`` (figure protocol, :func:`~strikecast.models.spec.darts_common_kwargs`),
    because the legacy builder ignored every global except ``COMMON_KWARGS_TAB``
    (no seed, no device).
    """
    from darts.models import LinearRegressionModel

    return LinearRegressionModel(
        **darts_common_kwargs(ctx),
        multi_models=True,
        **dict(params),
    )


def _build_arima(params: Mapping[str, Any], ctx: RunContext) -> Any:
    """``build_regressor("arima")`` -- ``_diff_regression.py``:392-397."""
    from darts.models import ARIMA

    p = dict(params)
    return ARIMA(
        p=p.pop("p"),
        d=p.pop("d"),
        q=p.pop("q"),
        random_state=ctx.seed,
        **p,
    )


def _naive_mean_fallback() -> Any:
    """``NaiveMean()`` -- the F5 fallback of the diff local runner."""
    from darts.models import NaiveMean

    return NaiveMean()


def _build_naive(params: Mapping[str, Any], ctx: RunContext) -> None:
    """``build_regressor(name)`` returns ``None`` for both naive names.

    The naive forecasts are produced by ``strikecast.backtest.naive``, not by a
    darts model, exactly as the legacy runners dispatched them.
    """
    del params, ctx
    return None


# --------------------------------------------------------------------------- #
# registration                                                                 #
# --------------------------------------------------------------------------- #
register(
    ModelSpec(
        name="linear",
        family="linear",
        kind="global",
        build=_build_linear,
        experiments=("diff",),
        defaults=MappingProxyType({}),
        search_space=None,
        is_neural=False,
        stochastic=False,
        needs_raw_past_covs=False,
        n_trials=None,
        device="cpu",
        threads_from_context=False,
        tags=("baseline", "diff"),
    )
)

register(
    ModelSpec(
        name="arima",
        family="arima",
        kind="local",
        build=_build_arima,
        experiments=("diff",),
        defaults=MappingProxyType({"p": 7, "d": 0, "q": 1}),
        search_space=None,
        is_neural=False,
        stochastic=False,
        needs_raw_past_covs=False,
        fallback=_naive_mean_fallback,
        n_trials=None,
        device="cpu",
        threads_from_context=False,
        tags=("baseline", "diff", "local"),
    )
)

for _naive in NAIVE_NAMES:
    register(
        ModelSpec(
            name=_naive,
            family="naive",
            kind="naive",
            build=_build_naive,
            experiments=("diff",),
            defaults=MappingProxyType({}),
            search_space=None,
            is_neural=False,
            stochastic=False,
            needs_raw_past_covs=False,
            n_trials=None,
            device="cpu",
            threads_from_context=False,
            tags=("baseline", "diff"),
        )
    )
