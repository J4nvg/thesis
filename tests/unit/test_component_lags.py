"""Figure protocol: tabular builders take EXACTLY the selected (feature, lag) pairs.

``RunContext.past_lags`` becomes darts per-component ``lags_past_covariates``
through :func:`strikecast.models.spec.darts_common_kwargs`. Each registered
tabular builder is fitted on a small synthetic panel shaped like the real
bundle (3 region series, one-hot region statics, past covariates, 2 future
covariates plus the cyclic encoders of the common kwargs), and the ``_pastcov``
columns of ``lagged_feature_names`` must be the selected pairs and nothing
else. darts orders design columns lag-major, so they are compared as sets.
The past covariates are subset to the selected components first, because
darts requires the dict keys to equal the fitted past components (the data
stage guarantees the same via ``past_keep``).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from strikecast.data.feature_selection import legacy_common_kwargs
from strikecast.models import registry  # noqa: F401  (import registers the specs)
from strikecast.models.spec import (
    RunContext,
    darts_common_kwargs,
    get_spec,
    past_lags_from_mapping,
)

darts = pytest.importorskip("darts")
from darts import TimeSeries  # noqa: E402

PAST_LAGS = (("a", (-7, -14)), ("c", (-1,)))
EXPECTED = {"a_pastcov_lag-7", "a_pastcov_lag-14", "c_pastcov_lag-1"}
N_STEPS = 120
N_SERIES = 3

#: (registered name, experiment, param overrides that keep the fit fast, binary target)
CASES = [
    ("lightgbm_poisson", "count", {"n_estimators": 5}, False),
    ("xgboost_tweedie", "count", {"n_estimators": 5}, False),
    ("catboost_tweedie", "count", {"iterations": 5}, False),
    ("linear", "diff", {}, False),
    ("spe_event_classifier", "hurdle", {"SPE_estm": 3}, True),
    ("catboost_tweedie_count_head", "hurdle", {"iterations": 5, "verbose": False}, False),
]


def _statics(i: int) -> pd.DataFrame:
    # one-hot region columns, as series._encode_statics produces them
    return pd.DataFrame({f"region_{j}": [float(i == j)] for j in range(N_SERIES)})


def _panel(binary: bool) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=N_STEPS, freq="D")
    fidx = pd.date_range("2020-01-01", periods=N_STEPS + 7, freq="D")
    targets, past, future = [], [], []
    for i in range(N_SERIES):
        y = rng.poisson(2.0, (N_STEPS, 1)).astype(float)
        if binary:
            y = (y > 2).astype(float)
        targets.append(
            TimeSeries.from_times_and_values(idx, y, columns=["y"]).with_static_covariates(
                _statics(i)
            )
        )
        p = TimeSeries.from_times_and_values(
            idx, rng.poisson(2.0, (N_STEPS, 3)).astype(float), columns=["a", "b", "c"]
        )
        past.append(p.with_static_covariates(_statics(i)))
        f = TimeSeries.from_times_and_values(fidx, rng.normal(size=(N_STEPS + 7, 2)), columns=["w1", "w2"])
        future.append(f.with_static_covariates(_statics(i)))
    return targets, past, future


def _build(name: str, experiment: str, overrides: dict[str, Any], ctx: RunContext) -> Any:
    spec = get_spec(name, experiment)
    return spec.build({**dict(spec.defaults), **overrides}, ctx)


@pytest.mark.parametrize(("name", "experiment", "overrides", "binary"), CASES)
def test_builder_fits_exactly_the_selected_pairs(name, experiment, overrides, binary):
    spec = get_spec(name, experiment)
    ctx = RunContext(seed=42, device=spec.device, threads=1, past_lags=PAST_LAGS)
    model = _build(name, experiment, overrides, ctx)
    targets, past, future = _panel(binary)
    keep = [comp for comp, _ in PAST_LAGS]
    past = [ts[keep] for ts in past]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(targets, past_covariates=past, future_covariates=future)
    names = list(model.lagged_feature_names)
    pastcov = {n for n in names if "_pastcov_" in n}
    assert pastcov == EXPECTED
    assert not any(n.startswith("b_") for n in names)
    # target lags, future covariates, encoders and statics still ride along
    assert sum("_target_lag" in n for n in names) == 7
    assert any(n.startswith("w1_futcov") for n in names)
    assert any("_cyc_" in n for n in names)
    assert any("statcov" in n for n in names)


@pytest.mark.parametrize(("name", "experiment", "overrides", "binary"), CASES)
def test_builder_without_past_lags_is_legacy(name, experiment, overrides, binary):
    spec = get_spec(name, experiment)
    ctx = RunContext(seed=42, device=spec.device, threads=1)
    model = _build(name, experiment, overrides, ctx)
    assert model.model_params["lags_past_covariates"] == [-1, -7, -14]
    assert model.model_params["lags_future_covariates"] == (2, 7)


def test_darts_common_kwargs_default_is_legacy():
    assert darts_common_kwargs(RunContext()) == legacy_common_kwargs()


def test_darts_common_kwargs_plain_sorted_int_lists():
    lags = past_lags_from_mapping({"a": [np.int64(-14), np.int64(-7)], "c": (-1,)})
    assert lags == (("a", (-14, -7)), ("c", (-1,)))
    kw = darts_common_kwargs(RunContext(past_lags=lags))
    assert kw["lags_past_covariates"] == {"a": [-14, -7], "c": [-1]}
    for v in kw["lags_past_covariates"].values():
        assert type(v) is list and all(type(x) is int for x in v)
    rest = {k: v for k, v in kw.items() if k != "lags_past_covariates"}
    legacy = legacy_common_kwargs()
    del legacy["lags_past_covariates"]
    assert rest == legacy
    assert kw["lags_future_covariates"] == (2, 7)


def test_for_head_semantics():
    reg = (("x", (-1,)),)
    ctx = RunContext(
        seed=3,
        threads=2,
        past_lags=(("z", (-7,)),),
        head_past_lags=(("classifier", PAST_LAGS), ("regressor", reg)),
    )
    hash(ctx)  # stays hashable
    c = ctx.for_head("classifier")
    assert c.past_lags == PAST_LAGS and c.head_past_lags == ()
    assert (c.seed, c.threads, c.device) == (3, 2, "cpu")
    assert ctx.for_head("regressor").past_lags == reg
    missing = ctx.for_head("other")
    assert missing.past_lags is None and missing.head_past_lags == ()
    assert RunContext().for_head("classifier") == RunContext()


def test_hurdle_builder_routes_per_head_lags():
    reg = (("x", (-14, -1)),)
    ctx = RunContext(
        seed=42, threads=1, head_past_lags=(("classifier", PAST_LAGS), ("regressor", reg))
    )
    spec = get_spec("hurdle")
    clf_builder, reg_builder = spec.build(spec.defaults, ctx)
    assert clf_builder().model_params["lags_past_covariates"] == {"a": [-14, -7], "c": [-1]}
    assert reg_builder().model_params["lags_past_covariates"] == {"x": [-14, -1]}
    clf_builder, reg_builder = spec.build(spec.defaults, RunContext(seed=42, threads=1))
    assert clf_builder().model_params["lags_past_covariates"] == [-1, -7, -14]
    assert reg_builder().model_params["lags_past_covariates"] == [-1, -7, -14]
