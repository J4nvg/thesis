"""Unit tests for ``strikecast.models.adapters``.

The global adapter is checked against a VERBATIM inline copy of the legacy loop
bodies (``src/evaluation_tools.py::run_expanding_cv`` lines 275-365 and
``run_final_test`` lines 368-450), so a drift in the covariate rule, the
sampling step or the log-link ordering fails the test.

Fixtures are local on purpose: ``tests/conftest.py`` is owned by another work
stream and this file must not depend on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LinearRegressionModel, NaiveMean

from strikecast.backtest.protocols import SINGLE_CHANNEL, Forecaster
from strikecast.models import GlobalDartsForecaster, LocalDartsForecaster

N_DAYS = 90
REGIONS = ["alpha", "beta", "gamma"]
HORIZON = 2
LAGS = 3
START_FRAC = 0.8
STRIDE = 1
RETRAIN_STRIDE = 7


# ---------------------------------------------------------------------------
# Synthetic 3-region dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset() -> dict[str, list[TimeSeries]]:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2022-01-01", periods=N_DAYS, freq="D")
    targets, past, future = [], [], []
    for i, _region in enumerate(REGIONS):
        # Already label-encoded, as `StaticCovariatesTransformer` leaves them.
        statics = pd.DataFrame({"region": [float(i)]})
        targets.append(
            TimeSeries.from_times_and_values(
                dates,
                rng.poisson(2.0 + i, N_DAYS).astype(float).reshape(-1, 1),
                columns=["y"],
                static_covariates=statics,
            )
        )
        past.append(
            TimeSeries.from_times_and_values(
                dates,
                np.column_stack(
                    [rng.normal(10.0 * (i + 1), 2.0, N_DAYS), np.arange(N_DAYS) * 3.0]
                ),
                columns=["p1", "p2"],
            )
        )
        future.append(
            TimeSeries.from_times_and_values(
                dates,
                (np.arange(N_DAYS) % 7).astype(float).reshape(-1, 1) + i,
                columns=["f1"],
            )
        )
    return {"target": targets, "past": past, "future": future}


def make_linear_builder():
    def _build() -> LinearRegressionModel:
        return LinearRegressionModel(
            lags=LAGS,
            lags_past_covariates=[-1, -2],
            lags_future_covariates=(1, 2),
            output_chunk_length=HORIZON,
        )

    return _build


def schedule(n_total: int, horizon: int, start_frac: float, stride: int):
    start_idx = int(start_frac * n_total)
    return start_idx, range(start_idx, n_total - horizon + 1, stride)


# ---------------------------------------------------------------------------
# Verbatim legacy loop bodies (src/evaluation_tools.py). Do not edit.
# ---------------------------------------------------------------------------
def legacy_run_expanding_cv(
    builder_fn,
    target_list,
    start_frac,
    *,
    is_local: bool = False,
    is_neural: bool = False,
    horizon: int,
    stride: int,
    retrain_stride: int | None = None,
    past_covs=None,
    future_covs=None,
):
    if retrain_stride is None:
        retrain_stride = stride

    ref_ts = target_list[0]
    n_total = len(ref_ts)
    start_idx = int(start_frac * n_total)

    n_regions = len(target_list)
    all_fold_preds = [[] for _ in range(n_regions)]
    model = None
    _local_builder = None

    if not is_neural:
        past_for_fit, fut_for_fit = past_covs, future_covs
    else:
        ps = Scaler()
        fs = Scaler()
        past_for_fit, fut_for_fit = ps.fit_transform(past_covs), fs.fit_transform(future_covs)

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        steps_since_start = t0 - start_idx
        split_time = ref_ts.time_index[t0]

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

    return all_fold_preds


def legacy_run_final_test(
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
):
    ref_ts = target_list[0]
    n_total = len(ref_ts)
    start_idx = int(start_frac * n_total)
    n_regions = len(target_list)

    all_fold_preds = [[] for _ in range(n_regions)]
    model = None
    _local_builder = None

    if not is_neural:
        past_for_fit, fut_for_fit = past_covs, future_covs
    else:
        ps = Scaler()
        fs = Scaler()
        past_for_fit, fut_for_fit = ps.fit_transform(past_covs), fs.fit_transform(future_covs)

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

        split_time = ref_ts.time_index[t0]
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

            if getattr(model, "_count_log_link", False):
                preds = [p.map(np.exp) for p in preds]

        for r_idx, pred in enumerate(preds):
            all_fold_preds[r_idx].append(pred)

    return all_fold_preds


# ---------------------------------------------------------------------------
# Minimal hand-rolled engine, matching protocols.py step by step.
# ---------------------------------------------------------------------------
def drive(
    forecaster: Any,
    target_list: list[TimeSeries],
    start_frac: float,
    *,
    horizon: int,
    stride: int,
    retrain_stride: int,
    past_covs=None,
    future_covs=None,
) -> list[list[TimeSeries]]:
    forecaster.prepare(past_covs, future_covs)
    ref_ts = target_list[0]
    n_total = len(ref_ts)
    start_idx, positions = schedule(n_total, horizon, start_frac, stride)

    all_fold_preds: list[list[TimeSeries]] = [[] for _ in target_list]
    for t0 in positions:
        cutoff = ref_ts.time_index[t0]
        if (t0 - start_idx) % retrain_stride == 0 and forecaster.retrains:
            forecaster.fit(
                [ts.drop_after(cutoff) for ts in target_list], cutoff=cutoff
            )
        out = forecaster.predict(
            horizon, [ts.drop_after(cutoff) for ts in target_list], cutoff=cutoff
        )
        for r_idx, pred in enumerate(out[SINGLE_CHANNEL]):
            all_fold_preds[r_idx].append(pred)
    return all_fold_preds


def assert_fold_preds_close(got, expected, atol=1e-9) -> None:
    assert len(got) == len(expected)
    for region_got, region_exp in zip(got, expected, strict=True):
        assert len(region_got) == len(region_exp)
        for a, b in zip(region_got, region_exp, strict=True):
            assert a.time_index.equals(b.time_index)
            np.testing.assert_allclose(a.values(), b.values(), atol=atol)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class StubModel:
    """Records the kwargs it is called with and returns constant series."""

    def __init__(
        self,
        *,
        supports_past: bool = True,
        supports_future: bool = True,
        likelihood: Any = None,
        count_log_link: bool = False,
        value: float = 1.0,
    ) -> None:
        self.supports_past_covariates = supports_past
        self.supports_future_covariates = supports_future
        self.likelihood = likelihood
        if count_log_link:
            self._count_log_link = True
        self.value = value
        self.fit_kwargs: dict[str, Any] | None = None
        self.predict_kwargs: list[dict[str, Any]] = []

    def fit(self, **kwargs: Any) -> StubModel:
        self.fit_kwargs = kwargs
        return self

    def predict(self, **kwargs: Any) -> list[TimeSeries]:
        self.predict_kwargs.append(kwargs)
        series = kwargs["series"]
        n = kwargs["n"]
        out = []
        for ts in series:
            idx = pd.date_range(
                ts.time_index[-1] + ts.freq, periods=n, freq=ts.freq
            )
            values = np.full((n, 1), self.value, dtype=float)
            if kwargs.get("num_samples", 1) > 1:
                # Mimic a sample dimension so `.quantile(0.5)` is exercised.
                values = np.repeat(
                    values[:, :, None], kwargs["num_samples"], axis=2
                )
            out.append(TimeSeries.from_times_and_values(idx, values))
        return out


class BoomModel:
    """Local model whose ``fit`` always raises (drives the F5 fallback)."""

    supports_past_covariates = False
    supports_future_covariates = False

    def fit(self, series: TimeSeries) -> BoomModel:
        raise RuntimeError("boom")

    def predict(self, n: int) -> TimeSeries:  # pragma: no cover - never reached
        raise AssertionError("predict should not be reached")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
def test_adapters_satisfy_the_frozen_forecaster_protocol() -> None:
    g = GlobalDartsForecaster(make_linear_builder())
    local = LocalDartsForecaster(lambda: NaiveMean())
    assert isinstance(g, Forecaster)
    assert isinstance(local, Forecaster)
    assert g.channels == (SINGLE_CHANNEL,) == local.channels
    assert g.retrains is True and local.retrains is True


def test_presets_set_exactly_the_three_legacy_combinations() -> None:
    builder = make_linear_builder()
    cv = GlobalDartsForecaster.for_cv(builder)
    test = GlobalDartsForecaster.for_test(builder)
    tuning = GlobalDartsForecaster.for_tuning(builder)
    assert (cv.sample_median, cv.apply_log_link) == (True, True)
    assert (test.sample_median, test.apply_log_link) == (False, True)
    assert (tuning.sample_median, tuning.apply_log_link) == (True, False)
    # The default constructor must equal one of the three, never a new mix.
    default = GlobalDartsForecaster(builder)
    assert (default.sample_median, default.apply_log_link) == (
        cv.sample_median,
        cv.apply_log_link,
    )


# ---------------------------------------------------------------------------
# GlobalDartsForecaster vs the verbatim legacy loops
# ---------------------------------------------------------------------------
def test_global_cv_matches_legacy_run_expanding_cv(dataset) -> None:
    expected = legacy_run_expanding_cv(
        make_linear_builder(),
        dataset["target"],
        START_FRAC,
        horizon=HORIZON,
        stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=dataset["past"],
        future_covs=dataset["future"],
    )
    got = drive(
        GlobalDartsForecaster.for_cv(make_linear_builder()),
        dataset["target"],
        START_FRAC,
        horizon=HORIZON,
        stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=dataset["past"],
        future_covs=dataset["future"],
    )
    assert len(got[0]) > 1
    assert_fold_preds_close(got, expected)


def test_global_test_matches_legacy_run_final_test(dataset) -> None:
    expected = legacy_run_final_test(
        make_linear_builder(),
        dataset["target"],
        START_FRAC,
        predict_stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        horizon=HORIZON,
        past_covs=dataset["past"],
        future_covs=dataset["future"],
    )
    got = drive(
        GlobalDartsForecaster.for_test(make_linear_builder()),
        dataset["target"],
        START_FRAC,
        horizon=HORIZON,
        stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
        past_covs=dataset["past"],
        future_covs=dataset["future"],
    )
    assert_fold_preds_close(got, expected)


def test_global_matches_legacy_without_covariates(dataset) -> None:
    def builder() -> LinearRegressionModel:
        return LinearRegressionModel(lags=LAGS, output_chunk_length=HORIZON)

    expected = legacy_run_expanding_cv(
        builder,
        dataset["target"],
        START_FRAC,
        horizon=HORIZON,
        stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )
    got = drive(
        GlobalDartsForecaster.for_cv(builder),
        dataset["target"],
        START_FRAC,
        horizon=HORIZON,
        stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )
    assert_fold_preds_close(got, expected)


def test_global_predict_before_fit_raises(dataset) -> None:
    f = GlobalDartsForecaster.for_cv(make_linear_builder())
    f.prepare(dataset["past"], dataset["future"])
    with pytest.raises(RuntimeError, match="before fit"):
        f.predict(HORIZON, dataset["target"], cutoff=dataset["target"][0].time_index[-1])


# ---------------------------------------------------------------------------
# Covariate handling in prepare()
# ---------------------------------------------------------------------------
def test_prepare_non_neural_stores_covariates_unchanged(dataset) -> None:
    f = GlobalDartsForecaster.for_cv(make_linear_builder(), is_neural=False)
    f.prepare(dataset["past"], dataset["future"])
    assert f.past_for_fit is dataset["past"]
    assert f.fut_for_fit is dataset["future"]


def test_prepare_neural_fits_scalers_on_full_length_covariates(dataset) -> None:
    """F1: the ``Scaler`` sees validation and test periods too."""
    f = GlobalDartsForecaster.for_cv(make_linear_builder(), is_neural=True)
    f.prepare(dataset["past"], dataset["future"])

    assert f.past_for_fit is not None and f.fut_for_fit is not None
    for scaled, raw in zip(f.past_for_fit, dataset["past"], strict=True):
        assert len(scaled) == len(raw) == N_DAYS
        vals = scaled.values()
        assert vals.min() >= 0.0 - 1e-12
        assert vals.max() <= 1.0 + 1e-12
        np.testing.assert_allclose(vals.min(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(vals.max(axis=0), 1.0, atol=1e-12)
    for scaled in f.fut_for_fit:
        vals = scaled.values()
        assert vals.min() >= 0.0 - 1e-12 and vals.max() <= 1.0 + 1e-12

    # Two independent Scaler instances: reproduces `_maybe_scale_covs` exactly.
    legacy_past = Scaler().fit_transform(dataset["past"])
    legacy_future = Scaler().fit_transform(dataset["future"])
    for a, b in zip(f.past_for_fit, legacy_past, strict=True):
        np.testing.assert_allclose(a.values(), b.values(), atol=1e-12)
    for a, b in zip(f.fut_for_fit, legacy_future, strict=True):
        np.testing.assert_allclose(a.values(), b.values(), atol=1e-12)


def test_prepare_neural_passes_none_through(dataset) -> None:
    """F53: legacy would raise here; the guard cannot change a recorded run."""
    f = GlobalDartsForecaster.for_cv(make_linear_builder(), is_neural=True)
    f.prepare(None, dataset["future"])
    assert f.past_for_fit is None
    assert f.fut_for_fit is not None


def test_covariates_are_skipped_when_the_model_does_not_support_them(dataset) -> None:
    stub = StubModel(supports_past=False, supports_future=True)
    f = GlobalDartsForecaster.for_cv(lambda: stub)
    f.prepare(dataset["past"], dataset["future"])
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    f.predict(HORIZON, dataset["target"], cutoff=cutoff)

    assert stub.fit_kwargs is not None
    assert "past_covariates" not in stub.fit_kwargs
    assert "future_covariates" in stub.fit_kwargs
    assert "past_covariates" not in stub.predict_kwargs[0]
    assert "future_covariates" in stub.predict_kwargs[0]
    assert stub.predict_kwargs[0]["show_warnings"] is False


# ---------------------------------------------------------------------------
# Log-link (F55) and sampling (F56) switches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("preset", "expect_exp"),
    [("for_cv", True), ("for_test", True), ("for_tuning", False)],
)
def test_count_log_link_exp_applies_per_preset(dataset, preset, expect_exp) -> None:
    stub = StubModel(count_log_link=True, value=2.0)
    f = getattr(GlobalDartsForecaster, preset)(lambda: stub)
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    out = f.predict(HORIZON, dataset["target"], cutoff=cutoff)[SINGLE_CHANNEL]

    expected = np.exp(2.0) if expect_exp else 2.0
    for ts in out:
        np.testing.assert_allclose(ts.values().ravel(), expected, atol=1e-12)


def test_log_link_is_ignored_when_the_flag_is_absent(dataset) -> None:
    stub = StubModel(count_log_link=False, value=2.0)
    f = GlobalDartsForecaster.for_cv(lambda: stub)
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    out = f.predict(HORIZON, dataset["target"], cutoff=cutoff)[SINGLE_CHANNEL]
    for ts in out:
        np.testing.assert_allclose(ts.values().ravel(), 2.0, atol=1e-12)


@pytest.mark.parametrize(
    ("preset", "expect_samples"),
    [("for_cv", True), ("for_tuning", True), ("for_test", False)],
)
def test_num_samples_200_per_preset_for_a_neural_likelihood_model(
    dataset, preset, expect_samples
) -> None:
    stub = StubModel(likelihood=object(), value=3.0)
    f = getattr(GlobalDartsForecaster, preset)(lambda: stub, is_neural=True)
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    out = f.predict(HORIZON, dataset["target"], cutoff=cutoff)[SINGLE_CHANNEL]

    kwargs = stub.predict_kwargs[0]
    assert ("num_samples" in kwargs) is expect_samples
    if expect_samples:
        assert kwargs["num_samples"] == 200
        assert all(ts.n_samples == 1 for ts in out)  # median collapsed the samples
    for ts in out:
        np.testing.assert_allclose(ts.values().ravel(), 3.0, atol=1e-12)


def test_sampling_requires_both_is_neural_and_a_likelihood(dataset) -> None:
    cutoff = dataset["target"][0].time_index[-1]

    no_likelihood = StubModel(likelihood=None)
    f = GlobalDartsForecaster.for_cv(lambda: no_likelihood, is_neural=True)
    f.prepare(None, None)
    f.fit(dataset["target"], cutoff=cutoff)
    f.predict(HORIZON, dataset["target"], cutoff=cutoff)
    assert "num_samples" not in no_likelihood.predict_kwargs[0]

    not_neural = StubModel(likelihood=object())
    g = GlobalDartsForecaster.for_cv(lambda: not_neural, is_neural=False)
    g.prepare(None, None)
    g.fit(dataset["target"], cutoff=cutoff)
    g.predict(HORIZON, dataset["target"], cutoff=cutoff)
    assert "num_samples" not in not_neural.predict_kwargs[0]


def test_num_samples_is_configurable(dataset) -> None:
    stub = StubModel(likelihood=object())
    f = GlobalDartsForecaster.for_cv(lambda: stub, is_neural=True, num_samples=17)
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    f.predict(HORIZON, dataset["target"], cutoff=cutoff)
    assert stub.predict_kwargs[0]["num_samples"] == 17


def test_model_is_rebuilt_on_every_fit(dataset) -> None:
    built: list[StubModel] = []

    def builder() -> StubModel:
        m = StubModel()
        built.append(m)
        return m

    f = GlobalDartsForecaster.for_cv(builder)
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    f.fit(dataset["target"], cutoff=cutoff)
    assert len(built) == 2
    assert f.model is built[-1]


# ---------------------------------------------------------------------------
# LocalDartsForecaster
# ---------------------------------------------------------------------------
def test_local_matches_the_legacy_local_branch(dataset) -> None:
    expected = legacy_run_expanding_cv(
        lambda: NaiveMean(),
        dataset["target"],
        START_FRAC,
        is_local=True,
        horizon=HORIZON,
        stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )
    got = drive(
        LocalDartsForecaster(lambda: NaiveMean()),
        dataset["target"],
        START_FRAC,
        horizon=HORIZON,
        stride=STRIDE,
        retrain_stride=RETRAIN_STRIDE,
    )
    assert_fold_preds_close(got, expected)


def test_local_fit_is_a_no_op_and_predict_refits_every_fold(dataset) -> None:
    built: list[NaiveMean] = []

    def builder() -> NaiveMean:
        m = NaiveMean()
        built.append(m)
        return m

    f = LocalDartsForecaster(builder)
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    assert built == []  # fit only captured the builder
    f.predict(HORIZON, dataset["target"], cutoff=cutoff)
    assert len(built) == len(REGIONS)  # one model per region, per fold
    f.predict(HORIZON, dataset["target"], cutoff=cutoff)
    assert len(built) == 2 * len(REGIONS)


def test_local_fallback_is_used_when_the_builder_model_raises(dataset, caplog) -> None:
    """F5: the diff runners wrap build/fit/predict in ``except Exception``."""
    f = LocalDartsForecaster(lambda: BoomModel(), fallback=lambda: NaiveMean())
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[len(dataset["target"][0]) - 1]
    context = [ts.drop_after(cutoff) for ts in dataset["target"]]
    f.fit(context, cutoff=cutoff)

    with caplog.at_level("WARNING", logger="strikecast.models.adapters"):
        out = f.predict(HORIZON, context, cutoff=cutoff)[SINGLE_CHANNEL]

    assert len(out) == len(REGIONS)
    for r_idx, (pred, ts) in enumerate(zip(out, context, strict=True)):
        expected = NaiveMean().fit(ts).predict(n=HORIZON)
        np.testing.assert_allclose(pred.values(), expected.values(), atol=1e-9)
        assert f"region index {r_idx}" in caplog.text


def test_local_without_a_fallback_propagates(dataset) -> None:
    f = LocalDartsForecaster(lambda: BoomModel())
    f.prepare(None, None)
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    with pytest.raises(RuntimeError, match="boom"):
        f.predict(HORIZON, dataset["target"], cutoff=cutoff)


def test_local_ignores_covariates(dataset) -> None:
    f = LocalDartsForecaster(lambda: NaiveMean())
    f.prepare(dataset["past"], dataset["future"])
    assert f.past_for_fit is dataset["past"]
    assert f.fut_for_fit is dataset["future"]
    cutoff = dataset["target"][0].time_index[-1]
    f.fit(dataset["target"], cutoff=cutoff)
    out = f.predict(HORIZON, dataset["target"], cutoff=cutoff)[SINGLE_CHANNEL]
    assert len(out) == len(REGIONS)
