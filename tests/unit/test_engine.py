"""Unit tests for `strikecast.backtest.engine.ExpandingWindowBacktest`.

No real models and no real transforms: the forecasters here are recording
fakes, and the two transforms are throw-away local copies (the real `Identity`
and `Diff` are written in parallel by another work stream). The last two tests
are the exception -- they wrap darts' `NaiveMean` and `LinearRegressionModel`
in trivial adapters and compare `run()` against the legacy
`src.evaluation_tools.run_expanding_cv` value for value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from darts import TimeSeries  # noqa: E402

from strikecast.backtest.engine import ExpandingWindowBacktest  # noqa: E402
from strikecast.backtest.hooks import CallbackHook  # noqa: E402
from strikecast.backtest.protocols import SINGLE_CHANNEL  # noqa: E402
from strikecast.config.schema import BacktestConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HORIZON = 7
CFG = BacktestConfig(start_frac=0.5, horizon=HORIZON, predict_stride=1, retrain_stride=7)


# --------------------------------------------------------------------------- #
# fixtures / fakes
# --------------------------------------------------------------------------- #
def make_targets(n_regions: int = 3, n: int = 40, seed: int = 0) -> list[TimeSeries]:
    index = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return [
        TimeSeries.from_times_and_values(
            index, (rng.poisson(2.0, n) + 10 * r).astype(float).reshape(-1, 1)
        )
        for r in range(n_regions)
    ]


class RecordingIdentity:
    """Stand-in for the real `Identity` transform."""

    name = "identity"

    def __init__(self) -> None:
        self.forward_calls: list[list[TimeSeries]] = []
        self.inverse_calls: list[tuple[TimeSeries, TimeSeries]] = []

    def forward(self, level_series: list[TimeSeries]) -> list[TimeSeries]:
        self.forward_calls.append(list(level_series))
        return list(level_series)

    def inverse(self, pred: TimeSeries, context: TimeSeries) -> TimeSeries:
        self.inverse_calls.append((pred, context))
        return pred


class TinyDiff:
    """Stand-in for the real `Diff` transform, mirroring `_diff_to_level`."""

    name = "diff"

    def __init__(self) -> None:
        self.forward_calls = 0
        self.inverse_contexts: list[TimeSeries] = []

    def forward(self, level_series: list[TimeSeries]) -> list[TimeSeries]:
        self.forward_calls += 1
        return [
            TimeSeries.from_times_and_values(
                ts.time_index[1:], np.diff(ts.values().ravel()).reshape(-1, 1)
            )
            for ts in level_series
        ]

    def inverse(self, pred: TimeSeries, context: TimeSeries) -> TimeSeries:
        self.inverse_contexts.append(context)
        anchor_idx = context.time_index.get_loc(pred.time_index[0]) - 1
        anchor = float(context.values()[anchor_idx, 0])
        return TimeSeries.from_times_and_values(
            pred.time_index, (anchor + np.cumsum(pred.values().ravel())).reshape(-1, 1)
        )


class RecordingForecaster:
    """Records every protocol call and returns deterministic, labelled series."""

    def __init__(self, channels: tuple[str, ...] = (SINGLE_CHANNEL,), retrains: bool = True):
        self.channels = channels
        self.retrains = retrains
        self.prepare_calls: list[tuple] = []
        self.fit_calls: list[dict] = []
        self.predict_calls: list[dict] = []

    def prepare(self, past_covs, future_covs) -> None:
        self.prepare_calls.append((past_covs, future_covs))

    def fit(self, train_series: list[TimeSeries], *, cutoff: pd.Timestamp) -> None:
        self.fit_calls.append(
            {
                "cutoff": cutoff,
                "ends": [ts.end_time() for ts in train_series],
                "lens": [len(ts) for ts in train_series],
                "series": train_series,
            }
        )

    def predict(self, n: int, context_series: list[TimeSeries], *, cutoff: pd.Timestamp):
        self.predict_calls.append(
            {
                "n": n,
                "cutoff": cutoff,
                "ends": [ts.end_time() for ts in context_series],
                "lens": [len(ts) for ts in context_series],
                "series": context_series,
            }
        )
        index = pd.date_range(cutoff, periods=n, freq="D")
        return {
            channel: [
                TimeSeries.from_times_and_values(
                    index, np.full((n, 1), 1000.0 * c_idx + r_idx, dtype=float)
                )
                for r_idx in range(len(context_series))
            ]
            for c_idx, channel in enumerate(self.channels)
        }


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def test_forward_and_prepare_run_once_before_the_loop() -> None:
    targets = make_targets()
    transform, forecaster = RecordingIdentity(), RecordingForecaster()
    past, future = ["past"], ["future"]

    ExpandingWindowBacktest(CFG, transform).run(forecaster, targets, past, future)  # type: ignore[arg-type]

    assert len(transform.forward_calls) == 1
    assert transform.forward_calls[0] == targets
    assert forecaster.prepare_calls == [(past, future)]


def test_iter_folds_is_lazy() -> None:
    """Nothing runs until the generator is advanced -- as `run_expanding_cv_iter`."""
    transform, forecaster = RecordingIdentity(), RecordingForecaster()
    it = ExpandingWindowBacktest(CFG, transform).iter_folds(forecaster, make_targets())

    assert transform.forward_calls == []
    assert forecaster.prepare_calls == []

    next(it)
    assert len(forecaster.prepare_calls) == 1
    assert len(forecaster.predict_calls) == 1


def test_works_without_covariates() -> None:
    forecaster = RecordingForecaster()
    out = ExpandingWindowBacktest(CFG, RecordingIdentity()).run(forecaster, make_targets())

    assert forecaster.prepare_calls == [(None, None)]
    assert len(out[SINGLE_CHANNEL][0]) == len(forecaster.predict_calls)


# --------------------------------------------------------------------------- #
# fit / predict cadence
# --------------------------------------------------------------------------- #
def test_fit_only_on_retrain_folds_predict_on_every_fold() -> None:
    targets = make_targets(n=40)  # start_idx 20, folds t0=20..33
    forecaster = RecordingForecaster()

    ExpandingWindowBacktest(CFG, RecordingIdentity()).run(forecaster, targets)

    assert len(forecaster.predict_calls) == 14
    assert len(forecaster.fit_calls) == 2
    index = targets[0].time_index
    assert [c["cutoff"] for c in forecaster.fit_calls] == [index[20], index[27]]
    assert [c["cutoff"] for c in forecaster.predict_calls] == list(index[20:34])
    assert all(c["n"] == HORIZON for c in forecaster.predict_calls)


def test_retrains_false_never_calls_fit() -> None:
    """The Chronos-2 shape: a fixed predictor, even with retrain_stride set."""
    forecaster = RecordingForecaster(retrains=False)
    ExpandingWindowBacktest(CFG, RecordingIdentity()).run(forecaster, make_targets())

    assert forecaster.fit_calls == []
    assert len(forecaster.predict_calls) == 14


def test_retrain_stride_none_never_calls_fit() -> None:
    cfg = BacktestConfig(start_frac=0.5, horizon=HORIZON, retrain_stride=None)
    forecaster = RecordingForecaster()
    ExpandingWindowBacktest(cfg, RecordingIdentity()).run(forecaster, make_targets())

    assert forecaster.fit_calls == []
    assert len(forecaster.predict_calls) == 14


# --------------------------------------------------------------------------- #
# slicing
# --------------------------------------------------------------------------- #
def test_slices_end_one_step_before_the_cutoff() -> None:
    """darts 0.43 `drop_after(cutoff)` drops the cutoff itself."""
    targets = make_targets(n=40)
    index = targets[0].time_index
    forecaster = RecordingForecaster()

    ExpandingWindowBacktest(CFG, RecordingIdentity()).run(forecaster, targets)

    for call in forecaster.predict_calls:
        t0 = index.get_loc(call["cutoff"])
        assert call["ends"] == [index[t0 - 1]] * 3
        assert call["lens"] == [t0] * 3
    for call in forecaster.fit_calls:
        t0 = index.get_loc(call["cutoff"])
        assert call["ends"] == [index[t0 - 1]] * 3


def test_retrain_slice_and_context_slice_are_separate_objects() -> None:
    """`run_expanding_cv` computes both `drop_after(split_time)` lists separately."""
    targets = make_targets(n=40)
    forecaster = RecordingForecaster()

    ExpandingWindowBacktest(CFG, RecordingIdentity()).run(forecaster, targets)

    fit_call = forecaster.fit_calls[0]
    predict_call = forecaster.predict_calls[0]
    assert fit_call["cutoff"] == predict_call["cutoff"]
    for a, b in zip(fit_call["series"], predict_call["series"], strict=True):
        assert a is not b
        np.testing.assert_array_equal(a.values(), b.values())
        assert a.time_index.equals(b.time_index)


def test_region_order_is_preserved() -> None:
    targets = make_targets(n_regions=4, n=40)
    out = ExpandingWindowBacktest(CFG, RecordingIdentity()).run(RecordingForecaster(), targets)

    preds = out[SINGLE_CHANNEL]
    assert len(preds) == 4
    for r_idx, region_folds in enumerate(preds):
        for pred in region_folds:
            assert float(pred.values()[0, 0]) == float(r_idx)


# --------------------------------------------------------------------------- #
# channels / output shape
# --------------------------------------------------------------------------- #
def test_channels_are_preserved() -> None:
    """The hurdle shape: three channels, each a region-ordered list per fold."""
    channels = ("prob", "count", "hurdle")
    forecaster = RecordingForecaster(channels=channels)
    out = ExpandingWindowBacktest(CFG, RecordingIdentity()).run(forecaster, make_targets())

    assert tuple(out) == channels
    for c_idx, channel in enumerate(channels):
        assert len(out[channel]) == 3
        assert len(out[channel][0]) == 14
        assert float(out[channel][2][0].values()[0, 0]) == 1000.0 * c_idx + 2


def test_run_output_is_the_legacy_shape() -> None:
    """outer = region, inner = fold; `collect_predictions_long` consumes this."""
    targets = make_targets(n_regions=3, n=40)
    out = ExpandingWindowBacktest(CFG, RecordingIdentity()).run(RecordingForecaster(), targets)

    preds = out[SINGLE_CHANNEL]
    assert [len(region_folds) for region_folds in preds] == [14, 14, 14]
    index = targets[0].time_index
    assert preds[0][0].time_index[0] == index[20]
    assert len(preds[0][0]) == HORIZON


def test_channel_mismatch_is_rejected() -> None:
    class WrongChannels(RecordingForecaster):
        def predict(self, n, context_series, *, cutoff):
            out = super().predict(n, context_series, cutoff=cutoff)
            return {"unexpected": out[SINGLE_CHANNEL]}

    with pytest.raises(ValueError, match="channels"):
        ExpandingWindowBacktest(CFG, RecordingIdentity()).run(WrongChannels(), make_targets())


def test_region_count_mismatch_is_rejected() -> None:
    class DropsARegion(RecordingForecaster):
        def predict(self, n, context_series, *, cutoff):
            out = super().predict(n, context_series, cutoff=cutoff)
            return {SINGLE_CHANNEL: out[SINGLE_CHANNEL][:-1]}

    with pytest.raises(ValueError, match="regions"):
        ExpandingWindowBacktest(CFG, RecordingIdentity()).run(DropsARegion(), make_targets())


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #
def test_hook_receives_cumulative_in_the_legacy_shape() -> None:
    seen: list[tuple[int, list[int]]] = []

    def record(result, cumulative):
        bundle = cumulative[SINGLE_CHANNEL]
        seen.append((result.fold.index, [len(region_folds) for region_folds in bundle]))

    engine = ExpandingWindowBacktest(CFG, RecordingIdentity(), hooks=[CallbackHook(record)])
    engine.run(RecordingForecaster(), make_targets())

    assert len(seen) == 14
    # cumulative grows by one fold per region per call, and the fold just
    # predicted is already in it (as `run_expanding_cv_iter` yields it)
    assert seen == [(i, [i + 1, i + 1, i + 1]) for i in range(14)]


def test_hook_snapshot_does_not_alias_engine_state() -> None:
    snapshots: list[dict] = []

    def grab(result, cumulative):
        snapshots.append(cumulative)
        cumulative[SINGLE_CHANNEL][0].clear()  # a hook must not be able to corrupt the run

    engine = ExpandingWindowBacktest(CFG, RecordingIdentity(), hooks=[CallbackHook(grab)])
    out = engine.run(RecordingForecaster(), make_targets())

    assert len(out[SINGLE_CHANNEL][0]) == 14
    assert snapshots[0] is not snapshots[1]


def test_hooks_run_in_order() -> None:
    order: list[str] = []
    hooks = [
        CallbackHook(lambda r, c: order.append(f"a{r.fold.index}")),
        CallbackHook(lambda r, c: order.append(f"b{r.fold.index}")),
    ]
    engine = ExpandingWindowBacktest(CFG, RecordingIdentity(), hooks=hooks)
    next(engine.iter_folds(RecordingForecaster(), make_targets()))

    assert order == ["a0", "b0"]


def test_hook_exception_aborts_the_run() -> None:
    class Boom(Exception):
        pass

    def explode(result, cumulative):
        if result.fold.index == 3:
            raise Boom

    forecaster = RecordingForecaster()
    engine = ExpandingWindowBacktest(CFG, RecordingIdentity(), hooks=[CallbackHook(explode)])

    with pytest.raises(Boom):
        engine.run(forecaster, make_targets())

    assert len(forecaster.predict_calls) == 4  # folds 0..3, then the abort


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #
def test_inverse_is_called_per_region_per_channel_with_the_full_level_series() -> None:
    targets = make_targets(n_regions=3, n=40)
    transform = RecordingIdentity()
    forecaster = RecordingForecaster(channels=("prob", "count"))

    ExpandingWindowBacktest(CFG, transform).run(forecaster, targets)

    assert len(transform.inverse_calls) == 14 * 2 * 3
    for _pred, context in transform.inverse_calls:
        assert any(context is ts for ts in targets)
        assert len(context) == 40


def test_diff_transform_schedules_on_model_space_and_inverts_on_level_space() -> None:
    """The diff branch: the model series is one step shorter than the level one."""
    targets = make_targets(n_regions=2, n=40)
    transform = TinyDiff()
    forecaster = RecordingForecaster()

    out = ExpandingWindowBacktest(CFG, transform).run(forecaster, targets)

    # schedule lives on the 39-step diffed reference: start_idx = int(0.5 * 39) = 19
    assert transform.forward_calls == 1
    diffed_index = targets[0].time_index[1:]
    assert forecaster.predict_calls[0]["cutoff"] == diffed_index[19]
    assert len(forecaster.predict_calls) == len(range(19, 39 - HORIZON + 1))

    # inverse always sees the FULL level series of that region, never a slice
    assert len(transform.inverse_contexts) == len(forecaster.predict_calls) * 2
    for context in transform.inverse_contexts:
        assert len(context) == 40
        assert any(context is ts for ts in targets)

    # and the level prediction is anchored on the last actual before the window
    first = out[SINGLE_CHANNEL][0][0]
    anchor_time = first.time_index[0] - pd.Timedelta(days=1)
    anchor = float(targets[0].values()[targets[0].time_index.get_loc(anchor_time), 0])
    np.testing.assert_allclose(first.values()[0, 0], anchor + 0.0, rtol=0, atol=1e-12)


def test_empty_target_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        ExpandingWindowBacktest(CFG, RecordingIdentity()).run(RecordingForecaster(), [])


# --------------------------------------------------------------------------- #
# hooks.py
# --------------------------------------------------------------------------- #
def test_progress_hook_logs_every_k_folds(caplog) -> None:
    from strikecast.backtest.hooks import ProgressHook  # noqa: PLC0415

    engine = ExpandingWindowBacktest(CFG, RecordingIdentity(), hooks=[ProgressHook(every=7)])
    with caplog.at_level("INFO", logger="strikecast.backtest.hooks"):
        engine.run(RecordingForecaster(), make_targets())

    lines = [r for r in caplog.records if r.name == "strikecast.backtest.hooks"]
    assert len(lines) == 2  # folds 0 and 7 of 14


class FakeTrial:
    """Only what `PruningHook` touches: `report` and `should_prune`."""

    def __init__(self, prune_at: int | None = None):
        self.reports: list[tuple[float, int]] = []
        self.prune_at = prune_at

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.prune_at is not None and len(self.reports) > self.prune_at


def test_pruning_hook_reports_every_fold_with_the_fold_index_as_step() -> None:
    from strikecast.backtest.hooks import PruningHook  # noqa: PLC0415

    trial = FakeTrial()
    # the legacy `_score_fold_preds` scores the CUMULATIVE bundle
    hook = PruningHook(lambda cum: float(len(cum[SINGLE_CHANNEL][0])), trial, "RMSSE_mean")

    ExpandingWindowBacktest(CFG, RecordingIdentity(), hooks=[hook]).run(
        RecordingForecaster(), make_targets()
    )

    assert trial.reports == [(float(i + 1), i) for i in range(14)]
    assert hook.last_value == 14.0


def test_pruning_hook_raises_trial_pruned_and_stops_the_run() -> None:
    optuna = pytest.importorskip("optuna")
    from strikecast.backtest.hooks import PruningHook  # noqa: PLC0415

    trial = FakeTrial(prune_at=3)
    hook = PruningHook(lambda cum: 1.0, trial)
    forecaster = RecordingForecaster()

    with pytest.raises(optuna.TrialPruned):
        ExpandingWindowBacktest(CFG, RecordingIdentity(), hooks=[hook]).run(
            forecaster, make_targets()
        )

    assert len(trial.reports) == 4
    assert len(forecaster.predict_calls) == 4


# --------------------------------------------------------------------------- #
# level D: equality with the legacy runner
# --------------------------------------------------------------------------- #
class LocalForecaster:
    """Legacy `is_local=True`: fit captures nothing, predict builds one model per series."""

    channels = (SINGLE_CHANNEL,)
    retrains = True

    def __init__(self, builder):
        self.builder = builder

    def prepare(self, past_covs, future_covs) -> None:
        pass

    def fit(self, train_series, *, cutoff) -> None:
        pass

    def predict(self, n, context_series, *, cutoff):
        preds = []
        for ts in context_series:
            model = self.builder()
            model.fit(ts)
            preds.append(model.predict(n=n))
        return {SINGLE_CHANNEL: preds}


class GlobalForecaster:
    """Legacy `is_local=False`: one model, rebuilt and refit at every retrain."""

    channels = (SINGLE_CHANNEL,)
    retrains = True

    def __init__(self, builder):
        self.builder = builder
        self.model = None

    def prepare(self, past_covs, future_covs) -> None:
        pass

    def fit(self, train_series, *, cutoff) -> None:
        self.model = self.builder()
        self.model.fit(series=train_series)

    def predict(self, n, context_series, *, cutoff):
        assert self.model is not None
        preds = self.model.predict(show_warnings=False, n=n, series=context_series)
        return {SINGLE_CHANNEL: list(preds)}


def assert_fold_preds_equal(new, legacy) -> None:
    assert len(new) == len(legacy)
    for r_idx, (new_region, legacy_region) in enumerate(zip(new, legacy, strict=True)):
        assert len(new_region) == len(legacy_region), f"region {r_idx}: fold count"
        for f_idx, (a, b) in enumerate(zip(new_region, legacy_region, strict=True)):
            assert a.time_index.equals(b.time_index), f"region {r_idx} fold {f_idx}: index"
            np.testing.assert_allclose(
                a.values(), b.values(), rtol=0, atol=1e-9,
                err_msg=f"region {r_idx} fold {f_idx}",
            )


@pytest.fixture(scope="module")
def legacy_runner():
    from src.evaluation_tools import run_expanding_cv  # noqa: PLC0415

    return run_expanding_cv


def test_run_matches_legacy_run_expanding_cv_local_naive_mean(legacy_runner) -> None:
    from darts.models import NaiveMean  # noqa: PLC0415

    targets = make_targets(n_regions=3, n=60, seed=7)
    cfg = BacktestConfig(start_frac=0.5, horizon=HORIZON, predict_stride=1, retrain_stride=7)

    new = ExpandingWindowBacktest(cfg, RecordingIdentity()).run(
        LocalForecaster(NaiveMean), targets
    )
    legacy = legacy_runner(
        NaiveMean, targets, 0.5,
        is_local=True, horizon=HORIZON, stride=1, retrain_stride=7, verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], legacy)


def test_run_matches_legacy_run_expanding_cv_global_linear(legacy_runner) -> None:
    from darts.models import LinearRegressionModel  # noqa: PLC0415

    targets = make_targets(n_regions=3, n=60, seed=11)
    cfg = BacktestConfig(start_frac=0.5, horizon=HORIZON, predict_stride=1, retrain_stride=7)

    def builder():
        return LinearRegressionModel(lags=3)

    new = ExpandingWindowBacktest(cfg, RecordingIdentity()).run(
        GlobalForecaster(builder), targets
    )
    legacy = legacy_runner(
        builder, targets, 0.5,
        is_local=False, horizon=HORIZON, stride=1, retrain_stride=7, verbose=False,
    )

    assert_fold_preds_equal(new[SINGLE_CHANNEL], legacy)
    assert len(new[SINGLE_CHANNEL][0]) == 24  # t0 = int(0.5*60) = 30 .. 53 inclusive
