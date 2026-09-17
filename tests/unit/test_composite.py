"""Equivalence tests for the composite forecasters.

``strikecast.models.composite`` is driven by hand over the same schedule the
legacy loops walk, with the SAME stub builders, and the per-fold channel outputs
are compared against the verbatim copies in ``tests/legacy_ref``:

* :class:`HurdleForecaster` vs ``hurdle_runners.run_hurdle_cv`` (cell 15),
  ``run_hurdle_test`` (cell 24) and ``run_hurdle_cv_per_region`` (cell 34,
  including the ``MIN_POSITIVE_SAMPLES`` dummy fallback);
* :class:`MultiTargetClassifierForecaster` vs ``damage_runners.run_damage_cv``
  (cell 17) and ``run_damage_test`` (cell 26).

The stubs stand in for darts models: they record every ``fit`` / ``predict``
kwarg and return deterministic values that depend on the fitted state, the
context and the sample weights, so a difference in the retrain schedule, in the
slicing or in which covariate list reaches which head shows up as a numeric
difference. The classifier stub returns TWO components, so the legacy
"last likelihood component" rule is actually exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import shim
    sys.path.insert(0, str(REPO_ROOT))

from tests.legacy_ref import damage_runners, hurdle_runners  # noqa: E402

from strikecast.backtest import protocols  # noqa: E402
from strikecast.models.composite import (  # noqa: E402
    LEGACY_MIN_POSITIVE_SAMPLES,
    HurdleForecaster,
    MultiTargetClassifierForecaster,
)

HORIZON = 7
START = pd.Timestamp("2022-01-01")
FREQ = "D"


# --------------------------------------------------------------------------- #
# stub darts models
# --------------------------------------------------------------------------- #
class _StubBase:
    """Records calls; deterministic, pure-function predictions."""

    n_components_out = 1

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.fit_calls: list[dict] = []
        self.predict_calls: list[dict] = []
        self._fit_len: float = 0.0
        self._fit_sum: float = 0.0
        self._weight_sum: float = 0.0

    def fit(self, series, past_covariates=None, future_covariates=None, sample_weight=None):
        self.fit_calls.append(
            {
                "series": series,
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
                "sample_weight": sample_weight,
            }
        )
        self._fit_len = float(sum(len(s) for s in series))
        self._fit_sum = float(sum(float(s.values().sum()) for s in series))
        if sample_weight is not None:
            self._weight_sum = float(sum(float(w.values().sum()) for w in sample_weight))
        return self

    def _future_index(self, s: TimeSeries, n: int) -> pd.DatetimeIndex:
        return pd.date_range(start=s.end_time() + s.freq, periods=n, freq=s.freq)

    def _signal(self, s: TimeSeries) -> float:
        vals = s.values().ravel()
        tail = float(vals[-5:].sum())
        return tail + 0.001 * self._fit_len + 0.01 * self._fit_sum + 0.1 * self._weight_sum

    def predict(
        self,
        n,
        series,
        past_covariates=None,
        future_covariates=None,
        predict_likelihood_parameters=False,
        show_warnings=True,
        **kwargs,
    ):
        self.predict_calls.append(
            {
                "n": n,
                "series": series,
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
                "predict_likelihood_parameters": predict_likelihood_parameters,
                "show_warnings": show_warnings,
                **kwargs,
            }
        )
        out = []
        for s_idx, s in enumerate(series):
            idx = self._future_index(s, n)
            base = self._signal(s) + 7.0 * s_idx
            steps = np.arange(n, dtype=float)
            out.append(self._make(idx, base, steps))
        return out

    def _make(self, idx, base, steps) -> TimeSeries:  # pragma: no cover - overridden
        raise NotImplementedError


class StubClassifier(_StubBase):
    """Two output components; the LAST one is the probability."""

    n_components_out = 2

    def _make(self, idx, base, steps) -> TimeSeries:
        n = len(steps)
        other = np.full(n, -1.0) - steps
        prob = 1.0 / (1.0 + np.exp(-(0.01 * base + 0.05 * steps)))
        return TimeSeries.from_times_and_values(
            idx, np.column_stack([other, prob]), columns=["not_the_prob", "p"]
        )


class StubRegressor(_StubBase):
    def _make(self, idx, base, steps) -> TimeSeries:
        vals = 0.5 + 0.1 * base + 0.25 * steps
        return TimeSeries.from_times_and_values(idx, vals.reshape(-1, 1), columns=["count"])


class Builders:
    """Zero-arg builders that hand out fresh stubs and keep every instance."""

    def __init__(self) -> None:
        self.clfs: list[StubClassifier] = []
        self.regs: list[StubRegressor] = []

    def classifier(self) -> StubClassifier:
        m = StubClassifier(f"clf{len(self.clfs)}")
        self.clfs.append(m)
        return m

    def regressor(self) -> StubRegressor:
        m = StubRegressor(f"reg{len(self.regs)}")
        self.regs.append(m)
        return m


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def _series(values: np.ndarray, name: str) -> TimeSeries:
    idx = pd.date_range(START, periods=len(values), freq=FREQ)
    return TimeSeries.from_times_and_values(idx, values.reshape(-1, 1), columns=[name])


def make_dataset(n_days: int = 60, n_regions: int = 3, seed: int = 0, dense: bool = False):
    """3 regions, ``n_days`` days: counts, binaries, weights and 4 covariate lists."""
    rng = np.random.default_rng(seed)
    counts, binaries = [], []
    for r in range(n_regions):
        lam = (2.0 if dense else 0.35) * (r + 1)
        vals = rng.poisson(lam, size=n_days).astype(float)
        counts.append(_series(vals, f"count_r{r}"))
        binaries.append(_series((vals > 0).astype(float), f"event_r{r}"))

    weights = hurdle_runners.make_positive_only_weights(counts)

    def covs(prefix: str, scale: float) -> list[TimeSeries]:
        return [
            _series(scale * rng.normal(size=n_days) + r, f"{prefix}_r{r}")
            for r in range(n_regions)
        ]

    return {
        "binary": binaries,
        "count": counts,
        "weights": weights,
        "clf_past": covs("cp", 1.0),
        "clf_future": covs("cf", 2.0),
        "reg_past": covs("rp", 3.0),
        "reg_future": covs("rf", 4.0),
        "region_names": [f"r{r}" for r in range(n_regions)],
    }


def split_three(series_list: list[TimeSeries], train_frac=0.7, val_frac=0.1):
    """Crude 70/10/20 split, only to feed the test-loop signature."""
    n = len(series_list[0])
    i1 = int(train_frac * n)
    i2 = int((train_frac + val_frac) * n)
    train = [ts[:i1] for ts in series_list]
    val = [ts[i1:i2] for ts in series_list]
    test = [ts[i2:] for ts in series_list]
    return train, val, test


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
def assert_ts_equal(actual: TimeSeries, expected: TimeSeries, where: str) -> None:
    assert actual.time_index.equals(expected.time_index), f"{where}: time index differs"
    assert actual.n_components == expected.n_components, f"{where}: n_components differs"
    np.testing.assert_allclose(
        actual.values(), expected.values(), rtol=0.0, atol=1e-12, err_msg=where
    )


def assert_channel_equal(actual, expected, channel: str) -> None:
    assert len(actual) == len(expected), f"{channel}: region count differs"
    for r_idx, (a_region, e_region) in enumerate(zip(actual, expected, strict=True)):
        assert len(a_region) == len(e_region), f"{channel} r{r_idx}: fold count differs"
        for f_idx, (a, e) in enumerate(zip(a_region, e_region, strict=True)):
            assert_ts_equal(a, e, f"{channel} region={r_idx} fold={f_idx}")


def schedule(reference: TimeSeries, start_frac: float, horizon: int, retrain_stride: int):
    """The legacy schedule: ``(cutoff, retrain)`` per fold, predict stride 1."""
    n_total = len(reference)
    start_idx = int(start_frac * n_total)
    return [
        (reference.time_index[t0], (t0 - start_idx) % retrain_stride == 0)
        for t0 in range(start_idx, n_total - horizon + 1, 1)
    ]


def drive(forecaster, targets: list[TimeSeries], folds, horizon: int = HORIZON):
    """Drive prepare/fit/predict by hand, the way the engine does."""
    forecaster.prepare(None, None)
    collected = {channel: [[] for _ in targets] for channel in forecaster.channels}
    for cutoff, retrain in folds:
        if retrain and forecaster.retrains:
            forecaster.fit([ts.drop_after(cutoff) for ts in targets], cutoff=cutoff)
        preds = forecaster.predict(
            horizon, [ts.drop_after(cutoff) for ts in targets], cutoff=cutoff
        )
        for channel, region_preds in preds.items():
            for r_idx, pred in enumerate(region_preds):
                collected[channel][r_idx].append(pred)
    return collected


# --------------------------------------------------------------------------- #
# protocol conformance
# --------------------------------------------------------------------------- #
def test_hurdle_satisfies_forecaster_protocol():
    data = make_dataset()
    f = HurdleForecaster(
        StubClassifier,
        StubRegressor,
        binary_targets=data["binary"],
        count_targets=data["count"],
        weights=data["weights"],
        clf_past=data["clf_past"],
        clf_future=data["clf_future"],
        reg_past=data["reg_past"],
        reg_future=data["reg_future"],
    )
    assert isinstance(f, protocols.Forecaster)
    assert f.channels == ("prob", "count", "hurdle")
    assert f.retrains is True


def test_multitarget_satisfies_forecaster_protocol():
    data = make_dataset()
    keys = ["b", "a"]  # channel order follows dict order, not sorted order
    f = MultiTargetClassifierForecaster(
        StubClassifier,
        {k: data["binary"] for k in keys},
        {k: data["clf_past"] for k in keys},
        {k: data["clf_future"] for k in keys},
    )
    assert isinstance(f, protocols.Forecaster)
    assert f.channels == ("b", "a")
    assert f.retrains is True


# --------------------------------------------------------------------------- #
# hurdle: global CV (final_hurdle.ipynb cell 15)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("retrain_stride", [7, 3])
def test_hurdle_matches_legacy_cv(retrain_stride):
    data = make_dataset()
    cv_start = 0.7

    legacy_b = Builders()
    exp_c, exp_r, exp_h = hurdle_runners.run_hurdle_cv(
        1,
        retrain_stride,
        target_for_cv_c=data["binary"],
        target_for_cv_r=data["count"],
        full_weights=data["weights"],
        full_past_covs_c=data["clf_past"],
        full_fut_covs_c=data["clf_future"],
        full_past_covs_r=data["reg_past"],
        full_fut_covs_r=data["reg_future"],
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        CV_START_VAL=cv_start,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_b = Builders()
    forecaster = HurdleForecaster(
        new_b.classifier,
        new_b.regressor,
        binary_targets=data["binary"],
        weights=data["weights"],
        clf_past=data["clf_past"],
        clf_future=data["clf_future"],
        reg_past=data["reg_past"],
        reg_future=data["reg_future"],
    )
    folds = schedule(data["binary"][0], cv_start, HORIZON, retrain_stride)
    got = drive(forecaster, data["count"], folds)

    assert len(folds) > 0
    assert_channel_equal(got["prob"], exp_c, "prob")
    assert_channel_equal(got["count"], exp_r, "count")
    assert_channel_equal(got["hurdle"], exp_h, "hurdle")
    assert len(new_b.clfs) == len(legacy_b.clfs) > 1
    assert len(new_b.regs) == len(legacy_b.regs)


def test_hurdle_cv_with_own_count_targets_matches_legacy():
    """``count_targets=`` given: the forecaster slices them itself."""
    data = make_dataset()
    cv_start = 0.7

    legacy_b = Builders()
    exp_c, exp_r, exp_h = hurdle_runners.run_hurdle_cv(
        1,
        HORIZON,
        target_for_cv_c=data["binary"],
        target_for_cv_r=data["count"],
        full_weights=data["weights"],
        full_past_covs_c=data["clf_past"],
        full_fut_covs_c=data["clf_future"],
        full_past_covs_r=data["reg_past"],
        full_fut_covs_r=data["reg_future"],
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        CV_START_VAL=cv_start,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_b = Builders()
    forecaster = HurdleForecaster(
        new_b.classifier,
        new_b.regressor,
        binary_targets=data["binary"],
        count_targets=data["count"],
        weights=data["weights"],
        clf_past=data["clf_past"],
        clf_future=data["clf_future"],
        reg_past=data["reg_past"],
        reg_future=data["reg_future"],
    )
    folds = schedule(data["binary"][0], cv_start, HORIZON, HORIZON)
    # The engine's series are deliberately WRONG here; they must be ignored.
    got = drive(forecaster, data["binary"], folds)

    assert_channel_equal(got["prob"], exp_c, "prob")
    assert_channel_equal(got["count"], exp_r, "count")
    assert_channel_equal(got["hurdle"], exp_h, "hurdle")


def test_hurdle_passes_each_covariate_list_to_the_right_head():
    data = make_dataset()
    b = Builders()
    forecaster = HurdleForecaster(
        b.classifier,
        b.regressor,
        binary_targets=data["binary"],
        weights=data["weights"],
        clf_past=data["clf_past"],
        clf_future=data["clf_future"],
        reg_past=data["reg_past"],
        reg_future=data["reg_future"],
    )
    # prepare() must ignore what the engine hands it (documented behaviour).
    forecaster.prepare(data["reg_past"], data["reg_future"])
    cutoff = data["count"][0].time_index[40]
    forecaster.fit([ts.drop_after(cutoff) for ts in data["count"]], cutoff=cutoff)
    forecaster.predict(HORIZON, [ts.drop_after(cutoff) for ts in data["count"]], cutoff=cutoff)

    clf, reg = b.clfs[0], b.regs[0]
    assert clf.fit_calls[0]["past_covariates"] is data["clf_past"]
    assert clf.fit_calls[0]["future_covariates"] is data["clf_future"]
    assert clf.fit_calls[0]["sample_weight"] is None
    assert reg.fit_calls[0]["past_covariates"] is data["reg_past"]
    assert reg.fit_calls[0]["future_covariates"] is data["reg_future"]
    assert reg.fit_calls[0]["sample_weight"] is not None
    # the classifier trains on the BINARY targets, sliced at the cutoff
    assert set(np.unique(clf.fit_calls[0]["series"][0].values())) <= {0.0, 1.0}
    assert clf.fit_calls[0]["series"][0].end_time() < cutoff
    # likelihood parameters on for the classifier, off for the regressor
    assert clf.predict_calls[0]["predict_likelihood_parameters"] is True
    assert clf.predict_calls[0]["show_warnings"] is False
    assert reg.predict_calls[0]["predict_likelihood_parameters"] is False
    assert reg.predict_calls[0]["show_warnings"] is False


# --------------------------------------------------------------------------- #
# hurdle: final test loop (final_hurdle.ipynb cell 24)
# --------------------------------------------------------------------------- #
def test_hurdle_matches_legacy_test_loop():
    data = make_dataset()
    train_c, val_c, test_c = split_three(data["binary"])
    train_r, val_r, test_r = split_three(data["count"])
    train_val_end = 0.7 + 0.1  # F17: 0.7999999999999999, on purpose

    legacy_b = Builders()
    exp_c, exp_r, exp_h, full_c, full_r = hurdle_runners.run_hurdle_test(
        1,
        HORIZON,
        train_target_c=train_c,
        val_target_c=val_c,
        test_target_c=test_c,
        train_target_r=train_r,
        val_target_r=val_r,
        test_target_r=test_r,
        full_past_covs_c=data["clf_past"],
        full_fut_covs_c=data["clf_future"],
        full_past_covs_r=data["reg_past"],
        full_fut_covs_r=data["reg_future"],
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        TRAIN_VAL_END=train_val_end,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_b = Builders()
    forecaster = HurdleForecaster(
        new_b.classifier,
        new_b.regressor,
        binary_targets=full_c,
        count_targets=full_r,
        weights=hurdle_runners.make_positive_only_weights(full_r),
        clf_past=data["clf_past"],
        clf_future=data["clf_future"],
        reg_past=data["reg_past"],
        reg_future=data["reg_future"],
    )
    folds = schedule(full_c[0], train_val_end, HORIZON, HORIZON)
    got = drive(forecaster, full_r, folds)

    assert len(folds) > 0
    assert_channel_equal(got["prob"], exp_c, "prob")
    assert_channel_equal(got["count"], exp_r, "count")
    assert_channel_equal(got["hurdle"], exp_h, "hurdle")


# --------------------------------------------------------------------------- #
# hurdle: per-region / dummy fallback (final_hurdle.ipynb cell 34)
# --------------------------------------------------------------------------- #
def _per_region_dataset():
    """90 days so that a dense region clears MIN_POSITIVE_SAMPLES=50 and a
    sparse one does not."""
    data = make_dataset(n_days=90, n_regions=3, seed=3, dense=True)
    n = len(data["count"][0])

    # region 0: 30 positive days before the CV start and a positive tail, so the
    # training slice stays under MIN_POSITIVE_SAMPLES=50 but well over any
    # smaller threshold -- this is what makes the test sensitive to the 50.
    medium = np.zeros(n)
    medium[:30] = np.arange(1.0, 31.0)
    medium[63:] = 3.0
    data["count"][0] = _series(medium, "count_r0")
    data["binary"][0] = _series((medium > 0).astype(float), "event_r0")

    # region 1: only 4 positive days in the whole series -> dummy at every retrain
    sparse = np.zeros(n)
    sparse[[3, 11, 29, 47]] = [2.0, 5.0, 1.0, 4.0]
    data["count"][1] = _series(sparse, "count_r1")
    data["binary"][1] = _series((sparse > 0).astype(float), "event_r1")

    # region 2 keeps the dense Poisson draw -> the regressor is always fit
    data["weights"] = hurdle_runners.make_positive_only_weights(data["count"])
    return data


def test_hurdle_per_region_matches_legacy_including_dummy():
    data = _per_region_dataset()
    cv_start = 0.7

    legacy_b = Builders()
    exp_c, exp_r, exp_h = hurdle_runners.run_hurdle_cv_per_region(
        1,
        HORIZON,
        target_for_cv_c=data["binary"],
        target_for_cv_r=data["count"],
        full_weights=data["weights"],
        full_past_covs_c=data["clf_past"],
        full_fut_covs_c=data["clf_future"],
        full_past_covs_r=data["reg_past"],
        full_fut_covs_r=data["reg_future"],
        get_event_classifier=legacy_b.classifier,
        get_count_regressor=legacy_b.regressor,
        region_names=data["region_names"],
        CV_START_VAL=cv_start,
        OUTPUT_CHUNK_LEN=HORIZON,
    )
    # the legacy run must have taken BOTH branches, or the test proves nothing
    assert len(legacy_b.regs) < len(legacy_b.clfs)
    assert len(legacy_b.regs) > 0

    got_c, got_r, got_h = [], [], []
    new_b = Builders()
    for i in range(len(data["count"])):
        forecaster = HurdleForecaster(
            new_b.classifier,
            new_b.regressor,
            binary_targets=[data["binary"][i]],
            count_targets=[data["count"][i]],
            weights=[data["weights"][i]],
            clf_past=[data["clf_past"][i]],
            clf_future=[data["clf_future"][i]],
            reg_past=[data["reg_past"][i]],
            reg_future=[data["reg_future"][i]],
            min_positive_samples=LEGACY_MIN_POSITIVE_SAMPLES,
        )
        folds = schedule(data["binary"][i], cv_start, HORIZON, HORIZON)
        one = drive(forecaster, [data["count"][i]], folds)
        got_c.append(one["prob"][0])
        got_r.append(one["count"][0])
        got_h.append(one["hurdle"][0])

    assert_channel_equal(got_c, exp_c, "prob")
    assert_channel_equal(got_r, exp_r, "count")
    assert_channel_equal(got_h, exp_h, "hurdle")
    assert len(new_b.regs) == len(legacy_b.regs)
    assert len(new_b.clfs) == len(legacy_b.clfs)


def test_hurdle_dummy_branch_shape_and_index():
    """The dummy prediction is ``np.full((n, 1), pos_vals.mean())`` on the
    CLASSIFIER's time index."""
    data = _per_region_dataset()
    i = 1  # the sparse region
    b = Builders()
    forecaster = HurdleForecaster(
        b.classifier,
        b.regressor,
        binary_targets=[data["binary"][i]],
        count_targets=[data["count"][i]],
        weights=[data["weights"][i]],
        clf_past=[data["clf_past"][i]],
        clf_future=[data["clf_future"][i]],
        reg_past=[data["reg_past"][i]],
        reg_future=[data["reg_future"][i]],
        min_positive_samples=50,
    )
    cutoff = data["count"][i].time_index[63]
    forecaster.fit([data["count"][i].drop_after(cutoff)], cutoff=cutoff)
    assert b.regs == []  # the regressor was never built
    out = forecaster.predict(HORIZON, [data["count"][i].drop_after(cutoff)], cutoff=cutoff)

    pred_r = out["count"][0]
    vals = data["count"][i].drop_after(cutoff).values()
    w = data["weights"][i].drop_after(cutoff).values()
    expected_mean = vals[w > 0].mean()
    assert pred_r.values().shape == (HORIZON, 1)
    np.testing.assert_allclose(pred_r.values(), np.full((HORIZON, 1), expected_mean), atol=0.0)
    # the index is the CLASSIFIER's prediction index: the context ends one step
    # before the cutoff, so the forecast window starts AT the cutoff
    assert pred_r.time_index.equals(pd.date_range(cutoff, periods=HORIZON, freq=FREQ))
    # all-zero training slice -> dummy_mean falls back to 1.0
    empty = _series(np.zeros(90), "zero")
    f2 = HurdleForecaster(
        b.classifier,
        b.regressor,
        binary_targets=[empty],
        count_targets=[empty],
        weights=hurdle_runners.make_positive_only_weights([empty]),
        clf_past=None,
        clf_future=None,
        reg_past=None,
        reg_future=None,
        min_positive_samples=50,
    )
    f2.fit([empty.drop_after(cutoff)], cutoff=cutoff)
    out2 = f2.predict(HORIZON, [empty.drop_after(cutoff)], cutoff=cutoff)
    np.testing.assert_allclose(out2["count"][0].values(), np.full((HORIZON, 1), 1.0))


def test_hurdle_rejects_multi_region_local_mode():
    data = make_dataset()
    with pytest.raises(ValueError, match="one region at a time"):
        HurdleForecaster(
            StubClassifier,
            StubRegressor,
            binary_targets=data["binary"],
            weights=data["weights"],
            clf_past=None,
            clf_future=None,
            reg_past=None,
            reg_future=None,
            min_positive_samples=50,
        )


# --------------------------------------------------------------------------- #
# damage classifiers (damage_classifier.ipynb cells 17 and 26)
# --------------------------------------------------------------------------- #
def _damage_classes(keys, cv_start, train_val_end):
    blocks = {}
    for k_idx, key in enumerate(keys):
        data = make_dataset(seed=10 + k_idx)
        train, val, test = split_three(data["binary"])
        blocks[key] = {
            "get_covs_and_encodings": {
                "R": data["region_names"],
                "Tr": train,
                "V": val,
                "Te": test,
                "FPC": data["clf_past"],
                "FFC": data["clf_future"],
                "TCV": data["binary"],
                "tve": train_val_end,
                "csv": cv_start,
            }
        }
    return blocks


def test_damage_cv_matches_legacy():
    keys = ["act_drone_strike_energy", "act_drone_strike_transport"]
    cv_start, train_val_end = 0.7, 0.8
    damage_classes = _damage_classes(keys, cv_start, train_val_end)

    legacy_b = Builders()
    expected = damage_runners.run_damage_cv(
        1,
        HORIZON,
        damage_classes=damage_classes,
        get_damage_classifier=legacy_b.classifier,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_b = Builders()
    forecaster = MultiTargetClassifierForecaster(
        new_b.classifier,
        {k: damage_classes[k]["get_covs_and_encodings"]["TCV"] for k in keys},
        {k: damage_classes[k]["get_covs_and_encodings"]["FPC"] for k in keys},
        {k: damage_classes[k]["get_covs_and_encodings"]["FFC"] for k in keys},
    )
    reference = damage_classes[keys[0]]["get_covs_and_encodings"]["TCV"][0]
    folds = schedule(reference, cv_start, HORIZON, HORIZON)
    got = drive(forecaster, damage_classes[keys[0]]["get_covs_and_encodings"]["TCV"], folds)

    assert forecaster.channels == tuple(keys)
    for key in keys:
        assert_channel_equal(got[key], expected[key], key)
    assert len(new_b.clfs) == len(legacy_b.clfs) == len(keys) * len(
        [f for f in folds if f[1]]
    )
    # the kept component is the LAST of the two the stub returns
    assert got[keys[0]][0][0].n_components == 1
    assert float(got[keys[0]][0][0].values()[0, 0]) > 0.0


def test_damage_test_loop_matches_legacy():
    keys = ["energy", "transport", "water"]
    cv_start, train_val_end = 0.7, 0.7 + 0.1
    damage_classes = _damage_classes(keys, cv_start, train_val_end)

    legacy_b = Builders()
    expected, full_target_d = damage_runners.run_damage_test(
        1,
        HORIZON,
        damage_classes=damage_classes,
        get_damage_classifier=legacy_b.classifier,
        OUTPUT_CHUNK_LEN=HORIZON,
    )

    new_b = Builders()
    forecaster = MultiTargetClassifierForecaster(
        new_b.classifier,
        full_target_d,
        {k: damage_classes[k]["get_covs_and_encodings"]["FPC"] for k in keys},
        {k: damage_classes[k]["get_covs_and_encodings"]["FFC"] for k in keys},
    )
    folds = schedule(full_target_d[keys[0]][0], train_val_end, HORIZON, HORIZON)
    got = drive(forecaster, full_target_d[keys[0]], folds)

    assert len(folds) > 0
    for key in keys:
        assert_channel_equal(got[key], expected[key], key)


def test_damage_uses_per_key_covariates():
    keys = ["a", "b"]
    damage_classes = _damage_classes(keys, 0.7, 0.8)
    b = Builders()
    targets = {k: damage_classes[k]["get_covs_and_encodings"]["TCV"] for k in keys}
    past = {k: damage_classes[k]["get_covs_and_encodings"]["FPC"] for k in keys}
    future = {k: damage_classes[k]["get_covs_and_encodings"]["FFC"] for k in keys}
    forecaster = MultiTargetClassifierForecaster(b.classifier, targets, past, future)
    forecaster.prepare(None, None)

    cutoff = targets[keys[0]][0].time_index[45]
    forecaster.fit([], cutoff=cutoff)
    forecaster.predict(HORIZON, [], cutoff=cutoff)

    assert len(b.clfs) == len(keys)
    for k_idx, key in enumerate(keys):
        model = b.clfs[k_idx]
        assert model.fit_calls[0]["past_covariates"] is past[key]
        assert model.fit_calls[0]["future_covariates"] is future[key]
        assert model.predict_calls[0]["predict_likelihood_parameters"] is True
        assert model.predict_calls[0]["show_warnings"] is False
        assert model.fit_calls[0]["series"][0].end_time() < cutoff


def test_damage_predict_before_fit_raises():
    keys = ["a"]
    damage_classes = _damage_classes(keys, 0.7, 0.8)
    forecaster = MultiTargetClassifierForecaster(
        StubClassifier,
        {k: damage_classes[k]["get_covs_and_encodings"]["TCV"] for k in keys},
        {k: damage_classes[k]["get_covs_and_encodings"]["FPC"] for k in keys},
        {k: damage_classes[k]["get_covs_and_encodings"]["FFC"] for k in keys},
    )
    with pytest.raises(RuntimeError, match="before fit"):
        forecaster.predict(HORIZON, [], cutoff=pd.Timestamp("2022-02-01"))
