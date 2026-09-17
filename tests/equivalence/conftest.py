"""Synthetic panel and shared helpers for the LEVEL D loop-equivalence tests.

Plan §6 level D: "new engine vs each old runner on a small synthetic panel with
deterministic models (naive, linear, LightGBM CPU single-thread), tolerance
1e-9". Everything in this package compares, on identical inputs:

* ``strikecast.backtest.ExpandingWindowBacktest`` driven end to end, against
* the legacy runner -- ``src.evaluation_tools`` for the count family and the
  verbatim copies in ``tests/legacy_ref`` for the diff, hurdle and damage
  families

fold by fold AND through ``PredictionSet.legacy_frame()`` versus
``src.evaluation_tools.collect_predictions_long``.

The panel
---------
Four regions, 121 daily steps, integer-valued non-negative counts with zeros
(Poisson draws, fixed ``numpy`` seed), three past covariates and two future
covariates per region at FULL length, one static ``region`` label plus an
``Activity_Level`` tier, both label-encoded by a ``StaticCovariatesTransformer``
exactly as ``strikecast.data.series::_encode_statics`` does it. The activity map
has two tiers, interleaved across the region order (``r0``/``r2`` -> tier 2,
``r1``/``r3`` -> tier 1) so that the Activity paradigm's
``for level in sorted(groups)`` order is NOT the region order and a wrapper that
forgot to scatter its results back would fail.

Why 121 days and not 120
------------------------
The diff family differences BEFORE splitting, so its CV view is
``Diff(...).fit_transform(full)`` and only then ``split_before(TRAIN_VAL_END)``.
Whether that list has the same length as the level CV view or one step fewer
depends on the panel length. On the real panel (847 days) the diffed CV view is
676, the same as the level CV view, while differencing the level CV view gives
675 -- the mismatch recorded as F51/F52. At 120 days both orderings give 94 and
the mismatch is invisible; at 121 days the real panel's pattern (95 / 95 / 94)
reproduces exactly. See ``test_diff_runners.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from darts import TimeSeries  # noqa: E402
from darts.dataprocessing.transformers import StaticCovariatesTransformer  # noqa: E402
from darts.models import (  # noqa: E402
    LightGBMModel,
    LinearRegressionModel,
    NaiveMean,
    SKLearnClassifierModel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import shim
    sys.path.insert(0, str(REPO_ROOT))

from strikecast.transforms import Diff  # noqa: E402

# --------------------------------------------------------------------------- #
# panel constants
# --------------------------------------------------------------------------- #
N_DAYS = 121
REGION_NAMES: list[str] = ["r0", "r1", "r2", "r3"]
#: Two tiers, deliberately NOT in region order (see the module docstring).
ACTIVITY_BY_REGION: dict[str, int] = {"r0": 2, "r1": 1, "r2": 2, "r3": 1}

TARGET_COL = "act_drone_strike_on_ua"
PAST_COLS: list[str] = ["past_a", "past_b", "past_c"]
FUTURE_COLS: list[str] = ["fut_a", "fut_b"]
GROUP_COL = "region"
TIME_COL = "event_date"

HORIZON = 7
RETRAIN_STRIDE = 7
PREDICT_STRIDE = 1

#: ``TRAIN_FRAC + VAL_FRAC`` written the legacy way: 0.7999999999999999 (F17).
TRAIN_VAL_END = 0.70 + 0.10
#: ``CV_START_VAL`` of the legacy pipeline.
CV_START_VAL = 0.875
#: A second, earlier CV start used by most tests so that a run has FOUR retrain
#: points instead of one; the schedule code path is identical.
CV_START_EARLY = 0.70

SEED = 20240917


# --------------------------------------------------------------------------- #
# panel / series construction
# --------------------------------------------------------------------------- #
def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2021-01-01", periods=N_DAYS, freq="D")
    frames: list[pd.DataFrame] = []
    for r_idx, region in enumerate(REGION_NAMES):
        lam = 0.4 + 0.9 * r_idx  # region 0 is sparse, region 3 is busy
        block = {
            GROUP_COL: region,
            TIME_COL: dates,
            TARGET_COL: rng.poisson(lam, N_DAYS).astype(float),
            "Activity_Level": ACTIVITY_BY_REGION[region],
        }
        for c_idx, col in enumerate(PAST_COLS):
            block[col] = rng.normal(loc=r_idx, scale=1.0 + c_idx, size=N_DAYS)
        for c_idx, col in enumerate(FUTURE_COLS):
            block[col] = np.sin(np.arange(N_DAYS) / (3.0 + c_idx)) + r_idx
        frames.append(pd.DataFrame(block))
    return pd.concat(frames, ignore_index=True)


def _encode(series_list: list[TimeSeries]) -> list[TimeSeries]:
    """``strikecast.data.series::_encode_statics``: one fresh encoder per list."""
    return list(StaticCovariatesTransformer().fit_transform(list(series_list)))


class Panel:
    """Every series list the equivalence tests need, built once per session."""

    def __init__(self) -> None:
        panel = _panel()

        targets = TimeSeries.from_group_dataframe(
            panel,
            group_cols=GROUP_COL,
            time_col=TIME_COL,
            value_cols=TARGET_COL,
            static_cols=["Activity_Level"],
        )
        past = TimeSeries.from_group_dataframe(
            panel, group_cols=GROUP_COL, time_col=TIME_COL, value_cols=PAST_COLS
        )
        future = TimeSeries.from_group_dataframe(
            panel, group_cols=GROUP_COL, time_col=TIME_COL, value_cols=FUTURE_COLS
        )

        # Region names are captured BEFORE encoding, as `_encode_and_split` does.
        self.region_names = [ts.static_covariates[GROUP_COL].iloc[0] for ts in targets]
        assert self.region_names == REGION_NAMES

        self.target_full = _encode(list(targets))
        self.past_covs = _encode(list(past))
        self.future_covs = _encode(list(future))

        # CV view: `split_before(TRAIN_VAL_END)` on the LEVEL list.
        self.target_cv = [ts.split_before(TRAIN_VAL_END)[0] for ts in self.target_full]

        # Diff family: difference the FULL list, then split (the legacy order).
        self.target_full_diff = Diff().forward(self.target_full)
        self.target_cv_diff = [
            ts.split_before(TRAIN_VAL_END)[0] for ts in self.target_full_diff
        ]

        self.activity_by_region = dict(ACTIVITY_BY_REGION)


@pytest.fixture(scope="session")
def panel() -> Panel:
    return Panel()


# --------------------------------------------------------------------------- #
# hurdle / damage series
# --------------------------------------------------------------------------- #
def _binary(series_list: list[TimeSeries]) -> list[TimeSeries]:
    return [
        TimeSeries.from_times_and_values(
            ts.time_index,
            (ts.values().ravel() > 0).astype(float).reshape(-1, 1),
            static_covariates=ts.static_covariates,
        )
        for ts in series_list
    ]


class HurdlePanel:
    """Counts + binary events + positive-only weights + two covariate pairs.

    Region 1's counts are overwritten with a sparse pattern that has FEWER than
    ``MIN_POSITIVE_SAMPLES = 50`` positive days in every training slice, so the
    per-region (local) hurdle runner takes its dummy-regressor branch there and
    the regressor branch everywhere else.
    """

    def __init__(self, base: Panel) -> None:
        from tests.legacy_ref import hurdle_runners  # noqa: PLC0415

        counts_full = list(base.target_full)

        sparse = np.zeros(N_DAYS)
        sparse[[3, 17, 29, 44, 58, 71, 83]] = [2.0, 5.0, 1.0, 4.0, 3.0, 2.0, 6.0]
        counts_full[1] = TimeSeries.from_times_and_values(
            counts_full[1].time_index,
            sparse.reshape(-1, 1),
            static_covariates=counts_full[1].static_covariates,
        )

        self.region_names = list(base.region_names)
        self.activity_by_region = dict(base.activity_by_region)

        self.count_full = counts_full
        self.binary_full = _binary(counts_full)
        self.count_cv = [ts.split_before(TRAIN_VAL_END)[0] for ts in self.count_full]
        self.binary_cv = [ts.split_before(TRAIN_VAL_END)[0] for ts in self.binary_full]

        self.weights_cv = hurdle_runners.make_positive_only_weights(self.count_cv)
        self.weights_full = hurdle_runners.make_positive_only_weights(self.count_full)

        # The hurdle family gives the two heads DIFFERENT covariate lists; using
        # the same list twice would hide a head/covariate mix-up.
        self.clf_past = base.past_covs
        self.clf_future = base.future_covs
        # a second, distinguishable past list
        self.reg_past = [ts * 1.5 for ts in base.past_covs]
        self.reg_future = [ts * 0.5 for ts in base.future_covs]

        # Test-stage 70/10/20 split, fed to `run_hurdle_test`'s signature.
        self.train_c, self.val_c, self.test_c = _split_three(self.binary_full)
        self.train_r, self.val_r, self.test_r = _split_three(self.count_full)


def _split_three(
    series_list: list[TimeSeries],
) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
    """``strikecast.data.series.split_series_list`` / the legacy 70/10/20 split."""
    train, val, test = [], [], []
    for ts in series_list:
        tr, temp = ts.split_after(0.7)
        vl, te = temp.split_after(1 / 3)
        train.append(tr)
        val.append(vl)
        test.append(te)
    return train, val, test


@pytest.fixture(scope="session")
def hurdle_panel(panel: Panel) -> HurdlePanel:
    return HurdlePanel(panel)


DAMAGE_KEYS: list[str] = ["act_damage_energy", "act_damage_transport"]


class DamagePanel:
    """``damage_classes``-shaped blocks for two damage keys.

    The structure is the notebook's: ``key -> {'get_covs_and_encodings': {...}}``
    with the sub-keys ``R/Tr/V/Te/FPC/FFC/TCV/tve/csv`` documented in
    ``tests/legacy_ref/damage_runners.py``.
    """

    def __init__(self, base: Panel) -> None:
        rng = np.random.default_rng(SEED + 1)
        self.keys = list(DAMAGE_KEYS)
        self.region_names = list(base.region_names)
        self.classes: dict[str, dict[str, Any]] = {}

        for k_idx, key in enumerate(self.keys):
            targets = [
                TimeSeries.from_times_and_values(
                    ts.time_index,
                    (rng.random(N_DAYS) < (0.25 + 0.1 * k_idx + 0.05 * r_idx))
                    .astype(float)
                    .reshape(-1, 1),
                    static_covariates=ts.static_covariates,
                )
                for r_idx, ts in enumerate(base.target_full)
            ]
            train, val, test = _split_three(targets)
            scale = 1.0 + 0.3 * k_idx
            self.classes[key] = {
                "get_covs_and_encodings": {
                    "R": list(base.region_names),
                    "Tr": train,
                    "V": val,
                    "Te": test,
                    "FPC": [ts * scale for ts in base.past_covs],
                    "FFC": [ts * scale for ts in base.future_covs],
                    "TCV": [ts.split_before(TRAIN_VAL_END)[0] for ts in targets],
                    "tve": TRAIN_VAL_END,
                    "csv": CV_START_EARLY,
                }
            }

    def block(self, key: str) -> dict[str, Any]:
        return self.classes[key]["get_covs_and_encodings"]


@pytest.fixture(scope="session")
def damage_panel(panel: Panel) -> DamagePanel:
    return DamagePanel(panel)


# --------------------------------------------------------------------------- #
# deterministic model builders
# --------------------------------------------------------------------------- #
def linear_builder() -> LinearRegressionModel:
    """The level-D linear model: covariates on, single-threaded, deterministic."""
    return LinearRegressionModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=(1, 2),
        output_chunk_length=HORIZON,
    )


def linear_builder_no_covs() -> LinearRegressionModel:
    return LinearRegressionModel(lags=3, output_chunk_length=HORIZON)


def log_link_linear_builder() -> LinearRegressionModel:
    """A linear model carrying ``_count_log_link``, the flag ``build_lstm_count`` sets.

    Only the ``exp`` post-processing differs between the three legacy runners
    for a non-neural model, so this is what makes ``for_cv`` / ``for_test`` /
    ``for_tuning`` distinguishable end to end (F55 / F56).
    """
    model = LinearRegressionModel(lags=3, output_chunk_length=HORIZON)
    model._count_log_link = True  # noqa: SLF001 - legacy sets exactly this attribute
    return model


def classifier_builder() -> SKLearnClassifierModel:
    """Binary classifier with ``predict_likelihood_parameters=True`` support.

    darts 0.43's default estimator is ``LogisticRegression(n_jobs=-1)`` and the
    likelihood is ``classprobability``, so ``predict_likelihood_parameters=True``
    returns one component per class (``0_p0``, ``0_p1``) and the LAST component
    -- what every legacy hurdle/damage loop keeps -- is ``P(Y = 1)``.
    """
    return SKLearnClassifierModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=(1, 2),
        output_chunk_length=HORIZON,
    )


def count_head_builder() -> LinearRegressionModel:
    """Hurdle count head; ``fit(sample_weight=list[TimeSeries])`` is supported."""
    return LinearRegressionModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=(1, 2),
        output_chunk_length=HORIZON,
    )


def lgbm_count_head_builder() -> LightGBMModel:
    """The hurdle's real count head shape: Poisson LightGBM, one thread, seeded."""
    return LightGBMModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=(1, 2),
        output_chunk_length=HORIZON,
        objective="poisson",
        n_estimators=20,
        num_leaves=7,
        min_child_samples=5,
        n_jobs=1,
        num_threads=1,
        deterministic=True,
        seed=0,
        verbose=-1,
        random_state=0,
    )


class FlakyLocalModel:
    """A local model that RAISES for one region, to exercise the F5 fallback.

    ``fit`` raises when the series' encoded ``region`` static equals
    ``fail_region_code``; otherwise it behaves like ``NaiveMean``. The legacy
    diff runners catch bare ``Exception`` and fall back to ``NaiveMean()``, so a
    correct adapter produces ``NaiveMean`` predictions for that one region and
    ``NaiveMean`` predictions for the others too -- which is why ``_offset``
    shifts the non-failing regions, making the fallback visible in the values.
    """

    _offset = 0.25

    def __init__(self, fail_region_code: float = 1.0) -> None:
        self.fail_region_code = fail_region_code
        self._inner = NaiveMean()

    def fit(self, series: TimeSeries) -> FlakyLocalModel:
        statics = series.static_covariates
        code = None if statics is None else float(statics[GROUP_COL].iloc[0])
        if code == self.fail_region_code:
            raise RuntimeError(f"local model refuses region code {code}")
        self._inner.fit(series)
        return self

    def predict(self, n: int) -> TimeSeries:
        pred = self._inner.predict(n=n)
        return pred + self._offset


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
ATOL = 1e-9


def assert_fold_preds_equal(new: Any, legacy: Any, label: str = "") -> None:
    """Level-D comparison of two ``[[pred per fold] per region]`` bundles."""
    assert len(new) == len(legacy), f"{label}: region count {len(new)} != {len(legacy)}"
    for r_idx, (new_region, legacy_region) in enumerate(zip(new, legacy, strict=True)):
        assert len(new_region) == len(legacy_region), (
            f"{label}: region {r_idx} fold count {len(new_region)} != {len(legacy_region)}"
        )
        for f_idx, (a, b) in enumerate(zip(new_region, legacy_region, strict=True)):
            where = f"{label}region {r_idx} fold {f_idx}"
            assert a.time_index.equals(b.time_index), f"{where}: time index differs"
            assert a.n_components == b.n_components, f"{where}: n_components differs"
            np.testing.assert_allclose(
                a.values(), b.values(), rtol=0.0, atol=ATOL, err_msg=where
            )


def assert_long_frames_equal(
    prediction_set: Any,
    actuals: list[TimeSeries],
    legacy_fold_preds: Any,
    region_names: list[str],
    channel: str = "y_pred",
) -> None:
    """``PredictionSet.legacy_frame()`` vs ``collect_predictions_long``, dtypes included."""
    from src.evaluation_tools import collect_predictions_long  # noqa: PLC0415

    legacy_long = collect_predictions_long(actuals, legacy_fold_preds, region_names)
    prediction_set.assert_equal_legacy(legacy_long, channel)
    assert len(legacy_long) > 0
