"""Level C golden test: the fold schedule equals the legacy `for` loop, exactly.

For every family's CV and test configuration the real series lengths are built
through `strikecast.data`, then the `(t0, retrain)` list produced by
`strikecast.backtest.schedule.fold_schedule` is compared against a verbatim
inline copy of the legacy loop -- the copy below is the loop body of
`src/evaluation_tools.py::run_expanding_cv` / `run_final_test`,
`_diff_regression.py::run_expanding_cv` / `run_final_test_diff`,
`final_hurdle.ipynb` cells 15/24 and `damage_classifier.ipynb` cells 17/26,
with `chronos2_rolling_long` as the never-retrained variant.

Recorded counts on the real panel (20 regions, 847 daily steps, target
``act_drone_strike_on_ua``), horizon 7 and predict stride 1 throughout:

===============  =======  =========  =====  ========  ============  ============
family / stage   n_total  start_idx  folds  retrains  first cutoff  last cutoff
===============  =======  =========  =====  ========  ============  ============
count CV            676        591      79        12    2024-05-11    2024-07-28
count test          847        677     164        24    2024-08-05    2025-01-15
diff CV             676        591      79        12    2024-05-12    2024-07-29
diff test           846        676     164        24    2024-08-05    2025-01-15
hurdle CV           676        591      79        12    2024-05-11    2024-07-28
hurdle test         847        677     164        24    2024-08-05    2025-01-15
chronos test        847        677     164         0    2024-08-05    2025-01-15
===============  =======  =========  =====  ========  ============  ============

Two observations that the numbers make visible (both preserved, see the report):

* the diff CV view is 676 steps like the level CV view, not 675, because
  ``split_before(0.7999999999999999)`` is applied to the already-diffed
  846-step series. The fold *indices* therefore coincide with the count
  family's while the *dates* are shifted one day later (flag F33).
* Chronos-2's rolling start is ``int((TRAIN_FRAC + VAL_FRAC) * n)`` -- the same
  expression as every darts family -- so its test folds are date-identical to
  the count family's. Only the AutoGluon fit/holdout split uses
  ``round(TEST_FRAC * n)`` (flag F34, refines F3).
"""

from __future__ import annotations

import pytest

from strikecast.backtest.schedule import fold_schedule
from strikecast.config.schema import BacktestConfig, SeriesConfig, WindowTransformConfig
from strikecast.data import (
    build_bundle,
    build_panel_legacy_chronos,
    build_panel_legacy_regressor,
    split_covariates,
)

pytestmark = pytest.mark.golden

pytest.importorskip("darts")

TARGET = "act_drone_strike_on_ua"
TARGET_BINARY = "act_drone_strike_on_ua_binary"
HORIZON = 7  # OUTPUT_CHUNK_LEN
STRIDE = 1  # CV_STRIDE / predict_stride
RETRAIN_STRIDE = 7  # OUTPUT_CHUNK_LEN


# --------------------------------------------------------------------------- #
# verbatim legacy loops
# --------------------------------------------------------------------------- #
def legacy_loop(n_total, start_frac, horizon, stride, retrain_stride):
    """Verbatim copy of the legacy expanding-window loop body.

    Lifted from ``src/evaluation_tools.py::run_expanding_cv`` (lines 301-320);
    ``run_final_test``, ``run_expanding_cv_iter``, the diff twins and the
    hurdle / damage notebook loops are character-for-character the same, with
    ``stride`` spelled ``predict_stride``.
    """
    n_total = int(n_total)
    start_idx = int(start_frac * n_total)
    pairs = []
    for t0 in range(start_idx, n_total - horizon + 1, stride):
        steps_since_start = t0 - start_idx
        retrain = steps_since_start % retrain_stride == 0
        pairs.append((t0, retrain))
    return pairs


def legacy_loop_chronos(n_per_item, test_start_idx, horizon, stride):
    """Verbatim copy of ``_chronos2.py::chronos2_rolling_long``'s loop.

    The predictor is fit once in §5 and never refit, so no fold retrains.
    """
    pairs = []
    for t0 in range(test_start_idx, n_per_item - horizon + 1, stride):
        pairs.append((t0, False))
    return pairs


def as_pairs(folds):
    return [(f.t0, f.retrain) for f in folds]


# --------------------------------------------------------------------------- #
# real series
# --------------------------------------------------------------------------- #
def _bundle(inputs, target, *, binarize_target=False, target_raw_col=None):
    panel = build_panel_legacy_regressor(
        inputs, target, binarize_target=binarize_target, target_raw_col=target_raw_col
    )
    covs = split_covariates(panel.panel, panel.global_weather_columns, target)
    return build_bundle(
        panel=panel.panel,
        target=target,
        past_covariates=covs.past_covariates,
        future_covariates=covs.future_covariates,
        config=SeriesConfig(window=WindowTransformConfig(expdecay="legacy_alpha")),  # thesis: legacy expdecay7 (A1/D1)
        activity_by_region=inputs.activity_by_region,
    )


@pytest.fixture(scope="module")
def count_bundle(inputs):
    return _bundle(inputs, TARGET)


@pytest.fixture(scope="module")
def hurdle_bundle(inputs):
    """The hurdle's event-classifier bundle (`target_for_cv_c` in cell 15)."""
    return _bundle(inputs, TARGET_BINARY, binarize_target=True, target_raw_col=TARGET)


@pytest.fixture(scope="module")
def diff_series(count_bundle):
    """`target_full_diff` and `target_for_cv_diff` of `_diff_regression.py`.

    The notebook diffs the *un-encoded* target list with darts'
    ``Diff(lags=1, dropna=True)`` and then re-runs ``get_covs_and_encodings``,
    whose CV view is ``split_before(TRAIN_VAL_END)``. Static encoding does not
    change lengths or the time index, so the schedule only needs these two.
    """
    from darts.dataprocessing.transformers import Diff  # noqa: PLC0415

    full = Diff(lags=1, dropna=True).fit_transform(count_bundle.raw.target)
    cv_view = [ts.split_before(count_bundle.train_val_end)[0] for ts in full]
    return {"full": full, "cv_view": cv_view}


@pytest.fixture(scope="module")
def chronos_panel(inputs):
    """`n_full` and the timestamps of `_chronos2.py` §10's `tsdf`.

    ``n_full = int(tsdf.num_timesteps_per_item().min())`` on the MultiIndexed
    Chronos panel is the number of rows of its shortest item.
    """
    panel = build_panel_legacy_chronos(inputs, TARGET).panel
    per_item = panel.groupby(level="region").size()
    first_region = panel.index.get_level_values("region")[0]
    time_index = panel.xs(first_region, level="region").index
    return {"n": int(per_item.min()), "time_index": time_index}


# --------------------------------------------------------------------------- #
# per-family equality
# --------------------------------------------------------------------------- #
def _check(reference, start_frac, *, expected_folds, expected_retrains, retrain_stride=7):
    cfg = BacktestConfig(
        start_frac=start_frac,
        horizon=HORIZON,
        predict_stride=STRIDE,
        retrain_stride=retrain_stride,
    )
    n_total = len(reference)
    folds = fold_schedule(
        n_total=n_total,
        start_frac=cfg.start_frac,
        horizon=cfg.horizon,
        predict_stride=cfg.predict_stride,
        retrain_stride=cfg.retrain_stride,
        time_index=reference.time_index,
    )
    expected = legacy_loop(n_total, start_frac, HORIZON, STRIDE, retrain_stride)

    assert as_pairs(folds) == expected
    assert len(folds) == expected_folds
    assert sum(f.retrain for f in folds) == expected_retrains
    assert [f.index for f in folds] == list(range(len(folds)))
    assert all(f.cutoff == reference.time_index[f.t0] for f in folds)
    return folds


def test_count_cv(count_bundle) -> None:
    """`run_expanding_cv(target_for_cv, CV_START_VAL)`: 79 folds, 12 retrains."""
    folds = _check(
        count_bundle.target_cv_view[0],
        count_bundle.cv_start_frac,
        expected_folds=79,
        expected_retrains=12,
    )
    assert len(count_bundle.target_cv_view[0]) == 676
    assert folds[0].t0 == 591
    assert str(folds[0].cutoff.date()) == "2024-05-11"
    assert str(folds[-1].cutoff.date()) == "2024-07-28"


def test_count_test(count_bundle) -> None:
    """`run_final_test(target_full, TRAIN_VAL_END)`: 164 folds, 24 retrains."""
    folds = _check(
        count_bundle.target_full[0],
        count_bundle.train_val_end,
        expected_folds=164,
        expected_retrains=24,
    )
    assert len(count_bundle.target_full[0]) == 847
    assert folds[0].t0 == 677
    assert str(folds[0].cutoff.date()) == "2024-08-05"
    assert str(folds[-1].cutoff.date()) == "2025-01-15"


def test_diff_cv(diff_series, count_bundle) -> None:
    """Diff CV runs on `target_for_cv_diff` with the LEVEL `CV_START_VAL`."""
    reference = diff_series["cv_view"][0]
    folds = _check(
        reference,
        count_bundle.cv_start_frac,
        expected_folds=79,
        expected_retrains=12,
    )
    assert len(reference) == 676
    assert folds[0].t0 == 591
    # same indices as the count family, dates one day later (F33)
    assert str(folds[0].cutoff.date()) == "2024-05-12"
    assert str(folds[-1].cutoff.date()) == "2024-07-29"


def test_diff_test(diff_series, count_bundle) -> None:
    """`run_final_test_diff(target_full_diff, TRAIN_VAL_END)` on 846 steps."""
    reference = diff_series["full"][0]
    folds = _check(
        reference,
        count_bundle.train_val_end,
        expected_folds=164,
        expected_retrains=24,
    )
    assert len(reference) == len(count_bundle.target_full[0]) - 1 == 846
    assert folds[0].t0 == 676
    assert str(folds[0].cutoff.date()) == "2024-08-05"
    assert str(folds[-1].cutoff.date()) == "2025-01-15"


def test_hurdle_cv(hurdle_bundle, count_bundle) -> None:
    """`run_hurdle_cv` (cell 15) is the count schedule on the binary target."""
    reference = hurdle_bundle.target_cv_view[0]
    folds = _check(
        reference,
        hurdle_bundle.cv_start_frac,
        expected_folds=79,
        expected_retrains=12,
    )
    assert len(reference) == len(count_bundle.target_cv_view[0]) == 676
    assert as_pairs(folds) == as_pairs(
        fold_schedule(
            len(count_bundle.target_cv_view[0]),
            count_bundle.cv_start_frac,
            HORIZON,
            STRIDE,
            RETRAIN_STRIDE,
        )
    )


def test_hurdle_test(hurdle_bundle, count_bundle) -> None:
    """`run_final_test` (cell 24) is the count test schedule on the binary target."""
    reference = hurdle_bundle.target_full[0]
    folds = _check(
        reference,
        hurdle_bundle.train_val_end,
        expected_folds=164,
        expected_retrains=24,
    )
    assert len(reference) == len(count_bundle.target_full[0]) == 847
    assert str(folds[0].cutoff.date()) == "2024-08-05"


def test_chronos_test(chronos_panel, count_bundle) -> None:
    """`chronos2_rolling_long`: 164 folds, never retrained (F34)."""
    chronos_n = chronos_panel["n"]
    train_val_end = 0.70 + 0.10  # `TRAIN_FRAC + VAL_FRAC` in _chronos2.py §10
    test_start_ix = int(train_val_end * chronos_n)

    folds = fold_schedule(
        n_total=chronos_n,
        start_frac=train_val_end,
        horizon=HORIZON,
        predict_stride=STRIDE,
        retrain_stride=None,
        time_index=chronos_panel["time_index"],
    )
    expected = legacy_loop_chronos(chronos_n, test_start_ix, HORIZON, STRIDE)

    assert as_pairs(folds) == expected
    assert len(folds) == 164
    assert sum(f.retrain for f in folds) == 0
    # the Chronos panel has the same 847 steps as the darts panel, and the same
    # start expression, so its test folds are date-identical to the count test
    assert chronos_n == len(count_bundle.target_full[0]) == 847
    assert chronos_panel["time_index"].equals(count_bundle.target_full[0].time_index)
    assert test_start_ix == 677
    assert str(folds[0].cutoff.date()) == "2024-08-05"
    assert str(folds[-1].cutoff.date()) == "2025-01-15"


def test_damage_matches_the_count_schedule(count_bundle) -> None:
    """`run_damage_cv` / `run_final_test` (damage cells 17, 26) share the loop.

    The damage panels differ only in which column is binarised, so the series
    length -- and therefore the schedule -- is the count family's.
    """
    for reference, start_frac in (
        (count_bundle.target_cv_view[0], count_bundle.cv_start_frac),
        (count_bundle.target_full[0], count_bundle.train_val_end),
    ):
        n_total = len(reference)
        folds = fold_schedule(n_total, start_frac, HORIZON, STRIDE, RETRAIN_STRIDE)
        assert as_pairs(folds) == legacy_loop(
            n_total, start_frac, HORIZON, STRIDE, RETRAIN_STRIDE
        )
