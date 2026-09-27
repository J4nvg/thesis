"""Level H: the hurdle family's calibration, components and metrics vs `golden/`.

Plan §6 level H is "hurdle predictions, sigmoid and Venn-Abers views vs
`results/finalhurdle/`; 1e-6 for CPU CatBoost; **calibration exact given
identical inputs**". That sentence splits the level in two, and this module
covers both halves differently, because only one of them is reproducible:

**The calibration and scoring half -- exact, and asserted.**
`golden/results/finalhurdle/` stores the thesis' own CV and test prediction
frames for all three hurdle channels. Feeding exactly those frames through
`strikecast.pipeline.composite_stage` must reproduce, bit for bit, the
calibrated probability frames, the calibrated hurdle product and all five
metric components the notebook saved (cells 48-49). That pins everything this
stream added downstream of the backtest: the per-horizon sigmoid fit and its
in-sample / out-of-sample application (F69), the positional `prob * count`
product (cell 22), the positive-only count-head population (F65), the per-stage
naive scales (F6/F67) and the four-view / five-component metric layout. No
model is fitted, so it runs in seconds and needs no GPU.

**The prediction half -- run on demand, NOT asserted numerically.**
Reproducing `global_cv_hurdle_preds.parquet` itself would need the exact
top-100 feature sets the thesis ran with, and those cannot be recovered: the
covariate column order depends on `PYTHONHASHSEED` and the hurdle notebook
cached nothing (`cache: false` in `configs/experiment/hurdle.yaml`; the one
converted `golden/converted/feature_sets/zipoisson.json` is a SINGLE selection
while the hurdle ran TWO, and which head it belongs to is not recorded). That
is flag F16's "the cached feature sets are DATA to load, not output to
reproduce", and it is why level G exists at all. So the real-data test here
checks what IS invariant -- the fold schedule, the channel set, the row counts
and the per-channel `y_true` -- and REPORTS the numeric deviation instead of
asserting it.

It is opt-in (`STRIKECAST_GOLDEN_HURDLE=1`) and fold-limited
(`STRIKECAST_GOLDEN_HURDLE_FOLDS`, default 3) so that a full 79-fold CV run,
which is 12 retrains of a 100-tree Self-Paced Ensemble plus a CatBoost per
horizon, never lands in a normal test session. With the default fold limit one
retrain window is run, which is minutes rather than hours.

Everything skips gracefully when `golden/results/finalhurdle/` or `data/` is
absent.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.golden

pytest.importorskip("darts")

from strikecast.backtest.predictions import COLUMNS, PredictionSet  # noqa: E402
from strikecast.evaluation.calibration import (  # noqa: E402
    apply_calibrators_per_horizon,
    fit_calibrators_per_horizon,
)
from strikecast.pipeline import composite_stage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "golden" / "results" / "finalhurdle"
DATA_DIR = REPO_ROOT / "data"

TARGET = "act_drone_strike_on_ua"
SEASONALITY = 7

#: `final_hurdle.ipynb` cells 48-49: the five components, per stage, and the
#: `global_<stage>_global_<component>.json` / `per_*_<stage>_global_<component>.csv`
#: files each one produced.
COMPONENTS = ("classifier_raw", "classifier_cal", "regressor", "hurdle_raw", "hurdle_cal")
VIEWS = ("per_region", "per_horizon", "per_region_horizon")

#: Comparison tolerance. The inputs are the stored predictions themselves, so
#: the only source of deviation is the parquet/CSV float round trip.
RTOL = 1e-9


def _require_golden() -> None:
    if not GOLDEN.is_dir():
        pytest.skip(f"no golden hurdle material at {GOLDEN}")


def _frame(name: str) -> pd.DataFrame:
    path = GOLDEN / f"{name}.parquet"
    if not path.is_file():
        pytest.skip(f"missing golden frame {path.name}")
    return pd.read_parquet(path).reset_index(drop=True)


def _stage_frames(stage: str) -> dict[str, pd.DataFrame]:
    """The three channels the notebook saved for one stage."""
    _require_golden()
    return {
        "prob": _frame(f"global_{stage}_classifier_probs"),
        "prob_cal": _frame(f"global_{stage}_classifier_cal_probs"),
        "count": _frame(f"global_{stage}_regressor_preds"),
        "hurdle": _frame(f"global_{stage}_hurdle_preds"),
        "hurdle_cal": _frame(f"global_{stage}_hurdle_cal_preds"),
    }


def _as_prediction_set(frames: dict[str, pd.DataFrame]) -> PredictionSet:
    """Rebuild a three-channel `PredictionSet` from the stored legacy frames.

    The stored frames carry only the six `LEGACY_COLUMNS`; `origin_date` is
    backfilled as the per-fold minimum date, which is what `PredictionSet`
    computes from the first predicted timestamp of the fold.
    """
    blocks = []
    for channel, source in (("prob", "prob"), ("count", "count"), ("hurdle", "hurdle")):
        block = frames[source].rename(columns={"y_prob": "y_pred"}).copy()
        block["origin_date"] = block.groupby("fold")["date"].transform("min")
        block["channel"] = channel
        blocks.append(block.loc[:, list(COLUMNS)])
    frame = (
        pd.concat(blocks, ignore_index=True)
        .sort_values(["channel", "region", "fold", "horizon"], kind="stable")
        .reset_index(drop=True)
    )
    return PredictionSet(frame)


# --------------------------------------------------------------------------- #
# the calibration half: exact, from the stored predictions
# --------------------------------------------------------------------------- #
def test_cv_calibration_is_in_sample_and_matches_the_stored_frame() -> None:
    """F69 / cell 22: the CV view is calibrated by calibrators fit on its own rows."""
    frames = _stage_frames("cv")
    calibrators = fit_calibrators_per_horizon(frames["prob"])
    got = apply_calibrators_per_horizon(frames["prob"], calibrators)

    np.testing.assert_allclose(got, frames["prob_cal"]["y_prob"].to_numpy(), rtol=RTOL)
    # every horizon held both classes, so every horizon has a calibrator
    assert sorted(calibrators) == list(range(1, 8))
    assert all(c is not None for c in calibrators.values())


def test_test_calibration_applies_the_cv_calibrators_out_of_sample() -> None:
    """Cell 27: no leakage -- the calibrators come from the CV rows."""
    cv = _stage_frames("cv")
    test = _stage_frames("test")
    calibrators = fit_calibrators_per_horizon(cv["prob"])
    got = apply_calibrators_per_horizon(test["prob"], calibrators)

    np.testing.assert_allclose(got, test["prob_cal"]["y_prob"].to_numpy(), rtol=RTOL)
    # and refitting on the TEST rows would NOT give the stored frame, which is
    # what makes the out-of-sample claim testable rather than decorative
    leaky = apply_calibrators_per_horizon(
        test["prob"], fit_calibrators_per_horizon(test["prob"])
    )
    assert not np.allclose(leaky, test["prob_cal"]["y_prob"].to_numpy())


@pytest.mark.parametrize("stage", ["cv", "test"])
def test_components_rebuild_the_stored_calibrated_hurdle(stage: str) -> None:
    """Cell 22 / 27: `hurdle_cal = calibrated prob * count`, positionally."""
    frames = _stage_frames(stage)
    preds = _as_prediction_set(frames)
    cv = _stage_frames("cv")
    calibrators = fit_calibrators_per_horizon(cv["prob"])

    prob_rows = composite_stage._as_prob_frame(preds.for_channel("prob").frame)
    calibrated = {"prob": apply_calibrators_per_horizon(prob_rows, calibrators)}
    components, _ = composite_stage._hurdle_components(preds, calibrated)
    by_name = {c.name: c.frame for c in components}

    assert [c.name for c in components] == list(COMPONENTS)

    keys = ["region", "fold", "horizon"]
    expected_cal = frames["hurdle_cal"].sort_values(keys, kind="stable").reset_index(drop=True)
    got_cal = by_name["hurdle_cal"].sort_values(keys, kind="stable").reset_index(drop=True)
    np.testing.assert_allclose(
        got_cal["y_pred"].to_numpy(), expected_cal["y_pred"].to_numpy(), rtol=RTOL
    )

    expected_raw = frames["hurdle"].sort_values(keys, kind="stable").reset_index(drop=True)
    got_raw = by_name["hurdle_raw"].sort_values(keys, kind="stable").reset_index(drop=True)
    np.testing.assert_allclose(
        got_raw["y_pred"].to_numpy(), expected_raw["y_pred"].to_numpy(), rtol=RTOL
    )

    # F65: the count head component is the positive-event-day subset
    positive = frames["count"]["y_true"] > 0
    assert len(by_name["regressor"]) == int(positive.sum()) < len(frames["count"])


# --------------------------------------------------------------------------- #
# the metric half: every stored component file, recomputed
# --------------------------------------------------------------------------- #
def _naive_scales_from_golden(stage: str) -> tuple[dict, dict]:
    """The scales the notebook used, rebuilt from the real panel (F6/F67).

    CV scales on the training count target (cell 17), the test stage on
    train+val (cell 26). This needs `data/`, so the metric comparison skips
    without it.
    """
    if not DATA_DIR.is_dir():
        pytest.skip(f"no input data at {DATA_DIR}")

    from strikecast.config.schema import SeriesConfig, WindowTransformConfig  # noqa: PLC0415
    from strikecast.data.covariates import split_covariates  # noqa: PLC0415
    from strikecast.data.load import load_inputs  # noqa: PLC0415
    from strikecast.data.panel import build_panel_legacy_regressor  # noqa: PLC0415
    from strikecast.data.series import build_bundle  # noqa: PLC0415
    from strikecast.evaluation.metrics import compute_naive_scales  # noqa: PLC0415

    inputs = load_inputs(str(DATA_DIR / "fixed"), str(DATA_DIR / "dataset"))
    result = build_panel_legacy_regressor(inputs, TARGET)
    panel = result.panel
    split = split_covariates(panel, list(result.global_weather_columns), TARGET)
    bundle = build_bundle(
        panel,
        TARGET,
        list(split.past_covariates),
        list(split.future_covariates),
        # Thesis reproduction: pin the LEGACY expdecay7 filter explicitly
        # (audit 2026-09-26 A1/D1; the schema default is the fixed `leaky` filter).
        SeriesConfig(window=WindowTransformConfig(expdecay="legacy_alpha")),
        dict(inputs.activity_by_region),
    )
    if stage == "cv":
        targets = list(bundle.target_train)
    else:
        targets = [
            tr.append(vl)
            for tr, vl in zip(bundle.target_train, bundle.target_val, strict=True)
        ]
    return compute_naive_scales(targets, bundle.region_names, seasonality=SEASONALITY)


def _assert_global_equal(got: dict, path: Path) -> None:
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert set(got) == set(expected), path.name
    for name, value in expected.items():
        mine = got[name]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isnan(float(value)):
                assert math.isnan(float(mine)), f"{path.name}:{name}"
            else:
                assert math.isclose(
                    float(mine), float(value), rel_tol=RTOL, abs_tol=1e-12
                ), f"{path.name}:{name}: {mine} != {value}"
        else:
            assert mine == value, f"{path.name}:{name}"


@pytest.mark.parametrize("stage", ["cv", "test"])
def test_every_stored_component_metric_is_recomputed(stage: str) -> None:
    """The five `global_<stage>_global_<component>.json` files and their views.

    This is the component layout `composite_stage` writes, scored through the
    same `_evaluate_components` the pipeline calls, against the notebook's own
    numbers.
    """
    frames = _stage_frames(stage)
    preds = _as_prediction_set(frames)
    calibrators = fit_calibrators_per_horizon(_stage_frames("cv")["prob"])
    prob_rows = composite_stage._as_prob_frame(preds.for_channel("prob").frame)
    calibrated = {"prob": apply_calibrators_per_horizon(prob_rows, calibrators)}
    components, _ = composite_stage._hurdle_components(preds, calibrated)

    mae_scales, rmse_scales = _naive_scales_from_golden(stage)
    views, component_globals = composite_stage._evaluate_components(
        components, "hurdle_raw", mae_scales, rmse_scales, 0.5
    )

    checked = 0
    for component in COMPONENTS:
        global_path = GOLDEN / f"global_{stage}_global_{component}.json"
        if not global_path.is_file():
            continue
        _assert_global_equal(component_globals[component], global_path)
        checked += 1

        for view in VIEWS:
            csv_path = GOLDEN / f"{view}_{stage}_global_{component}.csv"
            if not csv_path.is_file():
                continue
            expected = pd.read_csv(csv_path)
            got = views[f"{component}@{view}"].reset_index(drop=True)
            if view == "per_region" and component.startswith("classifier"):
                # the `per_region` classification view sorts on F1 and ten
                # regions tie at F1 == 0, so the surviving order is the pandas
                # build's, not the methodology's (Stream A's finding, recorded
                # as a flag). Compare aligned on region instead of on position.
                expected = expected.sort_values("region").reset_index(drop=True)
                got = got.sort_values("region").reset_index(drop=True)
            pd.testing.assert_frame_equal(
                got[expected.columns], expected, check_exact=False, rtol=RTOL
            )
    assert checked == len(COMPONENTS), f"only {checked} components had golden files"


# --------------------------------------------------------------------------- #
# the prediction half: opt-in, fold-limited, reported not asserted
# --------------------------------------------------------------------------- #
RUN_REAL = os.environ.get("STRIKECAST_GOLDEN_HURDLE") == "1"
REAL_FOLDS = int(os.environ.get("STRIKECAST_GOLDEN_HURDLE_FOLDS", "3"))


@pytest.mark.slow
@pytest.mark.skipif(
    not RUN_REAL,
    reason="set STRIKECAST_GOLDEN_HURDLE=1 to run the real hurdle CV stage "
    "(fits a Self-Paced Ensemble and a CatBoost per retrain)",
)
def test_real_hurdle_cv_stage_matches_the_golden_schedule(tmp_path, capsys) -> None:
    """Run the real pipeline for `REAL_FOLDS` folds and compare what is comparable.

    ASSERTED: the fold schedule (dates and horizons), the three channels, the
    per-channel `y_true` -- binary for `prob`, counts for `count` / `hurdle` --
    and the row count per fold. These are properties of the data and the
    schedule, not of the fitted models, so they must match exactly.

    REPORTED, not asserted: the prediction values. The thesis' top-100 feature
    sets are unrecoverable (F16, see the module docstring), so a fresh selection
    trains different models. The deviation is printed so a reader can see how
    far apart the two runs are.
    """
    if not DATA_DIR.is_dir():
        pytest.skip(f"no input data at {DATA_DIR}")

    from strikecast.config.loader import load_experiment  # noqa: PLC0415
    from strikecast.pipeline.data_stage import prepare_composite_data  # noqa: PLC0415
    from strikecast.store import RunKey, RunStore  # noqa: PLC0415

    # thesis mode (legacy expdecay7, audit A1); the heads are selected here, as
    # `strikecast featsel` would, because the config requires a cached selection
    cfg = load_experiment(
        "hurdle", overrides=[f"++store.root={tmp_path / 'runs'}", "legacy=hurdle"]
    )
    store = RunStore(cfg.store.root)
    data = prepare_composite_data(cfg, store, compute_features=True)

    outcome = composite_stage.run_composite_stage(
        cfg, "hurdle", "global", 42, "cv", data, store=store, max_folds=REAL_FOLDS
    )
    assert outcome.n_folds == REAL_FOLDS

    key = RunKey(cfg.name, "hurdle", "global", 42)
    got = store.load_predictions(key, "cv", legacy_order=True).frame
    assert set(got["channel"]) == {"prob", "count", "hurdle"}

    golden = _stage_frames("cv")
    keys = ["region", "fold", "horizon"]
    for channel, source in (("prob", "prob"), ("count", "count"), ("hurdle", "hurdle")):
        mine = (
            got[got["channel"] == channel]
            .sort_values(keys, kind="stable")
            .reset_index(drop=True)
        )
        theirs = (
            golden[source][golden[source]["fold"] < REAL_FOLDS]
            .sort_values(keys, kind="stable")
            .reset_index(drop=True)
        )
        assert len(mine) == len(theirs), channel
        pd.testing.assert_frame_equal(
            mine[["region", "fold", "horizon", "date", "y_true"]],
            theirs[["region", "fold", "horizon", "date", "y_true"]],
            check_dtype=False,
        )
        value = "y_prob" if source == "prob" else "y_pred"
        deviation = float(
            np.nanmax(np.abs(mine["y_pred"].to_numpy() - theirs[value].to_numpy()))
        )
        with capsys.disabled():
            print(f"  level H {channel:7s}: max |new - golden| = {deviation:.6f}")
