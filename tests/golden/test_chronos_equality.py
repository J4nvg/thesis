"""Golden tests for the Chronos-2 family against ``golden/results/chronos2/``.

Two halves, because they cost three orders of magnitude apart.

**Cheap half — level F on the metric port (no AutoGluon, seconds).**
``golden/results/chronos2/predictions_long_*.parquet`` are the thesis' own
predictions.  Pushing them back through
:func:`strikecast.evaluation.metrics.chronos_evaluate_long`, with the naive
scales rebuilt from the real panel, must reproduce ``per_region_*.csv``,
``per_horizon_*.csv``, ``per_region_horizon_*.csv``, ``global_*.json`` and
``leaderboard.csv`` to floating-point noise.  This needs only the panel and
pandas, so it runs in the MAIN environment as well.

**Expensive half — level E/F on the model path (needs ``envs/autogluon``).**
Loads the thesis' own saved predictors (``golden/checkpoints/chronos2_zeroshot``
and ``.../chronos2_best``), replays the first ``CHRONOS_GOLDEN_FOLDS`` folds of
the shared schedule through :func:`strikecast.models.chronos.run_backtest`, and
compares the long frame with the stored parquet.

Running the expensive half (``envs/autogluon`` installs ``strikecast`` as an
editable path, so no ``PYTHONPATH`` is needed any more)::

    cd <repo root>
    uv sync --project envs/autogluon          # once; installs pytest too
    CHRONOS_GOLDEN_FOLDS=2 \\
        uv run --project envs/autogluon python -m pytest \\
        tests/golden/test_chronos_equality.py -q -p no:cacheprovider

**Pipeline half (audit C15, ``slow``).** :func:`test_the_pipeline_fits_what_the_thesis_fit`
runs ``strikecast.pipeline.run_stage.run_stage`` for ``chronos2_zero_shot``
with ``max_folds=1``: it FITS a fresh zero-shot predictor from the pipeline's own
data stage (about 75 s on an Apple-silicon CPU; needs the Chronos-2 weights,
downloaded from Hugging Face on first use) and compares fold 0 with the thesis's
predictions and the internal validation MASE with the thesis log (1.1459).

One fold costs about 30 s (zero-shot) / 40 s (fine-tuned) on an Apple-silicon
CPU, so the default of 2 folds per model keeps the file inside a few minutes;
``CHRONOS_GOLDEN_FOLDS=164`` reproduces the whole run in roughly two hours.

Tolerance (flag F134)
---------------------
The stored predictions were produced on a CUDA node; a replay on CPU/MPS agrees
to ``max |Δ| ≈ 5e-3`` and ``mean |Δ| ≈ 1e-4`` on a target whose values are
counts, with a Pearson correlation of 0.9999999 over 280 points.  That is
float non-determinism in a 100M-parameter transformer, not a port difference,
so this is a **level-F family tolerance** (plan §6, "family-specific tolerance,
recorded in the golden report"), not level E's 1e-6.  The metric half, which
does no model arithmetic, is held to 1e-9.  ``y_true`` and every date are
compared exactly in both halves -- those come from the panel and any drift
there would be a real bug.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "golden" / "results" / "chronos2"
CHECKPOINTS = REPO_ROOT / "golden" / "checkpoints"

TARGET = "act_drone_strike_on_ua"
HORIZON = 7
TRAIN_FRAC = 0.70
TRAIN_VAL_END = 0.7999999999999999  # TRAIN_FRAC + VAL_FRAC, flag F17
SEASONALITY = 7

MODELS = ("chronos2_zero_shot", "chronos2_fine_tuned")
PREDICTOR_DIRS = {
    "chronos2_zero_shot": CHECKPOINTS / "chronos2_zeroshot",
    "chronos2_fine_tuned": CHECKPOINTS / "chronos2_best",
}

#: Level-F family tolerance for a replayed Chronos-2 prediction (see F134).
PRED_ATOL = 2e-2
PRED_MEAN_ATOL = 1e-3
PRED_MIN_CORR = 0.9999
#: The metric port does no model arithmetic.
METRIC_ATOL = 1e-9

#: How many folds of the 164 to replay. 2 keeps the file at a few minutes.
N_FOLDS = int(os.environ.get("CHRONOS_GOLDEN_FOLDS", "2"))

pytestmark = pytest.mark.golden


def _has_autogluon() -> bool:
    try:
        return importlib.util.find_spec("autogluon.timeseries") is not None
    except ModuleNotFoundError:
        return False


HAS_AUTOGLUON = _has_autogluon()
needs_autogluon = pytest.mark.skipif(
    not HAS_AUTOGLUON,
    reason="needs the AutoGluon environment: uv run --project envs/autogluon ...",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def golden_dir() -> Path:
    if not (GOLDEN / "leaderboard.csv").exists():
        pytest.skip(f"golden Chronos results not present under {GOLDEN}")
    return GOLDEN


@pytest.fixture(scope="module")
def chronos_frames(inputs):
    """The AutoGluon long frames of the real panel (``_chronos2.py`` §2-§3)."""
    from strikecast.data.autogluon import panel_to_long_frames
    from strikecast.data.panel import build_panel_legacy_chronos

    result = build_panel_legacy_chronos(inputs, TARGET)
    frames = panel_to_long_frames(result.panel)
    return frames, result


@pytest.fixture(scope="module")
def scales(chronos_frames):
    """``MAE_SCALES`` / ``RMSE_SCALES`` of ``_chronos2.py:295-300``."""
    from strikecast.data.autogluon import scales_for_stage

    frames, _ = chronos_frames
    return scales_for_stage(frames, TARGET, train_frac=TRAIN_FRAC, seasonality=SEASONALITY)


@pytest.fixture(scope="module")
def future_covariates(chronos_frames):
    from strikecast.data.covariates import split_covariates

    _, result = chronos_frames
    return list(
        split_covariates(result.panel, result.global_weather_columns, TARGET).future_covariates
    )


def _golden_predictions(model: str) -> pd.DataFrame:
    return pd.read_parquet(GOLDEN / f"predictions_long_{model}.parquet")


# --------------------------------------------------------------------------- #
# 1. the panel and the frame the thesis actually fit on
# --------------------------------------------------------------------------- #
def test_the_frame_has_the_shape_the_stored_sidecar_records(
    golden_dir: Path, chronos_frames, future_covariates
) -> None:
    """``best_params.json`` records 27 known covariates and 83 past ones."""
    frames, _ = chronos_frames
    sidecar = json.loads((CHECKPOINTS / "chronos2_best" / "best_params.json").read_text())

    assert sidecar["target"] == TARGET
    assert sidecar["prediction_length"] == HORIZON
    assert sidecar["num_val_windows"] == 3
    assert set(sidecar["future_covariates"]) == set(future_covariates)
    assert len(future_covariates) == 27

    # F123: the frame carries one column MORE than `n_past_covariates` records,
    # the stray `index` column of F22, which AutoGluon treats as a past covariate.
    n_frame_cols = frames.values.shape[1]
    assert n_frame_cols == 1 + len(future_covariates) + sidecar["n_past_covariates"] + 1
    assert "index" in frames.values.columns
    assert len(frames.item_ids) == 20


def test_the_split_and_the_schedule_are_the_recorded_ones(chronos_frames) -> None:
    """Level C for this family, re-derived here: 847 steps, t0 = 677, 164 folds."""
    from strikecast.backtest.schedule import fold_schedule
    from strikecast.data.autogluon import test_split_length

    frames, _ = chronos_frames
    panel = frames.panel()
    n = int(panel.num_timesteps_per_item().min())
    assert n == 847

    # F34/F126: the AutoGluon fit window is 678 steps, one MORE than the first
    # rolling origin, so it overlaps fold 0's first predicted day.
    test_size = test_split_length(n, 0.20)
    assert test_size == 169
    assert n - test_size == 678

    timestamps = panel.loc[frames.item_ids[0]].index
    folds = fold_schedule(
        n_total=n,
        start_frac=TRAIN_VAL_END,
        horizon=HORIZON,
        predict_stride=1,
        retrain_stride=None,  # the fixed-predictor shape
        time_index=timestamps,
    )
    assert len(folds) == 164
    assert folds[0].t0 == 677
    assert not any(f.retrain for f in folds)
    assert folds[0].cutoff == pd.Timestamp("2024-08-05")
    assert folds[-1].cutoff == pd.Timestamp("2025-01-15")


def test_the_golden_predictions_line_up_with_that_schedule(golden_dir: Path) -> None:
    for model in MODELS:
        frame = _golden_predictions(model)
        assert len(frame) == 22960  # 164 folds x 20 regions x 7 horizons
        assert frame["fold"].nunique() == 164
        assert frame["region"].nunique() == 20
        assert sorted(frame["horizon"].unique()) == list(range(1, HORIZON + 1))
        assert frame["date"].min() == pd.Timestamp("2024-08-05")
        assert frame["date"].max() == pd.Timestamp("2025-01-21")


# --------------------------------------------------------------------------- #
# 2. cheap half: the metric port on the stored predictions (level F)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", MODELS)
def test_global_metrics_are_reproduced_from_the_stored_predictions(
    golden_dir: Path, scales, model: str
) -> None:
    from strikecast.evaluation.metrics import chronos_evaluate_long

    mae_scales, rmse_scales = scales
    views = chronos_evaluate_long(_golden_predictions(model), mae_scales, rmse_scales)
    stored = json.loads((golden_dir / f"global_{model}.json").read_text())

    assert set(views["global"]) == set(stored)
    for key, expected in stored.items():
        assert views["global"][key] == pytest.approx(expected, abs=METRIC_ATOL), key


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("view", ["per_region", "per_horizon", "per_region_horizon"])
def test_the_per_view_csvs_are_reproduced(golden_dir: Path, scales, model: str, view: str) -> None:
    from strikecast.evaluation.metrics import chronos_evaluate_long

    mae_scales, rmse_scales = scales
    got = chronos_evaluate_long(_golden_predictions(model), mae_scales, rmse_scales)[view]
    expected = pd.read_csv(golden_dir / f"{view}_{model}.csv")

    assert list(got.columns) == list(expected.columns)
    assert len(got) == len(expected)
    key = [c for c in ("region", "horizon") if c in expected.columns]
    a = got.sort_values(key).reset_index(drop=True)
    b = expected.sort_values(key).reset_index(drop=True)
    for column in expected.columns:
        if column in key:
            assert list(a[column]) == list(b[column])
        else:
            np.testing.assert_allclose(
                a[column].to_numpy(dtype=float),
                b[column].to_numpy(dtype=float),
                atol=METRIC_ATOL,
                rtol=1e-9,
                err_msg=f"{view}.{column}",
            )


def test_the_naive_scales_are_the_train_only_ones(golden_dir: Path, scales) -> None:
    """A wrong scale set would move MASE but not MAE; pin it against the stored MASE."""
    from strikecast.evaluation.metrics import chronos_evaluate_long

    mae_scales, rmse_scales = scales
    assert len(mae_scales) == 20
    views = chronos_evaluate_long(
        _golden_predictions("chronos2_zero_shot"), mae_scales, rmse_scales
    )
    stored = json.loads((golden_dir / "global_chronos2_zero_shot.json").read_text())
    assert views["global"]["MASE_mean"] == pytest.approx(stored["MASE_mean"], abs=METRIC_ATOL)


def test_the_leaderboard_is_reproduced(golden_dir: Path, scales) -> None:
    """``_chronos2.py:751-786``: four rows, sorted by ``MASE_mean`` ascending."""
    from strikecast.evaluation.metrics import chronos_evaluate_long, skill

    mae_scales, rmse_scales = scales
    expected = pd.read_csv(golden_dir / "leaderboard.csv")

    globals_by_model = {}
    for model in expected["model"]:
        path = golden_dir / f"predictions_long_{model}.parquet"
        if path.exists():
            globals_by_model[model] = chronos_evaluate_long(
                pd.read_parquet(path), mae_scales, rmse_scales
            )["global"]
        else:  # the two naive floors, whose frames the diff family also stores
            globals_by_model[model] = json.loads(
                (golden_dir / f"global_{model}.json").read_text()
            )

    ref = globals_by_model["naive_weekly"]
    rows = []
    for model, g in globals_by_model.items():
        rows.append(
            {
                "model": model,
                "MAE": g["MAE"],
                "RMSE": g["RMSE"],
                "MedAE": g["MedAE"],
                "ME": g["ME"],
                "ZeroAcc": g["ZeroAcc"],
                "MASE_mean": g["MASE_mean"],
                "MASE_median": g["MASE_median"],
                "RMSSE_mean": g["RMSSE_mean"],
                "SkillRMSE": skill(g["RMSE"], ref["RMSE"]),
                "SkillMAE": skill(g["MAE"], ref["MAE"]),
            }
        )
    got = (
        pd.DataFrame(rows)
        .sort_values("MASE_mean", ascending=True)
        .reset_index(drop=True)
    )

    assert list(got.columns) == list(expected.columns)
    assert list(got["model"]) == list(expected["model"])
    for column in expected.columns[1:]:
        np.testing.assert_allclose(
            got[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            atol=1e-9,
            rtol=1e-9,
            err_msg=column,
        )


# --------------------------------------------------------------------------- #
# 3. expensive half: replay the stored predictors (level E/F)
# --------------------------------------------------------------------------- #
@needs_autogluon
@pytest.mark.parametrize("model", MODELS)
def test_replaying_the_stored_predictor_reproduces_the_predictions(
    golden_dir: Path, chronos_frames, future_covariates, model: str
) -> None:
    from strikecast.data.autogluon import to_timeseries_dataframe
    from strikecast.models import chronos
    from strikecast.models.spec import RunContext, get_spec

    predictor_dir = PREDICTOR_DIRS[model]
    if not (predictor_dir / "predictor.pkl").exists():
        pytest.skip(f"no saved predictor at {predictor_dir}")

    frames, _ = chronos_frames
    tsdf = to_timeseries_dataframe(frames)
    regions = list(frames.item_ids)

    forecaster = chronos.make_forecaster(
        get_spec(model, "chronos2"),
        {},
        RunContext(seed=42),
        data=tsdf,
        target=TARGET,
        known_covariates=future_covariates,
        region_names=regions,
        predictor_dir=predictor_dir,
    )
    targets = chronos.level_targets_from_frames(tsdf, TARGET, regions)
    got = chronos.run_backtest(
        forecaster,
        targets,
        start_frac=TRAIN_VAL_END,
        horizon=HORIZON,
        max_folds=N_FOLDS,
    ).legacy_frame()

    expected = _golden_predictions(model)
    expected = expected[expected["fold"] < N_FOLDS]

    key = ["region", "fold", "horizon"]
    a = got.sort_values(key).reset_index(drop=True)
    b = expected.sort_values(key).reset_index(drop=True)

    assert len(a) == len(b) == N_FOLDS * 20 * HORIZON
    assert list(a["region"]) == list(b["region"])
    pd.testing.assert_series_equal(a["date"], b["date"], check_dtype=False)
    np.testing.assert_allclose(
        a["y_true"].to_numpy(dtype=float), b["y_true"].to_numpy(dtype=float), atol=0
    )

    delta = np.abs(a["y_pred"].to_numpy(dtype=float) - b["y_pred"].to_numpy(dtype=float))
    corr = float(
        np.corrcoef(a["y_pred"].to_numpy(dtype=float), b["y_pred"].to_numpy(dtype=float))[0, 1]
    )
    assert delta.max() < PRED_ATOL, f"{model}: max |Δ| = {delta.max():.3e} (F134)"
    assert delta.mean() < PRED_MEAN_ATOL, f"{model}: mean |Δ| = {delta.mean():.3e}"
    assert corr > PRED_MIN_CORR, f"{model}: corr = {corr:.9f}"


@needs_autogluon
def test_the_forecaster_never_refits(chronos_frames, future_covariates) -> None:
    """``retrains=False`` plus ``retrain_stride=None``: ``fit`` is unreachable."""
    from strikecast.data.autogluon import to_timeseries_dataframe
    from strikecast.models import chronos
    from strikecast.models.spec import RunContext, get_spec

    predictor_dir = PREDICTOR_DIRS["chronos2_zero_shot"]
    if not (predictor_dir / "predictor.pkl").exists():
        pytest.skip(f"no saved predictor at {predictor_dir}")

    frames, _ = chronos_frames
    tsdf = to_timeseries_dataframe(frames)
    regions = list(frames.item_ids)
    forecaster = chronos.make_forecaster(
        get_spec("chronos2_zero_shot", "chronos2"),
        {},
        RunContext(seed=42),
        data=tsdf,
        target=TARGET,
        known_covariates=future_covariates,
        region_names=regions,
        predictor_dir=predictor_dir,
    )
    assert forecaster.retrains is False
    assert forecaster.use_cache is False  # F124

    targets = chronos.level_targets_from_frames(tsdf, TARGET, regions)
    chronos.run_backtest(
        forecaster, targets, start_frac=TRAIN_VAL_END, horizon=HORIZON, max_folds=1
    )
    assert forecaster.n_predicts == 1


# --------------------------------------------------------------------------- #
# 4. the pipeline path (audit C15): data stage -> run_stage -> run store
# --------------------------------------------------------------------------- #
#: `-score_val` of `Chronos2ZeroShot` in golden/checkpoints/chronos2_zeroshot/logs.
THESIS_ZERO_SHOT_VAL_MASE = 1.1459


@needs_autogluon
@pytest.mark.slow
def test_the_pipeline_fits_what_the_thesis_fit(golden_dir: Path, tmp_path: Path) -> None:
    from strikecast.config.loader import load_experiment
    from strikecast.pipeline import data_stage, run_stage
    from strikecast.store import RunKey, RunStore

    cfg = load_experiment("chronos2", ["tracking=noop", f"++store.root={tmp_path}"])
    store = RunStore(tmp_path)
    data = data_stage.prepare_data(cfg, store)
    out = run_stage.run_stage(
        cfg, "chronos2_zero_shot", "global", 42, "test", data, store=store, max_folds=1
    )
    assert out.n_folds == 1 and out.n_rows == 20 * HORIZON

    key = RunKey("chronos2", "chronos2_zero_shot", "global", 42)
    board = json.loads((store.artifacts_dir(key) / "internal_leaderboard.json").read_text())
    assert -board[0]["score_val"] == pytest.approx(THESIS_ZERO_SHOT_VAL_MASE, abs=5e-4)

    got = store.load_predictions(key, "test", legacy_order=True).legacy_frame()
    expected = _golden_predictions("chronos2_zero_shot")
    expected = expected[expected["fold"] == 0]
    cols = ["region", "fold", "horizon"]
    a = got.sort_values(cols).reset_index(drop=True)
    b = expected.sort_values(cols).reset_index(drop=True)
    assert list(a["region"]) == list(b["region"])
    np.testing.assert_allclose(a["y_true"].to_numpy(float), b["y_true"].to_numpy(float), atol=0)
    delta = np.abs(a["y_pred"].to_numpy(float) - b["y_pred"].to_numpy(float))
    assert delta.max() < PRED_ATOL, f"max |d| = {delta.max():.3e} (F134)"
    assert delta.mean() < PRED_MEAN_ATOL
