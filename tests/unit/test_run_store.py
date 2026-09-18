"""Unit tests for `strikecast.store.run_store`.

No models, no engine: the fold results are built by hand from a fake backtest
loop that uses the real `fold_schedule`, so the retrain boundaries the
`PersistHook` flushes on are the ones the engine would hand it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from darts import TimeSeries  # noqa: E402

from strikecast.backtest.predictions import COLUMNS, PredictionSet  # noqa: E402
from strikecast.backtest.protocols import SINGLE_CHANNEL, FoldResult  # noqa: E402
from strikecast.backtest.schedule import fold_schedule  # noqa: E402
from strikecast.store import (  # noqa: E402
    PartWriter,
    PersistHook,
    RunKey,
    RunStore,
    StageState,
    stage_hash,
)

N_STEPS = 20
HORIZON = 3
RETRAIN_STRIDE = 3
START_FRAC = 0.5
REGIONS = ["r0", "r1"]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _series(offset: float) -> TimeSeries:
    index = pd.date_range("2024-01-01", periods=N_STEPS, freq="D")
    values = np.arange(N_STEPS, dtype=float) + offset
    return TimeSeries.from_times_and_values(index, values)


@pytest.fixture
def actuals() -> list[TimeSeries]:
    return [_series(0.0), _series(100.0)]


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "runs")


@pytest.fixture
def run() -> RunKey:
    return RunKey("count", "lightgbm_poisson", "global", 42)


def _folds(reference: TimeSeries):
    return fold_schedule(
        len(reference),
        start_frac=START_FRAC,
        horizon=HORIZON,
        predict_stride=1,
        retrain_stride=RETRAIN_STRIDE,
        time_index=reference.time_index,
    )


def _fake_loop(actuals: list[TimeSeries], *, n_folds: int | None = None) -> list[FoldResult]:
    """The folds an `ExpandingWindowBacktest` would produce, without a model.

    Predictions are the actual values plus 0.5, so every row is identifiable.
    """
    folds = _folds(actuals[0])
    results = []
    for fold in folds[:n_folds]:
        preds = []
        for actual in actuals:
            window = actual[fold.t0 : fold.t0 + HORIZON]
            preds.append(
                TimeSeries.from_times_and_values(
                    window.time_index, window.values().ravel() + 0.5
                )
            )
        results.append(FoldResult(fold=fold, predictions={SINGLE_CHANNEL: preds}))
    return results


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def test_layout_matches_plan_section_5_3(store, run):
    root = store.root
    assert store.experiment_dir("count") == root / "count"
    assert store.shared_path("count", "panel", "abc") == root / "count/shared/panel.abc.parquet"
    assert store.shared_path("count", "series", "abc") == root / "count/shared/series.abc"
    assert (
        store.shared_path("count", "feature_selection", "abc")
        == root / "count/shared/feature_selection.abc.json"
    )
    assert store.tuning_dir("count", "lightgbm_poisson") == root / "count/tuning/lightgbm_poisson"
    assert store.run_dir("count", "lightgbm_poisson", "global", 42) == (
        root / "count/lightgbm_poisson/global/seed=42"
    )
    assert store.resolve(run) == store.run_dir("count", "lightgbm_poisson", "global", 42)
    assert store.predictions_dir(run, "cv") == store.resolve(run) / "cv/predictions"
    assert store.metrics_dir(run, "test") == store.resolve(run) / "test/metrics"
    assert store.artifacts_dir(run) == store.resolve(run) / "artifacts"
    assert store.report_dir("count") == root / "count/report"


def test_path_segments_are_validated(store):
    with pytest.raises(ValueError, match="unsafe experiment"):
        store.experiment_dir("../escape")
    with pytest.raises(ValueError, match="unsafe stage"):
        store.stage_dir(RunKey("count", "m", "global", 1), "cv/../..")


def test_config_and_env_round_trip(store, run):
    cfg = {"model": "lightgbm_poisson", "backtest": {"horizon": 7, "retrain_stride": 7}}
    env = {"PYTHONHASHSEED": "0", "python": "3.13.0", "darts": "0.43.0"}
    store.write_config(run, cfg)
    store.write_env(run, env)
    assert (store.resolve(run) / "config.yaml").exists()
    assert (store.resolve(run) / "env.json").exists()
    assert store.read_config(run) == cfg
    assert store.read_env(run) == env


# --------------------------------------------------------------------------- #
# stage hash and state
# --------------------------------------------------------------------------- #
def test_stage_hash_is_order_insensitive_but_seed_sensitive():
    cfg = {"horizon": 7, "model": "lightgbm_poisson"}
    reordered = {"model": "lightgbm_poisson", "horizon": 7}
    assert stage_hash(cfg, ["panelhash"], 42) == stage_hash(reordered, ["panelhash"], 42)
    assert stage_hash(cfg, ["panelhash"], 42) != stage_hash(cfg, ["panelhash"], 1)
    assert stage_hash(cfg, ["panelhash"], 42) != stage_hash(cfg, ["otherhash"], 42)
    assert stage_hash(cfg, [], 42) != stage_hash({"horizon": 8, "model": "x"}, [], 42)


def test_stage_lifecycle_and_is_complete(store, run):
    h = stage_hash({"horizon": 7}, ["panel"], 42)
    assert store.read_state(run, "cv") is None
    assert store.is_complete(run, "cv", h) is False

    state = store.start_stage(run, "cv", h)
    assert state.status == "running"
    assert state.started is not None
    assert store.is_complete(run, "cv", h) is False

    store.complete_stage(run, "cv")
    assert store.is_complete(run, "cv", h) is True
    assert store.is_complete(run, "cv", "another-hash") is False
    assert store.is_complete(run, "test", h) is False

    failed = store.fail_stage(run, "test", "boom")
    assert failed.status == "failed"
    assert failed.error == "boom"
    # the two stages live side by side in one state.json
    assert set(store.read_states(run)) == {"cv", "test"}


def test_changed_stage_hash_discards_old_parts(store, run, actuals):
    old = stage_hash({"horizon": 7}, [], 42)
    store.start_stage(run, "cv", old)
    hook = PersistHook(store, run, "cv", RETRAIN_STRIDE, actuals=actuals, region_names=REGIONS)
    for result in _fake_loop(actuals):
        hook.on_fold(result, {})
    hook.close()
    assert store.part_paths(run, "cv")

    store.start_stage(run, "cv", stage_hash({"horizon": 8}, [], 42))
    assert store.part_paths(run, "cv") == []
    assert store.read_state(run, "cv").parts == []


def test_state_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        StageState.from_dict({"stage": "cv", "bogus": 1})


# --------------------------------------------------------------------------- #
# PersistHook / PartWriter
# --------------------------------------------------------------------------- #
def test_persist_hook_writes_one_part_per_retrain_window(store, run, actuals):
    results = _fake_loop(actuals)
    retrains = [r.fold.index for r in results if r.fold.retrain]
    assert retrains == [0, 3, 6], "fixture must exercise three retrains"

    hook = PersistHook(store, run, "cv", RETRAIN_STRIDE, actuals=actuals, region_names=REGIONS)
    for result in results:
        hook.on_fold(result, {})
    hook.close()

    names = [p.name for p in store.part_paths(run, "cv")]
    assert names == [
        "part-000000-000002.parquet",
        "part-000003-000005.parquet",
        "part-000006-000007.parquet",
    ]

    state = store.read_state(run, "cv")
    assert state.folds_done == len(results)
    assert [(p["from"], p["to"]) for p in state.parts] == [(0, 2), (3, 5), (6, 7)]
    assert state.last_retrain_fold is None  # final flush opens no new part


def test_loaded_predictions_have_global_fold_numbers(store, run, actuals):
    results = _fake_loop(actuals)
    hook = PersistHook(store, run, "cv", RETRAIN_STRIDE, actuals=actuals, region_names=REGIONS)
    for result in results:
        hook.on_fold(result, {})
    hook.close()

    loaded = store.load_predictions(run, "cv")
    assert list(loaded.frame.columns) == list(COLUMNS)
    assert sorted(loaded.frame["fold"].unique().tolist()) == list(range(len(results)))
    assert len(loaded) == len(REGIONS) * len(results) * HORIZON
    assert loaded.channels == (SINGLE_CHANNEL,)


def test_legacy_order_reproduces_a_single_shot_prediction_set(store, run, actuals):
    results = _fake_loop(actuals)
    hook = PersistHook(store, run, "cv", RETRAIN_STRIDE, actuals=actuals, region_names=REGIONS)
    for result in results:
        hook.on_fold(result, {})
    hook.close()

    # what the legacy collector would have produced from the whole run at once
    per_region = [[] for _ in REGIONS]
    for result in results:
        for r_idx, pred in enumerate(result.predictions[SINGLE_CHANNEL]):
            per_region[r_idx].append(pred)
    one_shot = PredictionSet.from_fold_preds(actuals, per_region, REGIONS)

    loaded = store.load_predictions(run, "cv", legacy_order=True)
    pd.testing.assert_frame_equal(
        loaded.frame.reset_index(drop=True),
        one_shot.frame.reset_index(drop=True),
        check_dtype=False,
    )


def test_resume_point_is_the_last_written_retrain_boundary(store, run, actuals):
    assert store.resume_point(run, "cv") is None  # nothing started

    # a crash after fold 6 has been reported but before the run finished
    hook = PersistHook(store, run, "cv", RETRAIN_STRIDE, actuals=actuals, region_names=REGIONS)
    store.start_stage(run, "cv", "h")
    for result in _fake_loop(actuals, n_folds=7):
        hook.on_fold(result, {})

    state = store.read_state(run, "cv")
    assert [(p["from"], p["to"]) for p in state.parts] == [(0, 2), (3, 5)]
    assert state.folds_done == 6
    assert state.last_retrain_fold == 6
    assert store.resume_point(run, "cv") == 6
    # the resume point is a retrain fold, so restarting there refits on exactly
    # the data the crashed run refit on
    assert _folds(actuals[0])[6].retrain is True

    store.complete_stage(run, "cv")
    assert store.resume_point(run, "cv") is None


def test_persist_hook_rejects_mismatched_region_names(store, run, actuals):
    with pytest.raises(ValueError, match="region names"):
        PersistHook(store, run, "cv", 7, actuals=actuals, region_names=["only-one"])


def test_part_writer_rejects_empty_range(store, run):
    writer = PartWriter(store, run, "cv")
    with pytest.raises(ValueError, match="empty part range"):
        writer.write(PredictionSet.concat([]), 5, 4)


# --------------------------------------------------------------------------- #
# atomicity
# --------------------------------------------------------------------------- #
def test_crash_mid_write_leaves_no_partial_part(store, run, actuals, monkeypatch):
    results = _fake_loop(actuals, n_folds=3)
    per_region = [[] for _ in REGIONS]
    for result in results:
        for r_idx, pred in enumerate(result.predictions[SINGLE_CHANNEL]):
            per_region[r_idx].append(pred)
    predictions = PredictionSet.from_fold_preds(actuals, per_region, REGIONS)

    writer = PartWriter(store, run, "cv")
    real_to_parquet = pd.DataFrame.to_parquet

    def exploding_to_parquet(self, path=None, *args, **kwargs):
        # simulate a process killed halfway: bytes on disk, then death
        from pathlib import Path as _P

        _P(path).write_bytes(b"PAR1-partial-garbage")
        raise RuntimeError("SIGKILL")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", exploding_to_parquet)
    with pytest.raises(RuntimeError, match="SIGKILL"):
        writer.write(predictions, 0, 2)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", real_to_parquet)
    directory = store.predictions_dir(run, "cv")
    assert (directory / "part-000000-000002.parquet").exists() is False
    assert list(directory.iterdir()) == []  # not even a stray .tmp
    assert store.part_paths(run, "cv") == []
    assert store.read_state(run, "cv") is None  # state is written only after the parquet
    assert store.resume_point(run, "cv") is None

    # and the retry lands normally
    writer.write(predictions, 0, 2)
    assert (directory / "part-000000-000002.parquet").exists()
    assert store.read_state(run, "cv").folds_done == 3


# --------------------------------------------------------------------------- #
# metrics and artifacts
# --------------------------------------------------------------------------- #
def test_write_metrics_uses_the_legacy_file_names(store, run):
    views = {
        "global": {"MAE": 1.0, "RMSSE_mean": np.float64(0.9), "n": np.int64(10)},
        "per_region": pd.DataFrame({"region": ["r0"], "MASE": [1.1]}),
        "per_horizon": pd.DataFrame({"horizon": [1, 2], "MASE": [1.0, 2.0]}),
        "per_region_horizon": pd.DataFrame({"region": ["r0"], "horizon": [1], "MASE": [1.0]}),
        "per_activity_level": pd.DataFrame({"activity_level": [1], "MASE": [1.0]}),
        "per_activity_horizon": pd.DataFrame(
            {"activity_level": [1], "horizon": [1], "MASE": [1.0]}
        ),
    }
    written = store.write_metrics(run, "cv", views)
    directory = store.metrics_dir(run, "cv")
    assert sorted(p.name for p in directory.iterdir()) == [
        "global.json",
        "per_activity_horizon.csv",
        "per_activity_level.csv",
        "per_horizon.csv",
        "per_region.csv",
        "per_region_horizon.csv",
    ]
    assert written["global"].name == "global.json"

    payload = json.loads((directory / "global.json").read_text())
    assert payload == {"MAE": 1.0, "RMSSE_mean": 0.9, "n": 10}
    read_back = store.read_metrics(run, "cv")
    pd.testing.assert_frame_equal(read_back["per_horizon"], views["per_horizon"])


def test_write_metrics_accepts_extra_named_views(store, run):
    # F65: the hurdle family scores the count head over two row populations
    store.write_metrics(
        run,
        "test",
        {
            "global": {"MASE": 1.0},
            "count_head@positive_days": pd.DataFrame({"horizon": [1], "MASE": [5.93]}),
        },
    )
    assert (store.metrics_dir(run, "test") / "count_head@positive_days.csv").exists()


def test_write_metrics_rejects_a_non_frame_view(store, run):
    with pytest.raises(TypeError, match="must be a DataFrame"):
        store.write_metrics(run, "cv", {"per_region": {"not": "a frame"}})


def test_write_artifact_json_and_parquet_only(store, run):
    frame = pd.DataFrame({"feature": ["a"], "gain": [1.0]})
    parquet_path = store.write_artifact(run, "gain_importance", frame)
    json_path = store.write_artifact(run, "calibrators.json", {"h1": {"a": 1.0, "b": 0.0}})

    assert parquet_path.name == "gain_importance.parquet"
    assert json_path.name == "calibrators.json"
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)
    assert json.loads(json_path.read_text()) == {"h1": {"a": 1.0, "b": 0.0}}

    with pytest.raises(ValueError, match="never pickle"):
        store.write_artifact(run, "study.pkl", {"a": 1})
    with pytest.raises(TypeError, match="must be a DataFrame"):
        store.write_artifact(run, "x.parquet", {"a": 1})
