"""P4 acceptance: the pipeline actually drives a tracker (plan §5.5, §8 P4).

`tests/unit/test_tracking.py` pins what `strikecast.tracking` does when it is
called; this file pins that the PIPELINE calls it, on the synthetic panel of
`tests/unit/test_pipeline_stages.py` and with a fake `wandb` module injected
into `sys.modules`. No network, no real wandb, no GPU.

The smoke stage is deliberately two folds (`predict_stride=5` over the six-fold
CV schedule, `retrain_stride=1` so both folds are mirrored), which is exactly
the "smoke-tested on a two-fold run" of §8 P4. What is asserted is the whole
mirror: the run name, group and job type, the per-fold steps, the tables after
`write_metrics`, the predictions and metrics artifacts, `finish`, and the W&B
run id recorded in `env.json` and `state.json` by the run store.

It also covers the two additive pieces the wiring needed: the `callbacks=`
argument of `strikecast.tuning.optuna_runner.tune` and the trial callback
`strikecast.pipeline.tune_stage.trial_callback` built on it.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from strikecast.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    ModelEntry,
    ParadigmConfig,
    SeedConfig,
    SeriesConfig,
    StageConfig,
    StoreConfig,
    TrackingConfig,
    TransformConfig,
    TuningConfig,
)
from strikecast.data.series import build_bundle  # noqa: E402
from strikecast.pipeline import context, run_stage, tune_stage  # noqa: E402
from strikecast.pipeline.data_stage import DataArtifacts, FeatureSets  # noqa: E402
from strikecast.store import RunKey, RunStore  # noqa: E402
from strikecast.tracking import NoopTracker, run_name, wandb_run_id  # noqa: E402

N_STEPS = 121
REGIONS = ("alpha", "beta", "gamma", "delta")
ACTIVITY = {"alpha": 1, "beta": 1, "gamma": 2, "delta": 3}
TARGET = "y"
PAST = ["past_a", "past_b"]
FUTURE = ["holiday_x"]

#: The smoke stage: `predict_stride=5` leaves two forecast origins in the CV
#: window, `retrain_stride=1` makes the tracker hook fire on both of them.
SMOKE_FOLDS = 2


# --------------------------------------------------------------------------- #
# the fake wandb (same shape as tests/unit/test_tracking.py's)
# --------------------------------------------------------------------------- #
class FakeRun:
    def __init__(self, init_kwargs: dict) -> None:
        self.init_kwargs = init_kwargs
        self.id = init_kwargs.get("id")
        self.logs: list[tuple[int | None, dict]] = []
        self.artifacts: list[FakeArtifact] = []
        self.exit_code: int | None = None

    def log(self, data, step=None):
        self.logs.append((step, dict(data)))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self.exit_code = exit_code

    # -- convenience for the assertions -------------------------------------
    def steps(self) -> list[int | None]:
        return [step for step, _ in self.logs]

    def payload_with(self, key: str) -> dict:
        for _, payload in self.logs:
            if key in payload:
                return payload
        raise AssertionError(f"no logged payload carries {key!r}: {self.logs}")


class FakeTable:
    def __init__(self, dataframe=None) -> None:
        self.dataframe = dataframe


class FakeArtifact:
    def __init__(self, name, type, metadata=None) -> None:  # noqa: A002 - wandb's name
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []
        self.dirs: list[str] = []

    def add_file(self, path):
        self.files.append(path)

    def add_dir(self, path):
        self.dirs.append(path)


@pytest.fixture
def fake_wandb(monkeypatch):
    module = types.ModuleType("wandb")
    module.runs = []
    module.init_calls = []

    def init(**kwargs):
        module.init_calls.append(kwargs)
        run = FakeRun(kwargs)
        module.runs.append(run)
        return run

    module.init = init
    module.Table = FakeTable
    module.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return module


# --------------------------------------------------------------------------- #
# the synthetic pipeline fixtures
# --------------------------------------------------------------------------- #
def make_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=N_STEPS, freq="D")
    frames = []
    for i, region in enumerate(REGIONS):
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "event_date": dates,
                    "Activity_Level": ACTIVITY[region],
                    TARGET: rng.poisson(2.0 + i, N_STEPS).astype(float),
                    "past_a": rng.normal(size=N_STEPS),
                    "past_b": np.arange(N_STEPS, dtype=float) % 7,
                    "holiday_x": (dates.dayofweek >= 5).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def bundle():
    return build_bundle(make_panel(), TARGET, PAST, FUTURE, SeriesConfig(), ACTIVITY)


@pytest.fixture
def data(bundle) -> DataArtifacts:
    return DataArtifacts(
        bundle=bundle,
        features=FeatureSets(list(PAST), list(FUTURE), "test", "feathash"),
        panel_hash="panelhash",
        series_hash="serieshash",
        activity_by_region=dict(ACTIVITY),
    )


def make_cfg(tmp_path, tracking: TrackingConfig) -> ExperimentConfig:
    """The `diff` family with a two-fold cv stage and one cheap model."""
    return ExperimentConfig(
        name="diff",
        data=DataConfig(target=TARGET),
        series=SeriesConfig(),
        transform=TransformConfig(kind="diff"),
        models=[ModelEntry(name="linear"), ModelEntry(name="naive_last")],
        paradigms=[ParadigmConfig(name="global")],
        stages={
            "cv": StageConfig(start="cv_start_frac", predict_stride=5, retrain_stride=1),
            "test": StageConfig(start="train_val_end", predict_stride=25, retrain_stride=1),
        },
        tuning=TuningConfig(objective="RMSSE_mean"),
        seeds=SeedConfig(tuning_seed=42, eval_seeds=[42]),
        tracking=tracking,
        store=StoreConfig(root=str(tmp_path / "runs")),
    )


@pytest.fixture
def cfg(tmp_path) -> ExperimentConfig:
    return make_cfg(
        tmp_path,
        TrackingConfig(
            backend="wandb",
            mode="offline",
            project="strikecast-test",
            strict=True,  # the smoke test WANTS a broken mirror to fail loudly
            tags=["smoke"],
        ),
    )


@pytest.fixture
def store(cfg: ExperimentConfig) -> RunStore:
    return RunStore(cfg.store.root)


# --------------------------------------------------------------------------- #
# the two-fold smoke run
# --------------------------------------------------------------------------- #
def test_the_smoke_stage_really_has_two_folds(cfg, data, store, fake_wandb) -> None:
    outcome = run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    assert outcome.n_folds == SMOKE_FOLDS


def test_a_two_fold_run_mirrors_name_group_job_type_and_tags(cfg, data, store, fake_wandb):
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)

    assert len(fake_wandb.init_calls) == 1
    kwargs = fake_wandb.init_calls[0]
    key = RunKey("diff", "linear", "global", 42)
    assert kwargs["name"] == run_name(key) == "diff/linear/global/seed=42"
    assert kwargs["group"] == "diff"  # group = experiment
    assert kwargs["job_type"] == "cv"  # job_type = stage
    assert kwargs["project"] == "strikecast-test"
    assert kwargs["mode"] == "offline"
    assert kwargs["id"] == wandb_run_id(run_name(key), "cv")
    # [family, kind] from the spec, then TrackingConfig.tags
    assert kwargs["tags"] == ["linear", "global", "smoke"]
    # the config W&B holds is the resolved config the store snapshots, plus the
    # run coordinates -- no state the store lacks
    assert kwargs["config"]["name"] == "diff"
    assert kwargs["config"]["run"] == {
        "experiment": "diff",
        "model": "linear",
        "paradigm": "global",
        "seed": 42,
        "stage": "cv",
        "dir": "diff/linear/global/seed=42",
    }


def test_a_two_fold_run_logs_one_point_per_fold(cfg, data, store, fake_wandb) -> None:
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    run = fake_wandb.runs[0]

    fold_logs = [(step, payload) for step, payload in run.logs if step is not None]
    assert [step for step, _ in fold_logs] == list(range(SMOKE_FOLDS))
    for step, payload in fold_logs:
        assert payload["fold"] == step
        # the stage prefixes its curve, so cv and test never share a chart
        assert any(k.startswith("cv/") for k in payload)
        assert "cv/MASE_mean" in payload
        assert all(isinstance(v, (int, float)) for v in payload.values())


def test_a_two_fold_run_mirrors_the_metric_views_as_tables(cfg, data, store, fake_wandb):
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    run = fake_wandb.runs[0]

    tables = run.payload_with("cv/per_region")
    # exactly the views the Global paradigm writes (no activity views)
    assert set(tables) == {"cv/global", "cv/per_region", "cv/per_horizon", "cv/per_region_horizon"}
    assert isinstance(tables["cv/per_region"], FakeTable)
    assert isinstance(tables["cv/per_region"].dataframe, pd.DataFrame)
    # the mirrored table is the one on disk
    written = pd.read_csv(store.metrics_dir(RunKey("diff", "linear", "global", 42), "cv")
                          / "per_region.csv")
    assert len(tables["cv/per_region"].dataframe) == len(written)


def test_a_two_fold_run_mirrors_the_predictions_and_metrics_directories(
    cfg, data, store, fake_wandb
) -> None:
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    run = fake_wandb.runs[0]
    key = RunKey("diff", "linear", "global", 42)

    kinds = {a.type: a for a in run.artifacts}
    assert set(kinds) == {"predictions", "metrics"}
    assert kinds["predictions"].dirs == [str(store.predictions_dir(key, "cv"))]
    assert kinds["metrics"].dirs == [str(store.metrics_dir(key, "cv"))]
    # W&B artifact names may not contain `/` or `=`
    for artifact in run.artifacts:
        assert "/" not in artifact.name and "=" not in artifact.name
        assert artifact.name.startswith("diff__linear__global__seed-42__cv__")


def test_a_two_fold_run_finishes_its_mirror(cfg, data, store, fake_wandb) -> None:
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    assert fake_wandb.runs[0].exit_code == 0


def test_the_wandb_run_id_is_recorded_in_env_json(cfg, data, store, fake_wandb) -> None:
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    key = RunKey("diff", "linear", "global", 42)
    expected = wandb_run_id(run_name(key), "cv")

    env = json.loads((store.resolve(key) / "env.json").read_text(encoding="utf-8"))
    assert env["tracker_run_id"] == expected
    assert env["tracker_run_ids"] == {"cv": expected}
    assert env["python"]  # the rest of `record_env` is untouched
    assert store.read_state(key, "cv").tracker_run_id == expected


def test_each_stage_is_its_own_mirror_and_both_ids_survive(cfg, data, store, fake_wandb):
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    run_stage.run_stage(cfg, "linear", "global", 42, "test", data, store=store)
    key = RunKey("diff", "linear", "global", 42)

    assert [c["job_type"] for c in fake_wandb.init_calls] == ["cv", "test"]
    ids = [c["id"] for c in fake_wandb.init_calls]
    assert len(set(ids)) == 2, "one W&B run per stage"

    env = json.loads((store.resolve(key) / "env.json").read_text(encoding="utf-8"))
    assert env["tracker_run_ids"] == {
        "cv": wandb_run_id(run_name(key), "cv"),
        "test": wandb_run_id(run_name(key), "test"),
    }
    assert env["tracker_run_id"] == env["tracker_run_ids"]["test"]
    assert store.read_state(key, "cv").tracker_run_id == env["tracker_run_ids"]["cv"]


def test_a_skipped_stage_does_not_open_a_second_mirror(cfg, data, store, fake_wandb) -> None:
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    again = run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    assert again.skipped
    assert len(fake_wandb.init_calls) == 1


def test_a_failing_stage_finishes_the_mirror_as_failed(cfg, data, store, fake_wandb, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(run_stage, "_run_backtest", boom)
    with pytest.raises(RuntimeError):
        run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    assert fake_wandb.runs[0].exit_code == 1
    assert store.read_state(RunKey("diff", "linear", "global", 42), "cv").status == "failed"


def test_the_store_is_the_same_with_and_without_a_tracker(tmp_path, data, fake_wandb) -> None:
    """§5.5: dropping the tracker cannot change what a run produces."""
    tracked = make_cfg(tmp_path / "a", TrackingConfig(backend="wandb", mode="offline"))
    untracked = make_cfg(tmp_path / "b", TrackingConfig(backend="noop"))
    key = RunKey("diff", "linear", "global", 42)

    a, b = RunStore(tracked.store.root), RunStore(untracked.store.root)
    run_stage.run_stage(tracked, "linear", "global", 42, "cv", data, store=a)
    run_stage.run_stage(untracked, "linear", "global", 42, "cv", data, store=b)

    left = a.load_predictions(key, "cv", legacy_order=True).frame
    right = b.load_predictions(key, "cv", legacy_order=True).frame
    pd.testing.assert_frame_equal(left, right)
    assert a.read_metrics(key, "cv")["global"] == b.read_metrics(key, "cv")["global"]
    # ... and the untracked run records no mirror at all
    assert "tracker_run_id" not in b.read_env(key)


# --------------------------------------------------------------------------- #
# the no-op path
# --------------------------------------------------------------------------- #
def test_a_noop_tracker_gets_no_fold_hook() -> None:
    """The hook re-scores every fold it mirrors; that must not be paid for
    metrics nobody reads. Both no-op implementations count."""
    assert context.make_fold_hook(NoopTracker(), lambda _: {}) is None
    assert context.make_fold_hook(context.NoopTracker(), lambda _: {}) is None
    assert context.is_noop_tracker(NoopTracker())
    assert not context.is_noop_tracker(object())


def test_a_real_tracker_does_get_a_fold_hook() -> None:
    from strikecast.tracking import TrackerFoldHook, WandbTracker

    hook = context.make_fold_hook(WandbTracker(), lambda _: {}, prefix="cv/", every=3)
    assert isinstance(hook, TrackerFoldHook)
    assert (hook.prefix, hook.every) == ("cv/", 3)


def test_make_tracker_forwards_dir_and_strict_from_the_config(tmp_path) -> None:
    cfg = make_cfg(
        tmp_path,
        TrackingConfig(backend="wandb", mode="offline", dir=str(tmp_path / "wandb"), strict=True),
    )
    tracker = context.make_tracker(cfg)
    assert tracker.dir == tmp_path / "wandb"
    assert tracker.strict is True
    assert tracker.mode == "offline"

    plain = context.make_tracker(make_cfg(tmp_path, TrackingConfig(backend="wandb")))
    assert plain.dir is None and plain.strict is False  # unchanged default behaviour


def test_tracking_config_defaults_preserve_the_previous_behaviour() -> None:
    cfg = TrackingConfig()
    assert (cfg.dir, cfg.strict, cfg.tags) == (None, False, [])


def test_tracker_tags_appends_config_tags_and_drops_duplicates(tmp_path) -> None:
    spec = context.get_spec("linear", "diff")
    cfg = make_cfg(tmp_path, TrackingConfig(backend="noop", tags=["cluster", "linear"]))
    assert context.tracker_tags(cfg, spec) == ("linear", "global", "cluster")
    assert context.tracker_tags(make_cfg(tmp_path, TrackingConfig()), spec) == (
        "linear",
        "global",
    )


# --------------------------------------------------------------------------- #
# tuning: the `callbacks=` argument and the trial callback
# --------------------------------------------------------------------------- #
class _FakeSpec:
    """The three attributes `optuna_runner.tune` reads off a `ModelSpec`."""

    name = "toy"
    n_trials = None

    @staticmethod
    def search_space(trial) -> dict[str, Any]:
        return {"x": trial.suggest_float("x", -1.0, 1.0)}


def _settings():
    from strikecast.tuning import TuningSettings

    return TuningSettings(metric="RMSSE_mean")


def test_tune_calls_every_callback_once_per_finished_trial(tmp_path) -> None:
    pytest.importorskip("optuna")
    from strikecast.tuning import tune

    seen: list[tuple[str, int, float | None]] = []

    def callback(study, trial) -> None:
        seen.append((study.study_name, trial.number, trial.value))

    result = tune(
        _FakeSpec(),
        lambda params, trial: float(params["x"] ** 2),
        _settings(),
        tmp_path / "optuna.sqlite3",
        study_name="toy",
        n_trials=3,
        callbacks=[callback],
    )
    assert [n for _, n, _ in seen] == [0, 1, 2]
    assert {name for name, _, _ in seen} == {"toy"}
    assert result.n_trials == 3
    assert all(value is not None for _, _, value in seen)


def test_callbacks_are_optional_and_change_no_optimize_option(tmp_path, monkeypatch) -> None:
    """Additive: without `callbacks=`, `optimize` sees the legacy option set."""
    optuna = pytest.importorskip("optuna")
    from strikecast.tuning import tune

    seen: dict[str, Any] = {}
    real_optimize = optuna.Study.optimize

    def spy(self, func, **kwargs):
        seen.update(kwargs)
        return real_optimize(self, func, **kwargs)

    monkeypatch.setattr(optuna.Study, "optimize", spy)
    tune(
        _FakeSpec(),
        lambda params, trial: float(params["x"] ** 2),
        _settings(),
        tmp_path / "optuna.sqlite3",
        study_name="toy",
        n_trials=2,
    )
    assert "callbacks" not in seen


def test_the_trial_callback_logs_number_value_and_params() -> None:
    logged: list[tuple[int, float, dict]] = []

    class Spy:
        def log_trial(self, number, value, params):
            logged.append((number, value, dict(params)))

    callback = tune_stage.trial_callback(Spy())
    callback(None, types.SimpleNamespace(number=3, value=0.5, params={"lr": 0.1}))
    callback(None, types.SimpleNamespace(number=4, value=None, params={"lr": 0.2}))  # pruned
    callback(None, types.SimpleNamespace(number=5, value=float("nan"), params={}))  # failed
    assert logged == [(3, 0.5, {"lr": 0.1})]


def test_the_trial_callback_cannot_abort_a_study() -> None:
    class Broken:
        def log_trial(self, number, value, params):
            raise RuntimeError("no network")

    callback = tune_stage.trial_callback(Broken())
    callback(None, types.SimpleNamespace(number=0, value=1.0, params={}))  # must not raise


def test_tune_model_mirrors_each_trial_through_the_callback(
    cfg, data, store, fake_wandb, monkeypatch
) -> None:
    pytest.importorskip("optuna")
    from strikecast.models.classical import _build_linear
    from strikecast.models.spec import ModelSpec

    spec = ModelSpec(
        name="linear",
        family="linear",
        kind="global",
        build=lambda params, ctx: _build_linear({}, ctx),
        experiments=("diff",),
        search_space=lambda trial: {"dummy": trial.suggest_categorical("dummy", [0, 1])},
        stochastic=False,
        n_trials=None,
    )
    monkeypatch.setattr(tune_stage, "get_spec", lambda name, exp=None: spec)

    tune_stage.tune_model(cfg, "linear", data, store=store, n_trials=2)

    assert fake_wandb.init_calls[0]["job_type"] == "tune"
    assert fake_wandb.init_calls[0]["name"] == "diff/linear/global/seed=42"
    run = fake_wandb.runs[0]
    trials = [payload for _, payload in run.logs if "trial/number" in payload]
    assert [p["trial/number"] for p in trials] == [0, 1]
    # the params come from the trial itself, which the old post-study replay
    # could not supply
    assert all("trial/params/dummy" in p for p in trials)
    assert run.exit_code == 0
