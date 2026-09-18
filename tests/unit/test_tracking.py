"""Phase 4 tracking: the Tracker protocol, NoopTracker, WandbTracker, the hook.

No network and no wandb process is ever involved: a fake ``wandb`` module is
injected into ``sys.modules``, and every assertion is about what the tracker
*would* hand wandb. The rule under test throughout is §5.5's -- the tracker
mirrors the run store and may neither hold state the store lacks nor be able to
abort a run.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from strikecast.backtest.protocols import Fold, FoldResult
from strikecast.store.run_store import RunKey
from strikecast.tracking import (
    NoopTracker,
    Tracker,
    TrackerFoldHook,
    WandbTracker,
    make_tracker,
    numeric_metrics,
    run_name,
    wandb_run_id,
)

# --------------------------------------------------------------------------- #
# the fake wandb
# --------------------------------------------------------------------------- #


class FakeRun:
    def __init__(self, init_kwargs: dict) -> None:
        self.init_kwargs = init_kwargs
        self.id = init_kwargs.get("id")
        self.logs: list[tuple[int | None, dict]] = []
        self.artifacts: list[FakeArtifact] = []
        self.exit_code: int | None = None
        self.fail_log = False

    def log(self, data, step=None):
        if self.fail_log:
            raise RuntimeError("no network")
        self.logs.append((step, dict(data)))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


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


def _fake_wandb_module(*, fail_init: bool = False) -> types.ModuleType:
    module = types.ModuleType("wandb")
    module.runs = []
    module.init_calls = []

    def init(**kwargs):
        module.init_calls.append(kwargs)
        if fail_init:
            raise RuntimeError("wandb backend unreachable")
        run = FakeRun(kwargs)
        module.runs.append(run)
        return run

    module.init = init
    module.Table = FakeTable
    module.Artifact = FakeArtifact
    return module


@pytest.fixture
def fake_wandb(monkeypatch):
    module = _fake_wandb_module()
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return module


@pytest.fixture
def run_key() -> RunKey:
    return RunKey(experiment="count", model="lightgbm_poisson", paradigm="global", seed=42)


CONFIG = {"name": "count", "models": ["lightgbm_poisson"], "seeds": {"tuning_seed": 42}}
TAGS = ("gbm", "global")


def _started(tracker: WandbTracker, run_key: RunKey, **kwargs) -> FakeRun:
    tracker.start(run_key, CONFIG, TAGS, **kwargs)
    return tracker.run


# --------------------------------------------------------------------------- #
# the protocol and the naming
# --------------------------------------------------------------------------- #


def test_both_trackers_satisfy_the_tracker_protocol():
    assert isinstance(NoopTracker(), Tracker)
    assert isinstance(WandbTracker(), Tracker)


def test_run_name_is_the_run_store_directory(run_key):
    assert run_name(run_key) == str(run_key.relative())
    assert run_name(run_key) == "count/lightgbm_poisson/global/seed=42"


def test_run_name_only_needs_the_four_coordinates():
    key = types.SimpleNamespace(experiment="diff", model="gru_w14", paradigm="local", seed=3)
    assert run_name(key) == "diff/gru_w14/local/seed=3"


def test_run_id_is_deterministic_and_per_stage(run_key):
    name = run_name(run_key)
    assert wandb_run_id(name, "cv") == wandb_run_id(name, "cv")
    assert wandb_run_id(name, "cv") != wandb_run_id(name, "test")
    assert wandb_run_id(name, "cv") != wandb_run_id("count/other/global/seed=42", "cv")
    assert "/" not in wandb_run_id(name, "cv")


# --------------------------------------------------------------------------- #
# NoopTracker
# --------------------------------------------------------------------------- #


def test_noop_tracker_is_a_noop(run_key, tmp_path):
    tracker = NoopTracker()
    path = tmp_path / "global.json"
    path.write_text("{}")
    assert tracker.start(run_key, CONFIG, TAGS, stage="cv") is None
    assert tracker.log_fold(0, {"RMSSE_mean": 1.0}) is None
    assert tracker.log_trial(0, 1.0, {"lr": 0.1}) is None
    assert tracker.log_tables({"per_region": pd.DataFrame({"a": [1]})}, stage="cv") is None
    assert tracker.log_artifact(path, "metrics") is None
    assert tracker.finish() is None
    assert tracker.__dict__ == {}  # it cannot drift from the store: it holds nothing


def test_noop_tracker_never_imports_wandb(monkeypatch, run_key):
    monkeypatch.setitem(sys.modules, "wandb", None)  # any `import wandb` now raises
    tracker = NoopTracker()
    tracker.start(run_key, CONFIG, TAGS)
    tracker.log_fold(0, {"RMSSE_mean": 1.0})
    tracker.finish()


def test_noop_tracker_is_a_context_manager(run_key):
    with NoopTracker() as tracker:
        assert tracker.start(run_key, CONFIG) is None


# --------------------------------------------------------------------------- #
# WandbTracker.start: naming, grouping, job type, tags, config
# --------------------------------------------------------------------------- #


def test_start_names_groups_and_types_the_run(fake_wandb, run_key):
    tracker = WandbTracker(project="strikecast", entity="jan")
    returned = _started(tracker, run_key, stage="cv")
    kwargs = fake_wandb.init_calls[0]
    assert kwargs["name"] == "count/lightgbm_poisson/global/seed=42"
    assert kwargs["group"] == "count"  # group = experiment
    assert kwargs["job_type"] == "cv"  # job_type = stage
    assert kwargs["tags"] == ["gbm", "global"]  # tags = [family, kind]
    assert kwargs["project"] == "strikecast"
    assert kwargs["entity"] == "jan"
    assert kwargs["id"] == wandb_run_id(run_name(run_key), "cv")
    assert tracker.run_id == returned.id


def test_one_run_per_experiment_model_paradigm_seed_and_stage(fake_wandb):
    ids = set()
    for model in ("lightgbm_poisson", "xgboost_tweedie"):
        for paradigm in ("global", "activity"):
            for seed in (42, 1):
                for stage in ("cv", "test"):
                    key = RunKey("count", model, paradigm, seed)
                    tracker = WandbTracker()
                    tracker.start(key, CONFIG, TAGS, stage=stage)
                    ids.add(tracker.run_id)
                    tracker.finish()
    assert len(ids) == 2 * 2 * 2 * 2


def test_config_group_overrides_the_experiment(fake_wandb, run_key):
    tracker = WandbTracker(group="ablation")
    _started(tracker, run_key, stage="test")
    assert fake_wandb.init_calls[0]["group"] == "ablation"


def test_config_is_the_resolved_config_plus_the_run_coordinates(fake_wandb, run_key):
    tracker = WandbTracker()
    _started(tracker, run_key, stage="cv")
    cfg = fake_wandb.init_calls[0]["config"]
    assert cfg["name"] == "count"
    assert cfg["models"] == ["lightgbm_poisson"]
    assert cfg["run"] == {
        "experiment": "count",
        "model": "lightgbm_poisson",
        "paradigm": "global",
        "seed": 42,
        "stage": "cv",
        "dir": str(run_key.relative()),
    }
    assert CONFIG == {  # the caller's mapping is not mutated
        "name": "count",
        "models": ["lightgbm_poisson"],
        "seeds": {"tuning_seed": 42},
    }


def test_starting_twice_raises(fake_wandb, run_key):
    tracker = WandbTracker()
    _started(tracker, run_key)
    with pytest.raises(RuntimeError, match="already started"):
        tracker.start(run_key, CONFIG, TAGS)


# --------------------------------------------------------------------------- #
# online / offline
# --------------------------------------------------------------------------- #


def test_online_is_the_default_and_leaves_wandb_mode_alone(fake_wandb, run_key, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    tracker = WandbTracker()
    assert tracker.mode == "online"
    _started(tracker, run_key)
    assert fake_wandb.init_calls[0]["mode"] == "online"
    assert "WANDB_MODE" not in __import__("os").environ


def test_offline_mode_is_passed_and_exported(fake_wandb, run_key, monkeypatch):
    import os

    monkeypatch.delenv("WANDB_MODE", raising=False)
    tracker = WandbTracker(mode="offline")
    _started(tracker, run_key)
    assert fake_wandb.init_calls[0]["mode"] == "offline"
    assert os.environ["WANDB_MODE"] == "offline"


def test_offline_dir_is_created_and_passed(fake_wandb, run_key, tmp_path):
    scratch = tmp_path / "scratch" / "wandb"
    tracker = WandbTracker(mode="offline", dir=scratch)
    _started(tracker, run_key)
    assert scratch.is_dir()
    assert fake_wandb.init_calls[0]["dir"] == str(scratch)


def test_no_dir_key_when_none_is_configured(fake_wandb, run_key):
    tracker = WandbTracker()
    _started(tracker, run_key)
    assert "dir" not in fake_wandb.init_calls[0]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown wandb mode"):
        WandbTracker(mode="sometimes")


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


def test_log_fold_uses_the_fold_as_the_step(fake_wandb, run_key):
    tracker = WandbTracker()
    run = _started(tracker, run_key, stage="cv")
    tracker.log_fold(0, {"RMSSE_mean": 1.5})
    tracker.log_fold(7, {"RMSSE_mean": 1.25})
    assert [step for step, _ in run.logs] == [0, 7]
    assert run.logs[1][1] == {"RMSSE_mean": 1.25, "fold": 7}


def test_log_fold_before_start_and_after_finish_is_a_noop(fake_wandb, run_key):
    tracker = WandbTracker()
    tracker.log_fold(0, {"RMSSE_mean": 1.0})  # before start
    run = _started(tracker, run_key)
    tracker.log_fold(1, {"RMSSE_mean": 1.0})
    tracker.finish()
    tracker.log_fold(2, {"RMSSE_mean": 1.0})  # after finish
    assert [step for step, _ in run.logs] == [1]


def test_log_fold_ignores_empty_metrics(fake_wandb, run_key):
    tracker = WandbTracker()
    run = _started(tracker, run_key)
    tracker.log_fold(0, {})
    assert run.logs == []


def test_log_trial_records_value_and_params_at_the_trial_number(fake_wandb, run_key):
    tracker = WandbTracker()
    run = _started(tracker, run_key, stage="tune")
    tracker.log_trial(3, 1.234, {"learning_rate": 0.05, "num_leaves": 31})
    step, payload = run.logs[0]
    assert step == 3
    assert payload == {
        "trial/number": 3,
        "trial/value": 1.234,
        "trial/params/learning_rate": 0.05,
        "trial/params/num_leaves": 31,
    }


def test_log_tables_wraps_frames_and_prefixes_with_the_stage(fake_wandb, run_key):
    tracker = WandbTracker()
    run = _started(tracker, run_key, stage="cv")
    frame = pd.DataFrame({"region": ["Kyiv"], "RMSSE": [0.9]})
    tracker.log_tables({"per_region": frame, "leaderboard": None}, stage="cv")
    _, payload = run.logs[0]
    assert list(payload) == ["cv/per_region"]
    assert isinstance(payload["cv/per_region"], FakeTable)
    assert payload["cv/per_region"].dataframe is frame


def test_log_tables_without_a_stage_keeps_the_bare_key(fake_wandb, run_key):
    tracker = WandbTracker()
    run = _started(tracker, run_key)
    tracker.log_tables({"leaderboard": pd.DataFrame({"a": [1]})})
    assert list(run.logs[0][1]) == ["leaderboard"]


def test_log_artifact_mirrors_a_file_with_the_kind_as_its_type(fake_wandb, run_key, tmp_path):
    tracker = WandbTracker()
    run = _started(tracker, run_key, stage="cv")
    path = tmp_path / "global.json"
    path.write_text("{}")
    tracker.log_artifact(path, "metrics", metadata={"stage": "cv"})
    artifact = run.artifacts[0]
    assert artifact.type == "metrics"
    assert artifact.name == "count__lightgbm_poisson__global__seed-42__cv__metrics"
    assert artifact.files == [str(path)]
    assert artifact.metadata == {"stage": "cv"}


def test_log_artifact_mirrors_a_directory(fake_wandb, run_key, tmp_path):
    tracker = WandbTracker()
    run = _started(tracker, run_key, stage="test")
    directory = tmp_path / "predictions"
    directory.mkdir()
    (directory / "part-000000-000006.parquet").write_bytes(b"")
    tracker.log_artifact(directory, "predictions")
    assert run.artifacts[0].dirs == [str(directory)]
    assert run.artifacts[0].files == []


def test_log_artifact_skips_a_missing_path(fake_wandb, run_key, tmp_path, caplog):
    tracker = WandbTracker()
    run = _started(tracker, run_key)
    with caplog.at_level(logging.WARNING):
        tracker.log_artifact(tmp_path / "nope.json", "metrics")
    assert run.artifacts == []
    assert "does not exist" in caplog.text


def test_finish_sets_the_exit_code_and_is_idempotent(fake_wandb, run_key):
    tracker = WandbTracker()
    run = _started(tracker, run_key)
    tracker.finish("failed")
    assert run.exit_code == 1
    tracker.finish()  # second call must not raise
    assert run.exit_code == 1
    assert tracker.run is None


def test_context_manager_finishes_with_the_exception_status(fake_wandb, run_key):
    tracker = WandbTracker()
    with pytest.raises(ValueError):
        with tracker:
            run = _started(tracker, run_key)
            raise ValueError("boom")
    assert run.exit_code == 1


# --------------------------------------------------------------------------- #
# the mirror may not abort the run
# --------------------------------------------------------------------------- #


def test_a_failing_init_disables_tracking_instead_of_raising(monkeypatch, run_key, caplog):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb_module(fail_init=True))
    tracker = WandbTracker()
    with caplog.at_level(logging.WARNING):
        assert tracker.start(run_key, CONFIG, TAGS) is None
    tracker.log_fold(0, {"RMSSE_mean": 1.0})
    tracker.log_tables({"a": pd.DataFrame({"x": [1]})})
    tracker.finish()
    assert "tracking disabled" in caplog.text


def test_a_failing_log_disables_the_tracker_rather_than_the_run(fake_wandb, run_key, caplog):
    tracker = WandbTracker()
    run = _started(tracker, run_key)
    run.fail_log = True
    with caplog.at_level(logging.WARNING):
        tracker.log_fold(0, {"RMSSE_mean": 1.0})
    assert run.logs == []
    run.fail_log = False
    tracker.log_fold(1, {"RMSSE_mean": 1.0})  # stays disabled
    assert run.logs == []
    assert "tracking disabled" in caplog.text


def test_strict_reraises(fake_wandb, run_key):
    tracker = WandbTracker(strict=True)
    run = _started(tracker, run_key)
    run.fail_log = True
    with pytest.raises(RuntimeError, match="no network"):
        tracker.log_fold(0, {"RMSSE_mean": 1.0})


def test_wandb_backend_without_wandb_installed_says_what_to_do(monkeypatch, run_key):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="noop"):
        WandbTracker().start(run_key, CONFIG, TAGS)


def test_the_package_imports_without_wandb():
    """`import strikecast.tracking` must not need wandb (it is imported lazily)."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'wandb' or name.startswith('wandb.'):\n"
        "            raise ImportError('wandb is not installed')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import strikecast.tracking as t\n"
        "assert t.WandbTracker and t.NoopTracker and t.make_tracker\n"
        "assert 'wandb' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------- #
# the fold hook
# --------------------------------------------------------------------------- #


class RecordingTracker(NoopTracker):
    """A NoopTracker that remembers what the hook handed it."""

    def __init__(self) -> None:
        self.folds: list[tuple[int, dict]] = []

    def log_fold(self, step, metrics):
        self.folds.append((step, dict(metrics)))


def _fold_result(index: int, *, retrain: bool = False) -> FoldResult:
    fold = Fold(index=index, t0=100 + index, cutoff=pd.Timestamp("2023-01-01"), retrain=retrain)
    return FoldResult(fold=fold, predictions={"y_pred": []})


def test_fold_hook_logs_running_metrics_with_step_equal_to_the_fold():
    tracker = RecordingTracker()
    seen: list[dict] = []

    def score(cumulative):
        seen.append(cumulative)
        return {"RMSSE_mean": 1.0 + len(seen)}

    hook = TrackerFoldHook(tracker, score)
    cumulative = {"y_pred": [[]]}
    for i in range(3):
        hook.on_fold(_fold_result(i), cumulative)
    assert tracker.folds == [
        (0, {"RMSSE_mean": 2.0}),
        (1, {"RMSSE_mean": 3.0}),
        (2, {"RMSSE_mean": 4.0}),
    ]
    assert seen[0] is cumulative  # it scores the cumulative bundle, like the pruner
    assert hook.last_metrics == {"RMSSE_mean": 4.0}


def test_fold_hook_prefix_separates_the_stages():
    tracker = RecordingTracker()
    hook = TrackerFoldHook(tracker, lambda c: {"RMSSE_mean": 1.0}, prefix="cv/")
    hook.on_fold(_fold_result(0), {})
    assert tracker.folds == [(0, {"cv/RMSSE_mean": 1.0})]


def test_fold_hook_every_scores_once_per_window():
    tracker = RecordingTracker()
    calls = []
    hook = TrackerFoldHook(
        tracker,
        lambda c: calls.append(1) or {"RMSSE_mean": 1.0},
        every=7,
    )
    for i in range(15):
        hook.on_fold(_fold_result(i), {})
    assert [step for step, _ in tracker.folds] == [0, 7, 14]
    assert len(calls) == 3


def test_fold_hook_rejects_a_bad_every():
    with pytest.raises(ValueError, match="every must be"):
        TrackerFoldHook(NoopTracker(), lambda c: {}, every=0)


def test_fold_hook_drops_non_numeric_and_empty_rows():
    tracker = RecordingTracker()
    hook = TrackerFoldHook(
        tracker,
        lambda c: {"model": "lightgbm_poisson", "RMSSE_mean": "1.5", "MAE": 2, "flag": True,
                   "missing": None},
    )
    hook.on_fold(_fold_result(0), {})
    assert tracker.folds == [(0, {"RMSSE_mean": 1.5, "MAE": 2.0})]

    empty = RecordingTracker()
    TrackerFoldHook(empty, lambda c: None).on_fold(_fold_result(0), {})
    TrackerFoldHook(empty, lambda c: {}).on_fold(_fold_result(1), {})
    assert empty.folds == []


def test_fold_hook_swallows_a_failing_score(caplog):
    tracker = RecordingTracker()

    def score(cumulative):
        raise ZeroDivisionError("no folds yet")

    with caplog.at_level(logging.WARNING):
        TrackerFoldHook(tracker, score).on_fold(_fold_result(0), {})
    assert tracker.folds == []
    assert "not mirrored" in caplog.text


def test_fold_hook_with_a_noop_tracker_is_inert():
    hook = TrackerFoldHook(NoopTracker(), lambda c: {"RMSSE_mean": 1.0})
    hook.on_fold(_fold_result(0), {})
    assert hook.last_metrics == {"RMSSE_mean": 1.0}


def test_numeric_metrics_keeps_only_numbers():
    row = {"a": 1, "b": "2.5", "c": "x", "d": None, "e": True, "f": float("nan")}
    out = numeric_metrics(row)
    assert set(out) == {"a", "b", "f"}
    assert out["a"] == 1.0 and out["b"] == 2.5


# --------------------------------------------------------------------------- #
# the factory
# --------------------------------------------------------------------------- #


def test_make_tracker_returns_a_noop_for_the_noop_backend():
    cfg = types.SimpleNamespace(backend="noop", mode="online", project="p", entity=None, group=None)
    assert isinstance(make_tracker(cfg), NoopTracker)


def test_make_tracker_without_a_config_is_a_noop():
    assert isinstance(make_tracker(None), NoopTracker)


def test_make_tracker_passes_the_config_fields_through(tmp_path):
    cfg = types.SimpleNamespace(
        backend="wandb", mode="offline", project="strikecast", entity="jan", group="ablation"
    )
    tracker = make_tracker(cfg, dir=tmp_path)
    assert isinstance(tracker, WandbTracker)
    assert (tracker.project, tracker.entity, tracker.mode, tracker.group) == (
        "strikecast",
        "jan",
        "offline",
        "ablation",
    )
    assert tracker.dir == tmp_path


def test_make_tracker_rejects_an_unknown_backend():
    cfg = types.SimpleNamespace(backend="comet")
    with pytest.raises(ValueError, match="unknown tracking backend"):
        make_tracker(cfg)


def test_make_tracker_accepts_the_real_tracking_config():
    from strikecast.config.schema import TrackingConfig

    tracker = make_tracker(TrackingConfig())
    assert isinstance(tracker, WandbTracker)
    assert tracker.mode == "online"  # online by default (§5.5)
    assert tracker.project == "strikecast"
    assert isinstance(make_tracker(TrackingConfig(backend="noop")), NoopTracker)
    assert make_tracker(TrackingConfig(mode="offline")).mode == "offline"
