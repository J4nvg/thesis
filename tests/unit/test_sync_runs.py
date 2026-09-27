"""`scripts/sync_runs.py`: merging a cluster run store into a laptop one (§8 P6).

Everything here runs against a synthetic store built under `tmp_path`, because
the guarantee under test is about `state.json` arithmetic, not about data: a
**complete** stage must never be replaced by an incomplete one, whichever
machine it came from and whichever file is newer, and a second run of the same
sync must change nothing.

The script is standard-library only and is loaded from its path, the way
`test_make_jobs.py` loads its generator.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "sync_runs.py"


def _load():
    spec = importlib.util.spec_from_file_location("sync_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["sync_runs"] = module
    spec.loader.exec_module(module)
    return module


sync_runs = _load()


# --------------------------------------------------------------------------- #
# a synthetic run store
# --------------------------------------------------------------------------- #
RUN = Path("count") / "lightgbm_poisson" / "global" / "seed=42"


def _state(stage: str, *, status: str, folds: int, tracker: str | None = None) -> dict:
    """A `StageState.to_dict()` payload, with only the fields the sync reads."""
    parts = []
    if folds:
        parts = [{"path": f"part-0-{folds - 1}.parquet", "from": 0, "to": folds - 1, "rows": folds}]
    return {
        "stage": stage,
        "status": status,
        "stage_hash": "h" * 8,
        "folds_done": folds,
        "last_retrain_fold": None,
        "parts": parts,
        "started": "2026-09-19T10:00:00+00:00",
        "finished": "2026-09-19T11:00:00+00:00" if status == "complete" else None,
        "error": None,
        "tracker_run_id": tracker,
    }


def make_run(
    root: Path,
    *,
    states: dict[str, dict],
    parts: dict[str, list[str]] | None = None,
    env: dict | None = None,
    config: str = "experiment: count\n",
    metrics: dict[str, str] | None = None,
) -> Path:
    """Write one run directory of a store rooted at ``root``."""
    run = root / RUN
    run.mkdir(parents=True, exist_ok=True)
    (run / "state.json").write_text(json.dumps(states, indent=2), encoding="utf-8")
    (run / "config.yaml").write_text(config, encoding="utf-8")
    if env is not None:
        (run / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    for stage, names in (parts or {}).items():
        preds = run / stage / "predictions"
        preds.mkdir(parents=True, exist_ok=True)
        for name in names:
            (preds / name).write_text(f"{stage}:{name}", encoding="utf-8")
    for stage, body in (metrics or {}).items():
        mdir = run / stage / "metrics"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "global.json").write_text(body, encoding="utf-8")
    return run


def sync(src: Path, dst: Path, **kw):
    plan = sync_runs.plan_sync(src, dst, **kw)
    sync_runs.apply_plan(plan)
    return plan


# --------------------------------------------------------------------------- #
# stage_rank: the one rule everything else follows
# --------------------------------------------------------------------------- #
def test_a_complete_stage_outranks_any_running_one():
    complete = _state("test", status="complete", folds=10)
    running = _state("test", status="running", folds=164)
    assert sync_runs.stage_rank(complete) > sync_runs.stage_rank(running)


def test_a_missing_stage_ranks_lowest():
    assert sync_runs.stage_rank(None) == (0, 0)
    assert sync_runs.stage_rank({}) == (0, 0)
    assert sync_runs.stage_rank(_state("cv", status="running", folds=1)) > (0, 0)


def test_between_two_incomplete_stages_more_folds_wins():
    few = _state("cv", status="running", folds=6)
    many = _state("cv", status="running", folds=24)
    assert sync_runs.stage_rank(many) > sync_runs.stage_rank(few)


def test_a_corrupt_folds_count_does_not_crash_the_rank():
    assert sync_runs.stage_rank({"status": "running", "folds_done": "many"}) == (0, 0)


# --------------------------------------------------------------------------- #
# the core guarantee
# --------------------------------------------------------------------------- #
def test_a_complete_destination_stage_is_never_replaced_by_an_incomplete_source(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"test": _state("test", status="running", folds=12)},
        parts={"test": ["part-0-11.parquet"]},
        metrics={"test": '{"MAE": 999.0}'},
    )
    make_run(
        dst,
        states={"test": _state("test", status="complete", folds=164)},
        parts={"test": ["part-0-163.parquet"]},
        metrics={"test": '{"MAE": 1.0}'},
    )

    plan = sync(src, dst)

    run = dst / RUN
    assert json.loads((run / "test" / "metrics" / "global.json").read_text())["MAE"] == 1.0
    assert (run / "test" / "predictions" / "part-0-163.parquet").exists()
    states = json.loads((run / "state.json").read_text())
    assert states["test"]["status"] == "complete"
    assert states["test"]["folds_done"] == 164
    assert not plan.of("replace")
    assert not plan.of("prune")


def test_the_incomplete_source_still_tops_up_files_the_destination_lacks(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"test": _state("test", status="running", folds=12)},
        parts={"test": ["part-0-11.parquet"]},
    )
    (src / RUN / "artifacts").mkdir(parents=True)
    (src / RUN / "artifacts" / "importances.csv").write_text("feature,gain\n", encoding="utf-8")
    make_run(dst, states={"test": _state("test", status="complete", folds=164)})

    sync(src, dst)

    assert (dst / RUN / "artifacts" / "importances.csv").exists()
    # ... but the complete stage's own state is untouched
    states = json.loads((dst / RUN / "state.json").read_text())
    assert states["test"]["status"] == "complete"


def test_a_complete_source_stage_replaces_an_incomplete_destination_one(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"test": _state("test", status="complete", folds=24)},
        parts={"test": ["part-0-11.parquet", "part-12-23.parquet"]},
        metrics={"test": '{"MAE": 1.0}'},
    )
    make_run(
        dst,
        states={"test": _state("test", status="running", folds=12)},
        parts={"test": ["part-0-11.parquet"]},
        metrics={"test": '{"MAE": 999.0}'},
    )

    plan = sync(src, dst)

    run = dst / RUN
    assert json.loads((run / "test" / "metrics" / "global.json").read_text())["MAE"] == 1.0
    assert (run / "test" / "predictions" / "part-12-23.parquet").exists()
    assert json.loads((run / "state.json").read_text())["test"]["status"] == "complete"
    assert [a.verb for a in plan.actions if a.path.endswith("global.json")] == ["replace"]


def test_a_stale_destination_part_is_pruned_when_the_source_stage_is_accepted(tmp_path):
    """Two schedules must not end up side by side in one predictions directory."""
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="complete", folds=12)},
        parts={"cv": ["part-0-11.parquet"]},
    )
    make_run(
        dst,
        states={"cv": _state("cv", status="failed", folds=18)},
        parts={"cv": ["part-0-11.parquet", "part-12-17.parquet"]},
    )

    plan = sync(src, dst)

    preds = dst / RUN / "cv" / "predictions"
    assert not (preds / "part-12-17.parquet").exists()
    assert (preds / "part-0-11.parquet").exists()
    assert [a.path for a in plan.of("prune")] == [
        str(RUN / "cv" / "predictions" / "part-12-17.parquet")
    ]


def test_nothing_is_pruned_when_the_destination_stage_is_kept(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="running", folds=6)},
        parts={"cv": ["part-0-5.parquet"]},
    )
    make_run(
        dst,
        states={"cv": _state("cv", status="complete", folds=12)},
        parts={"cv": ["part-0-5.parquet", "part-6-11.parquet"]},
    )

    sync(src, dst)

    assert (dst / RUN / "cv" / "predictions" / "part-6-11.parquet").exists()


def test_a_run_the_destination_has_never_seen_is_copied_whole(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="complete", folds=6)},
        parts={"cv": ["part-0-5.parquet"]},
        metrics={"cv": '{"MAE": 2.0}'},
        env={"PYTHONHASHSEED": "0", "hostname": "node042"},
    )

    sync(src, dst)

    run = dst / RUN
    assert (run / "config.yaml").read_text() == "experiment: count\n"
    assert (run / "cv" / "predictions" / "part-0-5.parquet").exists()
    assert json.loads((run / "env.json").read_text())["hostname"] == "node042"
    assert json.loads((run / "state.json").read_text())["cv"]["folds_done"] == 6


# --------------------------------------------------------------------------- #
# dry run and idempotence
# --------------------------------------------------------------------------- #
def test_dry_run_writes_nothing_and_still_reports_every_action(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="complete", folds=6)},
        parts={"cv": ["part-0-5.parquet"]},
    )
    dst.mkdir()

    plan = sync_runs.plan_sync(src, dst)
    n = sync_runs.apply_plan(plan, dry_run=True)

    assert n > 0
    assert list(dst.rglob("*")) == []
    assert "copy" in plan.report()


def test_the_second_sync_is_a_no_op(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={
            "cv": _state("cv", status="complete", folds=6),
            "test": _state("test", status="complete", folds=12),
        },
        parts={"cv": ["part-0-5.parquet"], "test": ["part-0-11.parquet"]},
        metrics={"cv": '{"MAE": 2.0}', "test": '{"MAE": 3.0}'},
        env={"PYTHONHASHSEED": "0"},
    )
    (src / "count" / "shared").mkdir(parents=True)
    (src / "count" / "shared" / "panel.abc123.parquet").write_text("panel", encoding="utf-8")

    sync(src, dst)
    second = sync_runs.plan_sync(src, dst)

    assert sync_runs.apply_plan(second, dry_run=True) == 0, second.report(verbose=True)
    assert second.counts() == {"keep": len(second.actions)}


def test_syncing_back_the_other_way_is_also_a_no_op(tmp_path):
    """laptop -> scratch right after scratch -> laptop must not undo anything."""
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="complete", folds=6)},
        parts={"cv": ["part-0-5.parquet"]},
        env={"PYTHONHASHSEED": "0"},
    )

    sync(src, dst)
    back = sync_runs.plan_sync(dst, src)

    assert sync_runs.apply_plan(back, dry_run=True) == 0, back.report(verbose=True)


# --------------------------------------------------------------------------- #
# json merges
# --------------------------------------------------------------------------- #
def test_state_json_keeps_the_best_of_each_stage_from_both_sides(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={
            "cv": _state("cv", status="running", folds=3),
            "test": _state("test", status="complete", folds=164),
        },
    )
    make_run(
        dst,
        states={
            "cv": _state("cv", status="complete", folds=79),
            "test": _state("test", status="failed", folds=2),
        },
    )

    sync(src, dst)

    states = json.loads((dst / RUN / "state.json").read_text())
    assert (states["cv"]["status"], states["cv"]["folds_done"]) == ("complete", 79)
    assert (states["test"]["status"], states["test"]["folds_done"]) == ("complete", 164)


def test_env_json_unions_the_tracker_ids_of_both_machines(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"test": _state("test", status="complete", folds=164, tracker="cluster1")},
        env={
            "PYTHONHASHSEED": "0",
            "hostname": "node042",
            "tracker_run_id": "cluster1",
            "tracker_run_ids": {"test": "cluster1"},
        },
    )
    make_run(
        dst,
        states={"cv": _state("cv", status="complete", folds=79, tracker="laptop1")},
        env={
            "PYTHONHASHSEED": "0",
            "hostname": "macbook",
            "tracker_run_id": "laptop1",
            "tracker_run_ids": {"cv": "laptop1"},
        },
    )

    sync(src, dst)

    env = json.loads((dst / RUN / "env.json").read_text())
    assert env["tracker_run_ids"] == {"cv": "laptop1", "test": "cluster1"}
    # the source contributed the accepted stage, so it describes the environment
    assert env["hostname"] == "node042"


def test_the_destination_environment_wins_when_it_contributed_every_stage(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="running", folds=1)},
        env={"hostname": "node042", "note": "only on the cluster"},
    )
    make_run(
        dst,
        states={"cv": _state("cv", status="complete", folds=79)},
        env={"hostname": "macbook"},
    )

    sync(src, dst)

    env = json.loads((dst / RUN / "env.json").read_text())
    assert env["hostname"] == "macbook"
    assert env["note"] == "only on the cluster"  # a key only one side has is kept


def test_merge_env_rebuilds_the_id_map_from_the_merged_states():
    merged = sync_runs.merge_env(
        {"tracker_run_ids": {"cv": "stale"}},
        {},
        {
            "cv": _state("cv", status="complete", folds=6, tracker="fresh"),
            "test": _state("test", status="complete", folds=12, tracker="other"),
        },
        source_wins=True,
    )
    assert merged["tracker_run_ids"] == {"cv": "fresh", "test": "other"}
    assert merged["tracker_run_id"] in {"fresh", "other"}


# --------------------------------------------------------------------------- #
# shared, tuning, report and conflicts
# --------------------------------------------------------------------------- #
def test_shared_artefacts_are_copied_but_never_overwritten(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    for root, body in ((src, "cluster"), (dst, "laptop")):
        shared = root / "count" / "shared"
        shared.mkdir(parents=True)
        (shared / "panel.abc123.parquet").write_text(body, encoding="utf-8")
    (src / "count" / "shared" / "series.def456").mkdir()
    (src / "count" / "shared" / "series.def456" / "manifest.json").write_text("{}", "utf-8")

    plan = sync(src, dst)

    assert (dst / "count" / "shared" / "panel.abc123.parquet").read_text() == "laptop"
    assert (dst / "count" / "shared" / "series.def456" / "manifest.json").exists()
    assert [a.path for a in plan.conflicts] == [str(Path("count/shared/panel.abc123.parquet"))]


def test_a_tuning_study_is_copied_when_missing_and_reported_when_it_differs(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    for root, body in ((src, "cluster study"), (dst, "laptop study")):
        tuning = root / "count" / "tuning" / "catboost_tweedie"
        tuning.mkdir(parents=True)
        (tuning / "optuna.sqlite3").write_text(body, encoding="utf-8")
    (src / "count" / "tuning" / "xgboost_tweedie").mkdir()
    (src / "count" / "tuning" / "xgboost_tweedie" / "best_params.json").write_text("{}", "utf-8")

    plan = sync(src, dst)

    assert (dst / "count/tuning/catboost_tweedie/optuna.sqlite3").read_text() == "laptop study"
    assert (dst / "count/tuning/xgboost_tweedie/best_params.json").exists()
    assert len(plan.conflicts) == 1


def test_force_resolves_a_conflict_in_favour_of_the_source(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    for root, body in ((src, "cluster study"), (dst, "laptop study")):
        tuning = root / "count" / "tuning" / "catboost_tweedie"
        tuning.mkdir(parents=True)
        (tuning / "optuna.sqlite3").write_text(body, encoding="utf-8")

    plan = sync(src, dst, force=True)

    assert (dst / "count/tuning/catboost_tweedie/optuna.sqlite3").read_text() == "cluster study"
    assert not plan.conflicts


def test_two_different_configs_under_one_run_key_are_a_conflict(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(src, states={}, config="experiment: count\nseed: 42\n")
    make_run(dst, states={}, config="experiment: count\nseed: 1\n")

    plan = sync(src, dst)

    assert (dst / RUN / "config.yaml").read_text().endswith("seed: 1\n")
    assert [a.path for a in plan.conflicts] == [str(RUN / "config.yaml")]


# --------------------------------------------------------------------------- #
# selection, reporting and the command line
# --------------------------------------------------------------------------- #
def test_only_the_named_experiments_are_considered(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    for exp in ("count", "diff"):
        shared = src / exp / "shared"
        shared.mkdir(parents=True)
        (shared / "panel.abc.parquet").write_text(exp, encoding="utf-8")

    sync(src, dst, experiments=["diff"])

    assert (dst / "diff" / "shared" / "panel.abc.parquet").exists()
    assert not (dst / "count").exists()


def test_a_missing_source_store_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        sync_runs.plan_sync(tmp_path / "nope", tmp_path / "dst")


def test_the_report_names_every_action_and_hides_the_untouched_ones(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="complete", folds=6)},
        parts={"cv": ["part-0-5.parquet"]},
    )
    sync(src, dst)
    plan = sync_runs.plan_sync(src, dst)

    assert plan.report() == f"-- {len(plan.actions)} keep"
    assert "part-0-5.parquet" in plan.report(verbose=True)


def test_main_dry_run_exits_zero_and_touches_nothing(tmp_path, capsys):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="complete", folds=6)},
        parts={"cv": ["part-0-5.parquet"]},
    )
    dst.mkdir()

    code = sync_runs.main([str(src), str(dst), "--dry-run"])

    assert code == 0
    assert list(dst.rglob("*")) == []
    assert "would perform" in capsys.readouterr().out


def test_main_strict_exits_one_on_a_conflict(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    for root, body in ((src, "a"), (dst, "b")):
        shared = root / "count" / "shared"
        shared.mkdir(parents=True)
        (shared / "panel.abc.parquet").write_text(body, encoding="utf-8")

    assert sync_runs.main([str(src), str(dst), "--strict"]) == 1
    assert sync_runs.main([str(src), str(dst)]) == 0


def test_a_large_file_is_compared_by_size_alone(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 1024)
    b.write_bytes(b"y" * 1024)
    assert sync_runs.same_file(a, b, hash_limit=10_000) is False
    assert sync_runs.same_file(a, b, hash_limit=10) is True


def test_an_unreadable_state_file_is_treated_as_absent(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "laptop"
    make_run(
        src,
        states={"cv": _state("cv", status="complete", folds=6)},
        parts={"cv": ["part-0-5.parquet"]},
    )
    make_run(dst, states={})
    (dst / RUN / "state.json").write_text("{ truncated", encoding="utf-8")

    sync(src, dst)

    assert json.loads((dst / RUN / "state.json").read_text())["cv"]["status"] == "complete"


def test_a_leftover_state_lock_is_not_a_stage(tmp_path) -> None:
    """RunStore's `.state.lock` directory (audit C23) must never be synced as a stage."""
    run = tmp_path / "src" / "count" / "m" / "global" / "seed=42"
    (run / ".state.lock").mkdir(parents=True)
    (run / ".state.lock" / "owner").write_text("node1 123")
    assert sync_runs._stage_names(run, {}) == []
