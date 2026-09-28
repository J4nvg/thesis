"""`scripts/slurm/submit_all.py`: the one-command SLURM DAG (audit 2026-09-26 C30).

Nothing here talks to SLURM: the DAG, the sbatch lines, the resource/time
planning, `status` and `timings` are pure functions over a matrix, a manifest,
logs and a run store, and are tested as such. `job.sbatch` is exercised for
real (bash, a stand-in venv) for its exit line and its signal handling.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "scripts" / "slurm"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SLURM / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


make_jobs = _load("make_jobs")
submit_all = _load("submit_all")
Job = make_jobs.Job


def run_job(exp, model, paradigm, stage, seed=42, *, resource="cpu", family="lightgbm",
            kind="global", env="main"):
    command = (
        f"tune experiment={exp} model={model}" if stage == "tune"
        else f"run experiment={exp} model={model} paradigm={paradigm} stage={stage} seed={seed}"
    )
    return Job(resource, command, model, stage, experiment=exp,
               paradigm=None if stage == "tune" else paradigm,
               seed=None if stage == "tune" else seed, family=family, kind=kind, env=env,
               tunable=stage == "tune")


def matrices():
    return {
        "count": [
            run_job("count", "lightgbm_poisson", None, "tune"),
            run_job("count", "lstm_w7", None, "tune", resource="gpu", family="lstm"),
            run_job("count", "lightgbm_poisson", "global", "test"),
            run_job("count", "lightgbm_poisson", "local", "test"),
            run_job("count", "lstm_w7", "global", "test", resource="gpu", family="lstm"),
        ],
        "hurdle": [
            run_job("hurdle", "hurdle", "global", "cv", family="hurdle", kind="composite"),
            run_job("hurdle", "hurdle", "global", "test", family="hurdle", kind="composite"),
        ],
        "chronos2": [
            run_job("chronos2", "chronos2_fine_tuned", None, "tune", resource="gpu",
                    family="chronos", kind="chronos", env="autogluon"),
            run_job("chronos2", "chronos2_fine_tuned", "global", "test", resource="gpu",
                    family="chronos", kind="chronos", env="autogluon"),
        ],
    }


def dag(opts=None, **kwargs):
    opts = opts or submit_all.Options(experiments=["count", "hurdle", "chronos2"])
    kwargs.setdefault("need_setup", {"main": True, "autogluon": True})
    kwargs.setdefault("importance_jobs", {"count": {"lightgbm_poisson": ("global",)}})
    kwargs.setdefault("verify_groups", {"count": ["cpu", "gpu"], "chronos2": ["gpu"]})
    nodes = submit_all.build_dag(opts, kwargs.pop("matrices", matrices()), **kwargs)
    submit_all.plan_resources(nodes, submit_all.load_time_table(None), opts)
    return {n.name: n for n in nodes}, nodes


# --------------------------------------------------------------------------- #
# the DAG
# --------------------------------------------------------------------------- #
def test_the_dag_is_topologically_ordered_and_names_are_unique() -> None:
    _, nodes = dag()
    seen: set[str] = set()
    for node in nodes:
        for dep in node.after_ok + node.after_any:
            assert dep in seen, f"{node.name} depends on {dep}, which comes later"
        seen.add(node.name)


def test_setup_then_featsel_then_tune() -> None:
    by, _ = dag()
    assert by["featsel:count"].after_ok == ["setup"]
    assert by["tune:count:lightgbm_poisson"].after_ok == ["setup", "featsel:count"]
    assert "featsel:chronos2" not in by  # Chronos-2 has no feature selection (F28)


def test_cv_and_test_wait_for_their_own_tune_only() -> None:
    """Per-model chains: a model's test starts when ITS tune is done."""
    by, _ = dag()
    test = by["test:count:lightgbm_poisson:global:s42"]
    assert "tune:count:lightgbm_poisson" in test.after_ok
    assert "tune:count:lstm_w7" not in test.after_ok
    assert "tune:count:lstm_w7" in by["test:count:lstm_w7:global:s42"].after_ok


def test_a_composite_test_waits_for_its_cv() -> None:
    by, _ = dag()
    assert "cv:hurdle:hurdle:global" in by["test:hurdle:hurdle:global:s42"].after_ok


def test_chronos_runs_in_the_autogluon_env_after_its_setup() -> None:
    by, _ = dag()
    tune = by["tune:chronos2:chronos2_fine_tuned"]
    assert tune.env == "autogluon"
    assert tune.after_ok == ["setup", "setup_ag"]
    assert by["setup_ag"].after_ok == ["setup"]
    line = submit_all.sbatch_args(tune, {}, submit_all.Options())
    assert line[line.index(submit_all.JOB_SCRIPT) + 1] == "autogluon"


def test_tune_jobs_are_requeueable_and_signalled() -> None:
    by, _ = dag()
    tune = by["tune:count:lightgbm_poisson"]
    assert tune.requeue and tune.partition == "GPU"
    line = submit_all.sbatch_args(tune, {}, submit_all.Options())
    assert "--requeue" in line and "--signal=B:USR1@900" in line
    assert "--open-mode=append" in line
    assert "--requeue" not in submit_all.sbatch_args(
        by["test:count:lightgbm_poisson:global:s42"], {}, submit_all.Options()
    )


def test_report_and_figures_wait_for_everything_with_afterany() -> None:
    by, _ = dag()
    report = by["report:count"]
    assert set(report.after_any) >= {
        "featsel:count", "tune:count:lightgbm_poisson", "test:count:lightgbm_poisson:local:s42",
        "importance:count:lightgbm_poisson:global",
    }
    assert "test:hurdle:hurdle:global:s42" not in report.after_any
    figures = by["figures"]
    assert figures.optional
    assert set(figures.after_any) == {"report:count", "report:hurdle", "report:chronos2"}
    line = submit_all.sbatch_args(figures, {}, submit_all.Options())
    assert "--export=ALL,STRIKECAST_MAX_REQUEUES=8,STRIKECAST_OPTIONAL=1" in line


def test_importance_waits_for_tune_and_the_tuning_seed_test() -> None:
    by, _ = dag()
    node = by["importance:count:lightgbm_poisson:global"]
    assert "tune:count:lightgbm_poisson" in node.after_ok
    assert "test:count:lightgbm_poisson:global:s42" in node.after_ok
    assert "seed=42" in node.argv


def test_verification_is_independent_of_the_publication_chain() -> None:
    by, _ = dag()
    assert by["verify_prep"].after_ok == ["setup"]
    assert by["verify:count:gpu"].after_ok == ["setup", "verify_prep"]
    assert by["verify:chronos2:gpu"].after_ok == ["setup", "setup_ag", "verify_prep"]
    assert by["verify:chronos2:gpu"].env == "autogluon"
    assert set(by["verify_report"].after_any) == {
        "verify:count:cpu", "verify:count:gpu", "verify:chronos2:gpu",
    }
    assert "--store-root" in by["verify:count:cpu"].argv
    assert by["verify:count:cpu"].argv[by["verify:count:cpu"].argv.index("--store-root") + 1] == (
        "runs_verify"
    )


def test_a_built_environment_needs_no_setup_job() -> None:
    by, _ = dag(need_setup={"main": False, "autogluon": False})
    assert "setup" not in by and "setup_ag" not in by
    assert by["featsel:count"].after_ok == []


def test_setup_only() -> None:
    opts = submit_all.options_from(submit_all._parser().parse_args(["--setup-only"]))
    assert opts.stages == ["setup"]
    nodes = submit_all.build_dag(opts, {}, need_setup={"main": True, "autogluon": True})
    assert [n.name for n in nodes] == ["setup", "setup_ag"]


def test_later_seeds_depend_on_jobs_still_queued_from_an_earlier_submission() -> None:
    opts = submit_all.Options(seeds=[1, 2], stages=["test"], experiments=["count"])
    m = {"count": [run_job("count", "lightgbm_poisson", "global", "test", seed=s) for s in (1, 2)]}
    by, _ = dag(opts, matrices=m, need_setup={}, importance_jobs={}, verify_groups={},
                queued={"tune:count:lightgbm_poisson": "777"})
    node = by["test:count:lightgbm_poisson:global:s1"]
    assert node.after_ok == ["tune:count:lightgbm_poisson"]
    line = submit_all.sbatch_args(node, {"tune:count:lightgbm_poisson": "777"}, opts)
    assert "--dependency=afterok:777" in line
    assert "report:count" not in by  # stage list is test only


def test_benchmark_runs_truncated_pilots_on_their_own_tuned_params() -> None:
    opts = submit_all.options_from(submit_all._parser().parse_args(["--benchmark"]))
    assert opts.store_root == "runs_benchmark" and opts.tracking == "noop"
    assert set(opts.stages) <= {"setup", "featsel", "tune", "cv", "test"}
    opts.experiments = ["count", "hurdle", "chronos2"]
    by, _ = dag(opts)
    test = by["test:count:lightgbm_poisson:global:s42"]
    # the pilot's test waits for its own 2-trial tune (the real run's code path);
    # default params would select the unused legacy RNN preset (precision 32-true)
    assert "tune:count:lightgbm_poisson" in test.after_ok
    assert "--allow-default-params" not in test.argv
    flags = test.argv[test.argv.index("--max-folds"):]
    assert flags[:2] == ["--max-folds", "14"]
    assert flags[2:4] == ["--store-root", "runs_benchmark"]
    assert by["tune:count:lightgbm_poisson"].argv[-5:-3] == ["--n-trials", "2"]
    assert "cv:hurdle:hurdle:global" in by["test:hurdle:hurdle:global:s42"].after_ok
    assert not any(n.startswith(("report", "importance", "verify", "figures")) for n in by)
    assert all(n.hours <= 6 for n in by.values() if n.stage in ("cv", "test", "tune"))


# --------------------------------------------------------------------------- #
# resources and time
# --------------------------------------------------------------------------- #
def test_gpu_and_cpu_resource_classes() -> None:
    by, _ = dag()
    gpu = submit_all.sbatch_args(by["test:count:lstm_w7:global:s42"], {}, submit_all.Options())
    assert "--gres=gpu:1" in gpu and "--gres-flags=disable-binding" in gpu
    assert "--cpus-per-task=8" in gpu and "--partition=GPU" in gpu
    cpu = submit_all.sbatch_args(
        by["test:count:lightgbm_poisson:global:s42"], {}, submit_all.Options()
    )
    assert "--cpus-per-task=16" in cpu and not any(a.startswith("--gres") for a in cpu)


def test_a_job_class_above_the_threshold_goes_to_gpuextended() -> None:
    opts = submit_all.Options(experiments=["count", "hurdle", "chronos2"], long_threshold=20)
    by, _ = dag(opts)
    local = by["test:count:lightgbm_poisson:local:s42"]  # 24 h in the default table
    assert local.partition == "GPUExtended" and local.time == "1-00:00:00"
    assert by["test:count:lightgbm_poisson:global:s42"].partition == "GPU"
    # a resumable tune job never leaves GPU, and never asks for more than 36 h
    table = submit_all.load_time_table(None)
    table["tune/lightgbm/cpu"] = {"hours": 80}
    nodes = list(by.values())
    submit_all.plan_resources(nodes, table, opts)
    assert by["tune:count:lightgbm_poisson"].partition == "GPU"
    assert by["tune:count:lightgbm_poisson"].time == "1-12:00:00"


def test_time_keys_go_from_specific_to_generic() -> None:
    keys = submit_all.time_keys("test", "catboost", "cpu", "local")
    assert keys[:3] == ["test/catboost/cpu/local", "test/catboost/cpu", "test/gbdt/cpu/local"]
    assert keys[-1] == "test"


def test_the_time_table_file_overrides_the_defaults(tmp_path) -> None:
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"_comment": "x", "test/gbdt/cpu": {"hours": 3, "evidence": "pilot"},
                                "report": 0.5}))
    table = submit_all.load_time_table(path)
    assert table["test/gbdt/cpu"]["hours"] == 3 and table["test/gbdt/cpu"]["evidence"] == "pilot"
    assert table["report"]["hours"] == 0.5
    assert "_comment" not in table


def test_format_time() -> None:
    assert submit_all.format_time(2) == "02:00:00"
    assert submit_all.format_time(0.25) == "00:15:00"
    assert submit_all.format_time(36) == "1-12:00:00"


def test_dependency_line_mixes_afterok_and_afterany() -> None:
    node = submit_all.Node("report:x", "report", "cpu_small", "main", ["-m", "x"], ["report"],
                           after_ok=["setup"], after_any=["a", "b"])
    node.time = "02:00:00"
    line = submit_all.sbatch_args(node, {"setup": "1", "a": "2", "b": "3"}, submit_all.Options())
    assert "--dependency=afterok:1,afterany:2:3" in line
    assert "--kill-on-invalid-dep=yes" in line
    assert line[-4:] == [submit_all.JOB_SCRIPT, "main", "-m", "x"]


# --------------------------------------------------------------------------- #
# submitting (fake sbatch), dirty checkout, manifest
# --------------------------------------------------------------------------- #
def test_submit_uses_the_returned_ids_for_dependencies(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(submit_all, "REPO", tmp_path)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=f"{100 + len(calls)};cluster\n", stderr="")

    monkeypatch.setattr(submit_all.subprocess, "run", fake_run)
    _, nodes = dag()
    ids = submit_all.submit(nodes, submit_all.Options(), dry_run=False, queued={})
    assert ids["setup"] == "101"
    featsel = next(c for c in calls if "--job-name=sc:featsel:count" in c)
    assert "--dependency=afterok:101" in featsel
    assert (tmp_path / "logs" / "slurm" / "test").is_dir()  # C27: log dirs exist before sbatch


def test_a_dirty_checkout_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(submit_all, "git_state", lambda: {
        "commit": "abc", "branch": "b", "upstream": None, "dirty": True,
        "dirty_files": [" M x.py"], "unpushed": False,
    })
    monkeypatch.setattr(submit_all.subprocess, "run", lambda *a, **k: pytest.fail("sbatch called"))
    with pytest.raises(SystemExit, match="refusing to submit"):
        submit_all.main(["--setup-only"])


def test_setup_only_submits_two_jobs_and_writes_a_manifest(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(submit_all, "REPO", tmp_path)
    monkeypatch.setattr(submit_all, "git_state", lambda: {
        "commit": "abc123", "branch": "refactor", "upstream": "origin/refactor",
        "dirty": False, "dirty_files": [], "unpushed": False,
    })
    monkeypatch.setattr(submit_all, "setup_needed", lambda env: True)
    monkeypatch.setattr(submit_all, "queued_from_earlier", lambda root: {})
    seq = iter(range(500, 600))
    monkeypatch.setattr(
        submit_all.subprocess, "run",
        lambda args, **k: subprocess.CompletedProcess(args, 0, stdout=f"{next(seq)}\n", stderr=""),
    )
    assert submit_all.main(["--setup-only"]) == 0
    manifest = json.loads(next((tmp_path / "logs" / "submissions").glob("*.json")).read_text())
    assert [j["name"] for j in manifest["jobs"]] == ["setup", "setup_ag"]
    assert [j["job_id"] for j in manifest["jobs"]] == ["500", "501"]
    assert manifest["git"]["commit"] == "abc123"
    assert manifest["jobs"][1]["sbatch"][-2:] == [submit_all.SETUP_SCRIPT, "autogluon"]


def test_setup_needed_follows_the_lock_hash(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(submit_all, "REPO", tmp_path)
    (tmp_path / ".venv").mkdir()
    (tmp_path / "uv.lock").write_text("lock-v1")
    assert submit_all.setup_needed("main")
    stamp = tmp_path / ".venv" / ".strikecast-setup.json"
    stamp.write_text(json.dumps({"lock_sha256": submit_all._sha256(tmp_path / "uv.lock")}))
    assert not submit_all.setup_needed("main")
    (tmp_path / "uv.lock").write_text("lock-v2")
    assert submit_all.setup_needed("main")


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def test_a_resubmit_never_re_emits_jobs_still_queued_from_an_earlier_submission() -> None:
    """2026-09-27: after one failed tune the resubmit re-emitted 110 live jobs."""
    opts = submit_all.Options(stages=["tune", "test"], experiments=["count"])
    m = {"count": [
        run_job("count", "lightgbm_poisson", None, "tune"),
        run_job("count", "lightgbm_poisson", "global", "test"),
        run_job("count", "xgboost_tweedie", None, "tune"),
        run_job("count", "xgboost_tweedie", "global", "test"),
    ]}
    queued = {"tune:count:lightgbm_poisson": "700", "test:count:lightgbm_poisson:global:s42": "701"}
    by, _ = dag(opts, matrices=m, need_setup={}, importance_jobs={}, verify_groups={},
                queued=queued)
    assert set(by) == {"tune:count:xgboost_tweedie", "test:count:xgboost_tweedie:global:s42"}
    assert by["test:count:xgboost_tweedie:global:s42"].after_ok == ["tune:count:xgboost_tweedie"]


def test_status_merges_submissions_and_ignores_cancelled_duplicates(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(submit_all, "REPO", tmp_path)
    directory = tmp_path / "logs" / "submissions"
    directory.mkdir(parents=True)
    common = {"git": {"commit": "abc"}, "options": {"store_root": "store"}}
    first = [{"name": "tune:count:a", "stage": "tune", "job_id": "10", "log": "logs/x-%j.out"}]
    second = [
        {"name": "tune:count:a", "stage": "tune", "job_id": "20", "log": "logs/x-%j.out"},
        {"name": "tune:count:b", "stage": "tune", "job_id": "21", "log": "logs/x-%j.out"},
    ]
    (directory / "20260927T100000Z.json").write_text(json.dumps({**common, "jobs": first}))
    (directory / "20260927T110000Z.json").write_text(json.dumps({**common, "jobs": second}))
    live = {"10": {"state": "RUNNING", "reason": "None", "elapsed": "1:00", "node": "n1"},
            "21": {"state": "PENDING", "reason": "Priority", "elapsed": "0:00", "node": ""}}
    monkeypatch.setattr(submit_all, "squeue_ids", lambda: live)
    assert submit_all.cmd_status([]) == 0
    out = capsys.readouterr().out
    assert "(+1 earlier for this store)" in out
    assert "running" in out and "10  tune:count:a" in out  # the original, not cancelled 20
    assert "21  tune:count:b" in out
    assert "failed" not in out


def _manifest(tmp_path, jobs):
    directory = tmp_path / "logs" / "submissions"
    directory.mkdir(parents=True)
    path = directory / "20260926T120000Z.json"
    path.write_text(json.dumps({"git": {"commit": "abc"}, "options": {"store_root": "store"},
                                "jobs": jobs}))
    return path


def _log(tmp_path, name, job_id, lines):
    path = tmp_path / "logs" / "slurm" / f"{name}-{job_id}.out"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return f"logs/slurm/{name}-%j.out"


def test_status_reads_squeue_logs_and_the_store(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(submit_all, "REPO", tmp_path)
    run_dir = tmp_path / "store" / "count" / "lightgbm_poisson" / "global" / "seed=42"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "test": {"status": "failed", "error": "MissingTunedParams: no best_params"},
    }))
    meta = {"experiment": "count", "model": "lightgbm_poisson", "paradigm": "global", "seed": 42}
    jobs = [
        {"name": "setup", "stage": "setup", "job_id": "1",
         "log": _log(tmp_path, "setup", 1, ["ok", "STRIKECAST_EXIT 0"])},
        {"name": "tune:count:x", "stage": "tune", "job_id": "2",
         "log": _log(tmp_path, "tune", 2, ["STRIKECAST_EXIT 99"])},
        {"name": "test:count:lightgbm_poisson:global:s42", "stage": "test", "job_id": "3",
         "meta": meta, "log": _log(tmp_path, "test", 3, ["Traceback", "STRIKECAST_EXIT 1"])},
        {"name": "report:count", "stage": "report", "job_id": "4", "log": "logs/slurm/r-%j.out"},
        {"name": "featsel:count", "stage": "featsel", "job_id": "5",
         "log": _log(tmp_path, "featsel", 5, ["killed"])},
    ]
    _manifest(tmp_path, jobs)
    monkeypatch.setattr(submit_all, "squeue_ids", lambda: {
        "4": {"state": "PENDING", "reason": "Dependency", "elapsed": "0:00", "node": ""},
    })
    assert submit_all.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "interrupted" in out and "tune:count:x" in out
    assert "MissingTunedParams" in out  # the store's error, not just "exit 1"
    assert "pending" in out and "Dependency" in out
    assert "killed (wall clock / OOM / node)" in out
    assert "--- failed: test:count:lightgbm_poisson:global:s42" in out
    assert "Traceback" in out
    by_name = {j["name"]: submit_all.job_status(j, {}, tmp_path / "store") for j in jobs}
    assert by_name["setup"] == ("done", "")


# --------------------------------------------------------------------------- #
# timings (pilot -> time table)
# --------------------------------------------------------------------------- #
def test_timings_turn_the_pilot_into_a_time_table(tmp_path) -> None:
    store = tmp_path / "runs_benchmark"
    run = store / "count" / "lightgbm_poisson" / "local" / "seed=42"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({"test": {
        "status": "complete", "folds_done": 14, "max_folds": 14,
        "attempts": [{"outcome": "complete", "seconds": 1400.0, "folds_done": 14}],
    }}))
    tuning = store / "count" / "tuning" / "catboost_tweedie"
    tuning.mkdir(parents=True)
    (tuning / "trials.csv").write_text(
        "number,value,datetime_start,datetime_complete,duration,state\n"
        "0,1.0,2026-09-26 10:00:00,2026-09-26 10:30:00,0 days 00:30:00,COMPLETE\n"
        "1,1.1,2026-09-26 10:30:00,2026-09-26 11:10:00,0 days 00:40:00,COMPLETE\n"
    )
    manifest = {"jobs": [
        {"name": "test:count:lightgbm_poisson:local:s42", "stage": "test", "resource": "cpu",
         "meta": {"experiment": "count", "model": "lightgbm_poisson", "paradigm": "local",
                  "family": "lightgbm"}},
        {"name": "tune:count:catboost_tweedie", "stage": "tune", "resource": "cpu",
         "meta": {"experiment": "count", "model": "catboost_tweedie", "family": "catboost"}},
    ]}
    rows = submit_all.collect_timings(store, manifest)
    assert {(r["stage"], r["model"]) for r in rows} == {
        ("test", "lightgbm_poisson"), ("tune", "catboost_tweedie"),
    }
    table = submit_all.timings_table(rows, safety=1.5)
    # 100 s/fold x 164 folds x 1.5 = 6.83 h -> 7.0 (quarter hours, rounded up)
    assert table["test/lightgbm/cpu/local"]["hours"] == 7.0
    # max trial 40 min x 50 trials x 1.5 = 50 h
    assert table["tune/catboost/cpu"]["hours"] == 50.0
    assert "pilot" in table["test/lightgbm/cpu/local"]["evidence"]


# --------------------------------------------------------------------------- #
# the real configs (needs strikecast)
# --------------------------------------------------------------------------- #
def test_dry_run_against_the_real_configs(monkeypatch, capsys) -> None:
    pytest.importorskip("hydra")
    pytest.importorskip("darts")
    monkeypatch.setattr(submit_all, "setup_needed", lambda env: False)
    code = submit_all.main([
        "--dry-run", "--experiments", "diff,hurdle", "--store-root", "does_not_exist_store",
        "--no-verify",
    ])
    assert code == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("sbatch ")]
    names = [ln.split("--job-name=sc:")[1].split()[0] for ln in lines]
    tests = [n for n in names if n.startswith("test:")]
    # diff: 9 tuned x 3 paradigms + 4 baselines x global; hurdle: composite x 3
    assert len([n for n in tests if n.startswith("test:diff:")]) == 31
    assert len([n for n in tests if n.startswith("test:hurdle:")]) == 3
    assert [n for n in names if n.startswith("cv:")] == [
        "cv:hurdle:hurdle:global", "cv:hurdle:hurdle:activity", "cv:hurdle:hurdle:local",
    ]
    assert len([n for n in names if n.startswith("tune:diff:")]) == 9
    assert not any("spe_event_classifier" in n or "count_head" in n for n in names)
    assert "featsel:diff" in names and "featsel:hurdle" in names
    assert "(dry run:" in out


# --------------------------------------------------------------------------- #
# job.sbatch, for real (bash + a stand-in venv)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_repo(tmp_path):
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    os.symlink(sys.executable, bin_dir / "python")
    return tmp_path


def _job(fake_repo, *args, env=None, background=False):
    environ = {**os.environ, "SLURM_SUBMIT_DIR": str(fake_repo), **(env or {})}
    environ.pop("SLURM_JOB_ID", None)
    cmd = ["bash", str(SLURM / "job.sbatch"), "main", *args]
    if background:
        return subprocess.Popen(cmd, env=environ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
    return subprocess.run(cmd, env=environ, capture_output=True, text=True, timeout=60)


def test_job_sbatch_is_valid_bash() -> None:
    for script in ("job.sbatch", "setup.sbatch", "run.sbatch", "tune.sbatch"):
        assert subprocess.run(["bash", "-n", str(SLURM / script)]).returncode == 0, script


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_job_sbatch_runs_the_prebuilt_interpreter_and_prints_the_exit_line(fake_repo) -> None:
    done = _job(fake_repo, "-c", "import os; print('HASHSEED', os.environ['PYTHONHASHSEED'])")
    assert done.returncode == 0
    assert "HASHSEED 0" in done.stdout
    assert done.stdout.strip().splitlines()[-1] == "STRIKECAST_EXIT 0"
    failed = _job(fake_repo, "-c", "raise SystemExit(3)")
    assert failed.returncode == 3 and "STRIKECAST_EXIT 3" in failed.stdout


def test_job_sbatch_refuses_a_missing_env(tmp_path) -> None:
    done = subprocess.run(["bash", str(SLURM / "job.sbatch"), "main", "-c", "pass"],
                          env={**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path)},
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 3 and "--setup-only" in done.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_job_sbatch_forwards_sigterm_and_exits_99(fake_repo) -> None:
    script = (
        "import signal, sys, time\n"
        "def h(s, f):\n    print('child got', s, flush=True); sys.exit(99)\n"
        "signal.signal(signal.SIGTERM, h)\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    proc = _job(fake_repo, "-c", script, background=True)
    assert proc.stdout is not None
    deadline = time.time() + 20
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.strip() == "ready":
            break
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=40)
    assert proc.returncode == 99
    assert "child got" in out and "STRIKECAST_EXIT 99" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_job_sbatch_optional_command_is_skipped_when_unknown(fake_repo) -> None:
    pkg = fake_repo / "strikecast" / "cli"
    pkg.mkdir(parents=True)
    (fake_repo / "strikecast" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text("import sys\nprint('usage: {run,tune}')\nsys.exit(0)\n")
    done = _job(fake_repo, "-m", "strikecast.cli.main", "figures", "--x",
                env={"STRIKECAST_OPTIONAL": "1", "PYTHONPATH": str(fake_repo)})
    assert done.returncode == 0
    assert "not available in this checkout; skipping" in done.stdout


def test_finished_verification_groups_are_not_resubmitted(monkeypatch, tmp_path) -> None:
    """A resubmit re-queued verification jobs that had already finished."""
    from types import SimpleNamespace

    from strikecast.verification import legacy

    cases = {
        ("count", "cpu"): [SimpleNamespace(id="count__lightgbm_poisson__global__test")],
        ("count", "gpu"): [SimpleNamespace(id="count__lstm_w7__global__test")],
        ("diff", "cpu"): [SimpleNamespace(id="diff__arima__global__test"),
                          SimpleNamespace(id="diff__linear__global__test")],
    }
    monkeypatch.setattr(legacy, "cases_for", lambda exp, resource=None: cases[(exp, resource)])
    results = legacy.results_dir(tmp_path)
    results.mkdir(parents=True)
    (results / "count__lightgbm_poisson__global__test.json").write_text('{"status": "PASS"}')
    (results / "count__lstm_w7__global__test.json").write_text('{"status": "ERROR"}')
    (results / "diff__arima__global__test.json").write_text('{"status": "FAIL"}')
    # diff linear has no result yet

    pending = submit_all.verification_pending(
        {"count": ["cpu", "gpu"], "diff": ["cpu"]}, tmp_path
    )
    assert pending == {"count": ["gpu"], "diff": ["cpu"]}


# --------------------------------------------------------------------------- #
# --top / --runs: a seed sweep over a subset of runs
# --------------------------------------------------------------------------- #
def test_top_reads_the_master_leaderboard_as_store_run_keys(tmp_path: Path) -> None:
    board = tmp_path / "master_leaderboard.csv"
    board.write_text(
        "Modelname,paradigm,model,SkillScore,mae,rmse\n"
        "chronos2,local,chronos2_fine_tuned,0.17,0.71,1.83\n"
        "gbdt,global,xgboost_tweedie,0.16,0.78,1.84\n"
        "lstm,activity,gru_tweedie_w14,0.15,0.77,1.85\n"
        "diff,global,arima,0.15,0.81,1.85\n"
        "finalhurdle,global,hurdle,0.09,0.80,2.01\n",
        encoding="utf-8",
    )
    assert submit_all.leaderboard_runs(board, 4) == {
        ("chronos2", "chronos2_fine_tuned", "global"),
        ("count", "xgboost_tweedie", "global"),
        ("count", "gru_tweedie_w14", "activity"),
        ("diff", "arima", "global"),
    }
    with pytest.raises(SystemExit):
        submit_all.leaderboard_runs(tmp_path / "missing.csv", 20)


def test_runs_option_parses_and_restricts_experiments() -> None:
    assert submit_all.parse_runs("hurdle:hurdle:global,count:lstm_w7:global") == {
        ("hurdle", "hurdle", "global"), ("count", "lstm_w7", "global"),
    }
    with pytest.raises(SystemExit):
        submit_all.parse_runs("hurdle:global")
    args = submit_all._parser().parse_args(["--runs", "hurdle:hurdle:global", "--seeds", "1,2"])
    opts = submit_all.options_from(args)
    assert opts.experiments == ["hurdle"]
    assert opts.runs == {("hurdle", "hurdle", "global")}


def test_manifest_serializes_run_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit_all, "REPO", tmp_path)
    opts = submit_all.Options(runs={("hurdle", "hurdle", "global")})
    path = submit_all.write_manifest([], opts, {"commit": "x"}, {}, ["--runs", "hurdle:hurdle:global"])
    assert json.loads(path.read_text())["options"]["runs"] == [["hurdle", "hurdle", "global"]]


def test_adopt_takes_live_ids_then_newest_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit_all, "REPO", tmp_path)
    by, nodes = dag()
    runs = [n for n in nodes if n.stage == "test"][:3]
    live_one, logged, missing = runs
    logs = tmp_path / submit_all.LOG_ROOT / "test"
    logs.mkdir(parents=True)
    for job_id in ("100", "205"):
        (logs / f"{submit_all.sanitize(logged.name)}-{job_id}.out").write_text("")
    kept = submit_all.adopt(runs, submit_all.Options(), {live_one.name: "300"})
    assert [(n.name, n.job_id) for n in kept] == [(live_one.name, "300"), (logged.name, "205")]
    assert missing.job_id is None
