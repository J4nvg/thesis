"""`scripts/slurm/make_jobs.py`: the SLURM array matrix (plan §5.6, §8 P6).

The generator is the only part of the cluster path that can be tested without a
cluster: the `.sbatch` templates are checked with `bash -n`, and what they read
-- one `strikecast` argument line per array task -- is produced here.

What matters and is asserted: the §5.4 seed rules (cv is single-seed, a
deterministic model is not swept), the `cpu`/`gpu` resource class, that a stage
already complete in the run store is not queued again, that a model with
`best_params.json` is not re-tuned (Appendix C), and that the written files are
indexable by `sed -n "${i}p"`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "slurm" / "make_jobs.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_jobs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["make_jobs"] = module
    spec.loader.exec_module(module)
    return module


make_jobs = _load()


# --------------------------------------------------------------------------- #
# fakes: a config and a registry, so the matrix is tested and not the registry
# --------------------------------------------------------------------------- #
class FakeSpec:
    def __init__(self, *, stochastic=True, device="cpu", tunable=True) -> None:
        self.stochastic = stochastic
        self.device = device
        self.tunable = tunable


SPECS = {
    "lightgbm_poisson": FakeSpec(),
    "lstm_w7": FakeSpec(device="gpu"),
    "naive_last": FakeSpec(stochastic=False, tunable=False),
}


class FakeSeeds:
    tuning_seed = 42
    eval_seeds = (42, 1, 2)


class FakeCfg:
    name = "count"
    seeds = FakeSeeds()

    def __init__(self, devices: dict[str, str] | None = None) -> None:
        self.devices = devices or {}

    def device_for(self, model: str) -> str:
        return self.devices.get(model, "cpu")


def get_spec(name, experiment=None):
    return SPECS[name]


def build(cfg=None, **kwargs):
    kwargs.setdefault("models", ["lightgbm_poisson"])
    kwargs.setdefault("paradigms", ["global"])
    kwargs.setdefault("stages", ["test"])
    kwargs.setdefault("seeds", [42, 1])
    return make_jobs.build_jobs(cfg or FakeCfg(), get_spec=get_spec, **kwargs)


# --------------------------------------------------------------------------- #
# argument splitting
# --------------------------------------------------------------------------- #
def test_selectors_are_split_from_hydra_overrides_and_flags() -> None:
    selectors, overrides, flags = make_jobs.split_args(
        ["experiment=count", "seed=1,2,3", "tracking=wandb_offline", "--out", "jobs", "--force"]
    )
    assert selectors == {"experiment": ["count"], "seed": ["1", "2", "3"]}
    assert overrides == ["tracking=wandb_offline"]
    assert flags == ["--out", "jobs", "--force"]


def test_overrides_are_appended_to_every_line() -> None:
    jobs = build(overrides=["tracking=wandb_offline", "store.root=/scratch/runs"])
    assert jobs[0].command.endswith(" tracking=wandb_offline store.root=/scratch/runs")


# --------------------------------------------------------------------------- #
# the matrix
# --------------------------------------------------------------------------- #
def test_one_line_per_model_paradigm_seed_and_stage() -> None:
    jobs = build(paradigms=["global", "local"], seeds=[42, 1], stages=["test"])
    assert [j.command for j in jobs] == [
        "run experiment=count model=lightgbm_poisson paradigm=global stage=test seed=42",
        "run experiment=count model=lightgbm_poisson paradigm=global stage=test seed=1",
        "run experiment=count model=lightgbm_poisson paradigm=local stage=test seed=42",
        "run experiment=count model=lightgbm_poisson paradigm=local stage=test seed=1",
    ]


def test_cv_is_single_seed_and_uses_the_tuning_seed() -> None:
    """§5.4: "validation CV stays single-seed"."""
    jobs = build(stages=["cv"], seeds=[1, 2, 3])
    assert len(jobs) == 1
    assert jobs[0].command.endswith("stage=cv seed=42")


def test_a_deterministic_model_is_not_swept_over_seeds() -> None:
    jobs = build(models=["naive_last"], stages=["test"], seeds=[42, 1, 2])
    assert [j.command for j in jobs] == [
        "run experiment=count model=naive_last paradigm=global stage=test seed=42"
    ]


def test_the_resource_class_comes_from_the_device() -> None:
    cfg = FakeCfg({"lstm_w7": "gpu", "lightgbm_poisson": "cpu"})
    jobs = build(cfg, models=["lightgbm_poisson", "lstm_w7"], seeds=[42])
    assert {j.model: j.resource for j in jobs} == {"lightgbm_poisson": "cpu", "lstm_w7": "gpu"}


def test_the_spec_device_is_the_fallback_when_the_config_has_no_entry() -> None:
    class NoDevice(FakeCfg):
        def device_for(self, model: str) -> str:
            raise KeyError(model)  # a model the experiment YAML does not list

    assert make_jobs.resource_class(NoDevice(), "lstm_w7", SPECS["lstm_w7"]) == "gpu"
    assert make_jobs.resource_class(NoDevice(), "naive_last", SPECS["naive_last"]) == "cpu"


# --------------------------------------------------------------------------- #
# tune jobs
# --------------------------------------------------------------------------- #
def test_tune_jobs_carry_no_paradigm_or_seed_axis() -> None:
    """Tuning runs once, Global, under the tuning seed (§1 "Seeds", F14)."""
    jobs = build(stages=["tune"], paradigms=["global", "local"], seeds=[42, 1])
    assert [j.command for j in jobs] == ["tune experiment=count model=lightgbm_poisson"]


def test_a_model_without_a_search_space_gets_no_tune_job() -> None:
    assert build(models=["naive_last"], stages=["tune"]) == []


def test_a_model_with_best_params_is_not_retuned(tmp_path) -> None:
    from strikecast.store import RunStore

    store = RunStore(tmp_path / "runs")
    directory = store.tuning_dir("count", "lightgbm_poisson")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "best_params.json").write_text("{}", encoding="utf-8")

    assert build(stages=["tune"], store=store) == []
    assert len(build(stages=["tune"], store=store, force=True)) == 1


# --------------------------------------------------------------------------- #
# skipping finished work
# --------------------------------------------------------------------------- #
def test_a_complete_stage_is_not_queued_again(tmp_path) -> None:
    from strikecast.store import RunKey, RunStore

    store = RunStore(tmp_path / "runs")
    key = RunKey("count", "lightgbm_poisson", "global", 42)
    store.start_stage(key, "test", "digest")
    store.complete_stage(key, "test")

    jobs = build(seeds=[42, 1], store=store)
    assert [j.command for j in jobs] == [
        "run experiment=count model=lightgbm_poisson paradigm=global stage=test seed=1"
    ]
    assert len(build(seeds=[42, 1], store=store, force=True)) == 2


def test_a_failed_stage_is_queued_again(tmp_path) -> None:
    from strikecast.store import RunKey, RunStore

    store = RunStore(tmp_path / "runs")
    key = RunKey("count", "lightgbm_poisson", "global", 42)
    store.start_stage(key, "test", "digest")
    store.fail_stage(key, "test", "boom")
    assert len(build(seeds=[42], store=store)) == 1


# --------------------------------------------------------------------------- #
# writing the files
# --------------------------------------------------------------------------- #
def test_one_file_per_resource_class_indexable_by_line(tmp_path) -> None:
    cfg = FakeCfg({"lstm_w7": "gpu"})
    jobs = build(cfg, models=["lightgbm_poisson", "lstm_w7"], seeds=[42, 1])
    written = make_jobs.write_jobs(jobs, tmp_path / "jobs")

    assert set(written) == {"cpu", "gpu"}
    assert written["cpu"].name == "jobs.cpu.txt"
    for resource, path in written.items():
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n"), "sed -n Np drops an unterminated last line"
        lines = text.splitlines()
        assert lines == [j.command for j in jobs if j.resource == resource]
        assert all(line.startswith("run experiment=count") for line in lines)


def test_nothing_is_written_for_a_resource_class_without_jobs(tmp_path) -> None:
    written = make_jobs.write_jobs(build(seeds=[42]), tmp_path / "jobs")
    assert set(written) == {"cpu"}
    assert not (tmp_path / "jobs.gpu.txt").exists()


# --------------------------------------------------------------------------- #
# end to end, against the real configs/ tree and registry
# --------------------------------------------------------------------------- #
def test_main_expands_a_real_experiment(tmp_path, capsys) -> None:
    pytest.importorskip("hydra")
    code = make_jobs.main(
        [
            "experiment=diff",
            "model=naive_last,linear",
            "paradigm=global,local",
            "stage=test",
            "seed=42,1",
            "--out",
            "-",
            "--store-root",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # both models are deterministic, so seed 42 only: 2 models x 2 paradigms
    assert len(lines) == 4
    assert all(ln.startswith("cpu\trun experiment=diff ") for ln in lines)
    assert all(ln.endswith("stage=test seed=42") for ln in lines)


def test_main_requires_an_experiment(capsys) -> None:
    assert make_jobs.main(["stage=test"]) == 2
    assert "experiment=<name> is required" in capsys.readouterr().err


def test_main_rejects_an_unknown_stage(capsys) -> None:
    assert make_jobs.main(["experiment=diff", "stage=validation"]) == 2
    assert "unknown stage" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the thesis-faithful matrix (audit 2026-09-26 C16/D8) and the C17 skip rules
# --------------------------------------------------------------------------- #
class FakeEntry:
    def __init__(self, paradigms) -> None:
        self.paradigms = paradigms


class MatrixCfg(FakeCfg):
    paradigm_names = ["global"]
    stages = {"cv": object(), "test": object()}

    class calibration:  # noqa: N801 - mimics the config attribute
        enabled = True

    def __init__(self, entries) -> None:
        super().__init__()
        self.entries = entries

    def model_entry(self, model):
        return FakeEntry(self.entries.get(model))


class Composite(FakeSpec):
    kind = "composite"


def test_without_a_paradigm_selector_each_model_uses_its_own_list() -> None:
    cfg = MatrixCfg({"lightgbm_poisson": ["global", "activity", "local"],
                     "lstm_w7": ["global", "activity"], "naive_last": None})
    jobs = make_jobs.build_jobs(cfg, models=["lightgbm_poisson", "lstm_w7", "naive_last"],
                                paradigms=None, stages=["test"], seeds=[42], get_spec=get_spec)
    got = {(j.model, j.paradigm) for j in jobs}
    assert got == {
        ("lightgbm_poisson", "global"), ("lightgbm_poisson", "activity"),
        ("lightgbm_poisson", "local"), ("lstm_w7", "global"), ("lstm_w7", "activity"),
        ("naive_last", "global"),  # None -> the experiment's paradigms
    }
    # an explicit selector still wins, for every model
    forced = make_jobs.build_jobs(cfg, models=["lstm_w7"], paradigms=["local"], stages=["test"],
                                  seeds=[42], get_spec=get_spec)
    assert [j.paradigm for j in forced] == ["local"]


def test_an_empty_paradigm_list_is_never_a_standalone_job() -> None:
    cfg = MatrixCfg({"lightgbm_poisson": []})
    assert make_jobs.build_jobs(cfg, models=["lightgbm_poisson"], paradigms=None,
                                stages=["tune", "cv", "test"], seeds=[42], get_spec=get_spec) == []


def test_cv_needed_keeps_cv_only_for_a_calibrated_composite() -> None:
    specs = {**SPECS, "hurdle": Composite()}
    cfg = MatrixCfg({"lightgbm_poisson": ["global"], "hurdle": ["global", "local"]})
    jobs = make_jobs.build_jobs(cfg, models=["lightgbm_poisson", "hurdle"], paradigms=None,
                                stages=["cv", "test"], seeds=[42], cv="needed",
                                get_spec=lambda n, e=None: specs[n])
    cv = [(j.model, j.paradigm) for j in jobs if j.stage == "cv"]
    assert cv == [("hurdle", "global"), ("hurdle", "local")]
    all_cv = make_jobs.build_jobs(cfg, models=["lightgbm_poisson"], paradigms=None,
                                  stages=["cv"], seeds=[42], cv="all", get_spec=get_spec)
    assert len(all_cv) == 1


def test_a_stage_the_experiment_does_not_define_is_never_emitted() -> None:
    cfg = MatrixCfg({"lightgbm_poisson": ["global"]})
    cfg.stages = {"test": object()}  # Chronos-2: no cv stage (F131)
    jobs = make_jobs.build_jobs(cfg, models=["lightgbm_poisson"], paradigms=None,
                                stages=["cv", "test"], seeds=[42], cv="all", get_spec=get_spec)
    assert [j.stage for j in jobs] == ["test"]


def test_jobs_carry_the_metadata_submit_all_needs() -> None:
    cfg = MatrixCfg({"lstm_w7": ["global"]})
    (job,) = make_jobs.build_jobs(cfg, models=["lstm_w7"], paradigms=None, stages=["test"],
                                  seeds=[42], get_spec=get_spec)
    assert (job.experiment, job.paradigm, job.seed, job.env) == ("count", "global", 42, "main")


def test_chronos_runs_in_the_autogluon_env() -> None:
    class Chronos(FakeSpec):
        kind = "chronos"

    assert make_jobs.env_for(FakeCfg(), Chronos()) == "autogluon"
    assert make_jobs.env_for(FakeCfg(), FakeSpec()) == "main"


def _complete(store, key, stage, **kwargs):
    store.start_stage(key, stage, "digest", **kwargs)
    store.complete_stage(key, stage)


def test_a_stage_run_on_default_params_is_not_done(tmp_path) -> None:
    from strikecast.store import RunKey, RunStore

    store = RunStore(tmp_path / "runs")
    key = RunKey("count", "lightgbm_poisson", "global", 42)
    _complete(store, key, "test", params_source="defaults")
    assert len(build(seeds=[42], store=store)) == 1  # tunable + defaults -> re-queued
    _complete(store, key, "test", params_source="tuned")
    assert build(seeds=[42], store=store) == []


def test_a_stage_older_than_its_best_params_is_not_done(tmp_path) -> None:
    import os

    from strikecast.store import RunKey, RunStore

    store = RunStore(tmp_path / "runs")
    key = RunKey("count", "lightgbm_poisson", "global", 42)
    _complete(store, key, "test", params_source="tuned")
    best = store.tuning_dir("count", "lightgbm_poisson") / "best_params.json"
    best.parent.mkdir(parents=True)
    best.write_text("{}")
    future = best.stat().st_mtime + 3600
    os.utime(best, (future, future))  # re-tuned after the stage finished
    assert len(build(seeds=[42], store=store)) == 1


def test_a_fold_truncated_stage_is_not_done(tmp_path) -> None:
    from strikecast.store import RunKey, RunStore

    store = RunStore(tmp_path / "runs")
    key = RunKey("count", "naive_last", "global", 42)
    _complete(store, key, "test", max_folds=14)
    assert len(build(models=["naive_last"], seeds=[42], store=store)) == 1


def test_main_emits_the_thesis_matrix_by_default(tmp_path, capsys) -> None:
    pytest.importorskip("hydra")
    code = make_jobs.main(["experiment=diff", "seed=42", "--out", "-",
                           "--store-root", str(tmp_path / "runs")])
    assert code == 0
    lines = [ln.split("\t")[1] for ln in capsys.readouterr().out.splitlines() if "\t" in ln]
    assert not any("stage=cv" in ln for ln in lines)  # diff needs no CV for any table
    assert sum("stage=test" in ln for ln in lines) == 31  # 9 x 3 + 4 baselines x global
    assert not any("model=naive_last paradigm=local" in ln for ln in lines)
    assert any("model=gru_w7 paradigm=local" in ln for ln in lines)  # diff RNNs DO run local
