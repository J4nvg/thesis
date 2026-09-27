"""Unit tests for `strikecast.cli.main`, the Hydra entry point (plan sec. 5.6).

The CLI is invoked IN PROCESS (`main([...])`) against a throw-away `configs/`
tree in `tmp_path`, so these tests do not depend on the repository's own
configs and cannot be broken by an edit to them. The data stage is stubbed with
the synthetic bundle from `test_pipeline_stages`: what is under test is the
argument grammar, the composition and the wiring, not the panel builder.

Every invocation passes `tracking=noop`, so nothing here can touch the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The synthetic panel lives next door; `tests/` is not a package, so the test
# directory goes on the path explicitly rather than relying on pytest's rootdir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("darts")
pytest.importorskip("hydra")

from test_pipeline_stages import (  # noqa: E402
    ACTIVITY,
    FUTURE,
    PAST,
    TARGET,
    make_panel,
)

from strikecast.cli import main as cli  # noqa: E402
from strikecast.config.schema import SeriesConfig  # noqa: E402
from strikecast.data.series import build_bundle  # noqa: E402
from strikecast.pipeline import data_stage  # noqa: E402
from strikecast.pipeline.data_stage import DataArtifacts, FeatureSets  # noqa: E402
from strikecast.store import RunKey, RunStore  # noqa: E402

EXPERIMENT_YAML = """# @package _global_
defaults:
  - /tracking: noop
  - /paradigm: global
  - _self_

name: diff

data:
  target: y
  fixed_dir: unused/fixed
  dataset_dir: unused/dataset

transform:
  kind: diff

models:
  - linear
  - naive_last
  - naive_weekly

stages:
  cv:
    start: cv_start_frac
  test:
    start: train_val_end

seeds:
  tuning_seed: 42
  eval_seeds: [42, 1]

store:
  root: runs
"""

NOOP_YAML = """# @package _global_
tracking:
  backend: noop
  mode: offline
"""

GLOBAL_PARADIGM_YAML = """# @package _global_
paradigms:
  - name: global
"""

LOCAL_PARADIGM_YAML = """# @package _global_
paradigms:
  - name: local
"""


@pytest.fixture
def config_dir(tmp_path):
    root = tmp_path / "configs"
    (root / "experiment").mkdir(parents=True)
    (root / "tracking").mkdir(parents=True)
    (root / "paradigm").mkdir(parents=True)
    (root / "experiment" / "diff.yaml").write_text(EXPERIMENT_YAML, encoding="utf-8")
    (root / "tracking" / "noop.yaml").write_text(NOOP_YAML, encoding="utf-8")
    (root / "paradigm" / "global.yaml").write_text(GLOBAL_PARADIGM_YAML, encoding="utf-8")
    (root / "paradigm" / "local.yaml").write_text(LOCAL_PARADIGM_YAML, encoding="utf-8")
    return root


@pytest.fixture
def stub_data(monkeypatch):
    """Replace the data stage with the synthetic bundle."""
    panel = make_panel()
    bundle = build_bundle(panel, TARGET, PAST, FUTURE, SeriesConfig(), ACTIVITY)
    artifacts = DataArtifacts(
        bundle=bundle,
        features=FeatureSets(list(PAST), list(FUTURE), "test", "feathash"),
        panel_hash="panelhash",
        series_hash="serieshash",
        activity_by_region=dict(ACTIVITY),
    )
    monkeypatch.setattr(data_stage, "prepare_data", lambda *a, **k: artifacts)
    return artifacts


def _run(config_dir, store_root, *tokens: str) -> int:
    return cli.main(
        [
            *tokens,
            "--config-dir",
            str(config_dir),
            "--store-root",
            str(store_root),
        ]
    )


# --------------------------------------------------------------------------- #
# argument grammar
# --------------------------------------------------------------------------- #
def test_split_overrides_separates_selectors_from_hydra_overrides() -> None:
    experiment, selectors, passthrough = cli.split_overrides(
        [
            "experiment=count",
            "model=lightgbm_poisson,xgboost_tweedie",
            "paradigm=global,activity,local",
            "stage=cv,test",
            "seed=1,2,3",
            "tracking=noop",
            "seeds.eval_seeds=[1,2]",
        ]
    )
    assert experiment == "count"
    assert selectors == {
        "model": ["lightgbm_poisson", "xgboost_tweedie"],
        "paradigm": ["global", "activity", "local"],
        "stage": ["cv", "test"],
        "seed": ["1", "2", "3"],
    }
    # a Hydra value that happens to contain a comma is NOT split
    assert passthrough == ["tracking=noop", "seeds.eval_seeds=[1,2]"]


def test_a_bare_token_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.split_overrides(["run"])


def test_experiment_is_required(config_dir, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _run(config_dir, tmp_path / "runs", "run", "model=linear")


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def test_verify_lists_every_golden_level(capsys, tmp_path) -> None:
    assert cli.main(["verify", "--golden", str(tmp_path / "nothing")]) == 0
    out = capsys.readouterr().out
    for level, what, _how in cli.GOLDEN_LEVELS:
        assert f"  {level}  {what}" in out
    assert "MISSING" in out  # the golden directory does not exist


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def test_run_cv_and_test_for_one_model(config_dir, tmp_path, stub_data, capsys) -> None:
    store_root = tmp_path / "runs"
    code = _run(
        config_dir,
        store_root,
        "run",
        "experiment=diff",
        "model=linear",
        "paradigm=global",
        "stage=cv,test",
        "seed=42",
        "tracking=noop",
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "diff/linear/global/seed=42/cv" in out
    assert "diff/linear/global/seed=42/test" in out

    store = RunStore(store_root)
    key = RunKey("diff", "linear", "global", 42)
    states = store.read_states(key)
    assert {name: s.status for name, s in states.items()} == {
        "cv": "complete",
        "test": "complete",
    }
    assert store.read_metrics(key, "test")["global"]["MASE_mean"] is not None

    # the store root really is the one the CLI was given, resolved absolute
    assert (store_root / "diff" / "linear" / "global" / "seed=42").is_dir()


def test_a_second_run_skips_the_completed_stage(config_dir, tmp_path, stub_data, capsys) -> None:
    store_root = tmp_path / "runs"
    args = ("run", "experiment=diff", "model=naive_last", "stage=test", "seed=42", "tracking=noop")
    _run(config_dir, store_root, *args)
    capsys.readouterr()
    _run(config_dir, store_root, *args)
    out = capsys.readouterr().out
    assert "skip  diff/naive_last/global/seed=42/test" in out
    assert "1 stage(s) run, 0 skipped" not in out


def test_a_single_paradigm_selector_also_composes_its_group(
    config_dir, tmp_path, stub_data, capsys
) -> None:
    """`paradigm=local` selects the runs AND lands in the config snapshot."""
    store_root = tmp_path / "runs"
    assert (
        _run(
            config_dir,
            store_root,
            "run",
            "experiment=diff",
            "model=naive_last",
            "paradigm=local",
            "stage=test",
            "seed=42",
            "tracking=noop",
        )
        == 0
    )
    store = RunStore(store_root)
    key = RunKey("diff", "naive_last", "local", 42)
    assert store.read_config(key)["paradigms"] == [{"name": "local"}]


def test_multirun_flag_expands_the_product_in_process(
    config_dir, tmp_path, stub_data, capsys
) -> None:
    store_root = tmp_path / "runs"
    code = cli.main(
        [
            "run",
            "-m",
            "experiment=diff",
            "model=naive_last,naive_weekly",
            "stage=test",
            "seed=42",
            "tracking=noop",
            "--config-dir",
            str(config_dir),
            "--store-root",
            str(store_root),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "diff/naive_last/global/seed=42/test" in out
    assert "diff/naive_weekly/global/seed=42/test" in out


# --------------------------------------------------------------------------- #
# evaluate and report
# --------------------------------------------------------------------------- #
def test_evaluate_recomputes_metrics_without_refitting(
    config_dir, tmp_path, stub_data, capsys
) -> None:
    store_root = tmp_path / "runs"
    _run(config_dir, store_root, "run", "experiment=diff", "model=naive_last",
         "stage=test", "seed=42", "tracking=noop")
    store = RunStore(store_root)
    key = RunKey("diff", "naive_last", "global", 42)
    metrics_path = store.metrics_dir(key, "test") / "global.json"
    metrics_path.unlink()
    capsys.readouterr()

    assert _run(config_dir, store_root, "evaluate", "experiment=diff", "model=naive_last",
                "stage=test", "seed=42", "tracking=noop") == 0
    assert "1 stage(s) re-evaluated" in capsys.readouterr().out
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["MASE_mean"] is not None


def test_report_writes_the_leaderboard(config_dir, tmp_path, stub_data, capsys) -> None:
    store_root = tmp_path / "runs"
    _run(config_dir, store_root, "run", "experiment=diff", "model=naive_last,naive_weekly",
         "stage=test", "seed=42", "tracking=noop")
    capsys.readouterr()

    assert _run(config_dir, store_root, "report", "experiment=diff", "seed=42",
                "tracking=noop") == 0
    out = capsys.readouterr().out
    assert "leaderboard:" in out
    assert (store_root / "diff" / "report" / "leaderboard.csv").is_file()


# --------------------------------------------------------------------------- #
# tune
# --------------------------------------------------------------------------- #
def test_tune_skips_a_model_without_a_search_space(
    config_dir, tmp_path, stub_data, capsys
) -> None:
    store_root = tmp_path / "runs"
    assert _run(config_dir, store_root, "tune", "experiment=diff", "model=naive_last",
                "tracking=noop") == 0
    assert "naive_last: skip (not tunable)" in capsys.readouterr().out


def test_a_paradigm_group_that_cannot_compose_stays_a_selector(
    config_dir, tmp_path, stub_data, capsys
) -> None:
    """An experiment whose defaults list has no `paradigm` entry still runs.

    The group override is a provenance nicety, not a requirement: without it
    the run still lands under `.../local/seed=42/`, it is only the config
    snapshot that keeps naming the experiment's own paradigm list.
    """
    text = (config_dir / "experiment" / "diff.yaml").read_text(encoding="utf-8")
    (config_dir / "experiment" / "nogroup.yaml").write_text(
        text.replace("  - /paradigm: global\n", "").replace(
            "name: diff", "name: diff\nparadigms:\n  - name: global"
        ),
        encoding="utf-8",
    )
    store_root = tmp_path / "runs"
    assert (
        _run(
            config_dir,
            store_root,
            "run",
            "experiment=nogroup",
            "model=naive_last",
            "paradigm=local",
            "stage=test",
            "seed=42",
            "tracking=noop",
        )
        == 0
    )
    assert (store_root / "diff" / "naive_last" / "local" / "seed=42").is_dir()


# --------------------------------------------------------------------------- #
# audit 2026-09-26 C17 / C21: no silent defaults, fold-limited runs, signals
# --------------------------------------------------------------------------- #
TUNABLE_YAML = EXPERIMENT_YAML.replace("models:\n  - linear\n", "models:\n  - lightgbm\n  - linear\n")


@pytest.fixture
def tunable_dir(config_dir):
    (config_dir / "experiment" / "difft.yaml").write_text(TUNABLE_YAML, encoding="utf-8")
    return config_dir


def test_a_tunable_model_without_best_params_is_an_error(tunable_dir, tmp_path, stub_data) -> None:
    from strikecast.pipeline.run_stage import MissingTunedParams

    store_root = tmp_path / "runs"
    with pytest.raises(MissingTunedParams, match="--allow-default-params"):
        _run(tunable_dir, store_root, "run", "experiment=difft", "model=lightgbm",
             "stage=cv", "seed=42", "tracking=noop")
    state = RunStore(store_root).read_state(RunKey("diff", "lightgbm", "global", 42), "cv")
    assert state.status == "failed" and "MissingTunedParams" in state.error


def test_allow_default_params_and_max_folds(tunable_dir, tmp_path, stub_data, capsys) -> None:
    pytest.importorskip("lightgbm")
    store_root = tmp_path / "runs"
    code = _run(tunable_dir, store_root, "run", "experiment=difft", "model=lightgbm",
                "stage=cv", "seed=42", "tracking=noop", "--allow-default-params",
                "--max-folds", "3")
    assert code == 0
    assert "folds=3" in capsys.readouterr().out
    store = RunStore(store_root)
    key = RunKey("diff", "lightgbm", "global", 42)
    state = store.read_state(key, "cv")
    assert (state.status, state.params_source, state.max_folds, state.folds_done) == (
        "complete", "defaults", 3, 3,
    )
    assert sorted(store.load_predictions(key, "cv").frame["fold"].unique()) == [0, 1, 2]


def test_max_folds_is_part_of_the_stage_identity(config_dir, tmp_path, stub_data, capsys) -> None:
    store_root = tmp_path / "runs"
    args = ("run", "experiment=diff", "model=linear", "stage=cv", "seed=42", "tracking=noop")
    _run(config_dir, store_root, *args, "--max-folds", "2")
    capsys.readouterr()
    _run(config_dir, store_root, *args)  # the full stage is NOT skipped as "complete"
    out = capsys.readouterr().out
    assert "done  diff/linear/global/seed=42/cv" in out
    state = RunStore(store_root).read_state(RunKey("diff", "linear", "global", 42), "cv")
    assert state.max_folds is None and state.folds_done > 2


def test_sigterm_marks_the_stage_interrupted_and_exits_99(
    config_dir, tmp_path, stub_data, monkeypatch
) -> None:
    import os
    import signal

    from strikecast.pipeline import run_stage

    def killed(**kwargs):
        os.kill(os.getpid(), signal.SIGTERM)  # what SLURM does at the wall clock
        raise AssertionError("the handler should have raised")

    monkeypatch.setattr(run_stage, "_run_backtest", killed)
    before = signal.getsignal(signal.SIGTERM)
    store_root = tmp_path / "runs"
    code = _run(config_dir, store_root, "run", "experiment=diff", "model=linear", "stage=cv",
                "seed=42", "tracking=noop")
    assert code == 99
    state = RunStore(store_root).read_state(RunKey("diff", "linear", "global", 42), "cv")
    assert state.status == "interrupted" and "SIGTERM" in state.error
    assert state.attempts[-1]["outcome"] == "interrupted"
    assert signal.getsignal(signal.SIGTERM) is before  # handlers restored after main()
