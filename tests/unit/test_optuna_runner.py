"""Unit tests for `strikecast.tuning.optuna_runner`.

Fast by construction: the objective is a quadratic with a fake fold cadence, so
a 5-trial study runs in well under a second and the `MedianPruner` still fires
on the same schedule the real objectives use (report every fold, check
immediately after).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

optuna = pytest.importorskip("optuna")

from strikecast.store import RunStore  # noqa: E402
from strikecast.tuning import (  # noqa: E402
    BEST_PARAMS_KEYS,
    TuningSettings,
    load_best_params,
    load_tuning_meta,
    make_objective,
    tune,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_TUNING = REPO_ROOT / "golden/converted/tuning/checkpoints_tune"
N_FAKE_FOLDS = 10


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakeSpec:
    """The three `ModelSpec` fields the runner reads."""

    name: str = "toy_quadratic"
    n_trials: int | None = None
    search_space: Any = None


def _search_space(trial):
    return {"x": trial.suggest_float("x", -5.0, 5.0)}


def _run_trial(params, trial) -> float:
    """A quadratic with the legacy pruning cadence.

    Mirrors `make_gbm_objective`: score the running result after every fold,
    `trial.report(value, step=fold)`, check `should_prune()` immediately, and
    return the last score.
    """
    value = 0.0
    for step in range(N_FAKE_FOLDS):
        # a "running" score that converges on the true objective
        value = (params["x"] - 2.0) ** 2 + 1.0 / (step + 1)
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned
    return value


def _settings(**kwargs) -> TuningSettings:
    return TuningSettings(metric="RMSSE_mean", **kwargs)


def _tune(tmp_path, n_trials, **kwargs):
    spec = FakeSpec(search_space=_search_space)
    return tune(
        spec,
        _run_trial,
        kwargs.pop("cfg", _settings()),
        tmp_path / "optuna.sqlite3",
        n_trials=n_trials,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# study configuration: the legacy call, verbatim
# --------------------------------------------------------------------------- #
def _rng_state(sampler: Any) -> tuple[Any, ...]:
    """A comparable snapshot of a sampler's numpy RNG state."""
    state = sampler._rng.rng.get_state()
    return (state[0], tuple(state[1].tolist()), *state[2:])


def test_study_reproduces_the_legacy_create_study_and_optimize_options(tmp_path, monkeypatch):
    seen: dict[str, Any] = {}
    real_create = optuna.create_study
    real_optimize = optuna.Study.optimize

    def spy_create(**kwargs):
        seen["create"] = kwargs
        # The sampler's RNG advances as trials are drawn, so snapshot its
        # state HERE, before `optimize` touches it.
        seen["rng_state"] = _rng_state(kwargs["sampler"])
        return real_create(**kwargs)

    def spy_optimize(self, func, **kwargs):
        seen["optimize"] = kwargs
        return real_optimize(self, func, **kwargs)

    monkeypatch.setattr(optuna, "create_study", spy_create)
    monkeypatch.setattr(optuna.Study, "optimize", spy_optimize)

    _tune(tmp_path, 3)

    create = seen["create"]
    assert create["direction"] == "minimize"
    assert create["load_if_exists"] is True
    assert create["study_name"] == "toy_quadratic"
    assert create["storage"] == f"sqlite:///{(tmp_path / 'optuna.sqlite3').resolve()}"
    assert isinstance(create["sampler"], optuna.samplers.TPESampler)
    # `TPESampler(seed=RANDOM_STATE)`: optuna keeps no readable seed attribute
    # (`_rng` is a `LazyRandomState`), so the seeding is pinned by the RNG
    # state a freshly seeded sampler starts from.
    assert seen["rng_state"] == _rng_state(optuna.samplers.TPESampler(seed=42))
    assert seen["rng_state"] != _rng_state(optuna.samplers.TPESampler(seed=43))
    assert isinstance(create["pruner"], optuna.pruners.MedianPruner)
    assert create["pruner"]._n_warmup_steps == 5

    opt = seen["optimize"]
    assert opt == {
        "n_trials": 3,
        "timeout": None,
        "n_jobs": 1,
        "catch": (),
        "gc_after_trial": False,
        "show_progress_bar": False,
    }


def test_tuning_settings_requires_an_explicit_metric():
    # F39: the two legacy `_score_fold_preds` copies disagree on their default
    # metric, so the config field has no default here.
    with pytest.raises(TypeError):
        TuningSettings()  # type: ignore[call-arg]
    assert _settings().metric == "RMSSE_mean"


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def test_writes_best_params_and_trials_with_the_golden_key_set(tmp_path):
    result = _tune(tmp_path, 5)

    assert (tmp_path / "optuna.sqlite3").exists()
    assert result.best_params_path.name == "best_params.json"
    assert result.trials_path.name == "trials.csv"
    assert result.best_params_path.exists()
    assert result.trials_path.exists()

    payload = json.loads(result.best_params_path.read_text())
    assert tuple(payload) == BEST_PARAMS_KEYS
    assert payload["variant"] == "toy_quadratic"
    assert payload["sampler"] == "TPESampler"
    assert payload["pruner"] == "MedianPruner"
    assert payload["direction"] == "MINIMIZE"
    assert payload["n_trials"] == 5
    assert payload["n_complete"] + payload["n_pruned"] + payload["n_failed"] == 5
    assert set(payload["best_params"]) == {"x"}
    assert payload["best_value"] == pytest.approx(result.best_value)

    header = result.trials_path.read_text().splitlines()[0].split(",")
    assert header[:5] == ["number", "value", "datetime_start", "datetime_complete", "duration"]
    assert header[-1] == "state"
    assert "params_x" in header
    assert len(result.trials_path.read_text().splitlines()) == 6  # header + 5 trials


@pytest.mark.skipif(not GOLDEN_TUNING.exists(), reason="golden/converted not present")
def test_format_matches_the_converted_golden_artifacts(tmp_path):
    golden_dir = sorted(p for p in GOLDEN_TUNING.iterdir() if p.is_dir())[0]
    golden_meta = json.loads((golden_dir / "best_params.json").read_text())
    golden_header = (golden_dir / "trials.csv").read_text().splitlines()[0].split(",")

    result = _tune(tmp_path, 3)
    meta = json.loads(result.best_params_path.read_text())
    header = result.trials_path.read_text().splitlines()[0].split(",")

    assert list(meta) == list(golden_meta)
    assert header[:5] == golden_header[:5]
    assert header[-1] == golden_header[-1] == "state"


def test_load_best_params_reads_store_and_golden_directories(tmp_path):
    result = _tune(tmp_path, 3)
    assert load_best_params(tmp_path) == result.best_params
    assert load_best_params(result.best_params_path) == result.best_params
    assert load_tuning_meta(tmp_path)["study_name"] == "toy_quadratic"

    with pytest.raises(FileNotFoundError):
        load_best_params(tmp_path / "nope")

    if GOLDEN_TUNING.exists():
        golden_dir = sorted(p for p in GOLDEN_TUNING.iterdir() if p.is_dir())[0]
        assert load_best_params(golden_dir)  # non-empty flat param dict


def test_writes_into_a_run_store_tuning_directory(tmp_path):
    store = RunStore(tmp_path / "runs")
    directory = store.tuning_dir("count", "lightgbm_poisson")
    result = tune(
        FakeSpec(name="lightgbm_poisson", n_trials=3, search_space=_search_space),
        _run_trial,
        _settings(),
        directory / "optuna.sqlite3",
    )
    assert sorted(p.name for p in directory.iterdir()) == [
        "best_params.json",
        "optuna.sqlite3",
        "trials.csv",
    ]
    assert result.study_name == "lightgbm_poisson"


# --------------------------------------------------------------------------- #
# resume (§5.3) and the trial budget
# --------------------------------------------------------------------------- #
def test_sqlite_resume_continues_to_exactly_n_trials(tmp_path):
    first = _tune(tmp_path, 3)
    assert first.n_trials == 3

    # same budget: nothing to add
    again = _tune(tmp_path, 3)
    assert again.n_trials == 3
    assert again.study_name == first.study_name

    # larger budget: top up to exactly 8 finished trials in the SAME study
    third = _tune(tmp_path, 8)
    assert third.n_trials == 8
    assert json.loads(third.best_params_path.read_text())["n_trials"] == 8
    assert len(third.trials_path.read_text().splitlines()) == 9

    reopened = optuna.load_study(
        study_name="toy_quadratic", storage=f"sqlite:///{(tmp_path / 'optuna.sqlite3').resolve()}"
    )
    assert len(reopened.trials) == 8
    assert reopened.user_attrs["tuning_seed"] == 42
    assert reopened.user_attrs["metric"] == "RMSSE_mean"


def test_n_trials_resolution_prefers_the_spec_over_the_config(tmp_path):
    # `n_trials = 25 if is_nn else OPTUNA_N_TRIALS` is per variant in the legacy
    # script, so ModelSpec.n_trials beats the experiment-wide config value.
    spec = FakeSpec(name="from_spec", n_trials=2, search_space=_search_space)
    result = tune(spec, _run_trial, _settings(n_trials=7), tmp_path / "optuna.sqlite3")
    assert result.n_trials == 2

    # ...and the explicit argument beats both
    result = tune(spec, _run_trial, _settings(n_trials=7), tmp_path / "b.sqlite3", n_trials=1)
    assert result.n_trials == 1

    # ...and the config is the fallback when the spec has none
    result = tune(
        FakeSpec(search_space=_search_space),
        _run_trial,
        _settings(n_trials=1),
        tmp_path / "c.sqlite3",
    )
    assert result.n_trials == 1


def test_missing_trial_budget_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no trial budget"):
        tune(FakeSpec(search_space=_search_space), _run_trial, _settings(), tmp_path / "s.sqlite3")


# --------------------------------------------------------------------------- #
# pruning and failures
# --------------------------------------------------------------------------- #
def test_pruning_is_recorded_and_the_best_trial_is_complete(tmp_path):
    # a wide search space and 12 trials make the MedianPruner fire
    result = _tune(tmp_path, 12)
    assert result.n_complete >= 1
    assert result.n_pruned >= 1
    assert result.n_complete + result.n_pruned + result.n_failed == 12
    assert result.best_value == pytest.approx(
        (result.best_params["x"] - 2.0) ** 2 + 1.0 / N_FAKE_FOLDS
    )


def test_a_failing_trial_aborts_the_study_unless_catch_is_set(tmp_path):
    def boom(params, trial):
        raise ValueError("model blew up")

    spec = FakeSpec(search_space=_search_space)
    # legacy default: no `catch=`, so the exception propagates and the study dies
    with pytest.raises(ValueError, match="model blew up"):
        tune(spec, boom, _settings(), tmp_path / "a.sqlite3", n_trials=3)

    result = tune(
        spec,
        boom,
        _settings(catch=(ValueError,)),
        tmp_path / "b.sqlite3",
        n_trials=3,
    )
    assert result.n_failed == 3
    assert result.best_value is None
    assert result.best_params == {}


def test_make_objective_refuses_a_model_without_a_search_space():
    with pytest.raises(ValueError, match="not tunable"):
        make_objective(FakeSpec(name="naive_last"), _run_trial)


def test_make_objective_suggests_every_parameter_before_the_first_fold():
    order: list[str] = []

    def spy_space(trial):
        order.append("suggest")
        return _search_space(trial)

    def spy_run(params, trial):
        order.append("run")
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(FakeSpec(search_space=spy_space), spy_run), n_trials=2)
    assert order == ["suggest", "run", "suggest", "run"]
