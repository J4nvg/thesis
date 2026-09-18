"""Unit tests for `strikecast.pipeline`: the data, tune, run and report stages.

Everything runs on a synthetic 4-region, 121-step panel and the cheapest real
model in the registry (`linear`, i.e. darts' `LinearRegressionModel` with the
legacy `COMMON_KWARGS_TAB`). No real data, no GPU, no network: the point is the
ORCHESTRATION -- stage identity and resume, the paradigm grouping, the part
layout, the metric views, the seed loop and the Optuna wiring -- not the
numbers, which the level-D equivalence suite already pins.

The experiment is called `diff` because that is the family `linear`,
`arima` and the two naives are registered for, which also makes these tests
exercise the F80 model-space override (the diff CV stage schedules on
`Diff(full).split_before(train_val_end)`, one step longer than differencing the
level CV view).
"""

from __future__ import annotations

import json
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
from strikecast.pipeline import data_stage, report_stage, run_stage, tune_stage  # noqa: E402
from strikecast.pipeline.data_stage import DataArtifacts, FeatureSets  # noqa: E402
from strikecast.store import RunKey, RunStore  # noqa: E402

N_STEPS = 121
REGIONS = ("alpha", "beta", "gamma", "delta")
ACTIVITY = {"alpha": 1, "beta": 1, "gamma": 2, "delta": 3}
TARGET = "y"
PAST = ["past_a", "past_b"]
FUTURE = ["holiday_x"]

# The schedule these fixtures produce, derived in the tests rather than
# hard-coded twice: cv has 6 folds / 1 retrain, test has 19 folds / 3 retrains.
EXPECTED_CV_FOLDS = 6
EXPECTED_TEST_FOLDS = 19
EXPECTED_TEST_PARTS = 3


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_panel(seed: int = 0) -> pd.DataFrame:
    """A long panel with the columns `build_bundle` expects."""
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
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture(scope="module")
def bundle(panel: pd.DataFrame):
    return build_bundle(panel, TARGET, PAST, FUTURE, SeriesConfig(), ACTIVITY)


@pytest.fixture
def cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="diff",
        data=DataConfig(target=TARGET),
        series=SeriesConfig(),
        transform=TransformConfig(kind="diff"),
        models=[
            ModelEntry(name="linear"),
            ModelEntry(name="naive_last"),
            ModelEntry(name="naive_weekly"),
        ],
        paradigms=[ParadigmConfig(name="global"), ParadigmConfig(name="local")],
        stages={
            "cv": StageConfig(start="cv_start_frac"),
            "test": StageConfig(start="train_val_end"),
        },
        tuning=TuningConfig(objective="RMSSE_mean"),
        seeds=SeedConfig(tuning_seed=42, eval_seeds=[42, 1]),
        tracking=TrackingConfig(backend="noop"),
        store=StoreConfig(root=str(tmp_path / "runs")),
    )


@pytest.fixture
def data(bundle) -> DataArtifacts:
    return DataArtifacts(
        bundle=bundle,
        features=FeatureSets(list(PAST), list(FUTURE), "test", "feathash"),
        panel_hash="panelhash",
        series_hash="serieshash",
        activity_by_region=dict(ACTIVITY),
    )


@pytest.fixture
def store(cfg: ExperimentConfig) -> RunStore:
    return RunStore(cfg.store.root)


# --------------------------------------------------------------------------- #
# data stage
# --------------------------------------------------------------------------- #
def test_bundle_cache_round_trips_through_the_shared_directory(cfg, store, panel) -> None:
    """`build_or_load_bundle` writes `shared/series.<hash>/` and reads it back."""
    first, digest = data_stage.build_or_load_bundle(
        cfg, store, panel, [], dict(ACTIVITY), panel_hash="panelhash"
    )
    directory = store.shared_path(cfg.name, "series", digest)
    assert (directory / "manifest.json").is_file()

    second, again = data_stage.build_or_load_bundle(
        cfg, store, panel, [], dict(ACTIVITY), panel_hash="panelhash"
    )
    assert again == digest
    assert second.region_names == first.region_names
    assert len(second.target_full[0]) == len(first.target_full[0])
    np.testing.assert_allclose(
        second.target_full[0].values(), first.target_full[0].values()
    )


def test_series_hash_changes_with_the_series_config(cfg, store, panel) -> None:
    _, first = data_stage.build_or_load_bundle(
        cfg, store, panel, [], dict(ACTIVITY), panel_hash="panelhash"
    )
    other = cfg.model_copy(update={"series": SeriesConfig(static_cols=[])})
    assert data_stage._series_hash(other, "panelhash") != first


def test_feature_sets_load_the_converted_legacy_schema(cfg, store, bundle, tmp_path) -> None:
    """A `cache_path` is loaded verbatim: the legacy sets are data, not a target (F16)."""
    legacy = tmp_path / "diffreg.json"
    legacy.write_text(
        json.dumps(
            {
                "past_covariate_components": ["past_b", "past_a"],
                "future_covariate_components": ["holiday_x"],
            }
        ),
        encoding="utf-8",
    )
    cfg = cfg.model_copy(
        update={"feature_selection": cfg.feature_selection.model_copy(
            update={"cache_path": str(legacy)}
        )}
    )
    sets = data_stage.build_or_load_features(cfg, store, bundle)
    assert sets.source == "cache_path"
    assert sets.past_keep == ["past_a", "past_b"]  # sorted, as `past_keep` is
    assert sets.future_keep == ["holiday_x"]


# --------------------------------------------------------------------------- #
# run stage: targets, schedule and the F80 override
# --------------------------------------------------------------------------- #
def test_stage_targets_are_the_legacy_lists(data) -> None:
    assert stage_targets_len(data, "cv") == len(data.bundle.target_cv_view[0])
    assert stage_targets_len(data, "test") == len(data.bundle.target_full[0])
    with pytest.raises(KeyError):
        run_stage.stage_targets(data, "holdout")


def stage_targets_len(data: DataArtifacts, stage: str) -> int:
    return len(run_stage.stage_targets(data, stage)[0])


def test_diff_cv_gets_an_explicit_model_space_one_step_longer(cfg, data) -> None:
    """F80: `Diff(full).split_before(train_val_end)` beats `Diff(level CV view)`."""
    override = run_stage.model_targets_for(cfg, data, "cv")
    assert override is not None
    level_cv = data.bundle.target_cv_view[0]
    from strikecast.transforms.diff import Diff

    natural = Diff().forward([level_cv])[0]
    assert len(override[0]) == len(natural) + 1

    # the test stage needs no override: forward() of the full series is it
    assert run_stage.model_targets_for(cfg, data, "test") is None


def test_identity_experiments_never_get_an_override(cfg, data) -> None:
    identity = cfg.model_copy(update={"transform": TransformConfig(kind="identity")})
    assert run_stage.model_targets_for(identity, data, "cv") is None


# --------------------------------------------------------------------------- #
# run stage: cv + test, parts, metrics, resume
# --------------------------------------------------------------------------- #
def test_cv_and_test_stages_write_parts_metrics_and_state(cfg, data, store) -> None:
    key = RunKey(cfg.name, "linear", "global", 42)

    cv = run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    assert not cv.skipped
    assert cv.n_folds == EXPECTED_CV_FOLDS
    assert cv.n_rows > 0
    assert cv.params_source == "defaults"

    test = run_stage.run_stage(cfg, "linear", "global", 42, "test", data, store=store)
    assert test.n_folds == EXPECTED_TEST_FOLDS
    assert len(store.part_paths(key, "test")) == EXPECTED_TEST_PARTS

    # the config snapshot and the environment record are next to the stages
    assert (store.resolve(key) / "config.yaml").is_file()
    assert store.read_env(key)["python"]

    # both stages are complete, with different identities
    states = store.read_states(key)
    assert {s.status for s in states.values()} == {"complete"}
    assert states["cv"].stage_hash != states["test"].stage_hash

    # the Global paradigm writes the four legacy views and NO activity views,
    # because the legacy Global `evaluate_long` call passes no activity map
    assert set(store.read_metrics(key, "cv")) == {
        "global",
        "per_region",
        "per_horizon",
        "per_region_horizon",
    }
    assert "MASE_mean" in store.read_metrics(key, "cv")["global"]

    # every fold of every region is on disk, numbered contiguously from 0
    preds = store.load_predictions(key, "test", legacy_order=True)
    frame = preds.frame
    assert sorted(frame["fold"].unique()) == list(range(EXPECTED_TEST_FOLDS))
    assert set(frame["region"]) == set(REGIONS)
    assert set(frame["channel"]) == {"y_pred"}


def test_a_complete_stage_is_skipped_on_the_second_call(cfg, data, store) -> None:
    first = run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    second = run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    assert not first.skipped
    assert second.skipped
    assert second.stage_hash == first.stage_hash
    assert second.n_folds == 0  # nothing was run


def test_force_reruns_a_complete_stage(cfg, data, store) -> None:
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    again = run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store, force=True)
    assert not again.skipped
    assert again.n_folds == EXPECTED_CV_FOLDS


def test_changing_the_stage_config_changes_the_identity(cfg, data, store) -> None:
    run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    changed = cfg.model_copy(
        update={
            "stages": {
                "cv": StageConfig(start="cv_start_frac", horizon=5),
                "test": cfg.stage("test"),
            }
        }
    )
    outcome = run_stage.run_stage(changed, "linear", "global", 42, "cv", data, store=store)
    assert not outcome.skipped, "a different horizon must not reuse the old stage"


def test_local_paradigm_partitions_and_adds_the_activity_views(cfg, data, store) -> None:
    key = RunKey(cfg.name, "linear", "local", 42)
    outcome = run_stage.run_stage(cfg, "linear", "local", 42, "test", data, store=store)

    assert outcome.n_folds == EXPECTED_TEST_FOLDS
    frame = store.load_predictions(key, "test").frame
    assert set(frame["region"]) == set(REGIONS)
    # Activity and Local DO pass the activity map to `evaluate_long` (:1478ff)
    assert set(store.read_metrics(key, "test")) == {
        "global",
        "per_region",
        "per_horizon",
        "per_region_horizon",
        "per_activity_level",
        "per_activity_horizon",
    }


def test_local_and_global_agree_for_a_single_region_model(cfg, data, store) -> None:
    """Grouping is a partition, not a different loop: region blocks must match."""
    run_stage.run_stage(cfg, "naive_last", "global", 42, "test", data, store=store)
    run_stage.run_stage(cfg, "naive_last", "local", 42, "test", data, store=store)
    g = store.load_predictions(
        RunKey(cfg.name, "naive_last", "global", 42), "test", legacy_order=True
    ).frame
    loc = store.load_predictions(
        RunKey(cfg.name, "naive_last", "local", 42), "test", legacy_order=True
    ).frame
    pd.testing.assert_frame_equal(g, loc)


def test_naive_models_run_without_the_engine(cfg, data, store) -> None:
    outcome = run_stage.run_stage(cfg, "naive_weekly", "global", 42, "test", data, store=store)
    assert outcome.n_folds == EXPECTED_TEST_FOLDS
    assert outcome.n_rows > 0


# --------------------------------------------------------------------------- #
# the sweep: seeds and stages
# --------------------------------------------------------------------------- #
def test_run_experiment_keeps_cv_single_seed_and_broadcasts_deterministic_models(
    cfg, data, store, monkeypatch
) -> None:
    """Sec. 5.4: CV is single-seed; `stochastic=False` models run once."""
    seen: list[tuple[str, str, int, str]] = []
    real = run_stage.run_stage

    def _spy(cfg_, model, paradigm, seed, stage, data_, **kw):
        seen.append((model, str(paradigm), int(seed), stage))
        return real(cfg_, model, paradigm, seed, stage, data_, **kw)

    monkeypatch.setattr(run_stage, "run_stage", _spy)
    run_stage.run_experiment(
        cfg,
        data=data,
        models=["naive_last"],
        paradigms=["global"],
        seeds=[42, 1],
        stages=("cv", "test"),
        store=store,
    )
    # `naive_last` is deterministic, so both stages run under seed 42 only
    assert seen == [
        ("naive_last", "global", 42, "cv"),
        ("naive_last", "global", 42, "test"),
    ]


def test_run_experiment_sweeps_seeds_on_the_test_stage_of_a_stochastic_model(
    cfg, data, store, monkeypatch
) -> None:
    from strikecast.pipeline import context

    spec = context.get_spec("linear", "diff")
    stochastic = spec.__class__(**{**spec.__dict__, "stochastic": True})
    monkeypatch.setattr(run_stage, "get_spec", lambda name, exp=None: stochastic)

    outcomes = run_stage.run_experiment(
        cfg,
        data=data,
        models=["linear"],
        paradigms=["global"],
        seeds=[42, 1],
        stages=("test",),
        store=store,
    )
    assert [o.run_key.seed for o in outcomes] == [42, 1]
    assert all(not o.skipped for o in outcomes)
    # a seed is part of the stage identity, so the two runs are separate
    assert outcomes[0].stage_hash != outcomes[1].stage_hash


# --------------------------------------------------------------------------- #
# tune stage
# --------------------------------------------------------------------------- #
def _tunable_linear_spec():
    """`linear` with a one-parameter search space, so a study is cheap.

    The registry is NOT touched: the spec is built here and injected, so this
    test cannot change what any other test or any run sees.
    """
    from strikecast.models.classical import _build_linear
    from strikecast.models.spec import ModelSpec, RunContext

    def build(params, ctx: RunContext):
        return _build_linear({}, ctx)  # the suggested parameter is a dummy

    def search_space(trial) -> dict[str, Any]:
        return {"dummy": trial.suggest_categorical("dummy", [0, 1])}

    return ModelSpec(
        name="linear",
        family="linear",
        kind="global",
        build=build,
        experiments=("diff",),
        search_space=search_space,
        stochastic=False,
        n_trials=None,
    )


def test_tune_model_writes_best_params_and_is_not_repeated(cfg, data, store, monkeypatch) -> None:
    pytest.importorskip("optuna")
    spec = _tunable_linear_spec()
    monkeypatch.setattr(tune_stage, "get_spec", lambda name, exp=None: spec)

    outcome = tune_stage.tune_model(cfg, "linear", data, store=store, n_trials=2)
    assert not outcome.skipped
    assert outcome.n_trials == 2
    assert outcome.best_value is not None

    directory = store.tuning_dir(cfg.name, "linear")
    assert (directory / "optuna.sqlite3").is_file()
    assert (directory / "trials.csv").is_file()
    payload = json.loads((directory / "best_params.json").read_text(encoding="utf-8"))
    assert payload["study_name"] == "linear"
    assert payload["n_trials"] == 2

    # Appendix C: a second call does not re-tune
    again = tune_stage.tune_model(cfg, "linear", data, store=store, n_trials=2)
    assert again.skipped and "best_params.json" in again.reason


def test_a_model_without_a_search_space_is_not_tunable(cfg, data, store) -> None:
    outcome = tune_stage.tune_model(cfg, "naive_last", data, store=store)
    assert outcome.skipped and outcome.reason == "not tunable"


def test_run_stage_picks_up_the_tuned_parameters(cfg, data, store) -> None:
    directory = store.tuning_dir(cfg.name, "linear")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "best_params.json").write_text(
        json.dumps({"variant": "linear", "best_params": {}}), encoding="utf-8"
    )
    outcome = run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    assert outcome.params_source == "tuned"


def test_experiment_params_win_over_a_stored_study(cfg, data, store) -> None:
    directory = store.tuning_dir(cfg.name, "linear")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "best_params.json").write_text(
        json.dumps({"variant": "linear", "best_params": {}}), encoding="utf-8"
    )
    cfg = cfg.model_copy(
        update={"models": [ModelEntry(name="linear", params={"multi_models": True})]}
    )
    from strikecast.pipeline.context import get_spec

    params, source = run_stage.resolve_params(cfg, get_spec("linear", "diff"), "linear", store)
    assert source == "config" and params == {"multi_models": True}


# --------------------------------------------------------------------------- #
# report stage
# --------------------------------------------------------------------------- #
def test_report_writes_a_leaderboard_from_the_stored_metrics(cfg, data, store) -> None:
    run_stage.run_stage(cfg, "naive_last", "global", 42, "test", data, store=store)
    run_stage.run_stage(cfg, "naive_weekly", "global", 42, "test", data, store=store)

    written = report_stage.report(cfg, store=store, seed=42, eval_seeds=[42])
    assert "leaderboard" in written
    frame = pd.read_csv(written["leaderboard"])
    assert len(frame) >= 2
    assert {"naive_last", "naive_weekly"} <= set(frame["model"])
