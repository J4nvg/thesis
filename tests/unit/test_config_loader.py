"""The ``configs/`` Hydra tree and :func:`strikecast.config.loader.load_experiment`.

Three things are asserted here:

1. **Composition works.** Each of the four in-scope experiments composes from
   its group defaults and validates into an :class:`ExperimentConfig`, and
   overrides behave like they do on the command line.
2. **Typos fail loudly.** Every schema sets ``extra="forbid"``, so a misspelled
   key is either a Hydra composition error or a pydantic validation error --
   never a silently ignored setting.
3. **The per-family settings of plan sec. 2.3 are what the legacy scripts ran.**
   This is the regression test for the frozen methodology: if somebody edits a
   YAML and changes a device, a loss, a stride or a naive-scale denominator,
   this file fails.

The model lineups are cross-checked against the registry rather than against a
hard-coded list, so a model added to ``models/*.py`` without a YAML line (or the
other way round) is caught here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Importing the four model modules is what populates the registry; there is no
# import side effect in `strikecast.models.__init__` that does it for us.
import strikecast.models.classical  # noqa: F401
import strikecast.models.classifiers  # noqa: F401
import strikecast.models.gbm  # noqa: F401
import strikecast.models.rnn  # noqa: F401
from strikecast.config.loader import default_config_dir, load_experiment
from strikecast.config.schema import ExperimentConfig
from strikecast.models.spec import get_spec, registered_names

EXPERIMENTS = ("count", "diff", "hurdle", "damage")

#: Every file under `configs/experiment/`. `chronos2` is the P5 family added by
#: Stream H; it is deliberately NOT in `EXPERIMENTS`, because the assertions below
#: are the frozen plan-sec.-2.3 settings of the four P3 families and Chronos-2
#: shares none of them (no CV stage, no feature selection, no darts lags).
#: Its own config assertions live in `tests/unit/test_chronos_adapter.py`.
EXPERIMENT_FILES = (*EXPERIMENTS, "chronos2")


@pytest.fixture(scope="module")
def configs() -> dict[str, ExperimentConfig]:
    """Every experiment, composed once."""
    return {name: load_experiment(name) for name in EXPERIMENTS}


# --------------------------------------------------------------------------- #
# 1. composition                                                               #
# --------------------------------------------------------------------------- #
def test_the_configs_tree_sits_next_to_the_package() -> None:
    root = default_config_dir()
    assert root.is_dir()
    assert sorted(p.stem for p in (root / "experiment").glob("*.yaml")) == sorted(
        EXPERIMENT_FILES
    )


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_composes_and_validates(name: str) -> None:
    cfg = load_experiment(name)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == name


@pytest.mark.parametrize(
    "spelling",
    ["count", "experiment/count", "configs/experiment/count.yaml"],
)
def test_an_experiment_can_be_named_three_ways(spelling: str) -> None:
    assert load_experiment(spelling).name == "count"


def test_an_unknown_experiment_names_the_ones_that_exist() -> None:
    with pytest.raises(FileNotFoundError, match="no experiment config 'nope'"):
        load_experiment("nope")


def test_a_missing_config_dir_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_experiment("count", config_dir=tmp_path / "absent")


def test_composing_twice_in_one_process_works(configs) -> None:
    """Hydra's global singleton is cleared around every call."""
    first = load_experiment("count")
    second = load_experiment("count")
    assert first.resolved_hash() == second.resolved_hash()


# --------------------------------------------------------------------------- #
# 2. overrides and typos                                                       #
# --------------------------------------------------------------------------- #
def test_a_group_override_swaps_the_group() -> None:
    cfg = load_experiment("count", ["tracking=noop"])
    assert cfg.tracking.backend == "noop"
    assert load_experiment("count").tracking.backend == "wandb"


def test_a_value_override_replaces_a_leaf() -> None:
    cfg = load_experiment("count", ["seeds.eval_seeds=[1,2]", "models=[lightgbm_poisson]"])
    assert cfg.seeds.eval_seeds == [1, 2]
    assert cfg.model_names == ["lightgbm_poisson"]


def test_the_paradigm_group_is_swappable() -> None:
    assert load_experiment("count", ["paradigm=activity"]).paradigm_names == ["activity"]
    assert load_experiment("count", ["paradigm=local"]).paradigm_names == ["local"]


def test_a_typo_in_an_existing_key_is_refused_by_hydra() -> None:
    from hydra.errors import ConfigCompositionException

    with pytest.raises(ConfigCompositionException, match="data.targett"):
        load_experiment("count", ["data.targett=x"])


def test_a_typo_forced_in_with_plus_is_refused_by_pydantic() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_experiment("count", ["+data.targett=x"])


def test_an_unknown_group_option_is_refused() -> None:
    from hydra.errors import MissingConfigException

    with pytest.raises(MissingConfigException):
        load_experiment("count", ["tracking=does_not_exist"])


def test_a_typo_written_into_a_yaml_file_is_refused(tmp_path) -> None:
    """The same guard, but reached from a file rather than an override."""
    (tmp_path / "experiment").mkdir()
    (tmp_path / "experiment" / "typo.yaml").write_text(
        "# @package _global_\nname: typo\ntransfrom:\n  kind: identity\n"
    )
    with pytest.raises(ValidationError, match="transfrom"):
        load_experiment("typo", config_dir=tmp_path)


# --------------------------------------------------------------------------- #
# 3. the frozen per-family settings of plan sec. 2.3                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_lineup_is_exactly_what_the_registry_holds(name: str, configs) -> None:
    cfg = configs[name]
    assert sorted(cfg.model_names) == sorted(registered_names(name))
    assert len(cfg.model_names) == len(set(cfg.model_names))
    for model in cfg.model_names:
        get_spec(model, name)  # raises if the name is not registered for `name`


def test_the_count_lineup_is_the_legacy_one(configs) -> None:
    """``GBM_VARIANTS`` (_regression_GBDT.py:118) + the 15 RNN variants
    (_regression_LSTM.py:1126-1140), and no baselines (F84, F86)."""
    names = configs["count"].model_names
    assert names[:6] == [
        "lightgbm_poisson",
        "lightgbm_tweedie",
        "xgboost_poisson",
        "xgboost_tweedie",
        "catboost_poisson",
        "catboost_tweedie",
    ]
    assert len(names) == 21
    assert not {"naive_last", "naive_weekly", "linear", "arima"} & set(names)


def test_the_diff_lineup_keeps_its_baselines(configs) -> None:
    """``REGRESSORS_TO_RUN`` (_diff_regression.py:123-128) really ran here."""
    cfg = configs["diff"]
    assert {"linear", "arima", "naive_last", "naive_weekly"} <= set(cfg.model_names)
    # F5: ARIMA falls back to NaiveMean on any exception.
    assert cfg.model_entry("arima").fallback == "naive_mean"
    assert cfg.model_entry("linear").fallback is None


@pytest.mark.parametrize(
    ("name", "kind"),
    [("count", "identity"), ("diff", "diff"), ("hurdle", "identity"), ("damage", "identity")],
)
def test_the_target_transform_is_per_family(name: str, kind: str, configs) -> None:
    assert configs[name].transform.kind == kind


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_darts_skeleton_is_shared(name: str, configs) -> None:
    """``get_common_kwargs()`` with its defaults, in every family (F15, F26)."""
    ck = configs[name].common_kwargs
    assert ck.lags == 7
    assert ck.lags_past_covariates == [-1, -7, -14]
    assert ck.lags_future_covariates == (2, 7)
    assert isinstance(ck.as_kwargs()["lags_future_covariates"], tuple)
    assert ck.output_chunk_length == 7
    assert ck.output_chunk_shift == 0
    assert ck.add_encoders == {
        "cyclic": {"future": ["month", "week", "dayofyear", "dayofweek", "day"]}
    }


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_split_is_70_10_20(name: str, configs) -> None:
    split = configs[name].split
    assert (split.train, split.val, split.test) == (0.70, 0.10, 0.20)
    # F17: train_val_end is the float-rounding artefact the legacy code has.
    assert split.train_val_end == 0.7999999999999999
    assert split.cv_start_frac == 0.875


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_retrain_cadence_is_weekly_with_daily_predictions(name: str, configs) -> None:
    cfg = configs[name]
    assert sorted(cfg.stages) == ["cv", "test"]
    assert cfg.stage("cv").start == "cv_start_frac"
    assert cfg.stage("test").start == "train_val_end"
    for stage_name in ("cv", "test"):
        stage = cfg.stage(stage_name)
        assert stage.horizon == 7
        assert stage.predict_stride == 1
        assert stage.retrain_stride == 7
    assert cfg.stage("cv").start_frac(cfg.split) == 0.875
    assert cfg.stage("test").start_frac(cfg.split) == 0.7999999999999999


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_only_the_global_cv_stage_takes_the_sampled_median(name: str, configs) -> None:
    """F81: the Activity/Local CV wrappers delegate to ``run_final_test``."""
    cfg = configs[name]
    assert cfg.stage("cv").adapter_for("global") == "for_cv"
    assert cfg.stage("test").adapter_for("global") == "for_test"
    for paradigm in ("activity", "local"):
        assert cfg.stage("cv").adapters[paradigm] == "for_test"
        assert cfg.stage("test").adapters[paradigm] == "for_test"


def test_the_count_family_runs_its_gbdts_on_cpu(configs) -> None:
    """Plan sec. 1 "Device policy" and sec. 2.3 "Device"."""
    cfg = configs["count"]
    assert cfg.device_for("lightgbm_poisson") == "cpu"
    assert cfg.device_for("xgboost_tweedie") == "cpu"
    assert cfg.device_for("catboost_poisson") == "CPU"


def test_the_diff_family_runs_xgboost_and_catboost_on_the_gpu(configs) -> None:
    cfg = configs["diff"]
    assert cfg.device_for("lightgbm") == "cpu"
    assert cfg.device_for("xgboost") == "cuda"
    assert cfg.device_for("catboost") == "GPU"


def test_the_hurdle_count_head_runs_on_cpu(configs) -> None:
    assert configs["hurdle"].device_for("catboost_tweedie_count_head") == "CPU"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_configured_device_matches_its_spec(name: str, configs) -> None:
    cfg = configs[name]
    for model in cfg.model_names:
        assert cfg.device_for(model) == get_spec(model, name).device


@pytest.mark.parametrize(
    ("name", "selector", "device", "threads"),
    [
        # One `build_regressor("lightgbm_tweedie")` on level counts, GPU, top-100.
        ("count", "countreg", "gpu", None),
        # Its own selection: objective "regression" on CPU with 4 threads.
        ("diff", "diffreg", "cpu", 4),
        # Library-default LightGBM, Poisson with positive-only weights (F27).
        ("hurdle", "zipoisson_regressor", "cpu", 8),
        # The damage family reuses the hurdle classifier selection, per key.
        ("damage", "zipoisson_classifier", "cpu", 8),
    ],
)
def test_feature_selection_is_per_family(
    name: str, selector: str, device: str, threads: int | None, configs
) -> None:
    """The THESIS selector settings, i.e. under ``legacy=<name>``.

    (hurdle/damage passed no device/thread kwarg at all, so their legacy
    overrides keep the deterministic publication settings: there is no thesis
    cache to reproduce, A11.)
    """
    fs = load_experiment(name, [f"legacy={name}"]).feature_selection
    assert fs.selector == selector
    assert fs.device == device
    assert fs.num_threads == threads
    assert fs.top_k == 100  # F25: 100 LAGGED names, not 100 base features
    # the publication config: same selector, deterministic CPU, cached once
    pub = configs[name].feature_selection
    assert pub.selector == selector
    assert (pub.device, pub.num_threads, pub.deterministic) == ("cpu", 8, True)
    assert pub.top_k == 100


def test_the_hurdle_family_has_two_feature_selections(configs) -> None:
    """F27: a binary selection for the classifier, a Poisson one for the count
    head, each subsetting its own head's covariates (final_hurdle.ipynb cell 11)."""
    cfg = configs["hurdle"]
    assert cfg.feature_selection_for("classifier").selector == "zipoisson_classifier"
    assert cfg.feature_selection_for("regressor").selector == "zipoisson_regressor"
    # An unlisted head falls back to the family's own selection.
    assert cfg.feature_selection_for("nope").selector == "zipoisson_regressor"
    assert cfg.feature_selection_for() is cfg.feature_selection


def test_only_the_damage_family_selects_per_key(configs) -> None:
    assert configs["damage"].feature_selection.per_key is True
    for name in ("count", "diff", "hurdle"):
        assert configs[name].feature_selection.per_key is False


def test_only_the_regression_families_cache_their_selection(configs) -> None:
    # Thesis mode (`legacy=<name>`): the converted phase-0 JSON, not the legacy
    # pickle -- `build_or_load_features` reads `cache_path` as JSON (F16).
    assert (
        load_experiment("count", ["legacy=count"]).feature_selection.cache_path
        == "golden/converted/feature_sets/countreg.json"
    )
    assert (
        load_experiment("diff", ["legacy=diff"]).feature_selection.cache_path
        == "golden/converted/feature_sets/diffreg.json"
    )
    for name in ("hurdle", "damage"):
        legacy = load_experiment(name, [f"legacy={name}"])
        assert legacy.feature_selection.cache_path is None
    # Publication mode (audit A13/D2): no golden set, selected once by
    # `strikecast featsel`, cached, and required by every later job.
    for name in ("count", "diff", "hurdle", "damage"):
        fs = configs[name].feature_selection
        assert fs.cache_path is None
        assert fs.cache is True and fs.require_cached is True and fs.deterministic is True
        for head in configs[name].feature_selections.values():
            assert head.cache is True and head.require_cached is True and head.deterministic


def test_publication_configs_use_the_leaky_filter_and_legacy_pins_expdecay7(configs) -> None:
    """Audit A1/D1: publication = ``leaky``; ``legacy=<name>`` = the thesis filter."""
    for name in ("count", "diff", "hurdle", "damage"):
        assert configs[name].series.window.expdecay == "leaky"
        assert configs[name].series.window.transforms[-1]["function_name"] == "leaky7"
        legacy = load_experiment(name, [f"legacy={name}"])
        assert legacy.series.window.expdecay == "legacy_alpha"
        assert legacy.series.window.transforms[-1]["function_name"] == "expdecay7"
        assert legacy.resolved_hash() != configs[name].resolved_hash()


def test_only_the_hurdle_test_stage_scales_on_train_plus_val(configs) -> None:
    """F67: ``compute_naive_scales([tr.append(vl) ...])`` in final_hurdle.ipynb
    cell 26, against a train-only denominator everywhere else."""
    assert configs["hurdle"].stage("test").naive_scales.fit_on == "train_val"
    assert configs["hurdle"].stage("cv").naive_scales.fit_on == "train"
    for name in ("count", "diff", "damage"):
        for stage in ("cv", "test"):
            assert configs[name].stage(stage).naive_scales.fit_on == "train"
    for name in EXPERIMENTS:
        for stage in ("cv", "test"):
            assert configs[name].stage(stage).naive_scales.seasonality == 7


def test_calibration_is_on_for_the_composites_only(configs) -> None:
    for name in ("count", "diff"):
        assert configs[name].calibration.enabled is False
    for name in ("hurdle", "damage"):
        cal = configs[name].calibration
        assert cal.enabled is True
        assert cal.method == "sigmoid"  # CAL_METHOD
        assert cal.group_by == "horizon"
        assert cal.threshold == 0.5  # F70
        assert cal.cv_application == "in_sample"  # F69
        assert cal.oof_splits == 5
        assert cal.oof_random_state == 42
    # Venn-Abers is reported by the hurdle notebook only.
    assert configs["hurdle"].calibration.venn_abers is True
    assert configs["damage"].calibration.venn_abers is False


def test_the_tuned_families_minimise_rmsse_mean_over_50_trials(configs) -> None:
    """F39: both Optuna objectives pass ``metric="RMSSE_mean"`` explicitly."""
    for name in ("count", "diff"):
        tuning = configs[name].tuning
        assert tuning is not None
        assert tuning.objective == "RMSSE_mean"
        assert tuning.direction == "minimize"
        assert tuning.sampler == "tpe"
        assert tuning.pruner.kind == "median"
        assert tuning.pruner.n_warmup_steps == 5
        assert tuning.timeout_s is None
        assert tuning.load_if_exists is True  # F11
        assert tuning.trials_for("gbdt") == 50
        assert tuning.trials_for("rnn") == 50


def test_the_composite_families_were_never_tuned(configs) -> None:
    """F7: no Optuna study exists for the hurdle or damage family."""
    for name in ("hurdle", "damage"):
        assert configs[name].tuning is None
        for model in configs[name].model_names:
            assert get_spec(model, name).search_space is None


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_seed_policy_is_tune_once_evaluate_over_five(name: str, configs) -> None:
    seeds = configs[name].seeds
    assert seeds.tuning_seed == 42  # RANDOM_STATE
    assert seeds.eval_seeds == [42, 1, 2, 3, 4]
    assert seeds.eval_seeds[0] == seeds.tuning_seed


def test_the_thread_count_is_the_legacy_one(configs) -> None:
    """``_diff_regression.py:90`` hard-codes 4. The count/hurdle/damage scripts call
    ``get_available_threads()`` (_regression_GBDT.py:86); the configs pin what that
    returned in the thesis runs (saved notebook outputs, audit B5), so a 64-CPU
    cluster node does not change the thread count."""
    expected = {"diff": 4, "count": 12, "hurdle": 4, "damage": 16}
    for name, n in expected.items():
        assert configs[name].threads == n
        assert configs[name].resolved_threads == n


def test_the_panel_variant_and_binarisation_are_per_family(configs) -> None:
    count = configs["count"].data
    assert count.target == "act_drone_strike_on_ua"
    assert count.binarize == []
    assert count.panel_variant == "regressor"
    assert count.binarize_stage == "early"
    assert count.add_interactions is True

    assert configs["diff"].data.binarize == []

    # The hurdle event classifier trains on `<TARGET>_binary` (cell 3).
    hurdle = configs["hurdle"].data
    assert hurdle.binarize == ["act_drone_strike_on_ua"]
    assert hurdle.panel_variant == "regressor"

    # Q4/Q7: the damage variant binarises late and has no single target column.
    damage = configs["damage"].data
    assert damage.panel_variant == "damage"
    assert damage.binarize_stage == "late"
    assert damage.validate_target is False
    assert damage.target == ""
    assert damage.binarize == [
        "act_drone_infra_ua_health_intent",
        "act_drone_infra_ua_education_intent",
        "act_drone_infra_ua_residential_intent",
        "act_drone_infra_ua_energy_intent",
    ]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_family_shares_the_panel_filters(name: str, configs) -> None:
    data = configs[name].data
    assert data.fixed_dir == "data/fixed"
    assert data.dataset_dir == "data/dataset"
    assert data.activity_min_level == 1  # F24
    assert data.low_prevalence_ratio == 0.1  # F23/Q6


def test_the_damage_family_is_global_only(configs) -> None:
    """`damage_classifier.ipynb` has no per-activity or per-region wrapper."""
    assert configs["damage"].paradigm_names == ["global"]


def test_the_tracking_groups_cover_online_offline_and_noop() -> None:
    online = load_experiment("count")
    assert (online.tracking.backend, online.tracking.mode) == ("wandb", "online")
    offline = load_experiment("count", ["tracking=wandb_offline"])
    assert (offline.tracking.backend, offline.tracking.mode) == ("wandb", "offline")
    assert load_experiment("count", ["tracking=noop"]).tracking.backend == "noop"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_resolved_config_round_trips_and_hashes(name: str, configs) -> None:
    cfg = configs[name]
    again = ExperimentConfig(**cfg.model_dump(mode="json"))
    assert again.resolved_hash() == cfg.resolved_hash()
    assert again.common_kwargs.lags_future_covariates == (2, 7)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_hashes_differently(name: str, configs) -> None:
    others = {c.resolved_hash() for key, c in configs.items() if key != name}
    assert configs[name].resolved_hash() not in others
