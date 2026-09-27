"""Unit tests for `strikecast.pipeline.composite_stage` and `strikecast.models.hurdle`.

Everything runs on the same synthetic 4-region / 121-step panel shape the rest
of the pipeline tests use, with CHEAP deterministic heads (darts'
`SKLearnClassifierModel` over a logistic regression, `LinearRegressionModel` as
the count head) substituted for the real Self-Paced Ensemble and CatBoost. The
point is the ORCHESTRATION -- two panels, three channels, per-channel `y_true`,
the part layout, the calibration hand-off from cv to test, the component metric
files, the paradigm partition and the stage identity -- not the numbers, which
`tests/equivalence/test_hurdle_runners.py` and `test_damage_runners.py` already
pin fold by fold against the verbatim notebook loops.

The real registry specs are never built here: one SPE fit is 100 decision trees
per horizon per retrain, which would make this file minutes long for no extra
coverage. `composite_stage.get_spec` is monkeypatched with an unregistered
`ModelSpec` carrying the cheap builders, which is exactly the seam the real
specs sit in.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from darts.models import (  # noqa: E402
    LightGBMModel,
    LinearRegressionModel,
    SKLearnClassifierModel,
)

from strikecast.config.schema import (  # noqa: E402
    CalibrationConfig,
    DataConfig,
    ExperimentConfig,
    FeatureSelectionStageConfig,
    ModelEntry,
    NaiveScalesConfig,
    ParadigmConfig,
    SeedConfig,
    SeriesConfig,
    StageConfig,
    StoreConfig,
    TrackingConfig,
    TransformConfig,
)
from strikecast.data.series import build_bundle  # noqa: E402
from strikecast.models import hurdle as hurdle_wiring  # noqa: E402
from strikecast.models.spec import ModelSpec  # noqa: E402
from strikecast.pipeline import composite_stage, data_stage  # noqa: E402
from strikecast.pipeline.data_stage import CompositeData, DataArtifacts, FeatureSets  # noqa: E402
from strikecast.pipeline.run_stage import run_stage  # noqa: E402
from strikecast.store import RunKey, RunStore  # noqa: E402

N_STEPS = 121
REGIONS = ("alpha", "beta", "gamma", "delta")
ACTIVITY = {"alpha": 1, "beta": 1, "gamma": 2, "delta": 3}
COUNT_TARGET = "act_drone_strike_on_ua"
BINARY_TARGET = f"{COUNT_TARGET}_binary"
PAST = ["past_a", "past_b"]
FUTURE = ["holiday_x"]
DAMAGE_KEYS = ("act_drone_infra_ua_health_intent", "act_drone_infra_ua_energy_intent")

#: The schedule this panel produces with `start="cv_start_frac"` / horizon 7.
EXPECTED_CV_FOLDS = 6
EXPECTED_TEST_FOLDS = 19


# --------------------------------------------------------------------------- #
# panel / bundle fixtures
# --------------------------------------------------------------------------- #
def make_panel(target: str, *, binary: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=N_STEPS, freq="D")
    frames = []
    for i, region in enumerate(REGIONS):
        # sparse enough that the binary panel holds BOTH classes in every
        # horizon and every region -- a single-class horizon is a real legacy
        # case (calibration quirk Q3) but it is not what these tests are about.
        counts = rng.poisson(0.35 + 0.25 * i, N_STEPS).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "event_date": dates,
                    "Activity_Level": ACTIVITY[region],
                    target: (counts > 0).astype(float) if binary else counts,
                    "past_a": rng.normal(size=N_STEPS),
                    "past_b": np.arange(N_STEPS, dtype=float) % 7,
                    "holiday_x": (dates.dayofweek >= 5).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _artifacts(panel: pd.DataFrame, target: str, tag: str) -> DataArtifacts:
    bundle = build_bundle(panel, target, PAST, FUTURE, SeriesConfig(), ACTIVITY)
    return DataArtifacts(
        bundle=bundle,
        features=FeatureSets(list(PAST), list(FUTURE), "test", f"feat-{tag}"),
        panel_hash=f"panel-{tag}",
        series_hash=f"series-{tag}",
        activity_by_region=dict(ACTIVITY),
    )


@pytest.fixture(scope="module")
def hurdle_data() -> CompositeData:
    regressor = _artifacts(make_panel(COUNT_TARGET, binary=False), COUNT_TARGET, "reg")
    classifier = _artifacts(
        make_panel(BINARY_TARGET, binary=True, seed=1), BINARY_TARGET, "clf"
    )
    return CompositeData(
        bundle=regressor.bundle,
        features=regressor.features,
        panel_hash=regressor.panel_hash,
        series_hash=regressor.series_hash,
        activity_by_region=dict(ACTIVITY),
        heads={"regressor": regressor, "classifier": classifier},
        primary="regressor",
        family="hurdle",
    )


@pytest.fixture(scope="module")
def damage_data() -> CompositeData:
    heads = {
        key: _artifacts(
            make_panel(f"{key}_binary", binary=True, seed=i + 5), f"{key}_binary", f"d{i}"
        )
        for i, key in enumerate(DAMAGE_KEYS)
    }
    first = heads[DAMAGE_KEYS[0]]
    return CompositeData(
        bundle=first.bundle,
        features=first.features,
        panel_hash=first.panel_hash,
        series_hash=first.series_hash,
        activity_by_region=dict(ACTIVITY),
        heads=heads,
        primary=DAMAGE_KEYS[0],
        family="damage",
    )


# --------------------------------------------------------------------------- #
# cheap heads, substituted for SPE + CatBoost
# --------------------------------------------------------------------------- #
def _classifier() -> SKLearnClassifierModel:
    """`predict_likelihood_parameters=True` gives `0_p0` / `0_p1`; the loops keep
    the LAST component, which is `P(Y = 1)`."""
    return SKLearnClassifierModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=(1, 2),
        output_chunk_length=7,
    )


def _lgbm_count_head() -> LightGBMModel:
    """The real count head's shape -- a gradient booster that HAS
    `feature_importances_` -- at a size a unit test can afford."""
    return LightGBMModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=(1, 2),
        output_chunk_length=7,
        objective="poisson",
        n_estimators=15,
        num_leaves=7,
        min_child_samples=5,
        n_jobs=1,
        num_threads=1,
        deterministic=True,
        seed=0,
        verbose=-1,
        random_state=0,
    )


def _count_head() -> LinearRegressionModel:
    return LinearRegressionModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=(1, 2),
        output_chunk_length=7,
    )


class _Counting:
    def __init__(self, builder) -> None:
        self._builder = builder
        self.n = 0

    def __call__(self):
        self.n += 1
        return self._builder()


def hurdle_spec(counters: dict[str, _Counting] | None = None) -> ModelSpec:
    clf = _Counting(_classifier)
    reg = _Counting(_count_head)
    if counters is not None:
        counters.update({"classifier": clf, "regressor": reg})
    return ModelSpec(
        name="hurdle",
        family="hurdle",
        kind="composite",
        build=lambda params, ctx: (clf, reg),
        experiments=("hurdle",),
        stochastic=True,
    )


def damage_spec(counters: dict[str, _Counting] | None = None) -> ModelSpec:
    clf = _Counting(_classifier)
    if counters is not None:
        counters["classifier"] = clf
    return ModelSpec(
        name="damage",
        family="damage",
        kind="composite",
        build=lambda params, ctx: (clf,),
        experiments=("damage",),
        stochastic=True,
    )


def patch_spec(monkeypatch, spec: ModelSpec) -> None:
    monkeypatch.setattr(composite_stage, "get_spec", lambda name, experiment=None: spec)


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
def _stages() -> dict[str, StageConfig]:
    return {
        "cv": StageConfig(start="cv_start_frac"),
        # F67: the hurdle's TEST stage alone scales on train+val.
        "test": StageConfig(
            start="train_val_end", naive_scales=NaiveScalesConfig(fit_on="train_val")
        ),
    }


@pytest.fixture
def hurdle_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="hurdle",
        data=DataConfig(target=COUNT_TARGET, binarize=[COUNT_TARGET]),
        series=SeriesConfig(),
        transform=TransformConfig(kind="identity"),
        feature_selections={
            "regressor": FeatureSelectionStageConfig(selector="zipoisson_regressor"),
            "classifier": FeatureSelectionStageConfig(selector="zipoisson_classifier"),
        },
        models=[ModelEntry(name="hurdle")],
        paradigms=[
            ParadigmConfig(name="global"),
            ParadigmConfig(name="activity"),
            ParadigmConfig(name="local"),
        ],
        stages=_stages(),
        seeds=SeedConfig(tuning_seed=42, eval_seeds=[42, 1]),
        calibration=CalibrationConfig(enabled=True),
        tracking=TrackingConfig(backend="noop"),
        store=StoreConfig(root=str(tmp_path / "runs")),
    )


@pytest.fixture
def damage_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="damage",
        data=DataConfig(target="", binarize=list(DAMAGE_KEYS), panel_variant="damage"),
        series=SeriesConfig(),
        transform=TransformConfig(kind="identity"),
        models=[ModelEntry(name="damage")],
        paradigms=[ParadigmConfig(name="global")],
        stages=_stages(),
        seeds=SeedConfig(tuning_seed=42, eval_seeds=[42]),
        calibration=CalibrationConfig(enabled=True),
        tracking=TrackingConfig(backend="noop"),
        store=StoreConfig(root=str(tmp_path / "runs")),
    )


@pytest.fixture
def store(hurdle_cfg: ExperimentConfig) -> RunStore:
    return RunStore(hurdle_cfg.store.root)


def _key(cfg: ExperimentConfig, paradigm: str = "global", seed: int = 42) -> RunKey:
    return RunKey(cfg.name, cfg.model_names[0], paradigm, seed)


def _fit_calibrators(cfg: ExperimentConfig, data: CompositeData, store: RunStore) -> None:
    """A (fold-limited) tuning-seed CV stage, so a test stage finds its calibrators."""
    composite_stage.run_composite_stage(
        cfg, cfg.model_names[0], "global", int(cfg.seeds.tuning_seed), "cv", data,
        store=store, max_folds=3,
    )


# --------------------------------------------------------------------------- #
# data stage: two panels, two selections, distinct hashes
# --------------------------------------------------------------------------- #
def test_hurdle_head_configs_build_the_two_legacy_panels() -> None:
    """`final_hurdle.ipynb` cell 3: an unbinarised count panel and a binarised
    event panel, each with its own selection (F27)."""
    cfg = ExperimentConfig(
        name="hurdle",
        data=DataConfig(target=COUNT_TARGET, binarize=[COUNT_TARGET]),
        feature_selections={
            "regressor": FeatureSelectionStageConfig(selector="zipoisson_regressor"),
            "classifier": FeatureSelectionStageConfig(selector="zipoisson_classifier"),
        },
        models=[ModelEntry(name="hurdle")],
    )
    assert data_stage.is_composite_family(cfg)
    heads = data_stage.head_configs(cfg)

    assert list(heads) == ["regressor", "classifier"]
    assert heads["regressor"].data.target == COUNT_TARGET
    assert heads["regressor"].data.binarize == []
    assert heads["classifier"].data.target == BINARY_TARGET
    assert heads["classifier"].data.binarize == [COUNT_TARGET]
    assert heads["regressor"].feature_selection.selector == "zipoisson_regressor"
    assert heads["classifier"].feature_selection.selector == "zipoisson_classifier"


def test_damage_head_configs_build_one_panel_per_key() -> None:
    cfg = ExperimentConfig(
        name="damage",
        data=DataConfig(target="", binarize=list(DAMAGE_KEYS), panel_variant="damage"),
        models=[ModelEntry(name="damage")],
    )
    heads = data_stage.head_configs(cfg)

    assert list(heads) == list(DAMAGE_KEYS)
    for key, head in heads.items():
        assert head.data.binarize == [key]
        assert head.data.target == f"{key}_binary"
        assert head.data.panel_variant == "damage"


def test_every_head_gets_its_own_shared_cache_entry(hurdle_cfg, store) -> None:
    """Distinct `data`/`feature_selection` slices mean distinct content hashes,
    so two heads can never read each other's panel."""
    heads = data_stage.head_configs(hurdle_cfg)
    panel_hashes = {
        name: data_stage._panel_hash(head) for name, head in heads.items()
    }
    feature_hashes = {
        name: data_stage._features_hash(
            head, panel_hashes[name], data_stage._series_hash(head, panel_hashes[name])
        )
        for name, head in heads.items()
    }
    assert len(set(panel_hashes.values())) == len(heads)
    assert len(set(feature_hashes.values())) == len(heads)

    paths = {
        store.shared_path(hurdle_cfg.name, "panel", digest)
        for digest in panel_hashes.values()
    }
    assert len(paths) == len(heads)
    assert all(p.parent.name == "shared" for p in paths)


def test_composite_data_is_a_data_artifacts_over_the_primary_head(hurdle_data) -> None:
    assert isinstance(hurdle_data, DataArtifacts)
    assert hurdle_data.bundle is hurdle_data.head("regressor").bundle
    assert hurdle_data.region_names == sorted(REGIONS)
    # every head contributes to the stage identity
    assert set(hurdle_data.upstream) >= set(hurdle_data.head("classifier").upstream)
    assert set(hurdle_data.upstream) >= set(hurdle_data.head("regressor").upstream)
    with pytest.raises(KeyError, match="no head 'nope'"):
        hurdle_data.head("nope")


def test_only_the_hurdle_count_selection_is_weighted(hurdle_cfg, hurdle_data) -> None:
    """F27 / Q7: `final_hurdle.ipynb` cell 10 is the only selection in the code
    base that passes `sample_weight`."""
    heads = data_stage.head_configs(hurdle_cfg)
    bundle = hurdle_data.head("regressor").bundle

    weights = data_stage._selection_sample_weight(heads["regressor"], bundle)
    assert weights is not None
    assert len(weights) == len(REGIONS)
    assert all(set(np.unique(w.values())) <= {0.0, 1.0} for w in weights)
    assert all(
        len(w) == len(t) for w, t in zip(weights, bundle.target_train, strict=True)
    )
    assert data_stage._selection_sample_weight(heads["classifier"], bundle) is None


# --------------------------------------------------------------------------- #
# series wiring
# --------------------------------------------------------------------------- #
def test_hurdle_series_wires_the_two_stages_as_the_notebook_does(hurdle_data) -> None:
    clf = hurdle_data.head("classifier")
    reg = hurdle_data.head("regressor")

    cv = hurdle_wiring.hurdle_series(clf.bundle, reg.bundle, "cv")
    assert cv.binary_targets[0].values().tolist() == clf.bundle.target_cv_view[0].values().tolist()
    assert cv.count_targets[0].values().tolist() == reg.bundle.target_cv_view[0].values().tolist()
    # cell 10: the CV weights are FULL length, taken from the un-encoded list
    assert len(cv.weights[0]) == len(reg.bundle.target_full[0])
    assert cv.level_targets is cv.count_targets
    # per-channel actuals: prob against the binary list, count/hurdle against counts
    assert cv.actuals["prob"] is cv.binary_targets
    assert cv.actuals["count"] is cv.count_targets
    assert cv.actuals["hurdle"] is cv.count_targets

    test = hurdle_wiring.hurdle_series(clf.bundle, reg.bundle, "test")
    assert len(test.count_targets[0]) == len(reg.bundle.target_full[0])
    assert len(test.weights[0]) == len(test.count_targets[0])
    expected = (reg.bundle.target_full[0].values().ravel() > 0).astype(float)
    np.testing.assert_array_equal(test.weights[0].values().ravel(), expected)


def test_hurdle_series_rejects_bundles_with_different_regions(hurdle_data) -> None:
    from dataclasses import replace  # noqa: PLC0415

    clf = hurdle_data.head("classifier")
    shuffled = replace(
        clf.bundle, region_names=list(reversed(clf.bundle.region_names))
    )
    with pytest.raises(ValueError, match="disagree on the region order"):
        hurdle_wiring.hurdle_series(
            shuffled, hurdle_data.head("regressor").bundle, "cv"
        )


def test_min_positive_samples_only_fires_for_the_local_paradigm() -> None:
    """F62: `MIN_POSITIVE_SAMPLES = 50` exists in cell 34 alone."""
    assert hurdle_wiring.min_positive_samples_for("local") == 50
    assert hurdle_wiring.min_positive_samples_for("global") is None
    assert hurdle_wiring.min_positive_samples_for("activity") is None


def test_damage_series_takes_the_schedule_from_the_first_key(damage_data) -> None:
    """F68: both legacy loops schedule on the first key and apply it to all."""
    series = hurdle_wiring.damage_series(damage_data.bundles, "cv")
    assert series.keys == list(DAMAGE_KEYS)
    assert series.level_targets is series.targets_by_key[DAMAGE_KEYS[0]]
    assert set(series.actuals) == set(DAMAGE_KEYS)
    assert hurdle_wiring.damage_label(DAMAGE_KEYS[0]) == "health"


def test_hurdle_forecaster_slices_every_list_by_the_group(hurdle_data) -> None:
    series = hurdle_wiring.hurdle_series(
        hurdle_data.head("classifier").bundle, hurdle_data.head("regressor").bundle, "cv"
    )
    forecaster = hurdle_wiring.make_hurdle_forecaster(
        (_classifier, _count_head), series, indices=[2], min_positive_samples=50
    )
    assert len(forecaster.binary_targets) == 1
    assert len(forecaster.count_targets) == 1
    assert len(forecaster.clf_past) == 1
    assert forecaster.min_positive_samples == 50
    assert forecaster.channels == ("prob", "count", "hurdle")


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def test_hurdle_cv_writes_three_channels_with_per_channel_actuals(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    counters: dict[str, _Counting] = {}
    patch_spec(monkeypatch, hurdle_spec(counters))

    outcome = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store
    )
    assert not outcome.skipped
    assert outcome.n_folds == EXPECTED_CV_FOLDS
    assert counters["classifier"].n == counters["regressor"].n == 1  # one retrain

    preds = store.load_predictions(_key(hurdle_cfg), "cv", legacy_order=True)
    assert set(preds.channels) == {"prob", "count", "hurdle"}

    prob = preds.for_channel("prob").frame.reset_index(drop=True)
    count = preds.for_channel("count").frame.reset_index(drop=True)
    hurdle = preds.for_channel("hurdle").frame.reset_index(drop=True)

    # the three channels describe exactly the same rows...
    keys = ["region", "fold", "horizon", "date"]
    assert prob[keys].equals(count[keys])
    assert hurdle[keys].equals(count[keys])
    # ...but `prob` is scored against the BINARY actuals and the others against
    # the counts (cell 16), which is the whole reason for a per-channel hook.
    assert set(np.unique(prob["y_true"])) <= {0.0, 1.0}
    assert count["y_true"].max() > 1.0
    assert count["y_true"].equals(hurdle["y_true"])
    # hurdle = prob * count, row by row
    np.testing.assert_allclose(
        hurdle["y_pred"].to_numpy(),
        prob["y_pred"].to_numpy() * count["y_pred"].to_numpy(),
        rtol=1e-12,
    )


def test_hurdle_cv_writes_the_five_notebook_components(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    """`final_hurdle.ipynb` cells 48-49: five components x four views."""
    patch_spec(monkeypatch, hurdle_spec())
    composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store
    )

    directory = store.metrics_dir(_key(hurdle_cfg), "cv")
    names = {p.name for p in directory.iterdir()}

    # the primary component ALSO keeps the plain names, so a leaderboard finds
    # this family where it finds every other one
    assert {"global.json", "per_region.csv", "per_horizon.csv", "per_region_horizon.csv"} <= names
    for component in (
        "classifier_raw",
        "classifier_cal",
        "regressor",
        "hurdle_raw",
        "hurdle_cal",
    ):
        assert f"{component}@global.json" in names
        for view in ("per_region", "per_horizon", "per_region_horizon"):
            assert f"{component}@{view}.csv" in names
    # `hurdle_cal` is the primary (audit B8/D6: the thesis reports the
    # calibrated hurdle): the two spellings hold the same row
    assert (
        json.loads((directory / "hurdle_cal@global.json").read_text())
        == json.loads((directory / "global.json").read_text())
    )
    assert (
        json.loads((directory / "hurdle_raw@global.json").read_text())
        != json.loads((directory / "global.json").read_text())
    )
    # no activity views: neither legacy aggregator had that parameter
    assert not any("per_activity" in n for n in names)

    # the classifier components carry classification metrics, the hurdle ones
    # the scaled count metrics
    clf = json.loads((directory / "classifier_raw@global.json").read_text())
    assert {"F1", "ROC_AUC", "PR_AUC", "Brier"} <= set(clf)
    primary = json.loads((directory / "global.json").read_text())
    assert {"MASE_mean", "RMSSE_mean", "R2"} <= set(primary)

    # F65: the count head is scored on positive-event days only
    regressor = json.loads((directory / "regressor@global.json").read_text())
    assert regressor["n"] < primary["n"]


def test_calibration_is_fit_on_cv_and_applied_out_of_sample_on_test(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    """F69: the CV view is calibrated in sample by calibrators fit on those same
    rows (cell 22); the test view applies the CV-fitted ones (cell 27)."""
    patch_spec(monkeypatch, hurdle_spec())
    composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store
    )

    key = _key(hurdle_cfg)
    path = store.artifacts_dir(key) / composite_stage.CALIBRATORS_ARTIFACT
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert list(payload) == ["prob"]
    assert payload["prob"]["method"] == "sigmoid"
    assert set(payload["prob"]["calibrators"]) <= {str(h) for h in range(1, 8)}

    # the diagnostic of cell 21 is recorded rather than printed
    diagnostic = json.loads(
        (store.artifacts_dir(key) / composite_stage.CALIBRATION_DIAGNOSTIC_ARTIFACT).read_text()
    )
    assert {"raw", "oof_calibrated"} == set(diagnostic["prob"])

    composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "test", hurdle_data, store=store
    )
    metrics = store.metrics_dir(key, "test")
    raw = pd.read_csv(metrics / "classifier_raw@per_horizon.csv")
    cal = pd.read_csv(metrics / "classifier_cal@per_horizon.csv")
    # calibration is a monotone map, so ranking metrics are unchanged while the
    # probability-scale ones move -- the notebook's "Brier x -> y" print
    np.testing.assert_allclose(raw["ROC_AUC"], cal["ROC_AUC"], rtol=1e-9)
    assert not np.allclose(raw["Brier"], cal["Brier"])


def test_a_test_stage_without_calibrators_is_a_loud_error(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    """Audit C18: fitting calibrators on the test rows is the one thing the
    notebooks are careful not to do, and silently keeping the raw probabilities
    would report an uncalibrated model under the calibrated name -- so a missing
    CV stage is an error, raised before any fold runs."""
    patch_spec(monkeypatch, hurdle_spec())
    with pytest.raises(composite_stage.CalibratorsMissing, match="seed=42"):
        composite_stage.run_composite_stage(
            hurdle_cfg, "hurdle", "global", 1, "test", hurdle_data, store=store
        )
    # nothing was started, so nothing is left half-written
    assert not store.part_paths(_key(hurdle_cfg, seed=1), "test")


def test_a_test_stage_at_any_seed_reads_the_tuning_seed_calibrators(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    """Audit C18: CV runs only at the tuning seed, so the seed-1 test stage must
    apply the calibrators of `seed=42`, not look for its own."""
    patch_spec(monkeypatch, hurdle_spec())
    composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store, max_folds=3
    )
    assert composite_stage.calibration_key(hurdle_cfg, _key(hurdle_cfg, seed=1)) == _key(
        hurdle_cfg, seed=42
    )
    outcome = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 1, "test", hurdle_data, store=store
    )
    assert not outcome.skipped
    assert not (store.artifacts_dir(_key(hurdle_cfg, seed=1)) / "calibrators.json").exists()

    metrics = store.metrics_dir(_key(hurdle_cfg, seed=1), "test")
    raw = pd.read_csv(metrics / "classifier_raw@per_horizon.csv")
    cal = pd.read_csv(metrics / "classifier_cal@per_horizon.csv")
    assert not np.allclose(raw["Brier"], cal["Brier"]), "the probabilities were calibrated"
    assert (
        json.loads((metrics / "hurdle_cal@global.json").read_text())
        == json.loads((metrics / "global.json").read_text())
    )

    # refitting the calibrators moves the test stage's identity
    again = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 1, "test", hurdle_data, store=store
    )
    assert again.skipped
    composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store, max_folds=4
    )
    moved = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 1, "test", hurdle_data, store=store
    )
    assert not moved.skipped and moved.stage_hash != outcome.stage_hash


def test_a_calibrator_file_without_the_channel_is_a_loud_error(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    patch_spec(monkeypatch, hurdle_spec())
    store.write_artifact(_key(hurdle_cfg), composite_stage.CALIBRATORS_ARTIFACT, {})
    with pytest.raises(composite_stage.CalibratorsMissing, match="no channel 'prob'"):
        composite_stage.run_composite_stage(
            hurdle_cfg, "hurdle", "global", 42, "test", hurdle_data, store=store, max_folds=1
        )


def test_stage_identity_covers_every_head_and_is_skipped_on_rerun(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    from dataclasses import replace  # noqa: PLC0415

    patch_spec(monkeypatch, hurdle_spec())
    first = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store
    )
    again = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store
    )
    assert again.skipped
    assert again.stage_hash == first.stage_hash

    forced = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store, force=True
    )
    assert not forced.skipped

    # moving ONLY the classifier head's selection moves the stage identity
    moved_head = replace(
        hurdle_data.head("classifier"),
        features=FeatureSets(list(PAST), list(FUTURE), "test", "feat-other"),
    )
    moved = replace(
        hurdle_data, heads={**hurdle_data.heads, "classifier": moved_head}
    )
    outcome = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", moved, store=store
    )
    assert not outcome.skipped
    assert outcome.stage_hash != first.stage_hash


@pytest.mark.parametrize("paradigm", ["activity", "local"])
def test_grouped_paradigms_cover_every_region_once(
    monkeypatch, hurdle_cfg, hurdle_data, store, paradigm
) -> None:
    """The legacy `*_per_activity` / `*_per_region` wrappers restore the original
    region order; the local one also arms the F62 dummy branch."""
    counters: dict[str, _Counting] = {}
    patch_spec(monkeypatch, hurdle_spec(counters))

    outcome = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", paradigm, 42, "cv", hurdle_data, store=store
    )
    assert outcome.n_folds == EXPECTED_CV_FOLDS

    preds = store.load_predictions(_key(hurdle_cfg, paradigm), "cv", legacy_order=True)
    frame = preds.frame
    assert set(frame["region"]) == set(REGIONS)
    assert set(frame["channel"]) == {"prob", "count", "hurdle"}
    # one classifier per group per retrain; 3 activity tiers / 4 regions
    expected_groups = 3 if paradigm == "activity" else len(REGIONS)
    assert counters["classifier"].n == expected_groups


def test_local_paradigm_can_fall_back_to_the_constant_count_head(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    """F62: below 50 positive training days the count head becomes a constant,
    built on the CLASSIFIER's time index."""
    patch_spec(monkeypatch, hurdle_spec())
    series = hurdle_wiring.hurdle_series(
        hurdle_data.head("classifier").bundle, hurdle_data.head("regressor").bundle, "cv"
    )
    forecaster = hurdle_wiring.make_hurdle_forecaster(
        (_classifier, _count_head), series, indices=[0], min_positive_samples=10_000
    )
    cutoff = series.count_targets[0].time_index[60]
    forecaster.fit([series.count_targets[0].drop_after(cutoff)], cutoff=cutoff)
    assert forecaster._use_dummy
    out = forecaster.predict(7, [series.count_targets[0].drop_after(cutoff)], cutoff=cutoff)
    assert len(set(out["count"][0].values().ravel())) == 1


def test_parts_are_written_once_per_retrain_window(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    patch_spec(monkeypatch, hurdle_spec())
    _fit_calibrators(hurdle_cfg, hurdle_data, store)
    composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "test", hurdle_data, store=store
    )
    parts = store.part_paths(_key(hurdle_cfg), "test")
    assert len(parts) == 3  # 19 folds, retrain every 7

    # fold numbering is globally contiguous across parts
    preds = store.load_predictions(_key(hurdle_cfg), "test")
    assert sorted(preds.frame["fold"].unique().tolist()) == list(range(EXPECTED_TEST_FOLDS))


def test_run_stage_dispatches_a_composite_to_composite_stage(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    """The one wiring point in `run_stage.py`."""
    spec = hurdle_spec()
    patch_spec(monkeypatch, spec)
    monkeypatch.setattr(
        "strikecast.pipeline.run_stage.get_spec", lambda name, experiment=None: spec
    )
    outcome = run_stage(
        hurdle_cfg, "hurdle", "global", 42, "cv", hurdle_data, store=store
    )
    assert outcome.n_folds == EXPECTED_CV_FOLDS
    assert (store.metrics_dir(_key(hurdle_cfg), "cv") / "hurdle_cal@global.json").is_file()


def test_max_folds_truncates_the_schedule_and_changes_the_identity(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    patch_spec(monkeypatch, hurdle_spec())
    _fit_calibrators(hurdle_cfg, hurdle_data, store)
    limited = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "test", hurdle_data, store=store, max_folds=3
    )
    assert limited.n_folds == 3
    preds = store.load_predictions(_key(hurdle_cfg), "test")
    assert sorted(preds.frame["fold"].unique().tolist()) == [0, 1, 2]

    full = composite_stage.run_composite_stage(
        hurdle_cfg, "hurdle", "global", 42, "test", hurdle_data, store=store
    )
    assert not full.skipped
    assert full.stage_hash != limited.stage_hash


def test_native_importances_follow_the_legacy_contract(hurdle_data) -> None:
    """`src/evaluation_tools.py::feature_importances_per_horizon`: one column per
    horizon estimator, `None` when the model exposes no importances at all."""
    series = hurdle_wiring.hurdle_series(
        hurdle_data.head("classifier").bundle, hurdle_data.head("regressor").bundle, "test"
    )
    cutoff = series.count_targets[0].time_index[90]

    linear = _count_head()
    linear.fit(
        series=[ts.drop_after(cutoff) for ts in series.count_targets],
        past_covariates=series.reg_past,
        future_covariates=series.reg_future,
    )
    # a plain LinearRegression has neither `feature_importances_` nor
    # `get_feature_importance()`, so the legacy helper returns None
    assert composite_stage._importance_frame(linear, linear.lagged_feature_names) is None

    lgbm = _lgbm_count_head()
    lgbm.fit(
        series=[ts.drop_after(cutoff) for ts in series.count_targets],
        past_covariates=series.reg_past,
        future_covariates=series.reg_future,
        sample_weight=[w.drop_after(cutoff) for w in series.weights],
    )
    frame = composite_stage._importance_frame(lgbm, lgbm.lagged_feature_names)
    assert frame is not None
    assert list(frame.columns) == ["Feature", *[f"h{h}_importance" for h in range(1, 8)]]
    assert len(frame) == len(lgbm.lagged_feature_names)

    ranked = composite_stage._with_mean_importance(frame)
    assert ranked["mean_importance"].is_monotonic_decreasing


def test_hurdle_importances_come_off_one_fit_on_train_val(hurdle_data) -> None:
    """Cells E-F: ONE clean fit up to the test start, then native importances.

    The count head is a Poisson LightGBM here, which is the shape the real
    CatBoost head has; the classifier stays the cheap logistic one, whose
    estimators expose no importances -- exactly the `None` branch the legacy
    helper has, so only the regressor frame is produced.
    """
    series = hurdle_wiring.hurdle_series(
        hurdle_data.head("classifier").bundle, hurdle_data.head("regressor").bundle, "test"
    )
    train_val_end = hurdle_data.bundle.train_val_end
    frames = composite_stage._hurdle_importances(
        (_classifier, _lgbm_count_head), series, train_val_end
    )
    assert "feature_importance_regressor_per_horizon" in frames

    frame = frames["feature_importance_regressor_per_horizon"]
    assert frame["mean_importance"].is_monotonic_decreasing
    assert frame["Feature"].is_unique

    # the fit really stopped at the test start: refitting by hand on the same
    # slice reproduces the importances, refitting on everything does not
    cutoff = series.binary_targets[0].time_index[
        int(train_val_end * len(series.binary_targets[0]))
    ]
    manual = _lgbm_count_head()
    manual.fit(
        series=[ts.drop_after(cutoff) for ts in series.count_targets],
        past_covariates=series.reg_past,
        future_covariates=series.reg_future,
        sample_weight=[w.drop_after(cutoff) for w in series.weights],
    )
    expected = composite_stage._with_mean_importance(
        composite_stage._importance_frame(manual, manual.lagged_feature_names)
    )
    pd.testing.assert_frame_equal(frame, expected)


# --------------------------------------------------------------------------- #
# damage
# --------------------------------------------------------------------------- #
def test_damage_writes_one_channel_and_two_components_per_key(
    monkeypatch, damage_cfg, damage_data
) -> None:
    patch_spec(monkeypatch, damage_spec())
    store = RunStore(damage_cfg.store.root)

    composite_stage.run_composite_stage(
        damage_cfg, "damage", "global", 42, "cv", damage_data, store=store
    )
    key = _key(damage_cfg)
    preds = store.load_predictions(key, "cv", legacy_order=True)
    assert set(preds.channels) == set(DAMAGE_KEYS)
    # every channel is a probability scored against its OWN binary target
    for channel in DAMAGE_KEYS:
        frame = preds.for_channel(channel).frame
        assert set(np.unique(frame["y_true"])) <= {0.0, 1.0}
        assert frame["y_pred"].between(0.0, 1.0).all()

    names = {p.name for p in store.metrics_dir(key, "cv").iterdir()}
    assert "global.json" in names  # the first key's raw component is primary
    for label in ("health", "energy"):
        for suffix in ("raw", "cal"):
            if (label, suffix) == ("health", "raw"):
                continue  # the primary
            assert f"{label}_{suffix}@global.json" in names


def test_damage_test_stage_reads_every_key_from_the_tuning_seed_calibrators(
    monkeypatch, damage_cfg, damage_data
) -> None:
    """Audit C18 for the damage family: one calibrator set per key, read from
    `seed=<tuning_seed>`; missing -> loud error."""
    patch_spec(monkeypatch, damage_spec())
    store = RunStore(damage_cfg.store.root)
    with pytest.raises(composite_stage.CalibratorsMissing):
        composite_stage.run_composite_stage(
            damage_cfg, "damage", "global", 3, "test", damage_data, store=store, max_folds=2
        )
    composite_stage.run_composite_stage(
        damage_cfg, "damage", "global", 42, "cv", damage_data, store=store, max_folds=3
    )
    payload = json.loads(
        (store.artifacts_dir(_key(damage_cfg)) / composite_stage.CALIBRATORS_ARTIFACT).read_text()
    )
    assert set(payload) == set(DAMAGE_KEYS)
    outcome = composite_stage.run_composite_stage(
        damage_cfg, "damage", "global", 3, "test", damage_data, store=store, max_folds=2
    )
    assert not outcome.skipped


def test_damage_refuses_a_paradigm_the_notebook_never_ran(
    monkeypatch, damage_cfg, damage_data
) -> None:
    patch_spec(monkeypatch, damage_spec())
    with pytest.raises(ValueError, match="global only"):
        composite_stage.run_composite_stage(
            damage_cfg,
            "damage",
            "activity",
            42,
            "cv",
            damage_data,
            store=RunStore(damage_cfg.store.root),
        )


def test_composite_stage_refuses_plain_data_artifacts(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    patch_spec(monkeypatch, hurdle_spec())
    plain = DataArtifacts(
        bundle=hurdle_data.bundle,
        features=hurdle_data.features,
        panel_hash="p",
        series_hash="s",
    )
    with pytest.raises(TypeError, match="CompositeData"):
        composite_stage.run_composite_stage(
            hurdle_cfg, "hurdle", "global", 42, "cv", plain, store=store
        )


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def test_the_sweep_keeps_cv_single_seed_and_sweeps_the_test_stage(
    monkeypatch, hurdle_cfg, hurdle_data, store
) -> None:
    patch_spec(monkeypatch, hurdle_spec())
    outcomes = composite_stage.run_composite_experiment(
        hurdle_cfg,
        data=hurdle_data,
        paradigms=["global"],
        store=store,
        max_folds=2,
    )
    by_stage: dict[str, list[Any]] = {}
    for outcome in outcomes:
        by_stage.setdefault(outcome.stage, []).append(outcome.run_key.seed)

    assert by_stage["cv"] == [42]                     # §5.4: CV is single-seed
    assert sorted(by_stage["test"]) == [1, 42]        # stochastic -> swept
    assert len({o.stage_hash for o in outcomes if o.stage == "test"}) == 2
