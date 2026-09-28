"""The feature-selection cache guard, ``strikecast featsel`` and the selectors.

Audit 2026-09-26 A12/A13/A14, decision D2:

* a cached selection records the window/selector config it was made under and
  is REFUSED under another one; the converted thesis sets (no provenance) are
  valid only under ``series.window.expdecay = "legacy_alpha"``;
* ``feature_selection.require_cached`` turns the selection into a separate,
  once-per-family/head step (``select_features`` = ``strikecast featsel``) and
  every other caller fails loudly when it is missing;
* the deterministic selection (CPU, ``deterministic=True``, fixed threads,
  ``PYTHONHASHSEED=0``) is repeatable, including the hurdle and damage
  selectors, which had no test at all (A, "What the tests do not prove");
* the panel hash covers the input files' content (A14).

Everything runs on a small synthetic panel; nothing reads ``golden/`` except
the one test that checks the real thesis sets are refused under ``leaky``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")
pytest.importorskip("lightgbm")

from strikecast.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    FeatureSelectionStageConfig,
    SeriesConfig,
    StoreConfig,
    TrackingConfig,
    WindowTransformConfig,
)
from strikecast.data.feature_selection import (  # noqa: E402
    select_top_k,
    zipoisson_classifier_config,
    zipoisson_regressor_config,
)
from strikecast.data.series import build_bundle, positive_only_weights  # noqa: E402
from strikecast.pipeline import data_stage  # noqa: E402
from strikecast.pipeline.data_stage import (  # noqa: E402
    FeatureCacheMismatch,
    FeatureSelectionMissing,
)
from strikecast.store import RunStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
N_STEPS = 140
REGIONS = ("alpha", "beta", "gamma", "delta")
ACTIVITY = {"alpha": 1, "beta": 1, "gamma": 2, "delta": 3}
TARGET = "act_drone_strike_on_ua"
BINARY = f"{TARGET}_binary"
PAST = ["past_a", "past_b", "past_c", "past_d"]
FUTURE = ["fut_x", "fut_y"]


def make_panel(target: str = TARGET, *, binary: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=N_STEPS, freq="D")
    frames = []
    for i, region in enumerate(REGIONS):
        driver = rng.poisson(1.0 + i, N_STEPS).astype(float)
        counts = rng.poisson(0.5 + 0.3 * driver).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "event_date": dates,
                    "Activity_Level": ACTIVITY[region],
                    target: (counts > 0).astype(float) if binary else counts,
                    "past_a": np.roll(driver, 1),
                    "past_b": rng.normal(size=N_STEPS),
                    "past_c": np.arange(N_STEPS, dtype=float) % 7,
                    "past_d": rng.poisson(3.0, N_STEPS).astype(float),
                    "fut_x": (dates.dayofweek >= 5).astype(float),
                    "fut_y": rng.normal(size=N_STEPS),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def window(mode: str) -> SeriesConfig:
    return SeriesConfig(window=WindowTransformConfig(expdecay=mode))  # type: ignore[arg-type]


def fs(selector: str = "countreg", **overrides) -> FeatureSelectionStageConfig:
    base = dict(
        selector=selector, device="cpu", num_threads=2, top_k=12, cache=True,
        deterministic=True, require_cached=True,
    )
    base.update(overrides)
    return FeatureSelectionStageConfig(**base)


def make_cfg(tmp_path: Path, mode: str = "leaky", **fs_overrides) -> ExperimentConfig:
    return ExperimentConfig(
        name="count",
        data=DataConfig(target=TARGET),
        series=window(mode),
        feature_selection=fs(**fs_overrides),
        tracking=TrackingConfig(backend="noop"),
        store=StoreConfig(root=str(tmp_path / "runs")),
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture(scope="module")
def bundles(panel):
    return {
        mode: build_bundle(panel, TARGET, PAST, FUTURE, window(mode), ACTIVITY)
        for mode in ("leaky", "legacy_alpha")
    }


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_deterministic_needs_cpu_and_a_fixed_thread_count() -> None:
    with pytest.raises(ValueError, match="device=cpu"):
        fs(device="gpu")
    with pytest.raises(ValueError, match="num_threads"):
        fs(num_threads=None)
    with pytest.raises(ValueError, match="cache=true"):
        fs(cache=False)


@pytest.mark.parametrize(
    "selector",
    [
        "countreg", "diffreg", "zipoisson_regressor", "zipoisson_classifier",
        # the figure selectors (plan 2026-09-28) share the same switch
        "diff_l2", "count_poisson", "count_tweedie", "hurdle_binary", "hurdle_tweedie_pos",
    ],
)
def test_deterministic_build_pins_lightgbm(selector: str) -> None:
    kwargs = fs(selector, num_threads=3).build().model_kwargs
    assert kwargs["device_type"] == "cpu"
    assert kwargs["num_threads"] == 3
    assert kwargs["deterministic"] is True
    assert kwargs["force_col_wise"] is True
    assert kwargs["random_state"] == 42
    # the legacy build is untouched when the switch is off
    legacy = FeatureSelectionStageConfig(selector=selector, device="cpu").build().model_kwargs
    assert "deterministic" not in legacy


# --------------------------------------------------------------------------- #
# require_cached / featsel
# --------------------------------------------------------------------------- #
def test_require_cached_refuses_to_select(tmp_path, bundles) -> None:
    cfg = make_cfg(tmp_path)
    with pytest.raises(FeatureSelectionMissing, match="strikecast featsel experiment=count"):
        data_stage.build_or_load_features(cfg, RunStore(cfg.store.root), bundles["leaky"])


def test_featsel_selects_once_and_later_jobs_load_it(tmp_path, bundles) -> None:
    cfg = make_cfg(tmp_path)
    store = RunStore(cfg.store.root)
    made = data_stage.build_or_load_features(cfg, store, bundles["leaky"], compute=True)
    assert made.source == "computed"
    assert made.path is not None and made.path.is_file()
    assert made.past_keep

    provenance = json.loads(made.path.read_text(encoding="utf-8"))["provenance"]
    assert provenance["expdecay"] == "leaky"
    assert provenance["window"] == cfg.series.window.model_dump(mode="json")
    assert provenance["feature_selection"]["deterministic"] is True
    assert provenance["pythonhashseed"] == "0"
    assert provenance["target"] == TARGET

    loaded = data_stage.build_or_load_features(cfg, store, bundles["leaky"])
    assert loaded.source == "store"
    assert (loaded.past_keep, loaded.future_keep) == (made.past_keep, made.future_keep)
    assert loaded.hash == made.hash


def test_leaky_features_are_named_leaky7(bundles) -> None:
    leaky = set(bundles["leaky"].past_covs[0].components)
    legacy = set(bundles["legacy_alpha"].past_covs[0].components)
    assert {f"ewm_leaky7_{c}" for c in PAST} <= leaky
    assert not any("expdecay7" in c for c in leaky)
    assert {f"ewm_expdecay7_{c}" for c in PAST} <= legacy


def test_deterministic_selection_is_repeatable(tmp_path, bundles) -> None:
    runs = []
    for i in range(2):
        cfg = make_cfg(tmp_path / f"r{i}")
        sets = data_stage.build_or_load_features(
            cfg, RunStore(cfg.store.root), bundles["leaky"], compute=True
        )
        payload = json.loads(sets.path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        runs.append((payload["top_features"], sets.past_keep, sets.future_keep))
    assert runs[0] == runs[1]


def test_deterministic_selection_requires_hash_seed_zero(tmp_path, bundles, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    cfg = make_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        data_stage.build_or_load_features(
            cfg, RunStore(cfg.store.root), bundles["leaky"], compute=True
        )
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        data_stage.select_features(cfg, RunStore(cfg.store.root))


# --------------------------------------------------------------------------- #
# the A13 guard
# --------------------------------------------------------------------------- #
def _thesis_set(tmp_path: Path, name: str = "countreg") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "target_components": [TARGET],
                "past_covariate_components": ["ewm_expdecay7_past_a", "past_b"],
                "future_covariate_components": ["fut_x"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_thesis_set_is_refused_under_the_leaky_filter(tmp_path, bundles) -> None:
    path = _thesis_set(tmp_path)
    cfg = make_cfg(tmp_path, "leaky", cache_path=str(path))
    with pytest.raises(FeatureCacheMismatch, match="legacy_alpha"):
        data_stage.build_or_load_features(cfg, RunStore(cfg.store.root), bundles["leaky"])

    legacy = make_cfg(tmp_path, "legacy_alpha", cache_path=str(path))
    sets = data_stage.build_or_load_features(
        legacy, RunStore(legacy.store.root), bundles["legacy_alpha"]
    )
    assert sets.source == "cache_path"
    assert sets.past_keep == ["ewm_expdecay7_past_a", "past_b"]


def test_a_thesis_set_is_refused_for_another_selector_or_target(tmp_path, bundles) -> None:
    path = _thesis_set(tmp_path, "countreg")
    other = make_cfg(tmp_path, "legacy_alpha", selector="diffreg", cache_path=str(path))
    with pytest.raises(FeatureCacheMismatch, match="countreg"):
        data_stage.build_or_load_features(other, None, bundles["legacy_alpha"])
    wrong_target = make_cfg(tmp_path, "legacy_alpha", cache_path=str(path)).model_copy(
        update={"data": DataConfig(target="something_else")}
    )
    with pytest.raises(FeatureCacheMismatch, match="target"):
        data_stage.build_or_load_features(wrong_target, None, bundles["legacy_alpha"])


def test_a_selection_made_on_other_features_is_refused(tmp_path, bundles) -> None:
    """The exact A13 failure: a cached set from one window config reused on another."""
    leaky = make_cfg(tmp_path)
    made = data_stage.build_or_load_features(
        leaky, RunStore(leaky.store.root), bundles["leaky"], compute=True
    )
    reused = make_cfg(tmp_path, "legacy_alpha", cache_path=str(made.path))
    with pytest.raises(FeatureCacheMismatch, match="window"):
        data_stage.build_or_load_features(reused, None, bundles["legacy_alpha"])
    # ... and a different selector identity (thread count) is refused too
    other_threads = make_cfg(tmp_path, cache_path=str(made.path), num_threads=5)
    with pytest.raises(FeatureCacheMismatch, match="feature_selection"):
        data_stage.build_or_load_features(other_threads, None, bundles["leaky"])
    # the same config re-reads it fine through cache_path
    same = make_cfg(tmp_path, cache_path=str(made.path))
    assert data_stage.build_or_load_features(same, None, bundles["leaky"]).past_keep == (
        made.past_keep
    )


@pytest.mark.parametrize("name", ["countreg", "diffreg"])
def test_the_real_thesis_sets_are_refused_under_leaky(tmp_path, bundles, name: str) -> None:
    path = REPO_ROOT / "golden" / "converted" / "feature_sets" / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"{path} not present")
    cfg = make_cfg(tmp_path, "leaky", selector=name, cache_path=str(path))
    with pytest.raises(FeatureCacheMismatch, match="LEGACY expdecay7"):
        data_stage.build_or_load_features(cfg, None, bundles["leaky"])


# --------------------------------------------------------------------------- #
# the hurdle / damage selectors (no test existed, audit A)
# --------------------------------------------------------------------------- #
def _bundle(target: str, *, binary: bool, seed: int):
    return build_bundle(
        make_panel(target, binary=binary, seed=seed), target, PAST, FUTURE, window("leaky"),
        ACTIVITY,
    )


def test_zipoisson_regressor_selector_uses_positive_only_weights() -> None:
    bundle = _bundle(TARGET, binary=False, seed=3)
    config = fs("zipoisson_regressor").build()
    weights = [
        w.slice_intersect(ts)
        for w, ts in zip(positive_only_weights(bundle.target_full), bundle.target_train,
                         strict=True)
    ]
    first = select_top_k(
        bundle.target_train, bundle.past_covs, bundle.future_covs, config, sample_weight=weights
    )
    again = select_top_k(
        bundle.target_train, bundle.past_covs, bundle.future_covs, config, sample_weight=weights
    )
    unweighted = select_top_k(bundle.target_train, bundle.past_covs, bundle.future_covs, config)

    assert first.top_features == again.top_features
    pd.testing.assert_frame_equal(first.gain_table, again.gain_table)
    assert len(first.top_features) == config.top_k
    assert first.past_keep and set(first.past_keep) <= set(bundle.past_covs[0].components)
    assert set(first.future_keep) <= set(bundle.future_covs[0].components)
    # the weights reach the fit: the gain table is not the unweighted one
    assert not np.allclose(
        first.gain_table.set_index("Feature")["mean_gain"].sort_index().to_numpy(),
        unweighted.gain_table.set_index("Feature")["mean_gain"].sort_index().to_numpy(),
    )
    # the legacy (non-deterministic) builder is the library-default LightGBM
    assert zipoisson_regressor_config().model_kwargs["random_state"] == 42


def test_zipoisson_classifier_selector_is_repeatable_on_a_binary_target() -> None:
    bundle = _bundle(BINARY, binary=True, seed=4)
    config = fs("zipoisson_classifier").build()
    assert config.kind == "classifier" and config.objective == "binary"
    assert config.model_kwargs["is_unbalance"] is True
    first = select_top_k(bundle.target_train, bundle.past_covs, bundle.future_covs, config)
    again = select_top_k(bundle.target_train, bundle.past_covs, bundle.future_covs, config)
    assert first.top_features == again.top_features
    assert first.past_keep and set(first.past_keep) <= set(bundle.past_covs[0].components)
    assert zipoisson_classifier_config().model_kwargs["is_unbalance"] is True


def test_featsel_selects_every_hurdle_head_once(tmp_path, monkeypatch) -> None:
    """``select_features`` = ``strikecast featsel``: one cached selection per head."""
    cfg = ExperimentConfig(
        name="hurdle",
        data=DataConfig(target=TARGET),
        series=window("leaky"),
        feature_selection=fs("zipoisson_regressor"),
        feature_selections={
            "regressor": fs("zipoisson_regressor"),
            "classifier": fs("zipoisson_classifier"),
        },
        tracking=TrackingConfig(backend="noop"),
        store=StoreConfig(root=str(tmp_path / "runs")),
    )
    panels = {
        TARGET: make_panel(TARGET, binary=False, seed=5),
        BINARY: make_panel(BINARY, binary=True, seed=6),
    }

    def fake_panel(head_cfg, store=None, **_):
        return panels[head_cfg.data.target], [], data_stage._panel_hash(head_cfg)

    monkeypatch.setattr(data_stage, "build_or_load_panel", fake_panel)
    import strikecast.data.covariates as covariates  # noqa: PLC0415

    monkeypatch.setattr(
        covariates,
        "split_covariates",
        lambda panel, weather, target: SimpleNamespace(
            past_covariates=list(PAST), future_covariates=list(FUTURE)
        ),
    )
    inputs = SimpleNamespace(activity_by_region=dict(ACTIVITY))
    store = RunStore(cfg.store.root)

    made = data_stage.select_features(cfg, store, inputs=inputs)  # type: ignore[arg-type]
    assert set(made) == {"regressor", "classifier"}
    assert {s.source for s in made.values()} == {"computed"}
    targets = {
        json.loads(s.path.read_text(encoding="utf-8"))["provenance"]["target"]  # type: ignore[union-attr]
        for s in made.values()
    }
    assert targets == {TARGET, BINARY}

    again = data_stage.select_features(cfg, store, inputs=inputs)  # type: ignore[arg-type]
    assert {s.source for s in again.values()} == {"store"}
    # and a cv/test job (require_cached, compute=None) now loads both heads
    loaded = data_stage.prepare_composite_data(cfg, store, inputs=inputs)  # type: ignore[arg-type]
    assert {h: a.features.source for h, a in loaded.heads.items()} == {
        "regressor": "store",
        "classifier": "store",
    }


# --------------------------------------------------------------------------- #
# A14: the panel hash covers the input files' content
# --------------------------------------------------------------------------- #
def test_panel_hash_follows_the_input_file_content(tmp_path) -> None:
    fixed, dataset = tmp_path / "fixed", tmp_path / "dataset"
    fixed.mkdir()
    dataset.mkdir()
    for name in ("regions.txt", "regions_activity_cat.json", "actors.json"):
        (fixed / name).write_text("x", encoding="utf-8")
    parquet = dataset / "master_combined_timeseries.parquet"
    pd.DataFrame({"a": [1.0, 2.0]}).to_parquet(parquet)

    cfg = ExperimentConfig(
        name="count", data=DataConfig(fixed_dir=str(fixed), dataset_dir=str(dataset))
    )
    first = data_stage._panel_hash(cfg)
    assert data_stage._panel_hash(cfg) == first
    pd.DataFrame({"a": [1.0, 3.0]}).to_parquet(parquet)
    changed = data_stage._panel_hash(cfg)
    assert changed != first
    (fixed / "regions.txt").write_text("y", encoding="utf-8")
    assert data_stage._panel_hash(cfg) != changed
    assert data_stage.input_digests(cfg)["master_combined_timeseries.parquet"] is not None
