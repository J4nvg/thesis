"""``sensitivity.feature_space``: the feature-ablation switch (plan 2026-10-01).

The switch must (1) leave every default run bit-for-bit unchanged (no key in the
config dump, legacy darts skeleton, the publication selection hash), (2) build
exactly the feature space each mode promises (``all``, ``random`` uniform and
stratified, ``groups``, ``selected_minus``, ``reselect_*``), (3) keep the
selection hash for every mode but ``reselect_*`` (so tuned parameters still
resolve) and give each re-selection its own hash, and (4) let a darts model fit
and predict with no past covariates, no future covariates and no encoders.

Synthetic data only: pure functions on hand-made component names, the data
stage on a small panel through ``build_bundle``, and a few tiny model fits.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

pytest.importorskip("darts")
pytest.importorskip("lightgbm")

from darts import TimeSeries  # noqa: E402

import strikecast.models.gbm  # noqa: E402,F401  (populates the registry)
from strikecast.config.loader import load_experiment  # noqa: E402
from strikecast.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    FeatureSelectionStageConfig,
    FeatureSpaceConfig,
    SensitivityConfig,
    SeriesConfig,
    StoreConfig,
    TrackingConfig,
    WindowTransformConfig,
)
from strikecast.data import feature_selection as fsel  # noqa: E402
from strikecast.data import feature_space as fspace  # noqa: E402
from strikecast.data.feature_selection import legacy_common_kwargs  # noqa: E402
from strikecast.data.series import build_bundle, subset_components  # noqa: E402
from strikecast.models.spec import RunContext, darts_common_kwargs, get_spec  # noqa: E402
from strikecast.pipeline import data_stage, run_stage  # noqa: E402
from strikecast.store import RunStore  # noqa: E402

FS = "+sensitivity.feature_space"

# one realistic base name per group (the importance rules match substrings)
NAMES = {
    "strikes": "act_total_daily_strike_events",
    "spatial": "dist_to_nearest_ru_km",
    "conflict": "acled_other_ua_armed_clash",
    "comms": "com_aid_military",
    "macro": "fin_usd_rub_rate",
    "missile": "act_confirmed_launched",
    "cyber": "act_cyber_rus_ua",
}
FUTURE_NAMES = {"weather": ["env_weather_rain_sum", "env_k_max"], "calendar": ["env_ua_holiday"]}
PAST = [f"rolling_rsum7_7_{base}" for base in NAMES.values()] + [
    f"ewm_leaky7_{base}" for base in NAMES.values()
]
FUTURE = [*FUTURE_NAMES["weather"], *FUTURE_NAMES["calendar"]]
LAGS = (-1, -7, -14)


def _count(*extra: str) -> ExperimentConfig:
    return load_experiment("count", ["paradigm=global", "hydra.job.chdir=false", *extra])


# --------------------------------------------------------------------------- #
# config: default unchanged, overrides, validation
# --------------------------------------------------------------------------- #
def test_default_config_has_no_feature_space_anywhere():
    cfg = _count()
    assert cfg.sensitivity is None
    assert "sensitivity" not in cfg.model_dump(mode="json")
    ctx = cfg.to_run_context("catboost_tweedie", 42)
    assert (ctx.calendar_encoders, ctx.future_covariates) == (True, True)
    assert darts_common_kwargs(RunContext()) == legacy_common_kwargs()


def test_the_futwin_dump_is_unchanged_by_the_new_field():
    # the futwin stores' stage identities must not move
    cfg = _count("+sensitivity.future_covariate_lags=[8,7]")
    assert cfg.model_dump(mode="json")["sensitivity"] == {"future_covariate_lags": [8, 7]}


def test_override_reaches_the_dump_and_round_trips():
    cfg = _count(f"{FS}.mode=groups", f"{FS}.groups=[weather,cyber,cyber]")
    dumped = cfg.model_dump(mode="json")["sensitivity"]
    assert dumped == {
        "feature_space": {
            "mode": "groups", "k": 100, "draw": 0, "stratified": False,
            "groups": ["cyber", "weather"],  # display order, de-duplicated
        }
    }
    assert ExperimentConfig.model_validate(cfg.model_dump(mode="json")) == cfg


def test_an_empty_group_list_is_the_core():
    cfg = _count(f"{FS}.mode=groups", f"{FS}.groups=[]")
    fs = cfg.sensitivity.feature_space
    assert fs.kept_groups == () and not fs.calendar_encoders and not fs.future_covariates
    ctx = cfg.to_run_context("catboost_tweedie", 42)
    assert (ctx.calendar_encoders, ctx.future_covariates) == (False, False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "groups", "groups": ["nope"]},
        {"mode": "all", "draw": 3},
        {"mode": "selected", "groups": ["weather"]},
        {"mode": "random", "groups": ["weather"]},
        {"mode": "random", "k": 0},
        {"mode": "groups", "stratified": True},
        {"mode": "selected_minus"},
        {"mode": "reselect_drop"},
        {"mode": "reselect_only", "groups": ["weather", "calendar"]},  # no past pool
        {"mode": "reselect_drop", "groups": list(fspace.PAST_GROUPS)},  # no past pool
        {"mode": "nonsense"},
    ],
)
def test_invalid_feature_spaces_are_refused(kwargs):
    with pytest.raises(ValidationError):
        FeatureSpaceConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "kept", "encoders", "future"),
    [
        ({"mode": "selected"}, fspace.GROUP_ORDER, True, True),
        ({"mode": "all"}, fspace.GROUP_ORDER, True, True),
        ({"mode": "random", "draw": 4, "stratified": True}, fspace.GROUP_ORDER, True, True),
        ({"mode": "groups", "groups": ["cyber"]}, ("cyber",), False, False),
        ({"mode": "groups", "groups": ["weather"]}, ("weather",), False, True),
        ({"mode": "groups", "groups": ["calendar"]}, ("calendar",), True, True),
        (
            {"mode": "selected_minus", "groups": ["calendar"]},
            tuple(g for g in fspace.GROUP_ORDER if g != "calendar"),
            False,
            True,
        ),
        ({"mode": "reselect_only", "groups": ["comms"]}, ("comms",), False, False),
        (
            {"mode": "reselect_drop", "groups": ["weather"]},
            tuple(g for g in fspace.GROUP_ORDER if g != "weather"),
            True,
            True,
        ),
    ],
)
def test_kept_groups_and_switches_per_mode(kwargs, kept, encoders, future):
    fs = FeatureSpaceConfig(**kwargs)
    assert fs.kept_groups == kept
    assert (fs.calendar_encoders, fs.future_covariates) == (encoders, future)
    assert fs.reselects == kwargs["mode"].startswith("reselect")


# --------------------------------------------------------------------------- #
# pure functions
# --------------------------------------------------------------------------- #
def test_every_group_name_maps_to_its_slug():
    for slug, base in NAMES.items():
        assert fspace.group_of(f"rolling_rsum14_14_{base}") == slug
    for slug, names in FUTURE_NAMES.items():
        for name in names:
            assert fspace.group_of(name, future=True) == slug


def test_unclassified_or_misplaced_components_raise():
    with pytest.raises(ValueError, match="no feature group"):
        fspace.group_of("totally_unknown_column")
    with pytest.raises(ValueError, match="only"):
        fspace.group_of("act_cyber_rus_ua", future=True)
    with pytest.raises(ValueError, match="past covariates must"):
        fspace.group_of("env_weather_rain_sum")


def test_pool_is_every_component_at_every_lag_in_canonical_order():
    pool = fspace.pool_pairs(PAST, [-14, -1, -7])
    assert len(pool) == 3 * len(PAST)
    assert pool[:3] == [(PAST[0], -14), (PAST[0], -7), (PAST[0], -1)]


def test_random_draw_is_exact_deterministic_and_draw_dependent():
    pool = fspace.pool_pairs(PAST, LAGS)
    a = fspace.random_pairs(pool, 10, draw=1)
    assert len(a) == 10 and len(set(a)) == 10 and set(a) <= set(pool)
    assert a == fspace.random_pairs(pool, 10, draw=1)
    assert a != fspace.random_pairs(pool, 10, draw=2)
    with pytest.raises(ValueError):
        fspace.random_pairs(pool, len(pool) + 1, draw=1)


def test_stratified_draw_matches_the_strata():
    pool = fspace.pool_pairs(PAST, LAGS)
    strata = {"conflict": 4, "comms": 3, "strikes": 1}
    picked = fspace.random_pairs(pool, 8, draw=5, strata=strata)
    got = {g: n for g, n in fspace.pairs_by_group(fspace._group_pairs(picked)).items() if n}
    assert got == strata
    with pytest.raises(ValueError, match="sum"):
        fspace.random_pairs(pool, 9, draw=5, strata=strata)
    with pytest.raises(ValueError, match="stratum"):
        fspace.random_pairs(pool, 7, draw=5, strata={"cyber": 7})  # 2 comps x 3 lags = 6


SELECTED = (
    (PAST[2], (-7, -1)),  # conflict (lags ascending, as selections store them)
    (PAST[3], (-14,)),  # comms
    (PAST[0], (-1,)),  # strikes
    (PAST[9], (-7,)),  # conflict (leaky7)
)


def _space(mode, **kw):
    return fspace.resolve_space(
        mode,
        selected_past_lags=SELECTED,
        past_components=PAST,
        future_components=FUTURE,
        lag_grid=LAGS,
        **kw,
    )


def test_all_is_every_pair_and_equals_groups_with_every_group():
    past_lags, future = _space("all")
    assert sum(len(lags) for _, lags in past_lags) == 3 * len(PAST)
    assert future == FUTURE
    assert _space("groups", groups=fspace.GROUP_ORDER) == (past_lags, future)


def test_core_has_no_past_and_no_future():
    assert _space("groups", groups=()) == ((), [])


def test_only_one_group_keeps_just_that_group():
    past_lags, future = _space("groups", groups=["cyber"])
    assert {fspace.group_of(c) for c, _ in past_lags} == {"cyber"}
    assert all(lags == (-14, -7, -1) for _, lags in past_lags) and future == []
    assert _space("groups", groups=["weather"]) == ((), FUTURE_NAMES["weather"])


def test_selected_minus_removes_exactly_that_groups_pairs():
    past_lags, future = _space("selected_minus", groups=["conflict"])
    assert past_lags == ((PAST[3], (-14,)), (PAST[0], (-1,)))
    assert future == FUTURE
    past_lags, future = _space("selected_minus", groups=["weather"])
    assert past_lags == SELECTED and future == FUTURE_NAMES["calendar"]


def test_random_keeps_every_future_input_and_stratifies_on_the_selection():
    past_lags, future = _space("random", k=5, draw=4, stratified=True)
    assert future == FUTURE
    counts = fspace.pairs_by_group(past_lags)
    assert counts == fspace.pairs_by_group(SELECTED)  # conflict 3, comms 1, strikes 1


def test_selected_is_the_selection_and_reselect_is_not_resolved_here():
    assert _space("selected") == (SELECTED, FUTURE)
    with pytest.raises(ValueError, match="re-selection"):
        _space("reselect_only", groups=["comms"])


def test_restricted_components_is_the_reselect_pool():
    past, future = fspace.restricted_components("reselect_drop", ["conflict", "weather"], PAST, FUTURE)
    assert all(fspace.group_of(c) != "conflict" for c in past) and len(past) == len(PAST) - 2
    assert future == FUTURE_NAMES["calendar"]


def test_describe_counts_pairs_per_group():
    payload = fspace.describe(SELECTED, FUTURE, mode="selected")
    assert payload["mode"] == "selected" and payload["n_past_pairs"] == 5
    assert payload["pairs_by_group"]["conflict"] == 3 and payload["pairs_by_group"]["cyber"] == 0


# --------------------------------------------------------------------------- #
# darts skeleton and fits
# --------------------------------------------------------------------------- #
def test_darts_kwargs_per_switch():
    kw = darts_common_kwargs(RunContext(past_lags=(), calendar_encoders=False, future_covariates=False))
    assert kw["lags_past_covariates"] is None
    assert kw["lags_future_covariates"] is None
    assert kw["add_encoders"] is None
    assert kw["lags"] == 7  # the core's target lags stay
    kw = darts_common_kwargs(RunContext(calendar_encoders=False))
    assert kw["add_encoders"] is None and kw["lags_future_covariates"] == (2, 7)
    assert kw["lags_past_covariates"] == [-1, -7, -14]


def _statics(i: int, n: int) -> pd.DataFrame:
    return pd.DataFrame({f"region_{j}": [float(i == j)] for j in range(n)})


def _toy(n_series: int = 3, n_steps: int = 120):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n_steps, freq="D")
    fidx = pd.date_range("2020-01-01", periods=n_steps + 7, freq="D")
    targets, past, future = [], [], []
    for i in range(n_series):
        st = _statics(i, n_series)
        targets.append(
            TimeSeries.from_times_and_values(
                idx, rng.poisson(2.0, (n_steps, 1)).astype(float), columns=["y"]
            ).with_static_covariates(st)
        )
        past.append(
            TimeSeries.from_times_and_values(
                idx, rng.poisson(2.0, (n_steps, 2)).astype(float), columns=["a", "b"]
            ).with_static_covariates(st)
        )
        future.append(
            TimeSeries.from_times_and_values(
                fidx, rng.normal(size=(n_steps + 7, 1)), columns=["w"]
            ).with_static_covariates(st)
        )
    return targets, past, future


@pytest.mark.parametrize(
    ("name", "overrides"),
    [("lightgbm_poisson", {"n_estimators": 5}), ("catboost_tweedie", {"iterations": 5})],
)
@pytest.mark.parametrize(
    ("ctx_kw", "use_past", "use_future"),
    [
        ({"past_lags": (), "calendar_encoders": False, "future_covariates": False}, False, False),
        ({"past_lags": (), "calendar_encoders": False}, False, True),  # only weather
        ({"past_lags": (), "calendar_encoders": True}, False, False),  # encoders, no comps
        ({"past_lags": (("a", (-1,)),), "calendar_encoders": False, "future_covariates": False},
         True, False),  # one past group, no future
    ],
)
def test_a_core_like_model_fits_and_predicts(name, overrides, ctx_kw, use_past, use_future):
    spec = get_spec(name, "count")
    ctx = RunContext(seed=42, device=spec.device, threads=1, **ctx_kw)
    model = spec.build({**dict(spec.defaults), **overrides}, ctx)
    targets, past, future = _toy()
    fit = {"series": targets}
    pred = {"n": 7, "series": targets}
    # exactly what the adapter does: pass a kind only when the model takes it
    assert model.supports_past_covariates is use_past
    if use_past:
        fit["past_covariates"] = pred["past_covariates"] = [ts[["a"]] for ts in past]
    if use_future:
        fit["future_covariates"] = pred["future_covariates"] = future
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**fit)
        out = model.predict(**pred)
    assert len(out) == len(targets) and len(out[0]) == 7
    names = list(model.lagged_feature_names)
    assert sum("_target_lag" in n for n in names) == 7
    assert any("statcov" in n for n in names)  # the statics are core
    assert any("_pastcov_" in n for n in names) is use_past
    assert any(n.startswith("w_futcov") for n in names) is use_future
    assert any("_cyc_" in n for n in names) is ctx.calendar_encoders


def test_run_stage_passes_none_for_switched_off_kinds():
    bundle = SimpleNamespace(past_covs=["p"], raw_past_covs=["rp"], future_covs=["f"])
    spec = SimpleNamespace(needs_raw_past_covs=False)
    on = SimpleNamespace(bundle=bundle, features=data_stage.FeatureSets([], [], "s", "h"))
    assert run_stage._covariates(spec, on) == (["p"], ["f"])
    off = SimpleNamespace(
        bundle=bundle,
        features=data_stage.FeatureSets([], [], "s", "h", use_past=False, use_future=False),
    )
    assert run_stage._covariates(spec, off) == (None, None)


# --------------------------------------------------------------------------- #
# data stage on a small synthetic panel
# --------------------------------------------------------------------------- #
TARGET = "act_drone_strike_on_ua"
REGIONS = ("alpha", "beta", "gamma")
ACTIVITY = {"alpha": 1, "beta": 2, "gamma": 3}
PANEL_PAST = list(NAMES.values())


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=160, freq="D")
    frames = []
    for i, region in enumerate(REGIONS):
        driver = rng.poisson(1.0 + i, len(dates)).astype(float)
        cols = {name: rng.poisson(1.0, len(dates)).astype(float) for name in PANEL_PAST}
        cols[NAMES["conflict"]] = np.roll(driver, 1)  # the planted signal
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "event_date": dates,
                    "Activity_Level": ACTIVITY[region],
                    TARGET: rng.poisson(0.5 + 0.3 * driver).astype(float),
                    **cols,
                    "env_weather_rain_sum": rng.normal(size=len(dates)),
                    "env_k_max": rng.normal(size=len(dates)),
                    "env_ua_holiday": (dates.dayofweek >= 5).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def bundle():
    window = SeriesConfig(window=WindowTransformConfig(expdecay="leaky"))
    return build_bundle(_panel(), TARGET, PANEL_PAST, FUTURE, window, ACTIVITY)


@pytest.fixture(autouse=True)
def _hash_seed_zero(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")


def _cfg(tmp_path: Path, feature_space: FeatureSpaceConfig | None, top_k: int = 12):
    return ExperimentConfig(
        name="count",
        data=DataConfig(target=TARGET),
        series=SeriesConfig(window=WindowTransformConfig(expdecay="leaky")),
        feature_selection=FeatureSelectionStageConfig(
            selector="count_tweedie", device="cpu", num_threads=2, top_k=top_k, cache=True,
            deterministic=True, require_cached=True,
        ),
        tracking=TrackingConfig(backend="noop"),
        store=StoreConfig(root=str(tmp_path / "runs")),
        sensitivity=None if feature_space is None else SensitivityConfig(feature_space=feature_space),
    )


def _hash(cfg: ExperimentConfig) -> str:
    return data_stage._features_hash(cfg, "panel", "series")


def test_only_a_reselection_moves_the_features_hash(tmp_path):
    base = _hash(_cfg(tmp_path, None))
    for fs in (
        FeatureSpaceConfig(mode="selected"),
        FeatureSpaceConfig(mode="all"),
        FeatureSpaceConfig(mode="random", draw=2),
        FeatureSpaceConfig(mode="groups", groups=["cyber"]),
        FeatureSpaceConfig(mode="selected_minus", groups=["conflict"]),
    ):
        assert _hash(_cfg(tmp_path, fs)) == base, fs
    drop_a = _hash(_cfg(tmp_path, FeatureSpaceConfig(mode="reselect_drop", groups=["conflict"])))
    drop_b = _hash(_cfg(tmp_path, FeatureSpaceConfig(mode="reselect_drop", groups=["comms"])))
    only_a = _hash(_cfg(tmp_path, FeatureSpaceConfig(mode="reselect_only", groups=["conflict"])))
    assert len({base, drop_a, drop_b, only_a}) == 4


def test_the_count_features_hash_ignores_every_non_reselect_mode():
    base = _count().for_selection("tweedie")
    ph = data_stage._panel_hash(base)
    sh = data_stage._series_hash(base, ph)
    want = data_stage._features_hash(base, ph, sh)
    for extra in ([f"{FS}.mode=all"], [f"{FS}.mode=groups", f"{FS}.groups=[]"]):
        cfg = _count(*extra).for_selection("tweedie")
        assert data_stage._features_hash(cfg, ph, sh) == want


def _selection(tmp_path, bundle):
    cfg = _cfg(tmp_path, None)
    return data_stage.build_or_load_features(cfg, RunStore(cfg.store.root), bundle, compute=True)


@pytest.mark.parametrize(
    ("fs", "n_pairs", "use_past", "use_future"),
    [
        (FeatureSpaceConfig(mode="all"), None, True, True),
        (FeatureSpaceConfig(mode="groups", groups=[]), 0, False, False),
        (FeatureSpaceConfig(mode="groups", groups=["weather"]), 0, False, True),
        (FeatureSpaceConfig(mode="groups", groups=["cyber"]), None, True, False),
        (FeatureSpaceConfig(mode="random", k=12, draw=1), 12, True, True),
        (FeatureSpaceConfig(mode="random", k=12, draw=4, stratified=True), 12, True, True),
    ],
)
def test_apply_feature_space_on_the_real_bundle_shape(tmp_path, bundle, fs, n_pairs, use_past, use_future):
    selection = _selection(tmp_path, bundle)
    cfg = _cfg(tmp_path, fs)
    got = data_stage._apply_feature_space(cfg, selection, bundle)
    assert got.hash == selection.hash  # tuned params still resolve
    assert got.source == f"sensitivity:{fs.mode}"
    assert (got.use_past, got.use_future) == (use_past, use_future)
    pairs = sum(len(lags) for _, lags in got.past_lags)
    pool = data_stage._components_of(bundle.raw.past)
    if fs.mode == "all":
        assert pairs == 3 * len(pool)
    if n_pairs is not None:
        assert pairs == n_pairs
    assert got.past_keep == [c for c, _ in got.past_lags]
    if fs.stratified:
        assert fspace.pairs_by_group(got.past_lags) == fspace.pairs_by_group(selection.past_lags)
    # the data stage's subset must build (darts cannot hold a 0-component series)
    sub = subset_components(
        bundle,
        got.past_keep if got.use_past else pool,
        got.future_keep if got.use_future else data_stage._components_of(bundle.raw.future),
    )
    if got.use_past:
        assert list(sub.past_covs[0].components) == got.past_keep


def test_selected_minus_on_the_bundle(tmp_path, bundle):
    selection = _selection(tmp_path, bundle)
    cfg = _cfg(tmp_path, FeatureSpaceConfig(mode="selected_minus", groups=["conflict", "calendar"]))
    got = data_stage._apply_feature_space(cfg, selection, bundle)
    before = fspace.pairs_by_group(selection.past_lags)
    after = fspace.pairs_by_group(got.past_lags)
    assert after["conflict"] == 0
    assert {g: n for g, n in after.items() if g != "conflict"} == {
        g: n for g, n in before.items() if g != "conflict"
    }
    assert got.future_keep == FUTURE_NAMES["weather"]


def test_reselect_ranks_only_the_restricted_pool_with_matching_inputs(tmp_path, bundle, monkeypatch):
    calls = []
    real = fsel.select_top_k

    def spy(train_target, past, future, config, **kw):
        calls.append({"past": past, "future": future, "config": config})
        return real(train_target, past, future, config, **kw)

    monkeypatch.setattr(fsel, "select_top_k", spy)
    fs = FeatureSpaceConfig(mode="reselect_only", groups=["conflict"])
    cfg = _cfg(tmp_path, fs, top_k=5)
    store = RunStore(cfg.store.root)
    got = data_stage._reselect_features(
        cfg, store, bundle, panel_hash="p", series_hash="s", force=False, compute=None
    )
    (call,) = calls
    assert {fspace.group_of(c) for c in call["past"][0].components} == {"conflict"}
    assert call["future"] is None  # reselect_only_conflict has no future input ...
    assert call["config"].model_kwargs["add_encoders"] is None  # ... and no encoders
    assert call["config"].model_kwargs["lags_future_covariates"] is None
    assert {fspace.group_of(c) for c in got.past_keep} == {"conflict"}
    assert sum(len(lags) for _, lags in got.past_lags) == 5
    assert (got.use_past, got.use_future) == (True, False)
    assert got.source == "sensitivity:reselect_only:computed"
    assert got.hash != _hash(_cfg(tmp_path, None))

    # cached under its own hash: the second call is a store hit, no new fit
    again = data_stage._reselect_features(
        cfg, store, bundle, panel_hash="p", series_hash="s", force=False, compute=None
    )
    assert len(calls) == 1 and again.past_lags == got.past_lags
    assert again.source == "sensitivity:reselect_only:store"


def test_reselect_drop_weather_keeps_the_calendar(tmp_path, bundle, monkeypatch):
    calls = []
    real = fsel.select_top_k

    def spy(train_target, past, future, config, **kw):
        calls.append({"future": future, "config": config})
        return real(train_target, past, future, config, **kw)

    monkeypatch.setattr(fsel, "select_top_k", spy)
    cfg = _cfg(tmp_path, FeatureSpaceConfig(mode="reselect_drop", groups=["weather"]), top_k=5)
    got = data_stage._reselect_features(
        cfg, RunStore(cfg.store.root), bundle, panel_hash="p", series_hash="s", force=False,
        compute=None,
    )
    (call,) = calls
    assert list(call["future"][0].components) == FUTURE_NAMES["calendar"]
    assert call["config"].model_kwargs["add_encoders"] is not None
    assert got.future_keep == FUTURE_NAMES["calendar"] and got.use_future
