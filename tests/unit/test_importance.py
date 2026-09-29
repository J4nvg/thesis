"""Feature importance (audit C14): the port against the verbatim legacy code.

* :func:`strikecast.evaluation.importance.gbm_importances` against
  ``get_gbm_importances`` of ``_regression_GBDT.ipynb`` cell 57, inlined below
  as the oracle, on the SAME fitted darts models (LightGBM and CatBoost, with
  darts encoders so the ``encode_inference`` branch runs) -- exact equality.
* :func:`classify_feature` against ``results/analyse_results.ipynb`` cell 52
  (inlined) and the thesis table ``tab:semantic_categories``; the expdecay
  switch (``expdecay7`` -> ``leaky7``) must not move any feature's category.
* The category shares recomputed from the thesis's own
  ``results/gbdt/importance_all.csv`` reproduce every percentage the thesis
  text quotes for Fig. ``importancesharebycategory_grouped`` (main.tex
  1117-1119).
* :mod:`strikecast.pipeline.importance_stage`: outputs, labels, resume/skip,
  the Chronos branch (fake predictor) and ``collect_importance``'s order.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strikecast.evaluation import importance as imp

REPO_ROOT = Path(__file__).resolve().parents[2]
STORED_IMPORTANCE = REPO_ROOT / "results" / "gbdt" / "importance_all.csv"
OCL = 3


# --------------------------------------------------------------------------- #
# the legacy oracles, verbatim
# --------------------------------------------------------------------------- #
def legacy_get_gbm_importances(fitted_model, model_name_prefix, series, past_covs, fut_covs):
    """``_regression_GBDT.ipynb`` cell 57, verbatim (``display`` etc. removed)."""
    from sklearn.inspection import permutation_importance

    underlying = fitted_model.model
    estimators = getattr(underlying, "estimators_", [underlying])
    features = fitted_model.lagged_feature_names

    df_imps = {"Feature": features}

    # 1. Native Booster Importances (Gain)
    for h, est in enumerate(estimators, start=1):
        if "lightgbm" in model_name_prefix:
            imp_ = est.booster_.feature_importance(importance_type="gain")
        elif "catboost" in model_name_prefix:
            imp_ = est.get_feature_importance()
        else:
            imp_ = est.feature_importances_
        df_imps[f"h{h}_gain"] = imp_

    # 2. Sklearn Permutation Importances
    if hasattr(fitted_model, "encoders") and fitted_model.encoders.encoding_available:
        past_covs_enc, fut_covs_enc = fitted_model.encoders.encode_inference(
            n=fitted_model.output_chunk_length,
            target=series,
            past_covariates=past_covs,
            future_covariates=fut_covs,
        )
        series_enc = series
    else:
        series_enc, past_covs_enc, fut_covs_enc = series, past_covs, fut_covs

    lagged_data = fitted_model._create_lagged_data(
        series=series_enc,
        past_covariates=past_covs_enc,
        future_covariates=fut_covs_enc,
        max_samples_per_ts=None,
        sample_weight=None,
    )
    X, y_target = lagged_data[0], lagged_data[1]

    for h, est in enumerate(estimators, start=1):
        y_h = y_target[:, h - 1]
        r = permutation_importance(est, X, y_h, n_repeats=5, random_state=42, n_jobs=-1)
        df_imps[f"h{h}_perm"] = r.importances_mean

    df = pd.DataFrame(df_imps)
    gain_cols = [c for c in df.columns if c.endswith("_gain")]
    perm_cols = [c for c in df.columns if c.endswith("_perm")]
    df["agg_gain"] = df[gain_cols].mean(axis=1)
    df["agg_perm"] = df[perm_cols].mean(axis=1)
    return df.sort_values("agg_perm", ascending=False).reset_index(drop=True)


def legacy_classify_feature(name: str) -> str:
    """``results/analyse_results.ipynb`` cell 52, verbatim."""
    n = name.lower()
    if any(k in n for k in ("region_statcov", "dist_to_nearest", "dist_x_clash", "area_km2")):
        return "Spatial / static"
    if "env_weather" in n or "env_k_max" in n:
        return "Weather / geomag."
    if "holiday" in n:
        return "Calendar"
    if "act_cyber" in n:
        return "Cyber"
    if any(k in n for k in ("confirmed_launched", "confirmed_destroyed", "intercept_rate")):
        return "Missile / launch"
    if any(k in n for k in ("usd_rub", "usd_uah", "ch_export")):
        return "Macroeconomic"
    if any(k in n for k in ("com_aid", "com_diplo", "com_verbal", "com_restrictive",
                            "com_escalation", "com_logistic", "com_ners")):
        return "Comms / diplo / aid"
    if any(k in n for k in ("disrupted_weapons", "shelling", "armed_clash",
                            "damage_events", "drone_infra", "acled_other")):
        return "Conflict & damage"
    if any(k in n for k in ("total_daily_strike_events", "drone_strike_on", "_target",
                            "specialmilitary", "ratio_ua_rus")):
        return "Autoregressive strikes"
    return "Other"


# --------------------------------------------------------------------------- #
# synthetic darts GBDTs
# --------------------------------------------------------------------------- #
def _series(n_regions: int = 3, n: int = 90, seed: int = 0):
    from darts import TimeSeries

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    targets, past, future = [], [], []
    for _ in range(n_regions):
        p = rng.normal(size=(n, 2))
        f = rng.normal(size=(n + 10, 1))
        y = np.maximum(0, np.round(1 + p[:, 0] + 0.5 * np.roll(p[:, 1], 1) + rng.poisson(1, n)))
        targets.append(TimeSeries.from_times_and_values(idx, y.reshape(-1, 1), columns=["y"]))
        past.append(
            TimeSeries.from_times_and_values(idx, p, columns=["act_total_damage_events", "env_weather_rain"])
        )
        fidx = pd.date_range("2023-01-01", periods=n + 10, freq="D")
        future.append(TimeSeries.from_times_and_values(fidx, f, columns=["env_ua_holiday"]))
    return targets, past, future


ENCODERS = {"cyclic": {"future": ["dayofweek"], "past": ["month"]}}


def _lightgbm():
    from darts.models import LightGBMModel

    return LightGBMModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=[0],
        output_chunk_length=OCL,
        add_encoders=ENCODERS,
        n_estimators=15,
        num_leaves=4,
        min_child_samples=5,
        random_state=0,
        verbose=-1,
        num_threads=1,
    )


def _catboost():
    from darts.models import CatBoostModel

    return CatBoostModel(
        lags=3,
        lags_past_covariates=[-1, -2],
        lags_future_covariates=[0],
        output_chunk_length=OCL,
        add_encoders=ENCODERS,
        iterations=15,
        depth=3,
        random_state=0,
        verbose=False,
        thread_count=1,
    )


@pytest.fixture(scope="module")
def data():
    return _series()


@pytest.fixture(scope="module", params=["lightgbm", "catboost"])
def fitted(request, data):
    targets, past, future = data
    model = _lightgbm() if request.param == "lightgbm" else _catboost()
    model.fit(series=targets, past_covariates=past, future_covariates=future)
    return request.param, model


# --------------------------------------------------------------------------- #
# 1. the GBDT port equals cell 57
# --------------------------------------------------------------------------- #
def test_gbm_importances_equal_the_legacy_function(fitted, data) -> None:
    family, model = fitted
    targets, past, future = data
    got = imp.gbm_importances(model, family, targets, past, future)
    expected = legacy_get_gbm_importances(model, family, targets, past, future)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


def test_the_schema_is_the_stored_one(fitted, data) -> None:
    family, model = fitted
    targets, past, future = data
    got = imp.gbm_importances(model, family, targets, past, future, n_jobs=1)
    assert list(got.columns) == imp.importance_columns(OCL)
    assert len(got) == len(model.lagged_feature_names)
    assert got["agg_perm"].is_monotonic_decreasing
    gain = got[[f"h{h}_gain" for h in range(1, OCL + 1)]].mean(axis=1)
    np.testing.assert_allclose(got["agg_gain"], gain)
    # darts encoders were applied: their lagged columns are in the frame
    assert any(f.startswith("darts_enc") for f in got["Feature"])


def test_the_stored_csv_has_the_same_column_order() -> None:
    if not STORED_IMPORTANCE.exists():
        pytest.skip("results/gbdt/importance_all.csv not present")
    header = list(pd.read_csv(STORED_IMPORTANCE, nrows=1).columns)
    assert header == [*imp.importance_columns(7), "model"]


def test_n_jobs_never_changes_a_value(fitted, data) -> None:
    """LightGBM and CatBoost: sequential and loky workers give the same bits."""
    family, model = fitted
    targets, past, future = data
    a = imp.gbm_importances(model, family, targets, past, future, n_jobs=1)
    b = imp.gbm_importances(model, family, targets, past, future, n_jobs=2)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_gain_only_skips_the_permutation_half(data) -> None:
    targets, past, future = data
    model = _lightgbm()
    model.fit(series=targets, past_covariates=past, future_covariates=future)
    timings: dict[str, float] = {}
    got = imp.gbm_importances(model, "lightgbm", targets, past, future, permutation=False,
                              timings=timings)
    assert got["agg_perm"].isna().all()
    assert got["agg_gain"].is_monotonic_decreasing
    assert timings["perm_s"] == 0.0 and "gain_s" in timings


def test_lagged_design_matches_the_feature_names(fitted, data) -> None:
    _, model = fitted
    targets, past, future = data
    X, y = imp.lagged_design(model, targets, past, future)
    assert X.shape[1] == len(model.lagged_feature_names)
    assert y.shape == (X.shape[0], OCL)


# --------------------------------------------------------------------------- #
# 2. categories
# --------------------------------------------------------------------------- #
FEATURES = [
    "Activity_Level_statcov_target_act_drone_strike_on_ua",
    "region_statcov_target_act_drone_strike_on_ua",
    "act_drone_strike_on_ua_target_lag-1",
    "ewm_expdecay7_act_total_daily_strike_events_pastcov_lag-7",
    "ewm_expdecay7_acled_other_ua_disrupted_weapons_use_pastcov_lag-1",
    "ewm_expdecay7_total_ru_area_km2_pastcov_lag-1",
    "ewm_expdecay7_dist_to_nearest_contested_km_pastcov_lag-1",
    "rolling_rmean28_28_act_total_daily_strike_events_pastcov_lag-14",
    "ewm_ewma14_usd_rub_pastcov_lag-1",
    "env_weather_precipitation_hours_futcov_lag0",
    "env_k_max_futcov_lag1",
    "env_ua_holiday_futcov_lag2",
    "act_cyber_attacks_pastcov_lag-1",
    "act_missile_confirmed_launched_pastcov_lag-1",
    "com_aid_ru_west_pastcov_lag-7",
    "act_total_damage_events_pastcov_lag-1",
    "darts_enc_fc_cyc_dayofweek_sin_futcov_lag0",
]


#: Features whose category deliberately differs from the notebook (see CATEGORY_RULES).
DEVIATIONS = {"Activity_Level_statcov_target_act_drone_strike_on_ua": "Spatial / static"}


@pytest.mark.parametrize("name", [f for f in FEATURES if f not in DEVIATIONS])
def test_classify_feature_equals_the_notebook(name: str) -> None:
    assert imp.classify_feature(name) == legacy_classify_feature(name)


@pytest.mark.parametrize(("name", "category"), DEVIATIONS.items())
def test_activity_level_is_static_not_autoregressive(name: str, category: str) -> None:
    """The tier indicator is a static region descriptor (2026-09-29); the notebook's
    ``_target`` rule had made it autoregressive."""
    assert legacy_classify_feature(name) == "Autoregressive strikes"
    assert imp.classify_feature(name) == category


@pytest.mark.parametrize("name", [f for f in FEATURES if "expdecay7" in f])
def test_the_expdecay_switch_never_moves_a_category(name: str) -> None:
    """``series.window.expdecay: leaky`` renames ``expdecay7`` -> ``leaky7`` (D1)."""
    leaky = name.replace("ewm_expdecay7_", "ewm_leaky7_")
    assert imp.classify_feature(leaky) == imp.classify_feature(name)
    assert imp.window_transform(name) == "ewm_expdecay7"
    assert imp.window_transform(leaky) == "ewm_leaky7"
    assert imp.window_transform("env_weather_rain_sum_futcov_lag0") is None


def test_the_rules_are_the_thesis_table() -> None:
    """``tab:semantic_categories`` (main.tex ~1577), row for row, plus
    ``activity_level_statcov`` in Spatial / static (2026-09-29; the table gets the same
    row, docs/audits/2026-09-29/MAIN_TEX_PROPOSED_CHANGES.md)."""
    table = {
        "Autoregressive strikes": {"total_daily_strike_events", "drone_strike_on", "_target",
                                   "specialmilitary", "ratio_ua_rus"},
        "Spatial / static": {"region_statcov", "activity_level_statcov", "dist_to_nearest",
                             "dist_x_clash", "area_km2"},
        "Weather / geomag.": {"env_weather", "env_k_max"},
        "Conflict & damage": {"disrupted_weapons", "shelling", "armed_clash", "damage_events",
                              "drone_infra", "acled_other"},
        "Macroeconomic": {"usd_rub", "usd_uah", "ch_export"},
        "Comms / diplo / aid": {"com_aid", "com_diplo", "com_verbal", "com_restrictive",
                                "com_escalation", "com_logistic", "com_ners"},
        "Missile / launch": {"confirmed_launched", "confirmed_destroyed", "intercept_rate"},
        "Cyber": {"act_cyber"},
        "Calendar": {"holiday"},
    }
    assert {c: set(needles) for c, needles in imp.CATEGORY_RULES} == table
    assert set(imp.CATEGORY_ORDER) == {*table, "Other"}


# --------------------------------------------------------------------------- #
# 3. the thesis's percentages from the thesis's own importance file
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def thesis_matrix() -> pd.DataFrame:
    if not STORED_IMPORTANCE.exists():
        pytest.skip("results/gbdt/importance_all.csv not present")
    stored = pd.read_csv(STORED_IMPORTANCE)
    importancedict = {}
    for tier in range(1, 4):  # analysis notebook cell 47
        sub = stored[stored["model"] == f"activity_{tier}_catboost_tweedie"]
        importancedict[f"activity_{tier}"] = {
            "gain": imp.top_features(sub, "agg_gain"),
            "perm": imp.top_features(sub, "agg_perm"),
        }
    return imp.category_importance_matrix(importancedict)


@pytest.mark.parametrize(
    ("column", "category", "percent"),
    [
        # main.tex 1117-1119, "Reliance on different feature classes ..."
        ("Tier 1 gain", "Conflict & damage", 62),
        ("Tier 1 perm", "Conflict & damage", 48),
        ("Tier 1 perm", "Weather / geomag.", 30),
        ("Tier 1 perm", "Comms / diplo / aid", 15),
        ("Tier 2 gain", "Autoregressive strikes", 45),
        ("Tier 2 gain", "Conflict & damage", 32),
        ("Tier 2 gain", "Macroeconomic", 22),
        ("Tier 2 perm", "Weather / geomag.", 56),
        ("Tier 2 perm", "Autoregressive strikes", 28),
        ("Tier 2 perm", "Conflict & damage", 16),
        ("Tier 3 gain", "Autoregressive strikes", 82),
        ("Tier 3 perm", "Autoregressive strikes", 66),
        ("Tier 3 gain", "Spatial / static", 18),
        ("Tier 3 perm", "Spatial / static", 29),
    ],
)
def test_category_shares_reproduce_the_thesis_text(
    thesis_matrix: pd.DataFrame, column: str, category: str, percent: int
) -> None:
    assert round(100 * thesis_matrix.loc[category, column]) == percent


def test_every_matrix_column_sums_to_one(thesis_matrix: pd.DataFrame) -> None:
    np.testing.assert_allclose(thesis_matrix.sum(axis=0).to_numpy(), 1.0)
    assert list(thesis_matrix.columns) == [
        f"Tier {t} {m}" for t in (1, 2, 3) for m in ("gain", "perm")
    ]


def test_category_shares_long_covers_every_model() -> None:
    if not STORED_IMPORTANCE.exists():
        pytest.skip("results/gbdt/importance_all.csv not present")
    stored = pd.read_csv(STORED_IMPORTANCE)
    long = imp.category_shares_long(stored)
    assert list(long.columns) == ["model", "metric", "category", "share", "top_n"]
    sums = long.groupby(["model", "metric"])["share"].sum()
    assert len(sums) == 2 * stored["model"].nunique()
    np.testing.assert_allclose(sums.to_numpy(), 1.0)


# --------------------------------------------------------------------------- #
# 4. Chronos-2
# --------------------------------------------------------------------------- #
class FakeAGPredictor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def model_names(self):
        return ["Chronos2FT_best"]

    def feature_importance(self, data, **kwargs):
        self.calls.append({"data": data, **kwargs})
        return pd.DataFrame(
            {"importance": [0.3, 0.1], "stdev": [0.0, 0.0], "n": [5.0, 5.0],
             "p99_low": [0.3, 0.1], "p99_high": [0.3, 0.1]},
            index=["env_weather_precipitation_hours", "act_total_damage_events"],
        )


def test_chronos_importance_is_the_legacy_call() -> None:
    predictor = FakeAGPredictor()
    frame = imp.chronos_importance(predictor, "FULL_FRAME")
    (call,) = predictor.calls
    # `_chronos2.py:686`: data=test_data_ag (full frame), model=<name>,
    # relative_scores=True, every other argument AutoGluon's default.
    assert call == {"data": "FULL_FRAME", "model": "Chronos2FT_best", "relative_scores": True}
    assert list(frame.columns) == ["importance", "stdev", "n", "p99_low", "p99_high"]


def test_the_stored_chronos_csvs_have_autogluons_schema() -> None:
    for kind in ("zs", "ft"):
        path = REPO_ROOT / "results" / "chronos2" / f"feature_importance_chronos2_{kind}.csv"
        if not path.exists():
            pytest.skip(f"{path} not present")
        frame = pd.read_csv(path, index_col=0)
        assert list(frame.columns) == ["importance", "stdev", "n", "p99_low", "p99_high"]
        assert frame["importance"].is_monotonic_decreasing
        # 27 known + 84 past covariates (F123: the stray `index` column is one)
        # + the static `Activity_Level`
        assert len(frame) == 112
        assert {"index", "Activity_Level"} <= set(frame.index)


# --------------------------------------------------------------------------- #
# 5. the stage
# --------------------------------------------------------------------------- #
@pytest.fixture
def count_cfg(tmp_path: Path):
    from strikecast.config.loader import load_experiment

    return load_experiment("count", ["tracking=noop", f"++store.root={tmp_path / 'runs'}"])


def _fake_gbdt_setup(monkeypatch, data, family="lightgbm"):
    """A fake spec whose ``build`` is the synthetic model, and a fake DataArtifacts."""
    from strikecast.pipeline import importance_stage

    targets, past, future = data
    spec = SimpleNamespace(
        name="lightgbm_poisson", kind="global", family=family, tunable=False,
        defaults={}, fallback=None, from_best_params=lambda b: b,
        build=lambda params, ctx: _lightgbm() if family == "lightgbm" else _catboost(),
    )
    monkeypatch.setattr(importance_stage, "get_spec", lambda name, exp=None: spec)
    monkeypatch.setattr("strikecast.pipeline.run_stage.get_spec", lambda name, exp=None: spec)
    names = ["r_a", "r_b", "r_c"]
    bundle = SimpleNamespace(target_full=targets, past_covs=past, future_covs=future,
                             region_names=names)
    art = SimpleNamespace(
        bundle=bundle, region_names=names, activity_by_region={"r_a": 1, "r_b": 2, "r_c": 1},
        upstream=("p", "s", "f"),
        # resolve_params checks the tuned selection against this (plan §7)
        features=SimpleNamespace(hash="f", source="test", past_lags=None),
    )
    return art


def test_run_importance_writes_next_to_the_run(monkeypatch, data, count_cfg, tmp_path) -> None:
    from strikecast.pipeline import importance_stage as stage
    from strikecast.store import RunStore

    art = _fake_gbdt_setup(monkeypatch, data)
    store = RunStore(tmp_path / "runs")
    out = stage.run_importance(count_cfg, "lightgbm_poisson", "activity", 42, art, store=store,
                               n_jobs=1)
    assert not out.skipped
    assert out.labels == ["activity_1_lightgbm_poisson", "activity_2_lightgbm_poisson"]
    directory = tmp_path / "runs" / "count" / "lightgbm_poisson" / "activity" / "seed=42" / "importance"
    frame = pd.read_csv(directory / "importance.csv")
    assert list(frame.columns) == [*imp.importance_columns(OCL), "model"]
    assert list(dict.fromkeys(frame["model"])) == out.labels
    shares = pd.read_csv(directory / "category_shares.csv")
    assert set(shares["model"]) == set(out.labels)
    timings = json.loads((directory / "timings.json").read_text())
    assert set(timings["groups"]) == set(out.labels)
    state = json.loads((directory.parent / "state.json").read_text())
    assert state["importance"]["status"] == "complete"

    again = stage.run_importance(count_cfg, "lightgbm_poisson", "activity", 42, art, store=store,
                                 n_jobs=1)
    assert again.skipped
    gain_only = stage.run_importance(count_cfg, "lightgbm_poisson", "activity", 42, art,
                                     store=store, permutation=False)
    assert not gain_only.skipped  # a different identity


def test_activity_groups_are_fit_on_their_own_regions(monkeypatch, data, count_cfg, tmp_path):
    """Tier 1 = regions a and c; the frame must equal a direct fit on those two."""
    from strikecast.pipeline import importance_stage as stage
    from strikecast.store import RunStore

    art = _fake_gbdt_setup(monkeypatch, data)
    stage.run_importance(count_cfg, "lightgbm_poisson", "activity", 42, art,
                         store=RunStore(tmp_path / "runs"), n_jobs=1)
    got = pd.read_csv(
        tmp_path / "runs/count/lightgbm_poisson/activity/seed=42/importance/importance.csv"
    )
    got = got[got["model"] == "activity_1_lightgbm_poisson"].drop(columns="model")

    targets, past, future = data
    idx = [0, 2]
    model = _lightgbm()
    ts, pc, fc = [targets[i] for i in idx], [past[i] for i in idx], [future[i] for i in idx]
    model.fit(series=ts, past_covariates=pc, future_covariates=fc)
    # (the port == cell 57 on one fitted model is pinned above; n_jobs=1 for speed)
    expected = imp.gbm_importances(model, "lightgbm", ts, pc, fc, n_jobs=1)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected, check_exact=False,
                                  rtol=1e-12, atol=0)


def test_non_gbdt_models_are_refused(monkeypatch, count_cfg, tmp_path) -> None:
    from strikecast.pipeline import importance_stage as stage
    from strikecast.store import RunStore

    spec = SimpleNamespace(kind="global", family="lstm")
    monkeypatch.setattr(stage, "get_spec", lambda name, exp=None: spec)
    with pytest.raises(ValueError, match="GBDT families"):
        stage.run_importance(count_cfg, "lstm_w7", "global", 42, SimpleNamespace(upstream=()),
                             store=RunStore(tmp_path))


def test_collect_orders_like_importance_all(monkeypatch, data, count_cfg, tmp_path) -> None:
    """global first, then per tier: lightgbm_poisson before catboost_tweedie (cell 57)."""
    from strikecast.pipeline import importance_stage as stage
    from strikecast.store import RunStore

    art = _fake_gbdt_setup(monkeypatch, data)
    store = RunStore(tmp_path / "runs")
    for model in ("lightgbm_poisson", "catboost_tweedie"):
        for paradigm in ("global", "activity"):
            stage.run_importance(count_cfg, model, paradigm, 42, art, store=store,
                                 permutation=False)
    written = stage.collect_importance(store, count_cfg, 42,
                                       models=["lightgbm_poisson", "catboost_tweedie"])
    frame = pd.read_csv(written["importance_all"])
    assert list(dict.fromkeys(frame["model"])) == [
        "global_lightgbm_poisson",
        "global_catboost_tweedie",
        "activity_1_lightgbm_poisson",
        "activity_1_catboost_tweedie",
        "activity_2_lightgbm_poisson",
        "activity_2_catboost_tweedie",
    ]
    assert written["importance_all"].parent.name == "seed=42"
    assert "category_shares" in written
    # the CLI collects the union of every completed run (per-model jobs must not
    # overwrite each other): same file without the explicit model list
    union = stage.collect_importance(store, count_cfg, 42)
    pd.testing.assert_frame_equal(pd.read_csv(union["importance_all"]), frame)


def test_chronos_importance_stage(monkeypatch, tmp_path) -> None:
    from strikecast.config.loader import load_experiment
    from strikecast.pipeline import chronos_stage
    from strikecast.pipeline import importance_stage as stage
    from strikecast.store import RunKey, RunStore

    cfg = load_experiment("chronos2", ["tracking=noop"])
    store = RunStore(tmp_path / "runs")
    key = RunKey("chronos2", "chronos2_fine_tuned", "global", 42)
    from strikecast.pipeline.data_stage import FeatureSets

    art = chronos_stage.ChronosData(
        bundle=None,  # type: ignore[arg-type]
        features=FeatureSets([], [], "all", "f"),
        panel_hash="p",
        series_hash="s",
    )
    art.__dict__["tsdf"] = "FULL_FRAME"  # the cached_property, pre-filled

    with pytest.raises(RuntimeError, match="test stage is not complete"):
        stage.run_importance(cfg, "chronos2_fine_tuned", "global", 42, art, store=store)

    store.start_stage(key, "test", "h")
    store.complete_stage(key, "test")
    predictor = FakeAGPredictor()
    monkeypatch.setattr(chronos_stage, "load_stage_predictor", lambda s, k: predictor)
    out = stage.run_importance(cfg, "chronos2_fine_tuned", "global", 42, art, store=store)
    assert not out.skipped
    assert predictor.calls[0]["data"] == "FULL_FRAME"
    frame = pd.read_csv(out.path, index_col=0)
    assert list(frame.index) == ["env_weather_precipitation_hours", "act_total_damage_events"]

    written = stage.collect_importance(store, cfg, 42)
    assert written["feature_importance_chronos2_fine_tuned"].name == (
        "feature_importance_chronos2_fine_tuned.csv"
    )
    header = written["feature_importance_chronos2_fine_tuned"].read_text().splitlines()[0]
    assert header == ",importance,stdev,n,p99_low,p99_high"  # results/chronos2's format


def test_default_jobs_are_the_thesis_set() -> None:
    from strikecast.pipeline.importance_stage import default_jobs

    assert default_jobs("count") == {
        "lightgbm_poisson": ("global", "activity"),
        "catboost_tweedie": ("global", "activity"),
    }
    assert set(default_jobs("chronos2")) == {"chronos2_fine_tuned", "chronos2_zero_shot"}
    assert default_jobs("diff") == {}


def test_the_cli_knows_the_importance_command() -> None:
    from strikecast.cli import main

    assert "importance" in main.COMMANDS
    args = main._parser().parse_args(
        ["importance", "experiment=count", "--no-permutation", "--n-jobs", "4"]
    )
    assert args.no_permutation and args.n_jobs == 4


# --------------------------------------------------------------------------- #
# 6. parallelism and progress (never a value; cluster 2026-09-29)
# --------------------------------------------------------------------------- #
def test_permutation_workers_fit_next_to_the_model_threads() -> None:
    from strikecast.pipeline.importance_stage import permutation_workers

    # count on a 16-CPU SLURM job: 12 CatBoost threads leave room for ONE worker
    # (it used to be 12 workers x 12 threads = 144 threads on 16 CPUs)
    assert permutation_workers(-1, threads=12, cpus=16) == 1
    assert permutation_workers(None, threads=12, cpus=16) == 1
    assert permutation_workers(0, threads=12, cpus=16) == 1
    assert permutation_workers(-1, threads=12, cpus=64) == 5
    assert permutation_workers(-1, threads=4, cpus=16) == 4
    assert permutation_workers(-1, threads=1, cpus=8) == 8
    assert permutation_workers(-1, threads=24, cpus=16) == 1  # more threads than CPUs
    assert permutation_workers(-1, threads=None, cpus=16) == 1  # unpinned = every CPU
    # an explicit positive n_jobs always wins
    assert permutation_workers(4, threads=12, cpus=16) == 4
    assert permutation_workers(1, threads=None, cpus=64) == 1
    assert permutation_workers(32, threads=12, cpus=16) == 32


def test_available_cpus_prefers_the_affinity_then_slurm(monkeypatch) -> None:
    from strikecast.pipeline import importance_stage as stage

    monkeypatch.setattr(stage.os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert stage.available_cpus() == (16, "sched_getaffinity")

    monkeypatch.delattr(stage.os, "sched_getaffinity", raising=False)  # macOS
    assert stage.available_cpus() == (8, "SLURM_CPUS_PER_TASK")

    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setattr(stage.os, "cpu_count", lambda: 10)
    assert stage.available_cpus() == (10, "os.cpu_count")


def test_model_threads_follow_the_builder(count_cfg) -> None:
    from strikecast.config.loader import load_experiment
    from strikecast.models import get_spec
    from strikecast.pipeline.importance_stage import model_threads

    # every count GBDT pins cfg.threads (12 on the cluster)
    spec = get_spec("catboost_tweedie", "count")
    assert model_threads(count_cfg, spec, "catboost_tweedie") == count_cfg.resolved_threads
    # diff: LightGBM pins 4 threads; GPU XGBoost/CatBoost use the library default
    diff = load_experiment("diff", ["tracking=noop"])
    assert model_threads(diff, get_spec("lightgbm", "diff"), "lightgbm") == 4
    assert model_threads(diff, get_spec("xgboost", "diff"), "xgboost") is None
    assert model_threads(diff, get_spec("catboost", "diff"), "catboost") is None


def test_progress_reports_design_then_every_horizon(data) -> None:
    targets, past, future = data
    model = _lightgbm()
    model.fit(series=targets, past_covariates=past, future_covariates=future)
    events: list[tuple[str, dict]] = []
    got = imp.gbm_importances(model, "lightgbm", targets, past, future, n_jobs=1,
                              progress=lambda event, info: events.append((event, dict(info))))
    silent = imp.gbm_importances(model, "lightgbm", targets, past, future, n_jobs=1)
    pd.testing.assert_frame_equal(got, silent, check_exact=True)
    assert [e for e, _ in events] == ["design", *["horizon"] * OCL]
    design = events[0][1]
    assert design["features"] == len(model.lagged_feature_names) and design["rows"] > 0
    assert [info["horizon"] for _, info in events[1:]] == list(range(1, OCL + 1))
    assert all(info["n_horizons"] == OCL and info["seconds"] >= 0 for _, info in events[1:])


def test_the_stage_records_parallelism_and_mirrors_progress(
    monkeypatch, data, count_cfg, tmp_path
) -> None:
    from strikecast.pipeline import importance_stage as stage
    from strikecast.pipeline.context import NoopTracker
    from strikecast.store import RunStore

    class Recorder(NoopTracker):
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def start(self, run_key, config, tags=(), *, stage="run"):
            self.calls.append(("start", stage))
            return None

        def log_fold(self, step, metrics):
            self.calls.append(("log_fold", step, dict(metrics)))

        def finish(self, status="success"):
            self.calls.append(("finish", status))

    art = _fake_gbdt_setup(monkeypatch, data)
    tracker = Recorder()
    out = stage.run_importance(count_cfg, "lightgbm_poisson", "activity", 42, art,
                               store=RunStore(tmp_path / "runs"), tracker=tracker)
    timings = json.loads((out.path.parent / "timings.json").read_text())
    # the fake spec pins no threads -> treated as using every CPU -> one worker
    assert timings["n_jobs"] == 1 and timings["n_jobs_requested"] == -1
    assert timings["model_threads"] is None and timings["cpus_available"] >= 1
    assert tracker.calls[0] == ("start", "importance")
    assert ("finish", "success") not in tracker.calls  # a passed-in tracker is the caller's
    horizons = [c for c in tracker.calls if c[0] == "log_fold" and "importance/horizon" in c[2]]
    assert [c[1] for c in horizons] == list(range(1, 2 * OCL + 1))  # 2 tiers x OCL horizons
    assert horizons[-1][2]["importance/progress_frac"] == 1.0
    assert horizons[-1][2]["importance/eta_seconds"] == 0.0
    designs = [c for c in tracker.calls if c[0] == "log_fold" and "importance/design_rows" in c[2]]
    assert [c[2]["importance/group_index"] for c in designs] == [1, 2]
