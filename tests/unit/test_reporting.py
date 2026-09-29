"""``strikecast figures`` (WP4): the legacy source reproduces the thesis tables,
figures are byte-deterministic, and a sparse store degrades to skipped items.

Thesis numbers are copied from ``writing/thesis_writing_folder/main.tex`` (cited
by line); every place the thesis disagrees with the stored outputs carries its
audit ID (2026-09-26, C6-C11, D6, D7).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strikecast.reporting import compute as C
from strikecast.reporting import plots as P
from strikecast.reporting.build import _fmt_param, _macro_name, _native_model, build_all
from strikecast.reporting.sources import LegacySource, MissingInput, StoreSource

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DATA = REPO / "data"
HAVE_RESULTS = (RESULTS / "diff" / "leaderboard.csv").is_file()
HAVE_DATA = (DATA / "dataset" / "master_combined_timeseries.parquet").is_file()

needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="results/ not available")
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data/ not available")
# golden/ is git-ignored (absent in CI); the tuning table reads the converted studies.
HAVE_GOLDEN = (REPO / "golden" / "converted").is_dir()
# F2/F3 (choropleths) need the optional `figures` extra (geopandas).
try:
    import geopandas  # noqa: F401
    HAVE_GEOPANDAS = True
except ImportError:
    HAVE_GEOPANDAS = False


# --------------------------------------------------------------------------- #
# the legacy build, once
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def legacy_out(tmp_path_factory) -> Path:
    if not (HAVE_RESULTS and HAVE_DATA):
        pytest.skip("results/ or data/ not available")
    out = tmp_path_factory.mktemp("figs_legacy")
    items = build_all(
        LegacySource(RESULTS, REPO / "golden", REPO / "_regression_GBDT.ipynb"),
        out,
        data_dir=DATA,
    )
    failed = {i.id: i.reason for i in items if i.status == "failed"}
    assert not failed, failed
    return out


def _csv(out: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out / f"{name}.csv")


def test_legacy_every_item_generated_or_static(legacy_out: Path) -> None:
    manifest = json.loads((legacy_out / "MANIFEST.json").read_text())
    by_status: dict[str, list[str]] = {}
    for item in manifest["items"]:
        by_status.setdefault(item["status"], []).append(item["id"])
    if not HAVE_GEOPANDAS:  # the maps are skipped with a reason, not failed
        assert set(by_status.pop("skipped", [])) <= {"F2", "F3"}, by_status
    assert set(by_status) <= {"generated", "static"}, by_status
    assert sorted(by_status["static"]) == sorted(
        ["F1", "F8", "F9", "F10", "F11", "F22", "T1", "T7", "T11"]
    )
    for item in manifest["items"]:
        for name in item["outputs"] if item["status"] == "generated" else []:
            assert (legacy_out / name).is_file(), name
    assert (legacy_out / "MANIFEST.md").is_file()


#: main.tex:1549-1568 (appendix Top 20) with rank 20 restored: the thesis drops
#: diff/global/linear and lists rank 21 instead; the F and H statistics only
#: reproduce WITH it (C9, D7: master_df[:20]).
THESIS_TOP20 = [
    ("gbdt", "activity", "catboost_tweedie_tuned", 0.169634, 0.767909, 1.830539),
    ("chronos2", "local", "chronos2_fine_tuned", 0.169486, 0.713611, 1.830864),
    ("gbdt", "global", "catboost_tweedie_tuned", 0.169411, 0.785818, 1.831029),
    ("gbdt", "global", "lightgbm_poisson_tuned", 0.165593, 0.775707, 1.839445),
    ("lstm", "activity", "lstm_poisson_w28_tuned", 0.165131, 0.777548, 1.840464),
    ("chronos2", "local", "chronos2_zero_shot", 0.161110, 0.719448, 1.849330),
    ("diff", "global", "arima", 0.159221, 0.813287, 1.853493),
    ("gbdt", "activity", "lightgbm_poisson_tuned", 0.159210, 0.766848, 1.853518),
    ("gbdt", "activity", "catboost_poisson_tuned", 0.153791, 0.779569, 1.865464),
    ("gbdt", "global", "xgboost_tweedie_tuned", 0.152136, 0.778661, 1.869111),
    ("lstm", "activity", "gru_poisson_w28_tuned", 0.151309, 0.782510, 1.870936),
    ("gbdt", "activity", "xgboost_tweedie_tuned", 0.150523, 0.764992, 1.872668),
    ("lstm", "activity", "gru_poisson_w14_tuned", 0.149851, 0.781645, 1.874149),
    ("gbdt", "global", "xgboost_poisson_tuned", 0.149798, 0.790820, 1.874267),
    ("gbdt", "activity", "xgboost_poisson_tuned", 0.148924, 0.774119, 1.876192),
    ("gbdt", "local", "lightgbm_poisson_tuned", 0.147808, 0.783284, 1.878654),
    ("gbdt", "global", "catboost_poisson_tuned", 0.145142, 0.804140, 1.884530),
    ("lstm", "activity", "lstm_tweedie_w28_tuned", 0.144459, 0.792681, 1.886036),
    ("lstm", "global", "gru_poisson_w14_tuned", 0.144243, 0.799084, 1.886513),
    ("diff", "global", "linear", 0.143345, 0.875316, 1.888492),  # C9: missing in the thesis
]


def test_top20_matches_thesis_appendix(legacy_out: Path) -> None:
    top = _csv(legacy_out, "tab_top20")
    assert len(top) == 20
    for row, want in zip(top.itertuples(index=False), THESIS_TOP20, strict=True):
        assert (row.Modelname, row.paradigm, row.model) == want[:3]
        np.testing.assert_allclose([row.SkillScore, row.mae, row.rmse], want[3:], atol=5e-7)
    tex = (legacy_out / "tab_top20.tex").read_text()
    assert r"gbdt & activity & catboost\_tweedie\_tuned & 0.169634 & 0.767909 & 1.830539" in tex


#: main.tex:905-915 (tab:overall_performance): MAE, RMSE, Skill Score at 2 dp.
THESIS_TABLE4 = {
    "CatBoost": (0.77, 1.83, 0.17), "LightGBM": (0.78, 1.84, 0.17),
    "XGBoost": (0.78, 1.87, 0.15), "LSTM": (0.78, 1.84, 0.17), "GRU": (0.78, 1.87, 0.15),
    "Chronos-2-FT": (0.71, 1.83, 0.17), "Chronos-2-OS": (0.72, 1.85, 0.16),
    "ARIMA": (0.81, 1.85, 0.16), "Hurdle": (0.79, 2.00, 0.09),
    "Seasonal Naive": (0.89, 2.20, 0.00), "Naive": (1.03, 2.58, -0.17),
}


def test_table4_matches_thesis_except_the_hurdle(legacy_out: Path) -> None:
    t = _csv(legacy_out, "tab_overall_performance").set_index("Model")
    assert list(t.index) == list(THESIS_TABLE4)
    for model, (mae, rmse, skill) in THESIS_TABLE4.items():
        got = (round(t.loc[model, "MAE"], 2), round(t.loc[model, "RMSE"], 2),
               round(t.loc[model, "Skill Score"], 2))
        if model == "Hurdle":
            # D6/C7: the canonical hurdle row is hurdle_cal from the stored
            # predictions (0.80/2.02/0.09); the thesis printed 0.79/2.00 from
            # finalhurdle/leaderboard.csv (a session that no code reproduces).
            assert got == (0.80, 2.02, 0.09)
            continue
        assert got == (mae, rmse, skill), model


def test_top_models_matches_thesis(legacy_out: Path) -> None:
    t = _csv(legacy_out, "tab_top_models")
    # main.tex:929-935
    assert [round(x, 3) for x in t["SkillScore"]] == [0.170, 0.169, 0.169, 0.166, 0.165]
    assert [round(x, 2) for x in t["RMSE"]] == [1.83, 1.83, 1.83, 1.84, 1.84]
    # C8: rank 1 MAE is 0.767909 -> 0.77 (Table 4 prints 0.77, tab:top_models 0.78)
    assert [round(x, 2) for x in t["MAE"]] == [0.77, 0.71, 0.79, 0.78, 0.78]
    assert list(t["Model"]) == ["CatBoost (Tweedie)", "Chronos-2 (Fine-tuned)",
                                "CatBoost (Tweedie)", "LightGBM (Poisson)",
                                "LSTM (Poisson, $w=28$)"]


def test_horizon_statistics_match_thesis(legacy_out: Path) -> None:
    s = _csv(legacy_out, "tab_horizon_stats").set_index("metric")
    # main.tex:1084-1087 RMSE: Shapiro p ~ 0.41, Levene p ~ 0.54, F = 6.70, p < 0.001
    assert round(s.loc["RMSE", "shapiro_reported_p"], 2) == 0.41
    assert round(s.loc["RMSE", "levene_p"], 2) == 0.55  # printed 0.54 (0.5461, truncated)
    assert round(s.loc["RMSE", "anova_F"], 4) == 6.7041
    assert s.loc["RMSE", "anova_p"] < 1e-3
    assert s.loc["RMSE", "test"] == "anova"
    # main.tex:1079-1081 MAE: Shapiro p < 0.05, H = 5.49; C10: p = 0.483, not ~0.05
    assert s.loc["MAE", "shapiro_reported_p"] < 0.05
    assert round(s.loc["MAE", "kruskal_H"], 4) == 5.4895
    assert round(s.loc["MAE", "kruskal_p"], 3) == 0.483
    tk = _csv(legacy_out, "tab_horizon_tukey")
    p = {(int(a), int(b)): round(float(v), 3)
         for a, b, v in zip(tk["group1"], tk["group2"], tk["p-adj"], strict=True)}
    # main.tex:1088-1093
    assert p[(3, 5)] == 0.008
    assert p[(1, 5)] == 0.429 and p[(2, 5)] == 0.129 and p[(4, 5)] == 0.478
    assert p[(1, 6)] == 0.027 and p[(2, 6)] <= 0.004 and p[(3, 6)] < 0.001 and p[(4, 6)] == 0.034


def test_friedman_ranks_and_cd_match_thesis(legacy_out: Path) -> None:
    t = _csv(legacy_out, "tab_friedman_top5").set_index("model")
    # main.tex:972-980
    rmse = t["RMSE avg rank"].round(2).to_dict()
    assert rmse == {"CatBoost (Tw, Act)": 2.65, "LightGBM (Po, Glb)": 2.75, "LSTM (Po, Act)": 3.14,
                    "CatBoost (Tw, Glb)": 3.34, "ARIMA": 4.51, "Chronos-2 (FT)": 4.62}
    mae = t["MAE avg rank"].round(2).to_dict()
    assert mae["Chronos-2 (FT)"] == 1.59 and mae["LSTM (Po, Act)"] == 2.89
    assert mae["CatBoost (Tw, Act)"] == 2.91 and mae["ARIMA"] == 4.81
    assert round(t["RMSE CD"].iloc[0], 2) == 0.64
    assert (t["RMSE p"] < 1e-3).all() and (t["MAE p"] < 1e-3).all()


def test_hurdle_bias_counts_match_values_recomputed(legacy_out: Path) -> None:
    t = _csv(legacy_out, "tab_hurdle_bias").head(8)
    # main.tex:1046-1053: regions, counts and maxima reproduce
    assert list(t["Region"]) == ["sumy", "zaporizhia", "chernihiv", "donetsk", "dnipropetrovsk",
                                 "kharkiv", "kherson", "kyiv"]
    assert list(t["Count (Days > 0)"]) == [1093, 640, 962, 270, 803, 846, 987, 324]
    assert list(t["Max Actual Strikes"]) == [37.0, 12.0, 11.0, 11.0, 7.0, 8.0, 5.0, 6.0]
    # C6/D6: RMSE/MAE/bias are the stored-prediction values, not the thesis' 7.35/4.83/-4.17
    assert [round(x, 2) for x in t.iloc[0][["RMSE", "MAE", "Bias (Mean Error)"]]] == [
        6.79, 4.48, -3.13
    ]
    assert (t["Bias (Mean Error)"] < 0).all()


def test_classifier_numbers_match_thesis(legacy_out: Path) -> None:
    n = json.loads((legacy_out / "numbers.json").read_text())
    # main.tex:1016, 1027
    assert round(n["PRAUCUncal"], 3) == 0.817 and round(n["PRAUCCal"], 3) == 0.817
    assert round(n["PrevalenceTest"], 3) == 0.294
    reg = _csv(legacy_out, "tab_prauc_per_region")
    top = reg.sort_values("Delta", ascending=False).head(4)
    # main.tex:1031 says "0.06-0.10"; the data give 0.057-0.108 (audit C F17)
    assert round(top["Delta"].max(), 3) == 0.108 and round(top["Delta"].min(), 3) == 0.057


def test_data_tables_match_thesis(legacy_out: Path) -> None:
    split = _csv(legacy_out, "tab_splitdimensions")
    # C11: the thesis prints 592/85/169 = 846 days and 16,920 observations
    assert list(split["Days / region"]) == [593, 85, 169, 847]
    assert list(split["Observations"]) == [11860, 1700, 3380, 16940]
    tiers = _csv(legacy_out, "tab_activitytiers").set_index("Tier")
    # main.tex:306-309: 7 / 84 / 219 / 411
    assert [tiers.loc[t, "min_days"] for t in (1, 2, 3)] == [7, 84, 219]
    assert tiers.loc[3, "max_days"] == 411
    assert list(tiers["n_regions"]) == [5, 10, 4, 6]  # 20 modelled oblasts
    assert len(_csv(legacy_out, "tab_base_variables_rows")) == 112  # tab:base_variables
    n = json.loads((legacy_out / "numbers.json").read_text())
    # main.tex:426ff: D = 1.15 / 2.03 / 5.16, Tier-1 zero rate 0.98
    assert [round(n[f"DispersionTier{t}"], 2) for t in (1, 2, 3)] == [1.15, 2.03, 5.16]
    assert round(n["ZeroRateTier1"], 2) == 0.98
    # fig:STLDECOMP panel titles: F_T 0.73 -> 0.09, F_S 0.38 -> 0.36
    assert (round(n["STLFTraw"], 2), round(n["STLFTdiff"], 2)) == (0.73, 0.09)
    assert (round(n["STLFSraw"], 2), round(n["STLFSdiff"], 2)) == (0.38, 0.36)


@pytest.mark.skipif(not HAVE_GOLDEN, reason="golden/ not available (git-ignored)")
def test_tuning_best_all_matches_thesis(legacy_out: Path) -> None:
    tex = (legacy_out / "tab_tuning_best_all.tex").read_text()
    # main.tex:1418-1452, value for value
    for param, value in [
        ("depth", "4"), ("learning\\_rate", "0.01207"), ("iterations", "900"),
        ("l2\\_leaf\\_reg", "5.0"), ("subsample", "0.998"), ("tweedie\\_variance\\_power", "1.395"),
        ("num\\_leaves", "15"), ("max\\_depth", "8"), ("min\\_child\\_samples", "168"),
        ("learning\\_rate", "0.01221"), ("n\\_estimators", "700"), ("subsample", "0.797"),
        ("colsample\\_bytree", "0.826"), ("reg\\_alpha", "0.104"), ("reg\\_lambda", "0.642"),
        ("hidden\\_dim", "64"), ("n\\_rnn\\_layers", "1"), ("hidden\\_fc\\_sizes", "$[64]$"),
        ("dropout", "0.251"), ("batch\\_size", "32"), ("lr", r"$1.081 \times 10^{-3}$"),
        ("weight\\_decay", r"$4.960 \times 10^{-4}$"),
        ("fine\\_tune\\_lr", r"$8.72 \times 10^{-5}$"), ("fine\\_tune\\_steps", "1500"),
    ]:
        assert rf"\texttt{{{param}}} & {value} \\" in tex, (param, value)


def test_importance_share_reproduces_thesis_file(legacy_out: Path) -> None:
    """F20 from the thesis results: the leaderboard picks Activity CatBoost-Tweedie (rank 1)
    and the fine-tuned Chronos-2 (rank 2); the tier columns are the thesis numbers."""
    mat = _csv(legacy_out, "Feature-importancesharebycategory_grouped").set_index("category")
    tiers = [f"catboost_tweedie_activity Tier {t} {m}" for t in (1, 2, 3) for m in ("gain", "perm")]
    assert list(mat.columns) == [*tiers, "chronos2_fine_tuned_local perm"]
    np.testing.assert_allclose(mat.sum(axis=0).to_numpy(), 1.0)
    # main.tex:1120-1122: tier 1 conflict 62 % gain / 48 % perm, tier 3 autoregressive 82 % / 66 %
    conflict = mat.loc["Conflict & damage"]
    assert round(conflict[tiers[0]], 2) == 0.62 and round(conflict[tiers[1]], 2) == 0.48
    ar = mat.loc["Autoregressive strikes"]
    assert round(ar[tiers[4]], 2) == 0.82 and round(ar[tiers[5]], 2) == 0.66


def test_figures_same_size_as_thesis_svgs(legacy_out: Path) -> None:
    """Sizes of figures whose thesis copy is the current code's output (C5: DejaVu Sans)."""
    thesis_fig = Path("/Users/jan/projects/writing/thesis_writing_folder/fig")
    if not thesis_fig.is_dir():
        pytest.skip("thesis fig/ folder not available")
    import re

    # Feature-importancesharebycategory_grouped.svg is left out on purpose: since
    # 2026-09-29 it adds the Chronos-2 column to the thesis' three tier facets.
    same = ["top_5_combined.svg", "calibration_prcurve.svg", "prauc_vs_prevalence.svg",
            "top20_rmse_horizon.svg", "Chronos2LocalFeatureImportance.svg", "Top15FeatureImportancesperActivityLevel.svg",
            "fig_eda2c_stl_side_by_side.svg", "spatiotemporalintensityheatmap.svg",
            "strike_activity_per_region.svg", "strike_activity_per_region_activity_level.svg"]
    size = re.compile(r'<svg [^>]*?width="([^"]+)" height="([^"]+)"', re.DOTALL)
    for name in same:
        ours = size.search((legacy_out / name).read_text()).groups()
        theirs = size.search((thesis_fig / name).read_text()).groups()
        assert ours == theirs, name


# --------------------------------------------------------------------------- #
# determinism (C5)
# --------------------------------------------------------------------------- #
def _synthetic_per_horizon(seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return {
        f"m{i}_global": pd.DataFrame(
            {"horizon": range(1, 8), "RMSE": rng.random(7) + 1.8, "MAE": rng.random(7) / 5 + 0.7}
        )
        for i in range(5)
    }


def test_figure_bytes_are_deterministic_and_scoped(tmp_path: Path) -> None:
    import matplotlib as mpl

    from strikecast.reporting.style import save_svg, thesis_style

    before = dict(mpl.rcParams)
    per = _synthetic_per_horizon(0)
    rows = pd.DataFrame([{"model": k.rsplit("_", 1)[0], "paradigm": "global"} for k in per])
    blobs = []
    for run in range(2):
        with thesis_style():
            fig = P.top5_horizon_lines(per, P.model_colors(rows))
            blobs.append(save_svg(fig, tmp_path / f"f{run}.svg").read_bytes())
    assert blobs[0] == blobs[1]
    text = blobs[0].decode()
    assert "<dc:date>" not in text
    assert "DejaVu" in text or "DejaVuSans" in text
    after = dict(mpl.rcParams)
    assert {k: v for k, v in after.items() if before.get(k) != v} == {}


# --------------------------------------------------------------------------- #
# sparse / empty store
# --------------------------------------------------------------------------- #
def test_empty_store_skips_every_results_item(tmp_path: Path) -> None:
    store = tmp_path / "runs"
    store.mkdir()
    items = build_all(StoreSource(store), tmp_path / "out", data_dir=tmp_path / "nodata")
    status = {i.id: i.status for i in items}
    assert "failed" not in status.values(), {i.id: i.reason for i in items if i.status == "failed"}
    for i in items:
        if i.klass in ("RESULTS", "DATA", "EXTRA"):
            assert i.status == "skipped", i.id
            assert i.reason
    manifest = json.loads((tmp_path / "out" / "MANIFEST.json").read_text())
    assert len(manifest["items"]) == len(items)


@pytest.mark.skipif(not (REPO / "runs" / "diff").is_dir(), reason="no local runs/ store")
@needs_data
def test_real_sparse_store_does_not_crash(tmp_path: Path) -> None:
    items = build_all(StoreSource(REPO / "runs"), tmp_path, data_dir=DATA,
                      only=["T3", "T4", "T5", "X1", "F12", "F16", "F19", "F20"])
    status = {i.id: i.status for i in items}
    assert "failed" not in status.values(), {i.id: i.reason for i in items if i.status == "failed"}
    assert status["T3"] == "generated" and status["X1"] == "generated"
    assert status["F16"] == "skipped" and status["F20"] == "skipped"


def _mini_store(root: Path) -> None:
    """diff naive_weekly + arima and count lightgbm_poisson, seed 42, test stage only."""
    from strikecast.backtest.predictions import COLUMNS
    from strikecast.store.run_store import RunKey, RunStore

    store = RunStore(root)
    rng = np.random.default_rng(1)
    regions = ["kyiv", "sumy"]
    for exp, model, bias in [("diff", "naive_weekly", 0.9), ("diff", "arima", 0.4),
                             ("count", "lightgbm_poisson", 0.2), ("hurdle", "hurdle", 0.3)]:
        key = RunKey(exp, model, "global", 42)
        rows = []
        for r in regions:
            for f in range(4):
                for h in range(1, 8):
                    y = float(rng.poisson(1.0))
                    date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=f * 7 + h)
                    base = {"region": r, "fold": f, "horizon": h, "date": date, "y_true": y,
                            "origin_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=f * 7)}
                    if exp == "hurdle":
                        rows.append({**base, "y_true": float(y > 0), "y_pred": rng.random(),
                                     "channel": "prob"})
                        rows.append({**base, "y_pred": y + bias, "channel": "count"})
                        rows.append({**base, "y_pred": y, "channel": "hurdle"})
                    else:
                        rows.append({**base, "y_pred": y + bias * rng.standard_normal(),
                                     "channel": "y_pred"})
        frame = pd.DataFrame(rows)[list(COLUMNS)]
        pred_dir = store.predictions_dir(key, "test")
        pred_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(pred_dir / "part-000000-000003.parquet")
        metrics = store.metrics_dir(key, "test")
        metrics.mkdir(parents=True, exist_ok=True)
        main = frame[frame["channel"] != "prob"] if exp == "hurdle" else frame
        e = main["y_pred"] - main["y_true"]
        (metrics / "global.json").write_text(json.dumps(
            {"MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e**2).mean()))}))
        if exp == "hurdle":
            from strikecast.evaluation.calibration import (
                calibrators_to_json,
                fit_calibrators_per_horizon,
            )

            prob = frame[frame["channel"] == "prob"].rename(columns={"y_pred": "y_prob"})
            cal = fit_calibrators_per_horizon(prob.reset_index(drop=True), method="sigmoid")
            store.write_artifact(key, "calibrators.json", {"prob": calibrators_to_json(cal)})


def test_store_source_maps_rows_and_calibrates(tmp_path: Path) -> None:
    _mini_store(tmp_path)
    src = StoreSource(tmp_path)
    lb = src.leaderboards()
    assert set(lb["Modelname"]) == {"diff", "gbdt", "finalhurdle"}
    master = C.master_leaderboard(lb)
    ref = lb.loc[lb["model"] == "naive_weekly", "RMSE"].item()
    np.testing.assert_allclose(master["SkillScore"], 1 - master["rmse"] / ref)
    # the cell-11 `local` relabel of the seasonal naive maps back to the store's global run
    assert not src.predictions("diff", "local", "naive_weekly").empty
    probs = src.hurdle_classifier_probs()
    assert list(probs.columns[-2:]) == ["y_prob_uncal", "y_prob_cal"]
    assert not np.allclose(probs["y_prob_uncal"], probs["y_prob_cal"])
    hurdle = src.predictions("finalhurdle", "global", "finalhurdle")
    count = src.hurdle_regressor_predictions()
    np.testing.assert_allclose(hurdle["y_pred"], probs["y_prob_cal"] * count["y_pred"])
    with pytest.raises(MissingInput):
        src.gbdt_importance()


def test_cli_figures_on_a_store(tmp_path: Path) -> None:
    from strikecast.cli.main import main

    _mini_store(tmp_path / "runs")
    rc = main(["figures", "--store-root", str(tmp_path / "runs"), "--data-dir",
               str(tmp_path / "nodata"), "--only", "X1,T5"])
    assert rc == 0
    out = tmp_path / "runs" / "_figures"
    assert (out / "master_leaderboard.csv").is_file() and (out / "tab_top_models.tex").is_file()
    manifest = json.loads((out / "MANIFEST.json").read_text())
    status = {i["id"]: i["status"] for i in manifest["items"]}
    assert status["X1"] == "generated" and status["F12"] == "not-requested"


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #
def test_names_and_formats() -> None:
    assert C.pretty_model("catboost_tweedie_tuned") == "CatBoost (Tweedie)"
    assert C.pretty_model("lstm_poisson_w28") == "LSTM (Poisson, $w=28$)"
    assert C.pretty_model("chronos2_fine_tuned") == "Chronos-2 (Fine-tuned)"
    assert P.cd_label({"model": "lightgbm_poisson", "paradigm": "global"}) == "LightGBM (Po, Glb)"
    assert P.cd_label({"model": "lstm_poisson_w28_tuned", "paradigm": "activity"}) == "LSTM (Po, Act)"
    assert _macro_name("DispersionTier1") == "DispersionTierOne"
    assert _fmt_param("learning_rate", 0.012071584113760261) == "0.01207"
    assert _fmt_param("subsample", 0.9977756396798295) == "0.998"
    assert _fmt_param("lr", 0.0010814406628766157) == r"$1.081 \times 10^{-3}$"
    assert _fmt_param("hidden_fc_sizes", "64") == "$[64]$"


def test_native_model_keeps_chronos_fine_tuned_store_name() -> None:
    """The store runs Chronos-2 as ``chronos2_fine_tuned``; stripping ``_tuned`` there
    dropped the Chronos-2-FT row of Table 4 in the first publication run."""
    store = type("S", (), {"name": "store"})()
    legacy = type("L", (), {"name": "legacy"})()
    assert _native_model(store, "chronos2_fine_tuned") == "chronos2_fine_tuned"
    assert _native_model(store, "catboost_tweedie_tuned") == "catboost_tweedie"
    assert _native_model(store, "finalhurdle") == "hurdle"
    assert _native_model(legacy, "chronos2_fine_tuned") == "chronos2_fine_tuned"


def test_horizon_statistics_routing() -> None:
    per = list(_synthetic_per_horizon(3).values())
    h = C.horizon_statistics(per, "RMSE")
    assert h.test in ("anova", "kruskal")
    assert set(h.summary()) >= {"anova_F", "kruskal_H", "levene_p", "shapiro_reported_p"}
    assert h.n_models == 5


def test_split_dimensions_is_the_darts_split() -> None:
    t = C.split_dimensions(847, 20)
    assert list(t["Days / region"]) == [593, 85, 169, 847]


def test_legacy_source_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(MissingInput):
        LegacySource(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# F20/F21/F23: the two leading models of the leaderboard (2026-09-29)
# --------------------------------------------------------------------------- #
class _FISource(LegacySource):
    """A leaderboard plus importance frames, nothing else."""

    name = "store"

    def __init__(self, rows: list[tuple[str, str, str, float]], labels: list[str]) -> None:
        self.root, self.seed = Path("/runs"), 42
        self._lb = pd.DataFrame(rows, columns=["Modelname", "paradigm", "model", "RMSE"])
        self._lb["MAE"] = self._lb["RMSE"] / 2
        self._labels = labels

    def leaderboards(self) -> pd.DataFrame:
        return self._lb

    def gbdt_importance(self) -> pd.DataFrame:
        feats = ["act_drone_strike_on_ua_target_lag-1", "env_weather_rain_sum_futcov_lag0",
                 "acled_other_rus_armed_clash_pastcov_lag-1"]
        return pd.DataFrame([{"model": lab, "Feature": f, "agg_gain": 3.0 - i, "agg_perm": 1.0 + i}
                             for lab in self._labels for i, f in enumerate(feats)])

    def chronos_importance(self, model: str = "chronos2_fine_tuned") -> pd.DataFrame:
        return pd.DataFrame({"Feature": [f"{model}_env_weather_x", "com_diplo_y"],
                             "importance": [0.3, 0.1]})


_BASE_ROWS = [("diff", "global", "naive_weekly", 2.2), ("diff", "global", "arima", 1.85),
              ("chronos2", "local", "chronos2_zero_shot", 1.83),
              ("chronos2", "local", "chronos2_fine_tuned", 1.84),
              ("gbdt", "local", "lightgbm_poisson", 1.70),  # Local is outside the pool
              ("lstm", "activity", "lstm_poisson_w28", 1.75)]


def _fi_build(tmp_path: Path, rows, labels) -> dict[str, object]:
    items = build_all(_FISource(rows, labels), tmp_path, data_dir=tmp_path / "nodata",
                      only=["F20", "F21", "F23"])
    return {i.id: i for i in items}


def test_importance_figures_follow_the_leaderboard(tmp_path: Path) -> None:
    rows = [*_BASE_ROWS, ("gbdt", "global", "catboost_tweedie", 1.80),
            ("gbdt", "activity", "lightgbm_poisson", 1.81)]
    items = _fi_build(tmp_path, rows, ["global_catboost_tweedie"])
    assert {k: items[k].status for k in ("F20", "F21", "F23")} == dict.fromkeys(
        ("F20", "F21", "F23"), "generated")
    mat = pd.read_csv(tmp_path / "Feature-importancesharebycategory_grouped.csv").set_index("category")
    # best Global/Activity GBDT (not the Local one, not the RNN) + best Chronos-2 (zero-shot)
    assert list(mat.columns) == ["catboost_tweedie_global gain", "catboost_tweedie_global perm",
                                 "chronos2_zero_shot_local perm"]
    np.testing.assert_allclose(mat.sum(axis=0).to_numpy(), 1.0)
    assert "catboost_tweedie_global (rank 3), chronos2_zero_shot_local (rank 5)" in items["F20"].notes[0]


def test_activity_gbdt_gets_one_facet_per_tier(tmp_path: Path) -> None:
    rows = [*_BASE_ROWS, ("gbdt", "activity", "catboost_tweedie_tuned", 1.79)]
    labels = [f"activity_{t}_catboost_tweedie" for t in (1, 2, 3)]
    _fi_build(tmp_path, rows, labels)
    mat = pd.read_csv(tmp_path / "Feature-importancesharebycategory_grouped.csv").set_index("category")
    assert list(mat.columns) == [f"catboost_tweedie_activity Tier {t} {m}"
                                 for t in (1, 2, 3) for m in ("gain", "perm")] + [
        "chronos2_zero_shot_local perm"]


def test_missing_importance_names_the_job(tmp_path: Path) -> None:
    rows = [*_BASE_ROWS, ("gbdt", "global", "xgboost_poisson", 1.80)]
    items = _fi_build(tmp_path, rows, ["global_catboost_tweedie"])
    for item_id in ("F20", "F23"):
        assert items[item_id].status == "skipped"
        assert ("strikecast importance experiment=count model=xgboost_poisson paradigm=global "
                "seed=42 --store-root /runs") in items[item_id].reason
    assert items["F21"].status == "generated"  # the Chronos-2 figure does not need the GBDT
