"""Unit tests for the Chronos-2 adapter (Stream H, plan §8 P5).

Everything here runs in the MAIN environment, where ``autogluon`` cannot be
installed (plan §5.7).  Three things make that possible:

* every ``autogluon`` import in :mod:`strikecast.data.autogluon` and
  :mod:`strikecast.models.chronos` is inside a function;
* :class:`~strikecast.data.autogluon.LongPanel` re-implements exactly the
  ``TimeSeriesDataFrame`` members the Chronos code path uses, so the frame
  construction, the slicing and the naive scales are all exercisable;
* :class:`~strikecast.models.chronos.Chronos2Forecaster` takes the predictor as
  a constructor argument, so a fake one drives the real backtest engine.

The only thing that needs AutoGluon is ``TimeSeriesDataFrame`` itself; the two
tests that would build one are skipped when it is absent.  The real comparison
against ``golden/results/chronos2/`` lives in
``tests/golden/test_chronos_equality.py`` and runs in ``envs/autogluon``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from strikecast.backtest.predictions import COLUMNS, LEGACY_COLUMNS, PredictionSet
from strikecast.backtest.protocols import SINGLE_CHANNEL
from strikecast.data import autogluon as ag
from strikecast.models import chronos
from strikecast.models.spec import RunContext, get_spec, registered_names


def _has_autogluon() -> bool:
    try:
        return importlib.util.find_spec("autogluon.timeseries") is not None
    except ModuleNotFoundError:
        return False


HAS_AUTOGLUON = _has_autogluon()

REGIONS = ("alpha", "bravo", "charlie")
N_STEPS = 60
TARGET = "act_drone_strike_on_ua"
FUTURE_COVS = ("env_ua_holiday", "env_weather_temperature_2m_mean")
PAST_COVS = ("com_actor_a", "acled_other_ua_armed_clash")
HORIZON = 7


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel() -> pd.DataFrame:
    """A synthetic Chronos panel: MultiIndexed, with the stray ``index`` column.

    Shaped exactly like ``build_panel_legacy_chronos``'s output -- a
    ``(region, event_date)`` MultiIndex, an ``Activity_Level`` column, the
    target, future and past covariates, and the stray ``index`` column of flag
    F22/F123.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-09-28", periods=N_STEPS, freq="D")
    rows = []
    for r_idx, region in enumerate(REGIONS):
        for t, date in enumerate(dates):
            row = {
                "region": region,
                "event_date": date,
                "Activity_Level": r_idx + 1,
                "index": r_idx * N_STEPS + t,  # F123: the stray column
                TARGET: float(rng.poisson(0.5 + r_idx)),
            }
            for c in FUTURE_COVS:
                row[c] = float(rng.normal())
            for c in PAST_COVS:
                row[c] = float(rng.normal())
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.set_index(["region", "event_date"])


@pytest.fixture
def frames(panel: pd.DataFrame) -> ag.LongFrames:
    return ag.panel_to_long_frames(panel)


class FakePredictor:
    """Stands in for ``TimeSeriesPredictor``: a deterministic 0.5 quantile.

    Records every call so the tests can assert on the exact slices the
    forecaster hands AutoGluon.  The prediction for item ``i`` at step ``h`` is
    ``last_context_value + h``, which is enough to tell folds apart.
    """

    def __init__(self, horizon: int = HORIZON, target: str = TARGET) -> None:
        self.horizon = horizon
        self.target = target
        self.calls: list[dict[str, object]] = []

    def predict(self, data, known_covariates=None, use_cache=True):  # noqa: ANN001
        self.calls.append(
            {
                "context_len": int(data.num_timesteps_per_item().iloc[0]),
                "context_end": data.loc[REGIONS[0]].index[-1],
                "known_cols": None if known_covariates is None else list(known_covariates.columns),
                "known_start": None
                if known_covariates is None
                else known_covariates.loc[REGIONS[0]].index[0],
                "known_len": None
                if known_covariates is None
                else int(known_covariates.num_timesteps_per_item().iloc[0]),
                "use_cache": use_cache,
            }
        )
        blocks = []
        for region in data.item_ids:
            sub = data.loc[region]
            last_date = sub.index[-1]
            last_value = float(sub[self.target].to_numpy()[-1])
            index = pd.date_range(
                last_date + pd.Timedelta(days=1), periods=self.horizon, freq="D"
            )
            block = pd.DataFrame(
                {
                    "mean": [last_value + h for h in range(1, self.horizon + 1)],
                    "0.5": [last_value + h for h in range(1, self.horizon + 1)],
                },
                index=pd.MultiIndex.from_product(
                    [[region], index], names=[ag.ITEM_ID, ag.TIMESTAMP]
                ),
            )
            blocks.append(block)
        return ag.LongPanel(pd.concat(blocks))


@pytest.fixture
def forecaster(frames: ag.LongFrames) -> chronos.Chronos2Forecaster:
    return chronos.Chronos2Forecaster(
        FakePredictor(),
        frames,
        target=TARGET,
        known_covariates=list(FUTURE_COVS),
        region_names=list(REGIONS),
        horizon=HORIZON,
    )


# --------------------------------------------------------------------------- #
# 1. the forward adapter
# --------------------------------------------------------------------------- #
def test_panel_becomes_the_item_id_timestamp_frame(frames: ag.LongFrames) -> None:
    assert list(frames.values.index.names) == [ag.ITEM_ID, ag.TIMESTAMP]
    assert frames.item_ids == list(REGIONS)
    assert len(frames.values) == len(REGIONS) * N_STEPS


def test_activity_level_moves_to_the_static_frame(frames: ag.LongFrames) -> None:
    assert ag.STATIC_COL not in frames.values.columns
    assert list(frames.static.columns) == [ag.STATIC_COL]
    assert frames.static.index.name == ag.ITEM_ID
    assert frames.static.loc["bravo", ag.STATIC_COL] == 2


def test_the_stray_index_column_survives_as_a_covariate(frames: ag.LongFrames) -> None:
    """F123: `_chronos2.py` §3 keeps every column, `exclude_cols` never applies."""
    assert "index" in frames.values.columns
    expected = 1 + len(FUTURE_COVS) + len(PAST_COVS) + 1  # target + covs + `index`
    assert frames.values.shape[1] == expected


def test_drop_columns_is_not_the_legacy_path(panel: pd.DataFrame) -> None:
    dropped = ag.panel_to_long_frames(panel, drop_columns=["index"])
    assert "index" not in dropped.values.columns


def test_a_reset_panel_gives_the_same_frame(panel: pd.DataFrame, frames: ag.LongFrames) -> None:
    from_reset = ag.panel_to_long_frames(panel.reset_index())
    pd.testing.assert_frame_equal(from_reset.values, frames.values)
    pd.testing.assert_frame_equal(from_reset.static, frames.static)


def test_a_panel_without_activity_level_is_a_loud_error(panel: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Activity_Level"):
        ag.panel_to_long_frames(panel.drop(columns=[ag.STATIC_COL]))


def test_frames_match_the_verbatim_legacy_construction(panel: pd.DataFrame) -> None:
    """`_chronos2.py:202-218`, inlined here as the oracle."""
    ts_df = (
        panel.reset_index()
        .rename(columns={"region": "item_id", "event_date": "timestamp"})
        .sort_values(["item_id", "timestamp"])
        .set_index(["item_id", "timestamp"])
    )
    static_df = (
        ts_df.reset_index()[["item_id", "Activity_Level"]]
        .drop_duplicates()
        .set_index("item_id")
    )
    ts_df = ts_df.drop(columns=["Activity_Level"])

    got = ag.panel_to_long_frames(panel)
    pd.testing.assert_frame_equal(got.values, ts_df)
    pd.testing.assert_frame_equal(got.static, static_df)


# --------------------------------------------------------------------------- #
# 2. LongPanel: the TimeSeriesDataFrame surface
# --------------------------------------------------------------------------- #
def test_slice_by_timestep_is_per_item_iloc(frames: ag.LongFrames) -> None:
    panel = frames.panel()
    sliced = panel.slice_by_timestep(None, 10)
    assert list(sliced.num_timesteps_per_item()) == [10] * len(REGIONS)
    assert list(sliced.item_ids) == list(REGIONS)

    window = panel.slice_by_timestep(10, 17)
    assert list(window.num_timesteps_per_item()) == [7] * len(REGIONS)
    assert window.loc["alpha"].index[0] == panel.loc["alpha"].index[10]


def test_column_selection_keeps_the_index(frames: ag.LongFrames) -> None:
    known = frames.panel()[list(FUTURE_COVS)]
    assert list(known.columns) == list(FUTURE_COVS)
    assert list(known.index.names) == [ag.ITEM_ID, ag.TIMESTAMP]


def test_test_split_length_rounds(frames: ag.LongFrames) -> None:
    """F3/F34: Chronos ROUNDS where every darts family truncates."""
    assert ag.test_split_length(847, 0.20) == 169
    assert ag.test_split_length(N_STEPS, 0.20) == 12
    assert ag.test_split_length(7, 0.5) == 4  # round-half-even is python's


def test_naive_scales_are_the_seasonal_differences_of_the_train_head(
    frames: ag.LongFrames,
) -> None:
    mae, rmse = ag.scales_for_stage(frames, TARGET, train_frac=0.70, seasonality=7)
    train_only = int(0.70 * N_STEPS)
    y = frames.values.loc["alpha", TARGET].to_numpy(dtype=float)[:train_only]
    err = y[7:] - y[:-7]
    assert mae["alpha"] == pytest.approx(float(np.mean(np.abs(err))))
    assert rmse["alpha"] == pytest.approx(float(np.sqrt(np.mean(err**2))))
    assert set(mae) == set(REGIONS)


# --------------------------------------------------------------------------- #
# 3. the inverse adapter
# --------------------------------------------------------------------------- #
def test_median_column_prefers_the_float_then_the_string_then_the_mean() -> None:
    idx = pd.MultiIndex.from_tuples([("a", pd.Timestamp("2024-01-01"))])
    assert ag.median_column(pd.DataFrame({0.5: [1.0], "mean": [2.0]}, index=idx)) == 0.5
    assert ag.median_column(pd.DataFrame({"0.5": [1.0], "mean": [2.0]}, index=idx)) == "0.5"
    assert ag.median_column(pd.DataFrame({"mean": [2.0]}, index=idx)) == "mean"
    with pytest.raises(KeyError, match="median column"):
        ag.median_column(pd.DataFrame({"0.9": [2.0]}, index=idx))


def test_predictions_to_series_follows_region_order(forecaster) -> None:
    pred, _ = forecaster.predict_frames(30)
    series = ag.predictions_to_series(pred, list(reversed(REGIONS)))
    assert len(series) == len(REGIONS)
    assert len(series[0]) == HORIZON
    forward = ag.predictions_to_series(pred, list(REGIONS))
    assert forward[0].values().ravel().tolist() == series[-1].values().ravel().tolist()


def test_fold_long_frame_is_the_six_legacy_columns(forecaster) -> None:
    pred, future_slice = forecaster.predict_frames(30)
    frame = ag.fold_long_frame(pred, future_slice, 3, target=TARGET)

    assert list(frame.columns) == list(LEGACY_COLUMNS)
    assert set(frame["fold"]) == {3}
    assert sorted(frame["horizon"].unique()) == list(range(1, HORIZON + 1))
    assert list(frame["region"].unique()) == list(REGIONS)  # item-major (F43)
    assert len(frame) == len(REGIONS) * HORIZON


def test_prediction_set_appends_origin_date_and_channel(forecaster) -> None:
    fold_frames = [
        ag.fold_long_frame(*forecaster.predict_frames(t0), i, target=TARGET)
        for i, t0 in enumerate((30, 31))
    ]
    ps = ag.prediction_set(fold_frames)

    assert isinstance(ps, PredictionSet)
    assert list(ps.frame.columns) == list(COLUMNS)
    assert ps.channels == (SINGLE_CHANNEL,)
    for _fold, block in ps.frame.groupby("fold"):
        assert (block["origin_date"] == block["date"].min()).all()


def test_an_empty_prediction_set_is_still_typed() -> None:
    ps = ag.prediction_set([])
    assert list(ps.frame.columns) == list(COLUMNS)
    assert len(ps.frame) == 0


# --------------------------------------------------------------------------- #
# 4. the forecaster on the shared schedule
# --------------------------------------------------------------------------- #
def test_the_forecaster_is_a_fixed_predictor(forecaster) -> None:
    from strikecast.backtest.protocols import Forecaster

    assert isinstance(forecaster, Forecaster)
    assert forecaster.retrains is False
    assert forecaster.channels == (SINGLE_CHANNEL,)
    with pytest.raises(RuntimeError, match="fixed predictor"):
        forecaster.fit([], cutoff=pd.Timestamp("2022-10-01"))


def test_prepare_ignores_the_engine_covariates(forecaster) -> None:
    forecaster.prepare(None, None)  # no raise, no state


def test_predict_slices_the_frame_exactly_as_the_legacy_loop(forecaster, frames) -> None:
    timestamps = frames.values.loc[REGIONS[0]].index
    t0 = 40
    cutoff = timestamps[t0]

    forecaster.predict(HORIZON, [], cutoff=cutoff)
    call = forecaster.predictor.calls[-1]

    assert call["context_len"] == t0                     # slice_by_timestep(None, t0)
    assert call["context_end"] == timestamps[t0 - 1]     # drop_after(cutoff) semantics
    assert call["known_start"] == cutoff                 # slice_by_timestep(t0, t0 + h)
    assert call["known_len"] == HORIZON
    assert call["known_cols"] == list(FUTURE_COVS)
    assert call["use_cache"] is False                    # F124


def test_an_unknown_cutoff_is_a_loud_error(forecaster) -> None:
    with pytest.raises(KeyError, match="not a timestamp"):
        forecaster.predict(HORIZON, [], cutoff=pd.Timestamp("1999-01-01"))


def test_the_schedule_never_retrains_and_has_the_legacy_shape(forecaster, frames) -> None:
    from strikecast.backtest.schedule import fold_schedule

    start_frac = 0.7999999999999999  # train + val (F17)
    folds = fold_schedule(
        n_total=N_STEPS,
        start_frac=start_frac,
        horizon=HORIZON,
        predict_stride=1,
        retrain_stride=None,
    )
    start_idx = int(start_frac * N_STEPS)
    assert [f.t0 for f in folds] == list(range(start_idx, N_STEPS - HORIZON + 1))
    assert not any(f.retrain for f in folds)

    targets = chronos.level_targets_from_frames(frames, TARGET, list(REGIONS))
    ps = chronos.run_backtest(forecaster, targets, start_frac=start_frac, horizon=HORIZON)
    assert ps.frame["fold"].nunique() == len(folds)
    assert len(ps.frame) == len(folds) * len(REGIONS) * HORIZON


def test_run_backtest_reproduces_chronos2_rolling_long(forecaster, frames) -> None:
    """Level-D style: the engine against a verbatim copy of the legacy loop.

    ``chronos2_rolling_long`` (``_chronos2.py:406-456``) is inlined below with
    only the prints removed.  The two frames differ in row ORDER (F43: the
    legacy loop is fold-major, the darts collector region-major), so both are
    sorted before comparing; every value must be identical.
    """
    start_frac = 0.7999999999999999
    start_idx = int(start_frac * N_STEPS)
    panel = frames.panel()

    # ---- verbatim `chronos2_rolling_long` ------------------------------- #
    oracle_predictor = FakePredictor()
    long_rows = []
    fold_idx = 0
    for t0 in range(start_idx, N_STEPS - HORIZON + 1, 1):
        context = panel.slice_by_timestep(None, t0)
        future_slice = panel.slice_by_timestep(t0, t0 + HORIZON)
        known = future_slice[list(FUTURE_COVS)]
        pred = oracle_predictor.predict(context, known_covariates=known)
        median_col = ag.median_column(pred.frame)
        pred_df = pred.frame[[median_col]].rename(columns={median_col: "y_pred"}).reset_index()
        truth_df = (
            future_slice.frame[[TARGET]].rename(columns={TARGET: "y_true"}).reset_index()
        )
        merged = pred_df.merge(truth_df, on=["item_id", "timestamp"])
        merged = merged.sort_values(["item_id", "timestamp"])
        merged["horizon"] = merged.groupby("item_id").cumcount() + 1
        merged["fold"] = fold_idx
        merged = merged.rename(columns={"item_id": "region", "timestamp": "date"})
        long_rows.append(merged[list(LEGACY_COLUMNS)])
        fold_idx += 1
    legacy = pd.concat(long_rows, ignore_index=True)

    # ---- the port ------------------------------------------------------- #
    targets = chronos.level_targets_from_frames(frames, TARGET, list(REGIONS))
    ported = chronos.run_backtest(
        forecaster, targets, start_frac=start_frac, horizon=HORIZON
    ).legacy_frame()

    key = ["region", "fold", "horizon"]
    a = legacy.sort_values(key).reset_index(drop=True)
    b = ported.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_max_folds_truncates_the_schedule(forecaster, frames) -> None:
    targets = chronos.level_targets_from_frames(frames, TARGET, list(REGIONS))
    ps = chronos.run_backtest(
        forecaster, targets, start_frac=0.7999999999999999, horizon=HORIZON, max_folds=2
    )
    assert sorted(ps.frame["fold"].unique()) == [0, 1]


# --------------------------------------------------------------------------- #
# 5. specs and registration
# --------------------------------------------------------------------------- #
def test_both_variants_are_registered_for_chronos2() -> None:
    assert registered_names("chronos2") == [chronos.ZERO_SHOT, chronos.FINE_TUNED]


@pytest.mark.parametrize("name", [chronos.ZERO_SHOT, chronos.FINE_TUNED])
def test_spec_flags(name: str) -> None:
    spec = get_spec(name, "chronos2")
    assert spec.family == "chronos"
    assert spec.kind == "chronos"          # not one of the four darts kinds
    assert spec.experiments == ("chronos2",)
    assert spec.needs_raw_past_covs is False


def test_zero_shot_is_deterministic_and_untunable() -> None:
    spec = get_spec(chronos.ZERO_SHOT, "chronos2")
    assert spec.stochastic is False        # F127: nothing is trained
    assert spec.tunable is False
    assert spec.n_trials is None


def test_fine_tuned_is_stochastic_and_has_twelve_trials() -> None:
    spec = get_spec(chronos.FINE_TUNED, "chronos2")
    assert spec.stochastic is True
    assert spec.tunable is True
    assert spec.n_trials == 12             # OPTUNA_N_TRIALS


def test_zero_shot_builds_the_legacy_hyperparameters() -> None:
    builder = get_spec(chronos.ZERO_SHOT, "chronos2").build({}, RunContext(seed=42))
    assert isinstance(builder, chronos.Chronos2Builder)
    assert builder.hyperparameters == {"Chronos2": [{"ag_args": {"name_suffix": "ZeroShot"}}]}
    assert builder.fine_tune is False
    assert builder.random_seed == 42
    assert builder.num_val_windows == 3
    assert builder.prediction_length == 7
    assert builder.eval_metric == "MASE"
    assert builder.freq == "D"


def test_fine_tuned_builds_the_winning_configuration() -> None:
    builder = get_spec(chronos.FINE_TUNED, "chronos2").build({}, RunContext(seed=42))
    assert builder.hyperparameters == {
        "Chronos2": {
            "fine_tune": True,
            "fine_tune_lr": 8.721349828452045e-05,
            "fine_tune_steps": 1500,
            "ag_args": {"name_suffix": "FT_best"},
        }
    }
    assert builder.fine_tune is True


def test_the_defaults_are_the_stored_best_params(repo_root) -> None:
    import json

    stored = json.loads(
        (repo_root / "golden/checkpoints/chronos2_best/best_params.json").read_text()
    )["best_params"]
    assert dict(chronos.DEFAULT_FINE_TUNE_PARAMS) == stored


def test_from_best_params_accepts_both_shapes() -> None:
    spec = get_spec(chronos.FINE_TUNED, "chronos2")
    bare = spec.from_best_params({"fine_tune_lr": 1e-5, "fine_tune_steps": 300})
    nested = spec.from_best_params(
        {"best_params": {"fine_tune_lr": 1e-5, "fine_tune_steps": 300}, "n_trials": 12}
    )
    assert bare == nested == {"fine_tune": True, "fine_tune_lr": 1e-5, "fine_tune_steps": 300}


def test_search_space_is_the_legacy_two_knobs() -> None:
    class Trial:
        def __init__(self) -> None:
            self.float_calls: list[tuple] = []
            self.int_calls: list[tuple] = []

        def suggest_float(self, name, low, high, log=False):  # noqa: ANN001
            self.float_calls.append((name, low, high, log))
            return low

        def suggest_int(self, name, low, high, step=1):  # noqa: ANN001
            self.int_calls.append((name, low, high, step))
            return low

    trial = Trial()
    params = chronos.search_space(trial)
    assert trial.float_calls == [("fine_tune_lr", 1e-6, 1e-4, True)]
    assert trial.int_calls == [("fine_tune_steps", 200, 3000, 100)]
    assert params == {"fine_tune_lr": 1e-6, "fine_tune_steps": 200}


def test_the_optuna_trials_all_lie_inside_the_search_space(repo_root) -> None:
    trials = pd.read_csv(repo_root / "golden/results/chronos2/optuna_trials.csv")
    assert len(trials) == 12
    assert trials["params_fine_tune_lr"].between(1e-6, 1e-4).all()
    assert trials["params_fine_tune_steps"].between(200, 3000).all()
    assert (trials["params_fine_tune_steps"] % 100 == 0).all()
    best = trials.loc[trials["value"].idxmin()]
    assert best["params_fine_tune_lr"] == pytest.approx(
        chronos.DEFAULT_FINE_TUNE_PARAMS["fine_tune_lr"]
    )
    assert best["params_fine_tune_steps"] == chronos.DEFAULT_FINE_TUNE_PARAMS["fine_tune_steps"]


def test_internal_validation_score_rejects_a_multi_row_leaderboard() -> None:
    class P:
        def __init__(self, rows):
            self.rows = rows

        def leaderboard(self):
            return pd.DataFrame({"score_val": self.rows})

    assert chronos.internal_validation_score(P([-1.10])) == pytest.approx(1.10)
    with pytest.raises(ValueError, match="F130"):
        chronos.internal_validation_score(P([-1.10, -1.20]))


# --------------------------------------------------------------------------- #
# 6. the experiment config
# --------------------------------------------------------------------------- #
def test_the_experiment_config_is_the_chronos_shape() -> None:
    from strikecast.config.loader import load_experiment

    cfg = load_experiment("chronos2")
    assert cfg.name == "chronos2"
    assert [m.name for m in cfg.models] == [chronos.ZERO_SHOT, chronos.FINE_TUNED]
    assert [p.name for p in cfg.paradigms] == ["global"]
    assert cfg.data.panel_variant == "chronos"
    assert list(cfg.stages) == ["test"]                       # F131: no CV stage
    stage = cfg.stages["test"]
    assert stage.retrain_stride is None                       # the fixed-predictor shape
    assert stage.start == "train_val_end"
    assert stage.start_frac(cfg.series.split) == 0.7999999999999999  # F17
    assert stage.naive_scales.fit_on == "train"
    assert cfg.tuning is not None
    assert cfg.tuning.objective == "internal_MASE"            # F129, not RMSSE_mean
    assert cfg.tuning.n_trials == {"chronos": 12}
    assert cfg.device_for(chronos.ZERO_SHOT) == "auto"


# --------------------------------------------------------------------------- #
# 7. the environment boundary
# --------------------------------------------------------------------------- #
def test_the_modules_import_without_autogluon() -> None:
    """Both modules must import in the main env (plan §5.7)."""
    code = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_module(self, name, path=None):
                if name == "autogluon" or name.startswith("autogluon."):
                    raise ImportError("autogluon is blocked")
                return None

            def find_spec(self, name, path=None, target=None):
                if name == "autogluon" or name.startswith("autogluon."):
                    raise ImportError("autogluon is blocked")
                return None

        sys.meta_path.insert(0, Blocker())
        import strikecast.data.autogluon as a
        import strikecast.models.chronos as c
        assert c.ZERO_SHOT == "chronos2_zero_shot"
        assert a.ITEM_ID == "item_id"
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_to_timeseries_dataframe_says_what_it_needs(frames: ag.LongFrames) -> None:
    if HAS_AUTOGLUON:
        tsdf = ag.to_timeseries_dataframe(frames)
        assert tsdf.num_items == len(REGIONS)
        assert tsdf.freq == "D"
        assert ag.STATIC_COL in tsdf.static_features.columns
    else:
        with pytest.raises(ModuleNotFoundError):
            ag.to_timeseries_dataframe(frames)


@pytest.mark.skipif(not HAS_AUTOGLUON, reason="needs envs/autogluon")
def test_longpanel_agrees_with_the_real_timeseriesdataframe(frames: ag.LongFrames) -> None:
    """The stand-in must slice like the class it stands in for."""
    tsdf = ag.to_timeseries_dataframe(frames)
    panel = frames.panel()

    assert list(tsdf.item_ids) == list(panel.item_ids)
    assert list(tsdf.num_timesteps_per_item()) == list(panel.num_timesteps_per_item())
    for start, end in ((None, 10), (10, 17), (40, 47)):
        a = pd.DataFrame(tsdf.slice_by_timestep(start, end))
        b = panel.slice_by_timestep(start, end).frame
        pd.testing.assert_frame_equal(a, b, check_like=False)


# --------------------------------------------------------------------------- #
# 8. the bundle entry point
# --------------------------------------------------------------------------- #
def test_bundle_adapter_refuses_to_pretend_it_reproduces_the_thesis() -> None:
    with pytest.raises(ValueError, match="F125"):
        ag.bundle_to_long_frames(object(), target=TARGET)  # type: ignore[arg-type]


def test_bundle_adapter_rebuilds_the_same_items_and_statics(panel: pd.DataFrame) -> None:
    from strikecast.config.schema import SeriesConfig
    from strikecast.data.series import build_bundle

    flat = panel.reset_index()
    bundle = build_bundle(
        flat,
        target=TARGET,
        past_covariates=list(PAST_COVS),
        future_covariates=list(FUTURE_COVS),
        config=SeriesConfig(),
    )
    frames = ag.bundle_to_long_frames(bundle, target=TARGET, strict=False)

    assert frames.item_ids == list(REGIONS)
    assert list(frames.static.columns) == [ag.STATIC_COL]
    assert set(frames.values.columns) >= {TARGET, *FUTURE_COVS, *PAST_COVS}
    assert "index" not in frames.values.columns      # F125: the panel-only column
    np.testing.assert_allclose(
        frames.values.loc["alpha", TARGET].to_numpy(dtype=float),
        flat.loc[flat["region"] == "alpha", TARGET].to_numpy(dtype=float),
    )
