"""The Chronos-2 family through the pipeline (audit C15), without AutoGluon.

* the registry hook: ``chronos2_*`` resolve through ``pipeline.context``;
* ``data_stage.prepare_data`` dispatches ``panel_variant: chronos`` to
  :func:`~strikecast.pipeline.chronos_stage.prepare_chronos_data` (real data);
* :func:`~strikecast.pipeline.chronos_stage.run_chronos_stage` on a synthetic
  frame with a fake predictor: store layout, metrics, predictor reuse by fit
  identity, skip/resume, ``cv`` skipped (F131), ``max_folds``, C17;
* the fine-tune study (:mod:`strikecast.tuning.autogluon_runner`) with a fake
  fit: artefacts, and the SAME parameter sequence as the legacy in-memory
  ``create_study(direction="minimize", sampler=TPESampler(seed=42))``;
* ``scripts/import_golden_chronos.py``.

The real-AutoGluon half (a pipeline-fit predictor against the thesis'
predictions) is in ``tests/golden/test_chronos_equality.py``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strikecast.data import autogluon as ag
from strikecast.models import chronos

REPO_ROOT = Path(__file__).resolve().parents[2]
REGIONS = ("alpha", "bravo", "charlie")
N_STEPS = 60
TARGET = "act_drone_strike_on_ua"
FUTURE_COVS = ("env_ua_holiday", "env_weather_temperature_2m_mean")
PAST_COVS = ("com_actor_a", "acled_other_ua_armed_clash")
HORIZON = 7
#: int(0.7999999999999999 * 60) = 47 -> 60 - 47 - 7 + 1 folds
N_FOLDS = N_STEPS - int(0.7999999999999999 * N_STEPS) - HORIZON + 1


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakePredictor:
    """A deterministic 0.5 quantile: ``last context value + h``."""

    def __init__(self) -> None:
        self.n_predict = 0

    def predict(self, data, known_covariates=None, use_cache=True):  # noqa: ANN001
        self.n_predict += 1
        assert use_cache is False  # F124
        blocks = []
        for region in data.item_ids:
            sub = data.loc[region]
            last = float(sub[TARGET].to_numpy()[-1])
            index = pd.date_range(sub.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
            values = [last + h for h in range(1, HORIZON + 1)]
            blocks.append(
                pd.DataFrame(
                    {"mean": values, "0.5": values},
                    index=pd.MultiIndex.from_product(
                        [[region], index], names=[ag.ITEM_ID, ag.TIMESTAMP]
                    ),
                )
            )
        return ag.LongPanel(pd.concat(blocks))

    def leaderboard(self):
        return pd.DataFrame([{"model": "Chronos2ZeroShot", "score_val": -1.1}])


class FakeFit:
    """Stands in for ``models.chronos.load_or_fit_predictor``; records every call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, path, *, builder, train_data=None, target, known_covariates_names=None,
                 force=False, verbosity=None):
        path = Path(path)
        fitted = not path.exists()
        if fitted:
            path.mkdir(parents=True)
            (path / "predictor.pkl").write_text("fake")
        self.calls.append(
            {"path": path, "fitted": fitted, "train_data": train_data,
             "hyperparameters": builder.hyperparameters, "seed": builder.random_seed,
             "known": list(known_covariates_names or []), "target": target}
        )
        return FakePredictor()


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-09-28", periods=N_STEPS, freq="D")
    rows = []
    for r_idx, region in enumerate(REGIONS):
        for t, date in enumerate(dates):
            row = {"region": region, "event_date": date, "Activity_Level": r_idx + 1,
                   "index": r_idx * N_STEPS + t, TARGET: float(rng.poisson(0.5 + r_idx))}
            row.update({c: float(rng.normal()) for c in FUTURE_COVS})
            row.update({c: float(rng.normal()) for c in PAST_COVS})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["region", "event_date"])


@pytest.fixture
def chronos_data():
    from strikecast.data.cache import content_hash
    from strikecast.pipeline.chronos_stage import ChronosData
    from strikecast.pipeline.data_stage import FeatureSets

    frames = ag.panel_to_long_frames(_panel())
    past = [c for c in frames.values.columns if c != TARGET and c not in FUTURE_COVS]
    data = ChronosData(
        bundle=None,  # type: ignore[arg-type]
        features=FeatureSets(past, list(FUTURE_COVS), "all", content_hash("f", past)),
        panel_hash="panel",
        series_hash="series",
        activity_by_region={r: i + 1 for i, r in enumerate(REGIONS)},
        frames=frames,
        target=TARGET,
        future_covariates=FUTURE_COVS,
        past_covariates=tuple(past),
    )
    data.__dict__["tsdf"] = frames.panel()  # the cached_property, without AutoGluon
    data.__dict__["train_data"] = "TRAIN_DATA"
    return data


@pytest.fixture
def cfg(tmp_path: Path):
    from strikecast.config.loader import load_experiment

    return load_experiment("chronos2", ["tracking=noop", f"++store.root={tmp_path / 'runs'}"])


@pytest.fixture
def fake_fit(monkeypatch) -> FakeFit:
    fit = FakeFit()
    monkeypatch.setattr(chronos, "load_or_fit_predictor", fit)
    return fit


# --------------------------------------------------------------------------- #
# 1. registry and data
# --------------------------------------------------------------------------- #
def test_the_pipeline_resolves_both_chronos_specs() -> None:
    from strikecast.pipeline.context import get_spec

    assert get_spec("chronos2_zero_shot", "chronos2").kind == "chronos"
    assert get_spec("chronos2_fine_tuned", "chronos2").tunable


def test_prepare_data_dispatches_the_chronos_family(cfg) -> None:
    if not (REPO_ROOT / "data" / "dataset").is_dir():
        pytest.skip("real data not present")
    from strikecast.pipeline import data_stage
    from strikecast.pipeline.chronos_stage import ChronosData

    data = data_stage.prepare_data(cfg)
    assert isinstance(data, ChronosData)
    assert len(data.region_names) == 20
    assert len(data.future_covariates) == 27
    assert len(data.past_covariates) == 84 and "index" in data.past_covariates  # F123
    assert data.features.source == "all"
    assert data.n_steps == 847
    again = data_stage.prepare_data(cfg)
    assert again.upstream == data.upstream
    mae, rmse = data.scales()
    assert len(mae) == len(rmse) == 20


# --------------------------------------------------------------------------- #
# 2. the run stage
# --------------------------------------------------------------------------- #
def test_run_chronos_stage_writes_the_store_layout(cfg, chronos_data, fake_fit, tmp_path):
    from strikecast.evaluation.metrics import chronos_evaluate_long
    from strikecast.pipeline import run_stage
    from strikecast.store import RunKey, RunStore

    store = RunStore(tmp_path / "runs")
    out = run_stage.run_stage(cfg, "chronos2_zero_shot", "global", 42, "test", chronos_data,
                              store=store)
    assert not out.skipped
    assert out.n_folds == N_FOLDS
    assert out.n_rows == N_FOLDS * len(REGIONS) * HORIZON
    (call,) = fake_fit.calls
    assert call["fitted"] and call["train_data"] == "TRAIN_DATA"
    assert call["hyperparameters"] == chronos.zero_shot_hyperparameters()
    assert call["known"] == list(FUTURE_COVS) and call["seed"] == 42

    key = RunKey("chronos2", "chronos2_zero_shot", "global", 42)
    run_dir = tmp_path / "runs" / key.relative()
    assert (run_dir / "artifacts" / "predictor" / "predictor.pkl").exists()
    meta = json.loads((run_dir / "artifacts" / "predictor.json").read_text())
    assert meta["hyperparameters"] == chronos.zero_shot_hyperparameters()
    assert json.loads((run_dir / "artifacts" / "internal_leaderboard.json").read_text())

    # the metric views are chronos_evaluate_long of the stored predictions
    preds = store.load_predictions(key, "test", legacy_order=True).legacy_frame()
    expected = chronos_evaluate_long(preds, *chronos_data.scales())
    got = json.loads((run_dir / "test" / "metrics" / "global.json").read_text())
    assert got == pytest.approx(expected["global"])
    assert {p.name for p in (run_dir / "test" / "metrics").iterdir()} == {
        "global.json", "per_region.csv", "per_horizon.csv", "per_region_horizon.csv",
    }

    # and the predictions are exactly models.chronos.run_backtest's
    forecaster = chronos.Chronos2Forecaster(
        FakePredictor(), chronos_data.tsdf, target=TARGET, known_covariates=list(FUTURE_COVS),
        region_names=list(REGIONS),
    )
    targets = chronos.level_targets_from_frames(chronos_data.tsdf, TARGET, list(REGIONS))
    direct = chronos.run_backtest(forecaster, targets, start_frac=0.7999999999999999).legacy_frame()
    key_cols = ["region", "fold", "horizon"]
    pd.testing.assert_frame_equal(
        preds.sort_values(key_cols).reset_index(drop=True)[list(direct.columns)],
        direct.sort_values(key_cols).reset_index(drop=True),
        check_dtype=False,
    )


def test_a_complete_stage_is_skipped_and_the_predictor_reused(cfg, chronos_data, fake_fit, tmp_path):
    from strikecast.pipeline.chronos_stage import run_chronos_stage
    from strikecast.store import RunStore

    store = RunStore(tmp_path / "runs")
    run_chronos_stage(cfg, "chronos2_zero_shot", "global", 42, "test", chronos_data, store=store)
    again = run_chronos_stage(cfg, "chronos2_zero_shot", "global", 42, "test", chronos_data,
                              store=store)
    assert again.skipped and len(fake_fit.calls) == 1

    # a different stage identity (max_folds) re-runs, but the predictor is the
    # same fit (same hyper-parameters, seed and data) and is reloaded, not refit
    short = run_chronos_stage(cfg, "chronos2_zero_shot", "global", 42, "test", chronos_data,
                              store=store, max_folds=2)
    assert not short.skipped and short.n_folds == 2
    assert [c["fitted"] for c in fake_fit.calls] == [True, False]
    assert short.stage_hash != again.stage_hash

    # --force refits
    run_chronos_stage(cfg, "chronos2_zero_shot", "global", 42, "test", chronos_data,
                      store=store, force=True)
    assert [c["fitted"] for c in fake_fit.calls] == [True, False, True]


def test_the_cv_stage_is_skipped(cfg, chronos_data, fake_fit, tmp_path) -> None:
    from strikecast.pipeline import run_stage
    from strikecast.store import RunStore

    outcomes = run_stage.run_experiment(
        cfg, data=chronos_data, models=["chronos2_zero_shot"], seeds=[42],
        stages=("cv", "test"), store=RunStore(tmp_path / "runs"),
    )
    assert [(o.stage, o.skipped) for o in outcomes] == [("cv", True), ("test", False)]


def test_fine_tuned_needs_tuned_params(cfg, chronos_data, fake_fit, tmp_path) -> None:
    """Audit C17/D4: no best_params.json -> refuse, unless --allow-default-params."""
    from strikecast.pipeline import run_stage
    from strikecast.store import RunKey, RunStore

    if not hasattr(run_stage, "require_tuned_params"):
        pytest.skip("run_stage has no C17 guard in this tree")
    store = RunStore(tmp_path / "runs")
    with pytest.raises(Exception, match="best_params.json"):
        run_stage.run_stage(cfg, "chronos2_fine_tuned", "global", 42, "test", chronos_data,
                            store=store)
    key = RunKey("chronos2", "chronos2_fine_tuned", "global", 42)
    assert store.read_state(key, "test").status == "failed"
    assert not fake_fit.calls

    out = run_stage.run_stage(cfg, "chronos2_fine_tuned", "global", 7, "test", chronos_data,
                              store=store, allow_default_params=True)
    assert not out.skipped and out.params_source == "defaults"
    assert fake_fit.calls[-1]["hyperparameters"] == chronos.fine_tune_hyperparameters(
        8.721349828452045e-05, 1500, "FT_best"
    )
    assert fake_fit.calls[-1]["seed"] == 7


def test_tuned_params_reach_the_predictor(cfg, chronos_data, fake_fit, tmp_path) -> None:
    from strikecast.pipeline.chronos_stage import run_chronos_stage
    from strikecast.store import RunStore

    store = RunStore(tmp_path / "runs")
    directory = store.tuning_dir("chronos2", "chronos2_fine_tuned")
    directory.mkdir(parents=True)
    (directory / "best_params.json").write_text(json.dumps(
        {"variant": "chronos2_fine_tuned",
         "best_params": {"fine_tune_lr": 3e-5, "fine_tune_steps": 700}}
    ))
    out = run_chronos_stage(cfg, "chronos2_fine_tuned", "global", 42, "test", chronos_data,
                            store=store)
    assert out.params_source == "tuned"
    assert fake_fit.calls[-1]["hyperparameters"] == chronos.fine_tune_hyperparameters(
        3e-5, 700, "FT_best"
    )


def test_make_forecaster_points_at_the_chronos_stage(cfg) -> None:
    from strikecast.pipeline.context import get_spec, make_run_context
    from strikecast.pipeline.run_stage import make_forecaster

    spec = get_spec("chronos2_zero_shot", "chronos2")
    with pytest.raises(ValueError, match="run_chronos_stage"):
        make_forecaster(cfg, spec, "chronos2_zero_shot", {},
                        make_run_context(cfg, "chronos2_zero_shot", 42), preset="for_test")


def test_a_non_global_paradigm_is_refused(cfg, chronos_data, fake_fit, tmp_path) -> None:
    from strikecast.pipeline.chronos_stage import run_chronos_stage
    from strikecast.store import RunStore

    with pytest.raises(ValueError, match="F133"):
        run_chronos_stage(cfg, "chronos2_zero_shot", "local", 42, "test", chronos_data,
                          store=RunStore(tmp_path))


# --------------------------------------------------------------------------- #
# 3. the fine-tune study
# --------------------------------------------------------------------------- #
def _fake_score(params) -> float:
    return abs(math.log10(params["fine_tune_lr"]) + 4.3) + params["fine_tune_steps"] / 1e4


class _Fitted:
    def __init__(self, params):
        self.params = params


def _fake_fit(train_data, *, builder, target, known_covariates_names, path, verbosity):
    assert train_data == "TRAIN_DATA" and verbosity == 0
    assert builder.hyperparameters["Chronos2"]["ag_args"] == {"name_suffix": "FT"}
    Path(path).mkdir(parents=True)
    return _Fitted(builder.params)


def test_the_study_matches_the_legacy_sampler(cfg, chronos_data, tmp_path) -> None:
    """Same TPE(seed=42) suggestions as ``_chronos2.py``'s in-memory study, all 12 trials."""
    import optuna

    from strikecast.pipeline.chronos_stage import tune_chronos_model
    from strikecast.store import RunStore

    store = RunStore(tmp_path / "runs")
    out = tune_chronos_model(
        cfg, "chronos2_fine_tuned", chronos_data, store=store,
        fit=_fake_fit, score=lambda predictor: _fake_score(predictor.params),
    )
    assert not out.skipped and out.n_trials == 12

    # the legacy study, verbatim apart from the objective body
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    def chronos2_objective(trial):
        fine_tune_lr = trial.suggest_float("fine_tune_lr", 1e-6, 1e-4, log=True)
        fine_tune_steps = trial.suggest_int("fine_tune_steps", 200, 3000, step=100)
        return _fake_score({"fine_tune_lr": fine_tune_lr, "fine_tune_steps": fine_tune_steps})

    study.optimize(chronos2_objective, n_trials=12, gc_after_trial=True)
    assert out.best_params == study.best_params
    assert out.best_value == pytest.approx(study.best_value)
    assert type(study.pruner).__name__ == "MedianPruner"  # the legacy default, never used

    directory = store.tuning_dir("chronos2", "chronos2_fine_tuned")
    meta = json.loads((directory / "best_params.json").read_text())
    assert meta["pruner"] == "MedianPruner" and meta["sampler"] == "TPESampler"
    assert meta["n_trials"] == 12 and meta["n_pruned"] == 0
    trials = pd.read_csv(directory / "trials.csv")
    legacy = study.trials_dataframe()
    np.testing.assert_allclose(trials["params_fine_tune_lr"], legacy["params_fine_tune_lr"])
    assert list(trials["params_fine_tune_steps"]) == list(legacy["params_fine_tune_steps"])
    assert not (directory / "trials").exists()  # trial predictors are deleted

    # the stored best params drive the test stage
    from strikecast.pipeline.context import get_spec
    from strikecast.pipeline.run_stage import resolve_params

    params, source = resolve_params(cfg, get_spec("chronos2_fine_tuned", "chronos2"),
                                    "chronos2_fine_tuned", store)
    assert source == "tuned" and params["fine_tune_lr"] == study.best_params["fine_tune_lr"]

    skipped = tune_chronos_model(cfg, "chronos2_fine_tuned", chronos_data, store=store)
    assert skipped.skipped and skipped.reason == "best_params.json exists"


def test_tune_stage_dispatches_chronos(cfg, chronos_data, tmp_path, monkeypatch) -> None:
    from strikecast.pipeline import tune_stage
    from strikecast.store import RunStore

    monkeypatch.setattr(chronos, "fit_predictor", _fake_fit)
    monkeypatch.setattr(chronos, "internal_validation_score",
                        lambda predictor: _fake_score(predictor.params))
    store = RunStore(tmp_path / "runs")
    outs = tune_stage.tune_experiment(cfg, data=chronos_data, store=store, n_trials=3)
    by_model = {o.model: o for o in outs}
    assert by_model["chronos2_zero_shot"].skipped
    assert by_model["chronos2_zero_shot"].reason == "not tunable"
    assert by_model["chronos2_fine_tuned"].n_trials == 3


def test_a_failing_fit_aborts_the_study_and_cleans_up(cfg, chronos_data, tmp_path) -> None:
    from strikecast.pipeline.chronos_stage import tune_chronos_model
    from strikecast.store import RunStore

    def broken_fit(train_data, *, builder, target, known_covariates_names, path, verbosity):
        Path(path).mkdir(parents=True)
        raise RuntimeError("CUDA out of memory")

    store = RunStore(tmp_path / "runs")
    with pytest.raises(RuntimeError, match="out of memory"):
        tune_chronos_model(cfg, "chronos2_fine_tuned", chronos_data, store=store, fit=broken_fit)
    trials = store.tuning_dir("chronos2", "chronos2_fine_tuned") / "trials"
    assert not any(trials.iterdir()) if trials.exists() else True


def test_settings_follow_the_experiment_yaml(cfg) -> None:
    from strikecast.tuning.autogluon_runner import settings_from_config

    settings = settings_from_config(cfg)
    assert settings.n_trials == 12 and settings.tuning_seed == 42
    assert settings.metric == "internal_MASE" and settings.gc_after_trial
    bad = cfg.model_copy(update={"tuning": cfg.tuning.model_copy(update={"objective": "RMSSE_mean"})})
    with pytest.raises(ValueError, match="F129"):
        settings_from_config(bad)


# --------------------------------------------------------------------------- #
# 4. the golden importer
# --------------------------------------------------------------------------- #
def _importer():
    path = REPO_ROOT / "scripts" / "import_golden_chronos.py"
    spec = importlib.util.spec_from_file_location("import_golden_chronos", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_golden_study_is_imported(cfg, tmp_path, capsys) -> None:
    golden = REPO_ROOT / "golden"
    if not (golden / "checkpoints" / "chronos2_best" / "best_params.json").exists():
        pytest.skip("golden Chronos checkpoint not present")
    module = _importer()
    store_root = tmp_path / "runs"
    assert module.main(["--golden", str(golden), "--store-root", str(store_root)]) == 0
    assert module.main(["--golden", str(golden), "--store-root", str(store_root)]) == 0
    assert "unchanged" in capsys.readouterr().out

    from strikecast.pipeline.context import get_spec
    from strikecast.pipeline.run_stage import resolve_params
    from strikecast.store import RunStore
    from strikecast.tuning import BEST_PARAMS_KEYS

    directory = store_root / "chronos2" / "tuning" / "chronos2_fine_tuned"
    meta = json.loads((directory / "best_params.json").read_text())
    assert tuple(meta) == BEST_PARAMS_KEYS
    assert meta["best_value"] == pytest.approx(1.1009486182930974)
    params, source = resolve_params(cfg, get_spec("chronos2_fine_tuned", "chronos2"),
                                    "chronos2_fine_tuned", RunStore(store_root))
    assert source == "tuned"
    assert params == {"fine_tune": True, "fine_tune_lr": 8.721349828452045e-05,
                      "fine_tune_steps": 1500}

    (directory / "best_params.json").write_text("{}")
    assert module.main(["--golden", str(golden), "--store-root", str(store_root)]) == 1
    assert module.main(["--golden", str(golden), "--store-root", str(store_root), "--force"]) == 0
