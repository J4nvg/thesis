"""Golden levels E and F, end to end through the NEW pipeline.

Every other golden module replays a frozen artefact (a panel, a schedule, a
stored prediction frame) through one ported function. This one is the first
that compares what ``strikecast.pipeline.run_stage`` *produces on the real
data* with what the thesis produced: same panel, same schedule, same tuned
hyper-parameters, same cached feature set, then the predictions and the
``global.json`` metrics side by side with ``golden/results/``.

Why the subset is small
-----------------------
A full experiment is days of compute. This module covers a **cheap,
CPU-deterministic subset**, which is the set plan §6 level E names:

* ``diff``: ``naive_last`` / ``naive_weekly`` (no fit at all), ``linear``
  (closed form), ``arima`` (per region, one statsmodels fit per step) and
  ``lightgbm``, the one diff GBDT the thesis ran on CPU
  (``device: {lightgbm: cpu, xgboost: cuda, catboost: GPU}``);
* ``count``: ``lightgbm_poisson``, the only count variant with a converted
  Optuna study under ``golden/converted/tuning/checkpoints_tune/`` -- the
  other five count GBDTs were never tuned, so there are no stored best params
  to run them from.

What the runs showed, and why the GBDTs are level F here
--------------------------------------------------------
The baselines reproduce ``golden/results`` at level E. **The GBDTs do not, and
cannot on this machine** -- but the port is not the reason. Measured on
2026-09-18 on the diff CV stage, fold 0, with identical inputs:

======================================================  ==================
comparison                                              max ``|dy_pred|``
======================================================  ==================
port engine vs the verbatim legacy runner                        **0.0**
  (``tests/legacy_ref/diff_runners.run_expanding_cv_iter`` with
  ``diff_builders.build_gbm_from_params``, both fed the legacy
  ``features/diffreg_saved_sets.pkl`` covariates, 8 folds)
verbatim legacy runner vs ``golden/results/diff``                 1.17
port vs ``golden/results/diff``                                   1.18
======================================================  ==================

So the legacy code, run here, does not reproduce its own stored predictions
either: LightGBM 4.6.0 on this platform builds different trees from the
platform the thesis ran on, and a difference at one split cascades through
300 boosted trees. The covariate values themselves agree to 1.1e-13 (34 of
the 52 selected columns are ``ewm_*``, where the pandas that wrote the cache
and pandas 3.0.2 differ by ~1 ULP), and feeding the pipeline the legacy
pickle's covariates instead of its own moves the fold-0 deviation only from
1.18 to 1.17 -- so that is a second-order effect, not the cause.

This contradicts plan F9's "the count-family GBDTs run on CPU and are
therefore bit-reproducible". They are reproducible *given the same machine*;
they are not portable. Recorded as **F124**. The consequence for §6 is that
the GBDTs move from level E to level F ("family-specific tolerance, recorded
in the golden report") and that the real port assertion for them is
:func:`test_lightgbm_port_equals_the_legacy_runner`, which compares the two
implementations *in one process* and is exact.

How to produce the run store this module reads
----------------------------------------------
Nothing is trained by the golden comparisons -- a CV stage is 3-15 minutes and
a test stage 7-35 -- so the runs are made once, out of band, and the tests skip
when they are absent::

    python scripts/import_golden_params.py            # runs/<exp>/tuning/<model>/
    PYTHONHASHSEED=0 strikecast run experiment=diff legacy=diff \\
        model=naive_last,naive_weekly,linear,arima,lightgbm \\
        paradigm=global stage=cv,test seed=42 tracking=noop --store-root runs
    PYTHONHASHSEED=0 strikecast run experiment=count legacy=count \\
        model=lightgbm_poisson \\
        paradigm=global stage=cv,test seed=42 tracking=noop --store-root runs

``legacy=<exp>`` (``configs/legacy/``) is required: it restores the thesis'
buggy ``expdecay7`` window transform and the thesis' cached feature sets
(audit 2026-09-26 A1/A13). The publication configs use the fixed ``leaky7``
filter and refuse the golden sets.

``PYTHONHASHSEED=0`` is plan F16's rule for new runs. It does not change this
comparison, because ``feature_selection.cache_path`` loads the thesis' own
selected names *and their recorded order*, but it is what ``env.json`` records.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.golden

pytest.importorskip("darts")

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "golden" / "results"
CONVERTED = REPO_ROOT / "golden" / "converted"
STORE_ROOT = REPO_ROOT / "runs"

#: ``experiment -> golden/results/<dir>``. The count family's frames live under
#: ``gbdt`` (``_regression_GBDT.py`` wrote them), the diff family's under ``diff``.
GOLDEN_DIR_BY_EXPERIMENT = {"count": "gbdt", "diff": "diff"}

#: ``experiment -> golden/converted/feature_sets/<name>.json``.
FEATURE_SET_BY_EXPERIMENT = {"count": "countreg", "diff": "diffreg"}

#: The key columns of a legacy long frame, in ``legacy_frame`` order.
KEY_COLUMNS = ["region", "fold", "horizon", "date"]


@dataclass(frozen=True)
class Case:
    """One ``(run, golden file)`` pair and the tolerance it is held to."""

    experiment: str
    model: str
    stage: str
    golden_model: str
    level: str
    #: absolute tolerance on ``y_pred`` (0.0 means bit-exact).
    pred_tol: float
    #: relative tolerance on every metric of ``global.json``.
    metric_tol: float
    #: why a level-F tolerance is what it is; empty for level E.
    why: str = ""
    paradigm: str = "global"
    seed: int = 42
    #: per-metric ceilings that REPLACE ``metric_tol`` for those keys. One
    #: pathologically sensitive metric must not force every other metric of the
    #: same case up to its ceiling -- see F125 for the case that motivated this.
    metric_tol_by_key: Mapping[str, float] = field(default_factory=dict)
    #: ceiling on the MEAN absolute ``y_pred`` deviation (``None`` = unchecked).
    #: A max ceiling alone lets a real regression through as long as no single
    #: row moves by more than the platform gap (audit 2026-09-26 B6).
    pred_mean_tol: float | None = None

    def tol_for(self, metric: str) -> float:
        return self.metric_tol_by_key.get(metric, self.metric_tol)

    @property
    def golden_key(self) -> str:
        return f"{self.stage}_{self.paradigm}_{self.golden_model}"

    @property
    def golden_dir(self) -> Path:
        return GOLDEN / GOLDEN_DIR_BY_EXPERIMENT[self.experiment]

    @property
    def id(self) -> str:
        return f"{self.experiment}-{self.model}-{self.stage}"


#: The level-F ceilings below are the observed deviations rounded UP to the next
#: half order of magnitude, so they pin the numbers without pretending to a
#: precision the platform cannot deliver. The observed values are in the table
#: above each entry and in ``docs/REFACTOR_PROGRESS.md``.
_LGBM_WHY = (
    "F124: LightGBM trees are not portable across platforms; the verbatim legacy "
    "runner does not reproduce this file here either "
    "(see test_lightgbm_port_equals_the_legacy_runner)"
)

_ARIMA_WHY = (
    "F124: statsmodels' ARIMA(7,0,1) is an MLE fit, and the optimum it reaches "
    "depends on the BLAS/LAPACK this platform links; 72% of rows differ by more "
    "than 1e-6 and none by more than 0.17, which is the signature of a slightly "
    "different parameter estimate rather than a different model. The ported "
    "engine IS pinned against the legacy ARIMA runner, including the F5 NaiveMean "
    "fallback, by the level-D suite (tests/equivalence/test_diff_runners.py)"
)


def _cases() -> list[Case]:
    cases: list[Case] = []
    for stage in ("cv", "test"):
        for model in ("naive_last", "naive_weekly"):
            # No fit at all: the only float noise possible is the parquet round trip.
            cases.append(Case("diff", model, stage, model, "E", 0.0, 1e-9))
        # Closed-form solve: observed max |dy_pred| 2.3e-09 / 2.0e-09,
        # worst metric 4.3e-10 / 8.3e-10.
        cases.append(Case("diff", "linear", stage, "linear", "E", 1e-6, 1e-6))
    # Per-region statsmodels ARIMA(7,0,1) on the differences with the F5
    # NaiveMean fallback. Level F, not E -- see _ARIMA_WHY.
    # cv:   max |dy_pred| 0.167, worst metric 4.8e-03 (MedAE), MASE_mean 1.6e-03
    # test: max |dy_pred| 0.216, worst metric 5.1e-03 (MedAE), MASE_mean 1.1e-03
    # `TweedieDev` is held to its own, much looser ceiling (F125): at power=1.5 a
    # prediction clipped to EPS=1e-9 contributes 2*y*(1e-9)**-0.5 ~= 63,245, so
    # the statistic counts sign flips of near-zero predictions. Measured on the
    # test stage: 8 of 22,960 rows explain 88.8% of a 5.3e-2 gap, while every
    # other metric agrees to <= 3.1e-3. Keeping one ceiling for the case would
    # have loosened all eleven metrics to 1e-1 to cover this one.
    _arima_tweedie = {"TweedieDev": 1e-1}
    cases.append(Case("diff", "arima", "cv", "arima", "F", 0.5, 1e-2, _ARIMA_WHY,
                      metric_tol_by_key=_arima_tweedie))
    cases.append(Case("diff", "arima", "test", "arima", "F", 0.5, 1e-2, _ARIMA_WHY,
                      metric_tol_by_key=_arima_tweedie))
    # Level F -- see the module docstring and F124.
    # diff/lightgbm  cv:   max |dy_pred| 2.90, mean 0.128, worst metric 0.472
    #                      (TweedieDev), MASE_mean 0.0700, RMSSE_mean 0.0049
    #                test:  max |dy_pred| 3.69, mean 0.204, worst metric 1.140 (ME,
    #                      which is ~0 so its relative deviation is meaningless),
    #                      MASE_mean 0.0036, RMSSE_mean 0.00044
    # count/lightgbm_poisson (re-measured 2026-09-26, audit B6):
    #                cv:   max |dy_pred| 0.50, mean 0.011  (fold 0: 0.21)
    #                test: max |dy_pred| 0.95, mean 0.017  (fold 0: 0.17)
    # Ceilings: max ~1.4-2x the observed platform gap (it is not portable across
    # machines, F124), mean ~2.5-3x -- tight enough that a wrong feature set or
    # wrong params (mean deviations of the order of the predictions themselves)
    # fail, where the former 5.0 absolute ceiling asserted almost nothing.
    cases.append(
        Case("diff", "lightgbm", "cv", "lightgbm_tuned", "F", 5.0, 1.0, _LGBM_WHY,
             pred_mean_tol=0.35)
    )
    cases.append(
        Case("diff", "lightgbm", "test", "lightgbm_tuned", "F", 5.0, 2.0, _LGBM_WHY,
             pred_mean_tol=0.5)
    )
    cases.append(
        Case("count", "lightgbm_poisson", "cv", "lightgbm_poisson_tuned", "F", 1.0, 1.0,
             _LGBM_WHY, pred_mean_tol=0.03)
    )
    cases.append(
        Case("count", "lightgbm_poisson", "test", "lightgbm_poisson_tuned", "F", 2.0, 2.0,
             _LGBM_WHY, pred_mean_tol=0.05)
    )
    return cases


CASES = _cases()

#: Metrics whose *relative* deviation is not interpretable because the golden
#: value is a near-zero mean error; they are still compared, on the scale of the
#: frame's MAE, by :func:`_worst_metric`.
NEAR_ZERO_METRICS = frozenset({"ME"})

#: What this module deliberately does not run, and why.
NOT_RUN: dict[str, str] = {
    "diff/xgboost": "device=cuda in the legacy run (F9); no CUDA on this machine",
    "diff/catboost": "task_type=GPU in the legacy run (F9); no GPU on this machine",
    "*/lstm_*, */gru_*": "torch training, not bit-reproducible (F9); hours per stage",
    "count/{lightgbm_tweedie,xgboost_*,catboost_poisson}": (
        "full stages not run here (their params come from _regression_GBDT.ipynb "
        "cell 56, audit B1, imported by scripts/import_golden_params.py); count "
        "catboost_tweedie is held EXACTLY on one retrain window by "
        "test_count_catboost_tweedie_reproduces_golden_on_the_first_retrain_window"
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _require_golden(case: Case) -> Path:
    path = case.golden_dir / f"predictions_long_{case.golden_key}.parquet"
    if not path.is_file():
        pytest.skip(f"golden predictions missing: {path.relative_to(REPO_ROOT)}")
    return path


def _require_run(case: Case):
    """The stage's store handle, or a skip that says how to make the run."""
    from strikecast.store import RunKey, RunStore  # noqa: PLC0415

    if not STORE_ROOT.is_dir():
        pytest.skip(f"no run store at {STORE_ROOT}; see this module's docstring")
    store = RunStore(STORE_ROOT)
    key = RunKey(case.experiment, case.model, case.paradigm, case.seed)
    if not store.is_complete(key, case.stage):
        pytest.skip(
            f"no complete run for {key.relative()}/{case.stage}; produce it with\n"
            f"  PYTHONHASHSEED=0 strikecast run experiment={case.experiment} "
            f"model={case.model} paradigm={case.paradigm} stage={case.stage} "
            f"seed={case.seed} tracking=noop --store-root runs"
        )
    return store, key


def _relative_deviations(got: dict, want: dict) -> dict[str, float]:
    """``metric -> relative deviation`` over the golden file's own keys.

    ``ME`` is scored against ``MAE`` rather than against itself, because the
    mean error is a near-zero quantity whose relative deviation says nothing
    (a golden ``ME`` of 3.9e-05 with a run value of 8.4e-05 is a relative
    deviation of 1.14 and an absolute one of 4e-05).
    """
    out: dict[str, float] = {}
    for name, expected in want.items():
        expected = float(expected)
        actual = float(got[name])
        if math.isnan(expected) and math.isnan(actual):
            continue
        absolute = abs(actual - expected)
        scale = abs(float(want["MAE"])) if name in NEAR_ZERO_METRICS else abs(expected)
        out[name] = absolute / scale if scale else absolute
    return out


def _worst_metric(got: dict, want: dict) -> tuple[str, float]:
    """``(name, worst relative deviation)``; kept for callers that want one number."""
    deviations = _relative_deviations(got, want)
    if not deviations:
        return "", 0.0
    worst_key = max(deviations, key=lambda k: deviations[k])
    return worst_key, deviations[worst_key]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_predictions_match_golden(case: Case) -> None:
    """``PredictionSet.legacy_frame()`` against ``predictions_long_<key>.parquet``.

    Strict about shape and order whatever the level: the legacy collector is
    region-major and so is ``legacy_frame`` (quirk Q3 in
    ``backtest/predictions.py``), so the frames must line up row for row
    without sorting, ``y_true`` must be identical, and the row COUNT must
    match -- a count difference is a schedule bug (F52 already accounts for
    the one row the diff CV stage drops). Only ``y_pred`` gets a tolerance.
    """
    golden_path = _require_golden(case)
    store, key = _require_run(case)

    got = store.load_predictions(key, case.stage, legacy_order=True).legacy_frame()
    want = pd.read_parquet(golden_path)

    assert list(got.columns) == list(want.columns), (
        f"{case.id}: column mismatch\n  run:    {list(got.columns)}\n"
        f"  golden: {list(want.columns)}"
    )
    assert len(got) == len(want), f"{case.id}: {len(got)} rows, golden has {len(want)}"

    got = got.reset_index(drop=True)
    want = want.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        got[KEY_COLUMNS], want[KEY_COLUMNS], obj=f"{case.id} keys", check_dtype=False
    )
    np.testing.assert_array_equal(
        got["y_true"].to_numpy(float), want["y_true"].to_numpy(float), err_msg=f"{case.id} y_true"
    )

    deviation = np.abs(got["y_pred"].to_numpy(float) - want["y_pred"].to_numpy(float))
    worst = float(np.nanmax(deviation))
    row = int(np.nanargmax(deviation))
    assert worst <= case.pred_tol, (
        f"{case.id} (level {case.level}): max |dy_pred| = {worst:.6g} > {case.pred_tol:g} "
        f"at row {row} ({got.loc[row, 'region']}, fold {got.loc[row, 'fold']}, "
        f"h={got.loc[row, 'horizon']}): run {got.loc[row, 'y_pred']!r} vs "
        f"golden {want.loc[row, 'y_pred']!r}. {case.why}"
    )
    if case.pred_mean_tol is not None:
        mean = float(np.nanmean(deviation))
        assert mean <= case.pred_mean_tol, (
            f"{case.id} (level {case.level}): mean |dy_pred| = {mean:.6g} > "
            f"{case.pred_mean_tol:g}. {case.why}"
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_global_metrics_match_golden(case: Case) -> None:
    """``<stage>/metrics/global.json`` against ``global_<key>.json``.

    The metric port itself is pinned on FROZEN frames by
    ``tests/golden/test_metrics_equality.py``; what this adds is that the
    pipeline feeds it the right frame and the right scales (F6/F67: train-only
    for every count and diff stage).
    """
    golden_path = case.golden_dir / f"global_{case.golden_key}.json"
    if not golden_path.is_file():
        pytest.skip(f"golden metrics missing: {golden_path.relative_to(REPO_ROOT)}")
    store, key = _require_run(case)
    run_path = store.metrics_dir(key, case.stage) / "global.json"
    if not run_path.is_file():
        pytest.skip(f"no global.json for {key.relative()}/{case.stage}")

    want = json.loads(golden_path.read_text(encoding="utf-8"))
    got = json.loads(run_path.read_text(encoding="utf-8"))
    assert set(want) <= set(got), f"{case.id}: metrics missing from the run: {set(want) - set(got)}"
    # `n` is exact at every level: a row-count difference is a schedule bug.
    assert int(got["n"]) == int(want["n"]), f"{case.id}: n {got['n']} != golden {want['n']}"

    # Each metric is held to ITS OWN ceiling: `metric_tol` for almost all of
    # them, and a documented override for the few whose sensitivity is a
    # property of the statistic rather than of the port (F125).
    over = [
        (name, dev, case.tol_for(name))
        for name, dev in sorted(
            _relative_deviations(got, want).items(), key=lambda kv: -kv[1]
        )
        if dev > max(case.tol_for(name), 1e-12)
    ]
    assert not over, (
        f"{case.id} (level {case.level}): "
        + "; ".join(
            f"{name} deviates {dev:.6g} (run {got.get(name)!r} vs golden "
            f"{want.get(name)!r}), ceiling {tol:g}"
            for name, dev, tol in over
        )
        + f". {case.why}"
    )


# ---------------------------------------------------------------------------
# The port assertion for the GBDTs (F124)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_lightgbm_port_equals_the_legacy_runner(inputs) -> None:
    """The engine and the verbatim legacy diff CV runner, in ONE process.

    This is the assertion the GBDT golden comparison cannot make (F124): given
    the same inputs, the same builder and the same schedule, the ported engine
    and ``tests/legacy_ref/diff_runners.run_expanding_cv_iter`` must produce
    bit-identical predictions. Whatever this platform's LightGBM does, it does
    it to both sides.

    Fold-limited to :data:`N_FOLDS` because each side refits at fold 0 and
    every 7th fold; the full 79-fold stage is ~3 minutes per side.
    """
    pytest.importorskip("lightgbm")
    import pickle  # noqa: PLC0415
    import sys  # noqa: PLC0415

    cache = REPO_ROOT / "features" / "diffreg_saved_sets.pkl"
    best_params = CONVERTED / "tuning" / "checkpoints_tune_diff" / "lightgbm" / "best_params.json"
    if not cache.is_file() or not best_params.is_file():
        pytest.skip("the legacy feature cache or the converted study is not present")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from tests.legacy_ref.diff_builders import build_gbm_from_params  # noqa: PLC0415
    from tests.legacy_ref.diff_runners import run_expanding_cv_iter  # noqa: PLC0415

    from strikecast.backtest.engine import ExpandingWindowBacktest  # noqa: PLC0415
    from strikecast.config.loader import load_experiment  # noqa: PLC0415
    from strikecast.models.registry import get_spec  # noqa: PLC0415
    from strikecast.pipeline import data_stage, run_stage  # noqa: PLC0415
    from strikecast.pipeline.context import make_run_context  # noqa: PLC0415
    from strikecast.store import RunStore  # noqa: PLC0415

    with cache.open("rb") as handle:
        (_names, _tr, _vl, _te, past_covs, future_covs, target_cv, _tve, cv_start) = pickle.load(
            handle
        )
    best = json.loads(best_params.read_text(encoding="utf-8"))["best_params"]

    n_folds = 4
    cfg = load_experiment(
        "diff", overrides=["tracking=noop", "legacy=diff"], config_dir=str(REPO_ROOT / "configs")
    )
    store = RunStore(STORE_ROOT)
    data = data_stage.prepare_data(cfg, store)
    model_targets = run_stage.model_targets_for(cfg, data, "cv")
    level_targets = run_stage.stage_targets(data, "cv")
    stage_cfg = cfg.stage("cv")

    legacy = run_expanding_cv_iter(
        lambda: build_gbm_from_params("lightgbm", dict(best)),
        target_diff_list=model_targets,
        target_level_list=list(target_cv),
        start_frac=cv_start,
        past_covs=past_covs,
        future_covs=future_covs,
        is_neural=False,
    )
    legacy_folds = None
    for index, cumulative in enumerate(legacy):
        legacy_folds = [list(region) for region in cumulative]
        if index + 1 >= n_folds:
            break
    legacy.close()

    spec = get_spec("lightgbm", "diff")
    params, source = run_stage.resolve_params(cfg, spec, "lightgbm", store)
    assert source == "tuned", "the port must run on the imported study, not the spec defaults"
    forecaster = run_stage.make_forecaster(
        cfg, spec, "lightgbm", params, make_run_context(cfg, "lightgbm", 42),
        preset=stage_cfg.adapter_for("global"),
    )
    engine = ExpandingWindowBacktest(
        stage_cfg.backtest(cfg.split), run_stage._transform(cfg), hooks=[]
    )
    ported: list[list] = [[] for _ in level_targets]
    iterator = engine.iter_folds(
        forecaster, level_targets, list(past_covs), list(future_covs),
        model_targets=model_targets,
    )
    for index, result in enumerate(iterator):
        predictions = next(iter(result.predictions.values()))
        for region, series in enumerate(predictions):
            ported[region].append(series)
        if index + 1 >= n_folds:
            break
    iterator.close()

    assert legacy_folds is not None
    worst = max(
        float(np.nanmax(np.abs(a.values() - b.values())))
        for legacy_region, ported_region in zip(legacy_folds, ported, strict=True)
        for a, b in zip(legacy_region, ported_region, strict=True)
    )
    assert worst == 0.0, (
        f"the ported engine and the legacy diff CV runner disagree by {worst:.3e} "
        f"over {n_folds} folds with identical inputs; that IS a port bug, unlike "
        "the deviation against golden/results (F124)"
    )


# ---------------------------------------------------------------------------
# Level G: the pipeline's covariate ORDER, not just its names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("experiment", sorted(FEATURE_SET_BY_EXPERIMENT))
def test_selected_covariate_order_matches_the_cached_set(experiment: str, inputs) -> None:
    """``prepare_data`` must keep the cached component ORDER, not sort it.

    ``subset_components`` iterates the names as given and the resulting
    TimeSeries component order becomes the GBDT's feature-column order, which
    LightGBM's ``colsample_bytree`` draws against. Sorting the cached names --
    which ``_sets_from_payload`` used to do -- therefore changes every tree.
    F16 says the cached selection is data to load; this pins that its ORDER is
    part of that data.
    """
    name = FEATURE_SET_BY_EXPERIMENT[experiment]
    payload_path = CONVERTED / "feature_sets" / f"{name}.json"
    if not payload_path.is_file():
        pytest.skip(f"converted feature set missing: {payload_path.relative_to(REPO_ROOT)}")

    from strikecast.config.loader import load_experiment  # noqa: PLC0415
    from strikecast.pipeline import data_stage  # noqa: PLC0415
    from strikecast.store import RunStore  # noqa: PLC0415

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    cfg = load_experiment(
        experiment,
        overrides=["tracking=noop", f"legacy={experiment}"],
        config_dir=str(REPO_ROOT / "configs"),
    )
    bundle = data_stage.prepare_data(cfg, RunStore(STORE_ROOT)).bundle

    assert list(bundle.past_covs[0].components) == list(payload["past_covariate_components"])
    assert list(bundle.future_covs[0].components) == list(payload["future_covariate_components"])


# ---------------------------------------------------------------------------
# Level E on a fold subset: count CatBoost-Tweedie (Table 4 row 1), audit B1
# ---------------------------------------------------------------------------
#: One retrain window = one CatBoost fit per horizon; folds 0-6 share it.
CATBOOST_FOLDS = 7


@pytest.mark.slow
def test_count_catboost_tweedie_reproduces_golden_on_the_first_retrain_window() -> None:
    """The thesis' #1 model, exactly, through the NEW pipeline.

    Legacy mode (``legacy=count``: expdecay7 + the thesis' cached countreg set),
    the cell-56 params (``scripts/import_golden_params.py``), the global test
    stage, the first retrain window (folds 0-6): the predictions must equal
    ``golden/results/gbdt/predictions_long_test_global_catboost_tweedie_tuned``
    to 1e-9. CatBoost on CPU is portable where LightGBM is not (F124; audit B1
    measured 1.8e-15 at fold 0).
    """
    pytest.importorskip("catboost")
    golden_path = GOLDEN / "gbdt" / "predictions_long_test_global_catboost_tweedie_tuned.parquet"
    params_path = STORE_ROOT / "count" / "tuning" / "catboost_tweedie" / "best_params.json"
    if not golden_path.is_file():
        pytest.skip(f"golden predictions missing: {golden_path}")
    if not params_path.is_file():
        pytest.skip("run scripts/import_golden_params.py (cell-56 params, audit B1)")
    if not (CONVERTED / "feature_sets" / "countreg.json").is_file():
        pytest.skip("the converted countreg feature set is missing")

    from strikecast.backtest.engine import ExpandingWindowBacktest  # noqa: PLC0415
    from strikecast.backtest.predictions import PredictionSet  # noqa: PLC0415
    from strikecast.config.loader import load_experiment  # noqa: PLC0415
    from strikecast.models.registry import get_spec  # noqa: PLC0415
    from strikecast.pipeline import data_stage, run_stage  # noqa: PLC0415
    from strikecast.pipeline.context import make_run_context  # noqa: PLC0415
    from strikecast.store import RunStore  # noqa: PLC0415

    cfg = load_experiment(
        "count", overrides=["tracking=noop", "legacy=count"], config_dir=str(REPO_ROOT / "configs")
    )
    store = RunStore(STORE_ROOT)
    data = data_stage.prepare_data(cfg, store)
    spec = get_spec("catboost_tweedie", "count")
    params, source = run_stage.resolve_params(cfg, spec, "catboost_tweedie", store)
    assert source == "tuned"
    stage_cfg = cfg.stage("test")
    forecaster = run_stage.make_forecaster(
        cfg, spec, "catboost_tweedie", params, make_run_context(cfg, "catboost_tweedie", 42),
        preset=stage_cfg.adapter_for("global"),
    )
    engine = ExpandingWindowBacktest(stage_cfg.backtest(cfg.split), run_stage._transform(cfg))
    past, future = run_stage._covariates(spec, data)
    targets = run_stage.stage_targets(data, "test")
    cumulative: dict[str, list[list]] = {}
    iterator = engine.iter_folds(forecaster, targets, past, future)
    for index, result in enumerate(iterator):
        for channel, region_preds in result.predictions.items():
            bucket = cumulative.setdefault(channel, [[] for _ in targets])
            for region, series in enumerate(region_preds):
                bucket[region].append(series)
        if index + 1 >= CATBOOST_FOLDS:
            break
    iterator.close()

    got = (
        PredictionSet.from_fold_preds(targets, cumulative, data.region_names)
        .legacy_frame()
        .reset_index(drop=True)
    )
    want = pd.read_parquet(golden_path)
    want = want[want["fold"] < CATBOOST_FOLDS].reset_index(drop=True)
    assert len(got) == len(want) == len(data.region_names) * CATBOOST_FOLDS * 7
    pd.testing.assert_frame_equal(
        got[KEY_COLUMNS], want[KEY_COLUMNS], check_dtype=False, obj="catboost keys"
    )
    np.testing.assert_array_equal(got["y_true"].to_numpy(float), want["y_true"].to_numpy(float))
    worst = float(np.max(np.abs(got["y_pred"].to_numpy(float) - want["y_pred"].to_numpy(float))))
    assert worst <= 1e-9, f"count catboost_tweedie folds 0-6: max |dy_pred| = {worst:.3e}"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_the_cheap_subset_is_what_the_plan_names() -> None:
    """Level E/F's model list, pinned so a silent shrink of CASES is visible."""
    assert {(c.experiment, c.model) for c in CASES} == {
        ("diff", "naive_last"),
        ("diff", "naive_weekly"),
        ("diff", "linear"),
        ("diff", "arima"),
        ("diff", "lightgbm"),
        ("count", "lightgbm_poisson"),
    }
    assert {c.stage for c in CASES} == {"cv", "test"}
    assert {c.level for c in CASES} == {"E", "F"}
    # Every level-F case must say why its tolerance is what it is.
    assert all(c.why for c in CASES if c.level == "F")


def test_every_case_has_a_golden_counterpart() -> None:
    """Each case names a file that exists, so a typo cannot hide as a skip."""
    if not GOLDEN.is_dir():
        pytest.skip(f"golden corpus not available at {GOLDEN}")
    missing = [
        str((c.golden_dir / f"predictions_long_{c.golden_key}.parquet").relative_to(REPO_ROOT))
        for c in CASES
        if not (c.golden_dir / f"predictions_long_{c.golden_key}.parquet").is_file()
    ]
    assert not missing, f"cases point at files that do not exist: {missing}"


def test_the_tuned_cases_have_imported_best_params() -> None:
    """``resolve_params`` must find the thesis' own study, not the spec defaults.

    Running a tuned variant on the spec defaults would compare two different
    models and is the easiest way to get a mysterious level-E failure, so it is
    checked rather than assumed.
    """
    from strikecast.store import RunStore  # noqa: PLC0415

    if not STORE_ROOT.is_dir():
        pytest.skip(f"no run store at {STORE_ROOT}; run scripts/import_golden_params.py")
    store = RunStore(STORE_ROOT)
    tuned = {(c.experiment, c.model) for c in CASES if c.golden_model.endswith("_tuned")}
    missing = [
        f"{experiment}/{model}"
        for experiment, model in sorted(tuned)
        if not (store.tuning_dir(experiment, model) / "best_params.json").is_file()
    ]
    assert not missing, (
        f"no imported best_params.json for {missing}; run "
        "`python scripts/import_golden_params.py`"
    )
