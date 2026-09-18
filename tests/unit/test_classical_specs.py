"""Linear / ARIMA / naive specs against the verbatim legacy builders.

Oracles: ``tests/legacy_ref/diff_builders.py`` and
``tests/legacy_ref/count_builders.py``, byte-for-byte copies of
``build_regressor`` in ``_diff_regression.py``, ``_regression_GBDT.py`` and
``_regression_LSTM.py``.  Two darts models are "the same" when their class,
their ``model_params`` and their forwarded ``kwargs`` are equal.

Registered name -> legacy artefacts
-----------------------------------
====================  ==========  ======  =====================================================
registered name       experiment  tuned   result files
====================  ==========  ======  =====================================================
``linear``            diff        no      ``golden/results/diff/global_{cv,test}_global_linear.json``
``arima``             diff        no      ``golden/results/diff/global_{cv,test}_global_arima.json``
``naive_last``        diff        no      ``golden/results/diff/global_{cv,test}_global_naive_last.json``
``naive_weekly``      diff        no      ``golden/results/diff/global_{cv,test}_global_naive_weekly.json``
====================  ==========  ======  =====================================================

None of the four has a tuning directory: they skip Optuna
(``_diff_regression.py``:1203 "Linear and ARIMA skip tuning").  All four are
diff-only: the count family's ``REGRESSORS_TO_RUN`` is dead code and its
leaderboard has no baseline row -- see
:func:`test_count_family_persisted_no_baseline_results` (F84, F86).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from strikecast.models import classical
from strikecast.models.spec import RunContext, get_spec, registered_names

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "golden"
DIFF_RESULTS = GOLDEN / "results" / "diff"

# the verbatim oracles import the legacy top-level ``src`` package
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if not (REPO_ROOT / "src" / "prevalent_functions.py").exists():
    pytest.skip("legacy `src` package not present", allow_module_level=True)

from tests.legacy_ref import count_builders as legacy_count  # noqa: E402
from tests.legacy_ref import diff_builders as legacy_diff  # noqa: E402

SEED = 42  # RANDOM_STATE in all three scripts

#: ``REGRESSORS_TO_RUN`` of ``_diff_regression.py``:124-129, the four names that
#: never went through Optuna.
DIFF_UNTUNED_LINEUP = ("naive_last", "naive_weekly", "arima", "linear")


def fingerprint(model: Any) -> tuple[Any, ...]:
    if model is None:
        return (None,)
    return (type(model), dict(model.model_params), getattr(model, "kwargs", None))


# --------------------------------------------------------------------------- #
# lineups and experiment membership                                            #
# --------------------------------------------------------------------------- #
def test_diff_untuned_lineup_is_registered():
    """Every name of the diff script's ``REGRESSORS_TO_RUN`` has a spec."""
    diff_names = set(registered_names("diff"))
    assert set(DIFF_UNTUNED_LINEUP) <= diff_names


def test_linear_is_registered_for_diff_only():
    """``LinearRegressionModel`` exists in exactly one script.

    ``grep -n LinearRegressionModel _regression_GBDT.py _regression_LSTM.py``
    is empty; only ``_diff_regression.py``:361 has it.  Pinned by asking both
    count copies of ``build_regressor`` for it.
    """
    assert get_spec("linear").experiments == ("diff",)
    assert "linear" not in registered_names("count")
    for builder in (legacy_count.build_regressor_gbdt, legacy_count.build_regressor_lstm):
        with pytest.raises(ValueError, match="Unknown regressor name"):
            builder("linear")


def test_arima_is_registered_for_diff_only():
    """The count scripts define an ``arima`` branch but never ran it (F86).

    ``REGRESSORS_TO_RUN`` is dead in both count scripts: nothing iterates it,
    the CV and test loops iterate ``best_params_by_variant`` instead, and no
    ``arima`` file exists under ``golden/results/gbdt`` or
    ``golden/results/lstm``.  The reported ARIMA numbers come from the diff
    run (plan sec. 2.1, F13), so the spec belongs to ``diff``.
    """
    assert get_spec("arima").experiments == ("diff",)
    assert "arima" not in registered_names("count")
    # the dead count branch is nonetheless identical to the diff one
    assert fingerprint(legacy_count.build_regressor_gbdt("arima")) == fingerprint(
        legacy_diff.build_regressor("arima")
    )


@pytest.mark.parametrize("name", classical.NAIVE_NAMES)
def test_naives_are_registered_for_diff_only(name: str):
    """Both count scripts list the naives but never ran them (F84)."""
    spec = get_spec(name)
    assert spec.experiments == ("diff",)
    assert name in registered_names("diff")
    assert name not in registered_names("count")


def test_no_baseline_is_registered_for_the_count_experiment():
    """``registered_names("count")`` holds no baseline at all.

    ``golden/results/gbdt/leaderboard.csv`` is 36 rows: 6 GBDT variants x 3
    paradigms x 2 splits, with no ``naive_*``, ``linear`` or ``arima`` row.
    """
    count_names = set(registered_names("count"))
    assert count_names.isdisjoint({"linear", "arima", *classical.NAIVE_NAMES})


# --------------------------------------------------------------------------- #
# golden artefacts                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", DIFF_UNTUNED_LINEUP)
def test_registered_name_appears_in_golden_diff_results(name: str):
    if not DIFF_RESULTS.exists():
        pytest.skip(f"golden results not available at {DIFF_RESULTS}")
    for split in ("cv", "test"):
        assert (DIFF_RESULTS / f"global_{split}_global_{name}.json").exists()


def test_count_family_persisted_no_baseline_results():
    """F84: the count scripts' ``REGRESSORS_TO_RUN`` produced nothing.

    ``naive_last``, ``naive_weekly`` and ``arima`` are in the count lineup but
    no count result file carries those names, so the count leaderboard has no
    baseline row and its skill columns (which divide by ``naive_weekly``) stay
    NaN.  Pinned as a data fact, not repaired.
    """
    for family in ("gbdt", "lstm"):
        directory = GOLDEN / "results" / family
        if not directory.exists():
            pytest.skip(f"golden results not available at {directory}")
        for name in (*classical.NAIVE_NAMES, "arima", "linear"):
            assert not list(directory.glob(f"*_{name}.json")), f"{family}/{name}"

    leaderboard = GOLDEN / "results" / "gbdt" / "leaderboard.csv"
    rows = leaderboard.read_text().strip().splitlines()[1:]
    # 6 GBDT variants x 3 paradigms x 2 splits, and nothing else
    assert len(rows) == 36
    assert not [r for r in rows if "naive" in r or "linear" in r or "arima" in r]


def test_none_of_the_baselines_has_a_tuning_dir():
    root = GOLDEN / "converted" / "tuning"
    if not root.exists():
        pytest.skip(f"converted tuning artefacts not available at {root}")
    existing = {p.name for p in root.glob("*/*") if p.is_dir()}
    assert existing.isdisjoint({"linear", "arima", *classical.NAIVE_NAMES})


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #
def test_linear_build_matches_legacy():
    """``LinearRegressionModel(**COMMON_KWARGS_TAB, multi_models=True)``.

    No ``random_state`` and no device, so the ctx must not leak into the model:
    two different contexts have to give the same object.
    """
    spec = get_spec("linear")
    theirs = legacy_diff.build_regressor("linear")
    assert fingerprint(spec.build(spec.defaults, RunContext())) == fingerprint(theirs)
    assert fingerprint(
        spec.build(spec.defaults, RunContext(seed=7, device="cuda", threads=16))
    ) == fingerprint(theirs)
    assert dict(theirs.model_params)["random_state"] is None


def test_arima_build_matches_legacy():
    """``ARIMA(p=7, d=0, q=1, random_state=RANDOM_STATE)`` on the differences."""
    spec = get_spec("arima")
    ours = spec.build(spec.defaults, RunContext(seed=SEED))
    theirs = legacy_diff.build_regressor("arima")
    assert fingerprint(ours) == fingerprint(theirs)
    params = dict(ours.model_params)
    assert (params["p"], params["d"], params["q"]) == (7, 0, 1)
    assert params["random_state"] == SEED


def test_arima_seed_comes_from_the_context():
    spec = get_spec("arima")
    assert dict(spec.build(spec.defaults, RunContext(seed=3)).model_params)["random_state"] == 3


def test_arima_defaults_are_the_legacy_orders():
    assert dict(get_spec("arima").defaults) == {"p": 7, "d": 0, "q": 1}


def test_arima_fallback_is_naive_mean():
    """F5: the diff local runner falls back to ``NaiveMean()`` on any exception.

    ``tests/legacy_ref/diff_runners.py``:138, 221, 309.
    """
    from darts.models import NaiveMean

    spec = get_spec("arima")
    assert spec.fallback is not None
    assert isinstance(spec.fallback(), NaiveMean)
    # a fresh instance every call: the runner rebuilds it per region per fold
    assert spec.fallback() is not spec.fallback()


@pytest.mark.parametrize("name", classical.NAIVE_NAMES)
def test_naive_build_returns_none(name: str):
    """``build_regressor`` returns ``None`` for the naives in all three scripts."""
    spec = get_spec(name)
    assert spec.build(spec.defaults, RunContext()) is None
    assert legacy_diff.build_regressor(name) is None
    assert legacy_count.build_regressor_gbdt(name) is None
    assert legacy_count.build_regressor_lstm(name) is None


# --------------------------------------------------------------------------- #
# spec flags                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "kind"),
    [("linear", "global"), ("arima", "local"), ("naive_last", "naive"), ("naive_weekly", "naive")],
)
def test_spec_flags(name: str, kind: str):
    """Plan sec. 5.4: deterministic models run once and are broadcast."""
    spec = get_spec(name)
    assert spec.kind == kind
    assert spec.stochastic is False
    assert spec.is_neural is False
    assert spec.needs_raw_past_covs is False
    assert spec.search_space is None
    assert spec.tunable is False
    assert spec.n_trials is None
    assert spec.device == "cpu"
    assert spec.threads_from_context is False


@pytest.mark.parametrize("name", ["linear", "naive_last", "naive_weekly"])
def test_only_arima_has_a_fallback(name: str):
    """F5 is an ARIMA-only behaviour in the diff runner."""
    assert get_spec(name).fallback is None
