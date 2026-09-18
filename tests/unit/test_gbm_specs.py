"""GBDT specs against the verbatim legacy builders.

Every assertion here compares a ``ModelSpec.build`` result with the model that
``tests/legacy_ref/{count_builders,diff_builders}.py`` -- byte-for-byte copies of
``_regression_GBDT.py``, ``_regression_LSTM.py`` and ``_diff_regression.py`` --
returns for the same inputs.  Two darts models are "the same" when their class,
their ``model_params`` (which carries ``lags``, ``lags_past_covariates``,
``lags_future_covariates``, ``output_chunk_length``, ``output_chunk_shift`` and
``add_encoders`` as well as every model kwarg) and their ``kwargs`` (what darts
forwards to the underlying estimator) are equal.

Registered name -> legacy artefacts
-----------------------------------
=========================  ==========  =================================  ==========================================
registered name            experiment  tuning dir (best_params.json)      result files
=========================  ==========  =================================  ==========================================
``lightgbm_poisson``       count       ``checkpoints_tune/``              ``results/gbdt/*_lightgbm_poisson_tuned.*``
``lightgbm_tweedie``       count       missing (F85)                      ``results/gbdt/*_lightgbm_tweedie_tuned.*``
``xgboost_poisson``        count       missing (F85)                      ``results/gbdt/*_xgboost_poisson_tuned.*``
``xgboost_tweedie``        count       missing (F85)                      ``results/gbdt/*_xgboost_tweedie_tuned.*``
``catboost_poisson``       count       missing (F85)                      ``results/gbdt/*_catboost_poisson_tuned.*``
``catboost_tweedie``       count       missing (F85)                      ``results/gbdt/*_catboost_tweedie_tuned.*``
``lightgbm``               diff        ``checkpoints_tune_diff/``         ``results/diff/*_lightgbm_tuned.*``
``xgboost``                diff        ``checkpoints_tune_diff/``         ``results/diff/*_xgboost_tuned.*``
``catboost``               diff        ``checkpoints_tune_diff/``         ``results/diff/*_catboost_tuned.*``
=========================  ==========  =================================  ==========================================

The registered name is the legacy *variant* string, which is also the tuning
directory name; the legacy runners append ``_tuned`` when they persist results
(``_regression_GBDT.py``:1162, ``_diff_regression.py``:1268).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from strikecast.models import gbm  # noqa: F401  (import registers the specs)
from strikecast.models.spec import RunContext, get_spec, registered_names

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "golden"

# the verbatim oracles import the legacy top-level ``src`` package
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if not (REPO_ROOT / "src" / "prevalent_functions.py").exists():
    pytest.skip("legacy `src` package not present", allow_module_level=True)

from tests.legacy_ref import count_builders as legacy_count  # noqa: E402
from tests.legacy_ref import diff_builders as legacy_diff  # noqa: E402

TUNING_COUNT = GOLDEN / "converted" / "tuning" / "checkpoints_tune"
TUNING_DIFF = GOLDEN / "converted" / "tuning" / "checkpoints_tune_diff"

#: ``get_available_threads()`` on the machine that ran the count scripts is
#: ``min(os.cpu_count(), 64)``; any fixed value pins the same wiring.
COUNT_THREADS = 8
#: ``_diff_regression.py``:90 hard-codes ``available_threads = 4``.
DIFF_THREADS = 4
SEED = 42  # RANDOM_STATE in all three scripts

#: Device of the *default* ``build_regressor`` branch, per script.
GBDT_DEFAULT_DEVICE = {"lightgbm": "gpu", "xgboost": "cuda", "catboost": "GPU"}
#: Device of the *tuned* ``build_gbm_from_params`` branch in the count scripts.
COUNT_TUNED_DEVICE = {"lightgbm": "cpu", "xgboost": "cpu", "catboost": "CPU"}

#: A fixed synthetic params dict per family, shaped like a trial's output.
SYNTHETIC_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 44,
        "learning_rate": 0.037,
        "n_estimators": 400,
        "subsample": 0.73,
        "colsample_bytree": 0.91,
        "reg_alpha": 0.021,
        "reg_lambda": 3.5,
    },
    "xgboost": {
        "max_depth": 9,
        "min_child_weight": 6,
        "learning_rate": 0.088,
        "n_estimators": 900,
        "subsample": 0.64,
        "colsample_bytree": 0.77,
        "reg_alpha": 5.5,
        "reg_lambda": 0.004,
    },
    "catboost": {
        "depth": 6,
        "learning_rate": 0.019,
        "iterations": 700,
        "l2_leaf_reg": 4.5,
        "subsample": 0.82,
    },
}
#: The value a tweedie study would have suggested.
SYNTHETIC_TWEEDIE_POWER = 1.37


def fingerprint(model: Any) -> tuple[Any, ...]:
    """Class, constructor kwargs and forwarded estimator kwargs of a darts model."""
    return (
        type(model),
        dict(model.model_params),
        getattr(model, "kwargs", None),
    )


def synthetic_params(variant: str) -> dict[str, Any]:
    family, _, objective_kind = variant.partition("_")
    params = dict(SYNTHETIC_PARAMS[family])
    if objective_kind == "tweedie":
        params["tweedie_variance_power"] = SYNTHETIC_TWEEDIE_POWER
    return params


def stored_best_params(tuning_root: Path, variant: str) -> dict[str, Any] | None:
    path = tuning_root / variant / "best_params.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    assert payload["variant"] == variant
    return payload["best_params"]


def count_param_sets(variant: str) -> list[tuple[str, dict[str, Any]]]:
    spec = get_spec(variant)
    sets = [("defaults", dict(spec.defaults)), ("synthetic", synthetic_params(variant))]
    best = stored_best_params(TUNING_COUNT, variant)
    if best is not None:
        sets.append(("best_params.json", spec.from_best_params(best)))
    return sets


def diff_param_sets(variant: str) -> list[tuple[str, dict[str, Any]]]:
    spec = get_spec(variant)
    sets = [("defaults", dict(spec.defaults)), ("synthetic", synthetic_params(variant))]
    best = stored_best_params(TUNING_DIFF, variant)
    if best is not None:
        sets.append(("best_params.json", spec.from_best_params(best)))
    return sets


# --------------------------------------------------------------------------- #
# lineups                                                                      #
# --------------------------------------------------------------------------- #
def test_count_gbm_variants_match_legacy_lineup():
    """``GBM_VARIANTS`` of ``_regression_GBDT.py``:118-121."""
    assert gbm.COUNT_GBM_VARIANTS == (
        "lightgbm_poisson",
        "lightgbm_tweedie",
        "xgboost_poisson",
        "xgboost_tweedie",
        "catboost_poisson",
        "catboost_tweedie",
    )
    registered = registered_names("count")
    assert [n for n in registered if n in gbm.COUNT_GBM_VARIANTS] == list(
        gbm.COUNT_GBM_VARIANTS
    )
    # the count family ran nothing but the six tuned GBDTs: its
    # REGRESSORS_TO_RUN baselines are dead code (F84, F86), and the RNN specs
    # live in models/rnn.py, so with only this module imported the lineup is
    # exactly the six.
    if "strikecast.models.rnn" not in sys.modules:
        assert registered == list(gbm.COUNT_GBM_VARIANTS)


def test_diff_gbm_variants_match_legacy_lineup():
    """``GBM_VARIANTS`` of ``_diff_regression.py``:119."""
    assert gbm.DIFF_GBM_VARIANTS == ("lightgbm", "xgboost", "catboost")
    registered = registered_names("diff")
    assert [n for n in registered if n in gbm.DIFF_GBM_VARIANTS] == list(gbm.DIFF_GBM_VARIANTS)


@pytest.mark.parametrize(
    ("variant", "results_dir"),
    [(v, "gbdt") for v in gbm.COUNT_GBM_VARIANTS]
    + [(v, "diff") for v in gbm.DIFF_GBM_VARIANTS],
)
def test_registered_name_appears_in_golden_results(variant: str, results_dir: str):
    """Every registered GBDT name has stored results under ``<variant>_tuned``."""
    directory = GOLDEN / "results" / results_dir
    if not directory.exists():
        pytest.skip(f"golden results not available at {directory}")
    hits = sorted(directory.glob(f"global_cv_global_{variant}{gbm.RESULT_FILE_SUFFIX}.json"))
    assert hits, f"no golden result file for {variant!r} in {directory}"


@pytest.mark.parametrize("variant", gbm.DIFF_GBM_VARIANTS)
def test_diff_variant_has_a_tuning_dir(variant: str):
    if not TUNING_DIFF.exists():
        pytest.skip(f"converted tuning artefacts not available at {TUNING_DIFF}")
    assert (TUNING_DIFF / variant / "best_params.json").exists()


def test_only_lightgbm_poisson_has_a_converted_count_study():
    """F85: five of the six count GBDT studies were never converted to JSON.

    Pinned so the gap is visible; it is a data gap, not a behaviour change.
    """
    if not TUNING_COUNT.exists():
        pytest.skip(f"converted tuning artefacts not available at {TUNING_COUNT}")
    present = {v for v in gbm.COUNT_GBM_VARIANTS if (TUNING_COUNT / v).exists()}
    assert present == {"lightgbm_poisson"}


# --------------------------------------------------------------------------- #
# spec flags                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("variant", "experiment", "device", "threads_from_context"),
    [
        ("lightgbm_poisson", "count", "cpu", True),
        ("lightgbm_tweedie", "count", "cpu", True),
        ("xgboost_poisson", "count", "cpu", True),
        ("xgboost_tweedie", "count", "cpu", True),
        ("catboost_poisson", "count", "CPU", True),
        ("catboost_tweedie", "count", "CPU", True),
        ("lightgbm", "diff", "cpu", True),
        ("xgboost", "diff", "cuda", False),
        ("catboost", "diff", "GPU", False),
    ],
)
def test_spec_flags(variant: str, experiment: str, device: str, threads_from_context: bool):
    """Device and thread wiring as the *tuned* legacy builders had it (F9)."""
    spec = get_spec(variant)
    assert spec.experiments == (experiment,)
    assert spec.kind == "global"
    assert spec.device == device
    assert spec.threads_from_context is threads_from_context
    assert spec.stochastic is True  # subsample / colsample bagging
    assert spec.is_neural is False
    assert spec.needs_raw_past_covs is False
    assert spec.n_trials == 50  # OPTUNA_N_TRIALS in both scripts
    assert spec.tunable


@pytest.mark.parametrize(
    "variant", [*gbm.COUNT_GBM_VARIANTS, *gbm.DIFF_GBM_VARIANTS]
)
def test_from_best_params_is_the_identity(variant: str):
    """The legacy runners pass ``study.best_params`` straight to the builder.

    ``_regression_GBDT.py``:1175 and ``_diff_regression.py``:1281 both do
    ``lambda p=best_params, v=variant: build_gbm_from_params(v, p)``; the
    ``tweedie_variance_power`` pop happens inside the builder, not before it.
    """
    spec = get_spec(variant)
    payload = {"num_leaves": 31, "tweedie_variance_power": 1.5}
    assert spec.from_best_params(payload) == payload
    assert spec.from_best_params(payload) is not payload


# --------------------------------------------------------------------------- #
# build == build_gbm_from_params (the tuned path, which produced every number)  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", gbm.COUNT_GBM_VARIANTS)
def test_count_build_matches_legacy_tuned_builder(variant: str):
    spec = get_spec(variant)
    ctx = RunContext(seed=SEED, device=spec.device, threads=COUNT_THREADS)
    for label, params in count_param_sets(variant):
        ours = spec.build(params, ctx)
        theirs = legacy_count.build_gbm_from_params(
            variant,
            dict(params),
            COMMON_KWARGS_TAB=legacy_count.COMMON_KWARGS_TAB,
            RANDOM_STATE=SEED,
            available_threads=COUNT_THREADS,
        )
        assert fingerprint(ours) == fingerprint(theirs), f"{variant} / {label}"


@pytest.mark.parametrize("variant", gbm.DIFF_GBM_VARIANTS)
def test_diff_build_matches_legacy_tuned_builder(variant: str):
    spec = get_spec(variant)
    ctx = RunContext(
        seed=SEED,
        device=spec.device,
        threads=DIFF_THREADS if spec.threads_from_context else None,
    )
    for label, params in diff_param_sets(variant):
        ours = spec.build(params, ctx)
        theirs = legacy_diff.build_gbm_from_params(
            variant,
            dict(params),
            COMMON_KWARGS_TAB=legacy_diff.COMMON_KWARGS_TAB,
            RANDOM_STATE=SEED,
            available_threads=DIFF_THREADS,
        )
        assert fingerprint(ours) == fingerprint(theirs), f"{variant} / {label}"


def test_stored_best_params_are_actually_exercised():
    """Guard against the previous two tests silently degrading to synthetic only."""
    if not (TUNING_COUNT.exists() and TUNING_DIFF.exists()):
        pytest.skip("converted tuning artefacts not available")
    labels = [label for label, _ in count_param_sets("lightgbm_poisson")]
    assert "best_params.json" in labels
    for variant in gbm.DIFF_GBM_VARIANTS:
        assert "best_params.json" in [label for label, _ in diff_param_sets(variant)]


# --------------------------------------------------------------------------- #
# build(defaults) == build_regressor (the default branch, F8 feature selection) #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", gbm.COUNT_GBM_VARIANTS)
def test_count_defaults_match_gbdt_script_default_branch(variant: str):
    """``_regression_GBDT.py`` default branch: GPU, thread keyword commented out.

    This is the branch the feature-selection pass used (F8), and the source of
    the GPU quirk: the default branches run on GPU while every tuned builder,
    and therefore every reported number, runs on CPU.
    """
    spec = get_spec(variant)
    ctx = RunContext(seed=SEED, device=GBDT_DEFAULT_DEVICE[spec.family], threads=None)
    ours = spec.build(spec.defaults, ctx)
    theirs = legacy_count.build_regressor_gbdt(variant)
    assert fingerprint(ours) == fingerprint(theirs)


@pytest.mark.parametrize("variant", gbm.COUNT_GBM_VARIANTS)
def test_count_defaults_match_lstm_script_default_branch(variant: str):
    """``_regression_LSTM.py`` default branch: CPU with the thread keyword live.

    Drift between the two count scripts (F84): identical variants, identical
    parameters, different device and threading.
    """
    spec = get_spec(variant)
    ctx = RunContext(seed=SEED, device=COUNT_TUNED_DEVICE[spec.family], threads=COUNT_THREADS)
    ours = spec.build(spec.defaults, ctx)
    theirs = legacy_count.build_regressor_lstm(variant, available_threads=COUNT_THREADS)
    assert fingerprint(ours) == fingerprint(theirs)


@pytest.mark.parametrize("variant", ["lightgbm", "xgboost"])
def test_diff_defaults_match_default_branch(variant: str):
    spec = get_spec(variant)
    ctx = RunContext(
        seed=SEED,
        device=spec.device,
        threads=DIFF_THREADS if spec.threads_from_context else None,
    )
    ours = spec.build(spec.defaults, ctx)
    theirs = legacy_diff.build_regressor(variant)
    assert fingerprint(ours) == fingerprint(theirs)


def test_diff_catboost_default_branch_differs_only_in_boost_from_average():
    """F83: the diff CatBoost default branch omits ``boost_from_average``.

    ``_diff_regression.py``:340-355 (default) has no ``boost_from_average``,
    so CatBoost's own default (``True`` for RMSE) applies, while
    ``_diff_regression.py``:444-459 (tuned) passes ``boost_from_average=False``.
    ``build`` follows the tuned builder, which produced every reported diff
    number.  The discrepancy is pinned here, not repaired.
    """
    spec = get_spec("catboost")
    ctx = RunContext(seed=SEED, device=spec.device, threads=None)
    ours = dict(spec.build(spec.defaults, ctx).model_params)
    theirs = dict(legacy_diff.build_regressor("catboost").model_params)
    assert ours.pop("boost_from_average") is False
    assert "boost_from_average" not in theirs
    assert ours == theirs


# --------------------------------------------------------------------------- #
# search spaces                                                                #
# --------------------------------------------------------------------------- #
def legacy_suggester(variant: str):
    """The legacy ``_suggest_*_params`` bound the way the tuning loop binds it."""
    family, _, objective_kind = variant.partition("_")
    if objective_kind:  # count: the suggester takes the objective kind
        fn = legacy_count.SUGGESTERS_BY_FAMILY[family]
        return lambda trial: fn(trial, objective_kind)
    return legacy_diff.SUGGESTERS_BY_FAMILY[family]


ALL_VARIANTS = [*gbm.COUNT_GBM_VARIANTS, *gbm.DIFF_GBM_VARIANTS]


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_search_space_matches_legacy_on_a_fixed_trial(variant: str):
    """Same param names and values for a ``FixedTrial`` with the same inputs."""
    import optuna

    values = {
        "num_leaves": 40,
        "max_depth": 6,
        "min_child_samples": 99,
        "min_child_weight": 4,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "subsample": 0.77,
        "colsample_bytree": 0.83,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "depth": 5,
        "iterations": 500,
        "l2_leaf_reg": 3.0,
        "tweedie_variance_power": 1.42,
    }
    spec = get_spec(variant)
    assert spec.search_space is not None
    ours = spec.search_space(optuna.trial.FixedTrial(dict(values)))
    theirs = legacy_suggester(variant)(optuna.trial.FixedTrial(dict(values)))
    assert ours == theirs


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_search_space_matches_legacy_over_three_seeded_trials(variant: str):
    """Same distributions AND same drawn values for a seeded TPE sampler."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = get_spec(variant)
    assert spec.search_space is not None
    legacy = legacy_suggester(variant)

    def run(fn):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        seen: list[dict[str, Any]] = []

        def objective(trial):
            seen.append(fn(trial))
            return float(len(seen))

        study.optimize(objective, n_trials=3)
        return seen, [
            (t.params, {k: str(v) for k, v in t.distributions.items()}) for t in study.trials
        ]

    ours_params, ours_trials = run(spec.search_space)
    theirs_params, theirs_trials = run(legacy)
    assert ours_params == theirs_params
    assert ours_trials == theirs_trials
    # the search space is non-trivial: at least the shared knobs were drawn
    assert len(ours_params) == 3
    assert ours_params[0]


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_tweedie_power_only_in_the_tweedie_search_spaces(variant: str):
    import optuna

    spec = get_spec(variant)
    assert spec.search_space is not None
    drawn = spec.search_space(
        optuna.trial.FixedTrial(
            {
                "num_leaves": 40,
                "max_depth": 6,
                "min_child_samples": 99,
                "min_child_weight": 4,
                "learning_rate": 0.05,
                "n_estimators": 600,
                "subsample": 0.77,
                "colsample_bytree": 0.83,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "depth": 5,
                "iterations": 500,
                "l2_leaf_reg": 3.0,
                "tweedie_variance_power": 1.42,
            }
        )
    )
    assert ("tweedie_variance_power" in drawn) is variant.endswith("_tweedie")
