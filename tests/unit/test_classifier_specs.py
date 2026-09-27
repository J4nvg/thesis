"""The hurdle / damage registry entries build exactly the legacy objects.

Compared against ``tests/legacy_ref/hurdle_builders.py``, the verbatim copy of
``final_hurdle.ipynb`` cell 9 and ``damage_classifier.ipynb`` cell 10, on two
axes:

* the darts wrapper's constructor kwargs (``model_params``), which carry the
  ``get_common_kwargs()`` skeleton and, for CatBoost, the loss and device
  settings;
* the inner estimator's ``get_params()``, which carries the SPE / CatBoost
  hyper-parameters and the seed.

``models/spec.py`` is FROZEN, so nothing here touches it beyond
``get_spec``/``registered_names``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import shim for `tests.`
    sys.path.insert(0, str(REPO_ROOT))

from tests.legacy_ref import hurdle_builders as legacy  # noqa: E402

from strikecast.data.feature_selection import legacy_common_kwargs  # noqa: E402
from strikecast.models.classifiers import (  # noqa: E402
    DamageBuilders,
    HurdleBuilders,
    build_count_regressor,
    build_damage_classifier,
    build_event_classifier,
)
from strikecast.models.spec import RunContext, get_spec, registered_names  # noqa: E402

#: ``available_threads = get_available_threads()`` is machine dependent (F110),
#: so every comparison pins one value on both sides instead of reading the
#: machine's.
THREADS = 8
SEED = 42


def _legacy_kwargs(threads=THREADS, seed=SEED):
    """The three notebook globals, as ``hurdle_builders`` takes them."""
    return {
        "RANDOM_STATE": seed,
        "available_threads": threads,
        "COMMON_KWARGS": legacy_common_kwargs(),
    }


def _wrapper_kwargs(model):
    """``model_params`` minus the estimator itself, which is compared apart."""
    return {k: v for k, v in dict(model.model_params).items() if k != "model"}


def _spe_params(model):
    """``get_params()`` of the SPE, with the inner tree expanded.

    sklearn estimators compare by identity, so the nested ``DecisionTreeClassifier``
    has to be turned into its own parameter dict before the two sides can be
    compared with ``==``.
    """
    params = dict(model.model.get_params(deep=False))
    params["estimator"] = params["estimator"].get_params()
    return params


# --------------------------------------------------------------------------- #
# registry wiring
# --------------------------------------------------------------------------- #
def test_registered_names():
    assert set(registered_names("hurdle")) >= {
        "spe_event_classifier",
        "catboost_tweedie_count_head",
        "hurdle",
    }
    assert set(registered_names("damage")) >= {"spe_damage_classifier", "damage"}


@pytest.mark.parametrize(
    ("name", "kind", "family"),
    [
        ("spe_event_classifier", "global", "hurdle"),
        ("catboost_tweedie_count_head", "global", "hurdle"),
        ("spe_damage_classifier", "global", "damage"),
        ("hurdle", "composite", "hurdle"),
        ("damage", "composite", "damage"),
    ],
)
def test_spec_flags(name, kind, family):
    spec = get_spec(name)
    assert spec.kind == kind
    assert spec.family == family
    # F7: neither family was ever tuned.
    assert spec.search_space is None
    assert spec.tunable is False
    assert spec.n_trials is None
    # SPE undersampling and CatBoost subsampling are seeded but stochastic.
    assert spec.stochastic is True
    assert spec.is_neural is False
    assert spec.threads_from_context is True


def test_count_head_device_is_cpu():
    # Plan sec. 2.3 "Device" row: the hurdle CatBoost ran on CPU.
    assert get_spec("catboost_tweedie_count_head").device == "CPU"


# --------------------------------------------------------------------------- #
# SPE classifiers vs the verbatim builders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("build_new", "build_legacy"),
    [
        (build_event_classifier, legacy.get_event_classifier),
        (build_damage_classifier, legacy.get_damage_classifier),
    ],
    ids=["event", "damage"],
)
@pytest.mark.parametrize("threads", [THREADS, None])
def test_spe_classifier_matches_legacy(build_new, build_legacy, threads):
    new = build_new({}, RunContext(seed=SEED, threads=threads))
    old = build_legacy(**_legacy_kwargs(threads=threads))

    assert type(new) is type(old)
    assert _wrapper_kwargs(new) == _wrapper_kwargs(old)

    new_params = new.model.get_params(deep=False)
    old_params = old.model.get_params(deep=False)
    assert new_params.keys() == old_params.keys()
    for key in new_params:
        if key == "estimator":
            continue
        assert new_params[key] == old_params[key], key

    # the notebook's explicit SPE settings
    assert new_params["n_estimators"] == 100
    assert new_params["random_state"] == SEED
    assert new_params["n_jobs"] == threads

    # and the inner decision tree
    assert new.model.estimator.get_params() == old.model.estimator.get_params()
    assert new.model.estimator.get_params()["max_depth"] == 5
    assert new.model.estimator.get_params()["random_state"] == SEED


@pytest.mark.parametrize(
    ("build_new", "build_legacy"),
    [
        (build_event_classifier, legacy.get_event_classifier),
        (build_damage_classifier, legacy.get_damage_classifier),
    ],
    ids=["event", "damage"],
)
def test_spe_classifier_honours_the_two_kwargs(build_new, build_legacy):
    params = {"DTm_depth": 3, "SPE_estm": 7}
    new = build_new(params, RunContext(seed=SEED, threads=THREADS))
    old = build_legacy(**_legacy_kwargs(), **params)
    assert new.model.get_params(deep=False)["n_estimators"] == 7
    assert new.model.estimator.get_params()["max_depth"] == 3
    assert new.model.get_params(deep=False)["n_estimators"] == old.model.get_params(deep=False)[
        "n_estimators"
    ]
    assert new.model.estimator.get_params() == old.model.estimator.get_params()


def test_spe_classifier_ignores_unknown_params():
    """The notebook reads only two keys out of ``**kwargs`` and drops the rest."""
    new = build_event_classifier(
        {"nonsense": 1}, RunContext(seed=SEED, threads=THREADS)
    )
    old = legacy.get_event_classifier(**_legacy_kwargs(), nonsense=1)
    assert _wrapper_kwargs(new) == _wrapper_kwargs(old)
    assert _spe_params(new) == _spe_params(old)


def test_spe_classifier_seed_reaches_both_places():
    new = build_event_classifier({}, RunContext(seed=7, threads=THREADS))
    old = legacy.get_event_classifier(**_legacy_kwargs(seed=7))
    assert new.model.get_params(deep=False)["random_state"] == 7
    assert new.model.estimator.get_params()["random_state"] == 7
    assert old.model.get_params(deep=False)["random_state"] == 7


# --------------------------------------------------------------------------- #
# CatBoost count head vs the verbatim builder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("threads", [THREADS, None])
def test_count_regressor_matches_legacy(threads):
    new = build_count_regressor({}, RunContext(seed=SEED, threads=threads))
    old = legacy.get_count_regressor(**_legacy_kwargs(threads=threads))

    assert type(new) is type(old)
    assert dict(new.model_params) == dict(old.model_params)
    assert new.model.get_params() == old.model.get_params()

    params = new.model.get_params()
    assert params["loss_function"] == "Tweedie:variance_power=1.5"
    assert params["boost_from_average"] is False
    assert params["task_type"] == "CPU"
    assert params["random_seed"] == SEED
    # CatBoost drops `None` kwargs from `get_params()`, on both sides alike.
    assert params.get("thread_count") == threads


def test_count_regressor_forwards_extra_kwargs():
    """Unlike the classifier builders, this one passes ``**kwargs`` through."""
    new = build_count_regressor({"iterations": 11}, RunContext(seed=SEED, threads=THREADS))
    old = legacy.get_count_regressor(**_legacy_kwargs(), iterations=11)
    assert dict(new.model_params) == dict(old.model_params)
    assert new.model.get_params()["iterations"] == 11


def test_count_regressor_ignores_context_device():
    """The notebook hard-codes ``task_type="CPU"``; ``ctx.device`` must not win."""
    new = build_count_regressor({}, RunContext(seed=SEED, device="GPU", threads=THREADS))
    assert new.model.get_params()["task_type"] == "CPU"


# --------------------------------------------------------------------------- #
# composite entries
# --------------------------------------------------------------------------- #
def test_hurdle_composite_returns_two_fresh_builders():
    ctx = RunContext(seed=SEED, threads=THREADS)
    builders = get_spec("hurdle").build(get_spec("hurdle").defaults, ctx)
    assert isinstance(builders, HurdleBuilders)

    clf_builder, reg_builder = builders  # HurdleForecaster(*builders, ...)
    clf, reg = clf_builder(), reg_builder()

    old_clf = legacy.get_event_classifier(**_legacy_kwargs())
    old_reg = legacy.get_count_regressor(**_legacy_kwargs())
    assert _wrapper_kwargs(clf) == _wrapper_kwargs(old_clf)
    assert _spe_params(clf) == _spe_params(old_clf)
    assert dict(reg.model_params) == dict(old_reg.model_params)
    assert reg.model.get_params() == old_reg.model.get_params()

    # a fresh object per call, which is what every retrain needs
    assert clf_builder() is not clf
    assert reg_builder() is not reg


def test_damage_composite_returns_one_builder():
    ctx = RunContext(seed=SEED, threads=THREADS)
    builders = get_spec("damage").build(get_spec("damage").defaults, ctx)
    assert isinstance(builders, DamageBuilders)

    (clf_builder,) = builders
    clf = clf_builder()
    old = legacy.get_damage_classifier(**_legacy_kwargs())
    assert _wrapper_kwargs(clf) == _wrapper_kwargs(old)
    assert _spe_params(clf) == _spe_params(old)
    assert clf_builder() is not clf


def test_composite_builders_take_component_params():
    ctx = RunContext(seed=SEED, threads=THREADS)
    builders = get_spec("hurdle").build(
        {"classifier": {"SPE_estm": 3}, "regressor": {"iterations": 5}}, ctx
    )
    clf, reg = (b() for b in builders)
    assert clf.model.get_params(deep=False)["n_estimators"] == 3
    assert reg.model.get_params()["iterations"] == 5


# --------------------------------------------------------------------------- #
# the shared kwargs skeleton
# --------------------------------------------------------------------------- #
def test_common_kwargs_are_the_legacy_ones():
    common = legacy_common_kwargs()
    for model in (
        build_event_classifier({}, RunContext(seed=SEED, threads=THREADS)),
        build_damage_classifier({}, RunContext(seed=SEED, threads=THREADS)),
        build_count_regressor({}, RunContext(seed=SEED, threads=THREADS)),
    ):
        got = dict(model.model_params)
        for key, value in common.items():
            assert got[key] == value, key
        # F26: `lags_future_covariates` must stay a tuple, not become a list.
        assert isinstance(got["lags_future_covariates"], tuple)


def test_builders_do_not_share_the_common_kwargs_dict():
    """The notebook shares one ``COMMON_KWARGS`` dict; a fresh equal one is fine
    only as long as no builder mutates it."""
    first = build_event_classifier({}, RunContext(seed=SEED, threads=THREADS))
    second = build_event_classifier({}, RunContext(seed=SEED, threads=THREADS))
    assert dict(first.model_params)["add_encoders"] == legacy_common_kwargs()["add_encoders"]
    assert _wrapper_kwargs(first) == _wrapper_kwargs(second)
