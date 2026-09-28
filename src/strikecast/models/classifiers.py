"""Registry entries for the hurdle and damage classifier families.

Behaviour-preserving port of the three legacy builder functions that the two
classifier notebooks defined, plus the two composite entries that wire them into
:mod:`strikecast.models.composite`.

Provenance
----------
``final_hurdle.ipynb`` cell 9
    ``get_event_classifier`` -> :data:`spe_event_classifier`;
    ``get_count_regressor``  -> :data:`catboost_tweedie_count_head`.
``damage_classifier.ipynb`` cell 10
    ``get_damage_classifier`` -> :data:`spe_damage_classifier`.

The two feature-selection models defined in the same cells
(``event_classifier_feature_selection``, ``regressor_feature_selection``,
``get_feature_selector_classifier``) are NOT registered here: they are already
configuration in :mod:`strikecast.data.feature_selection`
(``zipoisson_classifier_config`` / ``zipoisson_regressor_config``).

What each spec builds
---------------------
Exactly the object the notebook builder returned for the same parameters. The
only substitutions are the three notebook globals:

==============================  ==================================
notebook global                 replacement
==============================  ==================================
``RANDOM_STATE`` (cell 1, 42)   ``ctx.seed``
``available_threads``           ``ctx.threads``
``COMMON_KWARGS``               ``darts_common_kwargs(ctx)``
==============================  ==================================

:func:`~strikecast.models.spec.darts_common_kwargs` is ``legacy_common_kwargs()``
unless ``ctx.past_lags`` is set (figure protocol, hurdle heads only via
:meth:`~strikecast.models.spec.RunContext.for_head`); damage never sets it.

``COMMON_KWARGS = get_common_kwargs()`` is a single module-level dict in the
notebook, shared by every model built in that session;
:func:`~strikecast.data.feature_selection.legacy_common_kwargs` returns a fresh
equal dict per call. darts copies the mapping into its encoder objects and never
mutates the argument, so the two are equivalent.

Tuning (F7)
-----------
Neither family was ever tuned, so every spec here has ``search_space=None`` and
its ``defaults`` are the literal notebook defaults. ``DTm_depth`` and
``SPE_estm`` exist as parameters only because the notebook's ``**kwargs``
``.get`` calls make them look tunable; no call site ever passed them.

Seeds (``stochastic=True``)
---------------------------
All three are seeded but genuinely stochastic under a seed change:
``SelfPacedEnsembleClassifier`` draws a new under-sample of the majority class
per boosting iteration from ``random_state``, and CatBoost's bootstrap sampling
is driven by ``random_seed``. They are therefore swept across seeds rather than
run once and broadcast (plan sec. 7.1).

Device
------
The hurdle count head is hard-coded to ``task_type="CPU"`` in the notebook, and
that is what every recorded run used (plan sec. 2.3, "Device" row: "hurdle
CatBoost CPU"). ``ctx.device`` is therefore NOT consulted by
:func:`build_count_regressor`; ``ModelSpec.device`` records the legacy string
so the pipeline can still report it. The two SPE specs have no device concept at
all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from darts.models import CatBoostModel, SKLearnClassifierModel
from imbens.ensemble import SelfPacedEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier

from strikecast.models.spec import ModelSpec, RunContext, darts_common_kwargs, register

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "DamageBuilders",
    "HurdleBuilders",
    "build_count_regressor",
    "build_damage_classifier",
    "build_event_classifier",
    "catboost_tweedie_count_head",
    "damage",
    "hurdle",
    "spe_damage_classifier",
    "spe_event_classifier",
]

#: ``final_hurdle.ipynb`` cell 9 / ``damage_classifier.ipynb`` cell 10:
#: ``kwargs.get('DTm_depth', 5)`` and ``kwargs.get('SPE_estm', 100)``.
LEGACY_DT_MAX_DEPTH = 5
LEGACY_SPE_N_ESTIMATORS = 100

#: ``final_hurdle.ipynb`` cell 9, ``get_count_regressor``. The variance power is
#: a literal inside the loss string, not a separate kwarg.
LEGACY_COUNT_LOSS_FUNCTION = "Tweedie:variance_power=1.5"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_event_classifier(
    params: Mapping[str, Any] | None = None, ctx: RunContext | None = None
) -> SKLearnClassifierModel:
    """``final_hurdle.ipynb`` cell 9, ``get_event_classifier``.

    A :class:`~imbens.ensemble.SelfPacedEnsembleClassifier` of depth-5 decision
    trees, wrapped in darts' :class:`SKLearnClassifierModel` with the shared
    ``get_common_kwargs()`` skeleton.

    ``params`` reproduces the notebook's ``**kwargs``: only ``DTm_depth`` and
    ``SPE_estm`` are read, every other key is IGNORED rather than forwarded to
    the model -- the notebook's ``kwargs.get(...)`` calls never unpack the rest.
    """
    params = {} if params is None else params
    ctx = RunContext() if ctx is None else ctx
    dt_max_depth = params.get("DTm_depth", LEGACY_DT_MAX_DEPTH)
    spe_n_estimators = params.get("SPE_estm", LEGACY_SPE_N_ESTIMATORS)
    clf = SelfPacedEnsembleClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=dt_max_depth,
            random_state=ctx.seed,
        ),
        n_estimators=spe_n_estimators,
        random_state=ctx.seed,
        n_jobs=ctx.threads,
    )
    return SKLearnClassifierModel(model=clf, **darts_common_kwargs(ctx))


def build_damage_classifier(
    params: Mapping[str, Any] | None = None, ctx: RunContext | None = None
) -> SKLearnClassifierModel:
    """``damage_classifier.ipynb`` cell 10, ``get_damage_classifier``.

    Byte-equivalent to :func:`build_event_classifier`: the two notebook bodies
    differ only in whitespace and in the name of the function. They are kept as
    two registry entries because they belong to two different experiments and a
    later change to one must not silently move the other.
    """
    params = {} if params is None else params
    ctx = RunContext() if ctx is None else ctx
    dt_max_depth = params.get("DTm_depth", LEGACY_DT_MAX_DEPTH)
    spe_n_estimators = params.get("SPE_estm", LEGACY_SPE_N_ESTIMATORS)
    clf = SelfPacedEnsembleClassifier(
        estimator=DecisionTreeClassifier(max_depth=dt_max_depth, random_state=ctx.seed),
        n_estimators=spe_n_estimators,
        random_state=ctx.seed,
        n_jobs=ctx.threads,
    )
    return SKLearnClassifierModel(model=clf, **darts_common_kwargs(ctx))


def build_count_regressor(
    params: Mapping[str, Any] | None = None, ctx: RunContext | None = None
) -> CatBoostModel:
    """``final_hurdle.ipynb`` cell 9, ``get_count_regressor``.

    CatBoost with a Tweedie(1.5) loss and ``boost_from_average=False``, fitted
    downstream with positive-only sample weights (F27: the thesis figure calls
    the ">0 only" part an objective; it is the weights).

    Unlike the classifier builders, the notebook forwards ``**kwargs`` straight
    into the ``CatBoostModel`` constructor, so ``params`` is passed through
    here as well. No recorded call site passes anything.
    """
    params = {} if params is None else params
    ctx = RunContext() if ctx is None else ctx
    return CatBoostModel(
        **darts_common_kwargs(ctx),
        loss_function=LEGACY_COUNT_LOSS_FUNCTION,
        boost_from_average=False,
        random_seed=ctx.seed,
        task_type="CPU",
        thread_count=ctx.threads,
        **params,
    )


# --------------------------------------------------------------------------- #
# composite builder bundles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HurdleBuilders:
    """What ``build`` returns for the ``hurdle`` composite spec.

    :class:`~strikecast.models.composite.HurdleForecaster` takes two
    zero-argument builders positionally and calls them afresh at every retrain,
    which is what the notebook loops do (``clf = get_event_classifier()`` /
    ``reg = get_count_regressor()`` inside ``if retrain:``). Iterating this
    object yields the pair in that order, so both spellings work::

        builders = get_spec("hurdle").build({}, ctx)
        HurdleForecaster(*builders, binary_targets=..., ...)
        HurdleForecaster(builders.classifier_builder, builders.regressor_builder, ...)
    """

    classifier_builder: Callable[[], Any]
    regressor_builder: Callable[[], Any]

    def __iter__(self):
        yield self.classifier_builder
        yield self.regressor_builder


@dataclass(frozen=True, slots=True)
class DamageBuilders:
    """What ``build`` returns for the ``damage`` composite spec.

    :class:`~strikecast.models.composite.MultiTargetClassifierForecaster` takes
    ONE zero-argument builder and calls it once per damage key at every retrain.
    The wrapper exists so the damage entry has the same shape as the hurdle one.
    """

    classifier_builder: Callable[[], Any]

    def __iter__(self):
        yield self.classifier_builder


def _build_hurdle(params: Mapping[str, Any], ctx: RunContext) -> HurdleBuilders:
    """Two fresh-per-retrain head builders. Each head gets its own selection
    (figure protocol: ``hurdle_binary`` for the classifier,
    ``hurdle_tweedie_pos`` for the count head), so each builds from
    ``ctx.for_head(<head>)``; with an empty ``ctx.head_past_lags`` both heads
    fall back to the legacy all-lags skeleton."""
    classifier_params = dict(params.get("classifier", {}))
    regressor_params = dict(params.get("regressor", {}))
    classifier_ctx = ctx.for_head("classifier")
    regressor_ctx = ctx.for_head("regressor")
    return HurdleBuilders(
        classifier_builder=lambda: build_event_classifier(classifier_params, classifier_ctx),
        regressor_builder=lambda: build_count_regressor(regressor_params, regressor_ctx),
    )


def _build_damage(params: Mapping[str, Any], ctx: RunContext) -> DamageBuilders:
    classifier_params = dict(params.get("classifier", {}))
    return DamageBuilders(
        classifier_builder=lambda: build_damage_classifier(classifier_params, ctx),
    )


# --------------------------------------------------------------------------- #
# registry entries
# --------------------------------------------------------------------------- #
spe_event_classifier = register(
    ModelSpec(
        name="spe_event_classifier",
        family="hurdle",
        kind="global",
        build=build_event_classifier,
        experiments=("hurdle",),
        defaults={"DTm_depth": LEGACY_DT_MAX_DEPTH, "SPE_estm": LEGACY_SPE_N_ESTIMATORS},
        search_space=None,  # F7: the hurdle family was never tuned
        stochastic=True,
        threads_from_context=True,
        tags=("classifier", "imbalanced", "spe"),
    )
)

catboost_tweedie_count_head = register(
    ModelSpec(
        name="catboost_tweedie_count_head",
        family="hurdle",
        kind="global",
        build=build_count_regressor,
        experiments=("hurdle",),
        defaults={},  # the notebook calls `get_count_regressor()` with no kwargs
        search_space=None,  # F7
        stochastic=True,
        device="CPU",  # the literal `task_type` the notebook passed
        threads_from_context=True,
        tags=("count_head", "catboost", "tweedie"),
    )
)

spe_damage_classifier = register(
    ModelSpec(
        name="spe_damage_classifier",
        family="damage",
        kind="global",
        build=build_damage_classifier,
        experiments=("damage",),
        defaults={"DTm_depth": LEGACY_DT_MAX_DEPTH, "SPE_estm": LEGACY_SPE_N_ESTIMATORS},
        search_space=None,  # F7
        stochastic=True,
        threads_from_context=True,
        tags=("classifier", "imbalanced", "spe"),
    )
)

hurdle = register(
    ModelSpec(
        name="hurdle",
        family="hurdle",
        kind="composite",
        build=_build_hurdle,
        experiments=("hurdle",),
        defaults={
            "classifier": dict(spe_event_classifier.defaults),
            "regressor": dict(catboost_tweedie_count_head.defaults),
        },
        search_space=None,  # F7
        stochastic=True,
        device="CPU",
        threads_from_context=True,
        tags=("composite", "hurdle"),
    )
)

damage = register(
    ModelSpec(
        name="damage",
        family="damage",
        kind="composite",
        build=_build_damage,
        experiments=("damage",),
        defaults={"classifier": dict(spe_damage_classifier.defaults)},
        search_space=None,  # F7
        stochastic=True,
        threads_from_context=True,
        tags=("composite", "damage"),
    )
)
