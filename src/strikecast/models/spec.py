"""Model registry contract (Phase 3).

One :class:`ModelSpec` per model variant the thesis ran. A spec is data: a name,
a family, how to build the darts model from parameters and a
:class:`RunContext`, the legacy default parameters, the Optuna search space,
and the flags the pipeline needs to schedule it. Registering a spec is one
call; the experiment YAML then lists names.

Rules, mirroring ``docs/REFACTOR_PLAN.md`` §5.2 and §5.4:

* ``build(params, ctx)`` returns exactly the object the legacy builder returned
  for the same parameters (``build_regressor``, ``build_gbm_from_params``,
  ``build_nn_from_params``, ``_build_lstm_from_best``, ``get_event_classifier``,
  ``get_count_regressor``, ``get_damage_classifier`` and the diff-script
  twins). Device, thread count and seed come from ``ctx`` instead of module
  globals, with the SAME values the thesis ran (F9, §2.3 "Device" row):
  count-family GBDTs on CPU, diff-branch XGBoost ``cuda`` / CatBoost ``GPU``,
  LightGBM always CPU, hurdle CatBoost CPU.
* ``defaults`` are the untuned parameters of the legacy default branch, used
  by the feature-selection pass and as the fallback lineup.
* ``search_space(trial)`` reproduces the legacy ``_suggest_*`` function for
  the variant and returns the params dict a trial builds with.
* ``from_best_params(best)`` maps a stored ``best_params.json`` payload (see
  ``golden/converted/tuning/*/best_params.json``) to the params ``build``
  expects, reproducing ``build_gbm_from_params`` / ``_build_lstm_from_best``.
* ``stochastic`` drives the seed sweep (§7.1): deterministic models run once
  and are broadcast across seeds.
* ``kind`` selects the forecaster adapter: ``global`` ->
  ``GlobalDartsForecaster``, ``local`` -> ``LocalDartsForecaster`` (with
  ``fallback`` where the legacy runner had one, F5), ``naive`` ->
  ``strikecast.backtest.naive`` (``build`` returns ``None``), ``composite`` ->
  the family's composite forecaster.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ModelKind",
    "ModelSpec",
    "RunContext",
    "all_specs",
    "get_spec",
    "register",
    "registered_experiments",
    "registered_names",
]

ModelKind = Literal["global", "local", "naive", "composite"]


@dataclass(frozen=True, slots=True)
class RunContext:
    """Per-run values the legacy code read from module globals.

    ``device`` is the family-specific string the legacy builder passed
    (``"cpu"``, ``"gpu"``, ``"cuda"``, ``"GPU"``, ``"CPU"``); the spec decides
    which keyword it maps to. ``threads`` is ``available_threads`` where the
    legacy builder used it (``None`` where it was commented out, so the
    library default applies, exactly as before).
    """

    seed: int = 42
    device: str = "cpu"
    threads: int | None = None


def _identity_params(best: Mapping[str, Any]) -> dict[str, Any]:
    return dict(best)


@dataclass(frozen=True)
class ModelSpec:
    """Everything the pipeline needs to know about one model variant."""

    name: str
    family: str
    kind: ModelKind
    build: Callable[[Mapping[str, Any], RunContext], Any]
    experiments: tuple[str, ...]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    search_space: Callable[[Any], dict[str, Any]] | None = None
    from_best_params: Callable[[Mapping[str, Any]], dict[str, Any]] = _identity_params
    is_neural: bool = False
    stochastic: bool = True
    needs_raw_past_covs: bool = False
    fallback: Callable[[], Any] | None = None
    n_trials: int | None = None
    device: str = "cpu"
    threads_from_context: bool = False
    tags: tuple[str, ...] = ()

    @property
    def tunable(self) -> bool:
        return self.search_space is not None


_REGISTRY: dict[str, list[ModelSpec]] = {}


def register(spec: ModelSpec, *, replace: bool = False) -> ModelSpec:
    """Add ``spec`` under ``spec.name``.

    The same name may be registered once per experiment family, because the
    legacy scripts reused names across families with different builders
    (``lstm_w7`` is an MSE-on-levels RNN in the count family and an
    MSE-on-differences RNN in the diff family). Two specs with the same name
    must therefore have disjoint ``experiments``; overlapping ones raise
    unless ``replace`` is set.
    """
    if not spec.experiments:
        raise ValueError(f"model {spec.name!r} declares no experiments")
    entries = _REGISTRY.setdefault(spec.name, [])
    for i, existing in enumerate(entries):
        overlap = set(existing.experiments) & set(spec.experiments)
        if overlap:
            if not replace:
                raise KeyError(
                    f"model {spec.name!r} is already registered for {sorted(overlap)}"
                )
            entries[i] = spec
            return spec
    entries.append(spec)
    return spec


def get_spec(name: str, experiment: str | None = None) -> ModelSpec:
    """Look up a spec by name, and by experiment family when the name is
    shared between families."""
    entries = _REGISTRY.get(name)
    if not entries:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    if experiment is not None:
        for spec in entries:
            if experiment in spec.experiments:
                return spec
        raise KeyError(
            f"model {name!r} is not registered for experiment {experiment!r}; "
            f"available for {[s.experiments for s in entries]}"
        )
    if len(entries) > 1:
        raise KeyError(
            f"model {name!r} is registered for several experiments "
            f"{[s.experiments for s in entries]}; pass experiment="
        )
    return entries[0]


def registered_names(experiment: str | None = None) -> list[str]:
    """Names in registration order, optionally filtered to one experiment
    family (``"count"``, ``"diff"``, ``"hurdle"``, ``"damage"``)."""
    if experiment is None:
        return list(_REGISTRY)
    return [
        name
        for name, entries in _REGISTRY.items()
        if any(experiment in s.experiments for s in entries)
    ]


def all_specs(experiment: str | None = None) -> list[ModelSpec]:
    """Every registered spec, in registration order.

    Filtered to one experiment family when ``experiment`` is given. A name that
    is registered for several families (``lstm_w7`` in ``count`` and ``diff``)
    contributes one spec per family to the unfiltered list, because the two are
    different models (see :func:`register`).
    """
    out: list[ModelSpec] = []
    for entries in _REGISTRY.values():
        for spec in entries:
            if experiment is None or experiment in spec.experiments:
                out.append(spec)
    return out


def registered_experiments() -> list[str]:
    """Every experiment family that has at least one spec, sorted."""
    return sorted({e for entries in _REGISTRY.values() for s in entries for e in s.experiments})
