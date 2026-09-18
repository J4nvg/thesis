"""Models: the backtest adapters, the composites and the model registry.

``adapters`` holds the two darts forecasters the engine drives, ``composite``
the hurdle and damage forecasters, and ``spec`` the registry contract. The
specs themselves live in ``gbm``, ``rnn``, ``classical`` and ``classifiers``
and register themselves when imported; :mod:`strikecast.models.registry`
imports all four, so that is the module to ask for a spec.

Everything here is resolved LAZILY through ``__getattr__``: importing the
package must not drag in torch, darts, lightgbm, xgboost and catboost for a
caller that only wanted a name. ``from strikecast.models import get_spec``
imports the registry; ``import strikecast.models`` alone imports nothing heavy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point
    from .adapters import GlobalDartsForecaster, LocalDartsForecaster
    from .composite import HurdleForecaster, MultiTargetClassifierForecaster
    from .registry import all_specs, specs_by_experiment
    from .spec import (
        ModelKind,
        ModelSpec,
        RunContext,
        get_spec,
        register,
        registered_experiments,
        registered_names,
    )

__all__ = [
    "GlobalDartsForecaster",
    "HurdleForecaster",
    "LocalDartsForecaster",
    "ModelKind",
    "ModelSpec",
    "MultiTargetClassifierForecaster",
    "RunContext",
    "all_specs",
    "get_spec",
    "register",
    "registered_experiments",
    "registered_names",
    "specs_by_experiment",
]

#: attribute -> the submodule that defines it. Registry lookups go through
#: ``.registry`` rather than ``.spec`` so that every spec module is imported
#: before the registry is read; the contract-only names come from ``.spec``.
_SOURCES: dict[str, str] = {
    "GlobalDartsForecaster": ".adapters",
    "LocalDartsForecaster": ".adapters",
    "HurdleForecaster": ".composite",
    "MultiTargetClassifierForecaster": ".composite",
    "ModelKind": ".spec",
    "ModelSpec": ".spec",
    "RunContext": ".spec",
    "register": ".spec",
    "all_specs": ".registry",
    "get_spec": ".registry",
    "registered_experiments": ".registry",
    "registered_names": ".registry",
    "specs_by_experiment": ".registry",
}


def __getattr__(name: str) -> Any:
    source = _SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # noqa: PLC0415

    value = getattr(import_module(source, __name__), name)
    globals()[name] = value  # resolve once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
