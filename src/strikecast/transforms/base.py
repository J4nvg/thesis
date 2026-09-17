"""Base class and factory for :class:`strikecast.backtest.protocols.TargetTransform`.

A target transform maps level-space targets into the space the model is trained
and predicted in, and maps a model-space prediction back to level space.

The contract is frozen in ``strikecast/backtest/protocols.py``:

* ``forward(level_series)`` is called ONCE per backtest on the full level-space
  list and returns the model-space list the engine schedules and slices on.
* ``inverse(pred, context)`` is called per fold per region, with ``context``
  being the FULL level-space series of that region (the legacy diff runners
  pass ``target_level_list[r_idx]``, never a sliced view).

Implementations here are deliberately tiny: they are verbatim ports of what the
legacy scripts do inline, and they must not "improve" on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from darts import TimeSeries

__all__ = ["BaseTargetTransform", "get_transform"]


class BaseTargetTransform(ABC):
    """Structural base for the two concrete transforms.

    Subclasses set the class attribute :attr:`name` and implement
    :meth:`forward` and :meth:`inverse`. Instances satisfy
    ``isinstance(obj, protocols.TargetTransform)``.
    """

    #: Registry key, also what the config/run manifest records.
    name: str = ""

    @abstractmethod
    def forward(self, level_series: list[TimeSeries]) -> list[TimeSeries]:
        """Level space -> model space, for the whole region list at once."""

    @abstractmethod
    def inverse(self, pred: TimeSeries, context: TimeSeries) -> TimeSeries:
        """Model space -> level space for one region's fold prediction.

        ``context`` is the FULL level-space series of that region.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"{type(self).__name__}(name={self.name!r})"


def get_transform(name: str) -> BaseTargetTransform:
    """Return the transform registered under ``name``.

    Known names are ``"identity"`` (the count and hurdle families) and
    ``"diff"`` (the differenced-regression branch and the ARIMA baseline that
    lives inside it). Lookup is case-insensitive on purpose: the legacy code
    lower-cases every model/family name it is handed.
    """
    from .diff import Diff  # noqa: PLC0415  (avoids an import cycle)
    from .identity import Identity  # noqa: PLC0415

    key = name.strip().lower()
    registry: dict[str, type[BaseTargetTransform]] = {
        Identity.name: Identity,
        Diff.name: Diff,
    }
    try:
        cls = registry[key]
    except KeyError:
        known = ", ".join(sorted(registry))
        raise KeyError(f"unknown target transform {name!r}; known: {known}") from None
    return cls()
