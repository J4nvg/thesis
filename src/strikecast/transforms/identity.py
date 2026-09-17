"""No-op target transform: the model trains and predicts in level space.

Used by the count family (``_regression_GBDT.py`` / ``_regression_LSTM.py`` via
``src/evaluation_tools.py``), the hurdle notebook and Chronos-2, all of which
never touch the target before fitting and never post-process the prediction
index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseTargetTransform

if TYPE_CHECKING:
    from darts import TimeSeries

__all__ = ["Identity"]


class Identity(BaseTargetTransform):
    """``forward`` copies the list, ``inverse`` returns ``pred`` unchanged."""

    name = "identity"

    def forward(self, level_series: list[TimeSeries]) -> list[TimeSeries]:
        """Return a NEW list holding the SAME ``TimeSeries`` objects.

        A new list so the engine can never alias the caller's container; the
        same objects because darts ``TimeSeries`` are immutable and the legacy
        count runners literally pass ``target_for_cv`` straight through.
        """
        return list(level_series)

    def inverse(self, pred: TimeSeries, context: TimeSeries) -> TimeSeries:
        """Return ``pred`` itself. ``context`` is accepted and ignored."""
        return pred
