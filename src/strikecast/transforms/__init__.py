"""Target transforms: level space <-> model space.

Implements ``strikecast.backtest.protocols.TargetTransform`` for the two
families in scope (plan §2.3, §5.2): ``identity`` for the count and hurdle
families, ``diff`` for the differenced-regression branch and the ARIMA
baseline that lives inside it.
"""

from .base import BaseTargetTransform, get_transform
from .diff import Diff
from .identity import Identity

__all__ = ["BaseTargetTransform", "Diff", "Identity", "get_transform"]
