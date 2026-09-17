"""Model adapters for the expanding-window backtest engine.

The darts adapters (global and local) and the composite forecasters (hurdle,
damage) live here. The model registry and the Chronos-2 fixed predictor land in
later phases of ``docs/REFACTOR_PLAN.md`` (P3 and P5).
"""

from .adapters import GlobalDartsForecaster, LocalDartsForecaster
from .composite import HurdleForecaster, MultiTargetClassifierForecaster

__all__ = [
    "GlobalDartsForecaster",
    "HurdleForecaster",
    "LocalDartsForecaster",
    "MultiTargetClassifierForecaster",
]
