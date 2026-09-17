"""Future vs past covariate split.

Behaviour-preserving port of
``src/prevalent_functions.py::split_future_and_past_cov``. Prints became
``logging`` calls; the returned lists and their ordering are identical.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from ._legacy_helpers import get_all_X

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_EXCLUDE_COLS", "CovariateSplit", "split_covariates"]

#: The legacy default exclusion set. Note it is a ``set``, so ``exclude_cols``
#: has no meaningful order; it is only ever used for membership tests.
DEFAULT_EXCLUDE_COLS = frozenset(
    {"Activity_Level", "index", "level_0", "region", "event_date"}
)


@dataclass(frozen=True)
class CovariateSplit:
    """The legacy 4-tuple ``(holiday_cols, future_covariates, exclude_cols, past_covariates)``."""

    holiday_cols: list[str]
    future_covariates: list[str]
    exclude_cols: set[str]
    past_covariates: list[str]

    def __iter__(self):
        """Allow tuple-unpacking exactly like the legacy return value."""
        return iter(
            (self.holiday_cols, self.future_covariates, self.exclude_cols, self.past_covariates)
        )


def split_covariates(
    panel: pd.DataFrame,
    global_weather_columns: list[str],
    target: str,
    exclude: Iterable[str] | None = None,
) -> CovariateSplit:
    """Split the panel's columns into future and past covariates.

    Parameters
    ----------
    panel:
        Either the reset-index panel (``for_global_reset``) or the
        MultiIndexed one -- ``_chronos2.py`` passes the latter. Only
        ``panel.columns`` is read, so both work.
    global_weather_columns:
        From :class:`strikecast.data.panel.PanelResult`. Appended verbatim to
        the holiday columns, so its (process-dependent, see panel quirk Q2)
        ordering carries through into ``future_covariates``.
    target:
        Added to the default exclusion set.
    exclude:
        Replaces the default exclusion set entirely when truthy. Legacy quirk:
        an empty collection is falsy, so ``exclude=[]`` or ``exclude=set()``
        silently falls back to the default set rather than excluding nothing.
        Preserved.

    Legacy quirks preserved
    -----------------------
    * ``future_covariates`` may name columns that are not in ``panel`` at all:
      ``global_weather_columns`` are *de-suffixed* names built from the wide
      master frame (``env_t_max_kyiv`` -> ``env_t_max``) and the pivot turns
      them into exactly those names, but nothing checks the correspondence.
    * ``past_covariates`` is everything else in column order, including
      derived/aggregated columns, and excluding nothing beyond
      ``future_covariates`` and ``exclude_cols``.
    """
    holiday_cols = get_all_X("holiday", panel)
    future_covariates = holiday_cols + global_weather_columns

    if not exclude:
        exclude_cols = {target, *DEFAULT_EXCLUDE_COLS}
    else:
        exclude_cols = set(exclude)

    past_covariates = [
        c
        for c in panel.columns
        if c not in future_covariates and c not in exclude_cols
    ]

    logger.info("# future covariates : %s", len(future_covariates))
    logger.info("# past covariates   : %s", len(past_covariates))
    return CovariateSplit(
        holiday_cols=holiday_cols,
        future_covariates=future_covariates,
        exclude_cols=exclude_cols,
        past_covariates=past_covariates,
    )
