"""First-order differencing target transform (the ``_diff_regression.py`` branch).

Verbatim port of two pieces of ``_diff_regression.py``:

``forward``
    ``_diff_regression.py`` lines 231-232 and 659-660 (the block is written
    twice, once before and once after feature selection, with identical
    arguments)::

        diff_transformer = Diff(lags=1, dropna=True)
        target_series_diff_list = diff_transformer.fit_transform(target_series_list)

    where ``Diff`` is ``darts.dataprocessing.transformers.Diff``. A fresh
    transformer is built for every call, exactly as the legacy script does.
    ``lags=1, dropna=True`` is identical to ``TimeSeries.diff()`` with its
    defaults (``n=1, periods=1, dropna=True``): verified component-for-component
    on darts 0.43. The result is ONE timestamp shorter and starts one step
    later; static covariates survive (checked empirically, see F54).

``inverse``
    ``_diff_regression.py`` lines 660-674, ``_diff_to_level``, copied
    expression for expression including the ``get_loc(...) - 1`` lookup and the
    ``reshape(-1, 1)``.

Ordering note (legacy, preserved by the engine, not by this class): the legacy
script differences the FULL un-encoded target list and only afterwards splits
it (``get_covs_and_encodings`` -> ``split_before(TRAIN_VAL_END)``). It never
differences an already-sliced CV view. ``forward`` must therefore be handed the
full level list, which is what ``protocols.TargetTransform`` prescribes.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Diff as _DartsDiff

from .base import BaseTargetTransform

__all__ = ["Diff"]


class Diff(BaseTargetTransform):
    """``y'_t = y_t - y_{t-1}``; un-differenced by anchoring on the last actual.

    Parameters mirror the legacy call site and default to it, so
    ``Diff()`` is ``Diff(lags=1, dropna=True)``.
    """

    name = "diff"

    def __init__(self, lags: int = 1, *, dropna: bool = True) -> None:
        self.lags = lags
        self.dropna = dropna

    # -- level space -> diff space ------------------------------------------
    def forward(self, level_series: list[TimeSeries]) -> list[TimeSeries]:
        transformer = _DartsDiff(lags=self.lags, dropna=self.dropna)
        return cast(list[TimeSeries], transformer.fit_transform(list(level_series)))

    # -- diff space -> level space ------------------------------------------
    def inverse(self, pred: TimeSeries, context: TimeSeries) -> TimeSeries:
        """Verbatim ``_diff_regression.py::_diff_to_level`` (lines 660-674).

        ``pred`` is ``diff_pred_ts``, ``context`` is ``level_anchor_ts`` (the
        legacy runners pass ``target_level_list[r_idx]``: the full-length level
        series of the region, NOT a view sliced at the cutoff).

        The ``- 1`` on a ``get_loc`` result is a plain Python index, so when the
        first predicted timestamp is the first timestamp of ``context`` the
        anchor index becomes ``-1`` and numpy wraps to the LAST level value.
        That is legacy behaviour and is preserved; see flag F50.
        """
        diff_pred_ts = pred
        level_anchor_ts = context

        first_pred_time = diff_pred_ts.time_index[0]
        # find the timestamp immediately before the first predicted timestamp
        anchor_idx = cast(Any, level_anchor_ts.time_index.get_loc(first_pred_time)) - 1
        anchor_value = float(level_anchor_ts.values()[anchor_idx, 0])

        diff_vals = diff_pred_ts.values().ravel()
        level_vals = anchor_value + np.cumsum(diff_vals)
        return TimeSeries.from_times_and_values(
            diff_pred_ts.time_index,
            level_vals.reshape(-1, 1),
        )
