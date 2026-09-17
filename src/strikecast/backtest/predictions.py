"""The long prediction frame every backtest produces and every evaluation reads.

:class:`PredictionSet` is a thin, validating wrapper around a long pandas
dataframe. It is the ONLY thing a backtest persists (plan §5.2); no darts
object is ever pickled.

Columns, in this exact order::

    region  fold  horizon  date  y_true  y_pred  origin_date  channel

The first six are exactly what the legacy
``src/evaluation_tools.py::collect_predictions_long`` produces, in exactly the
legacy row order (region-major, then fold, then horizon). Two columns are
appended:

``origin_date``
    ``pred.time_index[0]``, the first forecast date of the fold. The legacy
    frame identifies a fold only by ``fold``, a 0-based counter local to one
    region and one run, which makes cross-model joins and the block bootstrap
    of plan §7.1 impossible to express. ``origin_date`` is derivable from
    ``fold`` given the schedule (``fold`` k starts at ``t0 = start_idx +
    k * predict_stride``, so ``origin_date == time_index[t0]``), so adding it
    changes nothing and loses nothing. Note it is the first PREDICTED date, not
    the cutoff-minus-one; with the legacy ``drop_after(time_index[t0])``
    semantics the training/context window ends at ``time_index[t0 - 1]`` and
    the first predicted date is ``time_index[t0]``, which is also
    :attr:`strikecast.backtest.protocols.Fold.cutoff`.

``channel``
    Which output of the forecaster this row belongs to. Single-output models
    use :data:`~strikecast.backtest.protocols.SINGLE_CHANNEL` (``"y_pred"``);
    composite forecasters (hurdle: ``prob`` / ``count`` / ``hurdle``; damage:
    one channel per damage key) write one block per channel.

Behaviour-preservation notes (plan §6, level E):

* ``Q1`` :meth:`PredictionSet.legacy_frame` returns the six legacy columns in
  the legacy row order, so a stored ``predictions_long_*.parquet`` or a fresh
  ``collect_predictions_long`` call can be compared with
  ``pandas.testing.assert_frame_equal(check_dtype=True)``.
* ``Q2`` the ``if t in actual_map`` filter of the legacy collector is
  reproduced verbatim: forecast steps whose timestamp is past the end of the
  actual series are DROPPED, so the last folds of a run contribute fewer than
  ``horizon`` rows. (In practice the legacy schedule stops at
  ``n_total - horizon + 1`` so the filter never fires for a single-resolution
  run; it does fire whenever the prediction list is built against a shorter
  actual series, which is why it is kept.)
* ``Q3`` ``_chronos2.py::chronos2_rolling_long`` builds the same six columns
  but by an ``item_id``/``timestamp`` merge followed by
  ``sort_values(["item_id", "timestamp"])`` per fold, so its rows come out
  FOLD-major then region-major, the transpose of the legacy collector's
  region-major then fold order. It also has no ``actual_map`` membership test
  (the truth frame is the same slice it predicted on, so the join does the
  filtering) and its ``region`` values are AutoGluon ``item_id``s. Row order is
  irrelevant to every consumer (``evaluate_long`` only groups), but it matters
  to ``assert_frame_equal``, which is why comparisons go through
  :meth:`legacy_frame` and, for the Chronos family, need a sort first.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from .protocols import SINGLE_CHANNEL

if TYPE_CHECKING:
    from darts import TimeSeries

__all__ = [
    "COLUMNS",
    "LEGACY_COLUMNS",
    "PredictionSet",
]

#: The six columns ``collect_predictions_long`` produces, in its order.
LEGACY_COLUMNS: tuple[str, ...] = ("region", "fold", "horizon", "date", "y_true", "y_pred")

#: The full :class:`PredictionSet` schema.
COLUMNS: tuple[str, ...] = (*LEGACY_COLUMNS, "origin_date", "channel")

_EMPTY_DTYPES: Mapping[str, str] = {
    "region": "str",
    "fold": "int64",
    "horizon": "int64",
    "date": "datetime64[ns]",
    "y_true": "float64",
    "y_pred": "float64",
    "origin_date": "datetime64[ns]",
    "channel": "str",
}


def _empty_frame() -> pd.DataFrame:
    """An empty frame with the full schema and the dtypes a filled frame gets.

    The legacy collector returns ``pd.DataFrame([])`` for an empty run, which
    has NO columns at all. That is unusable as a typed container, so the empty
    :class:`PredictionSet` is explicitly typed instead. This is the one place
    where the new frame differs from the legacy one, and only in the
    zero-row case (see the report flag on empty runs).
    """
    return pd.DataFrame({name: pd.Series([], dtype=dtype) for name, dtype in _EMPTY_DTYPES.items()})


def _rows_for_channel(
    actuals: Sequence[TimeSeries],
    fold_preds_list: Sequence[Sequence[TimeSeries]],
    region_names: Sequence[str],
    channel: str,
) -> list[dict[str, Any]]:
    """``collect_predictions_long``'s body, verbatim, plus the two new columns.

    Copied from ``src/evaluation_tools.py`` lines 242-260. The loop order
    (region, then fold, then horizon), the ``enumerate(..., start=1)`` horizon
    numbering, the ``if t in actual_map`` filter and the ``float()`` casts are
    all exactly the legacy ones.
    """
    rows: list[dict[str, Any]] = []
    for r_idx, (actual_ts, fold_preds) in enumerate(
        zip(actuals, fold_preds_list, strict=False)
    ):
        region = region_names[r_idx]
        actual_map = dict(zip(actual_ts.time_index, actual_ts.values().ravel(), strict=False))
        for f_idx, pred in enumerate(fold_preds):
            pred_values = pred.values().ravel()
            origin_date = pred.time_index[0]
            for h, (t, y_pred) in enumerate(
                zip(pred.time_index, pred_values, strict=False), start=1
            ):
                if t in actual_map:
                    rows.append(
                        {
                            "region": region,
                            "fold": f_idx,
                            "horizon": h,
                            "date": t,
                            "y_true": float(actual_map[t]),
                            "y_pred": float(y_pred),
                            "origin_date": origin_date,
                            "channel": channel,
                        }
                    )
    return rows


class PredictionSet:
    """A validated long prediction frame with :data:`COLUMNS`."""

    __slots__ = ("_frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        if list(frame.columns) != list(COLUMNS):
            raise ValueError(
                f"PredictionSet expects columns {list(COLUMNS)}, got {list(frame.columns)}"
            )
        self._frame = frame

    # -- construction -------------------------------------------------------

    @classmethod
    def from_fold_preds(
        cls,
        actuals: Sequence[TimeSeries],
        fold_preds: Mapping[str, Sequence[Sequence[TimeSeries]]]
        | Sequence[Sequence[TimeSeries]],
        region_names: Sequence[str],
    ) -> PredictionSet:
        """Flatten per-region, per-fold predictions into the long frame.

        ``fold_preds`` is either a ``{channel: [[pred per fold] per region]}``
        mapping (what a composite forecaster's run produces) or the bare
        ``[[pred per fold] per region]`` list the legacy runners return, which
        is treated as the single channel
        :data:`~strikecast.backtest.protocols.SINGLE_CHANNEL`.

        Per channel the result is byte-for-byte the legacy
        ``collect_predictions_long`` frame (see :meth:`legacy_frame`); channel
        blocks are concatenated in the mapping's own iteration order.
        """
        by_channel: Mapping[str, Sequence[Sequence[TimeSeries]]]
        by_channel = (
            fold_preds
            if isinstance(fold_preds, Mapping)
            else {SINGLE_CHANNEL: fold_preds}
        )

        rows: list[dict[str, Any]] = []
        for channel, per_region in by_channel.items():
            rows.extend(_rows_for_channel(actuals, per_region, region_names, channel))
        if not rows:
            return cls(_empty_frame())
        return cls(pd.DataFrame(rows))

    @classmethod
    def from_parquet(cls, path: str | Path) -> PredictionSet:
        """Read a set written by :meth:`to_parquet`."""
        return cls(pd.read_parquet(Path(path)))

    @classmethod
    def concat(cls, sets: Iterable[PredictionSet]) -> PredictionSet:
        """Stack sets in the given order, renumbering the index only."""
        frames = [ps.frame for ps in sets]
        if not frames:
            return cls(_empty_frame())
        return cls(pd.concat(frames, ignore_index=True))

    # -- accessors ----------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """The full long frame, with :data:`COLUMNS`."""
        return self._frame

    @property
    def channels(self) -> tuple[str, ...]:
        """Channel names in order of first appearance."""
        return tuple(dict.fromkeys(self._frame["channel"].tolist()))

    def for_channel(self, name: str) -> PredictionSet:
        """The rows of one channel, as a :class:`PredictionSet`."""
        if name not in self.channels:
            raise KeyError(f"no channel {name!r}; have {self.channels}")
        sub = cast(pd.DataFrame, self._frame[self._frame["channel"] == name])
        return PredictionSet(sub.reset_index(drop=True))

    def legacy_frame(self, channel: str = SINGLE_CHANNEL) -> pd.DataFrame:
        """The six legacy columns of one channel, in legacy row order.

        This is what a level-E comparison against ``collect_predictions_long``
        or a stored ``predictions_long_*.parquet`` compares.
        """
        if channel not in self.channels:
            raise KeyError(f"no channel {channel!r}; have {self.channels}")
        sub = cast(pd.DataFrame, self._frame[self._frame["channel"] == channel])
        return sub.loc[:, list(LEGACY_COLUMNS)].reset_index(drop=True)

    # -- persistence --------------------------------------------------------

    def to_parquet(self, path: str | Path) -> Path:
        """Write the frame atomically (temp file in the same dir + ``os.replace``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        self._frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return path

    # -- test support -------------------------------------------------------

    def assert_equal_legacy(self, long_df: pd.DataFrame, channel: str = SINGLE_CHANNEL) -> None:
        """Assert :meth:`legacy_frame` equals a legacy long frame, dtypes included."""
        pd.testing.assert_frame_equal(
            self.legacy_frame(channel), long_df, check_dtype=True, obj="legacy_frame"
        )

    # -- dunders ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return f"PredictionSet(rows={len(self._frame)}, channels={list(self.channels)})"
