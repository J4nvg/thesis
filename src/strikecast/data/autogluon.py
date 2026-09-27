"""``SeriesBundle`` / panel  <->  AutoGluon ``TimeSeriesDataFrame`` (plan §5.1).

The Chronos-2 family (``_chronos2.py``) is the only one that does not speak
darts.  It builds an AutoGluon ``TimeSeriesDataFrame`` straight off the panel,
hands the whole thing to a ``TimeSeriesPredictor`` and reads a quantile column
back out.  This module is that boundary, in both directions:

forward
    :func:`panel_to_long_frames` / :func:`to_timeseries_dataframe` reproduce
    ``_chronos2.py`` §3 (lines 202-218) literally, including which column
    becomes the static feature and which columns survive into the frame.
    :func:`bundle_to_long_frames` is the plan's ``SeriesBundle`` entry point; it
    rebuilds the same frame out of a bundle's **un-encoded** series lists and is
    exact up to the columns a bundle cannot carry (see flag F125 below).

inverse
    :func:`median_column`, :func:`predictions_to_series` and
    :func:`fold_long_frame` turn a predictor's output back into darts series
    for the engine, or into the six legacy columns of
    ``chronos2_rolling_long``; :func:`prediction_set` assembles the
    :class:`~strikecast.backtest.predictions.PredictionSet` long frame.

**AutoGluon is imported lazily**, inside the two functions that construct a
``TimeSeriesDataFrame``.  Importing this module in the main environment (which
cannot install AutoGluon, plan §5.7) is free, and everything except those two
functions runs there: :class:`LongPanel` is a pure-pandas stand-in that
implements exactly the four ``TimeSeriesDataFrame`` members the Chronos code
path uses, so the slicing, the scales and the inverse direction are all
testable without AutoGluon.

Methodology flags raised here (numbers continue ``docs/REFACTOR_PLAN.md`` §4;
F122 was the highest in use)
---------------------------------------------------------------------------

F123
    **The Chronos frame carries the stray ``index`` column as a past
    covariate.**  ``split_future_and_past_cov`` puts ``index`` in
    ``exclude_cols`` (flag F22 records that it "is only excluded downstream"),
    but ``_chronos2.py`` §3 never uses ``past_covariates`` to build the frame:
    it hands AutoGluon *every* column of ``for_global`` except ``region``,
    ``event_date`` and ``Activity_Level``.  AutoGluon treats every column that
    is neither the target nor a known covariate as a past covariate, so
    Chronos-2 receives 84 past covariates, not the 83 that
    ``best_params.json`` records -- the extra one is ``index``, the row number
    of the wide master frame, i.e. a monotone time trend
    (verified on the real panel: 112 frame columns = 1 target + 27 known + 83
    past + ``index``; its first values are 238, 239, 240, ...).  No other
    family sees it.  Preserved; :func:`panel_to_long_frames` keeps it and
    ``drop_columns=`` exists only so a future run can *report* the difference.
    Publication: the covariate count in the Chronos section is 84, not 83.

F124
    **AutoGluon's prediction cache is left on for all 164 rolling folds.**
    ``TimeSeriesPredictor`` defaults to ``cache_predictions=True``, and
    ``chronos2_rolling_long`` calls ``predict`` 164 times with a different
    context each time, so every fold appends its full prediction frame to
    ``<predictor>/models/cached_predictions.pkl`` and rewrites the whole
    pickle -- which is why the stored ``checkpoints/chronos2_best`` is 20 MB.
    The cache is keyed by a hash of ``(data, known_covariates, static)``, so no
    fold ever hits it and the values are identical either way.  The port passes
    ``use_cache=False`` (see :mod:`strikecast.models.chronos`), which changes no
    number and removes the O(n²) write.  It also makes a stored predictor
    loadable at all: the cache path is the *absolute* path of the machine that
    fit it (``/home/u808440/...`` in ``golden/checkpoints/chronos2_best``), so
    the first cached write of a replay crashes with ``OSError: [Errno 45]``.

F125
    **The bundle path cannot reproduce the panel path column-for-column.**
    A ``SeriesBundle`` holds only the target, the future covariates and the
    past covariates that ``split_covariates`` returned; the panel's excluded
    columns (``index``, ``level_0``) are gone.  :func:`bundle_to_long_frames`
    is therefore the right adapter for a *new* Chronos run and NOT the way to
    reproduce the thesis' numbers -- use :func:`panel_to_long_frames` for that.
    :func:`bundle_to_long_frames` raises unless it is told which it is with
    ``strict=``.

F126
    **The frame is built before the split, so the AutoGluon fit window overlaps
    the first rolling fold by one step.**  ``train_test_split(prediction_length
    = int(round(0.2 * 847)) = 169)`` leaves 678 training steps (indices
    0..677), while the rolling backtest's first fold is ``t0 = int(0.8 * 847) =
    677`` and forecasts indices 677..683.  Index 677 is therefore inside the
    window Chronos-2 was fine-tuned on.  This refines flag F34, which records
    the 678-vs-677 difference but not that the overlap is a one-day leak into
    the first of 164 folds.  Preserved.  Publication: one sentence.

Quirks preserved verbatim (not flags -- they are internal to the legacy code)
---------------------------------------------------------------------------

Q1  ``sort_values(["item_id", "timestamp"])`` before ``set_index``: the panel
    is already sorted, so this is a no-op, but it fixes the item order that
    every downstream ``item_ids`` reads.
Q2  the static frame is built by ``drop_duplicates`` on ``(item_id,
    Activity_Level)``.  A region whose activity tier changed mid-series would
    silently produce two rows and a duplicated item; nothing checks it.
Q3  the predictor returns ``float32`` and the legacy parquet stores
    ``float32``; the :class:`PredictionSet` path casts to ``float``, so a
    golden comparison is numeric, never ``check_dtype=True``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from darts import TimeSeries

    from strikecast.backtest.predictions import PredictionSet
    from strikecast.data.series import SeriesBundle

logger = logging.getLogger(__name__)

__all__ = [
    "ITEM_ID",
    "MEDIAN_CANDIDATES",
    "STATIC_COL",
    "TIMESTAMP",
    "LongFrames",
    "LongPanel",
    "bundle_to_long_frames",
    "fold_long_frame",
    "median_column",
    "naive_scales",
    "panel_to_long_frames",
    "panel_to_timeseries_dataframe",
    "prediction_set",
    "predictions_to_series",
    "scales_for_stage",
    "test_split_length",
    "to_timeseries_dataframe",
]

#: AutoGluon's two index names (``_chronos2.py:206``).
ITEM_ID = "item_id"
TIMESTAMP = "timestamp"

#: The one static feature the Chronos run attaches (``_chronos2.py:211-215``).
STATIC_COL = "Activity_Level"

#: ``_pick_median_col``'s candidate list, in its order (``_chronos2.py:399-403``).
MEDIAN_CANDIDATES: tuple[Any, ...] = (0.5, "0.5", "mean")


# --------------------------------------------------------------------------- #
# forward: panel / bundle -> long frames
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LongFrames:
    """The two frames ``TimeSeriesDataFrame(values, static_features=static)`` takes.

    ``values`` is indexed by ``(item_id, timestamp)`` and holds the target and
    every covariate; ``static`` is indexed by ``item_id`` and holds
    :data:`STATIC_COL`.  Both are plain pandas, so this dataclass is the part of
    the adapter that the main environment can build and test.
    """

    values: pd.DataFrame
    static: pd.DataFrame

    @property
    def item_ids(self) -> list[str]:
        return list(self.values.index.get_level_values(ITEM_ID).unique())

    def panel(self) -> LongPanel:
        """A pure-pandas stand-in for ``TimeSeriesDataFrame`` (see :class:`LongPanel`)."""
        return LongPanel(self.values, self.static)


def panel_to_long_frames(
    panel: pd.DataFrame,
    *,
    group_col: str = "region",
    time_col: str = "event_date",
    static_col: str = STATIC_COL,
    drop_columns: Sequence[str] = (),
) -> LongFrames:
    """``_chronos2.py`` §3 lines 202-218, verbatim.

    Accepts the panel either with the ``(region, event_date)`` MultiIndex that
    :func:`strikecast.data.panel.build_panel_legacy_chronos` returns
    (``keep_index=True``) or already reset; the legacy call is the former.

    ``drop_columns`` is NOT part of the legacy path -- it exists only so a
    sensitivity run can drop the stray ``index`` column of flag F123 and report
    the difference.  Leave it empty to reproduce the thesis.
    """
    index_names = [n for n in panel.index.names if n is not None]
    if group_col in index_names and time_col in index_names:
        frame = panel.reset_index()
    else:
        frame = panel.copy()

    missing = [c for c in (group_col, time_col, static_col) if c not in frame.columns]
    if missing:
        raise KeyError(
            f"panel has no column(s) {missing}; expected the Chronos panel of "
            f"build_panel_legacy_chronos (got {list(frame.columns)[:8]}...)"
        )

    values = (
        frame.rename(columns={group_col: ITEM_ID, time_col: TIMESTAMP})
        .sort_values([ITEM_ID, TIMESTAMP])  # Q1
        .set_index([ITEM_ID, TIMESTAMP])
    )
    static = (
        values.reset_index()[[ITEM_ID, static_col]]
        .drop_duplicates()  # Q2
        .set_index(ITEM_ID)
    )
    values = values.drop(columns=[static_col])
    if drop_columns:
        values = values.drop(columns=[c for c in drop_columns if c in values.columns])
        logger.warning(
            "panel_to_long_frames dropped %s; this is NOT the legacy frame (F123)",
            list(drop_columns),
        )

    logger.info(
        "AutoGluon frame: %d items, %d columns, %d steps per item",
        static.shape[0],
        values.shape[1],
        len(values) // max(static.shape[0], 1),
    )
    return LongFrames(values=values, static=static)


def bundle_to_long_frames(
    bundle: SeriesBundle,
    *,
    target: str,
    static_col: str = STATIC_COL,
    columns: Sequence[str] | None = None,
    strict: bool = True,
) -> LongFrames:
    """The plan's ``SeriesBundle -> TimeSeriesDataFrame`` adapter (§5.1).

    Rebuilds the long frame from the bundle's **un-encoded** lists
    (``bundle.raw``): the target, the *un-windowed* past covariates
    (``raw_past``, which is what Chronos gets -- it does its own context
    modelling, so the six legacy window transforms are not applied, plan §2.3
    "lags_past_covariates": "known covariates only") and the future covariates.
    Region order follows ``bundle.region_names``.

    This is the adapter for a **new** Chronos run.  It cannot reproduce the
    thesis' frame, because a bundle does not carry the panel's excluded columns
    (flag F125) -- pass ``strict=False`` to acknowledge that, or use
    :func:`panel_to_long_frames` to reproduce.

    ``columns`` reorders the covariate block; without it the order is
    target, future, past, which is the order the bundle stores them in.
    """
    if strict:
        raise ValueError(
            "bundle_to_long_frames cannot reproduce the thesis' Chronos frame: a "
            "SeriesBundle drops the panel's excluded columns (F125, notably the "
            "stray 'index' column of F123). Use panel_to_long_frames to reproduce, "
            "or pass strict=False for a new run."
        )
    if bundle.raw is None:
        raise ValueError(
            "bundle_to_long_frames needs the un-encoded lists; this bundle has raw=None"
        )

    raw = bundle.raw
    blocks: list[pd.DataFrame] = []
    statics: dict[str, Any] = {}
    for idx, region in enumerate(bundle.region_names):
        parts = [
            _series_frame(raw.target[idx]),
            _series_frame(raw.future[idx]),
            _series_frame(raw.raw_past[idx]),
        ]
        block = pd.concat(parts, axis=1)
        block = block.loc[:, ~block.columns.duplicated()]
        block.index.name = TIMESTAMP
        block[ITEM_ID] = region
        blocks.append(block.reset_index().set_index([ITEM_ID, TIMESTAMP]))
        statics[region] = _static_value(raw.target[idx], static_col)

    values = pd.concat(blocks).sort_index()
    if target not in values.columns:
        raise KeyError(f"target {target!r} is not a component of the bundle's targets")
    if columns is not None:
        values = values.loc[:, [c for c in columns if c in values.columns]]

    static = pd.DataFrame({static_col: pd.Series(statics)})
    static.index.name = ITEM_ID
    return LongFrames(values=values, static=static)


def _series_frame(ts: TimeSeries) -> pd.DataFrame:
    frame = ts.to_dataframe()
    frame.index = pd.DatetimeIndex(ts.time_index)
    return frame


def _static_value(ts: TimeSeries, static_col: str) -> Any:
    sc = ts.static_covariates
    if sc is None or static_col not in sc.columns:
        raise KeyError(
            f"target series carries no static covariate {static_col!r}; "
            f"has {None if sc is None else list(sc.columns)}"
        )
    return sc[static_col].iloc[0]


def to_timeseries_dataframe(frames: LongFrames) -> Any:
    """``TimeSeriesDataFrame(values, static_features=static)``. Imports AutoGluon."""
    from autogluon.timeseries import TimeSeriesDataFrame  # noqa: PLC0415

    return TimeSeriesDataFrame(frames.values, static_features=frames.static)


def panel_to_timeseries_dataframe(panel: pd.DataFrame, **kwargs: Any) -> Any:
    """:func:`panel_to_long_frames` then :func:`to_timeseries_dataframe`."""
    return to_timeseries_dataframe(panel_to_long_frames(panel, **kwargs))


def test_split_length(n_steps: int, test_frac: float) -> int:
    """``test_size = int(round(TEST_FRAC * n))`` (``_chronos2.py:234``, flag F3/F34).

    Note this is ROUNDED, while every darts family truncates a fraction with
    ``int()``; on the real panel ``int(round(0.2 * 847)) = 169`` leaves 678
    training steps against the darts families' 677 (F34, F126).
    """
    return int(round(test_frac * n_steps))


# --------------------------------------------------------------------------- #
# a pure-pandas stand-in for TimeSeriesDataFrame
# --------------------------------------------------------------------------- #
class LongPanel:
    """The four ``TimeSeriesDataFrame`` members the Chronos code path uses.

    ``_chronos2.py`` only ever calls ``num_timesteps_per_item``, ``item_ids``,
    ``slice_by_timestep``, ``.loc[item_id]`` and ``frame[columns]`` on its
    ``tsdf``.  Re-implementing exactly those over a plain MultiIndexed frame
    means the fold slicing, the naive scales and the whole forecaster are
    exercisable in the main environment, where AutoGluon cannot be installed.

    It is a stand-in for *tests and scales*, never for ``predictor.predict``,
    which wants the real class.
    """

    __slots__ = ("_frame", "_static", "_item_ids")

    def __init__(self, frame: pd.DataFrame, static: pd.DataFrame | None = None) -> None:
        self._frame = frame
        self._static = static
        self._item_ids = pd.Index(frame.index.get_level_values(ITEM_ID).unique())

    # -- the TimeSeriesDataFrame surface ----------------------------------- #
    @property
    def item_ids(self) -> pd.Index:
        return self._item_ids

    @property
    def static_features(self) -> pd.DataFrame | None:
        return self._static

    @property
    def columns(self) -> pd.Index:
        return self._frame.columns

    @property
    def index(self) -> pd.MultiIndex:
        return self._frame.index

    @property
    def loc(self) -> Any:
        return self._frame.loc

    @property
    def num_items(self) -> int:
        return len(self._item_ids)

    def num_timesteps_per_item(self) -> pd.Series:
        return self._frame.groupby(level=ITEM_ID, sort=False).size()

    def slice_by_timestep(self, start: int | None, end: int | None) -> LongPanel:
        """``iloc[start:end]`` within every item, AutoGluon's semantics."""
        sliced = self._frame.groupby(level=ITEM_ID, sort=False, group_keys=False).apply(
            lambda g: g.iloc[slice(start, end)]
        )
        return LongPanel(sliced, self._static)

    def __getitem__(self, columns: Any) -> LongPanel:
        sub = self._frame[columns]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame()
        return LongPanel(sub, self._static)

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame


def naive_scales(
    train_frame: Any,
    target: str,
    seasonality: int = 7,
) -> tuple[dict[str, float], dict[str, float]]:
    """``compute_naive_scales_from_tsdf`` on a :class:`LongPanel` or a real tsdf.

    Thin forwarder to
    :func:`strikecast.evaluation.metrics.compute_naive_scales_from_tsdf`, which
    is already the verbatim port of ``_chronos2.py:279-292``.  It is re-exported
    here because that is where the Chronos caller looks for it, and because
    :class:`LongPanel` is what makes it callable without AutoGluon.
    """
    from strikecast.evaluation.metrics import (  # noqa: PLC0415
        compute_naive_scales_from_tsdf,
    )

    return compute_naive_scales_from_tsdf(train_frame, target, seasonality=seasonality)


# --------------------------------------------------------------------------- #
# inverse: predictions -> darts / long frame / PredictionSet
# --------------------------------------------------------------------------- #
def median_column(pred: Any) -> Any:
    """``_pick_median_col`` (``_chronos2.py:399-405``), verbatim.

    AutoGluon 1.5.0 returns the quantile columns as STRINGS (``'0.1'`` ...
    ``'0.9'``) plus ``'mean'``, so the second candidate wins and the backtest
    uses the **median quantile**, never the mean.  The float candidate is kept
    because the legacy function has it.
    """
    columns = list(pred.columns)
    for candidate in MEDIAN_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(f"Cannot find median column in {columns}")


def _prediction_frame(pred: Any) -> pd.DataFrame:
    frame = pred.frame if isinstance(pred, LongPanel) else pred
    return pd.DataFrame(frame)


def predictions_to_series(
    pred: Any,
    region_names: Sequence[str],
    *,
    column: Any | None = None,
    static_covariates: Sequence[Any] | None = None,
) -> list[TimeSeries]:
    """One darts ``TimeSeries`` per region, in ``region_names`` order.

    This is the inverse the backtest engine needs: the engine collects
    ``list[TimeSeries]`` per fold and
    :meth:`~strikecast.backtest.predictions.PredictionSet.from_fold_preds`
    flattens them into the long frame, so the Chronos family lands in exactly
    the same store layout as every other family.

    The values are the median quantile column (:func:`median_column`); the time
    index is the predictor's own, which starts at the first forecast date.
    """
    from darts import TimeSeries  # noqa: PLC0415

    frame = _prediction_frame(pred)
    col = median_column(frame) if column is None else column
    out: list[TimeSeries] = []
    for idx, region in enumerate(region_names):
        try:
            sub = frame.loc[region]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"predictor returned no item {region!r}") from exc
        statics = None if static_covariates is None else static_covariates[idx]
        out.append(
            TimeSeries.from_times_and_values(
                pd.DatetimeIndex(sub.index),
                np.asarray(sub[col], dtype=float).reshape(-1, 1),
                columns=["y_pred"],
                static_covariates=statics,
            )
        )
    return out


def fold_long_frame(
    pred: Any,
    future_slice: Any,
    fold_index: int,
    *,
    target: str,
) -> pd.DataFrame:
    """The body of ``chronos2_rolling_long``'s loop (``_chronos2.py:429-443``).

    Returns the six legacy columns ``[region, fold, horizon, date, y_true,
    y_pred]`` for ONE fold, in the legacy row order (item-major within the
    fold, which is the transpose of the darts collector's order -- flag F43,
    quirk Q3 of :mod:`strikecast.backtest.predictions`).
    """
    pred_frame = _prediction_frame(pred)
    truth_frame = _prediction_frame(future_slice)

    median = median_column(pred_frame)
    pred_df = pred_frame[[median]].rename(columns={median: "y_pred"}).reset_index()
    truth_df = truth_frame[[target]].rename(columns={target: "y_true"}).reset_index()

    merged = pred_df.merge(truth_df, on=[ITEM_ID, TIMESTAMP])
    merged = merged.sort_values([ITEM_ID, TIMESTAMP])
    merged["horizon"] = merged.groupby(ITEM_ID).cumcount() + 1
    merged["fold"] = fold_index
    merged = merged.rename(columns={ITEM_ID: "region", TIMESTAMP: "date"})
    return merged[["region", "fold", "horizon", "date", "y_true", "y_pred"]]


def prediction_set(long_frames: Iterable[pd.DataFrame]) -> PredictionSet:
    """Stack per-fold legacy frames into a :class:`PredictionSet`.

    ``origin_date`` is re-derived as the minimum ``date`` of each ``fold``
    (which for this schedule is the fold's own ``t0`` timestamp, because every
    region shares one index and horizon 1 is always present), and ``channel``
    is :data:`~strikecast.backtest.protocols.SINGLE_CHANNEL`.  Row order is the
    legacy fold-major one, NOT the darts collector's region-major one -- see
    flag F43; every consumer groups, and the golden comparison sorts first.
    """
    from strikecast.backtest.predictions import COLUMNS, PredictionSet  # noqa: PLC0415
    from strikecast.backtest.protocols import SINGLE_CHANNEL  # noqa: PLC0415

    frames = [f for f in long_frames if len(f)]
    if not frames:
        return PredictionSet.concat([])
    out = pd.concat(frames, ignore_index=True)
    out["origin_date"] = out.groupby("fold")["date"].transform("min")
    out["channel"] = SINGLE_CHANNEL
    out["y_true"] = out["y_true"].astype(float)
    out["y_pred"] = out["y_pred"].astype(float)
    return PredictionSet(out.loc[:, list(COLUMNS)])


def scales_for_stage(
    frames: LongFrames | Any,
    target: str,
    *,
    train_frac: float,
    seasonality: int = 7,
) -> tuple[Mapping[str, float], Mapping[str, float]]:
    """The naive MASE/RMSSE scales of ``_chronos2.py:295-300``.

    ``train_only_size = int(TRAIN_FRAC * n)`` -- **train only**, never the
    validation or test segment, which is what every other in-scope family does
    too (plan §2.3 "Naive scales for test", flag F67 records the hurdle as the
    one exception).  On the real panel that is ``int(0.70 * 847) = 592`` steps.
    """
    panel = frames.panel() if isinstance(frames, LongFrames) else frames
    n = int(panel.num_timesteps_per_item().min())
    train_only_size = int(train_frac * n)
    train_only = panel.slice_by_timestep(None, train_only_size)
    return naive_scales(train_only, target, seasonality=seasonality)
