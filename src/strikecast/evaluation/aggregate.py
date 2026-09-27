"""The one evaluation aggregator, generalised over the three legacy metric sets.

:func:`evaluate` replaces three near-copies that differ only in which metric
primitive they call:

``src/evaluation_tools.py::evaluate_long`` (lines 78-150)
    ``metric_set="count"``. Used by the GBDT, RNN and diff families for every
    persisted ``global_*.json`` and per-view CSV.
``final_hurdle.ipynb`` cell 14 ``evaluate_hurdle_long``
    ``metric_set="hurdle"``. ``evaluate_long`` with R² appended and one extra
    branch (Q2 below).
``final_hurdle.ipynb`` cell 14 ``evaluate_classif_long``
    ``metric_set="classification"``. Same view shapes, probability metrics,
    ``y_prob`` instead of ``y_pred``, and ``per_region`` sorted by F1
    descending instead of MASE ascending.

Column names, row order and dtypes are identical to the legacy functions; the
golden suite (``tests/golden/test_metrics_equality.py``) asserts that against
every stored view in ``golden/results/{gbdt,lstm,diff,finalhurdle}``.

Behaviour-preservation notes (plan §6, levels F and I)
-----------------------------------------------------

``Q1`` **Group keys are appended, not prepended.** Every legacy row is built as
    ``row = metrics(...)`` then ``row.update(key_map)``, so ``region`` and
    ``horizon`` are the LAST columns of each view and ``n`` sits in the middle.
    (The Chronos copy prepends them instead; see
    :data:`strikecast.evaluation.metrics.CHRONOS_DIFFERENCES`.)

``Q2`` **The hurdle adds MASE/RMSSE to non-region views.** ``evaluate_hurdle_long``
    has an ``elif "region" in sub.columns`` branch that, for a group which is
    not region-scoped (i.e. ``per_horizon``), averages the per-region MASE and
    RMSSE *within that group*, skipping NaNs. ``evaluate_long`` has no such
    branch, so the count families' ``per_horizon`` CSVs have no MASE column and
    the hurdle's do. This is why the two families' ``per_horizon`` files cannot
    be concatenated.

``Q3`` **``per_region`` is sorted by a metric, so the row order is data-dependent.**
    ``sort_values("MASE")`` for count/hurdle and ``sort_values("F1",
    ascending=False)`` for classification, both with pandas' default
    non-stable quicksort. Ties therefore have no defined order. Reproduced
    exactly, including the sort kind.

``Q4`` **NaN activity levels are dropped from the activity views.** The level is
    attached with ``df2["region"].map(activity_by_region)``, and a region
    missing from the map becomes NaN. ``groupby`` drops NaN groups by default,
    so such a region contributes to ``per_region`` and to the global row but to
    neither activity view, and the activity views' ``n`` then does not sum to
    the global ``n``. It also flips ``activity_level`` from ``int64`` to
    ``float64`` in the output. This is the aggregation-side counterpart of plan
    F24, which records the *tier filter* keeping NaN-tiered regions; see report
    flag F120.

``Q5`` **``MASE_mean`` in the global row is a mean over regions, not over rows.**
    ``per_region["MASE"].mean()`` skips NaN regions silently, so a family with
    a degenerate scale in one region averages over fewer regions than another
    without saying so.

``Q6`` **The activity-level MASE_mean is taken from ``per_region``, not recomputed.**
    ``per_region[per_region["region"].isin(sub["region"].unique())]``: the
    per-region MASE is computed over ALL rows of that region and then filtered
    by membership, so it is not the MASE of the activity-level subset.

``Q7`` **An empty frame raises.** See :func:`evaluate`'s docstring; the legacy
    functions raise ``KeyError: 'MASE'`` on a zero-row input and this port does
    the same (plan F42, report flag F121).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .metrics import base_metrics, classification_metrics, hurdle_metrics, scaled_metrics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strikecast.backtest.predictions import PredictionSet

__all__ = [
    "LEADERBOARD_COLUMNS",
    "METRIC_SETS",
    "VIEWS",
    "MetricSet",
    "evaluate",
    "leaderboard",
]

MetricSet = Literal["count", "hurdle", "classification"]

#: The three key columns every ``leaderboard.csv`` starts with, before the
#: metric block. Mirrors
#: :data:`strikecast.evaluation.leaderboard.LEADERBOARD_KEYS`, which cannot be
#: imported here (that module imports this one).
LEADERBOARD_COLUMNS: tuple[str, ...] = ("split", "paradigm", "model")

#: The three legacy metric sets, in the order they appear in the plan.
METRIC_SETS: tuple[str, ...] = ("count", "hurdle", "classification")

#: Every view :func:`evaluate` can return. The last two only when an activity
#: map is supplied, exactly as ``evaluate_long``'s ``regions_activity`` gate.
VIEWS: tuple[str, ...] = (
    "per_region",
    "per_horizon",
    "per_region_horizon",
    "global",
    "per_activity_level",
    "per_activity_horizon",
)


def _as_long_frame(preds: PredictionSet | pd.DataFrame) -> pd.DataFrame:
    """Accept either a :class:`PredictionSet` or a bare legacy long frame.

    A multi-channel :class:`PredictionSet` is rejected rather than silently
    pooled: ``evaluate_long`` only ever saw one model's rows, and pooling a
    hurdle's ``prob`` channel with its ``count`` channel would produce a
    meaningless MAE. Select the channel first with
    :meth:`PredictionSet.for_channel`.
    """
    if isinstance(preds, pd.DataFrame):
        return preds
    channels = preds.channels
    if len(channels) > 1:
        raise ValueError(
            f"evaluate() takes a single-channel PredictionSet; got {list(channels)}. "
            "Select one with .for_channel(name) first."
        )
    return preds.frame


def _value_column(long_df: pd.DataFrame, metric_set: str) -> str:
    """Which column holds the model output.

    ``evaluate_classif_long`` reads ``y_prob``; the hurdle notebook renames
    ``y_pred`` to ``y_prob`` right after ``collect_predictions_long``. A
    :class:`PredictionSet` always calls it ``y_pred``, so the classification
    path accepts either and prefers the legacy name when both exist.
    """
    if metric_set == "classification":
        return "y_prob" if "y_prob" in long_df.columns else "y_pred"
    return "y_pred"


def _metric_row(
    sub: pd.DataFrame, metric_set: str, value_col: str, threshold: float
) -> dict[str, Any]:
    if metric_set == "count":
        return base_metrics(sub["y_true"], sub[value_col])
    if metric_set == "hurdle":
        return hurdle_metrics(sub["y_true"], sub[value_col])
    if metric_set == "classification":
        return classification_metrics(sub["y_true"], sub[value_col], threshold)
    raise ValueError(f"unknown metric_set {metric_set!r}; expected one of {METRIC_SETS}")


def evaluate(
    pred_set_or_long_df: PredictionSet | pd.DataFrame,
    mae_scales: Mapping[str, float] | None = None,
    rmse_scales: Mapping[str, float] | None = None,
    activity_by_region: Mapping[str, int] | None = None,
    metric_set: MetricSet = "count",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate metrics over region, horizon, (region, horizon), activity and globally.

    Parameters
    ----------
    pred_set_or_long_df
        A single-channel :class:`~strikecast.backtest.predictions.PredictionSet`
        or the legacy long frame (``region, fold, horizon, date, y_true,
        y_pred``; ``y_prob`` for the classification set).
    mae_scales, rmse_scales
        ``{region: float}`` from
        :func:`~strikecast.evaluation.metrics.compute_naive_scales`. WHICH
        slice they were computed on is a per-family decision the caller owns
        (plan F6/F67): train only for every count family and for the hurdle's
        CV stage, train+val for the hurdle's test stage. Ignored by
        ``metric_set="classification"``, which has no scaled metrics; they may
        then be omitted.
    activity_by_region
        ``{region: level}``. When given, ``per_activity_level`` and
        ``per_activity_horizon`` are added -- exactly ``evaluate_long``'s
        ``regions_activity is not None`` gate. Neither hurdle aggregator had
        this parameter, so for ``metric_set`` ``"hurdle"`` / ``"classification"``
        passing a map produces views with NO legacy counterpart (the natural
        generalisation: the same two groupings, scored with that set's metric).
        Leave it ``None`` to reproduce the stored hurdle files.
    metric_set
        ``"count"`` reproduces ``evaluate_long``, ``"hurdle"``
        ``evaluate_hurdle_long``, ``"classification"`` ``evaluate_classif_long``.
    threshold
        The classification cut (plan F70). 0.5 is what every recorded run used.

    Returns
    -------
    dict
        ``per_region``, ``per_horizon``, ``per_region_horizon`` as dataframes,
        ``global`` as a plain dict, plus the two activity views when asked for.

    Notes
    -----
    **Empty input** (plan F42). A zero-row frame is NOT handled: every
    ``groupby`` yields nothing, ``pd.DataFrame([])`` has no columns, and the
    first ``sort_values("MASE")`` raises ``KeyError: 'MASE'`` (``'F1'`` for the
    classification set). That is the legacy behaviour verbatim and it is
    preserved rather than replaced with an empty-view convention, because no
    recorded run produces zero rows and inventing one would be a methodology
    change. ``tests/unit/test_metrics.py`` pins the exception.
    """
    long_df = _as_long_frame(pred_set_or_long_df)
    mae_scales = {} if mae_scales is None else mae_scales
    rmse_scales = {} if rmse_scales is None else rmse_scales
    value_col = _value_column(long_df, metric_set)
    scaled = metric_set in ("count", "hurdle")
    group_average_scaled = metric_set == "hurdle"

    def _rows_from_groupby(group_cols: str | list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for keys, sub in long_df.groupby(group_cols):
            row = _metric_row(sub, metric_set, value_col, threshold)
            if isinstance(group_cols, str):
                key_map: dict[str, Any] = {group_cols: keys}
            else:
                key_map = dict(
                    zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=False)
                )
            region_key = key_map.get("region")
            if scaled and region_key is not None:
                row.update(
                    scaled_metrics(sub, mae_scales.get(region_key), rmse_scales.get(region_key))
                )
            elif group_average_scaled and "region" in sub.columns:
                # Q2: average the per-region MASE/RMSSE across the regions in
                # this group, skipping NaNs. Hurdle only.
                mase_vals: list[float] = []
                rmsse_vals: list[float] = []
                for reg, reg_sub in sub.groupby("region"):
                    sm = scaled_metrics(reg_sub, mae_scales.get(reg), rmse_scales.get(reg))
                    if not np.isnan(sm["MASE"]):
                        mase_vals.append(sm["MASE"])
                    if not np.isnan(sm["RMSSE"]):
                        rmsse_vals.append(sm["RMSSE"])
                row["MASE"] = float(np.mean(mase_vals)) if mase_vals else np.nan
                row["RMSSE"] = float(np.mean(rmsse_vals)) if rmsse_vals else np.nan
            row.update(key_map)
            rows.append(row)
        return pd.DataFrame(rows)

    if metric_set == "classification":
        per_region = (
            _rows_from_groupby("region")
            .sort_values("F1", ascending=False)
            .reset_index(drop=True)
        )
    else:
        per_region = (
            _rows_from_groupby("region").sort_values("MASE", ascending=True).reset_index(drop=True)
        )
    per_horizon = _rows_from_groupby("horizon").sort_values("horizon").reset_index(drop=True)
    per_region_horizon = (
        _rows_from_groupby(["region", "horizon"])
        .sort_values(["region", "horizon"])
        .reset_index(drop=True)
    )

    global_row = _metric_row(long_df, metric_set, value_col, threshold)
    if scaled:
        global_row["MASE_mean"] = float(per_region["MASE"].mean())
        global_row["MASE_median"] = float(per_region["MASE"].median())
        global_row["RMSSE_mean"] = float(per_region["RMSSE"].mean())

    result: dict[str, Any] = {
        "per_region": per_region,
        "per_horizon": per_horizon,
        "per_region_horizon": per_region_horizon,
        "global": global_row,
    }

    if activity_by_region is not None:
        df2 = long_df.copy()
        df2["activity_level"] = df2["region"].map(activity_by_region)  # Q4

        act_rows: list[dict[str, Any]] = []
        for level, sub in df2.groupby("activity_level"):
            row = _metric_row(sub, metric_set, value_col, threshold)
            if scaled:
                pr_sub = per_region[per_region["region"].isin(sub["region"].unique())]  # Q6
                row["MASE_mean"] = float(pr_sub["MASE"].mean())
                row["RMSSE_mean"] = float(pr_sub["RMSSE"].mean())
            row["activity_level"] = level
            act_rows.append(row)
        result["per_activity_level"] = (
            pd.DataFrame(act_rows).sort_values("activity_level").reset_index(drop=True)
        )

        ah_rows: list[dict[str, Any]] = []
        for (level, h), sub in df2.groupby(["activity_level", "horizon"]):
            row = _metric_row(sub, metric_set, value_col, threshold)
            row["activity_level"] = level
            row["horizon"] = int(h)
            ah_rows.append(row)
        result["per_activity_horizon"] = (
            pd.DataFrame(ah_rows)
            .sort_values(["activity_level", "horizon"])
            .reset_index(drop=True)
        )

    return result


def leaderboard(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The ``leaderboard.csv`` of ``_regression_GBDT.py`` lines 1525-1560.

    The legacy construction is

    .. code-block:: python

        rows.append({"split": split, "paradigm": paradigm, "model": name, **res["global"]})
        ...
        pd.DataFrame(lb_rows).sort_values(["split", "MASE_mean"]).reset_index(drop=True)

    so this function takes the already-assembled ``lb_rows`` and only does the
    frame, the sort and the index reset. Identical in ``_regression_LSTM.py``
    (:1531-1540) and ``_diff_regression.py`` (:1691-1700).

    Quirks, preserved:

    * The sort is by ``split`` FIRST, so the whole ``cv`` block precedes the
      whole ``test`` block and the file is not a single ranking.
    * ``split`` sorts lexicographically: ``cv`` before ``test`` by accident,
      not by design.
    * ``sort_values`` is not stable (quicksort), so rows with equal
      ``MASE_mean`` have no defined order.
    * Rows whose ``MASE_mean`` is NaN sort LAST within their split
      (``na_position="last"`` is the default).
    * Nothing validates that every row has the same keys; a missing metric
      becomes NaN in the frame.
    * The CV leaderboards built earlier in the same scripts sort by
      ``"MASE_mean"`` alone and are not persisted; only this one reaches disk.

    One addition, for the report stage rather than for the legacy scripts: an
    EMPTY ``rows`` returns the empty frame with the three key columns instead
    of raising ``KeyError: 'split'``. ``pd.DataFrame([])`` has no columns, so
    the legacy expression cannot sort it; the legacy scripts never hit that
    because they always had at least one model, while ``strikecast report`` is
    routinely pointed at a store that holds nothing for the requested seed yet
    (plan sec. 7, "every output must degrade gracefully"). Non-empty input is
    untouched, so nothing the thesis produced changes.
    """
    rows = list(rows)
    if not rows:
        return pd.DataFrame(columns=list(LEADERBOARD_COLUMNS))
    return pd.DataFrame(rows).sort_values(["split", "MASE_mean"]).reset_index(drop=True)
