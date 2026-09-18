"""Metric primitives, ported verbatim from the legacy evaluation code.

Sources, all copied with their arithmetic unchanged:

``src/evaluation_tools.py`` lines 17-75
    :func:`compute_naive_scales`, :func:`base_metrics`, :func:`scaled_metrics`
    (legacy ``_scaled_metrics``), :func:`skill` (legacy ``_skill``).
``final_hurdle.ipynb`` cell 14
    :func:`hurdle_metrics` (legacy ``_hurdle_metrics``) and
    :func:`classification_metrics` (legacy ``_classif_metrics_hurdle``).
``_chronos2.py`` lines 279-400
    :func:`compute_naive_scales_from_tsdf`, :func:`chronos_base_metrics`,
    :func:`chronos_scaled_metrics` and :func:`chronos_evaluate_long`.

The methodology is frozen (plan §1): every questionable detail below is
documented as a quirk and reproduced, never repaired.

Behaviour-preservation notes (plan §6, levels F and I)
-----------------------------------------------------

``Q1`` **Predictions are clipped at zero, truth is not.**
    ``base_metrics`` and ``_scaled_metrics`` both apply
    ``np.clip(y_pred, 0.0, None)`` (``src/evaluation_tools.py:41``) before any
    error is formed, so a model that predicts -3 is scored as if it had
    predicted 0. Every reported MAE/RMSE/MASE/RMSSE in the thesis is therefore
    a *clipped* error. The two deviance metrics clip a second time, to ``EPS``,
    because ``mean_poisson_deviance`` needs strictly positive predictions.

``Q2`` **``ZeroAcc`` compares two different thresholds.**
    ``(y_true == 0) == (y_pred < 0.5)``: exact equality on the truth side, a
    0.5 cut on the prediction side. It is an agreement rate on "is this a zero
    day", not an accuracy against a rounded forecast.

``Q3`` **The scaled metrics recompute MAE and RMSE.**
    ``_scaled_metrics`` does not divide ``base_metrics``' output; it
    recomputes ``mae``/``rmse`` from the same subset. The two agree to floating
    point but the duplication is preserved so that any future change to one
    does not silently move the other.

``Q4`` **A degenerate scale yields NaN, and ``0`` and ``None`` are the same.**
    The guard is ``if mae_scale and mae_scale > 0``. A missing region (the
    ``.get`` returned ``None``), a zero scale and a NaN scale all fall through
    to ``np.nan``; ``nan > 0`` is ``False``, and a NaN scale is also falsy-safe
    because ``bool(nan)`` is ``True`` but ``nan > 0`` is ``False``.

``Q5`` **``skill`` returns NaN for a zero, missing or NaN reference**, so a
    skill score against a perfect reference is undefined rather than infinite.
    A NaN *model* value propagates: ``1 - nan/ref`` is NaN.

``Q6`` **The hurdle's R² is computed on the clipped predictions** but through
    ``sklearn.metrics.r2_score``, which has its own constant-truth behaviour
    (it returns 0.0 when ``y_true`` is constant and the fit is perfect, and
    ``-inf``/NaN otherwise). Nothing guards it.

``Q7`` **Classification is cut at a hard-coded 0.5** (plan F70). The threshold
    is a parameter here, but no legacy call site ever passed anything else, so
    the default reproduces every recorded run. ``confusion_matrix(...,
    labels=[0, 1])`` forces a 2x2 matrix, so single-class inputs still unpack;
    the AUC metrics, which genuinely cannot be defined on one class, are
    wrapped in ``_safe`` and come back NaN.

Differences between the ``src`` copies and the Chronos copies
-------------------------------------------------------------

See :data:`CHRONOS_DIFFERENCES` for the machine-readable list; the functions
themselves carry the detail.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_poisson_deviance,
    mean_tweedie_deviance,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from darts import TimeSeries

__all__ = [
    "BASE_METRIC_KEYS",
    "CHRONOS_BASE_METRIC_KEYS",
    "CHRONOS_DIFFERENCES",
    "CLASSIFICATION_METRIC_KEYS",
    "EPS",
    "HURDLE_METRIC_KEYS",
    "SCALED_METRIC_KEYS",
    "base_metrics",
    "chronos_base_metrics",
    "chronos_evaluate_long",
    "chronos_scaled_metrics",
    "classification_metrics",
    "compute_naive_scales",
    "compute_naive_scales_from_tsdf",
    "hurdle_metrics",
    "scaled_metrics",
    "skill",
]

#: ``src/evaluation_tools.py:12``. Guards the deviance metrics at zero.
EPS = 1e-9

#: The keys :func:`base_metrics` emits, in insertion order (= column order).
BASE_METRIC_KEYS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "MedAE",
    "ME",
    "PoissonDev",
    "TweedieDev",
    "ZeroAcc",
    "n",
)

#: The keys :func:`chronos_base_metrics` emits. Note the two missing deviances.
CHRONOS_BASE_METRIC_KEYS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "MedAE",
    "ME",
    "ZeroAcc",
    "n",
)

#: The keys :func:`scaled_metrics` emits.
SCALED_METRIC_KEYS: tuple[str, ...] = ("MASE", "RMSSE")

#: The keys :func:`hurdle_metrics` emits: :data:`BASE_METRIC_KEYS` plus ``R2``.
HURDLE_METRIC_KEYS: tuple[str, ...] = (*BASE_METRIC_KEYS, "R2")

#: The keys :func:`classification_metrics` emits.
CLASSIFICATION_METRIC_KEYS: tuple[str, ...] = (
    "F1",
    "Precision",
    "Recall",
    "ROC_AUC",
    "PR_AUC",
    "Brier",
    "TP",
    "FP",
    "TN",
    "FN",
    "n",
)


# ---------------------------------------------------------------------------
# Naive scales
# ---------------------------------------------------------------------------


def compute_naive_scales(
    target_list: Sequence[TimeSeries],
    region_names: Sequence[str],
    seasonality: int = 7,
) -> tuple[dict[str, float], dict[str, float]]:
    """In-sample mean ``|d_m|`` and ``sqrt(mean d_m^2)`` per region (``m = seasonality``).

    Verbatim port of ``src/evaluation_tools.py::compute_naive_scales``
    (lines 17-34). Returns the two ``{region_name: scale}`` dicts
    :func:`strikecast.evaluation.aggregate.evaluate` consumes.

    Which slice is passed in is the caller's decision and it differs per
    family (plan F6/F67): every count family passes the TRAIN split for both
    the CV and the test stage, while the hurdle passes train for CV and
    train+val for test.

    Quirks, all preserved:

    * ``zip(region_names, target_list)`` is not length-checked. A mismatch
      silently truncates to the shorter of the two.
    * A series of length ``<= seasonality`` yields ``nan`` for BOTH scales,
      which :func:`scaled_metrics` then turns into NaN MASE/RMSSE rather than
      an error.
    * The differences are taken over the whole slice including any leading
      window-transform warm-up, and ``np.mean`` of an empty array is never
      reached because of the length guard.
    """
    mae_s: dict[str, float] = {}
    rmse_s: dict[str, float] = {}
    for name, ts in zip(region_names, target_list, strict=False):
        y = np.asarray(ts.values(), dtype=float).ravel()
        if len(y) <= seasonality:
            mae_s[name] = rmse_s[name] = np.nan
            continue
        err = y[seasonality:] - y[:-seasonality]
        mae_s[name] = float(np.mean(np.abs(err)))
        rmse_s[name] = float(np.sqrt(np.mean(err**2)))
    return mae_s, rmse_s


def compute_naive_scales_from_tsdf(
    train_tsdf: Any,
    target_col: str,
    seasonality: int = 7,
) -> tuple[dict[str, float], dict[str, float]]:
    """The Chronos twin of :func:`compute_naive_scales` (``_chronos2.py:279-292``).

    Identical arithmetic; it only reads its series out of an AutoGluon
    ``TimeSeriesDataFrame`` instead of a list of darts ``TimeSeries``, and it
    keys the result by ``item_id`` rather than by a separately supplied
    ``region_names`` list. ``train_tsdf`` is typed ``Any`` because AutoGluon
    is not installed in the main environment (plan §5.7); only ``.item_ids``
    and ``.loc[item_id][target_col]`` are used.

    Differences from :func:`compute_naive_scales`: see
    :data:`CHRONOS_DIFFERENCES` entry ``compute_naive_scales``.
    """
    mae_s: dict[str, float] = {}
    rmse_s: dict[str, float] = {}
    for item_id in train_tsdf.item_ids:
        y = train_tsdf.loc[item_id][target_col].to_numpy(dtype=float).ravel()
        if len(y) <= seasonality:
            mae_s[item_id] = rmse_s[item_id] = np.nan
            continue
        err = y[seasonality:] - y[:-seasonality]
        mae_s[item_id] = float(np.mean(np.abs(err)))
        rmse_s[item_id] = float(np.sqrt(np.mean(err**2)))
    return mae_s, rmse_s


# ---------------------------------------------------------------------------
# Point-forecast metric primitives
# ---------------------------------------------------------------------------


def base_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Verbatim port of ``src/evaluation_tools.py::base_metrics`` (lines 40-57).

    Keys in :data:`BASE_METRIC_KEYS` order. ``n`` is a python ``int``, every
    other value a python ``float``; that is what makes the persisted CSVs come
    back with an ``int64`` ``n`` column and ``float64`` everywhere else.

    See Q1 (clip at zero), Q2 (``ZeroAcc``) in the module docstring.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0.0, None)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MedAE": float(np.median(np.abs(err))),
        "ME": float(np.mean(err)),
        "PoissonDev": float(mean_poisson_deviance(y_true, np.maximum(y_pred, EPS))),
        "TweedieDev": float(mean_tweedie_deviance(y_true, np.maximum(y_pred, EPS), power=1.5)),
        "ZeroAcc": float(np.mean((y_true == 0) == (y_pred < 0.5))),
        "n": int(len(y_true)),
    }


def scaled_metrics(
    sub_df: pd.DataFrame,
    mae_scale: float | None,
    rmse_scale: float | None,
) -> dict[str, float]:
    """MASE and RMSSE for a region-scoped subset.

    Verbatim port of ``src/evaluation_tools.py::_scaled_metrics``
    (lines 60-72). NaN when the scale is degenerate (Q4).

    It takes a dataframe rather than two arrays because that is the legacy
    signature and the legacy call sites hand it a groupby subset.
    """
    y_true = sub_df["y_true"].to_numpy(dtype=float)
    y_pred = np.clip(sub_df["y_pred"].to_numpy(dtype=float), 0.0, None)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {
        "MASE": float(mae / mae_scale) if mae_scale and mae_scale > 0 else np.nan,
        "RMSSE": float(rmse / rmse_scale) if rmse_scale and rmse_scale > 0 else np.nan,
    }


def skill(model_val: float | None, ref_val: float | None) -> float:
    """``1 - model / reference``; NaN for a missing, zero or NaN reference (Q5).

    Verbatim port of ``src/evaluation_tools.py::_skill`` (lines 68-71). Used
    only by the leaderboard's ``SkillMAE``/``SkillRMSE``/``SkillMASE``/
    ``SkillRMSSE`` columns against the ``naive_weekly`` reference row.
    """
    if ref_val is None or ref_val == 0 or np.isnan(ref_val):
        return np.nan
    return float(1.0 - model_val / ref_val)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Hurdle-family metric primitives (final_hurdle.ipynb cell 14)
# ---------------------------------------------------------------------------


def hurdle_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """``base_metrics`` extended with R². Verbatim port of ``_hurdle_metrics``.

    ``final_hurdle.ipynb`` cell 14. ``R2`` is appended AFTER ``n``, which is
    why the hurdle's persisted CSVs carry ``n`` in the middle of the frame
    rather than at the end (see Q6).
    """
    base = base_metrics(y_true, y_pred)
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0.0, None)
    base["R2"] = float(r2_score(y_true_arr, y_pred_arr))
    return base


def classification_metrics(y_true: Any, y_prob: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Verbatim port of ``_classif_metrics_hurdle`` (``final_hurdle.ipynb`` cell 14).

    ``threshold`` is the hard-coded 0.5 of plan F70, exposed as a parameter but
    never passed anything else by any recorded run.

    Quirks (Q7): ``confusion_matrix(labels=[0, 1])`` keeps the unpack safe on
    single-class inputs; ``_safe`` swallows ``ValueError`` and
    ``ZeroDivisionError`` from the AUC metrics and returns NaN; ``F1``,
    ``Precision`` and ``Recall`` use ``zero_division=0`` so an empty positive
    class scores 0.0 rather than NaN. ``TP``/``FP``/``TN``/``FN``/``n`` are
    python ``int``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def _safe(fn_: Callable[[], float]) -> float:
        try:
            return float(fn_())
        except (ValueError, ZeroDivisionError):
            return np.nan

    return {
        "F1": _safe(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "Precision": _safe(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "Recall": _safe(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": _safe(lambda: roc_auc_score(y_true, y_prob)),
        "PR_AUC": _safe(lambda: average_precision_score(y_true, y_prob)),
        "Brier": _safe(lambda: brier_score_loss(y_true, y_prob)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# The Chronos-2 copies (_chronos2.py lines 279-400)
# ---------------------------------------------------------------------------

#: Every difference between the ``src`` metric functions and their Chronos-2
#: twins, as ``{function: [difference, ...]}``. Reproduced, not reconciled.
CHRONOS_DIFFERENCES: Mapping[str, tuple[str, ...]] = {
    "compute_naive_scales": (
        "reads an AutoGluon TimeSeriesDataFrame (`.item_ids`, "
        "`.loc[item_id][target_col]`) instead of a list of darts TimeSeries",
        "keys the result by `item_id`; the src version keys it by a separately "
        "supplied `region_names` list, so the src version can silently "
        "mis-pair names and series while the Chronos one cannot",
        "arithmetic, the `<= seasonality` guard and the NaN fallback are identical",
    ),
    "base_metrics": (
        "DROPS `PoissonDev` and `TweedieDev`; the Chronos dict has 6 keys, "
        "the src one 8. The stored golden/results/chronos2 CSVs therefore have "
        "no deviance columns and cannot be compared column-wise against the "
        "count families'",
        "does not import or need sklearn; EPS is unused in the Chronos copy",
        "MAE/RMSE/MedAE/ME/ZeroAcc/n and the clip at 0.0 are byte-identical",
    ),
    "_scaled_metrics": (
        "identical, line for line; only the docstring is shorter (the src copy "
        "says 'NaN if the scale is degenerate')",
    ),
    "evaluate_long": (
        "takes ONLY `long_df`; the scales come from the module-level globals "
        "`MAE_SCALES` / `RMSE_SCALES`, which the src version deliberately "
        "removed ('no globals')",
        "has no `regions_activity` parameter, so it never produces the "
        "`per_activity_level` / `per_activity_horizon` views",
        "groups regions with `sort=False`, so `per_region` and the "
        "`per_region_horizon` region blocks come out in first-appearance order; "
        "the src version uses pandas' default `sort=True`",
        "does NOT sort `per_region` by MASE, does not sort `per_horizon` by "
        "horizon and does not `reset_index(drop=True)` any view; the src "
        "version sorts all three and resets every index",
        "puts the group keys FIRST in each row (`{'region': region, **m, **s}`), "
        "so the CSV column order is `region, MAE, ...`; the src version appends "
        "them LAST (`row.update(key_map)`), giving `MAE, ..., MASE, RMSSE, region`",
        "casts the horizon key with `int(h)`; the src version stores the raw "
        "groupby key for `per_horizon`/`per_region_horizon` and only casts in "
        "the per-activity-horizon view",
        "builds `per_region_horizon` with `sort=True` while `per_region` uses "
        "`sort=False`, so the two views disagree on region order within one call",
        "passes `skipna=True` explicitly to `mean`/`median` for MASE_mean, "
        "MASE_median and RMSSE_mean; that is pandas' default, so the values match",
        "copies the input frame (`df = long_df.copy()`) before grouping; the "
        "src version groups the caller's frame directly and only copies when "
        "adding `activity_level`",
    ),
}


def chronos_base_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """``_chronos2.py:307-320``. :func:`base_metrics` without the two deviances."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0.0, None)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MedAE": float(np.median(np.abs(err))),
        "ME": float(np.mean(err)),
        "ZeroAcc": float(np.mean((y_true == 0) == (y_pred < 0.5))),
        "n": int(len(y_true)),
    }


def chronos_scaled_metrics(
    sub_df: pd.DataFrame,
    mae_scale: float | None,
    rmse_scale: float | None,
) -> dict[str, float]:
    """``_chronos2.py:323-334``. Line-for-line identical to :func:`scaled_metrics`."""
    y_true = sub_df["y_true"].to_numpy(dtype=float)
    y_pred = np.clip(sub_df["y_pred"].to_numpy(dtype=float), 0.0, None)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {
        "MASE": float(mae / mae_scale) if mae_scale and mae_scale > 0 else np.nan,
        "RMSSE": float(rmse / rmse_scale) if rmse_scale and rmse_scale > 0 else np.nan,
    }


def chronos_evaluate_long(
    long_df: pd.DataFrame,
    mae_scales: Mapping[str, float],
    rmse_scales: Mapping[str, float],
) -> dict[str, Any]:
    """``_chronos2.py:341-375``, with the two module globals turned into arguments.

    This is the ONLY deviation from the Chronos original: it took
    ``MAE_SCALES`` / ``RMSE_SCALES`` from module scope. Everything else --
    the missing activity views, the absent sorting, the key-first column
    order, ``sort=False`` for regions -- is preserved exactly, which is why
    this cannot be expressed as a ``metric_set`` of
    :func:`strikecast.evaluation.aggregate.evaluate`. See
    :data:`CHRONOS_DIFFERENCES`.

    It reproduces ``golden/results/chronos2/{per_region,per_horizon,
    per_region_horizon,global}_*``; that family stores no activity views.
    """
    df = long_df.copy()

    rows: list[dict[str, Any]] = []
    for region, sub in df.groupby("region", sort=False):
        m = chronos_base_metrics(sub["y_true"], sub["y_pred"])
        s = chronos_scaled_metrics(sub, mae_scales.get(region), rmse_scales.get(region))
        rows.append({"region": region, **m, **s})
    per_region = pd.DataFrame(rows)

    rows = []
    for h, sub in df.groupby("horizon", sort=True):
        m = chronos_base_metrics(sub["y_true"], sub["y_pred"])
        rows.append({"horizon": int(h), **m})
    per_horizon = pd.DataFrame(rows)

    rows = []
    for (region, h), sub in df.groupby(["region", "horizon"], sort=True):
        m = chronos_base_metrics(sub["y_true"], sub["y_pred"])
        s = chronos_scaled_metrics(sub, mae_scales.get(region), rmse_scales.get(region))
        rows.append({"region": region, "horizon": int(h), **m, **s})
    per_region_horizon = pd.DataFrame(rows)

    global_metrics = chronos_base_metrics(df["y_true"], df["y_pred"])
    global_metrics["MASE_mean"] = float(per_region["MASE"].mean(skipna=True))
    global_metrics["MASE_median"] = float(per_region["MASE"].median(skipna=True))
    global_metrics["RMSSE_mean"] = float(per_region["RMSSE"].mean(skipna=True))

    return {
        "per_region": per_region,
        "per_horizon": per_horizon,
        "per_region_horizon": per_region_horizon,
        "global": global_metrics,
    }
