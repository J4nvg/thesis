"""Per-horizon probability calibration for the two classifier families.

Behaviour-preserving port of the calibration cells of the hurdle and damage
notebooks. The two notebooks define the same five functions byte for byte, so
one module serves both.

Provenance
----------
``final_hurdle.ipynb`` cell 20 == ``damage_classifier.ipynb`` cell 22
    ``_logit``, ``fit_sigmoid_cal``, ``apply_sigmoid_cal``, ``_oof_one_group``,
    ``oof_calibrated_probs``, ``fit_final_calibrators_per_horizon``,
    ``apply_calibrator_per_horizon``.
``final_hurdle.ipynb`` cells 21-22, 27, 33, 35; ``damage_classifier.ipynb``
cells 23-24
    the call sites: fit on the CV rows of a paradigm, apply to those same rows
    (the reported CV view) and to the test rows (the reported test view).
``final_hurdle.ipynb`` cells 40-42
    ``collect_va_cal_data_per_horizon`` / ``apply_venn_abers_per_horizon``,
    the Venn-Abers path. Test stage only, global/activity/local paradigms.

Byte-for-byte copies of all of the above live in
``tests/legacy_ref/hurdle_builders.py`` and are what ``tests/unit`` compares
against.

The long frame
--------------
Every function takes the long prediction frame that
``collect_predictions_long`` produces for a classifier channel, renamed
``y_pred -> y_prob``: columns ``region, fold, horizon, date, y_true, y_prob``.
In the refactored pipeline that is
``PredictionSet.legacy_frame("prob")`` with the same rename.

Quirks preserved
----------------
Q1  ``method`` is dead. Both notebooks accept a ``method`` argument on four of
    the five functions, thread ``CAL_METHOD = "sigmoid"`` through it, and then
    never read it: the body always calls ``fit_sigmoid_cal``. The
    ``IsotonicRegression`` import at the top of the cell is unused, and the
    diagnostic loops read ``for method in ["sigmoid"]``. The argument is kept
    with the same name and the same non-effect; passing anything else logs a
    warning and still fits a sigmoid (F111).
Q2  ``apply_calibrators_per_horizon`` writes into the output array with
    ``out[sub.index.to_numpy()]``, i.e. it treats the frame's INDEX LABELS as
    positions. That is correct only for a frame whose index is
    ``RangeIndex(0, len(df))``, which is why every legacy call site is spelled
    ``apply_calibrator_per_horizon(df.reset_index(drop=True), ...)``. A frame
    with any other index is logged at WARNING and otherwise processed exactly
    as the legacy code would (F113).
Q3  A horizon whose rows hold a single class gets ``None`` for a calibrator and
    its RAW probabilities are passed through unchanged, in both the fit and the
    apply step (F112).
Q4  The sigmoid is Platt scaling on the LOGIT of the raw probability,
    ``LogisticRegression(C=1e6, solver="lbfgs")`` -- practically unpenalised,
    but still fitted with sklearn's default ``class_weight=None``, so the class
    prior of the fitting rows is baked into the intercept (F114).
Q5  ``_oof_one_group`` falls back to an IN-SAMPLE sigmoid whenever the group
    has fewer than two minority-class rows, so the "out of fold" diagnostic is
    not out of fold for such groups (F115). With ZERO minority rows that
    fallback fits a ``LogisticRegression`` on a single class and sklearn
    raises ``ValueError``, so the diagnostic CRASHES on a frame that has a
    single-class group -- unlike :func:`fit_calibrators_per_horizon`, which
    takes the ``None`` path (Q3). That is the legacy behaviour, verbatim; it is
    preserved rather than guarded, and the unit test pins both sides raising
    the same error (F115).

In-sample application (F69)
---------------------------
The reported calibrated CV rows are calibrated in sample: the calibrators are
fitted on all CV rows of a horizon and then applied to those same rows. That is
the DEFAULT of :func:`calibrate_per_horizon` (``apply_rows=None``). The test
view passes the held-out rows explicitly and is genuinely out of sample. The
returned :class:`CalibrationResult` carries ``application`` so the run store can
label the two without anyone re-deriving which is which. Nothing about what is
computed changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from venn_abers import VennAbersCalibrator

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CAL_EPS",
    "CAL_METHOD",
    "CalibrationResult",
    "Calibrator",
    "apply_calibrators_per_horizon",
    "apply_sigmoid_calibrator",
    "apply_venn_abers_per_horizon",
    "calibrate_per_horizon",
    "calibrators_from_json",
    "calibrators_to_json",
    "collect_venn_abers_data_per_horizon",
    "fit_calibrators_per_horizon",
    "fit_sigmoid_calibrator",
    "oof_calibrated_probs",
    "venn_abers_data_from_json",
    "venn_abers_data_to_json",
]

logger = logging.getLogger(__name__)

#: ``_CAL_EPS`` of the notebook cell: the clip applied before the logit.
CAL_EPS = 1e-6

#: ``CAL_METHOD`` of ``final_hurdle.ipynb`` cell 22 and
#: ``damage_classifier.ipynb`` cell 24. The only value either notebook ran, and
#: the only value the bodies implement (Q1).
CAL_METHOD = "sigmoid"

#: ``RANDOM_STATE`` of notebook cell 1, the default ``random_state`` of
#: :func:`oof_calibrated_probs`.
LEGACY_RANDOM_STATE = 42

#: What a fitted per-horizon calibrator can be. ``None`` is the single-class
#: horizon of Q3.
Calibrator = LogisticRegression | IsotonicRegression | None

#: How a set of calibrated probabilities relates to the rows the calibrators
#: were fitted on (F69).
Application = Literal["in_sample", "out_of_sample"]


@dataclass(frozen=True)
class CalibrationResult:
    """One fit-and-apply pass of :func:`calibrate_per_horizon`."""

    #: ``horizon -> calibrator``, ``None`` where the horizon held one class.
    calibrators: dict[int, Calibrator]
    #: Calibrated probabilities, positionally aligned to the applied rows.
    probs: np.ndarray
    #: The ``method`` argument as given -- dead, see Q1.
    method: str
    #: ``"in_sample"`` for the legacy CV view, ``"out_of_sample"`` for the test
    #: view. A label only; it changes nothing that is computed.
    application: Application

    @property
    def n_fitted(self) -> int:
        """How many horizons got a calibrator, the notebook's print line."""
        return sum(c is not None for c in self.calibrators.values())


# --------------------------------------------------------------------------- #
# sigmoid (Platt) primitives
# --------------------------------------------------------------------------- #
def _logit(p: Any) -> np.ndarray:
    """``_logit`` of the notebook cell: clip to ``CAL_EPS`` then log-odds."""
    p = np.clip(p, CAL_EPS, 1 - CAL_EPS)
    return np.log(p / (1 - p))


def fit_sigmoid_calibrator(y_prob: Any, y_true: Any) -> LogisticRegression:
    """``fit_sigmoid_cal``. Note the argument order: probabilities FIRST."""
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(y_prob).reshape(-1, 1), np.asarray(y_true, dtype=int))
    return lr


def apply_sigmoid_calibrator(lr: LogisticRegression, y_prob: Any) -> np.ndarray:
    """``apply_sigmoid_cal``: ``P(class 1)`` of the Platt model."""
    return lr.predict_proba(_logit(y_prob).reshape(-1, 1))[:, 1]


def _check_method(method: str) -> None:
    """Q1/F111: ``method`` selects nothing. Warn, do not raise, do not branch."""
    if method != CAL_METHOD:
        logger.warning(
            "calibration method %r is ignored: the legacy calibration cells "
            "accept a `method` argument and always fit a sigmoid (flag F111); "
            "fitting a sigmoid",
            method,
        )


def _check_positional_index(long_df: pd.DataFrame) -> None:
    """Q2/F113: the legacy apply step indexes positionally with index labels."""
    index = long_df.index
    if not (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1):
        logger.warning(
            "long frame index is %s, not RangeIndex(0, %d, 1); the legacy "
            "per-horizon apply step writes results with `out[sub.index]`, so a "
            "frame that has not been through `reset_index(drop=True)` is "
            "scattered by label rather than by position (flag F113)",
            type(index).__name__,
            len(long_df),
        )


# --------------------------------------------------------------------------- #
# per-horizon fit / apply  (the reported path)
# --------------------------------------------------------------------------- #
def fit_calibrators_per_horizon(
    long_df: pd.DataFrame, method: str = CAL_METHOD
) -> dict[int, Calibrator]:
    """``fit_final_calibrators_per_horizon``: one calibrator per horizon.

    Fitted on ALL rows of that horizon in ``long_df``. Horizons whose rows hold
    a single class map to ``None`` (Q3).
    """
    _check_method(method)
    out: dict[int, Calibrator] = {}
    for h, sub in long_df.groupby("horizon"):
        y_h = sub["y_true"].to_numpy().astype(int)
        p_h = sub["y_prob"].to_numpy().astype(float)
        if len(np.unique(y_h)) < 2:
            out[int(h)] = None
            continue
        out[int(h)] = fit_sigmoid_calibrator(p_h, y_h)
    return out


def apply_calibrators_per_horizon(
    long_df: pd.DataFrame,
    calibrators: Mapping[int, Calibrator],
    method: str = CAL_METHOD,
) -> np.ndarray:
    """``apply_calibrator_per_horizon``: calibrated ``y_prob`` for every row.

    Rows of a horizon with no calibrator keep their RAW probability (Q3). The
    result is positionally aligned to ``long_df`` (Q2).
    """
    _check_method(method)
    _check_positional_index(long_df)
    p_raw = long_df["y_prob"].to_numpy().astype(float)
    out = p_raw.copy()
    for h, sub in long_df.groupby("horizon"):
        cal = calibrators.get(int(h))
        if cal is None:
            continue
        idx = sub.index.to_numpy()
        sub_p = sub["y_prob"].to_numpy().astype(float)
        out[idx] = _apply_calibrator(cal, sub_p)
    return out


def _apply_calibrator(cal: LogisticRegression | IsotonicRegression, y_prob: Any) -> np.ndarray:
    """Dispatch on the fitted object, not on ``method`` (Q1).

    Only the :class:`LogisticRegression` branch is ever taken by the legacy
    path; the isotonic branch exists so that a calibrator restored from JSON
    round-trips through the same code.
    """
    if isinstance(cal, IsotonicRegression):
        return np.asarray(cal.predict(np.asarray(y_prob, dtype=float)))
    return apply_sigmoid_calibrator(cal, y_prob)


def calibrate_per_horizon(
    *,
    fit_rows: pd.DataFrame,
    apply_rows: pd.DataFrame | None = None,
    method: str = CAL_METHOD,
) -> CalibrationResult:
    """Fit on ``fit_rows``, apply to ``apply_rows``; both legacy call shapes.

    ``apply_rows=None`` means "apply to the rows the calibrators were fitted
    on", which is the legacy CV view and therefore the DEFAULT (F69). Pass the
    test frame to get the out-of-sample application of
    ``final_hurdle.ipynb`` cell 27. Either way the computation is the legacy
    fit/apply pair; the only thing the distinction buys is the ``application``
    label on the result.
    """
    calibrators = fit_calibrators_per_horizon(fit_rows, method=method)
    in_sample = apply_rows is None
    rows = fit_rows if apply_rows is None else apply_rows
    probs = apply_calibrators_per_horizon(rows, calibrators, method=method)
    return CalibrationResult(
        calibrators=calibrators,
        probs=probs,
        method=method,
        application="in_sample" if in_sample else "out_of_sample",
    )


# --------------------------------------------------------------------------- #
# out-of-fold diagnostic (F69: printed, never saved)
# --------------------------------------------------------------------------- #
def _oof_one_group(
    y: np.ndarray, p: np.ndarray, method: str, n_splits: int, random_state: int
) -> np.ndarray:
    """``_oof_one_group``: OOF calibrated probs for one contiguous subset.

    ``n_splits`` is capped at the minority-class count; below two minority rows
    the function degrades to an in-sample sigmoid (Q5).
    """
    minority = int(min(np.bincount(y))) if set(y) == {0, 1} else 0
    n_splits_eff = max(2, min(n_splits, minority)) if minority >= 2 else 0
    if n_splits_eff < 2:
        return apply_sigmoid_calibrator(fit_sigmoid_calibrator(p, y), p)
    out = np.zeros_like(p)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    for fit_idx, eval_idx in skf.split(p, y):
        c = fit_sigmoid_calibrator(p[fit_idx], y[fit_idx])
        out[eval_idx] = apply_sigmoid_calibrator(c, p[eval_idx])
    return out


def oof_calibrated_probs(
    long_df: pd.DataFrame,
    method: str = CAL_METHOD,
    n_splits: int = 5,
    random_state: int = LEGACY_RANDOM_STATE,
    group_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``oof_calibrated_probs``: ``(y_true, p_raw, p_calibrated)``.

    The diagnostic path of F69: the notebooks call this with
    ``group_col="horizon"`` in a print cell only, and no saved table is built
    from it. Included so the diagnostic can be reproduced, NOT so it can
    replace :func:`calibrate_per_horizon`.

    The shuffled :class:`StratifiedKFold` mixes rows across regions, folds and
    dates, so the "out of fold" split has nothing to do with the backtest folds
    (see F116).
    """
    _check_method(method)
    y = long_df["y_true"].to_numpy().astype(int)
    p = long_df["y_prob"].to_numpy().astype(float)
    if group_col is None:
        return y, p, _oof_one_group(y, p, method, n_splits, random_state)
    p_cal = np.zeros_like(p)
    groups = long_df[group_col].to_numpy()
    for g in pd.unique(groups):
        mask = groups == g
        p_cal[mask] = _oof_one_group(y[mask], p[mask], method, n_splits, random_state)
    return y, p, p_cal


# --------------------------------------------------------------------------- #
# Venn-Abers  (final_hurdle.ipynb cells 40-42, test stage only)
# --------------------------------------------------------------------------- #
def collect_venn_abers_data_per_horizon(
    long_df: pd.DataFrame,
) -> dict[int, dict[str, np.ndarray] | None]:
    """``collect_va_cal_data_per_horizon``: the calibration set per horizon.

    Venn-Abers here is INDUCTIVE (IVAP): there is no fitted object, only the
    stored calibration scores and labels, which
    :func:`apply_venn_abers_per_horizon` hands to
    ``VennAbersCalibrator.predict_proba`` together with the test scores
    (F117). Single-class horizons map to ``None``, as for the sigmoid (Q3).
    """
    out: dict[int, dict[str, np.ndarray] | None] = {}
    for h, sub in long_df.groupby("horizon"):
        y_h = sub["y_true"].to_numpy().astype(int)
        p_h = sub["y_prob"].to_numpy().astype(float)
        if len(np.unique(y_h)) < 2:
            out[int(h)] = None
            continue
        out[int(h)] = {
            "p_cal": np.column_stack([1.0 - p_h, p_h]),
            "y_cal": y_h,
        }
    return out


def apply_venn_abers_per_horizon(
    long_df: pd.DataFrame,
    va_cal_data_per_h: Mapping[int, dict[str, np.ndarray] | None],
) -> np.ndarray:
    """``apply_venn_abers_per_horizon``: calibrated probabilities per row.

    One :class:`~venn_abers.VennAbersCalibrator` instance is reused across
    horizons; it is stateless in this mode, everything travels through the
    ``predict_proba`` arguments. The returned column is ``p_prime[:, 1]``, the
    merged/regularised probability, not the ``(p0, p1)`` interval (F117).
    Rows of a horizon with no calibration data keep their RAW probability.
    """
    _check_positional_index(long_df)
    p_raw = long_df["y_prob"].to_numpy().astype(float)
    out = p_raw.copy()
    va = VennAbersCalibrator()
    for h, sub in long_df.groupby("horizon"):
        cal_data = va_cal_data_per_h.get(int(h))
        if cal_data is None:
            continue
        idx = sub.index.to_numpy()
        p_test = np.column_stack([1.0 - sub["y_prob"].to_numpy(), sub["y_prob"].to_numpy()])
        p_prime = va.predict_proba(
            p_cal=cal_data["p_cal"],
            y_cal=cal_data["y_cal"],
            p_test=p_test.astype(float),
        )
        out[idx] = p_prime[:, 1]
    return out


# --------------------------------------------------------------------------- #
# JSON serialisation  (nothing is pickled)
# --------------------------------------------------------------------------- #
def _sigmoid_to_json(cal: LogisticRegression) -> dict[str, Any]:
    return {
        "type": "sigmoid",
        "coef": np.asarray(cal.coef_, dtype=float).tolist(),
        "intercept": np.asarray(cal.intercept_, dtype=float).tolist(),
        "classes": np.asarray(cal.classes_).astype(int).tolist(),
    }


def _sigmoid_from_json(payload: Mapping[str, Any]) -> LogisticRegression:
    cal = LogisticRegression(C=1e6, solver="lbfgs")
    cal.coef_ = np.asarray(payload["coef"], dtype=float)
    cal.intercept_ = np.asarray(payload["intercept"], dtype=float)
    cal.classes_ = np.asarray(payload["classes"], dtype=int)
    cal.n_features_in_ = 1
    return cal


def _isotonic_to_json(cal: IsotonicRegression) -> dict[str, Any]:
    return {
        "type": "isotonic",
        "x_thresholds": np.asarray(cal.X_thresholds_, dtype=float).tolist(),
        "y_thresholds": np.asarray(cal.y_thresholds_, dtype=float).tolist(),
        "x_min": float(cal.X_min_),
        "x_max": float(cal.X_max_),
        "increasing": bool(cal.increasing_),
        "out_of_bounds": cal.out_of_bounds,
        "y_min": cal.y_min,
        "y_max": cal.y_max,
    }


def _isotonic_from_json(payload: Mapping[str, Any]) -> IsotonicRegression:
    cal = IsotonicRegression(
        y_min=payload["y_min"],
        y_max=payload["y_max"],
        increasing=payload["increasing"],
        out_of_bounds=payload["out_of_bounds"],
    )
    cal.X_thresholds_ = np.asarray(payload["x_thresholds"], dtype=float)
    cal.y_thresholds_ = np.asarray(payload["y_thresholds"], dtype=float)
    cal.X_min_ = payload["x_min"]
    cal.X_max_ = payload["x_max"]
    cal.increasing_ = payload["increasing"]
    cal.n_features_in_ = 1
    # sklearn's `IsotonicRegression._build_f`, reproduced so that `predict`
    # takes exactly the interpolation path a freshly fitted object would.
    cal.f_ = interpolate.interp1d(
        cal.X_thresholds_,
        cal.y_thresholds_,
        kind="linear",
        bounds_error=payload["out_of_bounds"] == "raise",
    )
    return cal


def calibrators_to_json(calibrators: Mapping[int, Calibrator]) -> dict[str, Any]:
    """A JSON-ready payload for ``{horizon: calibrator}``.

    Horizons are string keys because JSON object keys are strings; ``None``
    (the single-class horizon of Q3) round-trips as ``null``. No estimator is
    pickled: a sigmoid is three arrays, an isotonic fit is its threshold pairs.
    """
    return {
        "version": 1,
        "method": CAL_METHOD,
        "calibrators": {
            str(int(h)): (
                None
                if cal is None
                else (
                    _isotonic_to_json(cal)
                    if isinstance(cal, IsotonicRegression)
                    else _sigmoid_to_json(cal)
                )
            )
            for h, cal in calibrators.items()
        },
    }


def calibrators_from_json(payload: Mapping[str, Any]) -> dict[int, Calibrator]:
    """Inverse of :func:`calibrators_to_json`.

    The restored objects reproduce the originals' predictions exactly: sklearn
    reads only ``coef_``/``intercept_``/``classes_`` in
    :meth:`LogisticRegression.predict_proba` and only ``f_`` (plus the clip
    bounds) in :meth:`IsotonicRegression.predict`.
    """
    out: dict[int, Calibrator] = {}
    for h, entry in payload["calibrators"].items():
        if entry is None:
            out[int(h)] = None
        elif entry["type"] == "isotonic":
            out[int(h)] = _isotonic_from_json(entry)
        elif entry["type"] == "sigmoid":
            out[int(h)] = _sigmoid_from_json(entry)
        else:
            raise ValueError(f"unknown calibrator type {entry['type']!r} for horizon {h}")
    return out


def venn_abers_data_to_json(
    va_cal_data_per_h: Mapping[int, dict[str, np.ndarray] | None],
) -> dict[str, Any]:
    """JSON payload for the Venn-Abers calibration sets.

    Only the raw ``y_prob`` column is stored per horizon; ``p_cal`` is
    ``[1-p, p]``, so keeping both columns would be redundant.
    """
    return {
        "version": 1,
        "method": "venn_abers",
        "calibration_sets": {
            str(int(h)): (
                None
                if data is None
                else {
                    "p_cal": np.asarray(data["p_cal"][:, 1], dtype=float).tolist(),
                    "y_cal": np.asarray(data["y_cal"], dtype=int).tolist(),
                }
            )
            for h, data in va_cal_data_per_h.items()
        },
    }


def venn_abers_data_from_json(
    payload: Mapping[str, Any],
) -> dict[int, dict[str, np.ndarray] | None]:
    """Inverse of :func:`venn_abers_data_to_json`."""
    out: dict[int, dict[str, np.ndarray] | None] = {}
    for h, entry in payload["calibration_sets"].items():
        if entry is None:
            out[int(h)] = None
            continue
        p_h = np.asarray(entry["p_cal"], dtype=float)
        out[int(h)] = {
            "p_cal": np.column_stack([1.0 - p_h, p_h]),
            "y_cal": np.asarray(entry["y_cal"], dtype=int),
        }
    return out
