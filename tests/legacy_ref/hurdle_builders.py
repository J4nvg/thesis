"""Verbatim extraction of the hurdle / damage builders and calibration cells.

Source
------
``final_hurdle.ipynb`` and ``damage_classifier.ipynb``, at git commit
``b1bfd795e70f004578ee760ffe7c5fd5ccfb3d79`` (branch ``refactor``).

================================  ==========================================
function here                     notebook source
================================  ==========================================
``get_event_classifier``          ``final_hurdle.ipynb`` cell 9
``get_count_regressor``           ``final_hurdle.ipynb`` cell 9
``get_damage_classifier``         ``damage_classifier.ipynb`` cell 10
``_logit``                        ``final_hurdle.ipynb`` cell 20
``fit_sigmoid_cal``               cell 20
``apply_sigmoid_cal``             cell 20
``_oof_one_group``                cell 20
``oof_calibrated_probs``          cell 20
``fit_final_calibrators_per_horizon``  cell 20
``apply_calibrator_per_horizon``  cell 20
``collect_va_cal_data_per_horizon``    cell 40
``apply_venn_abers_per_horizon``  cell 40
``_classif_metrics_hurdle``       ``damage_classifier.ipynb`` cell 16
``evaluate_classif_long``         ``damage_classifier.ipynb`` cell 16
================================  ==========================================

``final_hurdle.ipynb`` cell 20 and ``damage_classifier.ipynb`` cell 22 hold the
same seven calibration functions; the two cells are byte-identical, verified by
comparing the two notebooks' cell sources, so they are copied once.
``final_hurdle.ipynb`` cell 14 and ``damage_classifier.ipynb`` cell 16 both
define ``_classif_metrics_hurdle`` and ``evaluate_classif_long``; the damage
cell is the copy taken here, and the hurdle cell's two definitions are
byte-identical to it (the hurdle cell additionally defines the regression
metrics, which belong to another teammate's module).

Notebook globals -> keyword parameters
--------------------------------------
Per ``tests/legacy_ref/README.md`` rule 3, each notebook global became a
keyword-only parameter **with the same name**, so the bodies stay
byte-identical:

==========================  ================================================
notebook global             parameter
==========================  ================================================
``RANDOM_STATE``            ``RANDOM_STATE``   (notebook cell 1, value 42)
``available_threads``       ``available_threads`` (``get_available_threads()``)
``COMMON_KWARGS``           ``COMMON_KWARGS``  (``get_common_kwargs()``)
==========================  ================================================

The three builders take ``**kwargs`` in the notebook, so the three parameters
above are keyword-only and precede it; ``get_count_regressor`` forwards
``**kwargs`` into ``CatBoostModel`` exactly as the notebook does, and the two
classifier builders read only ``DTm_depth`` / ``SPE_estm`` out of it, exactly
as the notebook does.

``oof_calibrated_probs`` has ``random_state=RANDOM_STATE`` as a DEFAULT rather
than a body reference, so the module constant ``RANDOM_STATE = 42`` supplies it
here, the same device ``hurdle_runners.py`` uses for ``OUTPUT_CHUNK_LEN``.

This module is TEST-ONLY. Nothing under ``src/`` may import it.
"""

# The bodies below are copied verbatim from the notebooks, so the lint rules
# that would ask us to rewrite them are disabled for this file rather than
# applied:
#   E702 - `try: return float(fn_())` one-liners in `_classif_metrics_hurdle`
#   E741 - not triggered, kept for symmetry with the other legacy_ref modules
#   N803 - upper-case parameter names (the notebook globals)
#   B905 - `zip()` without `strict=`
#   E501 - long metric lines (also globally ignored)
# ruff: noqa: E702, E741, N803, B905, E501

from __future__ import annotations

import numpy as np
import pandas as pd
from darts.models import CatBoostModel, SKLearnClassifierModel
from imbens.ensemble import SelfPacedEnsembleClassifier
from sklearn.isotonic import IsotonicRegression  # noqa: F401  (unused in the cell too)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from venn_abers import VennAbersCalibrator

#: ``final_hurdle.ipynb`` / ``damage_classifier.ipynb`` cell 1.
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# final_hurdle.ipynb cell 9
# --------------------------------------------------------------------------- #
def get_event_classifier(*, RANDOM_STATE, available_threads, COMMON_KWARGS, **kwargs):
    DTm_depth = kwargs.get('DTm_depth',5)
    SPE_estm = kwargs.get('SPE_estm',100)
    clf = SelfPacedEnsembleClassifier(
        estimator    = DecisionTreeClassifier(
            max_depth=DTm_depth,
            random_state=RANDOM_STATE
            ),
        n_estimators = SPE_estm,
        random_state = RANDOM_STATE,
        n_jobs       = available_threads,
    )

    return SKLearnClassifierModel(model=clf, **COMMON_KWARGS)

def get_count_regressor(*, RANDOM_STATE, available_threads, COMMON_KWARGS, **kwargs):
        return CatBoostModel(
            **COMMON_KWARGS,
            loss_function     = "Tweedie:variance_power=1.5",
            boost_from_average= False,
            random_seed       = RANDOM_STATE,
            task_type         = "CPU",
            thread_count = available_threads,
            **kwargs
        )


# --------------------------------------------------------------------------- #
# damage_classifier.ipynb cell 10
# --------------------------------------------------------------------------- #
def get_damage_classifier(*, RANDOM_STATE, available_threads, COMMON_KWARGS, **kwargs):
    DTm_depth = kwargs.get('DTm_depth', 5)
    SPE_estm  = kwargs.get('SPE_estm', 100)
    clf = SelfPacedEnsembleClassifier(
        estimator    = DecisionTreeClassifier(max_depth=DTm_depth, random_state=RANDOM_STATE),
        n_estimators = SPE_estm,
        random_state = RANDOM_STATE,
        n_jobs       = available_threads,
    )
    return SKLearnClassifierModel(model=clf, **COMMON_KWARGS)


# --------------------------------------------------------------------------- #
# final_hurdle.ipynb cell 20 == damage_classifier.ipynb cell 22
# --------------------------------------------------------------------------- #
_CAL_EPS = 1e-6

def _logit(p):
    p = np.clip(p, _CAL_EPS, 1 - _CAL_EPS)
    return np.log(p / (1 - p))

def fit_sigmoid_cal(y_prob, y_true):
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(y_prob).reshape(-1, 1), np.asarray(y_true, dtype=int))
    return lr

def apply_sigmoid_cal(lr, y_prob):
    return lr.predict_proba(_logit(y_prob).reshape(-1, 1))[:, 1]

def _oof_one_group(y, p, method, n_splits, random_state):
    """OOF calibrated probs for one contiguous subset."""
    minority = int(min(np.bincount(y))) if set(y) == {0, 1} else 0
    n_splits_eff = max(2, min(n_splits, minority)) if minority >= 2 else 0
    if n_splits_eff < 2:
        return apply_sigmoid_cal(fit_sigmoid_cal(p, y), p)
    out = np.zeros_like(p)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    for fit_idx, eval_idx in skf.split(p, y):
            c = fit_sigmoid_cal(p[fit_idx], y[fit_idx])
            out[eval_idx] = apply_sigmoid_cal(c, p[eval_idx])
    return out


def oof_calibrated_probs(long_df, method, n_splits=5, random_state=RANDOM_STATE, group_col=None):
    """Returns (y_true, p_raw, p_calibrated) arrays aligned to long_df rows."""
    y = long_df["y_true"].to_numpy().astype(int)
    p = long_df["y_prob"].to_numpy().astype(float)
    if group_col is None:
        return y, p, _oof_one_group(y, p, method, n_splits, random_state)
    p_cal  = np.zeros_like(p)
    groups = long_df[group_col].to_numpy()
    for g in pd.unique(groups):
        mask = (groups == g)
        p_cal[mask] = _oof_one_group(y[mask], p[mask], method, n_splits, random_state)
    return y, p, p_cal


def fit_final_calibrators_per_horizon(long_df, method):
    """Fit one calibrator per horizon on all CV rows. Returns {horizon: calibrator}."""
    out = {}
    for h, sub in long_df.groupby("horizon"):
        y_h = sub["y_true"].to_numpy().astype(int)
        p_h = sub["y_prob"].to_numpy().astype(float)
        if len(np.unique(y_h)) < 2:
            out[int(h)] = None
            continue
        out[int(h)] = fit_sigmoid_cal(p_h, y_h)
    return out


def apply_calibrator_per_horizon(long_df, calibrators_per_h, method):
    """Apply per-horizon calibrator to y_prob column. Returns calibrated prob array."""
    p_raw = long_df["y_prob"].to_numpy().astype(float)
    out   = p_raw.copy()
    for h, sub in long_df.groupby("horizon"):
        cal = calibrators_per_h.get(int(h))
        if cal is None:
            continue
        idx   = sub.index.to_numpy()
        sub_p = sub["y_prob"].to_numpy().astype(float)
        out[idx] =apply_sigmoid_cal(cal, sub_p)
    return out


# --------------------------------------------------------------------------- #
# final_hurdle.ipynb cell 40
# --------------------------------------------------------------------------- #
def collect_va_cal_data_per_horizon(long_df):
    """Store {p_cal, y_cal} per horizon from CV predictions for manual Venn-Abers."""
    out = {}
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


def apply_venn_abers_per_horizon(long_df, va_cal_data_per_h):
    """Apply per-horizon Venn-Abers calibration. Returns calibrated prob array."""
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
# damage_classifier.ipynb cell 16 (== final_hurdle.ipynb cell 14, same two defs)
# --------------------------------------------------------------------------- #
def _classif_metrics_hurdle(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    def _safe(fn_):
        try: return float(fn_())
        except (ValueError, ZeroDivisionError): return np.nan
    return {
        "F1":        _safe(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "Precision": _safe(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    _safe(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC":   _safe(lambda: roc_auc_score(y_true, y_prob)),
        "PR_AUC":    _safe(lambda: average_precision_score(y_true, y_prob)),
        "Brier":     _safe(lambda: brier_score_loss(y_true, y_prob)),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), "n": int(len(y_true)),
    }


def evaluate_classif_long(long_df, threshold=0.5):
    def _rows(group_cols):
        rows = []
        for keys, sub in long_df.groupby(group_cols):
            m = _classif_metrics_hurdle(sub["y_true"], sub["y_prob"], threshold)
            key_map = (
                {group_cols: keys} if isinstance(group_cols, str)
                else dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            )
            m.update(key_map)
            rows.append(m)
        return pd.DataFrame(rows)
    return {
        "per_region":         _rows("region").sort_values("F1", ascending=False).reset_index(drop=True),
        "per_horizon":        _rows("horizon").sort_values("horizon").reset_index(drop=True),
        "per_region_horizon": _rows(["region", "horizon"]).sort_values(["region", "horizon"]).reset_index(drop=True),
        "global":             _classif_metrics_hurdle(long_df["y_true"], long_df["y_prob"], threshold),
    }
