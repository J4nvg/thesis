"""Verbatim extraction of the hurdle backtest loops.

Source
------
``final_hurdle.ipynb`` cells 15, 24, 29 and 34, at git commit
``b1bfd795e70f004578ee760ffe7c5fd5ccfb3d79`` (branch ``refactor``).

============================  =========================================
function here                 notebook source
============================  =========================================
``run_hurdle_cv``             cell 15, ``run_hurdle_cv``
``run_hurdle_test``           cell 24, ``run_final_test`` (renamed)
``run_hurdle_cv_per_activity``  cell 29, ``run_hurdle_cv_per_activity``
``run_hurdle_cv_per_region``  cell 34, ``run_hurdle_cv_per_region``
``make_positive_only_weights``  ``src/ts_specific_tools.py:105``
============================  =========================================

Cell 29 ALSO defines ``run_hurdle_cv_per_region``; cell 34 redefines it later in
the notebook, so at run time cell 34's definition is the one that is called
(cell 35 runs the local CV after cell 34). The two bodies are byte-identical
apart from a trailing blank line -- verified with ``difflib`` -- so the choice
does not change behaviour, but cell 34 is what is copied here. See flag F66.
Cell 29's ``run_final_test_per_activity`` / ``run_final_test_per_region`` are
out of scope for this module (the task asks for the four functions above).

Notebook globals -> keyword parameters
--------------------------------------
Every function in the notebook reads module-level notebook state. Each such
global became a keyword-only parameter **with the same name**, so that the loop
bodies stay byte-identical to the notebook. ``OUTPUT_CHUNK_LEN``,
``CV_START_VAL`` and ``TRAIN_VAL_END`` therefore appear as upper-case parameter
names; that is deliberate.

==========================  ================================================
notebook global             parameter
==========================  ================================================
``target_for_cv_c``         ``target_for_cv_c``   binary (event) targets, CV view
``target_for_cv_r``         ``target_for_cv_r``   count targets, CV view
``full_weights``            ``full_weights``      positive-only sample weights
``full_past_covs_c``        ``full_past_covs_c``  classifier past covariates
``full_fut_covs_c``         ``full_fut_covs_c``   classifier future covariates
``full_past_covs_r``        ``full_past_covs_r``  regressor past covariates
``full_fut_covs_r``         ``full_fut_covs_r``   regressor future covariates
``get_event_classifier``    ``get_event_classifier``  zero-arg builder
``get_count_regressor``     ``get_count_regressor``   zero-arg builder
``train_target_c/_r``       ``train_target_c`` / ``train_target_r``
``val_target_c/_r``         ``val_target_c`` / ``val_target_r``
``test_target_c/_r``        ``test_target_c`` / ``test_target_r``
``make_positive_only_weights``  ``make_positive_only_weights`` (defaults to the
                            copy in this module)
``region_names``            ``region_names``
``regions_activity``        ``regions_activity``
``CV_START_VAL``            ``CV_START_VAL``  (from ``get_covs_and_encodings``)
``TRAIN_VAL_END``           ``TRAIN_VAL_END`` (from ``get_covs_and_encodings``)
``OUTPUT_CHUNK_LEN``        ``OUTPUT_CHUNK_LEN`` (notebook cell 1, value 7)
==========================  ================================================

``predict_stride`` and ``retrain_stride`` keep their notebook signature and
defaults (``1`` and ``OUTPUT_CHUNK_LEN``). The module constant
``OUTPUT_CHUNK_LEN = 7`` exists only to supply that default, exactly as the
notebook global did; passing ``OUTPUT_CHUNK_LEN=...`` overrides the value used
inside the body but not the ``retrain_stride`` default, which is bound at
definition time -- as it was in the notebook.

This module is TEST-ONLY. Nothing under ``src/`` may import it.
"""

# The bodies below are copied verbatim from the notebook, so the lint rules that
# would ask us to rewrite them are disabled for this file rather than applied:
#   E702 - `a(); b(); c()` one-liners in the per-activity / per-region loops
#   B905 - `zip()` without `strict=`
# ruff: noqa: E702, B905

from __future__ import annotations

from collections import defaultdict  # final_hurdle.ipynb cell 29

import numpy as np
from darts import TimeSeries

# final_hurdle.ipynb cell 1
OUTPUT_CHUNK_LEN = 7  # how many days ahead each model predicts in one shot


# --------------------------------------------------------------------------
# src/ts_specific_tools.py:105 (imported into the notebook via `from src import *`)
# --------------------------------------------------------------------------

def make_positive_only_weights(target_list):
    """Weight 1.0 where y > 0, 0.0 where y == 0. Keeps the time index intact."""
    weights = []
    for ts in target_list:
        vals = ts.values().ravel()
        w = (vals > 0).astype(float)
        weights.append(
            TimeSeries.from_times_and_values(
                ts.time_index, w, static_covariates=ts.static_covariates
            )
        )
    return weights


# --------------------------------------------------------------------------
# final_hurdle.ipynb cell 15
# --------------------------------------------------------------------------

def run_hurdle_cv(
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    *,
    target_for_cv_c,
    target_for_cv_r,
    full_weights,
    full_past_covs_c,
    full_fut_covs_c,
    full_past_covs_r,
    full_fut_covs_r,
    get_event_classifier,
    get_count_regressor,
    CV_START_VAL,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
):
    """Expanding-window CV for the hurdle model.

    Makes a 7-day prediction every predict_stride days (default: every day).
    Retrains both models every retrain_stride days (default: every 7 days).
    The frozen model handles all daily predictions within each retrain window,
    but predict() is called with a growing context series each day:

        x * * * * * * *          <- predict day 0  (model trained at day 0)
        ...
        x x x x x x x * * * * * * *  <- predict day 6 then RETRAIN
        x x x x x x x x * * * * * * *  <- predict day 7 (new model)

    Returns
    -------
    fold_preds_c : list[list[TimeSeries]]  P(Y>0) per region / prediction day
    fold_preds_r : list[list[TimeSeries]]  E(Y|Y>0) per region / prediction day
    fold_preds_h : list[list[TimeSeries]]  hurdle = prob * count
    """
    ref_ts    = target_for_cv_c[0]
    n_total   = len(ref_ts)
    start_idx = int(CV_START_VAL * n_total)
    n_regions = len(target_for_cv_c)

    fold_preds_c = [[] for _ in range(n_regions)]
    fold_preds_r = [[] for _ in range(n_regions)]
    fold_preds_h = [[] for _ in range(n_regions)]
    n_preds    = 0
    n_retrains = 0
    clf = None
    reg = None

    for t0 in range(start_idx, n_total - OUTPUT_CHUNK_LEN + 1, predict_stride):
        steps_since_start = t0 - start_idx

        # --- Retrain every retrain_stride steps ---
        if steps_since_start % retrain_stride == 0:
            retrain_time  = ref_ts.time_index[t0]
            train_c       = [ts.drop_after(retrain_time) for ts in target_for_cv_c]
            train_r       = [ts.drop_after(retrain_time) for ts in target_for_cv_r]
            train_weights = [w.drop_after(retrain_time)  for w  in full_weights]

            clf = get_event_classifier()
            clf.fit(series=train_c, past_covariates=full_past_covs_c,
                    future_covariates=full_fut_covs_c)

            reg = get_count_regressor()
            reg.fit(series=train_r, past_covariates=full_past_covs_r,
                    future_covariates=full_fut_covs_r, sample_weight=train_weights)

            n_retrains += 1
            print(f"   retrain {n_retrains}  (data up to {retrain_time.date()})")

        # --- Daily prediction: growing context, frozen model ---
        split_time    = ref_ts.time_index[t0]
        pred_series_c = [ts.drop_after(split_time) for ts in target_for_cv_c]
        pred_series_r = [ts.drop_after(split_time) for ts in target_for_cv_r]

        preds_c = clf.predict(
            n=OUTPUT_CHUNK_LEN, series=pred_series_c,
            past_covariates=full_past_covs_c, future_covariates=full_fut_covs_c,
            predict_likelihood_parameters=True, show_warnings=False,
        )
        preds_r = reg.predict(
            n=OUTPUT_CHUNK_LEN, series=pred_series_r,
            past_covariates=full_past_covs_r, future_covariates=full_fut_covs_r,
            show_warnings=False,
        )

        for r_idx, (pred_c, pred_r) in enumerate(zip(preds_c, preds_r)):
            prob_ts = pred_c.univariate_component(pred_c.n_components - 1)
            probs   = prob_ts.values().ravel()
            counts  = pred_r.values().ravel()
            hurdle_ts = TimeSeries.from_times_and_values(
                pred_r.time_index, (probs * counts).reshape(-1, 1)
            )
            fold_preds_c[r_idx].append(prob_ts)
            fold_preds_r[r_idx].append(pred_r)
            fold_preds_h[r_idx].append(hurdle_ts)

        n_preds += 1

    print(f"   {n_preds} daily predictions, {n_retrains} retrains complete")
    return fold_preds_c, fold_preds_r, fold_preds_h


# --------------------------------------------------------------------------
# final_hurdle.ipynb cell 24 (`run_final_test`)
# --------------------------------------------------------------------------

def run_hurdle_test(
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    *,
    train_target_c,
    val_target_c,
    test_target_c,
    train_target_r,
    val_target_r,
    test_target_r,
    full_past_covs_c,
    full_fut_covs_c,
    full_past_covs_r,
    full_fut_covs_r,
    get_event_classifier,
    get_count_regressor,
    TRAIN_VAL_END,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
    make_positive_only_weights=make_positive_only_weights,
):
    """Evaluate hurdle model on the held-out test set.

    Makes a 7-day prediction every predict_stride days (default: every day).
    Retrains both models every retrain_stride days (default: every 7 days).
    Training always starts from 80% of the full series.
    """
    full_target_c = [tr.append(vl).append(te)
                     for tr, vl, te in zip(train_target_c, val_target_c, test_target_c)]
    full_target_r = [tr.append(vl).append(te)
                     for tr, vl, te in zip(train_target_r, val_target_r, test_target_r)]
    full_weights_final = make_positive_only_weights(full_target_r)

    ref_ts         = full_target_c[0]
    n_total        = len(ref_ts)
    test_start_idx = int(TRAIN_VAL_END * n_total)
    n_regions      = len(full_target_c)

    fold_preds_c = [[] for _ in range(n_regions)]
    fold_preds_r = [[] for _ in range(n_regions)]
    fold_preds_h = [[] for _ in range(n_regions)]
    n_preds    = 0
    n_retrains = 0
    clf = None
    reg = None

    for t0 in range(test_start_idx, n_total - OUTPUT_CHUNK_LEN + 1, predict_stride):
        steps_since_start = t0 - test_start_idx

        # --- Retrain every retrain_stride steps ---
        if steps_since_start % retrain_stride == 0:
            retrain_time  = ref_ts.time_index[t0]
            train_c       = [ts.drop_after(retrain_time) for ts in full_target_c]
            train_r       = [ts.drop_after(retrain_time) for ts in full_target_r]
            train_weights = [w.drop_after(retrain_time)  for w  in full_weights_final]

            clf = get_event_classifier()
            clf.fit(series=train_c, past_covariates=full_past_covs_c,
                    future_covariates=full_fut_covs_c)

            reg = get_count_regressor()
            reg.fit(series=train_r, past_covariates=full_past_covs_r,
                    future_covariates=full_fut_covs_r, sample_weight=train_weights)

            n_retrains += 1
            print(f"   retrain {n_retrains}  (data up to {retrain_time.date()})")

        # --- Daily prediction: growing context, frozen model ---
        split_time    = ref_ts.time_index[t0]
        pred_series_c = [ts.drop_after(split_time) for ts in full_target_c]
        pred_series_r = [ts.drop_after(split_time) for ts in full_target_r]

        preds_c = clf.predict(
            n=OUTPUT_CHUNK_LEN, series=pred_series_c,
            past_covariates=full_past_covs_c, future_covariates=full_fut_covs_c,
            predict_likelihood_parameters=True, show_warnings=False,
        )
        preds_r = reg.predict(
            n=OUTPUT_CHUNK_LEN, series=pred_series_r,
            past_covariates=full_past_covs_r, future_covariates=full_fut_covs_r,
            show_warnings=False,
        )

        for r_idx, (pred_c, pred_r) in enumerate(zip(preds_c, preds_r)):
            prob_ts   = pred_c.univariate_component(pred_c.n_components - 1)
            hurdle_ts = TimeSeries.from_times_and_values(
                pred_r.time_index,
                (prob_ts.values().ravel() * pred_r.values().ravel()).reshape(-1, 1),
            )
            fold_preds_c[r_idx].append(prob_ts)
            fold_preds_r[r_idx].append(pred_r)
            fold_preds_h[r_idx].append(hurdle_ts)

        n_preds += 1

    print(f"   {n_preds} daily predictions, {n_retrains} retrains complete")
    return fold_preds_c, fold_preds_r, fold_preds_h, full_target_c, full_target_r


# --------------------------------------------------------------------------
# final_hurdle.ipynb cell 29
# --------------------------------------------------------------------------

def run_hurdle_cv_per_activity(
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    *,
    target_for_cv_c,
    target_for_cv_r,
    full_weights,
    full_past_covs_c,
    full_fut_covs_c,
    full_past_covs_r,
    full_fut_covs_r,
    get_event_classifier,
    get_count_regressor,
    region_names,
    regions_activity,
    CV_START_VAL,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
):
    """Expanding-window CV — one hurdle model per activity-level group."""
    groups = defaultdict(list)
    for i, region in enumerate(region_names):
        groups[regions_activity[region]].append(i)

    n_regions = len(target_for_cv_c)
    all_fp_c = [None] * n_regions
    all_fp_r = [None] * n_regions
    all_fp_h = [None] * n_regions

    for level in sorted(groups):
        indices = groups[level]
        print(f"\n--- Activity level {level} ({len(indices)} regions) ---")
        g_tc = [target_for_cv_c[i]  for i in indices]
        g_tr = [target_for_cv_r[i]  for i in indices]
        g_w  = [full_weights[i]     for i in indices]
        g_pc = [full_past_covs_c[i] for i in indices]
        g_fc = [full_fut_covs_c[i]  for i in indices]
        g_pr = [full_past_covs_r[i] for i in indices]
        g_fr = [full_fut_covs_r[i]  for i in indices]

        ref_ts    = g_tc[0]
        n_total   = len(ref_ts)
        start_idx = int(CV_START_VAL * n_total)
        fp_c = [[] for _ in indices]
        fp_r = [[] for _ in indices]
        fp_h = [[] for _ in indices]
        clf = reg = None
        n_preds = n_retrains = 0

        for t0 in range(start_idx, n_total - OUTPUT_CHUNK_LEN + 1, predict_stride):
            if (t0 - start_idx) % retrain_stride == 0:
                rt  = ref_ts.time_index[t0]
                clf = get_event_classifier()
                clf.fit(series=[ts.drop_after(rt) for ts in g_tc],
                        past_covariates=g_pc, future_covariates=g_fc)
                reg = get_count_regressor()
                reg.fit(series=[ts.drop_after(rt) for ts in g_tr],
                        past_covariates=g_pr, future_covariates=g_fr,
                        sample_weight=[w.drop_after(rt) for w in g_w])
                n_retrains += 1
                print(f"   retrain {n_retrains} (data up to {rt.date()})")

            st      = ref_ts.time_index[t0]
            preds_c = clf.predict(n=OUTPUT_CHUNK_LEN,
                                  series=[ts.drop_after(st) for ts in g_tc],
                                  past_covariates=g_pc, future_covariates=g_fc,
                                  predict_likelihood_parameters=True, show_warnings=False)
            preds_r = reg.predict(n=OUTPUT_CHUNK_LEN,
                                  series=[ts.drop_after(st) for ts in g_tr],
                                  past_covariates=g_pr, future_covariates=g_fr,
                                  show_warnings=False)
            for g_i, (pc, pr) in enumerate(zip(preds_c, preds_r)):
                prob   = pc.univariate_component(pc.n_components - 1)
                hurdle = TimeSeries.from_times_and_values(
                    pr.time_index,
                    (prob.values().ravel() * pr.values().ravel()).reshape(-1, 1),
                )
                fp_c[g_i].append(prob); fp_r[g_i].append(pr); fp_h[g_i].append(hurdle)
            n_preds += 1

        print(f"   {n_preds} predictions, {n_retrains} retrains complete")
        for g_i, orig_i in enumerate(indices):
            all_fp_c[orig_i] = fp_c[g_i]
            all_fp_r[orig_i] = fp_r[g_i]
            all_fp_h[orig_i] = fp_h[g_i]
    return all_fp_c, all_fp_r, all_fp_h


# --------------------------------------------------------------------------
# final_hurdle.ipynb cell 34 (redefines cell 29's identical version)
# --------------------------------------------------------------------------

def run_hurdle_cv_per_region(
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    *,
    target_for_cv_c,
    target_for_cv_r,
    full_weights,
    full_past_covs_c,
    full_fut_covs_c,
    full_past_covs_r,
    full_fut_covs_r,
    get_event_classifier,
    get_count_regressor,
    region_names,
    CV_START_VAL,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
):
    """Expanding-window CV — one hurdle model per region (local)."""
    n_regions = len(target_for_cv_c)
    all_fp_c, all_fp_r, all_fp_h = [], [], []
    MIN_POSITIVE_SAMPLES = 50

    for i in range(n_regions):
        print(f"\n--- Region {i + 1}/{n_regions} ({region_names[i]}) ---")
        g_tc = [target_for_cv_c[i]];  g_tr = [target_for_cv_r[i]]; g_w = [full_weights[i]]
        g_pc = [full_past_covs_c[i]]; g_fc = [full_fut_covs_c[i]]
        g_pr = [full_past_covs_r[i]]; g_fr = [full_fut_covs_r[i]]

        ref_ts    = g_tc[0]
        n_total   = len(ref_ts)
        start_idx = int(CV_START_VAL * n_total)
        fp_c, fp_r, fp_h = [[]], [[]], [[]]
        clf = reg = None
        n_preds = n_retrains = 0
        use_dummy = False
        dummy_mean = 1.0

        for t0 in range(start_idx, n_total - OUTPUT_CHUNK_LEN + 1, predict_stride):
            if (t0 - start_idx) % retrain_stride == 0:
                rt  = ref_ts.time_index[t0]
                clf = get_event_classifier()
                clf.fit(series=[g_tc[0].drop_after(rt)], past_covariates=g_pc, future_covariates=g_fc)
                w_series = g_w[0].drop_after(rt)
                n_positive = (w_series.values() > 0).sum()
                if n_positive < MIN_POSITIVE_SAMPLES:
                    use_dummy = True
                    target_vals = g_tr[0].drop_after(rt).values()
                    pos_vals = target_vals[w_series.values() > 0]
                    dummy_mean = pos_vals.mean() if len(pos_vals) > 0 else 1.0
                    print(f"  [t0={t0}] Only {n_positive} samples. Using Dummy Regressor (mean={dummy_mean:.2f}).")
                else:
                    use_dummy = False
                    reg = get_count_regressor()
                    reg.fit(series=[g_tr[0].drop_after(rt)], past_covariates=g_pr, future_covariates=g_fr,
                            sample_weight=[w_series])
                n_retrains += 1

            st     = ref_ts.time_index[t0]
            pred_c = clf.predict(n=OUTPUT_CHUNK_LEN, series=[g_tc[0].drop_after(st)],
                                 past_covariates=g_pc, future_covariates=g_fc,
                                 predict_likelihood_parameters=True, show_warnings=False)[0]
            if use_dummy:
                pred_r = TimeSeries.from_times_and_values(
                    pred_c.time_index,
                    np.full((OUTPUT_CHUNK_LEN, 1), dummy_mean)
                )
            else:
                pred_r = reg.predict(n=OUTPUT_CHUNK_LEN, series=[g_tr[0].drop_after(st)],
                                     past_covariates=g_pr, future_covariates=g_fr,
                                     show_warnings=False)[0]
            prob   = pred_c.univariate_component(pred_c.n_components - 1)
            hurdle = TimeSeries.from_times_and_values(
                pred_r.time_index,
                (prob.values().ravel() * pred_r.values().ravel()).reshape(-1, 1),
            )
            fp_c[0].append(prob); fp_r[0].append(pred_r); fp_h[0].append(hurdle)
            n_preds += 1

        all_fp_c.append(fp_c[0]); all_fp_r.append(fp_r[0]); all_fp_h.append(fp_h[0])
    return all_fp_c, all_fp_r, all_fp_h
