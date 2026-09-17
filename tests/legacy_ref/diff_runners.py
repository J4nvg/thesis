"""Verbatim extraction of the diff-family backtest loops.

Source
------
``_diff_regression.py`` lines 660-1005, at git commit
``b1bfd795e70f004578ee760ffe7c5fd5ccfb3d79`` (branch ``refactor``).

Functions copied: ``_diff_to_level``, ``run_expanding_cv``,
``run_expanding_cv_iter``, ``run_final_test_diff``,
``run_final_test_diff_per_activity``, ``run_final_test_diff_per_region``.

Deviations from the source (there are only three, all mechanical):

1. ``_diff_regression.py`` gets ``np``, ``TimeSeries``, ``NaiveMean`` and
   ``_maybe_scale_covs`` from its own module namespace (``from src import *``
   plus explicit imports on lines 74-84). They are imported explicitly here
   from the same origins: ``numpy``, ``darts``, ``darts.models`` and
   ``src.evaluation_tools``.
2. ``from collections import defaultdict`` sits on line 948 of the script,
   between two function definitions. It is hoisted to the top of this module.
3. ``OUTPUT_CHUNK_LEN`` (line 106) and ``CV_STRIDE`` (line 111) are module-level
   globals in the script that are used as DEFAULT ARGUMENT VALUES in the
   signatures below. They are redefined here as module constants with the same
   values (7 and 1). Nothing else changed; every loop body is byte-identical to
   the source.

This module is TEST-ONLY. Nothing under ``src/`` may import it.
"""

from __future__ import annotations

from collections import defaultdict  # _diff_regression.py:948

import numpy as np
from darts import TimeSeries
from darts.models import NaiveMean
from src.evaluation_tools import _maybe_scale_covs

# --- module globals of _diff_regression.py used as default argument values ---
OUTPUT_CHUNK_LEN = 7  # _diff_regression.py:106
CV_STRIDE = 1  # _diff_regression.py:111


# --------------------------------------------------------------------------
# _diff_regression.py In[25]
# --------------------------------------------------------------------------

def _diff_to_level(diff_pred_ts, level_anchor_ts):
    """Un-diff a single fold-prediction: anchor = last actual level before the
    forecast window. Returns a TimeSeries on the same time index as the input.
    """
    first_pred_time = diff_pred_ts.time_index[0]
    # find the timestamp immediately before the first predicted timestamp
    anchor_idx = level_anchor_ts.time_index.get_loc(first_pred_time) - 1
    anchor_value = float(level_anchor_ts.values()[anchor_idx, 0])

    diff_vals  = diff_pred_ts.values().ravel()
    level_vals = anchor_value + np.cumsum(diff_vals)
    return TimeSeries.from_times_and_values(
        diff_pred_ts.time_index,
        level_vals.reshape(-1, 1),
    )


# --------------------------------------------------------------------------
# _diff_regression.py In[28]
# --------------------------------------------------------------------------

def run_expanding_cv(
    builder_fn,
    target_diff_list,
    target_level_list,
    start_frac,
    *,
    is_local=False,
    is_neural=False,
    horizon=OUTPUT_CHUNK_LEN,
    stride=CV_STRIDE,
    retrain_stride=OUTPUT_CHUNK_LEN,
    past_covs=None,
    future_covs=None,
    verbose=True,
):
    """Expanding-window CV on diffed targets. Predictions returned in LEVEL space.

    target_diff_list  -- what the model is trained / predicted on
    target_level_list -- used for the un-diff anchor (and is also what the
                         caller will compare against via evaluate_long)
    start_frac        -- fraction of the diffed series at which CV starts
    retrain_stride    -- retrain every this many prediction steps (default: OUTPUT_CHUNK_LEN)
    """
    ref_diff  = target_diff_list[0]
    n_total   = len(ref_diff)
    start_idx = int(start_frac * n_total)

    n_regions      = len(target_diff_list)
    all_fold_preds = [[] for _ in range(n_regions)]
    n_preds    = 0
    n_retrains = 0
    model      = None
    _local_builder = None

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        steps_since_start = t0 - start_idx
        split_time        = ref_diff.time_index[t0]

        if steps_since_start % retrain_stride == 0:
            train_series = [ts.drop_after(split_time) for ts in target_diff_list]
            if is_local:
                _local_builder = builder_fn
            else:
                model = builder_fn()
                fit_kwargs = {"series": train_series}
                if past_for_fit is not None and model.supports_past_covariates:
                    fit_kwargs["past_covariates"] = past_for_fit
                if fut_for_fit is not None and model.supports_future_covariates:
                    fit_kwargs["future_covariates"] = fut_for_fit
                model.fit(**fit_kwargs)
            n_retrains += 1
            if verbose:
                print(f"   retrain {n_retrains}  (data up to {split_time.date()})")

        pred_series = [ts.drop_after(split_time) for ts in target_diff_list]

        if is_local:
            # ARIMA per region, fit on diffed series with d=0.
            diff_preds = []
            for ts in pred_series:
                try:
                    m = _local_builder()
                    m.fit(ts)
                    pred = m.predict(n=horizon)
                except Exception:
                    fallback = NaiveMean()
                    fallback.fit(ts)
                    pred = fallback.predict(n=horizon)
                diff_preds.append(pred)
        else:
            pred_kwargs = {"n": horizon, "series": pred_series}
            if past_for_fit is not None and model.supports_past_covariates:
                pred_kwargs["past_covariates"] = past_for_fit
            if fut_for_fit is not None and model.supports_future_covariates:
                pred_kwargs["future_covariates"] = fut_for_fit
            diff_preds = model.predict(show_warnings=False, **pred_kwargs)

        # Un-diff each region's prediction back to level space.
        level_preds = [
            _diff_to_level(dp, target_level_list[r_idx])
            for r_idx, dp in enumerate(diff_preds)
        ]

        for r_idx, p in enumerate(level_preds):
            all_fold_preds[r_idx].append(p)
        n_preds += 1

    if verbose:
        print(f"   {n_preds} predictions, {n_retrains} retrains complete")
    return all_fold_preds


def run_expanding_cv_iter(
    builder_fn,
    target_diff_list,
    target_level_list,
    start_frac,
    *,
    is_local=False,
    is_neural=False,
    horizon=OUTPUT_CHUNK_LEN,
    stride=CV_STRIDE,
    retrain_stride=OUTPUT_CHUNK_LEN,
    past_covs=None,
    future_covs=None,
    verbose=False,
):
    """Generator twin of run_expanding_cv — yields cumulative fold preds after
    each prediction step so Optuna's MedianPruner can fire."""
    ref_diff  = target_diff_list[0]
    n_total   = len(ref_diff)
    start_idx = int(start_frac * n_total)
    n_regions = len(target_diff_list)
    all_fold_preds = [[] for _ in range(n_regions)]
    model      = None
    _local_builder = None

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, stride):
        steps_since_start = t0 - start_idx
        split_time        = ref_diff.time_index[t0]

        if steps_since_start % retrain_stride == 0:
            train_series = [ts.drop_after(split_time) for ts in target_diff_list]
            if is_local:
                _local_builder = builder_fn
            else:
                model = builder_fn()
                fit_kwargs = {"series": train_series}
                if past_for_fit is not None and model.supports_past_covariates:
                    fit_kwargs["past_covariates"] = past_for_fit
                if fut_for_fit is not None and model.supports_future_covariates:
                    fit_kwargs["future_covariates"] = fut_for_fit
                model.fit(**fit_kwargs)

        pred_series = [ts.drop_after(split_time) for ts in target_diff_list]

        if is_local:
            diff_preds = []
            for ts in pred_series:
                try:
                    m = _local_builder()
                    m.fit(ts)
                    pred = m.predict(n=horizon)
                except Exception:
                    fallback = NaiveMean()
                    fallback.fit(ts)
                    pred = fallback.predict(n=horizon)
                diff_preds.append(pred)
        else:
            pred_kwargs = {"n": horizon, "series": pred_series}
            if past_for_fit is not None and model.supports_past_covariates:
                pred_kwargs["past_covariates"] = past_for_fit
            if fut_for_fit is not None and model.supports_future_covariates:
                pred_kwargs["future_covariates"] = fut_for_fit
            diff_preds = model.predict(show_warnings=False, **pred_kwargs)

        level_preds = [
            _diff_to_level(dp, target_level_list[r_idx])
            for r_idx, dp in enumerate(diff_preds)
        ]
        for r_idx, p in enumerate(level_preds):
            all_fold_preds[r_idx].append(p)

        yield [list(rp) for rp in all_fold_preds]

def run_final_test_diff(
    builder_fn,
    target_diff_list,
    target_level_list,
    start_frac,
    *,
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    horizon=OUTPUT_CHUNK_LEN,
    past_covs=None,
    future_covs=None,
    is_local=False,
    is_neural=False,
    verbose=True,
):
    """Like run_expanding_cv but predict_stride and retrain_stride are decoupled.

    Trains in diff space; returns predictions in level space (via _diff_to_level).
    """
    ref_diff  = target_diff_list[0]
    n_total   = len(ref_diff)
    start_idx = int(start_frac * n_total)
    n_regions = len(target_diff_list)

    all_fold_preds = [[] for _ in range(n_regions)]
    n_preds    = 0
    n_retrains = 0
    model      = None
    _local_builder = None

    past_for_fit, fut_for_fit = _maybe_scale_covs(
        past_covs, future_covs, do_scale=is_neural,
    )

    for t0 in range(start_idx, n_total - horizon + 1, predict_stride):
        steps_since_start = t0 - start_idx

        if steps_since_start % retrain_stride == 0:
            retrain_time = ref_diff.time_index[t0]
            train_series = [ts.drop_after(retrain_time) for ts in target_diff_list]

            if is_local:
                _local_builder = builder_fn
            else:
                model = builder_fn()
                fit_kwargs = {"series": train_series}
                if past_for_fit is not None and model.supports_past_covariates:
                    fit_kwargs["past_covariates"] = past_for_fit
                if fut_for_fit is not None and model.supports_future_covariates:
                    fit_kwargs["future_covariates"] = fut_for_fit
                model.fit(**fit_kwargs)

            n_retrains += 1
            if verbose:
                print(f"   retrain {n_retrains}  (data up to {retrain_time.date()})")

        split_time   = ref_diff.time_index[t0]
        pred_series  = [ts.drop_after(split_time) for ts in target_diff_list]

        if is_local:
            diff_preds = []
            for ts in pred_series:
                try:
                    m = _local_builder()
                    m.fit(ts)
                    pred = m.predict(n=horizon)
                except Exception:
                    fallback = NaiveMean()
                    fallback.fit(ts)
                    pred = fallback.predict(n=horizon)
                diff_preds.append(pred)
        else:
            pred_kwargs = {"n": horizon, "series": pred_series, "show_warnings": False}
            if past_for_fit is not None and model.supports_past_covariates:
                pred_kwargs["past_covariates"] = past_for_fit
            if fut_for_fit is not None and model.supports_future_covariates:
                pred_kwargs["future_covariates"] = fut_for_fit
            diff_preds = model.predict(**pred_kwargs)

        level_preds = [
            _diff_to_level(dp, target_level_list[r_idx])
            for r_idx, dp in enumerate(diff_preds)
        ]
        for r_idx, p in enumerate(level_preds):
            all_fold_preds[r_idx].append(p)
        n_preds += 1

    if verbose:
        print(f"   {n_preds} daily predictions, {n_retrains} retrains complete")
    return all_fold_preds


# --------------------------------------------------------------------------
# _diff_regression.py In[ ]  (the per-activity / per-region wrappers)
# --------------------------------------------------------------------------

def run_final_test_diff_per_activity(
    builder_fn, target_diff_list, target_level_list,
    region_names, regions_activity, start_frac, *,
    horizon=OUTPUT_CHUNK_LEN, predict_stride=1, retrain_stride,
    past_covs=None, future_covs=None, is_local=False, is_neural=False, verbose=True,
):
    """Train one model per activity-level group in diff space; return level-space preds."""
    groups = defaultdict(list)
    for i, region in enumerate(region_names):
        groups[regions_activity[region]].append(i)
    all_fold_preds = [None] * len(target_diff_list)
    for level in sorted(groups):
        indices = groups[level]
        if verbose:
            print(f"\n--- Activity level {level} ({len(indices)} regions) ---")
        group_diff  = [target_diff_list[i]  for i in indices]
        group_level = [target_level_list[i] for i in indices]
        group_past  = [past_covs[i]   for i in indices] if past_covs   is not None else None
        group_fut   = [future_covs[i] for i in indices] if future_covs is not None else None
        group_preds = run_final_test_diff(
            builder_fn, group_diff, group_level, start_frac,
            predict_stride=predict_stride, retrain_stride=retrain_stride,
            horizon=horizon, past_covs=group_past, future_covs=group_fut,
            is_local=is_local, is_neural=is_neural, verbose=verbose,
        )
        for group_idx, orig_idx in enumerate(indices):
            all_fold_preds[orig_idx] = group_preds[group_idx]
    return all_fold_preds


def run_final_test_diff_per_region(
    builder_fn, target_diff_list, target_level_list, start_frac, *,
    horizon=OUTPUT_CHUNK_LEN, predict_stride=1, retrain_stride,
    past_covs=None, future_covs=None, is_neural=False, verbose=True,
):
    """Train one model per region in diff space; return level-space preds."""
    all_fold_preds = []
    for i in range(len(target_diff_list)):
        if verbose:
            print(f"\n--- Region {i + 1}/{len(target_diff_list)} ---")
        region_preds = run_final_test_diff(
            builder_fn,
            [target_diff_list[i]],
            [target_level_list[i]],
            start_frac,
            predict_stride=predict_stride,
            retrain_stride=retrain_stride,
            horizon=horizon,
            past_covs=[past_covs[i]]    if past_covs   is not None else None,
            future_covs=[future_covs[i]] if future_covs is not None else None,
            is_neural=is_neural,
            verbose=verbose,
        )
        all_fold_preds.append(region_preds[0])
    return all_fold_preds
