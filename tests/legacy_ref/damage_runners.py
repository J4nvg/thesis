"""Verbatim extraction of the damage-classifier backtest loops.

Source
------
``damage_classifier.ipynb`` cells 17 and 26, at git commit
``b1bfd795e70f004578ee760ffe7c5fd5ccfb3d79`` (branch ``refactor``).

=====================  =====================================
function here          notebook source
=====================  =====================================
``run_damage_cv``      cell 17, ``run_damage_cv``
``run_damage_test``    cell 26, ``run_final_test`` (renamed)
=====================  =====================================

Notebook globals -> keyword parameters
--------------------------------------
Same treatment as ``hurdle_runners``: each notebook global became a keyword-only
parameter with the SAME name, so the loop bodies are byte-identical to the
notebook.

=========================  ==================================================
notebook global            parameter
=========================  ==================================================
``damage_classes``         ``damage_classes``
``get_damage_classifier``  ``get_damage_classifier`` (zero-arg builder)
``OUTPUT_CHUNK_LEN``       ``OUTPUT_CHUNK_LEN`` (notebook cell 1, value 7)
=========================  ==================================================

The ``damage_classes`` structure
--------------------------------
``damage_classes`` is an ordered ``dict`` keyed by infrastructure-damage target
name (``damage_classifier.ipynb`` cell 4: one key per entry of ``all_targets``).
Only the ``'get_covs_and_encodings'`` sub-dict matters to these loops; it is
built in cell 8 from the 9-tuple returned by
``src.ts_specific_tools.get_covs_and_encodings``:

===========  =========================================================
sub-key      meaning
===========  =========================================================
``'R'``      ``region_names``, list[str] (unused by the loops)
``'Tr'``     ``train_target``  list[TimeSeries], binary damage target
``'V'``      ``val_target``    list[TimeSeries]
``'Te'``     ``test_target``   list[TimeSeries]
``'FPC'``    ``full_past_covs``   list[TimeSeries], FULL length
``'FFC'``    ``full_fut_covs``    list[TimeSeries], FULL length
``'TCV'``    ``target_for_cv``    list[TimeSeries], the train+val CV view
``'tve'``    ``TRAIN_VAL_END``    float fraction, test start (see flag F17)
``'csv'``    ``CV_START_VAL``     float fraction, CV start on the CV view
===========  =========================================================

Both loops take the schedule (``n_total``, ``start_idx`` / ``test_start_idx``
and ``ref_ts``) from the FIRST key only and apply it to every key -- see flag
F68.

This module is TEST-ONLY. Nothing under ``src/`` may import it.
"""

# Copied verbatim from the notebook; B905 (`zip()` without `strict=`) would ask
# us to rewrite the source, so it is disabled for this file.
# ruff: noqa: B905

from __future__ import annotations

# damage_classifier.ipynb cell 1
OUTPUT_CHUNK_LEN = 7  # how many days ahead each model predicts in one shot


# --------------------------------------------------------------------------
# damage_classifier.ipynb cell 17
# --------------------------------------------------------------------------

def run_damage_cv(
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    *,
    damage_classes,
    get_damage_classifier,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
):
    """Expanding-window CV for the damage classifiers.
    Directly predicts P(damage) for each infrastructure type.

    Returns
    -------
    fold_preds_d : dict[key -> list[list[TimeSeries]]]   P(damage) per type, per region
    """
    first_key = next(iter(damage_classes))
    ref_ts    = damage_classes[first_key]['get_covs_and_encodings']['TCV'][0]
    n_total   = len(ref_ts)
    start_idx = int(damage_classes[first_key]['get_covs_and_encodings']['csv'] * n_total)
    n_regions = len(damage_classes[first_key]['get_covs_and_encodings']['TCV'])

    fold_preds_d = {key: [[] for _ in range(n_regions)] for key in damage_classes}
    n_preds    = 0
    n_retrains = 0
    damage_clfs = {key: None for key in damage_classes}

    for t0 in range(start_idx, n_total - OUTPUT_CHUNK_LEN + 1, predict_stride):
        steps_since_start = t0 - start_idx

        if steps_since_start % retrain_stride == 0:
            retrain_time = ref_ts.time_index[t0]
            for key in damage_classes:
                dc      = damage_classes[key]['get_covs_and_encodings']
                train_d = [ts.drop_after(retrain_time) for ts in dc['TCV']]
                dmg_clf = get_damage_classifier()
                dmg_clf.fit(
                    series            = train_d,
                    past_covariates   = dc['FPC'],
                    future_covariates = dc['FFC'],
                )
                damage_clfs[key] = dmg_clf
            n_retrains += 1
            print(f"   retrain {n_retrains}  (data up to {retrain_time.date()})")

        split_time = ref_ts.time_index[t0]
        for key in damage_classes:
            dc            = damage_classes[key]['get_covs_and_encodings']
            pred_series_d = [ts.drop_after(split_time) for ts in dc['TCV']]
            preds_d = damage_clfs[key].predict(
                n                             = OUTPUT_CHUNK_LEN,
                series                        = pred_series_d,
                past_covariates               = dc['FPC'],
                future_covariates             = dc['FFC'],
                predict_likelihood_parameters = True,
                show_warnings                 = False,
            )
            for r_idx, pred_d in enumerate(preds_d):
                fold_preds_d[key][r_idx].append(pred_d.univariate_component(pred_d.n_components - 1))
        n_preds += 1

    print(f"   {n_preds} daily predictions, {n_retrains} retrains complete")
    return fold_preds_d


# --------------------------------------------------------------------------
# damage_classifier.ipynb cell 26 (`run_final_test`)
# --------------------------------------------------------------------------

def run_damage_test(
    predict_stride=1,
    retrain_stride=OUTPUT_CHUNK_LEN,
    *,
    damage_classes,
    get_damage_classifier,
    OUTPUT_CHUNK_LEN=OUTPUT_CHUNK_LEN,
):
    """Evaluate damage classifiers on the held-out test set.
    Directly predicts P(damage) for each infrastructure type.

    Returns
    -------
    fold_preds_d  : dict[key -> list[list[TimeSeries]]]  P(damage) per type, per region
    full_target_d : dict[key -> list[TimeSeries]]        full damage targets per type
    """
    full_target_d = {}
    for key in damage_classes:
        gc = damage_classes[key]['get_covs_and_encodings']
        full_target_d[key] = [tr.append(vl).append(te)
                               for tr, vl, te in zip(gc['Tr'], gc['V'], gc['Te'])]

    first_key      = next(iter(damage_classes))
    ref_ts         = full_target_d[first_key][0]
    n_total        = len(ref_ts)
    test_start_idx = int(damage_classes[first_key]['get_covs_and_encodings']['tve'] * n_total)
    n_regions      = len(full_target_d[first_key])

    fold_preds_d = {key: [[] for _ in range(n_regions)] for key in damage_classes}
    n_preds    = 0
    n_retrains = 0
    damage_clfs = {key: None for key in damage_classes}

    for t0 in range(test_start_idx, n_total - OUTPUT_CHUNK_LEN + 1, predict_stride):
        steps_since_start = t0 - test_start_idx

        if steps_since_start % retrain_stride == 0:
            retrain_time = ref_ts.time_index[t0]
            for key in damage_classes:
                dc      = damage_classes[key]['get_covs_and_encodings']
                train_d = [ts.drop_after(retrain_time) for ts in full_target_d[key]]
                dmg_clf = get_damage_classifier()
                dmg_clf.fit(
                    series            = train_d,
                    past_covariates   = dc['FPC'],
                    future_covariates = dc['FFC'],
                )
                damage_clfs[key] = dmg_clf
            n_retrains += 1
            print(f"   retrain {n_retrains}  (data up to {retrain_time.date()})")

        split_time = ref_ts.time_index[t0]
        for key in damage_classes:
            dc            = damage_classes[key]['get_covs_and_encodings']
            pred_series_d = [ts.drop_after(split_time) for ts in full_target_d[key]]
            preds_d = damage_clfs[key].predict(
                n                             = OUTPUT_CHUNK_LEN,
                series                        = pred_series_d,
                past_covariates               = dc['FPC'],
                future_covariates             = dc['FFC'],
                predict_likelihood_parameters = True,
                show_warnings                 = False,
            )
            for r_idx, pred_d in enumerate(preds_d):
                fold_preds_d[key][r_idx].append(pred_d.univariate_component(pred_d.n_components - 1))
        n_preds += 1

    print(f"   {n_preds} daily predictions, {n_retrains} retrains complete")
    return fold_preds_d, full_target_d
