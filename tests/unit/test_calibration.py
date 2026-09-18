"""``strikecast.evaluation.calibration`` reproduces the notebook cells exactly.

Oracle: ``tests/legacy_ref/hurdle_builders.py``, the verbatim copy of
``final_hurdle.ipynb`` cell 20 (== ``damage_classifier.ipynb`` cell 22) and
cell 40.

Two comparisons:

* on a synthetic long frame, the new functions must return arrays that are
  bit-identical to the legacy ones, for the sigmoid path, the dead ``method``
  argument, the 5-fold OOF diagnostic and Venn-Abers;
* on the stored hurdle predictions in ``golden/results/finalhurdle``, fitting
  on the CV raw rows and applying to the CV and test raw rows must reproduce
  the stored calibrated rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import shim for `tests.`
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.isotonic import IsotonicRegression  # noqa: E402

from strikecast.evaluation import (  # noqa: E402
    apply_calibrators_per_horizon,
    apply_venn_abers_per_horizon,
    calibrate_per_horizon,
    calibrators_from_json,
    calibrators_to_json,
    collect_venn_abers_data_per_horizon,
    fit_calibrators_per_horizon,
    oof_calibrated_probs,
    venn_abers_data_from_json,
    venn_abers_data_to_json,
)
from tests.legacy_ref import hurdle_builders as legacy  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "golden" / "results" / "finalhurdle"

#: Anything at or below this is float-representation noise: the two code paths
#: run the identical arithmetic, and the golden frames additionally went
#: through a parquet round trip.
FLOAT_NOISE = 1e-12


# --------------------------------------------------------------------------- #
# synthetic long frames
# --------------------------------------------------------------------------- #
def _make_long_df(n_folds=40, n_regions=4, horizons=7, seed=0, dead_horizon=None):
    """A ``collect_predictions_long`` frame for a classifier channel.

    Columns and dtypes follow the real one: ``region, fold, horizon, date,
    y_true, y_prob`` with a float ``y_true``. ``y_prob`` is a noisy, badly
    calibrated function of the latent event probability, so the sigmoid has
    something to correct. ``dead_horizon`` forces one horizon to hold a single
    class, which is the ``None``-calibrator path.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2024-05-11")
    for fold in range(n_folds):
        for region in range(n_regions):
            base = 0.15 + 0.6 * rng.random()
            for h in range(1, horizons + 1):
                p_true = np.clip(base - 0.03 * h, 0.01, 0.99)
                y = float(rng.random() < p_true)
                if dead_horizon is not None and h == dead_horizon:
                    y = 0.0
                # a deliberately over-confident score
                raw = np.clip(p_true + rng.normal(0, 0.12) + 0.2 * (y - 0.5), 0.001, 0.999)
                rows.append(
                    {
                        "region": f"region_{region}",
                        "fold": fold,
                        "horizon": h,
                        "date": start + pd.Timedelta(days=fold + h - 1),
                        "y_true": y,
                        "y_prob": float(raw),
                    }
                )
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture(scope="module")
def cv_df():
    return _make_long_df(seed=0)


@pytest.fixture(scope="module")
def test_df():
    return _make_long_df(n_folds=25, seed=1)


@pytest.fixture(scope="module")
def dead_df():
    """A frame in which horizon 4 has a single class (the ``None`` path)."""
    return _make_long_df(seed=2, dead_horizon=4)


# --------------------------------------------------------------------------- #
# fit / apply vs the verbatim cell
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["sigmoid", "isotonic", "venn-abers", "anything"])
def test_fit_and_apply_match_legacy(cv_df, test_df, method):
    """F111: ``method`` is accepted and ignored on BOTH sides, identically."""
    new_cals = fit_calibrators_per_horizon(cv_df, method=method)
    old_cals = legacy.fit_final_calibrators_per_horizon(cv_df, method=method)
    assert sorted(new_cals) == sorted(old_cals)

    for frame in (cv_df, test_df):
        new = apply_calibrators_per_horizon(frame, new_cals, method=method)
        old = legacy.apply_calibrator_per_horizon(frame, old_cals, method=method)
        np.testing.assert_array_equal(new, old)


def test_fitted_calibrators_are_platt_sigmoids(cv_df):
    from sklearn.linear_model import LogisticRegression

    cals = fit_calibrators_per_horizon(cv_df)
    assert set(cals) == set(range(1, 8))
    for cal in cals.values():
        assert isinstance(cal, LogisticRegression)
        assert cal.C == 1e6
        assert cal.solver == "lbfgs"
        # F114: no class_prior / class_weight handling, sklearn's default.
        assert cal.class_weight is None


def test_single_class_horizon_is_none_and_passes_through(dead_df):
    """F112: a horizon with one class gets no calibrator and keeps raw probs."""
    new_cals = fit_calibrators_per_horizon(dead_df)
    old_cals = legacy.fit_final_calibrators_per_horizon(dead_df, "sigmoid")
    assert new_cals[4] is None
    assert old_cals[4] is None

    new = apply_calibrators_per_horizon(dead_df, new_cals)
    old = legacy.apply_calibrator_per_horizon(dead_df, old_cals, "sigmoid")
    np.testing.assert_array_equal(new, old)

    mask = (dead_df["horizon"] == 4).to_numpy()
    np.testing.assert_array_equal(new[mask], dead_df["y_prob"].to_numpy()[mask])
    assert not np.array_equal(new[~mask], dead_df["y_prob"].to_numpy()[~mask])


def test_calibration_actually_changes_the_probabilities(cv_df):
    cals = fit_calibrators_per_horizon(cv_df)
    out = apply_calibrators_per_horizon(cv_df, cals)
    assert np.abs(out - cv_df["y_prob"].to_numpy()).max() > 1e-3


# --------------------------------------------------------------------------- #
# the in-sample / out-of-sample wrapper (F69)
# --------------------------------------------------------------------------- #
def test_calibrate_per_horizon_defaults_to_in_sample(cv_df):
    """F69: the reported CV view is fitted and applied on the same rows."""
    result = calibrate_per_horizon(fit_rows=cv_df)
    assert result.application == "in_sample"
    assert result.n_fitted == 7

    old_cals = legacy.fit_final_calibrators_per_horizon(cv_df, "sigmoid")
    old = legacy.apply_calibrator_per_horizon(cv_df, old_cals, "sigmoid")
    np.testing.assert_array_equal(result.probs, old)


def test_calibrate_per_horizon_out_of_sample(cv_df, test_df):
    """``final_hurdle.ipynb`` cell 27: CV-fitted calibrators, test rows."""
    result = calibrate_per_horizon(fit_rows=cv_df, apply_rows=test_df)
    assert result.application == "out_of_sample"

    old_cals = legacy.fit_final_calibrators_per_horizon(cv_df, "sigmoid")
    old = legacy.apply_calibrator_per_horizon(test_df, old_cals, "sigmoid")
    np.testing.assert_array_equal(result.probs, old)

    # the label is the ONLY difference; the numbers do not depend on it
    same = calibrate_per_horizon(fit_rows=cv_df, apply_rows=cv_df)
    np.testing.assert_array_equal(same.probs, calibrate_per_horizon(fit_rows=cv_df).probs)
    assert same.application == "out_of_sample"


# --------------------------------------------------------------------------- #
# the OOF diagnostic (F69, F115, F116)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group_col", ["horizon", "region", None])
def test_oof_matches_legacy(cv_df, group_col):
    new = oof_calibrated_probs(cv_df, method="sigmoid", group_col=group_col)
    old = legacy.oof_calibrated_probs(cv_df, method="sigmoid", group_col=group_col)
    for a, b in zip(new, old, strict=True):
        np.testing.assert_array_equal(a, b)


def test_oof_matches_legacy_with_a_single_class_group(dead_df):
    """F115/Q5: a group with NO minority row makes both sides raise, alike.

    ``_oof_one_group`` degrades to an in-sample sigmoid below two minority
    rows; with zero minority rows that sigmoid is a ``LogisticRegression``
    fitted on one class, which sklearn refuses. The notebook cell does exactly
    this, so the port must reproduce the crash rather than paper over it. The
    reported hurdle diagnostic never hit it -- every horizon of the real CV
    frames holds both classes -- and the calibrators themselves take the
    ``None`` path instead (Q3, ``test_single_class_horizon_is_none_...``).
    """
    with pytest.raises(ValueError, match="at least 2 classes") as new_err:
        oof_calibrated_probs(dead_df, method="sigmoid", group_col="horizon")
    with pytest.raises(ValueError, match="at least 2 classes") as old_err:
        legacy.oof_calibrated_probs(dead_df, method="sigmoid", group_col="horizon")
    assert type(new_err.value) is type(old_err.value)
    assert str(new_err.value) == str(old_err.value)


def test_oof_matches_legacy_with_a_single_minority_row(dead_df):
    """Q5's in-sample fallback, the degenerate case that does NOT raise.

    One minority row means ``n_splits_eff < 2``, so the group is calibrated in
    sample: fitted on all its rows and applied to them. Both sides must do
    that, and the other groups must still be split 5-fold.
    """
    frame = dead_df.copy()
    frame.loc[frame.index[frame["horizon"] == 4][0], "y_true"] = 1.0
    assert int(frame.loc[frame["horizon"] == 4, "y_true"].sum()) == 1

    new = oof_calibrated_probs(frame, method="sigmoid", group_col="horizon")
    old = legacy.oof_calibrated_probs(frame, method="sigmoid", group_col="horizon")
    for a, b in zip(new, old, strict=True):
        np.testing.assert_array_equal(a, b)

    # the fallback group really is in-sample, i.e. NOT out of fold
    mask = (frame["horizon"] == 4).to_numpy()
    sub = frame.loc[mask]
    in_sample = legacy.apply_sigmoid_cal(
        legacy.fit_sigmoid_cal(
            sub["y_prob"].to_numpy().astype(float), sub["y_true"].to_numpy().astype(int)
        ),
        sub["y_prob"].to_numpy().astype(float),
    )
    np.testing.assert_array_equal(new[2][mask], in_sample)
    assert not np.array_equal(new[2][~mask], frame["y_prob"].to_numpy()[~mask])


def test_oof_default_random_state_is_the_notebook_seed(cv_df):
    """``random_state=RANDOM_STATE`` -- notebook cell 1, value 42."""
    explicit = oof_calibrated_probs(cv_df, "sigmoid", 5, 42, "horizon")
    default = oof_calibrated_probs(cv_df, "sigmoid", group_col="horizon")
    np.testing.assert_array_equal(explicit[2], default[2])


# --------------------------------------------------------------------------- #
# Venn-Abers (F117)
# --------------------------------------------------------------------------- #
def test_venn_abers_matches_legacy(cv_df, test_df):
    new_data = collect_venn_abers_data_per_horizon(cv_df)
    old_data = legacy.collect_va_cal_data_per_horizon(cv_df)
    assert sorted(new_data) == sorted(old_data)
    for h, entry in new_data.items():
        assert entry is not None and old_data[h] is not None
        np.testing.assert_array_equal(entry["p_cal"], old_data[h]["p_cal"])
        np.testing.assert_array_equal(entry["y_cal"], old_data[h]["y_cal"])

    new = apply_venn_abers_per_horizon(test_df, new_data)
    old = legacy.apply_venn_abers_per_horizon(test_df, old_data)
    np.testing.assert_array_equal(new, old)
    assert np.abs(new - test_df["y_prob"].to_numpy()).max() > 1e-3


def test_venn_abers_single_class_horizon(dead_df, test_df):
    new_data = collect_venn_abers_data_per_horizon(dead_df)
    old_data = legacy.collect_va_cal_data_per_horizon(dead_df)
    assert new_data[4] is None and old_data[4] is None
    new = apply_venn_abers_per_horizon(test_df, new_data)
    old = legacy.apply_venn_abers_per_horizon(test_df, old_data)
    np.testing.assert_array_equal(new, old)
    mask = (test_df["horizon"] == 4).to_numpy()
    np.testing.assert_array_equal(new[mask], test_df["y_prob"].to_numpy()[mask])


def test_venn_abers_json_round_trip(cv_df, test_df):
    data = collect_venn_abers_data_per_horizon(cv_df)
    restored = venn_abers_data_from_json(json.loads(json.dumps(venn_abers_data_to_json(data))))
    np.testing.assert_array_equal(
        apply_venn_abers_per_horizon(test_df, data),
        apply_venn_abers_per_horizon(test_df, restored),
    )


# --------------------------------------------------------------------------- #
# JSON serialisation (nothing is pickled)
# --------------------------------------------------------------------------- #
def test_calibrator_json_round_trip_reproduces_predictions(cv_df, test_df, dead_df):
    for fit_frame in (cv_df, dead_df):
        cals = fit_calibrators_per_horizon(fit_frame)
        payload = json.loads(json.dumps(calibrators_to_json(cals)))
        restored = calibrators_from_json(payload)

        assert sorted(restored) == sorted(cals)
        for h, cal in cals.items():
            assert (restored[h] is None) == (cal is None)

        for frame in (cv_df, test_df, dead_df):
            np.testing.assert_array_equal(
                apply_calibrators_per_horizon(frame, cals),
                apply_calibrators_per_horizon(frame, restored),
            )
            # and identical to the legacy application as well
            np.testing.assert_array_equal(
                apply_calibrators_per_horizon(frame, restored),
                legacy.apply_calibrator_per_horizon(
                    frame, legacy.fit_final_calibrators_per_horizon(fit_frame, "sigmoid"), "sigmoid"
                ),
            )


def test_calibrator_json_is_plain_data(cv_df):
    payload = calibrators_to_json(fit_calibrators_per_horizon(cv_df))
    text = json.dumps(payload)
    assert json.loads(text) == payload
    assert payload["calibrators"]["1"]["type"] == "sigmoid"
    assert set(payload["calibrators"]["1"]) == {"type", "coef", "intercept", "classes"}


def test_isotonic_json_round_trip(cv_df):
    """The isotonic branch is never produced by the legacy path (F111), but the
    serialiser has to survive one if a later config ever fits one."""
    rng = np.random.default_rng(3)
    x = rng.random(400)
    y = (rng.random(400) < x).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    restored = calibrators_from_json(
        json.loads(json.dumps(calibrators_to_json({1: iso, 2: None})))
    )
    assert isinstance(restored[1], IsotonicRegression)
    assert restored[2] is None
    np.testing.assert_array_equal(iso.predict(x), restored[1].predict(x))

    frame = cv_df[cv_df["horizon"] == 1].reset_index(drop=True)
    np.testing.assert_array_equal(
        apply_calibrators_per_horizon(frame, {1: iso}),
        apply_calibrators_per_horizon(frame, restored),
    )


# --------------------------------------------------------------------------- #
# golden: the stored hurdle predictions
# --------------------------------------------------------------------------- #
def _golden(name):
    path = GOLDEN_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"golden prediction frame not available: {path}")
    return pd.read_parquet(path).reset_index(drop=True)


@pytest.mark.golden
@pytest.mark.parametrize("stage", ["cv", "test"])
def test_golden_hurdle_calibrated_probs(stage):
    """Refit the CV calibrators and reproduce the stored calibrated columns.

    ``final_hurdle.ipynb`` cells 22 and 27: one sigmoid per horizon fitted on
    ``long_df_c`` (all CV rows), applied to those same rows for the CV view
    (in sample, F69) and to the test rows for the test view.
    """
    cv_raw = _golden("global_cv_classifier_probs")
    rows = _golden(f"global_{stage}_classifier_probs")
    expected = _golden(f"global_{stage}_classifier_cal_probs")

    result = calibrate_per_horizon(
        fit_rows=cv_raw,
        apply_rows=None if stage == "cv" else rows,
    )
    assert result.application == ("in_sample" if stage == "cv" else "out_of_sample")
    assert result.n_fitted == len(result.calibrators) == 7

    max_abs_diff = float(np.abs(result.probs - expected["y_prob"].to_numpy()).max())
    assert max_abs_diff <= FLOAT_NOISE, f"max abs diff {max_abs_diff:.3e}"


@pytest.mark.golden
def test_golden_calibrators_survive_a_json_round_trip():
    cv_raw = _golden("global_cv_classifier_probs")
    test_raw = _golden("global_test_classifier_probs")
    expected = _golden("global_test_classifier_cal_probs")

    cals = fit_calibrators_per_horizon(cv_raw)
    restored = calibrators_from_json(json.loads(json.dumps(calibrators_to_json(cals))))
    probs = apply_calibrators_per_horizon(test_raw, restored)

    np.testing.assert_array_equal(probs, apply_calibrators_per_horizon(test_raw, cals))
    assert float(np.abs(probs - expected["y_prob"].to_numpy()).max()) <= FLOAT_NOISE
