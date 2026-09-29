"""Feature importance: the thesis's GBDT and Chronos-2 measures (audit C14).

Pure functions; :mod:`strikecast.pipeline.importance_stage` wires them to the
run store and the CLI.

GBDT (``_regression_GBDT.ipynb`` cell 57, ``get_gbm_importances``)
------------------------------------------------------------------
For a fitted darts GBDT (one sklearn estimator per horizon in
``model.model.estimators_``) and the SAME series it was fit on:

``h<k>_gain``
    native importance of the horizon-``k`` estimator: LightGBM
    ``booster_.feature_importance(importance_type="gain")``, CatBoost
    ``get_feature_importance()`` (its default, ``PredictionValuesChange``),
    anything else ``feature_importances_``;
``h<k>_perm``
    ``sklearn.inspection.permutation_importance(est, X, y[:, k-1],
    n_repeats=5, random_state=42, n_jobs=-1).importances_mean`` with the
    estimator's own ``score`` (R^2), on the lagged design matrix darts builds
    from the (encoded) fitting series;
``agg_gain`` / ``agg_perm``
    the plain mean over the seven horizons;

rows sorted by ``agg_perm`` descending. Columns, in this order:
``Feature, h1_gain..h7_gain, h1_perm..h7_perm, agg_gain, agg_perm`` -- exactly
``results/gbdt/importance_*.csv``; ``importance_all.csv`` appends ``model``.

Text versus code (for the paper, no code change; methodology frozen): the
thesis (``main.tex`` §"Feature importance") defines permutation importance as
the increase of a *validation loss* on *held-out* data. The code permutes the
columns of the TRAINING design matrix of a model fit on the FULL series
(``target_full``, i.e. including the test period), and the score is the
estimator's R^2 decrease, not a loss increase.

Chronos-2 (``_chronos2.py:526, 686``)
-------------------------------------
``predictor.feature_importance(data, model=<name>, relative_scores=True)`` with
AutoGluon's defaults (``method="permutation"``, ``subsample_size=50``,
``num_iterations=None`` -> 5, ``random_seed=123``), where ``data`` is the full
frame (``test_data_ag``), so AutoGluon scores the last ``prediction_length``
steps. See :func:`chronos_importance`.

Semantic categories (``results/analyse_results.ipynb`` cell 52, ``classify_feature``)
-------------------------------------------------------------------------------------
Ordered substring rules on the lower-cased lagged feature name, first match
wins (table ``tab:semantic_categories``). The rules match the BASE variable, so
a window prefix never matters: ``ewm_expdecay7_<base>...`` (legacy
``series.window.expdecay: legacy_alpha``) and ``ewm_leaky7_<base>...`` (the
fixed ``leaky`` filter, D1) fall in the same category. :func:`window_transform`
reports which filter a feature came from, for tables that want to split them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CATEGORY_ORDER",
    "CATEGORY_RULES",
    "GAIN_SUFFIX",
    "PERM_SUFFIX",
    "TOP_N",
    "category_importance_matrix",
    "category_shares",
    "category_shares_long",
    "chronos_importance",
    "classify_feature",
    "gain_importance",
    "gbm_importances",
    "importance_columns",
    "lagged_design",
    "top_features",
    "window_transform",
]

logger = logging.getLogger(__name__)

GAIN_SUFFIX = "_gain"
PERM_SUFFIX = "_perm"
#: ``TOP_N = 15`` of the category-share figure (analysis notebook cell 52).
TOP_N = 15

#: ``classify_feature``'s rules, in its order (first match wins). The last
#: category, ``Other``, is the fall-through.
CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Spatial / static", ("region_statcov", "dist_to_nearest", "dist_x_clash", "area_km2")),
    ("Weather / geomag.", ("env_weather", "env_k_max")),
    ("Calendar", ("holiday",)),
    ("Cyber", ("act_cyber",)),
    ("Missile / launch", ("confirmed_launched", "confirmed_destroyed", "intercept_rate")),
    ("Macroeconomic", ("usd_rub", "usd_uah", "ch_export")),
    (
        "Comms / diplo / aid",
        (
            "com_aid",
            "com_diplo",
            "com_verbal",
            "com_restrictive",
            "com_escalation",
            "com_logistic",
            "com_ners",
        ),
    ),
    (
        "Conflict & damage",
        (
            "disrupted_weapons",
            "shelling",
            "armed_clash",
            "damage_events",
            "drone_infra",
            "acled_other",
        ),
    ),
    (
        "Autoregressive strikes",
        ("total_daily_strike_events", "drone_strike_on", "_target", "specialmilitary", "ratio_ua_rus"),
    ),
)

#: ``CATEGORY_ORDER`` of the notebook: the row order of the category matrix.
CATEGORY_ORDER: tuple[str, ...] = (
    "Autoregressive strikes",
    "Spatial / static",
    "Weather / geomag.",
    "Conflict & damage",
    "Macroeconomic",
    "Comms / diplo / aid",
    "Missile / launch",
    "Cyber",
    "Calendar",
    "Other",
)

#: Window-transform prefixes of the lagged feature names (``ewm_<name>_...`` /
#: ``rolling_<name>_...``). ``expdecay7`` is the legacy filter, ``leaky7`` the
#: fixed one (``series.window.expdecay``).
_WINDOW_RE = re.compile(r"^(ewm|rolling)_([A-Za-z0-9]+)_")


def classify_feature(name: str) -> str:
    """The semantic category of one feature name (analysis notebook cell 52)."""
    n = name.lower()
    for category, needles in CATEGORY_RULES:
        if any(k in n for k in needles):
            return category
    return "Other"


def window_transform(name: str) -> str | None:
    """``"ewm_expdecay7"``, ``"ewm_leaky7"``, ``"rolling_rsum7"``, ... or ``None``."""
    match = _WINDOW_RE.match(name)
    return f"{match.group(1)}_{match.group(2)}" if match else None


def importance_columns(n_horizons: int) -> list[str]:
    """``Feature, h1_gain.., h1_perm.., agg_gain, agg_perm`` (cell 57's order)."""
    return [
        "Feature",
        *[f"h{h}{GAIN_SUFFIX}" for h in range(1, n_horizons + 1)],
        *[f"h{h}{PERM_SUFFIX}" for h in range(1, n_horizons + 1)],
        "agg_gain",
        "agg_perm",
    ]


# --------------------------------------------------------------------------- #
# GBDT
# --------------------------------------------------------------------------- #
def _estimators(fitted_model: Any) -> list[Any]:
    underlying = fitted_model.model
    return list(getattr(underlying, "estimators_", [underlying]))


def gain_importance(estimator: Any, family: str) -> np.ndarray:
    """One horizon's native importance, with cell 57's library branch."""
    if "lightgbm" in family:
        return np.asarray(estimator.booster_.feature_importance(importance_type="gain"))
    if "catboost" in family:
        return np.asarray(estimator.get_feature_importance())
    return np.asarray(estimator.feature_importances_)


def lagged_design(
    fitted_model: Any,
    series: Sequence[Any],
    past_covs: Sequence[Any] | None,
    fut_covs: Sequence[Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)``: the tabular data darts builds for the underlying regressor.

    Cell 57, verbatim: apply the model's encoders with ``encode_inference``
    when it has any, then ``_create_lagged_data(..., max_samples_per_ts=None,
    sample_weight=None)``. ``y`` has one column per horizon.
    """
    encoders = getattr(fitted_model, "encoders", None)
    if encoders is not None and encoders.encoding_available:
        past_enc, fut_enc = encoders.encode_inference(
            n=fitted_model.output_chunk_length,
            target=series,
            past_covariates=past_covs,
            future_covariates=fut_covs,
        )
        series_enc = series
    else:
        series_enc, past_enc, fut_enc = series, past_covs, fut_covs
    lagged = fitted_model._create_lagged_data(
        series=series_enc,
        past_covariates=past_enc,
        future_covariates=fut_enc,
        max_samples_per_ts=None,
        sample_weight=None,
    )
    X, y = lagged[0], lagged[1]
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return np.asarray(X), y


def gbm_importances(
    fitted_model: Any,
    family: str,
    series: Sequence[Any],
    past_covs: Sequence[Any] | None,
    fut_covs: Sequence[Any] | None,
    *,
    permutation: bool = True,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int | None = -1,
    timings: dict[str, float] | None = None,
    progress: Callable[[str, Mapping[str, float]], None] | None = None,
) -> pd.DataFrame:
    """``get_gbm_importances`` (``_regression_GBDT.ipynb`` cell 57).

    ``family`` is the legacy ``model_name_prefix`` test: it only needs to
    contain ``"lightgbm"`` or ``"catboost"`` (else ``feature_importances_``).
    ``permutation=False`` skips the expensive half and fills the ``_perm``
    columns with NaN (``agg_perm`` then NaN and the sort falls back to
    ``agg_gain``); it is a test/benchmark affordance, never the thesis output.
    ``timings`` (optional) receives ``gain_s``, ``design_s`` and ``perm_s``.

    ``progress`` (optional) is told what the permutation half is doing, so a
    caller can log it; it never changes a value. It is called as
    ``progress("design", {"rows", "features", "seconds"})`` once the lagged
    design matrix is built, then ``progress("horizon", {"horizon",
    "n_horizons", "seconds"})`` after each horizon's ``permutation_importance``.
    """
    import time  # noqa: PLC0415

    estimators = _estimators(fitted_model)
    features = list(fitted_model.lagged_feature_names)
    frame: dict[str, Any] = {"Feature": features}

    t0 = time.perf_counter()
    for h, est in enumerate(estimators, start=1):
        frame[f"h{h}{GAIN_SUFFIX}"] = gain_importance(est, family)
    t1 = time.perf_counter()

    if permutation:
        from sklearn.inspection import permutation_importance  # noqa: PLC0415

        X, y = lagged_design(fitted_model, series, past_covs, fut_covs)
        if X.shape[1] != len(features):
            raise ValueError(
                f"design matrix has {X.shape[1]} columns but the model has "
                f"{len(features)} lagged feature names"
            )
        t2 = time.perf_counter()
        if progress is not None:
            progress("design", {"rows": X.shape[0], "features": X.shape[1], "seconds": t2 - t1})
        for h, est in enumerate(estimators, start=1):
            th = time.perf_counter()
            y_h = y[:, h - 1]
            result = permutation_importance(
                est, X, y_h, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
            )
            frame[f"h{h}{PERM_SUFFIX}"] = result.importances_mean
            if progress is not None:
                progress(
                    "horizon",
                    {
                        "horizon": h,
                        "n_horizons": len(estimators),
                        "seconds": time.perf_counter() - th,
                    },
                )
        t3 = time.perf_counter()
    else:
        t2 = t3 = time.perf_counter()
        for h in range(1, len(estimators) + 1):
            frame[f"h{h}{PERM_SUFFIX}"] = np.full(len(features), np.nan)

    df = pd.DataFrame(frame)
    gain_cols = [c for c in df.columns if c.endswith(GAIN_SUFFIX)]
    perm_cols = [c for c in df.columns if c.endswith(PERM_SUFFIX)]
    df["agg_gain"] = df[gain_cols].mean(axis=1)
    df["agg_perm"] = df[perm_cols].mean(axis=1)
    if timings is not None:
        timings.update({"gain_s": t1 - t0, "design_s": t2 - t1, "perm_s": t3 - t2})
        if permutation:
            timings["n_rows"] = float(X.shape[0])
    sort_key = "agg_perm" if permutation else "agg_gain"
    return df.sort_values(sort_key, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Chronos-2
# --------------------------------------------------------------------------- #
def chronos_importance(
    predictor: Any,
    data: Any,
    *,
    model: str | None = None,
    relative_scores: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """``predictor.feature_importance(data, model=..., relative_scores=True)``.

    ``_chronos2.py:526`` (zero-shot) / ``:686`` (fine-tuned). ``model``
    defaults to the predictor's single model (``Chronos2ZeroShot`` /
    ``Chronos2FT_best`` in the thesis; the name is ``"Chronos2" +
    name_suffix``). Everything else is AutoGluon's default, as in the script;
    ``kwargs`` exist for tests only. Returns AutoGluon's frame unchanged: index
    = feature name, columns ``importance, stdev, n, p99_low, p99_high``.
    """
    if model is None:
        names = list(predictor.model_names())
        if len(names) != 1:
            raise ValueError(f"predictor holds {len(names)} models {names}; pass model=")
        model = names[0]
    return predictor.feature_importance(
        data, model=model, relative_scores=relative_scores, **kwargs
    )


# --------------------------------------------------------------------------- #
# category shares (analysis notebook cell 52)
# --------------------------------------------------------------------------- #
def top_features(frame: pd.DataFrame, value_col: str, top_n: int = TOP_N) -> pd.DataFrame:
    """``frame[['Feature', value_col]].sort_values(value_col, ascending=False).head(top_n)``.

    Cells 44/47: the ``importancedict`` entries the figures plot.
    """
    feature_col = "Feature" if "Feature" in frame.columns else frame.columns[0]
    return (
        frame[[feature_col, value_col]]
        .sort_values([value_col], ascending=False)
        .head(top_n)
    )


def category_shares(two_col: pd.DataFrame, top_n: int = TOP_N) -> pd.Series:
    """``_col_share``: category shares of the top-``n`` rows of a 2-column frame.

    The frame's first column is the feature, the second the value, and it must
    already be sorted (as :func:`top_features` returns it). Negative values
    (permutation noise) are clipped to 0 before summing; the shares sum to 1.
    """
    feat_col, val_col = two_col.columns[0], two_col.columns[1]
    top = two_col.head(top_n).copy()
    top[val_col] = top[val_col].clip(lower=0)
    top["cat"] = top[feat_col].map(classify_feature)
    sums = top.groupby("cat")[val_col].sum()
    total = sums.sum()
    return sums / total if total > 0 else sums


def category_importance_matrix(
    importancedict: Mapping[str, Mapping[str, pd.DataFrame]], top_n: int = TOP_N
) -> pd.DataFrame:
    """``category_importance_matrix`` of cell 52: rows = categories, columns = ``"<group> <metric>"``.

    ``importancedict`` is ``{group: {metric: two-column frame}}`` as cell 47
    builds it (``{"activity_1": {"gain": ..., "perm": ...}, ...}``); the column
    label replaces ``activity_`` by ``Tier `` (``"Tier 1 gain"``). Rows follow
    :data:`CATEGORY_ORDER`; all-zero rows are dropped.
    """
    cols: dict[str, pd.Series] = {}
    for group, metrics in importancedict.items():
        for metric, frame in metrics.items():
            label = f"{group.replace('activity_', 'Tier ')} {metric}"
            cols[label] = category_shares(frame, top_n)
    mat = pd.DataFrame(cols).reindex(list(CATEGORY_ORDER)).fillna(0.0)
    return mat.loc[(mat != 0).any(axis=1)]


def category_shares_long(importance: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Every model label x metric of an ``importance_all``-shaped frame, long format.

    Columns ``model, metric, category, share, top_n``; ``metric`` is ``gain``
    or ``perm`` (``agg_gain`` / ``agg_perm`` ranked, top ``n``). A model whose
    ``agg_perm`` is all-NaN (permutation skipped) gets no ``perm`` rows.
    """
    rows: list[dict[str, Any]] = []
    labels = list(dict.fromkeys(importance["model"])) if "model" in importance else [""]
    for label in labels:
        sub = importance[importance["model"] == label] if label else importance
        for metric, col in (("gain", "agg_gain"), ("perm", "agg_perm")):
            if col not in sub or sub[col].isna().all():
                continue
            shares = category_shares(top_features(sub, col, top_n), top_n)
            for category in CATEGORY_ORDER:
                if category in shares.index:
                    rows.append(
                        {
                            "model": label,
                            "metric": metric,
                            "category": category,
                            "share": float(shares[category]),
                            "top_n": int(top_n),
                        }
                    )
    return pd.DataFrame(rows, columns=["model", "metric", "category", "share", "top_n"])
