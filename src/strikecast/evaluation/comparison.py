"""Pairwise model comparison on saved predictions (plan sec. 7.2).

Additive (plan sec. 1): everything here reads the long ``PredictionSet`` frame
(``region, fold, horizon, date, y_true, y_pred, origin_date, channel``, plan
sec. 5.2) out of the run store and answers "how much better is A than B", in the
metric's own units and with an honest interval. No pipeline is touched.

The four answers, in the order plan sec. 7.2 lists them:

1. :func:`paired_differences` -- the loss difference ``d = L(A) - L(B)`` on
   every shared ``(region, origin_date, horizon)``, and :func:`mean_difference`
   with a :func:`block_bootstrap_ci` over forecast origins. Negative ``d``
   means A is better, because ``d`` is A's loss minus B's.
2. :func:`diebold_mariano` -- DM on the per-origin loss differential with a
   Newey-West HAC variance and the Harvey-Leybourne-Newbold small-sample
   correction.
3. :func:`cliffs_delta` -- effect size on the per-origin differences.
4. :func:`compare_pair` / :func:`pairwise_table` -- the summary rows that
   ``report/pairwise_<metric>.csv`` is written from.

Why the blocking, in one line: forecasts from consecutive origins overlap
(plan flag F4), so neither the rows nor the per-origin differentials are
independent. The block bootstrap resamples *blocks of consecutive origins* and
the DM variance is HAC, which is the pair of corrections plan sec. 9 asks for.

Notes
-----

``Q1`` **``d`` is oriented A minus B throughout.** Negative is "A wins". The
    summary also reports ``pct_improvement``, which flips the sign so that a
    positive percentage always means "A is better than B", because that is how
    the paper reads.

``Q2`` **Pairing is an inner join on ``(region, origin_date, horizon)``.** Two
    models that were backtested on different schedules (the diff family's dates
    shift, flag F33/F51) therefore compare on their intersection, and
    :func:`paired_differences` reports ``n_a``, ``n_b`` and ``n`` so a silent
    partial overlap is visible. ``fold`` is deliberately **not** a join key: it
    is an index into each run's own schedule, and two runs with different start
    dates give the same ``origin_date`` different fold numbers. The merge is
    validated one-to-one, so a composite forecaster's multi-channel frame has
    to be narrowed with ``channel=`` first rather than silently fanning out.

``Q3`` **The block bootstrap resamples origins, never rows.** A drawn origin
    brings all its regions and horizons with it (plan sec. 7.1). The default is
    a *moving*-block bootstrap with ``block_length=7``, matching the 7-day
    overlap of a 7-day horizon; ``block_length=1`` degenerates to the i.i.d.
    origin bootstrap. The choice of 7 is the plan's, not a fitted value.

``Q4`` **Blocks are drawn on the sorted unique origins and then trimmed.**
    ``ceil(n / L)`` blocks are drawn and the concatenation is cut to exactly
    ``n`` origins, so every replicate has the same size as the sample. Origins
    are *not* wrapped around (a circular block bootstrap would tie the last
    origin to the first, which is meaningless for forecast time).

``Q5`` **The DM series is the mean differential per origin.** Regions are
    averaged within an origin before the HAC variance, because the HAC
    correction is for serial dependence along time; cross-sectional dependence
    between regions at the same origin is handled by keeping the origin whole
    (Q3). Per horizon, the lag is ``horizon - 1``; pooled, it is
    ``max(horizon) - 1``, which is the plan's "bandwidth ``horizon - 1``".

``Q6`` **Newey-West means Bartlett weights.** ``gamma_0 + 2 * sum_k w_k
    gamma_k`` with ``w_k = 1 - k / (L + 1)``. Passing ``weights="truncated"``
    gives the uniform weights of the original Diebold-Mariano paper instead;
    the HAC estimate can then go negative, in which case the statistic is NaN
    rather than a complex number. Bartlett is the default because plan sec. 7.2
    says Newey-West.

``Q7`` **HLN is applied by default.** ``DM* = DM * sqrt((n + 1 - 2h +
    h(h-1)/n) / n)`` and the p-value comes from ``t_{n-1}``, not from the
    normal. Plan sec. 7.2: "Harvey-Leybourne-Newbold small-sample correction
    applied." ``hln=False`` gives the uncorrected statistic against the normal.

``Q8`` **Cliff's delta is the paired form.** On the per-origin differences,
    ``delta = (#(d < 0) - #(d > 0)) / n``: the probability that A beats B on a
    random forecast origin minus the probability that B beats A, already in
    ``[-1, 1]``. Ties contribute zero. Oriented so that *positive delta means A
    is better*, i.e. the opposite sign of ``d`` (Q1); the column is named
    ``cliffs_delta_a_better`` to keep that explicit.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

__all__ = [
    "BREAKDOWN_COLUMNS",
    "DEFAULT_BLOCK_LENGTH",
    "DEFAULT_N_BOOT",
    "DMResult",
    "FAMILY_COLUMNS",
    "FAMILY_PREFIXES",
    "JOIN_KEYS",
    "LOSSES",
    "LossName",
    "PairSummary",
    "block_bootstrap_ci",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "compare_pair",
    "diebold_mariano",
    "family_comparison",
    "family_of",
    "loss_values",
    "mean_difference",
    "moving_block_indices",
    "paired_differences",
    "pairwise_breakdowns",
    "pairwise_table",
    "per_origin_mean",
    "tier_regions",
    "write_family_comparison",
    "write_pairwise",
]

logger = logging.getLogger(__name__)

LossName = Literal["squared", "absolute"]

#: The two losses plan sec. 7.2 asks for: "for squared and absolute error".
LOSSES: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "squared": lambda y_true, y_pred: (y_pred - y_true) ** 2,
    "absolute": lambda y_true, y_pred: np.abs(y_pred - y_true),
}

#: What makes two rows the same forecast instance (Q2).
JOIN_KEYS: tuple[str, ...] = ("region", "origin_date", "horizon")

#: Plan sec. 7.1: "a moving-block variant with block length 7 covers the overlap".
DEFAULT_BLOCK_LENGTH = 7

#: Plan sec. 7.1: "1000 replicates, percentile intervals".
DEFAULT_N_BOOT = 1000


# --------------------------------------------------------------------------- #
# losses and pairing
# --------------------------------------------------------------------------- #
def loss_values(frame: pd.DataFrame, loss: LossName | str = "squared") -> np.ndarray:
    """``L(y_true, y_pred)`` for one long prediction frame.

    No clipping. :mod:`strikecast.evaluation.metrics` clips predictions at zero
    before scoring (its Q1) because the legacy code does; the comparison layer
    is new, is not reproducing anything, and a clip here would quietly change
    which model wins on negative predictions. Clip upstream if that is wanted.
    """
    try:
        fn = LOSSES[loss]
    except KeyError:
        raise ValueError(f"unknown loss {loss!r}, expected one of {sorted(LOSSES)}") from None
    return np.asarray(
        fn(frame["y_true"].to_numpy(float), frame["y_pred"].to_numpy(float)), dtype=float
    )


def paired_differences(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    loss: LossName | str = "squared",
    keys: Sequence[str] = JOIN_KEYS,
    channel: str | None = None,
) -> pd.DataFrame:
    """Loss difference ``d = L(A) - L(B)`` on every shared forecast instance (Q1, Q2).

    Returns the join keys plus ``loss_a``, ``loss_b`` and ``d``, sorted by the
    keys so the row order is deterministic. Raises when the two frames share no
    instance at all, because a comparison of nothing is a bug and not a result.
    """
    keys = list(keys)
    a_sub = a[a["channel"] == channel] if channel is not None and "channel" in a else a
    b_sub = b[b["channel"] == channel] if channel is not None and "channel" in b else b
    left = a_sub.loc[:, keys].copy()
    left["loss_a"] = loss_values(a_sub, loss)
    right = b_sub.loc[:, keys].copy()
    right["loss_b"] = loss_values(b_sub, loss)
    try:
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    except pd.errors.MergeError as exc:  # duplicate keys: usually an unfiltered channel
        raise ValueError(
            f"{keys} is not unique in at least one frame ({exc}); "
            "pass channel=... if the PredictionSet holds several channels"
        ) from exc
    if merged.empty:
        raise ValueError(
            f"models share no forecast instance on {keys} "
            f"({len(left)} rows in A, {len(right)} rows in B)"
        )
    merged["d"] = merged["loss_a"] - merged["loss_b"]
    merged.attrs["n_a"] = int(len(left))
    merged.attrs["n_b"] = int(len(right))
    return merged.sort_values(keys).reset_index(drop=True)


def per_origin_mean(diff: pd.DataFrame, column: str = "d") -> pd.Series:
    """Mean of ``column`` per ``origin_date``, indexed by the sorted origins (Q5)."""
    return diff.groupby("origin_date")[column].mean().sort_index()


def mean_difference(diff: pd.DataFrame) -> dict[str, float]:
    """The headline numbers: mean loss of A and B, their difference, and the ratio.

    ``pct_improvement`` is ``100 * (mean_b - mean_a) / mean_b`` -- positive when
    A is better (Q1). For the squared loss the caller usually wants the RMSE
    version, so ``rmse_a``/``rmse_b``/``pct_improvement_rmse`` are included and
    are meaningful only when ``loss="squared"``.
    """
    mean_a = float(diff["loss_a"].mean())
    mean_b = float(diff["loss_b"].mean())
    rmse_a, rmse_b = math.sqrt(max(mean_a, 0.0)), math.sqrt(max(mean_b, 0.0))
    return {
        "mean_loss_a": mean_a,
        "mean_loss_b": mean_b,
        "mean_diff": float(diff["d"].mean()),
        "pct_improvement": float(100.0 * (mean_b - mean_a) / mean_b) if mean_b else float("nan"),
        "rmse_a": rmse_a,
        "rmse_b": rmse_b,
        "pct_improvement_rmse": float(100.0 * (rmse_b - rmse_a) / rmse_b)
        if rmse_b
        else float("nan"),
    }


# --------------------------------------------------------------------------- #
# block bootstrap over forecast origins
# --------------------------------------------------------------------------- #
def moving_block_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """One moving-block resample of ``range(n)``, trimmed to length ``n`` (Q3, Q4).

    Blocks start uniformly on ``0 .. n - block_length`` and are *not* circular,
    so the last ``block_length - 1`` origins are slightly under-sampled. That is
    the standard moving-block trade-off and is preferred here over wrapping
    forecast time around on itself.
    """
    if n <= 0:
        return np.empty(0, dtype=int)
    length = max(1, min(int(block_length), n))
    n_blocks = math.ceil(n / length)
    starts = rng.integers(0, n - length + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(length)[None, :]).ravel()
    return idx[:n]


def block_bootstrap_ci(
    diff: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float] | None = None,
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    column: str = "d",
) -> dict[str, float]:
    """Percentile CI for ``statistic(diff)`` under a moving-block origin bootstrap.

    ``statistic`` defaults to the mean of ``column``. A resample draws origins,
    not rows: each drawn origin contributes **all** of its regions and horizons
    (Q3). Deterministic given ``seed``. The returned dict carries the point
    estimate, the interval, the bootstrap SE and the settings used, so a row of
    ``pairwise_<metric>.csv`` is self-describing.
    """
    stat = statistic if statistic is not None else (lambda f: float(f[column].mean()))
    origins = np.sort(diff["origin_date"].unique())
    n = int(origins.size)
    rng = np.random.default_rng(seed)
    point = float(stat(diff))
    if n < 2:
        return {
            "estimate": point,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "boot_se": float("nan"),
            "n_origins": n,
            "block_length": int(block_length),
            "n_boot": int(n_boot),
        }
    reps = np.empty(int(n_boot), dtype=float)
    if statistic is None:
        # The default statistic is the mean of `column`, and the mean over a
        # concatenation of whole origins is (sum of the drawn origins' sums) /
        # (sum of their non-NaN counts). Same number as rebuilding the frame,
        # same draws, without 1000 `pd.concat` calls -- which is what makes a
        # report over a real store finish in seconds instead of minutes.
        grouped = diff.groupby("origin_date")[column]
        sums = grouped.sum().sort_index().to_numpy(float)
        counts = grouped.count().sort_index().to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            for i in range(int(n_boot)):
                idx = moving_block_indices(n, block_length, rng)
                total = counts[idx].sum()
                reps[i] = sums[idx].sum() / total if total else float("nan")
    else:
        groups = {o: sub for o, sub in diff.groupby("origin_date")}
        for i in range(int(n_boot)):
            idx = moving_block_indices(n, block_length, rng)
            sample = pd.concat([groups[origins[j]] for j in idx], ignore_index=True)
            reps[i] = float(stat(sample))
    lo, hi = np.percentile(reps, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "estimate": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "boot_se": float(reps.std(ddof=1)),
        "n_origins": n,
        "block_length": int(min(max(1, block_length), n)),
        "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DMResult:
    """A Diebold-Mariano test on one loss differential series."""

    stat: float
    p_value: float
    mean_d: float
    n: int
    lag: int
    hac_var: float
    hln: bool
    weights: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hac_variance(d: np.ndarray, lag: int, weights: str) -> float:
    """Long-run variance of the mean of ``d``: ``gamma_0 + 2 sum_k w_k gamma_k`` (Q6)."""
    n = d.size
    centred = d - d.mean()
    gamma0 = float(centred @ centred) / n
    total = gamma0
    for k in range(1, min(int(lag), n - 1) + 1):
        gamma_k = float(centred[k:] @ centred[:-k]) / n
        if weights == "bartlett":
            w = 1.0 - k / (lag + 1.0)
        elif weights == "truncated":
            w = 1.0
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"unknown HAC weights {weights!r}")
        total += 2.0 * w * gamma_k
    return total


def diebold_mariano(
    d: Sequence[float] | np.ndarray | pd.Series,
    horizon: int,
    *,
    lag: int | None = None,
    hln: bool = True,
    weights: str = "bartlett",
) -> DMResult:
    """Diebold-Mariano on a per-origin loss differential series (Q5-Q7).

    ``d`` is ordered by forecast origin (use :func:`per_origin_mean`).
    ``horizon`` drives both the default HAC lag (``horizon - 1``) and the HLN
    correction. A negative HAC variance -- possible with ``weights="truncated"``
    -- yields NaN rather than an imaginary statistic.

    The null is equal expected loss; the alternative is two-sided. Under
    ``hln=True`` the reference distribution is ``t_{n-1}``, otherwise the
    standard normal.
    """
    if weights not in ("bartlett", "truncated"):
        raise ValueError(f"unknown HAC weights {weights!r}, expected 'bartlett' or 'truncated'")
    arr = np.asarray(pd.Series(d).to_numpy(), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    h = max(1, int(horizon))
    L = (h - 1) if lag is None else int(lag)
    L = max(0, min(L, max(n - 1, 0)))
    if n < 2:
        return DMResult(float("nan"), float("nan"), float("nan"), n, L, float("nan"), hln, weights)
    var = _hac_variance(arr, L, weights)
    mean_d = float(arr.mean())
    if not np.isfinite(var) or var <= 0.0:
        return DMResult(float("nan"), float("nan"), mean_d, n, L, float(var), hln, weights)
    stat = mean_d / math.sqrt(var / n)
    if hln:
        factor = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
        if factor <= 0.0:
            return DMResult(float("nan"), float("nan"), mean_d, n, L, float(var), hln, weights)
        stat *= math.sqrt(factor)
        p = float(2.0 * _scipy_stats.t.sf(abs(stat), df=n - 1))
    else:
        p = float(2.0 * _scipy_stats.norm.sf(abs(stat)))
    return DMResult(float(stat), p, mean_d, n, L, float(var), hln, weights)


# --------------------------------------------------------------------------- #
# effect size
# --------------------------------------------------------------------------- #
def cliffs_delta(d: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Paired Cliff's delta on per-origin loss differences (Q8).

    ``(#(d < 0) - #(d > 0)) / n``: the probability that A beats B on a random
    forecast origin minus the probability that B beats A. Positive means A is
    better. NaN on an empty input.
    """
    arr = np.asarray(pd.Series(d).to_numpy(), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float((np.count_nonzero(arr < 0) - np.count_nonzero(arr > 0)) / arr.size)


def cliffs_delta_magnitude(delta: float) -> str:
    """Romano et al. (2006) thresholds: .147 / .33 / .474 -> negligible..large."""
    if not np.isfinite(delta):
        return "undefined"
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairSummary:
    """One row of ``report/pairwise_<metric>.csv`` (plan sec. 7.2)."""

    model_a: str
    model_b: str
    loss: str
    scope: str
    horizon: int | None
    n_instances: int
    n_origins: int
    mean_loss_a: float
    mean_loss_b: float
    mean_diff: float
    pct_improvement: float
    ci_lo: float
    ci_hi: float
    boot_se: float
    dm_stat: float
    dm_p: float
    dm_lag: int
    cliffs_delta_a_better: float
    cliffs_magnitude: str
    block_length: int
    n_boot: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_pair(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    label_a: str = "A",
    label_b: str = "B",
    loss: LossName | str = "squared",
    horizon: int | None = None,
    scope: str | None = None,
    channel: str | None = None,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    hln: bool = True,
    weights: str = "bartlett",
) -> PairSummary:
    """The full plan sec. 7.2 answer for one ordered pair, on one scope.

    ``horizon=None`` pools every horizon and uses ``max(horizon) - 1`` as the
    HAC lag (Q5); an integer restricts to that horizon and uses ``horizon - 1``.
    ``scope`` is a free-form label for the breakdown the row belongs to
    (``"pooled"``, ``"h=3"``, ``"tier=3"``, ...) and is carried through
    untouched; ``None`` derives it from ``horizon``.
    """
    if scope is None:
        scope = "pooled" if horizon is None else f"h={horizon}"
    diff = paired_differences(a, b, loss=loss, channel=channel)
    if horizon is not None:
        diff = diff[diff["horizon"] == horizon].reset_index(drop=True)
        if diff.empty:
            raise ValueError(f"no shared instances at horizon {horizon}")
        dm_h = int(horizon)
    else:
        dm_h = int(pd.to_numeric(diff["horizon"]).max())
    head = mean_difference(diff)
    boot = block_bootstrap_ci(
        diff, block_length=block_length, n_boot=n_boot, alpha=alpha, seed=seed
    )
    per_origin = per_origin_mean(diff)
    dm = diebold_mariano(per_origin, dm_h, hln=hln, weights=weights)
    delta = cliffs_delta(per_origin)
    return PairSummary(
        model_a=label_a,
        model_b=label_b,
        loss=str(loss),
        scope=scope,
        horizon=horizon,
        n_instances=int(len(diff)),
        n_origins=int(boot["n_origins"]),
        mean_loss_a=head["mean_loss_a"],
        mean_loss_b=head["mean_loss_b"],
        mean_diff=head["mean_diff"],
        pct_improvement=head["pct_improvement"],
        ci_lo=boot["ci_lo"],
        ci_hi=boot["ci_hi"],
        boot_se=boot["boot_se"],
        dm_stat=dm.stat,
        dm_p=dm.p_value,
        dm_lag=dm.lag,
        cliffs_delta_a_better=delta,
        cliffs_magnitude=cliffs_delta_magnitude(delta),
        block_length=int(boot["block_length"]),
        n_boot=int(boot["n_boot"]),
    )


def pairwise_table(
    frames: Mapping[str, pd.DataFrame],
    *,
    loss: LossName | str = "squared",
    horizons: Iterable[int | None] = (None,),
    channel: str | None = None,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    hln: bool = True,
    weights: str = "bartlett",
    ordered: bool = False,
) -> pd.DataFrame:
    """Every pair of ``frames``, on every requested horizon scope.

    ``ordered=False`` (the default) emits each unordered pair once, in the
    insertion order of ``frames``; ``ordered=True`` emits both directions, which
    is what a symmetric p-value matrix wants. Every pair gets its own bootstrap
    stream derived from ``seed``, so a row's interval does not depend on how
    many pairs preceded it.
    """
    labels = list(frames)
    rows: list[dict[str, Any]] = []
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if la == lb:
                continue
            if not ordered and j < i:
                continue
            for h in horizons:
                pair_seed = int(seed) + 1_000 * (i * len(labels) + j)
                rows.append(
                    compare_pair(
                        frames[la],
                        frames[lb],
                        label_a=la,
                        label_b=lb,
                        loss=loss,
                        horizon=h,
                        channel=channel,
                        block_length=block_length,
                        n_boot=n_boot,
                        alpha=alpha,
                        seed=pair_seed,
                        hln=hln,
                        weights=weights,
                    ).as_dict()
                )
    return pd.DataFrame.from_records(rows)


def write_pairwise(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``report/pairwise_<metric>.csv`` with ``index=False``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


# --------------------------------------------------------------------------- #
# breakdowns: global, per horizon, per activity tier (plan sec. 7.2 item 6)
# --------------------------------------------------------------------------- #
def _breakdown_columns() -> tuple[str, ...]:
    """:class:`PairSummary`'s fields with ``breakdown``/``tier`` after ``scope``."""
    names = [f.name for f in fields(PairSummary)]
    cut = names.index("scope") + 1
    return (*names[:cut], "breakdown", "tier", *names[cut:])


#: Column order of ``report/pairwise_<metric>.csv``.
BREAKDOWN_COLUMNS: tuple[str, ...] = _breakdown_columns()


def tier_regions(
    activity_by_region: Mapping[str, Any], regions: Iterable[str] | None = None
) -> dict[str, list[str]]:
    """``{tier label: [regions]}``, restricted to ``regions`` when given.

    The tier label is ``str(level)`` so it survives a CSV round trip; regions
    whose level is missing or NaN are dropped, which is the same thing
    :func:`strikecast.evaluation.aggregate.evaluate` does to its activity views
    (its Q4) and is why an activity breakdown can cover fewer regions than the
    global one.
    """
    keep = None if regions is None else set(regions)
    out: dict[str, list[str]] = {}
    for region, level in activity_by_region.items():
        if keep is not None and region not in keep:
            continue
        if level is None or (isinstance(level, float) and math.isnan(level)):
            continue
        out.setdefault(str(level), []).append(str(region))
    return {tier: sorted(members) for tier, members in sorted(out.items())}


def pairwise_breakdowns(
    frames: Mapping[str, pd.DataFrame],
    *,
    loss: LossName | str = "squared",
    horizons: Iterable[int] | None = None,
    activity_by_region: Mapping[str, Any] | None = None,
    channel: str | None = None,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    hln: bool = True,
    weights: str = "bartlett",
    ordered: bool = False,
) -> pd.DataFrame:
    """Every pair of ``frames`` on every plan sec. 7.2 item 6 breakdown.

    Three breakdowns per pair, each a row: ``global`` (every horizon pooled),
    one ``horizon`` row per horizon, and one ``tier`` row per activity level
    when ``activity_by_region`` is given. ``horizons=None`` uses every horizon
    present in the frames; pass ``()`` to suppress the per-horizon rows.

    Degrades rather than raises: a pair with no shared forecast instance on a
    breakdown (different schedules, a tier one model never saw) is logged and
    skipped, so a partial store still produces a table. The returned frame
    always has :data:`BREAKDOWN_COLUMNS`, empty or not.
    """
    labels = list(frames)
    if horizons is None:
        present: set[int] = set()
        for frame in frames.values():
            present.update(int(h) for h in pd.unique(frame["horizon"]))
        horizon_list: list[int] = sorted(present)
    else:
        horizon_list = [int(h) for h in horizons]
    tiers = (
        {}
        if activity_by_region is None
        else tier_regions(
            activity_by_region,
            {r for frame in frames.values() for r in pd.unique(frame["region"])},
        )
    )

    rows: list[dict[str, Any]] = []
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if la == lb or (not ordered and j < i):
                continue
            pair_seed = int(seed) + 1_000 * (i * len(labels) + j)
            jobs: list[tuple[str, str | None, int | None, str]] = [
                ("global", None, None, "pooled")
            ]
            jobs += [("horizon", None, h, f"h={h}") for h in horizon_list]
            jobs += [("tier", tier, None, f"tier={tier}") for tier in tiers]
            for offset, (breakdown, tier, horizon, scope) in enumerate(jobs):
                a, b = frames[la], frames[lb]
                if tier is not None:
                    members = set(tiers[tier])
                    a = a[a["region"].isin(members)]
                    b = b[b["region"].isin(members)]
                    if a.empty or b.empty:
                        logger.debug("%s vs %s: no rows in tier %s", la, lb, tier)
                        continue
                try:
                    summary = compare_pair(
                        a,
                        b,
                        label_a=la,
                        label_b=lb,
                        loss=loss,
                        horizon=horizon,
                        scope=scope,
                        channel=channel,
                        block_length=block_length,
                        n_boot=n_boot,
                        alpha=alpha,
                        seed=pair_seed + offset,
                        hln=hln,
                        weights=weights,
                    )
                except ValueError as exc:
                    logger.debug("%s vs %s on %s: %s", la, lb, scope, exc)
                    continue
                row = summary.as_dict()
                row["breakdown"] = breakdown
                row["tier"] = tier
                rows.append(row)
    return pd.DataFrame.from_records(rows, columns=list(BREAKDOWN_COLUMNS))


# --------------------------------------------------------------------------- #
# seed-aware family comparison (plan sec. 7.2 item 5)
# --------------------------------------------------------------------------- #
#: Model-variant prefix -> the "model type" plan sec. 7.2 compares. Longest
#: prefix wins; an unknown name falls back to its first underscore-separated
#: token, so a new variant lands in a family named after itself rather than in
#: the wrong one.
FAMILY_PREFIXES: Mapping[str, str] = {
    "lightgbm": "gbdt",
    "xgboost": "gbdt",
    "catboost": "gbdt",
    "lstm": "rnn",
    "gru": "rnn",
    "naive": "naive",
    "seasonal": "naive",
    "linear": "linear",
    "arima": "arima",
    "chronos": "chronos",
    "hurdle": "hurdle",
    "spe": "classifier",
    "damage": "damage",
}

#: Column order of ``report/family_comparison.csv``.
FAMILY_COLUMNS: tuple[str, ...] = (
    "experiment",
    "stage",
    "metric",
    "aggregate",
    "family_a",
    "family_b",
    "n_seeds",
    "seeds",
    "mean_a",
    "mean_b",
    "mean_diff",
    "pct_improvement",
    "sd_diff",
    "t_lo",
    "t_hi",
    "p_value",
    "n_runs_a",
    "n_runs_b",
    "deterministic",
)


def family_of(model: str, overrides: Mapping[str, str] | None = None) -> str:
    """Which "model type" a model variant belongs to (:data:`FAMILY_PREFIXES`).

    ``overrides`` is an explicit ``{model: family}`` map and always wins, so a
    caller that knows better than the prefix table -- the experiment YAML, or
    the paper's own grouping -- can say so.
    """
    if overrides is not None and model in overrides:
        return str(overrides[model])
    for prefix in sorted(FAMILY_PREFIXES, key=len, reverse=True):
        if model.startswith(prefix):
            return FAMILY_PREFIXES[prefix]
    return model.split("_", 1)[0]


def _family_values(
    frame: pd.DataFrame, aggregate: str, lower_is_better: bool
) -> pd.DataFrame:
    """One value per ``(experiment, stage, metric, family, seed)``.

    ``aggregate="mean"`` is plan sec. 7.2's "the family as a whole (all its
    variants and paradigms)"; ``aggregate="best"`` is its "the best
    configuration per family (as in Table 4)" and takes the minimum for an
    error metric (``lower_is_better``), the maximum otherwise.
    """
    keys = ["experiment", "stage", "metric", "family", "seed"]
    grouped = frame.groupby(keys, sort=True)
    if aggregate == "mean":
        values = grouped["value"].mean()
    elif aggregate == "best":
        values = grouped["value"].min() if lower_is_better else grouped["value"].max()
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}, expected 'mean' or 'best'")
    out = values.reset_index(name="value")
    out["n_runs"] = grouped["value"].size().to_numpy()
    return out


def family_comparison(
    rows: Any,
    *,
    families: Mapping[str, str] | None = None,
    metrics: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
    aggregate: str = "mean",
    lower_is_better: bool = True,
    stochastic: Mapping[str, bool] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Plan sec. 7.2 item 5: family A minus family B on the SAME seed.

    ``rows`` is the long metric frame of
    :func:`strikecast.evaluation.leaderboard.metric_frame` (or the
    :class:`~strikecast.evaluation.leaderboard.MetricRow` list it is built
    from). Per seed, each family is reduced to one number (``aggregate``); the
    two families are then joined **on the seed**, giving N paired differences,
    and the row reports their mean, SD, Student-t interval and the paired
    t-test p-value. That is what separates "A is better" from "A happened to
    get a lucky seed".

    Notes
    -----

    ``Q9`` **Paradigms are pooled into the family.** Plan sec. 7.2 defines the
        family level as "all its variants and paradigms", so a family's per-seed
        value averages (or takes the best of) every variant x paradigm cell that
        completed for that seed. A family that ran more paradigms than another
        is therefore not penalised by missing cells, but it *is* averaged over a
        different population; ``n_runs_a`` / ``n_runs_b`` record how many cells
        went into each side so that is visible.

    ``Q10`` **Two deterministic families get a marker, never an interval.**
        Deterministic models are run once and broadcast across seeds (plan
        sec. 5.4), so their per-seed differences are identical by construction
        and an interval computed from them would be a zero-width lie. When
        ``stochastic`` says every model of BOTH families is deterministic the
        row keeps its mean difference, sets ``sd_diff``/``t_lo``/``t_hi``/
        ``p_value`` to NaN and sets ``deterministic=True`` (plan sec. 7.1).

    ``Q11`` **Only seeds both families completed are paired.** A missing seed
        shrinks ``n_seeds``; it is never filled in. ``seeds`` lists the ones
        actually used.
    """
    from .leaderboard import metric_frame  # noqa: PLC0415 - avoids an import cycle
    from .seeds import t_interval  # noqa: PLC0415 - avoids an import cycle

    frame = rows if isinstance(rows, pd.DataFrame) else metric_frame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(FAMILY_COLUMNS))
    frame = frame.copy()
    if stages is not None:
        frame = frame[frame["stage"].isin(list(stages))]
    if metrics is not None:
        frame = frame[frame["metric"].isin(list(metrics))]
    if frame.empty:
        return pd.DataFrame(columns=list(FAMILY_COLUMNS))
    frame["family"] = [family_of(str(m), families) for m in frame["model"]]
    deterministic_family = {
        family: bool(
            stochastic is not None
            and len(models) > 0
            and all(not stochastic.get(str(m), True) for m in models)
        )
        for family, models in frame.groupby("family")["model"].unique().items()
    }

    values = _family_values(frame, aggregate, lower_is_better)
    records: list[dict[str, Any]] = []
    for (experiment, stage, metric), sub in values.groupby(
        ["experiment", "stage", "metric"], sort=True
    ):
        family_list = sorted(sub["family"].unique())
        for i, fa in enumerate(family_list):
            for fb in family_list[i + 1 :]:
                left = sub[sub["family"] == fa].set_index("seed").sort_index()
                right = sub[sub["family"] == fb].set_index("seed").sort_index()
                seeds = [int(s) for s in left.index.intersection(right.index)]
                if not seeds:
                    logger.debug("%s vs %s (%s): no shared seed", fa, fb, metric)
                    continue
                a = left.loc[seeds, "value"].to_numpy(float)
                b = right.loc[seeds, "value"].to_numpy(float)
                diff = a - b
                mean_b = float(np.mean(b))
                is_det = bool(
                    deterministic_family.get(fa, False) and deterministic_family.get(fb, False)
                )
                if is_det or diff.size < 2:
                    sd = float("nan")
                    lo = hi = float("nan")
                    p_value = float("nan")
                else:
                    sd = float(np.std(diff, ddof=1))
                    lo, hi = t_interval(diff, alpha)
                    p_value = float(_scipy_stats.ttest_1samp(diff, 0.0).pvalue)
                records.append(
                    {
                        "experiment": experiment,
                        "stage": stage,
                        "metric": metric,
                        "aggregate": aggregate,
                        "family_a": fa,
                        "family_b": fb,
                        "n_seeds": len(seeds),
                        "seeds": ";".join(str(s) for s in seeds),
                        "mean_a": float(np.mean(a)),
                        "mean_b": mean_b,
                        "mean_diff": float(np.mean(diff)),
                        "pct_improvement": (
                            float(100.0 * (mean_b - float(np.mean(a))) / mean_b)
                            if mean_b
                            else float("nan")
                        ),
                        "sd_diff": sd,
                        "t_lo": lo,
                        "t_hi": hi,
                        "p_value": p_value,
                        "n_runs_a": int(left.loc[seeds, "n_runs"].sum()),
                        "n_runs_b": int(right.loc[seeds, "n_runs"].sum()),
                        "deterministic": is_det,
                    }
                )
    return pd.DataFrame.from_records(records, columns=list(FAMILY_COLUMNS))


def write_family_comparison(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``report/family_comparison.csv`` with ``index=False``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
