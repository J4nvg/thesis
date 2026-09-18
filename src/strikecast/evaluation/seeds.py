"""Cross-seed aggregation and confidence intervals (plan sec. 5.4, sec. 7.1).

Additive (plan sec. 1): this module consumes the ``global.json`` metrics that
:mod:`strikecast.evaluation.leaderboard` collects and writes
``report/leaderboard_ci.csv``. It never reruns a stage and never changes a
metric.

What is reported, per ``(experiment, model, paradigm, stage, metric)``:

``n``, ``mean``, ``std``, ``min``, ``max``
    Over the evaluation seeds. ``std`` is the *sample* standard deviation
    (``ddof=1``), which is the one the t-interval below is built from.
``t_lo`` / ``t_hi``
    The Student-t interval ``mean +- t_{1-alpha/2, n-1} * std / sqrt(n)``.
    Plan sec. 7.1: "with five to ten seeds the t-interval is the honest choice".
``boot_lo`` / ``boot_hi``
    A percentile bootstrap of the *mean over seeds*: resample the ``n`` seed
    values with replacement ``n_boot`` times and take the ``alpha/2`` and
    ``1 - alpha/2`` percentiles of the resampled means. This is the
    training-variance bootstrap over seeds, **not** the evaluation-sample block
    bootstrap over forecast origins -- that one lives in
    :mod:`strikecast.evaluation.comparison` and resamples ``origin_date``.
``deterministic``
    True when the model was declared ``stochastic=False`` (plan sec. 5.4:
    naives, linear, ARIMA). Such a model is run once and *broadcast*: the
    single value is reported as a point value with ``std``/``t_*``/``boot_*``
    all NaN and ``n_seeds_declared`` recording how many seeds it stands for.
    An explicit marker, never a zero-width interval (plan sec. 7.1).

Notes
-----

``Q1`` **``std`` uses ``ddof=1`` and is NaN for a single seed.** A stochastic
    model that happens to have one completed seed is reported as ``n=1`` with
    NaN spread rather than as deterministic. The two cases are different and
    the ``deterministic`` column is what tells them apart.

``Q2`` **The bootstrap over five seeds is weak by construction.** With ``n``
    seed values there are only ``n**n`` distinct resamples and the percentile
    interval cannot reach outside ``[min, max]``. It is reported because plan
    sec. 7.1 asks for it; the t-interval is the one to quote. See the open
    question in ``docs/REFACTOR_PROGRESS.md``.

``Q3`` **Seeds are never pooled across stages or paradigms.** Each
    ``(stage, paradigm)`` cell is its own population, because the plan sweeps
    seeds over the test stage only and leaves CV single-seed (sec. 5.4). A CV
    cell therefore usually comes out with ``n=1``.

``Q4`` **A missing seed is a smaller ``n``, not an error.** Aggregation is over
    whatever completed. ``expected_seeds`` records what was asked for so the
    report can show ``4/5``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from .leaderboard import MetricRow, collect_metric_rows, metric_frame

__all__ = [
    "CI_COLUMNS",
    "DEFAULT_ALPHA",
    "DEFAULT_N_BOOT",
    "GROUP_KEYS",
    "aggregate_seeds",
    "bootstrap_mean_interval",
    "broadcast_deterministic",
    "leaderboard_ci",
    "t_interval",
    "write_leaderboard_ci",
]

#: Plan sec. 7.1: 95% intervals.
DEFAULT_ALPHA = 0.05

#: Plan sec. 7.1: "1000 replicates, percentile intervals".
DEFAULT_N_BOOT = 1000

#: The cell a seed population is formed over (plan sec. 7.1).
GROUP_KEYS: tuple[str, ...] = ("experiment", "model", "paradigm", "stage", "metric")

#: Column order of ``leaderboard_ci.csv``.
CI_COLUMNS: tuple[str, ...] = (
    *GROUP_KEYS,
    "n",
    "mean",
    "std",
    "min",
    "max",
    "t_lo",
    "t_hi",
    "boot_lo",
    "boot_hi",
    "deterministic",
    "expected_seeds",
    "seeds",
)


def t_interval(
    values: Sequence[float] | np.ndarray, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float]:
    """Two-sided Student-t interval for the mean.

    ``mean +- t_{1-alpha/2, n-1} * s / sqrt(n)`` with ``s`` the ``ddof=1``
    standard deviation. Returns ``(nan, nan)`` for fewer than two values, which
    is the honest answer rather than a degenerate point interval (Q1).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    half = float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, n - 1)) * sem
    return (mean - half, mean + half)


def bootstrap_mean_interval(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    An i.i.d. bootstrap: the seed values are exchangeable by construction, so
    there is nothing to block on here (unlike the origin bootstrap of
    :mod:`strikecast.evaluation.comparison`). Deterministic given ``seed``.
    Returns ``(nan, nan)`` for fewer than two values (Q1, Q2).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(int(n_boot), n))
    means = arr[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return (float(lo), float(hi))


def broadcast_deterministic(
    rows: Iterable[MetricRow],
    *,
    stochastic: Mapping[str, bool] | None = None,
    eval_seeds: Sequence[int] | None = None,
) -> list[MetricRow]:
    """Replicate a deterministic model's single run across ``eval_seeds``.

    Plan sec. 5.4: "Deterministic models (``stochastic: false``: naives, linear,
    ARIMA) run once and are broadcast across seeds in reports." ``stochastic``
    maps a model name to its :attr:`strikecast.models.spec.ModelSpec.stochastic`
    flag; a model absent from the map is treated as stochastic, because
    broadcasting something that is not deterministic would fabricate agreement.

    The broadcast only *adds* rows for seeds that are missing, so a
    deterministic model that was (wastefully) run under several seeds keeps its
    real rows and is not overwritten.
    """
    rows = list(rows)
    if stochastic is None or eval_seeds is None:
        return rows
    wanted = [int(s) for s in eval_seeds]
    by_key: dict[tuple[str, str, str, str], list[MetricRow]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)
    out = list(rows)
    for key, group in by_key.items():
        model = key[1]
        if stochastic.get(model, True):
            continue
        present = {r.seed for r in group}
        source = min(group, key=lambda r: r.seed)
        for seed in wanted:
            if seed in present:
                continue
            out.append(
                MetricRow(
                    experiment=source.experiment,
                    model=source.model,
                    paradigm=source.paradigm,
                    seed=seed,
                    stage=source.stage,
                    metrics=dict(source.metrics),
                    path=source.path,
                )
            )
    return out


def aggregate_seeds(
    rows: Iterable[MetricRow] | pd.DataFrame,
    *,
    stochastic: Mapping[str, bool] | None = None,
    eval_seeds: Sequence[int] | None = None,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = 0,
) -> pd.DataFrame:
    """Mean, SD, n and both intervals per :data:`GROUP_KEYS`.

    ``rows`` is either the :class:`~strikecast.evaluation.leaderboard.MetricRow`
    list or the long frame
    :func:`strikecast.evaluation.leaderboard.metric_frame` produces. A
    deterministic model (per ``stochastic``) is reported as a point value with
    NaN spread and ``deterministic=True``, whatever its ``n``.
    """
    frame = rows if isinstance(rows, pd.DataFrame) else metric_frame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(CI_COLUMNS))

    expected = None if eval_seeds is None else len(list(eval_seeds))
    records: list[dict[str, Any]] = []
    for key, sub in frame.groupby(list(GROUP_KEYS), sort=True):
        by_seed = sub.drop_duplicates(subset=["seed"]).sort_values("seed")
        values = by_seed["value"].to_numpy(float)
        seeds_present = [int(s) for s in by_seed["seed"].tolist()]
        model = key[GROUP_KEYS.index("model")]
        is_det = bool(stochastic is not None and not stochastic.get(model, True))
        finite = values[~np.isnan(values)]
        record: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
        record["n"] = int(finite.size)
        record["mean"] = float(finite.mean()) if finite.size else float("nan")
        record["min"] = float(finite.min()) if finite.size else float("nan")
        record["max"] = float(finite.max()) if finite.size else float("nan")
        if is_det:
            record["std"] = float("nan")
            record["t_lo"] = record["t_hi"] = float("nan")
            record["boot_lo"] = record["boot_hi"] = float("nan")
        else:
            record["std"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            record["t_lo"], record["t_hi"] = t_interval(finite, alpha)
            record["boot_lo"], record["boot_hi"] = bootstrap_mean_interval(
                finite, n_boot=n_boot, alpha=alpha, seed=boot_seed
            )
        record["deterministic"] = is_det
        record["expected_seeds"] = expected
        record["seeds"] = ";".join(str(s) for s in seeds_present)
        records.append(record)

    out = pd.DataFrame.from_records(records, columns=list(CI_COLUMNS))
    return out.sort_values(list(GROUP_KEYS)).reset_index(drop=True)


def write_leaderboard_ci(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``leaderboard_ci.csv`` (plan sec. 7.1) with ``index=False``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def leaderboard_ci(
    root: str | Path,
    *,
    experiment: str | None = None,
    stages: Sequence[str] = ("cv", "test"),
    eval_seeds: Sequence[int] | None = None,
    stochastic: Mapping[str, bool] | None = None,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = 0,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Collect, broadcast, aggregate and (optionally) write ``leaderboard_ci.csv``.

    ``out_path`` defaults to ``<root>/<experiment>/report/leaderboard_ci.csv``
    when a single ``experiment`` was named; with ``experiment=None`` the frame
    spans experiments and is only written if a path is given.
    """
    rows = collect_metric_rows(root, experiment=experiment, stages=stages, seeds=eval_seeds)
    rows = broadcast_deterministic(rows, stochastic=stochastic, eval_seeds=eval_seeds)
    frame = aggregate_seeds(
        rows,
        stochastic=stochastic,
        eval_seeds=eval_seeds,
        alpha=alpha,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    if out_path is None and experiment is not None:
        out_path = Path(root) / experiment / "report" / "leaderboard_ci.csv"
    if out_path is not None:
        write_leaderboard_ci(frame, out_path)
    return frame
