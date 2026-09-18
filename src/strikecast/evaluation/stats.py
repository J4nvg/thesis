"""Friedman / Nemenyi / critical-difference diagrams, ported from the notebook.

Source: ``golden/results/analyse_results.ipynb`` cells 27-30 (the stub
``golden/results/analyse_results.py`` holds only the ``q`` one-liner of cell
29). The arithmetic is unchanged -- this is the plan sec. 7.2 item 4, "the
Friedman and Nemenyi machinery already in ``analyse_results.ipynb`` moves to
``evaluation/stats.py`` unchanged", which keeps continuity with the thesis.

What moved, and what changed around it:

:func:`build_block_matrices`
    Cell 27. The notebook took a spec frame and derived a parquet path per
    model; here it takes ``{label: long prediction frame}`` so the caller
    decides where the frames come from (the run store, plan sec. 5.3). The
    arithmetic is verbatim: ``e = y_pred - y_true``, ``_se = e**2``,
    ``_ae = |e|``, ``groupby(["region", "horizon"])``, ``sqrt(mean(_se))`` and
    ``mean(_ae)``.
:func:`normality_report`
    Cell 28, verbatim, with ``print`` replaced by a returned dataclass so the
    result is usable in a report. The two-way residual is still
    ``x_ij - rowmean_i - colmean_j + grandmean``.
:func:`friedman_nemenyi`
    Cell 29, verbatim, including the Nemenyi critical distance
    ``q = studentized_range.ppf(1 - alpha, k, inf) / sqrt(2)`` and
    ``CD = q * sqrt(k (k + 1) / (6 N))``.
:func:`average_rank_table`
    The ``sig_top5_table`` construction of cell 29.
:func:`critical_difference_diagram` / :func:`save_critical_difference_diagram`
    Cell 30, including the y=0.5 -> y=1 crossbar nudge and the "CD =" label
    reposition.

Notes
-----

``Q1`` **Lower is better.** Every matrix is an *error* matrix, so
    ``rankdata`` per block gives rank 1 to the best model. Feeding a skill
    score in would invert the ranking silently.

``Q2`` **``friedmanchisquare`` needs at least three columns** and raises
    otherwise; that is scipy's rule and it is not caught here.

``Q3`` **The notebook asserted a complete matrix.** ``build_block_matrices``
    keeps the assertion as a ``require_complete`` flag, default True: a NaN
    cell means a model was not evaluated on that ``(region, horizon)`` block
    and Friedman on an incomplete design is meaningless.

``Q4`` **``rcParams`` are set inside the plotting call, not at import.** Cell
    30 mutated ``matplotlib.rcParams`` globally on execution; doing that on
    ``import strikecast.evaluation.stats`` would leak serif fonts into every
    other figure in the process. The same values are applied through
    ``matplotlib.rc_context`` around the draw, so the produced figure is
    identical. This is the only deliberate difference from the notebook and it
    is cosmetic, not arithmetic.

``Q5`` **matplotlib and scikit-posthocs are imported lazily.** Only the
    plotting helpers need matplotlib; ``scikit_posthocs`` is imported at module
    level because :func:`friedman_nemenyi` needs it and it is a declared
    dependency in ``pyproject.toml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, kurtosis, rankdata, shapiro, skew, studentized_range

__all__ = [
    "ALPHA",
    "CD_RCPARAMS",
    "FriedmanResult",
    "NormalityReport",
    "average_rank_table",
    "build_block_matrices",
    "critical_difference_diagram",
    "friedman_nemenyi",
    "nemenyi_critical_distance",
    "normality_report",
    "save_critical_difference_diagram",
    "tied_with_best",
]

#: Cell 29: ``ALPHA = 0.05``.
ALPHA = 0.05

#: Cell 30: ``mpl.rcParams.update({...})``, applied per figure instead (Q4).
CD_RCPARAMS: Mapping[str, Any] = {"font.family": "serif", "font.size": 11}


# --------------------------------------------------------------------------- #
# cell 27: block x model error matrices
# --------------------------------------------------------------------------- #
def build_block_matrices(
    frames: Mapping[str, pd.DataFrame],
    *,
    require_complete: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(RMSE, MAE)`` frames: rows = ``(region, horizon)`` blocks, cols = labels.

    Verbatim cell 27 arithmetic (Q1, Q3)::

        e = d["y_pred"].values - d["y_true"].values
        d = d.assign(_se=e ** 2, _ae=np.abs(e))
        g = d.groupby(["region", "horizon"])
        rmse[label] = np.sqrt(g["_se"].mean())
        mae[label] = g["_ae"].mean()
    """
    rmse: dict[str, pd.Series] = {}
    mae: dict[str, pd.Series] = {}
    for label, d in frames.items():
        e = d["y_pred"].values - d["y_true"].values
        d = d.assign(_se=e**2, _ae=np.abs(e))
        g = d.groupby(["region", "horizon"])
        rmse[label] = np.sqrt(g["_se"].mean())
        mae[label] = g["_ae"].mean()
    m_rmse, m_mae = pd.DataFrame(rmse), pd.DataFrame(mae)
    if require_complete and (m_rmse.isna().any().any() or m_mae.isna().any().any()):
        raise ValueError("incomplete block matrix")
    return m_rmse, m_mae


# --------------------------------------------------------------------------- #
# cell 28: normality diagnostic
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NormalityReport:
    """What cell 28 printed, as data."""

    name: str
    n: int
    shapiro_w: float
    shapiro_p: float
    skew: float
    kurtosis: float
    columns_rejecting: int
    n_columns: int
    alpha: float

    @property
    def normal(self) -> bool:
        """``p >= alpha`` -- cell 28's return value."""
        return bool(self.shapiro_p >= self.alpha)

    def __str__(self) -> str:
        verdict = "normal" if self.normal else "NON-normal"
        return (
            f"[{self.name}] ANOVA residuals (n={self.n}): Shapiro W={self.shapiro_w:.3f}, "
            f"p={self.shapiro_p:.2e}, skew={self.skew:.2f}, kurtosis={self.kurtosis:.2f} "
            f"-> {verdict}\n"
            f"         per-model columns: {self.columns_rejecting}/{self.n_columns} "
            f"reject normality (alpha={self.alpha})"
        )


def normality_report(M: pd.DataFrame, name: str, alpha: float = ALPHA) -> NormalityReport:
    """Shapiro-Wilk on the two-way (blocked) ANOVA residuals of ``M``.

    Verbatim cell 28: ``resid = X - X.mean(0) - X.mean(1) + X.mean()``. The
    point is to justify the rank-based Friedman test over a parametric blocked
    ANOVA, not to test the errors themselves.
    """
    X = M.values
    resid = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()
    r = resid.ravel()
    W, p = shapiro(r)
    rej = sum(shapiro(M[c].values)[1] < alpha for c in M.columns)
    return NormalityReport(
        name=name,
        n=int(r.size),
        shapiro_w=float(W),
        shapiro_p=float(p),
        skew=float(skew(r)),
        kurtosis=float(kurtosis(r)),
        columns_rejecting=int(rej),
        n_columns=int(M.shape[1]),
        alpha=float(alpha),
    )


# --------------------------------------------------------------------------- #
# cell 29: Friedman + Nemenyi
# --------------------------------------------------------------------------- #
def nemenyi_critical_distance(k: int, N: int, alpha: float = ALPHA) -> float:
    """Demsar (2006) critical distance, verbatim cell 29.

    ``q = studentized_range.ppf(1 - alpha, k, inf) / sqrt(2)`` and
    ``CD = q * sqrt(k (k + 1) / (6 N))``.
    """
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * N)))


@dataclass(frozen=True)
class FriedmanResult:
    """Cell 29's result dict, as a dataclass with the same field names."""

    chi: float
    p: float
    avg_rank: pd.Series
    CD: float
    nemenyi: pd.DataFrame
    k: int
    N: int
    ranks: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def __getitem__(self, key: str) -> Any:
        """Mapping access, so notebook code (``r["avg_rank"]``) keeps working."""
        return getattr(self, key)

    @property
    def best(self) -> str:
        """The lowest average rank -- cell 29's ``r["avg_rank"].index[0]``."""
        return str(self.avg_rank.index[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "chi": self.chi,
            "p": self.p,
            "avg_rank": self.avg_rank,
            "CD": self.CD,
            "nemenyi": self.nemenyi,
            "k": self.k,
            "N": self.N,
        }


def friedman_nemenyi(M: pd.DataFrame, alpha: float = ALPHA) -> FriedmanResult:
    """Friedman omnibus + Nemenyi post-hoc on a ``(blocks x models)`` error matrix.

    Verbatim cell 29 (Q1, Q2)::

        chi, p = friedmanchisquare(*[M[c].values for c in M.columns])
        ranks = M.apply(lambda row: rankdata(row.values), axis=1, result_type="expand")
        avg = ranks.mean().sort_values()
        q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
        CD = q * np.sqrt(k * (k + 1) / (6.0 * N))
        nem = sp.posthoc_nemenyi_friedman(M.values)
    """
    chi, p = friedmanchisquare(*[M[c].values for c in M.columns])
    ranks = M.apply(lambda row: rankdata(row.values), axis=1, result_type="expand")
    ranks.columns = M.columns
    avg = ranks.mean().sort_values()
    k, N = M.shape[1], M.shape[0]
    CD = nemenyi_critical_distance(k, N, alpha)
    nem = sp.posthoc_nemenyi_friedman(M.values)
    nem.index = nem.columns = M.columns
    return FriedmanResult(
        chi=float(chi), p=float(p), avg_rank=avg, CD=CD, nemenyi=nem, k=int(k), N=int(N), ranks=ranks
    )


def tied_with_best(result: FriedmanResult) -> list[str]:
    """Models within one CD of the best average rank -- cell 29's ``tied``."""
    avg = result.avg_rank
    return avg[avg <= avg.iloc[0] + result.CD].index.tolist()


def average_rank_table(
    results: Mapping[str, FriedmanResult], *, sort_by: str | None = None, decimals: int = 2
) -> pd.DataFrame:
    """The ``sig_top5_table`` of cell 29: one ``"<metric> avg rank"`` column per metric.

    ``sort_by`` defaults to the first metric, which is cell 29's
    ``sort_values("RMSE avg rank")``.
    """
    table = pd.DataFrame({f"{m} avg rank": r.avg_rank for m, r in results.items()})
    if table.empty:
        return table
    key = f"{sort_by} avg rank" if sort_by is not None else table.columns[0]
    return table.sort_values(key).round(decimals)


# --------------------------------------------------------------------------- #
# cell 30: critical-difference diagrams
# --------------------------------------------------------------------------- #
def critical_difference_diagram(
    result: FriedmanResult,
    *,
    ax: Any = None,
    title: str | None = None,
    grey: str = "0.15",
    cd_y: float = 1.0,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Draw Demsar's critical-difference diagram; returns the matplotlib ``Axes``.

    Verbatim cell 30 apart from Q4 (``rcParams`` scoped to the draw) and the
    ``savefig``, which moved to :func:`save_critical_difference_diagram`.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    order = result.avg_rank.index
    sig_mat = result.nemenyi.loc[order, order]
    with mpl.rc_context(dict(CD_RCPARAMS)):
        if ax is None:
            size = figsize if figsize is not None else (12, 0.30 * result.k + 2.6)
            _fig, ax = plt.subplots(figsize=size)
        sp.critical_difference_diagram(
            ranks=result.avg_rank,
            sig_matrix=sig_mat,
            ax=ax,
            cd=result.CD,
            label_fmt_left="{label} ({rank:.2f})  ",
            label_fmt_right="  ({rank:.2f}) {label}",
            color_palette={m: grey for m in order},
            crossbar_props={"color": "black", "linewidth": 2},
        )
        ax.grid(False)
        ax.xaxis.grid(False)
        for line in ax.lines:
            yd = np.atleast_1d(line.get_ydata())
            if len(yd) == 2 and np.allclose(yd, 0.5):  # the CD reference bar
                line.set_ydata([cd_y, cd_y])
        for text in ax.texts:
            if text.get_text().startswith("CD ="):
                x, _ = text.get_position()
                text.set_position((x, cd_y + 0.18))
        ax.set_ylim(ax.get_ylim()[0], cd_y + 0.45)
        if title is not None:
            ax.set_title(title, fontsize=12, pad=26)
        ax.annotate(
            "← better (lower average rank)",
            xy=(-0.2, 0.0),
            xycoords="axes fraction",
            ha="left",
            fontsize=9,
            style="italic",
        )
    return ax


def save_critical_difference_diagram(
    result: FriedmanResult,
    path: str | Path,
    *,
    title: str | None = None,
    **kwargs: Any,
) -> Path:
    """Draw and write ``report/cd_<metric>.svg`` (plan sec. 7.2), then close the figure."""
    import matplotlib.pyplot as plt

    ax = critical_difference_diagram(result, title=title, **kwargs)
    fig = ax.get_figure()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target
