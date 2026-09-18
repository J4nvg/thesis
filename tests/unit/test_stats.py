"""Unit tests for `strikecast.evaluation.stats`.

The port has to be arithmetically identical to `analyse_results.ipynb` cells
27-30, so most of these tests re-derive the notebook's expression inline and
compare, rather than pinning a number.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, studentized_range  # noqa: E402

from strikecast.evaluation.stats import (  # noqa: E402
    ALPHA,
    FriedmanResult,
    average_rank_table,
    build_block_matrices,
    critical_difference_diagram,
    friedman_nemenyi,
    nemenyi_critical_distance,
    normality_report,
    save_critical_difference_diagram,
    tied_with_best,
)

REGIONS = tuple(f"r{i}" for i in range(10))
HORIZONS = (1, 2, 3, 4, 5, 6, 7)


def make_frame(scale: float, *, seed: int = 0) -> pd.DataFrame:
    """A long prediction frame whose error size is controlled by `scale`."""
    rng = np.random.default_rng(seed)
    rows = []
    for region in REGIONS:
        for h in HORIZONS:
            for _day in range(8):
                y = float(rng.poisson(2.0))
                rows.append(
                    {
                        "region": region,
                        "horizon": h,
                        "y_true": y,
                        "y_pred": y + rng.normal(0.0, scale),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {
        "good": make_frame(0.3, seed=1),
        "middling": make_frame(0.8, seed=2),
        "poor": make_frame(1.6, seed=3),
        "worst": make_frame(2.4, seed=4),
    }
    return build_block_matrices(frames)


# --------------------------------------------------------------------------- #
# cell 27
# --------------------------------------------------------------------------- #
def test_block_matrices_are_blocks_by_models(matrices) -> None:
    rmse, mae = matrices
    assert rmse.shape == (len(REGIONS) * len(HORIZONS), 4)
    assert mae.shape == rmse.shape
    assert list(rmse.columns) == ["good", "middling", "poor", "worst"]
    assert rmse.index.names == ["region", "horizon"]


def test_block_matrix_arithmetic_is_the_notebook_expression() -> None:
    frame = make_frame(0.7, seed=9)
    rmse, mae = build_block_matrices({"m": frame})
    e = frame["y_pred"].values - frame["y_true"].values
    d = frame.assign(_se=e**2, _ae=np.abs(e))
    g = d.groupby(["region", "horizon"])
    np.testing.assert_allclose(rmse["m"].to_numpy(), np.sqrt(g["_se"].mean()).to_numpy())
    np.testing.assert_allclose(mae["m"].to_numpy(), g["_ae"].mean().to_numpy())


def test_a_bigger_error_scale_gives_a_bigger_block_error(matrices) -> None:
    rmse, _ = matrices
    means = rmse.mean()
    assert means["good"] < means["middling"] < means["poor"] < means["worst"]


def test_an_incomplete_block_matrix_is_refused() -> None:
    full = make_frame(0.5, seed=1)
    partial = full[full["region"] != "r0"]
    with pytest.raises(ValueError, match="incomplete block matrix"):
        build_block_matrices({"a": full, "b": partial})


def test_an_incomplete_matrix_is_allowed_when_asked_for() -> None:
    full = make_frame(0.5, seed=1)
    partial = full[full["region"] != "r0"]
    rmse, _ = build_block_matrices({"a": full, "b": partial}, require_complete=False)
    assert rmse["b"].isna().sum() == len(HORIZONS)


# --------------------------------------------------------------------------- #
# cell 28
# --------------------------------------------------------------------------- #
def test_normality_report_uses_the_two_way_residual(matrices) -> None:
    rmse, _ = matrices
    report = normality_report(rmse, "RMSE")
    X = rmse.values
    resid = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()
    assert report.n == resid.size
    assert report.name == "RMSE"
    assert report.n_columns == 4
    assert 0 <= report.columns_rejecting <= 4
    assert 0.0 <= report.shapiro_p <= 1.0


def test_normality_report_verdict_and_string(matrices) -> None:
    rmse, _ = matrices
    report = normality_report(rmse, "RMSE")
    assert report.normal == (report.shapiro_p >= ALPHA)
    assert "ANOVA residuals" in str(report)
    assert ("NON-normal" in str(report)) != report.normal


# --------------------------------------------------------------------------- #
# cell 29
# --------------------------------------------------------------------------- #
def test_critical_distance_matches_the_demsar_formula() -> None:
    for k, N in ((6, 140), (4, 70), (12, 140)):
        q = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
        assert nemenyi_critical_distance(k, N) == pytest.approx(q * np.sqrt(k * (k + 1) / (6.0 * N)))


def test_the_notebook_q_value_is_reproduced() -> None:
    """`golden/results/analyse_results.py` prints exactly this q for k=6."""
    q = studentized_range.ppf(1 - 0.05, 6, np.inf) / np.sqrt(2)
    assert nemenyi_critical_distance(6, 140) == pytest.approx(q * np.sqrt(6 * 7 / (6.0 * 140)))


def test_friedman_nemenyi_matches_the_notebook_cell(matrices) -> None:
    rmse, _ = matrices
    res = friedman_nemenyi(rmse)
    chi, p = friedmanchisquare(*[rmse[c].values for c in rmse.columns])
    ranks = rmse.apply(lambda row: rankdata(row.values), axis=1, result_type="expand")
    ranks.columns = rmse.columns
    avg = ranks.mean().sort_values()
    nem = sp.posthoc_nemenyi_friedman(rmse.values)
    nem.index = nem.columns = rmse.columns

    assert res.chi == pytest.approx(chi)
    assert res.p == pytest.approx(p)
    pd.testing.assert_series_equal(res.avg_rank, avg)
    pd.testing.assert_frame_equal(res.nemenyi, nem)
    assert res.k == 4 and res.N == len(REGIONS) * len(HORIZONS)
    assert res.CD == pytest.approx(nemenyi_critical_distance(4, res.N))


def test_lower_error_gets_the_lower_rank(matrices) -> None:
    rmse, _ = matrices
    res = friedman_nemenyi(rmse)
    assert res.avg_rank.index.tolist() == ["good", "middling", "poor", "worst"]
    assert res.best == "good"
    assert res.avg_rank.min() >= 1.0 and res.avg_rank.max() <= 4.0


def test_the_result_supports_the_notebooks_mapping_access(matrices) -> None:
    rmse, _ = matrices
    res = friedman_nemenyi(rmse)
    assert isinstance(res, FriedmanResult)
    assert res["CD"] == res.CD
    assert res["avg_rank"].index[0] == res.best
    assert set(res.as_dict()) == {"chi", "p", "avg_rank", "CD", "k", "N", "nemenyi"}


def test_three_identical_columns_tie_on_rank_and_give_a_degenerate_omnibus() -> None:
    """Every block is a perfect tie, so scipy's Friedman statistic is NaN.

    Preserved rather than guarded: a fully tied design has no rank variance and
    `friedmanchisquare` says so. Nemenyi still returns p=1 everywhere, so the
    "tied with best" window is the whole set.
    """
    frame = make_frame(0.6, seed=11)
    matrix, _ = build_block_matrices({"a": frame, "b": frame, "c": frame})
    res = friedman_nemenyi(matrix)
    assert np.isnan(res.chi) and np.isnan(res.p)
    assert np.allclose(res.avg_rank.to_numpy(), 2.0)
    assert np.allclose(res.nemenyi.to_numpy(), 1.0)
    assert set(tied_with_best(res)) == {"a", "b", "c"}


def test_barely_different_models_are_not_separated_by_nemenyi() -> None:
    res = friedman_nemenyi(
        build_block_matrices(
            {
                "a": make_frame(0.80, seed=21),
                "b": make_frame(0.81, seed=22),
                "c": make_frame(0.82, seed=23),
            }
        )[0]
    )
    assert res.p > 0.05
    assert set(tied_with_best(res)) == {"a", "b", "c"}
    assert (res.nemenyi.to_numpy() > 0.05).all()


def test_tied_with_best_is_the_one_cd_window(matrices) -> None:
    rmse, _ = matrices
    res = friedman_nemenyi(rmse)
    avg = res.avg_rank
    assert tied_with_best(res) == avg[avg <= avg.iloc[0] + res.CD].index.tolist()
    assert "good" in tied_with_best(res)


def test_a_clear_winner_is_flagged_significant_by_nemenyi(matrices) -> None:
    rmse, _ = matrices
    res = friedman_nemenyi(rmse)
    assert res.p < 0.05
    assert res.nemenyi.loc["good", "worst"] < 0.05


def test_average_rank_table_is_the_notebook_table(matrices) -> None:
    rmse, mae = matrices
    results = {"RMSE": friedman_nemenyi(rmse), "MAE": friedman_nemenyi(mae)}
    table = average_rank_table(results)
    assert list(table.columns) == ["RMSE avg rank", "MAE avg rank"]
    assert table["RMSE avg rank"].is_monotonic_increasing
    assert table.index.tolist()[0] == "good"
    assert (table.round(2) == table).all().all()


def test_average_rank_table_can_sort_on_another_metric(matrices) -> None:
    rmse, mae = matrices
    results = {"RMSE": friedman_nemenyi(rmse), "MAE": friedman_nemenyi(mae)}
    table = average_rank_table(results, sort_by="MAE")
    assert table["MAE avg rank"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# cell 30
# --------------------------------------------------------------------------- #
def test_the_diagram_draws_and_moves_the_cd_bar(matrices) -> None:
    import matplotlib.pyplot as plt

    rmse, _ = matrices
    res = friedman_nemenyi(rmse)
    ax = critical_difference_diagram(res, title="Critical-difference diagram - RMSE")
    assert ax.get_title().endswith("RMSE")
    # nothing is left at the library's default y=0.5 for the CD reference bar
    assert not any(
        len(np.atleast_1d(ln.get_ydata())) == 2 and np.allclose(np.atleast_1d(ln.get_ydata()), 0.5)
        for ln in ax.lines
    )
    assert ax.get_ylim()[1] == pytest.approx(1.45)
    plt.close(ax.get_figure())


def test_the_diagram_does_not_leak_rcparams(matrices) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    before = mpl.rcParams["font.family"]
    rmse, _ = matrices
    ax = critical_difference_diagram(friedman_nemenyi(rmse))
    plt.close(ax.get_figure())
    assert mpl.rcParams["font.family"] == before


def test_saving_the_diagram_writes_an_svg(matrices, tmp_path) -> None:
    rmse, _ = matrices
    path = save_critical_difference_diagram(
        friedman_nemenyi(rmse), tmp_path / "report" / "cd_rmse.svg", title="CD - RMSE"
    )
    assert path.exists()
    assert path.read_text(encoding="utf-8").lstrip().startswith(("<?xml", "<svg"))
