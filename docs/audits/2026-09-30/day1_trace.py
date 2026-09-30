"""Evidence for the gradient-boosted trees' Day-1 penalty (2026-09-30). Read-only, nothing trained.

    uv run python docs/audits/2026-09-30/day1_trace.py | tee docs/audits/2026-09-30/day1_trace_output.txt

Hypothesis: darts regression models with ``multi_models=True`` give every horizon's sub-model the
SAME future-covariate window, anchored at the forecast origin (``lags_future_covariates=(2, 7)``
-> origin-2 .. origin+6; darts ``sklearn_model.py`` applies no per-step shift when
``multi_models=True``). Relative to its own target day, the Day-h sub-model therefore sees weather
from h+1 days before to 7-h days after the target: Day 1 gets 2 days of pre-target weather and 6
days of post-target weather, Day 7 gets 8 days before and none after. Weather is supplied only as
a future covariate, so Day 1 has the shortest weather history. BlockRNN attaches each output
step's own-day future covariates (``block_rnn_model.py``), Chronos-2 handles known covariates per
time step and ARIMA uses none, so only the tree-based models (count GBDT, hurdle) have this
asymmetry.

Sections: [T1] error decomposition, [T2] which day each horizon's forecast aligns with,
[T3] per-horizon importance by weather-day offset, [T4] ARIMA / naive profiles (no weather).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import horizon_pipeline as hp  # noqa: E402

pd.set_option("display.width", 200)


def main() -> None:
    lb = pd.read_csv(hp.RUNS / "_figures/master_leaderboard.csv")
    top20 = lb.head(20)
    preds = hp.load_preds(top20)
    p0 = next(iter(preds.values()))
    dates = sorted(p0.date.unique())
    common = set.intersection(*[set(p0.loc[p0.horizon == h, "date"]) for h in hp.H])
    truth = p0.drop_duplicates(["region", "date"]).set_index(["region", "date"]).y_true
    fam = {hp.label(r): ("GBDT" if r.Modelname == "gbdt" else "other") for _, r in top20.iterrows()}

    def lookup(g: pd.DataFrame, days: pd.Series) -> np.ndarray:
        return truth.reindex(pd.MultiIndex.from_arrays([g.region, days])).values

    hp.section("T1", "Error decomposition per horizon, same 158 target dates (median over top-20 configs)")
    print("MSE = bias^2 + (sd_f - rho*sd_y)^2 [conditional bias] + (1-rho^2) var_y [discrimination]")
    rows = []
    for k, p in preds.items():
        q = p[p.date.isin(common)]
        for h, g in q.groupby("horizon"):
            y, f = g.y_true.values, g.y_pred.clip(lower=0).values
            rho = np.corrcoef(f, y)[0, 1]
            rows.append(dict(fam=fam[k], h=h, mse=np.mean((y - f) ** 2), bias2=(y.mean() - f.mean()) ** 2,
                             cond=(f.std() - rho * y.std()) ** 2, discrimination=(1 - rho ** 2) * y.var(),
                             rho=rho, sd_forecast=f.std()))
    print(pd.DataFrame(rows).groupby(["fam", "h"]).median().round(4).to_string())

    hp.section("T2", "corr(forecast for target day d, actual count on day d+delta), median over configs")
    inner = set(dates[9:-9])
    rows = []
    for k, p in preds.items():
        q = p[p.date.isin(inner)]
        for h, g in q.groupby("horizon"):
            f = g.y_pred.clip(lower=0).values
            r = {"fam": fam[k], "h": h}
            for dlt in (-2, -1, 0, 1, 2):
                r[f"d{dlt:+d}"] = np.corrcoef(f, lookup(g, g.date + pd.Timedelta(days=dlt)))[0, 1]
            rows.append(r)
    print(pd.DataFrame(rows).groupby(["fam", "h"]).median().round(4).to_string())
    print("Day d-1 is observed at the origin only for h = 1.")

    hp.section("T3", "Permutation / gain importance share by weather-day offset relative to the target")
    paths = sorted(glob.glob(str(hp.RUNS / "count/*/*/seed=42/importance/importance.csv")))
    print("configurations with saved per-horizon importance:", [Path(p).parts[-5] + "@" + Path(p).parts[-4] for p in paths])
    rows = []
    for path in paths:
        d = pd.read_csv(path)
        fut = d.Feature.str.extract(r"^(.*)_futcov_lag(-?\d+)$")
        lag = pd.to_numeric(fut[1], errors="coerce")
        weather = fut[0].fillna("").str.startswith("env_weather") | (fut[0] == "env_k_max")
        for h in hp.H:
            off = lag - (h - 1)
            for kind in ("perm", "gain"):
                v = d[f"h{h}_{kind}"].clip(lower=0)
                s = lambda m: v[m].sum() / v.sum()  # noqa: E731
                rows.append(dict(kind=kind, h=h, weather_3plus_days_before=s(weather & (off <= -3)),
                                 weather_d_minus2_to_d=s(weather & off.between(-2, 0)),
                                 weather_after_target=s(weather & (off > 0)),
                                 target_lags=s(d.Feature.str.contains("_target_lag")),
                                 past_covariates=s(d.Feature.str.contains("_pastcov_lag"))))
    r = pd.DataFrame(rows)
    for kind in ("perm", "gain"):
        print(f"\n{kind}:")
        print(r[r.kind == kind].drop(columns="kind").groupby("h").median().round(3).to_string())

    hp.section("T4", "Univariate baselines (no weather), % change vs Day 1, same 158 target dates")
    base = lb[lb.model.isin(["arima", "naive_weekly", "naive_last"])]
    Mb = hp.matrices_from_preds(hp.load_preds(base), dates=common)
    for m in hp.METRICS:
        print(m)
        print((100 * Mb[m].div(Mb[m][1], axis=0) - 100).round(2).to_string())


if __name__ == "__main__":
    main()
