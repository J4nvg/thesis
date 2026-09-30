"""Why the horizon profile is flat on common target dates, and why it rose in the thesis (2026-09-30).

Read-only: saved seed-42 test predictions of runs_publication_20260929 plus the raw target series in
data/dataset. Nothing is trained or written.

    uv run python docs/audits/2026-09-30/horizon_why_flat.py | tee docs/audits/2026-09-30/horizon_why_flat_output.txt

[W1] no look-ahead: naive_last forecasts every horizon with the last day before the first forecast
     day; naive_weekly with the value 7 days before the target.
[W2] memory of the target: autocorrelation at lags 1..14, raw, region-demeaned, and of the deviation
     from a trailing 28-day level.
[W3] leak-free rule forecasts on the common target dates (no model, no covariates).
[W4] weekday pattern behind the lag-7 peak.
[W5] thesis scoring: the extra target dates each horizon is graded on and their share of its
     squared error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import horizon_pipeline as hp  # noqa: E402

pd.set_option("display.width", 220)
DATA = hp.ROOT / "data/dataset/master_combined_timeseries.parquet"


def acf(Z: pd.DataFrame, lags) -> pd.Series:
    """Correlation of y(t) with y(t-k), pooled over regions."""
    out = {}
    for k in lags:
        a, b = Z.iloc[k:].values.ravel(), Z.shift(k).iloc[k:].values.ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        out[k] = np.corrcoef(a[ok], b[ok])[0, 1]
    return pd.Series(out)


def main() -> None:
    lb = pd.read_csv(hp.RUNS / "_figures/master_leaderboard.csv")
    preds = hp.load_preds(lb)
    top20 = {hp.label(r): preds[hp.label(r)] for _, r in lb.head(20).iterrows()}
    p0 = next(iter(top20.values()))
    regions = sorted(p0.region.unique())
    m = pd.read_parquet(DATA)
    Y = m.set_index(pd.to_datetime(m.event_date))[[f"act_drone_strike_on_ua_{r}" for r in regions]].astype(float)
    Y.columns = regions
    test = p0.drop_duplicates(["region", "date"]).pivot(index="date", columns="region", values="y_true")
    common = sorted(set.intersection(*[set(p0.loc[p0.horizon == h, "date"]) for h in hp.H]))
    S = Y.stack()

    hp.section("W1", "No look-ahead in the baselines (their forecasts are fixed functions of observed data)")
    print(f"raw series == y_true of the predictions on all {len(test)} test dates: "
          f"{np.allclose(Y.loc[test.index, regions].values, test[regions].values)}")
    for _, r in lb[lb.model.isin(["naive_last", "naive_weekly"])].iterrows():
        p = preds[hp.label(r)]
        for name, ref in (("y(first forecast day - 1)", p.origin_date - pd.Timedelta(days=1)),
                          ("y(target - 7)", p.date - pd.Timedelta(days=7))):
            v = S.reindex(pd.MultiIndex.from_arrays([ref, p.region])).values
            ok = np.isfinite(v)
            print(f"{r.model:12s} forecast == {name:26s}: {np.allclose(p.y_pred.values[ok], v[ok])}")

    hp.section("W2", "Memory of the target: autocorrelation at lag k, pooled over the 20 regions")
    for name, Z in {"full history": Y, "last 365 days": Y.iloc[-365:], "test window": Y.loc[test.index]}.items():
        level = Z.rolling(28, min_periods=14).mean().shift(1)
        t = pd.DataFrame({"raw": acf(Z, range(1, 15)), "region_demeaned": acf(Z - Z.mean(), range(1, 15)),
                          "minus_trailing28_level": acf(Z - level, range(1, 15))}).T
        print(f"\n{name} ({Z.index.min().date()} .. {Z.index.max().date()}, {len(Z)} days)")
        print(t.round(3).to_string())

    hp.section("W3", f"Leak-free rule forecasts on the {len(common)} common target dates: horizon h uses data up to t-h")
    rules = {"persistence y(t-h)": lambda h: Y.shift(h),
             "trailing 7-day mean": lambda h: Y.rolling(7).mean().shift(h),
             "trailing 28-day mean": lambda h: Y.rolling(28).mean().shift(h)}
    rows = []
    for name, f in rules.items():
        for metric in hp.METRICS:
            vals = []
            for h in hp.H:
                e = (Y - f(h)).loc[common].values
                vals.append(np.sqrt((e ** 2).mean()) if metric == "RMSE" else np.abs(e).mean())
            rows.append(dict(rule=name, metric=metric, **{f"h{h}": round(v, 3) for h, v in zip(hp.H, vals, strict=True)},
                             h7_vs_h1_pct=round(100 * (vals[-1] / vals[0] - 1), 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    B = hp.matrices_from_preds(top20, dates=set(common))
    for metric in hp.METRICS:
        print(f"top-20 models, median {metric}: " + " ".join(f"{v:.3f}" for v in B[metric].median()))

    hp.section("W4", "Weekday pattern (strikes/day summed over regions)")
    tot = Y.sum(axis=1)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for name, Z in {"full history": tot, "test window": tot.loc[test.index]}.items():
        g = Z.groupby(Z.index.day_name()).mean().reindex(days)
        print(f"{name:13s} " + "  ".join(f"{d[:3]}={v:.1f}" for d, v in g.items()))

    hp.section("W5", "Thesis scoring: target dates outside the common set, per horizon (strikes that day)")
    cs = set(common)
    rows = []
    for h in hp.H:
        share = []
        for p in top20.values():
            q = p[p.horizon == h]
            se = (q.y_true - q.y_pred.clip(lower=0)) ** 2
            share.append(se[~q.date.isin(cs)].sum() / se.sum())
        extra = sorted(set(p0.loc[p0.horizon == h, "date"]) - cs)
        rows.append(dict(h=h, extra_dates=" ".join(f"{d:%d%b}({int(tot[d])})" for d in extra),
                         strikes_on_extra=int(tot[extra].sum()),
                         share_of_squared_error_pct=round(100 * float(np.mean(share)), 1)))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"strikes/day: common dates {tot[common].mean():.1f}; the 6 early-only (Aug) "
          f"{tot[[d for d in test.index if d < common[0]]].mean():.1f}; the 6 late-only (Jan) "
          f"{tot[[d for d in test.index if d > common[-1]]].mean():.1f}")


if __name__ == "__main__":
    main()
