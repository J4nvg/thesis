"""Where vs when: cross-region ranking per day vs a static pre-test map; surge-day RMSE per horizon."""
import glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = "/Users/jan/projects/bsc_thesis_code/thesis/runs_publication_20260929"
MODELS = {"Chronos FT": "chronos2/chronos2_fine_tuned/global", "CatBoost-Tw G": "count/catboost_tweedie/global",
          "LightGBM-Po G": "count/lightgbm_poisson/global", "ARIMA": "diff/arima/global",
          "Seas. naive": "diff/naive_weekly/global"}
def load(p):
    d = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/{p}/seed=42/test/predictions/*.parquet")))
    return d[d.channel == "y_pred"].set_index(["region", "fold", "horizon"]).sort_index()
P = {k: load(v) for k, v in MODELS.items()}
base = P["Chronos FT"]
m = pd.read_parquet("/Users/jan/projects/bsc_thesis_code/thesis/data/dataset/master_combined_timeseries.parquet")
date = pd.to_datetime(m["event_date"])
regions = sorted(base.index.get_level_values("region").unique())
pre = m.loc[date < "2024-08-05", [f"act_drone_strike_on_ua_{r}" for r in regions]].mean()
static = pd.Series(pre.values, index=regions)
P["Static pre-test mean"] = base.assign(y_pred=base.index.get_level_values("region").map(static).values)

print("== Where: per (origin, horizon) across 20 regions; mean Spearman and top-3 overlap with actual top-3 ==")
rows = {}
for k, d in P.items():
    dd = d.reset_index()
    out = {}
    for h in (1, 4, 7):
        s = dd[dd.horizon == h]
        rho, top3 = [], []
        for _, g in s.groupby("fold"):
            if g.y_true.nunique() > 1:
                rho.append(spearmanr(g.y_pred, g.y_true).statistic)
            a = set(g.nlargest(3, "y_true").region); b = set(g.nlargest(3, "y_pred").region)
            top3.append(len(a & b) / 3)
        out[f"rho h{h}"] = np.nanmean(rho); out[f"top3 h{h}"] = np.mean(top3)
    rows[k] = out
print(pd.DataFrame(rows).T.round(3).to_string())

print("\n== Surge-day (y>=8) RMSE per horizon, and RMSE on non-surge days ==")
y = base.y_true.values; hor = base.index.get_level_values("horizon").values
res = {}
for k, d in P.items():
    e = d.y_pred.values - y
    res[k] = [np.sqrt((e[(y >= 8) & (hor == h)] ** 2).mean()) for h in range(1, 8)] + \
             [np.sqrt((e[(y < 8) & (hor == h)] ** 2).mean()) for h in (1, 4, 7)]
cols = [f"surge h{h}" for h in range(1, 8)] + ["rest h1", "rest h4", "rest h7"]
print(pd.DataFrame(res, index=cols).T.round(3).to_string())
