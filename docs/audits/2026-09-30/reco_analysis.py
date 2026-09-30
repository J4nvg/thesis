"""Read-only analysis of seed-42 publication-run test predictions for the Recommendations section."""
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "/Users/jan/projects/bsc_thesis_code/thesis/runs_publication_20260929"
TIERS = json.load(open("/Users/jan/projects/bsc_thesis_code/thesis/data/fixed/regions_activity_cat.json"))
TIER = {r: t for t, rs in TIERS.items() for r in rs}
MODELS = {
    "Chronos FT": "chronos2/chronos2_fine_tuned/global",
    "Chronos ZS": "chronos2/chronos2_zero_shot/global",
    "CatBoost-Tw G": "count/catboost_tweedie/global",
    "LightGBM-Po G": "count/lightgbm_poisson/global",
    "GRU-Po w28 G": "count/gru_poisson_w28/global",
    "ARIMA": "diff/arima/global",
    "Seas. naive": "diff/naive_weekly/global",
    "Naive": "diff/naive_last/global",
    "Hurdle": "hurdle/hurdle/global",
}

def load(path, channel="y_pred"):
    fs = sorted(glob.glob(f"{ROOT}/{path}/seed=42/test/predictions/*.parquet"))
    d = pd.concat(pd.read_parquet(f) for f in fs)
    if channel != "y_pred" or "hurdle" in path:
        d = d[d.channel == ("hurdle" if channel == "y_pred" else channel)]
    return d.drop(columns="channel").set_index(["region", "fold", "horizon"]).sort_index()

P = {k: load(v) for k, v in MODELS.items()}
base = P["CatBoost-Tw G"]
for k, d in P.items():
    assert d.index.equals(base.index), k
    assert np.allclose(d.y_true.values, base.y_true.values), k
y = base.y_true.values
reg = base.index.get_level_values("region").values
hor = base.index.get_level_values("horizon").values
tier = np.array([TIER[r] for r in reg])
print("rows", len(y), "test dates", base.date.min().date(), base.date.max().date())

print("\n== A. RMSE / MAE per tier, share of pooled MSE from Sumy and from tier high ==")
rows = []
for k, d in P.items():
    e = d.y_pred.values - y
    r = {"model": k, "RMSE": np.sqrt((e**2).mean()), "MAE": np.abs(e).mean()}
    for t in ("low", "medium", "high"):
        m = tier == t
        r[f"RMSE_{t}"] = np.sqrt((e[m]**2).mean()); r[f"MAE_{t}"] = np.abs(e[m]).mean()
    r["MSE share sumy"] = (e[reg == "sumy"]**2).sum() / (e**2).sum()
    r["MSE share high"] = (e[tier == "high"]**2).sum() / (e**2).sum()
    rows.append(r)
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\n== B. Surge days (y >= 8): pred/true ratio and mean pred, per horizon ==")
surge = y >= 8
print("surge region-days x horizons:", surge.sum(), "unique (region,date):",
      base[surge].reset_index().drop_duplicates(["region", "date"]).shape[0],
      "regions:", sorted(set(reg[surge])))
out = {}
for k, d in P.items():
    p = d.y_pred.values
    out[k] = [p[surge & (hor == h)].mean() / y[surge & (hor == h)].mean() for h in range(1, 8)]
print(pd.DataFrame(out, index=[f"h{h}" for h in range(1, 8)]).T.round(3).to_string())
print("mean y_true on surge days:", y[surge].mean().round(2))

print("\n== C. Within-region surge ranking (high tier): surge = region's top 10% test days; mean ROC-AUC over 6 regions ==")
high = sorted(TIERS["high"])
res = {}
for k, d in P.items():
    vals = []
    for h in range(1, 8):
        aucs = []
        for r in high:
            m = (reg == r) & (hor == h)
            thr = np.quantile(y[m], 0.9)
            lab = y[m] > thr if (y[m] > thr).sum() >= 5 else y[m] >= thr
            aucs.append(roc_auc_score(lab, d.y_pred.values[m]))
        vals.append(np.mean(aucs))
    res[k] = vals
print(pd.DataFrame(res, index=[f"h{h}" for h in range(1, 8)]).T.round(3).to_string())

print("\n== D. Within-region occurrence (y>0) ranking, medium + high tiers, mean PR-AUC minus prevalence, h1 and h7 ==")
prob = load("hurdle/hurdle/global", "prob")
P2 = dict(P); P2["Hurdle prob"] = prob
res = {}
for k, d in P2.items():
    row = {}
    for t in ("medium", "high"):
        for h in (1, 4, 7):
            lifts = []
            for r in TIERS[t]:
                m = (reg == r) & (hor == h)
                lab = y[m] > 0
                if 0 < lab.mean() < 1:
                    lifts.append(average_precision_score(lab, d.y_pred.values[m]) - lab.mean())
            row[f"{t} h{h}"] = np.mean(lifts)
    res[k] = row
print(pd.DataFrame(res).T.round(3).to_string())

print("\n== E. Horizon: pooled RMSE per horizon ==")
res = {}
for k, d in P.items():
    e = d.y_pred.values - y
    res[k] = [np.sqrt((e[hor == h]**2).mean()) for h in range(1, 8)]
print(pd.DataFrame(res, index=[f"h{h}" for h in range(1, 8)]).T.round(3).to_string())

print("\n== F. Bias (mean error) on high tier and overall ==")
for k, d in P.items():
    e = d.y_pred.values - y
    print(f"{k:14s} bias all {e.mean():+.3f}  high {e[tier=='high'].mean():+.3f}  "
          f"days y=0 in high {e[(tier=='high') & (y==0)].mean():+.3f}  surge {e[surge].mean():+.3f}")

print("\n== G. Weekday profile of y in the test set (mean over all regions) ==")
dow = base.date.dt.dayofweek.values
print(pd.Series(y[hor == 1]).groupby(dow[hor == 1]).mean().round(3).to_dict())
