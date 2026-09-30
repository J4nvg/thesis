# Horizon boxplots: why "degrades after Day 4" became "flat" (2026-09-30)

Read-only analysis of `runs_publication_20260929` (seed 42, top-20 of the master leaderboard) and
`data/dataset`. Nothing trained. `main.tex` untouched.

| File | What |
|---|---|
| `horizon_boxplots.py` (+ `_output.txt`) | Figures below; prints the medians and Holm-significant pairs |
| `horizon_boxplots_AvsB.png/svg` | Thesis-style boxplots, 2x2: thesis scoring vs same target dates, RMSE and MAE |
| `horizon_boxplots_B.png/svg` | Same-dates column alone at thesis print width (4.96 in) |
| `horizon_why_flat.py` (+ `_output.txt`) | Evidence W1-W5 quoted below |

Brackets are the pre-specified 11 Wilcoxon pairs (each day vs Day 1, each day vs the previous day),
Holm-adjusted, as in `horizon_pipeline.py`. Grey lines are the individual configurations.

## 1. The thesis result replicates, and it comes from the scoring calendar

The left column of `horizon_boxplots_AvsB` uses the thesis scoring and reproduces the thesis figure on the new
run: Day 3 lowest, a rise at Days 5-6, Day 6 > Day 1 (+1.5 %, p = 0.033), Day 5 > Day 4, Day 6 > Day 5. The paired
test is not what removes the effect; it is stricter than the thesis ANOVA and still finds it.

Scoring each horizon on its own shifted dates means Day h is graded on 6 dates the other horizons do not all
see (W5). The end of the test window (16-21 Jan 2025) is busy: 30.2 strikes/day against 21.1 on the common
dates, and the first days (5-10 Aug 2024) are quiet, 13.2/day. Day 1 gets only the quiet August days; each later
day swaps one quiet August day for one busy January day. The extra dates' share of the squared error:

| Day | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| share of SE on the 6 extra dates | 2.1 % | 3.9 % | 3.8 % | 3.8 % | **6.4 %** | 6.5 % | 6.5 % |

The jump at Day 5 is 19 January (35 strikes), the first horizon that is graded on it. That is the thesis's
"degrades beyond four days". On the same 158 dates (right column) the rise is gone: the one systematic effect
left is the tree-model Day-1 penalty (Day 2 and 3 better than Day 1), plus small alternating steps of < 1 %
(Day 4 and 6 slightly up) with no trend.

## 2. No look-ahead: Day 5 does not see Days 1-4

- Every fold truncates the target at the cutoff (`backtest/engine.py:153`, `drop_after(cutoff)`) and predicts all
  7 days in one call. GBDTs are darts direct multi-output (`multi_models=True`, `output_chunk_length=7`, no
  recursion); RNNs are `BlockRNNModel` with a 7-day output chunk; nothing feeds back observed or predicted values.
- Past covariates are lagged relative to the first forecast day ([-1, -7, -14]) for every horizon.
- The only inputs that reach past the cutoff are future covariates, which are holiday/calendar columns plus
  weather (`data/covariates.py:82`); none is built from strike counts. Weather being the realised weather is the
  optimism caveat the thesis already states, not target leakage.
- W1: `naive_last` equals the last day before the first forecast day at every horizon; `naive_weekly` equals the
  value 7 days before the target.

Leakage would also not produce this pattern: it would make every horizon look like Day 1, whereas the tree models
are *worse* on Day 1 (traced to the origin-anchored weather window, `day1_trace.py`, rerun in `FUTWIN_RERUN.md`).

## 3. Why a flat profile is plausible for this target

Lead time hurts when yesterday tells you more than last week. Here it does not (W2, test window, pooled over
regions):

| lag (days) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 14 |
|---|---|---|---|---|---|---|---|---|
| raw autocorrelation | 0.49 | 0.48 | 0.48 | 0.46 | 0.48 | 0.50 | **0.64** | 0.62 |
| deviation from trailing 28-day level | 0.01 | -0.02 | 0.00 | -0.03 | 0.00 | 0.03 | **0.31** | 0.27 |

What is predictable is (a) the region's current level, which moves over weeks, and (b) a weekly rhythm
(Saturdays 12.5 strikes/day vs 26 on Wednesdays and Fridays in the test window, W4). Both are equally known 1 or 7
days ahead. What is not predictable, the night of the next salvo, is equally unknown at every lead time. Rule
forecasts without any model show the same (W3, same 158 dates):

| rule (uses data up to t-h) | RMSE Day 1 | RMSE Day 7 | change |
|---|---|---|---|
| trailing 28-day mean | 1.885 | 1.901 | +0.9 % |
| trailing 7-day mean | 1.926 | 1.936 | +0.6 % |
| persistence y(t-h) | 2.615 | 2.214 | -15 % (Day 7 = same weekday) |

Side result worth a sentence in the paper: the trailing 28-day mean has only 0.6-2.8 % higher RMSE than the
top-20 median, depending on the day (1.885-1.901 vs 1.849-1.873), and 5-7 % higher MAE (0.83 vs 0.78). Same
message as ARIMA within 0.02 of the leaders.

## 4. If design B is adopted: `main.tex` lines that state the old result

84 (abstract), 134, 1079-1097 (Sub-Q2 results), 1143 (Sub-Q2 discussion), 1163 (conclusion). Proposed
direction for Sub-Q2, not final wording: *scored on the same target dates, error does not grow from Day 1 to
Day 7; the degradation under per-horizon scoring comes from later horizons being graded on the busiest days of
the test window; consistent with a target whose day-to-day deviations carry no memory beyond a weekly echo;
tree models are 1-3 % worse on Day 1 (weather window, sensitivity rerun pending).* The realised-weather caveat
stays.
