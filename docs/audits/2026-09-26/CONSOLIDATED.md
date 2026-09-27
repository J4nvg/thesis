# Consolidated audit and plan — 2026-09-26

Sources: `audit_A_data_features.md` (A1–A20), `audit_B_models_eval.md` (B1–B22),
`audit_C_outputs_cluster.md` (C1–C32), all in this folder. Every claim below cites the
finding ID; the evidence (file:line, numbers, scripts) is in those reports.

## 1. Verdict in one paragraph

The refactor is a faithful port of the legacy **computation** where it has been tested
(panels, windows, lags, splits, backtest schedule, naives, calibration: A2–A4, A7, A15, A16,
B10, B14). It is **not yet able to reproduce everything the thesis reports**, for three kinds
of reasons: (a) port/environment bugs that nothing tested (count XGBoost cannot be fitted at
all, B3; diff XGBoost silently changed strategy, B4; RNN re-tuning scores log-space
predictions, B7; hurdle leaderboard reports the raw instead of the calibrated hurdle, B8;
composite calibrators vanish for seeds ≠ 42, C18), (b) parts never ported (Chronos-2 path
C15, feature importance C14, cross-experiment SkillScore B16, horizon statistics B17, every
report figure/table from the run store C1/C2), and (c) the cluster path would run but give
wrong or incomplete results (global-only job matrix C16, no tune→cv→test dependencies C17,
no GPU request C19, no fold-level resume C20). Separately, the thesis text disagrees with the
code in ~20 places (list in §5), expdecay7 being the most consequential (A1).

## 2. Good news

- **The Colab tuning params are not lost** (B1). `_regression_GBDT.ipynb` cell 56 lost its
  source but kept its output: all six count GBDT best params at full float64 precision.
  Verified: the lightgbm_poisson entry equals the golden JSON digit for digit, and count
  catboost_tweedie (Table 4 #1) fitted with these params reproduces golden fold-0
  predictions to 1.8e-15.
- **Thread count does not change LightGBM or CatBoost results** (B5): fold-0 predictions
  are bit-identical at 4, 12, 15 and 64 threads. On the cluster it is a speed question only.
- **Dropping the log family loses nothing the thesis reports** (B18). The diff MSE variants
  must stay (the thesis quotes them, and SkillScore's naive comes from the diff run).
- Damage classifiers and Venn-Abers appear nowhere in the thesis (B19, B20): not needed for
  the thesis re-run.

## 3. Decisions needed from Jan (recommendation first)

| # | Question | Recommendation | Findings |
|---|---|---|---|
| D1 | Exact form of the fixed expdecay7 | Literal leaky integrator `s_t = x_t + 2^(-1/7) s_{t-1}` via darts `{"function":"sum","mode":"ewm","halflife":7}`, new name `leaky7`; equals the thesis equation exactly. Alternative: normalised EWM `mean/halflife=7/adjust=False`. Not the reviewer's `adjust=True` (distorts ~2 months per series). Scale is not free: LightGBM moves up to 0.29 between the two forms, so freeze one. | A1, §expdecay7 of A |
| D2 | Feature selection for the publication runs (must re-run: features change) | One cached, deterministic selection per family/head on **CPU** with `PYTHONHASHSEED=0`, run once as its own SLURM job; the thesis count selection ran on GPU and the hurdle re-selected in every session, which is what made the hurdle unreproducible (B9). | A12, A13, B9 |
| D3 | Weather "exempt" from selection (thesis 528) but the code filters it | Keep code behaviour, fix the text (methodology frozen); a switch is possible if you want weather exempt. | A9 |
| D4 | Hyper-parameters for the publication runs | Re-tune every model on the fixed features (thesis params were tuned on the buggy feature). Needs B7 fixed first. Alternative: reuse the thesis params (cheaper, less defensible). | B1, B7 |
| D5 | XGBoost multi-horizon strategy | Force one model per horizon (what the thesis text says, and what Colab's older darts did) for count and diff. | B3, B4 |
| D6 | Hurdle: which numbers are canonical | Session B (the stored predictions): Table 4 row becomes 0.80 / 2.02 / 0.09 and `tab:hurdle_bias` is regenerated; make `hurdle_cal` the leaderboard row. Session A cannot be reproduced by any code. | B8, B9, C6, C7 |
| D7 | Top-20 rule | `master_df[:20]` including linear (that is what reproduces the thesis F and H statistics); fix the appendix listing. | C9 |
| D8 | Seed-42 job matrix | B's 88-job thesis-faithful matrix (count GBDT × 3 paradigms; count RNN × global, activity; diff tuned × 3; diff baselines global; Chronos test; hurdle × 3 incl. CV for calibrators). Damage out. | B §matrix, C16 |
| D9 | How to verify the port without doubling compute | Legacy-mode **fold-subset** verification job: every model, first 1–2 retrain windows, compared to golden predictions; not full legacy re-runs. | B, C31 |
| D10 | Report outputs | Generated `.tex` fragments that `main.tex` `\input`s, SVGs with pinned DejaVu Sans and fixed metadata; figures built from the run store, not from `results/`. | C1, C2, C5 |
| D11 | W&B on the cluster | Online (compute nodes have internet) with `strict: false`, run store stays the truth. | C29 |

## 4. Implementation plan (after the decisions)

Order matters: correctness first, then completeness, then orchestration, then outputs.

**WP1 — correctness fixes (laptop, tests only).**
expdecay switch `series.window.expdecay: legacy_alpha | leaky` (default per D1, golden tests
pin legacy) with a feature-version tag in the run path (A1, A13 guard: refuse a cached selection
made on a different window config); XGBoost per-horizon (B3/B4) + a count XGBoost test;
RNN tuning objective = legacy (exp link, no per-fold RMSSE pruning) (B7); `hurdle_cal`
primary (B8); composite calibrators read from the tuning-seed CV (C18); import cell-56 params
for the five missing count GBDTs and cross-check the cluster's `lightgbm_tweedie_best.pkl` (B1);
deterministic feature selection (A12); panel cache keyed on file hash (A14);
`run.sbatch` must not export `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` (harmless for
LightGBM/CatBoost per B5, but keep thread settings explicit).

**WP2 — completeness.** Chronos-2 through the pipeline in `envs/autogluon` (C15); feature
importance jobs: GBDT gain + permutation across horizons, Chronos-2 permutation, category
shares (C14, needed for 4 thesis figures); cross-experiment SkillScore vs diff `naive_weekly`
(B16); horizon statistics (Shapiro, Levene, ANOVA, Tukey, Kruskal-Wallis) (B17); per-model
paradigm list in the schema (C16).

**WP3 — orchestration (`scripts/slurm/submit_all.py`).** One command, one DAG of SLURM jobs:
`setup` (install uv, build `.venv` + `envs/autogluon` once, CUDA + LightGBM smoke test) →
`featsel` (per family/head) → `tune` arrays → `cv` arrays → `test` arrays (seed list is an
argument, default 42) → `importance` → `report` → `figures`. `--dependency=afterok` between
stages, per-model resource class: GPU jobs `--gres=gpu:1 --gres-flags=disable-binding -c 8`,
CPU jobs `-c 48` (no GPU), partition `GPU` (36 h) with `GPUExtended` only for jobs that
cannot resume. Fold-level checkpoint/resume + `--signal=USR1@600` trap + requeue (C20, C21);
lock `state.json` (C23); no per-task `uv sync` (C24); logs dir created at submit (C27);
refuse to submit from a dirty/stale checkout; `--dry-run` prints the DAG; a `status` command
reads the run store + `squeue` (no `sacct` on this cluster). First submission: `--benchmark`
(1–2 folds, few trials per model) to size wall times.

**WP4 — outputs.** `strikecast figures` builds every DATA/RESULTS figure and table of the
thesis from the run store (and the data), writes SVG + `.tex` into one folder (C1, C2);
EDA dependencies (`geopandas`, `ruptures`) in a `figures` extra (C3); deterministic styling (C5);
generated `tab:splitdimensions`, `tab:activitytiers`, `tab:tuning-best-all` (C11, B2).

**WP5 — verification.** Legacy-mode fold-subset job for every model vs golden (D9); count
CatBoost as an exact level-E golden case (B); tighten the LightGBM ceiling (5.0 absolute is
too loose, B); hurdle/damage selector tests (A); replace the `xfail(strict=False)`
feature-selection "equality" tests with assertions (A).

## 5. Thesis/paper text that disagrees with the code (for Jan, no code change)

A1 expdecay7 (after D1 the text becomes true for the paper); A5 calendar encoders are past
covariates for the tuned RNNs; A6 RNNs do receive future covariates in darts 0.43 (the
reviewer's claim is wrong); A8 give RNN windows 7/14/28; A9 weather is filtered; A10 selector
objective differs per family; A17 "Cartesian product … six"; A18 115 vs 112 covariates;
B2 appendix params incomplete/rounded; B10 calibrator is fit on validation CV, not training;
B12 tuning backtest has ~79 daily folds, Chronos study has no pruner; B13 ARIMA(7,1,1) has a
drift constant and fixed orders; B22 hurdle count head is CatBoost-Tweedie with default
hyper-parameters trained with positive-only weights; C8 rank-1 MAE 0.78 vs 0.77; C9 Top-20
listing; C10 Kruskal-Wallis p = 0.483, not ≈ 0.05; C11 847 days, 593/85/169, 16,940
observations; C12 Chronos `fine_tune_steps` 200, not 600; B9/C6/C7 hurdle numbers (D6).

## 6. What the current tests do and do not prove

Pass: 1016 unit/equivalence, 245 golden (4 skipped). They prove the panel, windows, split,
backtest schedule, naives and calibration match the legacy code **run today** (not the pickled
thesis inputs, A §tests) and that LightGBM-Poisson, ARIMA and the diff subset agree with golden
within tolerances. They do **not** show that any count CatBoost/XGBoost, RNN, diff GPU GBDT,
hurdle prediction or Chronos model runs end to end on the real data; the feature-selection
"equality" tests are `xfail(strict=False)` and cannot fail; the LightGBM golden ceiling is too
loose to catch a regression.
