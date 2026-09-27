# Stream 4 (WP4) hand-off: every thesis figure and table generated automatically

Status: DONE (2026-09-27). Nothing committed. Decisions applied: D6 (hurdle = `hurdle_cal`
from the stored predictions), D7 (Top-20 = `master_df[:20]` incl. linear), D10 (SVG + `.tex`).

## Commands

```bash
# publication (what scripts/slurm/submit_all.py already runs after every report job)
PYTHONHASHSEED=0 strikecast figures --store-root runs_publication -v      # -> runs_publication/_figures/
# verification path: the thesis' stored outputs in results/ (+ golden/, _regression_GBDT.ipynb cell 56 for T8)
PYTHONHASHSEED=0 strikecast figures --source legacy --out /tmp/figs_legacy
# options: --out DIR, --seed 42 (store), --results-dir results, --golden golden, --data-dir data,
#          --only F12,T4,...   (item ids below)
```

No Hydra config is composed. Exit code 0 unless a builder *fails* (a missing input only
skips the item). Runtime ~8 s on the laptop (legacy, 30 items). `setup.sbatch` already
installs the `figures` extra when present (it is now), and `submit_all.py`'s `figures` node
(`figures --store-root <root> -v`) matches this CLI.

## Output layout (one folder, default `<store-root>/_figures/`, `/runs_*/` is gitignored)

`*.svg` figures under their thesis file names (drop-in for `writing/.../fig/`);
`tab_*.tex` + `tab_*.csv` per table (the `.tex` is the `tabular`/`tabularx` environment only;
caption and label stay in `main.tex`, which `\input`s it; T9 is longtable ROWS only);
`numbers.tex` (`\newcommand{\sc<Name>}{...}` macros for inline numbers: F, H, p-values, CD,
PR-AUC, prevalence, D, zero rates, STL strengths, 847/20/16940, tier bounds) + `numbers.json`;
`MANIFEST.md` + `MANIFEST.json` (id, thesis label, main.tex line, class, status, outputs,
source data, notes with audit IDs, source description, git commit).

## Code (files changed)

| file | what |
|---|---|
| `src/strikecast/reporting/__init__.py` (new) | package doc |
| `reporting/style.py` (new) | `thesis_style()` = `matplotlib.rc_context` (seaborn whitegrid dicts, DejaVu Sans, `svg.hashsalt=strikecast`, `svg.fonttype=path`), `save_svg()` (`metadata={"Date": None}` + `<dc:date>` strip). No global rcParams mutation (tested). |
| `reporting/sources.py` (new) | `ResultsSource` interface; `LegacySource` (AR notebook path rules incl. `*_local_naive_weekly`, chronos/finalhurdle cases; T8 params from `golden/converted/tuning`, `golden/checkpoints/chronos2_best` and `_regression_GBDT.ipynb` cell 56 parsed as a literal for CatBoost-Tweedie, B1); `StoreSource` (store -> AR vocabulary: count -> gbdt/lstm, hurdle -> finalhurdle, chronos `global` shown `local`; seed 42 default; hurdle calibrated probs = tuning-seed `artifacts/calibrators.json` applied to the `prob` channel, hurdle_cal = cal prob x count; GBDT importance concatenated from the per-run `importance/importance.csv` -- race-free, falls back to the collected `importance_all.csv`); `MissingInput`. |
| `reporting/compute.py` (new) | AR ports: `master_leaderboard` + `reference_rmse` (B16), `top_with_baselines`, `TABLE4_SPEC`/`table4`/`point_metrics` (cell 7), `top_models_table`, `region_matrix`, `horizon_statistics` (B17: Shapiro with the notebook's break/"reported p", Levene, ANOVA, Tukey, Kruskal-Wallis; ANOVA and KW both computed, `test` = the notebook's routing), `calibration_curves`, `prauc_vs_prevalence`, `hurdle_bias_table`, `activity_tier_table`, `split_dimensions` (darts `split_series_list`), `base_variables`, `pretty_model`. |
| `reporting/plots.py` (new) | F12/F13 per-region tier grids, F18, F19, F16, F17, F20 (via `evaluation.importance.category_importance_matrix`, so expdecay7 and leaky7 names map identically), F21/F23 grids, `cd_label`, `model_colors` (cell 12 palette). CD diagrams reuse `evaluation.stats`. |
| `reporting/eda.py` (new) | `load_eda` (eda_full cells 1-2), F5, F6, F7, F4, F2/F3 geopandas maps (`MissingInput` if geopandas is absent), `marginal_statistics`, `tiers_from_json`. |
| `reporting/build.py` (new) | `ITEMS` registry (audit C Part 1 + extras X1-X5), `BuildContext`, one builder per item, `build_all`, `write_numbers`, `write_manifest`. `bbox_inches="tight"` only where the notebook cell used it. |
| `src/strikecast/cli/main.py` (additive) | subcommand `figures` + flags `--source --out --results-dir --data-dir --seed --only` (reuses `--store-root`, `--golden`). |
| `pyproject.toml`, `uv.lock` | extra `figures = [geopandas>=1.0, ruptures>=1.1]` (C3); `uv lock` added geopandas 1.1.4, pyogrio, pyproj, shapely, ruptures 1.1.10. Installed in `.venv` with `uv sync --frozen --inexact --extra dev --extra figures`. |
| `tests/unit/test_reporting.py` (new, 21 tests) | see Tests. |

## Per-item status (legacy run -> `/tmp/s4/legacy`; sparse `runs/` store -> `/tmp/s4/store`)

| id | label | legacy | sparse `runs/` |
|---|---|---|---|
| F1, F8-F11, F22, T1, T7, T11 | hand-drawn / typed | static | static |
| F2, F3 | fig:strikeactivity_perregion(_activity_levels) | generated | generated |
| F4-F7 | heatmap, target distribution, PACF/ACF, STL | generated | generated |
| F12, F13 | fig:rmse/mae_per_region | generated | generated (top-5 of the 6 runs present) |
| F14, F15 | CD top-5 + ARIMA | generated | generated |
| F16, F17, T6, X4 | hurdle classifier / bias | generated | skipped: no hurdle test predictions |
| F18 | fig:top_5_horizon | generated | generated |
| F19, X3 | top-20 horizon boxplot / statistics | generated | skipped: leaderboard has 6 rows, need 20 |
| F20, F23 | GBDT importance | generated | skipped: no count importance |
| F21 | Chronos importance | generated | skipped: no Chronos importance |
| T2, T3, T9 | tiers, split, base variables | generated | generated |
| T4 | tab:overall_performance | generated | generated, 4 of 11 rows (missing rows listed in the manifest) |
| T5, T10, X1, X2 | top-5, top-20, master leaderboard, Friedman ranks | generated | generated (from what exists) |
| T8 | tab:tuning-best-all | generated | generated (`runs/*/tuning/*/best_params.json`) |
| X5 | EDA marginals | generated | generated |

Legacy: 30 generated, 9 static, 0 skipped/failed. Store: 21 generated, 9 skipped (each with
its reason in the manifest), 9 static, exit 0. Two legacy builds are byte-identical (`diff -rq`
empty).

## Verification against the thesis (legacy source; asserted in `tests/unit/test_reporting.py`)

Reproduce at the printed precision: Top-20 appendix values (6 dp, all 20 rows, with rank 20 =
diff/global/linear, D7/C9); Table 4 for 10 of 11 rows; tab:top_models Skill Score and RMSE;
tab:tuning-best-all value for value (all 24 entries, CatBoost-Tweedie from cell 56); Friedman
average ranks (RMSE 2.65/2.75/3.14/3.34/4.51/4.62; MAE 1.59/2.89/2.91/4.81), CD = 0.64, both
p < 0.001; horizon RMSE: Shapiro p 0.41, F = 6.7041, ANOVA p < 0.001, Tukey p 3v5 0.008,
1v5 0.43, 2v5 0.13, 4v5 0.48, 1v6 0.027, 2v6 0.003, 3v6 < 0.001, 4v6 0.033; horizon MAE:
Shapiro p < 0.05, H = 5.4895; PR-AUC 0.817/0.817, prevalence 0.294; tier thresholds
7/84/219/411; 112 base variables; D = 1.15/2.03/5.16, Tier-1 zero rate 0.98; STL F_T
0.73 -> 0.09, F_S 0.38 -> 0.36; hurdle-bias regions, counts and maxima.

SVG sizes equal the thesis copies for 11 figures (asserted; skips without the thesis folder):
top_5_combined, calibration_prcurve, prauc_vs_prevalence, top20_rmse_horizon, F20, F21, F23,
STL, heatmap, both maps. Sub-point differences remain for F5 (+0.7 pt), F6 (+0.7 pt height),
F12/F13 (+0.3 pt height): the thesis runs had other global rcParams in force (C5).

## Known mismatches (documented in the manifest notes, not "fixed" in code)

| audit | item | thesis | generated (stored outputs) |
|---|---|---|---|
| C7/D6 | T4 Hurdle row | 0.79 / 2.00 / 0.09 | 0.80 / 2.02 / 0.09 (hurdle_cal predictions) |
| C6/D6 | T6 tab:hurdle_bias | Sumy 7.35 / 4.83 / -4.17 ... | Sumy 6.79 / 4.48 / -3.13 ... (counts, maxima identical) |
| C8 | T5 rank-1 MAE | 0.78 | 0.77 (0.767909) |
| C9 | T10 Top-20 | drops rank 20 (linear), lists rank 21 | `master_df[:20]` incl. linear (D7) |
| C10 | KW p (text 1081) | p ~ 0.05 | 0.483 |
| C11 | T3 split | 592/85/169 = 846; 11,844/1,692/3,384 = 16,920 | 593/85/169 = 847; 11,860/1,700/3,380 = 16,940 |
| audit C F17 | Delta range (text 1031) | 0.06-0.10 | 0.057-0.108 |
| -- | Levene RMSE (text 1085) | 0.54 | 0.5461 |
| audit C F14 | CD figure height | 472.6 pt | 309.9 pt (thesis SVG from an older code version; ranks identical) |
| C12 | T7 fine_tune_steps | {600..3000} | static item, not generated (code: 200..3000) |

## Not done / for Jan

- `main.tex` is not edited: switch the tables to `\input{.../tab_*.tex}` (+ `numbers.tex`) and
  copy the SVGs into `fig/` once the publication run is in. T2 needs the `Y` column type that
  main.tex already defines; T6's `.tex` keeps the thesis' top 8.
- Single seed (default 42, `--seed`); seed means/CIs stay in `strikecast report`.
- F12-F15/F18/F19 pick the top-k from the store's own master leaderboard, so re-run models can
  change the selection (one explicit rule, C9).
- The store paths for hurdle probabilities/`hurdle_cal`, importance and Chronos are covered by
  a synthetic mini-store test only; `runs/` has no such runs yet.
- Paper-only `tab:covariates` (P1) is not generated.

## Tests

`tests/unit/test_reporting.py`, 21 tests: legacy vs thesis numbers (11), byte determinism +
rcParams isolation, empty store (every item skipped with a reason, no failure), the real sparse
`runs/` store, a synthetic mini store (mapping, naive relabel, hurdle calibration, hurdle_cal),
the CLI, helpers.
Full suite after this stream (`tests/unit tests/equivalence tests/golden`): 1498 passed, 13 skipped, 1 xfailed (5 min 28 s);
`ruff check .` clean.
