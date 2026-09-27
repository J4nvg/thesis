# Legacy code retirement audit

Plan §8 P8 says "delete the old scripts once golden passes". This is the audit
of what *can* go, what *cannot*, and why.

**Nothing has been moved.** The one precondition — golden levels E/F green on
the full model set — is **not met**: E/F pass on the cheap deterministic subset
only (`diff`: naives, `linear`, `arima`, `lightgbm`; `count`:
`lightgbm_poisson`). The rest of the thesis' models are days of compute and
have never been re-run into the run store. Until they are, the legacy code is
the only record of how they were produced.

Audited 2026-09-19 against the suite at **1290 passed, 5 skipped, 3 xfailed**.

## The rule

A file is retirable only when **both** hold:

1. nothing in `src/strikecast/`, `tests/`, or `scripts/` *imports or reads* it
   (a provenance citation in a comment does not count); and
2. its replacement is verified — by a golden level, an equivalence test, or a
   measured in-process comparison.

Condition 1 was checked by grep for real imports and file reads, separately
from the ~300 comment citations of the form `_regression_GBDT.py:123`. Those
citations are **why the refactor is auditable** and are not a dependency.

## Verdicts

| Path | Replaced by | Live dependency | Verdict |
|---|---|---|---|
| `src/__init__.py`, `src/prevalent_functions.py`, `src/evaluation_tools.py`, `src/general_tools.py`, `src/feature_tools.py`, `src/ts_specific_tools.py`, `src/dataset_specific_tools.py`, `src/dummy_tools.py` | `strikecast.data`, `.evaluation`, `.backtest` | **YES — imported at test time.** `tests/conftest.py::legacy_src`; `tests/unit/test_predictions.py`; `tests/golden/test_{panel,covariates,naive}_equality.py`; and `tests/legacy_ref/{diff_builders,count_builders,diff_runners,rnn_builders}.py` import `src.prevalent_functions`, `src.evaluation_tools`, `src.general_tools` directly | **NEVER MOVE.** This is the oracle the whole golden/equivalence suite compares against. Moving it turns levels A, B, D and G from assertions into skips. |
| `_regression_GBDT.py` / `.ipynb` | `configs/experiment/count.yaml` + `models/gbm.py` + `pipeline/run_stage.py` | No import; cited widely | **Keep until full E/F.** Only `lightgbm_poisson` of this family has been re-run; the other 5 GBDT variants have no store counterpart. |
| `_diff_regression.py` / `.ipynb` | `configs/experiment/diff.yaml` + `transforms/diff.py` | No import; cited widely | **Keep until full E/F.** 5 of 31 models re-run. The ARIMA path is the thesis' Table 4 baseline and is level-F pinned, but the GBDT/RNN variants are not. |
| `_regression_LSTM.py` / `.ipynb` | `models/rnn.py` + the `lstm_*`/`gru_*` registry entries | No import; cited widely | **Keep.** Torch training is not bit-reproducible (F9) and no RNN has been re-run; the notebook is the only record of the reported numbers. |
| `_chronos2.py` | `models/chronos.py`, `data/autogluon.py`, `configs/experiment/chronos2.yaml`, `envs/autogluon/` | No import; cited widely | **Keep.** The adapter exists and has unit + golden tests, but the family has never been run end to end here (it needs the separate AutoGluon environment). |
| `final_hurdle.ipynb` | `pipeline/composite_stage.py` + `models/hurdle.py` | No import. `tests/legacy_ref/hurdle_builders.py` is a *transcription* of its loops, not an import | **Keep.** Golden level H asserts calibration and metrics exactly, but the *prediction* half is unassertable: the thesis' two feature selections cannot be recovered (open question, see the progress log). |
| `damage_classifier.ipynb` | `composite_stage.py` damage path | No import; transcribed into `tests/legacy_ref/damage_runners.py` | **Keep.** The family persisted nothing in the thesis, so there is no golden counterpart at all — correctness rests on level D plus unit tests. |
| `event_classifiers_prehurdle.ipynb`, `regressor_part_prehurdle.ipynb` | not yet ported (plan §10, level I) | No import | **Keep.** Still in scope and not yet replaced. Also the likely source of `golden/converted/feature_sets/zipoisson.json`, which level H needs identified. |
| `tune_gbdt.sh`, `tune_lstm.sh`, `tune_diff_reg.sh` | `strikecast tune` + `scripts/slurm/` | No import. Referenced in `scripts/slurm/tune.sbatch` and the plan **as prose**, for the hyper-parameter ranges they document | **Keep for now.** Superseded functionally, but they record the trial budgets and ranges the thesis used. Retire together with the `_*.py` exports. |
| `eda.ipynb`, `eda_full.ipynb`, `results/analyse_results.ipynb` | — | Actively used | **Keep — rewired, not retired.** They now read through `strikecast.data` and `strikecast.evaluation`; see the progress log for the verified-identical outputs. |
| `archive/**` | — | No | Already retired in P0 (the log-transformed family, out of scope). |

## What unblocks the next pass

1. Re-run the remaining `count` and `diff` variants into the run store and
   extend `CASES` in `tests/golden/test_pipeline_equality.py`. The GBDTs will
   land at level F, not E (F124: CPU GBDT results are not portable across
   machines — the *verbatim legacy runner* misses its own stored predictions
   here by the same margin the port does).
2. Identify which hurdle head `golden/converted/feature_sets/zipoisson.json`
   belongs to; that makes the prediction half of level H assertable.
3. Run `chronos2` once in `envs/autogluon/`.

Only then does `_regression_GBDT.py`, `_diff_regression.py`,
`_regression_LSTM.py`, `_chronos2.py` and the three `tune_*.sh` become movable
— and they move to `archive/`, never deleted: `golden/` is git-ignored, so
these files plus the notebooks are the only in-repo record of how the thesis'
numbers were produced.

**The legacy `src/` package never moves**, regardless of golden state.
