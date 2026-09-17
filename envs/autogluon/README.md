# `envs/autogluon` — isolated environment for the Chronos-2 experiment

Standalone uv environment used **only** to run [`_chronos2.py`](../../_chronos2.py)
(AutoGluon TimeSeries 1.5.0 / Chronos-2). It is deliberately *not* part of the
main project's uv workspace and has its own `pyproject.toml`, `uv.lock` and
`.venv`.

## Why it exists

AutoGluon 1.5.0 caps `pandas>=2.0,<2.4`, `numpy<2.4`, `scipy<1.17` and
`pyarrow<21`. The main project is frozen on pandas 3.0.2 / numpy 2.4.4 /
scipy 1.17.1 / pyarrow 24.0.0, so the two cannot be satisfied together.
Forcing AutoGluon into the main env with uv `override-dependencies` installs,
but fails at import time:

```
ImportError: cannot import name 'ArrayManager' from pandas.core.internals
```

(`ArrayManager` was removed in pandas 3.0.) See the comment block at the top of
[`../../pyproject.toml`](../../pyproject.toml). The historical `pip freeze`
(`git show bcf16c8:requirements.txt`) lists both `autogluon==1.5.0` and
`pandas==3.0.2`, which cannot have been the environment that actually produced
the Chronos-2 results — hence this separate, reproducible env.

## Sync

From the repo root (uv installs the right Python itself; no conda needed):

```bash
uv sync --project envs/autogluon
```

Or from inside this directory: `uv lock && uv sync`.

## Run `_chronos2.py`

Always run it **from the repo root**, because the script does `from src import *`
and uses `PROJECT_DIR = './'` for relative `data/`, `features/`, `results/` paths:

```bash
cd /path/to/thesis
uv run --project envs/autogluon python _chronos2.py
```

Quick smoke test:

```bash
uv run --project envs/autogluon python -c "from autogluon.timeseries import TimeSeriesPredictor; import src; print('ok')"
```

### macOS (Apple Silicon)

Works out of the box; torch resolves to the CPU/MPS wheel and
`torch.cuda.is_available()` is `False`, which the script handles. Fine-tuning
Chronos-2 on CPU/MPS is slow — use it for smoke tests, run the real thing on the
cluster.

### SLURM cluster (conda-free)

`uv` downloads its own CPython 3.12 build, so no module load / conda env is
needed:

```bash
# once, on the login node (needs network):
cd $HOME/thesis
uv sync --project envs/autogluon

# in the job script:
cd $HOME/thesis
uv run --project envs/autogluon python _chronos2.py
```

On a Linux + CUDA node, `torch==2.9.1` pulls the CUDA wheels together with the
`nvidia-*`/`triton` transitive packages automatically; they are pinned inside
`uv.lock` and are not listed as direct dependencies. Set
`HF_HOME`/`HUGGINGFACE_HUB_CACHE` to a shared scratch path so the Chronos-2
weights are downloaded once, and keep `UV_CACHE_DIR` on a filesystem the compute
nodes can read.

## The Chronos-2 model

Registered under aliases `Chronos2`, `Chronos-2` and `Chronos`(v1 is a separate
`ChronosModel`). The class backing `hyperparameters={"Chronos2": {...}}` in
`_chronos2.py` is:

```
autogluon.timeseries.models.chronos.chronos2.Chronos2Model
# also importable as: from autogluon.timeseries.models import Chronos2Model
```

so the legacy script's hyperparameter key is correct for 1.5.0 and needs no
change.

## Resolved key versions

| package | version |
| --- | --- |
| Python | 3.12.14 |
| autogluon.timeseries (+core/common/features/tabular) | 1.5.0 |
| chronos-forecasting | 2.3.2 |
| pandas | 2.3.3 |
| numpy | 2.3.5 |
| scipy | 1.16.3 |
| pyarrow | 20.0.0 |
| torch | 2.9.1 |
| darts | 0.43.0 |
| scikit-learn | 1.6.0 |
| matplotlib | 3.10.9 |
| optuna / optuna-integration | 4.8.0 |
| lightning / pytorch-lightning | 2.5.6 / 2.5.2 |
| transformers | 4.57.6 |

## Pin deviations from the historical freeze (`bcf16c8:requirements.txt`)

Everything below is forced by AutoGluon 1.5.0's own metadata; nothing was
downgraded by choice.

| package | historical | here | reason |
| --- | --- | --- | --- |
| Python | 3.13 (main project) | 3.12 | autogluon.timeseries 1.5.0 is `>=3.10,<3.14`; 3.12 has the widest wheel coverage for the whole tree. |
| `pandas` | 3.0.2 | 2.3.3 | autogluon `<2.4.0`; pandas 3 removed `ArrayManager` (the import error above). darts 0.43.0 needs `>=2.2.0`, so 2.3.x is the only overlap. |
| `numpy` | 2.4.4 | 2.3.5 | autogluon `<2.4.0`; darts `>=2.2.0`. |
| `scipy` | 1.17.1 | 1.16.3 | autogluon `<1.17`. |
| `pyarrow` | 24.0.0 | 20.0.0 | autogluon-common `<21`. |
| `chronos-forecasting` | 2.2.2 | 2.3.2 | autogluon allows `>=2.2.2,<2.4`; left floating within that range (2.2.2 was a `pip freeze` snapshot, not a real pin). |
| `holidays`, `shap`, `gluonts`, other transitives | frozen by `pip freeze` | resolver's choice | only direct dependencies are pinned here; transitive versions are frozen by `uv.lock`, same convention as the main `pyproject.toml`. |
| `comet-ml` | 3.57.3 | dropped | tracking moved to W&B (same decision as the main project). |
| `autogluon` (meta-package) | 1.5.0 | `autogluon.timeseries==1.5.0` only | the script only uses the timeseries module; the full meta-package also drags in `autogluon.multimodal` (spacy / nlpaug / openmim / mmcv-style deps) which is unused and slow to resolve. `autogluon.tabular[catboost,lightgbm,xgboost]` still comes in transitively. |
| `imbalanced-ensemble`, `venn-abers`, `hydra-core`, `wandb`, `seaborn`, `xgboost`… | pinned in the main env | not here | `_chronos2.py` + `src/` never import them. Keep this env minimal; the main env is unchanged. |

Unchanged from the historical/main pins: `torch==2.9.1` (autogluon allows
`>=2.6,<2.10`), `darts==0.43.0`, `scikit-learn==1.6.0`, `matplotlib==3.10.9`,
`statsmodels==0.14.6`, `optuna==4.8.0`, `optuna-integration==4.8.0`,
`pytorch-lightning==2.5.2`, `tqdm==4.67.3`.
