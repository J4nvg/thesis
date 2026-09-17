# Machine and Deep Learning for Conflict Event Forecasting

**Multi-Horizon Spatiotemporal Prediction of Drone Strikes on Ukrainian Regions**

This repository contains the code, models, and results for my Bachelor's thesis in Cognitive Science & Artificial Intelligence at Tilburg University.

## Author
**Jan van Gestel**

## Overview
This project evaluates the extent to which machine learning and deep learning algorithms can produce multi-horizon spatiotemporal forecasts of drone-strike events across Ukrainian regions. It compares various predictive models including gradient boosting trees (XGBoost, LightGBM, CatBoost), LSTMs, and foundational time-series model Chronos-2. 

## Repository Structure
* **Notebooks (`*.ipynb`)**: Jupyter notebooks containing the training, evaluation, and hurdle models (e.g., `_chronos2.ipynb`, `_regression_GBDT.ipynb`, `damage_classifier.ipynb`).
* **`src/`**: Helper scripts for feature engineering, evaluation, and time-series specific tools.
* **`data/`**: Processed datasets, region mappings, and actor information.
* **`checkpoints*/`**: Saved model weights and fine-tuned configurations.
* **`results/`**: Evaluation metrics, generated forecasts, and performance plots.

Notebooks were converted to .py files before ran on a cluster using:


```sh
$ jupyter nbconvert --to script [NB].ipynb

```

## Repository status (September 2026)

This code base is being refactored into the `strikecast` package. The plan, scope, and open questions
live in [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md).

* `archive/` holds the log-transformed regression experiment and pre-refactor leftovers. These are not
  reported in the thesis and are kept read-only.
* `golden/` (git-ignored) holds a frozen copy of the thesis results, checkpoints, and features, used as
  the reference for regression checks against refactored code.
* `envs/autogluon/` is a standalone uv environment for the Chronos-2 experiment (`_chronos2.py`):
  AutoGluon 1.5.0 caps `pandas<2.4` and cannot coexist with the main pandas 3.0.2 pin. Sync it with
  `uv sync --project envs/autogluon` and run `uv run --project envs/autogluon python _chronos2.py`.
