# Machine and Deep Learning for Conflict Event Forecasting

**Multi-Horizon Spatiotemporal Prediction of Drone Strikes on Ukrainian Regions**

This repository contains the code, models, and results for my Bachelor's thesis in Cognitive Science & Artificial Intelligence at Tilburg University.

## Author
**Jan van Gestel**

## Overview
This project evaluates the extent to which machine learning and deep learning algorithms can produce multi-horizon spatiotemporal forecasts of drone-strike events across Ukrainian regions. It compares various predictive models including gradient boosting trees (XGBoost, LightGBM, CatBoost), LSTMs, and foundational time-series models like Chronos. 

## Repository Structure
* **`BscThesisJanvanGestel.pdf`**: The final thesis document detailing the methodology, experiments, and conclusions.
* **Notebooks (`*.ipynb`)**: Jupyter notebooks containing the training, evaluation, and hurdle models (e.g., `_chronos2.ipynb`, `_regression_GBDT.ipynb`, `damage_classifier.ipynb`).
* **`src/`**: Helper scripts for feature engineering, evaluation, and time-series specific tools.
* **`data/`**: Processed datasets, region mappings, and actor information.
* **`checkpoints*/`**: Saved model weights and fine-tuned configurations.
* **`results/`**: Evaluation metrics, generated forecasts, and performance plots.
