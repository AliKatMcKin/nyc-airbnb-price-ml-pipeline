# NYC Airbnb Price Prediction | Reproducible ML Pipeline

An end-to-end machine learning pipeline that predicts nightly rental prices for New York City Airbnb listings, built so that the entire workflow can be re-run from a single command against new data.

- **Stack:** Python · scikit-learn · MLflow · Hydra · Weights & Biases · pandas
- **Model:** Random Forest Regressor
- **Data:** NYC Airbnb listings, prices constrained to $10–$350

---

## Overview

The emphasis of this project is reproducibility rather than model performance. Each stage of the workflow is an independently versioned MLflow component with its own environment specification, configuration is centralized in a single Hydra-managed file, and every run produces artifacts and metrics tracked in Weights & Biases.

The pipeline includes an automated data validation stage that compares incoming data against a reference dataset and fails the run if the distributions have diverged, which prevents a silently corrupted input from producing a plausible-looking model.

## Pipeline Stages

| Stage | Function |
|---|---|
| `download` | Retrieves the raw listings sample and registers it as a versioned W&B artifact |
| `basic_cleaning` | Applies price boundaries ($10–$350) and geographic constraints, outputs a cleaned dataset |
| `data_check` | Validates the cleaned data against a reference version using a KL-divergence threshold to detect distribution drift |
| `data_split` | Partitions into train, validation, and test sets, stratified by `neighbourhood_group` |
| `train_random_forest` | Trains the model and exports it as a versioned MLflow artifact |

A `test_regression_model` stage exists but is excluded from default execution. It runs only against a model that has been explicitly promoted to production in W&B, so that final test-set evaluation happens once rather than being optimized against.

## Configuration

All parameters are managed through `config.yaml` and overridable at the command line via Hydra.

| Parameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 15 |
| min_samples_split | 4 |
| min_samples_leaf | 3 |
| max_features | 0.5 |
| criterion | squared_error |
| Test size | 0.2 |
| Validation size | 0.2 |
| Stratification | `neighbourhood_group` |
| Random seed | 42 |

Feature engineering includes TF-IDF extraction on listing text fields, imputation for missing values, and ordinal and one-hot encoding of categorical features, all composed into a single scikit-learn pipeline so that preprocessing is exported alongside the model.

## Results

Experiment runs, artifact lineage, and model metrics are tracked in Weights & Biases:

**[W&B project report](https://wandb.ai/alissa-mckinney-western-governors-university/nyc_airbnb/reports/nyc_airbnb-report--VmlldzoxNzExMzU1Mg?accessToken=eunvig99reyqw4rxpevay2k07e1tddzz08dlqahfgb8lhbmyngqa2q1a85o7ih7w)**

## Setup and Usage

Requires conda and a Weights & Biases account.

```bash
git clone https://github.com/AliKatMcKin/nyc-airbnb-price-ml-pipeline.git
cd nyc-airbnb-price-ml-pipeline

conda env create -f environment.yml
conda activate nyc_airbnb_dev
wandb login
```

| Task | Command |
|---|---|
| Run the full pipeline | `mlflow run .` |
| Run selected stages | `mlflow run . -P steps=download,basic_cleaning` |
| Override a parameter | `mlflow run . -P hydra_options="modeling.random_forest.max_depth=20"` |
| Run a hyperparameter sweep | `mlflow run . -P hydra_options="modeling.random_forest.max_depth=10,15,20 -m"` |

## Repository Structure

```
├── main.py            # Pipeline orchestration
├── config.yaml        # Hydra configuration: all parameters
├── MLproject          # MLflow entry point definition
├── environment.yml    # Conda environment
├── src/               # Pipeline stage implementations
├── components/        # Reusable components (data retrieval, splitting)
└── images/            # Pipeline and artifact diagrams
```

## Attribution

Project scaffolding, reusable components, and starter structure were provided as part of the Udacity/WGU Machine Learning DevOps curriculum. The cleaning logic, data validation tests, model configuration, feature engineering pipeline, and pipeline orchestration are my own work.

*Completed for the Machine Learning DevOps course, BS in Data Analytics, Western Governors University. Author: Alissa McKinney.*
