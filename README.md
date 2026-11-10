# Appliance Energy Prediction

Predicts how much energy household appliances use (in Wh) in the next 10 minutes, using the UCI Appliances Energy Prediction dataset. Machine learning models (Linear Regression, Ridge, XGBoost, LightGBM, CatBoost) are compared with deep learning models (GRU, LSTM, CNN-LSTM).

## Setup

You need [uv](https://docs.astral.sh/uv/) (install it with `pip install uv`).

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run jupyter notebook
```

## How to run

Run the notebooks in the `notebooks/` folder in this order.

### 1. EDA.ipynb
Explores the raw data (`data/energy_data_set.csv`) and saves a cleaned copy to `notebooks/data/eda.parquet`.

### 2. Feature_Engineer.ipynb
Builds the model input data. Change the settings in `FeatureConfig` (which features to create, how many to keep), then run the notebook.

It saves these files to `notebooks/data/`:

| File | What it is |
|---|---|
| `processed.npz` | the final train, validation and test data |
| `feature_scaler.pkl`, `target_scaler.pkl` | scalers, used to turn predictions back into Wh |
| `feature_config.json` | the feature settings and the list of features used |
| `feature_ranking.csv` | the rank of every feature from feature selection (RFE) |
| `feature_count_comparison.csv` | validation scores for different numbers of features |

`processed.npz` is not stored on GitHub because it is large, so run this notebook before training.

### 3. Train_ML.ipynb and Train_DL.ipynb
Train the models. There are two ways to run them:

- **One model:** set `model_name` and `model_params` in the config, then run `run_ml(config)` or `run_dl(config)`.
- **Grid search:** write the values to try in `ml.json` or `dl.json`, then run `run_ml_grid(...)` or `run_dl_grid(...)`. Every combination is trained.

Example `ml.json`:
```json
{
  "linear": {},
  "ridge": {"alpha": [0.1, 1, 10]}
}
```
`{}` means one run with the default settings.

Keep `evaluate_test=False` while experimenting. Set it to `True` only for the final run, to get the test score.

## Results

Every run is saved in `models/training_runs/`:

| File | What it is |
|---|---|
| `results.csv` | one row per run: scores, model settings and feature settings |
| `<date_time>_<run name>/` | one folder per run |

Each run folder contains:

| File | What it is |
|---|---|
| `config.json` | the model settings |
| `feature_config.json` | the feature settings |
| `metrics.json` | MAE, RMSE, MAPE and R² for each split |
| `predictions_val.csv`, `predictions_test.csv` | actual and predicted values |
| `history.csv` | train and validation loss per epoch (DL only) |
| `model.joblib` / `model.pt` | the trained model |

## Code (`src/`)

The notebooks only hold the settings. The code that does the work is in `src/`.

| File | What it does |
|---|---|
| `feature_engineer.py` | runs all the data preparation steps in order |
| `preprocessing/cleaning.py` | caps very high energy values (outliers) |
| `preprocessing/features.py` | creates the features: past energy use, rolling averages, changes, time of day, holidays |
| `preprocessing/splitting.py` | splits the data by time into train, validation and test |
| `preprocessing/scaling.py` | scales the data (fitted on train only) |
| `preprocessing/selection.py` | feature selection with RFE |
| `preprocessing/data_io.py` | loads and saves files |
| `ml_runner.py` | trains ML models, one at a time or from `ml.json` |
| `dl_runner.py` | trains DL models, one at a time or from `dl.json` |
| `training/ml_models.py` | the list of ML models |
| `training/dl_models.py` | the GRU, LSTM and CNN-LSTM models |
| `training/windows.py` | cuts the past rows each DL model looks at |
| `training/torch_loop.py` | the DL training loop with early stopping |
| `training/grid.py` | reads `ml.json` / `dl.json` and makes every combination |
| `training/data.py` | loads the processed data |
| `training/metrics.py` | calculates MAE, RMSE, MAPE and R² |
| `training/results.py` | saves run folders and `results.csv` |

## Other folders

| Folder | What it is |
|---|---|
| `data/` | the raw dataset |
| `reports/` | the report and its figures |
| `models/` | training runs and saved models |