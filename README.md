# Appliance Energy Prediction

Predicting how much energy the household appliances of a house in Belgium will use in the **next 10 minutes**, from the UCI *Appliances Energy Prediction* dataset (19,735 readings, one every 10 minutes, 11 January – 27 May 2016).

Ten models are compared: seven machine learning models (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost) and three deep learning models (GRU, LSTM, CNN-LSTM). Every experiment is driven by settings in the notebooks and JSON files, and every run is logged automatically, so results can be compared and repeated.

The full write-up is in [`reports/Assignment Report - Manith Ratnayake.pdf`](reports/).

---

## Results

Test set scores (the last 15% of the data, used once at the end). All values are in Wh, measured against the capped target (maximum 175 Wh, see [Data preparation](#2-feature_engineeripynb--data-preparation)).

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|
| **XGBoost** | **13.17** | **20.52** | **16.44** | **0.72** |
| CNN-LSTM | 13.93 | 20.90 | 18.34 | 0.71 |
| Ridge | 13.58 | 21.07 | 17.48 | 0.71 |
| Linear Regression (baseline) | 13.63 | 21.10 | 17.62 | 0.71 |
| Decision Tree | 13.48 | 21.17 | 16.54 | 0.71 |
| GRU | 15.37 | 21.60 | 20.87 | 0.69 |
| CatBoost | 15.11 | 21.63 | 20.19 | 0.69 |
| LSTM | 15.23 | 21.67 | 20.57 | 0.69 |
| Random Forest | 15.58 | 22.60 | 18.61 | 0.66 |
| Gradient Boosting | 17.62 | 23.80 | 23.60 | 0.63 |

Main findings:

- **XGBoost** was the best model, about 2.7% lower RMSE than the Linear Regression baseline. **CNN-LSTM** was the best deep learning model, about 1% better than the baseline.
- **Features mattered far more than the model.** Capping the spikes and adding past-energy features lowered the GRU's MAE from about 27 Wh to about 13 Wh. Tuning changed results by less than 1 Wh.
- **The most recent reading (`lag_1`) is by far the strongest predictor.**
- Several models that were best on validation (CatBoost, the tuned GRU) did worse on the test month, while Linear Regression stayed stable. The test month (May) is warmer than anything in the training data.

---

## Project structure

```
Assignment/
├── data/
│   └── energy_data_set.csv          # raw dataset
├── notebooks/
│   ├── EDA.ipynb                    # 1. explore the data
│   ├── Feature_Engineer.ipynb       # 2. build features, split, scale, select features
│   ├── Train_ML.ipynb               # 3. train the machine learning models
│   ├── Train_DL.ipynb               # 3. train the deep learning models
│   ├── Results.ipynb                # 4. test-set scores, tables and figures for the report
│   └── data/                        # outputs of steps 1 and 2
├── src/                             # all the code the notebooks call
│   ├── feature_engineer.py
│   ├── ml_runner.py
│   ├── dl_runner.py
│   ├── preprocessing/
│   └── training/
├── models/
│   ├── ml.json                      # grid search values for the ML models
│   ├── dl.json                      # grid search values for the DL models
│   ├── ml_rest.json                 # LightGBM grid (see Known issues)
│   └── training_runs/               # one folder per run + results.csv
├── reports/
│   ├── Assignment Report - Manith Ratnayake.pdf
│   └── figures/                     # every figure used in the report
├── requirements.txt
└── README.md
```

The notebooks only hold **settings** (as Python dataclasses) and call functions in `src/`. All the logic lives in `src/`, so it can be reused and tested.

---

## Setup

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/) (install it with `pip install uv`).

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run jupyter notebook
```

Open the notebooks from the `notebooks/` folder. They use paths relative to that folder (`../src`, `data/`, `../models/`).

A GPU is optional. The deep learning code uses one automatically if PyTorch can see it, otherwise it runs on the CPU (about 20 seconds to 2 minutes per DL run).

---

## How to run

Run the notebooks in this order.

### 1. `EDA.ipynb`: explore the data

Loads `data/energy_data_set.csv` and checks it:

- no missing values, no duplicate rows, no gaps in the 10-minute timeline
- distribution and outliers of the target (`Appliances`)
- daily and weekly patterns (hour × day heatmap, weekday vs weekend)
- autocorrelation, used to choose the lag features, plus a small test of which lags work best
- correlation between every feature and the target

Saves the cleaned data to `notebooks/data/eda.parquet`.

### 2. `Feature_Engineer.ipynb`: data preparation

All settings are in the `FeatureConfig` dataclass. The pipeline (`src/feature_engineer.py`) runs these steps in order:

1. **Cap outliers.** Target values above the IQR upper fence (175 Wh) are lowered to 175 Wh. The fence is worked out from the training rows only. Rows are not deleted, so the 10-minute timeline stays complete.
2. **Shift sensor readings by one step.** Every input only uses information available *before* the reading being predicted.
3. **Create features:**

   | Group | Features |
   |---|---|
   | Past energy use | `lag_1` … `lag_144` (10 min to 24 h ago), rolling means and standard deviations over 30 min – 24 h, differences over 10 min – 24 h |
   | Time | `hour_sin`, `hour_cos`, `day_sin`, `day_cos` (time as a circle), `is_weekend` |
   | Domain | `is_holiday` (Belgian public holidays in 2016) |
   | Interactions | temperature × humidity for the kitchen, living room and outside (`T1_x_RH_1`, `T2_x_RH_2`, `T6_x_RH_6`) |
   | Sensor summaries | average and spread of the 8 indoor temperature and humidity sensors, indoor–outdoor temperature gap |

   The random columns `rv1` and `rv2` are dropped.
4. **Drop the first 145 rows**, which don't have enough history for the 24-hour features.
5. **Split by time**, without shuffling: 70% train (13,712 rows), 15% validation (2,939), 15% test (2,939).
6. **Scale** features and target with `StandardScaler`, fitted on the training rows only.
7. **Select features with RFE** (Recursive Feature Elimination with a Ridge model). All 67 features are ranked on the training rows, the top 5, 10, 15, 20, 30, 40 and all 67 are compared on validation, and the top `n_selected_features` (30) are kept.

Outputs, saved to `notebooks/data/`:

| File | What it is |
|---|---|
| `processed.npz` | scaled train / validation / test data (not on GitHub, too large; run this notebook to create it) |
| `feature_scaler.pkl`, `target_scaler.pkl` | fitted scalers, used to turn predictions back into Wh |
| `feature_config.json` | the settings used and the final list of features |
| `feature_ranking.csv` | the RFE rank of every feature (1 = most important) |
| `feature_count_comparison.csv` | validation scores for different numbers of features |

### 3. `Train_ML.ipynb` and `Train_DL.ipynb`: train the models

Settings are in the `MLConfig` and `DLConfig` dataclasses. There are two ways to run:

**One model:**

```python
config = MLConfig(experiment_name="xgb_test", model_name="xgboost",
                  model_params={"max_depth": 3, "n_estimators": 300})
run_ml(config)
```

**Grid search:** every combination of the values in a JSON file is trained and logged.

```python
run_ml_grid("../models/ml.json", MLConfig())
run_dl_grid("../models/dl.json", DLConfig(epochs=20, patience=5))
```

Grid file format: model name → setting → list of values to try. `{}` means one run with the default settings.

```json
{
  "linear": {},
  "ridge": {"alpha": [0.1, 1.0, 10.0]},
  "xgboost": {"n_estimators": [300, 600], "max_depth": [3, 5], "learning_rate": [0.05]}
}
```

How the models are trained:

- **ML models** see one row of features per time step.
- **DL models** see a window of the last `sequence_length` rows (36 = 6 hours). Windows are cut on the fly by `WindowDataset`, so the 3D data is never stored. The first validation and test windows reach back into the end of the previous split (past data only), so ML and DL models are scored on exactly the same rows.
- DL training uses the **AdamW** optimizer, **MSE** loss, gradient clipping and **early stopping** on validation RMSE. The best epoch's weights are restored.
- A fixed seed (42) makes runs repeatable.
- `evaluate_test` stays `False` while experimenting, so the test set isn't used to make choices.

### 4. `Results.ipynb`: final scores and figures

Reads `models/training_runs/results.csv`, picks the best run of each model on validation, then loads each saved model and scores it **once on the test set**. Nothing is retrained. It produces every table and figure in the report and saves them to `reports/figures/`:

- validation and test MAE / RMSE for every model
- grid search values and best settings, dropout comparison, early-stopping epochs
- GRU before vs after tuning
- actual vs predicted over time, predicted vs actual scatter, residual distributions (every model)
- learning curves (DL: loss per epoch; tree models: error vs number of trees)
- train vs validation vs test RMSE, and XGBoost feature importance

Run it straight after training, without re-running `Feature_Engineer.ipynb` in between. The saved models must be scored on the same data they were trained on.

---

## Configuration reference

### `FeatureConfig` (`Feature_Engineer.ipynb`)

| Setting | Value used | What it controls |
|---|---|---|
| `lags` | `[1, 2, 3, 4, 5, 6, 18, 36, 144]` | past energy values to add (in 10-minute steps) |
| `rolling` | mean and std over `[3, 6, 18, 36, 144]` | rolling statistics of past energy use |
| `diffs` | `[1, 2, 3, 4, 5, 6, 18, 36, 144]` | how much energy use changed over each window |
| `time_features` | `["is_weekend", "is_holiday"]` | calendar flags |
| `cyclical_time` | `True` | hour and day as sin/cos |
| `interactions` | `(T1, RH_1)`, `(T2, RH_2)`, `(T6, RH_6)` | pairs to multiply |
| `sensor_aggregates` | `True` | indoor sensor averages, spreads and indoor–outdoor gap |
| `drop_columns` | `["rv1", "rv2"]` | columns to remove |
| `n_selected_features` | `30` | features kept by RFE (`0` = keep all) |
| `fixed_warmup` | `145` | first rows dropped |
| `train_ratio`, `val_ratio` | `0.7`, `0.15` | split sizes (test = the rest) |

### `RunConfig` (shared by `MLConfig` and `DLConfig`)

| Setting | Default | What it controls |
|---|---|---|
| `experiment_name` | `"baseline"` | name of the run folder and the row in `results.csv` |
| `notes` | `""` | free text saved with the run |
| `output_dir` | `"../models/training_runs"` | where runs are saved |
| `seed` | `42` | random seed |
| `evaluate_test` | `False` | also score the test set |

`MLConfig` adds `model_name` and `model_params`. `DLConfig` adds `model_name`, `model_params` (`hidden_size`, `num_layers`, `dropout`), `sequence_length`, `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `loss` (`mse` / `mae` / `huber`), `patience`, `monitor` and `grad_clip`.

Available models: `linear`, `ridge`, `lasso`, `elasticnet`, `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `xgboost`, `lightgbm`, `catboost` (ML) and `gru`, `lstm`, `cnn_lstm` (DL).

---

## Outputs

Every run is saved in `models/training_runs/`:

| File | What it is |
|---|---|
| `results.csv` | one row per run: train / validation / test scores, training time, every model setting and every feature setting |
| `<date_time>_<run name>/` | one folder per run |

Each run folder contains:

| File | What it is |
|---|---|
| `config.json` | the model settings |
| `feature_config.json` | the feature settings and feature list |
| `metrics.json` | MAE, RMSE, MAPE and R² for each split |
| `predictions_val.csv`, `predictions_test.csv` | actual and predicted values in Wh |
| `history.csv` | train and validation loss per epoch (DL only) |
| `model.pt` / `model.joblib` | the trained model |

---

## Code reference (`src/`)

| File | What it does |
|---|---|
| `feature_engineer.py` | runs all the data preparation steps in order |
| `preprocessing/cleaning.py` | IQR bounds and outlier capping |
| `preprocessing/features.py` | creates lag, rolling, difference, time, holiday, interaction and sensor summary features |
| `preprocessing/splitting.py` | chronological train / validation / test split |
| `preprocessing/scaling.py` | fits and applies the scalers (train only) |
| `preprocessing/selection.py` | RFE ranking and feature-count comparison |
| `preprocessing/data_io.py` | loads and saves data, scalers and settings |
| `ml_runner.py` | `run_ml` and `run_ml_grid` |
| `dl_runner.py` | `run_dl` and `run_dl_grid` |
| `training/ml_models.py` | the ML model list |
| `training/dl_models.py` | GRU, LSTM and CNN-LSTM networks |
| `training/windows.py` | cuts the past-reading windows for the DL models |
| `training/torch_loop.py` | DL training loop with early stopping |
| `training/grid.py` | reads a grid JSON file and makes every combination |
| `training/data.py` | loads the processed data |
| `training/metrics.py` | MAE, RMSE, MAPE and R², converted back to Wh |
| `training/results.py` | saves run folders and appends to `results.csv` |

---

## Known issues and notes

- **Large files are not on GitHub.** `processed.npz` and the ML models (`*.joblib`) are excluded because of their size. Run `Feature_Engineer.ipynb` and the training notebooks to recreate them before running `Results.ipynb`. The DL models (`model.pt`) are small and included.
- **LightGBM** is in the model list and `requirements.txt`, but it crashed (access violation) on the Windows machine used for this project, so it has no results. Its grid is kept separately in `models/ml_rest.json`.
- **Scores are measured on the capped target** (maximum 175 Wh), so they are lower than scores on the raw values would be.
