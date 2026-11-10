# Appliance Energy Prediction

Predicting household appliance energy use (Wh) from the UCI Appliances Energy Prediction dataset, using machine learning baselines and deep learning models (LSTM, GRU, CNN-LSTM).

## Project structure

```
Assignment/
├── data/energy_data_set.csv     # raw dataset
├── notebooks/
│   ├── EDA.ipynb                # data exploration
│   ├── Feature_Engineer.ipynb   # builds the model input data
│   ├── Train_ML.ipynb           # Linear Regression, XGBoost, CatBoost, ...
│   └── Train_DL.ipynb           # LSTM, GRU, CNN-LSTM
├── src/                         # code used by the notebooks
├── reports/                     # report and figures
├── requirements.txt
└── README.md
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) (install with `pip install uv`).

1. Create the environment with Python 3.12:
   ```bash
   uv venv --python 3.12
   ```

2. Install the packages:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Start Jupyter:
   ```bash
   uv run jupyter notebook
   ```

## How to run

Open the notebooks in the `notebooks/` folder and run them in this order:

1. **EDA.ipynb**: explores the data and saves `data/eda.parquet`
2. **Feature_Engineer.ipynb**: creates the features and saves the processed data
3. **Train_ML.ipynb** and/or **Train_DL.ipynb**: train the models

Every training run is saved in `notebooks/training_runs/`, and all results are collected in `notebooks/training_runs/results.csv`.