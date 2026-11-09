import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, target="Appliances", include_interactions=False) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    # Forecast Appliances at time t using only information available before time t.
    features = df.drop(columns=[target]).shift(1)
    y = df[target]

    for lag in [1, 6, 144]:
        features[f"lag_{lag}"] = y.shift(lag)

    for window in [6, 144]:
        history = y.shift(1).rolling(window)
        features[f"rolling_mean_{window}"] = history.mean()
        features[f"rolling_std_{window}"] = history.std()

    features["hour"] = df.index.hour
    features["day_of_week"] = df.index.dayofweek
    features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    for lag in [1, 6, 144]:
        features[f"diff_{lag}"] = y.shift(1).diff(lag)

    if include_interactions:
        for i in [2, 1, 6]:
            features[f"T{i}_RH{i}"] = features[f"T{i}"] * features[f"RH_{i}"]

    features[target] = y
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features
