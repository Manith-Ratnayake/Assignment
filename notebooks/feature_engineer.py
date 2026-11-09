import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def load_data(config):
    df = pd.read_parquet(config["data_path"]).sort_index()
    return df


def build_features(df, config):
    target = config["target"]
    y = df[target]

    features = df.drop(columns=[target]).shift(1).copy()
    features["past_target"] = y.shift(1)

    # Lag features
    for lag in config["lags"]:
        features[f"lag_{lag}"] = y.shift(lag)

    # Rolling statistics
    for statistic, windows in config["rolling"].items():
        for window in windows:
            history = y.shift(1).rolling(window)
            features[f"rolling_{statistic}_{window}"] = getattr(history, statistic)()

    # Difference features
    for lag in config["diffs"]:
        features[f"diff_{lag}"] = y.shift(1).diff(lag)

    # Time features
    hour = df.index.hour
    day = df.index.dayofweek

    time_values = {
        "hour": hour,
        "day_of_week": day,
        "is_weekend": (day >= 5).astype(int),
        "month": df.index.month
    }

    for name in config["time_features"]:
        features[name] = time_values[name]

    # Cyclical features
    if config["cyclical_time"]:
        fractional_hour = df.index.hour + df.index.minute / 60

        features["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24)
        features["day_sin"] = np.sin(2 * np.pi * day / 7)
        features["day_cos"] = np.cos(2 * np.pi * day / 7)

    # Environmental aggregates
    if config["sensor_aggregates"]:
        temp_cols = [f"T{i}" for i in range(1, 10)]
        humidity_cols = [f"RH_{i}" for i in range(1, 10)]

        features["temp_mean"] = features[temp_cols].mean(axis=1)
        features["temp_std"] = features[temp_cols].std(axis=1)
        features["humidity_mean"] = features[humidity_cols].mean(axis=1)
        features["humidity_std"] = features[humidity_cols].std(axis=1)
        features["indoor_outdoor_temp_gap"] = features["temp_mean"] - features["T_out"]

    # Interaction features
    for col1, col2 in config["interactions"]:
        features[f"{col1}_x_{col2}"] = features[col1] * features[col2]

    features = features.drop(columns=config["drop_columns"])
    features[target] = y

    return features


def prepare_features(df, config):
    df = build_features(df, config)
    df = df.iloc[config["fixed_warmup"]:].copy()

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("NaN or infinite values detected. Check features or fixed_warmup.")

    print(f"Engineered shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")

    return df


def split_data(df, config):
    n = len(df)
    train_end = int(n * config["train_ratio"])
    val_end = int(n * (config["train_ratio"] + config["val_ratio"]))

    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:]
    }

    for name, split in splits.items():
        print(f"{name.capitalize()}: {len(split)}")

    return splits


def scale_data(splits, config):
    scalers = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler
    }

    target = config["target"]
    feature_scaler = scalers[config["feature_scaler"]]()
    target_scaler = StandardScaler()

    train = splits["train"]

    feature_scaler.fit(train.drop(columns=[target]))
    target_scaler.fit(train[[target]])

    scaled = {}

    for name, df in splits.items():
        X = feature_scaler.transform(df.drop(columns=[target])).astype(np.float32)
        y = target_scaler.transform(df[[target]]).ravel().astype(np.float32)

        scaled[name] = (X, y)

    return scaled, feature_scaler, target_scaler


def create_sequences(X, y, window):
    if len(X) < window:
        raise ValueError("Sequence length exceeds split size.")

    X_seq = np.array([X[i-window+1:i+1] for i in range(window-1, len(y))], dtype=np.float32)
    y_seq = y[window-1:]

    return X_seq, y_seq


def prepare_sequences(scaled, config):
    window = config["sequence_length"]
    sequences = {}

    for name, (X, y) in scaled.items():
        X_seq, y_seq = create_sequences(X, y, window)
        sequences[name] = (X_seq, y_seq)

        print(f"{name.capitalize()} X: {X_seq.shape} | y: {y_seq.shape}")

    return sequences


def save_data(sequences, feature_scaler, target_scaler, config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}

    for name, (X, y) in sequences.items():
        arrays[f"X_{name}"] = X
        arrays[f"y_{name}"] = y

    np.savez_compressed(output_dir / "processed.npz", **arrays)

    joblib.dump(feature_scaler, output_dir / "feature_scaler.pkl")
    joblib.dump(target_scaler, output_dir / "target_scaler.pkl")

    print("Dataset and scalers saved successfully.")


def main(config):
    df = load_data(config)
    df = prepare_features(df, config)
    splits = split_data(df, config)
    scaled, feature_scaler, target_scaler = scale_data(splits, config)
    sequences = prepare_sequences(scaled, config)
    save_data(sequences, feature_scaler, target_scaler, config)