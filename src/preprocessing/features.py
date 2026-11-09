import numpy as np


def create_features(df, config):
    """Build the model input table: all features, then the target as the last column."""
    target = config["target"]
    y = df[target]

    features = shift_sensors(df, target)
    features["past_target"] = y.shift(1)

    features = add_lag_features(features, y, config["lags"])
    features = add_rolling_features(features, y, config["rolling"])
    features = add_diff_features(features, y, config["diffs"])
    features = add_time_features(features, df.index, config["time_features"])

    if config["cyclical_time"]:
        features = add_cyclical_features(features, df.index)

    if config["sensor_aggregates"]:
        features = add_sensor_aggregates(features)

    features = add_interactions(features, config["interactions"])

    features = features.drop(columns=config["drop_columns"])
    features[target] = y
    return features


def shift_sensors(df, target):
    """Return every non-target column shifted by one step (row t holds readings from t-1)."""
    return df.drop(columns=[target]).shift(1)


def add_lag_features(features, y, lags):
    """Add the target's value from `lag` steps ago, e.g. lag_6 = usage 1 hour ago."""
    for lag in lags:
        features[f"lag_{lag}"] = y.shift(lag)
    return features


def add_rolling_features(features, y, rolling):
    """Add rolling statistics of past usage, e.g. {"mean": [6, 18]} -> rolling_mean_6, rolling_mean_18.

    The target is shifted first so each window ends at t-1 and never includes row t.
    """
    past = y.shift(1)
    for statistic, windows in rolling.items():
        for window in windows:
            features[f"rolling_{statistic}_{window}"] = getattr(past.rolling(window), statistic)()
    return features


def add_diff_features(features, y, diffs):
    """Add how much past usage changed over `lag` steps (diff_1 = y[t-1] - y[t-2])."""
    past = y.shift(1)
    for lag in diffs:
        features[f"diff_{lag}"] = past.diff(lag)
    return features


def add_time_features(features, index, names):
    """Add the calendar features listed in `names` (hour, day_of_week, is_weekend, month)."""
    day = index.dayofweek
    available = {
        "hour": index.hour,
        "day_of_week": day,
        "is_weekend": (day >= 5).astype(int),
        "month": index.month,
    }

    for name in names:
        features[name] = available[name]
    return features


def add_cyclical_features(features, index):
    """Add sin/cos encodings of hour and weekday, so 23:00 sits next to 00:00 and Sunday next to Monday."""
    fractional_hour = index.hour + index.minute / 60
    day = index.dayofweek

    features["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24)
    features["day_sin"] = np.sin(2 * np.pi * day / 7)
    features["day_cos"] = np.cos(2 * np.pi * day / 7)
    return features


def add_sensor_aggregates(features):
    """Add the mean/spread of the 9 indoor temperature and humidity sensors, and the indoor-outdoor gap."""
    temp_cols = [f"T{i}" for i in range(1, 10)]
    humidity_cols = [f"RH_{i}" for i in range(1, 10)]

    features["temp_mean"] = features[temp_cols].mean(axis=1)
    features["temp_std"] = features[temp_cols].std(axis=1)
    features["humidity_mean"] = features[humidity_cols].mean(axis=1)
    features["humidity_std"] = features[humidity_cols].std(axis=1)
    features["indoor_outdoor_temp_gap"] = features["temp_mean"] - features["T_out"]
    return features


def add_interactions(features, pairs):
    """Add the product of each (col1, col2) pair, e.g. ("T_out", "RH_out") -> T_out_x_RH_out."""
    for col1, col2 in pairs:
        features[f"{col1}_x_{col2}"] = features[col1] * features[col2]
    return features


def drop_warmup_rows(df, n_rows):
    """Drop the first n_rows, where long lags and rolling windows don't have enough history yet."""
    return df.iloc[n_rows:].copy()


def check_no_missing(df):
    """Raise an error if any NaN or infinite value is left in the table."""
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("NaN or infinite values detected. Check the features or increase fixed_warmup.")
