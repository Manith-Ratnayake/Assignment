import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def get_scaler(name):
    """Return a new, unfitted scaler by name: "standard", "robust" or "minmax"."""
    if name not in SCALERS:
        raise ValueError(f"Unknown scaler '{name}'. Choose from {list(SCALERS)}.")
    return SCALERS[name]()


def separate_target(df, target):
    """Split a table into its feature columns (X) and its target column (y)."""
    return df.drop(columns=[target]), df[[target]]


def fit_scalers(train_df, target, feature_scaler_name, target_scaler_name="standard"):
    """Fit one scaler on the training features and one on the training target."""
    X, y = separate_target(train_df, target)
    feature_scaler = get_scaler(feature_scaler_name).fit(X)
    target_scaler = get_scaler(target_scaler_name).fit(y)
    return feature_scaler, target_scaler


def apply_scalers(splits, target, feature_scaler, target_scaler):
    """Scale every split with the already-fitted scalers and return {name: (X, y)} as float32 arrays."""
    scaled = {}
    for name, df in splits.items():
        X, y = separate_target(df, target)
        X_scaled = feature_scaler.transform(X).astype(np.float32)
        y_scaled = target_scaler.transform(y).ravel().astype(np.float32)
        scaled[name] = (X_scaled, y_scaled)
    return scaled
