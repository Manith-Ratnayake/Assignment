from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


def load_data(path: str, target="Appliances") -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Missing {file}. Run from the Assignment project root.")
    df = pd.read_csv(file, parse_dates=["date"]).sort_values("date").set_index("date")
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected; resolve them before training.")
    intervals = df.index.to_series().diff().dropna()
    if not intervals.eq(pd.Timedelta(minutes=10)).all():
        raise ValueError("Dataset is not consistently spaced at 10-minute intervals.")
    return df


def prepare_data(df: pd.DataFrame, target="Appliances", train_ratio=0.70, val_ratio=0.15):
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Ratios must leave nonempty train, validation, and test splits.")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if train_end < 2 or val_end <= train_end or val_end >= n:
        raise ValueError("Insufficient rows for the requested split ratios.")

    cols = df.columns.drop(target).tolist()
    X_raw = df[cols].to_numpy(dtype=np.float64)
    y_raw = df[target].to_numpy(dtype=np.float64)
    feature_scaler = StandardScaler().fit(X_raw[:train_end])
    target_scaler = StandardScaler().fit(y_raw[:train_end, None])
    X = feature_scaler.transform(X_raw).astype(np.float32)
    y_scaled = target_scaler.transform(y_raw[:, None]).ravel().astype(np.float32)

    return {
        "X": X, "y": y_raw, "y_scaled": y_scaled,
        "train_end": train_end, "val_end": val_end,
        "feature_columns": cols, "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


class SequenceDataset(Dataset):
    def __init__(self, X, y, start, end, window):
        if window < 1 or start < window - 1 or end > len(X) or start >= end:
            raise ValueError("Invalid sequence window or target index range.")
        self.X = X
        self.y = y
        self.start = start
        self.end = end
        self.window = window

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        t = self.start + index
        return torch.from_numpy(self.X[t - self.window + 1:t + 1]), torch.tensor(self.y[t], dtype=torch.float32)
