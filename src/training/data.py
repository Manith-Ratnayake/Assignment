import json
from pathlib import Path

import joblib
import numpy as np


def load_processed_data(data_dir, include_test=False):
    """Load what feature_engineer saved.

    Returns ({split: (X, y)}, target_scaler, feature_config). X is the 2D table
    (rows x features) and y is still scaled. Splits come back in time order
    (train, val, test); the test split is only loaded when include_test is True.
    """
    data_dir = Path(data_dir)
    names = ["train", "val"] + (["test"] if include_test else [])

    with np.load(data_dir / "processed.npz") as arrays:
        splits = {name: (arrays[f"X_{name}"], arrays[f"y_{name}"]) for name in names}

    target_scaler = joblib.load(data_dir / "target_scaler.pkl")
    feature_config = load_feature_config(data_dir)
    return splits, target_scaler, feature_config


def load_feature_config(data_dir):
    """Read feature_config.json, the record of which feature settings built processed.npz."""
    path = Path(data_dir) / "feature_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run feature engineering again to create it.")
    return json.loads(path.read_text())
