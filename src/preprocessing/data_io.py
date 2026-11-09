from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_data(path):
    """Load the EDA parquet file, sorted by timestamp."""
    return pd.read_parquet(path).sort_index()


def save_sequences(sequences, output_dir):
    """Save every split's (X, y) arrays to output_dir/processed.npz and return the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for name, (X, y) in sequences.items():
        arrays[f"X_{name}"] = X
        arrays[f"y_{name}"] = y

    path = output_dir / "processed.npz"
    np.savez_compressed(path, **arrays)
    return path


def save_scalers(feature_scaler, target_scaler, output_dir):
    """Save the fitted scalers so predictions can be converted back to Wh later."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(feature_scaler, output_dir / "feature_scaler.pkl")
    joblib.dump(target_scaler, output_dir / "target_scaler.pkl")
