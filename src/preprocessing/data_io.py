import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_data(path):
    """Load the EDA parquet file, sorted by timestamp."""
    return pd.read_parquet(path).sort_index()


def save_splits(splits, output_dir):
    """Save every split's scaled (X, y) arrays to output_dir/processed.npz and return the file path.

    X is the 2D table (rows x features). Windows for the sequence models are cut
    during training instead of being stored, which keeps the file small.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for name, (X, y) in splits.items():
        arrays[f"X_{name}"] = X
        arrays[f"y_{name}"] = y

    path = output_dir / "processed.npz"
    np.savez(path, **arrays)
    return path


def save_scalers(feature_scaler, target_scaler, output_dir):
    """Save the fitted scalers so predictions can be converted back to Wh later."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(feature_scaler, output_dir / "feature_scaler.pkl")
    joblib.dump(target_scaler, output_dir / "target_scaler.pkl")


def save_feature_selection(ranking, comparison, output_dir):
    """Save the RFE ranking and the feature-count comparison as CSV files for the report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking.to_csv(output_dir / "feature_ranking.csv", index_label="feature")
    comparison.to_csv(output_dir / "feature_count_comparison.csv", index=False)


def save_feature_config(config, feature_columns, output_dir):
    """Save the feature settings and final column names next to processed.npz.

    Training reads this file, so every result in results.csv records which features produced it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record = {**config, "n_features": len(feature_columns), "feature_columns": list(feature_columns)}
    (output_dir / "feature_config.json").write_text(json.dumps(record, indent=2))