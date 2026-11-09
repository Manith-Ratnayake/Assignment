import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.linear_models import MODELS as LINEAR_MODELS
from models.tree_models import MODELS as TREE_MODELS
from models.ensemble_models import MODELS as ENSEMBLE_MODELS, create_ensemble
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.cnn_lstm import CNNLSTMModel
from src.config import TrainConfig
from src.evaluate import predict_neural, regression_metrics
from src.feature_engineering import build_features
from src.preprocessing import SequenceDataset, load_data, prepare_data

DEEP_MODELS = {"lstm": LSTMModel, "gru": GRUModel, "cnn_lstm": CNNLSTMModel}
CLASSICAL_MODELS = {**LINEAR_MODELS, **TREE_MODELS, **ENSEMBLE_MODELS}
EXTRA_ENSEMBLES = {"xgboost", "lightgbm", "catboost"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_classical(config, data, output):
    name = config.model_name
    if name in CLASSICAL_MODELS:
        model = CLASSICAL_MODELS[name](**config.model_params)
    else:
        model = create_ensemble(name, config.model_params)

    X, y = data["X"], data["y"]
    n, m = data["train_end"], data["val_end"]
    model.fit(X[:n], y[:n])
    results = {"model": name, "validation": regression_metrics(y[n:m], model.predict(X[n:m]))}
    if config.evaluate_test:
        results["test"] = regression_metrics(y[m:], model.predict(X[m:]))

    joblib.dump({"model": model, "scaler": data["feature_scaler"], "feature_columns": data["feature_columns"]}, output / "checkpoints" / f"{name}.joblib")
    return results


def train_deep(config, data, output):
    X, y = data["X"], data["y_scaled"]
    n, m = data["train_end"], data["val_end"]
    window = config.sequence_length
    if window > n:
        raise ValueError(f"Sequence length {window} exceeds training samples {n}.")

    train_set = SequenceDataset(X, y, window - 1, n, window)
    val_set = SequenceDataset(X, y, n, m, window)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DEEP_MODELS[config.model_name](input_size=X.shape[1], **config.model_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    best_loss, bad_epochs = float("inf"), 0
    checkpoint = output / "checkpoints" / f"{config.model_name}_best.pt"

    for epoch in range(config.epochs):
        model.train()
        train_sum = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(batch_y)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                val_sum += criterion(model(batch_X), batch_y).item() * len(batch_y)
        train_loss = train_sum / len(train_set)
        val_loss = val_sum / len(val_set)
        val_rmse = np.sqrt(val_loss) * data["target_scaler"].scale_[0]
        print(f"Epoch {epoch + 1:02d} | Train MSE (scaled): {train_loss:.4f} | Val RMSE: {val_rmse:.2f}")

        if val_loss < best_loss - 1e-6:
            best_loss, bad_epochs = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "feature_columns": data["feature_columns"]}, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    actual, predicted = predict_neural(model, val_loader, device, data["target_scaler"])
    results = {"model": config.model_name, "validation": regression_metrics(actual, predicted)}
    if config.evaluate_test:
        test_set = SequenceDataset(X, y, m, len(X), window)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
        actual, predicted = predict_neural(model, test_loader, device, data["target_scaler"])
        results["test"] = regression_metrics(actual, predicted)
    return results


def run_experiment(config: TrainConfig):
    if config.model_name not in DEEP_MODELS and config.model_name not in CLASSICAL_MODELS and config.model_name not in EXTRA_ENSEMBLES:
        raise ValueError(f"Unknown model: {config.model_name}")
    if config.patience < 1 or config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")

    set_seed(config.seed)
    output = Path(config.output_dir)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "scalers").mkdir(parents=True, exist_ok=True)

    df = load_data(config.data_path, config.target)
    features = build_features(df, config.target, config.include_interactions)
    data = prepare_data(features, config.target, config.train_ratio, config.val_ratio)
    joblib.dump(data["feature_scaler"], output / "scalers" / "feature_scaler.pkl")
    joblib.dump(data["target_scaler"], output / "scalers" / "target_scaler.pkl")

    if config.model_name in DEEP_MODELS:
        results = train_deep(config, data, output)
    else:
        results = train_classical(config, data, output)

    (output / f"{config.model_name}_metrics.json").write_text(json.dumps({"config": asdict(config), "results": results}, indent=2))
    print(json.dumps(results, indent=2))
    return results
