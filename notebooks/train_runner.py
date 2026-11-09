import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import random
import joblib


MODEL_CLASSES = {"lstm": ("lstm.py", "LSTMModel"), "gru": ("gru.py", "GRUModel"), "cnn_lstm": ("cnn_lstm.py", "CNNLSTMModel")}


def load_data(config):
    path = Path(config["data_path"])
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path.resolve()}. Run feature engineering first.")

    with np.load(path) as data:
        names = ["train", "val"] + (["test"] if config["evaluate_test"] else [])
        arrays = {name: (np.asarray(data[f"X_{name}"], dtype=np.float32), np.asarray(data[f"y_{name}"], dtype=np.float32)) for name in names}

    scaler = joblib.load(config["target_scaler_path"])
    if not hasattr(scaler, "scale_") or len(scaler.scale_) != 1:
        raise ValueError("Expected a fitted StandardScaler for the target variable.")
    target_std = float(scaler.scale_[0])

    for name, (X, y) in arrays.items():
        if X.ndim != 3 or y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"Invalid {name} shapes: X={X.shape}, y={y.shape}")
        print(f"{name.capitalize():5s} | X: {X.shape} | y: {y.shape}", flush=True)

    return arrays, target_std


def create_loaders(arrays, batch_size):
    loaders = {}
    for name, (X, y) in arrays.items():
        # from_numpy avoids copying large sequence arrays into another CPU tensor.
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))
    return loaders


def create_model(config, input_size, device):
    model_name = config["model_name"].lower()
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_CLASSES)}")

    file_name, class_name = MODEL_CLASSES[model_name]
    model_path = (Path(__file__).resolve().parent / config["models_dir"] / file_name).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location(f"training_{model_name}", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, class_name)
    model = model_cls(input_size=input_size, **config["model_params"]).to(device)
    return model


def create_loss(config):
    name = config["loss"].lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=config["huber_delta"])
    raise ValueError("Loss must be 'mse' or 'huber'.")


def run_epoch(model, loader, criterion, device, target_std, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_sq = total_abs = 0.0
    count = len(loader.dataset)

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            predictions = model(X)
            loss = criterion(predictions, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        error = (predictions.detach() - y).float()
        total_loss += loss.detach().item() * len(y)
        total_sq += error.square().sum().item()
        total_abs += error.abs().sum().item()

    return {
        "loss": total_loss / count,
        "mse": total_sq / count * target_std ** 2,
        "rmse": (total_sq / count) ** 0.5 * target_std,
        "mae": total_abs / count * target_std,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu" if config["device"] == "auto" else config["device"])
    arrays, target_std = load_data(config)
    loaders = create_loaders(arrays, config["batch_size"])
    model = create_model(config, arrays["train"][0].shape[2], device)
    criterion = create_loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = config["experiment_name"]
    print(f"\nExperiment: {name} | Model: {config['model_name']} | Device: {device} | Loss: {config['loss']}", flush=True)
    print(model, flush=True)

    best_score = float("inf")
    best_state = None
    best_epoch = 0
    best_val = None
    bad_epochs = 0
    history = []
    metric = config["monitor"]
    if metric not in ("mae", "rmse"):
        raise ValueError("monitor must be 'mae' or 'rmse'.")

    for epoch in range(1, config["epochs"] + 1):
        train = run_epoch(model, loaders["train"], criterion, device, target_std, optimizer)
        val = run_epoch(model, loaders["val"], criterion, device, target_std)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()}})

        print(
            f"Epoch {epoch:02d}/{config['epochs']} | Train {config['loss']}: {train['loss']:.4f} | Val {config['loss']}: {val['loss']:.4f}"
            f" | Train MAE: {train['mae']:.2f} | Val MAE: {val['mae']:.2f}"
            f" | Train RMSE: {train['rmse']:.2f} | Val RMSE: {val['rmse']:.2f}"
            f" | Val MSE: {val['mse']:.2f}", flush=True
        )

        if val[metric] < best_score:
            best_score = val[metric]
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_val = val.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if config["patience"] is not None and bad_epochs >= config["patience"]:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    checkpoint = output_dir / f"{name}_best.pt"
    torch.save({"state_dict": best_state, "model_name": config["model_name"], "model_params": config["model_params"], "input_size": arrays["train"][0].shape[2], "best_epoch": best_epoch}, checkpoint)
    pd.DataFrame(history).to_csv(output_dir / f"{name}_history.csv", index=False)

    results = {"experiment": name, "best_epoch": best_epoch, "validation": best_val}
    if config["evaluate_test"]:
        results["test"] = run_epoch(model, loaders["test"], criterion, device, target_std)
    (output_dir / f"{name}_results.json").write_text(json.dumps({"config": config, "results": results}, indent=2), encoding="utf-8")

    print(f"\nBest epoch: {best_epoch} | Val MAE: {best_val['mae']:.2f} Wh | Val RMSE: {best_val['rmse']:.2f} Wh | Val MSE: {best_val['mse']:.2f} Wh²", flush=True)
    print(f"Saved best model and history to: {output_dir.resolve()}", flush=True)
    return results
