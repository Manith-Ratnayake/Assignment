import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

LOSSES = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
}


def set_seed(seed):
    """Seed Python, NumPy and PyTorch so a run gives the same result when repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Use the GPU when one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_loss(name, huber_delta=1.0):
    """Return the loss function called `name`: "mse", "mae" or "huber"."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSSES)}.")
    return nn.HuberLoss(delta=huber_delta) if name == "huber" else LOSSES[name]()


def make_train_loader(dataset, batch_size, seed):
    """Shuffled mini-batches of the training windows; the seed fixes the shuffle order."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


@torch.no_grad()
def predict(model, dataset, batch_size, device):
    """Predict every sample of the dataset in order and return the (still scaled) predictions as a NumPy array."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return torch.cat([model(X.to(device)).cpu() for X, _ in loader]).numpy()


def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_clip=0):
    """One pass over the training data. Returns the average training loss. grad_clip=0 means no clipping."""
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


def fit(model, train_loader, val_set, y_val, loss_fn, optimizer, device, epochs, patience, monitor,
        score_val, grad_clip=0, batch_size=256):
    """Train with early stopping and restore the weights from the best validation epoch.

    score_val turns scaled validation predictions into a metrics dict in Wh
    (mae, rmse, ...). `monitor` names the key used to pick the best epoch;
    lower is better. Returns (history, best_epoch).
    """
    if monitor not in ("rmse", "mae", "mape", "loss"):
        raise ValueError("monitor must be 'rmse', 'mae', 'mape' or 'loss' (lower is better for all of them).")

    y_val_tensor = torch.from_numpy(y_val)
    best_score, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip)

        val_pred = predict(model, val_set, batch_size, device)
        val_metrics = score_val(val_pred)
        val_metrics["loss"] = loss_fn(torch.from_numpy(val_pred), y_val_tensor).item()
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        improved = val_metrics[monitor] < best_score
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_metrics['loss']:.4f}"
              f" | val MAE {val_metrics['mae']:.2f} | val RMSE {val_metrics['rmse']:.2f}" + (" *" if improved else ""))

        if improved:
            best_score, best_state, best_epoch, bad_epochs = val_metrics[monitor], deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    return history, best_epoch
