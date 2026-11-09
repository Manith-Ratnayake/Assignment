import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    mse = mean_squared_error(actual, predicted)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


@torch.no_grad()
def predict_neural(model, loader, device, target_scaler):
    model.eval()
    targets, predictions = [], []
    for X, y in loader:
        pred = model(X.to(device)).cpu().numpy()
        predictions.append(pred)
        targets.append(y.numpy())
    actual = np.concatenate(targets)
    predicted = np.concatenate(predictions)
    actual = target_scaler.inverse_transform(actual[:, None]).ravel()
    predicted = target_scaler.inverse_transform(predicted[:, None]).ravel()
    return actual, predicted
