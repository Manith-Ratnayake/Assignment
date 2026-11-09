import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def to_wh(values, target_scaler):
    """Convert scaled target values back to Wh."""
    return target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def compute_metrics(actual, predicted):
    """MAE, RMSE (Wh), MAPE (%) and R² between actual and predicted values in Wh."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_splits(predict_fn, splits, target_scaler):
    """Score predict_fn on every split.

    splits maps a split name to (inputs, scaled targets); inputs are the 2D table
    for ML models or a WindowDataset for DL models. predict_fn takes the inputs
    and returns scaled predictions. Returns
    ({split: metrics}, {split: (actual_wh, predicted_wh)}).
    """
    metrics, predictions = {}, {}
    for name, (X, y) in splits.items():
        actual = to_wh(y, target_scaler)
        predicted = to_wh(predict_fn(X), target_scaler)
        metrics[name] = compute_metrics(actual, predicted)
        predictions[name] = (actual, predicted)
    return metrics, predictions
