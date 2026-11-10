import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

from training.metrics import compute_metrics, to_wh

COMPARE_SIZES = [5, 10, 15, 20, 30, 40]


def rank_features(X_train, y_train, columns, alpha=1.0):
    """Rank every feature with RFE (Recursive Feature Elimination) using a Ridge model.

    RFE trains the model, removes the weakest feature, and repeats until one is left.
    Rank 1 is the most important feature (the last one left).
    """
    rfe = RFE(Ridge(alpha=alpha), n_features_to_select=1, step=1)
    rfe.fit(X_train, y_train)
    return pd.Series(rfe.ranking_, index=columns, name="rank").sort_values()


def compare_feature_counts(splits, columns, ranking, target_scaler, sizes=COMPARE_SIZES, alpha=1.0):
    """Train Ridge on the top-k ranked features for several k and score each on the validation set (Wh)."""
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    actual = to_wh(y_val, target_scaler)

    rows = []
    for k in sorted({k for k in sizes if k < len(columns)} | {len(columns)}):
        positions = column_positions(columns, ranking.index[:k])
        model = Ridge(alpha=alpha).fit(X_train[:, positions], y_train)
        metrics = compute_metrics(actual, to_wh(model.predict(X_val[:, positions]), target_scaler))
        rows.append({"features": k, "val_mae": metrics["mae"], "val_rmse": metrics["rmse"]})
    return pd.DataFrame(rows).round(2)


def top_features(columns, ranking, n):
    """Return the n best-ranked features, in their original column order."""
    if n > len(columns):
        raise ValueError(f"n_selected_features is {n}, but there are only {len(columns)} features.")
    best = set(ranking.index[:n])
    return [column for column in columns if column in best]


def keep_columns(splits, columns, keep):
    """Return every split with only the `keep` columns."""
    positions = column_positions(columns, keep)
    return {name: (X[:, positions], y) for name, (X, y) in splits.items()}


def column_positions(columns, names):
    """Positions of `names` inside `columns`."""
    wanted = set(names)
    return [i for i, column in enumerate(columns) if column in wanted]