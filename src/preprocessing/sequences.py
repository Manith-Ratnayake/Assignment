import numpy as np


def create_sequences(X, y, window):
    """Turn 2D features into 3D windows: each sample is the last `window` rows ending at row t, labelled y[t]."""
    if len(X) < window:
        raise ValueError("Sequence length exceeds split size.")

    X_seq = np.array([X[i - window + 1:i + 1] for i in range(window - 1, len(y))], dtype=np.float32)
    y_seq = y[window - 1:]
    return X_seq, y_seq


def create_split_sequences(scaled, window):
    """Apply create_sequences to every split and return {name: (X_seq, y_seq)}."""
    return {name: create_sequences(X, y, window) for name, (X, y) in scaled.items()}
