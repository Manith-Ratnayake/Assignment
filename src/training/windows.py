import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Serves (window, target) pairs cut on the fly from a 2D feature table.

    Sample i is the `window` rows ending at row targets[i], labelled y[targets[i]].
    Only the 2D table is kept in memory; each window is a slice of it, so the
    3D windows never have to be stored.
    """

    def __init__(self, X, y, targets, window):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.targets = targets
        self.window = window

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        t = int(self.targets[i])
        return self.X[t - self.window + 1:t + 1], self.y[t]


def make_window_datasets(splits, window):
    """Build one WindowDataset per split from the 2D splits (given in time order).

    The splits are joined back into one table, so the first windows of val and
    test reach back into the end of the previous split. That is past data, so
    nothing leaks, and every val/test row gets a prediction: the same rows the
    ML models are scored on. Only the first window - 1 training rows are skipped.

    Returns {split: (dataset, y)} where y holds the scaled targets of that dataset.
    """
    if window > len(splits["train"][0]):
        raise ValueError(f"sequence_length {window} is longer than the training split.")

    X = np.concatenate([X for X, _ in splits.values()])
    y = np.concatenate([y for _, y in splits.values()])

    datasets, start = {}, 0
    for name, (X_split, _) in splits.items():
        end = start + len(X_split)
        targets = np.arange(max(start, window - 1), end)
        datasets[name] = (WindowDataset(X, y, targets, window), y[targets])
        start = end
    return datasets
