from torch import nn


class LSTMModel(nn.Module):
    """LSTM over the whole window; the output at the last time step predicts the next value.

    Spatial dropout (Dropout1d) drops whole feature channels across every time step,
    which regularises better than dropping single values the LSTM can fill in from neighbours.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout1d(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.dropout(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class GRUModel(nn.Module):
    """Same structure as LSTMModel with a GRU cell, which has fewer parameters for the same size."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout1d(dropout)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.dropout(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.gru(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class CNNLSTMModel(nn.Module):
    """A 1D convolution finds short local patterns, then an LSTM models the longer sequence."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, cnn_filters=64, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout)
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.dropout(self.relu(self.conv(x.transpose(1, 2)))).transpose(1, 2)
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


DL_MODELS = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cnn_lstm": CNNLSTMModel,
}


def build_dl_model(name, input_size, params):
    """Create the network called `name` for `input_size` features; params override the class defaults."""
    if name not in DL_MODELS:
        raise ValueError(f"Unknown DL model '{name}'. Choose from {list(DL_MODELS)}.")
    return DL_MODELS[name](input_size=input_size, **params)
