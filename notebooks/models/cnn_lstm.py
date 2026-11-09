import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, cnn_filters=64):
        super().__init__()
        self.conv = nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        prediction = self.fc(output[:, -1, :])
        return prediction.squeeze(-1)