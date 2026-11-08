import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.gelu = nn.GeLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        prediction = self.fc2(x)
        return prediction.squeeze(-1)


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         output, _ = self.lstm(x)
#         prediction = self.fc(output[:, -1, :])
#         return prediction.squeeze(-1)