import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout1d(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.dropout(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :]).squeeze(-1)



# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
#         super().__init__()

#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 32),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(32, 1)
#         )

#     def forward(self, x):
#         output, _ = self.lstm(x)
#         x = output[:, -1, :]
#         prediction = self.mlp(x)
#         return prediction.squeeze(-1)


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         output, _ = self.lstm(x)
#         prediction = self.fc(output[:, -1, :])
#         return prediction.squeeze(-1)