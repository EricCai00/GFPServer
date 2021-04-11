import torch
import torch.nn as nn


def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class Semilabel(nn.Module):
    def __init__(self, input_dim=135, hidden_dim=256, feature_dim=256, output_dim=20):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self.ln3 = nn.LayerNorm(2 * hidden_dim)

        self.hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.predict = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, arrays):
        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        hidden_states, _ = self.lstm(hidden_states.view(len(hidden_states), 1, -1))
        return hidden_states


class MLPReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.fc2 = nn.Linear(stacked_dim, 1)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, arrays):
        hidden1 = self.ln(torch.relu(self.fc1(arrays)))
        hidden2 = torch.relu(self.fc2(hidden1.transpose(0, 1)))
        output = self.output(hidden2.transpose(0, 1))
        return output
