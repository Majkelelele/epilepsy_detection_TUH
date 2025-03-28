from constants import GLOBAL_DATA
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
import torch.nn.functional as F
from constants import EPS, NO_EPS



class EEG_dataset(Dataset):
    def __init__(self, path_no_eps, path_eps):
        self.paths = path_eps + path_no_eps

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.tensor(np.load(self.paths[idx])["arr_0"], dtype=torch.float32)

        label = torch.tensor(NO_EPS if "no_eps" in self.paths[idx] else EPS, dtype=torch.float32)
        return data, label

class Conv1d_lstm(nn.Module):
    def __init__(self, input_channels, seq_length, num_classes=1):
        super(EEG_net, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(128 * 2, num_classes) 
    def forward(self, x):
        batch_size, channels, seq_len = x.shape  

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Permute for LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM processing
        x, _ = self.lstm(x)

        # Take last timestep's features
        x = x[:, -1, :]  # (batch, hidden_size * 2)

        # Output layer
        x = self.fc(x)
        return torch.sigmoid(x)  # Sigmoid for binary classification

        




        


