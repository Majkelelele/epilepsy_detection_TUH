from typing import Tuple, List
from constants import GLOBAL_DATA
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
from constants import EPS, NO_EPS, SEED
import torch.nn.functional as F
from dataset import EEG_dataset
from preprocess import make_spectogram


def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, path_to_save: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path_to_save)


def load_model_to_finetune(model: nn.Module, optimizer: torch.optim.Optimizer, model_path: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


class Conv1d_lstm(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        self.name: str = "conv1d_lstm"
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def get_model_name(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return torch.sigmoid(x)


class Conv2d_lstm(nn.Module):
    def __init__(self, num_classes: int = 1, transform: str = "") -> None:
        super().__init__()
        self.name: str = "conv2d_lstm"
        self.transform: str = transform

        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 51), stride=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 21), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def get_model_name(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform == "stft":
            x = make_spectogram(x)
        x = x.unsqueeze(dim=1)
        x = self.seq1(x)
        x = self.seq2(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return torch.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class ResNet_lstm(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 1, stride: int = 1) -> None:
        super().__init__()
        self.name: str = "resnet_lstm"

        self.seq1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(1, 9), stride=(1, stride), padding=(0, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 9), stride=(1, stride), padding=(0, 4), bias=False),
            nn.BatchNorm2d(32)
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(1, 9), stride=(1, stride), padding=(0, 4), bias=False),
            nn.BatchNorm2d(32)
        )
        self.seq2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 9), stride=(1, stride), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 9), stride=(1, stride), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 9), stride=(1, stride), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64)
        )

        self.se_block = SEBlock(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2, num_classes)
        self.out = nn.Sigmoid()

    def get_model_name(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.seq1(x) + self.res1(x)
        x = F.relu(x)
        x = self.seq2(x) + self.res2(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))
        x = x.squeeze(2)
        x = self.se_block(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return self.out(x)


class EEG_Conv2d_LSTM(nn.Module):
    def __init__(self, input_freq_bins: int, input_time_bins: int, num_classes: int = 1, transform: str = "") -> None:
        super().__init__()
        self.name: str = "conv2d_lstm"
        self.transform: str = transform

        self.conv1 = nn.Conv2d(19, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.lstm = nn.LSTM(input_size=663, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def get_model_name(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: List[torch.Tensor] = []
        if self.transform == "stft":
            for i in range(GLOBAL_DATA["batch_size"]):
                res.append(make_spectogram(x[i, :, :]))
            x = torch.stack(res)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.view(x.size(0), x.size(2) * x.size(3), -1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return torch.sigmoid(x)


class ensemble_models(nn.Module):
    def __init__(self, list_of_models: List[nn.Module]) -> None:
        super().__init__()
        self.list_of_models = nn.ModuleList(list_of_models)
        self.name: str = "ensemble_models"

    def get_model_name(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions: List[torch.Tensor] = []
        for model in self.list_of_models:
            out = model(x).squeeze(dim=-1)
            pred = (out > 0.5).float()
            predictions.append(pred)

        stacked_preds = torch.stack(predictions)
        votes = stacked_preds.sum(dim=0)
        majority = (votes > (len(self.list_of_models) / 2)).float()
        return majority.unsqueeze(dim=-1)
