from constants import GLOBAL_DATA
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
from constants import EPS, NO_EPS, SEED
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from dataset import EEG_dataset
from preprocess import make_spectogram





def save_model(model:nn.Module, optimizer:torch.optim.Optimizer, path_to_save:str) -> None:
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, path_to_save)
    
def load_model_to_finetune(model:nn.Module, optimizer:torch.optim, model_path:str) -> tuple[nn.Module, torch.optim.Optimizer]:
    checkpoint = torch.load(model_path)

    # Restore model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer
    



class Conv1d_lstm(nn.Module):
    def __init__(self, input_channels:int, num_classes=1) -> None:
        super(Conv1d_lstm, self).__init__()
        self.name = "conv1d_lstm"
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(128 * 2, num_classes) 
    
    def get_model_name(self):
        return self.name
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, channels, seq_len = x.shape  
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Permute for LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = x[:, -1, :]  

        x = self.fc(x)
        return torch.sigmoid(x) 
    
class Conv2d_lstm(nn.Module):
    def __init__(self, num_classes=1, transform=""):
        super(Conv2d_lstm, self).__init__()
        self.name = "conv2d_lstm"
        self.transform = transform

        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,51), stride=(1,4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        )
        self.seq2 = nn.Sequential( 
            nn.Conv2d(64, 128, kernel_size=(1,21), stride=(1,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # self.seq3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(1,9), stride=(1,2)),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        # )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256,num_classes)
    
    def get_model_name(self):
        return self.name
     
    def forward(self, x):
        # making one fake channel
        if self.transform == "stft":
            x = make_spectogram(x)
        
        x = x.unsqueeze(dim = 1)
        x = self.seq1(x)
        x = self.seq2(x)
        x = x.view(x.shape[0],x.shape[1],-1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return F.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x) 
        scale = self.fc(scale)  
        return x * scale

class ResNet_lstm(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, stride=1):
        super().__init__()
        self.name = "resnet_lstm"

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

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2, num_classes)  # *2 for bidirectional
        self.out = nn.Sigmoid()

    def get_model_name(self):
        return self.name

    def forward(self, x):
        # x: (B, Channels, Time) => Add a dummy spatial dim

        x = x.unsqueeze(1)  # (B, 1, Channels, Time)

        x = self.seq1(x) + self.res1(x)
        x = F.relu(x)

        x = self.seq2(x) + self.res2(x)
        x = F.relu(x)

        # Collapse spatial dim, keep time axis
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))  # (B, C, 1, T)
        x = x.squeeze(2)  # (B, C, T)

        x = self.se_block(x)

        # Prepare for LSTM: (B, T, C)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last time step

        x = self.dropout(x)
        x = self.fc(x)
        return self.out(x)



class EEG_Conv2d_LSTM(nn.Module):
    def __init__(self, input_freq_bins, input_time_bins, num_classes=1, transform=""):
        super(EEG_Conv2d_LSTM, self).__init__()
        self.name = "conv2d_lstm"
        self.transform = transform

        # CNN layers: Process multi-channel EEG signals (19 channels)
        self.conv1 = nn.Conv2d(19, 64, kernel_size=(3, 3), padding=(1, 1))  # Apply convolution across time and frequency
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Additional Conv2D layer (optional) to enhance feature extraction
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # Reshape for LSTM: Flatten the spatial dimensions (time, freq) into one feature dimension
        self.lstm = nn.LSTM(input_size=663, hidden_size=256, num_layers=2, batch_first=True)

        # Final fully connected layer to output the class prediction
        self.fc = nn.Linear(256, num_classes)

    def get_model_name(self):
        return self.name

    def forward(self, x):
        # x: [batch, 19, time_bins, freq_bins]
        res = []
        if self.transform == "stft":
            for i in range(GLOBAL_DATA["batch_size"]):
                res.append(make_spectogram(x[i,:,:]))
        x = torch.stack(res)
         
        # Apply the first CNN layer
        x = self.conv1(x)  # [batch, 64, time_bins, freq_bins]
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply the second CNN layer
        x = self.conv2(x)  # [batch, 128, time_bins, freq_bins]
        x = self.bn2(x)
        x = self.relu2(x)

        # Flatten the spatial dimensions (time_bins * freq_bins)
        x = x.view(x.size(0), x.size(2) * x.size(3), -1)  # [batch, time, feature_dim]

        # Apply LSTM to process the sequence
        x = x.permute(0, 2, 1)  # [batch, feature_dim, time]
        x, _ = self.lstm(x)     # [batch, time, hidden_size]

        # Take the output from the last time step (final time-step representation)
        x = x[:, -1, :]  # [batch, hidden_size]

        # Pass through fully connected layer
        x = self.fc(x)  # [batch, num_classes]

        return torch.sigmoid(x)


class ensemble_models(nn.Module):
    def __init__(self, list_of_models):
        super().__init__()
        self.list_of_models = nn.ModuleList(list_of_models)
        self.name = "ensemble_models"

    def get_model_name(self):
        return self.name

    def forward(self, x):
        predictions = []
        for model in self.list_of_models:
            out = model(x).squeeze(dim=-1)
            pred = (out > 0.5).float()
            predictions.append(pred)

        # Stack predictions: shape => (num_models, batch_size)
        stacked_preds = torch.stack(predictions)  

        # Majority vote: sum across models, then check if majority voted 1
        votes = stacked_preds.sum(dim=0)
        majority = (votes > (len(self.list_of_models) / 2)).float()

        return majority.unsqueeze(dim=-1)

        
            