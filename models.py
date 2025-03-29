from constants import GLOBAL_DATA
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
from constants import EPS, NO_EPS, SEED
import torch.nn.functional as F
from dataset import EEG_dataset




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
        self.path_to_save = "saved_models/conv1d_lstm_15epochs"
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(128 * 2, num_classes) 
    
    def get_path_to_model(self):
        return self.path_to_save
    
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
    def __init__(self, num_classes=1):
        super(Conv2d_lstm, self).__init__()
        self.path_to_save = "saved_models/conv2d_lstm_15epochs"

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
    
    def get_path_to_model(self):
        return self.path_to_save
     
    def forward(self, x):
        # making one fake channel
        x = x.unsqueeze(dim = 1)
        x = self.seq1(x)
        x = self.seq2(x)
        x = x.view(x.shape[0],x.shape[1],-1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return F.sigmoid(x)
    

        



        


