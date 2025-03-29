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

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(128 * 2, num_classes) 
        
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
    
# class 

        



        


