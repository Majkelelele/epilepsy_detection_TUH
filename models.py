from constants import GLOBAL_DATA
import numpy as np
from preprocess import get_paths
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from constants import EPS, NO_EPS
import time



class EEG_dataset(Dataset):
    def __init__(self, path_no_eps, path_eps):
        self.paths = path_eps + path_no_eps

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.tensor(np.load(self.paths[idx])["arr_0"], dtype=torch.float32)

        label = torch.tensor(NO_EPS if "no_eps" in self.paths[idx] else EPS, dtype=torch.float32)
        return data, label

class EEG_net(nn.Module):
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

        


if __name__ == "__main__":  
    batch_size = GLOBAL_DATA["batch_size"]

    paths_eps = get_paths("sliced_data/train/eps",".npz")
    paths_no_eps = get_paths("sliced_data/train/no_eps",".npz")[:len(paths_eps)]
    print(len(paths_eps))
    print(len(paths_no_eps))
    dataset = EEG_dataset(paths_no_eps, paths_eps)
    data_train, data_test = random_split(dataset, [0.8, 0.2])
    
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = EEG_net(len(GLOBAL_DATA['labels']), GLOBAL_DATA['resampled_freq'] * GLOBAL_DATA["slice_size_scs"])



    optimizer = AdamW(model.parameters())
    criterion = nn.BCELoss()
    epochs = 1
    
    for epoch in range(epochs):
        model.train()
        running_correct = 0
        start_time = time.time()
        for i, (data, labels) in enumerate(tqdm(train_loader,desc="progres")):
            model.zero_grad()
            out = model(data).squeeze(dim=-1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            predictions = (out > 0.5).float()
            predicted_corr = (predictions == labels).sum().item()
            
            running_correct += predicted_corr
        
        end_time = time.time()
        print(f"epoch time = {end_time - start_time}") 
        
        print(f"train accuracy = {running_correct/len(train_loader.dataset)}")
        
    model.eval()
    running_correct = 0
    running_loss = 0.0
    for i, (data, labels) in enumerate(tqdm(test_loader,desc="progres")):
        out = model(data).squeeze(dim=-1)
        loss = criterion(out, labels)
        running_loss += loss.item()
        predictions = (out > 0.5).float()
        predicted_corr = (predictions == labels).sum().item()
        running_correct += predicted_corr
    print(f"test accuracy = {running_correct/len(test_loader.dataset)}")
    print(f"loss = {running_loss}")

        


