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
from models import EEG_dataset, Conv1d_lstm
import time


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
    
    model = Conv1d_lstm(len(GLOBAL_DATA['labels']), GLOBAL_DATA['resampled_freq'] * GLOBAL_DATA["slice_size_scs"])



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
        print(f"epoch time = {end_time - start_time} seconds") 
        
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