from constants import GLOBAL_DATA
import numpy as np
from preprocess import get_paths
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from constants import EPS, NO_EPS, SEED
from models import EEG_dataset, Conv1d_lstm, save_model, load_model_to_finetune, Conv2d_lstm
import time


def train(model, optimizer, train_loader, criterion=nn.BCELoss(), epochs=5, save=True):
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
        print(f"{epoch} epoch time = {end_time - start_time} seconds") 
        print(f"{epoch} epoch train accuracy = {running_correct/len(train_loader.dataset)}")
    if save:
        save_model(model, optimizer, model.get_path_to_model())
    
def test(model, test_loader, criterion=nn.BCELoss()):
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
    
def prepare_loaders():
    torch.manual_seed(SEED)
    paths_eps = get_paths(GLOBAL_DATA["epilepsy_path"],".npz")
    paths_no_eps = get_paths(GLOBAL_DATA["no_epilepsy_path"],".npz")[:len(paths_eps)]
    print(f"epilepsy files count = {len(paths_eps)}")
    print(f" no epilepsy files count = {len(paths_no_eps)}")
    dataset = EEG_dataset(paths_no_eps, paths_eps)
    data_train, data_test = random_split(dataset, [0.8, 0.2])
     
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader

def test_model_from_memory(model, optim, path_to_read_model):
    _, test_loader = prepare_loaders()
    
    model_1, _ = load_model_to_finetune(model, optim, path_to_read_model)
    test(model_1, test_loader)


if __name__ == "__main__":  
    torch.manual_seed(SEED)
    batch_size = GLOBAL_DATA["batch_size"]

    
    # model = Conv1d_lstm(len(GLOBAL_DATA['labels']))
    model = Conv2d_lstm()

    train_loader, test_loader = prepare_loaders()

    optimizer = AdamW(model.parameters())
    criterion = nn.BCELoss()
    epochs = GLOBAL_DATA["epochs"]

    
    train(model,optimizer,train_loader, epochs=epochs)
    test(model,test_loader)
    test_model_from_memory(model, optimizer, model.get_path_to_model())

    
    
    