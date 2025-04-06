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
from models import EEG_dataset, Conv1d_lstm, save_model, load_model_to_finetune, Conv2d_lstm, ResNet_lstm
import time
import torch.multiprocessing as mp


# torch.set_num_threads(GLOBAL_DATA["cores_count"] - 1)

def make_path(model, epochs):
    return GLOBAL_DATA["saving_models_path"] + model.get_model_name() + "_" + str(epochs) + "epochs"

def train(model, optimizer, train_loader, criterion=nn.BCELoss(), epochs=5, save=True):
    for epoch in range(epochs):
        model.train()
        running_correct = 0
        start_time = time.time()
        loading_start = time.time()
        for i, (data, labels) in enumerate(tqdm(train_loader,desc="progres")):
            loading_end = time.time()
            # print(f"loading time = {loading_end-loading_start} seconds")
            training_start = time.time()
            model.zero_grad()
            out = model(data).squeeze(dim=-1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            predictions = (out > 0.5).float()
            predicted_corr = (predictions == labels).sum().item()
            training_end = time.time()
            # print(f"training time = {training_end - training_start} seconds")
            
            running_correct += predicted_corr
            loading_start = time.time()
        
        end_time = time.time()
        print(f"{epoch} epoch time = {end_time - start_time} seconds") 
        print(f"{epoch} epoch train accuracy = {running_correct/len(train_loader.dataset)}")
    if save:
        save_model(model, optimizer, make_path(model,epochs))
    
def test(model, test_loader, criterion=nn.BCELoss()):
    with torch.no_grad():
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
    paths_eps = get_paths(GLOBAL_DATA["epilepsy_path"],".npz")[:10000]
    paths_no_eps = get_paths(GLOBAL_DATA["no_epilepsy_path"],".npz")[:len(paths_eps)]
    print(f"epilepsy files count = {len(paths_eps)}")
    print(f" no epilepsy files count = {len(paths_no_eps)}")
    dataset = EEG_dataset(paths_no_eps, paths_eps)
    data_train, data_test = random_split(dataset, [0.8, 0.2])
    batch_size = GLOBAL_DATA["batch_size"]

     
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=GLOBAL_DATA["cores_count"] - 1)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=GLOBAL_DATA["cores_count"] - 1)
    return train_loader, test_loader

def test_model_from_memory(model, optim, path_to_read_model):
    _, test_loader = prepare_loaders()
    
    model_1, _ = load_model_to_finetune(model, optim, path_to_read_model)
    test(model_1, test_loader)


if __name__ == "__main__":  
    torch.manual_seed(SEED)
    print(f"cores availabe = {GLOBAL_DATA["cores_count"]}")
    
    if torch.backends.mps.is_available():
        print("Using Apple GPU (Metal Performance Shaders - MPS)")
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU (CUDA)")
    else:
        print("Using CPU")
    
    # model = Conv1d_lstm(len(GLOBAL_DATA['labels']))
    # model = Conv2d_lstm()
    model = ResNet_lstm()

    train_loader, test_loader = prepare_loaders()

    optimizer = AdamW(model.parameters())
    criterion = nn.BCELoss()
    epochs = GLOBAL_DATA["epochs"]

    
    train(model,optimizer,train_loader, epochs=epochs)
    test(model,test_loader)
    test_model_from_memory(model, optimizer, make_path(model,epochs))

    
    
    