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
from models import EEG_dataset, Conv1d_lstm, save_model, load_model_to_finetune, Conv2d_lstm, ResNet_lstm, EEG_Conv2d_LSTM, ensemble_models
import time
import torch.multiprocessing as mp
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# torch.set_num_threads(GLOBAL_DATA["cores_count"] - 1)


def make_path(model: nn.Module, epochs: int, transform: Optional[str] = None) -> str:
    transform_part = f"{transform}_" if transform else ""
    return f"{GLOBAL_DATA['saving_models_path']}{model.get_model_name()}_{transform_part}{epochs}epochs"

def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    criterion: nn.Module = nn.BCELoss(),
    epochs: int = 5,
    save: bool = True,
) -> None:
    print(f"training { make_path(model, epochs, model.transform)}")
    for epoch in range(epochs):
        model.train()
        running_correct = 0
        start_time = time.time()
        loading_start = time.time()
        for i, (data, labels) in enumerate(tqdm(train_loader, desc="train progres")):
            loading_end = time.time()
            training_start = time.time()
            # print(f"loading time = {loading_end-loading_start} seconds")
            model.zero_grad()
            out = model(data).squeeze(dim=-1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            predictions = (out > 0.5).float()
            predicted_corr = (predictions == labels).sum().item()
            training_end = time.time()
            running_correct += predicted_corr
            loading_start = time.time()
            # print(f"training time = {training_end - training_start} seconds")

        end_time = time.time()
        print(f"{epoch} epoch time = {end_time - start_time} seconds") 
        print(f"{epoch} epoch train accuracy = {running_correct / len(train_loader.dataset)}")
    if save:
        save_model(model, optimizer, make_path(model, epochs, model.transform))

def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module = nn.BCELoss()
) -> None:
    with torch.no_grad():
        model.eval()
        running_correct = 0
        running_loss = 0.0
        for i, (data, labels) in enumerate(tqdm(test_loader, desc="test progres")):
            out = model(data).squeeze(dim=-1)
            loss = criterion(out, labels)
            running_loss += loss.item()
            predictions = (out > 0.5).float()
            predicted_corr = (predictions == labels).sum().item()
            running_correct += predicted_corr
        print(f"test accuracy = {running_correct / len(test_loader.dataset)}")
        print(f"loss = {running_loss}")

def prepare_loaders() -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(SEED)
    paths_eps = get_paths(GLOBAL_DATA["epilepsy_path"], ".npz")
    paths_no_eps = get_paths(GLOBAL_DATA["no_epilepsy_path"], ".npz")[:len(paths_eps)]
    print(f"epilepsy files count = {len(paths_eps)}")
    print(f" no epilepsy files count = {len(paths_no_eps)}")
    dataset = EEG_dataset(paths_no_eps, paths_eps)
    data_train, data_test = random_split(dataset, [0.8, 0.2])
    batch_size = GLOBAL_DATA["batch_size"]
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=GLOBAL_DATA["workers"])
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=GLOBAL_DATA["workers"])
    return train_loader, test_loader

def test_model_from_memory(
    model: nn.Module,
    optim: Optimizer,
    epochs: int,
    test_loader: DataLoader
) -> None:
    path_to_read_model = make_path(model, epochs, model.transform)
    print(f"testing model {path_to_read_model}")

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
    # model = Conv2d_lstm(transform="stft")
    # model = ResNet_lstm(transform="stft")
    
    input_freq_bins = 51 
    input_time_bins = 100  
    # Initialize the model
    model = EEG_Conv2d_LSTM(input_freq_bins=input_freq_bins, input_time_bins=input_time_bins, num_classes=1, transform="stft")

    train_loader, test_loader = prepare_loaders()

    optimizer = AdamW(model.parameters())
    criterion = nn.BCELoss()
    epochs = GLOBAL_DATA["epochs"]
    


    
    
    # model = Conv1d_lstm(len(GLOBAL_DATA['labels']))
    # optimizer = AdamW(model.parameters())

    # model_1, _ = load_model_to_finetune(model, optimizer, "saved_models/conv1d_lstm_15epochs")
    
    # model = Conv2d_lstm()
    # optimizer = AdamW(model.parameters())
    
    # model_2, _ = load_model_to_finetune(model, optimizer, "saved_models/conv2d_lstm_15epochs")
    
    # model = Conv2d_lstm()
    # optimizer = AdamW(model.parameters())
    
    # model_5, _ = load_model_to_finetune(model, optimizer, "saved_models/conv2d_lstm_30epochs")
    
    
    
    
    model = ResNet_lstm(transform="stft")
    optimizer = AdamW(model.parameters())
    
    model_3, _ = load_model_to_finetune(model, optimizer, "saved_models/resnet_lstm_stft_15epochs")

    model = ResNet_lstm(transform="stft")
    optimizer = AdamW(model.parameters())
    
    model_4, _ = load_model_to_finetune(model, optimizer, "saved_models/resnet_lstm_stft_5epochs")
    
    
    model = EEG_Conv2d_LSTM(input_freq_bins=input_freq_bins, input_time_bins=input_time_bins, num_classes=1, transform="stft")
    optimizer = AdamW(model.parameters())
    
    model_6, _ = load_model_to_finetune(model, optimizer, "saved_models/conv2d_lstm_stft_5epochs")
    

    list_of_models = [model_6, model_4, model_3]
    
    # # list_of_models = [model_2]
    model = ensemble_models(list_of_models)
    
    # train(model,optimizer,train_loader, epochs=epochs)
    test(model,test_loader)
    # test_model_from_memory(model, optimizer, epochs, test_loader)

    