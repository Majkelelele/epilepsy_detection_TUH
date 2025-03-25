from constants import GLOBAL_DATA
import numpy as np
from preprocess import get_paths
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F







class EEG_dataset(Dataset):
    def __init__(self, files):
        self.files = files
        print(f"size = {len(files)}")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = np.load(self.files[idx])
        return item["arr_0"]

class EEG_net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        )
    def forward(self,x):
        batch_size, channels, l = x.shape
        x = x.reshape(batch_size, -1)
        return F.sigmoid(self.seq(x))
        


if __name__ == "__main__":  
    paths_eps = get_paths("sliced_data/train/eps",".npz")  
    # paths_no_eps = get_paths("sliced_data/train/no_eps",".npz")
    batch_size = GLOBAL_DATA["batch_size"]
    
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")
    dataset = EEG_dataset(paths_eps)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    window_len = GLOBAL_DATA['resampled_freq'] * GLOBAL_DATA["slice_size_scs"]
    input_size = len(GLOBAL_DATA['labels']) * window_len
    model = EEG_net(input_size)



    optimizer = AdamW(model.parameters())
    criterion = nn.BCELoss()
    epochs = 1
    labels = torch.ones(GLOBAL_DATA["batch_size"]).unsqueeze(dim=-1)
    for epoch in range(epochs):
        model.train()
        running_acc = 0
        for i, batch in enumerate(tqdm(train_loader,desc="progres")):
            model.zero_grad()
            out = model(batch)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            predictions = (out > 0.5).float()
            predicted_corr = (predictions == labels).sum().item()
            if i % 1000 == 0:
                print(f"predicted {predicted_corr} correctly in a batch size = {batch.shape[0]}")
            running_acc += predicted_corr
            
        


