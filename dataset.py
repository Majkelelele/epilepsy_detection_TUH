import numpy as np
import torch
from torch.utils.data import Dataset
from constants import EPS, NO_EPS


class EEG_dataset(Dataset):
    def __init__(self, path_no_eps: list[str], path_eps: list[str]) -> None:
        self.paths = path_eps + path_no_eps

    def __len__(self) -> None:
        return len(self.paths)

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.tensor(np.load(self.paths[idx])["arr_0"], dtype=torch.float32)

        label = torch.tensor(NO_EPS if "no_eps" in self.paths[idx] else EPS, dtype=torch.float32)
        return data, label