# dataloader.py (replace existing dataset class)

import numpy as np
import torch
from torch.utils.data import Dataset

class CAPNPZDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        X, y = data['X'], data['y']
        cap_mask = (y > 0)
        self.X = X[cap_mask].astype(np.float32)
        self.y = (y[cap_mask] - 1).astype(np.int64)  # shift labels 1→0, 2→1, 3→2
        self.transform = transform

        # scale to [-1,1] for Tanh generator
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        self.X = 2 * self.X - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # shape (1,640)
        y = self.y[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

