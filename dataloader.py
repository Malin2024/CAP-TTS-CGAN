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

        # Check label range
        assert self.y.min() >= 0 and self.y.max() < 3, \
            f"Labels should be in [0, 2] but got range [{self.y.min()}, {self.y.max()}]"

        self.transform = transform

        # Scale X to [-1,1] for Tanh activation output
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        self.X = 2 * self.X - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Ensure x has channel dimension (e.g. (1, 640))
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
