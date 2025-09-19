import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CAPDataset(Dataset):
    def __init__(self, data_dir, window_sec=30, fs=100, cache_npy=None):
        self.data_dir = data_dir
        self.window_sec = window_sec
        self.fs = fs
        self.cache_npy = cache_npy
        self.samples = []

        self._build_dataset()

    def _build_dataset(self):
        import pyedflib

        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.edf')]
        for fpath in files:
            with pyedflib.EdfReader(fpath) as f:
                n_signals = f.signals_in_file
                # Read all signals
                sigs = [f.readSignal(i) for i in range(n_signals)]
                # Crop to shortest channel length
                min_len = min(len(sig) for sig in sigs)
                sigs = np.array([sig[:min_len] for sig in sigs])  # shape: (channels, time)

                # --- windowing ---
                window_size = self.window_sec * self.fs
                n_windows = min_len // window_size
                for w in range(n_windows):
                    start = w * window_size
                    end = start + window_size
                    window = sigs[:, start:end]
                    self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32)
