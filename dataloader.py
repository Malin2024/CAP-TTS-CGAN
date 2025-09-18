import os
import numpy as np
import pyedflib
from torch.utils.data import Dataset

from utils import parse_cap_annotations, window_signal

class CAPDataset(Dataset):
    def __init__(self, raw_dir, channels=None, window_sec=30, fs=100, use_labels=True, cache_npy=None):
        self.raw_dir = raw_dir
        self.window_sec = window_sec
        self.fs = fs
        self.use_labels = use_labels
        self.channels = channels
        self.windows = []
        self.labels = []

        if cache_npy and os.path.exists(cache_npy):
            print(f"Loading preprocessed data from {cache_npy}")
            data = np.load(cache_npy, allow_pickle=True)
            self.windows = data["windows"].tolist()
            self.labels = data["labels"].tolist()
        else:
            self._build_dataset()
            if cache_npy:
                np.savez(cache_npy, windows=self.windows, labels=self.labels)

    def _build_dataset(self):
        files = os.listdir(self.raw_dir)
        edf_files = [f for f in files if f.lower().endswith('.edf')]
        for edf in edf_files:
            edf_path = os.path.join(self.raw_dir, edf)
            base = os.path.splitext(edf)[0]
            ann_path = os.path.join(self.raw_dir, base + '_annotations.txt')

            try:
                f = pyedflib.EdfReader(edf_path)
            except Exception as e:
                print(f"Failed to read {edf_path}: {e}")
                continue

            n_signals = f.signals_in_file
            labels = f.getSignalLabels()
            sigs = np.array([f.readSignal(i) for i in range(n_signals)])
            f._close()
            del f

            if self.channels is not None:
                if all(isinstance(c, int) for c in self.channels):
                    sigs = sigs[self.channels]
                else:
                    idxs = [labels.index(c) for c in self.channels if c in labels]
                    sigs = sigs[idxs]

            anns = None
            if self.use_labels and os.path.exists(ann_path):
                anns = parse_cap_annotations(ann_path)

            wlen = int(self.window_sec * self.fs)
            windows, win_labels = window_signal(sigs, anns, wlen, fs=self.fs)
            for win, lab in zip(windows, win_labels):
                win = (win - win.mean(axis=1, keepdims=True)) / (win.std(axis=1, keepdims=True) + 1e-8)
                self.windows.append(win.astype(np.float32))
                self.labels.append(lab if lab is not None else -1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.labels[idx]
        return x, y
