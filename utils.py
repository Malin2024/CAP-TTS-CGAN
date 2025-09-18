import numpy as np

def parse_cap_annotations(ann_path):
    anns = []
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) >= 3:
                onset = float(parts[0])
                dur = float(parts[1])
                lab = parts[2]
                lab_int = 1 if 'A' in lab else 0
                anns.append((int(onset), int(dur), lab_int))
    return anns

def window_signal(sigs, anns, wlen, fs=100):
    n_channels, n_samples = sigs.shape
    step = wlen
    windows = []
    labels = []
    for start in range(0, n_samples - wlen + 1, step):
        win = sigs[:, start:start + wlen]
        label = -1
        if anns:
            start_sec = start / fs
            end_sec = (start + wlen) / fs
            counts = {}
            for a_on, a_dur, a_lab in anns:
                a_end = a_on + a_dur
                if (a_on < end_sec) and (a_end > start_sec):
                    counts[a_lab] = counts.get(a_lab, 0) + 1
            if counts:
                label = max(counts.items(), key=lambda x: x[1])[0]
        windows.append(win)
        labels.append(label)
    return windows, labels
