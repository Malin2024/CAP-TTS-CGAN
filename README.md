# CAP-TTS-CGAN (skeleton)

This repository provides a skeleton for training a Time-series GAN (TTS-CGAN style) on the CAP Sleep Database (CAPSLPDB) to produce synthetic CAP EEG segments.

## Quick steps
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download CAPSLPDB from PhysioNet and place EDF + annotation files into a folder, e.g. `data/raw`.
3. Preprocess & create .npy dataset with provided dataloader utilities.
4. Train:
   ```bash
   python train.py --data data/processed/cap_windows.npy --epochs 200 --batch_size 64
   ```
5. Generate samples:
   ```bash
   python generate.py --model checkpoints/generator.pth --n 100 --out_dir generated
   ```

## Notes
- Replace the annotation parsing and channel selection with channels you need (e.g., C3-A2, C4-A1).
- Conditioning is optional; there's an example of including label conditioning for CAP phase.
