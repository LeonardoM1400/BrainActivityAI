# Brain Activity AI

A machine learning project that uses AI to detect and classify brain activity from EEG (electroencephalogram) signals. Includes two deep learning models, a full preprocessing pipeline, real EEG dataset support, and a web interface.

## What it does

- Trains on real EEG data from the **PhysioNet Motor Imagery dataset** (auto-downloaded)
- Loads any EEG recording (`.edf`, `.bdf`, `.fif`, `.set`) or generates synthetic data for testing
- Preprocesses signals with bandpass and notch filters
- Extracts frequency-band power features (delta, theta, alpha, beta, gamma)
- Classifies brain states: Rest, Left Fist, Right Fist, Both Hands
- Serves a **web interface** to upload EEG files and visualize predictions live

## Models

| Model | Description |
|---|---|
| **EEGNet** | Compact CNN designed specifically for EEG. Fast to train, works well with limited data. |
| **EEG Transformer** | Treats each EEG channel as a token. Better at capturing long-range patterns across channels. |

## Project Structure

```
BrainActivityAI/
├── main.py                      # Demo pipeline on synthetic data
├── train_physionet.py           # Train on real PhysioNet EEG dataset
├── requirements.txt
├── app/
│   ├── server.py                # FastAPI web server
│   └── static/
│       └── index.html           # Web UI (upload EEG, view predictions)
└── src/
    ├── data/
    │   ├── loader.py            # Load EEG files or generate synthetic data
    │   └── physionet.py         # PhysioNet dataset downloader & preprocessor
    ├── preprocessing/
    │   └── filters.py           # Bandpass, notch filter, normalization
    ├── features/
    │   └── spectral.py          # Per-band power extraction
    ├── models/
    │   └── eeg_cnn.py           # EEGNet and EEG Transformer architectures
    ├── training/
    │   └── trainer.py           # Training loop, evaluation, data loaders
    └── visualization/
        └── plots.py             # Loss curves, confusion matrix, EEG waveform plots
```

## Setup

**Requirements:** Python 3.11+

```bash
# Clone the repo
git clone git@github.com:LeonardoM1400/BrainActivityAI.git
cd BrainActivityAI

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train on real EEG data (PhysioNet)

Downloads the dataset automatically on first run.

```bash
python train_physionet.py
```

Options:
```bash
python train_physionet.py --subjects 1 2 3 4 5 --model transformer --epochs 40
```

| Flag | Default | Description |
|---|---|---|
| `--subjects` | `1 2 3` | Which subjects to train on (1–109 available) |
| `--model` | `cnn` | `cnn` (EEGNet) or `transformer` |
| `--epochs` | `30` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--batch-size` | `32` | Batch size |

Saves the trained model to `checkpoints/`.

### 2. Launch the web interface

```bash
uvicorn app.server:app --reload
```

Then open **http://localhost:8000** in your browser.

- Click **Run Demo** to test with synthetic EEG data instantly
- Or upload a real `.edf` / `.bdf` / `.fif` file to analyze it
- See the predicted brain state, confidence per class, and a live signal preview

### 3. Run the synthetic demo (no dataset needed)

```bash
python main.py
```

Trains both models on generated data and saves plots to `outputs/`.

## Output

| File | Description |
|---|---|
| `checkpoints/*.pt` | Trained model weights |
| `outputs/*_training.png` | Loss and accuracy curves |
| `outputs/*_confusion.png` | Confusion matrix |

## Dependencies

- [MNE](https://mne.tools) — EEG/MEG data loading, PhysioNet dataset
- [PyTorch](https://pytorch.org) — Deep learning models
- [FastAPI](https://fastapi.tiangolo.com) — Web server
- [scikit-learn](https://scikit-learn.org) — Metrics and utilities
- [SciPy](https://scipy.org) — Signal filtering
- [Matplotlib](https://matplotlib.org) / [Seaborn](https://seaborn.pydata.org) — Visualization
