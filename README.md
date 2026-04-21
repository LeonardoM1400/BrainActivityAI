# Brain Activity AI

A machine learning project that uses AI to detect and classify brain activity from EEG (electroencephalogram) signals. It includes two deep learning models, a full preprocessing pipeline, and spectral feature extraction.

## What it does

- Loads real EEG recordings (`.edf`, `.bdf`, `.fif`, `.set`) or generates synthetic EEG data for testing
- Preprocesses signals with bandpass and notch filters
- Extracts frequency-band power features (delta, theta, alpha, beta, gamma)
- Trains two AI models to classify brain states
- Outputs training curves and confusion matrices

## Models

| Model | Description |
|---|---|
| **EEGNet** | Compact CNN designed specifically for EEG. Fast to train, works well with limited data. |
| **EEG Transformer** | Treats each EEG channel as a token. Better at capturing long-range patterns across channels. |

## Project Structure

```
BrainActivityAI/
├── main.py                  # Run both pipelines end-to-end
├── requirements.txt
└── src/
    ├── data/
    │   └── loader.py        # Load EEG files or generate synthetic data
    ├── preprocessing/
    │   └── filters.py       # Bandpass, notch filter, normalization
    ├── features/
    │   └── spectral.py      # Per-band power extraction
    ├── models/
    │   └── eeg_cnn.py       # EEGNet and EEG Transformer architectures
    ├── training/
    │   └── trainer.py       # Training loop, evaluation, data loaders
    └── visualization/
        └── plots.py         # Loss curves, confusion matrix, EEG waveform plots
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

### Run the demo (synthetic data)

```bash
python main.py
```

This generates synthetic EEG, trains both models, and saves plots to the `outputs/` folder.

### Use your own EEG data

Edit `main.py` and replace `generate_synthetic_eeg(...)` with a call to `load_eeg_file`:

```python
from src.data.loader import load_eeg_file

raw = load_eeg_file("path/to/your/file.edf")
```

Supported formats: `.edf`, `.bdf`, `.fif`, `.set`

### Customize training

In `main.py` you can adjust:

```python
# Change number of EEG channels, duration, classes
epochs, labels, sfreq = generate_synthetic_eeg(
    n_channels=64,
    duration_sec=300.0,
    sfreq=256.0,
    n_classes=4,
)

# Change training hyperparameters
history = train(model, train_loader, val_loader, n_epochs=50, lr=5e-4)
```

## Output

After running, the `outputs/` folder will contain:

- `cnn_training.png` — EEGNet loss and accuracy curves
- `cnn_confusion.png` — EEGNet confusion matrix
- `transformer_training.png` — Transformer loss and accuracy curves

## Dependencies

- [MNE](https://mne.tools) — EEG/MEG data loading and processing
- [PyTorch](https://pytorch.org) — Deep learning models
- [scikit-learn](https://scikit-learn.org) — Metrics and utilities
- [SciPy](https://scipy.org) — Signal filtering
- [Matplotlib](https://matplotlib.org) / [Seaborn](https://seaborn.pydata.org) — Visualization
