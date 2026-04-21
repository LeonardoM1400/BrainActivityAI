import numpy as np
import pandas as pd
import mne
from pathlib import Path


def load_eeg_file(path: str) -> mne.io.Raw:
    """Load an EEG file using MNE (supports .edf, .fif, .bdf, .set)."""
    path = Path(path)
    ext = path.suffix.lower()

    loaders = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".fif": mne.io.read_raw_fif,
        ".set": mne.io.read_raw_eeglab,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported file format: {ext}")

    return loaders[ext](path, preload=True, verbose=False)


def generate_synthetic_eeg(
    n_channels: int = 64,
    duration_sec: float = 60.0,
    sfreq: float = 256.0,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate synthetic EEG data for testing.
    Returns (signals, labels, sfreq).
    signals shape: (n_epochs, n_channels, n_times)
    """
    rng = np.random.default_rng(seed)
    epoch_len = int(sfreq * 2)  # 2-second epochs
    n_epochs = int(duration_sec / 2)
    times = np.arange(epoch_len) / sfreq

    signals = []
    labels = []

    for i in range(n_epochs):
        label = i % n_classes
        # Base pink noise
        noise = rng.standard_normal((n_channels, epoch_len)) * 10.0

        # Class-specific oscillations
        if label == 0:
            # Alpha band (8-13 Hz) dominant
            alpha = 20.0 * np.sin(2 * np.pi * 10 * times)
            noise += alpha
        else:
            # Beta band (13-30 Hz) dominant
            beta = 15.0 * np.sin(2 * np.pi * 20 * times)
            noise += beta

        signals.append(noise)
        labels.append(label)

    return np.array(signals), np.array(labels), sfreq
