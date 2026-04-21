import numpy as np
from scipy.signal import welch


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    """Integrate PSD within a frequency band. psd: (..., n_freqs)."""
    idx = np.logical_and(freqs >= low, freqs <= high)
    return psd[..., idx].mean(axis=-1)


def extract_spectral_features(
    epochs: np.ndarray, sfreq: float, nperseg: int = 256
) -> np.ndarray:
    """
    Compute per-band power for each channel in each epoch.
    Input:  (n_epochs, n_channels, n_times)
    Output: (n_epochs, n_channels * n_bands)
    """
    n_epochs, n_channels, _ = epochs.shape
    n_bands = len(BANDS)
    features = np.zeros((n_epochs, n_channels * n_bands))

    for e in range(n_epochs):
        col = 0
        for ch in range(n_channels):
            freqs, psd = welch(epochs[e, ch], fs=sfreq, nperseg=nperseg)
            for low, high in BANDS.values():
                features[e, col] = band_power(psd, freqs, low, high)
                col += 1

    return features


def feature_names(n_channels: int) -> list[str]:
    names = []
    for ch in range(n_channels):
        for band in BANDS:
            names.append(f"ch{ch:02d}_{band}")
    return names
