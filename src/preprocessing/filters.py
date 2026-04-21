import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt_zi


def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, sfreq: float, order: int = 5
) -> np.ndarray:
    """Butterworth bandpass filter. data shape: (..., n_times)."""
    nyq = sfreq / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(
    data: np.ndarray, freq: float, sfreq: float, quality: float = 30.0
) -> np.ndarray:
    """Notch filter to remove power-line noise (50 or 60 Hz)."""
    b, a = iirnotch(freq / (sfreq / 2.0), quality)
    from scipy.signal import sosfiltfilt as _f, tf2sos
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, data, axis=-1)


def normalize_epochs(data: np.ndarray) -> np.ndarray:
    """Z-score normalize each epoch per channel. data: (epochs, channels, times)."""
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True) + 1e-8
    return (data - mean) / std
