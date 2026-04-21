"""
PhysioNet EEG Motor Movement/Imagery Dataset loader.
Uses MNE's built-in downloader — no manual setup needed.

Classes:
  0 = rest
  1 = left fist
  2 = right fist
  3 = both hands
"""

import numpy as np
import mne
from mne.datasets import eegbci

TASK_RUNS = {
    "rest": [1],
    "left_right_fist_imagery": [4, 8, 12],
    "both_hands_feet_imagery": [6, 10, 14],
}

EVENT_ID = {
    "rest": 1,
    "left_fist": 2,
    "right_fist": 3,
    "both_hands": 4,
}

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
CLASS_NAMES = ["Rest", "Left Fist", "Right Fist", "Both Hands"]


def load_subject(
    subject: int = 1,
    tmin: float = -0.1,
    tmax: float = 4.0,
    sfreq: float = 160.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Download (if needed) and epoch data for one subject.
    Returns (epochs, labels, channel_names).
    epochs shape: (n_epochs, n_channels, n_times)
    """
    runs = [1, 4, 6, 8, 10, 12, 14]
    raw_files = eegbci.load_data(subject, runs=runs, verbose=verbose)

    raws = [mne.io.read_raw_edf(f, preload=True, verbose=verbose) for f in raw_files]
    raw = mne.concatenate_raws(raws, verbose=verbose)

    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, verbose=verbose)

    raw.filter(1.0, 40.0, fir_design="firwin", verbose=verbose)

    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq, verbose=verbose)

    events, event_id = mne.events_from_annotations(raw, verbose=verbose)

    picks = mne.pick_types(raw.info, eeg=True)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=verbose,
    )

    X = epochs.get_data()
    y_raw = epochs.events[:, -1]

    # Remap to 0-indexed labels
    unique = sorted(set(y_raw))
    remap = {v: i for i, v in enumerate(unique)}
    y = np.array([remap[v] for v in y_raw])

    channel_names = epochs.ch_names
    return X, y, channel_names


def load_subjects(
    subjects: list[int] | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and concatenate multiple subjects."""
    if subjects is None:
        subjects = list(range(1, 6))  # subjects 1-5 by default

    all_X, all_y, ch_names = [], [], []
    for subj in subjects:
        print(f"Loading subject {subj}...")
        X, y, ch_names = load_subject(subj, **kwargs)
        all_X.append(X)
        all_y.append(y)

    return np.concatenate(all_X), np.concatenate(all_y), ch_names
