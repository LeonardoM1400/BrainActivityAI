"""
Brain Activity AI — end-to-end demo using synthetic EEG data.

Run:  python main.py
"""

import numpy as np
import torch

from src.data.loader import generate_synthetic_eeg
from src.preprocessing.filters import bandpass_filter, normalize_epochs
from src.features.spectral import extract_spectral_features
from src.models.eeg_cnn import EEGNet, EEGTransformer
from src.training.trainer import build_dataloaders, train
from src.visualization.plots import plot_training_history, plot_confusion_matrix


def run_cnn_pipeline():
    print("=== EEGNet (CNN) Pipeline ===")
    epochs, labels, sfreq = generate_synthetic_eeg(
        n_channels=32, duration_sec=120.0, sfreq=256.0, n_classes=2
    )

    # Preprocess
    epochs = bandpass_filter(epochs, lowcut=1.0, highcut=40.0, sfreq=sfreq)
    epochs = normalize_epochs(epochs)

    # Shape: (batch, 1, channels, times)
    X = torch.tensor(epochs[:, np.newaxis], dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    train_loader, val_loader = build_dataloaders(X, y, batch_size=16)

    model = EEGNet(n_classes=2, n_channels=32, n_times=epochs.shape[-1])
    history = train(model, train_loader, val_loader, n_epochs=20, lr=1e-3)

    plot_training_history(history, save_path="outputs/cnn_training.png")

    # Confusion matrix on val set
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            all_preds.extend(model(X_b.to(device)).argmax(1).cpu().numpy())
            all_true.extend(y_b.numpy())
    plot_confusion_matrix(
        np.array(all_true),
        np.array(all_preds),
        class_names=["Alpha", "Beta"],
        save_path="outputs/cnn_confusion.png",
    )


def run_transformer_pipeline():
    print("\n=== EEG Transformer Pipeline ===")
    epochs, labels, sfreq = generate_synthetic_eeg(
        n_channels=32, duration_sec=120.0, sfreq=256.0, n_classes=3
    )

    epochs = bandpass_filter(epochs, lowcut=1.0, highcut=40.0, sfreq=sfreq)
    epochs = normalize_epochs(epochs)

    X = torch.tensor(epochs, dtype=torch.float32)   # (batch, channels, times)
    y = torch.tensor(labels, dtype=torch.long)

    train_loader, val_loader = build_dataloaders(X, y, batch_size=16)

    model = EEGTransformer(n_classes=3, n_channels=32, n_times=epochs.shape[-1])
    history = train(model, train_loader, val_loader, n_epochs=20, lr=1e-3)

    plot_training_history(history, save_path="outputs/transformer_training.png")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    run_cnn_pipeline()
    run_transformer_pipeline()

    print("\nDone! Plots saved to outputs/")
