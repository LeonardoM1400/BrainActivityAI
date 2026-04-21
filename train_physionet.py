"""
Train on the PhysioNet EEG Motor Imagery dataset.

Run:  python train_physionet.py
      python train_physionet.py --subjects 1 2 3 --model transformer --epochs 40
"""

import argparse
import os
import numpy as np
import torch
import joblib

from src.data.physionet import load_subjects, CLASS_NAMES
from src.preprocessing.filters import normalize_epochs
from src.models.eeg_cnn import EEGNet, EEGTransformer
from src.training.trainer import build_dataloaders, train
from src.visualization.plots import plot_training_history, plot_confusion_matrix


def main(args):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Loading subjects: {args.subjects}")
    X, y, ch_names = load_subjects(subjects=args.subjects)
    print(f"Data shape: {X.shape}  |  Classes: {np.unique(y, return_counts=True)}")

    X = normalize_epochs(X)
    n_classes = len(np.unique(y))
    n_channels, n_times = X.shape[1], X.shape[2]

    if args.model == "cnn":
        X_tensor = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
        model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_times=n_times)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        model = EEGTransformer(n_classes=n_classes, n_channels=n_channels, n_times=n_times)

    y_tensor = torch.tensor(y, dtype=torch.long)
    train_loader, val_loader = build_dataloaders(X_tensor, y_tensor, batch_size=args.batch_size)

    history = train(model, train_loader, val_loader, n_epochs=args.epochs, lr=args.lr)

    plot_training_history(history, save_path=f"outputs/{args.model}_physionet_training.png")

    device = next(model.parameters()).device
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            all_preds.extend(model(X_b.to(device)).argmax(1).cpu().numpy())
            all_true.extend(y_b.numpy())

    label_names = CLASS_NAMES[:n_classes]
    plot_confusion_matrix(
        np.array(all_true),
        np.array(all_preds),
        class_names=label_names,
        save_path=f"outputs/{args.model}_physionet_confusion.png",
    )

    ckpt_path = f"checkpoints/{args.model}_physionet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": args.model,
            "n_classes": n_classes,
            "n_channels": n_channels,
            "n_times": n_times,
            "class_names": label_names,
            "ch_names": ch_names,
        },
        ckpt_path,
    )
    print(f"\nModel saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
