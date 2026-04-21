import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history: dict, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(history["val_acc"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | None = None,
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_eeg_sample(
    epoch: np.ndarray,
    sfreq: float,
    channel_names: list[str] | None = None,
    n_channels_shown: int = 8,
    save_path: str | None = None,
):
    """Plot a few channels of a single EEG epoch."""
    n_ch = min(n_channels_shown, epoch.shape[0])
    times = np.arange(epoch.shape[1]) / sfreq
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, n_ch * 1.2), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(times, epoch[i], linewidth=0.7)
        label = channel_names[i] if channel_names else f"Ch {i}"
        ax.set_ylabel(label, fontsize=7)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("EEG Epoch Sample")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
