"""
FastAPI web server for Brain Activity AI.

Run:  uvicorn app.server:app --reload
"""

import io
import tempfile
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.data.loader import generate_synthetic_eeg
from src.preprocessing.filters import bandpass_filter, normalize_epochs
from src.models.eeg_cnn import EEGNet, EEGTransformer

app = FastAPI(title="Brain Activity AI")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_model_cache: dict = {}


def load_model(checkpoint_path: str) -> dict:
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    n_classes = ckpt["n_classes"]
    n_channels = ckpt["n_channels"]
    n_times = ckpt["n_times"]

    if model_type == "cnn":
        model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_times=n_times)
    else:
        model = EEGTransformer(n_classes=n_classes, n_channels=n_channels, n_times=n_times)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    result = {**ckpt, "model": model}
    _model_cache[checkpoint_path] = result
    return result


def find_checkpoint() -> str | None:
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    return str(ckpts[0]) if ckpts else None


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text()


@app.get("/api/status")
async def status():
    ckpt = find_checkpoint()
    return {
        "model_loaded": ckpt is not None,
        "checkpoint": ckpt,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an EEG file (.edf, .bdf, .fif) and get brain state predictions."""
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Run train_physionet.py first.",
        )

    ckpt_data = load_model(ckpt_path)
    model = ckpt_data["model"]
    n_channels = ckpt_data["n_channels"]
    n_times = ckpt_data["n_times"]
    class_names = ckpt_data["class_names"]
    model_type = ckpt_data["model_type"]

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".edf", ".bdf", ".fif", ".set"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        import mne
        raw = mne.io.read_raw(tmp_path, preload=True, verbose=False)
        raw.filter(1.0, 40.0, verbose=False)

        data = raw.get_data()[:n_channels]
        # Slice into n_times windows
        n_windows = data.shape[1] // n_times
        if n_windows == 0:
            raise HTTPException(status_code=400, detail="File too short for model input.")

        epochs = np.stack([data[:, i * n_times:(i + 1) * n_times] for i in range(n_windows)])
        epochs = normalize_epochs(epochs)

        if model_type == "cnn":
            X = torch.tensor(epochs[:, np.newaxis], dtype=torch.float32)
        else:
            X = torch.tensor(epochs, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).numpy()

        avg_probs = probs.mean(axis=0).tolist()
        predicted_class = int(np.argmax(avg_probs))

        # Return first epoch's raw signal for visualization (first 8 channels)
        sample_signal = epochs[0, :8].tolist()

        return JSONResponse({
            "predicted_class": predicted_class,
            "predicted_label": class_names[predicted_class],
            "probabilities": {class_names[i]: round(p, 4) for i, p in enumerate(avg_probs)},
            "n_epochs_analyzed": n_windows,
            "sample_signal": sample_signal,
        })
    finally:
        os.unlink(tmp_path)


@app.post("/api/demo")
async def demo():
    """Run prediction on synthetic EEG data (no file needed)."""
    ckpt_path = find_checkpoint()

    epochs, labels, sfreq = generate_synthetic_eeg(
        n_channels=64, duration_sec=10.0, sfreq=256.0, n_classes=2
    )
    epochs = bandpass_filter(epochs, 1.0, 40.0, sfreq)
    epochs = normalize_epochs(epochs)

    if ckpt_path:
        ckpt_data = load_model(ckpt_path)
        model = ckpt_data["model"]
        model_type = ckpt_data["model_type"]
        n_channels = ckpt_data["n_channels"]
        n_times = ckpt_data["n_times"]
        class_names = ckpt_data["class_names"]

        epochs_trimmed = epochs[:, :n_channels, :n_times]
        if model_type == "cnn":
            X = torch.tensor(epochs_trimmed[:, np.newaxis], dtype=torch.float32)
        else:
            X = torch.tensor(epochs_trimmed, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).numpy()

        avg_probs = probs.mean(axis=0).tolist()
        predicted_class = int(np.argmax(avg_probs))
    else:
        class_names = ["Alpha State", "Beta State"]
        avg_probs = [0.72, 0.28]
        predicted_class = 0

    sample_signal = epochs[0, :8].tolist()

    return JSONResponse({
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "probabilities": {class_names[i]: round(p, 4) for i, p in enumerate(avg_probs)},
        "n_epochs_analyzed": len(epochs),
        "sample_signal": sample_signal,
        "note": "Demo mode using synthetic EEG data",
    })
