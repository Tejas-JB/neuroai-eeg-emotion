"""Evaluation entry point for trained SNN or baseline checkpoints.

Usage:
    python src/evaluate.py --model snn --checkpoint results/snn_checkpoint.pt --use-synthetic
    python src/evaluate.py --model baseline --checkpoint results/baseline_checkpoint.pt --use-synthetic

Produces:
    - Printed classification report (precision / recall / F1 per class + accuracy)
    - Confusion-matrix PNG at results/confusion_matrix_{model}.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data
from src.models.baseline_mlp import BaselineMLP
from src.models.snn_model import SpikingNN
from src.spike_encoder import encode_spikes, normalize_features
from src.visualize import CLASS_NAMES, plot_confusion_matrix

logger = logging.getLogger(__name__)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_type: str,
    timesteps: int = 25,
) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """Run inference on the test set and return metrics.

    Returns:
        preds: predicted class indices, shape (N,).
        labels: ground-truth class indices, shape (N,).
        report: sklearn classification_report string.
        cm: 3x3 confusion matrix.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if model_type == "snn":
                spikes = encode_spikes(X_batch, method="rate", T=timesteps)
                out = model(spikes)
                preds = out["spike_counts"].argmax(dim=1)
            else:
                preds = model(X_batch).argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    report = classification_report(
        labels_arr, preds_arr, target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1, 2])
    return preds_arr, labels_arr, report, cm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained SNN or baseline MLP")
    parser.add_argument("--model", choices=["snn", "baseline"], default="snn")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results/")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = _auto_device() if args.device == "auto" else torch.device(args.device)
    save_dir = Path(args.save_dir)

    X_train, X_test, y_train, y_test = load_data(
        use_synthetic=args.use_synthetic, data_path=args.data_path, seed=args.seed,
    )
    X_train, X_test, _ = normalize_features(X_train, X_test)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    if args.model == "snn":
        model = SpikingNN(timesteps=args.timesteps).to(device)
    else:
        model = BaselineMLP().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    preds, labels, report, cm = evaluate_model(
        model, test_loader, device, args.model, timesteps=args.timesteps,
    )
    acc = float((preds == labels).mean())
    model_name = "SNN" if args.model == "snn" else "MLP"

    print(f"\n=== {model_name} evaluation (checkpoint epoch {ckpt.get('epoch', '?')}) ===")
    print(report)
    print(f"Overall accuracy: {acc:.4f}\n")
    print("Confusion matrix:")
    print(cm)

    cm_path = save_dir / f"confusion_matrix_{args.model}.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path, model_name)
    print(f"\nSaved confusion matrix → {cm_path}")

    report_path = save_dir / f"report_{args.model}.txt"
    report_path.write_text(f"{model_name} test accuracy: {acc:.4f}\n\n{report}\n")


if __name__ == "__main__":
    main()
