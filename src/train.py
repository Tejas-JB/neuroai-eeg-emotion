"""Training loop for SNN and baseline MLP on EEG emotion classification.

Usage:
    python src/train.py --model snn --epochs 5 --use-synthetic
    python src/train.py --model baseline --epochs 5 --use-synthetic
    python src/train.py --model snn --epochs 50 --data-path data/ExtractedFeatures/

Both models are trained with Adam + CrossEntropyLoss. The SNN path additionally
tracks network sparsity (fraction of zero activations across hidden layers)
and the best checkpoint is selected by test-set accuracy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Allow running as `python src/train.py` from the project root by putting the
# repo root on sys.path ahead of the src/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data
from src.models.baseline_mlp import BaselineMLP
from src.models.snn_model import SpikingNN
from src.spike_encoder import encode_spikes, normalize_features

logger = logging.getLogger(__name__)

DARK = {
    "bg": "#0f1419",
    "text": "#8899a6",
    "grid": "#1a2530",
    "edge": "#2a3a4a",
    "green": "#00e676",
    "blue": "#1da1f2",
    "red": "#ff5252",
}


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": DARK["bg"],
        "axes.facecolor": DARK["bg"],
        "savefig.facecolor": DARK["bg"],
        "axes.edgecolor": DARK["edge"],
        "axes.labelcolor": DARK["text"],
        "xtick.color": DARK["text"],
        "ytick.color": DARK["text"],
        "text.color": DARK["text"],
        "grid.color": DARK["grid"],
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "figure.dpi": 180,
    })


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_loaders(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_snn(
    model: SpikingNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    timesteps: int,
    save_dir: Path,
) -> Dict[str, list]:
    """Train the SNN, tracking loss / accuracy / sparsity per epoch."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "snn_checkpoint.pt"
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "sparsity": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        spikes_sum = 0.0
        spikes_possible = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            spikes = encode_spikes(X_batch, method="rate", T=timesteps)
            out = model(spikes)
            loss = criterion(out["spike_counts"], y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = out["spike_counts"].argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

            h1 = out["hidden1_spikes"]
            h2 = out["hidden2_spikes"]
            spikes_sum += float(h1.sum().item() + h2.sum().item())
            spikes_possible += h1.numel() + h2.numel()

        train_loss = total_loss / total
        train_acc = correct / total
        sparsity = 1.0 - (spikes_sum / max(spikes_possible, 1))

        test_loss, test_acc = _evaluate_snn(model, test_loader, criterion, device, timesteps)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        print(
            f"[SNN] epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  "
            f"sparsity={sparsity * 100:.1f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "timesteps": timesteps,
                },
                ckpt_path,
            )

    return history


def _evaluate_snn(
    model: SpikingNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    timesteps: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            spikes = encode_spikes(X_batch, method="rate", T=timesteps)
            out = model(spikes)
            loss = criterion(out["spike_counts"], y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = out["spike_counts"].argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
    return total_loss / total, correct / total


def train_baseline(
    model: BaselineMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    save_dir: Path,
) -> Dict[str, list]:
    """Train the baseline MLP on raw features."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "baseline_checkpoint.pt"
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        test_loss, test_acc = _evaluate_baseline(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"[MLP] epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                },
                ckpt_path,
            )
    return history


def _evaluate_baseline(
    model: BaselineMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)
    return total_loss / total, correct / total


def plot_training_curves(
    snn_history: Optional[Dict[str, list]],
    baseline_history: Optional[Dict[str, list]],
    save_path: Path,
) -> None:
    """Plot loss + accuracy curves for one or both models."""
    _apply_dark_theme()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax_loss, ax_acc = axes

    if snn_history is not None:
        xs = range(1, len(snn_history["train_loss"]) + 1)
        ax_loss.plot(xs, snn_history["train_loss"], color=DARK["green"], label="SNN train", linewidth=2)
        ax_loss.plot(xs, snn_history["test_loss"], color=DARK["green"], linestyle="--", label="SNN test")
        ax_acc.plot(xs, snn_history["train_acc"], color=DARK["green"], label="SNN train", linewidth=2)
        ax_acc.plot(xs, snn_history["test_acc"], color=DARK["green"], linestyle="--", label="SNN test")

    if baseline_history is not None:
        xs = range(1, len(baseline_history["train_loss"]) + 1)
        ax_loss.plot(xs, baseline_history["train_loss"], color=DARK["blue"], label="MLP train", linewidth=2)
        ax_loss.plot(xs, baseline_history["test_loss"], color=DARK["blue"], linestyle="--", label="MLP test")
        ax_acc.plot(xs, baseline_history["train_acc"], color=DARK["blue"], label="MLP train", linewidth=2)
        ax_acc.plot(xs, baseline_history["test_acc"], color=DARK["blue"], linestyle="--", label="MLP test")

    ax_loss.set_title("Training / Test Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(facecolor=DARK["bg"], edgecolor=DARK["edge"], labelcolor=DARK["text"])

    ax_acc.set_title("Training / Test Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(facecolor=DARK["bg"], edgecolor=DARK["edge"], labelcolor=DARK["text"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _merge_history_into_curves(save_dir: Path, model_type: str, history: Dict[str, list]) -> None:
    """Persist history as JSON and regenerate combined training_curves.png.

    Keeps per-model JSON files so a later run of the other model can produce
    a combined SNN-vs-MLP chart without retraining.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    hist_path = save_dir / f"{model_type}_history.json"
    with hist_path.open("w") as f:
        json.dump(history, f)

    snn_hist = None
    base_hist = None
    snn_file = save_dir / "snn_history.json"
    base_file = save_dir / "baseline_history.json"
    if snn_file.exists():
        with snn_file.open() as f:
            snn_hist = json.load(f)
    if base_file.exists():
        with base_file.open() as f:
            base_hist = json.load(f)

    plot_training_curves(snn_hist, base_hist, save_dir / "training_curves.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SNN or baseline MLP on SEED / synthetic data")
    parser.add_argument("--model", choices=["snn", "baseline"], default="snn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--data-path", type=str, default="data/")
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
        use_synthetic=args.use_synthetic,
        data_path=args.data_path,
        seed=args.seed,
    )
    X_train, X_test, _ = normalize_features(X_train, X_test)

    train_loader, test_loader = _build_loaders(
        X_train, X_test, y_train, y_test, batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    print(f"Device: {device}")
    print(f"Train/Test samples: {X_train.shape[0]}/{X_test.shape[0]}   Features: {X_train.shape[1]}")

    if args.model == "snn":
        model = SpikingNN(timesteps=args.timesteps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"SNN parameters: {model.count_parameters():,}")
        history = train_snn(
            model, train_loader, test_loader, optimizer, criterion,
            device, args.epochs, args.timesteps, save_dir,
        )
        _merge_history_into_curves(save_dir, "snn", history)
        print(f"\nBest SNN test accuracy: {max(history['test_acc']):.4f}")
    else:
        model = BaselineMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"MLP parameters: {model.count_parameters():,}")
        history = train_baseline(
            model, train_loader, test_loader, optimizer, criterion, device, args.epochs, save_dir,
        )
        _merge_history_into_curves(save_dir, "baseline", history)
        print(f"\nBest MLP test accuracy: {max(history['test_acc']):.4f}")


if __name__ == "__main__":
    main()
