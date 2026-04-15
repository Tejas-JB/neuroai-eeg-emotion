"""Visualization suite for neuroai-eeg-emotion.

All plots use the shared dark theme specified in PRD §8. Call
``setup_dark_theme()`` once before any plotting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

DARK = {
    "bg": "#0f1419",
    "text": "#8899a6",
    "grid": "#1a2530",
    "edge": "#2a3a4a",
    "box": "#1a2530",
    "green": "#00e676",
    "red": "#ff5252",
    "blue": "#1da1f2",
    "yellow": "#ffab40",
}

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
CLASS_COLORS = [DARK["red"], DARK["blue"], DARK["green"]]


def setup_dark_theme() -> None:
    """Apply the repo-wide dark matplotlib theme (PRD §8)."""
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


def _dark_cmap() -> LinearSegmentedColormap:
    """Custom dark-to-green colormap for confusion matrices."""
    return LinearSegmentedColormap.from_list("dark_green", [DARK["bg"], DARK["green"]])


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    model_name: str,
) -> None:
    """Render a confusion matrix with the dark theme."""
    setup_dark_theme()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=_dark_cmap(), aspect="equal")

    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            color = DARK["bg"] if count > vmax / 2 else DARK["text"]
            ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=14)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spike_rasters(
    model,
    data_batches: Dict[int, torch.Tensor],
    encoder,
    timesteps: int,
    device: torch.device,
    save_path: Path,
    n_neurons_display: int = 50,
) -> None:
    """3-panel raster of hidden-layer-1 spikes, one panel per emotion class.

    Args:
        model: trained ``SpikingNN``.
        data_batches: dict ``{class_idx: features_tensor}`` with pre-normalized features.
        encoder: callable ``(X, T) -> spike train`` (see ``encode_spikes``).
        timesteps: T used for encoding + forward.
        device: device for the model.
        save_path: output PNG.
        n_neurons_display: number of hidden-1 neurons to show (top rows).
    """
    setup_dark_theme()
    model.eval()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    with torch.no_grad():
        for cls_idx, ax in enumerate(axes):
            X = data_batches[cls_idx].to(device)
            spikes = encoder(X, method="rate", T=timesteps)
            out = model(spikes)
            h1 = out["hidden1_spikes"].mean(dim=1)   # average over batch → (T, H1)
            matrix = h1[:, :n_neurons_display].cpu().numpy().T  # (H1, T)
            times, neurons = (matrix > matrix.mean()).nonzero()
            ax.scatter(neurons, times, s=6, c=CLASS_COLORS[cls_idx], alpha=0.9)
            ax.set_title(CLASS_NAMES[cls_idx], color=CLASS_COLORS[cls_idx])
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Hidden-1 neuron")
            ax.set_xlim(-0.5, timesteps - 0.5)
            ax.set_ylim(-1, n_neurons_display)

    fig.suptitle("SNN Spike Activity by Emotion Class", color=DARK["text"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(
    snn_accuracy: float,
    baseline_accuracy: float,
    save_path: Path,
) -> None:
    """Side-by-side bar chart of SNN vs MLP test accuracy."""
    setup_dark_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ["SNN", "MLP"]
    accs = [snn_accuracy * 100, baseline_accuracy * 100]
    colors = [DARK["green"], DARK["blue"]]
    bars = ax.bar(models, accs, color=colors, edgecolor=DARK["edge"], linewidth=1.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{acc:.2f}%", ha="center", color=DARK["text"], fontsize=14,
        )

    ax.set_ylim(0, max(accs) + 15 if max(accs) > 50 else 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Emotion Classification Accuracy: SNN vs MLP")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_membrane_traces(
    model,
    samples: Dict[int, torch.Tensor],
    encoder,
    timesteps: int,
    device: torch.device,
    save_path: Path,
) -> None:
    """Plot output-layer membrane potentials over time, overlaid per class."""
    setup_dark_theme()
    model.eval()

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    with torch.no_grad():
        traces_by_class = {}
        for cls_idx, X_single in samples.items():
            X = X_single.unsqueeze(0).to(device) if X_single.ndim == 1 else X_single[:1].to(device)
            spikes = encoder(X, method="rate", T=timesteps)
            out = model(spikes)
            traces_by_class[cls_idx] = out["membrane_traces"][:, 0, :].cpu().numpy()   # (T, 3)

    for output_neuron, ax in enumerate(axes):
        for cls_idx, traces in traces_by_class.items():
            ax.plot(
                range(timesteps),
                traces[:, output_neuron],
                color=CLASS_COLORS[cls_idx],
                label=f"input={CLASS_NAMES[cls_idx]}",
                linewidth=2,
                alpha=0.9,
            )
        ax.axhline(1.0, linestyle="--", color=DARK["text"], alpha=0.6, label="threshold")
        ax.set_title(f"Output Neuron {output_neuron} ({CLASS_NAMES[output_neuron]})",
                     color=CLASS_COLORS[output_neuron])
        ax.set_ylabel("Membrane potential")
        ax.grid(True, alpha=0.3)
        if output_neuron == 0:
            ax.legend(facecolor=DARK["bg"], edgecolor=DARK["edge"], labelcolor=DARK["text"], loc="upper right")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Output-Layer Membrane Potentials per Emotion Input", color=DARK["text"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sparsity_over_training(
    training_history: Dict[str, list],
    save_path: Path,
) -> None:
    """Line plot of SNN sparsity (%) across training epochs."""
    setup_dark_theme()
    if "sparsity" not in training_history or not training_history["sparsity"]:
        raise ValueError("training_history must contain non-empty 'sparsity' key")
    sparsity = np.array(training_history["sparsity"]) * 100
    xs = range(1, len(sparsity) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, sparsity, color=DARK["green"], linewidth=2, marker="o", markersize=5)
    ax.fill_between(xs, sparsity, color=DARK["green"], alpha=0.2)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Network Sparsity During Training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
