"""Generate all Phase 6 visualizations from existing checkpoints + history.

Produces:
    results/spike_rasters.png
    results/accuracy_comparison.png
    results/membrane_traces.png
    results/sparsity_training.png

Run:
    python scripts/generate_all_viz.py --use-synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from src.data_loader import load_data
from src.models.snn_model import SpikingNN
from src.spike_encoder import encode_spikes, normalize_features
from src.visualize import (
    plot_accuracy_comparison,
    plot_membrane_traces,
    plot_spike_rasters,
    plot_sparsity_over_training,
)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _auto_device()
    results_dir = Path(args.results_dir)

    X_train, X_test, y_train, y_test = load_data(
        use_synthetic=args.use_synthetic, data_path=args.data_path, seed=args.seed,
    )
    X_train, X_test, _ = normalize_features(X_train, X_test)

    snn = SpikingNN(timesteps=args.timesteps).to(device)
    snn_ckpt = torch.load(results_dir / "snn_checkpoint.pt", map_location=device, weights_only=False)
    snn.load_state_dict(snn_ckpt["model_state"])

    # Build per-class data batches for raster + membrane plots.
    class_batches = {}
    class_singles = {}
    for cls in range(3):
        mask = (y_test == cls).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            mask = (y_train == cls).nonzero(as_tuple=True)[0]
            pool = X_train
        else:
            pool = X_test
        picks = mask[: min(32, mask.numel())]
        class_batches[cls] = pool[picks]
        class_singles[cls] = pool[picks[0]]

    plot_spike_rasters(
        snn, class_batches, encode_spikes, args.timesteps, device,
        results_dir / "spike_rasters.png",
    )
    plot_membrane_traces(
        snn, class_singles, encode_spikes, args.timesteps, device,
        results_dir / "membrane_traces.png",
    )

    snn_hist_path = results_dir / "snn_history.json"
    base_hist_path = results_dir / "baseline_history.json"
    with snn_hist_path.open() as f:
        snn_hist = json.load(f)
    with base_hist_path.open() as f:
        base_hist = json.load(f)

    snn_best = max(snn_hist["test_acc"])
    base_best = max(base_hist["test_acc"])
    plot_accuracy_comparison(snn_best, base_best, results_dir / "accuracy_comparison.png")
    plot_sparsity_over_training(snn_hist, results_dir / "sparsity_training.png")

    print("Generated:")
    for name in [
        "spike_rasters.png",
        "membrane_traces.png",
        "accuracy_comparison.png",
        "sparsity_training.png",
    ]:
        path = results_dir / name
        print(f"  {path}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
