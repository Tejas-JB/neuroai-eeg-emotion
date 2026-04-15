"""Phase 4 smoke test — forward pass + parameter count for both models."""
from __future__ import annotations

import torch

from src.models.baseline_mlp import BaselineMLP
from src.models.snn_model import SpikingNN


def _test() -> None:
    torch.manual_seed(0)
    batch = 8
    T = 25
    F = 310

    print("=" * 60)
    print("Phase 4 smoke test — model forward passes")
    print("=" * 60)

    snn = SpikingNN(timesteps=T)
    mlp = BaselineMLP()

    # SNN expects pre-encoded spikes (T, batch, F).
    x_snn = (torch.rand(T, batch, F) < 0.5).float()
    out = snn(x_snn)

    assert set(out.keys()) == {
        "spike_counts", "membrane_traces", "output_spikes", "hidden1_spikes", "hidden2_spikes"
    }, f"SNN returned unexpected keys: {set(out.keys())}"

    print("SNN outputs:")
    print(f"  spike_counts    shape={tuple(out['spike_counts'].shape)}   "
          f"(expected ({batch}, 3))")
    print(f"  membrane_traces shape={tuple(out['membrane_traces'].shape)}  "
          f"(expected ({T}, {batch}, 3))")
    print(f"  output_spikes   shape={tuple(out['output_spikes'].shape)}  "
          f"(expected ({T}, {batch}, 3))")
    print(f"  hidden1_spikes  shape={tuple(out['hidden1_spikes'].shape)} "
          f"(expected ({T}, {batch}, 256))")
    print(f"  hidden2_spikes  shape={tuple(out['hidden2_spikes'].shape)} "
          f"(expected ({T}, {batch}, 128))")

    assert out["spike_counts"].shape == (batch, 3)
    assert out["membrane_traces"].shape == (T, batch, 3)
    assert out["output_spikes"].shape == (T, batch, 3)
    assert out["hidden1_spikes"].shape == (T, batch, 256)
    assert out["hidden2_spikes"].shape == (T, batch, 128)

    # MLP expects raw features (batch, F).
    x_mlp = torch.rand(batch, F)
    logits = mlp(x_mlp)
    print(f"\nMLP output logits shape={tuple(logits.shape)} (expected ({batch}, 3))")
    assert logits.shape == (batch, 3)

    snn_params = snn.count_parameters()
    mlp_params = mlp.count_parameters()
    ratio = snn_params / mlp_params

    print(f"\nParameter counts:")
    print(f"  SNN  = {snn_params:,}")
    print(f"  MLP  = {mlp_params:,}")
    print(f"  ratio SNN/MLP = {ratio:.4f}")
    assert 0.9 <= ratio <= 1.1, f"SNN/MLP param ratio {ratio:.3f} outside [0.9, 1.1]"

    # Sanity: SNN argmax prediction falls in {0, 1, 2}.
    preds = out["spike_counts"].argmax(dim=1)
    assert set(preds.unique().tolist()) <= {0, 1, 2}
    print(f"\nSNN argmax predictions: {preds.tolist()}")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    _test()
