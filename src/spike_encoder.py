"""Spike encoding for neuroai-eeg-emotion.

Converts continuous DE feature tensors into binary spike trains suitable for
SNN processing. Two encoders are provided:

    - Rate coding (default): Poisson spike trains whose firing probability
      equals the (normalized) feature value at every timestep. Relies on
      snntorch.spikegen.rate.
    - Delta coding (alternative): spikes are emitted where the absolute
      temporal difference in a (noised) time-varying version of the input
      exceeds a threshold. Useful for exploring temporal-contrast coding.

Both encoders output shape ``(T, batch, n_features)`` — time-first, as
required by snnTorch's recurrent forward pass.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from snntorch import spikegen

logger = logging.getLogger(__name__)

DEFAULT_T = 25


def normalize_features(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Min-max normalize features to [0, 1] using train statistics only.

    Args:
        X_train: training features, shape (N_train, F).
        X_test: test features, shape (N_test, F).

    Returns:
        Tuple of (X_train_norm, X_test_norm, (feature_min, feature_max)).
        Both outputs are clamped to [0, 1] — test samples outside the
        training range are clipped rather than producing invalid spike rates.
    """
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("normalize_features expects 2D tensors (N, F)")

    feat_min = X_train.min(dim=0).values
    feat_max = X_train.max(dim=0).values
    span = (feat_max - feat_min).clamp(min=1e-8)

    X_train_norm = ((X_train - feat_min) / span).clamp(0.0, 1.0)
    X_test_norm = ((X_test - feat_min) / span).clamp(0.0, 1.0)
    return X_train_norm, X_test_norm, (feat_min, feat_max)


def encode_rate(
    X: torch.Tensor,
    T: int = DEFAULT_T,
    gain: float = 1.0,
) -> torch.Tensor:
    """Rate-code features into Poisson spike trains.

    Args:
        X: feature tensor, shape (batch, F). Values should be pre-normalized
           to [0, 1]; this function clamps defensively.
        T: number of timesteps.
        gain: scalar multiplier on spike probability.

    Returns:
        Binary spike tensor of shape (T, batch, F).
    """
    if X.ndim != 2:
        raise ValueError(f"encode_rate expects (batch, F) tensor; got shape {tuple(X.shape)}")

    X_clamped = X.clamp(0.0, 1.0)
    spikes = spikegen.rate(X_clamped, num_steps=T, gain=gain)
    if spikes.shape[0] != T or spikes.shape[1:] != X.shape:
        raise RuntimeError(
            f"spikegen.rate returned unexpected shape {tuple(spikes.shape)}; "
            f"expected ({T}, {X.shape[0]}, {X.shape[1]})"
        )
    return spikes


def encode_delta(
    X: torch.Tensor,
    T: int = DEFAULT_T,
    threshold: float = 0.1,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Delta-code features by spiking on large temporal differences.

    Builds a synthetic time-varying signal by adding small Gaussian noise
    per timestep, then emits a spike at each (t, i, f) where the absolute
    difference between consecutive frames exceeds ``threshold``.

    Args:
        X: feature tensor, shape (batch, F), expected in [0, 1].
        T: number of timesteps.
        threshold: absolute-difference threshold for spike generation.
        noise_std: stddev of per-timestep Gaussian noise.
        seed: optional RNG seed for deterministic output.

    Returns:
        Binary spike tensor of shape (T, batch, F).
    """
    if X.ndim != 2:
        raise ValueError(f"encode_delta expects (batch, F) tensor; got shape {tuple(X.shape)}")

    generator = torch.Generator(device=X.device)
    if seed is not None:
        generator.manual_seed(seed)

    X_base = X.clamp(0.0, 1.0).unsqueeze(0).expand(T, -1, -1)   # (T, batch, F)
    noise = torch.randn(X_base.shape, generator=generator, device=X.device) * noise_std
    time_series = (X_base + noise).clamp(0.0, 1.0)

    diffs = torch.zeros_like(time_series)
    diffs[1:] = (time_series[1:] - time_series[:-1]).abs()

    spikes = (diffs > threshold).to(torch.float32)
    return spikes


def encode_spikes(
    X: torch.Tensor,
    method: str = "rate",
    T: int = DEFAULT_T,
    **kwargs,
) -> torch.Tensor:
    """Unified spike-encoding dispatcher.

    Args:
        X: feature tensor, shape (batch, F).
        method: "rate" or "delta".
        T: number of timesteps.
        **kwargs: forwarded to the specific encoder (e.g. ``gain``, ``threshold``).

    Returns:
        Binary spike tensor of shape (T, batch, F).
    """
    if method == "rate":
        return encode_rate(X, T=T, **kwargs)
    if method == "delta":
        return encode_delta(X, T=T, **kwargs)
    raise ValueError(f"Unknown encoding method {method!r}; expected 'rate' or 'delta'")


def _sparsity_pct(spikes: torch.Tensor) -> float:
    """Return the percentage of zero elements in a spike tensor."""
    return float((1.0 - spikes.mean()) * 100.0)


def _apply_dark_theme() -> None:
    """Minimal dark theme applied before plot generation in the smoke test.

    Mirrors the spec in PRD §8. The full ``setup_dark_theme`` lives in
    ``src/visualize.py`` (Phase 6); this inline copy keeps Phase 3 self-
    contained without introducing a module not yet in scope.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#0f1419",
        "axes.facecolor": "#0f1419",
        "savefig.facecolor": "#0f1419",
        "axes.edgecolor": "#2a3a4a",
        "axes.labelcolor": "#8899a6",
        "xtick.color": "#8899a6",
        "ytick.color": "#8899a6",
        "text.color": "#8899a6",
        "grid.color": "#1a2530",
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "figure.dpi": 180,
    })


def _smoke_test() -> None:
    """Run Phase 3 checkpoint tests and emit results/spike_encoding_demo.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data_loader import generate_synthetic_data

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)

    X_np, y_np = generate_synthetic_data(n_samples=300, seed=0)
    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).long()

    T = DEFAULT_T
    print("=" * 60)
    print("Phase 3 smoke test — spike encoding")
    print("=" * 60)
    print(f"Input X: shape={tuple(X.shape)} dtype={X.dtype}")

    spikes_rate = encode_rate(X, T=T)
    print(f"Rate output:  shape={tuple(spikes_rate.shape)}  "
          f"unique={sorted(spikes_rate.unique().tolist())}  "
          f"sparsity={_sparsity_pct(spikes_rate):.2f}%")

    spikes_delta = encode_delta(X, T=T, threshold=0.1, seed=0)
    print(f"Delta output: shape={tuple(spikes_delta.shape)}  "
          f"unique={sorted(spikes_delta.unique().tolist())}  "
          f"sparsity={_sparsity_pct(spikes_delta):.2f}%")

    # Contract checks.
    assert spikes_rate.shape == (T, X.shape[0], X.shape[1])
    assert set(spikes_rate.unique().tolist()) <= {0.0, 1.0}, "rate output must be binary"
    rate_sparsity = _sparsity_pct(spikes_rate)
    assert 40.0 <= rate_sparsity <= 95.0, f"rate sparsity {rate_sparsity:.1f}% outside [40%, 95%]"
    assert set(spikes_delta.unique().tolist()) <= {0.0, 1.0}, "delta output must be binary"

    # Unified dispatcher parity.
    via_unified = encode_spikes(X, method="rate", T=T)
    assert via_unified.shape == spikes_rate.shape

    # Demo visualization: one sample per class, rate-coded.
    _apply_dark_theme()
    class_names = ["Negative", "Neutral", "Positive"]
    class_colors = ["#ff5252", "#1da1f2", "#00e676"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for cls_idx, (ax, name, color) in enumerate(zip(axes, class_names, class_colors)):
        sample_idx = int((y == cls_idx).nonzero(as_tuple=True)[0][0].item())
        sample_spikes = spikes_rate[:, sample_idx, :].T.cpu().numpy()     # (F, T)
        times, neurons = (sample_spikes > 0).nonzero()
        ax.scatter(neurons, times, s=1.2, c=color, alpha=0.85)
        ax.set_title(name, color=color)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Feature index")
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-5, X.shape[1] + 5)

    fig.suptitle("Rate-Coded Spike Trains by Emotion Class (synthetic)", color="#8899a6")
    fig.tight_layout()
    out_path = "results/spike_encoding_demo.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved demo visualization → {out_path}")
    print("All assertions passed.")


if __name__ == "__main__":
    _smoke_test()
