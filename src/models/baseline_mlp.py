"""Baseline MLP for parameter-matched comparison against the SNN.

Same layer dimensions as ``SpikingNN`` (310 → 256 → 128 → 3), ReLU
activations, dropout between hidden layers. Consumes raw (non-spike)
features and returns unnormalized logits suitable for
``nn.CrossEntropyLoss``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Dense feedforward classifier used as the non-spiking baseline."""

    def __init__(
        self,
        input_size: int = 310,
        hidden1: int = 256,
        hidden2: int = 128,
        output_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward.

        Args:
            x: raw features of shape (batch, input_size).

        Returns:
            Logits of shape (batch, output_size).
        """
        if x.ndim != 2:
            raise ValueError(f"BaselineMLP expects (batch, F); got {tuple(x.shape)}")
        return self.net(x)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
