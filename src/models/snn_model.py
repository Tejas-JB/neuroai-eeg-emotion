"""Spiking Neural Network model for EEG emotion classification.

Three-layer feedforward SNN with Leaky Integrate-and-Fire neurons
(snnTorch's ``snn.Leaky``). Architecture: 310 → 256 → 128 → 3.

Design choices:
    - Learnable per-layer decay ``beta`` initialized at 0.9.
    - Fixed threshold of 1.0 with subtract-on-spike reset.
    - ``fast_sigmoid`` surrogate gradient (slope 25) enables backprop
      through the discontinuous spike function.
    - Readout is the sum of output-layer spikes over T timesteps; argmax
      of this spike count yields the predicted emotion class.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from snntorch import Leaky, surrogate


class SpikingNN(nn.Module):
    """3-layer LIF SNN for EEG emotion classification.

    Args:
        input_size: input feature dimension (default 310 for SEED DE features).
        hidden1: first hidden layer size.
        hidden2: second hidden layer size.
        output_size: number of output classes.
        beta: initial membrane-decay coefficient (learnable per layer).
        threshold: firing threshold (fixed).
        timesteps: number of simulation timesteps T.
    """

    def __init__(
        self,
        input_size: int = 310,
        hidden1: int = 256,
        hidden2: int = 128,
        output_size: int = 3,
        beta: float = 0.9,
        threshold: float = 1.0,
        timesteps: int = 25,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps

        spike_grad = surrogate.fast_sigmoid(slope=25)
        leaky_kwargs = dict(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            learn_beta=True,
            reset_mechanism="subtract",
        )

        self.fc1 = nn.Linear(input_size, hidden1)
        self.lif1 = Leaky(**leaky_kwargs)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = Leaky(**leaky_kwargs)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.lif3 = Leaky(**leaky_kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the SNN forward in time.

        Args:
            x: pre-encoded spike train of shape (T, batch, input_size).

        Returns:
            Dict with keys:
                spike_counts: (batch, output_size) — sum of output spikes
                membrane_traces: (T, batch, output_size)
                output_spikes: (T, batch, output_size)
                hidden1_spikes: (T, batch, hidden1)
                hidden2_spikes: (T, batch, hidden2)
        """
        if x.ndim != 3 or x.shape[0] != self.timesteps:
            raise ValueError(
                f"SpikingNN.forward expected (T={self.timesteps}, batch, F); got {tuple(x.shape)}"
            )

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        out_spk, out_mem = [], []
        h1_spk, h2_spk = [], []

        for t in range(self.timesteps):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            h1_spk.append(spk1)
            h2_spk.append(spk2)
            out_spk.append(spk3)
            out_mem.append(mem3)

        output_spikes = torch.stack(out_spk, dim=0)
        membrane_traces = torch.stack(out_mem, dim=0)
        hidden1_spikes = torch.stack(h1_spk, dim=0)
        hidden2_spikes = torch.stack(h2_spk, dim=0)

        return {
            "spike_counts": output_spikes.sum(dim=0),
            "membrane_traces": membrane_traces,
            "output_spikes": output_spikes,
            "hidden1_spikes": hidden1_spikes,
            "hidden2_spikes": hidden2_spikes,
        }

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
