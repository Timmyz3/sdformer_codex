"""Refractory-Period Pruning Neuron (A5).

Wraps a base spiking neuron with a hardware-enforced refractory period.
After emitting a spike, the neuron is silenced for N timesteps.

Hardware mapping:
  2-bit saturating counter per neuron:
    - counter=0 → neuron active (evaluate normally)
    - counter>0 → output forced to zero, counter decremented
  Zero ALU overhead — just a register + comparator + AND gate on output.

Reference: Activity Pruning AT-LIF (NeurIPS 2024/2025) validates
post-spike silencing for sparsity with minimal accuracy impact.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RefractoryNeuron(nn.Module):
    """Wrap a base neuron with a refractory period after each spike.

    Parameters
    ----------
    base_neuron:
        Underlying spiking neuron (PSN, LIF, etc.).
    refractory_steps:
        Number of timesteps to suppress output after a spike. 1 = suppress
        the immediate next timestep only. 2-3 recommended for event data.
    mode:
        ``"hard"`` — once triggered, counter resets on each new spike (full
        refractory window after LAST spike).
        ``"soft"`` — refractory mask is proportional to spike magnitude
        (larger spikes → longer refractory).
    """

    def __init__(
        self,
        base_neuron: nn.Module,
        refractory_steps: int = 2,
        mode: str = "hard",
    ):
        super().__init__()
        if refractory_steps < 1:
            raise ValueError(f"refractory_steps must be >= 1, got {refractory_steps}")

        self.base = base_neuron
        self.refractory_steps = int(refractory_steps)
        self.mode = str(mode)
        self.register_buffer("_refractory_counter", torch.tensor(0))

    def _hard_refractory_mask(self, spike: torch.Tensor) -> torch.Tensor:
        """Full refractory: counter resets on each spike."""
        has_spike = (spike != 0).any()
        if has_spike:
            self._refractory_counter.fill_(self.refractory_steps)

        if self._refractory_counter > 0:
            self._refractory_counter.sub_(1)
            return torch.zeros_like(spike)
        return spike

    def _soft_refractory_mask(self, spike: torch.Tensor) -> torch.Tensor:
        """Soft refractory: magnitude-proportional suppression."""
        spike_magnitude = spike.abs().amax()
        if spike_magnitude > 0:
            steps = min(self.refractory_steps, int(spike_magnitude * self.refractory_steps))
            self._refractory_counter.fill_(max(self._refractory_counter.item(), steps))

        if self._refractory_counter > 0:
            self._refractory_counter.sub_(1)
            return torch.zeros_like(spike)
        return spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base neuron forward
        out = self.base(x)

        # Apply refractory mask (only during inference or always)
        if self.mode == "hard":
            out = self._hard_refractory_mask(out)
        else:
            out = self._soft_refractory_mask(out)

        return out

    def reset_refractory(self) -> None:
        """Reset refractory state (call between samples)."""
        self._refractory_counter.zero_()

    def get_state(self) -> dict:
        return {
            "refractory_steps": self.refractory_steps,
            "mode": self.mode,
            "counter": int(self._refractory_counter.item()),
        }
