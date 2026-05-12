"""ATLIF-style adaptive threshold wrapper for PSN.

Preserves PSN's parallel computation. Adds per-neuron adaptive threshold
with ATLIF's proven activity-pruning dynamics.

Follows the official ATLIF training paradigm (Activity-Pruning-SNN, NeurIPS 2024/2025):
  1. adaptive_threshold: nn.Parameter (learnable, optimized by gradient descent)
  2. Activity-driven update: thresh += update_value × lr  (after each optimizer step)
     update_value = sp × window(proxy) × spike_rate  [accumulated over forward]
  3. Firing-rate regularization: L2 penalty on act_value

Key adaptation for black-box PSN:
  - Original ATLIF: window = max(0, 1 - |mem - thresh|/thresh)  [needs membrane]
  - Our wrapper:     window ≈ max(0, 1 - |input - thresh|/thresh)  [input as proxy]
  - The spike signal comes from PSN output (binary, after the base neuron's threshold)

Usage in training loop (source-patch required):
  loss += atlif_spike_regularization(model) * args.eta2
  optimizer.step()
  atlif_threshold_update(model, optimizer.param_groups[0]["lr"])
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _window_proxy(input: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
    """Approximate ATLIF's membrane-proximity window without membrane access.

    Original: window = max(0, 1 - |(mem - thresh) / thresh|)
    Proxy:    window = max(0, 1 - |input / thresh|)
    """
    return (1.0 - (input / thresh.clamp_min(1e-6)).abs()).clamp_min(0.0)


class ATLIFThresholdNeuron(nn.Module):
    """PSN wrapper with per-neuron ATLIF adaptive threshold.

    Parameters
    ----------
    base_neuron:
        Underlying PSN (or any spiking neuron). Preserved as-is.
    v_th:
        Nominal threshold. adaptive_threshold is initialized to this value.
    sp:
        Activity-pruning strength (ATLIF's ``sp`` parameter).
        Typical: 1e-5 to 1e-3. 0 = no activity-driven update.
    """

    def __init__(
        self,
        base_neuron: nn.Module,
        v_th: float = 0.1,
        sp: float = 1e-4,
    ):
        super().__init__()
        self.base = base_neuron

        # ATLIF learnable threshold — optimized by gradient descent
        self.adaptive_threshold = nn.Parameter(torch.tensor(float(v_th)))

        # Activity-pruning hyperparams
        self.v_th_nominal = float(v_th)
        self.sp = float(sp)

        # Accumulators (reset after each threshold_update call)
        self.update_value: float = 0.0
        self.act_value: float = 0.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Modulate input by adaptive threshold
        #    Higher threshold → less input → harder to fire → fewer spikes
        scale = self.v_th_nominal / self.adaptive_threshold.clamp_min(1e-6)
        effective_x = x * scale

        # 2. PSN forward (parallel, unchanged)
        out = self.base(effective_x)

        # 3. ATLIF activity tracking (no grad)
        with torch.no_grad():
            # Window proxy: how close was input to threshold?
            window = _window_proxy(effective_x, self.adaptive_threshold)

            # Update value: sp * window * spike (ATLIF Eq.)
            # Aggregate over spatial dims, average over batch
            spike_mask = (out != 0).float()
            thre_updates = (self.sp * window * spike_mask).sum(dim=(2, 3, 4)).mean().item()
            self.update_value += thre_updates / max(getattr(self.base, 'T', 1), 1)

            # Activity value for regularization
            self.act_value = out.reshape(out.size(0), -1).mean(1).sum().item()

        return out

    # ------------------------------------------------------------------
    # Regularization (called from training loop)
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        """Firing-rate penalty (mirrors ATLIF's eta2 * regularize_spike)."""
        return torch.tensor(self.act_value, device=self.adaptive_threshold.device)

    def get_state(self) -> dict:
        return {
            "adaptive_threshold": float(self.adaptive_threshold.item()),
            "update_value": self.update_value,
            "act_value": self.act_value,
            "sp": self.sp,
        }


# ---------------------------------------------------------------------------
# Training-loop helpers (mirror ATLIF's utils.py)
# ---------------------------------------------------------------------------

def atlif_threshold_update(model: nn.Module, lr: float) -> None:
    """Apply accumulated activity-driven updates to ATLIF thresholds.

    Must be called AFTER optimizer.step(). Mirrors ATLIF's threshold_update().
    """
    for module in model.modules():
        if isinstance(module, ATLIFThresholdNeuron):
            v = module.update_value
            module.adaptive_threshold.data.add_(v * lr)
            module.update_value = 0.0


def atlif_spike_regularization(model: nn.Module) -> torch.Tensor:
    """Sum of act_value over all ATLIFThresholdNeuron modules.

    Should be added to the task loss: loss += atlif_spike_reg(model) * eta2
    """
    total = 0.0
    for module in model.modules():
        if isinstance(module, ATLIFThresholdNeuron):
            total += module.act_value
    return torch.tensor(total, device=next(model.parameters()).device)
