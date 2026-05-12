"""Gate-Threshold Co-Optimized Neuron (GTCN) for hardware-friendly sparse SNNs.

Fuses three mechanisms into a single hardware-mappable primitive:

1. **Coarse gate** (learnable scalar via STE):
   - gate=0 → entire neuron unit is clock-gated in hardware → zero power
   - gate=1 → neuron computes normally

2. **Adaptive threshold bias** (ATLIF-style activity-dependent regulation):
   - Firing rate too high → threshold_bias INCREASES → harder to spike
   - Firing rate too low  → threshold_bias DECREASES → easier to spike
   - Creates a negative-feedback loop that pushes firing rate toward target

3. **Running firing-rate tracking** (EMA):
   - Provides the signal for threshold adaptation
   - Exported as metadata for hardware performance monitoring

Hardware mapping (spike_unit.v):
   input x → subtract threshold_bias → base_neuron(x - bias) → × gate → output
              ^                                                       ^
              |__ adaptive feedback loop (slow)                       |
                                                                     |__ clock-gating (fast)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HardwareSparseNeuron(nn.Module):
    """Gate-threshold co-optimized spiking neuron wrapper.

    Parameters
    ----------
    base_neuron:
        The underlying spiking neuron (e.g. PSN, LIF, PLIF).
    init_logit:
        Initial value for the gate logit. Positive = gate starts open.
    threshold:
        Hard-gate binarization threshold in probability space.
    activity_eta:
        ATLIF learning rate for threshold adaptation. 0 = disabled.
    target_rate:
        Desired mean firing rate that threshold adaptation aims for.
    adapt_momentum:
        EMA decay for running firing rate (0 = no history, 1 = frozen).
    """

    def __init__(
        self,
        base_neuron: nn.Module,
        init_logit: float = 2.0,
        threshold: float = 0.5,
        activity_eta: float = 0.0,
        target_rate: float = 0.05,
        adapt_momentum: float = 0.99,
    ):
        super().__init__()
        self.base = base_neuron

        # --- Coarse gate (clock-gating in hardware) ---
        self.gate_logit = nn.Parameter(torch.tensor(float(init_logit)))
        self.threshold = float(threshold)

        # --- ATLIF-style adaptive threshold ---
        self.activity_eta = float(activity_eta)
        self.target_rate = float(target_rate)
        self.adapt_momentum = float(adapt_momentum)

        # Non-trainable state buffers
        self.register_buffer("running_firing_rate", torch.tensor(0.0))
        self.register_buffer("threshold_bias", torch.tensor(0.0))
        self.register_buffer("_step_count", torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gate_probability(self) -> torch.Tensor:
        """Sigmoid probability in [0, 1]."""
        return torch.sigmoid(self.gate_logit)

    @property
    def gate_is_open(self) -> bool:
        """True when the hard gate is active (prob >= threshold)."""
        return bool(self.gate_probability.item() >= self.threshold)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def hard_gate(self) -> torch.Tensor:
        """Straight-through estimator for binary gate."""
        prob = self.gate_probability
        hard = (prob >= self.threshold).to(prob.dtype)
        return hard.detach() - prob.detach() + prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Adaptive threshold: bias the input to modulate effective threshold
        effective_x = x - self.threshold_bias

        # 2. Base neuron with biased input
        out = self.base(effective_x)

        # 3. Coarse gate (hardware clock-gating)
        gated = out * self.hard_gate()

        # 4. ATLIF-style activity tracking and threshold update (training only)
        if self.activity_eta > 0 and self.training:
            with torch.no_grad():
                current_rate = (out != 0).float().mean()
                # EMA update
                self.running_firing_rate.mul_(self.adapt_momentum).add_(
                    current_rate * (1.0 - self.adapt_momentum)
                )
                # Threshold update: delta proportional to rate deviation
                delta = self.running_firing_rate - self.target_rate
                self.threshold_bias.add_(self.activity_eta * delta)
                self._step_count.add_(1.0)

        return gated

    # ------------------------------------------------------------------
    # Regularization (optional auxiliary loss)
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        """Quadratic penalty for firing-rate deviation from target."""
        if self.activity_eta <= 0:
            return torch.tensor(0.0, device=self.gate_logit.device)
        delta = self.running_firing_rate - self.target_rate
        return delta * delta

    # ------------------------------------------------------------------
    # Introspection (for logging / hardware metadata export)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Export state for logging and hardware metadata."""
        return {
            "gate_prob": float(self.gate_probability.item()),
            "gate_open": self.gate_is_open,
            "running_rate": float(self.running_firing_rate.item()),
            "threshold_bias": float(self.threshold_bias.item()),
            "activity_eta": self.activity_eta,
            "target_rate": self.target_rate,
        }
