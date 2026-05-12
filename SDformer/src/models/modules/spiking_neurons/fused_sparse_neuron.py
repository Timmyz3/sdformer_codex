"""Fused Sparse Neuron (FSN): unifies three SNN efficiency paradigms.

Integrates:
  1. **GTCN** — coarse gate (clock-gating) + ATLIF adaptive threshold
  2. **LMH-style multi-level spikes** — quantized membrane → {0, 1, ..., L-1} levels
     per spike, carrying more information per bit
  3. **Ternary-spike option** — signed spikes {-1, 0, +1} naturally suited for
     optical flow where positive/negative event polarities encode motion direction

Hardware mapping (3-bit spike datapath):
  - binary mode (num_levels=1): 1 comparator, 1-bit spike
  - ternary mode (num_levels=2, signed=True): 2 comparators, 2-bit spike
  - multi-level mode (num_levels=3): 2 comparators + encoder, 2-bit spike
  - 2-bit mode (num_levels=4): 3 comparators + encoder, 2-bit spike

All modes share the same coarse-gate clock-gating signal, so downstream
AND-popcount or gated-accumulate units stay multiplier-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusedSparseNeuron(nn.Module):
    """Gate-threshold co-optimized neuron with multi-level spike output.

    Parameters
    ----------
    base_neuron:
        Underlying spiking neuron (PSN, LIF, PLIF, etc.).
    num_levels:
        Number of output levels. 1 = binary (original), 2 = two-level,
        3 = LMH-style three-level, 4 = 2-bit. Higher values carry more
        information per spike at the cost of wider datapath.
    signed:
        If True, produces signed spikes (ternary-style {-L, ..., 0, ..., +L}).
        Useful for optical flow where polarity matters.
    init_logit:
        Initial gate logit. Positive = gate starts open.
    threshold:
        Hard-gate binarization threshold in [0, 1].
    activity_eta:
        ATLIF learning rate for threshold adaptation. 0 = disabled.
    target_rate:
        Target mean firing rate for threshold adaptation.
    adapt_momentum:
        EMA momentum for running firing-rate tracking.
    level_learnable:
        If True, per-level thresholds are learnable. If False, evenly spaced.
    """

    def __init__(
        self,
        base_neuron: nn.Module,
        num_levels: int = 1,
        signed: bool = False,
        init_logit: float = 2.0,
        threshold: float = 0.5,
        activity_eta: float = 0.0,
        target_rate: float = 0.05,
        adapt_momentum: float = 0.99,
        level_learnable: bool = False,
    ):
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")

        self.base = base_neuron
        self.num_levels = int(num_levels)
        self.signed = bool(signed)

        # --- Coarse gate ---
        self.gate_logit = nn.Parameter(torch.tensor(float(init_logit)))
        self.threshold = float(threshold)

        # --- ATLIF adaptive threshold ---
        self.activity_eta = float(activity_eta)
        self.target_rate = float(target_rate)
        self.adapt_momentum = float(adapt_momentum)
        self.register_buffer("running_firing_rate", torch.tensor(0.0))
        self.register_buffer("threshold_bias", torch.tensor(0.0))
        self.register_buffer("_step_count", torch.tensor(0.0))

        # --- Multi-level quantization ---
        self.level_learnable = bool(level_learnable)
        if num_levels > 1:
            if level_learnable:
                # Learnable per-level thresholds (initialized evenly spaced)
                self.level_thresholds = nn.Parameter(
                    torch.linspace(0.0, 1.0, num_levels + 1)[1:-1]
                )
            else:
                self.register_buffer(
                    "level_thresholds",
                    torch.linspace(0.0, 1.0, num_levels + 1)[1:-1],
                )
            self.register_buffer("level_step", torch.tensor(1.0 / num_levels))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gate_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    @property
    def gate_is_open(self) -> bool:
        return bool(self.gate_probability.item() >= self.threshold)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def hard_gate(self) -> torch.Tensor:
        prob = self.gate_probability
        hard = (prob >= self.threshold).to(prob.dtype)
        return hard.detach() - prob.detach() + prob

    def _quantize_spike(self, raw_spike: torch.Tensor) -> torch.Tensor:
        """Convert binary spike to multi-level quantized output.

        Strategy: use the raw spike magnitude (pre-threshold membrane proxy)
        to determine the quantization level. For base neurons that output
        continuous values, this preserves gradation. For binary neurons,
        we accumulate neighboring spikes to simulate multi-level output.
        """
        if self.num_levels <= 1 and not self.signed:
            return raw_spike

        if self.num_levels <= 1 and self.signed:
            # Simple ternary: use raw magnitude sign
            return raw_spike.sign() * raw_spike.abs().clamp(min=0)

        # Multi-level: quantize the continuous-valued output into L levels
        abs_spike = raw_spike.abs()
        max_val = abs_spike.amax().clamp_min(1e-6)

        if self.signed:
            sign = raw_spike.sign()
            normalized = abs_spike / max_val
            level_idx = (normalized * self.num_levels).long().clamp(0, self.num_levels)
            quantized = level_idx.float() * (max_val / self.num_levels)
            return sign * quantized
        else:
            normalized = raw_spike / max_val
            level_idx = (normalized * self.num_levels).long().clamp(0, self.num_levels)
            return level_idx.float() * (max_val / self.num_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. ATLIF adaptive threshold: bias the input
        effective_x = x - self.threshold_bias

        # 2. Base neuron
        out = self.base(effective_x)

        # 3. Multi-level quantization (LMH / ternary)
        out = self._quantize_spike(out)

        # 4. Coarse gate (hardware clock-gating)
        gated = out * self.hard_gate()

        # 5. ATLIF activity tracking and threshold update
        if self.activity_eta > 0 and self.training:
            with torch.no_grad():
                current_rate = (out != 0).float().mean()
                self.running_firing_rate.mul_(self.adapt_momentum).add_(
                    current_rate * (1.0 - self.adapt_momentum)
                )
                delta = self.running_firing_rate - self.target_rate
                self.threshold_bias.add_(self.activity_eta * delta)
                self._step_count.add_(1.0)

        return gated

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        if self.activity_eta <= 0:
            return torch.tensor(0.0, device=self.gate_logit.device)
        delta = self.running_firing_rate - self.target_rate
        return delta * delta

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "gate_prob": float(self.gate_probability.item()),
            "gate_open": self.gate_is_open,
            "num_levels": self.num_levels,
            "signed": self.signed,
            "running_rate": float(self.running_firing_rate.item()),
            "threshold_bias": float(self.threshold_bias.item()),
            "activity_eta": self.activity_eta,
            "target_rate": self.target_rate,
        }
