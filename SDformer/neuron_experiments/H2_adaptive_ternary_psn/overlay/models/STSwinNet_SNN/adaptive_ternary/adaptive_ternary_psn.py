"""Adaptive ternary PSN for attention Q/K neurons.

This keeps PSN's parallel temporal mixing, replaces the binary firing function
with an official Ternary-Spike-style straight-through ternary activation, and
uses a learnable positive threshold whose value scales the output.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_spike_activation(x: torch.Tensor, dead_zone: float = 0.5) -> torch.Tensor:
    """Ternary spike {-1, 0, +1} with clamp STE from Ternary-Spike."""
    out_s = torch.sign(x)
    out_s = torch.where(torch.abs(x) < dead_zone, torch.zeros_like(out_s), out_s)
    out_bp = torch.clamp(x, -1.0, 1.0)
    return (out_s - out_bp).detach() + out_bp


class AdaptiveTernaryPSN(nn.Module):
    """PSN temporal mixer with adaptive ternary threshold.

    Parameters
    ----------
    T:
        Number of timesteps.
    base_psn:
        Optional baseline PSN. When supplied, ``weight`` and ``bias`` are copied
        so finetuning starts from the baseline attention dynamics.
    theta_init:
        Initial threshold. Output values are ``{-theta, 0, +theta}``.
    learn_threshold:
        If true, threshold is a trainable parameter. Otherwise it is fixed.
    output_scale:
        ``"threshold"`` uses AT-LIF-style ``spike * theta`` output; ``"unit"``
        returns plain ternary spikes.
    """

    official_tsn_source = "https://github.com/yfguo91/Ternary-Spike"

    def __init__(
        self,
        T: int,
        base_psn: nn.Module | None = None,
        theta_init: float = 1.0,
        learn_threshold: bool = True,
        min_threshold: float = 1.0e-4,
        dead_zone: float = 0.5,
        output_scale: str = "threshold",
        target_rate: float = 0.0,
        activity_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        if output_scale not in {"threshold", "unit"}:
            raise ValueError(f"output_scale must be 'threshold' or 'unit', got {output_scale!r}")

        self.T = int(T)
        self.dead_zone = float(dead_zone)
        self.output_scale = output_scale
        self.min_threshold = float(min_threshold)
        self.target_rate = float(target_rate)
        self.activity_momentum = float(activity_momentum)

        if base_psn is not None and hasattr(base_psn, "weight") and hasattr(base_psn, "bias"):
            self.weight = nn.Parameter(base_psn.weight.detach().clone())
            self.bias = nn.Parameter(base_psn.bias.detach().clone())
            self.surrogate_function = getattr(base_psn, "surrogate_function", None)
        else:
            self.weight = nn.Parameter(torch.zeros([self.T, self.T]))
            self.bias = nn.Parameter(torch.zeros([self.T, 1]))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.constant_(self.bias, -1.0)
            self.surrogate_function = None

        theta_init = max(float(theta_init), self.min_threshold)
        raw_theta = math.log(math.exp(theta_init - self.min_threshold) - 1.0)
        if learn_threshold:
            self.raw_theta = nn.Parameter(torch.tensor(raw_theta, dtype=self.weight.dtype))
        else:
            self.register_buffer("raw_theta", torch.tensor(raw_theta, dtype=self.weight.dtype))

        self.register_buffer("running_activity", torch.tensor(0.0))
        self.register_buffer("running_pos_rate", torch.tensor(0.0))
        self.register_buffer("running_neg_rate", torch.tensor(0.0))

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw_theta) + self.min_threshold

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        theta = self.theta.to(device=h_seq.device, dtype=h_seq.dtype)
        ternary = ternary_spike_activation(h_seq / theta, self.dead_zone)
        out = ternary * theta if self.output_scale == "threshold" else ternary
        out = out.view(x_seq.shape)

        if self.training:
            with torch.no_grad():
                activity = (ternary != 0).float().mean()
                pos_rate = (ternary > 0).float().mean()
                neg_rate = (ternary < 0).float().mean()
                momentum = self.activity_momentum
                self.running_activity.mul_(momentum).add_(activity * (1.0 - momentum))
                self.running_pos_rate.mul_(momentum).add_(pos_rate * (1.0 - momentum))
                self.running_neg_rate.mul_(momentum).add_(neg_rate * (1.0 - momentum))

        return out

    def activity_regularization(self) -> torch.Tensor:
        if self.target_rate <= 0:
            return torch.tensor(0.0, device=self.weight.device)
        return (self.running_activity - self.target_rate) ** 2

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, theta={float(self.theta.detach().cpu()):.4f}, "
            f"dead_zone={self.dead_zone}, output_scale={self.output_scale}"
        )

