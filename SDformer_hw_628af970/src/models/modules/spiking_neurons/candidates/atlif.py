"""Adaptive-threshold LIF candidate.

Adapted from the official Activity-Pruning-SNN ATLIF implementation:
https://github.com/putshua/Activity-Pruning-SNN
"""

from __future__ import annotations

import torch
from torch import nn

from .common import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class ATLIFNode(CandidateNeuron):
    def __init__(
        self,
        T: int,
        v_threshold: float = 1.0,
        tau: float = 1.0,
        lens: float = 1.0,
        activity_scale: float = 0.0,
    ):
        super().__init__()
        self.T = T
        self.thresh = nn.Parameter(torch.tensor(float(v_threshold)))
        self.tau = float(tau)
        self.lens = float(lens)
        self.activity_scale = float(activity_scale)
        self.firing_rate = 0.0
        self.spike_count = 0.0
        self.threshold_update = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0])
        spikes = []
        update_value = x.new_tensor(0.0)

        for t in range(T):
            mem = mem * self.tau + x[t]
            spike01 = SpikeFn.apply(mem, self.thresh, self.lens)
            spike = spike01 * self.thresh
            mem = (1.0 - spike / self.thresh.detach().clamp_min(1e-6)) * mem
            spikes.append(spike)

            if self.activity_scale:
                window = (1.0 - ((mem - self.thresh) / self.thresh.clamp_min(1e-6)).abs()).clamp_min(0.0)
                update_value = update_value + self.activity_scale * (window * spike01).mean()

        out = torch.stack(spikes, dim=0)
        with torch.no_grad():
            normalized = out / self.thresh.detach().clamp_min(1e-6)
            self.firing_rate = normalized.mean().item()
            self.spike_count = normalized.mean(dim=1).sum().item()
            self.threshold_update = (update_value / max(T, 1)).item()
        return out

