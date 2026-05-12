"""Ternary spiking neuron candidate.

Adapted from the ternary spike activation in:
https://github.com/yfguo91/Ternary-Spike
"""

from __future__ import annotations

import torch
from torch import nn

from .common import CandidateNeuron, TernarySpikeFn, ensure_time_first, reset_like


class TSNNode(CandidateNeuron):
    def __init__(
        self,
        T: int,
        v_threshold: float = 1.0,
        decay: float = 0.25,
        fire_ratio: float = 1.0,
    ):
        super().__init__()
        self.T = T
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)), requires_grad=False)
        self.decay = float(decay)
        self.fire_ratio = float(fire_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0])
        outputs = []

        for t in range(T):
            mem = mem * self.decay + x[t]
            spike = TernarySpikeFn.apply(mem, self.v_threshold) * self.fire_ratio
            mem = mem * (1.0 - spike.abs().detach())
            outputs.append(spike)

        return torch.stack(outputs, dim=0)

