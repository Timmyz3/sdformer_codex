"""Simple spiking neuron candidate."""

from __future__ import annotations

import torch
from torch import nn

from .common import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class SNNode(CandidateNeuron):
    def __init__(
        self,
        T: int,
        v_threshold: float = 1.0,
        decay: float = 0.25,
        lens: float = 1.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.T = T
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)), requires_grad=False)
        self.decay = float(decay)
        self.lens = float(lens)
        self.detach_reset = bool(detach_reset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0])
        outputs = []
        for t in range(T):
            mem = mem * self.decay + x[t]
            spike = SpikeFn.apply(mem, self.v_threshold, self.lens)
            reset_spike = spike.detach() if self.detach_reset else spike
            mem = mem * (1.0 - reset_spike)
            outputs.append(spike * self.v_threshold)
        return torch.stack(outputs, dim=0)

