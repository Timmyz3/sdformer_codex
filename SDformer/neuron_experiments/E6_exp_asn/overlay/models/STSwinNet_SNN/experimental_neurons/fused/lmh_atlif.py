from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class LMHATLIFNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, tau: float = 1.0, lens: float = 1.0):
        super().__init__()
        self.T = T
        self.threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.tau = float(tau)
        self.lens = float(lens)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.temporal_mask = nn.Parameter(torch.zeros(T, T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0])
        outputs = []
        temporal_weight = 2.0 * self.temporal_mask.sigmoid() / T
        flat_x = x.flatten(1)
        for t in range(T):
            mixed = torch.matmul(temporal_weight[t], flat_x).view_as(x[0])
            mem = (self.alpha.sigmoid() + self.tau) * mem.detach() + mixed
            spike01 = SpikeFn.apply(mem, self.threshold, self.lens)
            spike = spike01 * self.threshold
            mem = mem - spike.detach()
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
