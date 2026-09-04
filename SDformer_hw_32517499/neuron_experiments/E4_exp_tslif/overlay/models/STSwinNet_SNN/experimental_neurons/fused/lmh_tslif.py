from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class LMHTSLIFNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, gamma: float = 0.5, lens: float = 1.0):
        super().__init__()
        self.T = T
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)), requires_grad=False)
        self.gamma = float(gamma)
        self.temporal_mask = nn.Parameter(torch.zeros(T, T))
        self.decay_factor = nn.Parameter(torch.tensor((0.8, 0.2, 0.3, 0.7), dtype=torch.float32))
        self.lens = float(lens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        flat_x = x.flatten(1)
        temporal_weight = 2.0 * self.temporal_mask.sigmoid() / T
        v_short = reset_like(x[0])
        v_long = reset_like(x[0])
        outputs = []
        for t in range(T):
            mixed = torch.matmul(temporal_weight[t], flat_x).view_as(x[0])
            v_short = self.decay_factor[0] * v_short + self.decay_factor[1] * mixed - 0.1 * v_long
            v_long = self.decay_factor[2] * v_long + self.decay_factor[3] * mixed - 0.8 * v_short
            spike_short = SpikeFn.apply(v_short, self.v_threshold, self.lens)
            spike_long = SpikeFn.apply(v_long, self.v_threshold, self.lens)
            spike = spike_short + spike_long
            v_short = v_short - spike_long.detach() * self.gamma
            v_long = v_long - spike_short.detach() * self.v_threshold
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
