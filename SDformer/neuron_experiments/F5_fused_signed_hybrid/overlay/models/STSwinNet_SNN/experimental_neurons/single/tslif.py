from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class TSLIFNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, gamma: float = 0.5, decay_factor=(0.8, 0.2, 0.3, 0.7), lens: float = 1.0):
        super().__init__()
        self.T = T
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)), requires_grad=False)
        self.gamma = float(gamma)
        self.decay_factor = nn.Parameter(torch.tensor(decay_factor, dtype=torch.float32))
        self.short_weight = nn.Parameter(torch.tensor(1.0))
        self.long_weight = nn.Parameter(torch.tensor(1.0))
        self.cross_short = nn.Parameter(torch.tensor(0.1))
        self.cross_long = nn.Parameter(torch.tensor(0.8))
        self.lens = float(lens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        v_short = reset_like(x[0])
        v_long = reset_like(x[0])
        outputs = []
        for t in range(T):
            v_short_next = self.decay_factor[0] * v_short + self.decay_factor[1] * x[t] - self.cross_short * v_long
            v_long_next = self.decay_factor[2] * v_long + self.decay_factor[3] * x[t] - self.cross_long * v_short_next
            v_short, v_long = v_short_next, v_long_next
            spike_short = SpikeFn.apply(v_short, self.v_threshold, self.lens)
            spike_long = SpikeFn.apply(v_long, self.v_threshold, self.lens)
            spike = self.short_weight * spike_short + self.long_weight * spike_long
            v_short = v_short - spike_long.detach() * self.gamma
            v_long = v_long - spike_short.detach() * self.v_threshold
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
