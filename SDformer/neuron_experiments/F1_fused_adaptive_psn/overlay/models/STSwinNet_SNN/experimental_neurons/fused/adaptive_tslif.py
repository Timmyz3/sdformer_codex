from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class AdaptiveTSLIFNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, gamma: float = 0.5, decay_factor=(0.8, 0.2, 0.3, 0.7), lens: float = 1.0):
        super().__init__()
        self.T = T
        self.threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.gamma = float(gamma)
        self.decay_factor = nn.Parameter(torch.tensor(decay_factor, dtype=torch.float32))
        self.short_weight = nn.Parameter(torch.tensor(1.0))
        self.long_weight = nn.Parameter(torch.tensor(1.0))
        self.lens = float(lens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        v_short = reset_like(x[0])
        v_long = reset_like(x[0])
        outputs = []
        for t in range(T):
            v_short = self.decay_factor[0] * v_short + self.decay_factor[1] * x[t] - 0.1 * v_long
            v_long = self.decay_factor[2] * v_long + self.decay_factor[3] * x[t] - 0.8 * v_short
            mem = self.short_weight * v_short + self.long_weight * v_long
            spike01 = SpikeFn.apply(mem, self.threshold, self.lens)
            spike = spike01 * self.threshold
            v_short = v_short - spike01.detach() * self.gamma
            v_long = v_long - spike01.detach() * self.threshold
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
