from __future__ import annotations

import math
import torch
from torch import nn

from ..base import CandidateNeuron, SpikeFn, ensure_time_first


class AdaptivePSNNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, lens: float = 1.0):
        super().__init__()
        self.T = T
        self.threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.lens = float(lens)
        self.weight = nn.Parameter(torch.empty(T, T))
        self.bias = nn.Parameter(torch.full((T, 1), -float(v_threshold)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensure_time_first(x, self.T)
        h = torch.addmm(self.bias, self.weight, x.flatten(1))
        spike = SpikeFn.apply(h, self.threshold, self.lens) * self.threshold
        return spike.view_as(x)
