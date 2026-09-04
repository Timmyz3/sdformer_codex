from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class ATLIFNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, tau: float = 1.0, lens: float = 1.0):
        super().__init__()
        self.T = T
        self.thresh = nn.Parameter(torch.tensor(float(v_threshold)))
        self.tau = float(tau)
        self.lens = float(lens)
        self.firing_rate = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0])
        outputs = []
        for t in range(T):
            mem = mem * self.tau + x[t]
            spike01 = SpikeFn.apply(mem, self.thresh, self.lens)
            spike = spike01 * self.thresh
            mem = (1.0 - spike / self.thresh.detach().clamp_min(1e-6)) * mem
            outputs.append(spike)
        out = torch.stack(outputs, dim=0)
        with torch.no_grad():
            self.firing_rate = (out / self.thresh.detach().clamp_min(1e-6)).mean().item()
        return out
