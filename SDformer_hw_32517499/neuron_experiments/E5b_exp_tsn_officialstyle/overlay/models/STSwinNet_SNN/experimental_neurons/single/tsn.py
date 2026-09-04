from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, ensure_time_first, reset_like


OFFICIAL_SOURCE_REPO = "https://github.com/yfguo91/Ternary-Spike"
OFFICIAL_SOURCE_COMMIT = "2aca58747f01d7960cb6f0284665bbb353d35aab"


def official_ternary_spike_activation(x: torch.Tensor, binary: bool = False, temp: float = 1.0) -> torch.Tensor:
    """Official Ternary-Spike STE from models/spike_layer.py."""
    if binary:
        out_s = torch.gt(x, 0.5)
        out_bp = torch.clamp(x, 0, 1)
        return (out_s.float() - out_bp).detach() + out_bp

    out_s = torch.sign(x)
    out_s = torch.where(torch.abs(x) < 0.5, torch.zeros_like(out_s), out_s)
    out_bp = torch.clamp(x, -1, 1)
    return (out_s.float() - out_bp).detach() + out_bp


def official_mem_update(
    x_in: torch.Tensor,
    mem: torch.Tensor,
    v_threshold: torch.Tensor | float,
    decay: float,
    fire_ratio: torch.Tensor | float,
    temp: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official Ternary-Spike membrane update from models/spike_layer.py."""
    mem = mem * decay + x_in
    spike = official_ternary_spike_activation(mem / v_threshold, temp=temp)
    mem = mem * (1 - torch.abs(spike))
    spike = spike * fire_ratio
    return mem, spike


class TSNNode(CandidateNeuron):
    official_source_repo = OFFICIAL_SOURCE_REPO
    official_source_commit = OFFICIAL_SOURCE_COMMIT

    def __init__(
        self,
        T: int,
        v_threshold: float = 1.0,
        decay: float = 0.25,
        fire_ratio: float = 1.0,
        temp: float = 3.0,
    ):
        super().__init__()
        self.T = T
        self.v_threshold = float(v_threshold)
        self.decay = float(decay)
        self.temp = float(temp)
        self.fire_ratio = nn.Parameter(torch.tensor(float(fire_ratio)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0])
        outputs = []
        for t in range(T):
            mem, spike = official_mem_update(
                x_in=x[t],
                mem=mem,
                v_threshold=self.v_threshold,
                decay=self.decay,
                fire_ratio=self.fire_ratio,
                temp=self.temp,
            )
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
