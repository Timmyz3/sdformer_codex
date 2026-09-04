from __future__ import annotations

import torch
from torch import nn

from ..single.sn import SNNode
from ..single.tsn import TSNNode


class SignedHybridNode(nn.Module):
    backend = "torch"

    @property
    def supported_backends(self):
        return ("torch",)

    def __init__(self, T: int, v_threshold: float = 1.0, decay: float = 0.25, gate_init: float = 0.0):
        super().__init__()
        self.binary = SNNode(T=T, v_threshold=v_threshold, decay=decay)
        self.signed = TSNNode(T=T, v_threshold=v_threshold, decay=decay)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate.sigmoid()
        return (1.0 - gate) * self.binary(x) + gate * self.signed(x)
