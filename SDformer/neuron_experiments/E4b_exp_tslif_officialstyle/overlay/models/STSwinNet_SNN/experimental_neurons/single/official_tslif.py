from __future__ import annotations

import copy
import math
from typing import Callable

import torch
from torch import nn
from spikingjelly.activation_based import base as sj_base

from ..base import CandidateNeuron, ensure_time_first


OFFICIAL_SOURCE_REPO = "https://github.com/kkking-kk/TS-LIF"
OFFICIAL_SOURCE_COMMIT = "a59826a6c7f62d0f16edbafdbb28db65bebd9f69"


class MemoryModule(sj_base.MemoryModule):
    """Local copy of TS-LIF's state container from SeqSNN/network/snn/TSLIF_base.py."""

    def __init__(self):
        super().__init__()
        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        assert not hasattr(self, name), f"{name} has been set as a member variable!"
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if "_memories" in self.__dict__:
            memories = self.__dict__["_memories"]
            if name in memories:
                return memories[name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        memories = self.__dict__.get("_memories")
        if memories is not None and name in memories:
            memories[name] = value
        else:
            super().__setattr__(name, value)

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        return super()._apply(fn)


@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)


@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


class atan(torch.autograd.Function):
    """Official TS-LIF surrogate, copied from SeqSNN/network/snn/surrogate.py."""

    @staticmethod
    def forward(ctx, x, alpha=2.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class OfficialTSLIFCore(MemoryModule):
    """Official TS-LIF step dynamics with SDFormer-compatible alpha shape."""

    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_function: Callable = atan.apply,
        detach_reset: bool = False,
        hard_reset: bool = False,
        step_mode: str = "s",
        k: int = 2,
        decay_factor: torch.Tensor | None = None,
        gamma: float = 0.5,
    ):
        super().__init__()
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)

        if v_reset is None:
            self.register_memory("v", 0.0)
            self.register_memory("v_s", 0.0)
        else:
            self.register_memory("v", v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.hard_reset = hard_reset
        self.k = k
        self.gamma = gamma

        for i in range(1, self.k + 1):
            self.register_memory("v" + str(i), 0.0)
        self.names = self._memories

        if decay_factor is None:
            decay_factor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float)
        self.decay_factor = nn.Parameter(decay_factor.clone().detach().float())
        self.kk = nn.Parameter(torch.tensor([0.8], dtype=torch.float))
        self.yy = nn.Parameter(torch.tensor([0.1], dtype=torch.float))
        self.alpha_s = nn.Parameter(torch.randn([1], dtype=torch.float))
        self.alpha_l = nn.Parameter(torch.randn([1], dtype=torch.float))

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        return (1.0 - spike) * v + spike * v_reset

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        return v - spike * v_threshold

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x.data, self.v)

    def neuronal_charge(self, x: torch.Tensor):
        self.names["v1"] = self.decay_factor[0] * self.names["v1"] + self.decay_factor[1] * x - self.yy * self.names["v2"]
        self.names["v2"] = self.decay_factor[2] * self.names["v2"] + self.decay_factor[3] * x - self.kk * self.names["v1"]
        self.v = self.names["v2"]
        self.v_s = self.names["v1"]

    def sl_neuronal_fire(self):
        s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
        s_l = self.surrogate_function(self.v_s - self.v_threshold, 2.0)
        return s_s, s_l

    def neuronal_reset(self, spike_s, spike_l):
        if self.detach_reset:
            spike_s = spike_s.detach()
            spike_l = spike_l.detach()
        if not self.hard_reset:
            self.names["v1"] = self.jit_soft_reset(self.names["v1"], spike_l, self.gamma)
            self.names["v2"] = self.jit_soft_reset(self.names["v2"], spike_s, self.v_threshold)
            return
        self.names["v1"] = self.jit_hard_reset(self.names["v1"], spike_l, float(self.v_reset or 0.0))
        self.names["v2"] = self.jit_hard_reset(self.names["v2"], spike_s, float(self.v_reset or 0.0))

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        s_s, s_l = self.sl_neuronal_fire()
        spike = self.alpha_s * s_s + self.alpha_l * s_l
        self.neuronal_reset(s_s, s_l)
        return spike

    def forward(self, x: torch.Tensor):
        return self.single_step_forward(x)


class TSLIFNode(CandidateNeuron):
    """SDFormerFlow wrapper around the official TS-LIF neuron dynamics."""

    official_source_repo = OFFICIAL_SOURCE_REPO
    official_source_commit = OFFICIAL_SOURCE_COMMIT

    def __init__(
        self,
        T: int,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        detach_reset: bool = False,
        hard_reset: bool = False,
        decay_factor: torch.Tensor | None = None,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.T = T
        self.core = OfficialTSLIFCore(
            v_threshold=float(v_threshold),
            v_reset=v_reset,
            surrogate_function=atan.apply,
            detach_reset=detach_reset,
            hard_reset=hard_reset,
            decay_factor=decay_factor,
            gamma=gamma,
        )

    @property
    def decay_factor(self):
        return self.core.decay_factor

    @property
    def kk(self):
        return self.core.kk

    @property
    def yy(self):
        return self.core.yy

    @property
    def alpha_s(self):
        return self.core.alpha_s

    @property
    def alpha_l(self):
        return self.core.alpha_l

    def reset_state(self) -> None:
        self.core.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        self.reset_state()
        outputs = []
        for t in range(T):
            outputs.append(self.core(x[t]))
        return torch.stack(outputs, dim=0)
