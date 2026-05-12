"""Scaffold self-contained neuron experiment overlays.

The generated experiments keep all modified code outside third_party/SDformerFlow.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


EXPERIMENTS = {
    "E0_psn_baseline": {"neuron_type": "psn", "module": None},
    "E1_exp_sn": {"neuron_type": "exp_sn", "module": "single/sn.py"},
    "E2_exp_atlif": {"neuron_type": "exp_atlif", "module": "single/atlif.py"},
    "E3_exp_lmh": {"neuron_type": "exp_lmh", "module": "single/lmh.py"},
    "E4_exp_tslif": {"neuron_type": "exp_tslif", "module": "single/tslif.py"},
    "E5_exp_tsn": {"neuron_type": "exp_tsn", "module": "single/tsn.py"},
    "F1_fused_adaptive_psn": {"neuron_type": "fused_adaptive_psn", "module": "fused/adaptive_psn.py"},
    "F2_fused_lmh_atlif": {"neuron_type": "fused_lmh_atlif", "module": "fused/lmh_atlif.py"},
    "F3_fused_adaptive_tslif": {"neuron_type": "fused_adaptive_tslif", "module": "fused/adaptive_tslif.py"},
    "F4_fused_lmh_tslif": {"neuron_type": "fused_lmh_tslif", "module": "fused/lmh_tslif.py"},
    "F5_fused_signed_hybrid": {"neuron_type": "fused_signed_hybrid", "module": "fused/signed_hybrid.py"},
}


BASE_CONFIGS = {
    "smoke": "third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml",
    "subset": "third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_subset.yml",
    "full": "third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch.yml",
}

SMOKE_SEQUENCE_FILES = {
    "train": "smoke_train_split_seq.csv",
    "valid": "smoke_valid_split_seq.csv",
}


ENTRYPOINT = '''"""Experiment-local SDFormerFlow entrypoint.

This file keeps launch-time changes inside the experiment directory. It runs the
baseline SDFormerFlow script while placing this experiment's overlay before the
baseline on sys.path.
"""

from __future__ import annotations

import argparse
import types
import os
import runpy
import sys
from pathlib import Path


TRAIN_BLOCK = """    if config["model"]["spiking_neuron"]["neuron_type"] == "if":
        neurontype = getattr(neuron, "IFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "glif":
        neurontype = GatedLIFNode
    elif config["model"]["spiking_neuron"]["neuron_type"] == "psn":
        neurontype = PSN
    elif config["model"]["spiking_neuron"]["neuron_type"] == "SLTTlif":
        neurontype = SLTTLIFNode
    else:
        raise "neurontype not implemented!"
"""

PATCHED_BLOCK = """    if config["model"]["spiking_neuron"]["neuron_type"] == "if":
        neurontype = getattr(neuron, "IFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "glif":
        neurontype = GatedLIFNode
    elif config["model"]["spiking_neuron"]["neuron_type"] == "psn":
        neurontype = PSN
    elif config["model"]["spiking_neuron"]["neuron_type"] == "SLTTlif":
        neurontype = SLTTLIFNode
    else:
        from models.STSwinNet_SNN.experimental_neurons.factory import resolve_backend_neuron_type
        neurontype = resolve_backend_neuron_type(config["model"]["spiking_neuron"]["neuron_type"])
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _install_optional_mlflow_stub() -> None:
    disabled = os.getenv("SDFORMER_USE_MLFLOW", "1").lower() in {"0", "false", "no"}
    if not disabled:
        return
    try:
        __import__("mlflow")
    except ModuleNotFoundError:
        sys.modules["mlflow"] = types.ModuleType("mlflow")


def _run_baseline(entry_name: str, config: str, extra_args: list[str]) -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    baseline_entry = baseline_root / entry_name

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    sys.argv = [str(baseline_entry), "--config", str(Path(config).resolve()), *extra_args]

    _install_optional_mlflow_stub()
    os.chdir(baseline_root)
    source = baseline_entry.read_text()
    factory = overlay_root / "models" / "STSwinNet_SNN" / "experimental_neurons" / "factory.py"
    if factory.exists():
        if TRAIN_BLOCK not in source:
            raise RuntimeError(f"Could not patch backend neuron block in {baseline_entry}")
        source = source.replace(TRAIN_BLOCK, PATCHED_BLOCK)
        code = compile(source, str(baseline_entry), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})
    else:
        runpy.run_path(str(baseline_entry), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()
    _run_baseline("__ENTRY_NAME__", args.config, extra_args)


if __name__ == "__main__":
    main()
'''


BASE = '''"""Shared utilities for experimental SDFormerFlow neurons."""

from __future__ import annotations

import torch
from torch import nn


def ensure_time_first(x: torch.Tensor, T: int | None = None) -> int:
    if x.ndim < 2:
        raise ValueError(f"expected [T, B, ...], got {tuple(x.shape)}")
    if T is not None and x.shape[0] != T:
        raise ValueError(f"expected {T} timesteps, got {x.shape[0]}")
    return x.shape[0]


def reset_like(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    return torch.full_like(x, value)


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor, lens: float):
        ctx.save_for_backward(input, threshold)
        ctx.lens = lens
        return (input >= threshold).to(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, threshold = ctx.saved_tensors
        lens = ctx.lens
        scaled = (input - threshold) / threshold.clamp_min(1e-6)
        grad = (1.0 - scaled.abs() / lens).clamp_min(0.0)
        grad_input = grad_output * grad
        grad_threshold = -(grad_output * grad).sum().view_as(threshold)
        return grad_input, grad_threshold, None


class TernarySpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor):
        ctx.save_for_backward(input, threshold)
        out = torch.zeros_like(input)
        out = torch.where(input >= threshold, torch.ones_like(out), out)
        out = torch.where(input <= -threshold, -torch.ones_like(out), out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, threshold = ctx.saved_tensors
        scaled = input / threshold.clamp_min(1e-6)
        grad = (1.0 - scaled.abs()).clamp_min(0.0)
        return grad_output * grad, None


class CandidateNeuron(nn.Module):
    backend = "torch"

    @property
    def supported_backends(self):
        return ("torch",)

    def reset_state(self) -> None:
        pass
'''


FACTORY = '''"""Factory for experiment-local SDFormerFlow neurons."""

from __future__ import annotations


EXPERIMENTAL_NEURONS = {
    "exp_sn",
    "exp_atlif",
    "exp_lmh",
    "exp_tslif",
    "exp_tsn",
    "fused_adaptive_psn",
    "fused_lmh_atlif",
    "fused_adaptive_tslif",
    "fused_lmh_tslif",
    "fused_signed_hybrid",
}


def is_experimental_neuron(neuron_type: str) -> bool:
    return neuron_type in EXPERIMENTAL_NEURONS


def build_experimental_neuron(
    neuron_type: str,
    num_steps: int,
    v_th: float = 1.0,
    v_reset=None,
    tau: float = 2.0,
    detach_reset: bool = True,
    surrogate_fun=None,
    **kwargs,
):
    if neuron_type == "exp_sn":
        from .single.sn import SNNode
        return SNNode(T=num_steps, v_threshold=v_th, decay=1.0 / tau, detach_reset=detach_reset)
    if neuron_type == "exp_atlif":
        from .single.atlif import ATLIFNode
        return ATLIFNode(T=num_steps, v_threshold=v_th, tau=1.0 / tau)
    if neuron_type == "exp_lmh":
        from .single.lmh import LMHNode
        return LMHNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "exp_tslif":
        from .single.tslif import TSLIFNode
        return TSLIFNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "exp_tsn":
        from .single.tsn import TSNNode
        return TSNNode(T=num_steps, v_threshold=v_th, decay=1.0 / tau)
    if neuron_type == "fused_adaptive_psn":
        from .fused.adaptive_psn import AdaptivePSNNode
        return AdaptivePSNNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "fused_lmh_atlif":
        from .fused.lmh_atlif import LMHATLIFNode
        return LMHATLIFNode(T=num_steps, v_threshold=v_th, tau=1.0 / tau)
    if neuron_type == "fused_adaptive_tslif":
        from .fused.adaptive_tslif import AdaptiveTSLIFNode
        return AdaptiveTSLIFNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "fused_lmh_tslif":
        from .fused.lmh_tslif import LMHTSLIFNode
        return LMHTSLIFNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "fused_signed_hybrid":
        from .fused.signed_hybrid import SignedHybridNode
        return SignedHybridNode(T=num_steps, v_threshold=v_th, decay=1.0 / tau)
    raise KeyError(f"unsupported experimental neuron: {neuron_type}")


def resolve_backend_neuron_type(neuron_type: str):
    return type(build_experimental_neuron(neuron_type=neuron_type, num_steps=1))
'''


SN = '''from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, SpikeFn, ensure_time_first, reset_like


class SNNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, decay: float = 0.25, lens: float = 1.0, detach_reset: bool = True):
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
            mem = mem * (1.0 - (spike.detach() if self.detach_reset else spike))
            outputs.append(spike * self.v_threshold)
        return torch.stack(outputs, dim=0)
'''


ATLIF = '''from __future__ import annotations

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
'''


LMH = '''from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, ensure_time_first, reset_like


class MultiLevelSpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor, levels: int):
        ctx.save_for_backward(input, threshold)
        ctx.levels = levels
        level = torch.floor(input / threshold.clamp_min(1e-6)).clamp(0, levels)
        return level * threshold

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, threshold = ctx.saved_tensors
        levels = ctx.levels
        mask = ((input >= 0.5 * threshold) & (input <= (levels + 0.5) * threshold)).to(grad_output)
        return grad_output * mask, None, None


class LMHNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, levels: int = 2, initial_mem: float = 0.0):
        super().__init__()
        self.T = T
        self.levels = int(levels)
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)), requires_grad=False)
        self.initial_mem = float(initial_mem)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.temporal_mask = nn.Parameter(torch.zeros(T, T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0], self.initial_mem * float(self.v_threshold.item()))
        outputs = []
        temporal_weight = 2.0 * self.temporal_mask.sigmoid() / T
        history_decay = self.alpha.sigmoid() + 0.5
        flat_x = x.flatten(1)
        for t in range(T):
            mixed = torch.matmul(temporal_weight[t], flat_x).view_as(x[0])
            mem = history_decay * mem.detach() + mixed
            spike = MultiLevelSpikeFn.apply(mem, self.v_threshold, self.levels)
            mem = mem - spike.detach()
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
'''


TSLIF = '''from __future__ import annotations

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
'''


TSN = '''from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, TernarySpikeFn, ensure_time_first, reset_like


class TSNNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, decay: float = 0.25, fire_ratio: float = 1.0):
        super().__init__()
        self.T = T
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)), requires_grad=False)
        self.decay = float(decay)
        self.fire_ratio = float(fire_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0])
        outputs = []
        for t in range(T):
            mem = mem * self.decay + x[t]
            spike = TernarySpikeFn.apply(mem, self.v_threshold) * self.fire_ratio
            mem = mem * (1.0 - spike.abs().detach())
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
'''


ADAPTIVE_PSN = '''from __future__ import annotations

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
'''


LMH_ATLIF = '''from __future__ import annotations

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
'''


ADAPTIVE_TSLIF = '''from __future__ import annotations

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
'''


LMH_TSLIF = '''from __future__ import annotations

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
'''


SIGNED_HYBRID = '''from __future__ import annotations

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
'''


SPKING_MODULES_IMPORT = "from models.STSwinNet_SNN.Spiking_submodules import *\n"
SPKING_MODULES_EXPERIMENTAL_IMPORT = (
    "from models.STSwinNet_SNN.Spiking_submodules import *\n"
    "from models.STSwinNet_SNN.experimental_neurons.factory import build_experimental_neuron, is_experimental_neuron\n"
)

SPKING_MODULES_ASSERT = '        assert neuron_type in ["lif", "if", "plif", "SLTTlif", "glif", "psn"]\n'
SPKING_MODULES_BRANCH = '''        if is_experimental_neuron(neuron_type):
            self.spiking_neuron = build_experimental_neuron(
                neuron_type=neuron_type,
                num_steps=num_steps,
                v_th=v_th,
                v_reset=v_reset,
                tau=tau,
                detach_reset=detach_reset,
                surrogate_fun=surrogate_fun,
            )
            return
        assert neuron_type in ["lif", "if", "plif", "SLTTlif", "glif", "psn"]
'''


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def resolve_data_path(repo_root: Path, config: dict) -> Path:
    data_path = Path(config["data"]["path"])
    if data_path.is_absolute():
        return data_path
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    return (baseline_root / data_path).resolve()


def write_smoke_sequence_lists(repo_root: Path) -> dict[str, Path]:
    smoke_config = load_config(repo_root / BASE_CONFIGS["smoke"])
    sequence_dir = resolve_data_path(repo_root, smoke_config) / "sequence_lists"
    output_dir = repo_root / "neuron_experiments" / "_templates"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split, file_name in SMOKE_SEQUENCE_FILES.items():
        source = sequence_dir / f"{split}_split_seq.csv"
        first_sample = source.read_text().splitlines()[0]
        target = output_dir / file_name
        target.write_text(f"{first_sample}\n")
        paths[split] = target
    return paths


def experiment_config(repo_root: Path, base: str, exp_id: str, neuron_type: str) -> dict:
    config = load_config(repo_root / BASE_CONFIGS[base])
    config["experiment"] = exp_id
    config.setdefault("runtime", {})
    config["runtime"]["snn_backend"] = "torch"
    config["spiking_neuron"]["neuron_type"] = neuron_type
    config.setdefault("experimental_neuron", {})
    config["experimental_neuron"]["enabled"] = neuron_type != "psn"
    if base == "smoke":
        smoke_lists = write_smoke_sequence_lists(repo_root)
        config["data"]["sequence_list_overrides"] = {
            split: str(path) for split, path in smoke_lists.items()
        }
        config.setdefault("test", {})
        config["test"]["sample"] = 1
        config["test"]["n_valid"] = 1
    return config


def write_entrypoints(exp_root: Path) -> None:
    write_text(exp_root / "entrypoints" / "train.py", ENTRYPOINT.replace("__ENTRY_NAME__", "train_flow_parallel_supervised_SNN.py"))
    write_text(exp_root / "entrypoints" / "eval.py", ENTRYPOINT.replace("__ENTRY_NAME__", "eval_DSEC_flow_SNN.py"))


def write_overlay_common(repo_root: Path, exp_root: Path) -> None:
    baseline_modules = repo_root / "third_party" / "SDformerFlow" / "models" / "STSwinNet_SNN" / "Spiking_modules.py"
    modules_text = baseline_modules.read_text()
    modules_text = modules_text.replace(SPKING_MODULES_IMPORT, SPKING_MODULES_EXPERIMENTAL_IMPORT)
    modules_text = modules_text.replace(SPKING_MODULES_ASSERT, SPKING_MODULES_BRANCH)
    write_text(exp_root / "overlay" / "models" / "__init__.py", "from pkgutil import extend_path\n\n__path__ = extend_path(__path__, __name__)\n")
    write_text(exp_root / "overlay" / "models" / "STSwinNet_SNN" / "Spiking_modules.py", modules_text)
    base_dir = exp_root / "overlay" / "models" / "STSwinNet_SNN" / "experimental_neurons"
    write_text(base_dir / "__init__.py", "")
    write_text(base_dir / "base.py", BASE)
    write_text(base_dir / "factory.py", FACTORY)


def write_neuron_modules(exp_root: Path) -> None:
    base_dir = exp_root / "overlay" / "models" / "STSwinNet_SNN" / "experimental_neurons"
    write_text(base_dir / "single" / "__init__.py", "")
    write_text(base_dir / "single" / "sn.py", SN)
    write_text(base_dir / "single" / "atlif.py", ATLIF)
    write_text(base_dir / "single" / "lmh.py", LMH)
    write_text(base_dir / "single" / "tslif.py", TSLIF)
    write_text(base_dir / "single" / "tsn.py", TSN)
    write_text(base_dir / "fused" / "__init__.py", "")
    write_text(base_dir / "fused" / "adaptive_psn.py", ADAPTIVE_PSN)
    write_text(base_dir / "fused" / "lmh_atlif.py", LMH_ATLIF)
    write_text(base_dir / "fused" / "adaptive_tslif.py", ADAPTIVE_TSLIF)
    write_text(base_dir / "fused" / "lmh_tslif.py", LMH_TSLIF)
    write_text(base_dir / "fused" / "signed_hybrid.py", SIGNED_HYBRID)


def scaffold_experiment(repo_root: Path, exp_id: str, neuron_type: str) -> None:
    exp_root = repo_root / "neuron_experiments" / exp_id
    exp_root.mkdir(parents=True, exist_ok=True)
    write_text(
        exp_root / "README.md",
        f"# {exp_id}\n\nNeuron type: `{neuron_type}`.\n\nRun smoke:\n\n```bash\npython neuron_experiments/{exp_id}/entrypoints/train.py --config neuron_experiments/{exp_id}/configs/smoke.yml\n```\n",
    )
    write_entrypoints(exp_root)
    for level in BASE_CONFIGS:
        write_config(exp_root / "configs" / f"{level}.yml", experiment_config(repo_root, level, exp_id, neuron_type))
    write_text(exp_root / "results" / "metrics.md", "# Metrics\n\n| run | config | status | train loss | valid AEE | activity rate | notes |\n|---|---|---|---:|---:|---:|---|\n")
    write_text(exp_root / "results" / "run_commands.md", "# Run Commands\n\n")
    if neuron_type == "psn":
        (exp_root / "overlay").mkdir(exist_ok=True)
        return
    write_overlay_common(repo_root, exp_root)
    write_neuron_modules(exp_root)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for exp_id, meta in EXPERIMENTS.items():
        scaffold_experiment(repo_root, exp_id, meta["neuron_type"])


if __name__ == "__main__":
    main()
