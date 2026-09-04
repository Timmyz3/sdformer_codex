"""Partial sparse gates for selected SDFormerFlow spiking nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


DEFAULT_TARGET_LAYERS = (
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.attn.proj_sn",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.attn.proj_sn",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2",
)


@dataclass(frozen=True)
class SparseGateConfig:
    enabled: bool = False
    target_layers: tuple[str, ...] = DEFAULT_TARGET_LAYERS
    init_logit: float = 2.0
    threshold: float = 0.5
    reg_lambda: float = 0.0
    freeze_backbone: bool = False


def config_from_dict(raw: dict | None) -> SparseGateConfig:
    raw = raw or {}
    targets = tuple(raw.get("target_layers") or DEFAULT_TARGET_LAYERS)
    return SparseGateConfig(
        enabled=bool(raw.get("enabled", False)),
        target_layers=targets,
        init_logit=float(raw.get("init_logit", 2.0)),
        threshold=float(raw.get("threshold", 0.5)),
        reg_lambda=float(raw.get("reg_lambda", 0.0)),
        freeze_backbone=bool(raw.get("freeze_backbone", False)),
    )


class HardSparseGate(nn.Module):
    """Wrap a spiking node with a scalar hard straight-through gate."""

    def __init__(self, base: nn.Module, init_logit: float = 2.0, threshold: float = 0.5):
        super().__init__()
        self.base = base
        self.gate_logit = nn.Parameter(torch.tensor(float(init_logit)))
        self.threshold = float(threshold)

    @property
    def gate_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def hard_gate(self) -> torch.Tensor:
        prob = self.gate_probability
        hard = (prob >= self.threshold).to(prob.dtype)
        return hard.detach() - prob.detach() + prob

    def forward(self, x):
        return self.base(x) * self.hard_gate()


def _module_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def install_sparse_gates(model: nn.Module, raw_config: dict | None) -> list[str]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []

    named_modules = dict(model.named_modules())
    missing = [name for name in cfg.target_layers if name not in named_modules]
    if missing:
        raise ValueError("G1 sparse gate target layers missing: " + ", ".join(missing))

    installed = []
    for name in cfg.target_layers:
        module = named_modules[name]
        if not hasattr(module, "spiking_neuron"):
            raise TypeError(f"Target layer does not look like Spiking_neuron wrapper: {name}")
        current = module.spiking_neuron
        if isinstance(current, HardSparseGate):
            installed.append(name)
            continue
        device = _module_device(module)
        module.spiking_neuron = HardSparseGate(
            current,
            init_logit=cfg.init_logit,
            threshold=cfg.threshold,
        ).to(device)
        installed.append(name)

    if cfg.freeze_backbone:
        freeze_non_gate_parameters(model)
    return installed


def iter_sparse_gates(model: nn.Module) -> Iterable[tuple[str, HardSparseGate]]:
    for name, module in model.named_modules():
        if isinstance(module, HardSparseGate):
            yield name, module


def sparse_gate_regularization(model: nn.Module, raw_config: dict | None) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled or cfg.reg_lambda <= 0:
        return None
    gates = [gate.gate_probability for _, gate in iter_sparse_gates(model)]
    if not gates:
        return None
    return torch.stack(gates).mean() * cfg.reg_lambda


def sparse_gate_summary(model: nn.Module) -> dict[str, float | int]:
    probs = [float(gate.gate_probability.detach().cpu().item()) for _, gate in iter_sparse_gates(model)]
    if not probs:
        return {"num_gates": 0, "mean_prob": 0.0, "open_gates": 0}
    open_gates = sum(prob >= 0.5 for prob in probs)
    return {
        "num_gates": len(probs),
        "mean_prob": sum(probs) / len(probs),
        "min_prob": min(probs),
        "max_prob": max(probs),
        "open_gates": int(open_gates),
    }


def freeze_non_gate_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for _, gate in iter_sparse_gates(model):
        gate.gate_logit.requires_grad = True
