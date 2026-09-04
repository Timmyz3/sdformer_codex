"""Install adaptive ternary PSN on selected SDFormer attention neurons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .adaptive_ternary_psn import AdaptiveTernaryPSN


@dataclass(frozen=True)
class AdaptiveTernaryConfig:
    enabled: bool = False
    target: str = "qk"
    stage_selection: str = "all"
    theta_init: float = 1.0
    learn_threshold: bool = True
    min_threshold: float = 1.0e-4
    dead_zone: float = 0.5
    output_scale: str = "threshold"
    target_rate: float = 0.0
    activity_momentum: float = 0.99
    reg_lambda: float = 0.0


def config_from_dict(raw: dict | None) -> AdaptiveTernaryConfig:
    raw = raw or {}
    return AdaptiveTernaryConfig(
        enabled=bool(raw.get("enabled", False)),
        target=str(raw.get("target", "qk")),
        stage_selection=str(raw.get("stage_selection", "all")),
        theta_init=float(raw.get("theta_init", 1.0)),
        learn_threshold=bool(raw.get("learn_threshold", True)),
        min_threshold=float(raw.get("min_threshold", 1.0e-4)),
        dead_zone=float(raw.get("dead_zone", 0.5)),
        output_scale=str(raw.get("output_scale", "threshold")),
        target_rate=float(raw.get("target_rate", 0.0)),
        activity_momentum=float(raw.get("activity_momentum", 0.99)),
        reg_lambda=float(raw.get("reg_lambda", 0.0)),
    )


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def _iter_attention_modules(model: nn.Module, stage_selection: str) -> Iterable[tuple[str, nn.Module]]:
    if not hasattr(model, "sttmultires_unet"):
        return []
    swin3d = model.sttmultires_unet.encoders.swin3d
    layers = list(swin3d.layers)
    if stage_selection == "all":
        stage_ids = range(len(layers))
    elif stage_selection.startswith("stage"):
        stage_ids = [int(stage_selection.replace("stage", ""))]
    elif stage_selection == "layer0_only":
        stage_ids = [0]
    else:
        raise ValueError("stage_selection must be all, layer0_only, or stage{index}")

    pairs: list[tuple[str, nn.Module]] = []
    for stage_idx in stage_ids:
        stage = layers[stage_idx]
        for block_idx, block in enumerate(stage.swin_blocks):
            pairs.append((f"layers.{stage_idx}.swin_blocks.{block_idx}.attn", block.attn))
    return pairs


def _target_names(target: str) -> tuple[str, ...]:
    if target == "qk":
        return ("sn_q", "sn_k")
    if target == "q":
        return ("sn_q",)
    if target == "k":
        return ("sn_k",)
    raise ValueError("target must be qk, q, or k")


def install_adaptive_ternary_qk(model: nn.Module, raw_config: dict | None) -> list[str]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []

    installed: list[str] = []
    for attn_name, attn in _iter_attention_modules(model, cfg.stage_selection):
        for child_name in _target_names(cfg.target):
            wrapper = getattr(attn, child_name)
            if not hasattr(wrapper, "spiking_neuron"):
                raise TypeError(f"{attn_name}.{child_name} has no spiking_neuron")
            current = wrapper.spiking_neuron
            if isinstance(current, AdaptiveTernaryPSN):
                installed.append(f"{attn_name}.{child_name}")
                continue
            device = _module_device(wrapper)
            wrapper.spiking_neuron = AdaptiveTernaryPSN(
                T=getattr(current, "T", getattr(wrapper, "num_steps", 10)),
                base_psn=current,
                theta_init=cfg.theta_init,
                learn_threshold=cfg.learn_threshold,
                min_threshold=cfg.min_threshold,
                dead_zone=cfg.dead_zone,
                output_scale=cfg.output_scale,
                target_rate=cfg.target_rate,
                activity_momentum=cfg.activity_momentum,
            ).to(device)
            installed.append(f"{attn_name}.{child_name}")
    return installed


def iter_adaptive_ternary(model: nn.Module) -> Iterable[tuple[str, AdaptiveTernaryPSN]]:
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveTernaryPSN):
            yield name, module


def adaptive_ternary_regularization(model: nn.Module, raw_config: dict | None) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled or cfg.reg_lambda <= 0 or cfg.target_rate <= 0:
        return None
    losses = [module.activity_regularization() for _, module in iter_adaptive_ternary(model)]
    if not losses:
        return None
    return torch.stack(losses).mean() * cfg.reg_lambda


def adaptive_ternary_summary(model: nn.Module) -> dict[str, float | int]:
    modules = list(iter_adaptive_ternary(model))
    if not modules:
        return {"num_modules": 0}
    theta = [float(module.theta.detach().cpu()) for _, module in modules]
    activity = [float(module.running_activity.detach().cpu()) for _, module in modules]
    pos = [float(module.running_pos_rate.detach().cpu()) for _, module in modules]
    neg = [float(module.running_neg_rate.detach().cpu()) for _, module in modules]
    return {
        "num_modules": len(modules),
        "theta_mean": sum(theta) / len(theta),
        "theta_min": min(theta),
        "theta_max": max(theta),
        "activity_mean": sum(activity) / len(activity),
        "pos_mean": sum(pos) / len(pos),
        "neg_mean": sum(neg) / len(neg),
    }

