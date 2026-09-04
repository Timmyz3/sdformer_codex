"""Q/K attention control modules for ablation-only H4 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


class ZeroLike(nn.Module):
    """Return a zero tensor with the same shape as the wrapped neuron input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


@dataclass(frozen=True)
class QKControlConfig:
    enabled: bool = False
    target: str = "qk"
    stage_selection: str = "all"
    mode: str = "zero"


def config_from_dict(raw: dict | None) -> QKControlConfig:
    raw = raw or {}
    return QKControlConfig(
        enabled=bool(raw.get("enabled", False)),
        target=str(raw.get("target", "qk")),
        stage_selection=str(raw.get("stage_selection", "all")),
        mode=str(raw.get("mode", "zero")),
    )


def _iter_attention_modules(model: nn.Module, stage_selection: str) -> Iterable[tuple[str, nn.Module]]:
    if not hasattr(model, "sttmultires_unet"):
        return []
    swin3d = model.sttmultires_unet.encoders.swin3d
    layers = list(swin3d.layers)
    if stage_selection == "all":
        stage_ids = range(len(layers))
    elif stage_selection == "layer0_only":
        stage_ids = [0]
    elif stage_selection.startswith("stage"):
        stage_ids = [int(stage_selection.replace("stage", ""))]
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


def install_qk_control(model: nn.Module, raw_config: dict | None) -> list[str]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []
    if cfg.mode != "zero":
        raise ValueError("H4 qk_control.mode currently supports only zero")

    installed: list[str] = []
    for attn_name, attn in _iter_attention_modules(model, cfg.stage_selection):
        for child_name in _target_names(cfg.target):
            wrapper = getattr(attn, child_name)
            if not hasattr(wrapper, "spiking_neuron"):
                raise TypeError(f"{attn_name}.{child_name} has no spiking_neuron")
            wrapper.spiking_neuron = ZeroLike()
            installed.append(f"{attn_name}.{child_name}")
    return installed
