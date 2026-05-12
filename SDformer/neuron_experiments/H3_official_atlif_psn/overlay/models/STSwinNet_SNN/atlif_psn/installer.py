"""Install H3 ATLIF-PSN neurons on SDFormerFlow attention modules."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Iterable

import torch
import torch.nn as nn

from .atlif_psn import ATLIFPSN


@dataclass(frozen=True)
class ATLIFPSNConfig:
    enabled: bool = False
    target: str = "qk"
    stage_selection: str = "layer0_only"
    target_paths: tuple[str, ...] = ()
    threshold_init: float = 1.0
    threshold_eta: float = 1.0e-3
    threshold_lr_scale: float = 1.0
    min_threshold: float | None = 1.0e-3
    max_threshold: float | None = None
    activity_eta: float = 0.0


def config_from_dict(raw: dict | None) -> ATLIFPSNConfig:
    raw = raw or {}
    target_paths = raw.get("target_paths", raw.get("extra_target_paths", ()))
    if isinstance(target_paths, str):
        target_paths = [target_paths]
    return ATLIFPSNConfig(
        enabled=bool(raw.get("enabled", False)),
        target=str(raw.get("target", "qk")),
        stage_selection=str(raw.get("stage_selection", "layer0_only")),
        target_paths=tuple(str(path) for path in target_paths),
        threshold_init=float(raw.get("threshold_init", 1.0)),
        threshold_eta=float(raw.get("threshold_eta", 1.0e-3)),
        threshold_lr_scale=float(raw.get("threshold_lr_scale", 1.0)),
        min_threshold=None if raw.get("min_threshold") is None else float(raw.get("min_threshold", 1.0e-3)),
        max_threshold=None if raw.get("max_threshold") is None else float(raw.get("max_threshold")),
        activity_eta=float(raw.get("activity_eta", 0.0)),
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
    if target in {"none", "custom"}:
        return ()
    if target == "qk":
        return ("sn_q", "sn_k")
    if target == "q":
        return ("sn_q",)
    if target == "k":
        return ("sn_k",)
    raise ValueError("target must be qk, q, k, none, or custom")


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    modules = dict(model.named_modules())
    if path not in modules:
        raise KeyError(f"Could not find ATLIFPSN target path: {path}")
    return modules[path]


def _install_on_wrapper(wrapper: nn.Module, cfg: ATLIFPSNConfig, name: str) -> bool:
    if not hasattr(wrapper, "spiking_neuron"):
        raise TypeError(f"{name} has no spiking_neuron")
    current = wrapper.spiking_neuron
    if isinstance(current, ATLIFPSN):
        return False
    device = _module_device(wrapper)
    wrapper.spiking_neuron = ATLIFPSN(
        T=getattr(current, "T", getattr(wrapper, "num_steps", 10)),
        base_psn=current,
        thresh=cfg.threshold_init,
        sparsity_eta=cfg.threshold_eta,
    ).to(device)
    return True


def install_atlif_psn_qk(model: nn.Module, raw_config: dict | None) -> list[str]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []

    installed: list[str] = []
    seen: set[str] = set()
    for attn_name, attn in _iter_attention_modules(model, cfg.stage_selection):
        for child_name in _target_names(cfg.target):
            full_name = f"{attn_name}.{child_name}"
            if full_name in seen:
                continue
            seen.add(full_name)
            wrapper = getattr(attn, child_name)
            _install_on_wrapper(wrapper, cfg, full_name)
            installed.append(full_name)
    for path in cfg.target_paths:
        if path in seen:
            continue
        seen.add(path)
        wrapper = _get_module_by_path(model, path)
        _install_on_wrapper(wrapper, cfg, path)
        installed.append(path)
    return installed


def apply_trainable_mode(model: nn.Module, raw_config: dict | None) -> dict[str, int | str]:
    raw = raw_config or {}
    mode = str(raw.get("trainable", "all"))
    if mode == "all":
        return {"mode": mode, "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad)}
    if mode not in {"atlif_only", "threshold_only"}:
        raise ValueError("atlif_psn.trainable must be all, atlif_only, or threshold_only")

    for param in model.parameters():
        param.requires_grad_(False)

    for _, module in iter_atlif_psn(model):
        if mode == "atlif_only":
            for param in module.parameters():
                param.requires_grad_(True)
        else:
            module.thresh.requires_grad_(True)

    return {"mode": mode, "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad)}


def iter_atlif_psn(model: nn.Module) -> Iterable[tuple[str, ATLIFPSN]]:
    for name, module in model.named_modules():
        if isinstance(module, ATLIFPSN):
            yield name, module


def regularize_activity(model: nn.Module, raw_config: dict | None) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled or cfg.activity_eta == 0.0:
        return None
    losses = [module.act_value for _, module in iter_atlif_psn(model) if torch.is_tensor(module.act_value)]
    if not losses:
        return None
    return torch.stack(losses).sum() * cfg.activity_eta


def threshold_update(model: nn.Module, lr: float, raw_config: dict | None) -> dict[str, float | int]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return {"num_modules": 0}
    updates: list[float] = []
    for _, module in iter_atlif_psn(model):
        update_value = module.update_value
        if isinstance(update_value, Number):
            update_tensor = module.thresh.detach().new_tensor(float(update_value))
        elif torch.is_tensor(update_value):
            update_tensor = update_value.detach().to(device=module.thresh.device, dtype=module.thresh.dtype)
        else:
            continue
        updates.append(float(update_tensor.detach().cpu()))
        module.thresh.data = module.thresh.data + update_tensor * float(lr) * cfg.threshold_lr_scale
        if cfg.min_threshold is not None or cfg.max_threshold is not None:
            min_value = -float("inf") if cfg.min_threshold is None else cfg.min_threshold
            max_value = float("inf") if cfg.max_threshold is None else cfg.max_threshold
            module.thresh.data.clamp_(min=min_value, max=max_value)
        module.update_value = 0.0
    summary = atlif_psn_summary(model)
    summary["raw_update_mean"] = sum(updates) / len(updates) if updates else 0.0
    summary["effective_update_mean"] = summary["raw_update_mean"] * float(lr) * cfg.threshold_lr_scale
    return summary


def atlif_psn_summary(model: nn.Module) -> dict[str, float | int]:
    modules = list(iter_atlif_psn(model))
    if not modules:
        return {"num_modules": 0}
    thresholds = [float(module.thresh.detach().cpu()) for _, module in modules]
    rates = [float(module.r) for _, module in modules]
    updates = [float(module.update_value) for _, module in modules]
    return {
        "num_modules": len(modules),
        "threshold_mean": sum(thresholds) / len(thresholds),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "firing_mean": sum(rates) / len(rates),
        "update_mean": sum(updates) / len(updates),
    }
