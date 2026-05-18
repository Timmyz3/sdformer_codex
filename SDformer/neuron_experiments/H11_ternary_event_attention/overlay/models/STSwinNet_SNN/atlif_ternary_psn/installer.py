"""Install H9 ATLIF ternary/binary PSN neurons on SDFormerFlow modules."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from numbers import Number
from typing import Any, Iterable

import torch
import torch.nn as nn

from .atlif_ternary_psn import ATLIFTernaryPSN


@dataclass(frozen=True)
class ATLIFTernaryPSNConfig:
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
    negative_threshold_scale: float = 5.0
    target_rate: float | None = None
    target_rate_eta: float = 0.0
    output_mode: str = "ternary"
    preserve_loaded_threshold: bool = False
    stage_activity_eta: Any = None
    stage_max_threshold: Any = None
    stage_negative_threshold_scale: Any = None
    stage_target_rate: Any = None
    target_groups: tuple[dict[str, Any], ...] = ()


def config_from_dict(raw: dict | None) -> ATLIFTernaryPSNConfig:
    raw = raw or {}
    target_paths = raw.get("target_paths", raw.get("extra_target_paths", ()))
    if isinstance(target_paths, str):
        target_paths = [target_paths]
    target_groups = raw.get("target_groups", ())
    if isinstance(target_groups, dict):
        target_groups = [target_groups]
    return ATLIFTernaryPSNConfig(
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
        negative_threshold_scale=float(raw.get("negative_threshold_scale", 5.0)),
        target_rate=None if raw.get("target_rate") is None else float(raw.get("target_rate")),
        target_rate_eta=float(raw.get("target_rate_eta", 0.0)),
        output_mode=str(raw.get("output_mode", "ternary")),
        preserve_loaded_threshold=bool(raw.get("preserve_loaded_threshold", False)),
        stage_activity_eta=raw.get("stage_activity_eta"),
        stage_max_threshold=raw.get("stage_max_threshold"),
        stage_negative_threshold_scale=raw.get("stage_negative_threshold_scale"),
        stage_target_rate=raw.get("stage_target_rate"),
        target_groups=tuple(dict(group) for group in target_groups),
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
        raise KeyError(f"Could not find ATLIFTernaryPSN target path: {path}")
    return modules[path]


def _stage_value(raw_value: Any, stage_idx: int | None, default: Any) -> Any:
    if raw_value is None or stage_idx is None:
        return default
    if isinstance(raw_value, (list, tuple)):
        return raw_value[stage_idx] if stage_idx < len(raw_value) else default
    if isinstance(raw_value, dict):
        return raw_value.get(stage_idx, raw_value.get(str(stage_idx), default))
    return raw_value


def _install_on_wrapper(
    wrapper: nn.Module,
    cfg: ATLIFTernaryPSNConfig,
    name: str,
    stage_idx: int | None = None,
) -> bool:
    if not hasattr(wrapper, "spiking_neuron"):
        raise TypeError(f"{name} has no spiking_neuron")
    current = wrapper.spiking_neuron
    if isinstance(current, ATLIFTernaryPSN):
        return False
    device = _module_device(wrapper)
    initial_threshold = cfg.threshold_init
    if cfg.preserve_loaded_threshold and hasattr(current, "thresh"):
        loaded_threshold = getattr(current, "thresh")
        if isinstance(loaded_threshold, torch.Tensor):
            initial_threshold = float(loaded_threshold.detach().float().mean().cpu())
        elif isinstance(loaded_threshold, Number):
            initial_threshold = float(loaded_threshold)
    wrapper.spiking_neuron = ATLIFTernaryPSN(
        T=getattr(current, "T", getattr(wrapper, "num_steps", 10)),
        base_psn=current,
        thresh=initial_threshold,
        sparsity_eta=cfg.threshold_eta,
        negative_threshold_scale=float(
            _stage_value(cfg.stage_negative_threshold_scale, stage_idx, cfg.negative_threshold_scale)
        ),
        activity_eta=float(_stage_value(cfg.stage_activity_eta, stage_idx, cfg.activity_eta)),
        min_threshold=cfg.min_threshold,
        max_threshold=None
        if _stage_value(cfg.stage_max_threshold, stage_idx, cfg.max_threshold) is None
        else float(_stage_value(cfg.stage_max_threshold, stage_idx, cfg.max_threshold)),
        threshold_lr_scale=cfg.threshold_lr_scale,
        target_rate=None
        if _stage_value(cfg.stage_target_rate, stage_idx, cfg.target_rate) is None
        else float(_stage_value(cfg.stage_target_rate, stage_idx, cfg.target_rate)),
        target_rate_eta=cfg.target_rate_eta,
        output_mode=cfg.output_mode,
    ).to(device)
    return True


def _config_for_group(base: ATLIFTernaryPSNConfig, group: dict[str, Any]) -> ATLIFTernaryPSNConfig:
    overrides: dict[str, Any] = {}
    for key in (
        "threshold_init",
        "threshold_eta",
        "threshold_lr_scale",
        "min_threshold",
        "max_threshold",
        "activity_eta",
        "negative_threshold_scale",
        "target_rate",
        "target_rate_eta",
        "output_mode",
    ):
        if key in group:
            value = group[key]
            if value is None:
                overrides[key] = None
            elif key in {"threshold_init", "threshold_eta", "threshold_lr_scale", "min_threshold", "max_threshold", "activity_eta", "negative_threshold_scale", "target_rate", "target_rate_eta"}:
                overrides[key] = float(value)
            else:
                overrides[key] = value
    return replace(base, **overrides)


def install_atlif_ternary_psn(model: nn.Module, raw_config: dict | None) -> list[str]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []

    installed: list[str] = []
    seen: set[str] = set()
    for attn_name, attn in _iter_attention_modules(model, cfg.stage_selection):
        stage_idx = int(attn_name.split(".", 2)[1])
        for child_name in _target_names(cfg.target):
            full_name = f"{attn_name}.{child_name}"
            if full_name in seen:
                continue
            seen.add(full_name)
            wrapper = getattr(attn, child_name)
            _install_on_wrapper(wrapper, cfg, full_name, stage_idx=stage_idx)
            installed.append(full_name)
    for path in cfg.target_paths:
        if path in seen:
            continue
        seen.add(path)
        wrapper = _get_module_by_path(model, path)
        _install_on_wrapper(wrapper, cfg, path, stage_idx=None)
        installed.append(path)
    for group_index, group in enumerate(cfg.target_groups):
        group_cfg = _config_for_group(cfg, group)
        paths = group.get("paths", ())
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            path = str(path)
            if path in seen:
                continue
            seen.add(path)
            wrapper = _get_module_by_path(model, path)
            _install_on_wrapper(wrapper, group_cfg, path, stage_idx=None)
            installed.append(f"group{group_index}:{path}")
    return installed


def apply_trainable_mode(model: nn.Module, raw_config: dict | None) -> dict[str, int | str]:
    raw = raw_config or {}
    mode = str(raw.get("trainable", "all"))
    if mode == "all":
        return {"mode": mode, "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad)}
    if mode not in {"atlif_only", "threshold_only"}:
        raise ValueError("atlif_ternary_psn.trainable must be all, atlif_only, or threshold_only")

    for param in model.parameters():
        param.requires_grad_(False)

    for _, module in iter_atlif_ternary_psn(model):
        if mode == "atlif_only":
            for param in module.parameters():
                param.requires_grad_(True)
        else:
            module.thresh.requires_grad_(True)

    return {"mode": mode, "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad)}


def iter_atlif_ternary_psn(model: nn.Module) -> Iterable[tuple[str, ATLIFTernaryPSN]]:
    for name, module in model.named_modules():
        if isinstance(module, ATLIFTernaryPSN):
            yield name, module


def regularize_activity(model: nn.Module, raw_config: dict | None) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled or cfg.activity_eta == 0.0:
        if not any(getattr(module, "activity_eta", 0.0) for _, module in iter_atlif_ternary_psn(model)):
            return None
    losses = [
        module.act_value * float(getattr(module, "activity_eta", cfg.activity_eta))
        for _, module in iter_atlif_ternary_psn(model)
        if torch.is_tensor(module.act_value) and float(getattr(module, "activity_eta", cfg.activity_eta)) != 0.0
    ]
    if not losses:
        return None
    return torch.stack(losses).sum()


def threshold_update(model: nn.Module, lr: float, raw_config: dict | None) -> dict[str, float | int]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return {"num_modules": 0}
    updates: list[float] = []
    feedbacks: list[float] = []
    for _, module in iter_atlif_ternary_psn(model):
        update_value = module.update_value
        if isinstance(update_value, Number):
            update_tensor = module.thresh.detach().new_tensor(float(update_value))
        elif torch.is_tensor(update_value):
            update_tensor = update_value.detach().to(device=module.thresh.device, dtype=module.thresh.dtype)
        else:
            continue
        target_rate = getattr(module, "target_rate", cfg.target_rate)
        target_feedback = 0.0
        if target_rate is not None and float(getattr(module, "target_rate_eta", cfg.target_rate_eta)) != 0.0:
            target_feedback = float(getattr(module, "target_rate_eta", cfg.target_rate_eta)) * (
                float(module.r) - float(target_rate)
            )
            update_tensor = update_tensor + module.thresh.detach().new_tensor(target_feedback)
        updates.append(float(update_tensor.detach().cpu()))
        feedbacks.append(float(target_feedback))
        module_scale = getattr(module, "threshold_lr_scale", None)
        threshold_lr_scale = cfg.threshold_lr_scale if module_scale is None else float(module_scale)
        module.thresh.data = module.thresh.data + update_tensor * float(lr) * float(threshold_lr_scale)
        min_threshold = getattr(module, "min_threshold", None)
        max_threshold = getattr(module, "max_threshold", None)
        min_threshold = cfg.min_threshold if min_threshold is None else min_threshold
        max_threshold = cfg.max_threshold if max_threshold is None else max_threshold
        if min_threshold is not None or max_threshold is not None:
            min_value = -float("inf") if min_threshold is None else float(min_threshold)
            max_value = float("inf") if max_threshold is None else float(max_threshold)
            module.thresh.data.clamp_(min=min_value, max=max_value)
        module.update_value = 0.0
    summary = atlif_ternary_summary(model)
    summary["raw_update_mean"] = sum(updates) / len(updates) if updates else 0.0
    summary["effective_update_mean"] = summary["raw_update_mean"] * float(lr) * cfg.threshold_lr_scale
    summary["target_feedback_mean"] = sum(feedbacks) / len(feedbacks) if feedbacks else 0.0
    return summary


def atlif_ternary_summary(model: nn.Module) -> dict[str, float | int]:
    modules = list(iter_atlif_ternary_psn(model))
    if not modules:
        return {"num_modules": 0}
    thresholds = [float(module.thresh.detach().cpu()) for _, module in modules]
    rates = [float(module.r) for _, module in modules]
    pos = [float(module.pos_r) for _, module in modules]
    neg = [float(module.neg_r) for _, module in modules]
    updates = [float(module.update_value) for _, module in modules]
    target_rates = [module.target_rate for _, module in modules if module.target_rate is not None]
    return {
        "num_modules": len(modules),
        "threshold_mean": sum(thresholds) / len(thresholds),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "activity_mean": sum(rates) / len(rates),
        "pos_mean": sum(pos) / len(pos),
        "neg_mean": sum(neg) / len(neg),
        "update_mean": sum(updates) / len(updates),
        "target_rate_mean": sum(float(value) for value in target_rates) / len(target_rates) if target_rates else 0.0,
    }
