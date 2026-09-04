"""Optimizer helpers for differential-LR continuation experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from models.STSwinNet_SNN.atlif_ternary_psn import iter_atlif_ternary_psn


def _is_norm_or_bias(name: str, param: nn.Parameter) -> bool:
    lowered = name.lower()
    return param.ndim <= 1 or lowered.endswith(".bias") or ".norm" in lowered or ".bn" in lowered


def _atlif_prefixes(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _ in iter_atlif_ternary_psn(model))


def _under_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _is_h9_new_attention_param(name: str) -> bool:
    markers = (
        ".linear_v.", ".bn_v.", ".sn_v.",
        "._h9_match_code_weight", "._h9_lc4_coefficients", "._h9_cf10_beta",
    )
    return any(marker in name for marker in markers)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build an optimizer, optionally using parameter groups for fine-tuning.

    The default baseline path keeps the original single-LR behavior. When
    ``optimizer.param_groups.enabled`` is true, replaced ATLIF neuron parameters
    receive a larger LR, pretrained backbone parameters receive a smaller LR,
    and norm/bias parameters receive a conservative LR with zero weight decay.
    """

    opt_cfg = config["optimizer"]
    name = str(opt_cfg["name"])
    base_lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("wd", 0.0))
    group_cfg = opt_cfg.get("param_groups") or {}

    if not bool(group_cfg.get("enabled", False)):
        if name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
            _attach_initial_lrs(optimizer)
            return optimizer
        optimizer_cls = getattr(torch.optim, name)
        optimizer = optimizer_cls(model.parameters(), lr=base_lr)
        _attach_initial_lrs(optimizer)
        return optimizer

    backbone_lr = float(group_cfg.get("backbone_lr", base_lr))
    neuron_lr = float(group_cfg.get("neuron_lr", group_cfg.get("new_module_lr", base_lr)))
    new_module_lr = float(group_cfg.get("v_branch_lr", group_cfg.get("new_module_lr", neuron_lr)))
    threshold_lr = float(group_cfg.get("threshold_lr", neuron_lr))
    norm_lr = float(group_cfg.get("norm_lr", backbone_lr))
    norm_wd = float(group_cfg.get("norm_wd", 0.0))
    threshold_wd = float(group_cfg.get("threshold_wd", 0.0))

    atlif_prefixes = _atlif_prefixes(model)
    buckets: dict[str, list[nn.Parameter]] = defaultdict(list)
    seen: set[int] = set()
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        if _under_prefix(param_name, atlif_prefixes):
            if param_name.endswith(".thresh"):
                buckets["atlif_threshold"].append(param)
            elif _is_h9_new_attention_param(param_name):
                if _is_norm_or_bias(param_name, param):
                    buckets["new_module_no_decay"].append(param)
                else:
                    buckets["new_module"].append(param)
            elif _is_norm_or_bias(param_name, param):
                buckets["atlif_neuron_no_decay"].append(param)
            else:
                buckets["atlif_neuron"].append(param)
        elif _is_h9_new_attention_param(param_name):
            if _is_norm_or_bias(param_name, param):
                buckets["new_module_no_decay"].append(param)
            else:
                buckets["new_module"].append(param)
        elif _is_norm_or_bias(param_name, param):
            buckets["backbone_norm_bias"].append(param)
        else:
            buckets["backbone"].append(param)

    groups: list[dict[str, Any]] = []
    specs = [
        ("backbone", backbone_lr, weight_decay),
        ("backbone_norm_bias", norm_lr, norm_wd),
        ("new_module", new_module_lr, weight_decay),
        ("new_module_no_decay", new_module_lr, norm_wd),
        ("atlif_neuron", neuron_lr, weight_decay),
        ("atlif_neuron_no_decay", neuron_lr, norm_wd),
        ("atlif_threshold", threshold_lr, threshold_wd),
    ]
    for group_name, lr, wd in specs:
        params = buckets.get(group_name, [])
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": wd, "name": group_name})

    if name == "AdamW":
        optimizer = torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)
        _attach_initial_lrs(optimizer)
        return optimizer
    optimizer_cls = getattr(torch.optim, name)
    optimizer = optimizer_cls(groups, lr=base_lr)
    _attach_initial_lrs(optimizer)
    return optimizer


def _attach_initial_lrs(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", float(group["lr"]))


def lr_warmup_factor(step: int, config: dict[str, Any]) -> float | None:
    warmup_cfg = config.get("optimizer", {}).get("lr_warmup") or {}
    if not bool(warmup_cfg.get("enabled", False)):
        return None
    warmup_steps = int(warmup_cfg.get("steps", 0) or 0)
    if warmup_steps <= 0:
        return None
    if step > warmup_steps:
        return None
    start_factor = float(warmup_cfg.get("start_factor", 0.2))
    start_factor = min(1.0, max(0.0, start_factor))
    progress = min(1.0, max(0.0, float(max(0, step - 1)) / float(warmup_steps)))
    return start_factor + (1.0 - start_factor) * progress


def apply_lr_warmup(
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
) -> dict[str, float] | None:
    """Linearly warm up all parameter groups for short fine-tuning.

    Warmup only rewrites lrs while ``step <= warmup_steps``. After that it gets
    out of the way so epoch schedulers such as MultiStepLR remain effective.
    """

    factor = lr_warmup_factor(step, config)
    if factor is None:
        return None
    lrs: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        base_lr = float(group.get("initial_lr", group["lr"]))
        group["lr"] = base_lr * factor
        lrs[str(group.get("name", f"group{index}"))] = float(group["lr"])
    return lrs


def describe_optimizer_groups(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    description: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        num_params = sum(param.numel() for param in group["params"])
        description.append(
            {
                "index": index,
                "name": group.get("name", f"group{index}"),
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "num_params": int(num_params),
            }
        )
    return description


def freeze_threshold_gradients(
    model: nn.Module,
    step: int,
    config: dict[str, Any],
) -> int:
    """Drop ATLIF threshold gradients after an explicitly requested boundary.

    ``threshold_freeze_after_step`` historically freezes only the separate
    homeostatic update. Gradient freezing is intentionally opt-in so existing
    experiments retain their original optimizer semantics.
    """
    atlif_cfg = config.get("atlif_ternary_psn") or {}
    if not bool(atlif_cfg.get("freeze_threshold_grad_after_step", False)):
        return 0
    freeze_after = atlif_cfg.get("threshold_freeze_after_step")
    if freeze_after is None or int(step) < int(freeze_after):
        return 0
    frozen = 0
    for _, module in iter_atlif_ternary_psn(model):
        threshold = getattr(module, "thresh", None)
        if threshold is not None and threshold.grad is not None:
            threshold.grad = None
            frozen += 1
    return frozen
