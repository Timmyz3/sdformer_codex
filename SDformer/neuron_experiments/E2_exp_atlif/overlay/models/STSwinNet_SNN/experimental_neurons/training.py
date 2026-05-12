from __future__ import annotations

from numbers import Number

import torch


def _experimental_config(config: dict) -> dict:
    return config.get("experimental_neuron", {}) if isinstance(config, dict) else {}


def regularize_activity(model: torch.nn.Module, config: dict) -> torch.Tensor:
    scale = float(_experimental_config(config).get("activity_eta", 0.0))
    if scale == 0.0:
        try:
            return next(model.parameters()).new_tensor(0.0)
        except StopIteration:
            return torch.tensor(0.0)

    loss = None
    for module in model.modules():
        act_value = getattr(module, "act_value", None)
        if torch.is_tensor(act_value):
            loss = act_value if loss is None else loss + act_value
    if loss is None:
        try:
            loss = next(model.parameters()).new_tensor(0.0)
        except StopIteration:
            loss = torch.tensor(0.0)
    return loss * scale


def sanitize_threshold_grads(model: torch.nn.Module, config: dict) -> dict[str, int]:
    exp_cfg = _experimental_config(config)
    if not bool(exp_cfg.get("sanitize_threshold_grads", False)):
        return {"checked": 0, "sanitized": 0}

    checked = 0
    sanitized = 0
    for module in model.modules():
        if not hasattr(module, "thresh"):
            continue
        grad = getattr(module.thresh, "grad", None)
        if grad is None:
            continue
        checked += 1
        finite = torch.isfinite(grad)
        if finite.all():
            continue
        grad.data = torch.where(finite, grad, torch.zeros_like(grad))
        sanitized += 1
    return {"checked": checked, "sanitized": sanitized}


def freeze_experimental_parameters(model: torch.nn.Module, config: dict) -> dict[str, int | str]:
    exp_cfg = _experimental_config(config)
    mode = str(exp_cfg.get("freeze_mode", "none")).lower()
    if mode in {"", "none", "false"}:
        return {"mode": "none", "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "frozen": 0}
    if mode != "threshold_only":
        raise ValueError(f"Unsupported experimental freeze_mode: {mode}")

    trainable = 0
    frozen = 0
    threshold_params = 0
    for name, parameter in model.named_parameters():
        is_threshold = name.endswith(".thresh") or ".thresh" in name
        parameter.requires_grad_(is_threshold)
        if is_threshold:
            trainable += parameter.numel()
            threshold_params += 1
        else:
            frozen += parameter.numel()
    return {
        "mode": mode,
        "trainable": trainable,
        "frozen": frozen,
        "threshold_params": threshold_params,
    }


def threshold_update(model: torch.nn.Module, lr: float, config: dict) -> None:
    exp_cfg = _experimental_config(config)
    if not bool(exp_cfg.get("threshold_update", True)):
        return

    lr_scale = float(exp_cfg.get("threshold_lr_scale", 1.0))
    min_threshold = exp_cfg.get("min_threshold")
    max_threshold = exp_cfg.get("max_threshold")
    for module in model.modules():
        if not hasattr(module, "thresh") or not hasattr(module, "update_value"):
            continue
        update_value = module.update_value
        if isinstance(update_value, Number):
            update_tensor = module.thresh.detach().new_tensor(float(update_value))
        elif torch.is_tensor(update_value):
            update_tensor = update_value.detach().to(device=module.thresh.device, dtype=module.thresh.dtype)
        else:
            continue
        module.thresh.data = module.thresh.data + update_tensor * float(lr) * lr_scale
        if min_threshold is not None or max_threshold is not None:
            min_value = -float("inf") if min_threshold is None else float(min_threshold)
            max_value = float("inf") if max_threshold is None else float(max_threshold)
            module.thresh.data.clamp_(min=min_value, max=max_value)
        module.update_value = 0.0
