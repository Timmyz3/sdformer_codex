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
    threshold_grad_scale: float = 1.0
    threshold_freeze_after_step: int | None = None
    min_threshold: float | None = 1.0e-3
    max_threshold: float | None = None
    activity_eta: float = 0.0
    negative_threshold_scale: float = 5.0
    target_rate: float | None = None
    target_rate_eta: float = 0.0
    target_rate_mode: str = "upper_bound"
    negative_target_rate: float | None = None
    negative_target_eta: float = 0.0
    negative_scale_min: float | None = None
    negative_scale_max: float | None = None
    center_mode: str = "zero"
    output_mode: str = "ternary"
    threshold_mode: str = "asymmetric_scale"
    quantile_q: float | None = None
    quantile_momentum: float = 0.9
    quantile_guard_margin: float = 0.25
    quantile_min_guard: float = 0.0
    quantile_sample_size: int = 4096
    importance_enabled: bool = False
    importance_momentum: float = 0.9
    importance_scale: float = 0.0
    importance_min_guard: float = 0.1
    temporal_factor_rank: int = 0
    temporal_factor_init: str = "balanced_svd"
    preserve_loaded_threshold: bool = False
    stage_threshold_eta: Any = None
    stage_threshold_lr_scale: Any = None
    stage_target_rate_eta: Any = None
    stage_activity_eta: Any = None
    stage_max_threshold: Any = None
    stage_negative_threshold_scale: Any = None
    stage_negative_target_rate: Any = None
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
        threshold_grad_scale=float(raw.get("threshold_grad_scale", 1.0)),
        threshold_freeze_after_step=None
        if raw.get("threshold_freeze_after_step") is None
        else int(raw.get("threshold_freeze_after_step")),
        min_threshold=None if raw.get("min_threshold") is None else float(raw.get("min_threshold", 1.0e-3)),
        max_threshold=None if raw.get("max_threshold") is None else float(raw.get("max_threshold")),
        activity_eta=float(raw.get("activity_eta", 0.0)),
        negative_threshold_scale=float(raw.get("negative_threshold_scale", 5.0)),
        target_rate=None if raw.get("target_rate") is None else float(raw.get("target_rate")),
        target_rate_eta=float(raw.get("target_rate_eta", 0.0)),
        target_rate_mode=str(raw.get("target_rate_mode", raw.get("target_feedback_mode", "upper_bound"))),
        negative_target_rate=None
        if raw.get("negative_target_rate") is None
        else float(raw.get("negative_target_rate")),
        negative_target_eta=float(raw.get("negative_target_eta", 0.0)),
        negative_scale_min=None if raw.get("negative_scale_min") is None else float(raw.get("negative_scale_min")),
        negative_scale_max=None if raw.get("negative_scale_max") is None else float(raw.get("negative_scale_max")),
        center_mode=str(raw.get("center_mode", "zero")),
        output_mode=str(raw.get("output_mode", "ternary")),
        threshold_mode=str(raw.get("threshold_mode", "asymmetric_scale")),
        quantile_q=None if raw.get("quantile_q") is None else float(raw.get("quantile_q")),
        quantile_momentum=float(raw.get("quantile_momentum", 0.9)),
        quantile_guard_margin=float(raw.get("quantile_guard_margin", 0.25)),
        quantile_min_guard=float(raw.get("quantile_min_guard", 0.0)),
        quantile_sample_size=int(raw.get("quantile_sample_size", 4096)),
        importance_enabled=bool(raw.get("importance_enabled", False)),
        importance_momentum=float(raw.get("importance_momentum", 0.9)),
        importance_scale=float(raw.get("importance_scale", 0.0)),
        importance_min_guard=float(raw.get("importance_min_guard", 0.1)),
        temporal_factor_rank=int(raw.get("temporal_factor_rank", 0)),
        temporal_factor_init=str(raw.get("temporal_factor_init", "balanced_svd")),
        preserve_loaded_threshold=bool(raw.get("preserve_loaded_threshold", False)),
        stage_threshold_eta=raw.get("stage_threshold_eta"),
        stage_threshold_lr_scale=raw.get("stage_threshold_lr_scale"),
        stage_target_rate_eta=raw.get("stage_target_rate_eta"),
        stage_activity_eta=raw.get("stage_activity_eta"),
        stage_max_threshold=raw.get("stage_max_threshold"),
        stage_negative_threshold_scale=raw.get("stage_negative_threshold_scale"),
        stage_negative_target_rate=raw.get("stage_negative_target_rate"),
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


def _is_qk_attention_path(path: str) -> bool:
    """Paths reserved for ternary Q/K expressiveness; everything else can be binary ATLIF."""
    return path.endswith(".sn_q") or path.endswith(".sn_k") or path.endswith(".sn2_q")


def iter_non_qk_spiking_neuron_paths(
    model: nn.Module,
    seen: set[str] | None = None,
    *,
    exclude_path_prefixes: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Return all Spiking_neuron wrapper paths except attention Q/K (and already-installed paths)."""
    skip = seen or set()
    prefixes = tuple(str(prefix) for prefix in exclude_path_prefixes)
    paths: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Spiking_neuron":
            continue
        if _is_qk_attention_path(name):
            continue
        if name in skip:
            continue
        if prefixes and any(name.startswith(prefix) for prefix in prefixes):
            continue
        paths.append(name)
    return sorted(paths)


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


def _apply_threshold_grad_hook(module: ATLIFTernaryPSN, scale: float) -> None:
    module.threshold_grad_scale = float(scale)
    handle = getattr(module, "_h9_threshold_grad_hook", None)
    old_scale = getattr(module, "_h9_threshold_grad_hook_scale", None)
    if handle is not None and old_scale != float(scale):
        handle.remove()
        delattr(module, "_h9_threshold_grad_hook")
        handle = None
    if float(scale) == 1.0:
        if handle is not None:
            handle.remove()
            delattr(module, "_h9_threshold_grad_hook")
        module._h9_threshold_grad_hook_scale = float(scale)
        return
    if handle is None:
        module._h9_threshold_grad_hook = module.thresh.register_hook(
            lambda grad, hook_scale=float(scale): grad * hook_scale
        )
    module._h9_threshold_grad_hook_scale = float(scale)


def _configure_existing_atlif(module: ATLIFTernaryPSN, cfg: ATLIFTernaryPSNConfig, stage_idx: int | None) -> None:
    if getattr(module, "output_mode", cfg.output_mode) != cfg.output_mode:
        raise RuntimeError(
            "Loaded ATLIFTernaryPSN output_mode does not match config. "
            "Reload from a baseline checkpoint when changing binary/ternary mode."
        )
    if getattr(module, "threshold_mode", cfg.threshold_mode) != cfg.threshold_mode:
        raise RuntimeError(
            "Loaded ATLIFTernaryPSN threshold_mode does not match config. "
            "Reload from a baseline checkpoint when changing ATLIF/ternary threshold semantics."
        )
    if getattr(module, "center_mode", cfg.center_mode) != cfg.center_mode:
        raise RuntimeError(
            "Loaded ATLIFTernaryPSN center_mode does not match config. "
            "Reload from a baseline checkpoint when changing centering semantics."
        )
    if int(getattr(module, "temporal_factor_requested_rank", 0)) != cfg.temporal_factor_rank:
        raise RuntimeError(
            "Loaded ATLIFTernaryPSN temporal_factor_rank does not match config. "
            "Instantiate the requested factor rank before loading the checkpoint."
        )
    module.sp = float(_stage_value(cfg.stage_threshold_eta, stage_idx, cfg.threshold_eta))
    module.negative_threshold_scale = float(
        _stage_value(cfg.stage_negative_threshold_scale, stage_idx, cfg.negative_threshold_scale)
    )
    module.activity_eta = float(_stage_value(cfg.stage_activity_eta, stage_idx, cfg.activity_eta))
    module.min_threshold = cfg.min_threshold
    module.max_threshold = (
        None
        if _stage_value(cfg.stage_max_threshold, stage_idx, cfg.max_threshold) is None
        else float(_stage_value(cfg.stage_max_threshold, stage_idx, cfg.max_threshold))
    )
    module.threshold_lr_scale = float(_stage_value(cfg.stage_threshold_lr_scale, stage_idx, cfg.threshold_lr_scale))
    module.target_rate = (
        None
        if _stage_value(cfg.stage_target_rate, stage_idx, cfg.target_rate) is None
        else float(_stage_value(cfg.stage_target_rate, stage_idx, cfg.target_rate))
    )
    module.target_rate_eta = float(_stage_value(cfg.stage_target_rate_eta, stage_idx, cfg.target_rate_eta))
    module.target_rate_mode = str(cfg.target_rate_mode)
    module.negative_target_rate = (
        None
        if _stage_value(cfg.stage_negative_target_rate, stage_idx, cfg.negative_target_rate) is None
        else float(_stage_value(cfg.stage_negative_target_rate, stage_idx, cfg.negative_target_rate))
    )
    module.negative_target_eta = float(cfg.negative_target_eta)
    module.negative_scale_min = cfg.negative_scale_min
    module.negative_scale_max = cfg.negative_scale_max
    module.quantile_q = cfg.quantile_q
    module.quantile_momentum = cfg.quantile_momentum
    module.quantile_guard_margin = cfg.quantile_guard_margin
    module.quantile_min_guard = cfg.quantile_min_guard
    module.quantile_sample_size = cfg.quantile_sample_size
    module.importance_enabled = cfg.importance_enabled
    module.importance_momentum = cfg.importance_momentum
    module.importance_scale = cfg.importance_scale
    module.importance_min_guard = cfg.importance_min_guard
    _apply_threshold_grad_hook(module, cfg.threshold_grad_scale)


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
        _configure_existing_atlif(current, cfg, stage_idx)
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
        sparsity_eta=float(_stage_value(cfg.stage_threshold_eta, stage_idx, cfg.threshold_eta)),
        negative_threshold_scale=float(
            _stage_value(cfg.stage_negative_threshold_scale, stage_idx, cfg.negative_threshold_scale)
        ),
        activity_eta=float(_stage_value(cfg.stage_activity_eta, stage_idx, cfg.activity_eta)),
        min_threshold=cfg.min_threshold,
        max_threshold=None
        if _stage_value(cfg.stage_max_threshold, stage_idx, cfg.max_threshold) is None
        else float(_stage_value(cfg.stage_max_threshold, stage_idx, cfg.max_threshold)),
        threshold_lr_scale=float(_stage_value(cfg.stage_threshold_lr_scale, stage_idx, cfg.threshold_lr_scale)),
        target_rate=None
        if _stage_value(cfg.stage_target_rate, stage_idx, cfg.target_rate) is None
        else float(_stage_value(cfg.stage_target_rate, stage_idx, cfg.target_rate)),
        target_rate_eta=float(_stage_value(cfg.stage_target_rate_eta, stage_idx, cfg.target_rate_eta)),
        target_rate_mode=cfg.target_rate_mode,
        negative_target_rate=None
        if _stage_value(cfg.stage_negative_target_rate, stage_idx, cfg.negative_target_rate) is None
        else float(_stage_value(cfg.stage_negative_target_rate, stage_idx, cfg.negative_target_rate)),
        negative_target_eta=cfg.negative_target_eta,
        negative_scale_min=cfg.negative_scale_min,
        negative_scale_max=cfg.negative_scale_max,
        center_mode=cfg.center_mode,
        output_mode=cfg.output_mode,
        threshold_mode=cfg.threshold_mode,
        quantile_q=cfg.quantile_q,
        quantile_momentum=cfg.quantile_momentum,
        quantile_guard_margin=cfg.quantile_guard_margin,
        quantile_min_guard=cfg.quantile_min_guard,
        quantile_sample_size=cfg.quantile_sample_size,
        importance_enabled=cfg.importance_enabled,
        importance_momentum=cfg.importance_momentum,
        importance_scale=cfg.importance_scale,
        importance_min_guard=cfg.importance_min_guard,
        temporal_factor_rank=cfg.temporal_factor_rank,
        temporal_factor_init=cfg.temporal_factor_init,
    ).to(device)
    _apply_threshold_grad_hook(wrapper.spiking_neuron, cfg.threshold_grad_scale)
    return True


def _config_for_group(base: ATLIFTernaryPSNConfig, group: dict[str, Any]) -> ATLIFTernaryPSNConfig:
    overrides: dict[str, Any] = {}
    for key in (
        "threshold_init",
        "threshold_eta",
        "threshold_lr_scale",
        "threshold_grad_scale",
        "min_threshold",
        "max_threshold",
        "activity_eta",
        "negative_threshold_scale",
        "target_rate",
        "target_rate_eta",
        "target_rate_mode",
        "negative_target_rate",
        "negative_target_eta",
        "negative_scale_min",
        "negative_scale_max",
        "center_mode",
        "output_mode",
        "threshold_mode",
        "quantile_q",
        "quantile_momentum",
        "quantile_guard_margin",
        "quantile_min_guard",
        "quantile_sample_size",
        "importance_enabled",
        "importance_momentum",
        "importance_scale",
        "importance_min_guard",
        "temporal_factor_rank",
        "temporal_factor_init",
    ):
        if key in group:
            value = group[key]
            if value is None:
                overrides[key] = None
            elif key in {
                "threshold_init",
                "threshold_eta",
                "threshold_lr_scale",
                "threshold_grad_scale",
                "min_threshold",
                "max_threshold",
                "activity_eta",
                "negative_threshold_scale",
                "target_rate",
                "target_rate_eta",
                "negative_target_rate",
                "negative_target_eta",
                "negative_scale_min",
                "negative_scale_max",
                "quantile_q",
                "quantile_momentum",
                "quantile_guard_margin",
                "quantile_min_guard",
                "importance_momentum",
                "importance_scale",
                "importance_min_guard",
            }:
                overrides[key] = float(value)
            elif key == "quantile_sample_size":
                overrides[key] = int(value)
            elif key == "temporal_factor_rank":
                overrides[key] = int(value)
            elif key == "importance_enabled":
                overrides[key] = bool(value)
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
        path_selection = str(group.get("path_selection", "")).strip()
        if path_selection == "all_non_qk":
            exclude_prefixes = group.get("exclude_path_prefixes", ())
            if isinstance(exclude_prefixes, str):
                exclude_prefixes = [exclude_prefixes]
            paths = iter_non_qk_spiking_neuron_paths(
                model,
                seen,
                exclude_path_prefixes=tuple(str(prefix) for prefix in exclude_prefixes),
            )
        else:
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
    if mode not in {
        "atlif_only", "threshold_only", "temporal_factor_only",
        "temporal_factor_atlif",
    }:
        raise ValueError(
            "atlif_ternary_psn.trainable must be all, atlif_only, threshold_only, "
            "temporal_factor_only, or temporal_factor_atlif"
        )

    for param in model.parameters():
        param.requires_grad_(False)

    for _, module in iter_atlif_ternary_psn(model):
        if mode == "atlif_only":
            for param in module.parameters():
                param.requires_grad_(True)
        elif mode == "threshold_only":
            module.thresh.requires_grad_(True)
        elif int(getattr(module, "temporal_factor_rank", 0)) > 0:
            module.temporal_factor_left.requires_grad_(True)
            module.temporal_factor_right.requires_grad_(True)
            if mode == "temporal_factor_atlif":
                module.bias.requires_grad_(True)
                module.thresh.requires_grad_(True)

    return {"mode": mode, "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad)}


def iter_atlif_ternary_psn(model: nn.Module) -> Iterable[tuple[str, ATLIFTernaryPSN]]:
    for name, module in model.named_modules():
        if isinstance(module, ATLIFTernaryPSN):
            yield name, module


def materialize_temporal_factor_state_dict(model: nn.Module):
    """Return an explicit dense-only state dict for a factorized ATLIF model.

    ``ATLIFTernaryPSN._save_to_state_dict`` first refreshes every dense
    migration weight to ``left @ right``.  This helper then strips factor keys
    so a rank-zero model can load the result deliberately.  Floating-point
    association (and later fixed-point intermediate requantization) differs
    between ``L(Rx)`` and ``(LR)x``; callers must evaluate the exported model
    and must not describe this conversion as bit-exact.
    """
    state = model.state_dict()
    factor_keys = [
        key
        for key in state
        if key.endswith(".temporal_factor_left")
        or key.endswith(".temporal_factor_right")
        or key in {"temporal_factor_left", "temporal_factor_right"}
    ]
    for key in factor_keys:
        del state[key]
    return state


def atlif_temporal_factor_diagnostics(model: nn.Module) -> dict[str, float | int]:
    """Summarize factor scale/conditioning without changing checkpoint state."""
    modules = [
        module
        for _, module in iter_atlif_ternary_psn(model)
        if int(getattr(module, "temporal_factor_rank", 0)) > 0
    ]
    if not modules:
        return {"temporal_factorized_modules": 0}
    left_norms = []
    right_norms = []
    left_absmax = []
    right_absmax = []
    balance_ratios = []
    latent_component_balance_ratios = []
    effective_conditions = []
    reference_relative_errors = []
    with torch.no_grad():
        for module in modules:
            left = module.temporal_factor_left.detach().float()
            right = module.temporal_factor_right.detach().float()
            effective = torch.mm(left, right)
            left_norm = float(torch.linalg.norm(left).cpu())
            right_norm = float(torch.linalg.norm(right).cpu())
            left_norms.append(left_norm)
            right_norms.append(right_norm)
            left_absmax.append(float(left.abs().max().cpu()))
            right_absmax.append(float(right.abs().max().cpu()))
            balance_ratios.append(
                max(left_norm, right_norm) / max(min(left_norm, right_norm), 1.0e-12)
            )
            for component in range(int(module.temporal_factor_rank)):
                left_component_norm = float(
                    torch.linalg.norm(left[:, component]).cpu()
                )
                right_component_norm = float(
                    torch.linalg.norm(right[component, :]).cpu()
                )
                latent_component_balance_ratios.append(
                    max(left_component_norm, right_component_norm)
                    / max(min(left_component_norm, right_component_norm), 1.0e-12)
                )
            singular = torch.linalg.svdvals(effective)
            rank = int(module.temporal_factor_rank)
            effective_conditions.append(
                float((singular[0] / singular[rank - 1].clamp_min(1.0e-12)).cpu())
            )
            reference = module.weight.detach().float()
            reference_relative_errors.append(
                float(
                    (
                        torch.linalg.norm(effective - reference)
                        / torch.linalg.norm(reference).clamp_min(1.0e-12)
                    ).cpu()
                )
            )

    def mean(values):
        return sum(values) / len(values)

    return {
        "temporal_factorized_modules": len(modules),
        "left_fro_norm_mean": mean(left_norms),
        "right_fro_norm_mean": mean(right_norms),
        "left_absmax_max": max(left_absmax),
        "right_absmax_max": max(right_absmax),
        "left_right_norm_balance_ratio_max": max(balance_ratios),
        "latent_component_balance_ratio_max": max(
            latent_component_balance_ratios
        ),
        "effective_rank_condition_mean": mean(effective_conditions),
        "effective_rank_condition_max": max(effective_conditions),
        "dense_reference_relative_error_mean": mean(reference_relative_errors),
        "dense_reference_relative_error_max": max(reference_relative_errors),
    }


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


def _apply_positive_update_guard(update_tensor: torch.Tensor, guard: float) -> torch.Tensor:
    if guard >= 0.999999:
        return update_tensor
    guard_tensor = update_tensor.detach().new_tensor(float(guard))
    return torch.where(update_tensor > 0, update_tensor * guard_tensor, update_tensor)


def threshold_update(model: nn.Module, lr: float, raw_config: dict | None) -> dict[str, float | int]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return {"num_modules": 0}
    global_step = raw_config.get("_global_step") if raw_config else None
    freeze_after_step = cfg.threshold_freeze_after_step
    freeze_updates = (
        freeze_after_step is not None
        and global_step is not None
        and int(global_step) >= int(freeze_after_step)
    )
    updates: list[float] = []
    raw_updates: list[float] = []
    feedbacks: list[float] = []
    negative_scale_feedbacks: list[float] = []
    quantile_guards: list[float] = []
    importance_guards: list[float] = []
    effective_updates: list[float] = []
    for _, module in iter_atlif_ternary_psn(model):
        update_value = module.update_value
        if isinstance(update_value, Number):
            update_tensor = module.thresh.detach().new_tensor(float(update_value))
        elif torch.is_tensor(update_value):
            update_tensor = update_value.detach().to(device=module.thresh.device, dtype=module.thresh.dtype)
        else:
            continue
        raw_updates.append(float(update_tensor.detach().cpu()))
        if getattr(module, "quantile_q", None) is not None and getattr(module, "_quantile_initialized", False):
            theta = float(module.thresh.detach().cpu())
            q_value = float(getattr(module, "quantile_value", 0.0))
            margin = max(abs(theta) * float(getattr(module, "quantile_guard_margin", 0.25)), 1.0e-12)
            guard = (q_value - theta) / margin
            guard = min(1.0, max(float(getattr(module, "quantile_min_guard", 0.0)), guard))
            update_tensor = _apply_positive_update_guard(update_tensor, guard)
            quantile_guards.append(float(guard))
        if bool(getattr(module, "importance_enabled", False)) and float(getattr(module, "importance_scale", 0.0)) > 0.0:
            importance = float(getattr(module, "importance_ema", 0.0))
            guard = 1.0 / (1.0 + float(getattr(module, "importance_scale", 0.0)) * max(0.0, importance))
            guard = min(1.0, max(float(getattr(module, "importance_min_guard", 0.1)), guard))
            update_tensor = _apply_positive_update_guard(update_tensor, guard)
            importance_guards.append(float(guard))
        target_rate = getattr(module, "target_rate", cfg.target_rate)
        target_feedback = 0.0
        official_mode = getattr(module, "threshold_mode", cfg.threshold_mode) == "official_atlif"
        if (
            not official_mode
            and target_rate is not None
            and float(getattr(module, "target_rate_eta", cfg.target_rate_eta)) != 0.0
        ):
            rate_error = float(module.r) - float(target_rate)
            mode = str(getattr(module, "target_rate_mode", cfg.target_rate_mode))
            if mode in {"upper_bound", "budget", "one_sided"}:
                rate_error = max(rate_error, 0.0)
            elif mode in {"bidirectional", "track", "legacy"}:
                pass
            else:
                raise ValueError("atlif_ternary_psn.target_rate_mode must be upper_bound or bidirectional")
            target_feedback = float(getattr(module, "target_rate_eta", cfg.target_rate_eta)) * rate_error
            update_tensor = update_tensor + module.thresh.detach().new_tensor(target_feedback)
        updates.append(float(update_tensor.detach().cpu()))
        feedbacks.append(float(target_feedback))
        module_scale = getattr(module, "threshold_lr_scale", None)
        threshold_lr_scale = cfg.threshold_lr_scale if module_scale is None else float(module_scale)
        before_thresh = module.thresh.detach().clone()
        if not freeze_updates:
            module.thresh.data = module.thresh.data + update_tensor * float(lr) * float(threshold_lr_scale)
        min_threshold = getattr(module, "min_threshold", None)
        max_threshold = getattr(module, "max_threshold", None)
        min_threshold = cfg.min_threshold if min_threshold is None else min_threshold
        max_threshold = cfg.max_threshold if max_threshold is None else max_threshold
        if not official_mode and (min_threshold is not None or max_threshold is not None):
            min_value = -float("inf") if min_threshold is None else float(min_threshold)
            max_value = float("inf") if max_threshold is None else float(max_threshold)
            module.thresh.data.clamp_(min=min_value, max=max_value)
        effective_updates.append(float((module.thresh.detach() - before_thresh).mean().cpu()))
        negative_target_rate = getattr(module, "negative_target_rate", cfg.negative_target_rate)
        negative_target_eta = float(getattr(module, "negative_target_eta", cfg.negative_target_eta))
        if (
            getattr(module, "output_mode", "ternary") == "ternary"
            and getattr(module, "threshold_mode", "asymmetric_scale") == "asymmetric_scale"
            and negative_target_rate is not None
            and negative_target_eta != 0.0
        ):
            negative_scale_feedback = negative_target_eta * (float(module.neg_r) - float(negative_target_rate))
            new_scale = float(module.negative_threshold_scale) + negative_scale_feedback
            min_scale = getattr(module, "negative_scale_min", cfg.negative_scale_min)
            max_scale = getattr(module, "negative_scale_max", cfg.negative_scale_max)
            if min_scale is not None:
                new_scale = max(float(min_scale), new_scale)
            if max_scale is not None:
                new_scale = min(float(max_scale), new_scale)
            module.negative_threshold_scale = new_scale
            negative_scale_feedbacks.append(float(negative_scale_feedback))
        module.update_value = 0.0
    summary = atlif_ternary_summary(model)
    summary["raw_update_mean"] = sum(raw_updates) / len(raw_updates) if raw_updates else 0.0
    summary["guarded_update_mean"] = sum(updates) / len(updates) if updates else 0.0
    summary["effective_update_mean"] = (
        sum(effective_updates) / len(effective_updates) if effective_updates else 0.0
    )
    summary["target_feedback_mean"] = sum(feedbacks) / len(feedbacks) if feedbacks else 0.0
    summary["negative_scale_feedback_mean"] = (
        sum(negative_scale_feedbacks) / len(negative_scale_feedbacks) if negative_scale_feedbacks else 0.0
    )
    summary["quantile_guard_mean"] = sum(quantile_guards) / len(quantile_guards) if quantile_guards else 1.0
    summary["importance_guard_mean"] = sum(importance_guards) / len(importance_guards) if importance_guards else 1.0
    summary["threshold_updates_frozen"] = int(freeze_updates)
    return summary


def atlif_ternary_summary(model: nn.Module) -> dict[str, float | int]:
    modules = list(iter_atlif_ternary_psn(model))
    if not modules:
        return {"num_modules": 0}
    thresholds = [float(module.thresh.detach().cpu()) for _, module in modules]
    rates = [float(module.r) for _, module in modules]
    pos = [float(module.pos_r) for _, module in modules]
    neg = [float(module.neg_r) for _, module in modules]
    ternary_modules = [(name, module) for name, module in modules if getattr(module, "output_mode", "ternary") == "ternary"]
    binary_modules = [(name, module) for name, module in modules if getattr(module, "output_mode", "ternary") == "binary"]
    ternary_rates = [float(module.r) for _, module in ternary_modules]
    ternary_pos = [float(module.pos_r) for _, module in ternary_modules]
    ternary_neg = [float(module.neg_r) for _, module in ternary_modules]
    per_module_ratios = [
        float(module.pos_r) / max(float(module.neg_r), 1.0e-12)
        for _, module in ternary_modules
    ]
    binary_rates = [float(module.r) for _, module in binary_modules]
    updates = [float(module.update_value) for _, module in modules]
    target_rates = [module.target_rate for _, module in modules if module.target_rate is not None]
    target_control_modules = [
        module
        for _, module in modules
        if module.target_rate is not None
        and float(getattr(module, "target_rate_eta", 0.0)) != 0.0
        and getattr(module, "threshold_mode", "asymmetric_scale") != "official_atlif"
    ]
    active_target_rate_modes = [str(getattr(module, "target_rate_mode", "upper_bound")) for module in target_control_modules]
    negative_scales = [float(module.negative_threshold_scale) for _, module in modules]
    positive_trigger_rates = [float(getattr(module, "positive_trigger_r", 0.0)) for _, module in modules]
    negative_trigger_rates = [float(getattr(module, "negative_trigger_r", 0.0)) for _, module in modules]
    threshold_modes = [
        str(getattr(module, "threshold_mode", "asymmetric_scale"))
        for _, module in modules
        if getattr(module, "output_mode", "ternary") == "ternary"
    ]
    center_modes = [str(getattr(module, "center_mode", "zero")) for _, module in modules]
    negative_target_rates = [
        module.negative_target_rate for _, module in modules if getattr(module, "negative_target_rate", None) is not None
    ]
    quantile_modules = [
        module for _, module in modules if getattr(module, "quantile_q", None) is not None
    ]
    importance_modules = [
        module for _, module in modules if bool(getattr(module, "importance_enabled", False))
    ]
    quantile_values = [
        float(getattr(module, "quantile_value", 0.0))
        for module in quantile_modules
        if bool(getattr(module, "_quantile_initialized", False))
    ]
    importance_values = [
        float(getattr(module, "importance_ema", 0.0))
        for module in importance_modules
        if bool(getattr(module, "_importance_initialized", False))
    ]
    factorized_modules = [
        module for _, module in modules
        if int(getattr(module, "temporal_factor_rank", 0)) > 0
    ]
    factor_ranks = [
        int(getattr(module, "temporal_factor_rank", 0))
        for module in factorized_modules
    ]
    factor_entries = [
        int(module.temporal_factor_left.numel() + module.temporal_factor_right.numel())
        for module in factorized_modules
    ]
    requested_factor_modules = [
        module for _, module in modules
        if int(getattr(module, "temporal_factor_requested_rank", 0)) > 0
    ]
    return {
        "num_modules": len(modules),
        "threshold_mean": sum(thresholds) / len(thresholds),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "activity_mean": sum(rates) / len(rates),
        "pos_mean": sum(pos) / len(pos),
        "neg_mean": sum(neg) / len(neg),
        "ternary_activity_mean": sum(ternary_rates) / len(ternary_rates) if ternary_rates else 0.0,
        "ternary_pos_mean": sum(ternary_pos) / len(ternary_pos) if ternary_pos else 0.0,
        "ternary_neg_mean": sum(ternary_neg) / len(ternary_neg) if ternary_neg else 0.0,
        "ternary_pos_min": min(ternary_pos) if ternary_pos else 0.0,
        "ternary_pos_max": max(ternary_pos) if ternary_pos else 0.0,
        "ternary_neg_min": min(ternary_neg) if ternary_neg else 0.0,
        "ternary_neg_max": max(ternary_neg) if ternary_neg else 0.0,
        "ternary_pos_neg_ratio": (
            (sum(ternary_pos) / len(ternary_pos)) / max(sum(ternary_neg) / len(ternary_neg), 1.0e-12)
            if ternary_pos and ternary_neg
            else 0.0
        ),
        "ternary_worst_pos_neg_ratio": max(per_module_ratios) if per_module_ratios else 0.0,
        "ternary_zero_pos_modules": sum(1 for value in ternary_pos if value <= 1.0e-8),
        "ternary_zero_neg_modules": sum(1 for value in ternary_neg if value <= 1.0e-8),
        "ternary_balance_error": (
            abs((sum(ternary_pos) / len(ternary_pos)) - (sum(ternary_neg) / len(ternary_neg)))
            if ternary_pos and ternary_neg
            else 0.0
        ),
        "binary_activity_mean": sum(binary_rates) / len(binary_rates) if binary_rates else 0.0,
        "update_mean": sum(updates) / len(updates),
        "target_rate_mean": sum(float(value) for value in target_rates) / len(target_rates) if target_rates else 0.0,
        "target_rate_control_modules": len(target_control_modules),
        "target_rate_upper_bound_modules": sum(
            1 for mode in active_target_rate_modes if mode in {"upper_bound", "budget", "one_sided"}
        ),
        "target_rate_bidirectional_modules": sum(
            1 for mode in active_target_rate_modes if mode in {"bidirectional", "track", "legacy"}
        ),
        "negative_scale_mean": sum(negative_scales) / len(negative_scales),
        "negative_scale_min": min(negative_scales),
        "negative_scale_max": max(negative_scales),
        "positive_trigger_mean": sum(positive_trigger_rates) / len(positive_trigger_rates),
        "negative_trigger_mean": sum(negative_trigger_rates) / len(negative_trigger_rates),
        "asymmetric_scale_modules": sum(1 for mode in threshold_modes if mode == "asymmetric_scale"),
        "symmetric_bsa_tsn_modules": sum(1 for mode in threshold_modes if mode == "symmetric_bsa_tsn"),
        "symmetric_target_rate_modules": sum(1 for mode in threshold_modes if mode == "symmetric_target_rate"),
        "official_atlif_modules": sum(
            1 for _, module in modules if getattr(module, "threshold_mode", "asymmetric_scale") == "official_atlif"
        ),
        "symmetric_binary_abs_modules": sum(
            1
            for _, module in modules
            if getattr(module, "threshold_mode", "asymmetric_scale") == "symmetric_binary_abs"
        ),
        "center_bias_modules": sum(1 for mode in center_modes if mode == "bias"),
        "center_calibrated_modules": sum(1 for mode in center_modes if mode == "calibrated"),
        "negative_target_rate_mean": sum(float(value) for value in negative_target_rates) / len(negative_target_rates)
        if negative_target_rates
        else 0.0,
        "quantile_modules": len(quantile_modules),
        "quantile_value_mean": sum(quantile_values) / len(quantile_values) if quantile_values else 0.0,
        "importance_modules": len(importance_modules),
        "importance_ema_mean": sum(importance_values) / len(importance_values) if importance_values else 0.0,
        "temporal_factorized_modules": len(factorized_modules),
        "temporal_factor_requested_modules": len(requested_factor_modules),
        "temporal_factor_dense_fallback_modules": (
            len(requested_factor_modules) - len(factorized_modules)
        ),
        "temporal_factor_rank_min": min(factor_ranks) if factor_ranks else 0,
        "temporal_factor_rank_max": max(factor_ranks) if factor_ranks else 0,
        "temporal_factor_parameter_entries": sum(factor_entries),
    }
