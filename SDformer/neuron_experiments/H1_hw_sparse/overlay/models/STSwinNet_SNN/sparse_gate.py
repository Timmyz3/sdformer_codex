"""Multi-stage sparse gates for SDFormerFlow spiking nodes.

Extends the G1 approach from 6 layer0 nodes to all encoder stages.
Supports HardwareSparseNeuron (fused BN-Gate-Spike) as well as the
original HardSparseGate wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Re-export original G1 primitives so existing source patches keep working
# ---------------------------------------------------------------------------

DEFAULT_TARGET_LAYERS = (
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.attn.proj_sn",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.attn.proj_sn",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2",
)

# ---------------------------------------------------------------------------
# Target layer generation
# ---------------------------------------------------------------------------

_STAGE_SELECTION_MODES = {"layer0_only", "all_stages_proj", "all_stages_full"}


def generate_target_layers(
    swin_depths: List[int],
    stage_selection: str = "all_stages_proj",
) -> tuple[str, ...]:
    """Enumerate spiking neuron dotted paths for the configured stages.

    Parameters
    ----------
    swin_depths:
        Number of swin blocks per stage, e.g. ``[2, 2, 6, 2]``.
    stage_selection:
        * ``"layer0_only"`` — current G1, only stage 0 proj + mlp sn nodes.
        * ``"all_stages_proj"`` — all stages, attn.proj_sn + mlp.sn{1,2}.
        * ``"all_stages_full"`` — all stages, adds attn.sn_q + attn.sn_k.

    Returns
    -------
    Sorted tuple of dotted paths.
    """
    if stage_selection not in _STAGE_SELECTION_MODES:
        raise ValueError(
            f"stage_selection must be one of {sorted(_STAGE_SELECTION_MODES)}, "
            f"got {stage_selection!r}"
        )

    prefix = "sttmultires_unet.encoders.swin3d.layers"

    if stage_selection == "layer0_only":
        stage = 0
        paths: List[str] = []
        for block_idx in range(swin_depths[stage]):
            paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.attn.proj_sn")
            paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.mlp.sn1")
            paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.mlp.sn2")
        return tuple(sorted(paths))

    # all_stages_proj / all_stages_full
    paths = []
    for stage in range(len(swin_depths)):
        for block_idx in range(swin_depths[stage]):
            paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.attn.proj_sn")
            paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.mlp.sn1")
            paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.mlp.sn2")
            if stage_selection == "all_stages_full":
                paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.attn.sn_q")
                paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.attn.sn_k")
    return tuple(sorted(paths))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseGateConfig:
    enabled: bool = False
    target_layers: tuple[str, ...] = DEFAULT_TARGET_LAYERS
    stage_selection: str = "layer0_only"
    init_logit: float = 2.0
    threshold: float = 0.5
    reg_lambda: float = 0.0
    freeze_backbone: bool = False
    use_hardware_neuron: bool = False
    activity_eta: float = 0.0
    target_rate: float = 0.05
    adapt_momentum: float = 0.99
    # FSN (FusedSparseNeuron) fields
    use_fsn: bool = False
    fsn_num_levels: int = 1
    fsn_signed: bool = False
    fsn_level_learnable: bool = False


def config_from_dict(raw: dict | None) -> SparseGateConfig:
    raw = raw or {}
    targets = tuple(raw.get("target_layers") or DEFAULT_TARGET_LAYERS)
    return SparseGateConfig(
        enabled=bool(raw.get("enabled", False)),
        target_layers=targets,
        stage_selection=str(raw.get("stage_selection", "layer0_only")),
        init_logit=float(raw.get("init_logit", 2.0)),
        threshold=float(raw.get("threshold", 0.5)),
        reg_lambda=float(raw.get("reg_lambda", 0.0)),
        freeze_backbone=bool(raw.get("freeze_backbone", False)),
        use_hardware_neuron=bool(raw.get("use_hardware_neuron", False)),
        activity_eta=float(raw.get("activity_eta", 0.0)),
        target_rate=float(raw.get("target_rate", 0.05)),
        adapt_momentum=float(raw.get("adapt_momentum", 0.99)),
        use_fsn=bool(raw.get("use_fsn", False)),
        fsn_num_levels=int(raw.get("fsn_num_levels", 1)),
        fsn_signed=bool(raw.get("fsn_signed", False)),
        fsn_level_learnable=bool(raw.get("fsn_level_learnable", False)),
    )


# ---------------------------------------------------------------------------
# Original G1 HardSparseGate (kept for backward compat)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gate installation
# ---------------------------------------------------------------------------


def install_sparse_gates(model: nn.Module, raw_config: dict | None) -> list[str]:
    """Original G1 install: wraps selected .spiking_neuron with HardSparseGate."""
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []

    named_modules = dict(model.named_modules())
    missing = [name for name in cfg.target_layers if name not in named_modules]
    if missing:
        raise ValueError("sparse gate target layers missing: " + ", ".join(missing))

    installed: list[str] = []
    for name in cfg.target_layers:
        module = named_modules[name]
        if not hasattr(module, "spiking_neuron"):
            raise TypeError(f"Target layer does not look like Spiking_neuron wrapper: {name}")
        current = module.spiking_neuron
        if isinstance(current, (HardSparseGate, HardwareSparseNeuron, FusedSparseNeuron)):
            installed.append(name)
            continue
        device = _module_device(module)
        module.spiking_neuron = HardSparseGate(
            current, init_logit=cfg.init_logit, threshold=cfg.threshold
        ).to(device)
        installed.append(name)

    if cfg.freeze_backbone:
        freeze_non_gate_parameters(model)
    return installed


def install_hw_sparse_gates(model: nn.Module, raw_config: dict | None) -> list[str]:
    """Install HardwareSparseNeuron wrappers for fused BN-gate-spike.

    Uses the ``stage_selection`` field to auto-generate target layers from
    the model's swin_depths.
    """
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []

    if cfg.stage_selection != "layer0_only" or not cfg.target_layers:
        layers_key = "sttmultires_unet.encoders.swin3d.layers"
        if hasattr(model, "sttmultires_unet"):
            swin3d = model.sttmultires_unet.encoders.swin3d
            swin_depths = [len(stage.swin_blocks) for stage in swin3d.layers]
            resolved_targets = generate_target_layers(swin_depths, cfg.stage_selection)
        else:
            resolved_targets = cfg.target_layers
    else:
        resolved_targets = cfg.target_layers

    named_modules = dict(model.named_modules())
    missing = [name for name in resolved_targets if name not in named_modules]
    if missing:
        raise ValueError(
            f"HW sparse gate target layers missing ({cfg.stage_selection}): "
            + ", ".join(missing[:5])
            + (f" ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )

    from src.models.modules.spiking_neurons.hw_sparse_neuron import HardwareSparseNeuron

    installed: list[str] = []
    for name in resolved_targets:
        module = named_modules[name]
        if not hasattr(module, "spiking_neuron"):
            raise TypeError(f"Target layer does not look like Spiking_neuron wrapper: {name}")
        current = module.spiking_neuron
        if isinstance(current, (HardSparseGate, HardwareSparseNeuron)):
            installed.append(name)
            continue
        device = _module_device(module)
        module.spiking_neuron = HardwareSparseNeuron(
            current,
            init_logit=cfg.init_logit,
            threshold=cfg.threshold,
            activity_eta=cfg.activity_eta,
            target_rate=cfg.target_rate,
            adapt_momentum=cfg.adapt_momentum,
        ).to(device)
        installed.append(name)

    if cfg.freeze_backbone:
        freeze_non_gate_parameters(model)
    return installed


def install_fsn_gates(model: nn.Module, raw_config: dict | None) -> list[str]:
    """Install FusedSparseNeuron (GTCN + LMH multi-level + ternary spike).

    Supports num_levels (LMH) and signed (ternary) options.
    """
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []

    if cfg.stage_selection != "layer0_only" or not cfg.target_layers:
        if hasattr(model, "sttmultires_unet"):
            swin3d = model.sttmultires_unet.encoders.swin3d
            swin_depths = [len(stage.swin_blocks) for stage in swin3d.layers]
            resolved_targets = generate_target_layers(swin_depths, cfg.stage_selection)
        else:
            resolved_targets = cfg.target_layers
    else:
        resolved_targets = cfg.target_layers

    named_modules = dict(model.named_modules())
    missing = [name for name in resolved_targets if name not in named_modules]
    if missing:
        raise ValueError(
            f"FSN target layers missing ({cfg.stage_selection}): "
            + ", ".join(missing[:5])
            + (f" ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )

    from src.models.modules.spiking_neurons.fused_sparse_neuron import FusedSparseNeuron

    installed: list[str] = []
    for name in resolved_targets:
        module = named_modules[name]
        if not hasattr(module, "spiking_neuron"):
            raise TypeError(f"Target layer does not look like Spiking_neuron wrapper: {name}")
        current = module.spiking_neuron
        if isinstance(current, _GATE_TYPES):
            installed.append(name)
            continue
        device = _module_device(module)
        module.spiking_neuron = FusedSparseNeuron(
            current,
            num_levels=cfg.fsn_num_levels,
            signed=cfg.fsn_signed,
            init_logit=cfg.init_logit,
            threshold=cfg.threshold,
            activity_eta=cfg.activity_eta,
            target_rate=cfg.target_rate,
            adapt_momentum=cfg.adapt_momentum,
            level_learnable=cfg.fsn_level_learnable,
        ).to(device)
        installed.append(name)

    if cfg.freeze_backbone:
        freeze_non_gate_parameters(model)
    return installed


# ---------------------------------------------------------------------------
# Iteration and introspection
# ---------------------------------------------------------------------------


_GATE_TYPES = (HardSparseGate,)

try:
    from src.models.modules.spiking_neurons.hw_sparse_neuron import HardwareSparseNeuron

    _GATE_TYPES = _GATE_TYPES + (HardwareSparseNeuron,)
except ImportError:
    pass

try:
    from src.models.modules.spiking_neurons.fused_sparse_neuron import FusedSparseNeuron

    _GATE_TYPES = _GATE_TYPES + (FusedSparseNeuron,)
except ImportError:
    pass


def iter_sparse_gates(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, _GATE_TYPES):
            yield name, module


def sparse_gate_regularization(model: nn.Module, raw_config: dict | None) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled or cfg.reg_lambda <= 0:
        return None
    probs: list[torch.Tensor] = []
    for _, gate in iter_sparse_gates(model):
        if hasattr(gate, "gate_probability"):
            probs.append(gate.gate_probability)
    if not probs:
        return None
    return torch.stack(probs).mean() * cfg.reg_lambda


def threshold_regularization_loss(model: nn.Module, raw_config: dict | None) -> torch.Tensor | None:
    """ATLIF-style adaptive threshold penalty for HardwareSparseNeuron gates."""
    cfg = config_from_dict(raw_config)
    if not cfg.enabled or cfg.activity_eta <= 0:
        return None
    losses: list[torch.Tensor] = []
    for _, gate in iter_sparse_gates(model):
        if hasattr(gate, "regularization_loss"):
            losses.append(gate.regularization_loss())
    if not losses:
        return None
    return torch.stack(losses).sum()


def sparse_gate_summary(model: nn.Module) -> dict[str, float | int]:
    probs: list[float] = []
    for _, gate in iter_sparse_gates(model):
        if hasattr(gate, "gate_probability"):
            probs.append(float(gate.gate_probability.detach().cpu().item()))
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
        if hasattr(gate, "gate_logit"):
            gate.gate_logit.requires_grad = True
