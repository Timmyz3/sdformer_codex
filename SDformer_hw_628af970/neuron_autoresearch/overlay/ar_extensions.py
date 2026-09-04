"""Autoresearch extensions — new gate types and regularizers.

Imported separately from H1's sparse_gate module to avoid shadowing issues.
"""

import torch
import torch.nn as nn
from typing import Iterable, List

# Import H1 primitives for reuse
from models.STSwinNet_SNN.sparse_gate import (
    generate_target_layers,
    iter_sparse_gates,
    sparse_gate_summary,
)


# ---------------------------------------------------------------------------
# A5: Refractory-Period Pruning
# ---------------------------------------------------------------------------

def install_refractory_neurons(
    model: nn.Module, raw_config: dict | None
) -> list[str]:
    """Wrap target spiking neurons with RefractoryNeuron.

    Config keys under ``refractory``:
        enabled, stage_selection, refractory_steps (default 2), mode (hard|soft).
    """
    rc = (raw_config or {}).get("refractory", {})
    if not rc.get("enabled", False):
        return []

    from src.models.modules.spiking_neurons.refractory_neuron import RefractoryNeuron

    if hasattr(model, "sttmultires_unet"):
        swin3d = model.sttmultires_unet.encoders.swin3d
        swin_depths = [len(stage.swin_blocks) for stage in swin3d.layers]
        rtargets = generate_target_layers(
            swin_depths, rc.get("stage_selection", "all_stages_proj")
        )
    else:
        rtargets = tuple(rc.get("target_layers", []))

    named_modules = dict(model.named_modules())
    missing = [n for n in rtargets if n not in named_modules]
    if missing:
        raise ValueError(f"Refractory targets missing: {missing[:5]}...")

    refractory_steps = int(rc.get("refractory_steps", 2))
    mode = str(rc.get("mode", "hard"))

    installed = []
    for name in rtargets:
        module = named_modules[name]
        if not hasattr(module, "spiking_neuron"):
            continue
        current = module.spiking_neuron
        if isinstance(current, RefractoryNeuron):
            installed.append(name)
            continue

        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        module.spiking_neuron = RefractoryNeuron(
            current, refractory_steps=refractory_steps, mode=mode
        ).to(device)
        installed.append(name)

    return installed


# ---------------------------------------------------------------------------
# A6: Bipolar Attention Gate helpers
# ---------------------------------------------------------------------------

def generate_attn_qk_targets(
    swin_depths: List[int],
    stages: Iterable[int] | None = None,
) -> tuple[str, ...]:
    """Generate target paths for attention Q/K projection neurons only."""
    prefix = "sttmultires_unet.encoders.swin3d.layers"
    stage_list = list(stages) if stages is not None else range(len(swin_depths))
    paths = []
    for stage in stage_list:
        for block_idx in range(swin_depths[stage]):
            paths.append(f"{prefix}.{stage}.swin_blocks.{block_idx}.attn.proj_sn")
    return tuple(sorted(paths))


# ---------------------------------------------------------------------------
# A8: Dual-Sparsity Regularizer
# ---------------------------------------------------------------------------

def dual_sparse_regularization(
    model: nn.Module,
    raw_config: dict | None,
) -> torch.Tensor | None:
    """Joint weight + activation sparsity penalty.

    Config under ``dual_sparse``: enabled, lambda_firing, lambda_weight, target_rate.
    """
    dc = (raw_config or {}).get("dual_sparse", {})
    if not dc.get("enabled", False):
        return None

    lambda_firing = float(dc.get("lambda_firing", 0.0))
    lambda_weight = float(dc.get("lambda_weight", 0.0))
    target_rate = float(dc.get("target_rate", 0.05))

    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)

    if lambda_firing > 0:
        for _, gate in iter_sparse_gates(model):
            if hasattr(gate, "running_firing_rate"):
                rate = gate.running_firing_rate
                total_loss = total_loss + lambda_firing * (rate - target_rate) ** 2

    if lambda_weight > 0:
        for _, gate in iter_sparse_gates(model):
            for param in gate.parameters():
                total_loss = total_loss + lambda_weight * param.abs().sum()

    return total_loss if total_loss.item() != 0.0 else None
