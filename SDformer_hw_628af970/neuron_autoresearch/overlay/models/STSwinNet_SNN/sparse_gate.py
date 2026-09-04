"""Autoresearch Extended Sparse Gate Module.

Re-exports H1's primitives + A5 (refractory) + A6 (attn QK targets) +
A8 (dual-sparse) + A9 (ATLIF threshold).
"""

from __future__ import annotations

# Re-export H1 core
import sys
import os
_h1_sparse = os.path.join(os.path.dirname(__file__), '../../../../../../SDformer/neuron_experiments/H1_hw_sparse/overlay/models/STSwinNet_SNN')
if os.path.isdir(_h1_sparse) and _h1_sparse not in sys.path:
    sys.path.insert(0, os.path.dirname(_h1_sparse))

try:
    from models.STSwinNet_SNN.sparse_gate import (  # noqa: F401 — re-exported
        config_from_dict, freeze_non_gate_parameters, generate_target_layers,
        HardSparseGate, install_fsn_gates, install_hw_sparse_gates,
        install_sparse_gates, iter_sparse_gates, SparseGateConfig,
        sparse_gate_regularization, sparse_gate_summary, threshold_regularization_loss,
    )
except ImportError:
    pass

import torch
import torch.nn as nn
from typing import Iterable, List


# ---------------------------------------------------------------------------
# A5: Refractory-Period Pruning
# ---------------------------------------------------------------------------

def install_refractory_neurons(model: nn.Module, raw_config: dict | None) -> list[str]:
    rc = (raw_config or {}).get("refractory", {})
    if not rc.get("enabled", False):
        return []
    from src.models.modules.spiking_neurons.refractory_neuron import RefractoryNeuron
    from models.STSwinNet_SNN.sparse_gate import generate_target_layers
    if hasattr(model, "sttmultires_unet"):
        swin3d = model.sttmultires_unet.encoders.swin3d
        swin_depths = [len(stage.swin_blocks) for stage in swin3d.layers]
        rtargets = generate_target_layers(swin_depths, rc.get("stage_selection", "all_stages_proj"))
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
            installed.append(name); continue
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        module.spiking_neuron = RefractoryNeuron(current, refractory_steps=refractory_steps, mode=mode).to(device)
        installed.append(name)
    return installed


# ---------------------------------------------------------------------------
# A6: Bipolar Attention Gate
# ---------------------------------------------------------------------------

def generate_attn_qk_targets(swin_depths: List[int], stages: Iterable[int] | None = None) -> tuple[str, ...]:
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

def dual_sparse_regularization(model: nn.Module, raw_config: dict | None) -> torch.Tensor | None:
    dc = (raw_config or {}).get("dual_sparse", {})
    if not dc.get("enabled", False):
        return None
    lambda_firing = float(dc.get("lambda_firing", 0.0))
    lambda_weight = float(dc.get("lambda_weight", 0.0))
    target_rate = float(dc.get("target_rate", 0.05))
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    if lambda_firing > 0:
        from models.STSwinNet_SNN.sparse_gate import iter_sparse_gates
        for _, gate in iter_sparse_gates(model):
            if hasattr(gate, "running_firing_rate"):
                rate = gate.running_firing_rate
                total_loss = total_loss + lambda_firing * (rate - target_rate) ** 2
    if lambda_weight > 0:
        from models.STSwinNet_SNN.sparse_gate import iter_sparse_gates
        for _, gate in iter_sparse_gates(model):
            for param in gate.parameters():
                total_loss = total_loss + lambda_weight * param.abs().sum()
    return total_loss if total_loss.item() != 0.0 else None


# ---------------------------------------------------------------------------
# A9: ATLIF-style adaptive threshold (PSN + ATLIF threshold, NO gate)
# ---------------------------------------------------------------------------

def install_atlif_threshold_neurons(model: nn.Module, raw_config: dict | None) -> list[str]:
    ac = (raw_config or {}).get("atlif_threshold", {})
    if not ac.get("enabled", False):
        return []
    from src.models.modules.spiking_neurons.atlif_threshold_neuron import ATLIFThresholdNeuron
    from models.STSwinNet_SNN.sparse_gate import generate_target_layers
    if hasattr(model, "sttmultires_unet"):
        swin3d = model.sttmultires_unet.encoders.swin3d
        swin_depths = [len(stage.swin_blocks) for stage in swin3d.layers]
        atargets = generate_target_layers(swin_depths, ac.get("stage_selection", "layer0_only"))
    else:
        atargets = tuple(ac.get("target_layers", []))
    named_modules = dict(model.named_modules())
    missing = [n for n in atargets if n not in named_modules]
    if missing:
        raise ValueError(f"ATLIF targets missing: {missing[:5]}...")
    sp_val = float(ac.get("sp", 1e-4))
    v_th = float(ac.get("v_th", 0.1))
    installed = []
    for name in atargets:
        module = named_modules[name]
        if not hasattr(module, "spiking_neuron"):
            continue
        current = module.spiking_neuron
        if isinstance(current, ATLIFThresholdNeuron):
            installed.append(name); continue
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        module.spiking_neuron = ATLIFThresholdNeuron(current, v_th=v_th, sp=sp_val).to(device)
        installed.append(name)
    return installed
