"""Teacher-model helpers for H55 distillation experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


def build_teacher_model(config: dict[str, Any], device: torch.device, remap=None) -> torch.nn.Module | None:
    """Build a frozen baseline PSN teacher from ``teacher_distill.checkpoint``.

    The teacher deliberately avoids H9 overlay installers: it is the original
    baseline architecture loaded from the configured checkpoint.
    """

    cfg = config.get("teacher_distill") or {}
    if not bool(cfg.get("enabled", False)):
        return None
    checkpoint = str(cfg.get("checkpoint", "") or "")
    if not checkpoint:
        raise ValueError("[H55] teacher_distill.enabled=true but checkpoint is empty")

    from models.STSwinNet_SNN.Spiking_STSwinNet import (  # type: ignore
        MS_SpikingformerFlowNet,
        MS_SpikingformerFlowNet_en4,
        SpikingformerFlowNet,
    )
    from spikingjelly.activation_based import functional
    from utils.runtime_backend import configure_snn_backend
    from utils.utils import load_model
    from spikingjelly.activation_based import neuron
    from models.STSwinNet_SNN import Spiking_submodules as spiking_submodules  # type: ignore

    constructors = {
        "SpikingformerFlowNet": SpikingformerFlowNet,
        "MS_SpikingformerFlowNet": MS_SpikingformerFlowNet,
        "MS_SpikingformerFlowNet_en4": MS_SpikingformerFlowNet_en4,
    }
    model_name = str(config["model"]["name"])
    if config["swin_transformer"]["use_arc"][0]:
        teacher = constructors[model_name](deepcopy(config["model"]), deepcopy(config["swin_transformer"]))
    else:
        teacher = constructors[model_name](deepcopy(config["model"]))
    teacher.to(device)
    teacher.init_weights()
    teacher = load_model(checkpoint, teacher, device, remap)

    neuron_type = str(config["model"]["spiking_neuron"]["neuron_type"])
    if neuron_type == "if":
        neurontype = getattr(neuron, "IFNode")
    elif neuron_type == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif neuron_type == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif neuron_type == "glif":
        neurontype = getattr(spiking_submodules, "GatedLIFNode")
    elif neuron_type == "psn":
        neurontype = getattr(spiking_submodules, "PSN")
    elif neuron_type == "SLTTlif":
        neurontype = getattr(spiking_submodules, "SLTTLIFNode")
    else:
        raise ValueError(f"[H55] unsupported teacher neuron type: {neuron_type}")

    functional.reset_net(teacher)
    functional.set_step_mode(teacher, config["data"]["step_mode"])
    configure_snn_backend(teacher, device, config, neurontype)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print(f"[H55] teacher loaded from {checkpoint}")
    return teacher


@torch.no_grad()
def teacher_forward(teacher: torch.nn.Module | None, chunk: torch.Tensor, config: dict[str, Any]):
    if teacher is None:
        return None
    from spikingjelly.activation_based import functional

    functional.reset_net(teacher)
    functional.set_step_mode(teacher, config["data"]["step_mode"])
    teacher.eval()
    return teacher(chunk)["flow"]
