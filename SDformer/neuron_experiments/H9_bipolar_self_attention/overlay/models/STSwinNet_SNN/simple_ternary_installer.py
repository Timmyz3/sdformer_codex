"""Install SimpleTernaryPSN on SDFormerFlow attention Q/K neurons."""

from __future__ import annotations

import torch
import torch.nn as nn

from .simple_ternary_psn import SimpleTernaryPSN


def _module_device(module: nn.Module) -> torch.device:
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device("cpu")


def install_simple_ternary_psn(
    model: nn.Module, raw_config: dict | None
) -> list[str]:
    """Replace attention sn_q/sn_k with SimpleTernaryPSN.

    Config key: ``simple_ternary_psn``
      enabled: true
      stage_selection: all | layer0_only | stage{idx}
      theta_init: 1.0
      center_mode: bias | zero
    """
    ac = (raw_config or {}).get("simple_ternary_psn", {})
    if not ac.get("enabled", False):
        return []

    swin3d = model.sttmultires_unet.encoders.swin3d
    stages = ac.get("stage_selection", "all")
    if stages == "all":
        stage_ids = range(len(swin3d.layers))
    elif stages == "layer0_only":
        stage_ids = [0]
    else:
        stage_ids = [int(stages.replace("stage", ""))]

    theta_init = float(ac.get("theta_init", 1.0))
    center_mode = str(ac.get("center_mode", "bias"))

    installed = []
    for si in stage_ids:
        stage = swin3d.layers[si]
        for bi, block in enumerate(stage.swin_blocks):
            for target in ("sn_q", "sn_k"):
                wrapper = getattr(block.attn, target)
                if not hasattr(wrapper, "spiking_neuron"):
                    continue
                current = wrapper.spiking_neuron
                if isinstance(current, SimpleTernaryPSN):
                    print(f"[ST] skip {si}:{bi}.attn.{target} — already installed")
                    continue
                device = _module_device(wrapper)
                T = getattr(current, "T", getattr(wrapper, "num_steps", 10))
                wrapper.spiking_neuron = SimpleTernaryPSN(
                    T=T,
                    base_psn=current,
                    theta_init=theta_init,
                    center_mode=center_mode,
                ).to(device)
                installed.append(f"layers.{si}.swin_blocks.{bi}.attn.{target}")

    print(f"[ST] installed SimpleTernaryPSN: {len(installed)} modules, "
          f"theta_init={theta_init}, center_mode={center_mode}")
    return installed
