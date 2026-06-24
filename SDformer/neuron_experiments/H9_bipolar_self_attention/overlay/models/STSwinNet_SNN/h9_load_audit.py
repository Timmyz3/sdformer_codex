"""Checkpoint/config compatibility checks for H9 overlay experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


H9_OVERLAY_KEY_MARKERS = (
    ".linear_v.",
    ".bn_v.",
    ".sn_v.",
    ".spiking_neuron.thresh",
    ".spiking_neuron.center",
)


def is_h9_overlay_key(key: str) -> bool:
    return any(marker in key for marker in H9_OVERLAY_KEY_MARKERS)


def config_requires_h9_overlay(config: dict[str, Any] | None) -> bool:
    config = config or {}
    return bool(
        config.get("atlif_ternary_psn", {}).get("enabled")
        or config.get("bsa_attention", {}).get("enabled")
        or config.get("simple_ternary_psn", {}).get("enabled")
    )


def extract_state_dict(pretrained_model: Any, *, test: bool = True) -> dict[str, torch.Tensor]:
    if hasattr(pretrained_model, "state_dict") and not isinstance(pretrained_model, dict):
        pretrained_dict = pretrained_model.state_dict()
    elif isinstance(pretrained_model, dict):
        if "model_state_dict" in pretrained_model:
            pretrained_dict = pretrained_model["model_state_dict"]
        elif "state_dict" in pretrained_model:
            pretrained_dict = pretrained_model["state_dict"]
        elif "model" in pretrained_model and hasattr(pretrained_model["model"], "state_dict"):
            pretrained_dict = pretrained_model["model"].state_dict()
        else:
            pretrained_dict = pretrained_model
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(pretrained_model)}")

    if test:
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    return pretrained_dict


def load_checkpoint_with_h9_audit(
    checkpoint: str | os.PathLike[str],
    model: nn.Module,
    device: torch.device,
    *,
    config: dict[str, Any] | None,
    remap: str | None = None,
    test: bool = True,
) -> nn.Module:
    """Load a local checkpoint and fail fast on H9 overlay/config mismatches.

    Baseline SDFormerFlow uses ``strict=False`` while loading. That is useful for
    pretrained interpolation, but dangerous for H9 experiments: using a baseline
    eval config with an H9 checkpoint silently drops overlay parameters and
    produces invalid metrics. This helper keeps strict=False for normal missing
    non-overlay keys, but treats overlay mismatch as fatal.
    """

    checkpoint = str(checkpoint)
    if not Path(checkpoint).is_file():
        from utils.utils import load_model

        return load_model(checkpoint, model, device, remap=remap, test=test)

    pretrained_model = torch.load(checkpoint, map_location=device, weights_only=False)
    pretrained_dict = extract_state_dict(pretrained_model, test=test)
    if remap == "v2":
        from utils.utils import remap_pretrained_keys_swin

        pretrained_dict = remap_pretrained_keys_swin(model, pretrained_dict)
    elif remap == "v1":
        from utils.utils import load_pretrained_interpolate

        load_pretrained_interpolate(model, pretrained_dict)
        del pretrained_model
        torch.cuda.empty_cache()
        print("Model restored from local checkpoint " + checkpoint + "\n")
        return model

    overlay_checkpoint_keys = [key for key in pretrained_dict if is_h9_overlay_key(key)]
    overlay_model_keys = [key for key in model.state_dict() if is_h9_overlay_key(key)]
    h9_enabled = config_requires_h9_overlay(config)
    if overlay_checkpoint_keys and not h9_enabled:
        raise RuntimeError(
            "Checkpoint contains H9 overlay parameters but the current config does not enable "
            "ATLIF/BSA/simple ternary modules; this requires an H9 config. "
            f"Example keys: {overlay_checkpoint_keys[:8]}"
        )
    if overlay_model_keys and not overlay_checkpoint_keys:
        raise RuntimeError(
            "Model contains stateful H9 overlay modules but checkpoint does not contain H9 overlay "
            f"parameters: {checkpoint}"
        )

    incompatible = model.load_state_dict(pretrained_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    overlay_missing = [key for key in missing if is_h9_overlay_key(key)]
    overlay_unexpected = [key for key in unexpected if is_h9_overlay_key(key)]
    print(
        f"[H9] load audit: checkpoint_overlay_keys={len(overlay_checkpoint_keys)}, "
        f"model_overlay_keys={len(overlay_model_keys)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[H9] missing keys sample: {missing[:12]}")
    if unexpected:
        print(f"[H9] unexpected keys sample: {unexpected[:12]}")
    if overlay_unexpected:
        raise RuntimeError(
            "[H9] overlay checkpoint keys were not registered before load: "
            + str(overlay_unexpected[:20])
        )
    if overlay_checkpoint_keys and overlay_missing:
        raise RuntimeError(
            "[H9] checkpoint contains overlay parameters but matching model keys are missing: "
            + str(overlay_missing[:20])
        )

    del pretrained_model
    torch.cuda.empty_cache()
    print("Model restored from local checkpoint " + checkpoint + "\n")
    return model
