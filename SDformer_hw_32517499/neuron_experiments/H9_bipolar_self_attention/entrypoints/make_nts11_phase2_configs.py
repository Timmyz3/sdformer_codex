"""NTS-11 phase-2 short-test configs: layered Q/K schedule + decoder ablations."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_two_neuron_only_configs import (
    BASE,
    apply_hparam_overrides,
    apply_two_neuron_only_policy,
    blocks_for,
    read_yaml,
    set_runtime,
    write_yaml,
)


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"

S0_FFN_PATHS = [
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2",
]

DECODER_HEAD_PATHS = [
    "sttmultires_unet.decoders.0.sn",
    "sttmultires_unet.decoders.1.sn",
    "sttmultires_unet.decoders.2.sn",
    "sttmultires_unet.decoders.3.sn",
    "sttmultires_unet.preds.0.sn",
    "sttmultires_unet.preds.1.sn",
    "sttmultires_unet.preds.2.sn",
    "sttmultires_unet.preds.3.sn",
    "sttmultires_unet.resblocks.0.sn1",
    "sttmultires_unet.resblocks.0.sn2",
    "sttmultires_unet.resblocks.1.sn1",
    "sttmultires_unet.resblocks.1.sn2",
]

DECODER_HEAD_EXCLUDE_PREFIXES = [
    "sttmultires_unet.decoders.",
    "sttmultires_unet.preds.",
    "sttmultires_unet.resblocks.",
]


def binary_group(name: str, **overrides: Any) -> dict[str, Any]:
    group = {
        "name": name,
        "output_mode": "binary",
        "threshold_mode": "official_atlif",
        "center_mode": "zero",
        "threshold_eta": 0.0,
        "activity_eta": 0.0,
        "target_rate": None,
        "target_rate_eta": 0.0,
    }
    group.update(overrides)
    return group


def apply_target_group_policy(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    policy = str(spec.get("target_group_policy", "all_non_qk"))
    atlif = cfg.setdefault("atlif_ternary_psn", {})

    if policy == "all_non_qk":
        return

    if policy == "layered_s0_plus_all_non_qk":
        atlif["target_groups"] = [
            binary_group("s0_ffn", paths=S0_FFN_PATHS, threshold_lr_scale=8000.0),
            binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
        ]
        return

    if policy == "decoder_head_soft_plus_encoder":
        atlif["target_groups"] = [
            binary_group("decoder_head_binary", paths=DECODER_HEAD_PATHS, threshold_lr_scale=3000.0),
            binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
        ]
        return

    if policy == "vanilla_decoder_head":
        atlif["target_groups"] = [
            binary_group(
                "encoder_only_binary_atlif",
                path_selection="all_non_qk",
                exclude_path_prefixes=DECODER_HEAD_EXCLUDE_PREFIXES,
            ),
        ]
        return

    raise ValueError(f"unknown target_group_policy: {policy}")


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    set_runtime(cfg, spec["name"], spec["note"])
    apply_two_neuron_only_policy(cfg)
    apply_target_group_policy(cfg, spec)
    apply_hparam_overrides(cfg, spec)

    if "stage_threshold_lr_scale" in spec:
        cfg.setdefault("atlif_ternary_psn", {})["stage_threshold_lr_scale"] = dict(
            spec["stage_threshold_lr_scale"]
        )

    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for(str(spec.get("scope", "s23")), base)

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts11h_hw_h60_s23_two_neuron_stage_qkscale_s1224",
            "note": (
                "NTS-11h: per-Swin-stage Q/K threshold_lr_scale "
                "{0:25k,1:35k,2:50k,3:50k}; shallow layers adapt slower."
            ),
            "stage_threshold_lr_scale": {0: 25000.0, 1: 35000.0, 2: 50000.0, 3: 50000.0},
        },
        {
            "name": "nts11i_hw_h60_s23_two_neuron_layered_s0ffn_s1224",
            "note": (
                "NTS-11i: restore NTS-10d s0_ffn path group (scale 8k) before all_non_qk. "
                "Tests shallow-encoder layering on top of two-neuron policy."
            ),
            "target_group_policy": "layered_s0_plus_all_non_qk",
        },
        {
            "name": "nts11j_hw_h60_s23_two_neuron_vanilla_decoder_s1224",
            "note": (
                "NTS-11j: binary ATLIF on encoder only; keep decoder/pred/head resblocks "
                "as vanilla PSN (12 modules). Ablation for decoder replacement necessity."
            ),
            "target_group_policy": "vanilla_decoder_head",
        },
        {
            "name": "nts11k_hw_h60_s23_two_neuron_decoder_soft_s1224",
            "note": (
                "NTS-11k: explicit decoder-head group (12 paths) with lower threshold_lr_scale=3k, "
                "then all_non_qk for the rest. Tests gentler decoder adaptation."
            ),
            "target_group_policy": "decoder_head_soft_plus_encoder",
        },
        {
            "name": "nts11l_hw_h60_s23_two_neuron_fastlr_freeze816_s1224",
            "note": (
                "NTS-11l: combine phase-1 best LR direction (fast neuron/backbone) "
                "with earlier threshold freeze816."
            ),
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
            "threshold_freeze_after_step": 816,
        },
        {
            "name": "nts11m_hw_h60_s23_two_neuron_stage_freeze816_s1224",
            "note": (
                "NTS-11m: stage Q/K threshold_lr_scale (11h) + freeze816 combo."
            ),
            "stage_threshold_lr_scale": {0: 25000.0, 1: 35000.0, 2: 50000.0, 3: 50000.0},
            "threshold_freeze_after_step": 816,
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())