"""NTS-11bd: unified H60 Shiftmax on all 12 blocks + two-neuron scope/LR sweep.

Deployment story (inference):
  - Attention: h60 on every encoder block (S0–S3, 12 blocks) — no Legacy split
  - Neurons: ternary Q/K + binary official ATLIF elsewhere (sn2_q explicit)

Short screen (1224 step, valid10) from NB0; winner promotes to full30.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_two_neuron_only_configs import apply_hparam_overrides, read_yaml, write_yaml

RECIPES: dict[str, dict[str, Any]] = {
    "w720_stdlr": {
        "neuron_lr": 3.0e-5,
        "backbone_lr": 1.0e-6,
        "warmup_steps": 720,
        "warmup_start": 0.05,
        "threshold_freeze_after_step": 1224,
    },
    "w720_fastlr": {
        "neuron_lr": 5.0e-5,
        "backbone_lr": 2.0e-6,
        "warmup_steps": 720,
        "warmup_start": 0.05,
        "threshold_freeze_after_step": 1224,
    },
    "w720_slowlr": {
        "neuron_lr": 2.0e-5,
        "backbone_lr": 5.0e-7,
        "warmup_steps": 720,
        "warmup_start": 0.05,
        "threshold_freeze_after_step": 1224,
    },
}


def apply_recipe(cfg: dict[str, Any], recipe: dict[str, Any]) -> None:
    optimizer = cfg.setdefault("optimizer", {})
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["neuron_lr"] = float(recipe["neuron_lr"])
    groups["backbone_lr"] = float(recipe["backbone_lr"])
    warmup = optimizer.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = int(recipe["warmup_steps"])
    warmup["start_factor"] = float(recipe["warmup_start"])
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["threshold_freeze_after_step"] = int(recipe["threshold_freeze_after_step"])

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
MANIFEST = GENERATED / "nts11bd_unified_attn_manifest.json"
AA_FULL = EXP_ROOT / "configs/nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_20260612_065413.yml"
NB0 = Path(__file__).resolve().parents[3] / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"

ALL12_BLOCKS = (
    "0:0",
    "0:1",
    "1:0",
    "1:1",
    "2:0",
    "2:1",
    "2:2",
    "2:3",
    "2:4",
    "2:5",
    "3:0",
    "3:1",
)

SN2Q_PATHS = [
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.1.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.1.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.3.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.4.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.5.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.attn.sn2_q",
    "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.1.attn.sn2_q",
]
DOWNSAMPLE_PATHS = [
    "sttmultires_unet.encoders.swin3d.layers.0.downsample.sn",
    "sttmultires_unet.encoders.swin3d.layers.1.downsample.sn",
    "sttmultires_unet.encoders.swin3d.layers.2.downsample.sn",
]
FFN_S0_PATHS = [
    f"sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.{b}.mlp.sn{n}"
    for b in (0, 1)
    for n in (1, 2)
]
FFN_S2_PATHS = [
    f"sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{b}.mlp.sn{n}"
    for b in range(6)
    for n in (1, 2)
]
STATIC_PATHS = {
    "sn2q": SN2Q_PATHS,
    "downsample": DOWNSAMPLE_PATHS,
    "ffn_s0": FFN_S0_PATHS,
    "ffn_s2": FFN_S2_PATHS,
}


def binary_group(name: str, *, paths: list[str] | None = None, path_selection: str = "") -> dict[str, Any]:
    group: dict[str, Any] = {
        "name": name,
        "output_mode": "binary",
        "threshold_mode": "official_atlif",
        "center_mode": "zero",
        "threshold_eta": 0.0,
        "activity_eta": 0.0,
        "target_rate": None,
        "target_rate_eta": 0.0,
    }
    if path_selection:
        group["path_selection"] = path_selection
    if paths is not None:
        group["paths"] = list(paths)
    return group


def ternary_group(name: str, paths: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "paths": list(paths),
        "output_mode": "ternary",
        "threshold_mode": "symmetric_bsa_tsn",
        "center_mode": "bias",
        "threshold_eta": 6.5e-4,
        "threshold_lr_scale": 50000.0,
        "activity_eta": 0.0,
        "target_rate": None,
        "target_rate_eta": 0.0,
    }


def apply_scope_policy(cfg: dict[str, Any], policy: str, paths: dict[str, list[str]]) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["enabled"] = True
    atlif["target"] = "qk"
    atlif["stage_selection"] = "all"
    atlif.pop("target_paths", None)
    groups: list[dict[str, Any]] = []

    if policy == "sn2q_binary":
        atlif["target_groups"] = [
            binary_group("sn2q_binary", paths=paths["sn2q"]),
            binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
        ]
        return
    if policy == "downsample_ternary":
        groups.extend(
            [
                ternary_group("downsample_ternary", paths["downsample"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "ffn_s2_ternary":
        groups.extend(
            [
                ternary_group("ffn_s2_ternary", paths["ffn_s2"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "downsample_ffn_s2_ternary":
        groups.extend(
            [
                ternary_group("downsample_ternary", paths["downsample"]),
                ternary_group("ffn_s2_ternary", paths["ffn_s2"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "downsample_ffn_s0_ternary":
        groups.extend(
            [
                ternary_group("downsample_ternary", paths["downsample"]),
                ternary_group("ffn_s0_ternary", paths["ffn_s0"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    else:
        raise ValueError(f"unknown scope policy: {policy}")

    groups.append(binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"))
    atlif["target_groups"] = groups


def apply_unified_h60_attention(cfg: dict[str, Any]) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "h60"
    attn.pop("stage_selection", None)
    attn["target_blocks"] = list(ALL12_BLOCKS)


def make_short_config(spec: dict[str, Any], paths: dict[str, list[str]]) -> tuple[Path, dict[str, Any]]:
    base = read_yaml(AA_FULL)
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]

    apply_scope_policy(cfg, str(spec["scope_policy"]), paths)
    apply_recipe(cfg, RECIPES[str(spec["recipe"])])
    if spec.get("attn"):
        apply_hparam_overrides(cfg, spec["attn"])
    apply_unified_h60_attention(cfg)

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 1224
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    out = GENERATED / f"{spec['name']}_s1224.yml"
    write_yaml(out, cfg)
    manifest = {
        "name": spec["name"],
        "config": str(out),
        "resume": str(spec["resume"]),
        "track": spec["track"],
        "full_epochs": int(spec["full_epochs"]),
        "scope_policy": spec["scope_policy"],
        "recipe": spec["recipe"],
        "attention": "h60_all12",
    }
    if spec.get("attn"):
        manifest["attn"] = spec["attn"]
    return out, manifest


def main() -> int:
    if not NB0.is_file():
        raise FileNotFoundError(f"missing NB0 checkpoint: {NB0}")
    if not AA_FULL.is_file():
        raise FileNotFoundError(f"missing 11aa base config: {AA_FULL}")

    paths = STATIC_PATHS
    specs: list[dict[str, Any]] = [
        {
            "name": "nts11bd_u12_sn2q_w720_stdlr",
            "scope_policy": "sn2q_binary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; Q/K ternary only (sn2q binary scope); warm720/stdlr.",
        },
        {
            "name": "nts11bd_u12_sn2q_w720_fastlr",
            "scope_policy": "sn2q_binary",
            "recipe": "w720_fastlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; sn2q scope; warm720/fastlr.",
        },
        {
            "name": "nts11bd_u12_sn2q_w720_slowlr",
            "scope_policy": "sn2q_binary",
            "recipe": "w720_slowlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; sn2q scope; warm720/slowlr.",
        },
        {
            "name": "nts11bd_u12_ds_w720_stdlr",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; downsample ternary scope; warm720/stdlr.",
        },
        {
            "name": "nts11bd_u12_ds_w720_fastlr",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_fastlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; downsample ternary; warm720/fastlr (11aq recipe).",
        },
        {
            "name": "nts11bd_u12_ds_w720_slowlr",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_slowlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; downsample ternary; warm720/slowlr.",
        },
        {
            "name": "nts11bd_u12_dsffn2_w720_stdlr",
            "scope_policy": "downsample_ffn_s2_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; downsample+S2 FFN ternary; stdlr.",
        },
        {
            "name": "nts11bd_u12_dsffn2_w720_fastlr",
            "scope_policy": "downsample_ffn_s2_ternary",
            "recipe": "w720_fastlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; downsample+S2 FFN ternary; fastlr.",
        },
        {
            "name": "nts11bd_u12_dsffn0_w720_stdlr",
            "scope_policy": "downsample_ffn_s0_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; downsample+S0 FFN ternary; stdlr.",
        },
        {
            "name": "nts11bd_u12_ffn2_w720_stdlr",
            "scope_policy": "ffn_s2_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "11bd: unified h60 all12; S2 FFN-only ternary; stdlr.",
        },
        {
            "name": "nts11bd_u12_ds_w720_stdlr_mu03",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "attn": {"bipolar_mu": 0.03},
            "note": "11bd: unified h60 all12; downsample scope; stdlr; bipolar_mu=0.03.",
        },
        {
            "name": "nts11bd_u12_ds_w720_stdlr_mu08",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "attn": {"bipolar_mu": 0.08},
            "note": "11bd: unified h60 all12; downsample scope; stdlr; bipolar_mu=0.08.",
        },
        {
            "name": "nts11bd_u12_ds_w720_fastlr_a001",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_fastlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "attn": {"alpha0": 0.01},
            "note": "11bd: unified h60 all12; downsample scope; fastlr; alpha0=0.01.",
        },
    ]

    manifests: list[dict[str, Any]] = []
    for spec in specs:
        _, manifest = make_short_config(spec, paths)
        manifests.append(manifest)
        print(manifest["config"])

    MANIFEST.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"# manifest: {MANIFEST} ({len(manifests)} specs, h60 blocks={len(ALL12_BLOCKS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())