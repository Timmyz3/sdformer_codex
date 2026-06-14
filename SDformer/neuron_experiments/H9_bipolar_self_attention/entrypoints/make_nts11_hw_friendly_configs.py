"""NTS-11 hardware-friendly scope configs (DATE 2027 co-design).

No model import — paths derived from 11aa yaml + block ids.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

RECIPES: dict[str, dict[str, Any]] = {
    "w720_stdlr": {
        "neuron_lr": 3.0e-5,
        "backbone_lr": 1.0e-6,
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
MANIFEST = GENERATED / "nts11_hw_friendly_manifest.json"
AA_FULL = EXP_ROOT / "configs/nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_20260612_065413.yml"
NB0 = Path(__file__).resolve().parents[3] / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
AAH0 = (
    EXP_ROOT
    / "results/nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15_bs8_20260612_194020_setsid"
    / "checkpoint_epoch0.pth"
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
S23_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5", "3:0", "3:1"]
S01_BLOCKS = ["0:0", "0:1", "1:0", "1:1"]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


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


def qk_paths_for_blocks(block_ids: list[str]) -> list[str]:
    paths: list[str] = []
    for block_id in block_ids:
        stage, blk = block_id.split(":")
        base = f"sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{blk}.attn"
        paths.extend([f"{base}.sn_q", f"{base}.sn_k"])
    return sorted(paths)


def apply_sn2q_scope(cfg: dict[str, Any]) -> None:
    """11r / 11aw: Q/K ternary (default target:qk) + sn2_q + all_non_qk binary; NO downsample ternary."""
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["enabled"] = True
    atlif["target"] = "qk"
    atlif["stage_selection"] = "all"
    atlif["output_mode"] = "ternary"
    atlif["threshold_mode"] = "symmetric_bsa_tsn"
    atlif["center_mode"] = "bias"
    atlif.pop("target_paths", None)
    atlif["target_groups"] = [
        binary_group("sn2q_binary", paths=SN2Q_PATHS),
        binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
    ]


def apply_qk_s23_scope(cfg: dict[str, Any]) -> None:
    """11ax: ternary Q/K only on H60 blocks; S0/S1 Q/K binary."""
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["enabled"] = True
    atlif["target"] = "custom"
    atlif["stage_selection"] = "all"
    atlif["output_mode"] = "ternary"
    atlif["threshold_mode"] = "symmetric_bsa_tsn"
    atlif["center_mode"] = "bias"
    atlif["target_paths"] = qk_paths_for_blocks(S23_BLOCKS)
    atlif["target_groups"] = [
        binary_group("s01_qk_binary", paths=qk_paths_for_blocks(S01_BLOCKS)),
        binary_group("sn2q_binary", paths=SN2Q_PATHS),
        binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
    ]


def make_config(spec: dict[str, Any], *, short: bool) -> Path:
    cfg = deepcopy(read_yaml(AA_FULL))
    cfg["experiment"] = spec["name"] + ("_s1224" if short else "_scope_full30")
    cfg["note"] = spec["note"]

    policy = str(spec["scope_policy"])
    if policy in {"sn2q_binary", "downsample_binary_ablation"}:
        apply_sn2q_scope(cfg)
    elif policy == "qk_s23_ternary":
        apply_qk_s23_scope(cfg)
    else:
        raise ValueError(policy)

    apply_recipe(cfg, RECIPES[str(spec["recipe"])])

    loader = cfg.setdefault("loader", {})
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10 if short else 40
    runtime = cfg.setdefault("runtime", {})
    runtime["use_mlflow_model_logging"] = False

    if short:
        loader["n_epochs"] = 1
        loader["batch_size"] = 8
        runtime["max_train_steps"] = 1224
        runtime["skip_state_save"] = True
        runtime["force_save_epochs"] = [0]
    else:
        loader["n_epochs"] = int(spec.get("full_epochs", 30))
        loader["batch_size"] = 8
        runtime["max_train_steps"] = 0
        runtime["skip_state_save"] = False
        runtime["force_save_epochs"] = [0, 9, 14, 19, 24, 28, 29]

    suffix = "_s1224.yml" if short else "_scope_full30.yml"
    out = GENERATED / f"{spec['name']}{suffix}"
    write_yaml(out, cfg)
    return out


SPECS: list[dict[str, Any]] = [
    {
        "name": "nts11aw_hw_h60_s23_sn2qbin_w720_stdlr",
        "scope_policy": "sn2q_binary",
        "recipe": "w720_stdlr",
        "resume": str(NB0),
        "track": "full30",
        "note": "HW-friendly 11aw: no downsample ternary, warm720/stdlr, NB0 full30.",
    },
    {
        "name": "nts11ax_hw_h60_s23_qks23_w720_stdlr",
        "scope_policy": "qk_s23_ternary",
        "recipe": "w720_stdlr",
        "resume": str(NB0),
        "track": "full30",
        "note": "HW-friendly 11ax: s23-only Q/K ternary, warm720/stdlr, NB0 full30.",
    },
    {
        "name": "nts11ay_hw_h60_s23_dsbin_w720_stdlr",
        "scope_policy": "downsample_binary_ablation",
        "recipe": "w720_stdlr",
        "resume": str(NB0),
        "track": "full30",
        "note": "HW ablation 11ay: identical scope to 11aw (downsample reverted to binary).",
    },
    {
        "name": "nts11az_hw_h60_s23_sn2qbin_w720_stdlr_ftaah0",
        "scope_policy": "sn2q_binary",
        "recipe": "w720_stdlr",
        "resume": str(AAH0),
        "track": "finetune",
        "full_epochs": 5,
        "note": "HW finetune 11az: sn2q scope from 11aah ft ep0, 5ep polish.",
    },
]


def main() -> int:
    if not NB0.is_file():
        raise FileNotFoundError(f"missing NB0: {NB0}")

    manifests: list[dict[str, Any]] = []
    for spec in SPECS:
        short_path = make_config(spec, short=True)
        full_path = make_config(spec, short=False)
        manifests.append(
            {
                "name": spec["name"],
                "config_short": str(short_path),
                "config_full": str(full_path),
                "resume": str(spec["resume"]),
                "track": spec["track"],
                "scope_policy": spec["scope_policy"],
                "recipe": spec["recipe"],
            }
        )
        print(short_path)
        print(full_path)

    MANIFEST.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"# manifest: {MANIFEST} ({len(manifests)} specs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())