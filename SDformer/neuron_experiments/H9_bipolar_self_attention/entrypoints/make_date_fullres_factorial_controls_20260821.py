#!/usr/bin/env python3
"""Generate same-parent full-resolution neuron/attention factorial controls."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
TEMPLATE = GENERATED / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
PARENT = (
    EXP
    / "results/dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/checkpoint_epoch29.pth"
)
STAMP = "20260821"

ROWS = (
    ("c00_psn_original", False, False, 0.0),
    ("c10_binary_original", True, False, 0.0),
    ("c01_psn_ttx", False, True, 0.0),
    ("c11_binary_ttx", True, True, 0.0),
    ("c12_binary_motion_ttx", True, True, 0.25),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_config(
    template: dict,
    name: str,
    atlif_enabled: bool,
    attention_enabled: bool,
    motion_alpha: float,
) -> dict:
    cfg = copy.deepcopy(template)
    experiment = f"dsec_fullres_w15_factorial_{name}_nb0ep29_ft10_{STAMP}"
    cfg["experiment"] = experiment
    cfg["swin_transformer"]["window_size"] = [2, 15, 15]
    cfg["swin_transformer"]["pretrained_window_size"] = [2, 15, 15]

    atlif = cfg["atlif_ternary_psn"]
    atlif["enabled"] = atlif_enabled
    if not atlif_enabled:
        atlif["target"] = "none"
        atlif["target_groups"] = []

    attention = cfg["bsa_attention"]
    attention["enabled"] = attention_enabled
    attention["mode"] = "h60"
    attention["binary_motion_xor_alpha"] = float(motion_alpha)
    if not attention_enabled:
        attention["target_blocks"] = []

    cfg["optimizer"]["milestones"] = []
    cfg["loader"]["n_epochs"] = 10
    cfg["loader"]["batch_size"] = 2
    cfg["loader"]["crop"] = None
    cfg["loader"]["resolution"] = [480, 640]
    cfg["loader"]["n_workers"] = 8
    cfg["loader"]["persistent_workers"] = True
    cfg["loader"]["prefetch_factor"] = 2
    cfg["test"]["bn_policy"] = "no_running"
    cfg["test"]["n_valid"] = 1
    runtime = cfg["runtime"]
    runtime["seed"] = 0
    runtime["force_save_epochs"] = [4, 9]
    runtime["state_save_epochs"] = [9]
    runtime["save_only_force_epochs"] = True
    runtime["full_resolution_protocol"] = (
        "date_same_parent_nb0ep29_fresh_optimizer_480x640_"
        "window2x15x15_ft10"
    )
    runtime["factorial_parent_checkpoint"] = str(PARENT.resolve())
    runtime["factorial_parent_checkpoint_sha256"] = sha256(PARENT)
    runtime["factorial_cell"] = name
    cfg["note"] = (
        "DATE same-parent causal control. Fresh optimizer, seed0, 10 full-resolution "
        f"epochs from frozen NB0 ep29; ATLIF={atlif_enabled}, "
        f"TTX={attention_enabled}, motion_alpha={motion_alpha}."
    )
    return cfg


def main() -> int:
    if not TEMPLATE.is_file() or not PARENT.is_file():
        raise FileNotFoundError(TEMPLATE if not TEMPLATE.is_file() else PARENT)
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    generated = []
    for name, atlif_enabled, attention_enabled, motion_alpha in ROWS:
        cfg = make_config(
            template, name, atlif_enabled, attention_enabled, motion_alpha
        )
        path = GENERATED / f"{cfg['experiment']}.yml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        generated.append(
            {
                "cell": name,
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
                "atlif_enabled": atlif_enabled,
                "attention_enabled": attention_enabled,
                "motion_alpha": motion_alpha,
            }
        )
    manifest = {
        "schema": "date_fullres_factorial_controls_v1",
        "parent_checkpoint": str(PARENT.resolve()),
        "parent_checkpoint_sha256": sha256(PARENT),
        "template": str(TEMPLATE.resolve()),
        "template_sha256": sha256(TEMPLATE),
        "shared_protocol": {
            "resolution": [480, 640],
            "crop": None,
            "window_size": [2, 15, 15],
            "bn_policy": "no_running",
            "seed": 0,
            "fresh_optimizer": True,
            "epochs": 10,
            "evaluation_epochs": [4, 9],
        },
        "controls": generated,
    }
    manifest_path = GENERATED / f"date_fullres_factorial_controls_{STAMP}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
