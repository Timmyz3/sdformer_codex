#!/usr/bin/env python3
"""Generate the Local5 same-parent control that follows the DATE factorial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
TEMPLATE = GENERATED / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml"
PARENT = (
    EXP
    / "results/dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/checkpoint_epoch29.pth"
)
CONFIG = GENERATED / "dsec_fullres_w15_factorial_c20_binary_local5_nb0ep29_ft10_20260821.yml"
MANIFEST = GENERATED / "date_fullres_local5_same_parent_control_20260821.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    for path in (TEMPLATE, PARENT):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    cfg["experiment"] = "dsec_fullres_w15_factorial_c20_binary_local5_nb0ep29_ft10_20260821"
    cfg["swin_transformer"]["window_size"] = [2, 15, 15]
    cfg["swin_transformer"]["pretrained_window_size"] = [2, 15, 15]
    cfg["loader"]["n_epochs"] = 10
    cfg["loader"]["batch_size"] = 2
    cfg["loader"]["resolution"] = [480, 640]
    cfg["loader"]["crop"] = None
    cfg["loader"]["n_workers"] = 8
    cfg["loader"]["persistent_workers"] = True
    cfg["loader"]["prefetch_factor"] = 2
    cfg["test"]["n_valid"] = 1
    cfg["test"]["bn_policy"] = "no_running"
    cfg["optimizer"]["milestones"] = []
    runtime = cfg["runtime"]
    runtime["seed"] = 0
    runtime["force_save_epochs"] = [4, 9]
    runtime["state_save_epochs"] = [4, 9]
    runtime["save_only_force_epochs"] = True
    runtime["full_resolution_protocol"] = (
        "date_same_parent_nb0ep29_fresh_optimizer_480x640_"
        "window2x15x15_local5_ft10"
    )
    runtime["factorial_parent_checkpoint"] = str(PARENT.resolve())
    runtime["factorial_parent_checkpoint_sha256"] = sha256(PARENT)
    runtime["factorial_cell"] = "c20_binary_local5"
    cfg["note"] = (
        "DATE Local5 same-parent control. Fresh optimizer, seed0, 10 full-resolution "
        "epochs from frozen NB0 ep29; all-binary ATLIF and all12 Local5."
    )
    CONFIG.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    manifest = {
        "schema": "date_fullres_local5_same_parent_control_v1",
        "config": str(CONFIG.resolve()),
        "config_sha256": sha256(CONFIG),
        "template": str(TEMPLATE.resolve()),
        "template_sha256": sha256(TEMPLATE),
        "parent_checkpoint": str(PARENT.resolve()),
        "parent_checkpoint_sha256": sha256(PARENT),
        "protocol": {
            "resolution": [480, 640],
            "crop": None,
            "window_size": [2, 15, 15],
            "bn_policy": "no_running",
            "seed": 0,
            "fresh_optimizer": True,
            "epochs": 10,
            "evaluation_epochs": [4, 9],
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
