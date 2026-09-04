#!/usr/bin/env python3
"""Generate H82 Class-Major TTX config from the H81 no-motion parent."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
OUTPUT = EXP / "configs/generated/dsec_fullres_w15_H82_class_major_ttx_ft15.yml"
H81_RANK1 = (
    EXP
    / "results/dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/checkpoint_epoch29.pth"
)


def main() -> int:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H82_class_major_ttx_ft15"
    attention = config["bsa_attention"]
    attention["mode"] = "h82"
    attention["binary_motion_xor_alpha"] = 0.0
    attention["hardware_quant_enabled"] = True
    attention["hardware_rtl_shiftmax_enabled"] = False
    attention["hardware_score_step"] = 0.0078125
    attention["hardware_score_min"] = -2.0
    attention["hardware_score_max"] = 2.0
    attention["hardware_gate_step"] = 0.0078125
    attention["hardware_gate_min"] = 0.0
    attention["hardware_gate_max"] = 2.0
    attention["class_stability_regularization_weight"] = 0.01
    config["optimizer"]["lr"] = 5.0e-5
    config["optimizer"]["param_groups"]["backbone_lr"] = 5.0e-5
    config["optimizer"]["milestones"] = [5, 10]
    config["loader"]["n_epochs"] = 15
    runtime = config["runtime"]
    runtime["force_save_epochs"] = [4, 9, 14]
    runtime["state_save_epochs"] = [14]
    runtime["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_h82_class_major_ttx_ft15"
    )
    runtime["h82_parent"] = "H81_nomotion_ep29"
    runtime["h82_init_checkpoint"] = str(H81_RANK1.resolve())
    config["note"] = (
        "H82 Class-Major TTX (C8.3 + C8.1). Parent is H81 no-motion. "
        "Shiftmax is over unique Q7 classes, not tokens. Motion and Local5 stay off. "
        "Q7 STE is on during training so the Class File exists in the graph."
    )
    OUTPUT.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
