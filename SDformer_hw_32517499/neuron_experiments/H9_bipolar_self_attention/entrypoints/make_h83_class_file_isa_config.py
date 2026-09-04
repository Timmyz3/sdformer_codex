#!/usr/bin/env python3
"""Generate H83 Class File ISA config. Do not launch while H82 owns the GPU."""

from __future__ import annotations

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H82_class_major_ttx_ft15.yml"
OUTPUT = EXP / "configs/generated/dsec_fullres_w15_H83_class_file_isa_ft15.yml"


def main() -> int:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H83_class_file_isa_ft15"
    attention = config["bsa_attention"]
    attention["mode"] = "h83"
    attention["binary_motion_xor_alpha"] = 0.0
    attention["preserve_mean"] = False
    attention["class_stability_regularization_weight"] = 0.01
    config["runtime"]["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_h83_class_file_isa_ft15"
    )
    config["note"] = (
        "H83 Class File ISA. Occupied class records are the executed object. "
        "K expands from that file. No Motion, no Local5, no preserve_mean*450. "
        "Do not start this job while H82 still owns the GPU."
    )
    OUTPUT.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
