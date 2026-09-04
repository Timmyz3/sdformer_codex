#!/usr/bin/env python3
"""Generate H85 config. Do not launch while H82 owns the GPU."""

from __future__ import annotations

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H84_packed_class_file_ft15.yml"
OUTPUT = EXP / "configs/generated/dsec_fullres_w15_H85_row_delta_class_file_ft15.yml"


def main() -> int:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H85_row_delta_class_file_ft15"
    attention = config["bsa_attention"]
    attention["mode"] = "h85"
    attention["preserve_mean"] = False
    attention["binary_motion_xor_alpha"] = 0.0
    attention["class_stability_regularization_weight"] = 0.01
    config["runtime"]["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_h85_row_delta_class_file_ft15"
    )
    config["note"] = (
        "H85 per-row Class File with shared/insert/delete. "
        "Expand forbids T450 token_gate/codes/member_mask. "
        "Do not start while H82 owns the GPU."
    )
    OUTPUT.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
