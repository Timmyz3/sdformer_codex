#!/usr/bin/env python3
"""Generate the full-network ATLIF-only MVSEC ablation config."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/mvsec_cicc_h67_motion_w8_seed0.yml"
OUTPUT = EXP / "configs/generated/mvsec_cicc_atlif_only_w8_seed0_20260825.yml"


def main() -> int:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["experiment"] = "mvsec_cicc_atlif_only_w8_seed0_20260825"
    config["bsa_attention"] = {"enabled": False}
    config["experimental_neuron"] = {"enabled": False}
    config["runtime"]["initialization"] = "same_mvsec_nb0_seed0_checkpoint"
    config["runtime"]["ablation_role"] = "full_network_binary_atlif_original_attention"
    config["note"] = (
        "DATE two-contribution MVSEC ablation: replace all 105 neuron sites with "
        "one-sided binary ATLIF while retaining the original SDformer attention. "
        "Same NB0 ep11 parent, day2-only split, seed0, optimizer, and 30-epoch budget "
        "as the H67 full ATLIF+TTX route. No partial-stage or mixed attention replacement."
    )
    OUTPUT.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(OUTPUT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
