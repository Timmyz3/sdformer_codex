#!/usr/bin/env python3
"""Generate H81 same-protocol MVSEC config and Local5 DSEC-to-day2 FT config."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
H81_SOURCE = GEN / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
LOCAL5_MVSEC = GEN / "mvsec_cicc_local5_w8_seed0.yml"


def dump(name: str, config: dict) -> Path:
    path = GEN / name
    rendered = yaml.safe_dump(config, sort_keys=False, width=100)
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"generated config drift: {path}")
    path.write_text(rendered, encoding="utf-8")
    print(path)
    return path


def main() -> int:
    h81_src = yaml.safe_load(H81_SOURCE.read_text(encoding="utf-8"))
    local5 = yaml.safe_load(LOCAL5_MVSEC.read_text(encoding="utf-8"))

    h81 = copy.deepcopy(local5)
    h81["experiment"] = "mvsec_cicc_h81_nomotion_w8_seed0"
    h81["atlif_ternary_psn"] = copy.deepcopy(h81_src["atlif_ternary_psn"])
    h81["atlif_ternary_psn"]["threshold_freeze_after_step"] = 4430
    h81["bsa_attention"] = copy.deepcopy(h81_src["bsa_attention"])
    h81["experimental_neuron"] = copy.deepcopy(h81_src.get("experimental_neuron") or {"enabled": False})
    h81["loader"]["n_epochs"] = 30
    h81["optimizer"]["milestones"] = [20, 25]
    h81["runtime"]["initialization"] = "same_mvsec_nb0_seed0_checkpoint"
    h81["runtime"]["protocol_family"] = "day2_scratch_same_as_h67_local5"
    h81["note"] = (
        "Same-protocol H81 MVSEC control: day2-only, same NB0 init as H67/Local5, "
        "Motion-XOR alpha=0. Completes the missing reviewer ablation on MVSEC."
    )
    dump("mvsec_cicc_h81_nomotion_w8_seed0.yml", h81)

    transfer = copy.deepcopy(local5)
    transfer["experiment"] = "mvsec_cicc_local5_dsec_ep44_ft15_w8_seed0"
    transfer["loader"]["n_epochs"] = 15
    transfer["swin_transformer"]["pretrained_window_size"] = [2, 15, 15]
    for key in ("lr", "backbone_lr", "neuron_lr", "norm_lr"):
        if key == "lr":
            transfer["optimizer"]["lr"] = 2.5e-5
        elif "param_groups" in transfer["optimizer"]:
            if key in transfer["optimizer"]["param_groups"]:
                transfer["optimizer"]["param_groups"][key] = 2.5e-5
    transfer["optimizer"]["param_groups"]["threshold_lr"] = 1.0e-6
    transfer["optimizer"]["milestones"] = []
    transfer["atlif_ternary_psn"]["threshold_freeze_after_step"] = 0
    transfer["runtime"]["initialization"] = "dsec_local5_ep44_fullres"
    transfer["runtime"]["protocol_family"] = "dsec_pretrain_day2_ft"
    transfer["runtime"]["protocol"] = (
        "cicc_day2_dt1_center256_seed0_from_dsec_local5_ep44"
    )
    transfer["note"] = (
        "Local5 MVSEC rescue: start from DSEC Local5 ep44 and fine-tune 15 day2 "
        "epochs at 2.5e-5. Separately labeled from the scratch day2 table. "
        "Window [2,8,8] with pretrained_window [2,15,15] for relative-bias interpolate."
    )
    dump("mvsec_cicc_local5_dsec_ep44_ft15_w8_seed0.yml", transfer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
