#!/usr/bin/env python3
"""Generate the full-resolution no-motion control matched to H67."""

from __future__ import annotations

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml"
OUTPUT = EXP / "configs/generated/dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
H81_SOURCE = (
    EXP
    / "results/h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30_bs8_full30_20260717_setsid/checkpoint_epoch19.pth"
)


def main() -> int:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H81_nomotion_bb1e4_ft40"
    config["bsa_attention"]["binary_motion_xor_alpha"] = 0.0
    config["optimizer"]["milestones"] = [10, 20]
    config["loader"]["n_epochs"] = 40

    runtime = config["runtime"]
    runtime["force_save_epochs"] = [29, 34, 39]
    runtime["state_save_epochs"] = [29, 39]
    runtime["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_h81_nomotion_equal40"
    )
    runtime["rescue_profile"] = "bb1e4"
    runtime["rescue_init"] = "own_crop_h81_ep19"
    runtime["source_crop_checkpoint"] = str(H81_SOURCE.resolve())
    runtime["source_crop_budget"] = 20
    runtime["fullres_budget"] = 40
    for key in (
        "epoch_offset",
        "rescue_continuation",
        "rescue_source_checkpoint",
        "resume_protocol",
        "resume_source_epoch",
    ):
        runtime.pop(key, None)

    config["note"] = (
        "DSEC full-resolution reviewer control matched to the H67 bb1e4 recipe. "
        "It starts from the equal-budget H81 crop ep19 checkpoint and differs from "
        "H67 only by binary_motion_xor_alpha=0.0. Geometry is 480x640 with "
        "window [2,15,15], crop=null, batch2, no-running-stat BN evaluation, and "
        "40 full-resolution epochs with checkpoints at equal budgets 30/35/40."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
