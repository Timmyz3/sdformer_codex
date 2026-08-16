"""Generate the fair full-resolution Local-5 bb1e4 training config."""

from __future__ import annotations

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_paper_w15_h66d_local5_ep29_ft30.yml"
OUTPUT = EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"


def build_config() -> dict:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H66d_local5_bb1e4_ft30"
    config["optimizer"].update(
        {
            "lr": 1.0e-4,
            "wd": 1.0e-3,
            "scheduler": "multistep",
            # H67's segmented rescue executed 1e-4 through epoch 12, then
            # 5e-5 through epoch 19. Match the observed LR trace, not only
            # the nominal milestones stored in its source YAML.
            "milestones": [13, 20],
            "use_amp": True,
            "num_acc": 1,
        }
    )
    config["optimizer"]["param_groups"] = {
        "enabled": True,
        "backbone_lr": 1.0e-4,
        "neuron_lr": 5.0e-5,
        "norm_lr": 1.0e-4,
        "norm_wd": 0.0,
        "threshold_lr": 5.0e-6,
        "threshold_wd": 0.0,
    }
    config["optimizer"]["lr_warmup"] = {
        "enabled": False,
        "steps": 720,
        "start_factor": 0.05,
    }
    config["atlif_ternary_psn"]["threshold_freeze_after_step"] = 1224
    config["loader"]["n_epochs"] = 30
    config["test"]["n_valid"] = 1

    runtime = config.setdefault("runtime", {})
    for stale in (
        "epoch_offset",
        "resume_protocol",
        "resume_source_epoch",
        "rescue_continuation",
        "rescue_source_checkpoint",
    ):
        runtime.pop(stale, None)
    runtime.update(
        {
            "force_save_epochs": [9, 14, 19, 24, 29],
            "state_save_epochs": [9, 19, 29],
            "save_only_force_epochs": True,
            "full_resolution_protocol": "paper_480x640_window2x15x15_local5_bb1e4_ft30",
            "physical_batch": 2,
            "gradient_accumulation": 1,
            "rescue_profile": "bb1e4",
            "rescue_init": "own_crop_rank1_epoch29",
        }
    )
    config["note"] = (
        "Fair Local-5 full-resolution rerun: same 480x640/window2x15x15, bb1e4 "
        "optimizer strength, effective LR trace, threshold freeze and valid825 protocol "
        "as the rescued H67 line. Starts from the completed Local-5 crop/full30 rank-1 ep29."
    )
    return config


def main() -> None:
    config = build_config()

    OUTPUT.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
