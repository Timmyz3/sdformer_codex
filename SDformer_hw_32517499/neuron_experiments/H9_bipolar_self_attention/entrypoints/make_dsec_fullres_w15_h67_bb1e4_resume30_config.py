"""Generate the H67 full-resolution ep15-to-ep30 continuation config."""

from __future__ import annotations

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep15.yml"
OUTPUT = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml"


def main() -> None:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H67_crop_bb1e4_resume_ep30"
    config["loader"]["n_epochs"] = 30

    runtime = config.setdefault("runtime", {})
    runtime.update(
        {
            # Internal epochs map to paper-facing epochs through epoch_offset=1.
            "force_save_epochs": [19, 24, 29],
            "state_save_epochs": [29],
            "save_only_force_epochs": True,
            "resume_protocol": "audited_model_optimizer_scaler_with_scheduler_counter_repair_ep15_to_ep30",
            "resume_source_epoch": 15,
        }
    )

    OUTPUT.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
