"""Generate the strict H67 full-resolution ep10-to-ep15 resume config."""

from __future__ import annotations

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep10.yml"
OUTPUT = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep15.yml"


def main() -> None:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H67_crop_bb1e4_strict_resume_ep15"
    config["loader"]["n_epochs"] = 15

    runtime = config.setdefault("runtime", {})
    runtime.update(
        {
            # Internal epochs 11 and 14 map to paper-facing checkpoints 12 and 15.
            "force_save_epochs": [11, 14],
            "state_save_epochs": [14],
            "save_only_force_epochs": True,
            "resume_protocol": "strict_local_model_optimizer_scheduler_scaler_ep10_to_ep15",
            "resume_source_epoch": 10,
        }
    )

    OUTPUT.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
