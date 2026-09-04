"""Generate additive DSEC window15 rescue configs for H67/H66d.

The first paper-window15 run inherited the crop-search optimizer, whose
backbone and norm learning rates were 50x/100x below the NB0 full-resolution
recipe.  These configs keep the model definitions and source checkpoints
fixed while testing stronger full-resolution adaptation.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"

SOURCE_CONFIGS = {
    "H67": GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
    "H66d": GEN / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml",
}
SOURCE_CHECKPOINTS = {
    "H67": EXP
    / (
        "results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_"
        "bs8_full30_20260711_setsid/checkpoint_epoch19.pth"
    ),
    "H66d": EXP
    / (
        "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_"
        "bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
    ),
}
NB0_FULLRES = EXP / (
    "results/dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728/"
    "checkpoint_epoch29.pth"
)

LR_PROFILES: dict[str, dict[str, float]] = {
    "bb2e5": {
        "lr": 2.0e-5,
        "backbone_lr": 2.0e-5,
        "norm_lr": 1.0e-5,
        "neuron_lr": 5.0e-5,
        "threshold_lr": 5.0e-6,
    },
    "bb1e4": {
        "lr": 1.0e-4,
        "backbone_lr": 1.0e-4,
        "norm_lr": 1.0e-4,
        "neuron_lr": 5.0e-5,
        "threshold_lr": 5.0e-6,
    },
}

SCREEN_SPECS = (
    {
        "id": "H67_crop_bb2e5",
        "model": "H67",
        "profile": "bb2e5",
        "init": "own_crop",
        "checkpoint": SOURCE_CHECKPOINTS["H67"],
    },
    {
        "id": "H67_crop_bb1e4",
        "model": "H67",
        "profile": "bb1e4",
        "init": "own_crop",
        "checkpoint": SOURCE_CHECKPOINTS["H67"],
    },
    {
        "id": "H67_nb0full_bb2e5",
        "model": "H67",
        "profile": "bb2e5",
        "init": "nb0_fullres_conversion",
        "checkpoint": NB0_FULLRES,
    },
)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_config(
    model_id: str,
    profile_name: str,
    *,
    name: str,
    batch_size: int,
    epochs: int,
    screen: bool,
    init: str = "own_crop",
) -> dict[str, Any]:
    config = deepcopy(load_yaml(SOURCE_CONFIGS[model_id]))
    profile = LR_PROFILES[profile_name]
    config["experiment"] = name
    config["data"]["path"] = "../../data/Datasets/DSEC/saved_flow_data"
    config["swin_transformer"]["window_size"] = [2, 15, 15]
    config["swin_transformer"]["pretrained_window_size"] = (
        [2, 15, 15] if init == "nb0_fullres_conversion" else [2, 9, 9]
    )

    optimizer = config["optimizer"]
    optimizer.update(
        {
            "lr": profile["lr"],
            "scheduler": "multistep",
            "milestones": [10, 20],
            "num_acc": 1,
        }
    )
    groups = optimizer.setdefault("param_groups", {})
    groups.update(
        {
            "enabled": True,
            "backbone_lr": profile["backbone_lr"],
            "norm_lr": profile["norm_lr"],
            "neuron_lr": profile["neuron_lr"],
            "threshold_lr": profile["threshold_lr"],
        }
    )
    # The original crop warmup hid most of a one-epoch LR diagnostic and was
    # not present in the successful NB0 full-resolution recipe.
    optimizer.setdefault("lr_warmup", {})["enabled"] = False

    config["loader"].update(
        {
            "n_epochs": epochs,
            "batch_size": batch_size,
            "resolution": [480, 640],
            "crop": None,
            "remap": "v1",
            "n_workers": 8,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "pin_memory": False,
            "non_blocking": True,
        }
    )
    config.setdefault("metrics", {})["name"] = ["AEE", "AAE", "AAE_Benchmark"]
    config.setdefault("test", {}).update(
        {
            "sample": 40,
            "n_valid": 1 if screen else 5,
            "scale_factor": 1,
            "bn_policy": "no_running",
            "eval_batch_size": 1,
        }
    )
    save_epochs = [0] if screen else [0, 4, 9, 14, 19, 24, 29]
    state_epochs = [] if screen else [9, 19, 29]
    config.setdefault("runtime", {}).update(
        {
            "seed": 0,
            "max_train_steps": 0,
            "skip_save": False,
            "skip_state_save": screen,
            "save_only_force_epochs": True,
            "force_save_epochs": save_epochs,
            "state_save_epochs": state_epochs,
            "use_mlflow_model_logging": False,
            "full_resolution_protocol": "paper_480x640_window2x15x15_lr_rescue",
            "physical_batch": batch_size,
            "gradient_accumulation": 1,
            "rescue_profile": profile_name,
            "rescue_init": init,
        }
    )
    config["note"] = (
        f"Additive DSEC full-resolution rescue for {model_id}; init={init}, "
        f"profile={profile_name}, 480x640, window [2,15,15], remap=v1, "
        "BN=no_running, no gradient accumulation. Model structure is unchanged."
    )
    return config


def specs_for(mode: str, profile: str) -> list[dict[str, Any]]:
    if mode == "screen":
        return [dict(item) for item in SCREEN_SPECS]
    return [
        {
            "id": f"{model_id}_{profile}",
            "model": model_id,
            "profile": profile,
            "init": "own_crop",
            "checkpoint": SOURCE_CHECKPOINTS[model_id],
        }
        for model_id in ("H67", "H66d")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "formal"), default="screen")
    parser.add_argument("--profile", choices=tuple(LR_PROFILES), default="bb2e5")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    screen = args.mode == "screen"
    epochs = 1 if screen else 30
    rows = []
    for order, spec in enumerate(specs_for(args.mode, args.profile), start=1):
        name = f"dsec_fullres_w15_rescue_{spec['id']}_{'screen1' if screen else 'ft30'}"
        config = build_config(
            spec["model"],
            spec["profile"],
            name=name,
            batch_size=args.batch_size,
            epochs=epochs,
            screen=screen,
            init=spec["init"],
        )
        output = GEN / f"{name}.yml"
        output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "order": order,
                **spec,
                "name": name,
                "config": str(output),
                "checkpoint": str(spec["checkpoint"]),
                "epochs": epochs,
                "batch_size": args.batch_size,
            }
        )
        print(output)

    manifest = GEN / f"dsec_fullres_w15_rescue_{args.mode}_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
