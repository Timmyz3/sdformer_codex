"""Generate hardware-consistent 480x640/window9 DSEC fine-tuning configs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
MANIFEST = GEN / "dsec_fullres_window9_manifest.json"
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]
STATE_EPOCHS = [4, 9, 14, 19, 24, 29]

CANDIDATES = [
    {
        "id": "NB0",
        "name": "dsec_fullres_w9_nb0_ep59_ft30",
        "config": REPO / "configs/generated/upstream_baseline_stride.yml",
        "checkpoint": REPO / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth",
        "expected_atlif": 0,
        "expected_attention": 0,
        "expected_overlay": 0,
    },
    {
        "id": "H67",
        "name": "dsec_fullres_w9_h67_motion_ep19_ft30",
        "config": GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
        "checkpoint": EXP / (
            "results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_"
            "bs8_full30_20260711_setsid/checkpoint_epoch19.pth"
        ),
        "expected_atlif": 105,
        "expected_attention": 12,
        "expected_overlay": 210,
    },
    {
        "id": "H66d",
        "name": "dsec_fullres_w9_h66d_local5_ep29_ft30",
        "config": GEN / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml",
        "checkpoint": EXP / (
            "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_"
            "bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
        ),
        "expected_atlif": 105,
        "expected_attention": 12,
        "expected_overlay": 210,
    },
]


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.effective_batch < args.batch_size:
        parser.error("require 1 <= batch-size <= effective-batch")
    if args.effective_batch % args.batch_size:
        parser.error("effective-batch must be divisible by batch-size")

    accumulation = args.effective_batch // args.batch_size
    rows = []
    for order, candidate in enumerate(CANDIDATES, start=1):
        config = deepcopy(load(candidate["config"]))
        suffix = "_smoke" if args.smoke else ""
        name = candidate["name"] + suffix
        config["experiment"] = name
        config["data"]["path"] = "../../data/Datasets/DSEC/saved_flow_data"
        config["loader"].update(
            {
                "n_epochs": 1 if args.smoke else args.epochs,
                "batch_size": args.batch_size,
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
        config["optimizer"]["num_acc"] = accumulation
        if "lr_warmup" in config["optimizer"]:
            base_steps = int(config["optimizer"]["lr_warmup"].get("steps", 0) or 0)
            if base_steps:
                config["optimizer"]["lr_warmup"]["steps"] = (
                    base_steps * args.effective_batch // args.batch_size
                )
        if config.get("atlif_ternary_psn", {}).get("threshold_freeze_after_step") is not None:
            base_freeze = int(config["atlif_ternary_psn"]["threshold_freeze_after_step"])
            config["atlif_ternary_psn"]["threshold_freeze_after_step"] = (
                base_freeze * args.effective_batch // args.batch_size
            )

        config.setdefault("metrics", {})["name"] = ["AEE", "AAE", "AAE_Benchmark"]
        config.setdefault("test", {}).update(
            {
                "sample": 1 if args.smoke else 40,
                "n_valid": 1 if args.smoke else 5,
                "scale_factor": 1,
            }
        )
        config.setdefault("runtime", {}).update(
            {
                "seed": 0,
                "max_train_steps": 2 if args.smoke else 0,
                "skip_save": bool(args.smoke),
                "skip_state_save": bool(args.smoke),
                "save_only_force_epochs": True,
                "force_save_epochs": [] if args.smoke else list(SAVE_EPOCHS),
                "state_save_epochs": [] if args.smoke else list(STATE_EPOCHS),
                "use_mlflow_model_logging": False,
                "full_resolution_protocol": "480x640_window9_hardware_consistent",
                "effective_batch": args.effective_batch,
            }
        )
        config["note"] = (
            f"DSEC full-resolution fine-tune from {candidate['id']} frozen crop checkpoint. "
            "Uses 480x640 with window [2,9,9], remap=v1 audited interpolation, "
            f"physical batch {args.batch_size}, accumulation {accumulation}, "
            f"effective batch {args.effective_batch}. This is hardware-consistent and "
            "must not be labeled as the paper window15 protocol."
        )
        output = GEN / f"{name}.yml"
        output.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        rows.append(
            {
                "order": order,
                "id": candidate["id"],
                "name": name,
                "config": str(output),
                "checkpoint": str(candidate["checkpoint"]),
                "expected_atlif": candidate["expected_atlif"],
                "expected_attention": candidate["expected_attention"],
                "expected_overlay": candidate["expected_overlay"],
                "batch_size": args.batch_size,
                "accumulation": accumulation,
                "effective_batch": args.effective_batch,
                "epochs": 1 if args.smoke else args.epochs,
                "smoke": bool(args.smoke),
            }
        )
        print(output)

    manifest = (
        GEN / "dsec_fullres_window9_smoke_manifest.json"
        if args.smoke
        else MANIFEST
    )
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
