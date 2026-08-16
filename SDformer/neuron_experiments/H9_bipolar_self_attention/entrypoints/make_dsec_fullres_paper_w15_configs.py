"""Generate paper-protocol DSEC 480x640/window15 fine-tuning configs.

This generator is additive. It does not modify the historical window9 configs.
The public SDformerFlow paper specifies 480x640, a 2x15x15 window, 30 extra
fine-tuning epochs, batch size 1 or 2, bicubic relative-position remapping, and
evaluation without BatchNorm running statistics.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
MANIFEST = GEN / "dsec_fullres_paper_w15_manifest.json"
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 29]
STATE_EPOCHS = [9, 19, 29]
REFERENCE_CROP_BATCH = 8

CANDIDATES = [
    {
        "id": "NB0",
        "name": "dsec_fullres_paper_w15_nb0_ep59_ft30",
        "config": REPO / "configs/generated/upstream_baseline_stride.yml",
        "checkpoint": REPO / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth",
        "crop_epochs": 60,
        "expected_atlif": 0,
        "expected_attention": 0,
        "expected_overlay": 0,
    },
    {
        "id": "H67",
        "name": "dsec_fullres_paper_w15_h67_motion_ep19_ft30",
        "config": GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
        "checkpoint": EXP
        / (
            "results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_"
            "bs8_full30_20260711_setsid/checkpoint_epoch19.pth"
        ),
        "crop_epochs": 20,
        "expected_atlif": 105,
        "expected_attention": 12,
        "expected_overlay": 210,
    },
    {
        "id": "H66d",
        "name": "dsec_fullres_paper_w15_h66d_local5_ep29_ft30",
        "config": GEN / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml",
        "checkpoint": EXP
        / (
            "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_"
            "bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
        ),
        "crop_epochs": 30,
        "expected_atlif": 105,
        "expected_attention": 12,
        "expected_overlay": 210,
    },
]


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def scale_step_schedule(config: dict, batch_size: int) -> None:
    """Preserve the crop schedule's sample count when physical batch changes."""
    factor = REFERENCE_CROP_BATCH // batch_size
    warmup = config.get("optimizer", {}).get("lr_warmup")
    if warmup and int(warmup.get("steps", 0) or 0):
        warmup["steps"] = int(warmup["steps"]) * factor
    atlif = config.get("atlif_ternary_psn", {})
    freeze = atlif.get("threshold_freeze_after_step")
    if freeze is not None:
        atlif["threshold_freeze_after_step"] = int(freeze) * factor
        # Historical H9 configs freeze only the auxiliary homeostatic update.
        # True optimizer-gradient freezing remains an explicit future ablation.
        atlif.setdefault("freeze_threshold_grad_after_step", False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = []
    for order, candidate in enumerate(CANDIDATES, start=1):
        config = deepcopy(load(candidate["config"]))
        suffix = "_smoke" if args.smoke else ""
        name = candidate["name"] + suffix
        config["experiment"] = name
        config["data"]["path"] = "../../data/Datasets/DSEC/saved_flow_data"
        config["swin_transformer"]["window_size"] = [2, 15, 15]
        config["swin_transformer"]["pretrained_window_size"] = [2, 9, 9]
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
        # The paper's full-resolution batch is a physical batch of 1 or 2.
        config["optimizer"]["num_acc"] = 1
        scale_step_schedule(config, args.batch_size)
        config.setdefault("metrics", {})["name"] = ["AEE", "AAE", "AAE_Benchmark"]
        config.setdefault("test", {}).update(
            {
                "sample": 1 if args.smoke else 40,
                "n_valid": 1 if args.smoke else 5,
                "scale_factor": 1,
                "bn_policy": "no_running",
                "eval_batch_size": 1,
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
                "full_resolution_protocol": "paper_480x640_window2x15x15_ft30",
                "physical_batch": args.batch_size,
                "gradient_accumulation": 1,
                "source_crop_epochs": candidate["crop_epochs"],
            }
        )
        config["note"] = (
            f"Paper-geometry DSEC full-resolution fine-tune from {candidate['id']}. "
            "Uses 480x640, window [2,15,15], crop=null, 30 additional epochs, "
            "physical batch 1 or 2 without gradient accumulation, audited remap=v1 "
            "bicubic relative-position interpolation, and no-running-stat BN evaluation "
            "with the released validation batch size 1. "
            f"This local source checkpoint contains {candidate['crop_epochs']} crop epochs; "
            "the paper's crop training budget is 80 epochs, so that source-budget "
            "difference must remain disclosed."
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
                "source_crop_epochs": candidate["crop_epochs"],
                "expected_atlif": candidate["expected_atlif"],
                "expected_attention": candidate["expected_attention"],
                "expected_overlay": candidate["expected_overlay"],
                "batch_size": args.batch_size,
                "accumulation": 1,
                "epochs": 1 if args.smoke else args.epochs,
                "smoke": bool(args.smoke),
                "protocol": {
                    "resolution": [480, 640],
                    "crop": None,
                    "window_size": [2, 15, 15],
                    "pretrained_window_size": [2, 9, 9],
                    "remap": "v1",
                    "bn_policy": "no_running",
                    "eval_batch_size": 1,
                },
            }
        )
        print(output)

    manifest = (
        GEN / "dsec_fullres_paper_w15_smoke_manifest.json"
        if args.smoke
        else MANIFEST
    )
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
