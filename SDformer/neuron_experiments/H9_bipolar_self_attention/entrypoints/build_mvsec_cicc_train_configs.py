#!/usr/bin/env python3
"""Generate the frozen NB0, H67, and Local5 direct-MVSEC training configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention"
CONFIG_ROOT = EXP_ROOT / "configs/generated"
UPSTREAM_BASE = REPO_ROOT / "third_party/SDformerFlow/configs/train_MDR_supervised_SDformerFlow.yml"
H67_SOURCE = CONFIG_ROOT / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
LOCAL5_SOURCE = CONFIG_ROOT / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml"
MANIFEST = "neuron_experiments/H9_bipolar_self_attention/manifests/mvsec_cicc_dt1_v1.json"


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def common_config() -> dict:
    config = read_yaml(UPSTREAM_BASE)
    config["data"].update(
        {
            "path": "data/Datasets/MVSEC",
            "preprocessed": True,
            "training_dataset": "mvsec_dt1",
            "test_sequence": "outdoor_day2",
            "event_interval": "dt1",
            "num_frames": 10,
            "num_chunks": 1,
            "mvsec_split_manifest": MANIFEST,
            "mvsec_train_split": "train",
            "mvsec_valid_split": "validation",
            "mvsec_source_valid_before_crop": True,
            "mvsec_direct_augmentation": {
                "enabled": True,
                "horizontal_flip_probability": 0.5,
                "vertical_flip_probability": 0.5,
            },
        }
    )
    config["model"]["num_bins"] = 10
    config["swin_transformer"]["window_size"] = [2, 8, 8]
    config["swin_transformer"]["pretrained_window_size"] = [2, 8, 8]
    config["spiking_neuron"]["num_steps"] = 10
    config["loader"].update(
        {
            "n_epochs": 50,
            "batch_size": 8,
            "validation_batch_size": 1,
            "resolution": [260, 346],
            "crop": [256, 256],
            "gpu": 0,
            "n_workers": 8,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "pin_memory": False,
            "drop_last_train": True,
            "drop_last_valid": False,
        }
    )
    config["optimizer"].update(
        {
            "lr": 1.0e-3,
            "wd": 0.01,
            "milestones": [10, 20, 30, 40],
            "use_amp": True,
            "num_acc": 1,
        }
    )
    config["metrics"].update(
        {
            "name": ["AEE"],
            "mask_events": True,
            "train_mask_events": False,
            "valid_mask_events": True,
        }
    )
    config["test"].update({"sample": 800, "n_valid": 1})
    config["vis"].update(
        {
            "enabled": False,
            "store": False,
            "store_grads": False,
            "store_spike_rates": False,
        }
    )
    config["runtime"] = {
        "snn_backend": "cupy",
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "checkpoint_metric": "valid_loss",
        "protocol": "cicc_spikeflownet_outdoor_day2_dt1_center256_seed0",
        "seed": 0,
        "reproducibility": "seeded_data_order_non_bit_exact",
    }
    return config


def candidate_config(source_path: Path, experiment: str) -> dict:
    config = common_config()
    source = read_yaml(source_path)
    for section in ("atlif_ternary_psn", "bsa_attention", "experimental_neuron"):
        if section in source:
            config[section] = copy.deepcopy(source[section])
    config["experiment"] = experiment
    config["loader"]["n_epochs"] = 30
    config["optimizer"] = copy.deepcopy(source["optimizer"])
    config["optimizer"]["milestones"] = [20, 25]
    config["optimizer"]["use_amp"] = True
    config["loss"].update(
        {key: value for key, value in source.get("loss", {}).items() if key not in {"training"}}
    )
    config["atlif_ternary_psn"]["threshold_freeze_after_step"] = 4430
    config["runtime"]["initialization"] = "same_mvsec_nb0_seed0_checkpoint"
    return config


def write_config(name: str, config: dict) -> None:
    path = CONFIG_ROOT / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path.resolve())


def main() -> int:
    nb0 = common_config()
    nb0["experiment"] = "mvsec_cicc_nb0_w8_seed0"
    nb0["atlif_ternary_psn"] = {"enabled": False}
    nb0["bsa_attention"] = {"enabled": False}
    nb0["experimental_neuron"] = {"enabled": False}
    write_config("mvsec_cicc_nb0_w8_seed0.yml", nb0)
    smoke = copy.deepcopy(nb0)
    smoke["experiment"] = "mvsec_cicc_nb0_w8_seed0_smoke"
    smoke["loader"]["n_epochs"] = 1
    smoke["runtime"]["protocol"] += "_single_batch_smoke"
    write_config("mvsec_cicc_nb0_w8_seed0_smoke.yml", smoke)
    write_config(
        "mvsec_cicc_h67_motion_w8_seed0.yml",
        candidate_config(H67_SOURCE, "mvsec_cicc_h67_motion_w8_seed0"),
    )
    write_config(
        "mvsec_cicc_local5_w8_seed0.yml",
        candidate_config(LOCAL5_SOURCE, "mvsec_cicc_local5_w8_seed0"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
