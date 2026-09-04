#!/usr/bin/env python3
"""Generate the H90/M71 five-epoch PAFT candidate and one-step smoke config."""

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
FULL = EXP / "configs/generated/dsec_fullres_w15_H90_h67_paft_k16q16_ft5_w001.yml"
SMOKE = EXP / "configs/generated/dsec_fullres_w15_H90_h67_paft_k16q16_smoke1_w001.yml"
CATALOG = (
    "hw_autoresearch_nts07/results/"
    "m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json"
)


def candidate():
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_H90_h67_paft_k16q16_ft5_w001"
    config["loader"]["n_epochs"] = 5
    optimizer = config["optimizer"]
    optimizer["lr"] = 2.5e-5
    optimizer["milestones"] = []
    groups = optimizer.setdefault("param_groups", {})
    groups["backbone_lr"] = 2.5e-5
    config["pattern_paft"] = {
        # Fail closed: the original M71 catalog was later proven to come from
        # valid825, not the training split.  A successor generator must replace
        # the catalog and dataset-role receipt before enabling PAFT.
        "enabled": False,
        "blocked_reason": "M71_VALID825_CATALOG_REVOKED_USE_TRAIN_ONLY_SUCCESSOR",
        "catalog": CATALOG,
        "partition_bits": 16,
        "patterns_per_partition": 16,
        "sample_vectors_per_module": 64,
        "partition_chunk": 36,
        "regularization_weight": 1.0e-3,
        "hardware_fanout_output_blocks": 8,
        "runtime_cost": "min(popcount,one_pwp_plus_nearest_signed_hamming)",
    }
    runtime = config.setdefault("runtime", {})
    runtime.update({
        "force_save_epochs": [4],
        "state_save_epochs": [4],
        "save_only_force_epochs": True,
        "skip_save": False,
        "skip_state_save": False,
        "max_train_steps": 0,
        "epoch_offset": 0,
        "resume_protocol": "model_only_h67_ep35_five_epoch_hardware_weighted_paft",
        "paft_catalog_split": "REVOKED_M71_VALID825_INTERNAL_SAMPLES_0_TO_4",
        "paft_heldout_split": "REVOKED_NOT_AN_INDEPENDENT_HELDOUT",
    })
    config["note"] = (
        "H90/M71 hardware-weighted PAFT: canonical H67 ep35 model-only anchor, "
        "five epochs at backbone lr 2.5e-5. This generated config is disabled: "
        "the M71 catalog was found to use valid825 samples and was revoked. "
        "Replace it with a disjoint training-only catalog before any run."
    )
    return config


def dump_yaml(value):
    try:
        return yaml.safe_dump(value, sort_keys=False)
    except TypeError:
        # The Synopsys host carries an older PyYAML without sort_keys.
        return yaml.safe_dump(value)


def main():
    full = candidate()
    FULL.write_text(dump_yaml(full), encoding="utf-8")
    smoke = yaml.safe_load(dump_yaml(full))
    smoke["experiment"] = "dsec_fullres_w15_H90_h67_paft_k16q16_smoke1_w001"
    smoke["loader"]["n_epochs"] = 1
    smoke["runtime"].update({
        "max_train_steps": 1,
        "force_save_epochs": [],
        "state_save_epochs": [],
        "save_only_force_epochs": True,
        "skip_save": True,
        "skip_state_save": True,
    })
    smoke["note"] = "One-step fail-closed smoke for the H90/M71 PAFT candidate."
    SMOKE.write_text(dump_yaml(smoke), encoding="utf-8")
    print(FULL)
    print(SMOKE)


if __name__ == "__main__":
    main()
