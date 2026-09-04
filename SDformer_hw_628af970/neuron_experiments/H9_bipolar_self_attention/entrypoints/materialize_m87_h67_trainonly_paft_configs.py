#!/usr/bin/env python3
"""Materialize enabled H67 PAFT configs from an admitted M77 catalog."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "neuron_experiments/H9_bipolar_self_attention"
SOURCE = EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
SOURCE_SHA256 = (
    "86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc")
CHECKPOINT_SHA256 = (
    "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158")
OPERATORS = [
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(path.resolve().relative_to(ROOT.resolve()))


def dump_yaml(value):
    try:
        return yaml.safe_dump(value, sort_keys=False)
    except TypeError:
        return yaml.safe_dump(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--admission-contract", required=True, type=Path)
    parser.add_argument("--train-trace-manifest", required=True, type=Path)
    parser.add_argument("--full-output", required=True, type=Path)
    parser.add_argument("--smoke-output", required=True, type=Path)
    parser.add_argument("--control-output", required=True, type=Path)
    args = parser.parse_args()
    for output in (args.full_output, args.smoke_output, args.control_output):
        require(not output.exists(), "refusing M87 config overwrite: " + str(output))
    require(sha256(SOURCE) == SOURCE_SHA256,
            "M87 forward source config SHA drift")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    contract = json.loads(args.admission_contract.read_text(encoding="utf-8"))
    trace = json.loads(args.train_trace_manifest.read_text(encoding="utf-8"))
    catalog_sha = sha256(args.catalog)
    contract_sha = sha256(args.admission_contract)
    trace_sha = sha256(args.train_trace_manifest)
    require(catalog["schema"] ==
            "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1",
            "M87 catalog schema mismatch")
    require(catalog["split"]["role"] == "DSEC_TRAIN_ONLY_PAFT_CALIBRATION"
            and catalog["split"]["train_catalog_eligible"] is True
            and catalog["split"]["test_or_validation_data_used"] is False,
            "M87 catalog split not admitted")
    require(contract["schema"] ==
            "m77_pattern_paft_catalog_admission_contract_v1"
            and contract["unit_test_only"] is False
            and contract["train_only_admitted"] is True
            and contract["catalog_sha256"] == catalog_sha
            and contract["train_trace_manifest_sha256"] == trace_sha,
            "M87 admission contract mismatch")
    require(contract.get("forward_base_config_sha256") == SOURCE_SHA256
            and catalog.get("identity", {}).get(
                "forward_base_config_sha256") == SOURCE_SHA256,
            "M87 catalog/contract forward config mismatch")
    require(trace["schema"] ==
            "m73_h67_ep35_train_calibration_packed_source_trace_v1"
            and trace["split_audit"]["full_train_valid825_key_overlap"] == 0
            and trace["split_audit"]["selected_valid825_key_overlap"] == 0,
            "M87 train trace split mismatch")
    require(trace.get("identity", {}).get("config_sha256") == SOURCE_SHA256
            and trace.get("identity", {}).get(
                "paft_forward_base_config_sha256") == SOURCE_SHA256,
            "M87 trace was not captured from the training forward config")

    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["experiment"] = "dsec_fullres_w15_M87_h67_trainonly_paft_k16q16_ft5_w001"
    config["loader"]["n_epochs"] = 5
    config["optimizer"]["lr"] = 2.5e-5
    config["optimizer"]["milestones"] = []
    config["optimizer"].setdefault("param_groups", {})["backbone_lr"] = 2.5e-5
    config["pattern_paft"] = {
        "enabled": True,
        "catalog": relative(args.catalog),
        "catalog_sha256": catalog_sha,
        "catalog_admission_contract": relative(args.admission_contract),
        "catalog_admission_contract_sha256": contract_sha,
        "runtime_train_sequence_list": (
            "data/Datasets/DSEC/saved_flow_data/sequence_lists/train_split_seq.csv"),
        "runtime_valid825_sequence_list": (
            "data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv"),
        "runtime_train_trace_manifest": relative(args.train_trace_manifest),
        "runtime_train_trace_manifest_sha256": trace_sha,
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "expected_forward_base_config_sha256": SOURCE_SHA256,
        "expected_operator_names": OPERATORS,
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
        "resume_protocol": "model_only_h67_ep35_five_epoch_trainonly_paft",
        "paft_catalog_split": "M73_DSEC_TRAIN_ONLY_S32_ALL18_SEQUENCES",
        "paft_heldout_split": "VALID825_NEVER_USED_FOR_CATALOG",
        "paired_arm": "PAFT_K16Q16",
    })
    config["note"] = (
        "M87 H67 five-epoch hardware-weighted PAFT using an M77 catalog "
        "derived only from the disjoint M73 DSEC training trace. Accuracy, "
        "cycle speedup and paper claims remain unadmitted until post-run validation."
    )
    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    args.full_output.write_text(dump_yaml(config), encoding="utf-8")

    # Paired continuation control: identical source checkpoint, optimizer,
    # seed, loader order and five-epoch budget, with only PAFT disabled.
    control = yaml.safe_load(dump_yaml(config))
    control["experiment"] = (
        "dsec_fullres_w15_M87_h67_trainonly_no_paft_control_ft5")
    control["pattern_paft"] = {
        "enabled": False,
        "paired_catalog_sha256": catalog_sha,
        "expected_forward_base_config_sha256": SOURCE_SHA256,
    }
    control["runtime"]["paired_arm"] = "NO_PAFT_CONTROL"
    control["note"] = (
        "M87 paired five-epoch no-PAFT continuation control. It shares the "
        "checkpoint, seed, loader, optimizer and budget with the PAFT arm."
    )
    args.control_output.parent.mkdir(parents=True, exist_ok=True)
    args.control_output.write_text(dump_yaml(control), encoding="utf-8")

    smoke = yaml.safe_load(dump_yaml(config))
    smoke["experiment"] = "dsec_fullres_w15_M87_h67_trainonly_paft_k16q16_smoke1_w001"
    smoke["loader"]["n_epochs"] = 1
    smoke["runtime"].update({
        "max_train_steps": 1,
        "force_save_epochs": [],
        "state_save_epochs": [],
        "save_only_force_epochs": True,
        "skip_save": True,
        "skip_state_save": True,
    })
    smoke["note"] = "M87 one-step production-contract smoke before H67 PAFT."
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_output.write_text(dump_yaml(smoke), encoding="utf-8")
    require(sha256(SOURCE) == SOURCE_SHA256,
            "M87 forward source config changed during materialization")
    print("PASS M87 full={} control={} smoke={} catalog={} contract={} trace={}".format(
        args.full_output, args.control_output, args.smoke_output,
        catalog_sha, contract_sha, trace_sha),
        flush=True)


if __name__ == "__main__":
    main()
