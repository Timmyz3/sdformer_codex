#!/usr/bin/env python3
"""Capture a disjoint DSEC-train calibration cohort for Phi-style PAFT.

The capture deliberately reuses M40's exact four-Conv hook/writer, but replaces
its valid825 cohort with 32 deterministic train samples spanning every training
sequence.  The manifest freezes both train and valid sequence-list identities
and proves zero key overlap before the GPU model is constructed.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
M40_PATH = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "trace_m40_bottleneck_packed_sources.py")
EXPECTED_TRAIN_LIST_SHA256 = (
    "919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10")
EXPECTED_VALID_LIST_SHA256 = (
    "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0")
EXPECTED_FORWARD_CONFIG_SHA256 = (
    "86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc")
SAMPLES = 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_m40():
    parent = str(M40_PATH.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("m73_m40", str(M40_PATH))
    require(spec is not None and spec.loader is not None, "cannot import M40")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_one_column_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    require(all(len(row) == 1 for row in rows),
            "M73 sequence list must have exactly one column")
    return [row[0] for row in rows]


def sequence_name(key):
    return "_".join(Path(key).stem.split("_")[:-1])


def select_train_indices(train_keys):
    """Cover all sequences once, then give the 14 largest a second sample."""
    grouped = defaultdict(list)
    for index, key in enumerate(train_keys):
        grouped[sequence_name(key)].append(index)
    require(len(grouped) == 18, "M73 expected exactly 18 train sequences")
    doubled = set(sorted(grouped, key=lambda name: (-len(grouped[name]), name))[:14])
    selected = []
    allocation = {}
    for name in sorted(grouped):
        indices = grouped[name]
        if name in doubled:
            chosen = [indices[len(indices) // 3], indices[(2 * len(indices)) // 3]]
        else:
            chosen = [indices[len(indices) // 2]]
        require(len(chosen) == len(set(chosen)), "M73 duplicate sequence quantile")
        allocation[name] = chosen
        selected.extend(chosen)
    selected.sort()
    require(len(selected) == SAMPLES and len(set(selected)) == SAMPLES,
            "M73 selected cohort extent mismatch")
    return selected, allocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    m40 = load_m40()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing M73 output overwrite")
    # M73 is a training catalog trace, so it must use the exact unmodified
    # forward configuration from which M87 is materialized.  The earlier
    # hardware-order deployment YAML has a different attention/quantization
    # path and is intentionally rejected here.
    require(sha256(args.config.resolve()) == EXPECTED_FORWARD_CONFIG_SHA256,
            "M73/M87 forward config identity drift")
    require(sha256(args.checkpoint.resolve()) == m40.EXPECTED_CHECKPOINT_SHA256,
            "M73 checkpoint identity drift")
    data_root = args.data_root.resolve()
    train_list = data_root / "sequence_lists/train_split_seq.csv"
    valid_list = data_root / "sequence_lists/valid_split_seq.csv"
    require(sha256(train_list) == EXPECTED_TRAIN_LIST_SHA256,
            "M73 train list identity drift")
    require(sha256(valid_list) == EXPECTED_VALID_LIST_SHA256,
            "M73 valid825 list identity drift")
    train_keys = read_one_column_csv(train_list)
    valid_keys = read_one_column_csv(valid_list)
    require(len(train_keys) == len(set(train_keys)) == 7345,
            "M73 train key population/uniqueness drift")
    require(len(valid_keys) == len(set(valid_keys)) == 825,
            "M73 valid825 key population/uniqueness drift")
    require(not (set(train_keys) & set(valid_keys)),
            "M73 train/valid825 key overlap")
    selected_indices, allocation = select_train_indices(train_keys)
    sample_keys = [train_keys[index] for index in selected_indices]
    require(not (set(sample_keys) & set(valid_keys)),
            "M73 selected cohort leaks valid825")

    if args.preflight_only:
        print(json.dumps({
            "schema": "m73_train_calibration_preflight_v1",
            "status": "PASS_M73_PREFLIGHT_NO_GPU_MODEL_CONSTRUCTED",
            "train_sequence_list_sha256": sha256(train_list),
            "valid825_sequence_list_sha256": sha256(valid_list),
            "train_population": len(train_keys),
            "valid825_population": len(valid_keys),
            "full_train_valid825_key_overlap": 0,
            "selected_samples": len(sample_keys),
            "selected_sequences": len(set(sequence_name(key) for key in sample_keys)),
            "selected_valid825_key_overlap": 0,
            "selected_train_indices": selected_indices,
            "selected_sample_keys": sample_keys,
        }, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True)
    profile = m40.load_profile_module()
    config, device = profile.load_config(args.config.resolve())
    config["data"]["path"] = str(data_root)
    require(device.type == "cuda" and torch.cuda.is_available(),
            "M73 exact train trace requires CUDA")
    dataset = profile.DSECDatasetLite(
        config, file_list="train", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    observed = [dataset.files[index][0] for index in selected_indices]
    require(observed == sample_keys, "M73 DataLoader/train-list identity drift")
    subset = torch.utils.data.Subset(dataset, selected_indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=0)
    transform = None
    if config["loader"].get("crop") is not None:
        transform = profile.Compose([
            profile.CenterCrop(tuple(config["loader"]["crop"]))])
    model = profile.build_model(config, args.checkpoint.resolve(), device)
    load_audit = profile.validate_h9_load_audit(model, config)
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    bn_changed = profile.configure_batch_norm_evaluation(model, bn_policy)
    writer = m40.PackedBottleneckWriter(output_dir, sample_keys)
    writer.attach(model)
    processed = 0
    try:
        with torch.no_grad():
            for chunk, mask, label in itertools.islice(loader, SAMPLES):
                profile.functional.reset_net(model)
                writer.begin_sample(processed, sample_keys[processed])
                x, transformed_label, transformed_mask = profile.preprocess_chunk(
                    config, chunk, label, mask, transform, device)
                del transformed_label, transformed_mask
                model(x)
                processed += 1
                print("[M73 train capture] {}/{} {}".format(
                    processed, SAMPLES, sample_keys[processed - 1]), flush=True)
    except BaseException:
        writer.detach()
        raise
    writer.finalize()
    require(processed == SAMPLES, "M73 cohort did not complete")

    rows = sorted(writer.rows, key=lambda row: (
        row["sample_id"], row["operator_index"]))
    dataset_receipts = m40.dataset_file_receipts(data_root, sample_keys)
    manifest = {
        "schema": "m73_h67_ep35_train_calibration_packed_source_trace_v1",
        "status": "PASS_M73_DSEC_TRAIN_ONLY_S32_ALL18_SEQUENCES_EXACT_H67_EP35_FOUR_BOTTLENECK_TRACE",
        "identity": {
            "tracer_sha256": sha256(Path(__file__).resolve()),
            "m40_hook_writer_sha256": sha256(M40_PATH),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint.resolve()),
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256(args.config.resolve()),
            "paft_forward_base_config_sha256": (
                EXPECTED_FORWARD_CONFIG_SHA256),
            "data_root": str(data_root),
            "train_sequence_list_path": str(train_list),
            "train_sequence_list_sha256": sha256(train_list),
            "valid825_sequence_list_path": str(valid_list),
            "valid825_sequence_list_sha256": sha256(valid_list),
            "checkpoint_load_audit": load_audit,
            "bn_policy": bn_policy,
            "bn_modules_changed": bn_changed,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "dataset_input_files": dataset_receipts,
        },
        "split_audit": {
            "role": "DSEC_TRAIN_ONLY_PAFT_CALIBRATION",
            "train_population": len(train_keys),
            "valid825_population": len(valid_keys),
            "full_train_valid825_key_overlap": 0,
            "selected_valid825_key_overlap": 0,
            "selected_samples": SAMPLES,
            "selected_sequences": len(set(sequence_name(key) for key in sample_keys)),
            "selection": "one temporal quantile from every train sequence plus a second quantile from the 14 largest sequences",
            "selected_train_indices": selected_indices,
            "selected_sample_keys": sample_keys,
            "per_sequence_train_indices": allocation,
        },
        "cohort": {
            "samples": SAMPLES,
            "operators": list(m40.TARGETS),
            "records": len(rows),
            "shape": list(m40.EXPECTED_SHAPE),
        },
        "records": rows,
        "admission": {
            "train_only_calibration_trace": True,
            "paft_catalog": False,
            "accuracy": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    manifest_path = output_dir / "m73_train_calibration_source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M73 samples={} sequences={} records={} manifest={}".format(
        SAMPLES, manifest["split_audit"]["selected_sequences"], len(rows),
        manifest_path), flush=True)


if __name__ == "__main__":
    main()
