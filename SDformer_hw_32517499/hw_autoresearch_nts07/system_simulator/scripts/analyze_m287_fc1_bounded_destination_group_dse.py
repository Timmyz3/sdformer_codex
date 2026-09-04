#!/usr/bin/env python3
"""Screen bounded destination-group pruning on frozen binary FC1 payloads."""

from __future__ import division

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m287_fc1_bounded_destination_group_dse_contract_v1",
            "M287 contract schema drift")
    hw = contract_path.parents[1]
    repo = hw.parent
    paths = {}
    identity = {
        "contract": {"path": str(contract_path.relative_to(hw)),
                     "sha256": sha256(contract_path)},
        "analyzer": {"path": str(source_path.relative_to(repo)),
                     "sha256": source_start},
    }
    for label, spec in contract["identity"].items():
        path = (hw / spec["path"]).resolve()
        require(path.is_file(), "missing M287 input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "M287 input SHA drift for {}: {}".format(label, digest))
        paths[label] = path
        identity[label] = {"path": spec["path"], "sha256": digest}

    import torch

    helper = load_module(paths["pickle_import_helper"], "m287_pickle_helper")
    helper.install_pickle_import_paths(repo)
    checkpoint = torch.load(str(paths["checkpoint"]), map_location="cpu",
                            weights_only=False)
    require(hasattr(checkpoint, "state_dict"),
            "M287 checkpoint is not a model object")
    state = checkpoint.state_dict()
    manifest = strict_json(paths["m51_manifest"])
    require(manifest["packing"]["layout"] == "C_ORDER_FLAT" and
            manifest["packing"]["bit_order"] == "LITTLE_WITHIN_BYTE",
            "M287 M51 packing drift")
    lookup = np.asarray([bin(value).count("1") for value in range(256)],
                        dtype=np.uint8)
    channel_activity = {}
    token_maximum = defaultdict(int)
    record_counts = defaultdict(int)
    for record in manifest["records"]:
        name = record["name"]
        if ".mlp.fc1" not in name:
            continue
        payload = paths["m51_manifest"].parent / record["relative_path"]
        if not payload.is_file():
            continue
        require(sha256(payload) == record["file_sha256"],
                "M287 FC1 payload SHA drift")
        shape = [int(value) for value in record["input_shape"]]
        channels = shape[-1]
        require(channels % 8 == 0 and
                product(shape) == int(record["input_elements"]),
                "M287 FC1 shape drift")
        packed = np.fromfile(str(payload), dtype=np.uint8)
        require(packed.size * 8 == int(record["input_elements"]),
                "M287 FC1 packed extent drift")
        bits = np.unpackbits(packed.reshape(-1, channels // 8), axis=1,
                             bitorder="little")[:, :channels]
        per_channel = bits.sum(axis=0, dtype=np.int64)
        per_token = bits.sum(axis=1, dtype=np.int64)
        require(int(per_channel.sum()) == int(record["active_elements"]),
                "M287 active element drift")
        if name not in channel_activity:
            channel_activity[name] = per_channel
        else:
            channel_activity[name] += per_channel
        token_maximum[name] = max(token_maximum[name], int(per_token.max()))
        record_counts[name] += 1
    require(len(channel_activity) == 10 and
            all(count == 10 for count in record_counts.values()),
            "M287 binary FC1 population drift")
    require(sum(int(row.sum()) for row in channel_activity.values()) == 112213979,
            "M287 aggregate FC1 event drift")

    group_sizes = [int(value) for value in
                   contract["dse"]["destination_group_sizes"]]
    betas = [int(value) for value in
             contract["dse"]["maximum_absolute_int8_weight_per_group"]]
    require(group_sizes == [4, 8, 16, 32, 96] and
            betas == [0, 8, 16, 24, 32, 48, 64, 80, 96],
            "M287 DSE grid drift")
    aggregates = {}
    per_module = []
    for group_size in group_sizes:
        aggregates[group_size] = dict((beta, {
            "baseline_group_tasks": 0,
            "kept_group_tasks": 0,
            "static_source_group_pairs": 0,
            "static_source_group_pairs_removed": 0,
            "maximum_trace_active_sources_per_token": 0,
        }) for beta in betas)

    for name in sorted(channel_activity):
        key = name + ".weight"
        require(key in state, "missing M287 FC1 weight: " + key)
        weight = state[key].detach().cpu().to(torch.float64)
        require(weight.ndim == 2 and weight.shape[1] == len(channel_activity[name]),
                "M287 FC1 weight geometry drift")
        row_maximum = weight.abs().amax(dim=1)
        scale = torch.where(row_maximum == 0, torch.ones_like(row_maximum),
                            row_maximum / 127.0)
        quantized = torch.clamp(torch.round(weight / scale[:, None]),
                                -127, 127).to(torch.int16).numpy()
        require(not bool((quantized == -128).any()), "M287 emitted -128")
        activity = channel_activity[name]
        module_row = {"module": name, "input_channels": int(weight.shape[1]),
                      "output_channels": int(weight.shape[0]),
                      "records": int(record_counts[name]),
                      "source_events": int(activity.sum()),
                      "maximum_active_sources_per_token": token_maximum[name],
                      "groups": {}}
        for group_size in group_sizes:
            require(int(weight.shape[0]) % group_size == 0,
                    "M287 destination group does not divide output")
            groups = int(weight.shape[0]) // group_size
            maximum = np.abs(quantized).reshape(
                groups, group_size, int(weight.shape[1])).max(axis=1)
            baseline = int(activity.sum()) * groups
            group_row = {}
            for beta in betas:
                keep = maximum > beta if beta else np.ones_like(maximum, dtype=bool)
                kept = int((keep * activity[None, :]).sum())
                removed_pairs = int((~keep).sum())
                data = {
                    "maximum_absolute_int8_weight": beta,
                    "baseline_group_tasks": baseline,
                    "kept_group_tasks": kept,
                    "weighted_group_task_fraction_removed":
                        float(baseline - kept) / float(baseline),
                    "ideal_task_compaction_speedup":
                        float(baseline) / float(kept) if kept else None,
                    "static_source_group_pairs": int(keep.size),
                    "static_source_group_pairs_removed": removed_pairs,
                    "per_omitted_task_per_destination_accumulator_bound_int8": beta,
                    "conservative_trace_per_destination_accumulator_bound_int8":
                        beta * token_maximum[name],
                }
                group_row[str(beta)] = data
                aggregate = aggregates[group_size][beta]
                aggregate["baseline_group_tasks"] += baseline
                aggregate["kept_group_tasks"] += kept
                aggregate["static_source_group_pairs"] += int(keep.size)
                aggregate["static_source_group_pairs_removed"] += removed_pairs
                aggregate["maximum_trace_active_sources_per_token"] = max(
                    aggregate["maximum_trace_active_sources_per_token"],
                    token_maximum[name])
            module_row["groups"][str(group_size)] = group_row
        per_module.append(module_row)

    envelope = strict_json(paths["m221_envelope"])["frozen_h67_compute_envelope"]
    envelope_cycles = int(envelope["cycles_per_frame"])
    fc1_cycles = int(envelope["hotspots"]["fc1"]["cycles"])
    require(envelope_cycles == 620302905 and fc1_cycles == 118370114,
            "M287 frozen envelope drift")
    target = float(contract["dse"]["target_full_envelope_speedup"])
    aggregate_rows = {}
    for group_size in group_sizes:
        rows = []
        for beta in betas:
            row = aggregates[group_size][beta]
            baseline = row["baseline_group_tasks"]
            kept = row["kept_group_tasks"]
            module_speedup = float(baseline) / float(kept) if kept else None
            projected = (float(envelope_cycles) /
                         float(envelope_cycles - fc1_cycles +
                               fc1_cycles / module_speedup))
            rows.append({
                "destination_group_size": group_size,
                "maximum_absolute_int8_weight": beta,
                "baseline_group_tasks": baseline,
                "kept_group_tasks": kept,
                "weighted_group_task_fraction_removed":
                    float(baseline - kept) / float(baseline),
                "ideal_task_compaction_speedup": module_speedup,
                "fc1_only_full_compute_envelope_sensitivity": projected,
                "crosses_1p15_sensitivity_gate": projected >= target,
                "static_source_group_fraction_removed":
                    float(row["static_source_group_pairs_removed"]) /
                    float(row["static_source_group_pairs"]),
                "per_omitted_task_per_destination_accumulator_bound_int8": beta,
                "conservative_trace_per_destination_accumulator_bound_int8":
                    beta * row["maximum_trace_active_sources_per_token"],
            })
        aggregate_rows[str(group_size)] = rows

    first_crossing = {}
    for group_size in group_sizes:
        candidates = [row for row in aggregate_rows[str(group_size)]
                      if row["crosses_1p15_sensitivity_gate"]]
        first_crossing[str(group_size)] = candidates[0] if candidates else None

    result = {
        "schema": "m287_fc1_bounded_destination_group_dse_v1",
        "status": "PASS_CHECKPOINT_TRACE_BOUND_FC1_OPTIMISTIC_COMPACTION_DSE",
        "identity": identity,
        "scope": {
            "binary_fc1_modules": len(channel_activity),
            "records": sum(record_counts.values()),
            "source_events": sum(int(row.sum()) for row in channel_activity.values()),
            "stage3_nonbinary_fc1_modules_excluded": 2,
            "frozen_compute_envelope_cycles": envelope_cycles,
            "frozen_fc1_cycles": fc1_cycles,
        },
        "mechanism": {
            "name": "bounded destination-group source-task elision",
            "zero_budget_behavior": "beta=0 performs no pruning and is the exact engine subset",
            "skip_predicate": "omit a source/destination-group task only when every per-row INT8 weight in that group has absolute value <= beta",
            "deterministic_bound": "each omitted task changes each destination INT8 accumulator by at most beta; a runtime omitted-source counter gives beta times count",
            "required_hardware_not_modeled": "group-task metadata, compactor, tag router, conflict-free accumulator service, and their cycles/area",
        },
        "aggregate_dse": aggregate_rows,
        "first_beta_crossing_fc1_only_1p15_sensitivity": first_crossing,
        "per_module": per_module,
        "decision": {
            "promote_to_accuracy_screen": "only group-size 4/8 points crossing the 1.15 optimistic sensitivity gate; a miss remains a hard NO-GO",
            "rtl_before_accuracy": False,
            "fc2_and_conv_capture_needed_for_less_aggressive_combined_budget": True,
        },
        "admission": {
            "checkpoint_bound_weight_dse": True,
            "trace_weighted_opportunity": True,
            "executable_cycle_schedule": False,
            "accuracy": False,
            "rtl": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    require(sha256(source_path) == source_start, "M287 analyzer self drift")
    output = args.output_dir.resolve()
    require(not output.exists(), "refusing to overwrite M287 output")
    output.mkdir(parents=True)
    target_path = output / "m287_fc1_bounded_destination_group_dse_r1.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("PASS M287 modules={} events={} first_g4={} output={}".format(
        len(channel_activity), result["scope"]["source_events"],
        first_crossing["4"]["maximum_absolute_int8_weight"]
        if first_crossing["4"] else None, target_path))


if __name__ == "__main__":
    main()
