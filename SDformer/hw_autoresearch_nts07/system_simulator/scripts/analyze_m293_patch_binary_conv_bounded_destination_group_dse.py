#!/usr/bin/env python3
"""Screen bounded destination-group pruning on frozen binary patch Conv2d."""

from __future__ import division

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
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
    answer = 1
    for value in values:
        answer *= int(value)
    return answer


def normalize_tbc_hw(shape):
    shape = tuple(int(value) for value in shape)
    if len(shape) == 4:
        return (shape[0], 1, shape[1], shape[2], shape[3])
    require(len(shape) == 5, "unsupported Conv shape")
    return shape


def infer_stride(height, width, out_height, out_width):
    candidates = []
    for stride in (1, 2):
        got_height = (height + 2 - 3) // stride + 1
        got_width = (width + 2 - 3) // stride + 1
        if (got_height, got_width) == (out_height, out_width):
            candidates.append(stride)
    require(len(candidates) == 1, "ambiguous Conv stride")
    return candidates[0]


def record_source_activity(bits, out_height, out_width, stride):
    """Return activity for source keys channel*9+tap and max token fan-in."""
    t_count, batch, channels, _height, _width = bits.shape
    padded = np.pad(bits, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)))
    activity = np.zeros((channels, 9), dtype=np.int64)
    per_output = np.zeros((t_count, batch, out_height, out_width),
                          dtype=np.uint16)
    for ky in range(3):
        for kx in range(3):
            tap = ky * 3 + kx
            sampled = padded[
                :, :, :, ky:ky + stride * out_height:stride,
                kx:kx + stride * out_width:stride,
            ]
            require(sampled.shape[-2:] == (out_height, out_width),
                    "bad Conv receptive-field slice")
            activity[:, tap] = sampled.sum(axis=(0, 1, 3, 4),
                                            dtype=np.int64)
            per_output += sampled.sum(axis=2, dtype=np.uint16)
    return activity.reshape(-1), int(per_output.max())


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
            "m293_patch_binary_conv_bounded_destination_group_dse_contract_v1",
            "M293 contract schema drift")
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
        require(path.is_file(), "missing M293 input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "M293 input SHA drift for {}: {}".format(label, digest))
        paths[label] = path
        identity[label] = {"path": spec["path"], "sha256": digest}

    import torch

    helper = load_module(paths["pickle_import_helper"], "m293_pickle_helper")
    helper.install_pickle_import_paths(repo)
    checkpoint = torch.load(str(paths["checkpoint"]), map_location="cpu",
                            weights_only=False)
    require(hasattr(checkpoint, "state_dict"),
            "M293 checkpoint is not a model object")
    state = checkpoint.state_dict()
    manifest = strict_json(paths["m51_manifest"])
    require(manifest["packing"]["layout"] == "C_ORDER_FLAT" and
            manifest["packing"]["bit_order"] == "LITTLE_WITHIN_BYTE" and
            "exact-binary" in manifest["claim_boundary"],
            "M293 M51 packing/scope drift")

    source_activity = {}
    token_maximum = defaultdict(int)
    record_counts = defaultdict(int)
    active_input_total = 0
    source_contribution_total = 0
    record_receipts = []
    for record in manifest["records"]:
        name = record["name"]
        payload = paths["m51_manifest"].parent / record["relative_path"]
        if ("patch_embed" not in name or record["operator"] != "Conv2d" or
                not payload.is_file()):
            continue
        require(sha256(payload) == record["file_sha256"],
                "M293 Conv payload SHA drift")
        shape = normalize_tbc_hw(record["input_shape"])
        out_shape = normalize_tbc_hw(record["output_shape"])
        require(product(shape) == int(record["input_elements"]),
                "M293 Conv input extent drift")
        require(out_shape[2] == 96 and shape[:2] == out_shape[:2],
                "M293 Conv output identity drift")
        packed = np.fromfile(str(payload), dtype=np.uint8)
        require(packed.size == int(record["packed_bytes"]),
                "M293 packed byte extent drift")
        bits = np.unpackbits(packed, bitorder="little")[:product(shape)]
        bits = bits.reshape(shape)
        active = int(bits.sum(dtype=np.uint64))
        require(active == int(record["active_elements"]),
                "M293 active input mismatch")
        stride = infer_stride(shape[-2], shape[-1],
                              out_shape[-2], out_shape[-1])
        activity, maximum = record_source_activity(
            bits, out_shape[-2], out_shape[-1], stride)
        source = int(activity.sum())
        if name not in source_activity:
            source_activity[name] = activity
        else:
            source_activity[name] += activity
        token_maximum[name] = max(token_maximum[name], maximum)
        record_counts[name] += 1
        active_input_total += active
        source_contribution_total += source
        record_receipts.append({
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "module": name,
            "stride": stride,
            "active_input_elements": active,
            "source_contributions": source,
            "maximum_active_sources_per_output": maximum,
        })
    require(len(source_activity) == 6 and len(record_receipts) == 60 and
            all(count == 10 for count in record_counts.values()),
            "M293 six-module/sixty-record population drift")
    require(active_input_total == 325287254 and
            source_contribution_total == 1774268587,
            "M293 M222 population cross-check drift")

    group_sizes = [int(value) for value in
                   contract["dse"]["destination_group_sizes"]]
    betas = [int(value) for value in
             contract["dse"]["maximum_absolute_int8_weight_per_group"]]
    require(group_sizes == [4, 8, 16, 32, 96] and
            betas == [0, 8, 16, 24, 32, 48, 64, 80, 96],
            "M293 DSE grid drift")
    aggregates = {}
    for group_size in group_sizes:
        aggregates[group_size] = dict((beta, {
            "baseline_group_tasks": 0,
            "kept_group_tasks": 0,
            "static_source_group_pairs": 0,
            "static_source_group_pairs_removed": 0,
            "maximum_trace_active_sources_per_output": 0,
        }) for beta in betas)

    per_module = []
    for name in sorted(source_activity):
        key = name + ".weight"
        require(key in state, "missing M293 Conv weight: " + key)
        weight = state[key].detach().cpu().to(torch.float64)
        require(weight.ndim == 4 and tuple(weight.shape[2:]) == (3, 3) and
                int(weight.shape[0]) == 96 and
                int(weight.shape[1]) * 9 == len(source_activity[name]),
                "M293 Conv weight geometry drift")
        flat = weight.reshape(int(weight.shape[0]), -1)
        row_maximum = flat.abs().amax(dim=1)
        scale = torch.where(row_maximum == 0, torch.ones_like(row_maximum),
                            row_maximum / 127.0)
        quantized = torch.clamp(torch.round(flat / scale[:, None]),
                                -127, 127).to(torch.int16).numpy()
        require(not bool((quantized == -128).any()), "M293 emitted -128")
        activity = source_activity[name]
        module_row = {
            "module": name,
            "input_channels": int(weight.shape[1]),
            "output_channels": int(weight.shape[0]),
            "kernel": [3, 3],
            "records": int(record_counts[name]),
            "source_contributions": int(activity.sum()),
            "maximum_active_sources_per_output": token_maximum[name],
            "groups": {},
        }
        for group_size in group_sizes:
            groups = int(weight.shape[0]) // group_size
            maximum = np.abs(quantized).reshape(
                groups, group_size, quantized.shape[1]).max(axis=1)
            baseline = int(activity.sum()) * groups
            group_row = {}
            for beta in betas:
                keep = maximum > beta if beta else np.ones_like(maximum,
                                                                dtype=bool)
                kept = int((keep * activity[None, :]).sum())
                removed_pairs = int((~keep).sum())
                row = {
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
                group_row[str(beta)] = row
                aggregate = aggregates[group_size][beta]
                aggregate["baseline_group_tasks"] += baseline
                aggregate["kept_group_tasks"] += kept
                aggregate["static_source_group_pairs"] += int(keep.size)
                aggregate["static_source_group_pairs_removed"] += removed_pairs
                aggregate["maximum_trace_active_sources_per_output"] = max(
                    aggregate["maximum_trace_active_sources_per_output"],
                    token_maximum[name])
            module_row["groups"][str(group_size)] = group_row
        per_module.append(module_row)

    envelope = strict_json(paths["m221_envelope"])["frozen_h67_compute_envelope"]
    scope = contract["frozen_cycle_scope"]
    require(int(envelope["cycles_per_frame"]) ==
            int(scope["compute_envelope_cycles"]), "M293 envelope drift")
    patch = envelope["hotspots"]["patch_embed_8_conv"]
    require(int(patch["cycles"]) == int(scope["all_patch_cycles"]) and
            int(scope["eligible_six_binary_patch_cycles"]) +
            int(scope["excluded_two_nonbinary_patch_cycles"]) ==
            int(scope["all_patch_cycles"]), "M293 patch scope drift")
    m222 = strict_json(paths["m222_independent_recompute"])
    require(int(m222["population"]["records"]) == 60 and
            int(m222["population"]["source_contributions"]) ==
            source_contribution_total,
            "M293 independent M222 cross-check drift")

    full_cycles = int(scope["compute_envelope_cycles"])
    eligible_cycles = int(scope["eligible_six_binary_patch_cycles"])
    all_patch_cycles = int(scope["all_patch_cycles"])
    excluded_cycles = int(scope["excluded_two_nonbinary_patch_cycles"])
    aggregate_rows = {}
    first_crossing = {}
    for group_size in group_sizes:
        rows = []
        for beta in betas:
            raw = aggregates[group_size][beta]
            ratio = (float(raw["baseline_group_tasks"]) /
                     float(raw["kept_group_tasks"]))
            scaled_eligible = eligible_cycles / ratio
            scaled_patch = excluded_cycles + scaled_eligible
            row = {
                "destination_group_size": group_size,
                "maximum_absolute_int8_weight": beta,
                "baseline_group_tasks": raw["baseline_group_tasks"],
                "kept_group_tasks": raw["kept_group_tasks"],
                "weighted_group_task_fraction_removed":
                    float(raw["baseline_group_tasks"] -
                          raw["kept_group_tasks"]) /
                    float(raw["baseline_group_tasks"]),
                "ideal_task_compaction_speedup": ratio,
                "static_source_group_pairs":
                    raw["static_source_group_pairs"],
                "static_source_group_pairs_removed":
                    raw["static_source_group_pairs_removed"],
                "static_source_group_pair_fraction_removed":
                    float(raw["static_source_group_pairs_removed"]) /
                    float(raw["static_source_group_pairs"]),
                "eligible_binary_patch_only_cycle_sensitivity": ratio,
                "all_patch_scope_corrected_cycle_sensitivity":
                    all_patch_cycles / scaled_patch,
                "full_envelope_scope_corrected_cycle_sensitivity":
                    full_cycles /
                    (full_cycles - eligible_cycles + scaled_eligible),
                "conservative_trace_per_destination_accumulator_bound_int8":
                    beta * raw["maximum_trace_active_sources_per_output"],
                "scope_warning": "ideal task compaction only; no router, bank-conflict, scan/commit, or executable cycle adapter",
            }
            rows.append(row)
        aggregate_rows[str(group_size)] = rows
        candidates = [row for row in rows if
                      row["full_envelope_scope_corrected_cycle_sensitivity"] >=
                      float(contract["dse"]["target_full_envelope_speedup"])]
        first_crossing[str(group_size)] = candidates[0] if candidates else None

    result = {
        "schema": "m293_patch_binary_conv_bounded_destination_group_dse_v1",
        "status": "PASS_OPPORTUNITY_SCREEN_NOT_CYCLE_OR_ACCURACY_ADMISSION",
        "identity": identity,
        "population": {
            "modules": len(source_activity),
            "records": len(record_receipts),
            "active_input_elements": active_input_total,
            "source_contributions": source_contribution_total,
            "product_updates": source_contribution_total * 96,
            "record_receipts": record_receipts,
        },
        "cycle_scope": scope,
        "mechanism": {
            "name": "bounded source-by-destination-group task elision",
            "zero_budget_behavior": "beta=0 performs no pruning and is the exact engine subset",
            "skip_predicate": "omit one active convolution source/destination-group task only when every per-row INT8 weight in that destination group has magnitude <= beta",
            "deterministic_bound": "each omitted task changes each destination INT8 accumulator by at most beta; beta times omitted-source count is a conservative integer-domain ledger",
        },
        "aggregate_grid": aggregate_rows,
        "first_beta_crossing_scope_corrected_full_envelope_1p15":
            first_crossing,
        "per_module": per_module,
        "decision_policy": {
            "accuracy_before_rtl": True,
            "maximum_absolute_aee_increase": 0.02,
            "crossing_is_only_eligible_for_paired_s10": True,
            "no_crossing_or_aggressive_cliff": "NO_GO_PRIMARY",
        },
        "admission": {
            "checkpoint_trace_weight_opportunity": True,
            "scope_correct_amdahl_sensitivity": True,
            "modified_forward_accuracy": False,
            "hardware_cycles": False,
            "rtl": False,
            "dc": False,
            "power": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / "m293_patch_binary_conv_bounded_destination_group_dse_r1.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    require(sha256(source_path) == source_start,
            "M293 analyzer changed during execution")
    print("PASS M293 modules={} records={} source={} output={}".format(
        len(source_activity), len(record_receipts),
        source_contribution_total, target))


if __name__ == "__main__":
    main()
