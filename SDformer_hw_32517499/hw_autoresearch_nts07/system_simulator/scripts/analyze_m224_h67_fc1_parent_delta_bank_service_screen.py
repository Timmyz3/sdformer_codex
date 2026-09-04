#!/usr/bin/env python3
"""Exact M51-s10 FC1 parent-delta and bank-service strong-baseline screen.

The screen keeps two effects separate: parent-delta changes signed source
work, while K-bank coissue changes service parallelism.  Every reported
composed serial cycle count charges activation/candidate scans, choice bits,
parent-output seeds, output slicing and final 96-channel commits.

This is a trace-driven premodel, not RTL, physical PPA or system speedup.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST_SHA256 = (
    "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
)
EXPECTED_RUNTIME_SHA256 = (
    "9cb5ccfc15b83c680ca8c96a816df1cdd4b5c4d956bd5c2462175b175b1b6c85"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
SELECTED_MODULE_INDICES = (6, 8, 11, 13, 16, 18, 20, 22, 24, 26)
POPCOUNT = np.array([bin(value).count("1") for value in range(256)],
                    dtype=np.uint8)
PARENTS = ("zero", "left", "up", "previous_timestep")
POINTS = (
    ("shared96", "K1_D96", 1, 96),
    ("shared96", "K2_D48", 2, 48),
    ("shared96", "K4_D24", 4, 24),
    ("shared96", "K8_D12", 8, 12),
    ("matched128", "K1_D128", 1, 128),
    ("matched128", "K2_D64", 2, 64),
    ("matched128", "K4_D32", 4, 32),
    ("matched128", "K8_D16", 8, 16),
)
MODE_ORDER = ("raw", "spatial", "temporal")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator, denominator):
    require(int(denominator) > 0, "zero ratio denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def decode_record(record, payload_root):
    path = payload_root / record["relative_path"]
    require(path.is_file(), "missing payload {}".format(path))
    require(sha256(path) == record["file_sha256"],
            "payload SHA drift {}".format(path))
    shape = tuple(int(value) for value in record["input_shape"])
    require(len(shape) == 5 and shape[-1] % 8 == 0,
            "unsupported FC1 geometry")
    expected_bytes = int(np.prod(shape)) // 8
    require(int(record["packed_bytes"]) == expected_bytes,
            "packed geometry mismatch")
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(int(packed.size) == expected_bytes, "packed size mismatch")
    packed = packed.reshape(shape[:-1] + (shape[-1] // 8,))
    active = int(POPCOUNT[packed].sum(dtype=np.int64))
    require(active == int(record["active_elements"]),
            "active element mismatch")
    return packed


def source_direction(current, residual):
    positive = int(POPCOUNT[np.bitwise_and(current, residual)].sum(
        dtype=np.int64))
    negative = int(POPCOUNT[np.bitwise_and(
        np.bitwise_not(current), residual)].sum(dtype=np.int64))
    total = int(POPCOUNT[residual].sum(dtype=np.int64))
    require(positive + negative == total, "signed source conservation")
    return total, positive, negative


def select_parent(current, mode):
    base_cost = POPCOUNT[current].sum(axis=-1, dtype=np.int32)
    choice = np.zeros(base_cost.shape, dtype=np.uint8)
    residual = current.copy()
    if mode == "raw":
        pass
    elif mode == "spatial":
        left = np.bitwise_xor(current[:, :, :, 1:, :],
                              current[:, :, :, :-1, :])
        left_cost = POPCOUNT[left].sum(axis=-1, dtype=np.int32)
        take = left_cost < base_cost[:, :, :, 1:]
        base_cost[:, :, :, 1:][take] = left_cost[take]
        choice[:, :, :, 1:][take] = 1
        residual[:, :, :, 1:, :][take] = left[take]

        up = np.bitwise_xor(current[:, :, 1:, :, :],
                            current[:, :, :-1, :, :])
        up_cost = POPCOUNT[up].sum(axis=-1, dtype=np.int32)
        take = up_cost < base_cost[:, :, 1:, :]
        base_cost[:, :, 1:, :][take] = up_cost[take]
        choice[:, :, 1:, :][take] = 2
        residual[:, :, 1:, :, :][take] = up[take]
    elif mode == "temporal":
        previous = np.bitwise_xor(current[1:, :, :, :, :],
                                  current[:-1, :, :, :, :])
        previous_cost = POPCOUNT[previous].sum(axis=-1, dtype=np.int32)
        take = previous_cost < base_cost[1:, :, :, :]
        base_cost[1:, :, :, :][take] = previous_cost[take]
        choice[1:, :, :, :][take] = 3
        residual[1:, :, :, :, :][take] = previous[take]
    else:
        raise RuntimeError("unknown parent mode {}".format(mode))
    total, positive, negative = source_direction(current, residual)
    counts = np.bincount(choice.reshape(-1), minlength=len(PARENTS))
    return residual, {
        "source_events": total,
        "positive_events": positive,
        "negative_events": negative,
        "choice_counts": dict((name, int(counts[index]))
                              for index, name in enumerate(PARENTS)),
    }


def bank_group_count(residual, banks):
    require(banks in (1, 2, 4, 8), "unsupported bank count")
    if banks == 1:
        counts = POPCOUNT[residual].sum(axis=-1, dtype=np.int32)
        return int(counts.sum(dtype=np.int64)), int(counts.max(initial=0))
    bank_counts = []
    for bank in range(banks):
        mask = sum(1 << bit for bit in range(bank, 8, banks))
        count = POPCOUNT[np.bitwise_and(residual, mask)].sum(
            axis=-1, dtype=np.int32)
        bank_counts.append(count)
    occupancy = np.stack(bank_counts, axis=-1)
    depth = occupancy.max(axis=-1)
    return int(depth.sum(dtype=np.int64)), int(depth.max(initial=0))


def fixed_overheads(shape, output_shape, choice_stats, mode):
    time, batch, height, width, channels = (int(x) for x in shape)
    output_channels = int(output_shape[-1])
    vectors = time * batch * height * width
    input_beats = ceil_div(channels, 256)
    output_blocks = ceil_div(output_channels, 96)
    current_scan = vectors * input_beats
    if mode == "spatial":
        candidate_vectors = (time * batch * height * max(0, width - 1) +
                             time * batch * max(0, height - 1) * width)
        choice_bits = 2
        chosen_parent = (choice_stats["choice_counts"]["left"] +
                         choice_stats["choice_counts"]["up"])
    elif mode == "temporal":
        candidate_vectors = max(0, time - 1) * batch * height * width
        choice_bits = 1
        chosen_parent = choice_stats["choice_counts"]["previous_timestep"]
    else:
        candidate_vectors = 0
        choice_bits = 0
        chosen_parent = 0
    return {
        "vectors": vectors,
        "input_channels": channels,
        "output_channels": output_channels,
        "input_256b_beats_per_vector": input_beats,
        "output_96lane_blocks_per_vector": output_blocks,
        "current_activation_scan_cycles": current_scan,
        "candidate_parent_scan_cycles": candidate_vectors * input_beats,
        "choice_metadata_cycles": ceil_div(vectors * choice_bits, 256)
        if choice_bits else 0,
        "chosen_parent_seed_cycles": chosen_parent * output_blocks,
        "final_commit_cycles": vectors * output_blocks,
    }


def point_metrics(residual, choice_stats, shape, output_shape,
                  family, point_name, banks, destination_lanes, mode):
    overhead = fixed_overheads(shape, output_shape, choice_stats, mode)
    groups, maximum_depth = bank_group_count(residual, banks)
    slices = ceil_div(overhead["output_channels"], destination_lanes)
    service = groups * slices
    serial = (overhead["current_activation_scan_cycles"] +
              overhead["candidate_parent_scan_cycles"] +
              overhead["choice_metadata_cycles"] +
              overhead["chosen_parent_seed_cycles"] + service +
              overhead["final_commit_cycles"])
    product_lanes = banks * destination_lanes
    bank_equivalents = banks * ceil_div(destination_lanes, 16)
    product_updates = (choice_stats["source_events"] *
                       overhead["output_channels"])
    physical_slots = service * product_lanes
    return {
        "family": family,
        "point": point_name,
        "mode": mode,
        "source_banks": banks,
        "destination_lanes_per_source": destination_lanes,
        "product_lanes": product_lanes,
        "weight_bits_per_issue": product_lanes * 8,
        "banks_128b_required": bank_equivalents,
        "fits_eight_128b_banks": bank_equivalents <= 8,
        "destination_slices": slices,
        "source_bank_groups": groups,
        "maximum_sources_in_one_bank_per_vector": maximum_depth,
        "service_cycles": service,
        "serial_cycles": serial,
        "source_events": choice_stats["source_events"],
        "positive_events": choice_stats["positive_events"],
        "negative_events": choice_stats["negative_events"],
        "product_updates": product_updates,
        "physical_product_slots": physical_slots,
        "product_slot_utilization": (float(product_updates) /
                                     float(physical_slots)
                                     if physical_slots else 1.0),
        "overheads": overhead,
    }


def add_point(total, point):
    scalar_keys = (
        "source_bank_groups", "service_cycles", "serial_cycles",
        "source_events", "positive_events", "negative_events",
        "product_updates", "physical_product_slots",
    )
    for key in scalar_keys:
        total[key] = total.get(key, 0) + int(point[key])
    total["maximum_sources_in_one_bank_per_vector"] = max(
        total.get("maximum_sources_in_one_bank_per_vector", 0),
        int(point["maximum_sources_in_one_bank_per_vector"]))
    overhead = total.setdefault("overheads", {})
    for key, value in point["overheads"].items():
        if key in ("input_channels", "output_channels",
                   "input_256b_beats_per_vector",
                   "output_96lane_blocks_per_vector"):
            continue
        overhead[key] = overhead.get(key, 0) + int(value)


def distribution(values):
    ordered = sorted(int(value) for value in values)
    require(ordered, "empty distribution")
    def nr(frac):
        return ordered[max(0, ceil_div(len(ordered) * frac[0], frac[1]) - 1)]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50_nearest_rank": nr((1, 2)),
        "p95_nearest_rank": nr((95, 100)),
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def load_stage3_rows(runtime_path):
    selected = []
    with runtime_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["name"].endswith(".mlp.fc1") and ".layers.3." in row["name"]:
                selected.append({
                    "name": row["name"],
                    "operator": row["operator"],
                    "input_shape_first": json.loads(row["input_shape_first"]),
                    "output_shape_first": json.loads(row["output_shape_first"]),
                    "input_sample_binary01_ratio": float(
                        row["input_sample_binary01_ratio"]),
                    "input_sample_ternary_ratio": float(
                        row["input_sample_ternary_ratio"]),
                    "policy": "CONVENTIONAL_NONBINARY_PATH",
                })
    require(len(selected) == 2, "expected two stage-3 FC1 rows")
    require(all(row["input_sample_binary01_ratio"] < 1.0
                for row in selected), "stage-3 binary exclusion drift")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--operator-runtime", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.operator_runtime) == EXPECTED_RUNTIME_SHA256,
            "operator runtime identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs/359 drift")
    manifest = json.loads(args.manifest.read_text())
    records = [row for row in manifest["records"]
               if int(row["module_index"]) in SELECTED_MODULE_INDICES]
    require(len(records) == 100, "expected 100 FC1 records")
    require(sorted(set(int(row["module_index"]) for row in records)) ==
            list(SELECTED_MODULE_INDICES), "FC1 module identity drift")
    require(all(row["operator"] == "Linear" and
                row["name"].endswith(".mlp.fc1") for row in records),
            "non-FC1 record selected")
    require(all(int(row["output_shape"][-1]) ==
                4 * int(row["input_shape"][-1]) for row in records),
            "FC1 expansion drift")

    aggregate = {}
    sample_totals = {}
    module_totals = {}
    per_record = []
    payload_identity = []
    for ordinal, record in enumerate(sorted(
            records, key=lambda row: (int(row["sample_id"]),
                                      int(row["module_index"])))):
        current = decode_record(record, args.payload_root)
        record_result = {
            "ordinal": ordinal,
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "name": record["name"],
            "input_shape": record["input_shape"],
            "output_shape": record["output_shape"],
            "modes": {},
        }
        payload_identity.append({
            "relative_path": record["relative_path"],
            "sha256": record["file_sha256"],
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
        })
        for mode in MODE_ORDER:
            residual, choice_stats = select_parent(current, mode)
            mode_result = {
                "choice": choice_stats,
                "points": {},
            }
            for family, point_name, banks, lanes in POINTS:
                point = point_metrics(
                    residual, choice_stats, record["input_shape"],
                    record["output_shape"], family, point_name, banks,
                    lanes, mode)
                mode_result["points"][point_name] = point
                key = "{}_{}".format(mode, point_name)
                template = aggregate.setdefault(key, {
                    "family": family,
                    "point": point_name,
                    "mode": mode,
                    "source_banks": banks,
                    "destination_lanes_per_source": lanes,
                    "product_lanes": banks * lanes,
                    "weight_bits_per_issue": banks * lanes * 8,
                    "banks_128b_required": banks * ceil_div(lanes, 16),
                    "fits_eight_128b_banks":
                        banks * ceil_div(lanes, 16) <= 8,
                })
                add_point(template, point)
                sample = sample_totals.setdefault(
                    str(record["sample_id"]), {}).setdefault(key, 0)
                sample_totals[str(record["sample_id"])][key] = (
                    sample + int(point["serial_cycles"]))
                module = module_totals.setdefault(
                    str(record["module_index"]), {}).setdefault(key, 0)
                module_totals[str(record["module_index"])][key] = (
                    module + int(point["serial_cycles"]))
            del residual
            mode_result["raw_source_work_reduction"] = ratio(
                int(record["active_elements"]),
                choice_stats["source_events"])
            record_result["modes"][mode] = mode_result
        per_record.append(record_result)

    for point in aggregate.values():
        point["product_slot_utilization"] = (
            float(point["product_updates"]) /
            float(point["physical_product_slots"])
            if point["physical_product_slots"] else 1.0)

    family_refs = {"shared96": "K1_D96", "matched128": "K1_D128"}
    for key, point in aggregate.items():
        mode = point["mode"]
        family = point["family"]
        raw_ref = aggregate["raw_{}".format(family_refs[family])]
        mode_ref = aggregate["{}_{}".format(mode, family_refs[family])]
        point["bank_parallelism_ratio_vs_same_parent_k1"] = ratio(
            mode_ref["serial_cycles"], point["serial_cycles"])
        point["parent_delta_ratio_at_family_k1"] = ratio(
            raw_ref["serial_cycles"], mode_ref["serial_cycles"])
        point["composed_ratio_vs_raw_family_k1"] = ratio(
            raw_ref["serial_cycles"], point["serial_cycles"])
        sample_ratios = []
        sample_key = "{}_{}".format(mode, point["point"])
        ref_key = "raw_{}".format(family_refs[family])
        for sample in sorted(sample_totals, key=int):
            sample_ratios.append(
                float(sample_totals[sample][ref_key]) /
                float(sample_totals[sample][sample_key]))
        point["per_sample_composed_ratio_distribution"] = {
            "count": len(sample_ratios),
            "min": min(sample_ratios),
            "max": max(sample_ratios),
            "mean": sum(sample_ratios) / len(sample_ratios),
        }

    ranked = sorted(
        aggregate.values(),
        key=lambda row: row["composed_ratio_vs_raw_family_k1"]["float"],
        reverse=True)
    best_by_family = {}
    for family in family_refs:
        candidates = [row for row in ranked
                      if row["family"] == family and
                      row["fits_eight_128b_banks"]]
        best_by_family[family] = candidates[0]

    mode_source_totals = {}
    for mode in MODE_ORDER:
        key = "{}_K1_D96".format(mode)
        mode_source_totals[mode] = {
            "source_events": aggregate[key]["source_events"],
            "positive_events": aggregate[key]["positive_events"],
            "negative_events": aggregate[key]["negative_events"],
            "raw_source_work_reduction": ratio(
                aggregate["raw_K1_D96"]["source_events"],
                aggregate[key]["source_events"]),
        }

    stage3 = load_stage3_rows(args.operator_runtime)
    advance = any(
        row["composed_ratio_vs_raw_family_k1"]["float"] >= 1.5
        and row["point"] not in family_refs.values()
        for row in best_by_family.values())
    output = {
        "schema": "m224_h67_fc1_parent_delta_bank_service_screen_v1",
        "status": ("PASS_ADVANCE_BEST_LEGAL_POINT_TO_RTL"
                   if advance else
                   "PASS_NO_GO_UNDER_STRONG_K1_AND_CURRENT_RESOURCE"),
        "identity": {
            "manifest_sha256": sha256(args.manifest),
            "operator_runtime_sha256": sha256(args.operator_runtime),
            "docs359_sha256": sha256(args.docs359),
            "selected_payloads": payload_identity,
        },
        "population": {
            "records": len(records),
            "samples": len(set(int(row["sample_id"]) for row in records)),
            "modules": len(SELECTED_MODULE_INDICES),
            "selected_module_indices": list(SELECTED_MODULE_INDICES),
            "selected_names": sorted(set(row["name"] for row in records)),
            "stage3_conventional_nonbinary": stage3,
        },
        "source_work": mode_source_totals,
        "aggregate_points": aggregate,
        "best_legal_by_family": best_by_family,
        "sample_serial_cycles": sample_totals,
        "module_serial_cycles": module_totals,
        "per_record": per_record,
        "admission": {
            "exact_trace_screen": True,
            "all_100_payload_sha_verified": True,
            "stage3_nonbinary_fallback_bound": True,
            "minimum_serial_speedup_gate": 1.5,
            "advance_to_rtl": advance,
            "rtl": False,
            "dc": False,
            "complete_fc1": False,
            "complete_ffn": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": {
            "ratios_are": "M51-s10 FC1 trace premodel ratios",
            "ratios_are_not": [
                "RTL measured throughput",
                "complete FC1 or FFN speedup",
                "system speedup",
                "macro-aware PPA or energy",
            ],
            "parent_delta_and_bank_parallelism_separately_reported": True,
            "cross_lane_family_comparison_prohibited": True,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / \
        "m224_h67_fc1_parent_delta_bank_service_screen_r1.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / \
        "m224_h67_fc1_parent_delta_bank_service_points_r1.csv"
    fields = [
        "family", "mode", "point", "source_banks",
        "destination_lanes_per_source", "product_lanes",
        "banks_128b_required", "source_events", "source_bank_groups",
        "service_cycles", "serial_cycles",
        "parent_delta_ratio_at_family_k1",
        "bank_parallelism_ratio_vs_same_parent_k1",
        "composed_ratio_vs_raw_family_k1",
        "per_sample_min", "per_sample_mean", "per_sample_max",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(aggregate.values(),
                          key=lambda item: (item["family"], item["mode"],
                                            item["source_banks"])):
            dist = row["per_sample_composed_ratio_distribution"]
            writer.writerow({
                "family": row["family"],
                "mode": row["mode"],
                "point": row["point"],
                "source_banks": row["source_banks"],
                "destination_lanes_per_source":
                    row["destination_lanes_per_source"],
                "product_lanes": row["product_lanes"],
                "banks_128b_required": row["banks_128b_required"],
                "source_events": row["source_events"],
                "source_bank_groups": row["source_bank_groups"],
                "service_cycles": row["service_cycles"],
                "serial_cycles": row["serial_cycles"],
                "parent_delta_ratio_at_family_k1":
                    row["parent_delta_ratio_at_family_k1"]["float"],
                "bank_parallelism_ratio_vs_same_parent_k1":
                    row["bank_parallelism_ratio_vs_same_parent_k1"]["float"],
                "composed_ratio_vs_raw_family_k1":
                    row["composed_ratio_vs_raw_family_k1"]["float"],
                "per_sample_min": dist["min"],
                "per_sample_mean": dist["mean"],
                "per_sample_max": dist["max"],
            })
    print("PASS M224 records={} best96={:.6f} best128={:.6f} advance={}".
          format(len(records),
                 best_by_family["shared96"]
                 ["composed_ratio_vs_raw_family_k1"]["float"],
                 best_by_family["matched128"]
                 ["composed_ratio_vs_raw_family_k1"]["float"],
                 advance))


if __name__ == "__main__":
    main()
