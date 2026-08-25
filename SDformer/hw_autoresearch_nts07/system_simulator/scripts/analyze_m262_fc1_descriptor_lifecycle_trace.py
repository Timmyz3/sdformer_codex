#!/usr/bin/env python3
"""Map the frozen M230 raw FC1 population onto the M262 lifecycle.

The comparison uses one identical small-width module and identical factor,
weight, Acc19 and commit ports for dense, bit-sparse and context-factorized
streams.  A frozen 96-output-channel block is serialized as twelve eight-lane
slices.  This is an exact aggregate mapping, not a full-trace RTL replay.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "m230_result": "6110dff1cac748ca934e05033ddabe39f06e8b54286699a7843c209ddfe4a6ca",
    "m230_seal": "133c32c37d6ff61d19ca119634b5604d8a9fe12dd510cd4d9425e59e967247e5",
    "m230_review_seal": "7b8e904a873d2b2abf95667a3b6dcff100400f2127db661cd59074905eddadc4",
    "m262_vcs_seal": "f60f3fa5639d7e9410a081afd6e285a7d83443867f2f0f4b110e4c4956450245",
    "m262_contract": "b1bbdee8d0b151af094eef9378b50936358695eced1553398d13a908dc824415",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
CONTEXTS = 8
RTL_LANES = 8
FROZEN_OUTPUT_BLOCK_LANES = 96
LANE_SLICES = FROZEN_OUTPUT_BLOCK_LANES // RTL_LANES
NONEMPTY_OVERHEAD = 34
EMPTY_CYCLES = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ratio(numerator, denominator):
    require(int(denominator) > 0, "zero ratio denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def lifecycle_point(group_streams, nonempty_streams, empty_streams,
                    descriptors, updates, mode):
    if mode == "dense":
        require(empty_streams == 0, "dense stream cannot use empty bypass")
        descriptor_cycles = 9 * descriptors
    elif mode == "bit_sparse":
        require(descriptors == updates, "bit-sparse one-hot drift")
        descriptor_cycles = 9 * descriptors
    elif mode == "context_factorized":
        require(descriptors <= updates, "factorized union drift")
        descriptor_cycles = 6 * descriptors + 3 * updates
    else:
        raise RuntimeError("unknown lifecycle mode")
    cycles_per_slice_population = (
        empty_streams * EMPTY_CYCLES
        + nonempty_streams * NONEMPTY_OVERHEAD
        + descriptor_cycles
    )
    slices = LANE_SLICES
    tile_instances = group_streams * slices
    nonempty_tiles = nonempty_streams * slices
    empty_tiles = empty_streams * slices
    descriptor_instances = descriptors * slices
    update_instances = updates * slices
    return {
        "mode": mode,
        "lane_slices_per_frozen_96lane_block": slices,
        "tile_instances": tile_instances,
        "nonempty_tiles": nonempty_tiles,
        "empty_bypass_tiles": empty_tiles,
        "descriptor_instances": descriptor_instances,
        "context_update_instances": update_instances,
        "factor_requests": descriptor_instances,
        "weight_requests": descriptor_instances,
        "acc_read_requests": update_instances + nonempty_tiles * CONTEXTS,
        "acc_write_requests": update_instances + nonempty_tiles * CONTEXTS,
        "commit_beats": nonempty_tiles * CONTEXTS,
        "descriptor_cycles": descriptor_cycles * slices,
        "lifecycle_cycles": cycles_per_slice_population * slices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m230-result", required=True, type=Path)
    parser.add_argument("--m230-seal", required=True, type=Path)
    parser.add_argument("--m230-review-seal", required=True, type=Path)
    parser.add_argument("--m262-vcs-seal", required=True, type=Path)
    parser.add_argument("--m262-vcs-receipt", required=True, type=Path)
    parser.add_argument("--m262-contract", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    identity = {
        "m230_result": sha256(args.m230_result),
        "m230_seal": sha256(args.m230_seal),
        "m230_review_seal": sha256(args.m230_review_seal),
        "m262_vcs_seal": sha256(args.m262_vcs_seal),
        "m262_contract": sha256(args.m262_contract),
        "docs359": sha256(args.docs359),
    }
    require(identity == EXPECTED, "frozen input identity drift")
    receipt = dict(line.split("=", 1)
                   for line in args.m262_vcs_receipt.read_text().splitlines()
                   if "=" in line)
    require(receipt.get("status") ==
            "PASS_M262_FC1_DESCRIPTOR_LIFECYCLE_EXACT_VCS",
            "M262 VCS status drift")
    require(receipt.get("lanes") == "8" and
            receipt.get("contexts") == "8", "M262 geometry drift")
    require(receipt.get("numeric_mismatches") == "0" and
            receipt.get("transaction_mismatches") == "0" and
            receipt.get("assertion_failures") == "0", "M262 VCS gate drift")

    prior = json.loads(args.m230_result.read_text())
    records = prior["per_record"]
    require(len(records) == 100, "M230 record population drift")
    per_record = []
    aggregate_input = {
        "group_streams": 0,
        "nonempty_group_streams": 0,
        "empty_group_streams": 0,
        "unique_source_weight_reads": 0,
        "source_context_updates": 0,
        "dense_source_context_descriptors": 0,
    }
    for record in records:
        raw = record["modes"]["raw"]
        groups = int(raw["group_streams"])
        nonempty = int(raw["nonempty_group_streams"])
        empty = int(raw["empty_group_streams"])
        unique = int(raw["unique_source_weight_reads"])
        updates = int(raw["service_cycles"]["1"])
        input_channels = int(raw["input_channels"])
        dense_descriptors = groups * CONTEXTS * input_channels
        dense = lifecycle_point(groups, groups, 0, dense_descriptors,
                                dense_descriptors, "dense")
        bit_sparse = lifecycle_point(groups, nonempty, empty, updates,
                                     updates, "bit_sparse")
        factorized = lifecycle_point(groups, nonempty, empty, unique,
                                     updates, "context_factorized")
        per_record.append({
            "ordinal": int(record["ordinal"]),
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "name": record["name"],
            "input_channels": input_channels,
            "points": {
                "dense": dense,
                "bit_sparse": bit_sparse,
                "context_factorized": factorized,
            },
            "speedup": {
                "bit_sparse_vs_dense": ratio(
                    dense["lifecycle_cycles"],
                    bit_sparse["lifecycle_cycles"]),
                "context_factorized_vs_dense": ratio(
                    dense["lifecycle_cycles"],
                    factorized["lifecycle_cycles"]),
                "context_factorized_vs_bit_sparse": ratio(
                    bit_sparse["lifecycle_cycles"],
                    factorized["lifecycle_cycles"]),
            },
        })
        aggregate_input["group_streams"] += groups
        aggregate_input["nonempty_group_streams"] += nonempty
        aggregate_input["empty_group_streams"] += empty
        aggregate_input["unique_source_weight_reads"] += unique
        aggregate_input["source_context_updates"] += updates
        aggregate_input["dense_source_context_descriptors"] += dense_descriptors

    frozen = prior["aggregate_trace"]["raw"]
    checks = {
        "group_streams": int(frozen["group_streams"]),
        "nonempty_group_streams": int(frozen["nonempty_group_streams"]),
        "empty_group_streams": int(frozen["empty_group_streams"]),
        "unique_source_weight_reads": int(frozen["unique_source_weight_reads"]),
        "source_context_updates": int(frozen["service_cycles"]["1"]),
    }
    require(all(aggregate_input[key] == value for key, value in checks.items()),
            "M230 raw aggregate reconciliation drift")
    groups = aggregate_input["group_streams"]
    nonempty = aggregate_input["nonempty_group_streams"]
    empty = aggregate_input["empty_group_streams"]
    unique = aggregate_input["unique_source_weight_reads"]
    updates = aggregate_input["source_context_updates"]
    dense_descriptors = aggregate_input["dense_source_context_descriptors"]
    points = {
        "dense": lifecycle_point(groups, groups, 0, dense_descriptors,
                                 dense_descriptors, "dense"),
        "bit_sparse": lifecycle_point(groups, nonempty, empty, updates,
                                      updates, "bit_sparse"),
        "context_factorized": lifecycle_point(groups, nonempty, empty, unique,
                                              updates, "context_factorized"),
    }
    speedup = {
        "bit_sparse_vs_dense": ratio(points["dense"]["lifecycle_cycles"],
                                     points["bit_sparse"]["lifecycle_cycles"]),
        "context_factorized_vs_dense": ratio(
            points["dense"]["lifecycle_cycles"],
            points["context_factorized"]["lifecycle_cycles"]),
        "context_factorized_vs_bit_sparse": ratio(
            points["bit_sparse"]["lifecycle_cycles"],
            points["context_factorized"]["lifecycle_cycles"]),
        "context_factorized_weight_read_reduction_vs_bit_sparse": ratio(
            points["bit_sparse"]["weight_requests"],
            points["context_factorized"]["weight_requests"]),
    }
    distributions = {}
    for metric in speedup:
        values = [float(row["speedup"][metric]["float"])
                  for row in per_record if metric in row["speedup"]]
        if values:
            distributions[metric] = {
                "count": len(values), "min": min(values),
                "mean": sum(values) / len(values), "max": max(values),
            }

    output = {
        "schema": "m262_fc1_descriptor_lifecycle_trace_v1",
        "status": "PASS_M262_EXACT_AGGREGATE_SMALL_WIDTH_LIFECYCLE_MAPPING",
        "identity": identity,
        "vcs_receipt_sha256": sha256(args.m262_vcs_receipt),
        "population": {
            "records": 100,
            "samples": 10,
            "binary_fc1_modules": 10,
            "stage3_nonbinary_fc1_modules": 2,
            "stage3_policy": "conventional fallback",
        },
        "same_port_contract": {
            "rtl_lanes": RTL_LANES,
            "frozen_output_block_lanes": FROZEN_OUTPUT_BLOCK_LANES,
            "serialized_lane_slices": LANE_SLICES,
            "contexts": CONTEXTS,
            "factor_ports": "one request / one response",
            "weight_ports": "one request / one response",
            "accumulator_ports": "one read request / response and one write",
            "commit_ports": 1,
            "factor_response_latency_cycles": 2,
            "weight_response_latency_cycles": 2,
            "acc_response_latency_cycles": 1,
            "backpressure": "none in trace mapping; charged and covered in VCS",
        },
        "cycle_equations": {
            "descriptor": "6 + 3*context_popcount",
            "nonempty_tile": "34 + sum(descriptor_cycles)",
            "empty_tile": 1,
            "dense_zero_factor_policy": "factor, weight, Acc read and Acc write are all charged",
        },
        "aggregate_input": aggregate_input,
        "points": points,
        "speedup": speedup,
        "per_record_speedup_distribution": distributions,
        "per_record": per_record,
        "admission": {
            "exact_m230_aggregate_reconciliation": True,
            "module_lifecycle_cycle_mapping": True,
            "small_width_vcs_bound": True,
            "full_96_lane_rtl": False,
            "full_trace_rtl_replay": False,
            "physical_sram": False,
            "macro_ppa": False,
            "complete_fc1": False,
            "complete_ffn": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": {
            "ratios_are": "same-port serialized 8-lane module-lifecycle cycles over the frozen raw binary-FC1 population",
            "ratios_are_not": [
                "96-lane RTL throughput", "full-trace VCS",
                "physical SRAM or macro-aware PPA", "complete FC1/FFN",
                "system or headline speedup"
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "m262_fc1_descriptor_lifecycle_trace_r1.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / "m262_fc1_descriptor_lifecycle_per_record_r1.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ordinal", "sample_id", "module_index", "input_channels",
                         "dense_cycles", "bit_sparse_cycles",
                         "context_factorized_cycles", "bit_sparse_vs_dense",
                         "context_factorized_vs_dense",
                         "context_factorized_vs_bit_sparse"])
        for row in per_record:
            writer.writerow([
                row["ordinal"], row["sample_id"], row["module_index"],
                row["input_channels"],
                row["points"]["dense"]["lifecycle_cycles"],
                row["points"]["bit_sparse"]["lifecycle_cycles"],
                row["points"]["context_factorized"]["lifecycle_cycles"],
                row["speedup"]["bit_sparse_vs_dense"]["float"],
                row["speedup"]["context_factorized_vs_dense"]["float"],
                row["speedup"]["context_factorized_vs_bit_sparse"]["float"],
            ])
    print("PASS M262 trace bit/dense={:.6f} factor/dense={:.6f} factor/bit={:.6f}".
          format(speedup["bit_sparse_vs_dense"]["float"],
                 speedup["context_factorized_vs_dense"]["float"],
                 speedup["context_factorized_vs_bit_sparse"]["float"]))


if __name__ == "__main__":
    main()
