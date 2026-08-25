#!/usr/bin/env python3
"""M225 exact held-weight K-context multicast screen for binary FC1.

One 96-channel INT8 weight vector is read at a time.  The vector is broadcast
to at most F of K row-adjacent contexts; if more contexts need the same source,
the weight is held and replayed without another SRAM read.  Parent-delta work
and multicast parallelism are reported independently and composed only through
the explicit serialized recurrence.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_m224_h67_fc1_parent_delta_bank_service_screen as m224


EXPECTED_M224_SHA256SUMS = (
    "5331eee02054be4da880767d3d0c9c9108499d2d63c8a8a966ff01db40deb7a3"
)
EXPECTED_M224_REVIEW_SHA256SUMS = (
    "a283d10539c35397568a82298620ae816de85df04830f7fc348be10582a5125b"
)
CONTEXT_POINTS = {
    1: (1,),
    2: (1, 2),
    4: (1, 2, 4),
    8: (1, 2, 4, 8),
}
ACCUMULATOR_BITS = 19
WEIGHT_VECTOR_CHANNELS = 96
WEIGHT_BITS = 8
WEIGHT_VECTOR_BITS = WEIGHT_VECTOR_CHANNELS * WEIGHT_BITS


def context_occupancy(residual, context_group):
    time, batch, height, width, channel_bytes = residual.shape
    rows = residual.reshape(time * batch * height, width, channel_bytes)
    groups_per_row = m224.ceil_div(width, context_group)
    padded_width = groups_per_row * context_group
    if padded_width != width:
        padded = np.zeros((rows.shape[0], padded_width, channel_bytes),
                          dtype=np.uint8)
        padded[:, :width, :] = rows
        rows = padded
    grouped = rows.reshape(rows.shape[0], groups_per_row, context_group,
                           channel_bytes)
    service_units = dict((fanout, 0)
                         for fanout in CONTEXT_POINTS[context_group])
    union_sources = 0
    source_occurrences = 0
    maximum_context_count = 0
    context_count_histogram = [0] * (context_group + 1)
    for bit in range(8):
        occurrence = np.bitwise_and(np.right_shift(grouped, bit), 1).sum(
            axis=2, dtype=np.uint8)
        union_sources += int(np.count_nonzero(occurrence))
        source_occurrences += int(occurrence.sum(dtype=np.uint64))
        maximum_context_count = max(
            maximum_context_count, int(occurrence.max(initial=0)))
        histogram = np.bincount(occurrence.reshape(-1),
                                minlength=context_group + 1)
        for count in range(context_group + 1):
            context_count_histogram[count] += int(histogram[count])
        occurrence16 = occurrence.astype(np.uint16)
        for fanout in CONTEXT_POINTS[context_group]:
            service_units[fanout] += int(
                ((occurrence16 + fanout - 1) // fanout).sum(dtype=np.uint64))
    return {
        "row_bounded_groups": int(rows.shape[0] * groups_per_row),
        "union_sources": union_sources,
        "source_occurrences": source_occurrences,
        "maximum_context_count_per_source": maximum_context_count,
        "context_count_histogram_including_zero": context_count_histogram,
        "service_units": service_units,
    }


def fixed_cycles(record, choice_stats, mode):
    overhead = m224.fixed_overheads(
        record["input_shape"], record["output_shape"], choice_stats, mode)
    input_channels = int(record["input_shape"][-1])
    output_channels = int(record["output_shape"][-1])
    weight_bytes = input_channels * output_channels
    weight_dma = m224.ceil_div(weight_bytes, 32)
    charged = (
        overhead["current_activation_scan_cycles"] +
        overhead["candidate_parent_scan_cycles"] +
        overhead["choice_metadata_cycles"] +
        overhead["chosen_parent_seed_cycles"] +
        overhead["final_commit_cycles"] + weight_dma
    )
    return overhead, weight_dma, charged


def build_point(record, choice_stats, mode, occupancy, context_group,
                fanout):
    overhead, weight_dma, fixed = fixed_cycles(record, choice_stats, mode)
    output_channels = int(record["output_shape"][-1])
    output_blocks = m224.ceil_div(output_channels, WEIGHT_VECTOR_CHANNELS)
    descriptor_cycles = occupancy["row_bounded_groups"] * output_blocks
    service_cycles = occupancy["service_units"][fanout] * output_blocks
    weight_reads = occupancy["union_sources"] * output_blocks
    source_occurrences = occupancy["source_occurrences"]
    m224.require(source_occurrences == choice_stats["source_events"],
                 "context source conservation")
    product_updates = source_occurrences * output_channels
    physical_slots = service_cycles * fanout * WEIGHT_VECTOR_CHANNELS
    serial = fixed + descriptor_cycles + service_cycles
    return {
        "mode": mode,
        "context_group_k": context_group,
        "context_fanout_f": fanout,
        "product_lanes": fanout * WEIGHT_VECTOR_CHANNELS,
        "resident_accumulator_contexts": context_group,
        "minimum_accumulator_state_bits": (
            context_group * WEIGHT_VECTOR_CHANNELS * ACCUMULATOR_BITS),
        "weight_read_width_bits": WEIGHT_VECTOR_BITS,
        "weight_read_width_constant": True,
        "weight_vector_reads": weight_reads,
        "weight_read_bits": weight_reads * WEIGHT_VECTOR_BITS,
        "held_replay_cycles_without_weight_reread":
            service_cycles - weight_reads,
        "source_occurrences": source_occurrences,
        "unique_group_sources": occupancy["union_sources"],
        "maximum_context_count_per_source":
            occupancy["maximum_context_count_per_source"],
        "context_count_histogram_including_zero":
            occupancy["context_count_histogram_including_zero"],
        "product_updates": product_updates,
        "physical_product_slots": physical_slots,
        "product_slot_utilization": (float(product_updates) /
                                     float(physical_slots)
                                     if physical_slots else 1.0),
        "weight_dma_256b_cycles": weight_dma,
        "group_descriptor_cycles": descriptor_cycles,
        "service_cycles": service_cycles,
        "serial_cycles": serial,
        "overheads": overhead,
    }


def add_point(total, point):
    summed = (
        "weight_vector_reads", "weight_read_bits",
        "held_replay_cycles_without_weight_reread", "source_occurrences",
        "unique_group_sources", "product_updates", "physical_product_slots",
        "weight_dma_256b_cycles", "group_descriptor_cycles",
        "service_cycles", "serial_cycles",
    )
    for key in summed:
        total[key] = total.get(key, 0) + int(point[key])
    total["maximum_context_count_per_source"] = max(
        total.get("maximum_context_count_per_source", 0),
        int(point["maximum_context_count_per_source"]))
    histogram = total.setdefault(
        "context_count_histogram_including_zero",
        [0] * len(point["context_count_histogram_including_zero"]))
    m224.require(len(histogram) ==
                 len(point["context_count_histogram_including_zero"]),
                 "histogram geometry drift")
    for index, value in enumerate(
            point["context_count_histogram_including_zero"]):
        histogram[index] += int(value)
    overhead = total.setdefault("overheads", {})
    for key, value in point["overheads"].items():
        if key in ("input_channels", "output_channels",
                   "input_256b_beats_per_vector",
                   "output_96lane_blocks_per_vector"):
            continue
        overhead[key] = overhead.get(key, 0) + int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m224-sha256sums", required=True, type=Path)
    parser.add_argument("--m224-review-sha256sums", required=True, type=Path)
    parser.add_argument("--m224-result", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    m224.require(m224.sha256(args.manifest) ==
                 m224.EXPECTED_MANIFEST_SHA256, "manifest drift")
    m224.require(m224.sha256(args.m224_sha256sums) ==
                 EXPECTED_M224_SHA256SUMS, "M224 seal drift")
    m224.require(m224.sha256(args.m224_review_sha256sums) ==
                 EXPECTED_M224_REVIEW_SHA256SUMS, "M224 review seal drift")
    m224.require(m224.sha256(args.docs359) ==
                 m224.EXPECTED_DOCS359_SHA256, "docs/359 drift")
    m224_result = json.loads(args.m224_result.read_text())
    m224.require(m224_result["admission"]["exact_trace_screen"] and
                 not m224_result["admission"]["advance_to_rtl"],
                 "M224 boundary drift")

    manifest = json.loads(args.manifest.read_text())
    records = [row for row in manifest["records"]
               if int(row["module_index"]) in m224.SELECTED_MODULE_INDICES]
    m224.require(len(records) == 100, "expected 100 FC1 records")
    aggregate = {}
    sample_totals = {}
    per_record = []
    for ordinal, record in enumerate(sorted(
            records, key=lambda row: (int(row["sample_id"]),
                                      int(row["module_index"])))):
        current = m224.decode_record(record, args.payload_root)
        record_result = {
            "ordinal": ordinal,
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "name": record["name"],
            "modes": {},
        }
        for mode in m224.MODE_ORDER:
            residual, choice_stats = m224.select_parent(current, mode)
            record_result["modes"][mode] = {
                "choice": choice_stats,
                "points": {},
            }
            for context_group, fanouts in CONTEXT_POINTS.items():
                occupancy = context_occupancy(residual, context_group)
                for fanout in fanouts:
                    point = build_point(record, choice_stats, mode, occupancy,
                                        context_group, fanout)
                    name = "{}_K{}_F{}".format(mode, context_group, fanout)
                    record_result["modes"][mode]["points"][name] = point
                    template = aggregate.setdefault(name, {
                        "mode": mode,
                        "context_group_k": context_group,
                        "context_fanout_f": fanout,
                        "product_lanes": fanout * WEIGHT_VECTOR_CHANNELS,
                        "resident_accumulator_contexts": context_group,
                        "minimum_accumulator_state_bits": (
                            context_group * WEIGHT_VECTOR_CHANNELS *
                            ACCUMULATOR_BITS),
                        "weight_read_width_bits": WEIGHT_VECTOR_BITS,
                        "weight_read_width_constant": True,
                    })
                    add_point(template, point)
                    sample_key = str(record["sample_id"])
                    sample = sample_totals.setdefault(sample_key, {})
                    sample[name] = sample.get(name, 0) + int(
                        point["serial_cycles"])
            del residual
        per_record.append(record_result)

    reference_name = "raw_K1_F1"
    reference = aggregate[reference_name]
    for name, point in aggregate.items():
        mode_reference = aggregate["{}_K1_F1".format(point["mode"])]
        point["parent_delta_ratio_at_k1_f1"] = m224.ratio(
            reference["serial_cycles"], mode_reference["serial_cycles"])
        point["context_multicast_ratio_vs_same_parent_k1_f1"] = m224.ratio(
            mode_reference["serial_cycles"], point["serial_cycles"])
        point["composed_ratio_vs_raw_k1_f1"] = m224.ratio(
            reference["serial_cycles"], point["serial_cycles"])
        point["weight_read_reduction_vs_same_parent_k1_f1"] = m224.ratio(
            mode_reference["weight_vector_reads"],
            point["weight_vector_reads"])
        point["product_slot_utilization"] = (
            float(point["product_updates"]) /
            float(point["physical_product_slots"])
            if point["physical_product_slots"] else 1.0)
        sample_ratios = []
        for sample in sorted(sample_totals, key=int):
            sample_ratios.append(
                float(sample_totals[sample][reference_name]) /
                float(sample_totals[sample][name]))
        point["per_sample_composed_ratio"] = {
            "count": len(sample_ratios),
            "min": min(sample_ratios),
            "mean": sum(sample_ratios) / len(sample_ratios),
            "max": max(sample_ratios),
        }

    best_by_fanout = {}
    for fanout in (1, 2, 4, 8):
        candidates = [row for row in aggregate.values()
                      if row["context_fanout_f"] == fanout]
        best_by_fanout[str(fanout)] = max(
            candidates,
            key=lambda row: row["composed_ratio_vs_raw_k1_f1"]["float"])
    advance_f2 = (best_by_fanout["2"]
                  ["composed_ratio_vs_raw_k1_f1"]["float"] >= 1.5)
    advance_f4 = (best_by_fanout["4"]
                  ["composed_ratio_vs_raw_k1_f1"]["float"] >= 2.0)

    output = {
        "schema": "m225_h67_fc1_held_weight_context_multicast_screen_v1",
        "status": ("PASS_ADVANCE_F2_F4_TO_MATCHED_RTL_DC"
                   if advance_f2 and advance_f4 else
                   "PASS_TRACE_SCREEN_WITHOUT_DUAL_ADVANCE"),
        "identity": {
            "manifest_sha256": m224.sha256(args.manifest),
            "m224_sha256sums_sha256": m224.sha256(args.m224_sha256sums),
            "m224_review_sha256sums_sha256":
                m224.sha256(args.m224_review_sha256sums),
            "m224_result_sha256": m224.sha256(args.m224_result),
            "docs359_sha256": m224.sha256(args.docs359),
        },
        "population": {
            "records": len(records),
            "samples": len(set(int(row["sample_id"]) for row in records)),
            "modules": len(m224.SELECTED_MODULE_INDICES),
            "stage3_conventional_nonbinary": m224_result["population"]
                                                  ["stage3_conventional_nonbinary"],
        },
        "resource_contract": {
            "weight_read_width_bits_all_points": WEIGHT_VECTOR_BITS,
            "weight_read_width_constant": True,
            "one_weight_vector_read_per_unique_group_source": True,
            "held_replay_has_no_weight_reread": True,
            "product_lanes": "96*F",
            "minimum_accumulator_state_bits": "K*96*19",
            "extra_logic_requires_matched_DC": True,
        },
        "reference": reference_name,
        "aggregate_points": aggregate,
        "best_by_fanout": best_by_fanout,
        "sample_serial_cycles": sample_totals,
        "per_record": per_record,
        "admission": {
            "trace_screen": True,
            "advance_f2_to_matched_rtl_dc": advance_f2,
            "advance_f4_to_matched_rtl_dc": advance_f4,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "macro_complete": False,
            "complete_fc1": False,
            "complete_ffn": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": {
            "ratios_are": "M51-s10 exact trace serialized premodel ratios",
            "ratios_are_not": [
                "measured RTL throughput",
                "area efficiency",
                "complete FC1/FFN speedup",
                "system speedup",
                "macro-aware energy or PPA"
            ],
            "speedup_per_product_lane_not_called_area_efficiency": True,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / \
        "m225_h67_fc1_held_weight_context_multicast_screen_r1.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / \
        "m225_h67_fc1_held_weight_context_multicast_points_r1.csv"
    fields = [
        "mode", "K", "F", "product_lanes", "accumulator_state_bits",
        "weight_read_width_bits", "weight_vector_reads",
        "held_replay_cycles", "product_slot_utilization",
        "service_cycles", "serial_cycles", "parent_delta_ratio",
        "context_multicast_ratio", "composed_ratio", "weight_read_reduction",
        "sample_min", "sample_mean", "sample_max",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in sorted(aggregate.values(), key=lambda row: (
                row["mode"], row["context_group_k"],
                row["context_fanout_f"])):
            dist = point["per_sample_composed_ratio"]
            writer.writerow({
                "mode": point["mode"],
                "K": point["context_group_k"],
                "F": point["context_fanout_f"],
                "product_lanes": point["product_lanes"],
                "accumulator_state_bits":
                    point["minimum_accumulator_state_bits"],
                "weight_read_width_bits": point["weight_read_width_bits"],
                "weight_vector_reads": point["weight_vector_reads"],
                "held_replay_cycles":
                    point["held_replay_cycles_without_weight_reread"],
                "product_slot_utilization":
                    point["product_slot_utilization"],
                "service_cycles": point["service_cycles"],
                "serial_cycles": point["serial_cycles"],
                "parent_delta_ratio":
                    point["parent_delta_ratio_at_k1_f1"]["float"],
                "context_multicast_ratio": point[
                    "context_multicast_ratio_vs_same_parent_k1_f1"]["float"],
                "composed_ratio":
                    point["composed_ratio_vs_raw_k1_f1"]["float"],
                "weight_read_reduction": point[
                    "weight_read_reduction_vs_same_parent_k1_f1"]["float"],
                "sample_min": dist["min"],
                "sample_mean": dist["mean"],
                "sample_max": dist["max"],
            })
    print("PASS M225 F2={:.6f} F4={:.6f} F8={:.6f} advance={}/{}".
          format(best_by_fanout["2"]["composed_ratio_vs_raw_k1_f1"]
                 ["float"],
                 best_by_fanout["4"]["composed_ratio_vs_raw_k1_f1"]
                 ["float"],
                 best_by_fanout["8"]["composed_ratio_vs_raw_k1_f1"]
                 ["float"], advance_f2, advance_f4))


if __name__ == "__main__":
    main()
