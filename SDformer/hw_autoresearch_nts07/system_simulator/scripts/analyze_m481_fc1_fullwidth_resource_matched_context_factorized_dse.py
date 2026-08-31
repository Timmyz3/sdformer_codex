#!/usr/bin/env python3
"""M481 exact-mask, full-width analytical FC1 resource-matched DSE.

M230 and M262 are reconciled as separate producer models.  Their ratios are
never multiplied.  Each reported speedup compares bit-sparse and context-
factorized execution at one identical structural resource vector.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import analyze_m224_h67_fc1_parent_delta_bank_service_screen as m224


EXPECTED = {
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "m230_result": "6110dff1cac748ca934e05033ddabe39f06e8b54286699a7843c209ddfe4a6ca",
    "m230_seal": "133c32c37d6ff61d19ca119634b5604d8a9fe12dd510cd4d9425e59e967247e5",
    "m262_result": "9aa24e2ef8889e6e697121817e5e27ca028db81e9e0dee4206fbc34394ec103a",
    "m262_seal": "23f10ed13167d6dc8c6b5c9dbeba42a0777e995becfc7241396746608066b16e",
    "m292_result": "02ce52761729dc842ea27a7419879fece9ba4c9e31c6ba44b4fc5c004da09242",
    "m292_seal": "5910278c824ce8cb9e78a4506df0b01167f8b74324e88c1e7d92fef21df82c2a",
    "contract": "53be88fea5beef747e09e1dff2f10a67508d9f89d26e640b93f51a5276a8d556",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

LANES = (8, 16, 32, 96)
FANOUTS = (1, 2, 4)
CHUNKS = (16, 32, 64)
BANKS = (1, 2, 4)
CONTEXTS = 8
OUTPUT_BLOCK_LANES = 96
ACC_BITS = 19
WEIGHT_BITS = 8
FACTOR_RESPONSE_LATENCY = 2
WEIGHT_RESPONSE_LATENCY = 2
FACTOR_FILL_CYCLES = 1 + FACTOR_RESPONSE_LATENCY
WEIGHT_FILL_CYCLES = 1 + WEIGHT_RESPONSE_LATENCY
ACC_UPDATE_ROUND_CYCLES = 3
HEADER_CYCLES = 1
INIT_CYCLES = CONTEXTS
COMMIT_CYCLES = CONTEXTS * 3
DONE_CYCLES = 1
EMPTY_BYPASS_CYCLES = 1
MAX_INPUT_CHANNELS = 384
POPCOUNT = np.array([bin(value).count("1") for value in range(256)],
                    dtype=np.int64)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def ratio(numerator, denominator):
    require(float(denominator) > 0.0, "zero ratio denominator")
    return float(numerator) / float(denominator)


def packed_context_masks(packed):
    """Return exact raw 8-context masks indexed by group and input source."""
    time, batch, height, width, channel_bytes = packed.shape
    rows = packed.reshape(time * batch * height, width, channel_bytes)
    groups_per_row = ceil_div(width, CONTEXTS)
    padded_width = groups_per_row * CONTEXTS
    if padded_width != width:
        padded = np.zeros((rows.shape[0], padded_width, channel_bytes),
                          dtype=np.uint8)
        padded[:, :width, :] = rows
        rows = padded
    grouped = rows.reshape(rows.shape[0] * groups_per_row,
                           CONTEXTS, channel_bytes)
    sources = channel_bytes * 8
    masks = np.zeros((grouped.shape[0], sources), dtype=np.uint8)
    for context in range(CONTEXTS):
        bits = np.unpackbits(grouped[:, context, :], axis=-1,
                             bitorder="little")
        masks |= bits.astype(np.uint8) << context
    return masks


def exact_record_histogram(record, payload_root):
    packed = m224.decode_record(record, payload_root)
    masks = packed_context_masks(packed)
    output_blocks = ceil_div(int(record["output_shape"][-1]),
                             OUTPUT_BLOCK_LANES)
    histogram = np.bincount(masks.reshape(-1), minlength=256).astype(np.int64)
    histogram *= output_blocks
    nonempty_groups = int(np.count_nonzero(np.any(masks != 0, axis=1)))
    group_streams = int(masks.shape[0]) * output_blocks
    chunk_rows = {}
    for chunk in CHUNKS:
        source_chunks = ceil_div(masks.shape[1], chunk)
        padded_sources = source_chunks * chunk
        if padded_sources != masks.shape[1]:
            padded = np.zeros((masks.shape[0], padded_sources), dtype=np.uint8)
            padded[:, :masks.shape[1]] = masks
            chunk_masks = padded
        else:
            chunk_masks = masks
        active = np.any(chunk_masks.reshape(masks.shape[0], source_chunks,
                                            chunk) != 0, axis=2)
        chunk_rows[str(chunk)] = {
            "source_chunks_per_group": source_chunks,
            "dense_chunk_streams_all_groups": (
                int(masks.shape[0]) * source_chunks * output_blocks),
            "directory_chunk_streams_nonempty_groups": (
                nonempty_groups * source_chunks * output_blocks),
            "active_chunk_streams": int(np.count_nonzero(active)) * output_blocks,
        }
    return {
        "sample_id": int(record["sample_id"]),
        "module_index": int(record["module_index"]),
        "input_channels": int(record["input_shape"][-1]),
        "output_blocks": output_blocks,
        "group_streams": group_streams,
        "nonempty_group_streams": nonempty_groups * output_blocks,
        "empty_group_streams": group_streams - nonempty_groups * output_blocks,
        "mask_histogram": histogram,
        "chunks": chunk_rows,
    }


def mask_issue_rounds(mask, fanout, banks):
    contexts = [index for index in range(CONTEXTS) if mask & (1 << index)]
    require(contexts, "zero mask has no descriptor")
    effective = min(int(fanout), int(banks))
    per_bank = [0] * int(banks)
    for context in contexts:
        per_bank[context % int(banks)] += 1
    return max(max(per_bank), ceil_div(len(contexts), effective))


def aggregate_rounds(histogram, fanout, banks):
    ideal = 0
    banked = 0
    for mask in range(1, 256):
        count = int(histogram[mask])
        if count == 0:
            continue
        ideal += count * ceil_div(int(POPCOUNT[mask]), fanout)
        banked += count * mask_issue_rounds(mask, fanout, banks)
    return ideal, banked


def resource_vector(lanes, fanout, chunk, banks):
    source_offset_bits = int(math.ceil(math.log(chunk, 2)))
    descriptor_entry_bits = source_offset_bits + CONTEXTS + CONTEXTS + 2
    descriptor_entries = CONTEXTS * chunk
    return {
        "lanes": lanes,
        "held_context_fanout": fanout,
        "source_chunk": chunk,
        "accumulator_banks": banks,
        "effective_update_contexts_per_round": min(fanout, banks),
        "allocated_lane_adders": lanes * fanout,
        "accumulator_payload_bits": CONTEXTS * lanes * ACC_BITS,
        "accumulator_read_ports": banks,
        "accumulator_write_ports": banks,
        "accumulator_port_width_bits": lanes * ACC_BITS,
        "factor_request_ports": 1,
        "factor_response_ports": 1,
        "factor_response_latency_cycles": FACTOR_RESPONSE_LATENCY,
        "weight_request_ports": 1,
        "weight_response_ports": 1,
        "weight_response_latency_cycles": WEIGHT_RESPONSE_LATENCY,
        "weight_response_width_bits": lanes * WEIGHT_BITS,
        "held_weight_payload_bits": lanes * WEIGHT_BITS,
        "commit_ports": 1,
        "commit_width_bits": lanes * ACC_BITS,
        "descriptor_buffer_entries_both_modes": descriptor_entries,
        "descriptor_entry_bits_both_modes": descriptor_entry_bits,
        "descriptor_buffer_bits_both_modes": (
            descriptor_entries * descriptor_entry_bits),
        "maximum_chunk_directory_bits_per_tile": ceil_div(MAX_INPUT_CHANNELS,
                                                           chunk),
        "resource_identical_between_modes": True,
    }


def cycle_point(aggregate, lanes, fanout, chunk, banks, mode):
    slices = ceil_div(OUTPUT_BLOCK_LANES, lanes)
    nonempty_tiles = aggregate["nonempty_group_streams"] * slices
    empty_tiles = aggregate["empty_group_streams"] * slices
    if mode == "bit_sparse":
        descriptors = aggregate["source_context_updates"]
        update_rounds = descriptors
        ideal_rounds = descriptors
    elif mode == "context_factorized":
        descriptors = aggregate["unique_source_weight_reads"]
        ideal_rounds, update_rounds = aggregate_rounds(
            aggregate["mask_histogram"], fanout, banks)
    else:
        raise RuntimeError("unsupported mode {}".format(mode))
    directory_chunks = aggregate["chunks"][str(chunk)][
        "directory_chunk_streams_nonempty_groups"] * slices
    components = {
        "empty_bypass": empty_tiles * EMPTY_BYPASS_CYCLES,
        "nonempty_header": nonempty_tiles * HEADER_CYCLES,
        "accumulator_zero_init": nonempty_tiles * INIT_CYCLES,
        "chunk_directory": directory_chunks,
        "factor_fill": descriptors * slices * FACTOR_FILL_CYCLES,
        "weight_fill": descriptors * slices * WEIGHT_FILL_CYCLES,
        "accumulator_update_drain": (
            update_rounds * slices * ACC_UPDATE_ROUND_CYCLES),
        "commit": nonempty_tiles * COMMIT_CYCLES,
        "done": nonempty_tiles * DONE_CYCLES,
    }
    legacy_components = dict(components)
    legacy_components["chunk_directory"] = 0
    return {
        "mode": mode,
        "lane_slices_per_96lane_block": slices,
        "tile_instances": aggregate["group_streams"] * slices,
        "nonempty_tiles": nonempty_tiles,
        "empty_tiles": empty_tiles,
        "descriptor_instances": descriptors * slices,
        "source_context_updates": (
            aggregate["source_context_updates"] * slices),
        "factor_requests": descriptors * slices,
        "weight_requests": descriptors * slices,
        "accumulator_update_requests": (
            aggregate["source_context_updates"] * slices),
        "accumulator_update_issue_rounds": update_rounds * slices,
        "ideal_no_bank_conflict_issue_rounds": ideal_rounds * slices,
        "bank_conflict_extra_issue_rounds": (
            update_rounds - ideal_rounds) * slices,
        "chunk_active_streams": aggregate["chunks"][str(chunk)][
            "active_chunk_streams"] * slices,
        "chunk_directory_streams": directory_chunks,
        "cycle_components": components,
        "lifecycle_cycles": sum(components.values()),
        "legacy_no_chunk_directory_cycles": sum(legacy_components.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m230-result", required=True, type=Path)
    parser.add_argument("--m230-seal", required=True, type=Path)
    parser.add_argument("--m262-result", required=True, type=Path)
    parser.add_argument("--m262-seal", required=True, type=Path)
    parser.add_argument("--m292-result", required=True, type=Path)
    parser.add_argument("--m292-seal", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    identity_paths = {
        "manifest": args.manifest,
        "m230_result": args.m230_result,
        "m230_seal": args.m230_seal,
        "m262_result": args.m262_result,
        "m262_seal": args.m262_seal,
        "m292_result": args.m292_result,
        "m292_seal": args.m292_seal,
        "contract": args.contract,
        "docs359": args.docs359,
    }
    identity = dict((key, sha256(path)) for key, path in identity_paths.items())
    require(identity == EXPECTED, "frozen input identity drift")
    contract = json.loads(args.contract.read_text())
    require(contract["scan"]["points"] ==
            len(LANES) * len(FANOUTS) * len(CHUNKS) * len(BANKS),
            "contract scan size drift")

    manifest = json.loads(args.manifest.read_text())
    m230 = json.loads(args.m230_result.read_text())
    m262 = json.loads(args.m262_result.read_text())
    m292 = json.loads(args.m292_result.read_text())
    records = [row for row in manifest["records"]
               if int(row["module_index"]) in m224.SELECTED_MODULE_INDICES]
    require(len(records) == 100, "expected 100 binary FC1 records")

    aggregate = {
        "group_streams": 0,
        "nonempty_group_streams": 0,
        "empty_group_streams": 0,
        "mask_histogram": np.zeros(256, dtype=np.int64),
        "chunks": dict((str(chunk), {
            "dense_chunk_streams_all_groups": 0,
            "directory_chunk_streams_nonempty_groups": 0,
            "active_chunk_streams": 0,
        }) for chunk in CHUNKS),
    }
    per_record_summary = []
    for record in sorted(records, key=lambda row: (
            int(row["sample_id"]), int(row["module_index"]))):
        row = exact_record_histogram(record, args.payload_root)
        for key in ("group_streams", "nonempty_group_streams",
                    "empty_group_streams"):
            aggregate[key] += int(row[key])
        aggregate["mask_histogram"] += row["mask_histogram"]
        for chunk in CHUNKS:
            for key in ("dense_chunk_streams_all_groups",
                        "directory_chunk_streams_nonempty_groups",
                        "active_chunk_streams"):
                aggregate["chunks"][str(chunk)][key] += int(
                    row["chunks"][str(chunk)][key])
        per_record_summary.append({
            "sample_id": row["sample_id"],
            "module_index": row["module_index"],
            "input_channels": row["input_channels"],
            "output_blocks": row["output_blocks"],
            "group_streams": row["group_streams"],
            "nonempty_group_streams": row["nonempty_group_streams"],
            "unique_source_weight_reads": int(row["mask_histogram"][1:].sum()),
            "source_context_updates": int(np.dot(row["mask_histogram"],
                                                   POPCOUNT)),
        })

    histogram = aggregate["mask_histogram"]
    aggregate["unique_source_weight_reads"] = int(histogram[1:].sum())
    aggregate["source_context_updates"] = int(np.dot(histogram, POPCOUNT))
    raw = m230["aggregate_trace"]["raw"]
    reconciliations = {
        "group_streams": aggregate["group_streams"] == int(raw["group_streams"]),
        "nonempty_group_streams": (
            aggregate["nonempty_group_streams"] ==
            int(raw["nonempty_group_streams"])),
        "empty_group_streams": (
            aggregate["empty_group_streams"] == int(raw["empty_group_streams"])),
        "unique_source_weight_reads": (
            aggregate["unique_source_weight_reads"] ==
            int(raw["unique_source_weight_reads"])),
        "source_context_updates": (
            aggregate["source_context_updates"] ==
            int(raw["service_cycles"]["1"])),
        "chunk32_active": (
            aggregate["chunks"]["32"]["active_chunk_streams"] ==
            int(raw["active_32source_chunk_streams"])),
        "chunk32_dense": (
            aggregate["chunks"]["32"]["dense_chunk_streams_all_groups"] ==
            int(raw["dense_32source_chunk_streams"])),
    }
    for fanout in FANOUTS:
        ideal, _ = aggregate_rounds(histogram, fanout, fanout)
        reconciliations["m230_ideal_f{}_service".format(fanout)] = (
            ideal == int(raw["service_cycles"][str(fanout)]))
    require(all(reconciliations.values()), "M230 histogram reconciliation drift")

    legacy_bit = cycle_point(aggregate, 8, 1, 32, 1, "bit_sparse")
    legacy_factor = cycle_point(aggregate, 8, 1, 32, 1,
                                "context_factorized")
    reconciliations["m262_bit_sparse_legacy_cycles"] = (
        legacy_bit["legacy_no_chunk_directory_cycles"] ==
        int(m262["points"]["bit_sparse"]["lifecycle_cycles"]))
    reconciliations["m262_factorized_legacy_cycles"] = (
        legacy_factor["legacy_no_chunk_directory_cycles"] ==
        int(m262["points"]["context_factorized"]["lifecycle_cycles"]))

    scope = m292["scope_partition"]
    envelope = int(scope["compute_envelope_cycles"])
    eligible = int(scope["eligible_binary_fc1_cycles"])
    fallback = int(scope["excluded_stage3_nonbinary_fc1_cycles"])
    all_fc1 = int(scope["all_fc1_cycles"])
    reconciliations["m292_scope_partition"] = (
        envelope == 620302905 and eligible == 100895624 and
        fallback == 17474490 and all_fc1 == 118370114 and
        eligible + fallback == all_fc1)
    require(all(reconciliations.values()), "M262/M292 reconciliation drift")

    points = []
    for lanes in LANES:
        for fanout in FANOUTS:
            for chunk in CHUNKS:
                for banks in BANKS:
                    baseline = cycle_point(aggregate, lanes, fanout, chunk,
                                           banks, "bit_sparse")
                    candidate = cycle_point(aggregate, lanes, fanout, chunk,
                                            banks, "context_factorized")
                    matched_speedup = ratio(baseline["lifecycle_cycles"],
                                            candidate["lifecycle_cycles"])
                    projected_eligible = float(eligible) / matched_speedup
                    projected_all_fc1 = projected_eligible + fallback
                    ideal_envelope = ratio(
                        envelope,
                        float(envelope - eligible) + projected_eligible)
                    weight_reduction = ratio(baseline["weight_requests"],
                                             candidate["weight_requests"])
                    resource = resource_vector(lanes, fanout, chunk, banks)
                    gate = {
                        "same_resource_module_speedup_ge_1p50": (
                            matched_speedup >= 1.5),
                        "scope_corrected_ideal_envelope_ge_1p08": (
                            ideal_envelope >= 1.08),
                        "weight_requests_nonincreasing": (
                            candidate["weight_requests"] <=
                            baseline["weight_requests"]),
                        "stage3_fallback_charged": True,
                    }
                    gate["numerical_opportunity_gate_pass"] = all(gate.values())
                    points.append({
                        "point_id": "L{}_F{}_C{}_B{}".format(
                            lanes, fanout, chunk, banks),
                        "resource": resource,
                        "baseline": baseline,
                        "candidate": candidate,
                        "same_resource_speedup": matched_speedup,
                        "weight_request_reduction": weight_reduction,
                        "scope_corrected_projection": {
                            "eligible_binary_fc1_baseline_cycles": eligible,
                            "eligible_binary_fc1_projected_cycles": projected_eligible,
                            "stage3_fallback_cycles_unchanged": fallback,
                            "all_fc1_projected_cycles": projected_all_fc1,
                            "compute_envelope_cycles": envelope,
                            "ideal_envelope_sensitivity_not_speedup": ideal_envelope,
                            "linear_trace_ratio_projection_assumption": True,
                        },
                        "gate": gate,
                        "admission": {
                            "measured_hardware_cycles": False,
                            "system_speedup": False,
                            "headline": False,
                        },
                    })

    require(len(points) == 108, "DSE point count drift")
    fullwidth = [row for row in points if row["resource"]["lanes"] == 96]
    best_fullwidth = min(fullwidth,
                         key=lambda row: row["candidate"]["lifecycle_cycles"])
    best_speed = max(fullwidth, key=lambda row: row["same_resource_speedup"])
    gate_points = [row for row in points
                   if row["gate"]["numerical_opportunity_gate_pass"]]
    fullwidth_gate_points = [row for row in fullwidth
                             if row["gate"]["numerical_opportunity_gate_pass"]]
    compact_fullwidth_gate = min(
        fullwidth_gate_points,
        key=lambda row: (
            row["resource"]["allocated_lane_adders"],
            row["resource"]["accumulator_read_ports"],
            row["resource"]["descriptor_buffer_bits_both_modes"],
            row["candidate"]["lifecycle_cycles"]))
    undominated_fullwidth = []
    for row in fullwidth:
        vector = (
            row["candidate"]["lifecycle_cycles"],
            row["resource"]["allocated_lane_adders"],
            row["resource"]["accumulator_read_ports"],
            row["resource"]["descriptor_buffer_bits_both_modes"])
        dominated = False
        for other in fullwidth:
            other_vector = (
                other["candidate"]["lifecycle_cycles"],
                other["resource"]["allocated_lane_adders"],
                other["resource"]["accumulator_read_ports"],
                other["resource"]["descriptor_buffer_bits_both_modes"])
            if (all(left <= right for left, right in
                    zip(other_vector, vector)) and
                    any(left < right for left, right in
                        zip(other_vector, vector))):
                dominated = True
                break
        if not dominated:
            undominated_fullwidth.append(row["point_id"])

    histogram_json = dict((str(mask), int(histogram[mask]))
                          for mask in range(1, 256)
                          if int(histogram[mask]) != 0)
    output = {
        "schema": "m481_fc1_fullwidth_resource_matched_context_factorized_dse_v2",
        "status": "PASS_EXACT_MASK_108_POINT_CPU_DSE_NO_PERFORMANCE_ADMISSION",
        "identity": identity,
        "population": {
            "records": 100,
            "samples": 10,
            "binary_fc1_modules": 10,
            "stage3_nonbinary_fc1_modules": 2,
            "stage3_policy": "conventional fallback",
            "payload_sha_verified_records": 100,
        },
        "cycle_model": contract["cycle_model"],
        "same_resource_rule": contract["same_resource_rule"],
        "aggregate_trace": {
            "group_streams": aggregate["group_streams"],
            "nonempty_group_streams": aggregate["nonempty_group_streams"],
            "empty_group_streams": aggregate["empty_group_streams"],
            "unique_source_weight_reads": aggregate["unique_source_weight_reads"],
            "source_context_updates": aggregate["source_context_updates"],
            "context_mask_histogram": histogram_json,
            "chunks": aggregate["chunks"],
        },
        "reconciliations": reconciliations,
        "scope_partition": {
            "compute_envelope_cycles": envelope,
            "eligible_binary_fc1_cycles": eligible,
            "excluded_stage3_nonbinary_fc1_cycles": fallback,
            "all_fc1_cycles": all_fc1,
            "partition_conserves": eligible + fallback == all_fc1,
        },
        "points": points,
        "decision": {
            "numerical_gate_point_count": len(gate_points),
            "best_fullwidth_absolute_candidate_point": best_fullwidth["point_id"],
            "best_fullwidth_absolute_candidate_cycles": (
                best_fullwidth["candidate"]["lifecycle_cycles"]),
            "best_fullwidth_same_resource_speedup_point": best_speed["point_id"],
            "best_fullwidth_same_resource_speedup": (
                best_speed["same_resource_speedup"]),
            "best_fullwidth_ideal_envelope_sensitivity_not_speedup": (
                best_speed["scope_corrected_projection"]
                ["ideal_envelope_sensitivity_not_speedup"]),
            "best_fullwidth_weight_request_reduction": (
                best_speed["weight_request_reduction"]),
            "compact_fullwidth_gate_point": compact_fullwidth_gate["point_id"],
            "compact_fullwidth_gate_same_resource_speedup": (
                compact_fullwidth_gate["same_resource_speedup"]),
            "compact_fullwidth_gate_ideal_envelope_sensitivity_not_speedup": (
                compact_fullwidth_gate["scope_corrected_projection"]
                ["ideal_envelope_sensitivity_not_speedup"]),
            "compact_fullwidth_gate_resource_vector": (
                compact_fullwidth_gate["resource"]),
            "undominated_fullwidth_point_ids": undominated_fullwidth,
            "undominated_definition": "Pareto over candidate cycles, allocated lane adders, accumulator read/write bank count and descriptor-buffer bits; no weighted area proxy",
            "m230_m262_ratios_multiplied": False,
            "rtl_promotion": False,
            "next_gate": contract["decision_gate"]["next_gate_if_pass"],
        },
        "per_record_summary": per_record_summary,
        "admission": {
            "trace_exact_context_mask_histogram": True,
            "fullwidth_analytical_dse": True,
            "same_resource_within_each_point": True,
            "fullwidth_rtl": False,
            "full_trace_vcs": False,
            "physical_sram": False,
            "macro_ppa": False,
            "measured_cycle_speedup": False,
            "complete_fc1": False,
            "complete_ffn": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": {
            "ratios_are": "clean-latency analytical lifecycle ratios at identical per-point structural resources",
            "ratios_are_not": [
                "M230 multiplied by M262", "full-width RTL throughput",
                "stalled physical SRAM timing", "complete FC1 or FFN speedup",
                "measured system speedup", "paper PPA or headline result"
            ],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / \
        "m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / \
        "m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "point_id", "lanes", "fanout", "chunk", "acc_banks",
            "effective_update_contexts", "descriptor_buffer_bits",
            "baseline_cycles", "candidate_cycles", "same_resource_speedup",
            "candidate_bank_conflict_rounds", "baseline_weight_requests",
            "candidate_weight_requests", "weight_request_reduction",
            "projected_eligible_fc1_cycles", "stage3_fallback_cycles",
            "projected_all_fc1_cycles", "ideal_envelope_sensitivity_not_speedup",
            "numerical_gate_pass", "system_speedup", "headline"
        ])
        for row in points:
            rv = row["resource"]
            projection = row["scope_corrected_projection"]
            writer.writerow([
                row["point_id"], rv["lanes"], rv["held_context_fanout"],
                rv["source_chunk"], rv["accumulator_banks"],
                rv["effective_update_contexts_per_round"],
                rv["descriptor_buffer_bits_both_modes"],
                row["baseline"]["lifecycle_cycles"],
                row["candidate"]["lifecycle_cycles"],
                row["same_resource_speedup"],
                row["candidate"]["bank_conflict_extra_issue_rounds"],
                row["baseline"]["weight_requests"],
                row["candidate"]["weight_requests"],
                row["weight_request_reduction"],
                projection["eligible_binary_fc1_projected_cycles"],
                projection["stage3_fallback_cycles_unchanged"],
                projection["all_fc1_projected_cycles"],
                projection["ideal_envelope_sensitivity_not_speedup"],
                row["gate"]["numerical_opportunity_gate_pass"], False, False
            ])
    readme = args.output_dir / "README.md"
    readme.write_text(
        "# M481 FC1 full-width resource-matched context-factorized DSE\n\n"
        "Re-decoded 100 frozen H67 ep35 raw-binary FC1 records, reconciled "
        "M230/M262 independently, and evaluated 108 clean-latency analytical "
        "points.  No M230 and M262 ratios were multiplied.\n\n"
        "Best full-width same-resource point: `{}` at {:.6f}x module-lifecycle "
        "ratio and {:.6f}x scope-corrected ideal envelope sensitivity.  "
        "The compact full-width gate point is `{}` at {:.6f}x.  "
        "This is not measured hardware, complete FC1/FFN, system speedup, "
        "physical SRAM, PPA, or a headline result.\n".format(
            best_speed["point_id"], best_speed["same_resource_speedup"],
            best_speed["scope_corrected_projection"]
            ["ideal_envelope_sensitivity_not_speedup"],
            compact_fullwidth_gate["point_id"],
            compact_fullwidth_gate["same_resource_speedup"]))
    print("PASS M481 points={} best={} module={:.6f} envelope={:.6f} gates={}".
          format(len(points), best_speed["point_id"],
                 best_speed["same_resource_speedup"],
                 best_speed["scope_corrected_projection"]
                 ["ideal_envelope_sensitivity_not_speedup"],
                 len(gate_points)))


if __name__ == "__main__":
    main()
