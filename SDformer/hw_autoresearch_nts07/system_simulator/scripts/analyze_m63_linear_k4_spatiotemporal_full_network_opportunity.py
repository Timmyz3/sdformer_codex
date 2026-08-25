#!/usr/bin/env python3
"""M63 all-24 Linear spatial/temporal K1/K2/K4 opportunity model.

The packed-bit implementation is independent of the M55 parent analyzer.  It
uses M55 only as a fail-closed reconciliation target.  Cycles are admitted
only for the explicit conservative model frozen in the M63 contract.
"""

from __future__ import print_function

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


PARENTS = ("zero", "left", "up", "previous_timestep")
MODES = ("spatial", "temporal")
K_POINTS = (1, 2, 4)
POPCOUNT = np.array([bin(value).count("1") for value in range(256)],
                    dtype=np.uint8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def fraction(value):
    require(value.denominator > 0, "fraction denominator")
    return {"numerator": int(value.numerator),
            "denominator": int(value.denominator),
            "float": float(value)}


def nearest_rank(values, percentile):
    require(values, "empty percentile population")
    ordered = sorted(int(value) for value in values)
    rank = int(math.ceil(float(percentile) * len(ordered)))
    return ordered[max(0, rank - 1)]


def distribution(values):
    require(values, "empty distribution")
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50_nearest_rank": nearest_rank(ordered, 0.50),
        "p95_nearest_rank": nearest_rank(ordered, 0.95),
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def packed_array(path, record):
    shape = [int(value) for value in record["input_shape"]]
    require(record["operator"] == "Linear" and len(shape) == 5,
            "Linear shape/operator mismatch")
    channels = shape[-1]
    require(channels % 8 == 0, "M63 requires byte-aligned channels")
    expected = int(record["input_elements"]) // 8
    require(int(record["input_elements"]) % 8 == 0 and
            int(record["packed_bytes"]) == expected,
            "packed geometry mismatch")
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(int(packed.size) == expected, "packed byte count mismatch")
    current = packed.reshape(shape[:-1] + [channels // 8])
    require(int(POPCOUNT[current].sum(dtype=np.int64)) ==
            int(record["active_elements"]), "packed popcount mismatch")
    return current


def apply_candidate(best, choice, residual, candidate, candidate_id,
                    target_slice):
    target_best = best[target_slice]
    target_choice = choice[target_slice]
    target_residual = residual[target_slice]
    cost = POPCOUNT[candidate].sum(axis=-1, dtype=np.int32)
    take = cost < target_best
    target_best[take] = cost[take]
    target_choice[take] = candidate_id
    target_residual[take] = candidate[take]


def select_residual(current, mode):
    best = POPCOUNT[current].sum(axis=-1, dtype=np.int32)
    choice = np.zeros(best.shape, dtype=np.uint8)
    residual = current.copy()
    if mode == "spatial":
        left = np.bitwise_xor(current[:, :, :, 1:, :],
                              current[:, :, :, :-1, :])
        apply_candidate(best, choice, residual, left, 1,
                        (slice(None), slice(None), slice(None),
                         slice(1, None)))
        del left
        up = np.bitwise_xor(current[:, :, 1:, :, :],
                            current[:, :, :-1, :, :])
        apply_candidate(best, choice, residual, up, 2,
                        (slice(None), slice(None), slice(1, None),
                         slice(None)))
        del up
    elif mode == "temporal":
        previous = np.bitwise_xor(current[1:, :, :, :, :],
                                  current[:-1, :, :, :, :])
        apply_candidate(best, choice, residual, previous, 3,
                        (slice(1, None), slice(None), slice(None),
                         slice(None)))
        del previous
    else:
        raise ValueError("unknown mode {}".format(mode))
    source_bits = int(POPCOUNT[residual].sum(dtype=np.int64))
    positive = int(POPCOUNT[np.bitwise_and(current, residual)].sum(
        dtype=np.int64))
    negative = int(POPCOUNT[np.bitwise_and(
        np.bitwise_not(current), residual)].sum(dtype=np.int64))
    require(positive + negative == source_bits,
            "signed source conservation")
    counts = np.bincount(choice.reshape(-1), minlength=4)
    return residual, choice, source_bits, positive, negative, dict(
        (PARENTS[index], int(counts[index])) for index in range(4))


def union_bank_metrics(residual, fanout_k, output_blocks):
    time, batch, height, width, channel_bytes = residual.shape
    rows = time * batch * height
    groups_per_row = ceil_div(width, fanout_k)
    padded_width = groups_per_row * fanout_k
    view = residual.reshape(rows, width, channel_bytes)
    total_cycles = 0
    total_union = 0
    total_zero_groups = 0
    maximum_bank_depth = 0
    for start in range(0, rows, 256):
        stop = min(rows, start + 256)
        block = view[start:stop]
        if padded_width != width:
            padded = np.zeros((stop - start, padded_width, channel_bytes),
                              dtype=np.uint8)
            padded[:, :width, :] = block
            block = padded
        grouped = block.reshape(stop - start, groups_per_row,
                                fanout_k, channel_bytes)
        union = np.bitwise_or.reduce(grouped, axis=2)
        cycles = np.zeros(union.shape[:2], dtype=np.int32)
        for bank in range(8):
            bank_count = np.right_shift(union, bank)
            bank_count = np.bitwise_and(bank_count, 1).sum(
                axis=-1, dtype=np.int32)
            np.maximum(cycles, bank_count, out=cycles)
            maximum_bank_depth = max(maximum_bank_depth,
                                     int(bank_count.max(initial=0)))
        total_cycles += int(cycles.sum(dtype=np.int64))
        total_union += int(POPCOUNT[union].sum(dtype=np.int64))
        total_zero_groups += int((cycles == 0).sum(dtype=np.int64))
    base_groups = rows * groups_per_row
    return {
        "row_bounded_groups": base_groups * output_blocks,
        "row_bounded_groups_before_output_blocks": base_groups,
        "source_issue_cycles": total_cycles * output_blocks,
        "source_issue_cycles_before_output_blocks": total_cycles,
        "source_union_indices": total_union * output_blocks,
        "source_union_indices_before_output_blocks": total_union,
        "source_bank_read_transactions": total_union * output_blocks,
        "zero_source_groups": total_zero_groups * output_blocks,
        "maximum_sources_in_one_bank_per_group": maximum_bank_depth,
    }


def capacity_ledger(contract, input_channels, fanout_k, mode):
    model = contract["capacity_model"]
    context_bits = (int(model["resident_contexts"]) - 1).bit_length()
    payload_bits = (fanout_k * context_bits + (fanout_k - 1) + 8 +
                    fanout_k * 8 * 2 + 1)
    aligned = ceil_div(payload_bits, model["response_alignment_bytes"] * 8) * \
        model["response_alignment_bytes"]
    components = {
        "single_int8_weight_tile_256x96":
            model["single_int8_weight_tile_256x96_bytes"],
        "bit_tight_parent_output_line":
            model["bit_tight_parent_output_line_bytes"],
        "support_line": model["support_line_bytes"],
        "two_15x20x96_signed19_output_frames":
            model["two_15x20x96_signed19_output_frames_bytes"],
        "ready_frontier": model["ready_frontier_bytes"],
        "complete_fifo": model["complete_fifo_bytes"],
        "resident_contexts": model["resident_contexts"] * (
            model["context_vector_bytes_signed19_x96"] +
            model["context_metadata_bytes"]),
        "response_metadata_fifo": model["response_entries"] * aligned,
    }
    input_vector_bytes = ceil_div(input_channels, 8)
    if mode == "spatial":
        parent_state = (model["tile_w"] + 1) * input_vector_bytes
        state_name = "spatial_up_row_plus_left_input_vector"
    else:
        parent_state = model["tile_h"] * model["tile_w"] * input_vector_bytes
        state_name = "previous_timestep_input_tile"
    components[state_name] = parent_state
    combined = sum(int(value) for value in components.values())
    maximum = int(model["local_residency_bytes"])
    return {
        "fanout_k": fanout_k,
        "mode": mode,
        "response_metadata_payload_bits": payload_bits,
        "response_metadata_aligned_bytes_per_entry": aligned,
        "input_parent_state_bytes": parent_state,
        "components_bytes": components,
        "combined_local_capacity_bytes": combined,
        "local_capacity_headroom_bytes": maximum - combined,
        "local_residency_bytes": maximum,
        "passes_without_external_state_spill": combined <= maximum,
    }


def transaction_and_cycle_metrics(contract, record, choice_counts, source_bits,
                                  union, fanout_k, mode):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    time, batch, height, width, input_channels = shape
    output_channels = output_shape[-1]
    require(output_shape[:-1] == shape[:-1], "Linear output geometry")
    vectors = time * batch * height * width
    output_blocks = ceil_div(output_channels, 96)
    input_vector_bytes = ceil_div(input_channels, 8)
    activation_beats_per_vector = ceil_div(input_vector_bytes, 32)
    valid_left = time * batch * height * max(0, width - 1)
    valid_up = time * batch * max(0, height - 1) * width
    valid_previous = max(0, time - 1) * batch * height * width
    if mode == "spatial":
        candidate_vectors = valid_left + valid_up
        nonzero_parent = (choice_counts["left"] + choice_counts["up"])
        choice_bits = 2
    else:
        candidate_vectors = valid_previous
        nonzero_parent = choice_counts["previous_timestep"]
        choice_bits = 1
    current_activation_transactions = vectors * activation_beats_per_vector
    candidate_activation_transactions = candidate_vectors * \
        activation_beats_per_vector
    choice_bytes = ceil_div(vectors * choice_bits, 8)
    choice_transactions = ceil_div(choice_bytes, 32)
    weight_bytes = input_channels * output_channels
    weight_transactions = ceil_div(weight_bytes, 32)
    parent_seed_transactions = nonzero_parent * output_blocks
    commit_transactions = vectors * output_blocks
    descriptor_transactions = union["row_bounded_groups"]
    serialized = (weight_transactions + current_activation_transactions +
                  candidate_activation_transactions + choice_transactions +
                  descriptor_transactions + union["source_issue_cycles"] +
                  parent_seed_transactions + commit_transactions)
    physical_slots = (union["source_issue_cycles"] * 8 * 96 * fanout_k)
    product_updates = source_bits * output_channels
    return {
        "geometry": {
            "time": time, "batch": batch, "height": height, "width": width,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "output_blocks_96": output_blocks,
            "vectors": vectors,
        },
        "source_work": {
            "source_bits": source_bits,
            "product_updates": product_updates,
            "physical_product_slots": physical_slots,
            "physical_product_slot_utilization": (
                float(product_updates) / float(physical_slots)
                if physical_slots else 1.0),
        },
        "reads_and_commits": {
            "current_activation_read_bytes": vectors * input_vector_bytes,
            "candidate_parent_activation_read_bytes":
                candidate_vectors * input_vector_bytes,
            "chosen_parent_output_logical_bytes_signed19":
                nonzero_parent * ceil_div(output_channels * 19, 8),
            "chosen_parent_output_allocated_bytes_x96_blocks":
                parent_seed_transactions * 228,
            "final_commit_logical_bytes_signed19":
                vectors * ceil_div(output_channels * 19, 8),
            "final_commit_allocated_bytes_x96_blocks":
                commit_transactions * 228,
            "weight_bytes_int8_hypothesis": weight_bytes,
            "choice_metadata_bytes": choice_bytes,
        },
        "transactions": {
            "weight_dma_256b": weight_transactions,
            "current_activation_selector_256b": current_activation_transactions,
            "candidate_parent_activation_selector_256b":
                candidate_activation_transactions,
            "choice_metadata_write_256b": choice_transactions,
            "group_descriptor": descriptor_transactions,
            "source_bank_reads": union["source_bank_read_transactions"],
            "chosen_parent_output_seed_vector": parent_seed_transactions,
            "final_commit_vector": commit_transactions,
        },
        "cycles": {
            "source_issue": union["source_issue_cycles"],
            "serialized_integrated_no_overlap": serialized,
            "overlap_credit": 0,
        },
        "union": union,
    }


def summarize_rows(rows, config_name):
    cycle_source = [row["configs"][config_name]["cycles"]["source_issue"]
                    for row in rows]
    cycle_integrated = [row["configs"][config_name]["cycles"][
        "serialized_integrated_no_overlap"] for row in rows]
    sum_paths = [
        ("source_work", "source_bits"),
        ("source_work", "product_updates"),
        ("source_work", "physical_product_slots"),
        ("union", "source_union_indices"),
        ("union", "source_bank_read_transactions"),
        ("transactions", "weight_dma_256b"),
        ("transactions", "current_activation_selector_256b"),
        ("transactions", "candidate_parent_activation_selector_256b"),
        ("transactions", "choice_metadata_write_256b"),
        ("transactions", "group_descriptor"),
        ("transactions", "chosen_parent_output_seed_vector"),
        ("transactions", "final_commit_vector"),
    ]
    totals = {}
    for section, field in sum_paths:
        totals[section + "." + field] = sum(
            int(row["configs"][config_name][section][field]) for row in rows)
    return {
        "source_cycle_distribution": distribution(cycle_source),
        "serialized_integrated_cycle_distribution":
            distribution(cycle_integrated),
        "totals": totals,
    }


def validate_inputs(arguments, contract):
    paths = {
        "manifest": arguments.manifest,
        "m52_result": arguments.m52_result,
        "m53_result": arguments.m53_result,
        "m55_result": arguments.m55_result,
        "m39_result": arguments.m39_result,
        "operator_transactions": arguments.operator_transactions,
        "dual_line_contract": arguments.dual_line_contract,
    }
    identity = contract["identity"]
    for name, path in paths.items():
        require(Path(path).is_file() and
                sha256_path(path) == identity[name + "_sha256"],
                "{} SHA mismatch".format(name))
    if identity["analyzer_sha256"] != "TO_FILL_AFTER_ANALYZER_FREEZE":
        require(sha256_path(Path(__file__).resolve()) ==
                identity["analyzer_sha256"], "analyzer SHA mismatch")


def category_for_name(name):
    if ".mlp.fc1" in name:
        return "ffn_expand"
    if ".mlp.fc2" in name:
        return "ffn_contract"
    if name.endswith(".downsample.reduction"):
        return "downsample"
    raise ValueError("unmapped Linear category {}".format(name))


def build_amdahl(contract, config_summaries):
    model = contract["m39_amdahl_model"]
    fixed = int(model["fixed_compute_reference_cycles"])
    captured = int(model["expected_captured_baseline_cycles"])
    outside = fixed - captured
    removable_m39 = int(model["m39_noneligible_plus_qk_cycles"])
    after_both_zero = outside - removable_m39
    rows = []
    for target in model["comparison_targets"]:
        ratio = Fraction(int(target["numerator"]), int(target["denominator"]))
        ceiling = Fraction(fixed, 1) / ratio
        rows.append({
            "name": target["name"],
            "target_speedup": fraction(ratio),
            "maximum_total_cycles": fraction(ceiling),
            "maximum_captured_linear_replacement_cycles_if_only_M63_changes":
                fraction(ceiling - outside),
            "still_required_savings_after_zero_cycle_captured_linear":
                fraction(Fraction(outside, 1) - ceiling),
            "still_required_savings_after_zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk":
                fraction(Fraction(after_both_zero, 1) - ceiling),
            "required_reduction_from_m39_conditional_ideal": {
                "spatial_local": fraction(Fraction(
                    int(model["m39_conditional_ideal_cycles"]["spatial_local"]), 1
                ) - ceiling),
                "temporal_motion": fraction(Fraction(
                    int(model["m39_conditional_ideal_cycles"]["temporal_motion"]), 1
                ) - ceiling),
            },
            "system_speedup_admitted": False,
        })
    candidates = {}
    for name, summary in sorted(config_summaries.items()):
        p95 = summary["serialized_integrated_cycle_distribution"][
            "p95_nearest_rank"]
        denominator = outside + p95
        candidates[name] = {
            "captured_linear_baseline_cycles": captured,
            "captured_linear_replacement_p95_cycles": p95,
            "fixed_outside_captured_linear_cycles": outside,
            "conditional_total_cycles_using_p95": denominator,
            "conditional_fixed_over_total_ratio_not_system_speedup":
                fraction(Fraction(fixed, denominator)),
            "system_speedup_admitted": False,
        }
    return {
        "fixed_compute_reference_cycles": fixed,
        "captured_linear_baseline_cycles": captured,
        "fixed_outside_captured_linear_cycles": outside,
        "captured_linear_zero_cycle_amdahl_ceiling_not_system_speedup":
            fraction(Fraction(fixed, outside)),
        "m39_noneligible_plus_qk_cycles": removable_m39,
        "zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk_ceiling_not_system_speedup":
            fraction(Fraction(fixed, after_both_zero)),
        "targets": rows,
        "candidate_p95_compositions": candidates,
        "system_speedup_admitted": False,
    }


def build_m53_overlap_reconciliation(contract, config_summaries):
    overlap = contract["m53_overlap_reconciliation"]
    outside = int(overlap["m53_outside_four_bottleneck_model_cycles"])
    late = int(overlap["m53_fixed_late_scale_plus_frontend_cycles"])
    pair = int(overlap["m53_pair_p95_cycles"])
    denominator = int(overlap["m53_conditional_denominator_cycles"])
    require(outside + late + pair == denominator == 201259510,
            "M53 overlap denominator equation")
    fixed_linear = int(contract["m39_amdahl_model"][
        "expected_captured_baseline_cycles"])
    scenarios = {}
    for config_name in ("spatial_K4", "temporal_K4"):
        replacement = config_summaries[config_name][
            "serialized_integrated_cycle_distribution"]["p95_nearest_rank"]
        naive_savings = fixed_linear - replacement
        scenarios[config_name] = {
            "m63_fixed_baseline_cycles": fixed_linear,
            "m63_replacement_p95_cycles": replacement,
            "m63_naive_fixed_baseline_savings_cycles": naive_savings,
            "naive_subtraction_from_m53_denominator_cycles_prohibited":
                denominator - naive_savings,
            "naive_joint_ratio_prohibited": fraction(Fraction(
                int(contract["m39_amdahl_model"][
                    "fixed_compute_reference_cycles"]),
                denominator - naive_savings)),
            "L24_inherited_inside_188824491": "UNKNOWN_NOT_OPERATOR_DECOMPOSED",
            "additive_savings_admitted": False,
            "joint_ratio_admitted": False,
        }
    return {
        "status": overlap["overlap_status"],
        "m53_exact_denominator_components_cycles": {
            "outside_four_bottleneck_model": outside,
            "fixed_late_scale_plus_frontend": late,
            "pair_p95": pair,
            "total": denominator,
        },
        "exact_denominator_equation": overlap["exact_denominator_equation"],
        "scope_relation": overlap["scope_relation"],
        "missing_evidence": overlap["missing_evidence"],
        "legal_replacement_equation": overlap["legal_replacement_equation"],
        "prohibited_equation": overlap["prohibited_equation"],
        "scenarios": scenarios,
        "savings_admitted_as_additive_to_m53": [],
        "savings_with_unknown_overlap_and_therefore_not_additive": [
            "spatial_K4 fixed-baseline savings",
            "temporal_K4 fixed-baseline savings",
        ],
        "required_closure": (
            "freeze an exact operator/category decomposition of the M53 "
            "188824491 outside term and identify L24_inherited_inside_outside "
            "from the same lineage; then replace that term with M63 R24 rather "
            "than subtracting fixed-baseline savings"),
        "joint_ratio_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--m52-result", required=True, type=Path)
    parser.add_argument("--m53-result", required=True, type=Path)
    parser.add_argument("--m55-result", required=True, type=Path)
    parser.add_argument("--m39-result", required=True, type=Path)
    parser.add_argument("--operator-transactions", required=True, type=Path)
    parser.add_argument("--dual-line-contract", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "refusing existing output")
    contract = strict_json(arguments.contract)
    require(contract["schema"] ==
            "m63_linear_k4_spatiotemporal_full_network_opportunity_contract_v1" and
            contract["status"] ==
            "FROZEN_ALL24_LINEAR_BANK_EXECUTABLE_OPPORTUNITY_ONLY",
            "contract schema/status")
    validate_inputs(arguments, contract)
    manifest = strict_json(arguments.manifest)
    m52 = strict_json(arguments.m52_result)
    m53 = strict_json(arguments.m53_result)
    m55 = strict_json(arguments.m55_result)
    m39 = strict_json(arguments.m39_result)
    dual = strict_json(arguments.dual_line_contract)
    require(m52["status"].startswith("PASS_PROMOTE_") and
            m53["status"] ==
            "PASS_M53_K4_CTX16_TEMPORAL_TRANSACTION_DSE_M54_RTL_REQUIRED" and
            m55["status"] ==
            "PASS_EXACT_SOURCE_BIT_WORK_NO_CYCLE_SPEEDUP_ENERGY_OR_PPA_CLAIM" and
            m39["status"] ==
            "PASS_M39_R3_CURRENT_ANCHORS_CONDITIONAL_BOTTLENECK_DSE_ONLY",
            "upstream status mismatch")
    require(m53["two_frame_capacity_ledger"][
                "combined_k4_ctx16_capacity_bytes"] == 176688 and
            m53["two_frame_capacity_ledger"][
                "local_capacity_headroom_bytes"] == 17040 and
            m53["two_frame_capacity_ledger"]["existing_two_frame_bytes"] ==
            136800, "M52/M53 capacity anchor mismatch")
    require(m53["conditional_frozen_compute_model"][
                "conditional_total_cycles"] == 201259510 and
            m53["conditional_frozen_compute_model"][
                "pair_p95_nearest_rank_cycles"] == 9798504 and
            contract["m53_overlap_reconciliation"][
                "m53_outside_four_bottleneck_model_cycles"] +
            contract["m53_overlap_reconciliation"][
                "m53_fixed_late_scale_plus_frontend_cycles"] +
            contract["m53_overlap_reconciliation"][
                "m53_pair_p95_cycles"] == 201259510,
            "M53 denominator reconciliation drift")
    categories = dual["coverage"]["categories"]
    require(categories["ffn_expand"]["eligible_cycles"] == 100895624 and
            categories["ffn_contract"]["eligible_cycles"] == 41413997 and
            categories["downsample"]["eligible_cycles"] == 12321697,
            "dual-line eligible category drift")

    target_indices = set(contract["population"]["target_module_indices"])
    target_records = [row for row in manifest["records"]
                      if int(row["module_index"]) in target_indices]
    require(len(manifest["records"]) == 310 and len(target_records) == 240 and
            all(row["operator"] == "Linear" for row in target_records),
            "manifest/target population")
    target_records.sort(key=lambda row: (int(row["sample_id"]),
                                         int(row["module_index"])))
    require([(int(row["sample_id"]), int(row["module_index"]))
             for row in target_records] ==
            [(sample, module) for sample in range(10)
             for module in sorted(target_indices)], "target Cartesian order")
    m55_records = dict(((int(row["sample_id"]), int(row["module_index"])), row)
                       for row in m55["per_record"])
    with arguments.operator_transactions.open("r", encoding="utf-8") as handle:
        operator_rows = dict((row["name"], row) for row in csv.DictReader(handle))

    per_record = []
    payload_root = arguments.payload_root.resolve()
    for ordinal, record in enumerate(target_records):
        path = payload_root / record["relative_path"]
        require(path.is_file() and path.stat().st_size ==
                int(record["packed_bytes"]) and
                sha256_path(path) == record["file_sha256"],
                "payload identity {}".format(ordinal))
        current = packed_array(path, record)
        output_blocks = ceil_div(int(record["output_shape"][-1]), 96)
        configs = {}
        mode_identity = {}
        for mode in MODES:
            residual, choice, source_bits, positive, negative, choices = \
                select_residual(current, mode)
            upstream = m55_records[(int(record["sample_id"]),
                                    int(record["module_index"]))]["analysis"]
            m55_mode = "local" if mode == "spatial" else "motion"
            require(source_bits == int(upstream["source_bits"][m55_mode]) and
                    choices == upstream["choice_counts"][m55_mode],
                    "M55 source/choice mismatch {} {}".format(ordinal, mode))
            mode_identity[mode] = {
                "choice_counts": choices,
                "negative_1_to_0_source_bits": negative,
                "positive_0_to_1_source_bits": positive,
                "selected_residual_packed_sha256": hashlib.sha256(
                    residual.tobytes(order="C")).hexdigest(),
                "source_bits": source_bits,
            }
            for fanout_k in K_POINTS:
                name = "{}_K{}".format(mode, fanout_k)
                union = union_bank_metrics(residual, fanout_k, output_blocks)
                metrics = transaction_and_cycle_metrics(
                    contract, record, choices, source_bits, union,
                    fanout_k, mode)
                metrics["capacity"] = capacity_ledger(
                    contract, int(record["input_shape"][-1]), fanout_k, mode)
                configs[name] = metrics
            del residual, choice
        category = category_for_name(record["name"])
        require(record["name"] in operator_rows and
                operator_rows[record["name"]]["category"] == category,
                "M39 operator category mismatch")
        per_record.append({
            "ordinal": ordinal,
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "module_name": record["name"],
            "category": category,
            "relative_path": record["relative_path"],
            "file_sha256": record["file_sha256"],
            "input_shape": record["input_shape"],
            "output_shape": record["output_shape"],
            "mode_identity": mode_identity,
            "configs": configs,
        })
        del current
        print("[M63] {}/240 sample={} module={}".format(
            ordinal + 1, record["sample_id"], record["module_index"]),
              flush=True)

    config_names = ["{}_K{}".format(mode, fanout_k)
                    for mode in MODES for fanout_k in K_POINTS]
    per_module = []
    module_baseline = {}
    for module_index in sorted(target_indices):
        rows = [row for row in per_record
                if row["module_index"] == module_index]
        require(len(rows) == 10, "module sample population")
        name = rows[0]["module_name"]
        source = operator_rows[name]
        baseline = int(source["activity_cycles_at_config_lanes"])
        module_baseline[module_index] = baseline
        summaries = dict((config, summarize_rows(rows, config))
                         for config in config_names)
        capacities = dict((config, rows[0]["configs"][config]["capacity"])
                          for config in config_names)
        per_module.append({
            "module_index": module_index,
            "module_name": name,
            "category": rows[0]["category"],
            "input_shape": rows[0]["input_shape"],
            "output_shape": rows[0]["output_shape"],
            "m39_activity_cycles_at_config_lanes": baseline,
            "config_summaries": summaries,
            "capacities": capacities,
        })

    captured_baseline = sum(module_baseline.values())
    require(captured_baseline == contract["m39_amdahl_model"][
        "expected_captured_baseline_cycles"], "captured M39 baseline drift")
    per_sample = []
    for sample_id in range(10):
        rows = [row for row in per_record if row["sample_id"] == sample_id]
        require(len(rows) == 24, "sample module population")
        item = {"sample_id": sample_id, "configs": {}}
        for config in config_names:
            item["configs"][config] = {
                "source_issue_cycles": sum(row["configs"][config]["cycles"][
                    "source_issue"] for row in rows),
                "serialized_integrated_no_overlap_cycles": sum(
                    row["configs"][config]["cycles"][
                        "serialized_integrated_no_overlap"] for row in rows),
                "source_bits": sum(row["configs"][config]["source_work"][
                    "source_bits"] for row in rows),
                "source_union_indices": sum(row["configs"][config]["union"][
                    "source_union_indices"] for row in rows),
            }
        per_sample.append(item)

    aggregate_configs = {}
    for config in config_names:
        rows_by_sample = [{"configs": {config: {
            "cycles": {
                "source_issue": item["configs"][config]["source_issue_cycles"],
                "serialized_integrated_no_overlap": item["configs"][config][
                    "serialized_integrated_no_overlap_cycles"],
            },
            "source_work": {
                "source_bits": item["configs"][config]["source_bits"],
                "product_updates": sum(row["configs"][config]["source_work"][
                    "product_updates"] for row in per_record
                    if row["sample_id"] == item["sample_id"]),
                "physical_product_slots": sum(row["configs"][config][
                    "source_work"]["physical_product_slots"] for row in per_record
                    if row["sample_id"] == item["sample_id"]),
            },
            "union": {
                "source_union_indices": item["configs"][config][
                    "source_union_indices"],
                "source_bank_read_transactions": item["configs"][config][
                    "source_union_indices"],
            },
            "transactions": dict((field, sum(row["configs"][config][
                "transactions"][field] for row in per_record
                if row["sample_id"] == item["sample_id"])) for field in (
                    "weight_dma_256b", "current_activation_selector_256b",
                    "candidate_parent_activation_selector_256b",
                    "choice_metadata_write_256b", "group_descriptor",
                    "chosen_parent_output_seed_vector", "final_commit_vector")),
        }}} for item in per_sample]
        aggregate_configs[config] = summarize_rows(rows_by_sample, config)
        mode, k_text = config.split("_K")
        aggregate_configs[config]["capacity_feasible_modules"] = sum(
            1 for row in per_module if row["capacities"][config][
                "passes_without_external_state_spill"])
        aggregate_configs[config]["capacity_infeasible_modules"] = 24 - \
            aggregate_configs[config]["capacity_feasible_modules"]
        aggregate_configs[config]["fanout_k"] = int(k_text)
        aggregate_configs[config]["mode"] = mode

    for mode in MODES:
        k1 = aggregate_configs[mode + "_K1"]
        for fanout_k in (2, 4):
            row = aggregate_configs["{}_K{}".format(mode, fanout_k)]
            row["ratios_not_system_speedup"] = {
                "k1_over_k_source_issue": fraction(Fraction(
                    k1["source_cycle_distribution"]["sum"],
                    row["source_cycle_distribution"]["sum"])),
                "k1_over_k_serialized_integrated": fraction(Fraction(
                    k1["serialized_integrated_cycle_distribution"]["sum"],
                    row["serialized_integrated_cycle_distribution"]["sum"])),
            }

    category_ledger = {}
    for category in ("ffn_expand", "ffn_contract", "downsample"):
        modules = [row for row in per_module if row["category"] == category]
        category_ledger[category] = {
            "captured_modules": len(modules),
            "captured_m39_activity_cycles": sum(
                row["m39_activity_cycles_at_config_lanes"] for row in modules),
            "dual_line_category_total_cycles":
                int(categories[category]["cycles"]),
            "dual_line_category_eligible_cycles":
                int(categories[category]["eligible_cycles"]),
        }
    require(sum(row["captured_m39_activity_cycles"]
                for row in category_ledger.values()) == captured_baseline,
            "category captured sum")

    result = {
        "schema": "m63_linear_k4_spatiotemporal_full_network_opportunity_result_v1",
        "status": "PASS_ALL24_LINEAR_K1_K2_K4_BANK_EXECUTABLE_OPPORTUNITY_RTL_NUMERIC_SYSTEM_OPEN",
        "identity": {
            "contract_sha256": sha256_path(arguments.contract),
            "analyzer_sha256": sha256_path(Path(__file__).resolve()),
            "inputs_sha256": dict((key, value) for key, value in
                                   contract["identity"].items()
                                   if key != "analyzer_sha256"),
        },
        "claim_boundary": contract["claim_boundary"],
        "population": {
            "manifest_records": 310,
            "target_records": 240,
            "target_modules": 24,
            "samples": 10,
            "operator": "Linear",
            "raw_payload_sha_size_popcount_checked": True,
        },
        "address_schedule": contract["address_schedule"],
        "aggregate_configurations": aggregate_configs,
        "per_sample": per_sample,
        "per_module": per_module,
        "per_record": per_record,
        "m39_category_ledger": category_ledger,
        "m39_amdahl": build_amdahl(contract, aggregate_configs),
        "m53_overlap_reconciliation": build_m53_overlap_reconciliation(
            contract, aggregate_configs),
        "kill_gates": {
            "spatial_capacity_infeasible_modules":
                aggregate_configs["spatial_K4"]["capacity_infeasible_modules"],
            "spatial_all24_fit_without_external_state":
                aggregate_configs["spatial_K4"]["capacity_infeasible_modules"] == 0,
            "temporal_capacity_infeasible_modules":
                aggregate_configs["temporal_K4"]["capacity_infeasible_modules"],
            "temporal_all24_fit_without_external_state":
                aggregate_configs["temporal_K4"]["capacity_infeasible_modules"] == 0,
            "temporal_headline_killed_by_external_state_requirement":
                aggregate_configs["temporal_K4"]["capacity_infeasible_modules"] > 0,
            "joint_m53_m63_ratio_killed_by_overlap_unknown": True,
        },
        "qualification": {
            "int8_numeric_qualified": False,
            "rtl_or_vcs_admitted": False,
            "synthesis_or_ppa_admitted": False,
            "system_speedup_admitted": False,
            "address_timed_dram_admitted": False,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        arguments.output.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(arguments.output))
    temporary.unlink()
    print(json.dumps({
        "output": str(arguments.output),
        "output_sha256": sha256_path(arguments.output),
        "spatial_k4_source_p95": aggregate_configs["spatial_K4"][
            "source_cycle_distribution"]["p95_nearest_rank"],
        "temporal_k4_source_p95": aggregate_configs["temporal_K4"][
            "source_cycle_distribution"]["p95_nearest_rank"],
        "status": result["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
