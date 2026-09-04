#!/usr/bin/env python3
"""Fail-closed validator for M63 all-24 Linear opportunity evidence."""

from __future__ import print_function

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


CONFIGS = ("spatial_K1", "spatial_K2", "spatial_K4",
           "temporal_K1", "temporal_K2", "temporal_K4")
PARENTS = ("zero", "left", "up", "previous_timestep")


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
        raise ValueError("non-standard JSON {}".format(raw))

    def pairs(raw):
        result = {}
        for key, value in raw:
            require(key not in result, "duplicate key {}".format(key))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def category(name):
    if ".mlp.fc1" in name:
        return "ffn_expand"
    if ".mlp.fc2" in name:
        return "ffn_contract"
    if name.endswith(".downsample.reduction"):
        return "downsample"
    raise ValueError("unmapped category")


def nearest(values, percentile):
    ordered = sorted(values)
    return ordered[int(math.ceil(len(ordered) * percentile)) - 1]


def check_distribution(row, values, label):
    ordered = sorted(int(value) for value in values)
    require(row == {"count": len(ordered), "min": ordered[0],
                    "p50_nearest_rank": nearest(ordered, 0.50),
                    "p95_nearest_rank": nearest(ordered, 0.95),
                    "max": ordered[-1], "sum": sum(ordered)},
            "distribution {}".format(label))


def capacity(contract, channels, fanout_k, mode):
    model = contract["capacity_model"]
    context_bits = (model["resident_contexts"] - 1).bit_length()
    payload = fanout_k * context_bits + fanout_k - 1 + 8 + fanout_k * 16 + 1
    aligned = ceil_div(payload, model["response_alignment_bytes"] * 8) * \
        model["response_alignment_bytes"]
    components = {
        "single_int8_weight_tile_256x96": 24576,
        "bit_tight_parent_output_line": 4560,
        "support_line": 640,
        "two_15x20x96_signed19_output_frames": 136800,
        "ready_frontier": 1280,
        "complete_fifo": 3904,
        "resident_contexts": 16 * (228 + 64),
        "response_metadata_fifo": 16 * aligned,
    }
    vector_bytes = ceil_div(channels, 8)
    if mode == "spatial":
        state = 21 * vector_bytes
        components["spatial_up_row_plus_left_input_vector"] = state
    else:
        state = 300 * vector_bytes
        components["previous_timestep_input_tile"] = state
    combined = sum(components.values())
    return {
        "fanout_k": fanout_k, "mode": mode,
        "response_metadata_payload_bits": payload,
        "response_metadata_aligned_bytes_per_entry": aligned,
        "input_parent_state_bytes": state,
        "components_bytes": components,
        "combined_local_capacity_bytes": combined,
        "local_capacity_headroom_bytes": 193728 - combined,
        "local_residency_bytes": 193728,
        "passes_without_external_state_spill": combined <= 193728,
    }


def validate_core(contract, manifest, m52, m53, m55, m39, dual,
                  operator_rows, result):
    require(contract["schema"] ==
            "m63_linear_k4_spatiotemporal_full_network_opportunity_contract_v1" and
            result["schema"] ==
            "m63_linear_k4_spatiotemporal_full_network_opportunity_result_v1" and
            result["status"] ==
            "PASS_ALL24_LINEAR_K1_K2_K4_BANK_EXECUTABLE_OPPORTUNITY_RTL_NUMERIC_SYSTEM_OPEN",
            "schema/status")
    require(result["claim_boundary"] == contract["claim_boundary"] and
            "system speedup" in " ".join(
                result["claim_boundary"]["forbidden"]).lower() and
            result["qualification"] == {
                "address_timed_dram_admitted": False,
                "int8_numeric_qualified": False,
                "rtl_or_vcs_admitted": False,
                "synthesis_or_ppa_admitted": False,
                "system_speedup_admitted": False}, "claim boundary")
    require(result["population"] == {
        "manifest_records": 310, "operator": "Linear",
        "raw_payload_sha_size_popcount_checked": True, "samples": 10,
        "target_modules": 24, "target_records": 240}, "population")
    require(m52["status"].startswith("PASS_PROMOTE_") and
            m53["conditional_frozen_compute_model"]["conditional_total_cycles"] ==
            201259510 and
            m55["status"] ==
            "PASS_EXACT_SOURCE_BIT_WORK_NO_CYCLE_SPEEDUP_ENERGY_OR_PPA_CLAIM" and
            m39["status"] ==
            "PASS_M39_R3_CURRENT_ANCHORS_CONDITIONAL_BOTTLENECK_DSE_ONLY",
            "upstream status")
    targets = [row for row in manifest["records"]
               if int(row["module_index"]) in set(range(6, 30))]
    targets.sort(key=lambda row: (int(row["sample_id"]),
                                  int(row["module_index"])))
    m55_rows = dict(((row["sample_id"], row["module_index"]), row)
                    for row in m55["per_record"])
    require(len(targets) == len(result["per_record"]) == 240,
            "record population")
    for index, (source, row) in enumerate(zip(targets, result["per_record"])):
        require(row["ordinal"] == index and
                row["sample_id"] == source["sample_id"] and
                row["module_index"] == source["module_index"] and
                row["module_name"] == source["name"] and
                row["relative_path"] == source["relative_path"] and
                row["file_sha256"] == source["file_sha256"] and
                row["input_shape"] == source["input_shape"] and
                row["output_shape"] == source["output_shape"] and
                row["category"] == category(source["name"]),
                "record identity {}".format(index))
        vector_count = product(source["input_shape"][:-1])
        upstream = m55_rows[(source["sample_id"], source["module_index"])][
            "analysis"]
        require(set(row["configs"]) == set(CONFIGS), "configs")
        for mode in ("spatial", "temporal"):
            identity = row["mode_identity"][mode]
            m55_mode = "local" if mode == "spatial" else "motion"
            require(identity["source_bits"] == upstream["source_bits"][m55_mode] and
                    identity["choice_counts"] ==
                    upstream["choice_counts"][m55_mode] and
                    sum(identity["choice_counts"].values()) == vector_count and
                    identity["positive_0_to_1_source_bits"] +
                    identity["negative_1_to_0_source_bits"] ==
                    identity["source_bits"] and
                    len(identity["selected_residual_packed_sha256"]) == 64,
                    "mode identity {} {}".format(index, mode))
            first_source = None
            for fanout_k in (1, 2, 4):
                name = "{}_K{}".format(mode, fanout_k)
                item = row["configs"][name]
                geometry = item["geometry"]
                require(geometry["vectors"] == vector_count and
                        geometry["input_channels"] == source["input_shape"][-1] and
                        geometry["output_channels"] == source["output_shape"][-1] and
                        geometry["output_blocks_96"] ==
                        ceil_div(source["output_shape"][-1], 96), "geometry")
                work = item["source_work"]
                if first_source is None:
                    first_source = work["source_bits"]
                require(work["source_bits"] == first_source ==
                        identity["source_bits"] and
                        work["product_updates"] == work["source_bits"] *
                        geometry["output_channels"] and
                        work["physical_product_slots"] ==
                        item["cycles"]["source_issue"] * 8 * 96 * fanout_k,
                        "work equation")
                union = item["union"]
                tx = item["transactions"]
                require(union["source_bank_read_transactions"] ==
                        union["source_union_indices"] == tx["source_bank_reads"] and
                        union["row_bounded_groups"] == tx["group_descriptor"] and
                        union["source_issue_cycles"] ==
                        item["cycles"]["source_issue"] and
                        union["source_union_indices"] <=
                        work["source_bits"] * geometry["output_blocks_96"],
                        "union equation")
                expected_integrated = (
                    tx["weight_dma_256b"] +
                    tx["current_activation_selector_256b"] +
                    tx["candidate_parent_activation_selector_256b"] +
                    tx["choice_metadata_write_256b"] +
                    tx["group_descriptor"] + item["cycles"]["source_issue"] +
                    tx["chosen_parent_output_seed_vector"] +
                    tx["final_commit_vector"])
                require(item["cycles"] == {
                    "overlap_credit": 0,
                    "serialized_integrated_no_overlap": expected_integrated,
                    "source_issue": item["cycles"]["source_issue"]},
                    "integrated equation")
                require(item["capacity"] == capacity(
                    contract, geometry["input_channels"], fanout_k, mode),
                    "capacity equation")

    require(len(result["per_module"]) == 24 and
            len(result["per_sample"]) == 10, "module/sample population")
    module_by_index = dict((row["module_index"], row)
                           for row in result["per_module"])
    captured = 0
    for module_index in range(6, 30):
        module = module_by_index[module_index]
        rows = [row for row in result["per_record"]
                if row["module_index"] == module_index]
        require(len(rows) == 10 and module["module_name"] in operator_rows,
                "module records")
        baseline = int(operator_rows[module["module_name"]][
            "activity_cycles_at_config_lanes"])
        captured += baseline
        require(module["m39_activity_cycles_at_config_lanes"] == baseline and
                module["category"] == operator_rows[module["module_name"]][
                    "category"], "module baseline")
        for name in CONFIGS:
            require(module["capacities"][name] == rows[0]["configs"][name][
                "capacity"], "module capacity")
            check_distribution(module["config_summaries"][name][
                "source_cycle_distribution"],
                [row["configs"][name]["cycles"]["source_issue"] for row in rows],
                "module source")
            check_distribution(module["config_summaries"][name][
                "serialized_integrated_cycle_distribution"],
                [row["configs"][name]["cycles"][
                    "serialized_integrated_no_overlap"] for row in rows],
                "module integrated")
    require(captured == 154631318, "captured baseline")

    sample_by_id = dict((row["sample_id"], row) for row in result["per_sample"])
    aggregate = result["aggregate_configurations"]
    for name in CONFIGS:
        source_values = []
        integrated_values = []
        for sample_id in range(10):
            rows = [row for row in result["per_record"]
                    if row["sample_id"] == sample_id]
            expected_source = sum(row["configs"][name]["cycles"][
                "source_issue"] for row in rows)
            expected_integrated = sum(row["configs"][name]["cycles"][
                "serialized_integrated_no_overlap"] for row in rows)
            sample = sample_by_id[sample_id]["configs"][name]
            require(sample["source_issue_cycles"] == expected_source and
                    sample["serialized_integrated_no_overlap_cycles"] ==
                    expected_integrated, "sample aggregate")
            source_values.append(expected_source)
            integrated_values.append(expected_integrated)
        check_distribution(aggregate[name]["source_cycle_distribution"],
                           source_values, "aggregate source")
        check_distribution(aggregate[name][
            "serialized_integrated_cycle_distribution"],
                           integrated_values, "aggregate integrated")
    require(aggregate["spatial_K4"]["source_cycle_distribution"][
                "p95_nearest_rank"] == 14117338 and
            aggregate["spatial_K4"][
                "serialized_integrated_cycle_distribution"][
                "p95_nearest_rank"] == 28535569 and
            aggregate["temporal_K4"]["source_cycle_distribution"][
                "p95_nearest_rank"] == 14414724 and
            aggregate["temporal_K4"][
                "serialized_integrated_cycle_distribution"][
                "p95_nearest_rank"] == 25245435 and
            aggregate["spatial_K4"]["capacity_infeasible_modules"] == 0 and
            aggregate["temporal_K4"]["capacity_infeasible_modules"] == 11,
            "headline anchors")

    category_ledger = result["m39_category_ledger"]
    require(category_ledger["ffn_expand"]["captured_m39_activity_cycles"] ==
            100895624 and
            category_ledger["ffn_contract"]["captured_m39_activity_cycles"] ==
            41413997 and
            category_ledger["downsample"]["captured_m39_activity_cycles"] ==
            12321697, "category ledger")
    amdahl = result["m39_amdahl"]
    require(amdahl["fixed_compute_reference_cycles"] == 620868243 and
            amdahl["captured_linear_baseline_cycles"] == 154631318 and
            amdahl["fixed_outside_captured_linear_cycles"] == 466236925 and
            amdahl["m39_noneligible_plus_qk_cycles"] == 162059820 and
            amdahl["system_speedup_admitted"] is False,
            "Amdahl anchors")
    require(abs(amdahl[
        "captured_linear_zero_cycle_amdahl_ceiling_not_system_speedup"][
            "float"] - 620868243.0 / 466236925.0) < 1e-15 and
            abs(amdahl[
        "zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk_ceiling_not_system_speedup"][
            "float"] - 620868243.0 / 304177105.0) < 1e-15,
            "Amdahl ceiling")
    targets_by_name = dict((row["name"], row) for row in amdahl["targets"])
    require(abs(targets_by_name["3.2x"][
        "still_required_savings_after_zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk"][
            "float"] - 110155779.0625) < 1e-9 and
            abs(targets_by_name["3.45x"][
        "still_required_savings_after_zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk"][
            "float"] - 124215295.43478261) < 1e-9,
            "target gates")

    overlap = result["m53_overlap_reconciliation"]
    require(overlap["status"] ==
            "OVERLAP_UNKNOWN_M63_FIXED_BASELINE_SAVINGS_NOT_ADDITIVE_TO_M53" and
            overlap["m53_exact_denominator_components_cycles"] == {
                "fixed_late_scale_plus_frontend": 2636515,
                "outside_four_bottleneck_model": 188824491,
                "pair_p95": 9798504, "total": 201259510} and
            overlap["savings_admitted_as_additive_to_m53"] == [] and
            overlap["joint_ratio_admitted"] is False and
            all(not row["additive_savings_admitted"] and
                not row["joint_ratio_admitted"] and
                row["L24_inherited_inside_188824491"] ==
                "UNKNOWN_NOT_OPERATOR_DECOMPOSED"
                for row in overlap["scenarios"].values()),
            "M53 overlap guard")
    require(result["kill_gates"] == {
        "joint_m53_m63_ratio_killed_by_overlap_unknown": True,
        "spatial_all24_fit_without_external_state": True,
        "spatial_capacity_infeasible_modules": 0,
        "temporal_all24_fit_without_external_state": False,
        "temporal_capacity_infeasible_modules": 11,
        "temporal_headline_killed_by_external_state_requirement": True},
        "kill gates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "analyzer", "manifest", "m52-result",
                 "m53-result", "m55-result", "m39-result",
                 "operator-transactions", "dual-line-contract", "result"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-result-sha256", required=True)
    arguments = parser.parse_args()
    contract = strict_json(arguments.contract)
    identity = contract["identity"]
    paths = {
        "analyzer": arguments.analyzer, "manifest": arguments.manifest,
        "m52_result": arguments.m52_result, "m53_result": arguments.m53_result,
        "m55_result": arguments.m55_result, "m39_result": arguments.m39_result,
        "operator_transactions": arguments.operator_transactions,
        "dual_line_contract": arguments.dual_line_contract,
    }
    for name, path in paths.items():
        require(sha256_path(path) == identity[name + "_sha256"],
                "input SHA {}".format(name))
    require(sha256_path(arguments.result) == arguments.expected_result_sha256,
            "result SHA")
    manifest = strict_json(arguments.manifest)
    m52 = strict_json(arguments.m52_result)
    m53 = strict_json(arguments.m53_result)
    m55 = strict_json(arguments.m55_result)
    m39 = strict_json(arguments.m39_result)
    dual = strict_json(arguments.dual_line_contract)
    result = strict_json(arguments.result)
    with arguments.operator_transactions.open("r", encoding="utf-8") as handle:
        operator_rows = dict((row["name"], row) for row in csv.DictReader(handle))
    require(result["identity"]["contract_sha256"] ==
            sha256_path(arguments.contract) and
            result["identity"]["analyzer_sha256"] ==
            sha256_path(arguments.analyzer), "result identity")
    validate_core(contract, manifest, m52, m53, m55, m39, dual,
                  operator_rows, result)
    print(json.dumps({
        "result_sha256": sha256_path(arguments.result),
        "spatial_k4_p95_cycles": 28535569,
        "temporal_k4_p95_cycles": 25245435,
        "temporal_capacity_fail_modules": 11,
        "m53_overlap_status": result["m53_overlap_reconciliation"]["status"],
        "status": "PASS_M63_ALL24_LINEAR_OPPORTUNITY_ONLY_JOINT_RATIO_KILLED",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
