#!/usr/bin/env python3
"""Independent fail-closed validator for the producer-owned M63 evidence."""

from __future__ import print_function

import argparse
import copy
import csv
import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REVIEW_SHA256 = \
    "4db187df89f509a0adc1296ae5209863bf248982968d7f54dbf641d79f09fe12"
CONFIGS = ["{}_K{}".format(mode, k) for mode in ("spatial", "temporal")
           for k in (1, 2, 4)]
RESULT_REL = Path("results/m63_linear_k4_spatiotemporal_full_network_opportunity_r1_20260823/m63_linear_k4_spatiotemporal_full_network_opportunity_result_r2.json")
PRODUCER_RECEIPT_REL = Path("results/m63_linear_k4_spatiotemporal_full_network_opportunity_r1_20260823/m63_linear_k4_spatiotemporal_full_network_validation_receipt_r1.json")
TAMPER_RECEIPT_REL = Path("results/m63_linear_k4_spatiotemporal_full_network_opportunity_r1_20260823/m63_linear_k4_spatiotemporal_full_network_tamper_receipt_r1.json")
CONTRACT_REL = Path("contracts/m63_linear_k4_spatiotemporal_full_network_opportunity_contract_r1_20260823.json")
MANIFEST_REL = Path("results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json")
M52_REL = Path("results/m52_high_fanout_context16_dse_r1_20260823/m52_high_fanout_context16_dse.json")
M53_REL = Path("results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M55_REL = Path("results/m55_h67_full_network_dual_parent_opportunity_r1_20260823/m55_h67_full_network_dual_parent_opportunity_result_r1.json")
M39_REL = Path("results/m39_remaining_bottleneck_r3_20260822/m39_remaining_bottleneck.json")
OPERATOR_CSV = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/h67_full_network_ledger_v2_multisample_vcs_20260821/operator_transactions.csv")
DUAL_CONTRACT = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/h67_dual_line_full_system_v0_20260821/dual_line_contract.json")


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
        raise ValueError("non-standard JSON constant {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def category(name):
    if ".mlp.fc1" in name:
        return "ffn_expand"
    if ".mlp.fc2" in name:
        return "ffn_contract"
    if name.endswith(".downsample.reduction"):
        return "downsample"
    raise ValueError("unmapped Linear module {}".format(name))


def distribution(values):
    ordered = sorted(int(value) for value in values)
    require(ordered, "empty distribution")
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50_nearest_rank": ordered[int(math.ceil(len(ordered) * 0.50)) - 1],
        "p95_nearest_rank": ordered[int(math.ceil(len(ordered) * 0.95)) - 1],
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def check_fraction(row, expected, label):
    require(row["numerator"] == expected.numerator and
            row["denominator"] == expected.denominator and
            abs(row["float"] - float(expected)) < 1e-14,
            "fraction {}".format(label))


def resolve_binding(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def validate_review(review, review_path):
    require(sha256_path(review_path) == EXPECTED_REVIEW_SHA256,
            "independent review SHA")
    require(review["schema"] == "m63_independent_hammer_review_v1" and
            review["status"] ==
            "PASS_OPPORTUNITY_MODEL_WITH_MANDATORY_TEMPORAL_AND_JOINT_KILL_GATES",
            "review schema/status")
    require(review["reviewer_is_producer"] is False and
            review["producer_evidence_modified"] is False and
            review["headline"] is False and
            review["system_speedup"] is False, "review independence/boundary")
    score = review["date_oriented_score"]
    require(score["score_0_to_100"] == 63 and
            sum(score["subscores"].values()) == 63, "DATE score")
    require(review["issues"]["P0"] == [] and
            len(review["issues"]["P1"]) == 4 and
            len(review["issues"]["P2"]) == 3, "issue ledger")
    observed = {}
    for name, expected in review["exact_sha_bindings"].items():
        path = resolve_binding(name)
        require(path.is_file(), "review binding missing {}".format(name))
        actual = sha256_path(path)
        require(actual == expected, "review binding drift {}".format(name))
        observed[name] = actual
    return observed


def validate_producer_receipt(receipt, result, tamper):
    require(receipt["schema"] ==
            "m63_linear_k4_spatiotemporal_full_network_validation_receipt_v1" and
            receipt["status"] ==
            "PASS_ALL24_LINEAR_OPPORTUNITY_ONLY_TEMPORAL_AND_JOINT_RATIO_KILLED",
            "producer receipt schema/status")
    require(receipt["artifacts"][RESULT_REL.name]["sha256"] ==
            sha256_path(ROOT / RESULT_REL) and
            receipt["artifacts"][TAMPER_RECEIPT_REL.name]["sha256"] ==
            sha256_path(ROOT / TAMPER_RECEIPT_REL), "producer artifact binding")
    source_paths = {
        "analyzer_sha256": ROOT / "system_simulator/scripts/analyze_m63_linear_k4_spatiotemporal_full_network_opportunity.py",
        "validator_sha256": ROOT / "system_simulator/scripts/validate_m63_linear_k4_spatiotemporal_full_network_opportunity.py",
        "tamper_runner_sha256": ROOT / "system_simulator/scripts/run_m63_linear_k4_spatiotemporal_full_network_tamper.py",
        "contract_sha256": ROOT / CONTRACT_REL,
        "manifest_sha256": ROOT / MANIFEST_REL,
        "m52_result_sha256": ROOT / M52_REL,
        "m53_result_sha256": ROOT / M53_REL,
        "m55_result_sha256": ROOT / M55_REL,
        "m39_result_sha256": ROOT / M39_REL,
        "operator_transactions_sha256": OPERATOR_CSV,
        "dual_line_contract_sha256": DUAL_CONTRACT,
    }
    for key, path in source_paths.items():
        require(receipt["sources"][key] == sha256_path(path),
                "producer receipt source {}".format(key))
    require(receipt["method"]["system_speedup_admitted"] is False and
            receipt["method"]["producer_not_self_review"] is True,
            "producer receipt method boundary")
    require(tamper["status"] == "PASS_ALL_SEMANTIC_TAMPERS_REJECTED" and
            tamper["attack_count"] == tamper["rejected_count"] == 22 and
            all(row["rejected"] for row in tamper["attacks"]),
            "producer tamper receipt")
    names = set(row["name"] for row in tamper["attacks"])
    require({"overlap_promoted", "overlap_savings_additive",
             "temporal_kill_removed", "system_admission_true"} <= names,
            "producer guard tamper coverage")
    require(result["claim_boundary"] == receipt["claim_boundary"] and
            result["m53_overlap_reconciliation"] ==
            receipt["m53_overlap_reconciliation"] and
            result["kill_gates"] == receipt["kill_gates"],
            "producer receipt semantic copies")


def validate_population_and_classification(review, result, manifest,
                                           operator_rows):
    require(result["population"] == {
        "manifest_records": 310, "operator": "Linear",
        "raw_payload_sha_size_popcount_checked": True, "samples": 10,
        "target_modules": 24, "target_records": 240}, "population")
    targets = [row for row in manifest["records"]
               if int(row["module_index"]) in set(range(6, 30))]
    targets.sort(key=lambda row: (int(row["sample_id"]),
                                  int(row["module_index"])))
    require(len(manifest["records"]) == 310 and len(targets) == 240 and
            all(row["operator"] == "Linear" for row in targets),
            "manifest Linear population")
    require([(int(row["sample_id"]), int(row["module_index"]))
             for row in targets] == [(sample, module)
                                     for sample in range(10)
                                     for module in range(6, 30)],
            "10x24 Cartesian identity")
    expected_indices = {
        "ffn_expand": [6, 8, 11, 13, 16, 18, 20, 22, 24, 26],
        "ffn_contract": [7, 9, 12, 14, 17, 19, 21, 23, 25, 27, 28, 29],
        "downsample": [10, 15],
    }
    expected_cycles = {
        "ffn_expand": 100895624,
        "ffn_contract": 41413997,
        "downsample": 12321697,
    }
    modules = dict((row["module_index"], row) for row in result["per_module"])
    require(set(modules) == set(range(6, 30)), "module index set")
    captured = 0
    observed_indices = dict((name, []) for name in expected_indices)
    observed_cycles = dict((name, 0) for name in expected_indices)
    for index in range(6, 30):
        row = modules[index]
        name = row["module_name"]
        expected_category = category(name)
        require(name in operator_rows and row["category"] == expected_category ==
                operator_rows[name]["category"], "module category {}".format(index))
        baseline = int(operator_rows[name]["activity_cycles_at_config_lanes"])
        require(row["m39_activity_cycles_at_config_lanes"] == baseline,
                "module baseline {}".format(index))
        observed_indices[expected_category].append(index)
        observed_cycles[expected_category] += baseline
        captured += baseline
    require(observed_indices == expected_indices and
            observed_cycles == expected_cycles and captured == 154631318,
            "24-Linear category totals")
    review_classes = review["independent_recomputation"][
        "module_classification"]
    for name in expected_indices:
        require(review_classes[name]["module_indices"] == expected_indices[name] and
                review_classes[name]["captured_m39_activity_cycles"] ==
                expected_cycles[name], "review category {}".format(name))
    for ordinal, (source, row) in enumerate(zip(targets, result["per_record"])):
        require(row["ordinal"] == ordinal and
                row["sample_id"] == int(source["sample_id"]) and
                row["module_index"] == int(source["module_index"]) and
                row["module_name"] == source["name"] and
                row["file_sha256"] == source["file_sha256"] and
                row["category"] == category(source["name"]),
                "record identity {}".format(ordinal))
    return {"indices": observed_indices, "cycles": observed_cycles,
            "captured_total": captured}


def validate_k_arithmetic(result):
    violations = 0
    for row in result["per_record"]:
        for mode in ("spatial", "temporal"):
            source_cycles = []
            integrated_cycles = []
            identity_source = row["mode_identity"][mode]["source_bits"]
            for k in (1, 2, 4):
                item = row["configs"]["{}_K{}".format(mode, k)]
                geometry = item["geometry"]
                work = item["source_work"]
                union = item["union"]
                tx = item["transactions"]
                source = item["cycles"]["source_issue"]
                integrated = item["cycles"]["serialized_integrated_no_overlap"]
                require(work["source_bits"] == identity_source and
                        work["product_updates"] == identity_source *
                        geometry["output_channels"] and
                        work["physical_product_slots"] == source * 8 * 96 * k,
                        "per-record work equation")
                require(union["source_issue_cycles"] == source and
                        union["source_union_indices"] ==
                        union["source_bank_read_transactions"] ==
                        tx["source_bank_reads"], "per-record union equation")
                expected_integrated = sum(tx[field] for field in (
                    "weight_dma_256b", "current_activation_selector_256b",
                    "candidate_parent_activation_selector_256b",
                    "choice_metadata_write_256b", "group_descriptor",
                    "chosen_parent_output_seed_vector", "final_commit_vector")) + source
                require(integrated == expected_integrated and
                        item["cycles"]["overlap_credit"] == 0,
                        "per-record serialized equation")
                source_cycles.append(source)
                integrated_cycles.append(integrated)
            if not (source_cycles[0] >= source_cycles[1] >= source_cycles[2] and
                    integrated_cycles[0] >= integrated_cycles[1] >=
                    integrated_cycles[2]):
                violations += 1
    require(violations == 0, "K1/K2/K4 monotonicity")

    expected = {
        "spatial_K1": (247950609, 426929343, 25488370, 43290601),
        "spatial_K2": (189290642, 345709376, 19455898, 35002129),
        "spatial_K4": (137719990, 282858724, 14117338, 28535569),
        "temporal_K1": (285144242, 427326627, 28700846, 42924787),
        "temporal_K2": (204197723, 323820108, 20707842, 32666553),
        "temporal_K4": (141812576, 250154961, 14414724, 25245435),
    }
    per_sample = dict((row["sample_id"], row) for row in result["per_sample"])
    for config in CONFIGS:
        source_values = []
        integrated_values = []
        for sample_id in range(10):
            rows = [row for row in result["per_record"]
                    if row["sample_id"] == sample_id]
            source = sum(row["configs"][config]["cycles"]["source_issue"]
                         for row in rows)
            integrated = sum(row["configs"][config]["cycles"][
                "serialized_integrated_no_overlap"] for row in rows)
            require(per_sample[sample_id]["configs"][config][
                "source_issue_cycles"] == source and
                per_sample[sample_id]["configs"][config][
                "serialized_integrated_no_overlap_cycles"] == integrated,
                "per-sample aggregate {}".format(config))
            source_values.append(source)
            integrated_values.append(integrated)
        source_dist = distribution(source_values)
        integrated_dist = distribution(integrated_values)
        aggregate = result["aggregate_configurations"][config]
        require(aggregate["source_cycle_distribution"] == source_dist and
                aggregate["serialized_integrated_cycle_distribution"] ==
                integrated_dist, "aggregate distribution {}".format(config))
        exp = expected[config]
        require(source_dist["sum"] == exp[0] and
                integrated_dist["sum"] == exp[1] and
                source_dist["p95_nearest_rank"] == exp[2] and
                integrated_dist["p95_nearest_rank"] == exp[3],
                "aggregate anchors {}".format(config))
    ratios = {}
    for mode in ("spatial", "temporal"):
        k1 = result["aggregate_configurations"][mode + "_K1"]
        for k in (2, 4):
            item = result["aggregate_configurations"][
                "{}_K{}".format(mode, k)]
            source_fraction = Fraction(k1["source_cycle_distribution"]["sum"],
                                       item["source_cycle_distribution"]["sum"])
            integrated_fraction = Fraction(
                k1["serialized_integrated_cycle_distribution"]["sum"],
                item["serialized_integrated_cycle_distribution"]["sum"])
            stored = item["ratios_not_system_speedup"]
            check_fraction(stored["k1_over_k_source_issue"], source_fraction,
                           "{} K{} source".format(mode, k))
            check_fraction(stored["k1_over_k_serialized_integrated"],
                           integrated_fraction,
                           "{} K{} integrated".format(mode, k))
            if k == 4:
                ratios[mode] = {
                    "source": float(source_fraction),
                    "serialized": float(integrated_fraction),
                }
    return {"expected_sums_and_p95": expected,
            "K1_over_K4_ratio_of_totals": ratios,
            "monotonic_records": 240}


def validate_capacity(contract, result):
    model = contract["capacity_model"]
    context_bits = (int(model["resident_contexts"]) - 1).bit_length()
    payload_bits = 4 * context_bits + 3 + 8 + 4 * 8 * 2 + 1
    aligned = ((payload_bits + 63) // 64) * 8
    base = (24576 + 4560 + 640 + 136800 + 1280 + 3904 +
            16 * (228 + 64) + 16 * aligned)
    require(payload_bits == 92 and aligned == 16 and base == 176688,
            "K4 fixed capacity base")
    feasible = []
    infeasible = []
    for module in result["per_module"]:
        index = module["module_index"]
        channels = int(module["input_shape"][-1])
        vector_bytes = (channels + 7) // 8
        temporal_combined = base + 300 * vector_bytes
        spatial_combined = base + 21 * vector_bytes
        temporal = module["capacities"]["temporal_K4"]
        spatial = module["capacities"]["spatial_K4"]
        require(temporal["combined_local_capacity_bytes"] == temporal_combined and
                temporal["local_capacity_headroom_bytes"] ==
                193728 - temporal_combined and
                temporal["passes_without_external_state_spill"] ==
                (temporal_combined <= 193728), "temporal capacity {}".format(index))
        require(spatial["combined_local_capacity_bytes"] == spatial_combined and
                spatial["passes_without_external_state_spill"] is True,
                "spatial capacity {}".format(index))
        (feasible if temporal_combined <= 193728 else infeasible).append(index)
    require(feasible == [6, 7, 8, 9, 10, 11, 13, 16, 18, 20, 22, 24, 26] and
            infeasible == [12, 14, 15, 17, 19, 21, 23, 25, 27, 28, 29],
            "temporal 13/24 split")
    gates = result["kill_gates"]
    require(gates == {
        "joint_m53_m63_ratio_killed_by_overlap_unknown": True,
        "spatial_all24_fit_without_external_state": True,
        "spatial_capacity_infeasible_modules": 0,
        "temporal_all24_fit_without_external_state": False,
        "temporal_capacity_infeasible_modules": 11,
        "temporal_headline_killed_by_external_state_requirement": True},
        "capacity kill gates")
    return {"base_bytes": base, "temporal_feasible_indices": feasible,
            "temporal_infeasible_indices": infeasible}


def validate_amdahl_and_overlap(contract, result, m53):
    amdahl = result["m39_amdahl"]
    fixed = 620868243
    captured = 154631318
    outside = fixed - captured
    removable = 162059820
    after_both = outside - removable
    require(outside == 466236925 and after_both == 304177105,
            "Amdahl integer anchors")
    require(amdahl["fixed_compute_reference_cycles"] == fixed and
            amdahl["captured_linear_baseline_cycles"] == captured and
            amdahl["fixed_outside_captured_linear_cycles"] == outside and
            amdahl["m39_noneligible_plus_qk_cycles"] == removable and
            amdahl["system_speedup_admitted"] is False, "Amdahl ledger")
    check_fraction(amdahl[
        "captured_linear_zero_cycle_amdahl_ceiling_not_system_speedup"],
        Fraction(fixed, outside), "zero Linear ceiling")
    check_fraction(amdahl[
        "zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk_ceiling_not_system_speedup"],
        Fraction(fixed, after_both), "both-zero ceiling")
    for config in CONFIGS:
        p95 = result["aggregate_configurations"][config][
            "serialized_integrated_cycle_distribution"]["p95_nearest_rank"]
        item = amdahl["candidate_p95_compositions"][config]
        denominator = outside + p95
        require(item["captured_linear_replacement_p95_cycles"] == p95 and
                item["conditional_total_cycles_using_p95"] == denominator and
                item["system_speedup_admitted"] is False,
                "Amdahl candidate {}".format(config))
        check_fraction(item[
            "conditional_fixed_over_total_ratio_not_system_speedup"],
            Fraction(fixed, denominator), "Amdahl candidate {}".format(config))
    targets = dict((row["name"], row) for row in amdahl["targets"])
    for name, ratio in (("3.2x", Fraction(16, 5)),
                        ("3.45x", Fraction(69, 20))):
        item = targets[name]
        ceiling = Fraction(fixed, 1) / ratio
        check_fraction(item["target_speedup"], ratio, name + " target")
        check_fraction(item["maximum_total_cycles"], ceiling,
                       name + " total ceiling")
        check_fraction(item[
            "still_required_savings_after_zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk"],
            Fraction(after_both, 1) - ceiling, name + " remaining")
        require(item["system_speedup_admitted"] is False,
                "target admission {}".format(name))

    overlap = result["m53_overlap_reconciliation"]
    components = overlap["m53_exact_denominator_components_cycles"]
    require(m53["conditional_frozen_compute_model"][
                "conditional_total_cycles"] == 201259510 and
            m53["conditional_frozen_compute_model"][
                "pair_p95_nearest_rank_cycles"] == 9798504 and
            components == {"outside_four_bottleneck_model": 188824491,
                           "fixed_late_scale_plus_frontend": 2636515,
                           "pair_p95": 9798504, "total": 201259510} and
            sum(value for key, value in components.items() if key != "total") ==
            components["total"], "M53 denominator components")
    require("188824491" not in json.dumps(m53, sort_keys=True),
            "M53 unexpectedly gained outside-term decomposition/value")
    require(overlap["status"] ==
            "OVERLAP_UNKNOWN_M63_FIXED_BASELINE_SAVINGS_NOT_ADDITIVE_TO_M53" and
            overlap["savings_admitted_as_additive_to_m53"] == [] and
            overlap["joint_ratio_admitted"] is False and
            result["kill_gates"][
                "joint_m53_m63_ratio_killed_by_overlap_unknown"] is True,
            "M53 overlap top-level guard")
    scenarios = overlap["scenarios"]
    for config, replacement in (("spatial_K4", 28535569),
                                ("temporal_K4", 25245435)):
        item = scenarios[config]
        savings = captured - replacement
        prohibited_denominator = 201259510 - savings
        require(item["m63_fixed_baseline_cycles"] == captured and
                item["m63_replacement_p95_cycles"] == replacement and
                item["m63_naive_fixed_baseline_savings_cycles"] == savings and
                item["naive_subtraction_from_m53_denominator_cycles_prohibited"] ==
                prohibited_denominator and
                item["L24_inherited_inside_188824491"] ==
                "UNKNOWN_NOT_OPERATOR_DECOMPOSED" and
                item["additive_savings_admitted"] is False and
                item["joint_ratio_admitted"] is False,
                "M53 overlap scenario {}".format(config))
        check_fraction(item["naive_joint_ratio_prohibited"],
                       Fraction(fixed, prohibited_denominator),
                       "prohibited naive ratio {}".format(config))
    forbidden = " ".join(result["claim_boundary"]["forbidden"]).lower()
    require("subtracting m63 fixed-baseline savings" in forbidden and
            "system speedup" in forbidden and
            result["qualification"]["system_speedup_admitted"] is False,
            "M53/system claim boundary")
    return {
        "fixed": fixed, "captured": captured, "outside": outside,
        "after_both_zero": after_both,
        "zero_linear_ceiling": float(Fraction(fixed, outside)),
        "both_zero_ceiling": float(Fraction(fixed, after_both)),
        "joint_ratio_admitted": False,
    }


def guard_errors(contract, result, producer_receipt, review):
    errors = []
    if review.get("headline") is not False:
        errors.append("review headline promotion")
    if review.get("system_speedup") is not False:
        errors.append("review system speedup promotion")
    if result.get("qualification", {}).get("system_speedup_admitted") is not False:
        errors.append("result system speedup promotion")
    overlap = result.get("m53_overlap_reconciliation", {})
    if overlap.get("joint_ratio_admitted") is not False:
        errors.append("joint ratio promotion")
    if overlap.get("savings_admitted_as_additive_to_m53") != []:
        errors.append("additive savings promotion")
    if any(item.get("additive_savings_admitted") is not False or
           item.get("joint_ratio_admitted") is not False
           for item in overlap.get("scenarios", {}).values()):
        errors.append("scenario overlap promotion")
    if result.get("kill_gates", {}).get(
            "temporal_headline_killed_by_external_state_requirement") is not True:
        errors.append("temporal capacity kill removal")
    if contract.get("m53_overlap_reconciliation", {}).get(
            "joint_ratio_admitted") is not False:
        errors.append("contract joint ratio promotion")
    if producer_receipt.get("method", {}).get(
            "system_speedup_admitted") is not False:
        errors.append("receipt system speedup promotion")
    return errors


def run_independent_tampers(contract, result, producer_receipt, review):
    mutations = []
    item = copy.deepcopy(review); item["headline"] = True
    mutations.append(("review_headline_promotion", contract, result,
                      producer_receipt, item))
    item = copy.deepcopy(review); item["system_speedup"] = True
    mutations.append(("review_system_speedup_promotion", contract, result,
                      producer_receipt, item))
    item = copy.deepcopy(result); item["qualification"][
        "system_speedup_admitted"] = True
    mutations.append(("result_system_speedup_promotion", contract, item,
                      producer_receipt, review))
    item = copy.deepcopy(result); item["m53_overlap_reconciliation"][
        "joint_ratio_admitted"] = True
    mutations.append(("joint_ratio_promotion", contract, item,
                      producer_receipt, review))
    item = copy.deepcopy(result); item["m53_overlap_reconciliation"][
        "savings_admitted_as_additive_to_m53"] = ["spatial_K4"]
    mutations.append(("additive_savings_promotion", contract, item,
                      producer_receipt, review))
    item = copy.deepcopy(result); item["m53_overlap_reconciliation"][
        "scenarios"]["spatial_K4"]["additive_savings_admitted"] = True
    mutations.append(("scenario_additive_promotion", contract, item,
                      producer_receipt, review))
    item = copy.deepcopy(result); item["kill_gates"][
        "temporal_headline_killed_by_external_state_requirement"] = False
    mutations.append(("temporal_capacity_kill_removal", contract, item,
                      producer_receipt, review))
    item = copy.deepcopy(contract); item["m53_overlap_reconciliation"][
        "joint_ratio_admitted"] = True
    mutations.append(("contract_joint_ratio_promotion", item, result,
                      producer_receipt, review))
    item = copy.deepcopy(producer_receipt); item["method"][
        "system_speedup_admitted"] = True
    mutations.append(("receipt_system_speedup_promotion", contract, result,
                      item, review))
    receipts = []
    for name, con, res, prod, rev in mutations:
        errors = guard_errors(con, res, prod, rev)
        require(errors, "independent tamper survived {}".format(name))
        receipts.append({"name": name, "result": "REJECTED",
                         "diagnostic": errors[0]})
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args()
    require(not arguments.receipt.exists(), "refusing existing independent receipt")
    review = strict_json(arguments.review)
    bindings = validate_review(review, arguments.review)
    contract = strict_json(ROOT / CONTRACT_REL)
    result = strict_json(ROOT / RESULT_REL)
    producer_receipt = strict_json(ROOT / PRODUCER_RECEIPT_REL)
    tamper = strict_json(ROOT / TAMPER_RECEIPT_REL)
    manifest = strict_json(ROOT / MANIFEST_REL)
    m53 = strict_json(ROOT / M53_REL)
    with OPERATOR_CSV.open("r", encoding="utf-8") as handle:
        operator_rows = dict((row["name"], row) for row in csv.DictReader(handle))
    require(result["identity"]["contract_sha256"] == sha256_path(ROOT / CONTRACT_REL) and
            result["identity"]["analyzer_sha256"] ==
            sha256_path(ROOT / "system_simulator/scripts/analyze_m63_linear_k4_spatiotemporal_full_network_opportunity.py"),
            "result exact-SHA identity")
    require(contract["claim_boundary"] == result["claim_boundary"],
            "contract/result claim boundary")
    validate_producer_receipt(producer_receipt, result, tamper)
    classification = validate_population_and_classification(
        review, result, manifest, operator_rows)
    k_summary = validate_k_arithmetic(result)
    capacity_summary = validate_capacity(contract, result)
    amdahl_summary = validate_amdahl_and_overlap(contract, result, m53)
    require(not guard_errors(contract, result, producer_receipt, review),
            "untampered guard failure")
    independent_tampers = run_independent_tampers(
        contract, result, producer_receipt, review)
    receipt = {
        "schema": "m63_independent_hammer_validation_receipt_v1",
        "status": "PASS_INDEPENDENT_EXACT_SHA_ARITHMETIC_CAPACITY_AND_OVERLAP_GUARD",
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "review": {"path": str(arguments.review.resolve()),
                   "sha256": sha256_path(arguments.review)},
        "validator": {"path": str(Path(__file__).resolve()),
                      "sha256": sha256_path(Path(__file__).resolve())},
        "producer_bindings_sha256": bindings,
        "producer_receipt_sha256": sha256_path(ROOT / PRODUCER_RECEIPT_REL),
        "classification_recomputed": classification,
        "k_arithmetic_recomputed": k_summary,
        "capacity_recomputed": capacity_summary,
        "amdahl_and_overlap_recomputed": amdahl_summary,
        "producer_tamper_attacks_rejected": 22,
        "independent_tamper_attacks": independent_tampers,
        "date_oriented_score_0_to_100": 63,
        "issues": {"P0": 0, "P1": 4, "P2": 3},
        "headline": False,
        "system_speedup": False,
        "claim_boundary": review["admission"],
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    with arguments.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "receipt_sha256": sha256_path(arguments.receipt),
        "score": 63,
        "status": receipt["status"],
        "headline": False,
        "system_speedup": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
