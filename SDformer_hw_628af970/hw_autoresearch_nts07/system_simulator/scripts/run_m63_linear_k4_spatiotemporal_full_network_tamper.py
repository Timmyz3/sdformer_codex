#!/usr/bin/env python3
"""Semantic negative tests for the M63 validator core."""

from __future__ import print_function

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path


def load_module(path):
    spec = importlib.util.spec_from_file_location("m63_validator", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validator", "contract", "manifest", "m52-result",
                 "m53-result", "m55-result", "m39-result",
                 "operator-transactions", "dual-line-contract", "result",
                 "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    arguments = parser.parse_args()
    validator = load_module(arguments.validator)
    documents = {
        "contract": validator.strict_json(arguments.contract),
        "manifest": validator.strict_json(arguments.manifest),
        "m52": validator.strict_json(arguments.m52_result),
        "m53": validator.strict_json(arguments.m53_result),
        "m55": validator.strict_json(arguments.m55_result),
        "m39": validator.strict_json(arguments.m39_result),
        "dual": validator.strict_json(arguments.dual_line_contract),
        "result": validator.strict_json(arguments.result),
    }
    with arguments.operator_transactions.open("r", encoding="utf-8") as handle:
        operator_rows = dict((row["name"], row) for row in csv.DictReader(handle))

    attacks = []

    def attack(name, mutation):
        docs = copy.deepcopy(documents)
        ops = copy.deepcopy(operator_rows)
        mutation(docs, ops)
        rejected = False
        diagnostic = ""
        try:
            validator.validate_core(
                docs["contract"], docs["manifest"], docs["m52"], docs["m53"],
                docs["m55"], docs["m39"], docs["dual"], ops, docs["result"])
        except Exception as error:
            rejected = True
            diagnostic = "{}: {}".format(type(error).__name__, error)
        attacks.append({"name": name, "rejected": rejected,
                        "diagnostic": diagnostic})

    attack("status_promoted_to_system", lambda d, o: d["result"].__setitem__(
        "status", "PASS_SYSTEM_SPEEDUP"))
    attack("population_239", lambda d, o: d["result"]["population"].__setitem__(
        "target_records", 239))
    attack("record_file_sha", lambda d, o: d["result"]["per_record"][0].__setitem__(
        "file_sha256", "0" * 64))
    attack("record_source_bits", lambda d, o: d["result"]["per_record"][0][
        "mode_identity"]["spatial"].__setitem__("source_bits", 0))
    attack("record_signed_conservation", lambda d, o: d["result"]["per_record"][0][
        "mode_identity"]["spatial"].__setitem__(
            "negative_1_to_0_source_bits", 0))
    attack("record_choice_population", lambda d, o: d["result"]["per_record"][0][
        "mode_identity"]["spatial"]["choice_counts"].__setitem__("zero", 0))
    attack("k4_source_drift", lambda d, o: d["result"]["per_record"][0][
        "configs"]["spatial_K4"]["source_work"].__setitem__("source_bits", 1))
    attack("k4_product_equation", lambda d, o: d["result"]["per_record"][0][
        "configs"]["spatial_K4"]["source_work"].__setitem__("product_updates", 1))
    attack("k4_physical_slots", lambda d, o: d["result"]["per_record"][0][
        "configs"]["spatial_K4"]["source_work"].__setitem__(
            "physical_product_slots", 1))
    attack("union_bank_reads", lambda d, o: d["result"]["per_record"][0][
        "configs"]["spatial_K4"]["union"].__setitem__(
            "source_bank_read_transactions", 0))
    attack("integrated_cycle", lambda d, o: d["result"]["per_record"][0][
        "configs"]["spatial_K4"]["cycles"].__setitem__(
            "serialized_integrated_no_overlap", 0))
    attack("capacity_pass", lambda d, o: d["result"]["per_record"][0][
        "configs"]["temporal_K4"]["capacity"].__setitem__(
            "passes_without_external_state_spill", False))
    attack("module_baseline", lambda d, o: d["result"]["per_module"][0].__setitem__(
        "m39_activity_cycles_at_config_lanes", 0))
    attack("operator_csv_baseline", lambda d, o: o[
        d["result"]["per_module"][0]["module_name"]].__setitem__(
            "activity_cycles_at_config_lanes", "0"))
    attack("aggregate_spatial_p95", lambda d, o: d["result"][
        "aggregate_configurations"]["spatial_K4"][
            "source_cycle_distribution"].__setitem__("p95_nearest_rank", 0))
    attack("amdahl_captured", lambda d, o: d["result"]["m39_amdahl"].__setitem__(
        "captured_linear_baseline_cycles", 0))
    attack("amdahl_target_gap", lambda d, o: d["result"]["m39_amdahl"][
        "targets"][0][
            "still_required_savings_after_zero_cycle_captured_linear_and_zero_M39_noneligible_plus_qk"].__setitem__(
                "float", 0.0))
    attack("overlap_promoted", lambda d, o: d["result"][
        "m53_overlap_reconciliation"].__setitem__("joint_ratio_admitted", True))
    attack("overlap_savings_additive", lambda d, o: d["result"][
        "m53_overlap_reconciliation"].__setitem__(
            "savings_admitted_as_additive_to_m53", ["spatial_K4"]))
    attack("temporal_kill_removed", lambda d, o: d["result"]["kill_gates"].__setitem__(
        "temporal_headline_killed_by_external_state_requirement", False))
    attack("system_admission_true", lambda d, o: d["result"][
        "qualification"].__setitem__("system_speedup_admitted", True))
    attack("claim_boundary_weakened", lambda d, o: d["result"][
        "claim_boundary"].__setitem__("forbidden", []))

    if not all(row["rejected"] for row in attacks):
        raise ValueError("one or more semantic tampers survived")
    payload = {
        "schema": "m63_linear_k4_spatiotemporal_full_network_tamper_receipt_v1",
        "status": "PASS_ALL_SEMANTIC_TAMPERS_REJECTED",
        "attack_count": len(attacks),
        "rejected_count": sum(1 for row in attacks if row["rejected"]),
        "attacks": attacks,
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    print(json.dumps({"attacks": len(attacks), "status": payload["status"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
