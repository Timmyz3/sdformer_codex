#!/usr/bin/env python3
"""Independent ledger and physical-assumption hammer for M45-r2."""

from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
RESULT_DIR = HW_ROOT / (
    "results/m45_dual_destination_bank_fused_integrated_schedule_r2_20260823")
CANONICAL = RESULT_DIR / "m45_r2_context8_primary_schedule.json"
CONTRACT = HW_ROOT / (
    "contracts/m45_dual_destination_bank_fused_integrated_schedule_contract_r2_20260823.json")
ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_r2_context8_primary_schedule.py")
R1_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
PRODUCER_VALIDATOR = HW_ROOT / (
    "system_simulator/scripts/validate_m45_r2_context8_primary_schedule.py")
MANIFEST = HW_ROOT / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
M43_RESULT = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatial_parent_delta_schedule_final.json")
TARGETED_REPLAY = RESULT_DIR / "m45_r2_independent_targeted_replay_samples3_7.json"

EXPECTED = {
    "canonical": "0f16e75601fdb18f31f9bc36f6aae8a17a9e62a20f5c07e18226562e9ba0d37c",
    "contract": "1c547c3ecd5d82c5dc8217297f19ca730748ac9526663f5449d8f13d867cd6b4",
    "analyzer": "1b07e6efea778561605f7a89d03505c3610ec96c19c21b278c347c2cf8d90885",
    "r1_analyzer": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "producer_validator": "2b81ce08bfef7113b25df9da4d23275f8ba207889a7c2cf7622cfb760a17db03",
}

MAX_FIELDS = ("maximum_metadata_occupancy", "maximum_complete_occupancy",
              "maximum_resident_occupancy")
RECORD_SUM_FIELDS = (
    "source_only_cycles", "integrated_cycles", "logical_source_updates",
    "unique_weight_issues", "descriptor_commands", "parent_partial_reads",
    "parent_partial_writes", "final_accumulator_reads",
    "final_accumulator_writes", "completed_outputs", "fusion_groups",
    "zero_source_groups", "parent_wait_cycles",
    "command_or_state_wait_cycles", "response_or_context_wait_cycles",
    "weight_dma_wait_cycles", "fusion_hold_wait_cycles", "late_join_groups",
    "signed_add_updates", "signed_subtract_updates", "weight_dma_bytes",
    "final_accumulator_read_bytes", "final_accumulator_write_bytes",
    "completed_output_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
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


def fraction(numerator, denominator):
    require(denominator > 0, "zero fraction denominator")
    return {"numerator": numerator, "denominator": denominator}


def distribution(values):
    ordered = sorted(values)
    require(ordered, "empty distribution")

    def nearest_rank(percent):
        rank = (percent * len(ordered) + 99) // 100
        return ordered[rank - 1]
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean_exact": fraction(sum(ordered), len(ordered)),
        "p50_nearest_rank": nearest_rank(50),
        "p95_nearest_rank": nearest_rank(95),
        "p99_nearest_rank": nearest_rank(99),
    }


def by_name(result):
    value = {}
    for config in result["configurations"]:
        require(config["name"] not in value, "duplicate configuration")
        value[config["name"]] = config
    return value


def validate_core(result, contract, manifest, m43):
    require(result["schema"] ==
            "m45_dual_destination_bank_fused_integrated_schedule_result_v2",
            "result schema drift")
    require(result["population"] == {"samples": 10, "operators": 4,
                                      "records": 40},
            "population drift")
    configs = by_name(result)
    expected_names = set(("K1_CTX4_REPRODUCTION", "K2_CTX8_PRIMARY",
                          "K2_CTX4_CAPACITY_ABLATION",
                          "K4_CTX4_KILLED_ABLATION"))
    require(set(configs) == expected_names, "configuration set drift")
    manifest_keys = set((row["sample_id"], row["operator"])
                        for row in manifest["records"])
    require(len(manifest_keys) == 40, "manifest key population drift")
    m43_by_key = dict(((row["sample_id"], row["operator"]), row)
                      for row in m43["records"])
    require(set(m43_by_key) == manifest_keys, "M43/manifest identity drift")

    fixed_record = {
        "descriptor_commands": 648000,
        "parent_partial_writes": 648000,
        "final_accumulator_reads": 624000,
        "final_accumulator_writes": 648000,
        "completed_outputs": 24000,
        "weight_dma_bytes": 53084160,
        "final_accumulator_read_bytes": 179712000,
        "final_accumulator_write_bytes": 186624000,
        "completed_output_bytes": 6912000,
    }
    reference_signed = {}
    for name, config in configs.items():
        records = config["records"]
        require(len(records) == 40, "record population drift: {}".format(name))
        record_map = {}
        for row in records:
            key = (row["sample_id"], row["operator"])
            require(key not in record_map, "duplicate record key")
            record_map[key] = row
            require(row["logical_source_updates"] ==
                    row["signed_add_updates"] + row["signed_subtract_updates"],
                    "signed source conservation drift")
            for field, expected_value in fixed_record.items():
                require(row[field] == expected_value,
                        "fixed record service/traffic drift: {}".format(field))
            require(row["maximum_complete_occupancy"] <= 16,
                    "complete FIFO overflow")
            require(row["maximum_resident_occupancy"] <=
                    config["resident_contexts"], "resident overflow")
            signed = (row["logical_source_updates"],
                      row["signed_add_updates"], row["signed_subtract_updates"],
                      row["parent_partial_reads"])
            if key in reference_signed:
                require(reference_signed[key] == signed,
                        "cross-configuration logical workload drift")
            else:
                reference_signed[key] = signed
        require(set(record_map) == manifest_keys, "record cohort drift")

        samples = config["per_sample"]
        require([row["sample_id"] for row in samples] == list(range(10)),
                "sample identity/order drift")
        for sample in samples:
            selected = [row for row in records
                        if row["sample_id"] == sample["sample_id"]]
            require(len(selected) == 4, "sample/operator reconciliation drift")
            for field in RECORD_SUM_FIELDS:
                require(sample[field] == sum(row[field] for row in selected),
                        "sample sum drift: {} {}".format(name, field))
            for field in MAX_FIELDS:
                require(sample[field] == max(row[field] for row in selected),
                        "sample max drift: {} {}".format(name, field))
            require(sample["integrated_over_source_only"] == fraction(
                sample["integrated_cycles"] - sample["source_only_cycles"],
                sample["source_only_cycles"]), "sample overhead fraction drift")
            require(sample["parent_wait_fraction"] == fraction(
                sample["parent_wait_cycles"], sample["integrated_cycles"]),
                "sample parent fraction drift")

        source_values = [row["source_only_cycles"] for row in samples]
        integrated_values = [row["integrated_cycles"] for row in samples]
        require(config["source_only_cycle_distribution"] ==
                distribution(source_values), "source distribution drift")
        require(config["integrated_cycle_distribution"] ==
                distribution(integrated_values), "integrated distribution drift")
        total_source = sum(source_values)
        total_integrated = sum(integrated_values)
        require(config["aggregate_source_only_cycles"] == total_source and
                config["aggregate_integrated_cycles"] == total_integrated,
                "aggregate cycle drift")
        require(config["aggregate_integrated_over_source_only"] == fraction(
            total_integrated - total_source, total_source),
            "aggregate overhead fraction drift")
        require(config["aggregate_parent_wait_fraction"] == fraction(
            sum(row["parent_wait_cycles"] for row in samples), total_integrated),
            "aggregate parent fraction drift")
        for aggregate, field in (
                ("aggregate_logical_source_updates", "logical_source_updates"),
                ("aggregate_unique_weight_issues", "unique_weight_issues"),
                ("aggregate_fusion_groups", "fusion_groups")):
            require(config[aggregate] == sum(row[field] for row in samples),
                    "aggregate workload drift: {}".format(aggregate))
        traffic_map = {
            "weight_dma": "weight_dma_bytes",
            "final_accumulator_read": "final_accumulator_read_bytes",
            "final_accumulator_write": "final_accumulator_write_bytes",
            "completed_output": "completed_output_bytes",
        }
        for short, field in traffic_map.items():
            values = set(row[field] for row in samples)
            require(len(values) == 1 and
                    config["traffic_bytes_per_sample"][short] in values,
                    "per-sample traffic drift: {}".format(short))

    k1 = configs["K1_CTX4_REPRODUCTION"]
    for row in k1["records"]:
        key = (row["sample_id"], row["operator"])
        require(row["source_only_cycles"] ==
                m43_by_key[key]["parent_delta_p8_l96_source_issue_cycles"],
                "K1/M43 per-record reproduction drift")
        require(row["unique_weight_issues"] == row["logical_source_updates"],
                "K1 unique/source identity drift")
    require(k1["aggregate_source_only_cycles"] == 116376872,
            "K1 aggregate reproduction drift")

    primary = configs["K2_CTX8_PRIMARY"]
    ctx4 = configs["K2_CTX4_CAPACITY_ABLATION"]
    k4 = configs["K4_CTX4_KILLED_ABLATION"]
    gates = contract["kill_gates"]
    overhead = gates["maximum_primary_integrated_over_source_only_fraction"]
    parent = gates["maximum_primary_parent_wait_fraction"]
    reduction = gates["minimum_primary_integrated_reduction_vs_k1_fraction"]
    justify = gates["minimum_ctx8_p95_improvement_over_ctx4_fraction"]
    calculated = {}
    calculated["primary_all_samples_integrated_over_source_only_le_10pct"] = all(
        (row["integrated_cycles"] - row["source_only_cycles"]) *
        overhead["denominator"] <= row["source_only_cycles"] * overhead["numerator"]
        for row in primary["per_sample"])
    calculated["primary_all_samples_parent_wait_le_5pct"] = all(
        row["parent_wait_cycles"] * parent["denominator"] <=
        row["integrated_cycles"] * parent["numerator"]
        for row in primary["per_sample"])
    k1_integrated = k1["aggregate_integrated_cycles"]
    primary_integrated = primary["aggregate_integrated_cycles"]
    calculated["primary_aggregate_integrated_reduction_vs_k1_ge_15pct"] = (
        (k1_integrated - primary_integrated) * reduction["denominator"] >=
        k1_integrated * reduction["numerator"])
    primary_p95 = primary["integrated_cycle_distribution"]["p95_nearest_rank"]
    ctx4_p95 = ctx4["integrated_cycle_distribution"]["p95_nearest_rank"]
    calculated["ctx8_p95_improvement_over_ctx4_ge_3pct"] = (
        (ctx4_p95 - primary_p95) * justify["denominator"] >=
        ctx4_p95 * justify["numerator"])
    calculated["primary_p95_integrated_cycles_le_15495075"] = (
        primary_p95 <= gates["maximum_primary_p95_integrated_cycles"])
    calculated["k4_ctx4_slower_than_k2_ctx8_and_killed"] = (
        k4["integrated_cycle_distribution"]["p95_nearest_rank"] >= primary_p95)
    for name, value in calculated.items():
        require(result["kill_gates"][name] is value,
                "stored kill gate drift: {}".format(name))
    require(result["kill_gates"]["ctx8_p95_improvement_over_ctx4"] ==
            fraction(ctx4_p95 - primary_p95, ctx4_p95),
            "stored CTX8 improvement fraction drift")
    require(result["kill_gates"]["all_kill_gates_pass"] is all(
        calculated.values()), "all-gates reduction drift")
    require(result["kill_gates"]["three_x_target_crossing_admitted"] is False,
            "improper 3x claim")
    return configs


def validate_capacity(result, m43):
    capacity = result["capacity"]
    require(capacity["context_vector_bytes"] == 96 * 3 and
            capacity["context_bytes_per_entry"] == 96 * 3 + 64,
            "context byte arithmetic drift")
    require(capacity["four_context_bytes"] == 4 * 352 and
            capacity["eight_context_bytes"] == 8 * 352 and
            capacity["extra_state_bytes_vs_four_contexts"] == 4 * 352,
            "context count byte arithmetic drift")
    bridge = m43["physical_layout_bridge"]
    require(bridge["local_scratch_bytes"] ==
            bridge["weight_double_buffer_bytes"] +
            bridge["parent_partial_buffer_bytes"] +
            bridge["support_parent_buffer_bytes"] +
            bridge["four_context_bytes"] == 56960,
            "M43 scratch decomposition drift")
    require(capacity["eight_context_local_scratch_bytes"] ==
            56960 + 1408 == 58368, "C8 scratch arithmetic drift")
    require(capacity["single_timestep_final_accumulator_bytes"] ==
            300 * 96 * 3 == 86400, "frame arithmetic drift")
    require(capacity["base_local_bytes_before_fifos"] ==
            58368 + 86400 == 144768, "base capacity arithmetic drift")
    require(capacity["metadata_fifo_storage_bytes"] == 16 * 64 == 1024 and
            capacity["complete_fifo_storage_bytes"] == 16 * (288 + 16) == 4864,
            "FIFO capacity arithmetic drift")
    require(capacity["combined_local_capacity_bytes"] ==
            144768 + 1024 + 4864 == 150656,
            "combined capacity arithmetic drift")
    require(capacity["local_capacity_headroom_bytes"] ==
            193728 - 150656 == 43072, "headroom arithmetic drift")


def run_mutation_matrix(canonical, contract, manifest, m43):
    attacks = []

    def attack(name, mutate):
        item = copy.deepcopy(canonical)
        mutate(item)
        rejected = False
        try:
            validate_core(item, contract, manifest, m43)
            validate_capacity(item, m43)
        except (ValueError, KeyError, TypeError, AssertionError):
            rejected = True
        require(rejected, "independent validator accepted attack: {}".format(name))
        attacks.append(name)

    attack("aggregate_integrated_cycle_plus_one",
           lambda d: d["configurations"][1].__setitem__(
               "aggregate_integrated_cycles",
               d["configurations"][1]["aggregate_integrated_cycles"] + 1))
    attack("record_signed_add_plus_one",
           lambda d: d["configurations"][1]["records"][0].__setitem__(
               "signed_add_updates",
               d["configurations"][1]["records"][0]["signed_add_updates"] + 1))
    attack("stored_p95_minus_one",
           lambda d: d["configurations"][1]["integrated_cycle_distribution"].__setitem__(
               "p95_nearest_rank",
               d["configurations"][1]["integrated_cycle_distribution"]["p95_nearest_rank"] - 1))
    attack("capacity_combined_plus_one",
           lambda d: d["capacity"].__setitem__(
               "combined_local_capacity_bytes",
               d["capacity"]["combined_local_capacity_bytes"] + 1))
    attack("duplicate_record_key",
           lambda d: d["configurations"][1]["records"][1].update({
               "sample_id": d["configurations"][1]["records"][0]["sample_id"],
               "operator": d["configurations"][1]["records"][0]["operator"]}))
    attack("complete_fifo_occupancy_17",
           lambda d: d["configurations"][1]["records"][0].__setitem__(
               "maximum_complete_occupancy", 17))
    attack("k1_source_reproduction_plus_one",
           lambda d: d["configurations"][0]["records"][0].__setitem__(
               "source_only_cycles",
               d["configurations"][0]["records"][0]["source_only_cycles"] + 1))
    attack("weight_traffic_plus_one",
           lambda d: d["configurations"][1]["records"][0].__setitem__(
               "weight_dma_bytes",
               d["configurations"][1]["records"][0]["weight_dma_bytes"] + 1))
    attack("ctx8_improvement_fraction_corruption",
           lambda d: d["kill_gates"].__setitem__(
               "ctx8_p95_improvement_over_ctx4", {"numerator": 1, "denominator": 1}))
    attack("three_x_claim_flip",
           lambda d: d["kill_gates"].__setitem__(
               "three_x_target_crossing_admitted", True))

    malformed_rejected = []
    with tempfile.TemporaryDirectory(prefix="m45_r2_json_attacks_") as tempdir:
        duplicate = Path(tempdir) / "duplicate.json"
        duplicate.write_text('{"x": 1, "x": 2}\n', encoding="utf-8")
        nan = Path(tempdir) / "nan.json"
        nan.write_text('{"x": NaN}\n', encoding="utf-8")
        for name, path in (("duplicate_json_key", duplicate),
                           ("nan_json_constant", nan)):
            try:
                read_json(path)
            except ValueError:
                malformed_rejected.append(name)
    require(len(malformed_rejected) == 2, "malformed JSON attack accepted")
    attacks += malformed_rejected
    return attacks


def producer_validator_gap(canonical):
    spec = importlib.util.spec_from_file_location(
        "m45_r2_producer_validator_gap", str(PRODUCER_VALIDATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    accepted = []
    mutations = []
    item = copy.deepcopy(canonical)
    item["configurations"][1]["records"][0]["signed_add_updates"] += 1
    mutations.append(("record_signed_add_plus_one", item))
    item = copy.deepcopy(canonical)
    item["configurations"][1]["integrated_cycle_distribution"][
        "p95_nearest_rank"] -= 1
    mutations.append(("stored_p95_minus_one", item))
    with tempfile.TemporaryDirectory(prefix="m45_r2_producer_gap_") as tempdir:
        for name, payload in mutations:
            path = Path(tempdir) / (name + ".json")
            path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                            encoding="utf-8")
            try:
                module.validate_result(path, require_frozen_sha=False)
            except ValueError:
                continue
            accepted.append(name)
    return accepted


def ratio_summary(configs):
    k1 = configs["K1_CTX4_REPRODUCTION"]
    primary = configs["K2_CTX8_PRIMARY"]
    ctx4 = configs["K2_CTX4_CAPACITY_ABLATION"]
    k4 = configs["K4_CTX4_KILLED_ABLATION"]
    k1_i = k1["aggregate_integrated_cycles"]
    p_i = primary["aggregate_integrated_cycles"]
    p_s = primary["aggregate_source_only_cycles"]
    c4_i = ctx4["aggregate_integrated_cycles"]
    primary_p95 = primary["integrated_cycle_distribution"]["p95_nearest_rank"]
    ctx4_p95 = ctx4["integrated_cycle_distribution"]["p95_nearest_rank"]
    k4_p95 = k4["integrated_cycle_distribution"]["p95_nearest_rank"]
    overheads = [(row["integrated_cycles"] - row["source_only_cycles"],
                  row["source_only_cycles"], row["sample_id"])
                 for row in primary["per_sample"]]
    parents = [(row["parent_wait_cycles"], row["integrated_cycles"],
                row["sample_id"]) for row in primary["per_sample"]]
    worst_overhead = max(overheads, key=lambda row: float(row[0]) / row[1])
    worst_parent = max(parents, key=lambda row: float(row[0]) / row[1])
    return {
        "k1_aggregate_integrated_cycles": k1_i,
        "k2_ctx8_aggregate_source_only_cycles": p_s,
        "k2_ctx8_aggregate_integrated_cycles": p_i,
        "k2_ctx8_integrated_speedup_vs_k1": fraction(k1_i, p_i),
        "k2_ctx8_integrated_cycle_reduction_vs_k1": fraction(k1_i - p_i, k1_i),
        "k2_ctx8_aggregate_over_source_only": fraction(p_i - p_s, p_s),
        "worst_per_sample_overhead": {
            "sample_id": worst_overhead[2],
            "fraction": fraction(worst_overhead[0], worst_overhead[1]),
        },
        "worst_per_sample_parent_wait": {
            "sample_id": worst_parent[2],
            "fraction": fraction(worst_parent[0], worst_parent[1]),
        },
        "ctx8_aggregate_integrated_reduction_vs_ctx4": fraction(c4_i - p_i, c4_i),
        "ctx8_p95_reduction_vs_ctx4": fraction(ctx4_p95 - primary_p95, ctx4_p95),
        "k4_p95_penalty_vs_ctx8": fraction(k4_p95 - primary_p95, primary_p95),
        "weight_dma_bytes_per_sample_unchanged":
            primary["traffic_bytes_per_sample"]["weight_dma"],
        "weight_dma_bytes_all10":
            primary["traffic_bytes_per_sample"]["weight_dma"] * 10,
    }


def build():
    for name, path in (("canonical", CANONICAL), ("contract", CONTRACT),
                       ("analyzer", ANALYZER), ("r1_analyzer", R1_ANALYZER),
                       ("producer_validator", PRODUCER_VALIDATOR)):
        require(sha256(path) == EXPECTED[name], "anchor drift: {}".format(name))
    canonical = read_json(CANONICAL)
    contract = read_json(CONTRACT)
    manifest = read_json(MANIFEST)
    m43 = read_json(M43_RESULT)
    replay = read_json(TARGETED_REPLAY)
    require(replay["status"] ==
            "PASS_EXACT_RECORD_REPLAY_K2_CTX8_SAMPLES_3_AND_7",
            "targeted replay is not PASS")
    require(replay["scope"]["sample_ids"] == [3, 7] and
            replay["scope"]["records"] == 8,
            "targeted replay scope drift")
    configs = validate_core(canonical, contract, manifest, m43)
    validate_capacity(canonical, m43)
    attacks = run_mutation_matrix(canonical, contract, manifest, m43)
    producer_gaps = producer_validator_gap(canonical)
    require(set(producer_gaps) ==
            set(("record_signed_add_plus_one", "stored_p95_minus_one")),
            "producer validator gap characterization drift")
    source = R1_ANALYZER.read_text(encoding="utf-8")
    require("min(16, len(ready))" in source,
            "metadata clamp evidence drift")
    require("maximum_live_before_group = COMPLETE_FIFO_ENTRIES - len(group)" in source,
            "atomic group-credit evidence drift")
    require("while complete_entries and complete_entries[0][0] <= response_ready" in source,
            "same-cycle retire evidence drift")
    raw_depths = [row["raw_spatial_dag_ready_depth"] for row in replay["records"]]
    require(max(raw_depths) > 16,
            "targeted replay did not expose the metadata clamp")
    complete_boundary_records = sum(
        1 for row in configs["K2_CTX8_PRIMARY"]["records"]
        if row["maximum_complete_occupancy"] == 16)
    return {
        "schema": "m45_r2_independent_hammer_review_v1",
        "date": "2026-08-23",
        "status": "GO_ALL10_TRANSACTION_SCHEDULE_METADATA_CAPACITY_NO_GO_PENDING_R3",
        "review": {
            "decision": "GO_ALL10_TRANSACTION_SCHEDULE_METADATA_CAPACITY_NO_GO_PENDING_R3",
            "score_0_to_100": 86,
            "p0": 0,
            "p1": 1,
            "p2": 5,
        },
        "anchors": {
            "canonical_result": sha256(CANONICAL),
            "contract": sha256(CONTRACT),
            "analyzer": sha256(ANALYZER),
            "r1_analyzer": sha256(R1_ANALYZER),
            "producer_validator": sha256(PRODUCER_VALIDATOR),
            "manifest": sha256(MANIFEST),
            "m43_result": sha256(M43_RESULT),
            "targeted_replay_receipt": sha256(TARGETED_REPLAY),
            "independent_reviewer": sha256(Path(__file__).resolve()),
        },
        "candidate_modified_by_reviewer": False,
        "independent_reconstruction": {
            "records_reconciled": 160,
            "samples_reconciled": 40,
            "all_record_to_sample_sums_exact": True,
            "all_sample_to_aggregate_sums_exact": True,
            "all_distributions_and_nearest_rank_percentiles_exact": True,
            "all_signed_add_subtract_conservation_exact": True,
            "all_service_counts_and_traffic_bytes_exact": True,
            "all_six_frozen_kill_gates_independently_recomputed": True,
            "K1_all40_records_reproduce_M43": True,
            "metrics": ratio_summary(configs),
        },
        "targeted_replay": {
            "sample_ids": [3, 7],
            "records": 8,
            "all_noninstrumented_fields_exact_match": True,
            "raw_spatial_dag_ready_depth_by_record": raw_depths,
            "maximum_raw_spatial_dag_ready_depth": max(raw_depths),
            "reported_clamped_metadata_occupancy": 16,
            "qualification": replay["scope"]["qualification"],
        },
        "physical_boundary_audit": {
            "complete_fifo_depth": 16,
            "primary_records_reaching_complete_fifo_depth_16":
                complete_boundary_records,
            "atomic_k2_credit_reservation_modeled": True,
            "atomic_credit_rule": "16 - len(group), therefore K2 requires two free entries",
            "same_cycle_retire_then_atomic_enqueue_assumed": True,
            "context_release_after_response_ready_modeled": True,
            "parent_DAG_runtime_conservation_assertions_passed": True,
            "weight_double_buffer_bytes_included_in_m43_scratch": 49152,
            "weight_replays_per_sample": 10,
            "weight_traffic_reduction_admitted": False,
            "metadata_fifo_capacity_proved": False,
        },
        "capacity": {
            "byte_arithmetic_reconciled": True,
            "eight_context_local_scratch_bytes": 58368,
            "single_timestep_final_accumulator_bytes": 86400,
            "fifo_storage_bytes": 5888,
            "combined_local_capacity_bytes": 150656,
            "nominal_headroom_bytes": 43072,
            "nominal_headroom_admitted_after_r2": False,
            "reason": "the 16-entry metadata quantity is a clamped selection-window report, not an enqueue/dequeue occupancy proof",
        },
        "adversarial_matrix": {
            "tested": len(attacks),
            "rejected_by_independent_validator": len(attacks),
            "rejected_attacks": attacks,
            "producer_validator_known_accepted_mutations": producer_gaps,
        },
        "findings": {
            "p0": [],
            "p1": [{
                "id": "P1_METADATA_FIFO_OCCUPANCY_IS_CLAMPED_NOT_MODELED",
                "detail": "The producer records min(16, len(ready)) and then checks it is <=16. Targeted sample3/sample7 instrumentation observes raw spatial-DAG ready depth above 16. That raw set may be a scoreboard/frontier rather than FIFO payload, but r2 does not separate the two, so the 1,024-byte metadata FIFO and 43,072-byte physical headroom are not proved.",
                "disposition": "transaction cycle ledger remains GO; physical metadata capacity and combined headroom are NO-GO until r3",
            }],
            "p2": [
                {
                    "id": "P2_ATOMIC_PUSH2_AND_SAME_CYCLE_POP_PUSH_REQUIRE_RTL",
                    "detail": "The schedule correctly reserves two credits for K2 and reaches 16/16, but assumes a FIFO can retire entries and atomically accept two complete vectors at response_ready in the same cycle.",
                },
                {
                    "id": "P2_TARGETED_REPLAY_REUSES_PINNED_SCHEDULER",
                    "detail": "Samples 3 and 7 were independently invoked and matched, but this is not a second scheduling implementation.",
                },
                {
                    "id": "P2_PARENT_DAG_HAS_NO_EDGE_LEVEL_LEDGER",
                    "detail": "The runtime scheduler asserts 300 commits per tile and nonnegative indegrees, but the sealed result does not expose per-edge admission/commit witnesses for an alternate oracle.",
                },
                {
                    "id": "P2_WEIGHT_PREFETCH_HAS_NO_SHARED_MEMORY_CONTENTION",
                    "detail": "All ten weight replays and exact bytes are counted, but the 64-byte/cycle DMA and cross-timestep overlap do not include SRAM/DRAM arbitration or macro timing.",
                },
                {
                    "id": "P2_RTL_PPA_SYSTEM_AND_GENERALITY_UNADMITTED",
                    "detail": "There is no RTL integer miter, VCS cycle measurement, SRAM macro, power/energy, end-to-end system result, or cross-sequence cohort in M45-r2.",
                },
            ],
        },
        "repair_gate": {
            "r3_required": True,
            "steps": [
                "Represent the spatial-DAG ready set separately as a 300-bit scoreboard or prove a bounded 20-row frontier and account its exact bytes.",
                "Model the response metadata FIFO with explicit enqueue/dequeue events and raw occupancy; remove min() from the reported occupancy and assert the real bound.",
                "State and later miter the atomic push2 plus simultaneous pop/push rule; throttle unless two credits exist after legal same-cycle retire.",
                "Recompute combined capacity/headroom and rerun all10 only if the separated metadata organization or its timing changes the frozen schedule.",
            ],
            "full_all10_schedule_rerun_required_if": "the repair adds descriptor delivery backpressure, changes ready selection, or changes command timing",
            "ledger_only_r3_permitted_if": "a separately accounted scoreboard/frontier supplies the identical deterministic first-16 window without schedule-visible backpressure",
        },
        "claim_boundary": "GO is limited to the SHA-bound all-ten, 40-record transaction-cycle ledger and its K1/K2-C8/K2-C4/K4 comparisons. The metadata FIFO physical capacity and 43,072-byte headroom are NO-GO in r2. RTL cycles, integer equivalence, SRAM/DRAM timing, PPA, power, energy, system speedup, 3x, external accelerator comparison, DATE headline, and best-paper claims are forbidden.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite review output")
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
