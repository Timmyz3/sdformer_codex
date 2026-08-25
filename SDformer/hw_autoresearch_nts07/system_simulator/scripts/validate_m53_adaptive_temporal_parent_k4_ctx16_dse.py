#!/usr/bin/env python3
"""Fail-closed validator and tamper suite for the M53 exact DSE."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m53_adaptive_temporal_parent_k4_ctx16_dse_contract_r1_20260823.json")
ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py")
RESULT = HW_ROOT / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M52_RESULT = HW_ROOT / (
    "results/m52_high_fanout_context16_dse_r1_20260823/"
    "m52_high_fanout_context16_dse.json")
EXPECTED_CONTRACT_SHA256 = (
    "e1dd6eb10a4b580115ff8cfe9d28605167256dfe81942ea2e2ea92d5fba88e03")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload):
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) +
           "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def nearest_rank(values, percentile):
    values = sorted(values)
    require(values, "empty percentile population")
    return values[int(math.ceil(len(values) * percentile)) - 1]


def validate_distribution(distribution, values, label):
    require(distribution["count"] == len(values), label + " count drift")
    require(distribution["minimum"] == min(values) and
            distribution["maximum"] == max(values),
            label + " min/max drift")
    require(distribution["mean_exact"] ==
            {"numerator": sum(values), "denominator": len(values)},
            label + " mean drift")
    require(distribution["p50_nearest_rank"] == nearest_rank(values, 0.50) and
            distribution["p95_nearest_rank"] == nearest_rank(values, 0.95) and
            distribution["p99_nearest_rank"] == nearest_rank(values, 0.99),
            label + " nearest-rank percentile drift")


def get_configuration(payload, name):
    matches = [row for row in payload["configuration_ledgers"]
               if row["name"] == name]
    require(len(matches) == 1, "M53 configuration identity drift: " + name)
    return matches[0]


def validate_configuration(configuration):
    records = configuration["record_ledger"]["records"]
    samples = configuration["per_sample"]
    require(configuration["record_ledger"]["record_count"] == 40 and
            len(records) == 40 and len(samples) == 10,
            "M53 configuration population drift")
    require(configuration["record_ledger"]["canonical_sha256"] ==
            canonical_sha(records), "M53 record ledger SHA drift")
    require(sorted(row["sample_id"] for row in samples) == list(range(10)),
            "M53 per-sample identity drift")
    sum_fields = (
        "source_only_cycles", "integrated_cycles", "logical_source_updates",
        "unique_weight_issues", "descriptor_commands", "parent_partial_reads",
        "parent_partial_writes", "final_accumulator_reads",
        "final_accumulator_writes", "completed_outputs", "fusion_groups",
        "zero_source_groups", "parent_wait_cycles",
        "command_or_state_wait_cycles", "response_or_context_wait_cycles",
        "weight_dma_wait_cycles", "fusion_hold_wait_cycles",
        "late_join_groups", "signed_add_updates", "signed_subtract_updates",
        "weight_dma_bytes", "final_accumulator_read_bytes",
        "final_accumulator_write_bytes", "completed_output_bytes",
    )
    maximum_fields = ("maximum_metadata_occupancy",
                      "maximum_complete_occupancy",
                      "maximum_resident_occupancy")
    parents = ("local_zero", "left", "up", "previous_timestep")
    for sample in samples:
        selected = [row for row in records
                    if row["sample_id"] == sample["sample_id"]]
        require(len(selected) == 4, "M53 sample record population drift")
        for field in sum_fields:
            require(sample[field] == sum(row[field] for row in selected),
                    "M53 per-sample sum drift: " + field)
        for field in maximum_fields:
            require(sample[field] == max(row[field] for row in selected),
                    "M53 per-sample maximum drift: " + field)
        for parent in parents:
            require(sample["parent_choice_by_tile"][parent] == sum(
                row["parent_selection"]["parent_choice_by_tile"][parent]
                for row in selected), "M53 per-sample parent choice drift")
        require(sample["previous_timestep_choices_after_timestep_zero"] == sum(
            row["parent_selection"][
                "previous_timestep_choices_after_timestep_zero"]
            for row in selected), "M53 per-sample temporal choice drift")
    source_values = [row["source_only_cycles"] for row in samples]
    integrated_values = [row["integrated_cycles"] for row in samples]
    require(configuration["aggregate_source_only_cycles"] == sum(source_values),
            "M53 aggregate source drift")
    require(configuration["aggregate_integrated_cycles"] ==
            sum(integrated_values), "M53 aggregate integrated drift")
    validate_distribution(configuration["source_only_cycle_distribution"],
                          source_values, "M53 source distribution")
    validate_distribution(configuration["integrated_cycle_distribution"],
                          integrated_values, "M53 integrated distribution")
    parent_aggregate = configuration["parent_selection_aggregate"]
    for parent in parents:
        require(parent_aggregate["parent_choice_by_tile"][parent] == sum(
            row["parent_selection"]["parent_choice_by_tile"][parent]
            for row in records), "M53 aggregate parent choice drift")
    require(sum(parent_aggregate["parent_choice_by_tile"].values()) ==
            40 * 10 * 300 * 27,
            "M53 aggregate parent-choice population drift")
    require(parent_aggregate["previous_timestep_choices_at_timestep_zero"] == 0,
            "M53 illegal timestep-zero temporal parent")


def validate_payload(payload):
    contract = read_json(CONTRACT)
    m52 = read_json(M52_RESULT)
    require(sha256_path(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M53 validator contract drift")
    for name, identity in contract["inputs"].items():
        path = HW_ROOT / identity["path"]
        require(path.is_file() and sha256_path(path) == identity["sha256"],
                "M53 validator input identity drift: " + name)
    require(payload["schema"] ==
            "m53_adaptive_temporal_parent_k4_ctx16_dse_result_v1",
            "M53 result schema drift")
    require(payload["status"] ==
            "PASS_M53_K4_CTX16_TEMPORAL_TRANSACTION_DSE_M54_RTL_REQUIRED",
            "M53 result status is not PASS")
    require(payload["identity"]["contract_sha256"] ==
            EXPECTED_CONTRACT_SHA256 and
            payload["identity"]["analyzer_sha256"] == sha256_path(ANALYZER),
            "M53 result identity drift")
    require(payload["population"] ==
            {"samples": 10, "operators": 4, "records": 40},
            "M53 result population drift")
    require([row["name"] for row in payload["configuration_ledgers"]] == [
        "K2_CTX16_TEMPORAL", "K4_CTX16_SPATIAL", "K4_CTX16_TEMPORAL"],
        "M53 configuration order drift")
    for configuration in payload["configuration_ledgers"]:
        validate_configuration(configuration)

    k2 = get_configuration(payload, "K2_CTX16_TEMPORAL")
    spatial = get_configuration(payload, "K4_CTX16_SPATIAL")
    temporal = get_configuration(payload, "K4_CTX16_TEMPORAL")
    require((k2["aggregate_source_only_cycles"],
             k2["aggregate_integrated_cycles"],
             k2["integrated_cycle_distribution"]["p95_nearest_rank"]) ==
            (83847720, 90755624, 9192368),
            "M53 K2 temporal frozen metrics drift")
    require((spatial["aggregate_source_only_cycles"],
             spatial["aggregate_integrated_cycles"],
             spatial["integrated_cycle_distribution"]["p95_nearest_rank"]) ==
            (70821488, 81921184, 8376280),
            "M53 K4 spatial frozen metrics drift")
    require((temporal["aggregate_source_only_cycles"],
             temporal["aggregate_integrated_cycles"],
             temporal["integrated_cycle_distribution"]["p95_nearest_rank"]) ==
            (68847096, 79869808, 8139624),
            "M53 K4 temporal frozen metrics drift")

    reproduction = payload["m52_spatial_reproduction"]
    require(reproduction["exact_match"] and
            reproduction["mismatch_count"] == 0 and
            reproduction["mismatch_fields"] == [],
            "M53 M52 spatial reproduction failed")
    m52_spatial = [row for row in m52["configuration_ledgers"]
                   if row["name"] == "K4_CTX16"][0]
    require(spatial["aggregate_source_only_cycles"] ==
            m52_spatial["aggregate_source_only_cycles"] and
            spatial["aggregate_integrated_cycles"] ==
            m52_spatial["aggregate_integrated_cycles"] and
            spatial["integrated_cycle_distribution"] ==
            m52_spatial["integrated_cycle_distribution"],
            "M53 independent M52 spatial anchor drift")

    selection = payload["adaptive_parent_selection_decomposition"]
    require(selection["spatial_parent_choice_by_tile"] ==
            {"left": 1094484, "local_zero": 1365035,
             "previous_timestep": 0, "up": 780481},
            "M53 spatial parent selection drift")
    require(selection["adaptive_temporal_parent_choice_by_tile"] ==
            {"left": 991660, "local_zero": 1240600,
             "previous_timestep": 301274, "up": 706466},
            "M53 temporal parent selection drift")
    require(selection["spatial_choices_displaced_by_previous_timestep"] ==
            {"left": 102824, "local_zero": 124435, "up": 74015} and
            selection["displaced_choice_count"] ==
            selection["previous_timestep_choice_count"] == 301274 and
            selection["choice_count_conserved"] and
            selection["previous_timestep_at_timestep_zero"] == 0 and
            selection["unfused_source_issue_cycle_gain"] == 3029128,
            "M53 adaptive parent displacement drift")

    for comparison in payload["cycle_gain_decomposition"].values():
        aggregate = comparison["aggregate"]
        require(aggregate["source_only_cycle_gain"] +
                aggregate["non_source_overhead_cycle_gain"] ==
                aggregate["integrated_cycle_gain"] and
                aggregate["decomposition_exact"],
                "M53 aggregate decomposition drift")
        for row in comparison["per_sample"]:
            require(row["source_only_cycle_gain"] +
                    row["non_source_overhead_cycle_gain"] ==
                    row["integrated_cycle_gain"] and
                    row["decomposition_exact"],
                    "M53 per-sample decomposition drift")

    capacity = payload["two_frame_capacity_ledger"]
    require(capacity["frame_bytes"] == 68400 and
            capacity["existing_frame_count"] == 2 and
            capacity["existing_two_frame_bytes"] == 136800 and
            capacity["new_third_frame_bytes"] == 0 and
            capacity["combined_k4_ctx16_capacity_bytes"] == 176688 and
            capacity["local_capacity_headroom_bytes"] == 17040 and
            capacity["headroom_unit"] == "bytes" and
            capacity["margin_above_16kib_gate_bytes"] == 656 and
            not capacity["rtl_fifo_feasibility_admitted"],
            "M53 exact two-frame byte ledger drift")
    proof = payload["timestep_then_tile_commit_proof"]
    require(proof["loop_order"] == ["timestep", "feature_tile"] and
            proof["feature_tiles_per_timestep"] == 27 and
            proof["rows_committed_per_tile"] == 300 and
            proof["modeled_previous_to_current_boundaries_all_40_records"] ==
            360 and proof["output_block_expanded_boundaries"] == 2880 and
            proof["previous_timestep_complete_before_current_starts"] and
            all(value == 1 for value in
                proof["source_snippet_occurrences"].values()),
            "M53 timestep-then-tile commit proof drift")

    gates = payload["predeclared_gates"]
    require(all(value is True for value in gates.values()),
            "M53 one or more predeclared gates failed")
    for before, after in zip(spatial["per_sample"], temporal["per_sample"]):
        require(after["source_only_cycles"] <= before["source_only_cycles"],
                "M53 temporal source regressed for a sample")
    require(payload["conditional_frozen_compute_model"][
                "system_or_end_to_end_speedup_admitted"] is False and
            payload["conditional_frozen_compute_model"][
                "address_timed_pair_replayed"] is False,
            "M53 conditional ratio promoted beyond evidence")
    admission = payload["admission"]
    for forbidden_true in (
            "adaptive_per_tile_temporal_parent_arithmetic_state_rtl_admitted",
            "finite_context_tag_allocation_admitted",
            "response_metadata_fifo_event_ledger_admitted",
            "new_configuration_vcs_or_synopsys_admitted",
            "sram_macro_port_feasibility_admitted",
            "full_network_or_system_speedup_admitted",
            "date_headline_or_best_paper_admitted"):
        require(admission[forbidden_true] is False,
                "M53 forbidden admission promoted: " + forbidden_true)
    for configuration in payload["configuration_ledgers"]:
        audit = configuration["dynamic_source_edit_audit"]
        expected_edits = 3 if configuration[
            "previous_timestep_parent_enabled"] else 1
        require(audit["edit_count"] == expected_edits and
                len(audit["edits"]) == expected_edits and
                audit["unlisted_source_edits"] == 0 and
                audit["canonical_m45_sha256"] ==
                contract["inputs"]["m45_canonical_analyzer"]["sha256"],
                "M53 dynamic source edit audit drift")
    return payload


def run_attacks(payload):
    attacks = []

    def attack(name, mutate):
        candidate = copy.deepcopy(payload)
        mutate(candidate)
        rejected = False
        error = ""
        try:
            validate_payload(candidate)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            rejected = True
            error = str(exc)
        require(rejected, "M53 attack was accepted: " + name)
        attacks.append({"name": name, "rejected": True, "error": error})

    attack("k4_temporal_aggregate_source_increment",
           lambda x: get_configuration(x, "K4_CTX16_TEMPORAL").__setitem__(
               "aggregate_source_only_cycles", 68847097))
    attack("k4_temporal_sample_cycle_increment",
           lambda x: get_configuration(x, "K4_CTX16_TEMPORAL")[
               "per_sample"][0].__setitem__("integrated_cycles",
                                            get_configuration(
                                                x, "K4_CTX16_TEMPORAL")[
                                                "per_sample"][0][
                                                "integrated_cycles"] + 1))
    attack("k4_temporal_p95_increment",
           lambda x: get_configuration(x, "K4_CTX16_TEMPORAL")[
               "integrated_cycle_distribution"].__setitem__(
                   "p95_nearest_rank", 8139625))
    attack("previous_timestep_choice_removed",
           lambda x: x["adaptive_parent_selection_decomposition"][
               "adaptive_temporal_parent_choice_by_tile"].__setitem__(
                   "previous_timestep", 0))
    attack("headroom_changed",
           lambda x: x["two_frame_capacity_ledger"].__setitem__(
               "local_capacity_headroom_bytes", 17039))
    attack("headroom_unit_bits",
           lambda x: x["two_frame_capacity_ledger"].__setitem__(
               "headroom_unit", "bits"))
    attack("third_frame_added",
           lambda x: x["two_frame_capacity_ledger"].__setitem__(
               "new_third_frame_bytes", 68400))
    attack("fifo_feasibility_promoted",
           lambda x: x["two_frame_capacity_ledger"].__setitem__(
               "rtl_fifo_feasibility_admitted", True))
    attack("system_speedup_promoted",
           lambda x: x["admission"].__setitem__(
               "full_network_or_system_speedup_admitted", True))
    attack("conditional_ratio_promoted",
           lambda x: x["conditional_frozen_compute_model"].__setitem__(
               "system_or_end_to_end_speedup_admitted", True))
    attack("unlisted_dynamic_edit_added",
           lambda x: get_configuration(x, "K4_CTX16_TEMPORAL")[
               "dynamic_source_edit_audit"].__setitem__(
                   "unlisted_source_edits", 1))
    attack("m52_spatial_reproduction_revoked",
           lambda x: x["m52_spatial_reproduction"].__setitem__(
               "exact_match", False))
    return attacks


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite M53 validator receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate_payload(read_json(RESULT))
    byte_identical = None
    if args.rerun:
        with tempfile.TemporaryDirectory(prefix="m53_validate_") as temporary:
            rebuilt = Path(temporary) / "rebuilt.json"
            subprocess.check_call([
                "/usr/bin/python3.6", str(ANALYZER), "--output", str(rebuilt)])
            byte_identical = rebuilt.read_bytes() == RESULT.read_bytes()
            require(byte_identical, "M53 deterministic rerun differs")
    attacks = run_attacks(payload)
    receipt = {
        "schema": "m53_adaptive_temporal_parent_k4_ctx16_dse_validator_receipt_v1",
        "status": "PASS_M53_EXACT_DSE_VALIDATOR_AND_TAMPER_SUITE",
        "contract_sha256": sha256_path(CONTRACT),
        "analyzer_sha256": sha256_path(ANALYZER),
        "result_sha256": sha256_path(RESULT),
        "validator_sha256": sha256_path(Path(__file__).resolve()),
        "rerun": args.rerun,
        "rerun_byte_identical": byte_identical,
        "negative_tamper_attacks": attacks,
        "negative_tamper_attack_count": len(attacks),
        "negative_tamper_attacks_rejected": sum(
            1 for row in attacks if row["rejected"]),
        "open_source_hdl_tools_run": False,
        "dc_run": False,
        "vcs_or_synopsys_result_admitted": False,
        "system_speedup_admitted": False,
    }
    if args.output is not None:
        write_output(args.output, receipt)
    print("PASS M53 exact DSE; attacks={}/{} rerun_byte_identical={}".format(
        receipt["negative_tamper_attacks_rejected"],
        receipt["negative_tamper_attack_count"], byte_identical))


if __name__ == "__main__":
    main()
