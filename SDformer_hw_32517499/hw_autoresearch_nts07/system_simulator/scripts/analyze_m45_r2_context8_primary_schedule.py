#!/usr/bin/env python3
"""M45-r2 exact all10 physical-context transaction-level schedule.

This promotes K2 with eight resident contexts only after the four-context r1
diagnostic failed.  It imports the frozen r1 scheduling algorithm and changes
only the enumerated context capacity.  Results are not RTL or system timing.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m45_dual_destination_bank_fused_integrated_schedule_contract_r2_20260823.json")
EXPECTED_CONTRACT_SHA256 = (
    "1c547c3ecd5d82c5dc8217297f19ca730748ac9526663f5449d8f13d867cd6b4")
R1_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
CONFIGURATIONS = (
    ("K1_CTX4_REPRODUCTION", 1, 4, "M43_SOURCE_CYCLE_REPRODUCTION"),
    ("K2_CTX8_PRIMARY", 2, 8, "PRIMARY_PHYSICAL_TRANSACTION_CANDIDATE"),
    ("K2_CTX4_CAPACITY_ABLATION", 2, 4, "CAPACITY_ABLATION"),
    ("K4_CTX4_KILLED_ABLATION", 4, 4, "COUNTERFACTUAL_KILLED_ABLATION"),
)


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
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_r1():
    spec = importlib.util.spec_from_file_location("m45_r2_pinned_r1", R1_ANALYZER)
    require(spec is not None and spec.loader is not None,
            "cannot import pinned M45-r1 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_contract():
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M45-r2 contract identity drift")
    contract = read_json(CONTRACT)
    require(contract["schema"] ==
            "m45_dual_destination_bank_fused_integrated_schedule_contract_v2",
            "M45-r2 schema drift")
    for name, item in contract["inputs"].items():
        path = HW_ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "M45-r2 upstream identity drift: {}".format(name))
    capacity = contract["capacity_model"]
    require(capacity["context_bytes_per_entry"] == 352 and
            capacity["extra_state_bytes_vs_four_contexts"] == 1408 and
            capacity["eight_context_local_scratch_bytes"] == 58368 and
            capacity["base_local_bytes_before_fifos"] == 144768 and
            capacity["fifo_storage_bytes"] == 5888 and
            capacity["combined_local_capacity_bytes"] == 150656 and
            capacity["local_capacity_headroom_bytes"] == 43072,
            "M45-r2 capacity identity drift")
    return contract


def analyze_cached_configuration(r1, m43, cached_records, name, fanout_k,
                                 contexts, qualification):
    per_record = []
    for index, item in enumerate(cached_records):
        record, masks, expected = item
        row = r1.analyze_record(m43, masks, expected, fanout_k, contexts)
        row["sample_id"] = record["sample_id"]
        row["operator"] = record["operator"]
        per_record.append(row)
        print("[M45-r2 {}] {}/40 sample={} operator={}".format(
            name, index + 1, record["sample_id"], record["operator"]))

    sum_fields = [field for field in r1.blank_counts()
                  if not field.startswith("maximum_")]
    sum_fields += ["signed_add_updates", "signed_subtract_updates",
                   "weight_dma_bytes", "final_accumulator_read_bytes",
                   "final_accumulator_write_bytes", "completed_output_bytes"]
    per_sample = []
    for sample_id in range(10):
        selected = [row for row in per_record
                    if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M45-r2 sample/operator population drift")
        sample = {"sample_id": sample_id}
        for field in sum_fields:
            sample[field] = sum(row[field] for row in selected)
        for field in ("maximum_metadata_occupancy",
                      "maximum_complete_occupancy",
                      "maximum_resident_occupancy"):
            sample[field] = max(row[field] for row in selected)
        sample["integrated_over_source_only"] = r1.fraction(
            sample["integrated_cycles"] - sample["source_only_cycles"],
            sample["source_only_cycles"])
        sample["parent_wait_fraction"] = r1.fraction(
            sample["parent_wait_cycles"], sample["integrated_cycles"])
        per_sample.append(sample)

    result = r1.aggregate_configuration(name, fanout_k, contexts, per_sample)
    result["qualification"] = qualification
    result["records"] = per_record
    return result


def build():
    contract = validate_contract()
    r1 = load_r1()
    r1.validate_contract()
    manifest = r1.read_json(r1.MANIFEST)
    m43_result = r1.read_json(r1.M43_RESULT)
    m43_records = dict(((row["sample_id"], row["operator"]), row)
                       for row in m43_result["records"])
    require(len(manifest["records"]) == 40 and len(m43_records) == 40,
            "M45-r2 frozen cohort drift")
    m43 = r1.load_m43_module()
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in m43_records, "M45-r2 M43 record identity mismatch")
        masks = m43.unpack_record_masks(r1.MANIFEST.parent, record)
        cached.append((record, masks, m43_records[key]))

    configurations = []
    for name, fanout_k, contexts, qualification in CONFIGURATIONS:
        config = analyze_cached_configuration(
            r1, m43, cached, name, fanout_k, contexts, qualification)
        configurations.append(config)
        print("[M45-r2 SUMMARY {}] source={} integrated={} p95={}".format(
            name, config["aggregate_source_only_cycles"],
            config["aggregate_integrated_cycles"],
            config["integrated_cycle_distribution"]["p95_nearest_rank"]))
    by_name = dict((row["name"], row) for row in configurations)
    k1 = by_name["K1_CTX4_REPRODUCTION"]
    primary = by_name["K2_CTX8_PRIMARY"]
    ctx4 = by_name["K2_CTX4_CAPACITY_ABLATION"]
    k4 = by_name["K4_CTX4_KILLED_ABLATION"]
    require(k1["aggregate_source_only_cycles"] == 116376872,
            "M45-r2 K1 does not reproduce M43")

    gates_contract = contract["kill_gates"]
    overhead_gate = gates_contract[
        "maximum_primary_integrated_over_source_only_fraction"]
    parent_gate = gates_contract["maximum_primary_parent_wait_fraction"]
    reduction_gate = gates_contract[
        "minimum_primary_integrated_reduction_vs_k1_fraction"]
    justification_gate = gates_contract[
        "minimum_ctx8_p95_improvement_over_ctx4_fraction"]
    primary_overhead_pass = all(
        (sample["integrated_cycles"] - sample["source_only_cycles"]) *
        overhead_gate["denominator"] <=
        sample["source_only_cycles"] * overhead_gate["numerator"]
        for sample in primary["per_sample"])
    primary_parent_wait_pass = all(
        sample["parent_wait_cycles"] * parent_gate["denominator"] <=
        sample["integrated_cycles"] * parent_gate["numerator"]
        for sample in primary["per_sample"])
    integrated_reduction = (k1["aggregate_integrated_cycles"] -
                            primary["aggregate_integrated_cycles"])
    primary_reduction_pass = (
        integrated_reduction * reduction_gate["denominator"] >=
        k1["aggregate_integrated_cycles"] * reduction_gate["numerator"])
    primary_p95 = primary["integrated_cycle_distribution"]["p95_nearest_rank"]
    ctx4_p95 = ctx4["integrated_cycle_distribution"]["p95_nearest_rank"]
    ctx8_improvement = ctx4_p95 - primary_p95
    ctx8_justified_pass = (
        ctx8_improvement * justification_gate["denominator"] >=
        ctx4_p95 * justification_gate["numerator"])
    p95_gate_pass = (
        primary_p95 <=
        gates_contract["maximum_primary_p95_integrated_cycles"])
    k4_p95 = k4["integrated_cycle_distribution"]["p95_nearest_rank"]
    k4_killed = k4_p95 >= primary_p95
    all_pass = all((primary_overhead_pass, primary_parent_wait_pass,
                    primary_reduction_pass, ctx8_justified_pass,
                    p95_gate_pass, k4_killed))
    gates = {
        "primary_all_samples_integrated_over_source_only_le_10pct":
            primary_overhead_pass,
        "primary_all_samples_parent_wait_le_5pct": primary_parent_wait_pass,
        "primary_aggregate_integrated_reduction_vs_k1_ge_15pct":
            primary_reduction_pass,
        "ctx8_p95_improvement_over_ctx4_ge_3pct": ctx8_justified_pass,
        "ctx8_p95_improvement_over_ctx4": r1.fraction(
            ctx8_improvement, ctx4_p95),
        "primary_p95_integrated_cycles_le_15495075": p95_gate_pass,
        "k4_ctx4_slower_than_k2_ctx8_and_killed": k4_killed,
        "all_kill_gates_pass": all_pass,
        "later_rtl_context_increment_area_gate_required": True,
        "three_x_target_crossing_admitted": False
    }
    capacity = contract["capacity_model"]
    return {
        "schema": "m45_dual_destination_bank_fused_integrated_schedule_result_v2",
        "status": ("PASS_M45_R2_TRANSACTION_GATES_RTL_AND_SYSTEM_UNADMITTED"
                   if all_pass else
                   "NO_GO_M45_R2_ONE_OR_MORE_TRANSACTION_GATES_FAILED"),
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "r1_analyzer_sha256": sha256(R1_ANALYZER)
        },
        "architecture": {
            "name": "DUAL_DESTINATION_BANK_FUSED_CONTEXT8_PRIMARY",
            "destination_fanout_k": 2,
            "resident_contexts": 8,
            "loop_order": contract["frozen_schedule"]["loop_order"],
            "context_release": contract["frozen_schedule"]["context_release"],
            "weight_replays_per_sample": 10
        },
        "capacity": capacity,
        "population": {"samples": 10, "operators": 4, "records": 40},
        "configurations": configurations,
        "kill_gates": gates,
        "qualification": {
            "exact": contract["claim_policy"]["admitted_after_validation"],
            "not_admitted": contract["claim_policy"]["forbidden"]
        },
        "deferred_interface": contract["deferred_interface"]
    }


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite M45-r2 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_output(args.output, build())
    print(args.output)


if __name__ == "__main__":
    main()
