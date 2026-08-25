#!/usr/bin/env python3
"""M52 exact all10 K4/K8 transaction, capacity, and complexity DSE.

This replays the frozen M45 scheduler after widening only its context-capacity
guard to 16.  It does not claim K4/K8 RTL, PPA, memory-port feasibility, or
system speedup.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / "contracts/m52_high_fanout_context16_dse_contract_r1_20260823.json"
R1_ANALYZER = HW_ROOT / "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py"
CONFIGURATIONS = (
    ("K2_CTX16", 2, 16, "CONTEXT_DEPTH_EFFICIENCY_CANDIDATE"),
    ("K4_CTX8", 4, 8, "CAPACITY_ABLATION"),
    ("K4_CTX16", 4, 16, "PRIMARY_RTL_EXPERIMENT_CANDIDATE"),
    ("K8_CTX16", 8, 16, "HIGH_FANOUT_KILL_ABLATION"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
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


def fraction(numerator, denominator):
    require(denominator > 0, "fraction denominator must be positive")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[int(math.ceil(len(ordered) * percentile)) - 1]


def validate_contract():
    contract = read_json(CONTRACT)
    require(contract["schema"] == "m52_high_fanout_context16_dse_contract_v1",
            "M52 contract schema drift")
    for name, identity in contract["inputs"].items():
        path = HW_ROOT / identity["path"]
        require(path.is_file() and sha256_path(path) == identity["sha256"],
                "M52 input identity drift: {}".format(name))
    require(contract["frozen_schedule_extension"]["basis"].endswith(
        "context_capacity<=16"), "M52 guard-widening policy drift")
    return contract


def load_guard_widened_r1():
    source = R1_ANALYZER.read_text(encoding="utf-8")
    needle = ("require(1 <= fanout_k <= context_capacity <= 8,\n"
              "            \"invalid fanout/context geometry\")")
    replacement = ("require(1 <= fanout_k <= context_capacity <= 16,\n"
                   "            \"invalid fanout/context geometry\")")
    require(source.count(needle) == 1,
            "M45-r1 context guard source identity drift")
    widened = source.replace(needle, replacement)
    namespace = {
        "__file__": str(R1_ANALYZER),
        "__name__": "m52_guard_widened_m45_r1",
    }
    exec(compile(widened, str(R1_ANALYZER) + "#M52_CTX16", "exec"), namespace)
    require(namespace["schedule_tile_timestep"].__globals__ is namespace,
            "M52 widened scheduler namespace mismatch")
    return namespace


def analyze_configuration(r1, m43, cached_records, name, fanout_k,
                          contexts, qualification):
    per_record = []
    for index, item in enumerate(cached_records):
        record, masks, expected = item
        row = r1["analyze_record"](
            m43, masks, expected, fanout_k, contexts)
        row["sample_id"] = record["sample_id"]
        row["operator"] = record["operator"]
        per_record.append(row)
        print("[M52 {}] {}/40 sample={} operator={}".format(
            name, index + 1, record["sample_id"], record["operator"]),
              flush=True)

    blank = r1["blank_counts"]()
    sum_fields = [field for field in blank
                  if not field.startswith("maximum_")]
    sum_fields += ["signed_add_updates", "signed_subtract_updates",
                   "weight_dma_bytes", "final_accumulator_read_bytes",
                   "final_accumulator_write_bytes", "completed_output_bytes"]
    per_sample = []
    for sample_id in range(10):
        selected = [row for row in per_record
                    if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M52 sample/operator population drift")
        sample = {"sample_id": sample_id}
        for field in sum_fields:
            sample[field] = sum(row[field] for row in selected)
        for field in ("maximum_metadata_occupancy",
                      "maximum_complete_occupancy",
                      "maximum_resident_occupancy"):
            sample[field] = max(row[field] for row in selected)
        sample["integrated_over_source_only"] = fraction(
            sample["integrated_cycles"] - sample["source_only_cycles"],
            sample["source_only_cycles"])
        sample["parent_wait_fraction"] = fraction(
            sample["parent_wait_cycles"], sample["integrated_cycles"])
        per_sample.append(sample)

    result = r1["aggregate_configuration"](
        name, fanout_k, contexts, per_sample)
    result["qualification"] = qualification
    record_bytes = (json.dumps(per_record, sort_keys=True,
                               separators=(",", ":")) + "\n").encode("utf-8")
    result["record_ledger"] = {
        "record_count": len(per_record),
        "canonical_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "records": per_record,
    }
    return result


def capacity_ledger(contract, fanout_k, contexts):
    model = contract["bit_tight_capacity_model"]
    context_id_bits = (contexts - 1).bit_length()
    payload_bits = (fanout_k * context_id_bits + (fanout_k - 1) +
                    model["source_banks"] +
                    fanout_k * model["source_banks"] * 2 + 1)
    alignment = model["response_metadata_alignment_bytes"]
    metadata_entry_bytes = ((payload_bits + alignment * 8 - 1) //
                            (alignment * 8)) * alignment
    components = dict(model["fixed_components_bytes"])
    components["resident_contexts"] = contexts * (
        model["vector_bytes"] + model["context_metadata_bytes"])
    components["response_metadata_fifo"] = (
        model["response_metadata_entries"] * metadata_entry_bytes)
    combined = sum(components.values())
    headroom = model["local_residency_bytes"] - combined
    return {
        "fanout_k": fanout_k,
        "resident_contexts": contexts,
        "context_id_bits": context_id_bits,
        "response_metadata_payload_bits": payload_bits,
        "response_metadata_aligned_bytes_per_entry": metadata_entry_bytes,
        "components": components,
        "combined_local_capacity_bytes": combined,
        "local_capacity_headroom_bytes": headroom,
        "minimum_headroom_bytes": model["minimum_headroom_bytes"],
        "headroom_gate_pass": headroom >= model["minimum_headroom_bytes"],
        "external_accumulator_spill_permitted": False,
        "double_weight_buffer_permitted": False,
    }


def complexity_ledger(contract, fanout_k):
    lanes = contract["bit_tight_capacity_model"]["output_lanes"]
    banks = contract["bit_tight_capacity_model"]["source_banks"]
    acc_bits = contract["bit_tight_capacity_model"]["signed_accumulator_bits"]
    return {
        "fanout_k": fanout_k,
        "accumulator_read_modify_write_paths_per_response": fanout_k * lanes,
        "signed_bank_terms_per_response": fanout_k * banks * lanes,
        "atomic_complete_vector_push_count": fanout_k,
        "atomic_complete_payload_bits_excluding_tags": fanout_k * lanes * acc_bits,
        "relative_to_k2": {
            "accumulator_paths": fraction(fanout_k, 2),
            "signed_bank_terms": fraction(fanout_k, 2),
            "atomic_push_width": fraction(fanout_k, 2),
        },
        "qualification": contract["complexity_model"]["qualification"],
    }


def configuration_summary(configuration, capacity, complexity):
    return {
        "name": configuration["name"],
        "destination_fanout_k": configuration["destination_fanout_k"],
        "resident_contexts": configuration["resident_contexts"],
        "qualification": configuration["qualification"],
        "aggregate_source_only_cycles": configuration["aggregate_source_only_cycles"],
        "aggregate_integrated_cycles": configuration["aggregate_integrated_cycles"],
        "source_only_cycle_distribution":
            configuration["source_only_cycle_distribution"],
        "integrated_cycle_distribution": configuration["integrated_cycle_distribution"],
        "capacity": capacity,
        "complexity": complexity,
    }


def replay_one_configuration(configuration):
    """Replay one independent DSE point in its own process."""
    validate_contract()
    r1 = load_guard_widened_r1()
    r1["validate_contract"]()
    manifest = r1["read_json"](r1["MANIFEST"])
    m43_result = r1["read_json"](r1["M43_RESULT"])
    m43_records = dict(((row["sample_id"], row["operator"]), row)
                       for row in m43_result["records"])
    require(len(manifest["records"]) == 40 and len(m43_records) == 40,
            "M52 frozen cohort drift")
    m43 = r1["load_m43_module"]()
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in m43_records, "M52 M43 record identity mismatch")
        masks = m43.unpack_record_masks(r1["MANIFEST"].parent, record)
        cached.append((record, masks, m43_records[key]))

    name, fanout_k, contexts, qualification = configuration
    result = analyze_configuration(
        r1, m43, cached, name, fanout_k, contexts, qualification)
    print("[M52 SUMMARY {}] source={} integrated={} p95={}".format(
        name, result["aggregate_source_only_cycles"],
        result["aggregate_integrated_cycles"],
        result["integrated_cycle_distribution"]["p95_nearest_rank"]),
          flush=True)
    return result


def build():
    contract = validate_contract()

    with ProcessPoolExecutor(max_workers=len(CONFIGURATIONS)) as executor:
        configurations = list(executor.map(
            replay_one_configuration, CONFIGURATIONS))

    by_name = dict((row["name"], row) for row in configurations)
    k2c16 = by_name["K2_CTX16"]
    k4c8 = by_name["K4_CTX8"]
    k4c16 = by_name["K4_CTX16"]
    k8c16 = by_name["K8_CTX16"]
    m45 = read_json(HW_ROOT / contract["inputs"]["m45_r2_result"]["path"])
    matches = [row for row in m45["configurations"]
               if row["name"] == "K2_CTX8_PRIMARY"]
    require(len(matches) == 1, "M52 inherited K2-C8 identity mismatch")
    k2c8 = matches[0]
    require(k2c8["aggregate_source_only_cycles"] == 88269520 and
            k2c8["aggregate_integrated_cycles"] == 95047672 and
            k2c8["integrated_cycle_distribution"]["p95_nearest_rank"] == 9681752,
            "M52 inherited K2-C8 values drift")

    capacity = {
        "K2_CTX8": capacity_ledger(contract, 2, 8),
        "K2_CTX16": capacity_ledger(contract, 2, 16),
        "K4_CTX8": capacity_ledger(contract, 4, 8),
        "K4_CTX16": capacity_ledger(contract, 4, 16),
        "K8_CTX16": capacity_ledger(contract, 8, 16),
    }
    complexity = {
        "K2_CTX8": complexity_ledger(contract, 2),
        "K2_CTX16": complexity_ledger(contract, 2),
        "K4_CTX8": complexity_ledger(contract, 4),
        "K4_CTX16": complexity_ledger(contract, 4),
        "K8_CTX16": complexity_ledger(contract, 8),
    }

    gates_contract = contract["promotion_and_kill_gates"]
    k2_p95 = k2c8["integrated_cycle_distribution"]["p95_nearest_rank"]
    k2c16_p95 = k2c16["integrated_cycle_distribution"]["p95_nearest_rank"]
    k4_p95 = k4c16["integrated_cycle_distribution"]["p95_nearest_rank"]
    k8_p95 = k8c16["integrated_cycle_distribution"]["p95_nearest_rank"]
    p95_gain = k2_p95 - k4_p95
    k2_depth_p95_gain = k2_p95 - k2c16_p95
    k4_over_k2c16_p95_gain = k2c16_p95 - k4_p95
    aggregate_gain = (k2c8["aggregate_integrated_cycles"] -
                      k4c16["aggregate_integrated_cycles"])
    p95_gate = gates_contract[
        "k4_ctx16_minimum_p95_improvement_vs_k2_ctx8_fraction"]
    k2_depth_gate = gates_contract[
        "k2_ctx16_minimum_p95_improvement_vs_k2_ctx8_fraction"]
    k4_over_k2c16_gate = gates_contract[
        "k4_ctx16_minimum_p95_improvement_vs_k2_ctx16_fraction"]
    aggregate_gate = gates_contract[
        "k4_ctx16_minimum_aggregate_integrated_improvement_vs_k2_ctx8_fraction"]
    overhead_gate = gates_contract[
        "k4_ctx16_maximum_integrated_over_source_fraction_each_sample"]
    p95_pass = (p95_gain * p95_gate["denominator"] >=
                k2_p95 * p95_gate["numerator"])
    k2_depth_p95_pass = (
        k2_depth_p95_gain * k2_depth_gate["denominator"] >=
        k2_p95 * k2_depth_gate["numerator"])
    k4_over_k2c16_p95_pass = (
        k4_over_k2c16_p95_gain * k4_over_k2c16_gate["denominator"] >=
        k2c16_p95 * k4_over_k2c16_gate["numerator"])
    aggregate_pass = (
        aggregate_gain * aggregate_gate["denominator"] >=
        k2c8["aggregate_integrated_cycles"] * aggregate_gate["numerator"])
    overhead_pass = all(
        (sample["integrated_cycles"] - sample["source_only_cycles"]) *
        overhead_gate["denominator"] <=
        sample["source_only_cycles"] * overhead_gate["numerator"]
        for sample in k4c16["per_sample"])
    occupancy_pass = all(
        sample["maximum_metadata_occupancy"] <= 16 and
        sample["maximum_complete_occupancy"] <= 16 and
        sample["maximum_resident_occupancy"] <= 16
        for sample in k4c16["per_sample"])
    context_exercised = max(sample["maximum_resident_occupancy"]
                            for sample in k4c16["per_sample"]) == 16
    capacity_pass = capacity["K4_CTX16"]["headroom_gate_pass"]
    k2_depth_occupancy_pass = all(
        sample["maximum_metadata_occupancy"] <= 16 and
        sample["maximum_complete_occupancy"] <= 16 and
        sample["maximum_resident_occupancy"] <= 16
        for sample in k2c16["per_sample"])
    k2_depth_context_exercised = max(
        sample["maximum_resident_occupancy"]
        for sample in k2c16["per_sample"]) == 16
    k2_depth_capacity_pass = capacity["K2_CTX16"]["headroom_gate_pass"]
    k2_depth_promotion_pass = all((
        k2_depth_p95_pass, k2_depth_occupancy_pass,
        k2_depth_context_exercised, k2_depth_capacity_pass))
    k4_promotion_pass = all((
        k4_over_k2c16_p95_pass, p95_pass, aggregate_pass,
        overhead_pass, occupancy_pass,
        context_exercised, capacity_pass))
    selected = k4c16 if k4_promotion_pass else k2c16
    selected_name = selected["name"]
    dse_pass = k2_depth_promotion_pass or k4_promotion_pass

    k8_increment = k4_p95 - k8_p95
    k8_gate = gates_contract[
        "kill_k8_if_incremental_p95_improvement_vs_k4_ctx16_below_fraction"]
    k8_increment_below_gate = (
        k8_increment * k8_gate["denominator"] <
        k4_p95 * k8_gate["numerator"])
    k8_source_not_lower = (
        k8c16["aggregate_source_only_cycles"] >=
        k4c16["aggregate_source_only_cycles"])
    k8_killed = k8_increment_below_gate or k8_source_not_lower

    pair_load = contract["conservative_pair_model"][
        "serialized_single_buffer_weight_load_cycles_per_sample"]
    pair_samples = []
    for sample in selected["per_sample"]:
        pair_samples.append({
            "sample_id": sample["sample_id"],
            "transaction_integrated_cycles": sample["integrated_cycles"],
            "serialized_weight_load_cycles_added": pair_load,
            "conservative_pair_upper_bound_cycles":
                sample["integrated_cycles"] + pair_load,
        })
    pair_values = [row["conservative_pair_upper_bound_cycles"]
                   for row in pair_samples]
    pair_p95 = nearest_rank(pair_values, 0.95)
    pair_model = contract["conservative_pair_model"]
    conditional_denominator = (
        pair_model["outside_four_bottleneck_model_cycles"] +
        pair_model["fixed_late_scale_plus_frontend_cycles"] + pair_p95)
    fixed_reference = pair_model["fixed_compute_reference_cycles"]

    local_source_cycles = read_json(
        HW_ROOT / contract["inputs"]["m43_result"]["path"]
    )["aggregate"]["local_p8_l96_source_issue_cycles"]
    require(local_source_cycles == 141484880,
            "M52 M43 local source reference drift")

    summaries = [
        configuration_summary(k2c16, capacity["K2_CTX16"], complexity["K2_CTX16"]),
        configuration_summary(k4c8, capacity["K4_CTX8"], complexity["K4_CTX8"]),
        configuration_summary(k4c16, capacity["K4_CTX16"], complexity["K4_CTX16"]),
        configuration_summary(k8c16, capacity["K8_CTX16"], complexity["K8_CTX16"]),
    ]
    return {
        "schema": "m52_high_fanout_context16_dse_result_v1",
        "status": ("PASS_PROMOTE_{}_TO_RTL_EXPERIMENT_K8_KILLED_SYSTEM_UNADMITTED".format(
                       selected_name)
                   if dse_pass and k8_killed else
                   "NO_GO_M52_PROMOTION_OR_K8_KILL_GATE_FAILED"),
        "identity": {
            "contract_sha256": sha256_path(CONTRACT),
            "analyzer_sha256": sha256_path(Path(__file__).resolve()),
            "inputs_sha256": dict((name, item["sha256"])
                                   for name, item in contract["inputs"].items()),
            "guard_widening_occurrences": 1,
        },
        "population": {"samples": 10, "operators": 4, "records": 40},
        "inherited_k2_ctx8_reference": {
            "aggregate_source_only_cycles": k2c8["aggregate_source_only_cycles"],
            "aggregate_integrated_cycles": k2c8["aggregate_integrated_cycles"],
            "integrated_cycle_distribution": k2c8["integrated_cycle_distribution"],
            "capacity": capacity["K2_CTX8"],
            "complexity": complexity["K2_CTX8"],
            "rtl_basis": "M49 K2-C8 VCS/SVA only; no K4/K8 RTL inheritance",
        },
        "configuration_summaries": summaries,
        "configuration_ledgers": configurations,
        "performance_comparisons": {
            "k2_ctx16_p95_improvement_vs_k2_ctx8": fraction(
                k2_depth_p95_gain, k2_p95),
            "k4_ctx16_p95_improvement_vs_k2_ctx16": fraction(
                k4_over_k2c16_p95_gain, k2c16_p95),
            "k4_ctx16_p95_improvement_vs_k2_ctx8": fraction(p95_gain, k2_p95),
            "k4_ctx16_aggregate_integrated_improvement_vs_k2_ctx8": fraction(
                aggregate_gain, k2c8["aggregate_integrated_cycles"]),
            "k4_ctx16_transaction_speedup_vs_k1_ctx4_integrated": fraction(
                122418024, k4c16["aggregate_integrated_cycles"]),
            "k4_ctx16_source_speedup_vs_local_zero_source": fraction(
                local_source_cycles, k4c16["aggregate_source_only_cycles"]),
            "k4_ctx8_p95_improvement_vs_k2_ctx8": fraction(
                k2_p95 - k4c8["integrated_cycle_distribution"]["p95_nearest_rank"],
                k2_p95),
            "k8_ctx16_incremental_p95_improvement_vs_k4_ctx16": fraction(
                k8_increment, k4_p95),
            "k8_ctx16_source_change_vs_k4_ctx16": fraction(
                k8c16["aggregate_source_only_cycles"] -
                k4c16["aggregate_source_only_cycles"],
                k4c16["aggregate_source_only_cycles"]),
        },
        "promotion_and_kill_gates": {
            "k2_ctx16_p95_improvement_vs_k2_ctx8_ge_2pct": k2_depth_p95_pass,
            "k2_ctx16_fifo_and_context_occupancy_within_capacity": k2_depth_occupancy_pass,
            "k2_ctx16_reaches_full_16_context_occupancy": k2_depth_context_exercised,
            "k2_ctx16_capacity_headroom_ge_16kib": k2_depth_capacity_pass,
            "k2_ctx16_all_promotion_gates_pass": k2_depth_promotion_pass,
            "k4_ctx16_p95_improvement_vs_k2_ctx16_ge_10pct": k4_over_k2c16_p95_pass,
            "k4_ctx16_p95_improvement_vs_k2_ctx8_ge_10pct": p95_pass,
            "k4_ctx16_aggregate_integrated_improvement_vs_k2_ctx8_ge_10pct": aggregate_pass,
            "k4_ctx16_each_sample_integrated_over_source_le_20pct": overhead_pass,
            "k4_ctx16_fifo_and_context_occupancy_within_capacity": occupancy_pass,
            "k4_ctx16_reaches_full_16_context_occupancy": context_exercised,
            "k4_ctx16_capacity_headroom_ge_16kib": capacity_pass,
            "k4_ctx16_all_promotion_gates_pass": k4_promotion_pass,
            "k8_incremental_p95_improvement_below_5pct": k8_increment_below_gate,
            "k8_source_cycles_not_lower_than_k4": k8_source_not_lower,
            "k8_killed_by_predeclared_complexity_gate": k8_killed,
            "selected_configuration": selected_name if dse_pass else "NO_GO",
            "selected_role": (
                gates_contract["k4_ctx16_role_after_pass"]
                if selected_name == "K4_CTX16" else
                gates_contract["k2_ctx16_role_after_pass"])
                if dse_pass else "NO_GO",
            "k8_role": gates_contract["k8_ctx16_role_after_kill"]
                if k8_killed else "REQUIRES_REVIEW",
        },
        "conservative_pair_upper_bound": {
            "construction": pair_model["construction"],
            "per_sample": pair_samples,
            "aggregate_cycles": sum(pair_values),
            "distribution": {
                "count": len(pair_values),
                "minimum": min(pair_values),
                "maximum": max(pair_values),
                "p50_nearest_rank": nearest_rank(pair_values, 0.50),
                "p95_nearest_rank": pair_p95,
                "p99_nearest_rank": nearest_rank(pair_values, 0.99),
            },
            "address_timed_pair_replayed": False,
        },
        "conditional_frozen_compute_model": {
            "fixed_compute_reference_cycles": fixed_reference,
            "outside_four_bottleneck_model_cycles":
                pair_model["outside_four_bottleneck_model_cycles"],
            "fixed_late_scale_plus_frontend_cycles":
                pair_model["fixed_late_scale_plus_frontend_cycles"],
            "candidate_p95_cycle_upper_bound": pair_p95,
            "conditional_total_cycles": conditional_denominator,
            "conditional_compute_speedup": fraction(
                fixed_reference, conditional_denominator),
            "three_x_crossing_in_conditional_model":
                fixed_reference >= 3 * conditional_denominator,
            "system_or_end_to_end_speedup_admitted": False,
            "qualification": pair_model["qualification"],
        },
        "admission": {
            "exact_all10_transaction_dse_admitted": dse_pass,
            "bit_exact_capacity_ledger_admitted": dse_pass,
            "structural_complexity_width_ledger_admitted": dse_pass,
            "k2_ctx16_promoted_to_rtl_experiment_only":
                dse_pass and selected_name == "K2_CTX16",
            "k4_ctx16_promoted_to_rtl_experiment_only":
                dse_pass and selected_name == "K4_CTX16",
            "k8_ctx16_killed_before_rtl": k8_killed,
            "new_configuration_rtl_vcs_synopsys_admitted": False,
            "sram_macro_port_feasibility_admitted": False,
            "address_timed_pair_schedule_admitted": False,
            "full_network_or_system_speedup_admitted": False,
            "date_headline_or_best_paper_admitted": False,
        },
        "claim_policy": contract["claim_policy"],
    }


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite M52 output")
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
