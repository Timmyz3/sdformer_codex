#!/usr/bin/env python3
"""Fail-closed validator for the canonical M52-r1 high-fanout DSE."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = HW_ROOT / "contracts/m52_high_fanout_context16_dse_contract_r1_20260823.json"
ANALYZER = HW_ROOT / "system_simulator/scripts/analyze_m52_high_fanout_context16_dse.py"
RESULT = HW_ROOT / "results/m52_high_fanout_context16_dse_r1_20260823/m52_high_fanout_context16_dse.json"
EXPECTED = {
    "contract": "9aab440911d8a1dbe5b0465ca4af31427131b7a27bebb4f9cfc9451689e5a173",
    "analyzer": "9202a7021bfaa993a3028fcf39a679dbafcfb9fee836ddbd2435f3fdc044fdbc",
    "result": "d60567fecd891e9da0fc1b5bb0d88f4bb7e8e93faa92092037fc46d63dcde50b",
}
EXPECTED_CONFIGS = {
    "K2_CTX16": {
        "fanout": 2, "contexts": 16, "source": 86239944,
        "integrated": 93116008, "p95": 9446640,
        "capacity": 176560, "headroom": 17168,
        "metadata_bits": 50, "metadata_bytes": 8,
        "rmw_paths": 192, "bank_terms": 1536, "push": 2,
    },
    "K4_CTX8": {
        "fanout": 4, "contexts": 8, "source": 73986456,
        "integrated": 87459112, "p95": 8860624,
        "capacity": 174352, "headroom": 19376,
        "metadata_bits": 88, "metadata_bytes": 16,
        "rmw_paths": 384, "bank_terms": 3072, "push": 4,
    },
    "K4_CTX16": {
        "fanout": 4, "contexts": 16, "source": 70821488,
        "integrated": 81921184, "p95": 8376280,
        "capacity": 176688, "headroom": 17040,
        "metadata_bits": 92, "metadata_bytes": 16,
        "rmw_paths": 384, "bank_terms": 3072, "push": 4,
    },
    "K8_CTX16": {
        "fanout": 8, "contexts": 16, "source": 71437992,
        "integrated": 78082032, "p95": 8029048,
        "capacity": 176816, "headroom": 16912,
        "metadata_bits": 176, "metadata_bytes": 24,
        "rmw_paths": 768, "bank_terms": 6144, "push": 8,
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise RuntimeError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[int(math.ceil(len(ordered) * percentile)) - 1]


def canonical_record_sha(records):
    payload = (json.dumps(records, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def by_name(rows):
    result = {}
    for row in rows:
        name = row["name"]
        require(name not in result, "duplicate configuration: {}".format(name))
        result[name] = row
    return result


def validate_capacity_and_complexity(summary, expected):
    fanout = expected["fanout"]
    contexts = expected["contexts"]
    context_id_bits = (contexts - 1).bit_length()
    metadata_bits = fanout * context_id_bits + (fanout - 1) + 8 + fanout * 16 + 1
    metadata_bytes = ((metadata_bits + 63) // 64) * 8
    fixed_bytes = 24576 + 4560 + 640 + 136800 + 1280 + 3904
    capacity_bytes = fixed_bytes + contexts * (228 + 64) + 16 * metadata_bytes
    require(metadata_bits == expected["metadata_bits"] and
            metadata_bytes == expected["metadata_bytes"] and
            capacity_bytes == expected["capacity"],
            "independent capacity arithmetic mismatch")

    capacity = summary["capacity"]
    require(capacity["context_id_bits"] == context_id_bits and
            capacity["response_metadata_payload_bits"] == metadata_bits and
            capacity["response_metadata_aligned_bytes_per_entry"] == metadata_bytes and
            capacity["combined_local_capacity_bytes"] == capacity_bytes and
            capacity["local_capacity_headroom_bytes"] ==
                193728 - capacity_bytes == expected["headroom"] and
            capacity["minimum_headroom_bytes"] == 16384 and
            capacity["headroom_gate_pass"] is True and
            capacity["external_accumulator_spill_permitted"] is False and
            capacity["double_weight_buffer_permitted"] is False,
            "reported capacity ledger mismatch")

    complexity = summary["complexity"]
    require(complexity["accumulator_read_modify_write_paths_per_response"] ==
                fanout * 96 == expected["rmw_paths"] and
            complexity["signed_bank_terms_per_response"] ==
                fanout * 8 * 96 == expected["bank_terms"] and
            complexity["atomic_complete_vector_push_count"] ==
                fanout == expected["push"] and
            complexity["atomic_complete_payload_bits_excluding_tags"] ==
                fanout * 96 * 19,
            "structural complexity arithmetic mismatch")


def validate_record_ledger(configuration, expected):
    ledger = configuration["record_ledger"]
    records = ledger["records"]
    require(ledger["record_count"] == len(records) == 40 and
            ledger["canonical_sha256"] == canonical_record_sha(records),
            "record ledger population/digest mismatch")
    operators_by_sample = {}
    for row in records:
        operators_by_sample.setdefault(row["sample_id"], set()).add(row["operator"])
    require(set(operators_by_sample) == set(range(10)) and
            all(len(value) == 4 for value in operators_by_sample.values()),
            "record sample/operator identity mismatch")

    per_sample = configuration["per_sample"]
    require(len(per_sample) == 10 and
            [row["sample_id"] for row in per_sample] == list(range(10)),
            "per-sample population/order mismatch")
    rebuilt_source = []
    rebuilt_integrated = []
    for sample in per_sample:
        records_for_sample = [row for row in records
                              if row["sample_id"] == sample["sample_id"]]
        source = sum(row["source_only_cycles"] for row in records_for_sample)
        integrated = sum(row["integrated_cycles"] for row in records_for_sample)
        require(sample["source_only_cycles"] == source and
                sample["integrated_cycles"] == integrated and
                sample["maximum_metadata_occupancy"] == max(
                    row["maximum_metadata_occupancy"] for row in records_for_sample) and
                sample["maximum_complete_occupancy"] == max(
                    row["maximum_complete_occupancy"] for row in records_for_sample) and
                sample["maximum_resident_occupancy"] == max(
                    row["maximum_resident_occupancy"] for row in records_for_sample),
                "per-sample record reconstruction mismatch")
        require(sample["maximum_metadata_occupancy"] <= 16 and
                sample["maximum_complete_occupancy"] <= 16 and
                sample["maximum_resident_occupancy"] <= expected["contexts"],
                "per-sample physical occupancy overflow")
        rebuilt_source.append(source)
        rebuilt_integrated.append(integrated)
    require(sum(rebuilt_source) == expected["source"] and
            sum(rebuilt_integrated) == expected["integrated"] and
            nearest_rank(rebuilt_integrated, 0.95) == expected["p95"] and
            configuration["aggregate_source_only_cycles"] == expected["source"] and
            configuration["aggregate_integrated_cycles"] == expected["integrated"] and
            configuration["integrated_cycle_distribution"]["p95_nearest_rank"] ==
                expected["p95"],
            "configuration aggregate/percentile mismatch")


def validate(rerun):
    require(sha256_path(CONTRACT) == EXPECTED["contract"], "contract SHA mismatch")
    require(sha256_path(ANALYZER) == EXPECTED["analyzer"], "analyzer SHA mismatch")
    require(sha256_path(RESULT) == EXPECTED["result"], "result SHA mismatch")
    contract = strict_json(CONTRACT)
    result = strict_json(RESULT)
    require(contract["schema"] == "m52_high_fanout_context16_dse_contract_v1" and
            result["schema"] == "m52_high_fanout_context16_dse_result_v1",
            "schema mismatch")
    for name, identity in contract["inputs"].items():
        path = HW_ROOT / identity["path"]
        require(path.is_file() and sha256_path(path) == identity["sha256"],
                "upstream identity mismatch: {}".format(name))
    require(result["identity"] == {
        "contract_sha256": EXPECTED["contract"],
        "analyzer_sha256": EXPECTED["analyzer"],
        "inputs_sha256": dict((name, row["sha256"])
                               for name, row in contract["inputs"].items()),
        "guard_widening_occurrences": 1,
    }, "embedded identity mismatch")

    summaries = by_name(result["configuration_summaries"])
    ledgers = by_name(result["configuration_ledgers"])
    require(set(summaries) == set(EXPECTED_CONFIGS) == set(ledgers),
            "configuration set mismatch")
    for name, expected in EXPECTED_CONFIGS.items():
        summary = summaries[name]
        ledger = ledgers[name]
        require(summary["destination_fanout_k"] == expected["fanout"] and
                summary["resident_contexts"] == expected["contexts"] and
                summary["aggregate_source_only_cycles"] == expected["source"] and
                summary["aggregate_integrated_cycles"] == expected["integrated"] and
                summary["integrated_cycle_distribution"]["p95_nearest_rank"] ==
                    expected["p95"],
                "configuration summary mismatch: {}".format(name))
        validate_capacity_and_complexity(summary, expected)
        validate_record_ledger(ledger, expected)

    inherited = result["inherited_k2_ctx8_reference"]
    require(inherited["aggregate_source_only_cycles"] == 88269520 and
            inherited["aggregate_integrated_cycles"] == 95047672 and
            inherited["integrated_cycle_distribution"]["p95_nearest_rank"] == 9681752 and
            inherited["capacity"]["combined_local_capacity_bytes"] == 174224 and
            inherited["capacity"]["local_capacity_headroom_bytes"] == 19504 and
            inherited["complexity"]["accumulator_read_modify_write_paths_per_response"] == 192,
            "inherited K2-C8 reference mismatch")

    gates = result["promotion_and_kill_gates"]
    k2_p95 = 9681752
    k2c16_p95 = EXPECTED_CONFIGS["K2_CTX16"]["p95"]
    k4_p95 = EXPECTED_CONFIGS["K4_CTX16"]["p95"]
    k8_p95 = EXPECTED_CONFIGS["K8_CTX16"]["p95"]
    require((k2_p95 - k2c16_p95) * 50 >= k2_p95 and
            (k2c16_p95 - k4_p95) * 10 >= k2c16_p95 and
            (k2_p95 - k4_p95) * 10 >= k2_p95 and
            (95047672 - 81921184) * 10 >= 95047672 and
            (k4_p95 - k8_p95) * 20 < k4_p95 and
            EXPECTED_CONFIGS["K8_CTX16"]["source"] >=
                EXPECTED_CONFIGS["K4_CTX16"]["source"],
            "independent promotion/kill arithmetic mismatch")
    require(gates["k2_ctx16_all_promotion_gates_pass"] is True and
            gates["k4_ctx16_all_promotion_gates_pass"] is True and
            gates["k8_incremental_p95_improvement_below_5pct"] is True and
            gates["k8_source_cycles_not_lower_than_k4"] is True and
            gates["k8_killed_by_predeclared_complexity_gate"] is True and
            gates["selected_configuration"] == "K4_CTX16",
            "reported promotion/kill gates mismatch")

    selected = ledgers["K4_CTX16"]
    pair = result["conservative_pair_upper_bound"]
    values = []
    for reported, transaction in zip(pair["per_sample"], selected["per_sample"]):
        value = transaction["integrated_cycles"] + 1658880
        require(reported["sample_id"] == transaction["sample_id"] and
                reported["transaction_integrated_cycles"] ==
                    transaction["integrated_cycles"] and
                reported["serialized_weight_load_cycles_added"] == 1658880 and
                reported["conservative_pair_upper_bound_cycles"] == value,
                "pair upper-bound reconstruction mismatch")
        values.append(value)
    require(sum(values) == pair["aggregate_cycles"] and
            min(values) == 9724960 and max(values) == 10035160 and
            nearest_rank(values, 0.50) == 9819960 and
            nearest_rank(values, 0.95) == 10035160 and
            pair["distribution"]["p95_nearest_rank"] == 10035160 and
            pair["address_timed_pair_replayed"] is False,
            "pair distribution mismatch")
    model = result["conditional_frozen_compute_model"]
    denominator = 188824491 + 2636515 + 10035160
    require(denominator == 201496166 and
            model["conditional_total_cycles"] == denominator and
            model["conditional_compute_speedup"]["numerator"] == 620868243 and
            model["conditional_compute_speedup"]["denominator"] == denominator and
            620868243 >= 3 * denominator and
            model["three_x_crossing_in_conditional_model"] is True and
            model["system_or_end_to_end_speedup_admitted"] is False,
            "conditional frozen-compute reconstruction mismatch")

    require(result["status"] ==
            "PASS_PROMOTE_K4_CTX16_TO_RTL_EXPERIMENT_K8_KILLED_SYSTEM_UNADMITTED" and
            result["admission"] == {
                "exact_all10_transaction_dse_admitted": True,
                "bit_exact_capacity_ledger_admitted": True,
                "structural_complexity_width_ledger_admitted": True,
                "k2_ctx16_promoted_to_rtl_experiment_only": False,
                "k4_ctx16_promoted_to_rtl_experiment_only": True,
                "k8_ctx16_killed_before_rtl": True,
                "new_configuration_rtl_vcs_synopsys_admitted": False,
                "sram_macro_port_feasibility_admitted": False,
                "address_timed_pair_schedule_admitted": False,
                "full_network_or_system_speedup_admitted": False,
                "date_headline_or_best_paper_admitted": False,
            } and result["claim_policy"] == contract["claim_policy"],
            "status/admission/claim boundary mismatch")

    if rerun:
        with tempfile.TemporaryDirectory(prefix="m52_validate_") as directory:
            output = Path(directory) / "rebuilt.json"
            subprocess.check_call([
                sys.executable, str(ANALYZER), "--output", str(output)
            ])
            require(sha256_path(output) == EXPECTED["result"],
                    "deterministic rerun SHA mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    validate(args.rerun)
    print("PASS M52-r1 fail-closed validation{}".format(
        " with deterministic all10 rerun" if args.rerun else ""))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("FAIL M52 validation: {}".format(error), file=sys.stderr)
        sys.exit(1)
