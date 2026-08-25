#!/usr/bin/env python3
"""Independent arithmetic/provenance audit of the M90 negative screen.

This script does not import or execute the M90 producer.  It verifies the
sealed identities, parses both raw logs, and rebuilds the published population,
distribution, baseline-comparison, and predeclared-gate arithmetic.
"""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT = HW / "results/m90_window_parent_integrated_probe_r1_20260823"
RECEIPT = RESULT / "m90_window_parent_integrated_probe_receipt.json"
CONTRACT = HW / "contracts/m90_window_parent_integrated_probe_contract_r1_20260823.json"
PROBE = HW / "system_simulator/scripts/probe_m90_window_parent_integrated.py"
M53 = HW / ("results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
            "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M89 = HW / ("results/m89_temporal_fanout_hold_screen_r1_20260823/"
            "m89_temporal_fanout_hold_screen_receipt.json")
OUTPUT = HERE / "m90_independent_recompute.json"

EXPECTED = {
    "receipt": "358d1f170b535704ddd032fe33ccc8b6f5492e5134a52f29370a24d69ccf3b09",
    "contract": "aeb339d356e2757333b161983e6063d8cb40ef3871720cdc735f5fe7aca2c450",
    "probe": "fcb75e30d780db2226a9a41c8f6316a3820d357613a7164fd9a51d7f6e9c26f4",
    "m53": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m89": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}


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
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key {} in {}".format(key, path))
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard JSON {}".format(value))))


def nearest_rank(values, percent):
    ordered = sorted(values)
    return ordered[int(math.ceil(percent * len(ordered))) - 1]


def parse_log(path, fanout):
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\[M90 K{} W64\] ([0-9]+)/40 sample=([0-9]+) operator=(\S+)$".format(fanout))
    rows = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            rows.append((int(match.group(1)), int(match.group(2)), match.group(3)))
    require([row[0] for row in rows] == list(range(1, 41)),
            "K{} progress is not exactly 1..40".format(fanout))
    identities = [(row[1], row[2]) for row in rows]
    require(len(set(identities)) == 40, "K{} duplicate record".format(fanout))
    require(all(sum(sample == expected for sample, _ in identities) == 4
                for expected in range(10)), "K{} sample/operator drift".format(fanout))
    markers = [line for line in text.splitlines()
               if line.startswith("M90_WINDOW_PARENT_PROBE=")]
    require(len(markers) == 1, "K{} compact marker population".format(fanout))
    return json.loads(markers[0].split("=", 1)[1])


def audit_result(payload, fanout, marker):
    samples = payload["per_sample"]
    require(len(samples) == 10 and [row["sample_id"] for row in samples] == list(range(10)),
            "K{} per-sample population/order".format(fanout))
    sources = [row["source_only_cycles"] for row in samples]
    integrated = [row["integrated_cycles"] for row in samples]
    require(sum(sources) == payload["aggregate_source_only_cycles"], "source sum")
    require(sum(integrated) == payload["aggregate_integrated_cycles"], "integrated sum")
    for key in ("logical_source_updates", "unique_weight_issues", "fusion_groups"):
        require(sum(row[key] for row in samples) == payload["aggregate_" + key], key + " sum")
    distribution = payload["integrated_cycle_distribution"]
    require(distribution["count"] == 10 and
            distribution["minimum"] == min(integrated) and
            distribution["maximum"] == max(integrated) and
            distribution["p50_nearest_rank"] == nearest_rank(integrated, 0.50) and
            distribution["p95_nearest_rank"] == nearest_rank(integrated, 0.95),
            "K{} distribution drift".format(fanout))
    require(marker["source"] == sum(sources) and marker["integrated"] == sum(integrated) and
            marker["p95"] == max(integrated) and marker["fanout"] == fanout,
            "K{} log/result marker mismatch".format(fanout))
    origins = payload["window_parent"]["origin_counts"]
    distances = payload["window_parent"]["distance_counts"]
    require(sum(origins.values()) == 3240000 and origins["window"] == 788279,
            "K{} origin population".format(fanout))
    require(sum(int(value) for value in distances.values()) == origins["window"] and
            all(1 <= int(key) <= 64 for key in distances),
            "K{} distance population".format(fanout))
    return {
        "source": sum(sources), "integrated": sum(integrated),
        "p95": max(integrated), "origins": origins,
        "distance_population": sum(int(value) for value in distances.values()),
        "waits": dict((field, sum(row[field] for row in samples)) for field in (
            "command_or_state_wait_cycles", "fusion_hold_wait_cycles",
            "parent_wait_cycles", "response_or_context_wait_cycles",
            "weight_dma_wait_cycles")),
    }


def main():
    paths = {"receipt": RECEIPT, "contract": CONTRACT, "probe": PROBE,
             "m53": M53, "m89": M89}
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name], name + " SHA drift")
    receipt = read_json(RECEIPT)
    contract = read_json(CONTRACT)
    artifacts = receipt["identity"]["artifacts"]
    payloads = {}
    markers = {}
    artifact_hashes = {}
    for fanout, prefix in ((4, "k4"), (6, "k6")):
        result_item = artifacts[prefix + "_result"]
        log_item = artifacts[prefix + "_log"]
        result_path = RESULT / result_item["path"]
        log_path = RESULT / log_item["path"]
        for label, item, path in ((prefix + "_result", result_item, result_path),
                                  (prefix + "_log", log_item, log_path)):
            actual = sha256(path)
            require(actual == item["sha256"], label + " SHA drift")
            artifact_hashes[label] = actual
        payloads[fanout] = read_json(result_path)
        markers[fanout] = parse_log(log_path, fanout)
    rows = dict((fanout, audit_result(payloads[fanout], fanout, markers[fanout]))
                for fanout in (4, 6))
    require(payloads[4]["window_parent"] == payloads[6]["window_parent"],
            "K4/K6 parent selection population differs")

    m53 = read_json(M53)
    legacy_rows = [row for row in m53["configuration_ledgers"]
                   if row["name"] == "K4_CTX16_TEMPORAL"]
    require(len(legacy_rows) == 1, "M53 K4 baseline population")
    k4_base = legacy_rows[0]
    m89 = read_json(M89)
    k6_rows = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(k6_rows) == 1, "M89 K6 baseline population")
    k6_base = k6_rows[0]

    comparisons = {
        "k4": {
            "source_delta_candidate_minus_baseline": rows[4]["source"] - k4_base["aggregate_source_only_cycles"],
            "integrated_delta_candidate_minus_baseline": rows[4]["integrated"] - k4_base["aggregate_integrated_cycles"],
            "p95_delta_candidate_minus_baseline": rows[4]["p95"] - k4_base["integrated_cycle_distribution"]["p95_nearest_rank"],
        },
        "k6": {
            "source_delta_candidate_minus_baseline": rows[6]["source"] - k6_base["source_cycles"],
            "integrated_delta_candidate_minus_baseline": rows[6]["integrated"] - k6_base["integrated_cycles"],
            "p95_delta_candidate_minus_baseline": rows[6]["p95"] - k6_base["p95_integrated_cycles"],
        },
    }
    published = receipt["comparisons"]
    require(comparisons["k4"] == {
        "source_delta_candidate_minus_baseline": -published["k4_window64_vs_m53_k4_temporal"]["source_cycle_improvement"],
        "integrated_delta_candidate_minus_baseline": published["k4_window64_vs_m53_k4_temporal"]["integrated_cycle_regression"],
        "p95_delta_candidate_minus_baseline": published["k4_window64_vs_m53_k4_temporal"]["p95_regression_cycles"]},
        "K4 published comparison drift")
    require(comparisons["k6"] == {
        "source_delta_candidate_minus_baseline": published["k6_window64_vs_m89_k6_temporal"]["source_cycle_regression"],
        "integrated_delta_candidate_minus_baseline": published["k6_window64_vs_m89_k6_temporal"]["integrated_cycle_regression"],
        "p95_delta_candidate_minus_baseline": published["k6_window64_vs_m89_k6_temporal"]["p95_regression_cycles"]},
        "K6 published comparison drift")

    gates = {
        "all_40_records_complete": True,
        "maximum_metadata_occupancy_le_16": all(
            max(row["maximum_metadata_occupancy"] for row in payloads[k]["per_sample"]) <= 16 for k in (4, 6)),
        "maximum_complete_occupancy_le_16": all(
            max(row["maximum_complete_occupancy"] for row in payloads[k]["per_sample"]) <= 16 for k in (4, 6)),
        "window_parent_exercised": rows[6]["origins"]["window"] > 0,
        "k6_source_cycles_improve_vs_m89_k6_by_at_least_2_percent":
            rows[6]["source"] * 100 <= k6_base["source_cycles"] * 98,
        "k6_integrated_cycles_improve_vs_m89_k6_by_at_least_1_percent":
            rows[6]["integrated"] * 100 <= k6_base["integrated_cycles"] * 99,
        "k6_p95_integrated_cycles_improves_vs_m89_k6":
            rows[6]["p95"] < k6_base["p95_integrated_cycles"],
    }
    require(all(gates[key] == receipt["predeclared_gate_results"][key] for key in gates),
            "gate result drift")
    output = {
        "schema": "m90_independent_recompute_v1",
        "status": "PASS_ARITHMETIC_NO_GO_PROMOTION",
        "producer_imported_or_executed": False,
        "artifact_sha256": artifact_hashes,
        "rows": rows,
        "comparisons": comparisons,
        "gates": gates,
        "all_performance_promotion_gates_pass": all(gates.values()),
        "window_fraction": 788279.0 / 3240000.0,
        "contract_status": contract["status"],
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
