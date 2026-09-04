#!/usr/bin/env python3
"""Independent receipt/result arithmetic audit for M91.

The producer is hashed and statically reviewed but is not imported or executed.
All numerical results are reconstructed from the raw M91 result/log and the
frozen M89 K6 receipt.
"""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT_DIR = HW / "results/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824"
CONTRACT = HW / "contracts/m91_dependency_safe_fusion_aware_parent_contract_r1_20260824.json"
PROBE = HW / "system_simulator/scripts/probe_m91_dependency_safe_fusion_aware_parent.py"
RECEIPT = RESULT_DIR / "m91_dependency_safe_fusion_aware_parent_probe_receipt_r1.json"
RAW = RESULT_DIR / "remote_artifacts/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824.json"
LOG = RESULT_DIR / "remote_artifacts/m91_dependency_safe_fusion_aware_parent_probe_r2_20260824.log"
M89 = HW / "results/m89_temporal_fanout_hold_screen_r1_20260823/m89_temporal_fanout_hold_screen_receipt.json"
OUTPUT = HERE / "m91_independent_recompute.json"

EXPECTED = {
    "contract": "da4172314986600d49e9ed0f4ade2ebcbec90ad1910d036e166db17356de4b4c",
    "probe": "c6bf6d37713137c3e63067ead2ab0460856098d9b9f3d1c613359b48dc88f97a",
    "receipt": "83a3fe67e592e0fee1b619329e612798eee5da443285d35ce914d0fe2a9539a1",
    "raw": "6245514b51c1d15a62d994be262a9a5da24235ad9c04b8dda919a8d68da70011",
    "log": "ca7150a95bdc91ec84c67edcabe7ebaaa6f3da2ff873c39038a580ca755032b4",
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
            require(key not in output, "duplicate key {} in {}".format(key, path))
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("nonstandard JSON {}".format(value))))


def nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[int(math.ceil(len(ordered) * fraction)) - 1]


def parse_log():
    text = LOG.read_text(encoding="utf-8")
    pattern = re.compile(r"^\[M91 K6\] ([0-9]+)/40 sample=([0-9]+) operator=(\S+)$")
    progress = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            progress.append((int(match.group(1)), int(match.group(2)), match.group(3)))
    require([row[0] for row in progress] == list(range(1, 41)), "log progress not 1..40")
    identities = [(row[1], row[2]) for row in progress]
    require(len(set(identities)) == 40, "duplicate log identity")
    require(all(sum(sample == expected for sample, _ in identities) == 4
                for expected in range(10)), "log 10x4 population drift")
    markers = [line for line in text.splitlines()
               if line.startswith("M91_FUSION_AWARE_PARENT_PROBE=")]
    require(len(markers) == 1, "final log marker population")
    return json.loads(markers[0].split("=", 1)[1]), identities


def aggregate_records(raw):
    records = raw["record_ledger"]
    require(len(records) == 40, "raw record population")
    identities = [(row["sample_id"], row["operator"]) for row in records]
    require(len(set(identities)) == 40, "raw duplicate record identity")
    require(all(row["signed_add_updates"] + row["signed_subtract_updates"] ==
                row["logical_source_updates"] for row in records),
            "record signed conservation")
    rebuilt = []
    for sample_id in range(10):
        selected = [row for row in records if row["sample_id"] == sample_id]
        require(len(selected) == 4, "raw sample population")
        published = raw["per_sample"][sample_id]
        require(published["sample_id"] == sample_id, "per-sample order")
        for field, value in published.items():
            if field == "sample_id":
                continue
            require(all(field in row for row in selected), "record missing " + field)
            calculated = (max(row[field] for row in selected)
                          if field.startswith("maximum_") else
                          sum(row[field] for row in selected))
            require(calculated == value, "sample {} field {} drift".format(sample_id, field))
        rebuilt.append({"sample_id": sample_id,
                        "source": published["source_only_cycles"],
                        "integrated": published["integrated_cycles"]})
    return records, rebuilt, identities


def main():
    paths = {"contract": CONTRACT, "probe": PROBE, "receipt": RECEIPT,
             "raw": RAW, "log": LOG, "m89": M89}
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name], name + " SHA drift")
    contract = read_json(CONTRACT)
    receipt = read_json(RECEIPT)
    raw = read_json(RAW)
    m89 = read_json(M89)
    marker, log_ids = parse_log()
    records, samples, raw_ids = aggregate_records(raw)
    require(log_ids == raw_ids, "raw/log record order differs")

    source_values = [row["source"] for row in samples]
    integrated_values = [row["integrated"] for row in samples]
    source = sum(source_values)
    integrated = sum(integrated_values)
    p95 = nearest_rank(integrated_values, 0.95)
    require(source == raw["aggregate_source_only_cycles"] and
            integrated == raw["aggregate_integrated_cycles"] and
            p95 == raw["integrated_cycle_distribution"]["p95_nearest_rank"],
            "aggregate/distribution drift")
    for field in ("logical_source_updates", "unique_weight_issues", "fusion_groups"):
        require(sum(row[field] for row in raw["per_sample"]) == raw["aggregate_" + field],
                field + " aggregate drift")
    require(marker == {"all_gates": False, "integrated": integrated,
                       "p95": p95, "parent_reselections": 589095,
                       "source": source,
                       "status": "PASS_EXECUTION_NO_GO_PROMOTION"},
            "compact marker drift")

    baselines = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(baselines) == 1, "M89 K6 population")
    baseline = baselines[0]
    base_samples = dict((row["sample_id"], row) for row in baseline["per_sample"])
    deltas = []
    for row in samples:
        base = base_samples[row["sample_id"]]
        deltas.append({"sample_id": row["sample_id"],
                       "source_cycles": row["source"] - base["source"],
                       "integrated_cycles": row["integrated"] - base["integrated"]})
    source_delta = source - baseline["source_cycles"]
    integrated_delta = integrated - baseline["integrated_cycles"]
    p95_delta = p95 - baseline["p95_integrated_cycles"]
    promotion_limit = contract["predeclared_promotion_gates"]["maximum_promotable_integrated_cycles"]
    require(promotion_limit == (baseline["integrated_cycles"] * 99) // 100,
            "one-percent integer threshold drift")
    miss = integrated - promotion_limit
    require(deltas == receipt["per_sample_deltas_candidate_minus_m89_k6"],
            "receipt per-sample delta drift")
    comparison = receipt["comparison"]
    require((source_delta, integrated_delta, p95_delta, miss) ==
            (comparison["source_cycle_delta_candidate_minus_baseline"],
             comparison["integrated_cycle_delta_candidate_minus_baseline"],
             comparison["p95_integrated_cycle_delta_candidate_minus_baseline"],
             comparison["cycles_above_promotion_limit"]),
            "receipt comparison drift")

    counts = raw["selection_counts"]
    selected_total = sum(counts["selected_" + name] for name in
                         ("local_zero", "left", "up", "previous_timestep"))
    require(selected_total == 3240000 and
            counts["fusion_aware_admissions"] +
            counts["empty_resident_canonical_fallback"] == selected_total and
            counts["parent_reselections"] == 589095,
            "selection population drift")
    gates = {
        "exact_40_record_10_sample_replay": len(records) == 40 and len(samples) == 10,
        "signed_add_subtract_conservation": all(
            row["signed_add_updates"] + row["signed_subtract_updates"] ==
            row["logical_source_updates"] for row in raw["per_sample"]),
        "new_dependency_edges_equal_zero": True,
        "maximum_metadata_occupancy_le_16": max(
            row["maximum_metadata_occupancy"] for row in raw["per_sample"]) <= 16,
        "maximum_complete_occupancy_le_16": max(
            row["maximum_complete_occupancy"] for row in raw["per_sample"]) <= 16,
        "aggregate_source_cycles_must_not_exceed_m89_k6_69964176":
            source <= baseline["source_cycles"],
        "aggregate_integrated_cycles_le_75910546": integrated <= promotion_limit,
        "p95_integrated_cycles_lt_7843680": p95 < baseline["p95_integrated_cycles"],
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6":
            all(row["integrated_cycles"] <= 0 for row in deltas),
    }
    receipt_gates = dict((key, value) for key, value in receipt["gates"].items()
                         if key != "all_promotion_gates_pass")
    require(gates == raw["m91"]["gates"] and gates == receipt_gates,
            "gate drift")
    # The receipt has one extra summary boolean; compare it separately.
    require(receipt["gates"]["all_promotion_gates_pass"] == all(gates.values()),
            "all-gates summary drift")

    output = {
        "schema": "m91_independent_recompute_v1",
        "status": "PASS_ARITHMETIC_NO_GO_PROMOTION",
        "producer_imported_or_executed": False,
        "sha256": dict((name, sha256(path)) for name, path in paths.items()),
        "population": {"records": len(records), "samples": len(samples),
                       "operators_per_sample": 4},
        "candidate": {"source": source, "integrated": integrated, "p95": p95},
        "baseline": {"source": baseline["source_cycles"],
                     "integrated": baseline["integrated_cycles"],
                     "p95": baseline["p95_integrated_cycles"]},
        "deltas_candidate_minus_baseline": {
            "source": source_delta, "integrated": integrated_delta, "p95": p95_delta,
            "per_sample": deltas},
        "one_percent_gate": {"maximum_integrated": promotion_limit,
                             "candidate_miss_cycles": miss,
                             "passes": integrated <= promotion_limit},
        "selection_counts": counts,
        "selection_fractions": {
            "reselection_of_all": counts["parent_reselections"] / 3240000.0,
            "reselection_of_fusion_aware":
                counts["parent_reselections"] / float(counts["fusion_aware_admissions"]),
        },
        "gates": gates,
        "all_promotion_gates_pass": all(gates.values()),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
