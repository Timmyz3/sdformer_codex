#!/usr/bin/env python3
"""Independent exact-artifact and arithmetic audit for M94.

The audit does not import or execute the M94 producer.  It reconstructs the
three policy ledgers from the raw result and frozen M89 K6 receipt and parses
all 120 completion markers from the interleaved raw log.
"""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT_DIR = HW / "results/m94_critical_first_fusion_seed_probe_r1_20260824"
CONTRACT = HW / "contracts/m94_critical_first_fusion_seed_contract_r1_20260824.json"
PROBE = HW / "system_simulator/scripts/probe_m94_critical_first_fusion_seed.py"
RAW = RESULT_DIR / "remote_artifacts/m94_critical_first_fusion_seed_probe_r1_20260824.json"
LOG = RESULT_DIR / "remote_artifacts/m94_critical_first_fusion_seed_probe_r1_20260824.log"
RECEIPT = RESULT_DIR / "m94_critical_first_fusion_seed_probe_receipt_r1.json"
M89 = HW / "results/m89_temporal_fanout_hold_screen_r1_20260823/m89_temporal_fanout_hold_screen_receipt.json"
OUTPUT = HERE / "m94_independent_recompute.json"

EXPECTED = {
    "contract": "c639654028525a03331f99a6393721bde45501450870ec15ea62bedabcf087ad",
    "probe": "a74bff6430269a4decf550ac60afd82b98b37eecc7e6896b80639b224304ce43",
    "raw": "a871355741e310508045a047da62659e718f237c6716a7e2fbd2a0be67d7f9a4",
    "log": "c17b47374051d57dd870631ca48e51c7f274f7771af9e9e5693254cd4e604b9a",
    "receipt": "37b0c4a95939dc3ddad9738840257447c199765b939845938e6a939d506c2eb8",
    "m89": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
POLICIES = ("oldest", "critical_first", "sparse_first")
LOG_NAMES = {
    "OLDEST": "oldest",
    "CRITICAL_FIRST": "critical_first",
    "SPARSE_FIRST": "sparse_first",
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
    pattern = re.compile(
        r"^\[M53 K6_CTX16_TEMPORAL_SEED_(OLDEST|CRITICAL_FIRST|SPARSE_FIRST)\] "
        r"([0-9]+)/40 sample=([0-9]+) operator=(\S+)$")
    by_policy = dict((policy, []) for policy in POLICIES)
    text = LOG.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            by_policy[LOG_NAMES[match.group(1)]].append(
                (int(match.group(2)), int(match.group(3)), match.group(4)))
    require(sum(len(rows) for rows in by_policy.values()) == 120,
            "completion marker population is not 120")
    identities = {}
    for policy, rows in by_policy.items():
        require([row[0] for row in rows] == list(range(1, 41)),
                policy + " ordinals are not 1..40")
        ids = [(row[1], row[2]) for row in rows]
        require(len(set(ids)) == 40 and
                all(sum(sample == expected for sample, _ in ids) == 4
                    for expected in range(10)), policy + " log population drift")
        identities[policy] = ids
    require(identities["oldest"] == identities["critical_first"] ==
            identities["sparse_first"], "policy log identity/order drift")
    markers = [line for line in text.splitlines()
               if line.startswith("M94_CRITICAL_FIRST_FUSION_SEED=")]
    require(len(markers) == 1, "summary marker population")
    return identities, json.loads(markers[0].split("=", 1)[1])


def audit_configuration(configuration, log_ids):
    policy = configuration["seed_selector"]["policy"]
    ledger = configuration["record_ledger"]
    records = ledger["records"]
    require(ledger["record_count"] == 40 and len(records) == 40,
            policy + " record count")
    ids = [(row["sample_id"], row["operator"]) for row in records]
    require(ids == log_ids and len(set(ids)) == 40, policy + " raw/log identity drift")
    require(all(row["signed_add_updates"] + row["signed_subtract_updates"] ==
                row["logical_source_updates"] for row in records),
            policy + " record signed conservation")
    per_sample = configuration["per_sample"]
    require(len(per_sample) == 10 and
            [row["sample_id"] for row in per_sample] == list(range(10)),
            policy + " per-sample population/order")
    rebuilt = []
    for sample_id in range(10):
        selected = [row for row in records if row["sample_id"] == sample_id]
        published = per_sample[sample_id]
        for field in ("source_only_cycles", "integrated_cycles", "fusion_groups",
                      "logical_source_updates", "unique_weight_issues",
                      "signed_add_updates", "signed_subtract_updates"):
            require(sum(row[field] for row in selected) == published[field],
                    "{} sample {} {} drift".format(policy, sample_id, field))
        for field in ("maximum_metadata_occupancy", "maximum_complete_occupancy"):
            require(max(row[field] for row in selected) == published[field],
                    "{} sample {} {} drift".format(policy, sample_id, field))
        require(published["signed_add_updates"] + published["signed_subtract_updates"] ==
                published["logical_source_updates"], policy + " sample signed drift")
        rebuilt.append({"sample_id": sample_id,
                        "source": published["source_only_cycles"],
                        "integrated": published["integrated_cycles"],
                        "groups": published["fusion_groups"]})
    sources = [row["source"] for row in rebuilt]
    integrated = [row["integrated"] for row in rebuilt]
    groups = [row["groups"] for row in rebuilt]
    require(sum(sources) == configuration["aggregate_source_only_cycles"] and
            sum(integrated) == configuration["aggregate_integrated_cycles"] and
            sum(groups) == configuration["aggregate_fusion_groups"],
            policy + " aggregate drift")
    dist = configuration["integrated_cycle_distribution"]
    require(dist["count"] == 10 and dist["minimum"] == min(integrated) and
            dist["maximum"] == max(integrated) and
            dist["p50_nearest_rank"] == nearest_rank(integrated, 0.5) and
            dist["p95_nearest_rank"] == nearest_rank(integrated, 0.95),
            policy + " distribution drift")
    selector = configuration["seed_selector"]
    require(selector["selection_events"] == configuration["aggregate_fusion_groups"] and
            selector["base_block_replication_factor"] == 8 and
            selector["additional_per_resident_metadata_bits"] == 6 and
            selector["additional_vector_payload_storage_bytes"] == 0,
            policy + " selector audit drift")
    return {
        "source": sum(sources), "integrated": sum(integrated),
        "p95": nearest_rank(integrated, 0.95), "groups": sum(groups),
        "logical_source_updates": configuration["aggregate_logical_source_updates"],
        "unique_weight_issues": configuration["aggregate_unique_weight_issues"],
        "per_sample": rebuilt, "selector": selector,
    }


def main():
    paths = {"contract": CONTRACT, "probe": PROBE, "raw": RAW,
             "log": LOG, "receipt": RECEIPT, "m89": M89}
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name], name + " SHA drift")
    contract = read_json(CONTRACT)
    raw = read_json(RAW)
    receipt = read_json(RECEIPT)
    m89 = read_json(M89)
    log_ids, marker = parse_log()
    require([row["seed_selector"]["policy"] for row in raw["configurations"]] ==
            list(POLICIES), "configuration policy order")
    rows = dict((policy, audit_configuration(configuration, log_ids[policy]))
                for policy, configuration in zip(POLICIES, raw["configurations"]))
    baselines = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(baselines) == 1, "M89 K6 baseline population")
    baseline = baselines[0]
    base_samples = dict((row["sample_id"], row) for row in baseline["per_sample"])
    require(rows["oldest"]["source"] == baseline["source_cycles"] and
            rows["oldest"]["integrated"] == baseline["integrated_cycles"] and
            rows["oldest"]["p95"] == baseline["p95_integrated_cycles"] and
            all(row["source"] == base_samples[row["sample_id"]]["source"] and
                row["integrated"] == base_samples[row["sample_id"]]["integrated"]
                for row in rows["oldest"]["per_sample"]),
            "oldest exact M89 reproduction drift")

    critical = rows["critical_first"]
    oldest = rows["oldest"]
    sparse = rows["sparse_first"]
    deltas = []
    for candidate in critical["per_sample"]:
        base = base_samples[candidate["sample_id"]]
        old = oldest["per_sample"][candidate["sample_id"]]
        deltas.append({"sample_id": candidate["sample_id"],
                       "source_cycles": candidate["source"] - base["source"],
                       "integrated_cycles": candidate["integrated"] - base["integrated"],
                       "fusion_groups": candidate["groups"] - old["groups"]})
    receipt_deltas = receipt["critical_first_per_sample_deltas_candidate_minus_m89_k6"]
    require([{key: row[key] for key in ("sample_id", "source_cycles", "integrated_cycles")}
             for row in deltas] == receipt_deltas, "critical receipt per-sample delta drift")
    comparison = {
        "source_delta": critical["source"] - oldest["source"],
        "integrated_delta": critical["integrated"] - oldest["integrated"],
        "p95_delta": critical["p95"] - oldest["p95"],
        "fusion_group_delta": critical["groups"] - oldest["groups"],
        "non_source_overhead_delta":
            (critical["integrated"] - critical["source"]) -
            (oldest["integrated"] - oldest["source"]),
    }
    published = receipt["critical_first_comparison_vs_m89_k6"]
    require(comparison == {
        "source_delta": published["source_cycle_delta"],
        "integrated_delta": published["integrated_cycle_delta"],
        "p95_delta": published["p95_integrated_cycle_delta"],
        "fusion_group_delta": published["fusion_group_delta"],
        "non_source_overhead_delta": published["non_source_overhead_delta"]},
        "critical comparison drift")
    require(oldest["logical_source_updates"] == critical["logical_source_updates"] ==
            sparse["logical_source_updates"], "logical work changed across seed policies")

    max_source = contract["predeclared_critical_first_promotion_gates"][
        "maximum_promotable_source_cycles"]
    max_integrated = contract["predeclared_critical_first_promotion_gates"][
        "maximum_promotable_integrated_cycles"]
    require(max_source == (baseline["source_cycles"] * 995) // 1000 and
            max_integrated == (baseline["integrated_cycles"] * 995) // 1000,
            "0.5 percent threshold drift")
    gates = {
        "oldest_exact_reproduction": True,
        "exact_40_record_10_sample_replay": True,
        "signed_add_subtract_conservation": True,
        "new_dependency_edges_equal_zero": True,
        "maximum_metadata_occupancy_le_16": max(
            row["maximum_metadata_occupancy"]
            for row in raw["configurations"][1]["per_sample"]) <= 16,
        "maximum_complete_occupancy_le_16": max(
            row["maximum_complete_occupancy"]
            for row in raw["configurations"][1]["per_sample"]) <= 16,
        "aggregate_source_cycles_le_69614355": critical["source"] <= max_source,
        "aggregate_integrated_cycles_le_76293933": critical["integrated"] <= max_integrated,
        "p95_integrated_cycles_lt_7843680": critical["p95"] < baseline["p95_integrated_cycles"],
        "each_sample_source_cycles_must_not_regress_vs_m89_k6":
            all(row["source_cycles"] <= 0 for row in deltas),
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6":
            all(row["integrated_cycles"] <= 0 for row in deltas),
        "critical_first_selector_score_population_must_be_positive":
            critical["selector"]["selected_standalone_cycle_sum"] > 0,
        "sparse_first_beats_critical_first": sparse["integrated"] < critical["integrated"],
    }
    receipt_gates = dict((key, value) for key, value in receipt["gates"].items()
                         if key != "all_promotion_gates_pass")
    require(gates == receipt_gates, "receipt gate drift")
    require(receipt["gates"]["all_promotion_gates_pass"] is False and
            raw["all_promotion_gates_pass"] is False,
            "all-gates summary drift")
    marker_rows = dict((row["policy"], row) for row in marker["configurations"])
    require(all(marker_rows[p]["source"] == rows[p]["source"] and
                marker_rows[p]["integrated"] == rows[p]["integrated"] and
                marker_rows[p]["p95"] == rows[p]["p95"] and
                marker_rows[p]["non_oldest"] == rows[p]["selector"]["non_oldest_selections"]
                for p in POLICIES), "log summary marker drift")

    output = {
        "schema": "m94_independent_recompute_v1",
        "status": "PASS_ARITHMETIC_NO_GO_PROMOTION",
        "producer_imported_or_executed": False,
        "sha256": dict((name, sha256(path)) for name, path in paths.items()),
        "completion_markers": 120,
        "policies": rows,
        "critical_first_deltas": {"aggregate": comparison, "per_sample": deltas},
        "critical_source_limit": max_source,
        "critical_integrated_limit": max_integrated,
        "critical_source_above_limit": critical["source"] - max_source,
        "critical_integrated_above_limit": critical["integrated"] - max_integrated,
        "gates": gates,
        "all_promotion_gates_pass": False,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
