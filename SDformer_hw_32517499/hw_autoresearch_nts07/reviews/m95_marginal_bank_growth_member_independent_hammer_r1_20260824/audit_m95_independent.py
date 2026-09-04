#!/usr/bin/env python3
"""Independent arithmetic/provenance audit for M95.

The producer is not imported or executed.  Raw per-record/per-sample ledgers,
the interleaved 120-marker log, the receipt, and frozen M89 K6 baseline are
checked directly.
"""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT_DIR = HW / "results/m95_marginal_bank_growth_member_probe_r1_20260824"
CONTRACT = HW / "contracts/m95_marginal_bank_growth_member_contract_r1_20260824.json"
PROBE = HW / "system_simulator/scripts/probe_m95_marginal_bank_growth_member.py"
RAW = RESULT_DIR / "remote_artifacts/m95_marginal_bank_growth_member_probe.json"
LOG = RESULT_DIR / "remote_artifacts/m95_marginal_bank_growth_member_probe_r1_20260824.log"
RECEIPT = RESULT_DIR / "m95_marginal_bank_growth_member_probe_receipt_r1.json"
M89 = HW / "results/m89_temporal_fanout_hold_screen_r1_20260823/m89_temporal_fanout_hold_screen_receipt.json"
OUTPUT = HERE / "m95_independent_recompute.json"

EXPECTED = {
    "contract": "eb17a8fe5d05bf3cdfef2b51dde9f9c16f16a58c77a64a0c292b4040f125f8a5",
    "probe": "819a392fe2fc8a7e9071877c3d49689e69d34213ea3f9eec20d1bb3d010e2d1b",
    "raw": "3fda60194dec1c47fc7b41cbd7b7d6840808b8060439c5c6472839ca728f9842",
    "log": "5dc8cd31486b818931fe3616f37d88afc4cdfe324bac67a6dacc5d1e0de29f70",
    "receipt": "80b51d52379f8215a2938dd28343a1771c57bc47b154946c2ce3baf55fcbab3b",
    "m89": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
POLICIES = (
    "saved_first_reproduction",
    "marginal_growth_primary",
    "standalone_heavy_negative_control",
)
LOG_NAMES = dict((name.upper(), name) for name in POLICIES)


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
    names = "|".join(LOG_NAMES)
    pattern = re.compile(
        r"^\[M53 K6_CTX16_TEMPORAL_MEMBER_(" + names +
        r")\] ([0-9]+)/40 sample=([0-9]+) operator=(\S+)$")
    by_policy = dict((policy, []) for policy in POLICIES)
    text = LOG.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            by_policy[LOG_NAMES[match.group(1)]].append(
                (int(match.group(2)), int(match.group(3)), match.group(4)))
    require(sum(len(rows) for rows in by_policy.values()) == 120,
            "completion marker population")
    ids = {}
    for policy, rows in by_policy.items():
        require([row[0] for row in rows] == list(range(1, 41)),
                policy + " progress ordinals")
        current = [(row[1], row[2]) for row in rows]
        require(len(set(current)) == 40 and
                all(sum(sample == expected for sample, _ in current) == 4
                    for expected in range(10)), policy + " 10x4 population")
        ids[policy] = current
    require(ids[POLICIES[0]] == ids[POLICIES[1]] == ids[POLICIES[2]],
            "policy identity/order drift")
    markers = [line for line in text.splitlines()
               if line.startswith("M95_MARGINAL_BANK_GROWTH_MEMBER=")]
    require(len(markers) == 1, "summary marker population")
    return ids, json.loads(markers[0].split("=", 1)[1])


def audit_configuration(configuration, log_ids):
    policy = configuration["member_selector"]["policy"]
    records = configuration["record_ledger"]["records"]
    require(configuration["record_ledger"]["record_count"] == 40 and
            len(records) == 40, policy + " record count")
    raw_ids = [(row["sample_id"], row["operator"]) for row in records]
    require(raw_ids == log_ids and len(set(raw_ids)) == 40,
            policy + " raw/log identities")
    require(all(row["signed_add_updates"] + row["signed_subtract_updates"] ==
                row["logical_source_updates"] for row in records),
            policy + " record signed conservation")
    samples = configuration["per_sample"]
    require(len(samples) == 10 and
            [row["sample_id"] for row in samples] == list(range(10)),
            policy + " sample population/order")
    rebuilt = []
    for sample_id, published in enumerate(samples):
        selected = [row for row in records if row["sample_id"] == sample_id]
        require(len(selected) == 4, policy + " operator population")
        for field in ("source_only_cycles", "integrated_cycles", "fusion_groups",
                      "unique_weight_issues", "logical_source_updates",
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
                        "groups": published["fusion_groups"],
                        "unique": published["unique_weight_issues"]})
    source = sum(row["source"] for row in rebuilt)
    integrated = sum(row["integrated"] for row in rebuilt)
    groups = sum(row["groups"] for row in rebuilt)
    unique = sum(row["unique"] for row in rebuilt)
    require((source, integrated, groups, unique) ==
            (configuration["aggregate_source_only_cycles"],
             configuration["aggregate_integrated_cycles"],
             configuration["aggregate_fusion_groups"],
             configuration["aggregate_unique_weight_issues"]),
            policy + " aggregate drift")
    values = [row["integrated"] for row in rebuilt]
    dist = configuration["integrated_cycle_distribution"]
    require(dist["count"] == 10 and dist["minimum"] == min(values) and
            dist["maximum"] == max(values) and
            dist["p50_nearest_rank"] == nearest_rank(values, 0.5) and
            dist["p95_nearest_rank"] == nearest_rank(values, 0.95),
            policy + " distribution drift")
    selector = configuration["member_selector"]
    require(selector["base_block_replication_factor"] == 8 and
            selector["candidate_evaluations"] > 0 and
            selector["additional_per_resident_metadata_bits"] == 0 and
            selector["additional_vector_payload_storage_bytes"] == 0 and
            selector["new_candidate_evaluation_lanes"] == 0,
            policy + " selector contract drift")
    for field in ("candidate_evaluations", "candidate_standalone_cycle_sum",
                  "current_union_cycle_sum", "fused_union_cycle_sum",
                  "saved_cycle_sum", "marginal_growth_cycle_sum"):
        require(selector[field] % 8 == 0, policy + " eight-block scale " + field)
    require(selector["saved_cycle_sum"] ==
            selector["current_union_cycle_sum"] +
            selector["candidate_standalone_cycle_sum"] -
            selector["fused_union_cycle_sum"] and
            selector["marginal_growth_cycle_sum"] ==
            selector["fused_union_cycle_sum"] -
            selector["current_union_cycle_sum"],
            policy + " selector audit identity")
    return {"source": source, "integrated": integrated,
            "p95": nearest_rank(values, 0.95), "groups": groups,
            "unique": unique,
            "logical": configuration["aggregate_logical_source_updates"],
            "per_sample": rebuilt, "selector": selector}


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
    require([row["member_selector"]["policy"] for row in raw["configurations"]] ==
            list(POLICIES), "policy order")
    rows = dict((policy, audit_configuration(configuration, log_ids[policy]))
                for policy, configuration in zip(POLICIES, raw["configurations"]))
    baselines = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(baselines) == 1, "M89 K6 baseline population")
    baseline = baselines[0]
    refs = dict((row["sample_id"], row) for row in baseline["per_sample"])
    saved = rows[POLICIES[0]]
    marginal = rows[POLICIES[1]]
    standalone = rows[POLICIES[2]]
    require(saved["source"] == baseline["source_cycles"] and
            saved["integrated"] == baseline["integrated_cycles"] and
            saved["p95"] == baseline["p95_integrated_cycles"] and
            all(row["source"] == refs[row["sample_id"]]["source"] and
                row["integrated"] == refs[row["sample_id"]]["integrated"]
                for row in saved["per_sample"]), "saved-first exact reproduction")
    deltas = []
    for row in marginal["per_sample"]:
        base = refs[row["sample_id"]]
        old = saved["per_sample"][row["sample_id"]]
        deltas.append({"sample_id": row["sample_id"],
                       "source_cycles": row["source"] - base["source"],
                       "integrated_cycles": row["integrated"] - base["integrated"],
                       "fusion_groups": row["groups"] - old["groups"],
                       "unique_weight_issues": row["unique"] - old["unique"]})
    receipt_deltas = receipt["marginal_growth_per_sample_deltas_candidate_minus_m89_k6"]
    require([{key: row[key] for key in ("sample_id", "source_cycles", "integrated_cycles")}
             for row in deltas] == receipt_deltas, "receipt per-sample deltas")
    comparison = {
        "source_delta": marginal["source"] - saved["source"],
        "integrated_delta": marginal["integrated"] - saved["integrated"],
        "p95_delta": marginal["p95"] - saved["p95"],
        "fusion_group_delta": marginal["groups"] - saved["groups"],
        "unique_weight_issue_delta": marginal["unique"] - saved["unique"],
        "non_source_overhead_delta":
            (marginal["integrated"] - marginal["source"]) -
            (saved["integrated"] - saved["source"]),
        "candidate_evaluation_delta":
            marginal["selector"]["candidate_evaluations"] -
            saved["selector"]["candidate_evaluations"],
    }
    published = receipt["marginal_growth_comparison_vs_m89_k6"]
    require({key: comparison[key] for key in (
        "source_delta", "integrated_delta", "p95_delta", "fusion_group_delta",
        "non_source_overhead_delta", "candidate_evaluation_delta")} == {
        "source_delta": published["source_cycle_delta"],
        "integrated_delta": published["integrated_cycle_delta"],
        "p95_delta": published["p95_integrated_cycle_delta"],
        "fusion_group_delta": published["fusion_group_delta"],
        "non_source_overhead_delta": published["non_source_overhead_delta"],
        "candidate_evaluation_delta": published["candidate_evaluation_delta"]},
        "receipt comparison drift")
    require(saved["logical"] == marginal["logical"] == standalone["logical"],
            "logical source work changed")
    max_source = contract["predeclared_marginal_growth_promotion_gates"][
        "maximum_promotable_source_cycles"]
    max_integrated = contract["predeclared_marginal_growth_promotion_gates"][
        "maximum_promotable_integrated_cycles"]
    require(max_source == (baseline["source_cycles"] * 995) // 1000 and
            max_integrated == (baseline["integrated_cycles"] * 995) // 1000,
            "0.5 percent threshold drift")
    gates = {
        "saved_first_exact_reproduction": True,
        "exact_40_record_10_sample_replay": True,
        "signed_add_subtract_conservation": True,
        "new_dependency_edges_equal_zero": True,
        "maximum_metadata_occupancy_le_16": max(
            row["maximum_metadata_occupancy"]
            for row in raw["configurations"][1]["per_sample"]) <= 16,
        "maximum_complete_occupancy_le_16": max(
            row["maximum_complete_occupancy"]
            for row in raw["configurations"][1]["per_sample"]) <= 16,
        "aggregate_source_cycles_le_69614355": marginal["source"] <= max_source,
        "aggregate_integrated_cycles_le_76293933": marginal["integrated"] <= max_integrated,
        "p95_integrated_cycles_lt_7843680": marginal["p95"] < baseline["p95_integrated_cycles"],
        "each_sample_source_cycles_must_not_regress_vs_m89_k6":
            all(row["source_cycles"] <= 0 for row in deltas),
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6":
            all(row["integrated_cycles"] <= 0 for row in deltas),
        "candidate_evaluation_population_must_be_positive":
            marginal["selector"]["candidate_evaluations"] > 0,
        "standalone_heavy_beats_marginal_growth":
            standalone["integrated"] < marginal["integrated"],
    }
    receipt_gates = dict((key, value) for key, value in receipt["gates"].items()
                         if key != "all_promotion_gates_pass")
    require(gates == receipt_gates and
            receipt["gates"]["all_promotion_gates_pass"] is False and
            raw["all_promotion_gates_pass"] is False, "gate drift")
    marker_rows = dict((row["policy"], row) for row in marker["configurations"])
    require(all(marker_rows[p]["source"] == rows[p]["source"] and
                marker_rows[p]["integrated"] == rows[p]["integrated"] and
                marker_rows[p]["p95"] == rows[p]["p95"] and
                marker_rows[p]["candidate_evaluations"] ==
                rows[p]["selector"]["candidate_evaluations"]
                for p in POLICIES), "log summary marker drift")
    output = {
        "schema": "m95_independent_recompute_v1",
        "status": "PASS_ARITHMETIC_NO_GO_PROMOTION",
        "producer_imported_or_executed": False,
        "sha256": dict((name, sha256(path)) for name, path in paths.items()),
        "completion_markers": 120,
        "policies": rows,
        "marginal_deltas": {"aggregate": comparison, "per_sample": deltas},
        "source_limit": max_source,
        "integrated_limit": max_integrated,
        "source_above_limit": marginal["source"] - max_source,
        "integrated_above_limit": marginal["integrated"] - max_integrated,
        "gates": gates,
        "all_promotion_gates_pass": False,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
