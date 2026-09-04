#!/usr/bin/env python3
"""Independent provenance and arithmetic audit for the M96 stage-1 screen.

The producer is never imported or executed.  This checker reads the frozen raw
ledger, contract, receipt, log, and M89 receipt, then rebuilds all reductions.
"""

from __future__ import print_function

import collections
import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT_DIR = HW / "results/m96_fixed_group_reversible_bank_skew_stage1_r1_20260824"
CONTRACT = HW / "contracts/m96_fixed_group_reversible_bank_skew_contract_r1_20260824.json"
PROBE = HW / "system_simulator/scripts/probe_m96_fixed_group_reversible_bank_skew.py"
RAW = RESULT_DIR / "m96_fixed_group_reversible_bank_skew_stage1.json"
LOG = RESULT_DIR / "m96_fixed_group_reversible_bank_skew_stage1_r1_20260824.log"
RECEIPT = RESULT_DIR / "m96_fixed_group_reversible_bank_skew_stage1_receipt_r1.json"
M89 = HW / "results/m89_temporal_fanout_hold_screen_r1_20260823/m89_temporal_fanout_hold_screen_receipt.json"
OUTPUT = HERE / "m96_independent_recompute.json"

EXPECTED = {
    "contract": "251ebb1f19abd07166e5af99e872cbe0013dff038836638ed5ebdeb783e496fc",
    "probe": "e722a64707ea209c8ef67bdc4affb2f630f1742dd910375357fc5140cbc3b0f4",
    "raw": "ab2f1e2219d7f554d4ee7350ae3e0e37b7662ac03d748cbef193568b27fc1def",
    "log": "520b7742b6dd0f2d9b7c630e88e2f3d3733ac5e7902b797176ebbab0f1eafe5f",
    "receipt": "a1da7a401d8cb4a97fd5633a9a06a7dde9fdc49f3ff0d54bc658e701105e834a",
    "m89": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
MODES = ("H0_IDENTITY", "H1_XOR_ROW", "H2_ADD_ROW", "H3_ADD_3ROW")
MAX_SOURCE = 69614355


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


def map_bank(mode, row, base):
    low = row & 7
    if mode == "H0_IDENTITY":
        return base
    if mode == "H1_XOR_ROW":
        return base ^ low
    if mode == "H2_ADD_ROW":
        return (base + low) & 7
    return (base + 3 * low) & 7


def prove_maps():
    output = {}
    for mode in MODES:
        rows = [[map_bank(mode, row, base) for base in range(8)]
                for row in range(32)]
        require(all(sorted(mapped) == list(range(8)) for mapped in rows),
                mode + " row mapping is not bijective")
        counts = [sum(map_bank(mode, row, base) == bank
                      for row in range(32) for base in range(8))
                  for bank in range(8)]
        require(counts == [32] * 8, mode + " global bank-depth drift")
        output[mode] = {
            "all_32_rows_bijective": True,
            "positions_per_bank": counts,
            "inverse_exists_per_row": True,
        }
    return output


def parse_log():
    pattern = re.compile(
        r"^\[M53 K6_CTX16_TEMPORAL_M96_FIXED_GROUP_H0\] "
        r"([0-9]+)/40 sample=([0-9]+) operator=(\S+)$")
    rows = []
    summaries = []
    failure_signatures = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            rows.append((int(match.group(1)), int(match.group(2)), match.group(3)))
        if line.startswith("M96_FIXED_GROUP_BANK_SKEW="):
            summaries.append(json.loads(line.split("=", 1)[1]))
        lowered = line.lower()
        if any(token in lowered for token in ("traceback", "assertionerror", "valueerror")):
            failure_signatures.append(line)
    require([row[0] for row in rows] == list(range(1, 41)), "log ordinals")
    ids = [(row[1], row[2]) for row in rows]
    require(len(ids) == 40 and len(set(ids)) == 40, "log record population")
    require(all(sum(sample == expected for sample, _ in ids) == 4
                for expected in range(10)), "log 10x4 population")
    require(len(summaries) == 1, "summary marker population")
    require(not failure_signatures, "failure signature in log")
    return ids, summaries[0]


def sum_fields(rows):
    result = collections.Counter()
    for row in rows:
        result["fusion_groups"] += row["fusion_groups"]
        result["union_popcount"] += row["union_popcount"]
        result["tight_eight_bank_lower_bound_cycles"] += row[
            "tight_eight_bank_lower_bound_cycles"]
        for mode in MODES:
            result[mode] += row["source_cycles_by_mode"][mode]
    return result


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
    map_proof = prove_maps()

    require(raw["fixed_group_record_ledger"]["record_count"] == 40,
            "published record count")
    records = raw["fixed_group_record_ledger"]["records"]
    require(len(records) == 40, "raw record population")
    raw_ids = [(row["sample_id"], row["operator"]) for row in records]
    require(raw_ids == log_ids and len(set(raw_ids)) == 40,
            "raw/log record identity/order")
    require(all(set(row["source_cycles_by_mode"]) == set(MODES)
                for row in records), "record mode population")
    require(all(row["tight_eight_bank_lower_bound_cycles"] <=
                min(row["source_cycles_by_mode"].values())
                for row in records), "invalid per-record lower bound")

    baseline_records = raw["baseline_replay"]["record_ledger"]["records"]
    baseline_by_id = dict(((row["sample_id"], row["operator"]), row)
                          for row in baseline_records)
    require(len(baseline_by_id) == 40, "baseline record population")
    for row in records:
        base = baseline_by_id[(row["sample_id"], row["operator"])]
        require((row["fusion_groups"], row["union_popcount"],
                 row["source_cycles_by_mode"]["H0_IDENTITY"]) ==
                (base["fusion_groups"], base["unique_weight_issues"],
                 base["source_only_cycles"]), "record H0 identity")

    by_operator = collections.defaultdict(list)
    by_sample = collections.defaultdict(list)
    for row in records:
        by_operator[row["operator"]].append(row)
        by_sample[row["sample_id"]].append(row)
    require(sorted(by_sample) == list(range(10)) and
            all(len(by_sample[sample]) == 4 for sample in by_sample),
            "sample record population")
    require(len(by_operator) == 4 and
            all(len(rows) == 10 for rows in by_operator.values()),
            "operator record population")

    operator_recompute = {}
    fixed_modes = {}
    for operator in sorted(by_operator):
        values = sum_fields(by_operator[operator])
        selected = min(MODES, key=lambda mode: (values[mode], MODES.index(mode)))
        fixed_modes[operator] = selected
        operator_recompute[operator] = {
            "operator": operator,
            "fusion_groups": values["fusion_groups"],
            "union_popcount": values["union_popcount"],
            "tight_eight_bank_lower_bound_cycles":
                values["tight_eight_bank_lower_bound_cycles"],
            "source_cycles_by_mode": dict((mode, values[mode]) for mode in MODES),
            "selected_mode": selected,
        }
    published_operators = dict((row["operator"], row) for row in raw["per_operator"])
    require(operator_recompute == published_operators, "per-operator reduction drift")
    require(fixed_modes == raw["fixed_operator_modes"] ==
            receipt["selected_fixed_modes"], "fixed mode selection drift")

    baselines = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(baselines) == 1, "M89 K6 baseline population")
    m89_k6 = baselines[0]
    m89_samples = dict((row["sample_id"], row) for row in m89_k6["per_sample"])
    sample_recompute = []
    for sample_id in range(10):
        rows = by_sample[sample_id]
        totals = sum_fields(rows)
        selected = sum(row["source_cycles_by_mode"][fixed_modes[row["operator"]]]
                       for row in rows)
        oracle = sum(min(row["source_cycles_by_mode"].values()) for row in rows)
        rebuilt = {
            "sample_id": sample_id,
            "baseline_source_cycles": m89_samples[sample_id]["source"],
            "selected_source_cycles": selected,
            "selected_delta_cycles": selected - m89_samples[sample_id]["source"],
            "source_cycles_by_global_mode":
                dict((mode, totals[mode]) for mode in MODES),
            "prohibited_per_record_oracle_source_cycles": oracle,
        }
        sample_recompute.append(rebuilt)
    require(sample_recompute == raw["per_sample"], "per-sample reduction drift")

    totals = sum_fields(records)
    selected_source = sum(row["selected_source_cycles"] for row in sample_recompute)
    oracle = sum(row["prohibited_per_record_oracle_source_cycles"]
                 for row in sample_recompute)
    aggregate = {
        "baseline_source_cycles": totals["H0_IDENTITY"],
        "selected_source_cycles": selected_source,
        "selected_source_cycle_gain": totals["H0_IDENTITY"] - selected_source,
        "selected_source_speedup": float(totals["H0_IDENTITY"]) / selected_source,
        "tight_eight_bank_lower_bound_cycles":
            totals["tight_eight_bank_lower_bound_cycles"],
        "remaining_cycles_above_tight_lower_bound":
            selected_source - totals["tight_eight_bank_lower_bound_cycles"],
        "fusion_groups": totals["fusion_groups"],
        "unique_weight_issues": totals["union_popcount"],
        "prohibited_per_record_oracle_source_cycles": oracle,
        "prohibited_oracle_source_gain": totals["H0_IDENTITY"] - oracle,
    }
    require(aggregate == receipt["aggregate"], "receipt aggregate drift")
    require(all(aggregate[key] == raw["aggregate"][key]
                for key in aggregate if key != "selected_source_speedup") and
            raw["aggregate"]["selected_source_speedup"] ==
            {"decimal": 1.0, "denominator": 1, "numerator": 1},
            "raw aggregate drift")
    require(aggregate["baseline_source_cycles"] == m89_k6["source_cycles"] ==
            raw["baseline_replay"]["aggregate_source_only_cycles"],
            "M89 aggregate source reproduction")
    require(all(row["baseline_source_cycles"] ==
                row["source_cycles_by_global_mode"]["H0_IDENTITY"]
                for row in sample_recompute), "M89 per-sample source reproduction")

    mode_totals = dict((mode, totals[mode]) for mode in MODES)
    mode_deltas = dict((mode, totals[mode] - totals["H0_IDENTITY"])
                       for mode in MODES)
    h0_wins_every_record = all(
        row["source_cycles_by_mode"]["H0_IDENTITY"] ==
        min(row["source_cycles_by_mode"].values()) for row in records)
    h0_strictly_beats_all_three_nonidentity_in_every_record = all(
        all(row["source_cycles_by_mode"][mode] >
            row["source_cycles_by_mode"]["H0_IDENTITY"]
            for mode in MODES[1:]) for row in records)
    require(h0_wins_every_record and oracle == totals["H0_IDENTITY"],
            "per-record oracle unexpectedly improves")
    require(h0_strictly_beats_all_three_nonidentity_in_every_record,
            "nonidentity mapping does not strictly regress every record")
    require(mode_deltas["H1_XOR_ROW"] > 0 and
            mode_deltas["H2_ADD_ROW"] > 0 and
            mode_deltas["H3_ADD_3ROW"] > 0, "global nonidentity mode result")

    gates = {
        "H0_exact_group_count_equal_10436792": totals["fusion_groups"] == 10436792,
        "H0_exact_source_cycles_equal_69964176": totals["H0_IDENTITY"] == 69964176,
        "logical_source_updates_equal_562451704":
            raw["baseline_replay"]["aggregate_logical_source_updates"] == 562451704,
        "unique_weight_issues_equal_416232640": totals["union_popcount"] == 416232640,
        "weight_dma_bytes_per_sample_equal_212336640":
            raw["baseline_replay"]["traffic_bytes_per_sample"]["weight_dma"] == 212336640,
        "zero_extra_ports_capacity_and_vector_storage": True,
        "each_operator_mode_is_fixed_across_all_samples": len(fixed_modes) == 4,
        "each_sample_source_cycles_must_not_regress":
            all(row["selected_delta_cycles"] <= 0 for row in sample_recompute),
        "selected_source_cycles_le_69614355": selected_source <= MAX_SOURCE,
    }
    receipt_gates = dict((key, value) for key, value in receipt["gates"].items()
                         if key != "all_stage1_gates_pass")
    require(gates == receipt_gates == raw["stage1_gates"], "gate drift")
    require(not all(gates.values()) and not receipt["gates"]["all_stage1_gates_pass"]
            and not raw["all_stage1_gates_pass"], "stage-1 verdict drift")
    require(raw["stage2"] == {"exact_integrated_replay_required": False,
                              "executed": False,
                              "integrated_cycles": None,
                              "p95_integrated_cycles": None},
            "stage-2 fail-closed drift")
    require(receipt["stage2"]["required"] is False and
            receipt["stage2"]["executed"] is False, "receipt stage-2 drift")

    require(marker["status"] == raw["status"] and
            marker["all_stage1_gates_pass"] is False and
            marker["operator_modes"] == fixed_modes and
            marker["baseline_source"] == aggregate["baseline_source_cycles"] and
            marker["selected_source"] == selected_source and marker["gain"] == 0 and
            marker["tight_lower"] == aggregate["tight_eight_bank_lower_bound_cycles"] and
            marker["per_sample_deltas"] == [0] * 10,
            "summary marker drift")
    require(raw["status"] == receipt["status"] ==
            "PASS_M96_STAGE1_EXECUTION_NO_GO", "status drift")
    require(receipt["decision"]["promotion"] == "NO_GO", "receipt decision drift")

    probe_text = PROBE.read_text(encoding="utf-8")
    require(probe_text.count("transformed = transformed.replace(hook_from, hook_to)") == 1,
            "audit hook replacement source drift")
    require(probe_text.count("AUDIT_ONLY_NO_SCHEDULER_FEEDBACK") == 1,
            "audit-only qualification drift")
    require('hook_from = "        for task in group\\n"' not in probe_text and
            'hook_from = "        for task in group:\\n            del resident[task]"' in probe_text and
            '"        audit_frozen_group(union_mask, group_cycles)\\n"' in probe_text,
            "post-selection hook text drift")

    exact_lower_speedup = float(totals["H0_IDENTITY"]) / float(
        totals["tight_eight_bank_lower_bound_cycles"])
    output = {
        "schema": "m96_fixed_group_reversible_bank_skew_independent_recompute_v1",
        "status": "PASS_EXACT_STAGE1_NEGATIVE_SCREEN_NO_GO_STAGE2_VCS_DC",
        "producer_imported_or_executed": False,
        "sha256": dict((name, sha256(path)) for name, path in paths.items()),
        "completion_markers": 40,
        "sample_count": 10,
        "operator_count": 4,
        "record_count": 40,
        "bank_map_proof": map_proof,
        "fixed_modes": fixed_modes,
        "mode_source_cycles": mode_totals,
        "mode_delta_vs_h0": mode_deltas,
        "h0_minimum_in_all_40_records": h0_wins_every_record,
        "h0_strictly_beats_all_three_nonidentity_in_all_40_records":
            h0_strictly_beats_all_three_nonidentity_in_every_record,
        "aggregate": aggregate,
        "per_sample": sample_recompute,
        "stage1_gates": gates,
        "all_stage1_gates_pass": False,
        "stage2_required": False,
        "stage2_executed": False,
        "lower_bound_context": {
            "h0_over_sum_ceil_group_popcount_div_8_speedup_ceiling":
                exact_lower_speedup,
            "remaining_cycles": aggregate["remaining_cycles_above_tight_lower_bound"],
            "qualification": "mapping-independent work-conservation lower bound; not achieved by H0-H3 and not an achievable implementation claim",
        },
        "claim_flags": raw["claim_policy"],
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M96 independent arithmetic/provenance audit")
    print("completion_markers=40 records=40 samples=10 operators=4")
    print("mode_cycles=" + json.dumps(mode_totals, sort_keys=True))
    print("selected_source={} gain=0 speedup=1.0".format(selected_source))
    print("tight_lower={} ceiling={:.9f}x".format(
        aggregate["tight_eight_bank_lower_bound_cycles"], exact_lower_speedup))
    print("stage2=false vcs=false dc=false")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
