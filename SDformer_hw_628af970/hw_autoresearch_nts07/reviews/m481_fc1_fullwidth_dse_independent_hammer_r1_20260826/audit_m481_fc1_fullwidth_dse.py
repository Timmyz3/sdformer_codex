#!/usr/bin/env python3
"""Independent arithmetic and CSV parity audit for the sealed M481 DSE."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


EXPECTED_RESULT = "2a7a1c917cb2f9aa1adb61092c7619de8d9b495aab5550f1fa41291188006578"
EXPECTED_CSV = "e4acb794cf3849a2c9fe4eaa4f5b4d4b39cf9e67caad7750ad9392b74fa54973"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right, tolerance=1e-12):
    return abs(float(left) - float(right)) <= tolerance * max(
        1.0, abs(float(left)), abs(float(right)))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    require(sha256(args.result) == EXPECTED_RESULT, "result identity drift")
    require(sha256(args.csv) == EXPECTED_CSV, "CSV identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359, "docs359 identity drift")
    data = json.loads(args.result.read_text())
    require(data["status"] ==
            "PASS_EXACT_MASK_108_POINT_CPU_DSE_NO_PERFORMANCE_ADMISSION",
            "status drift")
    require(all(data["reconciliations"].values()), "producer reconciliation drift")
    require(data["decision"]["m230_m262_ratios_multiplied"] is False,
            "forbidden ratio multiplication")
    require(data["decision"]["rtl_promotion"] is False, "unexpected RTL promotion")
    require(data["admission"]["system_speedup"] is False and
            data["admission"]["headline"] is False,
            "unexpected performance admission")

    expected_axes = set(itertools.product((8, 16, 32, 96), (1, 2, 4),
                                          (16, 32, 64), (1, 2, 4)))
    observed_axes = set()
    recomputed_gate_count = 0
    checks = 0
    points_by_id = {}
    for point in data["points"]:
        resource = point["resource"]
        axis = (int(resource["lanes"]),
                int(resource["held_context_fanout"]),
                int(resource["source_chunk"]),
                int(resource["accumulator_banks"]))
        observed_axes.add(axis)
        points_by_id[point["point_id"]] = point
        baseline = point["baseline"]
        candidate = point["candidate"]
        require(sum(int(value) for value in
                    baseline["cycle_components"].values()) ==
                int(baseline["lifecycle_cycles"]), "baseline cycle sum drift")
        require(sum(int(value) for value in
                    candidate["cycle_components"].values()) ==
                int(candidate["lifecycle_cycles"]), "candidate cycle sum drift")
        speed = (float(baseline["lifecycle_cycles"]) /
                 float(candidate["lifecycle_cycles"]))
        require(close(speed, point["same_resource_speedup"]), "speedup drift")
        weight = (float(baseline["weight_requests"]) /
                  float(candidate["weight_requests"]))
        require(close(weight, point["weight_request_reduction"]),
                "weight ratio drift")
        require(candidate["bank_conflict_extra_issue_rounds"] >= 0,
                "negative bank conflict")
        require(candidate["accumulator_update_issue_rounds"] >=
                candidate["ideal_no_bank_conflict_issue_rounds"],
                "banked rounds below ideal")
        require(resource["resource_identical_between_modes"] is True,
                "same-resource flag drift")
        require(resource["factor_request_ports"] == 1 and
                resource["weight_request_ports"] == 1 and
                resource["factor_response_latency_cycles"] == 2 and
                resource["weight_response_latency_cycles"] == 2,
                "port/latency drift")
        require(baseline["lane_slices_per_96lane_block"] == 96 // axis[0] and
                candidate["lane_slices_per_96lane_block"] == 96 // axis[0],
                "lane-slice drift")
        projection = point["scope_corrected_projection"]
        eligible = float(projection["eligible_binary_fc1_baseline_cycles"])
        fallback = float(projection["stage3_fallback_cycles_unchanged"])
        envelope = float(projection["compute_envelope_cycles"])
        projected = eligible / speed
        ideal_envelope = envelope / (envelope - eligible + projected)
        require(close(projected,
                      projection["eligible_binary_fc1_projected_cycles"]),
                "eligible projection drift")
        require(close(projected + fallback,
                      projection["all_fc1_projected_cycles"]),
                "stage3 fallback projection drift")
        require(close(ideal_envelope,
                      projection["ideal_envelope_sensitivity_not_speedup"]),
                "envelope arithmetic drift")
        gate = (speed >= 1.5 and ideal_envelope >= 1.08 and
                candidate["weight_requests"] <= baseline["weight_requests"])
        require(gate == point["gate"]["numerical_opportunity_gate_pass"],
                "gate drift")
        recomputed_gate_count += int(gate)
        require(point["admission"]["system_speedup"] is False and
                point["admission"]["headline"] is False,
                "point claim drift")
        checks += 18
    require(observed_axes == expected_axes, "108-point Cartesian coverage drift")
    require(recomputed_gate_count == 48, "gate count drift")

    fullwidth = [row for row in data["points"]
                 if row["resource"]["lanes"] == 96]
    best = max(fullwidth, key=lambda row: row["same_resource_speedup"])
    require(best["point_id"] == "L96_F4_C64_B4", "best point drift")
    require(data["decision"]["best_fullwidth_same_resource_speedup_point"] ==
            best["point_id"], "best decision drift")
    compact = points_by_id["L96_F2_C16_B2"]
    require(compact["gate"]["numerical_opportunity_gate_pass"] is True,
            "compact gate point drift")
    require(data["decision"]["compact_fullwidth_gate_point"] ==
            compact["point_id"], "compact decision drift")

    with args.csv.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 108, "CSV row count drift")
    for row in csv_rows:
        point = points_by_id[row["point_id"]]
        require(close(row["same_resource_speedup"],
                      point["same_resource_speedup"]), "CSV speed drift")
        require(int(row["baseline_cycles"]) ==
                point["baseline"]["lifecycle_cycles"], "CSV baseline drift")
        require(int(row["candidate_cycles"]) ==
                point["candidate"]["lifecycle_cycles"], "CSV candidate drift")
        require(row["system_speedup"] == "False" and
                row["headline"] == "False", "CSV claim drift")
        checks += 4

    output = {
        "schema": "m481_fc1_fullwidth_dse_independent_recompute_v1",
        "status": "PASS_108_POINT_ARITHMETIC_CSV_AND_CLAIM_RECOMPUTE",
        "checks": checks,
        "mismatches": 0,
        "gate_points": recomputed_gate_count,
        "compact_fullwidth_point": {
            "point_id": compact["point_id"],
            "same_resource_speedup": compact["same_resource_speedup"],
            "ideal_envelope_sensitivity_not_speedup": compact[
                "scope_corrected_projection"][
                    "ideal_envelope_sensitivity_not_speedup"],
            "weight_request_reduction": compact["weight_request_reduction"],
        },
        "maximum_fullwidth_point": {
            "point_id": best["point_id"],
            "same_resource_speedup": best["same_resource_speedup"],
            "ideal_envelope_sensitivity_not_speedup": best[
                "scope_corrected_projection"][
                    "ideal_envelope_sensitivity_not_speedup"],
        },
        "admission": {
            "performance": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("PASS M481 independent checks={} mismatches=0".format(checks))


if __name__ == "__main__":
    main()
