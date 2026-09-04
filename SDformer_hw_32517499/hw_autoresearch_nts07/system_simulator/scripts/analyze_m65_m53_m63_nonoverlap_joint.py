#!/usr/bin/env python3
"""Build the fail-closed M65 non-overlap joint conditional ledger."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_payload(
    *, numerator: int, outside: int, late: int, pair: int, replacement: int,
    inherited: int,
) -> Any:
    denominator = outside - inherited + late + pair + replacement
    return {
        "inherited_linear_cycles": inherited,
        "joint_conditional_cycles": denominator,
        "conditional_ratio_not_system_speedup": numerator / denominator,
        "replacement_minus_inherited_cycles": replacement - inherited,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    hw_root = contract_path.parent.parent
    contract = load_json(contract_path)
    require(contract["schema"] == "m65_m53_m63_nonoverlap_joint_contract_v1",
            "contract schema drift")

    payloads = {}  # type: dict[str, Any]
    identities = {}  # type: dict[str, Any]
    for item in contract["inputs"]:
        path = hw_root / item["path"]
        actual = sha256(path)
        require(actual == item["sha256"], "input SHA drift: " + item["path"])
        payloads[path.name] = load_json(path)
        identities[item["path"]] = actual

    m25 = payloads["m25_resource_bounded_tiled_cycles.json"]
    m39 = payloads["m39_remaining_bottleneck.json"]
    m53 = payloads["m53_adaptive_temporal_parent_k4_ctx16_dse.json"]
    m63 = payloads["m63_linear_k4_spatiotemporal_full_network_opportunity_result_r2.json"]
    frozen = contract["frozen_reconciliation"]

    local = m25["compute_envelopes"]["local"]["10"]
    require(local["m4_profiled_eligible_cycles"] ==
            frozen["m25_local_m4_profiled_eligible_cycles"], "M25 raw eligible drift")
    require(local["effective_m4_speed"] == frozen["m25_local_effective_m4_speed"],
            "M25 effective speed drift")
    require(local["accelerated_m4_cycles"] ==
            frozen["m25_local_accelerated_m4_cycles"], "M25 accelerated cycles drift")
    require(math.ceil(local["m4_profiled_eligible_cycles"] /
                      local["effective_m4_speed"]) == local["accelerated_m4_cycles"],
            "M25 global ceil rule does not reconstruct")

    dse_local = [row for row in m39["conditional_dse"]["four_bottleneck_rows"]
                 if row["line"] == "Local"]
    require(len(dse_local) == 2, "M39 Local DSE population drift")
    require(all(row["m38_model_substituted_ideal_before_scope_cycles"] ==
                frozen["m39_local_conditional_ideal_cycles"] for row in dse_local),
            "M39 Local ideal drift")
    require(all(row["before_cycles"] == frozen["four_bottleneck_before_cycles"]
                for row in dse_local), "M39 bottleneck population drift")
    outside = (frozen["m39_local_conditional_ideal_cycles"] -
               frozen["four_bottleneck_before_cycles"])
    require(outside == frozen["m53_outside_four_bottleneck_model_cycles"],
            "M39 outside term does not reconstruct")

    m53_model = m53["conditional_frozen_compute_model"]
    numerator = frozen["fixed_compute_reference_cycles"]
    late = frozen["m53_fixed_late_scale_plus_frontend_cycles"]
    pair = frozen["m53_pair_p95_cycles"]
    m53_denominator = outside + late + pair
    require(m53_model["fixed_compute_reference_cycles"] == numerator,
            "M53 numerator drift")
    require(m53_model["pair_p95_nearest_rank_cycles"] == pair,
            "M53 pair p95 drift")
    require(m53_denominator == frozen["m53_conditional_denominator_cycles"] ==
            m53_model["conditional_total_cycles"], "M53 denominator reconstruction failed")

    categories = m63["m39_category_ledger"]
    captured_raw = sum(int(row["captured_m39_activity_cycles"])
                       for row in categories.values())
    captured_modules = sum(int(row["captured_modules"]) for row in categories.values())
    require(captured_raw == frozen["m63_captured_linear_raw_m4_eligible_cycles"],
            "M63 captured raw cycles drift")
    require(captured_modules == frozen["m63_captured_linear_modules"] ==
            m63["population"]["target_modules"], "M63 module population drift")
    require(all(row["captured_m39_activity_cycles"] ==
                row["dual_line_category_eligible_cycles"] for row in categories.values()),
            "M63 captured categories are not wholly M25 eligible")
    require(captured_raw < local["m4_profiled_eligible_cycles"],
            "captured partition must be a strict subset")

    spatial = m63["aggregate_configurations"]["spatial_K4"]
    temporal = m63["aggregate_configurations"]["temporal_K4"]
    replacement = spatial["serialized_integrated_cycle_distribution"]["p95_nearest_rank"]
    require(replacement == frozen["m63_spatial_k4_replacement_p95_cycles"],
            "M63 spatial K4 replacement drift")
    require(spatial["capacity_feasible_modules"] ==
            frozen["m63_spatial_k4_capacity_feasible_modules"] == captured_modules,
            "M63 spatial capacity drift")
    require(temporal["capacity_infeasible_modules"] ==
            frozen["m63_temporal_k4_capacity_infeasible_modules"],
            "M63 temporal capacity kill drift")

    ideal_inherited = captured_raw / local["effective_m4_speed"]
    inherited_floor = math.floor(ideal_inherited)
    inherited_ceil = math.ceil(ideal_inherited)
    require(inherited_ceil - inherited_floor == 1,
            "expected one-cycle inherited rounding interval")
    complement_raw = local["m4_profiled_eligible_cycles"] - captured_raw
    require(captured_raw + complement_raw == local["m4_profiled_eligible_cycles"],
            "raw partition conservation failed")

    endpoints = [
        endpoint_payload(numerator=numerator, outside=outside, late=late, pair=pair,
                         replacement=replacement, inherited=value)
        for value in (inherited_floor, inherited_ceil)
    ]
    denominators = [row["joint_conditional_cycles"] for row in endpoints]
    ratios = [row["conditional_ratio_not_system_speedup"] for row in endpoints]
    deltas = [row["replacement_minus_inherited_cycles"] for row in endpoints]
    require(min(deltas) > 0, "M63 replacement unexpectedly beats inherited M25 interval")

    base_without_replacement = outside + late + pair
    target_replacement_limits = {}
    for label, target in (("preserve_m53", numerator / m53_denominator),
                          ("3p1", 3.1), ("3p2", 3.2), ("3p3", 3.3),
                          ("3p45", 3.45)):
        # Use the most generous inherited endpoint; still a conditional target.
        maximum = numerator / target - (base_without_replacement - inherited_ceil)
        target_replacement_limits[label] = {
            "target_ratio": target,
            "maximum_spatial_k4_replacement_cycles_not_system": maximum,
            "current_replacement_cycles": replacement,
            "additional_reduction_required_cycles": max(0.0, replacement - maximum),
        }

    result = {
        "schema": "m65_m53_m63_nonoverlap_joint_result_v1",
        "status": "PASS_ONE_CYCLE_TIGHT_NONOVERLAP_SPATIAL_K4_NO_GO",
        "identity": {
            "contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": identities,
        },
        "reconstruction": {
            "fixed_compute_reference_cycles": numerator,
            "m25_local_raw_eligible_cycles": local["m4_profiled_eligible_cycles"],
            "m25_local_effective_speed": local["effective_m4_speed"],
            "m25_local_accelerated_cycles_global_ceil": local["accelerated_m4_cycles"],
            "m39_local_ideal_cycles": frozen["m39_local_conditional_ideal_cycles"],
            "four_bottleneck_before_cycles": frozen["four_bottleneck_before_cycles"],
            "outside_four_bottleneck_cycles": outside,
            "fixed_late_scale_plus_frontend_cycles": late,
            "m53_pair_p95_cycles": pair,
            "m53_conditional_cycles": m53_denominator,
            "m53_conditional_ratio_not_system_speedup": numerator / m53_denominator,
        },
        "captured_linear_decomposition": {
            "modules": captured_modules,
            "raw_eligible_cycles": captured_raw,
            "complement_raw_eligible_cycles": complement_raw,
            "raw_partition_conserved": True,
            "ideal_inherited_cycles_before_global_ceil": ideal_inherited,
            "inherited_integral_interval": {
                "minimum": inherited_floor,
                "maximum": inherited_ceil,
                "width_cycles": inherited_ceil - inherited_floor,
            },
            "why_interval_not_point": (
                "M25 stored one ceil after aggregate effective-speed scaling and no "
                "per-operator accelerated-cycle allocation; floor/ceil is the tight "
                "integral interval without inventing a rounding-token owner."
            ),
        },
        "spatial_k4_nonoverlap_joint": {
            "replacement_p95_cycles": replacement,
            "endpoints": endpoints,
            "joint_conditional_cycle_interval": {
                "minimum": min(denominators), "maximum": max(denominators)},
            "conditional_ratio_interval_not_system_speedup": {
                "minimum": min(ratios), "maximum": max(ratios)},
            "replacement_regression_cycles_interval": {
                "minimum": min(deltas), "maximum": max(deltas)},
            "ratio_loss_vs_m53_interval": {
                "minimum": numerator / m53_denominator - max(ratios),
                "maximum": numerator / m53_denominator - min(ratios),
            },
            "decision": "NO_GO_AS_ADDITIVE_M53_ACCELERATOR",
            "reason": "spatial K4 p95 is slower than the inherited M25 contribution at both rounding endpoints",
        },
        "conditional_target_gates": target_replacement_limits,
        "temporal_k4": {
            "capacity_infeasible_modules": temporal["capacity_infeasible_modules"],
            "all24_joint_admitted": False,
            "decision": "KILLED_BY_11_OF_24_LOCAL_CAPACITY_FAILURES",
        },
        "claim_boundary": contract["claim_policy"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M65 one-cycle-tight nonoverlap; spatial K4 is a NO-GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
