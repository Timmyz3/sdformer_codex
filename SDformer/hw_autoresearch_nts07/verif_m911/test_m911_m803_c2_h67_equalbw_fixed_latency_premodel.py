#!/usr/bin/env python3
"""Static and adversarial checks for the additive M911 result."""

import argparse
import copy
import json
from pathlib import Path


EXPECTED_CYCLES = {"1": 504300928, "2": 508016984, "4": 515449096}


def check(data):
    assert data["status"] == \
        "PASS_H67_EQUAL_BANDWIDTH_FIXED_LATENCY_COMPONENT_PREMODEL"
    assert data["population"]["records"] == 120
    assert data["population"]["tokens"] == 5580000
    assert data["population"]["events"] == 143894510
    assert data["population"]["six_slice_weight_requests"] == 440284872
    assert set(data["latency_points"]) == {"1", "2", "4"}
    for latency, expected in EXPECTED_CYCLES.items():
        row = data["latency_points"][latency]
        assert row["latency_cycles"] == int(latency)
        assert row["k8_memory_service_cycles"] == expected
        assert row["k1x8_memory_service_cycles"] == expected
        assert row["equal_bandwidth_cycle_speedup_k8_vs_k1x8"]["float"] == 1.0
        assert row["controller_delta_modeled"] is False
    assert data["latency_points"]["4"][
        "k8_throughput_retention_vs_l1"]["float"] < 1.0
    assert data["fair_resource_boundary"]["physical_weight_banks_each"] == 8
    assert abs(data["fair_resource_boundary"][
        "k8_logic_area_saving_vs_k1x8_percent"] - 77.61043405612304) < 1e-12
    assert abs(data["latency_points"]["1"][
        "equal_bandwidth_throughput_per_logic_area_ratio_k8_vs_k1x8"]
        - 4.466366174791688) < 1e-15
    assert data["directed_vcs_crosscheck_separate_scope"][
        "not_extrapolated_to_120_record_points"] is True
    audit = data["direct_full_trace_replay_audit"]
    assert audit["available"] is False
    assert audit["raw_h67_payload_available"] is True
    assert len(audit["exact_blockers"]) == 3
    bounds = data["claim_boundary"]
    for key in ("direct_cycle_accurate_m803_full_trace", "complete_fc2",
                "complete_ffn", "physical_sram_macro", "power", "energy",
                "ppa", "system_speedup", "headline"):
        assert bounds[key] is False


def expect_failure(data, mutate):
    attacked = copy.deepcopy(data)
    mutate(attacked)
    failed = False
    try:
        check(attacked)
    except (AssertionError, KeyError, TypeError, ValueError):
        failed = True
    assert failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    check(data)
    attacks = [
        lambda d: d["latency_points"].pop("2"),
        lambda d: d["latency_points"]["4"].__setitem__(
            "k1x8_memory_service_cycles", 515449095),
        lambda d: d["fair_resource_boundary"].__setitem__(
            "physical_weight_banks_each", 1),
        lambda d: d["direct_full_trace_replay_audit"].__setitem__(
            "available", True),
        lambda d: d["claim_boundary"].__setitem__("system_speedup", True),
        lambda d: d["population"].__setitem__("records", 119),
    ]
    for attack in attacks:
        expect_failure(data, attack)
    print("PASS M911 static checks=1 attacks={} mismatches=0".format(
        len(attacks)))


if __name__ == "__main__":
    main()
