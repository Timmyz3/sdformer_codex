#!/usr/bin/env python3
"""Validate the canonical M43-r1 spatial parent-delta milestone."""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
RESULT = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatial_parent_delta_schedule_final.json")
ANALYZER = HW_ROOT / (
    "system_simulator/scripts/"
    "analyze_m43_tile_resident_parent_delta_schedule.py")
CONTRACT = HW_ROOT / (
    "contracts/m43_tile_resident_parent_delta_schedule_contract_r1_20260823.json")
TEST = HW_ROOT / (
    "system_simulator/tests/"
    "test_m43_tile_resident_parent_delta_schedule.py")
SPEC = HW_ROOT / "rtl_m43/M43_TILE_RESIDENT_PARENT_DELTA_R1.md"
EXPECTED_SHA256 = {
    "result": "70c52dfc8ef1b223391a1c0699f6ada8ff999d2079370bcd9d3917c198a1c329",
    "analyzer": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "contract": "c894b5fcdd6a6cd7d33bf736e8c084630f0ea297f632e1dd6a35889714772e44",
    "test": "c189935f3365beaa657eaa21ca7c40f275523a974ad12611e11f6f84331f197f",
    "spec": "5c2b53b7eb0ec4ca19559e65c0d454109a598c80d1647564b9f894e573dd13a6",
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


def load_analyzer():
    spec = importlib.util.spec_from_file_location("m43_validator_analyzer", ANALYZER)
    require(spec is not None and spec.loader is not None,
            "M43 analyzer import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(path=RESULT):
    canonical = Path(path).resolve() == RESULT.resolve()
    if canonical:
        require(sha256(path) == EXPECTED_SHA256["result"],
                "M43 canonical result identity drift")
    for name, item in (("analyzer", ANALYZER), ("contract", CONTRACT),
                       ("test", TEST), ("spec", SPEC)):
        require(item.is_file() and sha256(item) == EXPECTED_SHA256[name],
                "M43 {} identity drift".format(name))
    result = read_json(path)
    require(result["schema"] == "m43_tile_resident_parent_delta_schedule_v1",
            "M43 result schema drift")
    require(result["status"] == (
        "PASS_M43A_EXACT_FINITE_SOURCE_BANK_SCHEDULE_AND_CAPACITY_GATES_"
        "BUT_MULTICONTEXT_RTL_MEMORY_TIMING_AND_SYSTEM_SPEEDUP_UNADMITTED"),
        "M43 result status drift")
    identity = result["identity"]
    require(identity["contract_sha256"] == EXPECTED_SHA256["contract"] and
            identity["analyzer_sha256"] == EXPECTED_SHA256["analyzer"],
            "M43 result producer identity drift")
    architecture = result["architecture"]
    require(architecture["name"] ==
            "TILE_RESIDENT_SIGNED_SPATIAL_PARENT_DELTA" and
            architecture["temporal_parent_enabled"] is False and
            architecture["parents"] == ["local_zero", "left", "up"] and
            architecture["issue_width"] == 8 and
            architecture["output_lanes"] == 96 and
            architecture["peak_product_adds_per_cycle"] == 768,
            "M43 primary architecture drift")

    rows = result["records"]
    require(len(rows) == 40 and result["population"]["output_rows"] == 120000
            and result["population"]["tile_rows"] == 3240000,
            "M43 population drift")
    local_pairs = sum(row["local_source_destination_pairs"] for row in rows)
    delta_pairs = sum(row["parent_delta_source_destination_pairs"] for row in rows)
    local_cycles = sum(row["local_p8_l96_source_issue_cycles"] for row in rows)
    delta_cycles = sum(row["parent_delta_p8_l96_source_issue_cycles"]
                       for row in rows)
    aggregate = result["aggregate"]
    require(local_pairs == aggregate["local_source_destination_pairs"] == 92640472,
            "M43 Local pair reconciliation drift")
    require(delta_pairs == aggregate[
        "parent_delta_source_destination_pairs"] == 72716857,
        "M43 delta pair reconciliation drift")
    require(local_cycles == aggregate[
        "local_p8_l96_source_issue_cycles"] == 141484880,
        "M43 Local finite-bank cycle drift")
    require(delta_cycles == aggregate[
        "parent_delta_p8_l96_source_issue_cycles"] == 116376872,
        "M43 delta finite-bank cycle drift")
    require(aggregate["logical_pair_reduction"] ==
            {"numerator": 19923615, "denominator": 92640472} and
            aggregate["finite_bank_issue_cycle_reduction"] ==
            {"numerator": 25108008, "denominator": 141484880},
            "M43 reduction fraction drift")
    require(sum(aggregate["parent_choice_by_tile"].values()) == 3240000 and
            aggregate["parent_choice_by_tile"]["previous_timestep"] == 0,
            "M43 parent population drift")

    samples = result["per_sample"]
    require(len(samples) == 10 and
            sum(item["delta_issue_cycles"] for item in samples) == delta_cycles,
            "M43 sample issue-cycle conservation drift")
    require(all(item["source_issue_is_capacity_max"] is True and
                item["independent_service_capacity_max"] ==
                item["delta_issue_cycles"] and
                item["three_x_crossing_admitted"] is False
                for item in samples), "M43 independent capacity gate drift")
    gate = result["three_x_headroom_gate"]
    require(gate["m42_maximum_product_cycles"] == 15495075 and
            gate["p95_parent_delta_source_issue_cycles"] == 11883808 and
            gate["p95_visible_product_overhead_headroom_cycles"] == 3611267 and
            gate["all_samples_source_issue_below_gate"] is True and
            gate["all_samples_independent_service_capacity_below_source_issue"]
            is True and gate["target_crossing_admitted"] is False,
            "M43 3x headroom boundary drift")
    layout = result["physical_layout_bridge"]
    require(layout["local_scratch_bytes"] == 56960 and
            layout["frozen_local_residency_bytes"] == 193728 and
            layout["local_scratch_fits"] is True and
            layout["one_output_block_all_timestep_final_accumulator_bytes"] ==
            864000, "M43 storage capacity drift")
    forbidden = result["claim_policy"]["forbidden"]
    require(any("3x crossing" in item for item in forbidden) and
            any("full-network" in item for item in forbidden),
            "M43 fail-closed claim policy drift")
    return result


def rerun_and_compare():
    with tempfile.TemporaryDirectory() as tempdir:
        rebuilt = Path(tempdir) / "m43_rebuilt.json"
        subprocess.check_call(["/usr/bin/python3.6", str(ANALYZER),
                               "--output", str(rebuilt)])
        require(rebuilt.read_bytes() == RESULT.read_bytes(),
                "M43 full rebuild is not byte-identical")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    validate(args.result)
    if args.rerun:
        rerun_and_compare()
    print("PASS M43-r1 canonical validation")


if __name__ == "__main__":
    main()
