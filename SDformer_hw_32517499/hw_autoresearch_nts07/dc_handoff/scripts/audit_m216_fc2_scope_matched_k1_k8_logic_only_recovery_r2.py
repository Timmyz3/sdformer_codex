#!/usr/bin/env python3
"""Fail-closed recovery audit for the two completed M216 DC subruns."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED = {
    "contracts/m216_fc2_scope_matched_k1_k8_logic_only_dc_recovery_contract_r2_20260825.json": "08e3cef798f3b78e0a085ce2854c47a22ea1bd8146ef1d6f6055e1a4ad947e6e",
    "dc_handoff/scripts/run_dc_m216_fc2_scope_matched_k1_k8_logic_only.sh": "13e9d3960bc74e969a5681a3072a9e9e9074c6838294e1f1572c3eed750cc704",
    "dc_handoff/scripts/run_dc_m216_flattened_source_cap_logic_only.tcl": "2565e750551f6f2a03abff96b462558416f3b9531a693f43b6194af8026d61d5",
    "dc_handoff/filelists/date_m216_fc2_source_cap_rtl.f": "3380352827a201a750a8bdecad1e09d269479d2fdb691d23c84b6a09b7110e48",
    "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc": "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv": "8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv": "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

RUN_REL = Path(
    "dc_handoff/runs/"
    "m216_fc2_scope_matched_k1_k8_logic_only_dc_3p000ns_r1_sealed_20260825"
)
EXPECTED_RUN = {
    "RUN_FAILED_OR_INCOMPLETE.txt": "8afa55c2ab45edbfb33d1848560bd4e61f2b03cb44cc8ea2788adae9bf07a06a",
    "k1/RUN_COMPLETE.txt": "519dc606788c5b55d183c63a9b08d3d8d08b2b7a74a95ab4311e7134ba524c75",
    "k8/RUN_COMPLETE.txt": "9c213f608d2d4fd5453b075abea33fa2fe19694724a7edfa6baa63f1c8a15e15",
    "k1/evidence_manifest.sha256": "78f6660da206d4865685cee2ef68d7d3072c13c5e6da9bff355487514148dad8",
    "k8/evidence_manifest.sha256": "9f834c2fb9790a4919fc943a5dce75517512b4a4a77a2c302cda0b51b58699d6",
    "k1/netlist/m216_fc2_raw4_to_source_cap_frontend_mapped.v": "bb92c26402251fb9df6f8c7b0463c321397ea081ad94074cc287368e3b0a9778",
    "k8/netlist/m216_fc2_raw4_to_source_cap_frontend_mapped.v": "f0dfde0c69ecca2607c040664a7f8947d2e75f724634c1927b0929a8e7867e52",
}
EXPECTED_ONLY_K8_REGISTERS = {
    "paired_sink_group_source_count_q_reg_1_",
    "paired_sink_group_source_count_q_reg_2_",
    "paired_sink_group_source_count_q_reg_3_",
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_receipt(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def register_instances(path):
    instances = set()
    pattern = re.compile(r"^\s*\S+\s+(\S*_reg\S*)\s+\(")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            instances.add(match.group(1))
    return instances


def numeric(values, key, kind=float):
    require(key in values, "missing receipt field: {}".format(key))
    return kind(values[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")

    root = Path(__file__).resolve().parents[2]
    script_start = digest(Path(__file__).resolve())
    for relative, expected in EXPECTED.items():
        require(digest(root / relative) == expected,
                "input drift: {}".format(relative))
    run = root / RUN_REL
    for relative, expected in EXPECTED_RUN.items():
        require(digest(run / relative) == expected,
                "sealed DC evidence drift: {}".format(relative))

    parent = parse_receipt(run / "RUN_FAILED_OR_INCOMPLETE.txt")
    require(parent == {
        "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
        "runner_exit_code": "40",
    }, "r1 parent failure receipt changed")

    for cap in ("k1", "k8"):
        subprocess.run(
            ["sha256sum", "-c", "evidence_manifest.sha256", "--quiet"],
            cwd=run / cap, check=True)

    k1 = parse_receipt(run / "k1/RUN_COMPLETE.txt")
    k8 = parse_receipt(run / "k8/RUN_COMPLETE.txt")
    require(k1["status"] == "PASS_M216_K1_LOGIC_ONLY_DC_3NS",
            "K1 DC did not pass")
    require(k8["status"] == "PASS_M216_K8_LOGIC_ONLY_DC_3NS",
            "K8 DC did not pass")
    require(k1["elaboration_parameter"] == "SOURCE_CAP=1"
            and k8["elaboration_parameter"] == "SOURCE_CAP=8",
            "elaboration parameter mismatch")
    for receipt in (k1, k8):
        require(receipt["tool"] == "Synopsys_DC_V-2023.12-SP3",
                "tool mismatch")
        require(receipt["clock_period_ns"] == "3.000"
                and receipt["clock_network"] == "ideal"
                and receipt["wireload"] == "ZeroWireload",
                "flow mismatch")
        require(numeric(receipt, "cell_area_um2") < 22000,
                "area threshold failed")
        require(numeric(receipt, "cell_count", int) < 33000,
                "cell-count threshold failed")
        require(numeric(receipt, "sequential_cells", int) < 2900,
                "sequential-cell threshold failed")
        require(numeric(receipt, "logic_levels") <= 100,
                "logic-level threshold failed")
        require(numeric(receipt, "setup_worst_slack_ns") >= 0.0,
                "setup threshold failed")
        require(numeric(receipt, "hold_worst_slack_ns") >= 0.0,
                "hold threshold failed")
        require(receipt["macro_count"] == "0", "macro count changed")

    k1_regs = register_instances(
        run / "k1/netlist/m216_fc2_raw4_to_source_cap_frontend_mapped.v")
    k8_regs = register_instances(
        run / "k8/netlist/m216_fc2_raw4_to_source_cap_frontend_mapped.v")
    require(len(k1_regs) == numeric(k1, "sequential_cells", int),
            "K1 receipt/netlist sequential count mismatch")
    require(len(k8_regs) == numeric(k8, "sequential_cells", int),
            "K8 receipt/netlist sequential count mismatch")
    require(k1_regs - k8_regs == set(),
            "unexpected K1-only sequential instances")
    require(k8_regs - k1_regs == EXPECTED_ONLY_K8_REGISTERS,
            "sequential delta is not exactly the three folded count bits")

    k1_area = numeric(k1, "cell_area_um2")
    k8_area = numeric(k8, "cell_area_um2")
    result = {
        "schema": "m216_fc2_scope_matched_k1_k8_logic_only_dc_recovery_audit_v2",
        "status": "PASS_EXISTING_EXACT_SHA_MATCHED_DC_SUBRUN_RECOVERY",
        "identity": {
            "audit_script_start_sha256": script_start,
            "contract_sha256": EXPECTED[
                "contracts/m216_fc2_scope_matched_k1_k8_logic_only_dc_recovery_contract_r2_20260825.json"],
            "docs359_sha256": EXPECTED[
                "docs/359_DATE终局冻结_20260813.md"],
            "source_run": str(RUN_REL),
            "source_parent_failure_receipt_sha256": EXPECTED_RUN[
                "RUN_FAILED_OR_INCOMPLETE.txt"],
            "k1_evidence_manifest_sha256": EXPECTED_RUN[
                "k1/evidence_manifest.sha256"],
            "k8_evidence_manifest_sha256": EXPECTED_RUN[
                "k8/evidence_manifest.sha256"],
        },
        "parent_failure_preserved": {
            "status": parent["status"],
            "runner_exit_code": int(parent["runner_exit_code"]),
            "recovery_is_new_dc_run": False,
        },
        "k1": {
            "source_cap": 1,
            "cell_area_um2": k1_area,
            "cell_count": numeric(k1, "cell_count", int),
            "sequential_cells": len(k1_regs),
            "logic_levels": numeric(k1, "logic_levels"),
            "critical_path_length_ns": numeric(
                k1, "critical_path_length_ns"),
            "setup_worst_slack_ns": numeric(
                k1, "setup_worst_slack_ns"),
            "hold_worst_slack_ns": numeric(
                k1, "hold_worst_slack_ns"),
        },
        "k8": {
            "source_cap": 8,
            "cell_area_um2": k8_area,
            "cell_count": numeric(k8, "cell_count", int),
            "sequential_cells": len(k8_regs),
            "logic_levels": numeric(k8, "logic_levels"),
            "critical_path_length_ns": numeric(
                k8, "critical_path_length_ns"),
            "setup_worst_slack_ns": numeric(
                k8, "setup_worst_slack_ns"),
            "hold_worst_slack_ns": numeric(
                k8, "hold_worst_slack_ns"),
        },
        "matched_storage_audit": {
            "k1_register_instances": len(k1_regs),
            "k8_register_instances": len(k8_regs),
            "common_register_instances": len(k1_regs & k8_regs),
            "k1_only_register_instances": [],
            "k8_only_register_instances": sorted(k8_regs - k1_regs),
            "only_delta_is_constant_folded_source_count_encoding": True,
            "descriptor_queue_and_dual_d8_window_register_names_identical": True,
        },
        "area": {
            "k8_over_k1_ratio": k8_area / k1_area,
            "k8_area_overhead_um2": k8_area - k1_area,
            "k8_area_overhead_percent": (k8_area / k1_area - 1.0) * 100.0,
        },
        "claim_boundary": {
            "logic_only_pre_macro_area_timing_ablation": True,
            "complete_fc2": False,
            "complete_ffn": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    require(digest(Path(__file__).resolve()) == script_start,
            "audit script mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["area"], sort_keys=True))


if __name__ == "__main__":
    main()
