#!/usr/bin/python3.12
"""Fail-closed parser for one raw M2141/M2147 ICC2 library preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"M2141_CHECK_FAIL: {message}")


def parse_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            need(key not in result, f"duplicate key {key} in {path}")
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    need(work.is_dir() and not work.is_symlink(), "work is not a real directory")
    log = work / "icc2_preflight.log"
    rc_file = work / "icc2_preflight.rc"
    facts_file = work / "isolated_cwd" / "reports" / "machine_facts.txt"
    coverage = work / "isolated_cwd" / "reports" / "master_coverage.tsv"
    frame_dir = work / "isolated_cwd" / "frame_output"
    design_lib = work / "isolated_cwd" / "m2141_disposable_design.nlib"
    process_tree = work / "process_tree.json"
    for path in (log, rc_file, facts_file, coverage, frame_dir, design_lib, process_tree):
        need(path.exists() and not path.is_symlink(), f"missing or symlink output {path}")
    need(rc_file.read_text().strip() == "0", "ICC2 return code is nonzero")
    text = log.read_text(errors="replace")
    terminal = "RAW_PASS_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER"
    need(len(re.findall(rf"^{terminal}$", text, re.MULTILINE)) == 1,
         "missing or repeated anchored terminal token")
    for gate in range(1, 6):
        need(len(re.findall(rf"^M2141_GATE{gate}_[^\n]+$", text, re.MULTILINE)) == 1,
             f"anchored gate {gate} count is not one")
    need(not re.search(r"^M2141_FATAL_FAIL_CLOSED:", text, re.MULTILINE),
         "anchored runtime fail-closed token present")
    for token in ("CMD-104", "LIB-117", "FILE-001", "LIB-027"):
        need(token not in text, f"runtime failure diagnostic present: {token}")
    facts = parse_kv(facts_file)
    expected = {
        "status": "RAW_PASS_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148",
        "conversion_status": "1",
        "mapped_master_union_count": "94",
        "tt_master_coverage": "94",
        "ss_master_coverage": "94",
        "ff_master_coverage": "94",
        "physical_master_coverage": "94",
        "routing_layers": "M1,M2,M3,M4,M5,M6,M7,M8,M9",
        "via_layers": "VIA1,VIA2,VIA3,VIA4,VIA5,VIA6,VIA7,VIA8",
        "rc_technology_name": "crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical",
        "rtl_imported": "false",
        "pnr_invoked": "false",
    }
    for key, value in expected.items():
        need(facts.get(key) == value, f"machine fact {key} mismatch")
    lines = coverage.read_text().splitlines()
    need(lines[0] == "master\ttt\tss\tff\tphysical", "coverage header mismatch")
    need(len(lines) == 95, "coverage must contain exactly 94 masters")
    names = []
    for line in lines[1:]:
        fields = line.split("\t")
        need(len(fields) == 5 and all(int(x) >= 1 for x in fields[1:]), "coverage row invalid")
        names.append(fields[0])
    need(names == sorted(set(names)), "coverage master names not unique sorted")
    process = json.loads(process_tree.read_text())
    need(process["root_seen"] is True, "process root not observed")
    need(process["unique_process_identity_count"] >= 2, "ICC2 descendant census too small")
    need(isinstance(process["tool_spawned_conversion_child_count"], int), "child count not integer")
    need((work / "repo_root_before.sha256").read_bytes() == (work / "repo_root_after.sha256").read_bytes(),
         "repository root collateral changed")
    prior = work / "prior_m2135_collateral" / "icc2_output.txt"
    need(prior.is_file() and not prior.is_symlink(), "prior M2135 collateral was not absorbed")
    need(sha(prior) == "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6",
         "prior M2135 collateral identity mismatch")
    payload = {
        "schema": "m2141_m2147_icc2_library_import_preflight_raw_r1_v1",
        "milestone_source": "M2141",
        "milestone_execution": "M2147",
        "status": "RAW_PASS_M2147_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER",
        "claim_boundary": {
            "library_import_preflight_only": True,
            "rtl_imported": False,
            "pnr": False,
            "timing": False,
            "area": False,
            "power": False,
            "paper_ppa_ready": False,
            "authorizes_full_pnr": False,
        },
        "gates": {
            "option_round_trip": True,
            "frame_conversion_status": 1,
            "mapped_master_union_count": 94,
            "logical_physical_views_per_master": 4,
            "routing_metal_layers": 9,
            "via_layers": 8,
            "rc_technology_name": expected["rc_technology_name"],
        },
        "process_census": {
            "top_level_icc2_shell_runs": 1,
            "tool_spawned_conversion_child_count": process["tool_spawned_conversion_child_count"],
            "unique_process_identity_count": process["unique_process_identity_count"],
        },
        "identity": {
            "icc2_log_sha256": sha(log),
            "machine_facts_sha256": sha(facts_file),
            "master_coverage_sha256": sha(coverage),
            "process_tree_sha256": sha(process_tree),
            "prior_m2135_collateral_sha256": sha(prior),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PASS_RAW_M2147_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
