#!/usr/bin/python3.12
"""Read-only author selfcheck for the additive M2141 source package."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
TCL = HW / "dc_handoff/scripts/run_icc2_m2141_library_import_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2141_m2136_icc2_library_import_preflight_one_shot.sh"
MONITOR = HW / "dc_handoff/scripts/monitor_m2141_icc2_process_tree.py"
CHECKER = HW / "system_simulator/scripts/check_m2141_icc2_library_import_preflight.py"
CONTRACT = HW / "contracts/m2141_m2136_icc2_library_import_preflight_source_contract_r1_20260904.json"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
M2029 = HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
NETLISTS = [
    M2029 / "ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
    M2029 / "tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
]
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
MW_REF = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_errors(tcl: str, runner: str) -> list[str]:
    errors: list[str] = []
    required_tcl = [
        r"^\s*set_app_options -name lib\.configuration\.local_output_dir -value \$cache\s*$",
        r"^\s*set queried_cache \[file normalize \[get_app_option_value -name lib\.configuration\.local_output_dir\]\]\s*$",
        r"^\s*set conversion_status \[generate_frame_from_mw \$frame_name -mw_lib \$mw_ref \\\s*$",
        r"^\s*set created \[create_lib -ref_libs \[list \$frame_ndm\] \$design_lib\]\s*$",
        r"^\s*read_parasitic_tech -tlup \$nxtgrd -layermap \$layer_map \\\s*$",
        r"^\s*save_lib\s*$",
    ]
    for pattern in required_tcl:
        if not re.search(pattern, tcl, re.MULTILINE):
            errors.append(f"missing Tcl anchor {pattern}")
    if "-overwrite" in "\n".join(line for line in tcl.splitlines() if not line.lstrip().startswith("#")):
        errors.append("generate_frame_from_mw overwrite is present")
    banned_commands = (
        "read_verilog", "read_vhdl", "link_block", "current_block", "initialize_floorplan",
        "create_placement", "place_opt", "clock_opt", "route_auto", "route_opt",
        "write_parasitics", "report_timing", "report_area", "report_power",
    )
    for command in banned_commands:
        if re.search(rf"^\s*{re.escape(command)}(?:\s|$)", tcl, re.MULTILINE):
            errors.append(f"prohibited Tcl command {command}")
    if len(re.findall(r"^\s*generate_frame_from_mw\b|^\s*set conversion_status \[generate_frame_from_mw\b", tcl, re.MULTILINE)) != 1:
        errors.append("generate_frame_from_mw command count is not one")
    if len(re.findall(r"^\s*set created \[create_lib\b", tcl, re.MULTILINE)) != 1:
        errors.append("create_lib command count is not one")
    required_runner = [
        "cd -- \"${ISOLATED}\"",
        "top_level_icc2_shell_runs=1",
        "pnr_runs=0",
        "automatic_retry=false",
        "mv -- \"${PRIOR_COLLATERAL}\" \"${WORK}/prior_m2135_collateral/icc2_output.txt\"",
        "snapshot_repo_root \"${WORK}/repo_root_before.sha256\"",
        "snapshot_repo_root \"${WORK}/repo_root_after.sha256\"",
        "cmp -s -- \"${WORK}/repo_root_before.sha256\" \"${WORK}/repo_root_after.sha256\"",
        "--root-pid \"${icc2_pid}\"",
    ]
    for token in required_runner:
        if token not in runner:
            errors.append(f"missing runner anchor {token}")
    if runner.count('"${LMUTIL}" lmstat') != 1:
        errors.append("license query command count is not one")
    if runner.count('"${ICC2}" -f "${TCL}"') != 1:
        errors.append("top-level ICC2 command count is not one")
    if re.search(r"\b(for|while)\b[^\n]*(ICC2|lmstat)", runner):
        errors.append("EDA/license loop detected")
    return errors


def main() -> int:
    paths = [TCL, RUNNER, MONITOR, CHECKER, CONTRACT, MASTER_LIST, DOCS359, MW_MANIFEST, *NETLISTS]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"M2141_SELF_CHECK_FAIL missing/symlink: {path}")
    errors = text_errors(TCL.read_text(), RUNNER.read_text())
    if errors:
        raise SystemExit("M2141_SELF_CHECK_FAIL " + "; ".join(errors))
    masters = MASTER_LIST.read_text().splitlines()
    if len(masters) != 94 or masters != sorted(set(masters)):
        raise SystemExit("M2141_SELF_CHECK_FAIL master list is not 94 unique sorted names")
    netrefs: set[str] = set()
    for netlist in NETLISTS:
        netrefs.update(re.findall(r"^  ([A-Z0-9][A-Z0-9]*BWP35P140) \S+ \(", netlist.read_text(), re.MULTILINE))
    if sorted(netrefs) != masters:
        raise SystemExit("M2141_SELF_CHECK_FAIL union94 list differs from frozen netlists")
    if sha(MASTER_LIST) != "e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b":
        raise SystemExit("M2141_SELF_CHECK_FAIL union94 SHA")
    if sha(DOCS359) != "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4":
        raise SystemExit("M2141_SELF_CHECK_FAIL docs359 changed")
    mw_files = [path for path in MW_REF.rglob("*") if path.is_file() and not path.is_symlink()]
    if len(mw_files) != 1051:
        raise SystemExit("M2141_SELF_CHECK_FAIL frozen Milkyway count")
    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "SOURCE_ONLY_PENDING_M2146_INDEPENDENT_HAMMER":
        raise SystemExit("M2141_SELF_CHECK_FAIL contract status")
    if contract["exact_runtime_budget_after_m2146_pass"]["top_level_icc2_shell_runs"] != 1:
        raise SystemExit("M2141_SELF_CHECK_FAIL ICC2 budget")
    if contract["exact_runtime_budget_after_m2146_pass"]["pnr_runs"] != 0:
        raise SystemExit("M2141_SELF_CHECK_FAIL P&R budget")
    official = {
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat2/set_app_options.2": "ae28a2f50dc5ed7457adad00428a0c0e7fa57cc4555866015d4ab4563e4ec0da",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat2/get_app_option_value.2": "f0d7b2b4334d00f90432c7fcdb319fe80668578633dfbda0bcdc644302e4e47a",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat3/lib.configuration.local_output_dir.3": "5354ec5b5964e454395a8f8d8cfecd489470d5c6555ec78242213d5925c6d9ea",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/generate_frame_from_mw.2": "f9424346c44d9d48cbae5a3839f26cadad46b4d85e405deb19354356cd232952",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/create_lib.2": "c19f9fd04239f0be10b97816cb4913ba71868b2b02f7c760d443cebdd40d835b",
    }
    for raw, expected in official.items():
        if sha(Path(raw)) != expected:
            raise SystemExit(f"M2141_SELF_CHECK_FAIL official doc identity {raw}")
    collateral = REPO / "icc2_output.txt"
    if not collateral.is_file() or collateral.is_symlink() or sha(collateral) != "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6":
        raise SystemExit("M2141_SELF_CHECK_FAIL M2135 collateral identity")
    print("PASS_M2141_AUTHOR_SOURCE_SELFCHECK")
    print("mapped_master_union_count=94")
    print("milkyway_regular_file_count=1051")
    print("license_queries_after_hammer=1")
    print("top_level_icc2_shell_runs_after_hammer=1")
    print("pnr_runs=0")
    print("automatic_retry=false")
    print("source_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
