#!/usr/bin/python3.12
"""Read-only author selfcheck for the additive M2153 source package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
TCL = HW / "dc_handoff/scripts/run_icc2_m2153_library_import_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2153_m2146_icc2_library_import_preflight_one_shot.sh"
MONITOR = HW / "dc_handoff/scripts/monitor_m2153_icc2_process_tree.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
CHECKER = HW / "system_simulator/scripts/check_m2153_icc2_library_import_preflight.py"
CONTRACT = HW / "contracts/m2153_m2146_icc2_library_import_preflight_source_contract_r1_20260904.json"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
M2146 = HW / "reviews/m2146_m2141_m2136_icc2_library_import_preflight_source_hammer_r1_20260904"
M2029 = HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
NETLISTS = [
    M2029 / "ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
    M2029 / "tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
]
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
MW_REF = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
ICC2_WRAPPER = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell")
ICC2_REAL = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/dgcom_exec")
COLLATERAL = REPO / "icc2_output.txt"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_errors(tcl: str, runner: str, monitor: str, inventory: str, checker: str) -> list[str]:
    errors: list[str] = []
    required_tcl = [
        "set_app_options -name lib.configuration.local_output_dir -value $cache",
        "get_app_option_value -name lib.configuration.local_output_dir",
        "generate_frame_from_mw $frame_name -mw_lib $mw_ref",
        "create_lib -ref_libs [list $frame_ndm] $design_lib",
        "get_site_defs -quiet -exact core",
        "read_parasitic_tech -tlup $nxtgrd -layermap $layer_map",
        "m2153_tree_stats $frame_ndm \"frame NDM\"",
        "m2153_tree_stats $design_lib \"design library\"",
    ]
    for token in required_tcl:
        if token not in tcl:
            errors.append(f"missing Tcl anchor {token}")
    active_tcl = "\n".join(line for line in tcl.splitlines() if not line.lstrip().startswith("#"))
    if "-overwrite" in active_tcl:
        errors.append("frame overwrite enabled")
    if "get_site_defs -quiet *core*" in active_tcl or "get_site_defs *core*" in active_tcl:
        errors.append("wildcard core site present")
    banned = (
        "read_verilog", "read_vhdl", "link_block", "current_block", "compile_fusion",
        "initialize_floorplan", "create_placement", "place_opt", "clock_opt", "route_auto",
        "route_opt", "write_parasitics", "report_timing", "report_area", "report_power",
    )
    for command in banned:
        if re.search(rf"^\s*{re.escape(command)}(?:\s|$)", active_tcl, re.MULTILINE):
            errors.append(f"prohibited Tcl command {command}")
    if len(re.findall(r"^\s*set conversion_status \[generate_frame_from_mw\b", active_tcl, re.MULTILINE)) != 1:
        errors.append("generate_frame_from_mw count is not one")
    if len(re.findall(r"^\s*set created \[create_lib\b", active_tcl, re.MULTILINE)) != 1:
        errors.append("create_lib count is not one")

    required_runner = [
        'env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C',
        'HOME="${ISOLATED}/home" TMPDIR="${ISOLATED}/tmp" XDG_CACHE_HOME="${ISOLATED}/cache/xdg"',
        '"${ICC2}" -no_init -f "${TCL}"',
        'sha_executable_exact 4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c "${ICC2_REAL}"',
        '--wrapper-path "${ICC2}" --actual-exec-path "${ICC2_REAL}"',
        '"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_before.json"',
        '"${INVENTORY}" --root "${REPO_ROOT}" --output "${WORK}/repo_root_after.json"',
        'cp --reflink=never --preserve=mode,timestamps -- "${PRIOR_COLLATERAL}"',
        'cmp -s -- "${WORK}/repo_root_before.json" "${WORK}/repo_root_after.json"',
        'top_level_icc2_shell_runs=1',
        'pnr_runs=0',
        'retry=false',
    ]
    for token in required_runner:
        if token not in runner:
            errors.append(f"missing runner anchor {token}")
    if runner.count('"${LMUTIL}" lmstat') != 1:
        errors.append("license-query site count is not one")
    if runner.count('"${ICC2}" -no_init -f "${TCL}"') != 1:
        errors.append("top-level exact ICC2 site count is not one")
    if re.search(r'^\s*mv .*\$\{PRIOR_COLLATERAL\}', runner, re.MULTILINE):
        errors.append("M2135 collateral is moved rather than preserved")

    for token in (
        '"icc2_exec"', '"icc2_exec-sle"', '"dgcom_exec"', '"lm_shell_exec"',
        '"icc2_lm_shell_exec"', '"common_shell_exec"', '"milkyway_exec"',
        'starttime_ticks', 'parent_links', 'exec_observations', 'selected_environment',
    ):
        if token not in monitor:
            errors.append(f"monitor misses {token}")
    for token in (
        'observed_masters == frozen_masters', 'frame_stats["binary_files"] > 0',
        'design_stats["binary_files"] > 0', 'before == after',
        'actual ICC2 command lacks -no_init', 'process identity count/list mismatch',
        'non-root identity has no observed parent', 'actual dgcom_exec never observed',
    ):
        if token not in checker:
            errors.append(f"checker misses {token}")
    for token in (
        'stat.S_ISREG', 'stat.S_ISDIR', 'stat.S_ISLNK', 'stat.S_ISFIFO',
        'stat.S_ISSOCK', 'stat.S_ISBLK', 'stat.S_ISCHR',
    ):
        if token not in inventory:
            errors.append(f"inventory misses {token}")
    return errors


def main() -> int:
    paths = [TCL, RUNNER, MONITOR, INVENTORY, CHECKER, CONTRACT, MASTER_LIST,
             DOCS359, MW_MANIFEST, ICC2_WRAPPER, ICC2_REAL, COLLATERAL,
             M2146 / "review.json", *NETLISTS]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"M2153_SELF_CHECK_FAIL missing/symlink: {path}")
    errors = source_errors(
        TCL.read_text(), RUNNER.read_text(), MONITOR.read_text(),
        INVENTORY.read_text(), CHECKER.read_text(),
    )
    if errors:
        raise SystemExit("M2153_SELF_CHECK_FAIL " + "; ".join(errors))

    expected = {
        DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        M2146 / "review.json": "c70e9ce4867d1cbd6010a2da0f403c5ee155a07ee0329888c226a7623ebdd51b",
        MASTER_LIST: "e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b",
        MW_MANIFEST: "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
        ICC2_WRAPPER: "825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4",
        ICC2_REAL: "4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c",
        COLLATERAL: "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6",
        NETLISTS[0]: "f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0",
        NETLISTS[1]: "739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af",
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise SystemExit(f"M2153_SELF_CHECK_FAIL identity: {path}")
    if not M2146.joinpath("SHA256SUMS.seal.sha256").is_file():
        raise SystemExit("M2153_SELF_CHECK_FAIL M2146 seal absent")

    masters = MASTER_LIST.read_text().splitlines()
    if len(masters) != 94 or masters != sorted(set(masters)):
        raise SystemExit("M2153_SELF_CHECK_FAIL frozen union94 malformed")
    cell_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\(", re.MULTILINE)
    union: set[str] = set()
    for netlist in NETLISTS:
        union.update(match.group(1) for match in cell_re.finditer(netlist.read_text())
                     if match.group(1).endswith("BWP35P140"))
    if masters != sorted(union):
        raise SystemExit("M2153_SELF_CHECK_FAIL union94 differs from mapped netlists")

    mw_files = {str(path.relative_to(MW_REF)) for path in MW_REF.rglob("*")
                if path.is_file() and not path.is_symlink()}
    manifest_files = {
        line.split("  ", 1)[1] for line in MW_MANIFEST.read_text().splitlines() if line.strip()
    }
    if len(mw_files) != 1051 or mw_files != manifest_files:
        raise SystemExit("M2153_SELF_CHECK_FAIL Milkyway inventory mismatch")
    if any(path.is_symlink() for path in MW_REF.rglob("*")):
        raise SystemExit("M2153_SELF_CHECK_FAIL Milkyway symlink")

    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "SOURCE_ONLY_PENDING_M2154_INDEPENDENT_HAMMER":
        raise SystemExit("M2153_SELF_CHECK_FAIL contract status")
    if contract["exact_runtime_budget_after_m2154_pass"] != {
        "license_queries": 1,
        "top_level_icc2_shell_runs": 1,
        "pnr_runs": 0,
        "automatic_retry": False,
        "tool_spawned_children": "observed and counted, not additional top-level launches",
    }:
        raise SystemExit("M2153_SELF_CHECK_FAIL contract budget")
    official = {
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat1/icc2_shell.1": "2662ac4bfae4515c12e4f08e172c9754f2894267bb2891ff3ecc0b4f4674ff26",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat2/set_app_options.2": "ae28a2f50dc5ed7457adad00428a0c0e7fa57cc4555866015d4ab4563e4ec0da",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat2/get_app_option_value.2": "f0d7b2b4334d00f90432c7fcdb319fe80668578633dfbda0bcdc644302e4e47a",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat3/lib.configuration.local_output_dir.3": "5354ec5b5964e454395a8f8d8cfecd489470d5c6555ec78242213d5925c6d9ea",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/generate_frame_from_mw.2": "f9424346c44d9d48cbae5a3839f26cadad46b4d85e405deb19354356cd232952",
        "/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/create_lib.2": "c19f9fd04239f0be10b97816cb4913ba71868b2b02f7c760d443cebdd40d835b",
    }
    for raw, digest in official.items():
        if sha(Path(raw)) != digest:
            raise SystemExit(f"M2153_SELF_CHECK_FAIL official documentation: {raw}")
    if "-no_init" not in Path(next(iter(official))).read_text(errors="replace"):
        raise SystemExit("M2153_SELF_CHECK_FAIL official -no_init semantics unavailable")

    print("PASS_M2153_AUTHOR_SOURCE_SELFCHECK")
    print("mapped_master_union_count=94")
    print("milkyway_regular_file_count=1051")
    print("license_queries_after_hammer=1")
    print("top_level_icc2_shell_runs_after_hammer=1")
    print("pnr_runs=0")
    print("automatic_retry=false")
    print("p0=0")
    print("p1=0")
    print("p2=0")
    print("source_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
