#!/usr/bin/python3.12
"""Read-only author selfcheck for M2168; invokes no EDA or license client."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m2168_m2167_icc2_library_import_preflight_source_contract_r1_20260904.json"
RUNNER = HW / "dc_handoff/scripts/run_m2168_m2167_icc2_library_import_preflight_one_shot.sh"
CHECKER = HW / "system_simulator/scripts/check_m2164_icc2_library_import_preflight.py"
TCL = HW / "dc_handoff/scripts/run_icc2_m2153_library_import_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2153_icc2_process_tree.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
M2167 = HW / "reviews/m2167_m2166_m2164_icc2_library_preflight_startup_failure_hammer_r1_20260904"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2170_RESULT = HW / "dc_handoff/runs/m2170_m2168_icc2_library_import_preflight_raw_r1_20260904"
M2170_ATTEMPT = HW / "dc_handoff/runs/.m2170_m2168_icc2_library_import_preflight_attempt_consumed"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_isolated_layout_for_test(
    isolated_raw: Path,
    listed_raw: list[Path],
    design_lib: Path,
    frame_ndm: Path,
) -> None:
    """Pure mirror of the runner's startup postconditions for mutation tests."""
    if len(listed_raw) != 7:
        raise ValueError("seven exact isolation directories required")
    if not isolated_raw.is_dir() or isolated_raw.is_symlink():
        raise ValueError("isolated root is not a real directory")
    isolated = isolated_raw.resolve(strict=True)
    if isolated != isolated_raw.absolute():
        raise ValueError("isolated root traverses a symlink")
    for path in listed_raw:
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"isolation child is not a real directory: {path}")
        resolved = path.resolve(strict=True)
        if resolved == isolated or isolated not in resolved.parents:
            raise ValueError(f"isolation child escapes root: {path}")
        relative = path.absolute().relative_to(isolated)
        cursor = isolated
        for part in relative.parts:
            cursor = cursor / part
            mode = os.lstat(cursor).st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                raise ValueError(f"symlink or nondirectory component: {cursor}")
    for output in (design_lib, frame_ndm):
        if output.exists() or output.is_symlink():
            raise ValueError(f"stale output preexists: {output}")


def validate_runner_source(text: str) -> None:
    required = [
        "M2168_EXPECTED_RUNNER_SHA256",
        "PASS_M2169_M2168_SOURCE_HAMMER__M2170_ONE_SHOT_AUTHORIZED",
        "m2170_m2168_icc2_library_import_preflight",
        "M2168_LAYOUT_GATE_PASS paths=7 strictly_below=true symlinks=0",
        "M2168_OUTPUT_ABSENCE_GATE_PASS design_nlib=absent frame_ndm=absent",
        "M2168_EXECUTION_CONTRACT_WRITE_PASS",
        "M2168_EXECUTION_CONTRACT_REREAD_PASS",
        "path.resolve(strict=True)",
        "os.lstat(str(cursor)).st_mode",
        "not path.is_symlink()",
        "not out.is_symlink()",
        "json.loads(path.read_text()) == expected",
        "[[ ! -e \"${DESIGN_LIB}\" && ! -L \"${DESIGN_LIB}\" ]]",
        "[[ ! -e \"${FRAME_NDM}\" && ! -L \"${FRAME_NDM}\" ]]",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError(f"runner anchor missing: {missing}")
    if text.count("mkdir -p -- \\\n") != 1:
        raise ValueError("nested isolation creation is not one mkdir -p operation")
    if text.count('"${LMUTIL}" lmstat') != 1:
        raise ValueError("license site count is not one")
    if text.count('"${ICC2}" -no_init -f "${TCL}"') != 1:
        raise ValueError("top-level ICC2 site count is not one")
    if text.count(': >"${LAUNCH_GATE}"') != 1:
        raise ValueError("launch-gate release site count is not one")

    creation_start = text.index("mkdir -p -- \\\n")
    creation_end = text.index("/usr/libexec/platform-python3.6 -I - \"${ISOLATED}\"", creation_start)
    creation = text[creation_start:creation_end]
    exact_dirs = [
        '"${ISOLATED}/home"', '"${ISOLATED}/tmp"',
        '"${ISOLATED}/cache/xdg"', '"${ISOLATED}/cache/library"',
        '"${ISOLATED}/frame_output"', '"${ISOLATED}/frame_logs"',
        '"${ISOLATED}/reports"',
    ]
    if any(creation.count(path) != 1 for path in exact_dirs):
        raise ValueError("single creation block does not name seven exact paths once")

    order = [
        text.index('"${INVENTORY}" --root "${REPO_ROOT}"'),
        text.index("cp --reflink=never"),
        creation_start,
        text.index("M2168_LAYOUT_GATE_PASS"),
        text.index("M2168_OUTPUT_ABSENCE_GATE_PASS"),
        text.index("M2168_EXECUTION_CONTRACT_WRITE_PASS"),
        text.index("M2168_EXECUTION_CONTRACT_REREAD_PASS"),
        text.index('"${LMUTIL}" lmstat'),
    ]
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError("license site precedes startup/contract gates")

    wait_guard = text.index('while [[ ! -e "${LAUNCH_GATE}" ]]')
    icc2_site = text.index('"${ICC2}" -no_init -f "${TCL}"')
    monitor_site = text.index('"${MONITOR}" --root-pid')
    monitor_ready = text.index('[[ -e "${MONITOR_READY}" ]] || exit 5')
    gate_release = text.index(': >"${LAUNCH_GATE}"')
    if not (wait_guard < icc2_site < monitor_site < monitor_ready < gate_release):
        raise ValueError("ICC2 launch is not deferred until monitor readiness")


def main() -> int:
    expected = {
        CHECKER: "b732130841808f67791eeb9907a327c5149dc94f119507dfcccc65240fbeabc5",
        TCL: "4df768db7385fe2c6d2807104f650c925310012a5c21d96e7d396086a3433e65",
        MONITOR: "a4d002f50b3fc45a31f98a2863f1dd39477f81bd219fbc55454b470ec1be56d1",
        INVENTORY: "351db733e16f15895c7f1658b21c16901ff907ed5613cb89c2f4a85ce8928f94",
        MASTER_LIST: "e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b",
        M2167 / "review.json": "ee09260d23e4e8b140e4a943c2de58d2c9ad6694ad203b78542f2a6e1fbd3d1a",
        DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    }
    for path in (CONTRACT, RUNNER, *expected):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"M2168_SELFCHECK_FAIL missing/symlink {path}")
    for path, digest in expected.items():
        if sha(path) != digest:
            raise SystemExit(f"M2168_SELFCHECK_FAIL frozen identity {path}")
    subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=True)
    validate_runner_source(RUNNER.read_text())

    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "SOURCE_ONLY_PENDING_M2169_INDEPENDENT_HAMMER":
        raise SystemExit("M2168_SELFCHECK_FAIL contract status")
    if contract["author_authorization"] != {
        "m2169_independent_hammer": True,
        "m2170": False,
        "license_queries": 0,
        "top_level_icc2_shell_runs": 0,
        "pnr_runs": 0,
        "automatic_retry": False,
    }:
        raise SystemExit("M2168_SELFCHECK_FAIL author authorization")
    if contract["exact_runtime_budget_after_m2169_pass"] != {
        "license_queries": 1,
        "top_level_icc2_shell_runs": 1,
        "pnr_runs": 0,
        "automatic_retry": False,
        "tool_spawned_children": "observed and counted, not additional top-level launches",
    }:
        raise SystemExit("M2168_SELFCHECK_FAIL runtime budget")
    if any(path.exists() for path in (M2170_RESULT, M2170_ATTEMPT)):
        raise SystemExit("M2168_SELFCHECK_FAIL M2170 already consumed")
    if not (M2167 / "SHA256SUMS.seal.sha256").is_file():
        raise SystemExit("M2168_SELFCHECK_FAIL predecessor seal")
    print("PASS_M2168_AUTHOR_SOURCE_SELFCHECK")
    print("startup=single_mkdir_p+seven_real_dirs+no_stale_outputs+exact_contract_reread")
    print("launch=license_then_gated_wrapper+ready_monitor+single_icc2")
    print("m2170_authorized=false")
    print("eda_runs=0")
    print("license_queries=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
