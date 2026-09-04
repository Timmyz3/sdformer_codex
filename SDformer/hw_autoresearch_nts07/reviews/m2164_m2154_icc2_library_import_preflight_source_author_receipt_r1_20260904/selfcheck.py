#!/usr/bin/python3.12
"""Read-only author selfcheck for M2164; invokes no EDA or license client."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m2164_m2154_icc2_library_import_preflight_source_contract_r1_20260904.json"
RUNNER = HW / "dc_handoff/scripts/run_m2164_m2154_icc2_library_import_preflight_one_shot.sh"
CHECKER = HW / "system_simulator/scripts/check_m2164_icc2_library_import_preflight.py"
TCL = HW / "dc_handoff/scripts/run_icc2_m2153_library_import_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2153_icc2_process_tree.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
M2154 = HW / "reviews/m2154_m2153_m2146_icc2_library_import_preflight_source_hammer_r1_20260904"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
KNOWN_NDM = Path("/opt/synopsys/icc2/V-2023.12-SP3/libraries/syn/gtech.nlib/reflib.ndm")
M2166_RESULT = HW / "dc_handoff/runs/m2166_m2164_icc2_library_import_preflight_raw_r1_20260904"
M2166_ATTEMPT = HW / "dc_handoff/runs/.m2166_m2164_icc2_library_import_preflight_attempt_consumed"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    expected = {
        TCL: "4df768db7385fe2c6d2807104f650c925310012a5c21d96e7d396086a3433e65",
        MONITOR: "a4d002f50b3fc45a31f98a2863f1dd39477f81bd219fbc55454b470ec1be56d1",
        INVENTORY: "351db733e16f15895c7f1658b21c16901ff907ed5613cb89c2f4a85ce8928f94",
        MASTER_LIST: "e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b",
        M2154 / "review.json": "ebc29bcbddaa0837241bd23a8a473fdbbd762340009f052644946e2590908b39",
        DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        KNOWN_NDM: "56f9a2c14fc9ce7d3d7691146bbc89db35c58a4fe40543833a924a23e8ada829",
    }
    for path in (CONTRACT, RUNNER, CHECKER, *expected):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"M2164_SELFCHECK_FAIL missing/symlink {path}")
    for path, digest in expected.items():
        if sha(path) != digest:
            raise SystemExit(f"M2164_SELFCHECK_FAIL frozen identity {path}")
    subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=True)
    subprocess.run(["/usr/bin/python3.12", "-B", "-m", "py_compile", str(CHECKER)], check=True,
                   env={"PYTHONPYCACHEPREFIX": "/tmp/m2164_no_repo_pycache"})

    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "SOURCE_ONLY_PENDING_M2165_INDEPENDENT_HAMMER":
        raise SystemExit("M2164_SELFCHECK_FAIL contract status")
    if contract["author_authorization"]["m2166"] is not False:
        raise SystemExit("M2164_SELFCHECK_FAIL author authorized production")
    if contract["exact_runtime_budget_after_m2165_pass"] != {
        "license_queries": 1,
        "top_level_icc2_shell_runs": 1,
        "pnr_runs": 0,
        "automatic_retry": False,
        "tool_spawned_children": "observed and counted, not additional top-level launches",
    }:
        raise SystemExit("M2164_SELFCHECK_FAIL runtime budget")

    runner = RUNNER.read_text()
    checker = CHECKER.read_text()
    required_runner = [
        '"${ICC2}" -no_init -f "${TCL}"',
        "M2164_EXPECTED_RUNNER_SHA256",
        "PASS_M2165_M2164_SOURCE_HAMMER__M2166_ONE_SHOT_AUTHORIZED",
        "m2166_m2164_icc2_library_import_preflight",
        "automatic_retry': False",
        "56f9a2c14fc9ce7d3d7691146bbc89db35c58a4fe40543833a924a23e8ada829",
    ]
    if any(token not in runner for token in required_runner):
        raise SystemExit("M2164_SELFCHECK_FAIL runner anchor")
    if runner.count('"${ICC2}" -no_init -f "${TCL}"') != 1:
        raise SystemExit("M2164_SELFCHECK_FAIL top-level ICC2 site count")
    if runner.count('"${LMUTIL}" lmstat') != 1:
        raise SystemExit("M2164_SELFCHECK_FAIL license-query site count")
    for token in (
        "native_ndm_member(frame", 'design_lib / "reflib.ndm"',
        "known NDM identity mismatch", "root has an observed internal parent",
        "cycle in process identity graph", "process identity graph has a disconnected component",
        "parent starts after child", "reachable == identity_keys",
    ):
        if token not in checker:
            raise SystemExit(f"M2164_SELFCHECK_FAIL checker anchor {token}")
    if any(path.exists() for path in (M2166_RESULT, M2166_ATTEMPT)):
        raise SystemExit("M2164_SELFCHECK_FAIL M2166 already consumed")
    if not (M2154 / "SHA256SUMS.seal.sha256").is_file():
        raise SystemExit("M2164_SELFCHECK_FAIL predecessor seal")
    if KNOWN_NDM.read_bytes()[:68] != bytes.fromhex(
        "b2bdea03be02010000104c696272617279204d616e61676572002a562d323032332e31322d53503320666f72206c696e75783634202d2d204d61792030372c2032303234"
    ):
        raise SystemExit("M2164_SELFCHECK_FAIL same-version native header")
    print("PASS_M2164_AUTHOR_SOURCE_SELFCHECK")
    print("native_header_bytes=68")
    print("process_graph=unique_root+reachable+acyclic+time_ordered")
    print("m2166_authorized=false")
    print("eda_runs=0")
    print("license_queries=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
