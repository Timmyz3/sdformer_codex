#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA source authority checker for additive M1467 C2 successor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent
CHECKER = Path(__file__).resolve()
TESTS = HERE / "test_m1467_c2_debug_access_successor_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1467_m1432_c2_mapped_vcs_saif_ptpx_debug_access_successor_one_shot.py"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_one_shot.py"
CONTRACT = HW / "contracts/m1467_m1432_c2_debug_access_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1467_m1432_c2_debug_access_successor_source_author_r1_20260831"
M1468 = HW / "reviews/m1468_m1467_c2_debug_access_successor_source_blind_hammer_r1_20260831"
M1469 = HW / "contracts/m1469_m1468_m1467_c2_debug_access_successor_launch_release_r1_20260831.json"
M1472 = HW / "reviews/m1472_m1469_m1467_c2_debug_access_successor_final_launch_hammer_r1_20260831"
OLD_ATTEMPT = HW / "results/.m1432_c2_mapped_vcs_saif_ptpx_attempt_consumed"
OLD_FAILURE = HW / "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
OLD_PRIVATE = HW / "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite"
NEW_NAMESPACES = {
    "attempt": "results/.m1467_c2_mapped_vcs_saif_ptpx_attempt_consumed",
    "result": "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831",
    "failure": "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine",
    "private": "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite",
}
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
PTPX = HW / "dc_handoff/scripts/run_ptpx.tcl"

OLD_RUNNER_SHA = "314be83304d4b62cf2c4b73feb394fa2ab20e60a89afb9c3dfc07622d25a7156"
OLD_ATTEMPT_SHA = {
    "payload": "3552c04045e19446fd9521e2a6145d6cf0c2090286f3cd5aa180a3074076f82f",
    "manifest": "9a50caa634e99c943677158babe9765b74ccab89b27e425d22a570ef5a9941f6",
    "outer": "ee66123a569c45de3aa0573a1db09af833428af300da1fe842f9e5c1b5be50f9",
}
OLD_FAILURE_SHA = {
    "payload": "4d21019bd0145b84646fad055de9b52fa66574144276027fa61598bd4e7607c5",
    "manifest": "2a2835af25d3947e6e445a8a268d3c254c986d8530267289fdc951fe917e7e97",
    "outer": "12ef0ad6c390ac343c68dc9f6936a8e4a1609427387d12dc4b63e412c5d401ec",
}
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
UCLI_SHA = "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1"
PTPX_SHA = "879398c8b8708589d42346af10d4825afac19c7c0622601685d1ea3f72245368"
CLAIMS = {key: False for key in (
    "functional_vcs_verified", "production_saif", "ptpx", "power", "energy",
    "performance", "system_speedup", "paper_ppa_ready", "headline")}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    need(path.is_file() and not path.is_symlink(), "JSON not regular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> set[str]:
    need(root.is_dir() and not root.is_symlink(), "sealed directory invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha, "seal SHA drift")
    need(outer.read_text().split() == [manifest_sha, "SHA256SUMS"], "outer content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest fields")
        digest, name = fields
        name = name.lstrip("*")
        rel = Path(name)
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None
             and name not in listed and not rel.is_absolute() and ".." not in rel.parts,
             "manifest row")
        member = root / rel
        need(member.is_file() and not member.is_symlink()
             and stat.S_ISREG(member.lstat().st_mode) and sha(member) == digest,
             "manifest member")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(listed == actual, "sealed population")
    return listed


def check_predecessor_failure() -> dict[str, Any]:
    need(sha(OLD_RUNNER) == OLD_RUNNER_SHA, "M1432 runner drift")
    need(verify_seal(OLD_ATTEMPT, OLD_ATTEMPT_SHA["manifest"],
                     OLD_ATTEMPT_SHA["outer"]) == {"attempt.json"},
         "M1432 attempt members")
    need(sha(OLD_ATTEMPT / "attempt.json") == OLD_ATTEMPT_SHA["payload"],
         "M1432 attempt payload")
    need(verify_seal(OLD_FAILURE, OLD_FAILURE_SHA["manifest"],
                     OLD_FAILURE_SHA["outer"]) == {"failure.json"},
         "M1432 failure members")
    need(sha(OLD_FAILURE / "failure.json") == OLD_FAILURE_SHA["payload"],
         "M1432 failure payload")
    attempt = strict_json(OLD_ATTEMPT / "attempt.json")
    failure = strict_json(OLD_FAILURE / "failure.json")
    need(attempt.get("status") == "M1432_ATTEMPT_CONSUMED"
         and attempt.get("automatic_retry") is False, "M1432 attempt semantics")
    need(failure.get("status") == "FAILED_OR_INCOMPLETE"
         and failure.get("phase") == "SIM_k8_0"
         and failure.get("counts") == {"vcs_compiles": 1, "simv_runs": 1,
                                        "saif_files": 0, "ptpx_runs": 0}
         and failure.get("attempt_consumed") is True
         and failure.get("automatic_retry") is False
         and failure.get("partial_axis_citable") is False,
         "M1432 failure semantics")
    need(OLD_PRIVATE.is_dir(), "M1432 private residue unexpectedly absent")
    # Deliberately never enumerate or hash the unsealed private build.
    return {"phase": "SIM_k8_0", "vcs_compiles": 1, "simv_runs": 1,
            "saif_files": 0, "ptpx_runs": 0, "attempt_consumed": True,
            "private_build_read": False, "automatic_retry": False}


def check_minimal_delta_text(text: str) -> None:
    """Reject any source text that loses or duplicates the sole compile delta."""
    need('COMPILE_PREFIX = [str(BASE.VCS)' in text
         and '"-debug_access+r"' in text, "M1467 debug flag absent")
    compile_prefix = text[text.index("COMPILE_PREFIX ="):
                          text.index("\n\n\nclass Failure")]
    need(compile_prefix.count('"-debug_access+r"') == 1,
         "debug flag cardinality drift")


def check_runner_static() -> dict[str, Any]:
    old = OLD_RUNNER.read_text()
    text = RUNNER.read_text()
    need('"-debug_access+r"' not in old, "M1432 missing-flag premise drift")
    need(sha(UCLI) == UCLI_SHA and "-gate_level all mda sv" in UCLI.read_text(),
         "UCLI power command drift")
    need(sha(PTPX) == PTPX_SHA, "PTPX script drift")
    check_minimal_delta_text(text)
    need("for axis in (\"k8\", \"k1x8\")" in text
         and "for case in range(5)" in text, "workload loop drift")
    need(text.count('state["vcs_compiles"] += 1') == 1
         and text.count('state["simv_runs"] += 1') == 1
         and text.count('state["saif_files"] += 1') == 1
         and text.count('state["ptpx_runs"] += 1') == 1,
         "execution counter site drift")
    need(text.index('if any(state[key] != COUNTS[key]') <
         text.index('state["phase"] = f"PTPX_{axis}_{case}"'),
         "PTPX reachable before all SAIF gates")
    need(text.count("BASE.collision_gate()") == 2, "collision gate count")
    need(text.index("ATTEMPT.mkdir()") < text.index("BASE.run(command"),
         "attempt not consumed before first EDA")
    need("partial_axis_citable\": False" in text
         and '"automatic_retry": False' in text, "fail-close drift")
    need(all(token in text for token in ("M1468", "M1469", "M1472")),
         "future authority chain absent")
    return {"sole_delta": "vcs_compile_add_debug_access_r", "axes": 2,
            "cases_per_axis": 5, "vcs_compiles": 2, "simv_runs": 10,
            "saif_files": 10, "ptpx_runs": 10, "collision_gates": 2,
            "attempt_before_eda": True, "partial_axis_citable": False}


def expected_contract() -> dict[str, Any]:
    return {
        "schema": "m1467_m1432_c2_debug_access_successor_source_contract_r1_v1",
        "status": "M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE_READY__FRESH_M1468_REQUIRED__NO_EDA",
        "date": "2026-08-31",
        "purpose": "Additive successor to the consumed M1432 C2 mapped activity campaign. Preserve its exact two-axis/five-case VCS-SAIF-PTPX execution and add only the missing VCS -debug_access+r observability flag required by the frozen UCLI power command.",
        "identity": {
            "runner_path": RUNNER.relative_to(HW).as_posix(),
            "runner_sha256": sha(RUNNER),
            "checker_path": CHECKER.relative_to(HW).as_posix(),
            "checker_sha256": sha(CHECKER),
            "tests_path": TESTS.relative_to(HW).as_posix(),
            "tests_sha256": sha(TESTS),
            "m1432_runner_path": OLD_RUNNER.relative_to(HW).as_posix(),
            "m1432_runner_sha256": OLD_RUNNER_SHA,
        },
        "predecessor_failure": {
            "attempt_path": OLD_ATTEMPT.relative_to(HW).as_posix(),
            "attempt_payload_sha256": OLD_ATTEMPT_SHA["payload"],
            "attempt_manifest_sha256": OLD_ATTEMPT_SHA["manifest"],
            "attempt_outer_file_sha256": OLD_ATTEMPT_SHA["outer"],
            "failure_path": OLD_FAILURE.relative_to(HW).as_posix(),
            "failure_payload_sha256": OLD_FAILURE_SHA["payload"],
            "failure_manifest_sha256": OLD_FAILURE_SHA["manifest"],
            "failure_outer_file_sha256": OLD_FAILURE_SHA["outer"],
            "status": "FAILED_OR_INCOMPLETE", "phase": "SIM_k8_0",
            "counts": {"vcs_compiles": 1, "simv_runs": 1,
                       "saif_files": 0, "ptpx_runs": 0},
            "attempt_consumed": True, "automatic_retry": False,
            "canonical_result": False, "partial_axis_citable": False,
            "private_build_unsealed_do_not_cite": True,
            "private_build_read_by_author": False,
        },
        "root_cause": {
            "frozen_compile_has_debug_access_r": False,
            "frozen_ucli_requires_gate_level_power": True,
            "frozen_ucli_sha256": UCLI_SHA,
            "ucli_failure_code": "UCLI-117",
            "diagnosis": "vcs_compile_omitted_debug_access_r_before_ucli_gate_level_power",
            "hardware_or_protocol_failure": False,
        },
        "sole_repair": {
            "vcs_compile_add": ["-debug_access+r"],
            "vcs_compile_remove": [], "rtl_change": False,
            "netlist_change": False, "sdc_change": False,
            "testbench_change": False, "workload_change": False,
            "cycle_expectation_change": False, "ucli_change": False,
            "ptpx_script_change": False, "saif_scope_change": False,
        },
        "preserved_execution": {
            "axes": ["k8", "k1x8"], "cases": [0, 1, 2, 3, 4],
            "vcs_compiles": 2, "simv_runs": 10,
            "production_saif_files": 10, "ptpx_runs": 10,
            "ptpx_after_all_ten_saif_gates": True,
            "partial_axis_publication": False,
            "attempt_consumed_before_first_eda_tool": True,
            "same_uid_collision_gates_before_attempt": 2,
            "automatic_retry": False, "replacement_allowed": False,
            "fresh_namespaces": dict(NEW_NAMESPACES),
        },
        "future_authority": {
            "source_hammer": M1468.relative_to(HW).as_posix(),
            "launch_release": M1469.relative_to(HW).as_posix(),
            "final_hammer": M1472.relative_to(HW).as_posix(),
            "fresh_different_author_required": True,
            "launch_authorized": False,
        },
        "author_execution": {
            "source_authoring": True, "source_only_tests": True,
            "private_build_read": False, "license_query": False,
            "vcs": False, "simv": False, "saif": False,
            "pt": False, "ptpx": False, "eda": False,
            "attempt_consumed": False, "launch": False,
        },
        "claim_boundary": dict(CLAIMS),
        "protected": {"docs359_sha256": DOCS359_SHA,
                      "ucli_key_modified": False},
    }


def check_contract() -> dict[str, Any]:
    contract = strict_json(CONTRACT)
    need(contract == expected_contract(), "contract exact-set/value drift")
    return contract


def check_source(require_future_absent: bool = True) -> dict[str, Any]:
    need(sha(DOCS359) == DOCS359_SHA, "docs359 drift")
    failure = check_predecessor_failure()
    runner = check_runner_static()
    contract = check_contract()
    for value in NEW_NAMESPACES.values():
        need(not os.path.lexists(HW / value), "M1467 namespace residue")
    if require_future_absent:
        need(not any(os.path.lexists(path) for path in (M1468, M1469, M1472)),
             "future authority already exists")
    return {"schema": "m1467_c2_debug_access_successor_source_check_r1_v1",
            "status": "PASS_M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE__NO_EDA",
            "failure": failure, "runner": runner,
            "contract_status": contract["status"], "claim_boundary": CLAIMS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    parser.parse_args()
    print(json.dumps(check_source(require_future_absent=True), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
