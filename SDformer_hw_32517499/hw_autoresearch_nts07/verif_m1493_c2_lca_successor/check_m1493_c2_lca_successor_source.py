#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA source checker for the M1493 C2 ``-lca`` successor."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1493_m1467_c2_mapped_vcs_saif_ptpx_lca_successor_one_shot.py"
CHECKER = Path(__file__).resolve()
TESTS = CHECKER.parent / "test_m1493_c2_lca_successor_source.py"
CONTRACT = HW / "contracts/m1493_m1467_c2_lca_successor_source_contract_r1_20260831.json"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1467_m1432_c2_mapped_vcs_saif_ptpx_debug_access_successor_one_shot.py"
OLD_ATTEMPT = HW / "results/.m1467_c2_mapped_vcs_saif_ptpx_attempt_consumed"
OLD_FAILURE = HW / "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
M1484 = HW / "reviews/m1484_m1467_c2_second_production_failure_forensic_r1_20260831"
FUTURE = (
    HW / "reviews/m1494_m1493_c2_lca_successor_source_blind_hammer_r1_20260831",
    HW / "contracts/m1495_m1494_m1493_c2_lca_successor_launch_release_r1_20260831.json",
    HW / "reviews/m1496_m1495_m1493_c2_lca_successor_final_launch_hammer_r1_20260831",
)
NEW_NAMESPACES = {
    "attempt": "results/.m1493_c2_mapped_vcs_saif_ptpx_attempt_consumed",
    "result": "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831",
    "failure": "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine",
    "private": "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite",
}
OLD_RUNNER_SHA = "120cb1a8abe3df1e537de6797b3962fe0a7496be78954ba3b31fd9c8627e9a8a"
OLD_ATTEMPT_SHA = {"payload": "a3eead113c10d0134dd83972aaa06c6b26256f7459a37d784f98c5eeb2c68f92",
                   "manifest": "830d359dc80f2690913fb9b9f9a05b02073fd99e88639844412ac5f25138f526",
                   "outer": "eba291930799326b00d5460ce66f32fe29fef0a8b9a379bd05a28794a0cd13dc"}
OLD_FAILURE_SHA = {"payload": "39f3d5ffa39508db348cddf116584267e68e8796a008a7949bad88e02dd2c015",
                   "manifest": "233067e03f011cb1c3b4bd9fb4160d4fa7225246fc2eab9159933cf3e8792dcd",
                   "outer": "5503f1cc7db87e2cb1417f72167a5f11b6cc9fe86972c847b96f617357f80e82"}
M1484_SHA = {"review": "d26f73469d3d9e131cb776d47c6ee12c2ddd9f546e47fae690f73d7f8186d826",
             "manifest": "d61787c9a4c25e8cfe6fe2b0980605b09cae9ffaf1d4c8406b28d93cd43618b3",
             "outer": "86c26e7109931199578e22cba7795aeea2673ea5e57f2524ed76790ce9d1487d"}
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
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
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha, "seal SHA drift")
    need(outer.read_text().split() == [manifest_sha, "SHA256SUMS"], "outer content")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and name not in listed
             and not rel.is_absolute() and ".." not in rel.parts, "manifest row")
        member = root / rel
        need(member.is_file() and not member.is_symlink()
             and stat.S_ISREG(member.lstat().st_mode) and sha(member) == digest,
             "manifest member")
        listed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(listed == actual, "sealed population")
    return listed


def check_predecessor() -> dict[str, Any]:
    need(sha(OLD_RUNNER) == OLD_RUNNER_SHA, "M1467 runner drift")
    need(verify_seal(OLD_ATTEMPT, OLD_ATTEMPT_SHA["manifest"], OLD_ATTEMPT_SHA["outer"])
         == {"attempt.json"}, "M1467 attempt population")
    need(sha(OLD_ATTEMPT / "attempt.json") == OLD_ATTEMPT_SHA["payload"],
         "M1467 attempt payload")
    need(verify_seal(OLD_FAILURE, OLD_FAILURE_SHA["manifest"], OLD_FAILURE_SHA["outer"])
         == {"failure.json"}, "M1467 failure population")
    need(sha(OLD_FAILURE / "failure.json") == OLD_FAILURE_SHA["payload"],
         "M1467 failure payload")
    failure = strict_json(OLD_FAILURE / "failure.json")
    need(failure.get("phase") == "SIM_k8_0"
         and failure.get("counts") == {"vcs_compiles": 1, "simv_runs": 1,
                                        "saif_files": 0, "ptpx_runs": 0}
         and failure.get("automatic_retry") is False
         and failure.get("partial_axis_citable") is False,
         "M1467 failure semantics")
    need(verify_seal(M1484, M1484_SHA["manifest"], M1484_SHA["outer"])
         == {"mechanical_checks.txt", "review.json", "review.md"}, "M1484 population")
    need(sha(M1484 / "review.json") == M1484_SHA["review"], "M1484 review")
    forensic = strict_json(M1484 / "review.json")
    need(forensic["failure"]["first_error_code"] == "Error-[LCA_FEATURES_NEED_OPTION]"
         and forensic["authorization"]["additive_successor_source_authoring_allowed"] is True
         and forensic["authorization"]["successor_execution_allowed_by_m1484"] is False,
         "M1484 semantics")
    return {"phase": "SIM_k8_0", "vcs_compiles": 1, "simv_runs": 1,
            "saif_files": 0, "ptpx_runs": 0, "required_option": "-lca",
            "attempt_consumed": True, "automatic_retry": False}


def check_minimal_delta_text(text: str) -> None:
    region = text[text.index("COMPILE_PREFIX ="):text.index("\n\n\nclass Failure")]
    need(region.count('"-debug_access+r"') == 1, "debug_access cardinality")
    need(region.count('"-lca"') == 1, "lca cardinality")
    old = OLD_RUNNER.read_text()
    old_region = old[old.index("COMPILE_PREFIX ="):old.index("\n\n\nclass Failure")]
    need(old_region.count('"-debug_access+r"') == 1 and '"-lca"' not in old_region,
         "M1467 compile premise")
    need("for axis in (\"k8\", \"k1x8\")" in text
         and "for case in range(5)" in text, "campaign loop drift")
    need(text.count('state["vcs_compiles"] += 1') == 1
         and text.count('state["simv_runs"] += 1') == 1
         and text.count('state["saif_files"] += 1') == 1
         and text.count('state["ptpx_runs"] += 1') == 1, "counter sites")
    need(text.index("ATTEMPT.mkdir()") < text.index("BASE.BASE.run(command"),
         "attempt after EDA")
    need('"automatic_retry": False' in text and '"partial_axis_citable": False' in text,
         "fail-close semantics")


def expected_contract() -> dict[str, Any]:
    return {
        "schema": "m1493_m1467_c2_lca_successor_source_contract_r1_v1",
        "status": "M1493_C2_LCA_SUCCESSOR_SOURCE_READY__FRESH_M1494_REQUIRED__NO_EDA",
        "date": "2026-08-31",
        "purpose": "Additive successor to consumed M1467. Preserve its exact two-axis/five-case mapped VCS-SAIF-PTPX campaign, retain -debug_access+r, and add only VCS -lca required by the frozen SV-SAIF UCLI command.",
        "identity": {"runner_path": RUNNER.relative_to(HW).as_posix(),
                     "runner_sha256": sha(RUNNER),
                     "checker_path": CHECKER.relative_to(HW).as_posix(),
                     "checker_sha256": sha(CHECKER),
                     "tests_path": TESTS.relative_to(HW).as_posix(),
                     "tests_sha256": sha(TESTS),
                     "m1467_runner_sha256": OLD_RUNNER_SHA,
                     "m1484_review_sha256": M1484_SHA["review"]},
        "predecessor_failure": {"attempt_payload_sha256": OLD_ATTEMPT_SHA["payload"],
            "failure_payload_sha256": OLD_FAILURE_SHA["payload"],
            "phase": "SIM_k8_0", "counts": {"vcs_compiles": 1, "simv_runs": 1,
            "saif_files": 0, "ptpx_runs": 0}, "attempt_consumed": True,
            "automatic_retry": False, "partial_axis_citable": False,
            "first_error_code": "Error-[LCA_FEATURES_NEED_OPTION]"},
        "sole_repair": {"vcs_compile_keep": ["-debug_access+r"],
                        "vcs_compile_add": ["-lca"], "vcs_compile_remove": [],
                        "rtl_change": False, "netlist_change": False,
                        "sdc_change": False, "testbench_change": False,
                        "workload_change": False, "ucli_change": False,
                        "ptpx_script_change": False, "saif_scope_change": False},
        "preserved_execution": {"axes": ["k8", "k1x8"],
            "cases": [0, 1, 2, 3, 4], "vcs_compiles": 2, "simv_runs": 10,
            "production_saif_files": 10, "ptpx_runs": 10,
            "ptpx_after_all_ten_saif_gates": True,
            "attempt_consumed_before_first_eda_tool": True,
            "same_uid_collision_gates_before_attempt": 2,
            "automatic_retry": False, "partial_axis_publication": False,
            "fresh_namespaces": dict(NEW_NAMESPACES)},
        "future_authority": {"source_hammer": FUTURE[0].relative_to(HW).as_posix(),
            "launch_release": FUTURE[1].relative_to(HW).as_posix(),
            "final_hammer": FUTURE[2].relative_to(HW).as_posix(),
            "fresh_different_author_required": True, "launch_authorized": False},
        "author_execution": {"source_authoring": True, "source_only_tests": True,
            "license_query": False, "vcs": False, "simv": False, "saif": False,
            "pt": False, "ptpx": False, "eda": False,
            "attempt_consumed": False, "launch": False},
        "claim_boundary": dict(CLAIMS),
        "protected": {"docs359_sha256": DOCS359_SHA,
                      "ucli_key_modified": False},
    }


def check_contract() -> dict[str, Any]:
    value = strict_json(CONTRACT)
    need(value == expected_contract(), "contract exact-set/value drift")
    return value


def check_source(require_future_absent: bool = True) -> dict[str, Any]:
    need(sha(DOCS359) == DOCS359_SHA, "docs359 drift")
    predecessor = check_predecessor()
    check_minimal_delta_text(RUNNER.read_text())
    contract = check_contract()
    for value in NEW_NAMESPACES.values():
        need(not os.path.lexists(HW / value), "M1493 namespace residue")
    if require_future_absent:
        need(not any(os.path.lexists(path) for path in FUTURE), "future authority exists")
    return {"schema": "m1493_c2_lca_successor_source_check_r1_v1",
            "status": "PASS_M1493_C2_LCA_SUCCESSOR_SOURCE__NO_EDA",
            "predecessor": predecessor, "contract_status": contract["status"],
            "sole_delta": "vcs_compile_add_lca_after_debug_access_r",
            "claim_boundary": CLAIMS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    parser.parse_args()
    print(json.dumps(check_source(True), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
