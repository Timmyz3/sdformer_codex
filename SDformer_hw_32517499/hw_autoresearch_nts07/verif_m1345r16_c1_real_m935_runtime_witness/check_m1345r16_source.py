#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fail-closed source gate for the additive M1345/R16 C1 witness checker."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent
R15_DIR = HW / "verif_m1337r15_c1_real_m935_runtime_witness"
WITNESS = R15_DIR / "m1337r15_m935_runtime_witness.sv"
FILELIST = R15_DIR / "m1337r15_unit_delay_filelist.f"
R15_CHECKER = R15_DIR / "check_m1337r15_source.py"
R15_TEST = R15_DIR / "test_m1337r15_source.py"
R15_CONTRACT = HW / "contracts/m1337_c1_r15_real_m935_runtime_witness_source_contract_r1_20260831.json"
CHECKER = Path(__file__).resolve()
TEST = HERE / "test_m1345r16_source.py"
CONTRACT = HW / "contracts/m1345_c1_r16_real_m935_runtime_witness_source_contract_r1_20260831.json"
R15_AUTHOR = HW / "reviews/m1337_c1_r15_real_m935_runtime_witness_source_author_r1_20260831"
R15_FAILED = HW / "reviews/m1339_m1337_c1_r15_runtime_witness_source_blind_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED_NORMALIZED_WITNESS_SHA = "3a550df82d0fbdaa1db5591c651539943187796f391b4d92b5d0d19158d958c6"
EXPECTED_PATH_SHA = {
    WITNESS: "0ec7179e36f9af09e3020f76a5a927298d877b3cc20c6ac9ab4686bf465d18af",
    FILELIST: "87a8b5e7500808a8afbd4339668aae3a44db2de7924a948020e2c7bffce4289e",
    R15_CHECKER: "ba6d8c9b1e66854ee58cf3a3b247cceb1629495d2a5c6ca11aa93b7ba14c1326",
    R15_TEST: "ed2c92dde2ca6c96ec55f00b21188d6ea8bdf2426c89f188896351c314c6de9c",
    R15_CONTRACT: "49c55065bdafda15a75f5520d22428671ea3353a53c692270f47fbce5c80e5b8",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
R15_AUTHOR_SEAL = {
    "review": "fee38289e55bcb61b05cda5d75a4483a27c9bc053b976a018e4852db3cea0da7",
    "manifest": "59226f03c833ca657af7eacc60ada87ce75f0401ab7ca1737a823d25211e9374",
    "outer_file": "c56c890f41bcff07349af838ef390bf1764427ba9da7fc42f2708a39e932d2f0",
}
R15_FAILED_SEAL = {
    "review": "d3276f82ed2f19d46570930feb6aa858a24f25b1b6d8ca90354e7065f575e1a1",
    "manifest": "d3ef591063a44723d8e5e31a3474fc851e4dc4240778b0c2819af52c9b0ff5b7",
    "outer_file": "2770d6ee9ef82310d45b4a546bb7e9fdbb5bbaa461f53b48979f2558935d177a",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R15 = load_module("m1345_bound_m1337_checker", R15_CHECKER)


def normalized_sv(text: str) -> str:
    code = R15.strip_sv_comments(text)
    return re.sub(r"\s+", "", code)


def normalized_sha(text: str) -> str:
    return hashlib.sha256(normalized_sv(text).encode("utf-8")).hexdigest()


def stage_body(code: str, stage: str, next_stage: str) -> str:
    begin = code.find(stage + ": begin")
    end = code.find(next_stage + ": begin", begin + 1)
    require(begin >= 0 and end > begin, "stage body missing: " + stage)
    return re.sub(r"\s+", "", code[begin:end])


def check_registered_stage_semantics(code: str) -> None:
    requirements = {
        ("W_FIRST_REQUEST", "W_FIRST_ACCEPT"): (
            "(weight_request_fire===1'b0)",
            "responses_q<=4'd1", "core_accepts_q<=4'd1",
            "stage_q<=W_FIRST_ACCEPT"),
        ("W_SECOND_REQUEST", "W_SECOND_ACCEPT"): (
            "(psum_commit_fire===1'b0)",
            "responses_q<=4'd2", "core_accepts_q<=4'd2",
            "stage_q<=W_SECOND_ACCEPT"),
        ("W_SECOND_ACCEPT", "W_PSUM_COMMIT"): (
            "(row_complete_fire===1'b0)",
            "psum_commits_q<=4'd1", "stage_q<=W_PSUM_COMMIT"),
        ("W_PSUM_COMMIT", "W_ROW_DONE"): (
            "(task_done_fire===1'b0)",
            "row_completions_q<=4'd1", "stage_q<=W_ROW_DONE"),
    }
    for (stage, next_stage), snippets in requirements.items():
        body = stage_body(code, stage, next_stage)
        for snippet in snippets:
            require(body.count(snippet) == 1,
                    "registered-stage semantic drift: " + stage + " / " + snippet)


def check_complete_control_unknown(code: str) -> None:
    match = re.search(r"control_unknown\s*=\s*\$isunknown\s*\(\s*\{(.*?)\}\s*\)\s*;",
                      code, flags=re.S)
    require(match is not None, "control_unknown expression missing")
    actual = [item.strip() for item in match.group(1).split(",")]
    expected = [
        "weight_request_fire", "psum_request_fire", "response_accept",
        "core_accept", "psum_commit_fire", "row_complete_fire", "task_done_fire",
        "request_hold_attack_mode", "weight_service_attack_mode",
        "psum_service_attack_mode", "protocol_error", "boundary_fault",
        "core_fault", "m935_fault", "weight_service_fault", "psum_service_fault",
    ]
    require(actual == expected, "control_unknown complete ordered set drift")


def check_complete_final_oracle(code: str) -> None:
    start = code.find("final begin : witness_final_oracle")
    stop = code.find("end\nendmodule", start)
    require(start >= 0 and stop > start, "final oracle missing")
    oracle = re.sub(r"\s+", "", code[start:stop])
    required = (
        "(design_issue_accepts===64'd2)",
        "(design_psum_commits===64'd1)",
        "(design_row_completions===64'd1)",
    )
    for term in required:
        require(oracle.count(term) == 1, "final design-count conjunct drift: " + term)


def check_witness_text(text: str) -> None:
    code = R15.strip_sv_comments(text)
    R15.check_witness_text(text)
    check_registered_stage_semantics(code)
    check_complete_control_unknown(code)
    check_complete_final_oracle(code)
    require(normalized_sha(text) == EXPECTED_NORMALIZED_WITNESS_SHA,
            "exact normalized canonical R15 witness drift")


def check_contract_dict(contract: dict[str, Any]) -> None:
    require(contract.get("schema") ==
            "m1345_c1_r16_real_m935_runtime_witness_source_contract_r1_v1",
            "contract schema drift")
    require(contract.get("status") ==
            "SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_RELEASE_NO_VCS_NO_EDA",
            "contract status drift")
    require(contract.get("r15_failed_authority") == {
        "path": R15_FAILED.relative_to(HW.parent).as_posix(),
        "review_sha256": R15_FAILED_SEAL["review"],
        "manifest_sha256": R15_FAILED_SEAL["manifest"],
        "outer_file_sha256": R15_FAILED_SEAL["outer_file"],
        "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "false_negative_count": 14,
    }, "R15 failed authority drift")
    require(contract.get("frozen_r15_source") == {
        "witness_path": WITNESS.relative_to(HW.parent).as_posix(),
        "witness_sha256": EXPECTED_PATH_SHA[WITNESS],
        "filelist_path": FILELIST.relative_to(HW.parent).as_posix(),
        "filelist_sha256": EXPECTED_PATH_SHA[FILELIST],
        "r15_checker_sha256": EXPECTED_PATH_SHA[R15_CHECKER],
        "r15_test_sha256": EXPECTED_PATH_SHA[R15_TEST],
        "r15_contract_sha256": EXPECTED_PATH_SHA[R15_CONTRACT],
        "normalized_witness_sha256": EXPECTED_NORMALIZED_WITNESS_SHA,
        "r15_witness_modified": False,
    }, "frozen R15 source identity drift")
    require(contract.get("new_r16_source") == {
        "checker_path": CHECKER.relative_to(HW.parent).as_posix(),
        "checker_sha256": sha(CHECKER),
        "test_path": TEST.relative_to(HW.parent).as_posix(),
        "test_sha256": sha(TEST),
        "python_path": str(PYTHON),
        "python_sha256": sha(PYTHON),
    }, "R16 source identity drift")
    closure = contract.get("closure", {})
    require(closure.get("registered_stage_mutations") == 4
            and closure.get("control_unknown_mutations") == 7
            and closure.get("final_oracle_mutations") == 3
            and closure.get("new_mutations_total") == 14
            and closure.get("inherited_r15_tests") == 20
            and closure.get("directed_tests_total") == 34,
            "R16 closure/test-count drift")
    require(contract.get("launch_authorized") is False
            and contract.get("release_present") is False, "launch/release drift")
    execution = contract.get("author_execution", {})
    require(all(execution.get(key) is False for key in
                ("release", "vcs", "simv", "dc", "pt", "ptpx", "eda", "gpu", "remote")),
            "author execution boundary drift")
    boundary = contract.get("claim_boundary", {})
    require(boundary.get("source_only") is True and all(boundary.get(key) is False
            for key in ("source_admitted", "functional_vcs", "timing_verified",
                        "cycles_measured", "speedup", "ppa", "power", "energy",
                        "system_speedup", "headline")), "claim boundary drift")


def main() -> int:
    for path, digest in EXPECTED_PATH_SHA.items():
        regular(path)
        require(sha(path) == digest, "frozen identity drift: " + str(path))
    R15.verify_dir(R15_AUTHOR, R15_AUTHOR_SEAL)
    R15.verify_dir(R15_FAILED, R15_FAILED_SEAL)
    failed = json.loads((R15_FAILED / "review.json").read_text())
    require(failed.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
            and failed.get("false_negative_count") == 14,
            "M1339 verdict drift")
    for path in (CHECKER, TEST, CONTRACT):
        regular(path)
    check_witness_text(WITNESS.read_text())
    R15.check_witness_text(WITNESS.read_text())
    R15.check_contract_dict(json.loads(R15_CONTRACT.read_text()))
    check_contract_dict(json.loads(CONTRACT.read_text()))
    require(not list(HW.glob("contracts/m1345*c1*r16*release*.json")),
            "R16 release unexpectedly exists")
    print(json.dumps({
        "status": "PASS_M1345R16_SOURCE_AUTHORING__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED",
        "r15_witness_unchanged": True,
        "inherited_r15_tests": 20,
        "new_r16_mutations": 14,
        "directed_tests_total": 34,
        "launch_authorized": False,
        "release_present": False,
        "vcs_runs": 0,
        "eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
