#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact-set/value source checker for the additive M1506 C1 successor."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent
RUNNER = HW / "dc_handoff/scripts/run_m1506_m1497_c1_active_next_oracle_release_safe_successor_one_shot.py"
CHECKER = Path(__file__).resolve()
TESTS = HERE / "test_m1506_source.py"
CONTRACT = HW / "contracts/m1506_c1_active_next_oracle_release_safe_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1506_c1_active_next_oracle_release_safe_successor_source_author_r1_20260831"
HAMMER = HW / "reviews/m1507_m1506_c1_active_next_oracle_release_safe_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1508_m1507_m1506_c1_active_next_oracle_vcs_launch_release_r1_20260831.json"
FINAL = HW / "reviews/m1509_m1508_m1506_c1_active_next_oracle_final_launch_hammer_r1_20260831"
M1497_RUNNER = HW / "dc_handoff/scripts/run_m1497_m1459_c1_active_next_oracle_clean_result_successor_one_shot.py"
M1497_CHECKER = HW / "verif_m1497_c1_active_next_oracle_successor/check_m1497_source.py"
M1497_TESTS = HW / "verif_m1497_c1_active_next_oracle_successor/test_m1497_source.py"
M1497_TB = HW / "verif_m1497_c1_active_next_oracle_successor/tb_m1497_m1270r13_m1162_real_m935_protocol_unit_delay.sv"
M1497_FILELIST = HW / "verif_m1497_c1_active_next_oracle_successor/m1497_unit_delay_filelist.f"
R13 = HW / "verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
WITNESS = HW / "verif_m1337r15_c1_real_m935_runtime_witness/m1337r15_m935_runtime_witness.sv"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1498_FAIL = HW / "reviews/m1498_m1497_c1_active_next_oracle_source_blind_hammer_r1_20260831"
ATTEMPT = HW / "results/.m1506_c1_active_next_oracle_vcs_attempt_consumed"
RESULT = HW / "results/m1506_c1_active_next_oracle_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")

SOURCE_STATUS = "M1506_C1_ACTIVE_NEXT_ORACLE_RELEASE_SAFE_SOURCE_READY__NO_LAUNCH"
AUTHOR_STATUS = "PASS_M1506_C1_ACTIVE_NEXT_ORACLE_RELEASE_SAFE_SOURCE__NO_VCS_NO_EDA"
HAMMER_STATUS = "PASS_M1507_M1506_C1_ACTIVE_NEXT_ORACLE_RELEASE_SAFE_SOURCE__RELEASE_NOT_AUTHORED"
RELEASE_STATUS = "AUTHORIZE_ONE_M1506_C1_ACTIVE_NEXT_ORACLE_UNIT_DELAY_VCS_ATTEMPT"
FINAL_STATUS = "PASS_M1509_AUTHORIZE_ONE_M1506_C1_ACTIVE_NEXT_ORACLE_VCS_LAUNCH"
CLAIMS = {"source_only": True, "functional_vcs": False,
          "timing_verified": False, "cycles_measured": False,
          "speedup": False, "ppa": False, "power": False, "energy": False,
          "system_speedup": False, "headline": False}
M1497_PINS = {
    "runner_sha256": "db9154c1e8ab88afc209fefd39123ad812b2f6eeb566c031e7f1824d15ead708",
    "checker_sha256": "d29fd7c1fafa92ed572214b4ee2441bd5ec752adfb223f91632dd382489e74c0",
    "tests_sha256": "f15b007327e1394362ec818d48ee191656728c8fbebce75cd31c7b9dc2159110",
    "testbench_sha256": "e5604300f3e6cfcbdadfdafa8fae6a2faa6cdc1c18446fa8c48ba6ea10632526",
    "filelist_sha256": "de51bfdc95227ff7f8fbe2178465f1d088b9285067d9ed30770b357116a75e51",
    "frozen_r13_sha256": "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    "source_contract_sha256": "c3531c4c8d55046cf7f5eee5717a3ed5a3a6c475cbbb115e8c84a4a80e308375",
    "author_review_sha256": "aa7689c0401fe212d006b5e0b32b3cdad2237c6f53c6054c197295f26ad55919",
    "author_manifest_sha256": "570101c7002765d75ead34e084412050c0418eb2d10c26f8f014ea7435d6cae0",
    "author_outer_file_sha256": "02ec9968b4e0edf719cd8793091b4ac096f0d96d8c651db9c3b86affb9e9db46"
}
M1498_PINS = {
    "status": "FAIL_DO_NOT_CITE__M1497_ADDITIVE_SUCCESSOR_REQUIRED__NO_M1499",
    "review_sha256": "806cd6f629d17076e7f8bc1df0a633fb6d0a9cd68cf762d8f167123d3c7913b8",
    "manifest_sha256": "df0b581860be722c7c2e49bde4878dee317f72a5097d2b6e6c4e5c1861ddd300",
    "outer_file_sha256": "0e1d91e0dd700390abf78df87ab5a53fc3187eea1e4d53a8310ae77961eac2d4"
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_json(path: Path) -> dict[str, Any]:
    require(path.exists() and not path.is_symlink()
            and stat.S_ISREG(path.lstat().st_mode), "JSON not regular")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token:
                      (_ for _ in ()).throw(RuntimeError(token)))


def verify_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.read_text().split() == [sha(path), path.name],
            "contract sidecar")
    require(outer.read_text().split() == [sha(sidecar), sidecar.name],
            "contract outer")


def rel(path: Path) -> str:
    return path.relative_to(HW).as_posix()


def expected_contract() -> dict[str, Any]:
    return {
        "schema": "m1506_c1_active_next_oracle_release_safe_successor_source_contract_r1_v1",
        "status": SOURCE_STATUS,
        "scope": "Additive source only. M1497 TB/oracle and raw/clean isolation are unchanged. M1506 closes the exact-contract, frozen-input, log-admission, and post-attempt quarantine failures found by M1498. No VCS, simv, or EDA was invoked.",
        "identity": {
            "runner_path": rel(RUNNER), "runner_sha256": sha(RUNNER),
            "checker_path": rel(CHECKER), "checker_sha256": sha(CHECKER),
            "tests_path": rel(TESTS), "tests_sha256": sha(TESTS),
            "testbench_path": rel(M1497_TB), "testbench_sha256": sha(M1497_TB),
            "filelist_path": rel(M1497_FILELIST), "filelist_sha256": sha(M1497_FILELIST),
            "parent_rtl_path": rel(PARENT), "parent_rtl_sha256": sha(PARENT),
            "m935_rtl_path": rel(M935), "m935_rtl_sha256": sha(M935),
            "wrapper_rtl_path": rel(WRAPPER), "wrapper_rtl_sha256": sha(WRAPPER),
            "sva_path": rel(SVA), "sva_sha256": sha(SVA),
            "witness_path": rel(WITNESS), "witness_sha256": sha(WITNESS),
            "foundry_model_path": str(FOUNDRY),
            "foundry_model_sha256": sha(FOUNDRY),
            "vcs_binary_path": str(VCS), "vcs_binary_sha256": sha(VCS),
            "docs359_path": rel(DOCS359), "docs359_sha256": sha(DOCS359)
        },
        "predecessors": {
            "m1497": M1497_PINS,
            "m1498_failure": M1498_PINS
        },
        "oracle_contract": {
            "m1497_tb_oracle_byte_preserved": True,
            "active_next_weight_accepted": 0,
            "active_next_psum_accepted_equals_not_latched_first": True,
            "latched_first_equals_public_first": True,
            "latched_source_equals_public_source": True,
            "latched_source_differs_from_served_source": True,
            "unknown_active_first_source_or_accept_fails_closed": True
        },
        "result_hygiene": {
            "raw_build_separate_from_clean_result_stage": True,
            "raw_build_is_never_sealed_or_published": True,
            "clean_payload_regular_only": True,
            "clean_payload_members": [
                "compile.log", "sim.log",
                "m1506_c1_active_next_oracle_identity_r1.json",
                "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json"
            ],
            "symlink_rejection_relaxed": False,
            "recursive_manifest_and_outer_seal_required": True
        },
        "log_admission": {
            "exact_r13_pass_tokens": 1,
            "exact_r15_pass_tokens": 1,
            "weight_requests": 2,
            "psum_requests": 1,
            "responses": 2,
            "core_accepts": 2,
            "psum_commits": 1,
            "row_completions": 1,
            "task_completions": 1,
            "cp_nonfirst_min_matches": 1,
            "cp_ii2_min_matches": 1,
            "minimum_oracle_pass_records": 80,
            "assertion_failures": 0,
            "design_faults": 0,
            "error_fatal_assertion_failure_lines": 0
        },
        "failure_quarantine": {
            "all_post_attempt_operations_inside_guard": True,
            "raw_build_create_inside_guard": True,
            "post_attempt_failure_recursive_quarantine": True,
            "failure_claim_never_functional_vcs": True,
            "automatic_retry": False
        },
        "future_authority": {
            "source_hammer_path": rel(HAMMER),
            "source_hammer_required_status": HAMMER_STATUS,
            "launch_release_path": rel(RELEASE),
            "launch_release_required_status": RELEASE_STATUS,
            "final_launch_hammer_path": rel(FINAL),
            "final_launch_hammer_required_status": FINAL_STATUS,
            "maximum_future_vcs_compiles": 1,
            "maximum_future_simv_runs": 1,
            "automatic_retry": False
        },
        "authorization": {
            "vcs_compiles": 0, "simv_runs": 0,
            "all_other_eda_runs": 0, "automatic_retry": False
        },
        "author_execution": {
            "python_static_tests": True, "vcs": False, "simv": False,
            "eda": False, "ssh": False, "gpu": False
        },
        "claim_boundary": CLAIMS
    }


def validate_contract(value: Any, expected: dict[str, Any] | None = None) -> None:
    require(type(value) is dict, "contract must be object")
    canonical = expected_contract() if expected is None else expected
    require(value == canonical, "contract exact set/value drift")


def check_post_attempt_try_structure() -> None:
    tree = ast.parse(RUNNER.read_text())
    main_node = next(node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == "main")
    try_nodes = [node for node in main_node.body if isinstance(node, ast.Try)]
    require(len(try_nodes) == 1, "main post-attempt guard cardinality")
    guarded = ast.unparse(try_nodes[0])
    required = (
        "publish_no_replace(ATTEMPT_STAGE, ATTEMPT)",
        "RAW_BUILD.mkdir()", "run_tool(COMPILE_COMMAND",
        "run_tool(SIM_COMMAND", "validate_sim_log(",
        "publish_no_replace(CLEAN_RESULT_STAGE, RESULT)",
        "make_clean_evidence(FAILURE_STAGE", "publish_no_replace(FAILURE_STAGE, QUARANTINE)"
    )
    require(all(token in guarded for token in required),
            "post-attempt operation escaped guard")


def check_source(require_runtime_authority: bool = False) -> dict[str, Any]:
    verify_sidecar(CONTRACT)
    actual = strict_json(CONTRACT)
    validate_contract(actual)
    require(sha(M1497_RUNNER) == M1497_PINS["runner_sha256"], "M1497 runner drift")
    require(sha(M1497_CHECKER) == M1497_PINS["checker_sha256"], "M1497 checker drift")
    require(sha(M1497_TESTS) == M1497_PINS["tests_sha256"], "M1497 tests drift")
    require(sha(M1497_TB) == M1497_PINS["testbench_sha256"], "M1497 TB drift")
    require(sha(M1497_FILELIST) == M1497_PINS["filelist_sha256"], "M1497 filelist drift")
    require(sha(R13) == M1497_PINS["frozen_r13_sha256"], "R13 drift")
    require(sha(M1498_FAIL / "review.json") == M1498_PINS["review_sha256"],
            "M1498 failure review drift")
    require(sha(M1498_FAIL / "SHA256SUMS") == M1498_PINS["manifest_sha256"],
            "M1498 failure manifest drift")
    require(sha(M1498_FAIL / "SHA256SUMS.seal.sha256") ==
            M1498_PINS["outer_file_sha256"], "M1498 failure outer drift")
    runner_text = RUNNER.read_text()
    require("for path, digest in BASE.EXACT.items()" in runner_text,
            "runtime does not iterate frozen BASE.EXACT")
    require("validate_sim_log" in runner_text and "FORBIDDEN_LOG_RE" in runner_text,
            "strict runtime log admission absent")
    check_post_attempt_try_structure()
    require(not any(path.exists() for path in (ATTEMPT, RESULT, QUARANTINE)),
            "M1506 canonical namespace not fresh")
    if require_runtime_authority:
        require(AUTHOR.is_dir() and HAMMER.is_dir()
                and RELEASE.is_file() and FINAL.is_dir(),
                "runtime authority incomplete")
    return {
        "schema": "m1506_c1_active_next_oracle_release_safe_source_check_r1_v1",
        "status": AUTHOR_STATUS,
        "bindings": {
            "runner_sha256": sha(RUNNER), "checker_sha256": sha(CHECKER),
            "tests_sha256": sha(TESTS), "contract_sha256": sha(CONTRACT),
            "m1497_runner_sha256": M1497_PINS["runner_sha256"],
            "m1498_failure_review_sha256": M1498_PINS["review_sha256"]
        },
        "proofs": {
            "contract_exact_set_value": True,
            "runtime_exact_frozen_inputs": True,
            "strict_log_admission": True,
            "post_attempt_failure_guard": True
        },
        "claim_boundary": CLAIMS
    }


def main() -> int:
    if sys.argv[1:] not in (["--mode", "source_only"],
                            ["--mode", "runtime_present"]):
        raise SystemExit("usage: check_m1506_source.py --mode source_only|runtime_present")
    print(json.dumps(check_source(sys.argv[-1] == "runtime_present"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
