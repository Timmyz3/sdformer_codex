#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA source checker for the M1502 C2 SOURCE_CHAIN successor."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / (
    "dc_handoff/scripts/run_m1502_m1493_c2_source_chain_successor_"
    "one_shot.py")
CHECKER = Path(__file__).resolve()
TESTS = CHECKER.parent / "test_m1502_c2_source_chain_successor_source.py"
CONTRACT = HW / (
    "contracts/m1502_m1493_c2_source_chain_successor_source_contract_"
    "r1_20260831.json")
FUTURE = (
    HW / "reviews/m1503_m1502_c2_source_chain_successor_source_blind_hammer_r1_20260831",
    HW / "contracts/m1504_m1503_m1502_c2_source_chain_successor_launch_release_r1_20260831.json",
    HW / "reviews/m1505_m1504_m1502_c2_source_chain_successor_final_launch_hammer_r1_20260831",
)
NEW_NAMESPACES = {
    "attempt": "results/.m1502_c2_mapped_vcs_saif_ptpx_attempt_consumed",
    "result": "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831",
    "failure": "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine",
    "private": "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite",
}
OLD_NAMESPACES = {
    "attempt": "results/.m1493_c2_mapped_vcs_saif_ptpx_attempt_consumed",
    "result": "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831",
    "failure": "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine",
    "private": "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite",
}
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
OLD_RUNNER_SHA = "8d93d55ca600620eb903a7328f4cc38e0720ae45ce24d8128fac5924d2902677"
OLD_FAILURE_SHA = {
    "payload": "43497b8701400b6c7c5d3f0cc29a2a41955a135fff4be6720968cbeb736cc5e7",
    "manifest": "53e77670cd0f07ea457dc35f041e3885f7d73b304149c8d52e116fd06d6a5f88",
    "outer": "8cb2e41374f9b827c118b949e1a37b66baeec5bef578d81ee68a0d95a90d4a7e",
}
M1494_REVIEW_SHA = "65435aca804c486d50d8332774c70e87083d66d5c2e7acc30485dc84ba458340"
M1495_SHA = "838ea0f3714167c43c6f4e40829c2d1a59d1b84ee7468758798c82f21114eb94"
M1496_REVIEW_SHA = "ef0af9fbf0ab094f40052de8fc552b7b97e2519dd5db88c6f3c2bf7505acb810"
CLAIMS = {key: False for key in (
    "functional_vcs_verified", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready",
    "headline")}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("m1502_source_bound_runner", RUNNER)
    need(spec is not None and spec.loader is not None, "runner import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = load_runner()


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


def check_predecessor() -> dict[str, Any]:
    R.verify_predecessor_failure()
    need(sha(R.OLD_RUNNER) == OLD_RUNNER_SHA, "M1493 runner drift")
    need(sha(R.OLD_FAILURE / "failure.json") == OLD_FAILURE_SHA["payload"],
         "M1493 failure payload drift")
    need(sha(R.M1494 / "review.json") == M1494_REVIEW_SHA, "M1494 drift")
    need(sha(R.M1495) == M1495_SHA, "M1495 drift")
    need(sha(R.M1496 / "review.json") == M1496_REVIEW_SHA, "M1496 drift")
    need(not os.path.lexists(HW / OLD_NAMESPACES["attempt"]),
         "M1493 attempt unexpectedly exists")
    need(not os.path.lexists(HW / OLD_NAMESPACES["result"]),
         "M1493 result unexpectedly exists")
    need(not os.path.lexists(HW / OLD_NAMESPACES["private"]),
         "M1493 private unexpectedly exists")
    need(os.path.isdir(HW / OLD_NAMESPACES["failure"]),
         "M1493 sealed failure absent")
    return {"phase": "SOURCE_CHAIN", "error": "AttributeError",
            "attempt_consumed": False, "vcs_compiles": 0,
            "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0,
            "automatic_retry": False, "partial_axis_citable": False}


def check_corrected_callpath() -> dict[str, Any]:
    saved = {name: os.environ.pop(name, None) for name in R.ENV_PINS}
    try:
        try:
            R.verify_frozen_execution_inputs()
        except BaseException as error:
            need(type(error) is R.Failure, "corrected callpath wrong exception type")
            need(str(error) ==
                 "M1502 authority absent: required exact SHA environment",
                 "corrected callpath did not reach only future authority")
        else:
            raise RuntimeError("corrected callpath unexpectedly passed")
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    return {"called": "verify_frozen_execution_inputs",
            "attribute_error": False,
            "terminal": "M1502_FUTURE_AUTHORITY_ONLY"}


def check_execution_text(text: str) -> None:
    need("EXEC.verify_predecessor_failure()" not in text,
         "invalid predecessor method call present")
    need(R.COMPILE_PREFIX == R.OLD.COMPILE_PREFIX,
         "compile prefix object drift")
    need(R.COMPILE_PREFIX[-4:] ==
         ["-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc"],
         "compile flags drift")
    need(text.count('for axis in ("k8", "k1x8"):') == 4,
         "axis loops drift")
    need(text.count("for case in range(5):") == 2, "case loops drift")
    for token in ('state["vcs_compiles"] += 1',
                  'state["simv_runs"] += 1',
                  'state["saif_files"] += 1',
                  'state["ptpx_runs"] += 1'):
        need(text.count(token) == 1, "counter site drift: " + token)
    need(text.index("ATTEMPT.mkdir()") < text.index("EXEC.run(command"),
         "attempt after first EDA")
    need(text.index(
        'if any(state[key] != COUNTS[key] for key in\n'
        '               ("vcs_compiles", "simv_runs", "saif_files")):') <
        text.index('state["phase"] = f"PTPX_{axis}_{case}"'),
        "PTPX before all-SAIF gate")
    need('"automatic_retry": False' in text
         and '"partial_axis_citable": False' in text,
         "fail-close boundary drift")
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    target = functions.get("verify_frozen_execution_inputs")
    need(target is not None, "frozen-input function absent")
    for node in ast.walk(target):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            need(node.func.attr != "verify_predecessor_failure",
                 "invalid predecessor attribute call in corrected function")


def expected_contract() -> dict[str, Any]:
    return {
        "schema": "m1502_m1493_c2_source_chain_successor_source_contract_r1_v1",
        "status":
            "M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE_READY__FRESH_M1503_REQUIRED__NO_EDA",
        "date": "2026-08-31",
        "purpose": "Additive successor to the sealed pre-attempt M1493 SOURCE_CHAIN failure. Delete only the invalid predecessor-module verify_predecessor_failure call; preserve the exact per-file execution-stack pins and the complete mapped VCS-SAIF-PTPX campaign.",
        "identity": {
            "runner_path": RUNNER.relative_to(HW).as_posix(),
            "runner_sha256": sha(RUNNER),
            "checker_path": CHECKER.relative_to(HW).as_posix(),
            "checker_sha256": sha(CHECKER),
            "tests_path": TESTS.relative_to(HW).as_posix(),
            "tests_sha256": sha(TESTS),
            "m1493_runner_sha256": OLD_RUNNER_SHA,
            "m1494_review_sha256": M1494_REVIEW_SHA,
            "m1495_release_sha256": M1495_SHA,
            "m1496_review_sha256": M1496_REVIEW_SHA,
        },
        "predecessor_failure": {
            "payload_sha256": OLD_FAILURE_SHA["payload"],
            "manifest_sha256": OLD_FAILURE_SHA["manifest"],
            "outer_file_sha256": OLD_FAILURE_SHA["outer"],
            "phase": "SOURCE_CHAIN", "error": "AttributeError",
            "counts": {"vcs_compiles": 0, "simv_runs": 0,
                       "saif_files": 0, "ptpx_runs": 0},
            "attempt_consumed": False, "automatic_retry": False,
            "canonical_result": False, "partial_axis_citable": False,
        },
        "sole_repair": {
            "delete_invalid_call":
                "predecessor_module.verify_predecessor_failure",
            "per_file_exact_pin_loop_preserved": True,
            "compile_prefix_preserved_exactly": True,
            "debug_access_r_preserved": True, "lca_preserved": True,
            "rtl_change": False, "netlist_change": False,
            "sdc_change": False, "testbench_change": False,
            "workload_change": False, "ucli_change": False,
            "ptpx_script_change": False, "saif_scope_change": False,
        },
        "corrected_callpath_test": {
            "real_function_called": "verify_frozen_execution_inputs",
            "future_authority_absent": True,
            "expected_terminal":
                "M1502 authority absent: required exact SHA environment",
            "attribute_error": False,
        },
        "preserved_execution": {
            "axes": ["k8", "k1x8"], "cases": [0, 1, 2, 3, 4],
            "vcs_compiles": 2, "simv_runs": 10,
            "production_saif_files": 10, "ptpx_runs": 10,
            "ptpx_after_all_ten_saif_gates": True,
            "attempt_consumed_before_first_eda_tool": True,
            "automatic_retry": False, "partial_axis_publication": False,
            "fresh_namespaces": dict(NEW_NAMESPACES),
        },
        "future_authority": {
            "source_hammer": FUTURE[0].relative_to(HW).as_posix(),
            "launch_release": FUTURE[1].relative_to(HW).as_posix(),
            "final_hammer": FUTURE[2].relative_to(HW).as_posix(),
            "fresh_different_author_required": True,
            "launch_authorized": False,
        },
        "author_execution": {
            "source_authoring": True, "source_only_tests": True,
            "license_query": False, "vcs": False, "simv": False,
            "saif": False, "pt": False, "ptpx": False, "eda": False,
            "attempt_consumed": False, "launch": False,
        },
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
    callpath = check_corrected_callpath()
    check_execution_text(RUNNER.read_text())
    contract = check_contract()
    for rel in NEW_NAMESPACES.values():
        need(not os.path.lexists(HW / rel), "M1502 namespace residue")
    if require_future_absent:
        need(not any(os.path.lexists(path) for path in FUTURE),
             "future authority exists")
    return {"schema": "m1502_c2_source_chain_successor_source_check_r1_v1",
            "status": "PASS_M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE__NO_EDA",
            "predecessor": predecessor, "corrected_callpath": callpath,
            "contract_status": contract["status"],
            "sole_delta": "delete_invalid_source_chain_method_call",
            "claim_boundary": CLAIMS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    parser.parse_args()
    print(json.dumps(check_source(True), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
