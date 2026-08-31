#!/usr/bin/env python3
"""Fail-closed checker for source-only M1074; never advances M1072."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ENGINE = HERE / "execute_m1074_m1072_c1_full_exact_1rw_one_shot.py"
RUNNER = HERE / "run_m1074_m1072_c1_full_exact_1rw_one_shot.sh"
TEST = HW / "system_simulator/tests/test_m1074_c1_full_exact_1rw_one_shot_source.py"
ENGINE_SHA = "90ead8cb4a0196114dbb6c51f4fe9e042fee1bf2816855687327221c8c3274e5"
RUNNER_SHA = "cec9da5f0faaef281c705f46b41020fe6572be0f98317f6f8ab29f5e1a090812"
TEST_SHA = "6ad691f33962500bd1fd35aaf71040359dae95100384544fc63f7d726d526f4b"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_engine():
    require(sha256(ENGINE) == ENGINE_SHA and sha256(RUNNER) == RUNNER_SHA and
            sha256(TEST) == TEST_SHA, "M1074 engine/runner/test identity drift")
    spec = importlib.util.spec_from_file_location("m1074_checked_engine", ENGINE)
    require(spec is not None and spec.loader is not None, "cannot load M1074")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> dict[str, Any]:
    module = load_engine()
    source = module.validate_source_contract(RUNNER, require_fresh=True)
    oracle = module.source_self_test()
    require(oracle.get("status") ==
            "PASS_M1074_SOURCE_SELF_TEST__NO_FULL_REPLAY_NO_ATTEMPT" and
            oracle.get("m1072_generator_advanced") is False and
            oracle.get("canonical_rows_opened_or_hashed") is False and
            oracle.get("attempt_consumed") is False,
            "M1074 source self-test drift")

    function_source = inspect.getsource(module.execute_full)
    calls = [node for node in ast.walk(ast.parse(function_source))
             if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and
             node.func.attr == "iter_canonical_full_replay_results"]
    require(len(calls) == 1 and calls[0].args == [] and calls[0].keywords == [] and
            "403922" not in function_source,
            "M1074 unique M1072 zero-argument call drift")

    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            text=True, capture_output=True, check=False)
    require(syntax.returncode == 0, "M1074 runner shell syntax failure")
    tests = subprocess.run([sys.executable, str(TEST)], text=True,
                           capture_output=True, check=False)
    require(tests.returncode == 0 and "Ran 15 tests" in tests.stderr and
            "OK" in tests.stderr, "M1074 directed tests failed")

    results = HW / "results"
    forbidden = [module.RESULT, module.ATTEMPT]
    require(not any(path.exists() for path in forbidden) and
            not any(results.glob(module.WORK_PREFIX + "*")) and
            not any(results.glob(module.FAILURE_PREFIX + "*")),
            "M1074 run namespace exists")
    contract = module.strict_json(module.CONTRACT)
    require(contract["launch_now"] is False and
            contract["max_attempts_now"] == 0 and
            contract["claim_boundary"]["attempt_consumed"] is False and
            contract["claim_boundary"]["full_51840000_replay_executed"] is False,
            "M1074 claim boundary drift")
    return {
        "schema": "m1074_c1_full_exact_1rw_one_shot_source_check_v1",
        "status": "PASS_M1074_SOURCE_CHECK__M1075_REQUIRED_NO_LAUNCH",
        "engine_sha256": sha256(ENGINE),
        "runner_sha256": sha256(RUNNER),
        "tests_sha256": sha256(TEST),
        "contract_sha256": module.CONTRACT_SHA,
        "contract_sidecar_sha256": module.CONTRACT_SIDECAR_SHA,
        "contract_outer_seal_file_sha256": sha256(module.CONTRACT_OUTER),
        "m1072_source_sha256": module.M1072_SHA,
        "m1073_outer_seal_file_sha256": module.M1073_ID[2],
        "directed_tests": 15,
        "unique_zero_argument_m1072_cycle_call": True,
        "pre_attempt_rows_opened_or_hashed": False,
        "atomic_attempt_before_rows": True,
        "atomic_complete_result_seal": True,
        "sealed_failure_quarantine": True,
        "caller_cycle_capacity_coverage": False,
        "simple_raw_plus_403922": False,
        "attempt_consumed": False,
        "full_replay_executed": False,
        "eda_gpu_remote_used": False,
        "speedup_admitted": False,
        "rtl_cycles": False,
        "paper_ppa_ready": False,
        "docs359_sha256": module.DOCS359_SHA,
        "source": source,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
