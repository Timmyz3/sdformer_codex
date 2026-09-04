#!/usr/bin/env python3
"""Fail-closed checker for source-only M1064; never invokes the full iterator."""
from __future__ import annotations

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
SOURCE = HERE / "run_m1064_c1_frozen_exact_1rw_replay_source.py"
TEST = HW / "system_simulator/tests/test_m1064_c1_frozen_exact_1rw_replay_source.py"
SOURCE_SHA = "ecf2625ae60a9f7848fc32b852b67f8efd3439c5fb24b9904ef397d39aafed09"
TEST_SHA = "0956d82a8510a2307970b161f240df394d1bbd3e268f9f519997d7a205af864e"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    require(sha256(SOURCE) == SOURCE_SHA and sha256(TEST) == TEST_SHA,
            "M1064 source/test identity drift")
    spec = importlib.util.spec_from_file_location("m1064_checked_source", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1064")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> dict[str, Any]:
    module = load_source()
    contract = module.validate_sealed_contract()
    authorities = module.validate_frozen_authorities(hash_rows=True)
    oracle = module.small_oracle()
    require(oracle.get("status") ==
            "PASS_M1064_SMALL_ORACLE__M1065_REQUIRED_NO_FULL_REPLAY" and
            all(oracle.get("m1057_attacks_rejected", {}).values()) and
            oracle.get("full_iterator_called") is False,
            "M1064 small oracle drift")
    require(len(inspect.signature(module.derive_physical_capacity).parameters) == 0 and
            len(inspect.signature(module.iter_frozen_task_records).parameters) == 0 and
            len(inspect.signature(module.FrozenCoverage).parameters) == 0 and
            list(inspect.signature(module.replay_frozen_sample).parameters) ==
            ["records"],
            "caller-controlled production coordinate appeared")
    capacity = module.derive_physical_capacity()
    require(capacity["derived_total_bytes"] == 214_912 and
            capacity["psum"]["bytes"] == 122_880 and
            capacity["weight"]["bytes"] == 49_152 and
            capacity["parent_plus_other"]["bytes"] == 42_880 and
            capacity["capacity_bytes_pass"] is True and
            capacity["capacity_only_214912B_admitted"] is False,
            "physical capacity derivation drift")
    tests = subprocess.run(
        [sys.executable, str(TEST)], text=True, capture_output=True, check=False
    )
    require(tests.returncode == 0 and "Ran 15 tests" in tests.stderr and
            "OK" in tests.stderr, "M1064 directed tests failed")
    forbidden = [
        HW / "results/m1064_c1_frozen_exact_1rw_full_replay_r1_20260830",
        HW / "results/.m1064_c1_frozen_exact_1rw_full_replay_attempt_consumed",
        HW / "results/m1066_m1064_c1_frozen_exact_1rw_full_replay_r1_20260830",
        HW / "results/.m1066_m1064_c1_frozen_exact_1rw_full_replay_attempt_consumed",
    ]
    require(not any(path.exists() for path in forbidden),
            "M1064/M1066 full replay namespace exists")
    return {
        "schema": "m1064_c1_frozen_exact_1rw_source_check_v1",
        "status": "PASS_M1064_SOURCE_CHECK__M1065_REQUIRED_NO_FULL_REPLAY",
        "source_sha256": sha256(SOURCE),
        "tests_sha256": sha256(TEST),
        "contract": contract,
        "authorities": authorities,
        "directed_tests": 15,
        "all_m1057_attacks_rejected": True,
        "capacity_derived_internally": capacity,
        "full_iterator_called": False,
        "full_replay_executed": False,
        "eda_gpu_remote_used": False,
        "capacity_only_214912B_admitted": False,
        "matched_cycles_admitted": False,
        "speedup_admitted": False,
        "paper_ppa_ready": False,
        "docs359_sha256": module.DOCS359_SHA,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
