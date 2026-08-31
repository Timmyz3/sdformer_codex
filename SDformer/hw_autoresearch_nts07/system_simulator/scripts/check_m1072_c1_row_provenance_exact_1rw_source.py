#!/usr/bin/env python3
"""Fail-closed checker for source-only M1072; never starts the full iterator."""
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
SOURCE = HERE / "run_m1072_c1_row_provenance_exact_1rw_source.py"
TEST = HW / "system_simulator/tests/test_m1072_c1_row_provenance_exact_1rw_source.py"
SOURCE_SHA = "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c"
TEST_SHA = "051192f46e6fdd2d4803a44b56e556b8e2b54e409e30b07629d1820435707820"


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
            "M1072 source/test identity drift")
    spec = importlib.util.spec_from_file_location("m1072_checked_source", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1072")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> dict[str, Any]:
    module = load_source()
    contract = module.validate_sealed_contract()
    authorities = module.validate_frozen_authorities()
    oracle = module.small_oracle()
    require(oracle.get("status") ==
            "PASS_M1072_SMALL_ORACLE__M1073_REQUIRED_NO_FULL_REPLAY" and
            all(oracle.get("m1065_attacks_rejected", {}).values()) and
            oracle.get("file_identity_before_and_after_first_reads") is True and
            oracle.get("full_iterator_called") is False,
            "M1072 small oracle drift")

    production = module.iter_canonical_full_replay_results
    require(len(inspect.signature(production).parameters) == 0 and
            inspect.isgeneratorfunction(production) and
            len(inspect.signature(module.CanonicalRowReader).parameters) == 0 and
            len(inspect.signature(module.ProvenanceCoverage).parameters) == 0 and
            list(inspect.signature(
                module.validate_external_records_against_frozen).parameters) ==
            ["records"],
            "caller-controlled production coordinate appeared")

    capacity = module.M1064.derive_physical_capacity()
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
            "OK" in tests.stderr, "M1072 directed tests failed")

    forbidden = [
        HW / "results/m1072_c1_row_provenance_exact_1rw_full_replay_r1_20260830",
        HW / "results/.m1072_c1_row_provenance_exact_1rw_full_replay_attempt_consumed",
        HW / "results/m1074_m1072_c1_row_provenance_exact_1rw_full_replay_r1_20260830",
        HW / "results/.m1074_m1072_c1_row_provenance_exact_1rw_full_replay_attempt_consumed",
    ]
    require(not any(path.exists() for path in forbidden),
            "M1072/M1074 full replay namespace exists")
    require(module.sha256(module.DOCS359) == module.DOCS359_SHA,
            "docs/359 identity drift")

    return {
        "schema": "m1072_c1_row_provenance_exact_1rw_source_check_v1",
        "status": "PASS_M1072_SOURCE_CHECK__M1073_REQUIRED_NO_FULL_REPLAY",
        "source_sha256": sha256(SOURCE),
        "tests_sha256": sha256(TEST),
        "contract": contract,
        "authorities": authorities,
        "directed_tests": 15,
        "m1065_work_preprocess_forgery_rejected": True,
        "m1065_all_zero_mask_rejected": True,
        "row_reorder_rejected": True,
        "short_pread_rejected": True,
        "file_drift_rejected": True,
        "caller_cycle_fields_accepted": False,
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
