#!/usr/bin/env python3
"""Fail-closed source checker for M1056; never executes a full replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "run_m1056_c1_exact_1rw_arbitration_replay_source.py"
TEST = HW / "system_simulator/tests/test_m1056_c1_exact_1rw_arbitration_replay_source.py"
CONTRACT = HW / "contracts/m1056_m1051_c1_exact_1rw_arbitration_replay_source_contract_r1_20260829.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def load_source():
    spec = importlib.util.spec_from_file_location("m1056_checked_source", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> dict[str, Any]:
    contract = strict_json(CONTRACT)
    require(contract.get("status") ==
            "PASS_M1056_SOURCE_ONLY__M1057_REQUIRED_NO_FULL_REPLAY" and
            contract.get("launch_now") is False and
            contract.get("max_attempts_now") == 0,
            "contract authority drift")
    identities = contract.get("source_identity", {})
    for key, path in (("source", SOURCE), ("checker", Path(__file__).resolve()),
                      ("tests", TEST)):
        require(identities.get(key, {}).get("path") ==
                path.relative_to(HW).as_posix() and
                identities.get(key, {}).get("sha256") == sha256(path),
                key + " identity drift")
    require(sha256(DOCS359) == DOCS359_SHA and
            contract.get("docs359_sha256") == DOCS359_SHA,
            "docs359 drift")
    module = load_source()
    oracle = module.small_oracle()
    require(oracle.get("status") ==
            "PASS_M1056_SMALL_ORACLE__NO_FULL_REPLAY_NO_EDA" and
            oracle.get("naive_cycles_plus_conflicts_rejected") is True and
            oracle.get("capacity_and_port_gates_separate") is True,
            "small oracle drift")
    test_run = subprocess.run(
        [sys.executable, str(TEST)], text=True, capture_output=True, check=False
    )
    require(test_run.returncode == 0 and "OK" in test_run.stderr,
            "directed tests failed: " + test_run.stdout + test_run.stderr)
    forbidden = [
        HW / "results/m1056_c1_exact_1rw_arbitration_full_replay_r1_20260829",
        HW / "results/.m1056_c1_exact_1rw_arbitration_full_replay_attempt_consumed",
    ]
    require(not any(path.exists() for path in forbidden),
            "source milestone polluted by full replay namespace")
    return {
        "schema": "m1056_c1_exact_1rw_arbitration_source_check_v1",
        "status": "PASS_M1056_SOURCE_CHECK__M1057_REQUIRED_NO_FULL_REPLAY",
        "directed_tests": 13,
        "small_oracle_pass": True,
        "full_replay_executed": False,
        "eda_gpu_remote_used": False,
        "capacity_only_214912B_admitted": False,
        "matched_cycles_admitted": False,
        "speedup_admitted": False,
        "paper_ppa_ready": False,
        "docs359_sha256": sha256(DOCS359),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
