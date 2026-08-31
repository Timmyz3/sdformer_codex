#!/usr/bin/env python3
"""Static-only checker for the M972 additive evidence-safe source package."""

import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE / "execute_m972_m971_decoder_d2d3_10k_evidence_safe_r1.py"
RUNNER = HERE / "run_m972_m971_decoder_d2d3_10k_evidence_safe_r1_one_shot.sh"
TEST = HW / "system_simulator/tests/test_m972_m971_decoder_d2d3_10k_evidence_safe_source.py"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_driver():
    spec = importlib.util.spec_from_file_location("m972_static_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M972 driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check(contract: Path) -> dict:
    if (Path(sys.executable).resolve() != PYTHON or sha256(PYTHON) != PYTHON_SHA or
            tuple(sys.version_info[:3]) != (3, 10, 18)):
        raise RuntimeError("M972 checker interpreter drift")
    driver_text = DRIVER.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(driver_text, filename=str(DRIVER))
    imported = {alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    forbidden = sorted(imported.intersection({
        "subprocess", "socket", "requests", "urllib", "paramiko",
        "torch", "tensorflow", "cupy"}))
    if forbidden:
        raise RuntimeError("M972 forbidden import: " + ",".join(forbidden))
    if "SOURCE_FETCH_REQUESTS" in driver_text or "SOURCE_FETCH_ONLY" in driver_text:
        raise RuntimeError("M972 revived M961 byte/request or scope preassumption")
    if ('int(exact["compressed_transaction_count"]) == 1' in driver_text or
            "compressed == 1" in driver_text):
        raise RuntimeError("M972 revived one-transaction preassumption")
    required = (
        "M972_EXPECTED_RELEASE_SHA256", "--consume-attempt",
        "M972_WORK_ROOT_CREATED_BEFORE_D2", "--run-row D2",
        "--run-row D3", "--seal-failure-root", "--kill-after=30s 1800s",
    )
    missing = [token for token in required if token not in runner_text]
    if missing:
        raise RuntimeError("M972 runner safety token missing: " +
                           ",".join(missing))
    if runner_text.index("--run-row D2") >= runner_text.index("--run-row D3"):
        raise RuntimeError("M972 runner does not seal D2 before D3")
    if "100000" in runner_text or "--run-full" in runner_text:
        raise RuntimeError("M972 runner contains unauthorized expanded mode")
    driver = load_driver()
    validation = driver.validate_source_contract(contract, RUNNER)
    self_test = driver.source_self_test()
    return {
        "schema": "m972_evidence_safe_source_static_check_v1",
        "status": "PASS_M972_STATIC_SOURCE_CHECK__NO_REAL_10K",
        "driver_sha256": sha256(DRIVER), "runner_sha256": sha256(RUNNER),
        "test_sha256": sha256(TEST), "contract_sha256": sha256(contract),
        "forbidden_imports": forbidden,
        "byte_request_distinction": self_test["geometry"],
        "multi_transaction_accepted": self_test["multi_transaction_accepted"],
        "commit_coverage_accepted": self_test["commit_coverage_accepted"],
        "failure_trace_persisted_and_double_sealed":
            self_test["failure_trace_persisted_and_double_sealed"],
        "source_contract_status": validation["status"],
        "real_prefix_executed": False, "full_row_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.contract), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
