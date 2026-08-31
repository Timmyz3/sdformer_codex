#!/usr/bin/env python3
"""Static checker for the source-only M961 D2/D3 10K workflow."""

import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE / "execute_m961_m946_decoder_d2d3_10k_bounded_prefix_r1.py"
RUNNER = HERE / "run_m961_m946_decoder_d2d3_10k_bounded_prefix_r1_one_shot.sh"
TEST = HW / "system_simulator/tests/test_m961_m946_decoder_d2d3_10k_bounded_prefix_source_candidate.py"
PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_interpreter() -> dict:
    executable = Path(sys.executable).resolve()
    if (executable != PYTHON_PATH or sha256(executable) != PYTHON_SHA256 or
            tuple(sys.version_info[:3]) != (3, 10, 18)):
        raise RuntimeError("M961 checker requires exact frozen Python")
    return {"path": str(executable), "sha256": PYTHON_SHA256,
            "version": [3, 10, 18]}


def load_driver():
    spec = importlib.util.spec_from_file_location("m961_static_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M961 driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_check(contract: Path) -> dict:
    interpreter = validate_interpreter()
    source_text = DRIVER.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(DRIVER))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = sorted(imported.intersection({
        "subprocess", "socket", "requests", "urllib", "paramiko",
        "torch", "tensorflow", "cupy",
    }))
    if forbidden:
        raise RuntimeError("forbidden execution/network/GPU import")
    required_runner_tokens = (
        "M961_EXPECTED_RELEASE_SHA256",
        "M961_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "--run-exact-pair",
        "--consume-attempt",
        "--publish-no-replace",
        "16777216",
        "--kill-after=30s 1800s",
    )
    missing = [token for token in required_runner_tokens
               if token not in runner_text]
    if missing:
        raise RuntimeError("M961 runner safety token missing: " +
                           ",".join(missing))
    if "--bounded-prefix 100000" in runner_text or "--run-full" in runner_text:
        raise RuntimeError("M961 runner contains unauthorized expanded mode")
    driver = load_driver()
    validation = driver.validate_source_contract(contract, RUNNER)
    self_test = driver.source_self_test()
    if driver.SOURCE_FETCH_REQUESTS != {"D2": 231600, "D3": 465600}:
        raise RuntimeError("M961 source-fetch boundary drift")
    return {
        "schema": "m961_decoder_d2d3_10k_source_static_check_v1",
        "status": "PASS_M961_STATIC_SOURCE_CHECK__NO_PREFIX_EXECUTED",
        "interpreter": interpreter,
        "driver_sha256": sha256(DRIVER),
        "runner_sha256": sha256(RUNNER),
        "test_sha256": sha256(TEST),
        "contract_sha256": sha256(contract),
        "forbidden_imports": forbidden,
        "runner_safety_tokens_present": True,
        "source_contract_validation": validation,
        "source_self_test_status": self_test["status"],
        "prefix_executed": False,
        "d2_or_d3_100k_authorized": False,
        "full_row_authorized": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(static_check(args.contract), sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
