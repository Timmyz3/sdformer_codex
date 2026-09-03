#!/usr/bin/env python3
"""Run one manually reviewed environment-only successor to failed M2044.

M2044 was sealed before model construction because the contracted base Python
environment did not contain spikingjelly.  This wrapper changes only the Python
environment.  It imports the exact frozen M2044 execution engine, reuses the
independently reviewed M2044 bundle, and publishes into a new M2045 namespace.
It never removes or replaces the M2044 failure receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / (
    "contracts/m2045_ep34_valid825_sdformerflow_env_successor_"
    "contract_r1_20260902.json"
)
CONTRACT_SHA256 = (
    "4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0"
)
M2044_SOURCE = HW / (
    "system_handoff/scripts/"
    "run_m2044_ep34_valid825_attention_eight_operator_qdq.py"
)
M2044_SOURCE_SHA256 = (
    "edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20"
)
M2044_FAILURE = HW / (
    "results/"
    "m2044_ep34_valid825_attention_eight_operator_qdq_"
    "r1_20260902_FAILED_DO_NOT_CITE"
)
M2044_FAILURE_MANIFEST_SHA256 = (
    "6d366ccb3121a9b72e4e38bf12a112f6241e1be4e6fe341269685d7ceba6af58"
)
M2044_FAILURE_OUTER_SHA256 = (
    "ae7ebf05d56e4f409f09e1107f3c79fcebb7e61ced028593f282e1d7de8110a1"
)
M2044_FAILURE_LOG_SHA256 = (
    "a0dec1ac3481a6665deb3662b52a155bcfd4b019c57f857dd4104047cb8c7cc1"
)
M2044_FAILURE_TXT_SHA256 = (
    "52cc347333333875baebeee1fa12941d37c4ff01a2cd54815a392ed4db8a9ce7"
)
TENSOR_REVIEW = HW / (
    "reviews/m2044_ep34_derived_bundle_tensor_audit_r1_20260902"
)
TENSOR_REVIEW_MANIFEST_SHA256 = (
    "e2714d4a841e86fba30265d97e537c8f98a19af521a5ece8d8a47b9c33ae3ce9"
)
TENSOR_REVIEW_OUTER_SHA256 = (
    "d7a102fd964d7a1109fa309dc4e45a296d99646b2f23f23241bccb7e25548bea"
)
TENSOR_AUDIT_JSON_SHA256 = (
    "0e8905bde3d54b53518b0795ea42656a16c7da305788d200e06e16261b415fe6"
)
BUNDLE_MANIFEST_SHA256 = (
    "ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c"
)
REQUIRED_PREFIX = Path("/opt/conda/envs/sdformerflow")
OUTPUT = HW / (
    "results/m2045_ep34_valid825_sdformerflow_env_successor_r1_20260902"
)


class M2045Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M2045Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M2045Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA256 drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    result = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M2045Error("nonfinite JSON token: " + token)))
    require(type(result) is dict, "JSON root is not an object")
    return result


def verify_outer(directory: Path, manifest_sha: str, outer_sha: str) -> None:
    regular_exact(directory / "SHA256SUMS", manifest_sha,
                  directory.name + "/SHA256SUMS")
    regular_exact(directory / "SHA256SUMS.seal.sha256", outer_sha,
                  directory.name + "/SHA256SUMS.seal.sha256")
    require((directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
        directory.name + " outer-seal content drift")


def verify_environment() -> None:
    require(Path(sys.prefix).resolve() == REQUIRED_PREFIX.resolve(),
            "M2045 must run in /opt/conda/envs/sdformerflow")
    import spikingjelly
    import torch
    require(Path(spikingjelly.__file__).resolve().is_relative_to(
        REQUIRED_PREFIX.resolve()), "spikingjelly is outside required prefix")
    require(Path(torch.__file__).resolve().is_relative_to(
        REQUIRED_PREFIX.resolve()), "torch is outside required prefix")


def verify_authority(expected_source_sha256: str) -> dict[str, Any]:
    regular_exact(Path(__file__).resolve(), expected_source_sha256,
                  "running M2045 wrapper")
    regular_exact(CONTRACT, CONTRACT_SHA256, "M2045 contract")
    contract = strict_json(CONTRACT)
    require(contract.get("schema") ==
            "m2045_ep34_valid825_sdformerflow_env_successor_contract_r1_v1" and
            contract.get("status") ==
            "LOCKED_SOURCE_REVIEW_REQUIRED__ONE_MANUAL_SUCCESSOR_ATTEMPT_ONLY",
            "M2045 contract identity drift")
    regular_exact(M2044_SOURCE, M2044_SOURCE_SHA256, "frozen M2044 engine")
    verify_outer(M2044_FAILURE, M2044_FAILURE_MANIFEST_SHA256,
                 M2044_FAILURE_OUTER_SHA256)
    regular_exact(M2044_FAILURE / "eval.log", M2044_FAILURE_LOG_SHA256,
                  "M2044 failure eval.log")
    regular_exact(M2044_FAILURE / "FAILURE.txt", M2044_FAILURE_TXT_SHA256,
                  "M2044 FAILURE.txt")
    failure_log = (M2044_FAILURE / "eval.log").read_text(encoding="utf-8")
    require("ModuleNotFoundError: No module named 'spikingjelly'" in failure_log and
            "M2044 evaluator exit_code=1" in failure_log,
            "M2044 root cause is not exact")
    verify_outer(TENSOR_REVIEW, TENSOR_REVIEW_MANIFEST_SHA256,
                 TENSOR_REVIEW_OUTER_SHA256)
    regular_exact(TENSOR_REVIEW / "remote_cpu_audit.json",
                  TENSOR_AUDIT_JSON_SHA256, "M2044 remote tensor audit")
    audit = strict_json(TENSOR_REVIEW / "remote_cpu_audit.json")
    counts = audit.get("counts", {})
    require(audit.get("status") ==
            "PASS_M2044_EP34_DERIVED_BUNDLE_TENSOR_AUDIT" and
            counts.get("tensor_keys_checked") == 921 and
            counts.get("non_target_torch_equal") == 913 and
            counts.get("target_qdq_torch_equal") == 8 and
            counts.get("mismatches") == 0,
            "M2044 tensor audit admission drift")
    verify_environment()
    require(not OUTPUT.exists(), "M2045 canonical result already exists")
    require(not (OUTPUT.parent / ("." + OUTPUT.name + ".tmp")).exists(),
            "stale M2045 temporary result exists")
    require(not (OUTPUT.parent / (OUTPUT.name + "_FAILED_DO_NOT_CITE")).exists(),
            "M2045 failed-attempt namespace exists; retry forbidden")
    return contract


def load_m2044() -> Any:
    spec = importlib.util.spec_from_file_location("m2044_frozen_engine",
                                                  M2044_SOURCE)
    require(spec is not None and spec.loader is not None,
            "cannot load frozen M2044 engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha256", required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--preflight", action="store_true")
    phase.add_argument("--run", action="store_true")
    args = parser.parse_args()
    verify_authority(args.expected_source_sha256)
    engine = load_m2044()
    engine_contract = engine.load_contract(M2044_SOURCE_SHA256)
    inputs = engine.verify_inputs(engine_contract)
    bundle, _unused_m2044_output = engine.output_paths(engine_contract)
    engine.verify_bundle(bundle, M2044_SOURCE_SHA256, inputs,
                         BUNDLE_MANIFEST_SHA256)
    if args.preflight:
        print(json.dumps({
            "status": "PASS_M2045_ENV_SUCCESSOR_PREFLIGHT",
            "python": sys.executable,
            "python_prefix": sys.prefix,
            "m2044_failure_retained": True,
            "reviewed_bundle_verified": True,
            "m2045_result_exists": False,
        }, sort_keys=True))
        return 0

    engine.run_valid825(
        engine_contract, inputs, bundle, OUTPUT, M2044_SOURCE_SHA256,
        BUNDLE_MANIFEST_SHA256)
    require(OUTPUT.is_dir(), "M2045 result was not atomically published")
    print("PASS_M2045_ENV_SUCCESSOR_EXECUTION__RESULT_HAMMER_REQUIRED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (M2045Error, Exception) as error:
        print("FAIL_M2045: " + str(error), file=sys.stderr)
        raise SystemExit(2)
