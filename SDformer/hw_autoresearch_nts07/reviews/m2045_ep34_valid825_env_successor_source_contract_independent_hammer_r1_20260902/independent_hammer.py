#!/usr/bin/env python3
"""Read-only static hammer for M2045 environment-successor preflight admission."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_handoff/scripts/run_m2045_ep34_valid825_sdformerflow_env_successor.py"
CONTRACT = HW / "contracts/m2045_ep34_valid825_sdformerflow_env_successor_contract_r1_20260902.json"
ENGINE = HW / "system_handoff/scripts/run_m2044_ep34_valid825_attention_eight_operator_qdq.py"
FAILURE = HW / "results/m2044_ep34_valid825_attention_eight_operator_qdq_r1_20260902_FAILED_DO_NOT_CITE"
TENSOR_REVIEW = HW / "reviews/m2044_ep34_derived_bundle_tensor_audit_r1_20260902"
BUNDLE = HW / "system_handoff/generated/m2044_ep34_attention_hw_order_qdq8_bundle_r1_20260902"
OUTPUT = HW / "results/m2045_ep34_valid825_sdformerflow_env_successor_r1_20260902"

EXPECTED = {
    "source": "890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1",
    "contract": "4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0",
    "engine": "edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20",
    "failure_manifest": "6d366ccb3121a9b72e4e38bf12a112f6241e1be4e6fe341269685d7ceba6af58",
    "failure_outer": "ae7ebf05d56e4f409f09e1107f3c79fcebb7e61ced028593f282e1d7de8110a1",
    "failure_log": "a0dec1ac3481a6665deb3662b52a155bcfd4b019c57f857dd4104047cb8c7cc1",
    "failure_txt": "52cc347333333875baebeee1fa12941d37c4ff01a2cd54815a392ed4db8a9ce7",
    "bundle_manifest": "ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c",
    "tensor_review_manifest": "e2714d4a841e86fba30265d97e537c8f98a19af521a5ece8d8a47b9c33ae3ce9",
    "tensor_audit": "0e8905bde3d54b53518b0795ea42656a16c7da305788d200e06e16261b415fe6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    tests: list[dict[str, object]] = []

    def check(value: bool, name: str) -> None:
        tests.append({"name": name, "pass": bool(value)})

    for name, path in (
        ("source", SOURCE), ("contract", CONTRACT), ("engine", ENGINE),
    ):
        check(sha256(path) == EXPECTED[name], name + "_sha")
        ast.parse(path.read_text(encoding="utf-8"))
    check(True, "python_ast")
    check(sha256(FAILURE / "SHA256SUMS") == EXPECTED["failure_manifest"],
          "failure_manifest_sha")
    check(sha256(FAILURE / "SHA256SUMS.seal.sha256") == EXPECTED["failure_outer"],
          "failure_outer_sha")
    check(sha256(FAILURE / "eval.log") == EXPECTED["failure_log"],
          "failure_log_sha")
    check(sha256(FAILURE / "FAILURE.txt") == EXPECTED["failure_txt"],
          "failure_txt_sha")
    check(sha256(BUNDLE / "SHA256SUMS") == EXPECTED["bundle_manifest"],
          "reviewed_bundle_manifest_sha")
    check(sha256(TENSOR_REVIEW / "SHA256SUMS") ==
          EXPECTED["tensor_review_manifest"], "tensor_review_manifest_sha")
    check(sha256(TENSOR_REVIEW / "remote_cpu_audit.json") ==
          EXPECTED["tensor_audit"], "tensor_audit_sha")

    log = (FAILURE / "eval.log").read_text(encoding="utf-8")
    check("ModuleNotFoundError: No module named 'spikingjelly'" in log,
          "observed_root_cause_spikingjelly")
    check("M2044 evaluator exit_code=1" in log, "failed_evaluator_exit")
    check("Validating..." not in log and not (FAILURE / "spike_profile.json").exists(),
          "accuracy_not_executed")

    source = SOURCE.read_text(encoding="utf-8")
    required = (
        "Path(sys.prefix).resolve() == REQUIRED_PREFIX.resolve()",
        "Path(spikingjelly.__file__).resolve().is_relative_to",
        "Path(torch.__file__).resolve().is_relative_to",
        "M2045 failed-attempt namespace exists; retry forbidden",
        "engine.verify_bundle(bundle, M2044_SOURCE_SHA256, inputs,",
        "engine.run_valid825(",
        "bundle, OUTPUT, M2044_SOURCE_SHA256,",
        "BUNDLE_MANIFEST_SHA256",
    )
    for fragment in required:
        check(fragment in source, "source_fragment:" + fragment)

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    check(contract["frozen_execution_engine"]["semantic_changes"] == 0,
          "no_engine_semantic_change")
    check(contract["producer"]["required_python_prefix_on_A800"] ==
          "/opt/conda/envs/sdformerflow", "required_prefix")
    check(contract["accuracy_gate"] == {
        "baseline_AEE": 1.1995140134204518,
        "maximum_candidate_minus_baseline_AEE": 0.02,
        "unchanged_from_m2044": True,
    }, "aee_gate_unchanged")
    check(contract["outputs"]["result_schema_remains_m2044_engine_schema"] is True,
          "m2044_result_schema_preserved")

    audit = json.loads((TENSOR_REVIEW / "remote_cpu_audit.json").read_text())
    counts = audit["counts"]
    check(counts["tensor_keys_checked"] == 921 and
          counts["non_target_torch_equal"] == 913 and
          counts["target_qdq_torch_equal"] == 8 and
          counts["mismatches"] == 0, "reviewed_bundle_tensor_audit")
    check(not OUTPUT.exists() and
          not (OUTPUT.parent / ("." + OUTPUT.name + ".tmp")).exists() and
          not (OUTPUT.parent / (OUTPUT.name + "_FAILED_DO_NOT_CITE")).exists(),
          "m2045_namespaces_absent")

    passed = sum(row["pass"] is True for row in tests)
    result = {
        "schema": "m2045_env_successor_source_contract_hammer_r1_v1",
        "status": "PASS_PREFLIGHT_ONLY" if passed == len(tests) else "FAIL",
        "score": {"passed": passed, "total": len(tests)},
        "fixed_sha256": EXPECTED,
        "admission": {
            "one_preflight": passed == len(tests),
            "gpu_run": False,
            "reason": "A800 environment preflight has not yet produced a reviewed receipt",
        },
        "tests": tests,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
