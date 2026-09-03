#!/usr/bin/env python3
"""Read-only source/contract/evaluator hammer for M2044 prepare-only admission."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_handoff/scripts/run_m2044_ep34_valid825_attention_eight_operator_qdq.py"
CONTRACT = HW / "contracts/m2044_ep34_valid825_attention_eight_operator_qdq_contract_r1_20260902.json"
EVALUATOR = ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2042 = HW / "results/m2042_ep34_s40_eight_operator_int8_export_r1_20260902"
BASELINE = HW / "system_handoff/incoming/m2041_ep34_quant_binding_inputs/spike_profile.json"

EXPECTED = {
    "source": "edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20",
    "contract": "03f13063493d563cf0b26363498d18bde60c8bee5e785a4dfca95845555757d2",
    "evaluator": "84daee48291d8ab2ee644f43458b909e96190c0dce7f5ff4d4179b61be30faac",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m2042_result": "455c9fe7036779b890d4b85911cc42dc47bcb62c9fb6f6a6ce9c28a2c833cf29",
    "m2042_manifest": "519b8621a0c16f67ed33c8c624adc6bbfbc1c4a27224b2812542da3d92fc3881",
    "baseline": "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, name: str, tests: list[dict[str, object]]) -> None:
    tests.append({"name": name, "pass": bool(condition)})


def main() -> int:
    tests: list[dict[str, object]] = []
    check(sha256(SOURCE) == EXPECTED["source"], "source_sha", tests)
    check(sha256(CONTRACT) == EXPECTED["contract"], "contract_sha", tests)
    check(sha256(EVALUATOR) == EXPECTED["evaluator"], "evaluator_sha", tests)
    check(sha256(DOCS359) == EXPECTED["docs359"], "docs359_sha", tests)
    check(sha256(M2042 / "result.json") == EXPECTED["m2042_result"],
          "m2042_result_sha", tests)
    check(sha256(M2042 / "SHA256SUMS") == EXPECTED["m2042_manifest"],
          "m2042_manifest_sha", tests)
    check(sha256(BASELINE) == EXPECTED["baseline"], "baseline_sha", tests)

    source_text = SOURCE.read_text(encoding="utf-8")
    evaluator_text = EVALUATOR.read_text(encoding="utf-8")
    ast.parse(source_text)
    ast.parse(evaluator_text)
    check(True, "source_and_evaluator_ast", tests)

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    check(contract["phase_admission"]["valid825_run_requires_independent_derived_bundle_review"] is True,
          "bundle_review_required", tests)
    check(contract["accuracy_gate"]["maximum_candidate_minus_baseline_AEE"] == 0.02,
          "one_sided_aee_gate", tests)
    check(contract["producer"]["automatic_retry"] is False,
          "automatic_retry_false", tests)
    check(contract["candidate_contract"]["samples"] == 825,
          "contract_samples_825", tests)
    check(len(contract["candidate_contract"]["deployment_operator_forward_audit_targets"]) == 8,
          "contract_forward_targets_8", tests)

    required_source_fragments = (
        "regular_exact(CONTRACT, CONTRACT_SHA256",
        "derived-bundle failure already consumed this phase",
        "M2044 failed-attempt namespace exists; automatic retry forbidden",
        "expected_bundle_manifest_sha256",
        "non-target state drift:",
        "target QDQ content drift:",
        "verify_population_contract(profile, baseline)",
        "verify_forward_audit(profile)",
        "candidate deterministic backend audit failed",
        "gate = deltas[\"AEE\"] <=",
        "retain_failure(temporary, failed",
        '"paper_accuracy_result": False',
        '"paper_accuracy_result_requires_independent_result_hammer": True',
    )
    for fragment in required_source_fragments:
        check(fragment in source_text, "source_fragment:" + fragment, tests)

    required_evaluator_fragments = (
        "deployment_operator_forward_audit_targets",
        "operator_forward_counts[name] += 1",
        '"deployment_operator_forward_audit": operator_forward_audit',
        '"runtime_backend_audit": runtime_backend_audit',
        "torch.backends.cuda.matmul.allow_tf32 = allow_tf32",
        "torch.backends.cudnn.benchmark = bool(runtime_cfg",
    )
    for fragment in required_evaluator_fragments:
        check(fragment in evaluator_text, "evaluator_fragment:" + fragment, tests)

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    check(baseline["samples"] == 825, "baseline_samples_825", tests)
    check(baseline["eval_protocol"] == {
        "resolution": [480, 640], "crop": None, "window_size": [2, 15, 15],
        "remap": "v1", "bn_policy": "no_running", "bn_modules_changed": 78,
        "eval_batch_size": 1,
    }, "baseline_eval_protocol", tests)
    check(baseline["validation_file_list"]["sha256"] ==
          contract["inputs"]["validation_file_list_sha256"],
          "baseline_validation_population", tests)

    bundle = ROOT / contract["outputs"]["derived_bundle"]
    result = ROOT / contract["outputs"]["result_directory"]
    paths_expected_absent = (
        bundle,
        bundle.parent / (bundle.name + "_FAILED_DO_NOT_CITE"),
        bundle.parent / ("." + bundle.name + ".tmp"),
        result,
        result.parent / (result.name + "_FAILED_DO_NOT_CITE"),
        result.parent / ("." + result.name + ".tmp"),
    )
    check(all(not path.exists() for path in paths_expected_absent),
          "prepare_and_run_namespaces_absent", tests)

    passed = sum(test["pass"] is True for test in tests)
    output = {
        "schema": "m2044_source_contract_evaluator_independent_hammer_r2_v1",
        "status": "PASS_SOURCE_REVIEW_PREPARE_ONLY" if passed == len(tests) else "FAIL",
        "score": {"passed": passed, "total": len(tests)},
        "fixed_sha256": EXPECTED,
        "tests": tests,
        "admission": {
            "prepare_only_once": passed == len(tests),
            "valid825_gpu_run": False,
            "gpu_run_reason": "derived bundle does not exist and has not been independently reviewed",
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
