#!/usr/bin/env python3
"""Read-only independent hammer for the M2045 environment preflight receipt."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PREFLIGHT = HW / (
    "reviews/m2045_ep34_valid825_sdformerflow_env_preflight_r1_20260902/"
    "preflight.json"
)
SOURCE = HW / (
    "system_handoff/scripts/"
    "run_m2045_ep34_valid825_sdformerflow_env_successor.py"
)
CONTRACT = HW / (
    "contracts/"
    "m2045_ep34_valid825_sdformerflow_env_successor_contract_r1_20260902.json"
)
ENGINE = HW / (
    "system_handoff/scripts/"
    "run_m2044_ep34_valid825_attention_eight_operator_qdq.py"
)
FAILURE = HW / (
    "results/"
    "m2044_ep34_valid825_attention_eight_operator_qdq_"
    "r1_20260902_FAILED_DO_NOT_CITE"
)
OUTPUT = HW / (
    "results/m2045_ep34_valid825_sdformerflow_env_successor_r1_20260902"
)

EXPECTED_SHA = {
    "preflight": "41da22f4b5745e5919f6177267bfefeb4d168815f43196c53755e89cad74079a",
    "source": "890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1",
    "contract": "4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0",
    "engine": "edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20",
    "failure_manifest": "6d366ccb3121a9b72e4e38bf12a112f6241e1be4e6fe341269685d7ceba6af58",
    "failure_outer": "ae7ebf05d56e4f409f09e1107f3c79fcebb7e61ced028593f282e1d7de8110a1",
    "failure_log": "a0dec1ac3481a6665deb3662b52a155bcfd4b019c57f857dd4104047cb8c7cc1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AssertionError("duplicate JSON key: " + key)
            result[key] = value
        return result

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError("nonfinite JSON token: " + token)),
    )
    assert type(result) is dict
    return result


def regular_exact(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    assert stat.S_ISREG(mode) and not path.is_symlink(), path
    assert sha256(path) == expected, path


def main() -> int:
    tests: list[dict[str, Any]] = []

    def check(name: str, value: bool) -> None:
        tests.append({"name": name, "pass": bool(value)})
        assert value, name

    regular_exact(PREFLIGHT, EXPECTED_SHA["preflight"])
    regular_exact(SOURCE, EXPECTED_SHA["source"])
    regular_exact(CONTRACT, EXPECTED_SHA["contract"])
    regular_exact(ENGINE, EXPECTED_SHA["engine"])
    check("fixed_file_sha256", True)
    ast.parse(SOURCE.read_text(encoding="utf-8"))
    ast.parse(ENGINE.read_text(encoding="utf-8"))
    check("python_ast", True)

    receipt = strict_json(PREFLIGHT)
    expected_receipt = {
        "m2044_failure_retained": True,
        "m2045_result_exists": False,
        "python": "/opt/conda/envs/sdformerflow/bin/python",
        "python_prefix": "/opt/conda/envs/sdformerflow",
        "reviewed_bundle_verified": True,
        "status": "PASS_M2045_ENV_SUCCESSOR_PREFLIGHT",
    }
    check("receipt_exact_schema_and_values", receipt == expected_receipt)

    contract = strict_json(CONTRACT)
    check(
        "contract_one_manual_successor",
        contract.get("status") ==
        "LOCKED_SOURCE_REVIEW_REQUIRED__ONE_MANUAL_SUCCESSOR_ATTEMPT_ONLY"
        and contract["producer"]["automatic_retry"] is False
        and contract["producer"][
            "valid825_successor_attempts_authorized_after_independent_source_review"
        ] == 1,
    )
    check(
        "contract_environment_only",
        contract["claim_boundary"]["environment_only_successor"] is True
        and contract["frozen_execution_engine"]["semantic_changes"] == 0
        and contract["accuracy_gate"]["baseline_AEE"] ==
        1.1995140134204518
        and contract["accuracy_gate"][
            "maximum_candidate_minus_baseline_AEE"
        ] == 0.02,
    )

    regular_exact(FAILURE / "SHA256SUMS", EXPECTED_SHA["failure_manifest"])
    regular_exact(
        FAILURE / "SHA256SUMS.seal.sha256", EXPECTED_SHA["failure_outer"]
    )
    regular_exact(FAILURE / "eval.log", EXPECTED_SHA["failure_log"])
    log = (FAILURE / "eval.log").read_text(encoding="utf-8")
    check(
        "m2044_exact_observed_root_cause",
        "ModuleNotFoundError: No module named 'spikingjelly'" in log
        and "M2044 evaluator exit_code=1" in log,
    )
    check(
        "m2044_accuracy_not_reached",
        "Validating" not in log and "spike_profile.json" not in log,
    )

    source_text = SOURCE.read_text(encoding="utf-8")
    engine_text = ENGINE.read_text(encoding="utf-8")
    fragments = (
        "Path(sys.prefix).resolve() == REQUIRED_PREFIX.resolve()",
        "import spikingjelly",
        "import torch",
        "engine.verify_inputs(engine_contract)",
        "engine.verify_bundle(bundle, M2044_SOURCE_SHA256, inputs,",
        "engine.run_valid825(",
        "require(not OUTPUT.exists()",
        "M2045 failed-attempt namespace exists; retry forbidden",
    )
    check("preflight_checks_environment_inputs_bundle", all(
        fragment in source_text for fragment in fragments
    ))
    check(
        "frozen_engine_uses_active_interpreter",
        'sys.executable, "-u", str(EVALUATOR.relative_to(ROOT))' in engine_text,
    )
    check(
        "m2044_schema_and_result_hammer_retained",
        "m2044_ep34_valid825_attention_eight_operator_qdq_result_r1_v1"
        in engine_text
        and "paper_accuracy_result_requires_independent_result_hammer"
        in engine_text,
    )

    temporary = OUTPUT.parent / ("." + OUTPUT.name + ".tmp")
    failed = OUTPUT.parent / (OUTPUT.name + "_FAILED_DO_NOT_CITE")
    check(
        "local_mirror_production_namespaces_absent",
        not OUTPUT.exists() and not temporary.exists() and not failed.exists(),
    )

    result = {
        "schema": "m2045_env_preflight_independent_hammer_r1_v1",
        "status": "PASS_AUTHORIZE_EXACTLY_ONE_GPU_RUN",
        "score": {"passed": len(tests), "total": len(tests)},
        "fixed_sha256": EXPECTED_SHA,
        "severity": {"P0": 0, "P1": 2, "P2": 3},
        "authorization": {
            "gpu_runs": 1,
            "automatic_retry": False,
            "interpreter": "/opt/conda/envs/sdformerflow/bin/python",
            "source_sha256": EXPECTED_SHA["source"],
            "contract_sha256": EXPECTED_SHA["contract"],
            "result_hammer_required": True,
        },
        "tests": tests,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
