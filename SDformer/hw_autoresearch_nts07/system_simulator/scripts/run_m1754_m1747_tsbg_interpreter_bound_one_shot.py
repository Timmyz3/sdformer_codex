#!/usr/bin/env python3
"""Interpreter-bound one-shot executor for the exact M1747 TSBG analyzer.

M1754 changes no analysis code.  After exact M1755/M1756 authority validation,
it proves that it is running under the one fixed CPython binary, imports exact
torch/numpy versions without using a GPU, checks fresh result/work namespaces,
atomically consumes one wrapper attempt, and execs the exact M1747 source.
Source-self-check mode performs none of those production actions.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1754_m1747_tsbg_interpreter_bound_one_shot.py"
CONTRACT = HW / "contracts/m1754_m1747_tsbg_interpreter_bound_execution_source_contract_r1_20260901.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
M1747_SOURCE = HW / "system_simulator/scripts/analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py"
M1749_RELEASE = HW / "contracts/m1749_m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_analysis_release_r1_20260901.json"
M1748_REVIEW = HW / "reviews/m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_source_hammer_r1_20260901"
FAILURE = HW / "results/m1754_m1749_m1747_tsbg_interpreter_failure_receipt_r1_20260901.json"
FAILURE_SIDECAR = Path(str(FAILURE) + ".sha256")
FAILURE_OUTER = Path(str(FAILURE) + ".sha256.seal.sha256")
FUTURE_REVIEW = HW / "reviews/m1755_m1754_m1747_tsbg_interpreter_bound_source_hammer_r1_20260901"
FUTURE_RELEASE = HW / "contracts/m1756_m1755_m1754_tsbg_interpreter_bound_execution_release_r1_20260901.json"
RESULT = HW / "results/m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_r1_20260901"
WORK = HW / "results/.m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_r1_20260901.work"
ATTEMPT = HW / "results/.m1754_m1747_tsbg_interpreter_bound_execution_attempt_consumed"
INTERPRETER = Path("/opt/conda/envs/sdformerflow/bin/python3.10")

M1747_SOURCE_SHA256 = "3bc48502ab1cccf579cfc65dc0cba2747e5bd38a8a4df82dda3f626f7283683b"
M1749_RELEASE_SHA256 = "6114020ab8d4da7c9a7c6f149496ee3efb1e7d19aeff5e34becaf60c1d465806"
M1748_REVIEW_SHA256 = "f9c3e152bb10d67a1e0b2421565e0f72469804fab4330dae9c00518b684e1c47"
M1748_MANIFEST_SHA256 = "10683d2a63035841ef17572a5ca8b57a98eb260cb5b8c39d8d5eabbfb132e594"
M1748_OUTER_SHA256 = "d1ba7c36dff713385fc30817877f3228516f9a6fa862805a44e5f7d6355e07cc"
FAILURE_SHA256 = "57605ca6fa397429a4673f351cf2b01016dea7ff1dbcb29a01d1cbb4e4f12440"
FAILURE_SIDECAR_SHA256 = "199c647a948df47990078739e0c0ebff0861f7723f63f2d3f77970bd2e90b666"
FAILURE_OUTER_SHA256 = "382beb26c5e017f33041f99eb24cfd7d931d67dc918bfe8081f50cee0e9c8ebe"
INTERPRETER_SHA256 = "89520a3f2bc6e4f670921bd7a71a66eb0073775e685f6cbefda0dcda7bc42aa0"
PYTHON_VERSION = (3, 10, 20)
TORCH_VERSION = "2.2.2+cu121"
NUMPY_VERSION = "1.26.4"
REVIEW_SCHEMA = "m1755_m1754_m1747_tsbg_interpreter_bound_source_hammer_r1_v1"
REVIEW_STATUS = "PASS_M1755_M1754_SOURCE_HAMMER__M1756_RELEASE_MAY_BE_CREATED"
RELEASE_SCHEMA = "m1756_m1755_m1754_tsbg_interpreter_bound_execution_release_r1_v1"
RELEASE_STATUS = "AUTHORIZE_ONE_M1754_INTERPRETER_BOUND_M1747_ANALYSIS_EXECUTION"


class M1754Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1754Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1754Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(M1754Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_sidecar(path, sidecar, outer, label):
    path, sidecar, outer = Path(path), Path(sidecar), Path(outer)
    require(sidecar.is_file() and not sidecar.is_symlink() and outer.is_file() and not outer.is_symlink(),
            label + " double seal missing")
    require(sidecar.read_text(encoding="ascii").split() == [sha256(path), path.name], label + " sidecar drift")
    require(outer.read_text(encoding="ascii").split() == [sha256(sidecar), sidecar.name], label + " outer drift")


def verify_review(root, label):
    root = Path(root)
    sums, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink() and sums.is_file() and outer.is_file(), label + " missing")
    require(outer.read_text(encoding="ascii").split() == [sha256(sums), sums.name], label + " outer drift")
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        require(name and not Path(name).is_absolute() and ".." not in Path(name).parts, label + " unsafe member")
        regular_exact(root / name, digest, label + " member " + name)
    return {"review_sha256": sha256(root / "review.json"), "manifest_sha256": sha256(sums),
            "outer_seal_file_sha256": sha256(outer)}


def source_identities():
    return {"source_sha256": sha256(SOURCE), "test_sha256": sha256(TEST),
        "contract_sha256": sha256(CONTRACT), "contract_sidecar_sha256": sha256(CONTRACT_SIDECAR),
        "contract_outer_seal_file_sha256": sha256(CONTRACT_OUTER),
        "m1747_source_sha256": M1747_SOURCE_SHA256, "consumed_m1749_release_sha256": M1749_RELEASE_SHA256,
        "m1748_review_sha256": M1748_REVIEW_SHA256, "m1748_review_manifest_sha256": M1748_MANIFEST_SHA256,
        "m1748_review_outer_seal_file_sha256": M1748_OUTER_SHA256,
        "m1754_failure_receipt_sha256": FAILURE_SHA256,
        "m1754_failure_receipt_outer_seal_file_sha256": FAILURE_OUTER_SHA256,
        "interpreter_path": str(INTERPRETER), "interpreter_sha256": INTERPRETER_SHA256,
        "python_version": "3.10.20", "torch_version": TORCH_VERSION, "numpy_version": NUMPY_VERSION}


def validate_contract():
    verify_sidecar(CONTRACT, CONTRACT_SIDECAR, CONTRACT_OUTER, "M1754 contract")
    row = strict_json(CONTRACT)
    require(row.get("schema") == "m1754_m1747_tsbg_interpreter_bound_execution_source_contract_r1_v1" and
            row.get("status") == "SOURCE_ONLY__INTERPRETER_BOUND_WRAPPER__NO_ANALYSIS_NO_RELEASE" and
            row.get("source") == {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            row.get("test") == {"path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)} and
            row.get("authorization", {}).get("analysis_run") is False and
            row.get("claim_boundary", {}).get("paper_result") is False, "M1754 contract drift")
    return row


def validate_static():
    regular_exact(M1747_SOURCE, M1747_SOURCE_SHA256, "exact M1747 source")
    regular_exact(M1749_RELEASE, M1749_RELEASE_SHA256, "consumed M1749 release")
    regular_exact(FAILURE, FAILURE_SHA256, "M1754 failure receipt")
    regular_exact(FAILURE_SIDECAR, FAILURE_SIDECAR_SHA256, "failure sidecar")
    regular_exact(FAILURE_OUTER, FAILURE_OUTER_SHA256, "failure outer")
    verify_sidecar(FAILURE, FAILURE_SIDECAR, FAILURE_OUTER, "M1754 failure receipt")
    failure = strict_json(FAILURE)
    require(failure.get("absence_and_budget", {}).get("m1749_authority_consumed") is True and
            failure.get("absence_and_budget", {}).get("payload_replays") == 0 and
            failure.get("absence_and_budget", {}).get("automatic_retry") is False, "failure semantics drift")
    binding = verify_review(M1748_REVIEW, "M1748 review")
    require(binding == {"review_sha256": M1748_REVIEW_SHA256, "manifest_sha256": M1748_MANIFEST_SHA256,
                        "outer_seal_file_sha256": M1748_OUTER_SHA256}, "M1748 triple drift")
    return binding


def validate_future_review(root, identities):
    binding = verify_review(root, "M1755 review")
    row = strict_json(Path(root) / "review.json")
    require(row.get("schema") == REVIEW_SCHEMA and row.get("status") == REVIEW_STATUS and
            row.get("identity") == identities and row.get("authorization") == {
                "m1756_release_may_be_created": True, "execution": False, "analysis_run": False} and
            row.get("claim_boundary", {}).get("paper_result") is False, "M1755 authority drift")
    return binding


def validate_future_release(path, review, identities):
    path = Path(path); sidecar = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    verify_sidecar(path, sidecar, outer, "M1756 release")
    expected = dict(identities); expected.update({"m1755_review_sha256": review["review_sha256"],
        "m1755_review_outer_seal_file_sha256": review["outer_seal_file_sha256"]})
    row = strict_json(path)
    require(row.get("schema") == RELEASE_SCHEMA and row.get("status") == RELEASE_STATUS and
            row.get("identity") == expected and row.get("authorization") == {
                "wrapper_runs": 1, "interpreter_preflights": 1, "execs": 1, "analysis_runs": 1,
                "capture_verifications": 1, "result_publications": 1, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0} and
            row.get("claim_boundary", {}).get("paper_result") is False, "M1756 release drift")
    return {"release_sha256": sha256(path), "release_outer_seal_file_sha256": sha256(outer)}


def verify_authority():
    validate_contract(); validate_static(); identities = source_identities()
    review = validate_future_review(FUTURE_REVIEW, identities)
    release = validate_future_release(FUTURE_RELEASE, review, identities)
    return identities, review, release


def interpreter_preflight():
    require(Path(sys.executable) == INTERPRETER, "wrong interpreter path")
    regular_exact(INTERPRETER, INTERPRETER_SHA256, "production interpreter")
    require(tuple(sys.version_info[:3]) == PYTHON_VERSION, "Python version drift")
    import torch
    import numpy
    require(torch.__version__ == TORCH_VERSION and numpy.__version__ == NUMPY_VERSION,
            "torch/numpy version drift")
    return {"interpreter_sha256": INTERPRETER_SHA256, "python": "3.10.20",
            "torch": torch.__version__, "numpy": numpy.__version__}


def run_execution():
    identities, review, release = verify_authority()
    versions = interpreter_preflight()
    require(not os.path.lexists(str(RESULT)) and not os.path.lexists(str(WORK)), "fresh M1747 namespaces required")
    try:
        ATTEMPT.mkdir()
    except FileExistsError as error:
        raise M1754Error("M1754 attempt already consumed") from error
    receipt = {"schema": "m1754_interpreter_bound_execution_receipt_r1_v1",
        "status": "ATTEMPT_CONSUMED__EXEC_EXACT_M1747", "identity": identities,
        "m1755_review_sha256": review["review_sha256"], "m1756_release_sha256": release["release_sha256"],
        "versions": versions, "target_source_sha256": M1747_SOURCE_SHA256,
        "automatic_retry": False, "gpu_runs": 0, "eda_runs": 0}
    (ATTEMPT / "launch_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = dict(os.environ); env["PYTHONNOUSERSITE"] = "1"; env["CUDA_VISIBLE_DEVICES"] = ""
    os.execve(str(INTERPRETER), [str(INTERPRETER), str(M1747_SOURCE), "--run-analysis"], env)


def source_self_check():
    validate_contract(); validate_static()
    require(not os.path.lexists(str(RESULT)) and not os.path.lexists(str(WORK)) and
            not os.path.lexists(str(ATTEMPT)), "M1754 namespaces not fresh")
    return {"status": "PASS_M1754_SOURCE_SELF_CHECK__NO_EXECUTION", "live_interpreter_checked": False,
        "interpreter_expected_sha256": INTERPRETER_SHA256, "m1747_algorithm_changed": False,
        "attempt_created": False, "capture_touched": False, "analysis_runs": 0,
        "gpu_runs": 0, "eda_runs": 0, "network_access": False, "paper_result": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-self-check", action="store_true")
    mode.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args(argv)
    if args.source_self_check:
        print(json.dumps(source_self_check(), indent=2, sort_keys=True)); return 0
    run_execution(); return 99


if __name__ == "__main__":
    sys.exit(main())
