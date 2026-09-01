#!/usr/bin/env python3
"""Read-only release-integrity check for exact M1560 source bytes.

No call to run_once/stream_actual_call is permitted.  Compatible with
CPython 3.6.
"""
from __future__ import print_function

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m1560_ep34_decoder_d0_call0_nonproduct_one_shot.py"
TEST = HW / "system_simulator/tests/test_m1560_ep34_decoder_d0_call0_nonproduct_one_shot_source.py"
CONTRACT = HW / "contracts/m1560_ep34_decoder_d0_call0_nonproduct_one_shot_source_contract_r1_20260901.json"
AUTHOR = HW / "results/m1560_ep34_decoder_d0_call0_nonproduct_one_shot_source_r1_20260901/receipt.json"
M1556 = HW / "system_simulator/scripts/build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
M1559_REVIEW = HW / "reviews/m1559_m1556_decoder_immutable_snapshot_regression_r1_20260901/review.json"
M1559_OUTER = HW / "reviews/m1559_m1556_decoder_immutable_snapshot_regression_r1_20260901/SHA256SUMS.seal.sha256"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RUNNER: "890a7cf66b8132b23df77d864d08d75766e0f967b194b0dd40c2f244e76c674f",
    TEST: "df821787e7f89f1088930a8f722178e36625a743b392f96c1185445b39eadfeb",
    CONTRACT: "9dd2737b25cb4f76fefea39c35326592748a46944c533e4d21677cb434720731",
    AUTHOR: "00438a04019144302bd42cf334de76743057ca064a761726cb4da4d812333624",
    M1556: "a2fd0e3b1d5fbadcb18ccbadd7b4f709114abb22a19b6c92eec940afab5f9dfa",
    M1559_REVIEW: "9b34ec5d2e2fd7fb3a934e864b0cd975b6a9c2306c8f7fe80e5c77f6530c1185",
    M1559_OUTER: "ae36fe2c2a6643623c6840577cf828587d07508dcc15f36d3ec4922fc0921399",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


for _path, _digest in EXPECTED.items():
    assert _path.is_file() and sha256(_path) == _digest, (
        "identity drift: " + str(_path))

SPEC = importlib.util.spec_from_file_location("m1566_bound_m1560", str(RUNNER))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def main():
    description = M.describe()
    assert description["status"] == (
        "SOURCE_ONLY__INDEPENDENT_RELEASE_REVIEW_REQUIRED__ATTEMPT_NOT_CONSUMED")
    assert description["population"] == {
        "call_ordinal": 0, "sample_id": 10, "module_ordinal": 0,
        "timesteps": 10, "configurations": list(M.CONFIGS)}
    assert M.CONFIGS == (
        "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
    assert "PRODUCT_CAPTURE_TYPED_K8" not in M.CONFIGS
    assert description["one_shot"] == {
        "fresh_output": True, "exclusive_lock": True,
        "automatic_retry": False,
        "partial_after_each_configuration": True}
    assert description["execution"] == {
        "attempt_consumed": False, "pilot": False,
        "production": False, "product": False}
    assert description["claim_boundary"]["paper_citable_performance"] is False

    m1559 = M.verify_m1559()
    assert m1559["decision"][
        "prerequisite_for_one_d0_call0_three_nonproduct_diagnostic_met"] is True
    assert m1559["decision"]["actual_run_authorized_by_this_review"] is False
    bound = M.load_bound_source()
    assert tuple(bound.M.CONFIGS) == M.CONFIGS
    assert bound.describe()["pilot"] == {
        "call_ordinal": 0, "sample_id": 10, "module_ordinal": 0,
        "timesteps": 10, "execution": False}
    assert bound.FORBIDDEN_CONFIG == "PRODUCT_CAPTURE_TYPED_K8"

    contract = M.strict_json(CONTRACT)
    author = M.strict_json(AUTHOR)
    assert contract["one_shot"] == {
        "fresh_output_required": True,
        "exclusive_lock_required": True,
        "attempt_marker_written_before_first_replay": True,
        "automatic_retry": False,
        "partial_result_written_after_each_configuration": True,
        "same_commit_sequence_required": True,
        "same_resource_manifest_required": True}
    assert contract["resource_gates"]["mem_available_strict_min_bytes"] == (
        16 * 1024 * 1024 * 1024)
    assert contract["resource_gates"]["disk_free_strict_min_bytes"] == (
        16 * 1024 * 1024 * 1024)
    assert author["identity"]["m1556_source_sha256"] == EXPECTED[M1556]
    assert author["identity"]["m1559_review_sha256"] == EXPECTED[M1559_REVIEW]
    assert author["identity"]["m1559_outer_file_sha256"] == EXPECTED[M1559_OUTER]
    assert author["claim_boundary"]["actual_attempts"] == 0
    assert author["claim_boundary"]["paper_result"] is False

    # Ordinary source-order/AST integrity check.  It does not invoke run_once.
    source = RUNNER.read_text(encoding="utf-8")
    run_text = source[source.index("def run_once(output):"):
                      source.index("\ndef describe():")]
    positions = [run_text.index(token) for token in (
        "before = preflight(output, full_payload=True)",
        "output.mkdir()",
        "write_new(output / \"WORK_STARTED.json\"",
        "module = load_bound_source()",
        "row = module.stream_actual_call(config)",
        "write_new(output / (\"partial_%d_%s.json\"",
        "write_new(output / \"result.json\"")]
    assert positions == sorted(positions)
    tree = ast.parse(source)
    run_node = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == "run_once")
    assert not any(isinstance(node, ast.While) for node in ast.walk(run_node))
    assert sum(1 for node in ast.walk(run_node)
               if isinstance(node, ast.Call) and
               isinstance(node.func, ast.Name) and
               node.func.id == "run_once") == 0
    assert '"automatic_retry": False' in run_text
    assert 'row["diagnostic_only"] is True' in run_text
    assert 'row["product_capture"] is False' in run_text
    assert 'row["production"] is False' in run_text
    assert '"paper_citable_performance": False' in run_text
    assert '"paper_result": False' in M1556.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="m1566_preflight.",
                                     dir=str(HERE)) as directory:
        root = Path(directory)
        output = root / "fresh_output"
        checked, parent = M.validate_output(output)
        assert checked == output and parent == root.resolve()
        preflight = M.preflight(output, False)
        assert preflight["status"] == (
            "PASS_M1560_ONE_SHOT_SOURCE_PREFLIGHT__NO_EXECUTION")
        assert preflight["attempt_consumed"] is False
        assert preflight["actual_run_authorized_by_author_source"] is False
        assert preflight["memory_available_bytes"] > M.MIN_MEMORY_BYTES
        assert preflight["disk_free_bytes"] > M.MIN_DISK_BYTES
        assert not output.exists()

    author_test = subprocess.check_output([sys.executable, str(TEST)],
                                          stderr=subprocess.STDOUT).decode("utf-8")
    assert "PASS M1560 source tests attacks=5 pilot=0 production=0 product=0" in author_test

    result = {
        "schema": "m1566_m1560_decoder_one_shot_release_independent_check_output_r1_v1",
        "status": "PASS_M1566_M1560_RELEASE_INTEGRITY__ONE_DIAGNOSTIC_ATTEMPT_AUTHORIZED__NO_EXECUTION",
        "python": sys.version.split()[0],
        "bindings": dict((path.name, digest) for path, digest in EXPECTED.items()),
        "verified": {
            "population": description["population"],
            "m1556_exact_sha": True,
            "m1559_review_and_outer_exact_sha": True,
            "fresh_output_gate": True,
            "exclusive_lock_declared_and_acquired_before_preflight_in_source": True,
            "memory_strictly_over_16gib": True,
            "disk_strictly_over_16gib": True,
            "attempt_marker_before_first_replay": True,
            "automatic_retry_path_absent": True,
            "partial_after_each_configuration": True,
            "same_commit_and_resource_required": True,
            "diagnostic_only_rows_required": True,
            "partial_paper_result_false_from_exact_bound_source": True,
            "final_paper_citable_performance_false": True,
            "author_source_test_pass": True,
            "preflight_pass_without_namespace_creation": True,
        },
        "execution": {
            "actual_replay": False, "attempt_consumed": False,
            "output_created": False, "production": False,
            "product_configuration": False, "gpu": False,
            "ssh": False, "rtl_eda": False},
        "authorization": {
            "one_d0_call0_sample10_three_nonproduct_diagnostic_attempt": True,
            "attempts": 1, "automatic_retry": False,
            "production": False, "product_configuration": False,
            "paper_citable_performance": False,
            "independent_result_hammer_required_after_run": True},
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
