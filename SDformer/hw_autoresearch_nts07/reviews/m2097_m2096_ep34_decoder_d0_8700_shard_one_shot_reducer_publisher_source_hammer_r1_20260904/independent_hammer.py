#!/usr/bin/env python3
"""Independent source-only hammer for M2096; synthetic state only.

This checker never invokes execute(), any reducer, or any production-result,
shard, payload, GPU, or EDA path.  Its only dynamic authority probe builds a
temporary synthetic M2097/M2098 pair and asks the frozen future-gate parser
whether that pair is accepted without predecessor result-hammer evidence.
"""
from __future__ import print_function

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / (
    "system_simulator/scripts/run_m2096_ep34_decoder_d0_8700_shard_"
    "one_shot_reducer_publisher.py")
CONTRACT = HW / (
    "contracts/m2096_ep34_decoder_d0_8700_shard_one_shot_reducer_"
    "publisher_source_contract_r1_20260904.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED_SOURCE_SHA = (
    "7138800e32493e90bc1f1c8c4c56c52678397e6b6d54ea8ffabf028c8b5bd272")
EXPECTED_CONTRACT_SHA = (
    "68d3137d0e9617cf4f12e2ddcc025f410d2cd858a04b650107d64d79f9efbebd")
EXPECTED_DOCS359_SHA = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")


def load_target():
    spec = importlib.util.spec_from_file_location("m2097_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_names(function_node):
    rows = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            cursor = node.func
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
            name = ".".join(reversed(parts))
        else:
            name = "<dynamic>"
        rows.append({"name": name, "line": node.lineno})
    return sorted(rows, key=lambda row: (row["line"], row["name"]))


def synthetic_gate_without_predecessor_result_hammers(M):
    """Prove the runtime gate accepts no M2090/M2093 result-hammer identity."""
    with tempfile.TemporaryDirectory(prefix="m2097_synthetic_gate_") as temp:
        root = Path(temp)
        review_dir = root / "review"
        review_dir.mkdir()
        review_json = review_dir / "review.json"
        review_json.write_text("{}\n", encoding="ascii")
        release_path = root / "release.json"
        release_path.write_text("{}\n", encoding="ascii")
        old_review = M.FUTURE_REVIEW
        old_release = M.FUTURE_RELEASE
        old_tree = M.B.verify_sealed_tree
        old_json = M.B.strict_json
        old_double = M.B.verify_double_sealed_file
        try:
            M.FUTURE_REVIEW = review_dir
            M.FUTURE_RELEASE = release_path
            identity = M._identity()
            seal = {"manifest_sha256": "1" * 64,
                    "outer_file_sha256": "2" * 64}
            review_row = {
                "status": M.REVIEW_STATUS,
                "score_over_100": 100,
                "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
                "identity": identity,
                "authorization": {"m2098_release_authoring": 1,
                    "reducer_execution": 0, "payload_open": 0,
                    "shard_execution": 0}}
            release_identity = dict(identity,
                review_sha256=M.sha256(review_json),
                review_manifest_sha256=seal["manifest_sha256"],
                review_outer_file_sha256=seal["outer_file_sha256"])
            release_row = {
                "schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
                "identity": release_identity,
                "authorization": {"detached_launcher_runs": 1,
                    "outer_attempt_writes": 1, "reducer_runs": 1,
                    "sealed_shard_receipt_reads": M.TOTAL_SHARDS,
                    "payload_opens": 0, "shard_runs": 0,
                    "automatic_retry": False, "gpu_runs": 0,
                    "eda_runs": 0},
                "reducer": {
                    "implementation": "exact M1704.reduce_complete_sealed_shards",
                    "strong_verifier": "exact M1688.verify_sealed_shard",
                    "required_shards": M.TOTAL_SHARDS,
                    "ratio_policy": "integer ratio-of-sums"},
                "claim_boundary": {
                    "d0_candidate_pending_independent_hammer": True,
                    "monolithic_full_call": False, "full_decoder": False,
                    "system_speedup": False, "paper_result": False}}

            def fake_json(path):
                return (review_row if Path(path) == review_json
                        else release_row)

            M.B.verify_sealed_tree = lambda *args, **kwargs: seal
            M.B.strict_json = fake_json
            M.B.verify_double_sealed_file = (
                lambda *args, **kwargs: "3" * 64)
            accepted_sha = M._validate_future_gate()

            altered_review = copy.deepcopy(review_row)
            altered_review["severity_counts"]["p1"] = 1
            review_row.clear()
            review_row.update(altered_review)
            severity_rejected = False
            try:
                M._validate_future_gate()
            except M.M2096Error:
                severity_rejected = True
            return {
                "gate_accepted_without_m2090_result_hammer":
                    accepted_sha == "3" * 64,
                "gate_accepted_without_m2093_result_hammer":
                    accepted_sha == "3" * 64,
                "synthetic_review_p1_rejected": severity_rejected,
                "production_results_opened": 0,
                "production_shard_receipts_opened": 0,
                "reducer_runs": 0,
                "payload_opens": 0}
        finally:
            M.FUTURE_REVIEW = old_review
            M.FUTURE_RELEASE = old_release
            M.B.verify_sealed_tree = old_tree
            M.B.strict_json = old_json
            M.B.verify_double_sealed_file = old_double


def main():
    M = load_target()
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(SOURCE))
    functions = dict((node.name, node) for node in tree.body
                     if isinstance(node, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)))
    execute_calls = call_names(functions["execute"])
    locations = dict((row["name"], row["line"]) for row in execute_calls)
    output = {
        "schema": "m2097_m2096_source_hammer_mechanical_output_r1_v1",
        "identity": {
            "source_sha256": M.sha256(SOURCE),
            "contract_sha256": M.sha256(CONTRACT),
            "docs359_sha256": M.sha256(DOCS359)},
        "identity_exact": {
            "source": M.sha256(SOURCE) == EXPECTED_SOURCE_SHA,
            "contract": M.sha256(CONTRACT) == EXPECTED_CONTRACT_SHA,
            "docs359": M.sha256(DOCS359) == EXPECTED_DOCS359_SHA},
        "static": {
            "total_shards": M.TOTAL_SHARDS,
            "m1704_m1688_exact_edge": (
                M.M1704.reduce_complete_sealed_shards.__module__ ==
                M.M1704.__name__ and
                M.M1704.M1688.reduce_complete_sealed_shards.__module__ ==
                M.M1704.M1688.__name__),
            "outer_attempt_before_predecessor_reads": (
                locations["_consume_attempt"] <
                locations["_verify_predecessor_successes"]),
            "outer_attempt_before_reducer": (
                locations["_consume_attempt"] <
                locations["M1704.reduce_complete_sealed_shards"]),
            "success_rename_noreplace": text.count(
                "_rename_noreplace(WORK, RESULT)") == 1,
            "failure_rename_noreplace": text.count(
                "_rename_noreplace(WORK, FAILURE)") == 1,
            "m2090_raw_pending_true_is_required": text.count(
                'row.get("independent_result_hammer_pending") is True') >= 2,
            "predecessor_result_hammer_paths_referenced": any(token in text
                for token in ("M2090_RESULT_HAMMER", "M2093_RESULT_HAMMER",
                              "m2099_m2093", "result_hammer_review")),
            "execute_call_order": execute_calls},
        "synthetic_authority_attack":
            synthetic_gate_without_predecessor_result_hammers(M),
        "review_actions": {"execute_called": False,
            "production_result_opened": False,
            "production_shard_opened": False, "payload_opened": False,
            "reducer_run": False, "shard_run": False,
            "gpu_run": False, "eda_run": False}}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
