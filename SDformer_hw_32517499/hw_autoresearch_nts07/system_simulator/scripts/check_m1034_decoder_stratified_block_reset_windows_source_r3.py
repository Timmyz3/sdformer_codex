#!/usr/bin/env python3
"""Static source-only checker for M1034 r3."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "analyze_m1034_decoder_stratified_block_reset_windows_source_r3.py"
TEST = HW / "system_simulator/tests/test_m1034_decoder_stratified_block_reset_windows_source_r3.py"
CONTRACT = HW / "contracts/m1034_decoder_stratified_block_reset_windows_source_r3_contract_r1_20260829.json"
BASE = HERE / "analyze_m1023_decoder_stratified_block_reset_windows_source_r2.py"
BASE_SHA256 = "8e9ce843499cbcfdfe1856e5f829218e0329cd299ce25d1ba93e3b45cd74d2b2"
SCHEMA = "m1034_decoder_stratified_block_reset_windows_source_r3_v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def static_check(contract_path=CONTRACT):
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    require(contract["schema"] == SCHEMA and
            contract["status"] == "R3_SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "contract drift")
    require(sha256(BASE) == BASE_SHA256, "M1023 r2 changed")
    for name, path in (("source", SOURCE), ("checker", Path(__file__).resolve()),
                       ("test", TEST)):
        require(sha256(path) == contract["source_identity"][name]["sha256"],
                name + " identity drift")
    text = SOURCE.read_text(encoding="utf-8")
    functions = {node.name for node in ast.walk(ast.parse(text))
                 if isinstance(node, ast.FunctionDef)}
    require({"_coverage_projection", "_walk_numeric_paths",
             "validate_publication_envelope", "publication_projection",
             "estimate_paired_totals", "self_test"} <= functions,
            "recursive publication function absent")
    require('points = None' in text and
            '"candidate_mean_cycles" not in row' in text and
            'set(row) == COVERAGE_ROW_KEYS' in text and
            'numeric value outside recursive publication allowlist' in text,
            "hard-stop recursive guard absent")
    require("BASE.BASE.estimate_paired_totals(" in text,
            "raw estimator is not projected once at publication boundary")
    forbidden = ("M946.prefix_transactions(",
                 "M896.real_prefix_transactions(",
                 "iter_record_transactions(", "--run-real", "--full-row",
                 "--production", "--output")
    require(not any(token in text for token in forbidden),
            "real execution surface present")
    require(text.count("add_argument(") == 2, "CLI expanded")
    require(all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return {
        "status": "PASS_M1034_R3_SOURCE_STATIC_CHECK__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "source_sha256": sha256(SOURCE),
        "checker_sha256": sha256(Path(__file__)),
        "test_sha256": sha256(TEST),
        "m1023_r2_unchanged_sha256": sha256(BASE),
        "launch_now": False,
        "real_payload_execution": False,
        "eda_gpu_remote_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    require(args.check, "select --check")
    print(json.dumps(static_check(), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
