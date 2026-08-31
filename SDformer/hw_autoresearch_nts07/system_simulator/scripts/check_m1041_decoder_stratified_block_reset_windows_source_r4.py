#!/usr/bin/env python3
"""Static source-only checker for M1041 r4."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "analyze_m1041_decoder_stratified_block_reset_windows_source_r4.py"
TEST = HW / "system_simulator/tests/test_m1041_decoder_stratified_block_reset_windows_source_r4.py"
CONTRACT = HW / "contracts/m1041_decoder_stratified_block_reset_windows_source_r4_contract_r1_20260829.json"
BASE = HERE / "analyze_m1034_decoder_stratified_block_reset_windows_source_r3.py"
BASE_SHA256 = "155ebe3e19cb42e42afe3f26358f0598e8d33bad9558f450237cffc53eb4691a"
SCHEMA = "m1041_decoder_stratified_block_reset_windows_source_r4_v1"


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
            contract["status"] == "R4_SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "contract drift")
    require(sha256(BASE) == BASE_SHA256, "M1034 r3 changed")
    for name, path in (("source", SOURCE), ("checker", Path(__file__).resolve()),
                       ("test", TEST)):
        require(sha256(path) == contract["source_identity"][name]["sha256"],
                name + " identity drift")
    text = SOURCE.read_text(encoding="utf-8")
    functions = {node.name for node in ast.walk(ast.parse(text))
                 if isinstance(node, ast.FunctionDef)}
    require({"_walk_public_json", "_reject_semantic_point_keys",
             "_validate_bounds", "_validate_uncertainty",
             "_validate_coverage", "validate_publication_envelope",
             "publication_projection", "_m1035_attacks", "self_test"} <= functions,
            "strong value-shape functions absent")
    required = (
        "bound must be flat finite length-2 scalar interval",
        "uncertainty leaf must be finite scalar",
        "coverage population must be positive exact int",
        "coverage sample must be positive exact int within population",
        "semantic point key forbidden at depth",
        "len(rejected) == 11",
    )
    require(all(token in text for token in required),
            "M1035 strong-type repair surface absent")
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
        "status": "PASS_M1041_R4_SOURCE_STATIC_CHECK__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "source_sha256": sha256(SOURCE),
        "checker_sha256": sha256(Path(__file__).resolve()),
        "test_sha256": sha256(TEST),
        "m1034_r3_unchanged_sha256": sha256(BASE),
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
