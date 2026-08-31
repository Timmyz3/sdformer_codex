#!/usr/bin/env python3
"""Fail-closed static checker for the M1014 source-only implementation."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "analyze_m1014_decoder_stratified_block_reset_windows_source.py"
TEST = HW / "system_simulator/tests/test_m1014_decoder_stratified_block_reset_windows_source.py"
CONTRACT = HW / "contracts/m1014_decoder_stratified_block_reset_windows_source_contract_r1_20260829.json"
SCHEMA = "m1014_decoder_stratified_block_reset_windows_source_v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def static_check(contract_path=CONTRACT):
    contract = strict_json(contract_path)
    require(contract["schema"] == SCHEMA and
            contract["status"] == "SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "contract identity drift")
    required = {
        "source": SOURCE,
        "checker": Path(__file__).resolve(),
        "test": TEST,
    }
    for name, path in required.items():
        require(path.is_file() and not path.is_symlink(), name + " absent")
        require(sha256(path) == contract["source_identity"][name]["sha256"],
                name + " hash drift")
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    require({"frozen_route", "classify_stratum", "deterministic_select",
             "block_reset_transactions", "exact_replay", "paired_replay",
             "estimate_paired_totals", "validate_source", "self_test"}
            <= functions, "required source function absent")
    forbidden_cli = ("--run-real", "--run-production", "--full-row",
                     "--window-run", "--launch", "--output")
    require(not any(token in text for token in forbidden_cli),
            "real execution CLI surface present")
    require(text.count("add_argument(") == 2 and
            'add_argument("--validate-source"' in text and
            'add_argument("--self-test"' in text,
            "CLI is not exactly two source-only modes")
    require("M946.prefix_transactions(" not in text and
            "M896.real_prefix_transactions(" not in text and
            "iter_record_transactions(" not in text,
            "source-only implementation contains real payload execution call")
    require("WINDOW_EXPANDED_REQUEST_CAP = 10000" in text and
            "PILOT_PER_STRATUM = 8" in text and
            "MAX_PER_STRATUM = 32" in text,
            "sampling bound drift")
    require('reject_d1(layer)' in text and
            '"D1_STRICT_COMMON_CHARGE_NO_GENERATOR_OR_SCHEDULER_CALL"' in text,
            "D1 fail-closed boundary absent")
    require("transaction_ratio_is_speedup\": False" in text,
            "transaction ratio claim guard absent")
    require(all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return {
        "status": "PASS_M1014_SOURCE_STATIC_CHECK__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "source_sha256": sha256(SOURCE),
        "checker_sha256": sha256(Path(__file__)),
        "test_sha256": sha256(TEST),
        "cli_modes": ["--validate-source", "--self-test"],
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
