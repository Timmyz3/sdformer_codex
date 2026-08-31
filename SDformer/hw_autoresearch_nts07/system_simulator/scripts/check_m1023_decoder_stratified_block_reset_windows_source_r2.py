#!/usr/bin/env python3
"""Static M1023 r2 checker; source-only and no real payload execution."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "analyze_m1023_decoder_stratified_block_reset_windows_source_r2.py"
TEST = HW / "system_simulator/tests/test_m1023_decoder_stratified_block_reset_windows_source_r2.py"
CONTRACT = HW / "contracts/m1023_decoder_stratified_block_reset_windows_source_r2_contract_r1_20260829.json"
M1014 = HERE / "analyze_m1014_decoder_stratified_block_reset_windows_source.py"
M1014_SHA256 = "c1fb987bd6d9921286fd9c53f3c9374d9c4779d9b3617946ab9b3d7ab11e2c64"
SCHEMA = "m1023_decoder_stratified_block_reset_windows_source_r2_v1"


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
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def static_check(contract_path=CONTRACT):
    contract = strict_json(contract_path)
    require(contract["schema"] == SCHEMA and
            contract["status"] == "R2_SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "contract identity drift")
    require(sha256(M1014) == M1014_SHA256, "M1014 r1 changed")
    for name, path in (("source", SOURCE), ("checker", Path(__file__).resolve()),
                       ("test", TEST)):
        require(path.is_file() and not path.is_symlink(), name + " absent")
        require(sha256(path) == contract["source_identity"][name]["sha256"],
                name + " hash drift")
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}
    require({"validate_metadata_row", "_semantic_field_scan",
             "deterministic_select", "_reset_service_semantics",
             "paired_replay", "_relative_halfwidth",
             "apply_ci_publication_gate", "estimate_paired_totals",
             "validate_source", "self_test"} <= functions,
            "required r2 repair function absent")
    require("METADATA_SCHEMA =" in source and
            "unknown pre-cycle metadata field" in source and
            "_semantic_field_scan(row)" in source,
            "strict recursive allowlist absent")
    require("creset == breset" in source and
            '"service_cycle_charge"' in source and
            '"return_distance_cycles"' in source and
            '"paired_reset_exact_equal": True' in source,
            "reset semantic/cycle equality absent")
    require("worst > 0.10" in source and "worst > 0.05" in source and
            source.count("_total_cycles_estimate\"] = None") == 2 and
            'output["paired_speedup_estimate"] = None' in source,
            "CI public hard-stop absent")
    forbidden_calls = ("M946.prefix_transactions(",
                       "M896.real_prefix_transactions(",
                       "iter_record_transactions(")
    require(not any(token in source for token in forbidden_calls),
            "real payload execution call present")
    forbidden_cli = ("--run-real", "--production", "--full-row",
                     "--window-run", "--launch", "--output")
    require(not any(token in source for token in forbidden_cli),
            "execution CLI present")
    require(source.count("add_argument(") == 2 and
            'add_argument("--validate-source"' in source and
            'add_argument("--self-test"' in source,
            "CLI surface expanded")
    require(all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return {
        "status": "PASS_M1023_R2_SOURCE_STATIC_CHECK__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "source_sha256": sha256(SOURCE),
        "checker_sha256": sha256(Path(__file__)),
        "test_sha256": sha256(TEST),
        "m1014_r1_unchanged_sha256": sha256(M1014),
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
