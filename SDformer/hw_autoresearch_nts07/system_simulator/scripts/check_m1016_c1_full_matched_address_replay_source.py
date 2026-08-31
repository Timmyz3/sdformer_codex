#!/usr/bin/env python3
"""Static/source checker for M1016. Never runs production replay or EDA."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ENGINE = HERE / "run_m1016_c1_full_matched_address_replay.py"
ENGINE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
RUNNER = HERE / "run_m1016_c1_full_matched_address_replay_one_shot.sh"
RUNNER_SHA = "e11de2a48e87700aeb927a837c3fb50605bda3fa4020d24c58d702c6c622a54e"
TEST = HW / "system_simulator/tests/test_m1016_c1_full_matched_address_replay_source.py"
CONTRACT = HW / "contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
M1010 = HW / "reviews/m1010_m1007_c1_matched_common_charge_address_replay_source_hammer_r1_20260829"
M1010_ID = (
    "c74812b03ca17b698ec5f80d086427937aea312668fd8d34df35544a930d669e",
    "5bc8ea19bfb658cf737e227d632461a21096d5035efad8e88a20fc5cdb704e27",
    "4885bee6283a09551fa5f95088a01683ce2b561e9305a33365ad807bfeb618f7",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory, identity):
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) == identity, "M1010 seal drift")
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        require(sha(directory / name.lstrip("*")) == expected, "M1010 member drift")


def load_engine():
    require(sha(ENGINE) == ENGINE_SHA and sha(RUNNER) == RUNNER_SHA,
            "M1016 engine/runner drift")
    spec = importlib.util.spec_from_file_location("m1016_engine_checked", ENGINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_static(contract=CONTRACT):
    verify_flat(M1010, M1010_ID)
    source = ENGINE.read_text()
    runner = RUNNER.read_text()
    tree = ast.parse(source)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    require(not imported.intersection({"subprocess", "socket", "requests", "urllib"}),
            "execution/network import in engine")
    require("--coverage-complete" not in source and "coverage_complete" not in source and
            "COVERAGE_COMPLETE" not in runner, "naked coverage override survived")
    for token in ("RAW_ROWS = SAMPLES * OPERATORS * PARTITIONS * ROWS_PER_PHASE",
                  "unique_tiles", "all_17280_phases_have_3000_rows",
                  "all_designs_have_6497280_blocks",
                  "all_three_service_merges_finished", "service_digests_equal",
                  "parent_conservation", "caller_supplied_coverage\": False"):
        require(token in source, "derived coverage gate missing: " + token)
    for token in ("M1016_RELEASE_JSON", "M1016_RELEASE_HAMMER_DIR",
                  "M1016_EXPECTED_RUNNER_SHA256", "M1016_EXPECTED_CONTRACT_SHA256",
                  "PASS_M1016_FULL_REPLAY_RELEASE_HAMMER", "max_attempts"):
        require(token in runner, "future one-shot gate missing: " + token)
    for forbidden in ("/opt/synopsys", "dc_shell", "pt_shell", "nvidia-smi", "ssh "):
        require(forbidden not in runner.lower(), "forbidden tool in CPU runner: " + forbidden)
    module = load_engine()
    require(inspect.isgeneratorfunction(module.iter_parent_address_events),
            "parent address path not streaming")
    require(module.RAW_ROWS == 51_840_000 and module.TASKS == 812_160 and
            module.BLOCK_TASKS == 6_497_280 and module.PHASES == 17_280,
            "frozen geometry drift")
    oracle = module.small_oracle()
    require(oracle["empty_coverage_rejected"] and oracle["tiny_coverage_rejected"] and
            not oracle["capacity_admitted"] and not oracle["speedup_admitted"],
            "M1016 small oracle fail-open")
    value = strict_json(contract)
    require(value["status"] == "PASS_M1016_SOURCE_ONLY__NO_FULL_REPLAY_NO_EDA" and
            value["launch_now"] is False, "M1016 source contract state drift")
    identity = value["source_identity"]
    require(identity["engine"]["sha256"] == sha(ENGINE) and
            identity["runner"]["sha256"] == sha(RUNNER) and
            identity["tests"]["sha256"] == sha(TEST), "source identity drift")
    boundary = value["claim_boundary"]
    require(all(boundary[key] is False for key in
                ("full_replay_executed", "result_created", "capacity_only_214912B_admitted",
                 "matched_cycles_admitted", "speedup_admitted", "vcs_executed",
                 "dc_executed", "pt_executed", "ptpx_executed", "gpu_remote_used")),
            "M1016 false source claim")
    require(not (HW / "results/m1016_m1010_c1_full_matched_address_replay_r1_20260829").exists() and
            not (HW / "results/.m1016_m1010_c1_full_matched_address_replay_attempt_consumed").exists(),
            "M1016 source namespace consumed")
    return {
        "schema": "m1016_c1_full_matched_address_replay_source_check_v1",
        "status": "PASS_M1016_FULL_REPLAY_SOURCE_CHECK__NO_EXECUTION",
        "engine_sha256": sha(ENGINE), "runner_sha256": sha(RUNNER),
        "test_sha256": sha(TEST), "contract_sha256": sha(contract),
        "tests_mode": "small_only", "coverage_cli_override": False,
        "empty_and_tiny_coverage_rejected": True,
        "full_51840000_replayed": False, "eda_gpu_remote_used": False,
        "capacity_only_214912B_admitted": False, "speedup_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    print(json.dumps(validate_static(args.contract), sort_keys=True))


if __name__ == "__main__":
    main()
