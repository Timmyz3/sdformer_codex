#!/opt/anaconda3/envs/python310/bin/python3.10
"""M579 r4 mechanical symlink-hardening wrapper over frozen M594 r3.

R4 changes no support arithmetic, parent choice, M505 recurrence, task order,
payload identity, accuracy evidence, capacity coordinate, decision threshold or
claim boundary.  It closes only M598-P2-01: every execution contract and
staging/result path visible to the analyzer is checked with lexists semantics
and symlinks are rejected.  The companion M601 runner applies the same rule to
result, attempt, consumed, staging and quarantine coordinates while retaining
RENAME_NOREPLACE publication.

This remains source/cycle support only.  Arithmetic-work, local-cycle and
trained-activity ratios must never be multiplied into system, RTL, PPA, energy
or headline results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
R3_PATH = ROOT / "system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r3.py"
R3_SHA256 = "c684ac4ddc4cbea46e1eca7b088c303d8b0cf3acf6284e2a98d66d6e83136fd2"
R4_RUNNER_PATH = ROOT / "system_simulator/scripts/run_m601_m579_paft_control_single_port_product_capture_r4_exact_sha.sh"
RESULT_REL = "results/m579_paft_control_single_port_product_capture_r4_20260828"
ATTEMPT_REL = "results/m579_paft_control_single_port_product_capture_r4_20260828.attempt"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def require_regular_nosymlink(path: Path, label: str) -> Path:
    require(path_lexists(path), "missing " + label)
    require(not path.is_symlink(), label + " must not be a symlink")
    require(path.is_file(), label + " must be a regular file")
    return path.resolve(strict=True)


def require_directory_nosymlink(path: Path, label: str) -> Path:
    require(path_lexists(path), "missing " + label)
    require(not path.is_symlink(), label + " must not be a symlink")
    require(path.is_dir(), label + " must be a directory")
    return path.resolve(strict=True)


def load_frozen_r3() -> Any:
    require_regular_nosymlink(R3_PATH, "frozen M594 r3 analyzer")
    require(sha256_file(R3_PATH) == R3_SHA256, "M594 r3 analyzer SHA drift")
    spec = importlib.util.spec_from_file_location("m601_m579_frozen_r3", str(R3_PATH))
    require(spec is not None and spec.loader is not None, "cannot import frozen M594 r3")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R3 = load_frozen_r3()
BASE_STRICT_JSON = R3.BASE_STRICT_JSON
BASE_VALIDATE_EXECUTION_CONTRACT = R3.BASE_VALIDATE_EXECUTION_CONTRACT
BASE_POSTPROCESS_OUTPUT = R3.BASE_POSTPROCESS_OUTPUT


def worker_init() -> None:
    """Spawn-importable r4 entrypoint retaining the exact r3 initializer."""

    R3.worker_init()


def spawn_probe() -> dict[str, Any]:
    """Spawn-importable r4 entrypoint retaining the exact r3 probe."""

    return R3.spawn_probe()


def analyze_record(job: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    """Spawn-importable r4 entrypoint retaining exact r3 record arithmetic."""

    return R3.analyze_record(job)


# The frozen production core pickles these top-level r4 entrypoints under
# spawn.  Their bodies are exact delegations to the exact-SHA r3/r2/r1 chain.
R3.R2.R1.worker_init = worker_init
R3.R2.R1.analyze_record = analyze_record
R3.R2.__file__ = str(Path(__file__).resolve())
R3.R2.R1.__file__ = str(Path(__file__).resolve())


HISTORICAL_REQUIRED_INPUTS = copy.deepcopy(R3.HISTORICAL_REQUIRED_INPUTS)
RUNTIME_INPUT_PATHS = {
    "m601_r4_analyzer": "system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r4.py",
    "m601_r4_runner": "system_simulator/scripts/run_m601_m579_paft_control_single_port_product_capture_r4_exact_sha.sh",
}
REQUIRED_INPUT_KEYS = frozenset(HISTORICAL_REQUIRED_INPUTS) | frozenset(RUNTIME_INPUT_PATHS)
ACTIVE_CONTRACT: Path | None = None


def require_sha256(value: str, label: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        label + " must be lowercase SHA256",
    )


def strict_json_for_frozen_r2(path: Path) -> dict[str, Any]:
    """Present v4 as v2 only to the frozen M586 validator/postprocessor."""

    data = BASE_STRICT_JSON(path)
    if ACTIVE_CONTRACT is not None and Path(path).resolve() == ACTIVE_CONTRACT:
        require(
            data["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v4",
            "M579 r4 execution schema drift at frozen r2 boundary",
        )
        data = copy.deepcopy(data)
        data["schema"] = "m579_paft_control_single_port_product_capture_execution_contract_v2"
    return data


def strict_json_for_frozen_r1(path: Path) -> dict[str, Any]:
    """Present v4 as v1 only to the frozen production core."""

    data = BASE_STRICT_JSON(path)
    if ACTIVE_CONTRACT is not None and Path(path).resolve() == ACTIVE_CONTRACT:
        require(
            data["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v4",
            "M579 r4 execution schema drift at frozen r1 boundary",
        )
        data = copy.deepcopy(data)
        data["schema"] = "m579_paft_control_single_port_product_capture_execution_contract_v1"
    return data


R3.R2.strict_json = strict_json_for_frozen_r2
R3.R2.R1.strict_json = strict_json_for_frozen_r1


def validate_required_inputs(contract: dict[str, Any]) -> None:
    inputs = contract["inputs"]
    require(set(inputs) == REQUIRED_INPUT_KEYS, "execution input key set drift")
    for name, expected in HISTORICAL_REQUIRED_INPUTS.items():
        require(inputs[name] == expected, "frozen execution input drift: " + name)

    analyzer = require_regular_nosymlink(Path(__file__), "M601 r4 analyzer")
    runner = require_regular_nosymlink(R4_RUNNER_PATH, "M601 r4 runner")
    for name, path in (("m601_r4_analyzer", analyzer), ("m601_r4_runner", runner)):
        require(inputs[name]["path"] == RUNTIME_INPUT_PATHS[name], name + " input path drift")
        require(inputs[name]["sha256"] == sha256_file(path), name + " input SHA drift")
    require(
        contract["analyzer_sha256"] == inputs["m601_r4_analyzer"]["sha256"],
        "top-level/analyzer input SHA mismatch",
    )
    require(
        contract["runner_sha256"] == inputs["m601_r4_runner"]["sha256"],
        "top-level/runner input SHA mismatch",
    )


def validate_output_coordinate(contract: dict[str, Any]) -> None:
    output = contract["output"]
    require(output["result_dir"] == RESULT_REL, "result coordinate drift")
    require(output["attempt_dir"] == ATTEMPT_REL, "attempt coordinate drift")
    for label, relative in (
        ("canonical result", RESULT_REL),
        ("canonical attempt", ATTEMPT_REL),
        ("consumed attempt", ATTEMPT_REL + ".consumed"),
    ):
        path = ROOT / relative
        require(not path.is_symlink(), label + " must not be a symlink")


def validate_execution_contract(
    contract_path_raw: Path,
    expected_contract_sha256: str,
    expected_runner_sha256: str,
) -> dict[str, Any]:
    global ACTIVE_CONTRACT

    contract_path = require_regular_nosymlink(contract_path_raw, "M601 r4 execution contract")
    require_sha256(expected_contract_sha256, "expected contract SHA")
    require_sha256(expected_runner_sha256, "expected runner SHA")
    require(sha256_file(contract_path) == expected_contract_sha256, "execution contract changed")
    contract = BASE_STRICT_JSON(contract_path)
    require(
        contract["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v4",
        "execution contract schema drift",
    )
    validate_required_inputs(contract)
    validate_output_coordinate(contract)
    require(contract["runner_sha256"] == expected_runner_sha256, "launch runner SHA drift")

    ACTIVE_CONTRACT = contract_path
    R3.ACTIVE_CONTRACT = contract_path
    R3.R2.ACTIVE_CONTRACT = contract_path
    validated = BASE_VALIDATE_EXECUTION_CONTRACT(contract_path)
    require(sha256_file(contract_path) == expected_contract_sha256, "contract changed during validation")
    require(sha256_file(R4_RUNNER_PATH) == expected_runner_sha256, "runner changed during validation")
    require(sha256_file(Path(__file__).resolve()) == contract["analyzer_sha256"],
            "r4 analyzer changed during validation")
    validated["contract"] = contract
    validated["contract_sha256_start"] = expected_contract_sha256
    validated["runner_sha256_start"] = expected_runner_sha256
    validated["required_input_keys"] = sorted(REQUIRED_INPUT_KEYS)
    return validated


def postprocess_output(output_dir: Path, validated: dict[str, Any]) -> Path:
    result_r2 = BASE_POSTPROCESS_OUTPUT(output_dir, validated)
    payload = BASE_STRICT_JSON(result_r2)
    contract = validated["contract"]
    expected_contract_sha256 = validated["contract_sha256_start"]
    expected_runner_sha256 = validated["runner_sha256_start"]
    require(sha256_file(ACTIVE_CONTRACT) == expected_contract_sha256,
            "contract changed before r4 result binding")
    require(sha256_file(R4_RUNNER_PATH) == expected_runner_sha256,
            "runner changed before r4 result binding")
    require(sha256_file(Path(__file__).resolve()) == contract["analyzer_sha256"],
            "r4 analyzer changed before result binding")
    require(set(payload["identity"]) == REQUIRED_INPUT_KEYS, "result input identity set drift")
    for name in REQUIRED_INPUT_KEYS:
        require(payload["identity"][name] == contract["inputs"][name],
                "result input identity drift: " + name)

    payload["schema"] = "m579_paft_control_single_port_product_capture_v4"
    payload["execution_contract_identity"] = {
        "path": str(ACTIVE_CONTRACT),
        "sha256_start": expected_contract_sha256,
        "bytes_stable_through_result_binding": True,
        "required_input_keys": sorted(REQUIRED_INPUT_KEYS),
        "required_input_count": len(REQUIRED_INPUT_KEYS),
    }
    payload["runner_identity"] = {
        "path": RUNTIME_INPUT_PATHS["m601_r4_runner"],
        "sha256_start": expected_runner_sha256,
        "bytes_stable_through_result_binding": True,
    }
    payload["mechanical_overlay"] = {
        "m594_r3_analyzer_sha256": R3_SHA256,
        "support_arithmetic_changed": False,
        "canonical_path_policy": "LEXISTS_AND_REJECT_SYMLINK_THEN_RENAME_NOREPLACE",
        "m598_p2_01_closed": True,
    }
    payload["claim_boundary"].update({
        "m601_r4_mechanical_symlink_overlay_only": True,
        "support_arithmetic_changed_from_m594_r3": False,
        "accuracy_performance_pareto": False,
        "ratios_may_be_multiplied": False,
        "system_speedup": False,
        "headline": False,
    })

    result_r4 = output_dir / "m579_paft_control_single_port_product_capture_r4.json"
    temporary = output_dir / ".m579_r4_result.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_r4)
    result_r2.unlink()
    csv_r2 = output_dir / "m579_per_sample_cycles_r2.csv"
    require_regular_nosymlink(csv_r2, "r2 sample CSV during r4 binding")
    csv_r2.replace(output_dir / "m579_per_sample_cycles_r4.csv")
    return result_r4


def terminal_rehash(
    contract_path: Path,
    output_dir_raw: Path,
    expected_contract_sha256: str,
    expected_runner_sha256: str,
) -> dict[str, Any]:
    output_dir = require_directory_nosymlink(output_dir_raw, "r4 staging output")
    validated = validate_execution_contract(
        contract_path,
        expected_contract_sha256,
        expected_runner_sha256,
    )
    result = require_regular_nosymlink(
        output_dir / "m579_paft_control_single_port_product_capture_r4.json",
        "terminal r4 result",
    )
    sample_csv = require_regular_nosymlink(
        output_dir / "m579_per_sample_cycles_r4.csv",
        "terminal r4 sample CSV",
    )
    payload = BASE_STRICT_JSON(result)
    require(payload["schema"] == "m579_paft_control_single_port_product_capture_v4",
            "terminal schema drift")
    require(payload["execution_contract_identity"]["sha256_start"] == expected_contract_sha256,
            "result/launch contract SHA mismatch")
    require(payload["runner_identity"]["sha256_start"] == expected_runner_sha256,
            "result/launch runner SHA mismatch")
    require(payload["execution_contract_identity"]["required_input_keys"]
            == sorted(REQUIRED_INPUT_KEYS), "result required-input set drift")
    require(set(payload["identity"]) == REQUIRED_INPUT_KEYS, "terminal result input set drift")
    for name in REQUIRED_INPUT_KEYS:
        require(payload["identity"][name] == validated["contract"]["inputs"][name],
                "terminal result identity drift: " + name)
    require(payload["task_order"]["order"] == "sample_operator_row_chunk_partition",
            "terminal order drift")
    require(payload["accuracy_limitations"]["accuracy_performance_pareto"] is False,
            "terminal Pareto drift")
    require(payload["accuracy_scope"]["full_hardware_trace_sequence"]["direction"] == "PAFT_WORSE",
            "terminal full-sequence disclosure drift")
    require(sha256_file(contract_path.resolve()) == expected_contract_sha256,
            "contract changed at terminal exit")
    require(sha256_file(R4_RUNNER_PATH) == expected_runner_sha256,
            "runner changed at terminal exit")
    require(sha256_file(Path(__file__).resolve()) == validated["contract"]["analyzer_sha256"],
            "analyzer changed at terminal exit")
    return {
        "schema": "m601_m579_r4_terminal_rehash_receipt_v1",
        "status": "PASS_TERMINAL_SAME_CONTRACT_15_INPUTS_80_PAYLOADS_LEXISTS_NOSYMLINK",
        "contract_sha256_start": expected_contract_sha256,
        "contract_sha256_terminal": sha256_file(contract_path.resolve()),
        "contract_bytes_unchanged": True,
        "analyzer_sha256_terminal": sha256_file(Path(__file__).resolve()),
        "runner_sha256_start": expected_runner_sha256,
        "runner_sha256_terminal": sha256_file(R4_RUNNER_PATH),
        "runner_bytes_unchanged": True,
        "result_sha256": sha256_file(result),
        "sample_csv_sha256": sha256_file(sample_csv),
        "required_input_keys": sorted(REQUIRED_INPUT_KEYS),
        "required_inputs_rehashed": len(REQUIRED_INPUT_KEYS),
        "packed_payloads_rehashed": (
            validated["paft"]["payload_hashes_rechecked"]
            + validated["control"]["payload_hashes_rechecked"]
        ),
        "canonical_path_policy": "LEXISTS_AND_REJECT_SYMLINK_THEN_RENAME_NOREPLACE",
        "docs359_sha256": sha256_file(validated["paths"]["docs359"]),
        "task_order": validated["task_order"]["order"],
        "accuracy_performance_pareto": False,
    }


def preflight_only() -> int:
    runtime = R3.R2.verify_runtime()
    order = R3.R2.task_order_self_test()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=worker_init) as pool:
        probe = pool.submit(spawn_probe).result(timeout=60)
    print(json.dumps({
        "schema": "m601_m579_r4_preflight_v1",
        "status": "PASS_LIGHTWEIGHT_IMPORT_SPAWN_RECURRENCE_LEXISTS_OVERLAY_ONLY",
        "runtime": runtime,
        "task_order": order,
        "probe": probe,
        "required_input_keys": sorted(REQUIRED_INPUT_KEYS),
        "formal_trace_records_processed": 0,
        "result_or_attempt_created": False,
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument("--terminal-rehash", action="store_true")
    args = parser.parse_args()

    if args.preflight_only:
        require(args.contract is None and args.output_dir is None, "preflight takes no formal paths")
        require(args.expected_contract_sha256 is None and args.expected_runner_sha256 is None,
                "preflight takes no launch identities")
        require(not args.terminal_rehash and not args.validate_contract_only,
                "preflight/validation/terminal modes are exclusive")
        return preflight_only()

    require(args.contract is not None, "formal validation requires contract")
    require(args.expected_contract_sha256 is not None, "missing launch contract SHA")
    require(args.expected_runner_sha256 is not None, "missing launch runner SHA")
    require(1 <= args.workers <= 3, "workers must be 1..3")
    validated = validate_execution_contract(
        args.contract,
        args.expected_contract_sha256,
        args.expected_runner_sha256,
    )

    if args.validate_contract_only:
        require(args.output_dir is None and not args.terminal_rehash,
                "contract validation takes no output and is exclusive")
        print(json.dumps({
            "schema": "m601_m579_r4_contract_preflight_v1",
            "status": "PASS_SAME_CONTRACT_15_INPUTS_80_PAYLOADS_LEXISTS_NOSYMLINK",
            "contract_sha256_start": args.expected_contract_sha256,
            "runner_sha256_start": args.expected_runner_sha256,
            "required_inputs_rehashed": len(REQUIRED_INPUT_KEYS),
            "packed_payloads_rehashed": (
                validated["paft"]["payload_hashes_rechecked"]
                + validated["control"]["payload_hashes_rechecked"]
            ),
            "formal_trace_records_processed": 0,
            "result_or_attempt_created": False,
        }, sort_keys=True))
        return 0

    require(args.output_dir is not None, "formal/terminal mode requires output-dir")
    if args.terminal_rehash:
        receipt = terminal_rehash(
            args.contract,
            args.output_dir,
            args.expected_contract_sha256,
            args.expected_runner_sha256,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0

    require(not path_lexists(args.output_dir), "refuse existing or dangling-symlink r4 output")
    output_dir = args.output_dir.resolve()
    global ACTIVE_CONTRACT
    ACTIVE_CONTRACT = args.contract.resolve(strict=True)
    R3.ACTIVE_CONTRACT = ACTIVE_CONTRACT
    R3.R2.ACTIVE_CONTRACT = ACTIVE_CONTRACT
    R3.R2.R1.__file__ = str(Path(__file__).resolve())
    saved_argv = list(sys.argv)
    try:
        sys.argv = [
            str(Path(__file__).resolve()),
            "--contract", str(ACTIVE_CONTRACT),
            "--output-dir", str(output_dir),
            "--workers", str(args.workers),
        ]
        rc = int(R3.R2.R1.main())
    finally:
        sys.argv = saved_argv
    require(rc == 0, "frozen r1 worker returned nonzero")
    result = postprocess_output(output_dir, validated)
    require(sha256_file(ACTIVE_CONTRACT) == args.expected_contract_sha256,
            "contract changed after r4 production")
    require(sha256_file(R4_RUNNER_PATH) == args.expected_runner_sha256,
            "runner changed after r4 production")
    require(sha256_file(Path(__file__).resolve()) == validated["contract"]["analyzer_sha256"],
            "r4 analyzer changed after production")
    print("PASS M579_R4 staging_result={}".format(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
