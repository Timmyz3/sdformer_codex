#!/opt/anaconda3/envs/python310/bin/python3.10
"""M579 r3 identity-stable wrapper around the frozen M586 r2 replay.

R3 changes no support arithmetic, parent selection, M505 recurrence, cost
constant, task order, accuracy evidence or capacity coordinate.  It closes
only the source-publication findings from the independent M592 hammer:

* the execution-contract bytes are SHA-bound at launch and must be identical
  at validation, production completion and terminal publication;
* the exact required execution-input key/path/SHA set is enforced rather than
  accepting an arbitrary set of merely declared inputs;
* the terminal receipt binds the runner and result to the same launch
  contract identity.

The companion M594 runner owns attempt/quarantine atomicity.  This analyzer
remains support/cycle only; arithmetic, cycle and trained-activity ratios must
never be multiplied into a system, RTL, PPA, energy or headline result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
R2_PATH = ROOT / "system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r2.py"
R2_SHA256 = "70eb07465bb008569967f69ae0ea0d51057d64dd0d51669b604a8f1cd4d4b471"
R3_RUNNER_PATH = ROOT / "system_simulator/scripts/run_m594_m579_paft_control_single_port_product_capture_r3_exact_sha.sh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_r2() -> Any:
    require(R2_PATH.is_file(), "missing frozen M586 r2 analyzer")
    require(sha256_file(R2_PATH) == R2_SHA256, "M586 r2 analyzer SHA drift")
    spec = importlib.util.spec_from_file_location("m594_m579_frozen_r2", str(R2_PATH))
    require(spec is not None and spec.loader is not None, "cannot import frozen M586 r2")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R2 = load_frozen_r2()
# Frozen r2's analyzer-identity check must refer to the actually executed r3
# wrapper, exactly as r2 redirected the frozen r1 identity to itself.
R2.__file__ = str(Path(__file__).resolve())
BASE_STRICT_JSON = R2.strict_json
BASE_VALIDATE_EXECUTION_CONTRACT = R2.validate_execution_contract
BASE_POSTPROCESS_OUTPUT = R2.postprocess_output


def worker_init() -> None:
    """Spawn-importable r3 entrypoint retaining the exact r2 initializer."""

    R2.worker_init()


def spawn_probe() -> dict[str, Any]:
    """Spawn-importable r3 entrypoint retaining the exact r2 probe."""

    return R2.spawn_probe()


def analyze_record(job: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    """Spawn-importable r3 entrypoint retaining exact r2 record arithmetic."""

    return R2.analyze_record(job)


# ProcessPool pickles these r3 top-level entrypoints.  Their bodies are pure
# delegations to the exact-SHA r2 implementation.
R2.R1.worker_init = worker_init
R2.R1.analyze_record = analyze_record


HISTORICAL_REQUIRED_INPUTS = {
    "paft_trace": {
        "path": "system_handoff/incoming/m248_paft_ep4_running_bn_bottleneck_sources_s10_r1_20260825/m248_paft_ep4_running_bn_bottleneck_source_manifest.json",
        "sha256": "6ba74414093edc1bf7d165b8904d8ac68bfdcdb3a49151203932e5c3aea92b0b",
    },
    "control_trace": {
        "path": "system_handoff/incoming/m252_control_ep4_running_bn_bottleneck_sources_s10_r1_20260825/m252_control_ep4_running_bn_bottleneck_source_manifest.json",
        "sha256": "2b806bb9faa1e458bc207a4f3002c730017651145e6d555d1e14e4a2a1c2a59c",
    },
    "paired_valid825": {
        "path": "results/m247_paft_vs_control_paired_valid825_r1_20260825/m247_paft_vs_control_paired_valid825_r1.json",
        "sha256": "1a3469117656499229b4cde3be7a2dc6b36d50da21a2ee4d72d067d011fc07cb",
    },
    "m43_support_unpacker": {
        "path": "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py",
        "sha256": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    },
    "m504_recurrence_dependency": {
        "path": "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py",
        "sha256": "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e",
    },
    "m505_recurrence": {
        "path": "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py",
        "sha256": "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced",
    },
    "m579_r1_worker_base": {
        "path": "system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture.py",
        "sha256": "4b990906fa76543cbbccb9d244a26974914902e0b1ad546d1ad197e7edbaf1ee",
    },
    "m586_r2_analyzer_dependency": {
        "path": "system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r2.py",
        "sha256": R2_SHA256,
    },
    "m528_result_hammer": {
        "path": "reviews/m528_r4_result_hammer_r1_20260827/review.json",
        "sha256": "4f70610dcb5c0778fd7874b8f70239f9139c5f98732ae439ab246129ede53d6e",
    },
    "m528_result_json": {
        "path": "results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json",
        "sha256": "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1",
    },
    "m528_capacity_ledger": {
        "path": "results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_capacity_ledger_r1.csv",
        "sha256": "7a5d2e4e52b172f0feb1450a447ba52fa0e520cc947f68df8fd8d359c8d95cd2",
    },
    "m255_paft_control_hammer": {
        "path": "results/m255_m254_independent_hammer_r1_20260825/m255_m254_independent_hammer_r1.json",
        "sha256": "f311f0958c9d7573362ea3a91d4cfe881d7474e207a519e33552814e2a864c5f",
    },
    "docs359": {
        "path": "docs/359_DATE终局冻结_20260813.md",
        "sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    },
}

RUNTIME_INPUT_PATHS = {
    "m594_r3_analyzer": "system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r3.py",
    "m594_r3_runner": "system_simulator/scripts/run_m594_m579_paft_control_single_port_product_capture_r3_exact_sha.sh",
}

REQUIRED_INPUT_KEYS = frozenset(HISTORICAL_REQUIRED_INPUTS) | frozenset(RUNTIME_INPUT_PATHS)
ACTIVE_CONTRACT: Path | None = None
EXPECTED_CONTRACT_SHA256: str | None = None
EXPECTED_RUNNER_SHA256: str | None = None


def require_sha256(value: str, label: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        label + " must be lowercase SHA256",
    )


def strict_json_adapter(path: Path) -> dict[str, Any]:
    """Present the v3 execution contract as v2/v1 only to frozen base code."""

    data = BASE_STRICT_JSON(path)
    if ACTIVE_CONTRACT is not None and Path(path).resolve() == ACTIVE_CONTRACT:
        require(
            data["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v3",
            "M579 r3 execution schema drift",
        )
        data = copy.deepcopy(data)
        data["schema"] = "m579_paft_control_single_port_product_capture_execution_contract_v2"
    return data


# The frozen r2 validator and r2->r1 schema bridge both resolve this global.
R2.strict_json = strict_json_adapter


def strict_json_for_frozen_r1(path: Path) -> dict[str, Any]:
    """Present the v3 contract directly as v1 to the frozen production core."""

    data = BASE_STRICT_JSON(path)
    if ACTIVE_CONTRACT is not None and Path(path).resolve() == ACTIVE_CONTRACT:
        require(
            data["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v3",
            "M579 r3 execution schema drift at frozen r1 boundary",
        )
        data = copy.deepcopy(data)
        data["schema"] = "m579_paft_control_single_port_product_capture_execution_contract_v1"
    return data


R2.R1.strict_json = strict_json_for_frozen_r1


def validate_required_inputs(contract: dict[str, Any]) -> None:
    inputs = contract["inputs"]
    require(set(inputs) == REQUIRED_INPUT_KEYS, "execution input key set drift")
    for name, expected in HISTORICAL_REQUIRED_INPUTS.items():
        require(inputs[name] == expected, "frozen execution input drift: " + name)

    analyzer = Path(__file__).resolve()
    runner = R3_RUNNER_PATH.resolve()
    require(analyzer.is_file(), "missing M594 r3 analyzer")
    require(runner.is_file(), "missing M594 r3 runner")
    require(
        inputs["m594_r3_analyzer"]["path"] == RUNTIME_INPUT_PATHS["m594_r3_analyzer"],
        "M594 r3 analyzer input path drift",
    )
    require(
        inputs["m594_r3_analyzer"]["sha256"] == sha256_file(analyzer),
        "M594 r3 analyzer input SHA drift",
    )
    require(
        inputs["m594_r3_runner"]["path"] == RUNTIME_INPUT_PATHS["m594_r3_runner"],
        "M594 r3 runner input path drift",
    )
    require(
        inputs["m594_r3_runner"]["sha256"] == sha256_file(runner),
        "M594 r3 runner input SHA drift",
    )
    require(
        contract["analyzer_sha256"] == inputs["m594_r3_analyzer"]["sha256"],
        "top-level/analyzer input SHA mismatch",
    )
    require(
        contract["runner_sha256"] == inputs["m594_r3_runner"]["sha256"],
        "top-level/runner input SHA mismatch",
    )


def validate_execution_contract(
    contract_path: Path,
    expected_contract_sha256: str,
    expected_runner_sha256: str,
) -> dict[str, Any]:
    global ACTIVE_CONTRACT

    contract_path = contract_path.resolve()
    require_sha256(expected_contract_sha256, "expected contract SHA")
    require_sha256(expected_runner_sha256, "expected runner SHA")
    require(sha256_file(contract_path) == expected_contract_sha256, "execution contract changed")
    contract = BASE_STRICT_JSON(contract_path)
    require(
        contract["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v3",
        "execution contract schema drift",
    )
    validate_required_inputs(contract)
    require(contract["runner_sha256"] == expected_runner_sha256, "launch runner SHA drift")

    ACTIVE_CONTRACT = contract_path
    R2.ACTIVE_CONTRACT = contract_path
    validated = BASE_VALIDATE_EXECUTION_CONTRACT(contract_path)
    require(sha256_file(contract_path) == expected_contract_sha256, "contract changed during validation")
    require(sha256_file(R3_RUNNER_PATH) == expected_runner_sha256, "runner changed during validation")
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
    require(
        sha256_file(ACTIVE_CONTRACT) == expected_contract_sha256,
        "contract changed before r3 result binding",
    )
    require(sha256_file(R3_RUNNER_PATH) == expected_runner_sha256, "runner changed before result binding")
    require(set(payload["identity"]) == REQUIRED_INPUT_KEYS, "result input identity set drift")
    for name in REQUIRED_INPUT_KEYS:
        require(payload["identity"][name] == contract["inputs"][name], "result input identity drift: " + name)

    payload["schema"] = "m579_paft_control_single_port_product_capture_v3"
    payload["execution_contract_identity"] = {
        "path": str(ACTIVE_CONTRACT),
        "sha256_start": expected_contract_sha256,
        "bytes_stable_through_result_binding": True,
        "required_input_keys": sorted(REQUIRED_INPUT_KEYS),
        "required_input_count": len(REQUIRED_INPUT_KEYS),
    }
    payload["runner_identity"] = {
        "path": RUNTIME_INPUT_PATHS["m594_r3_runner"],
        "sha256_start": expected_runner_sha256,
        "bytes_stable_through_result_binding": True,
    }
    payload["claim_boundary"].update({
        "m594_r3_identity_wrapper_only": True,
        "support_arithmetic_changed_from_m586_r2": False,
        "accuracy_performance_pareto": False,
        "ratios_may_be_multiplied": False,
        "system_speedup": False,
        "headline": False,
    })

    result_r3 = output_dir / "m579_paft_control_single_port_product_capture_r3.json"
    temporary = output_dir / ".m579_r3_result.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_r3)
    result_r2.unlink()
    csv_r2 = output_dir / "m579_per_sample_cycles_r2.csv"
    require(csv_r2.is_file(), "missing r2 sample CSV during r3 binding")
    csv_r2.replace(output_dir / "m579_per_sample_cycles_r3.csv")
    return result_r3


def terminal_rehash(
    contract_path: Path,
    output_dir: Path,
    expected_contract_sha256: str,
    expected_runner_sha256: str,
) -> dict[str, Any]:
    validated = validate_execution_contract(
        contract_path,
        expected_contract_sha256,
        expected_runner_sha256,
    )
    result = output_dir / "m579_paft_control_single_port_product_capture_r3.json"
    sample_csv = output_dir / "m579_per_sample_cycles_r3.csv"
    require(result.is_file() and sample_csv.is_file(), "terminal r3 staging output incomplete")
    payload = BASE_STRICT_JSON(result)
    require(payload["schema"] == "m579_paft_control_single_port_product_capture_v3", "terminal schema drift")
    require(
        payload["execution_contract_identity"]["sha256_start"] == expected_contract_sha256,
        "result/launch contract SHA mismatch",
    )
    require(
        payload["runner_identity"]["sha256_start"] == expected_runner_sha256,
        "result/launch runner SHA mismatch",
    )
    require(
        payload["execution_contract_identity"]["required_input_keys"]
        == sorted(REQUIRED_INPUT_KEYS),
        "result required-input set drift",
    )
    require(set(payload["identity"]) == REQUIRED_INPUT_KEYS, "terminal result input set drift")
    for name in REQUIRED_INPUT_KEYS:
        require(
            payload["identity"][name] == validated["contract"]["inputs"][name],
            "terminal result identity drift: " + name,
        )
    require(payload["task_order"]["order"] == "sample_operator_row_chunk_partition", "terminal order drift")
    require(payload["accuracy_limitations"]["accuracy_performance_pareto"] is False, "terminal Pareto drift")
    require(
        payload["accuracy_scope"]["full_hardware_trace_sequence"]["direction"] == "PAFT_WORSE",
        "terminal full-sequence disclosure drift",
    )
    require(sha256_file(contract_path) == expected_contract_sha256, "contract changed at terminal exit")
    require(sha256_file(R3_RUNNER_PATH) == expected_runner_sha256, "runner changed at terminal exit")
    return {
        "schema": "m594_m579_r3_terminal_rehash_receipt_v1",
        "status": "PASS_TERMINAL_SAME_CONTRACT_ALL_REQUIRED_INPUTS_AND_80_PAYLOAD_REHASH",
        "contract_sha256_start": expected_contract_sha256,
        "contract_sha256_terminal": sha256_file(contract_path),
        "contract_bytes_unchanged": True,
        "analyzer_sha256_terminal": sha256_file(Path(__file__).resolve()),
        "runner_sha256_start": expected_runner_sha256,
        "runner_sha256_terminal": sha256_file(R3_RUNNER_PATH),
        "runner_bytes_unchanged": True,
        "result_sha256": sha256_file(result),
        "sample_csv_sha256": sha256_file(sample_csv),
        "required_input_keys": sorted(REQUIRED_INPUT_KEYS),
        "required_inputs_rehashed": len(REQUIRED_INPUT_KEYS),
        "packed_payloads_rehashed": (
            validated["paft"]["payload_hashes_rechecked"]
            + validated["control"]["payload_hashes_rechecked"]
        ),
        "docs359_sha256": sha256_file(validated["paths"]["docs359"]),
        "task_order": validated["task_order"]["order"],
        "accuracy_performance_pareto": False,
    }


def preflight_only() -> int:
    runtime = R2.verify_runtime()
    order = R2.task_order_self_test()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=worker_init) as pool:
        probe = pool.submit(spawn_probe).result(timeout=60)
    print(json.dumps({
        "schema": "m594_m579_r3_preflight_v1",
        "status": "PASS_LIGHTWEIGHT_IMPORT_SPAWN_RECURRENCE_ONLY",
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
    contract_path = args.contract.resolve()
    validated = validate_execution_contract(
        contract_path,
        args.expected_contract_sha256,
        args.expected_runner_sha256,
    )

    if args.validate_contract_only:
        require(args.output_dir is None and not args.terminal_rehash,
                "contract validation takes no output and is exclusive")
        print(json.dumps({
            "schema": "m594_m579_r3_contract_preflight_v1",
            "status": "PASS_SAME_EXECUTION_CONTRACT_EXACT_REQUIRED_INPUTS_AND_80_PAYLOADS",
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
    output_dir = args.output_dir.resolve()
    if args.terminal_rehash:
        receipt = terminal_rehash(
            contract_path,
            output_dir,
            args.expected_contract_sha256,
            args.expected_runner_sha256,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0

    global ACTIVE_CONTRACT, EXPECTED_CONTRACT_SHA256, EXPECTED_RUNNER_SHA256
    ACTIVE_CONTRACT = contract_path
    EXPECTED_CONTRACT_SHA256 = args.expected_contract_sha256
    EXPECTED_RUNNER_SHA256 = args.expected_runner_sha256
    R2.ACTIVE_CONTRACT = contract_path
    R2.R1.__file__ = str(Path(__file__).resolve())
    require(not output_dir.exists(), "refuse to overwrite M579 r3 output")
    saved_argv = list(sys.argv)
    try:
        sys.argv = [
            str(Path(__file__).resolve()),
            "--contract", str(contract_path),
            "--output-dir", str(output_dir),
            "--workers", str(args.workers),
        ]
        rc = int(R2.R1.main())
    finally:
        sys.argv = saved_argv
    require(rc == 0, "frozen r1 worker returned nonzero")
    result = postprocess_output(output_dir, validated)
    require(sha256_file(contract_path) == args.expected_contract_sha256,
            "contract changed after r3 production")
    require(sha256_file(R3_RUNNER_PATH) == args.expected_runner_sha256,
            "runner changed after r3 production")
    require(sha256_file(Path(__file__).resolve()) == validated["contract"]["analyzer_sha256"],
            "r3 analyzer changed during production")
    print("PASS M579_R3 staging_result={}".format(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
