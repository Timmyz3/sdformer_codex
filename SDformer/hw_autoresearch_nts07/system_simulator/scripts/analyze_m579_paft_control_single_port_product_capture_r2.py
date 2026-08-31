#!/opt/anaconda3/envs/python310/bin/python3.10
"""M579 r2 paired PAFT/control replay on the frozen M528 task stream.

R2 is a fail-closed wrapper around the byte-frozen r1 arithmetic worker.  It
repairs the execution boundary without changing the M43 unpacker or the M505
dead-write-only recurrence:

* an exact NumPy-capable Python 3.10 runtime is identity-bound;
* the r1 partition-major worker arrays are transposed into M528's frozen
  sample/operator/row-chunk/partition cycle order;
* M504 is bound as a direct transitive dependency;
* M255's global, ten-frame and complete 64-frame accuracy scopes are emitted
  together and an accuracy/performance Pareto is explicitly rejected;
* every record/cohort/plane identity is checked before replay and every input
  plus all 80 payloads can be rechecked at terminal publication.

This remains a support/cycle experiment.  Its three ratios are separate and
must never be multiplied into an RTL, system, energy or headline result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R1_PATH = ROOT / "system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture.py"
R1_SHA256 = "4b990906fa76543cbbccb9d244a26974914902e0b1ad546d1ad197e7edbaf1ee"
M504_PATH = ROOT / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
M504_SHA256 = "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e"

FROZEN_PYTHON = Path("/opt/anaconda3/envs/python310/bin/python3.10")
FROZEN_PYTHON_SHA256 = "4cd88f501216f7553ce8b80cc4c85c72ca09b0c6f03d62debfa16e8726546b0f"
FROZEN_PYTHON_VERSION = "3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0]"
FROZEN_NUMPY_VERSION = "2.0.1"
FROZEN_NUMPY_INIT = Path(
    "/opt/anaconda3/envs/python310/lib/python3.10/site-packages/numpy/__init__.py"
)
FROZEN_NUMPY_INIT_SHA256 = "c09e25b58f6b2f8e2cb3c158168f902d447f8171e5ea6513c0aca41ecbda7c2b"

SAMPLES = 10
OPERATORS = 4
ROWS = 3000
PARTITIONS = 432
PARTITION_BITS = 16
ROW_TILE = 64
CHUNKS = int(math.ceil(ROWS / float(ROW_TILE)))
FEATURE_BITS = 768 * 3 * 3
PACKED_SHAPE = [10, 1, 768, 15, 20]
PACKED_ELEMENTS = 10 * 768 * 15 * 20
PLANE_BYTES = (PACKED_ELEMENTS + 7) // 8
PACKING = (
    "C_ORDER_FLAT_NP_PACKBITS_LITTLE_POSITIVE_THEN_NEGATIVE_THEN_"
    "EXACT_FLOAT_VALUE_CHANGED_VS_PREVIOUS_T_WITH_T0_ZERO"
)

ACTIVE_CONTRACT: Path | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise RuntimeError("non-standard JSON number: " + token)

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def load_module(name: str, path: Path, expected_sha: str) -> Any:
    require(path.is_file(), "missing frozen module: " + str(path))
    require(sha256_file(path) == expected_sha, "frozen module SHA drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1 = load_module("m586_m579_frozen_r1", R1_PATH, R1_SHA256)
R1_ANALYZE_RECORD = R1.analyze_record
R1_WORKER_INIT = R1.worker_init
R1_STRICT_JSON = R1.strict_json


def verify_runtime() -> dict[str, str]:
    observed_executable = Path(sys.executable).resolve()
    require(observed_executable == FROZEN_PYTHON.resolve(), "Python executable drift")
    require(sha256_file(FROZEN_PYTHON) == FROZEN_PYTHON_SHA256, "Python SHA drift")
    require(sys.version == FROZEN_PYTHON_VERSION, "Python version drift")
    require(np.__version__ == FROZEN_NUMPY_VERSION, "NumPy version drift")
    require(Path(np.__file__).resolve() == FROZEN_NUMPY_INIT.resolve(), "NumPy path drift")
    require(
        sha256_file(FROZEN_NUMPY_INIT) == FROZEN_NUMPY_INIT_SHA256,
        "NumPy init SHA drift",
    )
    return {
        "python": str(observed_executable),
        "python_sha256": FROZEN_PYTHON_SHA256,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "numpy_init": str(Path(np.__file__).resolve()),
        "numpy_init_sha256": FROZEN_NUMPY_INIT_SHA256,
    }


def task_order_self_test() -> dict[str, Any]:
    require(PARTITIONS * PARTITION_BITS == FEATURE_BITS, "16-bit partition extent drift")
    require(CHUNKS == 47 and ROWS - (CHUNKS - 1) * ROW_TILE == 56, "row tail drift")
    tags = np.arange(PARTITIONS * CHUNKS, dtype=np.int64).reshape(PARTITIONS, CHUNKS)
    reordered = tags.T.reshape(-1)
    require(
        reordered[:4].tolist() == [0, CHUNKS, 2 * CHUNKS, 3 * CHUNKS],
        "chunk-major order anchor drift",
    )
    require(int(reordered[-1]) == PARTITIONS * CHUNKS - 1, "terminal order anchor drift")
    return {
        "order": "sample_operator_row_chunk_partition",
        "tasks_per_operator": PARTITIONS * CHUNKS,
        "first_partition_major_source_indices": reordered[:4].tolist(),
        "last_partition_major_source_index": int(reordered[-1]),
        "last_row_chunk_rows": 56,
    }


def worker_init() -> None:
    verify_runtime()
    require(M504_PATH.is_file(), "missing frozen M504 dependency")
    require(sha256_file(M504_PATH) == M504_SHA256, "frozen M504 SHA drift")
    R1_WORKER_INIT()


def spawn_probe() -> dict[str, Any]:
    require(R1.M505 is not None and R1.M43 is not None, "worker initializer did not load modules")
    masks = np.asarray([0, 1, 3, 7, 15, 5, 0, 9], dtype=np.uint16)
    dead = R1.M505.simulate_liveness_task(masks, False)
    require(
        int(dead["ideal_1r1w_issue_cycles"]) <= int(dead["liveness_cycles"]),
        "synthetic recurrence lower-bound failure",
    )
    return {
        "pid_import": True,
        "row_count": int(dead["row_count"]),
        "parent_edges": int(dead["parent_edges"]),
        "issue_cycles": int(dead["ideal_1r1w_issue_cycles"]),
        "liveness_cycles": int(dead["liveness_cycles"]),
    }


def analyze_record(job: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    row = R1_ANALYZE_RECORD(job)
    expected = PARTITIONS * CHUNKS
    for name, values in row["arrays"].items():
        values = np.asarray(values, dtype=np.int64)
        require(values.shape == (expected,), "r1 task-array extent drift: " + name)
        row["arrays"][name] = values.reshape(PARTITIONS, CHUNKS).T.reshape(-1)
    row["task_order"] = "sample_operator_row_chunk_partition"
    row["task_order_anchor"] = task_order_self_test()
    return row


def strict_json_for_r1(path: Path) -> dict[str, Any]:
    data = R1_STRICT_JSON(path)
    if ACTIVE_CONTRACT is not None and Path(path).resolve() == ACTIVE_CONTRACT:
        require(
            data["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v2",
            "M579 r2 execution schema drift",
        )
        data = copy.deepcopy(data)
        data["schema"] = "m579_paft_control_single_port_product_capture_execution_contract_v1"
    return data


R1.worker_init = worker_init
R1.analyze_record = analyze_record
R1.strict_json = strict_json_for_r1
# R1's two analyzer-identity checks must refer to this executed r2 wrapper.
R1.__file__ = str(Path(__file__).resolve())


def validate_trace_manifest(path: Path, expected_status: str) -> dict[str, Any]:
    manifest = strict_json(path)
    require(manifest["status"] == expected_status, "trace status drift")
    cohort = manifest["cohort"]
    require(cohort["samples"] == SAMPLES and cohort["records"] == SAMPLES * OPERATORS,
            "trace cohort count drift")
    require(cohort["shape"] == PACKED_SHAPE, "trace cohort shape drift")
    require(len(cohort["sample_keys"]) == SAMPLES, "sample key count drift")
    require(len(cohort["operators"]) == OPERATORS, "operator count drift")
    records = sorted(
        manifest["records"], key=lambda record: (record["sample_id"], record["operator_index"])
    )
    require(len(records) == SAMPLES * OPERATORS, "trace record count drift")
    observed_files: set[str] = set()
    payloads: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample = index // OPERATORS
        operator = index % OPERATORS
        require(record["sample_id"] == sample and record["operator_index"] == operator,
                "record order drift")
        require(record["sample_key"] == cohort["sample_keys"][sample], "record sample key drift")
        require(record["operator"] == cohort["operators"][operator], "record operator drift")
        require(record["shape"] == PACKED_SHAPE and record["output_shape"] == PACKED_SHAPE,
                "record shape drift")
        require(record["elements"] == PACKED_ELEMENTS, "record element count drift")
        require(record["positive_plane_bytes"] == PLANE_BYTES, "positive plane extent drift")
        require(record["negative_plane_offset_bytes"] == PLANE_BYTES,
                "negative plane offset drift")
        require(record["numeric_change_plane_offset_bytes"] == 2 * PLANE_BYTES,
                "numeric plane offset drift")
        require(record["numeric_change_plane_bytes"] == PLANE_BYTES,
                "numeric plane extent drift")
        require(record["packed_file_bytes"] == 3 * PLANE_BYTES, "packed extent drift")
        require(record["packing"] == PACKING, "packing contract drift")
        require(record["negative_count"] == 0, "negative support is outside M579 scope")
        require(record["positive_count"] == record["nonzero_count"], "support count drift")
        require(sum(record["local_nonzero_count_by_timestep"]) == record["positive_count"],
                "timestep support count drift")
        name = record["packed_file"]
        require(Path(name).name == name and name not in observed_files, "unsafe/duplicate payload name")
        observed_files.add(name)
        payload = path.parent / name
        require(payload.is_file(), "missing packed payload: " + str(payload))
        require(payload.stat().st_size == record["packed_file_bytes"], "payload size drift")
        observed_sha = sha256_file(payload)
        require(observed_sha == record["packed_file_sha256"], "payload SHA drift")
        payloads.append({"path": str(payload), "sha256": observed_sha})
    return {
        "manifest": manifest,
        "payloads": payloads,
        "records": len(records),
        "payload_hashes_rechecked": len(payloads),
    }


def validate_execution_contract(contract_path: Path) -> dict[str, Any]:
    contract = strict_json(contract_path)
    require(
        contract["schema"] == "m579_paft_control_single_port_product_capture_execution_contract_v2",
        "execution contract schema drift",
    )
    require(contract["authorization"]["launch_now"] is True, "launch_now not authorized")
    require(contract["authorization"]["run_cpu"] is True, "CPU run not authorized")
    require(contract["authorization"]["max_attempts"] == 1, "attempt count drift")
    require(contract["authorization"]["run_gpu"] is False, "GPU forbidden")
    require(contract["authorization"]["run_eda"] is False, "EDA forbidden")
    require(contract["authorization"]["run_remote"] is False, "remote forbidden")
    require(contract["analyzer_sha256"] == sha256_file(Path(__file__).resolve()),
            "r2 analyzer SHA drift")
    runtime = verify_runtime()
    require(contract["runtime"] == runtime, "frozen runtime identity drift")

    paths: dict[str, Path] = {}
    for name, spec in contract["inputs"].items():
        path = ROOT / spec["path"]
        require(path.is_file(), "missing execution input: " + name)
        require(sha256_file(path) == spec["sha256"], "execution input SHA drift: " + name)
        paths[name] = path
    require(paths["m504_recurrence_dependency"].resolve() == M504_PATH.resolve(),
            "M504 path drift")
    require(sha256_file(paths["m504_recurrence_dependency"]) == M504_SHA256,
            "M504 direct SHA drift")

    paft = validate_trace_manifest(
        paths["paft_trace"],
        "PASS_PAFT_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE",
    )
    control = validate_trace_manifest(
        paths["control_trace"],
        "PASS_CONTROL_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE",
    )
    require(
        paft["manifest"]["cohort"]["sample_keys"]
        == control["manifest"]["cohort"]["sample_keys"],
        "paired sample cohort drift",
    )
    require(
        paft["manifest"]["cohort"]["operators"]
        == control["manifest"]["cohort"]["operators"],
        "paired operator cohort drift",
    )
    require(
        paft["manifest"]["identity"]["capture_bn_policy"] == "running"
        and control["manifest"]["identity"]["capture_bn_policy"] == "running",
        "running-BN identity drift",
    )
    accuracy = strict_json(paths["paired_valid825"])
    require(accuracy["status"] == "PASS_SINGLE_SEED_SMALL_POSITIVE_RUNNING_BN_DIRECTION",
            "paired valid825 status drift")
    m255 = strict_json(paths["m255_paft_control_hammer"])
    require(
        m255["status"]
        == "PASS_GO_PAIRED_TRACE_LEVEL_ISOLATED_CONV_DIRECTION_NO_PARETO_OR_HEADLINE",
        "M255 status drift",
    )
    require(m255["admission"]["accuracy_performance_pareto_profiled_sequence"] is False,
            "M255 Pareto boundary drift")
    require(m255["accuracy_scope"]["full_hardware_trace_sequence"]["direction"] == "PAFT_WORSE",
            "M255 full-sequence fallback drift")
    m528 = strict_json(paths["m528_result_hammer"])
    require(m528["status"].startswith("PASS_M528_R4_RESULT_HAMMER"), "M528 status drift")
    capacity = m528["validated_metrics"]["capacity"]
    require(capacity["budget_bytes"] == R1.BUDGET_BYTES, "M528 budget drift")
    require(capacity["m505_dead_write_only_macro_rounded_bytes"] == R1.CAPACITY_BYTES,
            "M528 capacity drift")
    m528_result = strict_json(paths["m528_result_json"])
    require(m528_result["capacity"]["budget_bytes"] == R1.BUDGET_BYTES,
            "M528 result budget drift")
    require(
        m528_result["capacity"]["m505_dead_write_only_1rw"]
        ["macro_rounded_total_bytes"] == R1.CAPACITY_BYTES,
        "M528 result candidate capacity drift",
    )
    with paths["m528_capacity_ledger"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames == ["design", "item", "logical_bytes", "macro_rounded_bytes"],
            "capacity ledger schema drift",
        )
        capacity_rows = list(reader)
    candidate_rows = [
        row for row in capacity_rows if row["design"] == "m505_dead_write_only_1rw"
    ]
    require(len(candidate_rows) == 9, "capacity ledger candidate row count drift")
    require(len({row["item"] for row in candidate_rows}) == len(candidate_rows),
            "duplicate capacity ledger item")
    require(all(int(row["logical_bytes"]) >= 0 and int(row["macro_rounded_bytes"]) >= 0
                for row in candidate_rows), "negative capacity ledger value")
    require(sum(int(row["macro_rounded_bytes"]) for row in candidate_rows) == R1.CAPACITY_BYTES,
            "capacity ledger sum drift")
    return {
        "contract": contract,
        "paths": paths,
        "paft": paft,
        "control": control,
        "accuracy": accuracy,
        "m255": m255,
        "m528": m528,
        "m528_result": m528_result,
        "m528_capacity_rows": candidate_rows,
        "task_order": task_order_self_test(),
        "runtime": runtime,
    }


def postprocess_output(output_dir: Path, validated: dict[str, Any]) -> Path:
    old_result = output_dir / "m579_paft_control_single_port_product_capture_r1.json"
    old_csv = output_dir / "m579_per_sample_cycles_r1.csv"
    require(old_result.is_file() and old_csv.is_file(), "r1 staging output incomplete")
    payload = strict_json(old_result)
    m255 = validated["m255"]
    accuracy = validated["accuracy"]
    scope = copy.deepcopy(m255["accuracy_scope"])
    global_scope = scope["global_valid825"]
    require(
        math.isclose(
            float(global_scope["control_running_bn_aee"]),
            float(accuracy["paired_validation"]["running"]["metrics"]["AEE"]["control_frame_equal_mean"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "M247/M255 control AEE drift",
    )
    require(
        math.isclose(
            float(global_scope["paft_running_bn_aee"]),
            float(accuracy["paired_validation"]["running"]["metrics"]["AEE"]["paft_frame_equal_mean"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "M247/M255 PAFT AEE drift",
    )
    payload["schema"] = "m579_paft_control_single_port_product_capture_v2"
    payload["task_order"] = validated["task_order"]
    payload["runtime_identity"] = validated["runtime"]
    payload["accuracy_scope"] = scope
    payload["accuracy_limitations"] = {
        "single_seed": True,
        "multi_seed_significance": False,
        "same_evaluator_runtime_sha_bound_for_both_arms": False,
        "accuracy_performance_pareto": False,
        "profiled_full_sequence_direction": "PAFT_WORSE",
        "m247_limitations": copy.deepcopy(accuracy["limitations"]),
    }
    payload["trained_activity_increment"]["accuracy_performance_pareto"] = False
    payload["trained_activity_increment"]["profiled_full_sequence"] = copy.deepcopy(
        scope["full_hardware_trace_sequence"]
    )
    payload["decision"]["accuracy_performance_pareto"] = False
    payload["resource_coordinate"]["capacity_provenance"] = {
        "m528_result_hammer": validated["contract"]["inputs"]["m528_result_hammer"],
        "m528_result_json": validated["contract"]["inputs"]["m528_result_json"],
        "m528_capacity_ledger": validated["contract"]["inputs"]["m528_capacity_ledger"],
        "candidate_capacity_ledger_rows": len(validated["m528_capacity_rows"]),
        "budget_margin_bytes": R1.BUDGET_BYTES - R1.CAPACITY_BYTES,
        "generated_macro_integration_ppa_energy": "OPEN_NOT_ADMITTED",
        "capacity_is_not_integrated_macro_ppa": True,
    }
    payload["claim_boundary"].update({
        "accuracy_performance_pareto": False,
        "profiled_full_sequence_paft_regresses": True,
        "task_order_matches_frozen_m528": True,
        "integrated_macro_ppa": False,
        "ratios_may_be_multiplied": False,
    })

    new_result = output_dir / "m579_paft_control_single_port_product_capture_r2.json"
    temp_result = output_dir / ".m579_r2_result.tmp"
    temp_result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_result.replace(new_result)
    old_result.unlink()
    old_csv.replace(output_dir / "m579_per_sample_cycles_r2.csv")
    return new_result


def terminal_rehash(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    validated = validate_execution_contract(contract_path)
    result = output_dir / "m579_paft_control_single_port_product_capture_r2.json"
    sample_csv = output_dir / "m579_per_sample_cycles_r2.csv"
    require(result.is_file() and sample_csv.is_file(), "terminal staging output incomplete")
    payload = strict_json(result)
    require(payload["schema"] == "m579_paft_control_single_port_product_capture_v2",
            "terminal result schema drift")
    require(payload["task_order"]["order"] == "sample_operator_row_chunk_partition",
            "terminal task order drift")
    require(payload["accuracy_limitations"]["accuracy_performance_pareto"] is False,
            "terminal Pareto boundary drift")
    require(payload["accuracy_scope"]["full_hardware_trace_sequence"]["direction"] == "PAFT_WORSE",
            "terminal full-sequence disclosure drift")
    return {
        "schema": "m586_m579_r2_terminal_rehash_receipt_v1",
        "status": "PASS_TERMINAL_ALL_INPUT_AND_80_PAYLOAD_REHASH",
        "contract_sha256": sha256_file(contract_path),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "result_sha256": sha256_file(result),
        "sample_csv_sha256": sha256_file(sample_csv),
        "contract_inputs_rehashed": len(validated["contract"]["inputs"]),
        "packed_payloads_rehashed": (
            validated["paft"]["payload_hashes_rechecked"]
            + validated["control"]["payload_hashes_rechecked"]
        ),
        "docs359_sha256": sha256_file(validated["paths"]["docs359"]),
        "task_order": validated["task_order"]["order"],
        "accuracy_performance_pareto": False,
    }


def preflight_only() -> int:
    runtime = verify_runtime()
    order = task_order_self_test()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=worker_init,
    ) as pool:
        probe = pool.submit(spawn_probe).result(timeout=60)
    print(json.dumps({
        "schema": "m586_m579_r2_preflight_v1",
        "status": "PASS_LIGHTWEIGHT_IMPORT_SPAWN_RECURRENCE_ONLY",
        "runtime": runtime,
        "task_order": order,
        "probe": probe,
        "formal_trace_records_processed": 0,
        "result_or_attempt_created": False,
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument("--terminal-rehash", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        require(args.contract is None and args.output_dir is None, "preflight takes no formal paths")
        require(not args.terminal_rehash and not args.validate_contract_only,
                "preflight/validation/terminal modes are exclusive")
        return preflight_only()
    if args.validate_contract_only:
        require(args.contract is not None and args.output_dir is None,
                "contract validation takes contract and no output-dir")
        require(not args.terminal_rehash, "validation/terminal modes are exclusive")
        validated = validate_execution_contract(args.contract.resolve())
        print(json.dumps({
            "schema": "m586_m579_r2_contract_preflight_v1",
            "status": "PASS_EXECUTION_CONTRACT_INPUTS_AND_80_PAYLOADS",
            "contract_sha256": sha256_file(args.contract.resolve()),
            "contract_inputs_rehashed": len(validated["contract"]["inputs"]),
            "packed_payloads_rehashed": (
                validated["paft"]["payload_hashes_rechecked"]
                + validated["control"]["payload_hashes_rechecked"]
            ),
            "formal_trace_records_processed": 0,
            "result_or_attempt_created": False,
        }, sort_keys=True))
        return 0
    require(args.contract is not None and args.output_dir is not None,
            "formal/terminal mode requires contract and output-dir")
    require(1 <= args.workers <= 3, "workers must be 1..3")
    contract_path = args.contract.resolve()
    output_dir = args.output_dir.resolve()
    if args.terminal_rehash:
        receipt = terminal_rehash(contract_path, output_dir)
        print(json.dumps(receipt, sort_keys=True))
        return 0

    global ACTIVE_CONTRACT
    ACTIVE_CONTRACT = contract_path
    validated = validate_execution_contract(contract_path)
    require(not output_dir.exists(), "refuse to overwrite M579 r2 output")
    saved_argv = list(sys.argv)
    try:
        sys.argv = [
            str(Path(__file__).resolve()),
            "--contract", str(contract_path),
            "--output-dir", str(output_dir),
            "--workers", str(args.workers),
        ]
        rc = int(R1.main())
    finally:
        sys.argv = saved_argv
    require(rc == 0, "frozen r1 worker returned nonzero")
    result = postprocess_output(output_dir, validated)
    require(sha256_file(Path(__file__).resolve()) == validated["contract"]["analyzer_sha256"],
            "r2 analyzer changed during run")
    print("PASS M579_R2 staging_result={}".format(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
