#!/usr/bin/env python3
"""Materialize and replay the final ep34 C1 same-ledger cycle model.

The producer consumes the authenticated M1458 ep34 support planes through the
frozen M1524 mapping.  It writes one 51.84-million-row ledger and derives the
strong-zero, same-coordinate bit, fused 1R1W ceiling, all-write 1RW, and
dead-write-only 1RW schedules from that *same* row stream through the frozen
M504/M505/M528 recurrence.

This is an operator-level cycle model, not RTL timing, wall-clock latency,
energy, or full-network speedup.  A release file must bind this source, the
exact output path, the exact ledger path, and a single execution.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
M1524_PATH = HW / "system_simulator/scripts/build_m1524_ep34_c1_same_ledger_rebind_source.py"
M528_PATH = HW / "system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute.py"
M505_PATH = HW / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
M504_PATH = HW / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

M1524_SHA256 = "a089650bad2e6acb338cb19a6ffea52bf4a823d6e32b6fb70ef3b101ed96e416"
M528_SHA256 = "c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a"
M505_SHA256 = "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"
M504_SHA256 = "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1579_ep34_c1_same_ledger_cycle_model_r1_v1"
STATUS = "PASS_M1579_EP34_C1_SAME_LEDGER_CYCLE_MODEL"
RELEASE_SCHEMA = "m1579_ep34_c1_same_ledger_cycle_model_release_r1_v1"
RELEASE_STATUS = "RELEASED_EXACTLY_ONE_CPU_CYCLE_MODEL__NO_EDA_NO_GPU"

SAMPLES = 10
OPERATORS = 4
PARTITIONS = 432
ROWS_PER_PHASE = 3000
PHASES = SAMPLES * OPERATORS * PARTITIONS
SOURCE_ROWS = PHASES * ROWS_PER_PHASE
ROW_BYTES = 9
LEDGER_BYTES = SOURCE_ROWS * ROW_BYTES
CHUNKS = math.ceil(ROWS_PER_PHASE / 64)
BLOCK_BANKS = 8
BYTES_PER_PARENT_VECTOR = 144
CAPACITY_BYTES = 213_376
CAPACITY_BUDGET_BYTES = 245_760

M1524: Any = None
M528: Any = None


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON token: " + token)),
    )
    require(type(value) is dict, "JSON root is not an object")
    return value


def load_exact(path: Path, digest: str, name: str) -> Any:
    require(path.is_file() and not path.is_symlink(), "missing frozen " + name)
    require(sha256(path) == digest, "frozen " + name + " SHA drift")
    module_name = "m1579_frozen_" + name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_modules() -> tuple[Any, Any]:
    global M1524, M528
    require(sha256(M505_PATH) == M505_SHA256 and sha256(M504_PATH) == M504_SHA256,
            "transitive recurrence SHA drift")
    require(sha256(DOCS359) == DOCS359_SHA256, "docs/359 SHA drift")
    if M1524 is None:
        M1524 = load_exact(M1524_PATH, M1524_SHA256, "m1524")
    if M528 is None:
        M528 = load_exact(M528_PATH, M528_SHA256, "m528")
    return M1524, M528


def verify_release(path: Path, output: Path, ledger: Path, workers: int) -> dict[str, Any]:
    value = strict_json(path)
    require(value.get("schema") == RELEASE_SCHEMA and
            value.get("status") == RELEASE_STATUS,
            "release schema/status drift")
    require(value.get("source_sha256") == sha256(SOURCE),
            "release does not bind source")
    require(Path(value.get("output", "")).resolve() == output.resolve(),
            "release output drift")
    require(Path(value.get("ledger", "")).resolve() == ledger.resolve(),
            "release ledger drift")
    require(value.get("cpu_runs") == 1 and value.get("gpu_runs") == 0 and
            value.get("eda_runs") == 0,
            "release execution budget drift")
    require(type(value.get("maximum_workers")) is int and
            1 <= workers <= value["maximum_workers"] <= 3,
            "worker budget drift")
    frozen = value.get("frozen_inputs", {})
    require(frozen == {
        "m1524": M1524_SHA256,
        "m528": M528_SHA256,
        "m505": M505_SHA256,
        "m504": M504_SHA256,
        "docs359": DOCS359_SHA256,
    }, "release frozen-input map drift")
    return value


def materialize_ledger(path: Path) -> dict[str, Any]:
    m1524, _ = load_modules()
    records, _ = m1524.collect_records()
    require(len(records) == SAMPLES * OPERATORS, "ep34 C1 record population drift")
    digest = hashlib.sha256()
    line_count = 0
    active_bits = 0
    per_operator_active = [0] * OPERATORS
    with path.open("xb", buffering=1 << 20) as stream:
        for sample in range(SAMPLES):
            for operator in range(OPERATORS):
                record = records[sample * OPERATORS + operator]
                require(record["name"] == m1524.MODULES[operator],
                        "record order drift")
                support = m1524.decode_support(record)
                count = int(support.sum())
                require(count == int(record["input"]["active"]),
                        "support population drift")
                active_bits += count
                per_operator_active[operator] += count
                for partition in range(PARTITIONS):
                    block = m1524.m528_compatible_lines(
                        m1524.phase_masks(support, partition))
                    require(len(block) == ROWS_PER_PHASE * ROW_BYTES,
                            "phase byte extent drift")
                    stream.write(block)
                    digest.update(block)
                    line_count += ROWS_PER_PHASE
        stream.flush()
        os.fsync(stream.fileno())
    require(path.stat().st_size == LEDGER_BYTES and line_count == SOURCE_ROWS,
            "materialized ledger extent drift")
    return {
        "path": path.name,
        "sha256": digest.hexdigest(),
        "bytes": LEDGER_BYTES,
        "rows": SOURCE_ROWS,
        "line_format": "0000<support16_lowercase_hex>\\n",
        "phase_order": "sample,operator,partition",
        "row_order": "timestep,output_y,output_x",
        "captured_input_active_values": active_bits,
        "captured_input_active_values_by_operator": per_operator_active,
    }


def worker_init(ledger: str) -> None:
    _, m528 = load_modules()
    m528.worker_init(ledger)


def worker_phase(index: int) -> tuple[int, dict[str, np.ndarray]]:
    _, m528 = load_modules()
    return m528.worker_phase(index)


def describe(values: list[float]) -> dict[str, float | int]:
    require(values and all(math.isfinite(value) and value > 0 for value in values),
            "invalid distribution")
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "arithmetic_mean": mean,
        "geometric_mean": math.exp(math.fsum(math.log(value) for value in values) /
                                   len(values)),
        "minimum": min(values),
        "maximum": max(values),
        "coefficient_of_variation_population": statistics.pstdev(values) / mean,
    }


def replay(ledger: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _, m528 = load_modules()
    require(ledger.is_file() and ledger.stat().st_size == LEDGER_BYTES,
            "ledger unavailable or wrong extent")
    shape = (SAMPLES, OPERATORS, CHUNKS, PARTITIONS)
    arrays = {name: np.zeros(shape, dtype=np.int32) for name in m528.FIELD_NAMES}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=worker_init, initargs=(str(ledger),)) as pool:
        for phase, fields in pool.map(worker_phase, range(PHASES), chunksize=2):
            require(type(phase) is int and 0 <= phase < PHASES,
                    "worker phase identity drift")
            sample = phase // (OPERATORS * PARTITIONS)
            operator = (phase // PARTITIONS) % OPERATORS
            partition = phase % PARTITIONS
            for name in m528.FIELD_NAMES:
                values = np.asarray(fields[name], dtype=np.int32)
                require(values.shape == (CHUNKS,), "worker field shape drift")
                arrays[name][sample, operator, :, partition] = values

    aggregate = {name: int(values.astype(np.int64).sum())
                 for name, values in arrays.items()}
    require(aggregate["row_count"] == SOURCE_ROWS,
            "source-row conservation mismatch")
    require(aggregate["ideal_issue_cycles"] ==
            aggregate["residual_nnz"] + aggregate["exact_parent_rows"],
            "arithmetic issue conservation mismatch")
    require(aggregate["dead_reads"] + aggregate["dead_forwards"] ==
            aggregate["parent_edges"], "parent-edge conservation mismatch")
    require(aggregate["dead_writes"] + aggregate["dead_elisions"] ==
            aggregate["active_rows"], "completion conservation mismatch")

    samples: list[dict[str, Any]] = []
    operators: list[dict[str, Any]] = []
    for sample in range(SAMPLES):
        cycles = m528.cycle_row(arrays, sample, None)
        samples.append({
            "sample": sample,
            "aggregation_semantics":
                "sample_major_four_operator_continuous_pipeline_plus_commit",
            **cycles,
            **m528.ratio_fields(cycles),
        })
        for operator in range(OPERATORS):
            isolated = m528.cycle_row(arrays, sample, operator)
            operators.append({
                "sample": sample,
                "operator": operator,
                "module": M1524.MODULES[operator],
                "aggregation_semantics":
                    "operator_isolated_pipeline_no_commit_not_summable",
                **isolated,
                **m528.ratio_fields(isolated),
            })

    cycle_keys = [key for key in samples[0] if key.endswith("_cycles")]
    totals = {key: sum(int(row[key]) for row in samples) for key in cycle_keys}
    ratios = m528.ratio_fields(totals)
    ratio_keys = tuple(ratios)
    distributions = {
        "sample_major": {
            "cycles": {key: describe([float(row[key]) for row in samples])
                       for key in cycle_keys},
            "ratios": {key: describe([float(row[key]) for row in samples])
                       for key in ratio_keys},
        },
        "operator_isolated": {
            "cycles": {key: describe([float(row[key]) for row in operators])
                       for key in cycle_keys},
            "ratios": {key: describe([float(row[key]) for row in operators])
                       for key in ratio_keys},
        },
    }
    parent_accesses = aggregate["dead_reads"] + aggregate["dead_writes"]
    summary = {
        "aggregate_cycles": {**totals, **ratios},
        "ratio_semantics": "ratio_of_sums_over_ten_ep34_samples",
        "distribution": distributions,
        "conservation": {
            "source_rows": aggregate["row_count"],
            "input_nonzero_bit_issues_per_output_block": aggregate["input_nnz"],
            "residual_nonzero_bit_issues_per_output_block": aggregate["residual_nnz"],
            "exact_parent_only_issues_per_output_block": aggregate["exact_parent_rows"],
            "product_arithmetic_issues_per_output_block": aggregate["ideal_issue_cycles"],
            "product_arithmetic_issues_all_eight_output_blocks":
                aggregate["ideal_issue_cycles"] * BLOCK_BANKS,
            "parent_edges_per_output_block": aggregate["parent_edges"],
            "dead_reads_plus_forwards": aggregate["dead_reads"] + aggregate["dead_forwards"],
            "active_rows_per_output_block": aggregate["active_rows"],
            "dead_writes_plus_elisions": aggregate["dead_writes"] + aggregate["dead_elisions"],
            "all_equalities_pass": True,
        },
        "traffic": {
            "dead_write_only_parent_read_bytes_all_eight_blocks":
                aggregate["dead_reads"] * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR,
            "dead_write_only_parent_write_bytes_all_eight_blocks":
                aggregate["dead_writes"] * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR,
            "dead_write_only_parent_total_bytes_all_eight_blocks":
                parent_accesses * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR,
            "traffic_scope": "parent scratch only; not total SRAM or DRAM traffic",
        },
        "capacity": {
            "macro_rounded_bytes": CAPACITY_BYTES,
            "budget_bytes": CAPACITY_BUDGET_BYTES,
            "margin_bytes": CAPACITY_BUDGET_BYTES - CAPACITY_BYTES,
            "fits": CAPACITY_BYTES <= CAPACITY_BUDGET_BYTES,
        },
    }
    return summary, samples, operators


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal(directory: Path, members: list[Path]) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(not manifest.exists() and not outer.exists(), "seal already exists")
    manifest.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                for path in members), encoding="ascii")
    outer.write_text("{}  SHA256SUMS\n".format(sha256(manifest)), encoding="ascii")
    return {"manifest_sha256": sha256(manifest), "outer_file_sha256": sha256(outer)}


def execute(release: Path, output: Path, ledger: Path, workers: int) -> dict[str, Any]:
    require(not output.exists() and not ledger.exists(),
            "refuse to overwrite output or ledger")
    verify_release(release, output, ledger, workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(ledger.parent.resolve() == output.resolve(),
            "ledger must be a member of the canonical result directory")
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=str(output.parent)))
    try:
        staged_ledger = stage / ledger.name
        ledger_identity = materialize_ledger(staged_ledger)
        summary, samples, operators = replay(staged_ledger, workers)
        sample_csv = stage / "sample_major_cycles.csv"
        operator_csv = stage / "operator_isolated_cycles.csv"
        write_csv(sample_csv, samples)
        write_csv(operator_csv, operators)
        result = {
            "schema": SCHEMA,
            "status": STATUS,
            "identity": {
                "checkpoint_sha256": M1524.CHECKPOINT_SHA256,
                "capture_manifest_sha256": M1524.CAPTURE_MANIFEST_SHA256,
                "ordered_records_sha256": M1524.ORDERED_SHA256,
                "source_sha256": sha256(SOURCE),
                "release_sha256": sha256(release),
                "frozen_m1524_sha256": M1524_SHA256,
                "frozen_m528_sha256": M528_SHA256,
                "frozen_m505_sha256": M505_SHA256,
                "frozen_m504_sha256": M504_SHA256,
            },
            "scope": {
                "checkpoint": "Motion C12 ep34 live93",
                "samples": SAMPLES,
                "operators": list(M1524.MODULES),
                "operator_class": "four bottleneck Conv3x3 only",
                "sequence": "zurich_city_09_a",
                "cycle_model": True,
                "same_ledger_all_baselines": True,
            },
            "ledger": ledger_identity,
            **summary,
            "claim_boundary": {
                "paper_citable_after_independent_result_hammer": False,
                "cycle_model": True,
                "cpu_replay": True,
                "rtl_cycle": False,
                "wall_clock": False,
                "full_network": False,
                "system_speedup": False,
                "energy": False,
                "power": False,
                "ppa": False,
                "multi_sequence": False,
                "external_official_simulator": False,
            },
        }
        result_path = stage / "m1579_ep34_c1_same_ledger_cycle_model_result_r1.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n", encoding="utf-8")
        complete = stage / "RUN_COMPLETE.txt"
        complete.write_text(STATUS + "\n", encoding="ascii")
        seal(stage, [staged_ledger, result_path, sample_csv, operator_csv, complete])
        os.replace(str(stage), str(output))
        return result
    except BaseException:
        # Preserve the failed staging directory for forensics; never publish it.
        raise


def source_audit() -> dict[str, Any]:
    m1524, m528 = load_modules()
    audit = m1524.audit(run_numeric=True, run_checkpoint=True)
    require(audit["identity"]["retained_records"] == SAMPLES * OPERATORS,
            "M1524 retained population drift")
    require(m528.SAMPLES == SAMPLES and m528.OPERATORS == OPERATORS and
            m528.PARTITIONS == PARTITIONS and m528.ROWS_PER_PHASE == ROWS_PER_PHASE,
            "M528 geometry drift")
    return {
        "schema": "m1579_ep34_c1_same_ledger_cycle_model_source_audit_r1_v1",
        "status": "PASS_SOURCE_AUDIT__NO_EXECUTION",
        "source_sha256": sha256(SOURCE),
        "geometry": {"samples": SAMPLES, "operators": OPERATORS,
                     "partitions": PARTITIONS, "rows_per_phase": ROWS_PER_PHASE,
                     "source_rows": SOURCE_ROWS, "ledger_bytes": LEDGER_BYTES},
        "checkpoint_sha256": m1524.CHECKPOINT_SHA256,
        "capture_manifest_sha256": m1524.CAPTURE_MANIFEST_SHA256,
        "same_ledger_required": True,
        "old_ep35_cycles_reusable": False,
        "production": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    require(args.source_audit != args.execute, "select exactly one mode")
    if args.source_audit:
        print(json.dumps(source_audit(), indent=2, sort_keys=True,
                         allow_nan=False))
        return 0
    require(args.release is not None and args.out is not None and
            args.ledger is not None, "execute requires release/out/ledger")
    result = execute(args.release.resolve(), args.out.resolve(),
                     args.ledger.resolve(), args.workers)
    print(json.dumps(result["aggregate_cycles"], indent=2, sort_keys=True))
    print(STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
