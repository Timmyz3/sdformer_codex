#!/usr/bin/env python3
"""M528 same-ledger recompute for the realizable H67 one-port parent scratch.

This is deliberately a recompute, not a post-hoc division of four aggregate
numbers.  It replays the frozen row stream once, reconstructs M468 strong-zero,
M473 bit/fused-ceiling, M504 all-write, M505 dead-write-only, and the combined
PVRF ablation on the same task order, and emits distinct sample-major and
operator-isolated distributions.

The operator-isolated rows restart the preprocess/work pipeline and omit the
sample commit.  They are useful for heterogeneity analysis, but must never be
summed and presented as the sample-major/full-four-Conv runtime.
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
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTION_CONTRACT = (
    ROOT
    / "contracts"
    / "m528_h67_single_port_same_ledger_execution_contract_r1_20260827.json"
)
DEFAULT_OUT = (
    ROOT
    / "results"
    / "m528_h67_single_port_same_ledger_recompute_r1_20260827"
)

M505_ANALYZER = (
    ROOT
    / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
)
M504_ANALYZER = (
    ROOT
    / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
)
EXPECTED_M505_ANALYZER_SHA256 = (
    "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"
)
EXPECTED_M504_ANALYZER_SHA256 = (
    "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e"
)

SAMPLES = 10
OPERATORS = 4
PARTITIONS = 432
ROWS_PER_PHASE = 3000
ROW_TILE = 64
CHUNKS = math.ceil(ROWS_PER_PHASE / ROW_TILE)
BLOCK_BANKS = 8
BYTES_PER_PARENT_VECTOR = 144
COMMIT_PER_SAMPLE = 96000
WEIGHT_DMA_CYCLES = 160
TAIL_CYCLES = 2
CAM_COMPARE_LANES = 64

M505: Any = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise RuntimeError(f"non-standard JSON token: {token}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def load_frozen_m505() -> Any:
    global M505
    if M505 is not None:
        return M505
    require(M505_ANALYZER.is_file(), "missing frozen M505 analyzer")
    require(M504_ANALYZER.is_file(), "missing frozen M504 analyzer")
    require(
        sha256_file(M505_ANALYZER) == EXPECTED_M505_ANALYZER_SHA256,
        "frozen M505 analyzer SHA drift before import",
    )
    require(
        sha256_file(M504_ANALYZER) == EXPECTED_M504_ANALYZER_SHA256,
        "frozen M504 analyzer SHA drift before import",
    )
    spec = importlib.util.spec_from_file_location("m505_frozen_for_m528", M505_ANALYZER)
    require(spec is not None and spec.loader is not None, "cannot load frozen M505 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    M505 = module
    return module


def verify_manifest(directory: Path, expected_manifest_sha: str | None = None) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), f"missing double seal: {directory}")
    actual_manifest_sha = sha256_file(manifest)
    if expected_manifest_sha is not None:
        require(actual_manifest_sha == expected_manifest_sha, f"manifest SHA drift: {directory}")
    outer_parts = outer.read_text(encoding="utf-8").strip().split()
    require(
        len(outer_parts) == 2
        and outer_parts[0] == actual_manifest_sha
        and outer_parts[1] == "SHA256SUMS",
        f"outer seal mismatch: {directory}",
    )
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in entries, f"duplicate manifest entry: {name}")
        path = directory / name
        require(path.is_file(), f"missing sealed file: {path}")
        require(sha256_file(path) == digest, f"sealed file SHA drift: {path}")
        entries[name] = digest
    return entries


def pipeline_cycles(preprocess: np.ndarray, work: np.ndarray) -> int:
    preprocess = np.asarray(preprocess, dtype=np.int64)
    work = np.asarray(work, dtype=np.int64)
    require(preprocess.shape == work.shape and preprocess.size > 0, "pipeline shape mismatch")
    total = int(preprocess[0])
    if preprocess.size > 1:
        total += int(np.maximum(work[:-1], preprocess[1:]).sum())
        total += (preprocess.size - 1) * TAIL_CYCLES
    total += int(work[-1]) + TAIL_CYCLES
    return total


def geometric_mean(values: list[float]) -> float:
    require(values and all(value > 0.0 for value in values), "geomean requires positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def describe(values: list[float]) -> dict[str, float | int]:
    require(values, "empty distribution")
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "arithmetic_mean": mean,
        "geometric_mean": geometric_mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "coefficient_of_variation_population": statistics.pstdev(values) / mean,
    }


FIELD_NAMES = (
    "row_count",
    "input_nnz",
    "active_rows",
    "search_rows",
    "residual_nnz",
    "exact_parent_rows",
    "parent_edges",
    "ideal_issue_cycles",
    "m504_cycles",
    "m504_reads",
    "m504_writes",
    "dead_cycles",
    "dead_stalls",
    "dead_reads",
    "dead_writes",
    "dead_forwards",
    "dead_elisions",
    "combined_cycles",
    "combined_stalls",
    "combined_reads",
    "combined_writes",
    "combined_forwards",
    "combined_dead_elisions",
    "combined_single_use_elisions",
)


def worker_init(rows_path: str) -> None:
    module = load_frozen_m505()
    module.worker_init(rows_path)


def worker_phase(phase_index: int) -> tuple[int, dict[str, np.ndarray]]:
    module = load_frozen_m505()
    m504 = module.M504
    masks = m504.read_phase(phase_index)
    fields = {
        name: np.zeros(CHUNKS, dtype=np.int32) for name in FIELD_NAMES
    }
    for chunk, start in enumerate(range(0, ROWS_PER_PHASE, ROW_TILE)):
        tile = masks[start : min(start + ROW_TILE, ROWS_PER_PHASE)]
        residual, parent = m504.cleanroom_subset(tile)
        original_pc = m504.POPCOUNT[tile].astype(np.int32)
        residual_pc = m504.POPCOUNT[residual].astype(np.int32)
        active = tile != 0
        exact_parent = (parent >= 0) & (residual == 0) & active
        old = m504.simulate_single_port_task(tile, "deadline_lookahead")
        dead = module.simulate_liveness_task(tile, False)
        combined = module.simulate_liveness_task(tile, True)
        require(old["parent_edges"] == dead["parent_edges"] == combined["parent_edges"], "parent-edge drift")
        require(old["ideal_1r1w_issue_cycles"] == dead["ideal_1r1w_issue_cycles"] == combined["ideal_1r1w_issue_cycles"], "issue drift")
        require(dead["liveness_cycles"] <= old["single_port_issue_window_cycles"], "dead-only regression")
        require(combined["liveness_cycles"] <= dead["liveness_cycles"], "combined regression")
        values = {
            "row_count": int(tile.size),
            "input_nnz": int(original_pc.sum()),
            "active_rows": int(np.count_nonzero(active)),
            "search_rows": int(np.count_nonzero(original_pc > 1)),
            "residual_nnz": int(residual_pc.sum()),
            "exact_parent_rows": int(np.count_nonzero(exact_parent)),
            "parent_edges": int(old["parent_edges"]),
            "ideal_issue_cycles": int(old["ideal_1r1w_issue_cycles"]),
            "m504_cycles": int(old["single_port_issue_window_cycles"]),
            "m504_reads": int(old["macro_reads"]),
            "m504_writes": int(old["macro_writes"]),
            "dead_cycles": int(dead["liveness_cycles"]),
            "dead_stalls": int(dead["liveness_stall_cycles"]),
            "dead_reads": int(dead["macro_reads"]),
            "dead_writes": int(dead["macro_writes"]),
            "dead_forwards": int(dead["forwarded_reads"]),
            "dead_elisions": int(dead["dead_writes_elided"]),
            "combined_cycles": int(combined["liveness_cycles"]),
            "combined_stalls": int(combined["liveness_stall_cycles"]),
            "combined_reads": int(combined["macro_reads"]),
            "combined_writes": int(combined["macro_writes"]),
            "combined_forwards": int(combined["forwarded_reads"]),
            "combined_dead_elisions": int(combined["dead_writes_elided"]),
            "combined_single_use_elisions": int(combined["single_use_forwarded_writes_elided"]),
        }
        require(
            values["ideal_issue_cycles"]
            == values["residual_nnz"] + values["exact_parent_rows"],
            "arithmetic issue conservation mismatch",
        )
        for name, value in values.items():
            fields[name][chunk] = value
    return phase_index, fields


def flatten_sample(array: np.ndarray, sample: int) -> np.ndarray:
    return np.asarray(array[sample]).reshape(-1).astype(np.int64)


def flatten_operator(array: np.ndarray, sample: int, operator: int) -> np.ndarray:
    return np.asarray(array[sample, operator]).reshape(-1).astype(np.int64)


def prework(arrays: dict[str, np.ndarray], sample: int, operator: int | None = None) -> dict[str, np.ndarray]:
    take = (
        (lambda name: flatten_sample(arrays[name], sample))
        if operator is None
        else (lambda name: flatten_operator(arrays[name], sample, operator))
    )
    rows = take("row_count")
    input_nnz = take("input_nnz")
    search_rows = take("search_rows")
    nonempty = input_nnz != 0
    m468_frontend = rows + 5
    bit_capture = (rows + 7) // 8
    bit_frontend = bit_capture + 2
    product_frontend = (
        bit_capture
        + search_rows * ((rows + CAM_COMPARE_LANES - 1) // CAM_COMPARE_LANES)
        + 17 * bit_capture
        + 2
    )
    return {
        "m468_pre": np.where(nonempty, np.maximum(m468_frontend, WEIGHT_DMA_CYCLES), m468_frontend),
        "bit_pre": np.where(nonempty, np.maximum(bit_frontend, WEIGHT_DMA_CYCLES), bit_frontend),
        "product_pre": np.where(nonempty, np.maximum(product_frontend, WEIGHT_DMA_CYCLES), product_frontend),
        "m468_work": input_nnz * BLOCK_BANKS,
        "bit_work": input_nnz * BLOCK_BANKS,
        "ceiling_work": take("ideal_issue_cycles") * BLOCK_BANKS,
        "m504_work": take("m504_cycles") * BLOCK_BANKS,
        "dead_work": take("dead_cycles") * BLOCK_BANKS,
        "combined_work": take("combined_cycles") * BLOCK_BANKS,
    }


def cycle_row(arrays: dict[str, np.ndarray], sample: int, operator: int | None) -> dict[str, int]:
    payload = prework(arrays, sample, operator)
    result = {
        "m468_strong_zero_cycles": pipeline_cycles(payload["m468_pre"], payload["m468_work"]),
        "m473_same_coordinate_bit_cycles": pipeline_cycles(payload["bit_pre"], payload["bit_work"]),
        "m473_fused_concurrent_1r1w_ceiling_cycles": pipeline_cycles(payload["product_pre"], payload["ceiling_work"]),
        "m504_all_write_1rw_cycles": pipeline_cycles(payload["product_pre"], payload["m504_work"]),
        "m505_dead_write_only_1rw_cycles": pipeline_cycles(payload["product_pre"], payload["dead_work"]),
        "m505_combined_pvrf_1rw_cycles": pipeline_cycles(payload["product_pre"], payload["combined_work"]),
    }
    if operator is None:
        for key in result:
            result[key] += COMMIT_PER_SAMPLE
    return result


def ratio_fields(row: dict[str, int]) -> dict[str, float]:
    candidate = row["m505_dead_write_only_1rw_cycles"]
    ceiling = row["m473_fused_concurrent_1r1w_ceiling_cycles"]
    return {
        "speedup_vs_m468_strong_zero": row["m468_strong_zero_cycles"] / candidate,
        "speedup_vs_m473_same_coordinate_bit": row["m473_same_coordinate_bit_cycles"] / candidate,
        "port_tax_vs_m473_ceiling": candidate / ceiling - 1.0,
        "m504_to_dead_write_speedup": row["m504_all_write_1rw_cycles"] / candidate,
        "dead_to_combined_speedup": candidate / row["m505_combined_pvrf_1rw_cycles"],
    }


def build_distribution_stats(rows: list[dict[str, Any]], semantics: str) -> dict[str, Any]:
    cycle_keys = [key for key in rows[0] if key.endswith("_cycles")]
    ratio_keys = [
        "speedup_vs_m468_strong_zero",
        "speedup_vs_m473_same_coordinate_bit",
        "port_tax_vs_m473_ceiling",
        "m504_to_dead_write_speedup",
        "dead_to_combined_speedup",
    ]
    return {
        "semantics": semantics,
        "cycles": {key: describe([float(row[key]) for row in rows]) for key in cycle_keys},
        "ratios": {key: describe([float(row[key]) for row in rows]) for key in ratio_keys},
        "ratio_of_sums": {
            "speedup_vs_m468_strong_zero": sum(row["m468_strong_zero_cycles"] for row in rows) / sum(row["m505_dead_write_only_1rw_cycles"] for row in rows),
            "speedup_vs_m473_same_coordinate_bit": sum(row["m473_same_coordinate_bit_cycles"] for row in rows) / sum(row["m505_dead_write_only_1rw_cycles"] for row in rows),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", type=Path, default=DEFAULT_EXECUTION_CONTRACT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.execution_contract = args.execution_contract.resolve()
    args.out = args.out.resolve()
    require(not args.out.exists(), "refuse to overwrite M528 output")

    execution = strict_json(args.execution_contract)
    require(execution["schema"] == "m528_h67_single_port_same_ledger_execution_contract_v1", "execution schema drift")
    require(1 <= args.workers <= int(execution["runtime"]["maximum_workers"]), "worker count outside contract")
    governing_path = ROOT / execution["governing_contract"]["path"]
    require(sha256_file(governing_path) == execution["governing_contract"]["sha256"], "governing contract SHA drift")
    governing = strict_json(governing_path)
    require(governing["schema"] == "m528_single_port_same_ledger_recompute_contract_v1", "governing schema drift")
    require(int(governing["authorization"]["cpu_runs"]) == 1, "CPU authorization drift")
    require(int(governing["authorization"]["eda_runs"]) == 0 and int(governing["authorization"]["gpu_runs"]) == 0, "forbidden authorization drift")

    for item in governing["frozen_inputs"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing frozen input: {path}")
        require(sha256_file(path) == item["sha256"], f"frozen SHA drift: {path}")
    for item in execution["additional_frozen_inputs"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing additional input: {path}")
        require(sha256_file(path) == item["sha256"], f"additional SHA drift: {path}")

    m468_path = ROOT / governing["frozen_inputs"]["m468_r6_result"]["path"]
    m473_path = ROOT / governing["frozen_inputs"]["m473_r3_result"]["path"]
    m505_path = ROOT / governing["frozen_inputs"]["m505_result"]["path"]
    m468 = strict_json(m468_path)
    m473 = strict_json(m473_path)
    m505_result = strict_json(m505_path)

    m505_dir = m505_path.parent
    m505_entries = verify_manifest(
        m505_dir, governing["frozen_inputs"]["m505_manifest"]["sha256"]
    )
    require(m505_entries[m505_path.name] == governing["frozen_inputs"]["m505_result"]["sha256"], "M505 result not sealed")
    prior_csv = m505_dir / "m505_operator_sample_summary_r1.csv"
    require(m505_entries[prior_csv.name] == sha256_file(prior_csv), "M505 prior CSV not sealed")

    m473_dir = m473_path.parent
    m473_entries = verify_manifest(m473_dir)
    require(m473_entries[m473_path.name] == governing["frozen_inputs"]["m473_r3_result"]["sha256"], "M473 result not sealed")
    for output in m473["output_files"].values():
        require(m473_entries[output["path"]] == output["sha256"], "M473 output identity drift")

    module = load_frozen_m505()
    raw_item = m505_result["identity"]["m410r2_rows"]
    rows_path = ROOT / raw_item["path"]
    require(sha256_file(rows_path) == raw_item["sha256"], "transitively frozen row ledger drift")
    require(raw_item == execution["additional_frozen_inputs"]["m410r2_rows"], "execution/raw row identity mismatch")
    require(m505_result["identity"]["m505_analyzer"]["sha256"] == EXPECTED_M505_ANALYZER_SHA256, "M505 result analyzer identity drift")
    require(m505_result["identity"]["m504_analyzer"]["sha256"] == EXPECTED_M504_ANALYZER_SHA256, "M504 result analyzer identity drift")

    mapping = strict_json(ROOT / governing["frozen_inputs"]["sram_mapping"]["path"])
    inventory = mapping["generated_view_inventory"]
    require(inventory["logical_shape"] == "128x128b 1RW SP", "generated macro shape drift")
    require(
        math.isclose(
            float(inventory["area_um2"]) * 9,
            float(governing["capacity_recompute"]["generated_1rw_scratch_area_um2"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "nine-macro area drift",
    )

    shape = (SAMPLES, OPERATORS, CHUNKS, PARTITIONS)
    arrays = {name: np.zeros(shape, dtype=np.int32) for name in FIELD_NAMES}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(rows_path),),
    ) as pool:
        for phase, fields in pool.map(worker_phase, range(SAMPLES * OPERATORS * PARTITIONS), chunksize=args.chunksize):
            sample = phase // (OPERATORS * PARTITIONS)
            operator = (phase // PARTITIONS) % OPERATORS
            partition = phase % PARTITIONS
            for name in FIELD_NAMES:
                arrays[name][sample, operator, :, partition] = fields[name]

    aggregate = {name: int(value.astype(np.int64).sum()) for name, value in arrays.items()}
    require(aggregate["row_count"] == int(governing["frozen_coordinate"]["trace_rows"]), "trace-row conservation mismatch")
    require(aggregate["ideal_issue_cycles"] == aggregate["residual_nnz"] + aggregate["exact_parent_rows"], "aggregate arithmetic conservation mismatch")
    require(aggregate["dead_reads"] + aggregate["dead_forwards"] == aggregate["parent_edges"], "dead-only parent-edge conservation mismatch")
    require(aggregate["combined_reads"] + aggregate["combined_forwards"] == aggregate["parent_edges"], "combined parent-edge conservation mismatch")
    require(aggregate["dead_writes"] + aggregate["dead_elisions"] == aggregate["active_rows"], "dead-only completion conservation mismatch")
    require(
        aggregate["combined_writes"]
        + aggregate["combined_dead_elisions"]
        + aggregate["combined_single_use_elisions"]
        == aggregate["active_rows"],
        "combined completion conservation mismatch",
    )

    sample_rows: list[dict[str, Any]] = []
    for sample in range(SAMPLES):
        cycles = cycle_row(arrays, sample, None)
        row: dict[str, Any] = {
            "sample": sample,
            "aggregation_semantics": "sample_major_one_continuous_pipeline_plus_96000_commit",
            **cycles,
            **ratio_fields(cycles),
        }
        sample_rows.append(row)

    operator_rows: list[dict[str, Any]] = []
    for sample in range(SAMPLES):
        for operator in range(OPERATORS):
            cycles = cycle_row(arrays, sample, operator)
            row = {
                "sample": sample,
                "operator": operator,
                "aggregation_semantics": "operator_isolated_pipeline_no_commit_not_summable",
                **cycles,
                **ratio_fields(cycles),
            }
            operator_rows.append(row)

    totals = {
        key: sum(int(row[key]) for row in sample_rows)
        for key in sample_rows[0]
        if key.endswith("_cycles")
    }
    anchors = governing["frozen_aggregate_anchors"]
    expected_totals = {
        "m468_strong_zero_cycles": int(anchors["m468_strong_zero_cycles"]),
        "m473_same_coordinate_bit_cycles": int(anchors["m473_same_coordinate_bit_cycles"]),
        "m473_fused_concurrent_1r1w_ceiling_cycles": int(anchors["m473_fused_concurrent_1r1w_ceiling_cycles"]),
        "m505_dead_write_only_1rw_cycles": int(anchors["m505_dead_write_only_cycles"]),
        "m505_combined_pvrf_1rw_cycles": int(anchors["m505_combined_pvrf_cycles"]),
    }
    for key, expected in expected_totals.items():
        require(totals[key] == expected, f"frozen cycle anchor drift: {key}")
    require(
        totals["m504_all_write_1rw_cycles"]
        == int(m505_result["cycle_comparison"]["m504_single_port_cycles"]),
        "M504 all-write cycle anchor drift",
    )

    # Old M505 CSV is a sealed, independently generated operator-isolated
    # recurrence.  Cross-check its three single-port slices without using it as
    # the new sample-major denominator.
    prior_by_key: dict[tuple[int, int], dict[str, str]] = {}
    with prior_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prior_by_key[(int(row["sample"]), int(row["operator"]))] = row
    require(len(prior_by_key) == SAMPLES * OPERATORS, "prior M505 CSV population drift")
    for row in operator_rows:
        prior = prior_by_key[(int(row["sample"]), int(row["operator"]))]
        require(int(prior["m504_pipeline_slice_cycles_no_commit"]) == row["m504_all_write_1rw_cycles"], "M504 operator recurrence drift")
        require(int(prior["dead_only_pipeline_slice_cycles_no_commit"]) == row["m505_dead_write_only_1rw_cycles"], "dead-only operator recurrence drift")
        require(int(prior["m505_pipeline_slice_cycles_no_commit"]) == row["m505_combined_pvrf_1rw_cycles"], "combined operator recurrence drift")

    dead_accesses = aggregate["dead_reads"] + aggregate["dead_writes"]
    combined_accesses = aggregate["combined_reads"] + aggregate["combined_writes"]
    require(dead_accesses == int(anchors["m505_dead_write_only_parent_accesses_one_output_block"]), "dead-only access anchor drift")
    require(combined_accesses == int(anchors["m505_combined_parent_accesses_one_output_block"]), "combined access anchor drift")
    dead_bytes = dead_accesses * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR
    combined_bytes = combined_accesses * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR
    require(dead_bytes == int(anchors["m505_dead_write_only_parent_bytes_all_eight_output_blocks"]), "dead-only parent-byte anchor drift")
    require(combined_bytes == int(anchors["m505_combined_parent_bytes_all_eight_output_blocks"]), "combined parent-byte anchor drift")

    selected_m468 = min(
        (
            point
            for point in m468["points"]
            if point["mode"] == "strong_zero"
            and point["fits_both_240k_gates"] is True
            and int(point["resident_block_banks"]) == 8
            and str(point["bandwidth_bytes_per_cycle"]) == "128"
        ),
        key=lambda point: int(point["cycles"]),
    )
    selected_m473 = m473["best_128Bps_feasible_point"]
    require(int(selected_m468["cycles"]) == totals["m468_strong_zero_cycles"], "M468 selected-point drift")
    require(int(selected_m473["bit_cycles"]) == totals["m473_same_coordinate_bit_cycles"], "M473 bit selected-point drift")
    require(int(selected_m473["product_cycles"]) == totals["m473_fused_concurrent_1r1w_ceiling_cycles"], "M473 ceiling selected-point drift")
    require(int(selected_m468["weight_dram_bytes"]) == int(governing["frozen_coordinate"]["weight_dram_bytes"]), "weight traffic drift")
    require(int(selected_m468["source_sram_bytes"]) == int(governing["frozen_coordinate"]["source_sram_bytes"]), "source traffic drift")
    require(int(selected_m468["dma_commands"]) == int(selected_m473["weight_dma_commands"]), "DMA command drift")

    logical = dict(selected_m473["capacity"]["logical_items"])
    rounded = dict(selected_m473["capacity"]["macro_rounded_items"])
    require(logical["one_block_parent_scratch_signed12_row_indexed"] == int(governing["capacity_recompute"]["replace_nominal_scratch_bytes"]), "nominal scratch logical drift")
    require(rounded["one_block_parent_scratch_signed12_row_indexed"] == int(governing["capacity_recompute"]["replace_nominal_scratch_bytes"]), "nominal scratch rounded drift")
    rounded["one_block_parent_scratch_signed12_row_indexed"] = int(governing["capacity_recompute"]["generated_1rw_scratch_bytes"])
    logical["parent_liveness_class_2bit_per_row"] = 16
    rounded["parent_liveness_class_2bit_per_row"] = int(governing["capacity_recompute"]["conservative_extra_live_bitmap_macro_rounded_bytes"])
    m505_logical_total = sum(int(value) for value in logical.values())
    m505_rounded_total = sum(int(value) for value in rounded.values())
    require(m505_rounded_total == int(governing["capacity_recompute"]["m505_conservative_macro_rounded_bytes"]), "M505 rounded capacity drift")
    require(int(governing["capacity_recompute"]["budget_bytes"]) - m505_rounded_total == int(governing["capacity_recompute"]["m505_budget_margin_bytes"]), "M505 budget margin drift")
    capacity = {
        "budget_bytes": int(governing["capacity_recompute"]["budget_bytes"]),
        "m468_strong_zero": selected_m468["capacity"],
        "m473_concurrent_1r1w_ceiling": selected_m473["capacity"],
        "m505_dead_write_only_1rw": {
            "logical_items": logical,
            "macro_rounded_items": rounded,
            "logical_total_bytes": m505_logical_total,
            "macro_rounded_total_bytes": m505_rounded_total,
            "budget_margin_bytes": int(governing["capacity_recompute"]["budget_bytes"]) - m505_rounded_total,
            "generated_parent_scratch": {
                "organization": "9 x 128x128-bit 1RW SP; lower 64 rows used",
                "physical_capacity_bytes": int(governing["capacity_recompute"]["generated_1rw_scratch_bytes"]),
                "logical_payload_bytes": int(governing["capacity_recompute"]["replace_nominal_scratch_bytes"]),
                "area_um2": float(governing["capacity_recompute"]["generated_1rw_scratch_area_um2"]),
                "area_is_reported_separately_from_capacity": True,
            },
            "capacity_obligation_map": {
                "matcher_candidate_masks": "source_mask_pingpong",
                "parent_row_directory_and_order_tags": "descriptor32_pingpong",
                "one_cycle_parent_response_and_scheduler_queues": "fifo_control_reserve (at least 288 B of the 16384-B reserve is the two-entry 2x1152-bit response queue)",
                "liveness_metadata": "parent_liveness_class_2bit_per_row",
                "resident_accumulator": "psum plus psum_valid_bitmap",
                "ping_pong_load_compute_ownership": "source_mask_pingpong, descriptor32_pingpong, and weight_payload",
                "standard_cell_matcher_scheduler_area": "reported separately; never converted into free SRAM bytes",
            },
        },
    }

    traffic_rows = [
        {
            "design": "m468_strong_zero",
            "weight_dram_bytes": int(selected_m468["weight_dram_bytes"]),
            "source_sram_bytes": int(selected_m468["source_sram_bytes"]),
            "descriptor_write_bytes": 0,
            "candidate_store_search_read_bytes": 0,
            "descriptor_order_scan_read_bytes": 0,
            "parent_scratch_read_bytes": 0,
            "parent_scratch_write_bytes": 0,
            "dma_commands": int(selected_m468["dma_commands"]),
            "commit_cycles": int(selected_m468["commit_cycles"]),
        },
        {
            "design": "m473_fused_concurrent_1r1w_ceiling",
            "weight_dram_bytes": int(selected_m473["weight_dram_bytes"]),
            "source_sram_bytes": int(selected_m473["source_sram_bytes"]),
            "descriptor_write_bytes": int(selected_m473["descriptor_write_bytes"]),
            "candidate_store_search_read_bytes": int(selected_m473["candidate_store_search_read_bytes"]),
            "descriptor_order_scan_read_bytes": int(selected_m473["descriptor_order_scan_read_bytes"]),
            "parent_scratch_read_bytes": int(selected_m473["parent_scratch_read_bytes"]),
            "parent_scratch_write_bytes": int(selected_m473["parent_scratch_write_bytes"]),
            "dma_commands": int(selected_m473["weight_dma_commands"]),
            "commit_cycles": int(selected_m473["commit_cycles"]),
        },
        {
            "design": "m505_dead_write_only_1rw",
            "weight_dram_bytes": int(selected_m473["weight_dram_bytes"]),
            "source_sram_bytes": int(selected_m473["source_sram_bytes"]),
            "descriptor_write_bytes": int(selected_m473["descriptor_write_bytes"]),
            "candidate_store_search_read_bytes": int(selected_m473["candidate_store_search_read_bytes"]),
            "descriptor_order_scan_read_bytes": int(selected_m473["descriptor_order_scan_read_bytes"]),
            "parent_scratch_read_bytes": aggregate["dead_reads"] * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR,
            "parent_scratch_write_bytes": aggregate["dead_writes"] * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR,
            "dma_commands": int(selected_m473["weight_dma_commands"]),
            "commit_cycles": int(selected_m473["commit_cycles"]),
        },
        {
            "design": "m505_combined_pvrf_1rw_ablation",
            "weight_dram_bytes": int(selected_m473["weight_dram_bytes"]),
            "source_sram_bytes": int(selected_m473["source_sram_bytes"]),
            "descriptor_write_bytes": int(selected_m473["descriptor_write_bytes"]),
            "candidate_store_search_read_bytes": int(selected_m473["candidate_store_search_read_bytes"]),
            "descriptor_order_scan_read_bytes": int(selected_m473["descriptor_order_scan_read_bytes"]),
            "parent_scratch_read_bytes": aggregate["combined_reads"] * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR,
            "parent_scratch_write_bytes": aggregate["combined_writes"] * BLOCK_BANKS * BYTES_PER_PARENT_VECTOR,
            "dma_commands": int(selected_m473["weight_dma_commands"]),
            "commit_cycles": int(selected_m473["commit_cycles"]),
        },
    ]

    aggregate_cycles = {
        **totals,
        **ratio_fields(totals),
        "candidate_scope": "ten frozen samples and four H67 bottleneck Conv3x3 operators only",
        "m473_ceiling_is_diagnostic_not_rejection_denominator": True,
    }
    require(abs(aggregate_cycles["speedup_vs_m468_strong_zero"] - float(anchors["m505_speedup_vs_m468_strong_zero"])) < 1e-15, "M468 speedup anchor drift")
    require(abs(aggregate_cycles["speedup_vs_m473_same_coordinate_bit"] - float(anchors["m505_speedup_vs_same_coordinate_bit"])) < 1e-15, "bit speedup anchor drift")
    require(abs(aggregate_cycles["port_tax_vs_m473_ceiling"] - float(anchors["m505_port_tax_vs_m473_ceiling"])) < 1e-15, "ceiling tax anchor drift")

    conservation = {
        "trace_rows": aggregate["row_count"],
        "input_nonzero_bit_issues_per_output_block": aggregate["input_nnz"],
        "residual_nonzero_bit_issues_per_output_block": aggregate["residual_nnz"],
        "exact_parent_only_issues_per_output_block": aggregate["exact_parent_rows"],
        "product_arithmetic_issues_per_output_block": aggregate["ideal_issue_cycles"],
        "product_arithmetic_issues_all_eight_output_blocks": aggregate["ideal_issue_cycles"] * BLOCK_BANKS,
        "parent_edges_per_output_block": aggregate["parent_edges"],
        "dead_only_parent_reads_plus_forwards": aggregate["dead_reads"] + aggregate["dead_forwards"],
        "combined_parent_reads_plus_forwards": aggregate["combined_reads"] + aggregate["combined_forwards"],
        "active_rows_per_output_block": aggregate["active_rows"],
        "dead_only_writes_plus_dead_elisions": aggregate["dead_writes"] + aggregate["dead_elisions"],
        "combined_writes_plus_all_elisions": aggregate["combined_writes"] + aggregate["combined_dead_elisions"] + aggregate["combined_single_use_elisions"],
        "operator_row_tile_commits": SAMPLES * OPERATORS * CHUNKS,
        "committed_accumulator_vectors": int(governing["frozen_coordinate"]["commit_cycles"]),
        "commit_cycles": int(governing["frozen_coordinate"]["commit_cycles"]),
        "all_equalities_pass": True,
    }

    gates = governing["cpu_decision_gates"]
    decision = {
        "identity_and_conservation_pass": conservation["all_equalities_pass"],
        "fits_240k_macro_rounded_pass": m505_rounded_total <= int(governing["capacity_recompute"]["budget_bytes"]),
        "speedup_vs_m468_strong_zero_pass": aggregate_cycles["speedup_vs_m468_strong_zero"] >= float(gates["minimum_speedup_vs_m468_strong_zero"]),
        "speedup_vs_same_coordinate_bit_pass": aggregate_cycles["speedup_vs_m473_same_coordinate_bit"] >= float(gates["minimum_speedup_vs_same_coordinate_bit"]),
        "no_cycle_regression_from_frozen_dead_only_pass": totals["m505_dead_write_only_1rw_cycles"] <= int(anchors["m505_dead_write_only_cycles"]) * (1.0 + float(gates["maximum_cycle_regression_from_frozen_m505_dead_write_only"])),
        "m473_ceiling_distance_used_as_gate": False,
    }
    decision["bounded_rtl_preflight_authorized"] = all(
        value for key, value in decision.items() if key.endswith("_pass")
    )
    decision["verdict"] = (
        "GO_ONE_BOUNDED_DEAD_WRITE_ONLY_1RW_RTL_PREFLIGHT"
        if decision["bounded_rtl_preflight_authorized"]
        else "NO_GO_CLOSE_SINGLE_PORT_RTL_LINE"
    )

    result = {
        "schema": "m528_h67_single_port_same_ledger_recompute_result_v1",
        "date": governing["date"],
        "status": "PASS_EXACT_SAME_LEDGER_RECOMPUTE_PENDING_INDEPENDENT_RESULT_HAMMER",
        "scope": "one-sequence ten-sample four-bottleneck-Conv CPU cycle/traffic/capacity recompute",
        "identity": {
            "execution_contract": {"path": str(args.execution_contract.relative_to(ROOT)), "sha256": sha256_file(args.execution_contract)},
            "governing_contract": execution["governing_contract"],
            "analyzer": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__).resolve())},
            "row_ledger": raw_item,
            "m468_result": governing["frozen_inputs"]["m468_r6_result"],
            "m473_result": governing["frozen_inputs"]["m473_r3_result"],
            "m505_result": governing["frozen_inputs"]["m505_result"],
            "m505_manifest": governing["frozen_inputs"]["m505_manifest"],
            "sram_mapping": governing["frozen_inputs"]["sram_mapping"],
            "docs359": governing["frozen_inputs"]["docs359"],
        },
        "population": {
            "samples": SAMPLES,
            "operators": OPERATORS,
            "partitions": PARTITIONS,
            "rows_per_phase": ROWS_PER_PHASE,
            "row_tile": ROW_TILE,
            "tasks": int(np.prod(shape)),
            "task_order": governing["frozen_coordinate"]["task_order"],
        },
        "aggregate_cycles": aggregate_cycles,
        "distribution": {
            "sample_major": build_distribution_stats(sample_rows, "ten equal samples; continuous four-operator pipeline per sample plus one 96000-cycle commit"),
            "operator_isolated": build_distribution_stats(operator_rows, "forty equal operator slices; every slice restarts pipeline; no commit; diagnostic and not summable"),
        },
        "ablation": {
            "m504_all_write_1rw_cycles": totals["m504_all_write_1rw_cycles"],
            "m505_dead_write_only_1rw_cycles": totals["m505_dead_write_only_1rw_cycles"],
            "m505_combined_pvrf_1rw_cycles": totals["m505_combined_pvrf_1rw_cycles"],
            "m473_fused_concurrent_1r1w_ceiling_cycles": totals["m473_fused_concurrent_1r1w_ceiling_cycles"],
            "combined_pvrf_cycle_benefit_over_dead_only": totals["m505_dead_write_only_1rw_cycles"] - totals["m505_combined_pvrf_1rw_cycles"],
            "combined_pvrf_not_nominated": True,
        },
        "traffic": {
            "scope": "logical on-chip access bytes plus off-chip weight payload; not physical SRAM/DRAM energy",
            "rows": traffic_rows,
            "dead_only_parent_bytes_all_eight_blocks": dead_bytes,
            "combined_parent_bytes_all_eight_blocks": combined_bytes,
        },
        "capacity": capacity,
        "conservation": conservation,
        "decision": decision,
        "claim_boundary": governing["claim_boundary"],
    }

    args.out.mkdir(parents=True, exist_ok=False)
    result_path = args.out / "m528_h67_single_port_same_ledger_recompute_result_r1.json"
    sample_path = args.out / "m528_sample_major_distribution_r1.csv"
    operator_path = args.out / "m528_operator_isolated_distribution_r1.csv"
    traffic_path = args.out / "m528_traffic_ledger_r1.csv"
    capacity_path = args.out / "m528_capacity_ledger_r1.csv"
    readme_path = args.out / "README.md"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(sample_path, sample_rows, list(sample_rows[0].keys()))
    write_csv(operator_path, operator_rows, list(operator_rows[0].keys()))
    write_csv(traffic_path, traffic_rows, list(traffic_rows[0].keys()))
    capacity_rows: list[dict[str, Any]] = []
    for design, payload in (
        ("m468_strong_zero", selected_m468["capacity"]),
        ("m473_concurrent_1r1w_ceiling", selected_m473["capacity"]),
        ("m505_dead_write_only_1rw", capacity["m505_dead_write_only_1rw"]),
    ):
        for item, logical_bytes in payload["logical_items"].items():
            capacity_rows.append({
                "design": design,
                "item": item,
                "logical_bytes": int(logical_bytes),
                "macro_rounded_bytes": int(payload["macro_rounded_items"][item]),
            })
    write_csv(capacity_path, capacity_rows, list(capacity_rows[0].keys()))
    readme_path.write_text(
        "# M528 same-ledger recompute\n\n"
        "`m528_sample_major_distribution_r1.csv` is the only distribution with the "
        "frozen continuous four-operator pipeline and per-sample commit. "
        "`m528_operator_isolated_distribution_r1.csv` restarts the pipeline for every "
        "operator and omits commit; its rows are diagnostic and must not be summed.\n\n"
        "All cycle results remain CPU-model, four-bottleneck-Conv, one-sequence values. "
        "They are not RTL, Synopsys PPA, energy, full-network speedup, or a DATE headline.\n",
        encoding="utf-8",
    )
    manifest = args.out / "SHA256SUMS"
    payload_paths = sorted((result_path, sample_path, operator_path, traffic_path, capacity_path, readme_path))
    manifest.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in payload_paths),
        encoding="utf-8",
    )
    (args.out / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256_file(manifest)}  SHA256SUMS\n", encoding="utf-8"
    )
    print(json.dumps(aggregate_cycles, sort_keys=True))
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
