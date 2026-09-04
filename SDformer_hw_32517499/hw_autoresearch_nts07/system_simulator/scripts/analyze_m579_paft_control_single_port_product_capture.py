#!/usr/bin/env python3
"""Replay paired PAFT/control support traces through the M528 1RW schedule.

This is a support/cycle experiment, not a numeric Conv implementation.  It
reconstructs the exact 16-bit source rows from each packed trace, applies the
frozen M505 subset-parent and dead-write-only single-port recurrence, and
prices bit and product paths on the same task order and resource coordinate.
The result deliberately keeps three ratios separate:

  * bit issues / product issues: arithmetic opportunity;
  * bit cycles / 1RW product cycles: executable local cycle speedup;
  * control product cycles / PAFT product cycles: trained-activity increment.

The ratios must never be multiplied into a system or RTL headline.
"""

from __future__ import division, print_function

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "contracts/m579_paft_control_single_port_product_capture_execution_contract_r1_20260828.json"
DEFAULT_OUTPUT = ROOT / "results/m579_paft_control_single_port_product_capture_r1_20260828"

SAMPLES = 10
OPERATORS = 4
ROWS = 3000
PARTITIONS = 432
PARTITION_BITS = 16
ROW_TILE = 64
BLOCK_BANKS = 8
WEIGHT_DMA_CYCLES = 160
TAIL_CYCLES = 2
CAM_COMPARE_LANES = 64
COMMIT_PER_SAMPLE = 96000
CAPACITY_BYTES = 213376
BUDGET_BYTES = 240 * 1024

M43_PATH = ROOT / "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py"
M505_PATH = ROOT / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
M43_SHA256 = "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3"
M505_SHA256 = "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"

M43: Any = None
M505: Any = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Dict[str, Any]:
    def pairs(items: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise RuntimeError("non-standard JSON number: " + token)

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def load_module(name: str, path: Path, expected_sha: str) -> Any:
    require(path.is_file(), "missing frozen module: " + str(path))
    require(sha256_file(path) == expected_sha, "frozen module SHA drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_init() -> None:
    global M43, M505
    M43 = load_module("m579_frozen_m43", M43_PATH, M43_SHA256)
    M505 = load_module("m579_frozen_m505", M505_PATH, M505_SHA256)


def pipeline_cycles(preprocess: np.ndarray, work: np.ndarray) -> int:
    require(preprocess.shape == work.shape and preprocess.size > 0,
            "pipeline vector shape mismatch")
    total = int(preprocess[0])
    if preprocess.size > 1:
        total += int(np.maximum(work[:-1], preprocess[1:]).sum())
        total += (preprocess.size - 1) * TAIL_CYCLES
    total += int(work[-1]) + TAIL_CYCLES
    return total


def decode_partition_rows(masks: List[int], partition: int) -> np.ndarray:
    tile, subtile = divmod(partition, M43.TILE_BITS // PARTITION_BITS)
    shift = subtile * PARTITION_BITS
    values = np.empty(ROWS, dtype=np.uint16)
    for row in range(ROWS):
        values[row] = (masks[row * M43.TILES + tile] >> shift) & 0xffff
    return values


def analyze_record(job: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    arm, trace_dir_raw, record = job
    trace_dir = Path(trace_dir_raw)
    masks = M43.unpack_record_masks(trace_dir, record)
    require(len(masks) == ROWS * M43.TILES, "decoded mask extent drift")

    arrays = dict((name, []) for name in (
        "m468_pre", "bit_pre", "product_pre", "m468_work", "bit_work",
        "product_issue_work", "dead_1rw_work"))
    totals = Counter()

    for partition in range(PARTITIONS):
        rows = decode_partition_rows(masks, partition)
        for start in range(0, ROWS, ROW_TILE):
            tile = rows[start:min(start + ROW_TILE, ROWS)]
            row_count = int(tile.size)
            residual, parent = M505.M504.cleanroom_subset(tile)
            original_pc = M505.M504.POPCOUNT[tile].astype(np.int32)
            residual_pc = M505.M504.POPCOUNT[residual].astype(np.int32)
            active = tile != 0
            exact_parent = (parent >= 0) & (residual == 0) & active
            dead = M505.simulate_liveness_task(tile, False)

            input_nnz = int(original_pc.sum())
            search_rows = int(np.count_nonzero(original_pc > 1))
            product_issues = int(residual_pc.sum()) + int(np.count_nonzero(exact_parent))
            require(product_issues == int(dead["ideal_1r1w_issue_cycles"]),
                    "product issue conservation mismatch")
            require(int(dead["liveness_cycles"]) >= product_issues,
                    "single-port recurrence below arithmetic issue lower bound")

            bit_capture = (row_count + 7) // 8
            m468_frontend = row_count + 5
            bit_frontend = bit_capture + 2
            product_frontend = (
                bit_capture
                + search_rows * ((row_count + CAM_COMPARE_LANES - 1) // CAM_COMPARE_LANES)
                + 17 * bit_capture + 2)
            nonempty = input_nnz != 0

            arrays["m468_pre"].append(max(m468_frontend, WEIGHT_DMA_CYCLES)
                                      if nonempty else m468_frontend)
            arrays["bit_pre"].append(max(bit_frontend, WEIGHT_DMA_CYCLES)
                                    if nonempty else bit_frontend)
            arrays["product_pre"].append(max(product_frontend, WEIGHT_DMA_CYCLES)
                                        if nonempty else product_frontend)
            arrays["m468_work"].append(input_nnz * BLOCK_BANKS)
            arrays["bit_work"].append(input_nnz * BLOCK_BANKS)
            arrays["product_issue_work"].append(product_issues * BLOCK_BANKS)
            arrays["dead_1rw_work"].append(int(dead["liveness_cycles"]) * BLOCK_BANKS)

            totals["rows"] += row_count
            totals["input_nnz"] += input_nnz
            totals["active_rows"] += int(np.count_nonzero(active))
            totals["search_rows"] += search_rows
            totals["residual_nnz"] += int(residual_pc.sum())
            totals["exact_parent_rows"] += int(np.count_nonzero(exact_parent))
            totals["parent_edges"] += int(dead["parent_edges"])
            totals["product_issues"] += product_issues
            totals["dead_cycles"] += int(dead["liveness_cycles"])
            totals["dead_stalls"] += int(dead["liveness_stall_cycles"])
            totals["macro_reads"] += int(dead["macro_reads"])
            totals["macro_writes"] += int(dead["macro_writes"])
            totals["forwarded_reads"] += int(dead["forwarded_reads"])
            totals["dead_writes_elided"] += int(dead["dead_writes_elided"])

    expected_rows = ROWS * PARTITIONS
    require(totals["rows"] == expected_rows, "record row-partition population drift")
    require(totals["product_issues"] ==
            totals["residual_nnz"] + totals["exact_parent_rows"],
            "record arithmetic conservation mismatch")
    return {
        "arm": arm,
        "sample_id": int(record["sample_id"]),
        "operator_index": int(record["operator_index"]),
        "operator": record["operator"],
        "arrays": dict((key, np.asarray(value, dtype=np.int64))
                       for key, value in arrays.items()),
        "totals": dict(totals),
    }


def describe(values: List[float]) -> Dict[str, float]:
    require(values and all(value > 0.0 for value in values),
            "distribution requires positive values")
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "arithmetic_mean": mean,
        "geometric_mean": math.exp(sum(math.log(value) for value in values) /
                                   len(values)),
        "minimum": min(values),
        "maximum": max(values),
        "coefficient_of_variation_population": statistics.pstdev(values) / mean,
    }


def ordered_records(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = sorted(trace["records"],
                     key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(records) == SAMPLES * OPERATORS, "record count drift")
    for index, record in enumerate(records):
        require(record["sample_id"] == index // OPERATORS and
                record["operator_index"] == index % OPERATORS,
                "record sample/operator ordering drift")
        require(record["negative_count"] == 0,
                "M579 support replay currently requires nonnegative trace")
    return records


def summarize_arm(arm: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = dict(((row["sample_id"], row["operator_index"]), row)
                  for row in records)
    require(len(by_key) == SAMPLES * OPERATORS, "duplicate analyzed record")
    aggregate = Counter()
    sample_rows = []
    operator_totals = [Counter() for _ in range(OPERATORS)]

    for sample in range(SAMPLES):
        joined = defaultdict(list)
        sample_counter = Counter()
        for operator in range(OPERATORS):
            row = by_key[(sample, operator)]
            for key, value in row["arrays"].items():
                joined[key].append(value)
            sample_counter.update(row["totals"])
            operator_totals[operator].update(row["totals"])
        joined_np = dict((key, np.concatenate(value)) for key, value in joined.items())
        expected_tasks = OPERATORS * PARTITIONS * int(math.ceil(ROWS / float(ROW_TILE)))
        require(len(joined_np["bit_work"]) == expected_tasks,
                "sample task extent drift")
        cycles = {
            "strong_zero_cycles": pipeline_cycles(joined_np["m468_pre"], joined_np["m468_work"]) + COMMIT_PER_SAMPLE,
            "bit_cycles": pipeline_cycles(joined_np["bit_pre"], joined_np["bit_work"]) + COMMIT_PER_SAMPLE,
            "ideal_product_ceiling_cycles": pipeline_cycles(joined_np["product_pre"], joined_np["product_issue_work"]) + COMMIT_PER_SAMPLE,
            "single_port_product_cycles": pipeline_cycles(joined_np["product_pre"], joined_np["dead_1rw_work"]) + COMMIT_PER_SAMPLE,
        }
        cycles["arithmetic_work_reduction"] = sample_counter["input_nnz"] / float(sample_counter["product_issues"])
        cycles["local_cycle_speedup_vs_bit"] = cycles["bit_cycles"] / float(cycles["single_port_product_cycles"])
        cycles["local_cycle_speedup_vs_strong_zero"] = cycles["strong_zero_cycles"] / float(cycles["single_port_product_cycles"])
        cycles["single_port_tax_vs_ideal_product_ceiling"] = cycles["single_port_product_cycles"] / float(cycles["ideal_product_ceiling_cycles"]) - 1.0
        sample_rows.append(dict({"arm": arm, "sample_id": sample}, **cycles))
        aggregate.update(sample_counter)
        for key in ("strong_zero_cycles", "bit_cycles", "ideal_product_ceiling_cycles", "single_port_product_cycles"):
            aggregate[key] += int(cycles[key])

    aggregate["bit_issues_all_output_blocks"] = aggregate["input_nnz"] * BLOCK_BANKS
    aggregate["product_issues_all_output_blocks"] = aggregate["product_issues"] * BLOCK_BANKS
    require(aggregate["product_issues"] ==
            aggregate["residual_nnz"] + aggregate["exact_parent_rows"],
            "arm arithmetic conservation mismatch")

    ratios = {
        "arithmetic_work_reduction": aggregate["bit_issues_all_output_blocks"] /
                                     float(aggregate["product_issues_all_output_blocks"]),
        "local_cycle_speedup_vs_bit": aggregate["bit_cycles"] /
                                      float(aggregate["single_port_product_cycles"]),
        "local_cycle_speedup_vs_strong_zero": aggregate["strong_zero_cycles"] /
                                              float(aggregate["single_port_product_cycles"]),
        "single_port_tax_vs_ideal_product_ceiling":
            aggregate["single_port_product_cycles"] /
            float(aggregate["ideal_product_ceiling_cycles"]) - 1.0,
    }
    distributions = {}
    for key in ratios:
        distributions[key] = describe([float(row[key]) for row in sample_rows])

    return {
        "arm": arm,
        "aggregate": dict(aggregate),
        "ratios": ratios,
        "per_sample": sample_rows,
        "per_operator_support_counters": [dict(row) for row in operator_totals],
        "per_sample_ratio_distributions": distributions,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refuse to overwrite M579 output")
    require(1 <= args.workers <= 3, "M579 workers must be 1..3")

    contract = strict_json(contract_path)
    require(contract["schema"] ==
            "m579_paft_control_single_port_product_capture_execution_contract_v1",
            "contract schema drift")
    require(contract["authorization"]["launch_now"] is True and
            contract["authorization"]["run_cpu"] is True and
            contract["authorization"]["max_attempts"] == 1,
            "M579 production admission not enabled")
    require(contract["authorization"]["run_gpu"] is False and
            contract["authorization"]["run_eda"] is False,
            "forbidden execution authorization")

    identities = {}
    paths = {}
    for name, spec in contract["inputs"].items():
        path = ROOT / spec["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256_file(path)
        require(observed == spec["sha256"], "input SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}
    require(sha256_file(Path(__file__).resolve()) ==
            contract["analyzer_sha256"], "analyzer SHA drift")

    paft = strict_json(paths["paft_trace"])
    control = strict_json(paths["control_trace"])
    accuracy = strict_json(paths["paired_valid825"])
    require(paft["status"] == "PASS_PAFT_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE",
            "PAFT trace status drift")
    require(control["status"] == "PASS_CONTROL_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE",
            "control trace status drift")
    require(paft["identity"]["capture_bn_policy"] == "running" and
            control["identity"]["capture_bn_policy"] == "running",
            "running-BN identity drift")
    require(paft["cohort"]["sample_keys"] == control["cohort"]["sample_keys"] and
            paft["cohort"]["operators"] == control["cohort"]["operators"],
            "paired cohort/operator identity drift")
    require(accuracy["status"] == "PASS_SINGLE_SEED_SMALL_POSITIVE_RUNNING_BN_DIRECTION",
            "paired valid825 status drift")

    jobs = []
    for arm, trace, path in (("control", control, paths["control_trace"]),
                             ("paft", paft, paths["paft_trace"])):
        for record in ordered_records(trace):
            jobs.append((arm, str(path.parent), record))

    analyzed = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=worker_init) as pool:
        futures = [pool.submit(analyze_record, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            analyzed.append(future.result())
            print("[M579] analyzed {}/{} records".format(index, len(futures)),
                  flush=True)

    arms = {}
    for arm in ("control", "paft"):
        arms[arm] = summarize_arm(arm, [row for row in analyzed
                                        if row["arm"] == arm])
    trained_increment = {
        "candidate_cycle_reduction_percent":
            (1.0 - arms["paft"]["aggregate"]["single_port_product_cycles"] /
             float(arms["control"]["aggregate"]["single_port_product_cycles"])) * 100.0,
        "candidate_throughput_gain_control_over_paft":
            arms["control"]["aggregate"]["single_port_product_cycles"] /
            float(arms["paft"]["aggregate"]["single_port_product_cycles"]),
        "bit_cycle_reduction_percent":
            (1.0 - arms["paft"]["aggregate"]["bit_cycles"] /
             float(arms["control"]["aggregate"]["bit_cycles"])) * 100.0,
        "global_valid825_control_aee": accuracy["paired_validation"]["running"]["metrics"]["AEE"]["control_frame_equal_mean"],
        "global_valid825_paft_aee": accuracy["paired_validation"]["running"]["metrics"]["AEE"]["paft_frame_equal_mean"],
        "global_valid825_paft_aee_improvement_percent": accuracy["hardware_decision"]["paft_running_aee_improvement_percent"],
        "single_seed": True,
    }

    pass_gate = all(
        arms[arm]["ratios"]["local_cycle_speedup_vs_bit"] >= 1.5
        for arm in ("control", "paft")) and CAPACITY_BYTES <= BUDGET_BYTES
    payload = {
        "schema": "m579_paft_control_single_port_product_capture_v1",
        "status": ("PASS_PAIRED_SUPPORT_ONLY_SINGLE_PORT_PRODUCT_CAPTURE_LOCAL_CYCLE"
                   if pass_gate else
                   "NO_GO_PAIRED_SUPPORT_ONLY_SINGLE_PORT_PRODUCT_CAPTURE"),
        "identity": identities,
        "scope": {
            "samples": SAMPLES,
            "operators": paft["cohort"]["operators"],
            "rows_per_operator": ROWS,
            "partitions_per_operator": PARTITIONS,
            "row_tile": ROW_TILE,
            "output_blocks": BLOCK_BANKS,
            "checkpoint_arms": {
                "control": control["identity"]["checkpoint_sha256"],
                "paft": paft["identity"]["checkpoint_sha256"],
            },
        },
        "resource_coordinate": {
            "single_port_parent_scratch": "one synchronous 1152-bit 1RW access per cycle",
            "resident_output_block_banks": BLOCK_BANKS,
            "macro_rounded_capacity_bytes": CAPACITY_BYTES,
            "capacity_budget_bytes": BUDGET_BYTES,
            "capacity_ratio": CAPACITY_BYTES / float(BUDGET_BYTES),
            "weight_dma_cycles_per_task": WEIGHT_DMA_CYCLES,
            "commit_cycles_per_sample": COMMIT_PER_SAMPLE,
        },
        "arms": arms,
        "trained_activity_increment": trained_increment,
        "decision": {
            "minimum_local_cycle_speedup_each_arm": 1.5,
            "capacity_must_fit_240k": True,
            "pass": pass_gate,
        },
        "claim_boundary": {
            "paired_support_trace": True,
            "same_cycle_model_and_resource_coordinate": True,
            "exact_support_parent_residual_conservation": True,
            "arithmetic_work_reduction": True,
            "local_cpu_cycle_speedup": True,
            "single_seed_paft_accuracy_direction": True,
            "numeric_conv_or_acc24_equivalence": False,
            "rtl": False,
            "vcs": False,
            "synopsys_ppa": False,
            "energy": False,
            "decoder_complete": False,
            "system_speedup": False,
            "headline": False,
            "ratios_may_be_multiplied": False,
        },
    }

    output_dir.mkdir(parents=True)
    result_path = output_dir / "m579_paft_control_single_port_product_capture_r1.json"
    sample_rows = arms["control"]["per_sample"] + arms["paft"]["per_sample"]
    write_csv(output_dir / "m579_per_sample_cycles_r1.csv", sample_rows)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(sha256_file(Path(__file__).resolve()) == contract["analyzer_sha256"],
            "analyzer changed during run")
    print("PASS M579 result={}".format(result_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
