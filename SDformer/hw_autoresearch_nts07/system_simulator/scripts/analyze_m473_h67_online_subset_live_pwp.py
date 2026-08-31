#!/usr/bin/env python3
"""M473 exact online-subset and live-parent-scratch H67 Conv DSE."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTION_CONTRACT = (
    ROOT
    / "contracts"
    / "m473_h67_online_subset_live_pwp_execution_contract_r1_20260826.json"
)
DEFAULT_OUT = (
    ROOT / "results" / "m473_h67_online_subset_live_pwp_r1_20260826"
)
BYTES_PER_ROW = 9
ROWS_PER_PHASE = 3000
PARTITIONS = 432
SAMPLES = 10
OPERATORS = 4
PHASES = SAMPLES * OPERATORS * PARTITIONS
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
POPCOUNT = np.array([value.bit_count() for value in range(1 << 16)], dtype=np.uint8)

_ROWS_FD: int | None = None
_TILES: tuple[int, ...] = ()


TASK_FIELDS = (
    "row_count",
    "input_nnz",
    "active_rows",
    "zero_rows",
    "search_rows",
    "parent_rows",
    "partial_parent_rows",
    "exact_parent_rows",
    "residual_nnz",
    "product_issue_per_block",
    "live_parent_rows",
    "peak_live_rows",
    "max_refcount",
    "future_parent_rows",
    "maximum_parent_span",
    "reconstruction_mismatches",
    "topological_mismatches",
)


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
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError(f"non-standard JSON token: {token}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def git_stdout(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def bit_bytes(bits: int) -> int:
    return (int(bits) + 7) // 8


def depth64(rows: int) -> int:
    if rows <= 0:
        return 0
    return int(math.ceil(rows / 64.0) * 64)


def cleanroom_subset(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce official find_product_sparsity on uint16 masks exactly."""

    masks = np.asarray(masks, dtype=np.uint16)
    count = masks.size
    indices = np.arange(count, dtype=np.int32)
    residual = masks.copy()
    parent = np.full(count, -1, dtype=np.int16)
    for row, current in enumerate(masks):
        if POPCOUNT[int(current)] < 2:
            continue
        subset = np.bitwise_and(masks, current) == masks
        equal_later = (masks == current) & (indices >= row)
        candidate_indices = np.flatnonzero(subset & ~equal_later)
        if candidate_indices.size == 0:
            continue
        candidate_popcounts = POPCOUNT[masks[candidate_indices]]
        # Match the official simulator's fail-closed rule exactly: an all-zero
        # subset is not a reusable parent even though it is a set-theoretic
        # subset of every row.
        if int(candidate_popcounts.max(initial=0)) < 1:
            continue
        chosen = int(candidate_indices[int(np.argmax(candidate_popcounts))])
        parent[row] = chosen
        residual[row] = np.uint16(current ^ masks[chosen])
    return residual, parent


def task_metrics(masks: np.ndarray) -> dict[str, int]:
    masks = np.asarray(masks, dtype=np.uint16)
    residual, parent = cleanroom_subset(masks)
    original_pc = POPCOUNT[masks].astype(np.int32)
    residual_pc = POPCOUNT[residual].astype(np.int32)
    active = masks != 0
    has_parent = parent >= 0
    exact_parent = has_parent & (residual == 0) & active
    partial_parent = has_parent & (residual != 0)
    product_issue = int(residual_pc.sum() + np.count_nonzero(exact_parent))

    reconstruction_mismatches = 0
    for row in range(masks.size):
        reconstructed = int(residual[row])
        if parent[row] >= 0:
            reconstructed ^= int(masks[parent[row]])
        if reconstructed != int(masks[row]):
            reconstruction_mismatches += 1

    pop_order = np.lexsort(
        (np.arange(masks.size, dtype=np.int32), original_pc)
    )
    position = np.empty(masks.size, dtype=np.int32)
    position[pop_order] = np.arange(masks.size, dtype=np.int32)
    topological_mismatches = int(
        sum(
            parent[row] >= 0
            and position[parent[row]] >= position[row]
            for row in range(masks.size)
        )
    )

    referenced = parent[parent >= 0].astype(np.int64)
    refcount = np.bincount(referenced, minlength=masks.size).astype(np.int32)
    original_refcount = refcount.copy()
    live = 0
    peak = 0
    for row in pop_order:
        # Conservative capacity: a newly live result is charged before the
        # last-read parent slot is reclaimed in the same issue step.
        transient = live + int(refcount[row] > 0)
        peak = max(peak, transient)
        chosen = int(parent[row])
        if chosen >= 0:
            refcount[chosen] -= 1
            require(refcount[chosen] >= 0, "negative parent refcount")
            if refcount[chosen] == 0:
                live -= 1
        if refcount[row] > 0:
            live += 1
    require(live == 0, "live parent scratch does not drain")
    require(np.all(refcount == 0), "parent refcounts do not drain")

    parent_rows = int(np.count_nonzero(has_parent))
    parent_span = (
        np.abs(np.flatnonzero(has_parent) - parent[has_parent]).max(initial=0)
        if parent_rows
        else 0
    )
    return {
        "row_count": int(masks.size),
        "input_nnz": int(original_pc.sum()),
        "active_rows": int(np.count_nonzero(active)),
        "zero_rows": int(np.count_nonzero(~active)),
        "search_rows": int(np.count_nonzero(original_pc > 1)),
        "parent_rows": parent_rows,
        "partial_parent_rows": int(np.count_nonzero(partial_parent)),
        "exact_parent_rows": int(np.count_nonzero(exact_parent)),
        "residual_nnz": int(residual_pc.sum()),
        "product_issue_per_block": product_issue,
        "live_parent_rows": int(np.count_nonzero(original_refcount)),
        "peak_live_rows": int(peak),
        "max_refcount": int(original_refcount.max(initial=0)),
        "future_parent_rows": int(
            np.count_nonzero(has_parent & (parent > np.arange(masks.size)))
        ),
        "maximum_parent_span": int(parent_span),
        "reconstruction_mismatches": int(reconstruction_mismatches),
        "topological_mismatches": int(topological_mismatches),
    }


def worker_init(rows_path: str, tiles: tuple[int, ...]) -> None:
    global _ROWS_FD, _TILES
    _ROWS_FD = os.open(rows_path, os.O_RDONLY)
    _TILES = tiles


def read_phase(phase_index: int) -> np.ndarray:
    require(_ROWS_FD is not None, "worker rows file is not open")
    phase_bytes = ROWS_PER_PHASE * BYTES_PER_ROW
    raw = os.pread(_ROWS_FD, phase_bytes, phase_index * phase_bytes)
    require(len(raw) == phase_bytes, f"short phase read: {phase_index}")
    lines = raw.splitlines()
    require(len(lines) == ROWS_PER_PHASE, f"phase row mismatch: {phase_index}")
    words = np.fromiter((int(line, 16) for line in lines), dtype=np.uint32)
    require(words.size == ROWS_PER_PHASE, "decoded row population mismatch")
    return np.bitwise_and(words, np.uint32(0xFFFF)).astype(np.uint16)


def worker_phase(phase_index: int) -> tuple[int, dict[int, dict[str, np.ndarray]]]:
    masks = read_phase(phase_index)
    tile_payload: dict[int, dict[str, np.ndarray]] = {}
    for tile in _TILES:
        chunks = int(math.ceil(ROWS_PER_PHASE / tile))
        fields = {
            field: np.zeros(chunks, dtype=np.int32) for field in TASK_FIELDS
        }
        for chunk, start in enumerate(range(0, ROWS_PER_PHASE, tile)):
            metrics = task_metrics(masks[start : min(start + tile, ROWS_PER_PHASE)])
            for field in TASK_FIELDS:
                fields[field][chunk] = metrics[field]
        tile_payload[tile] = fields
    return phase_index, tile_payload


def official_find_product_sparsity(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_prosperity_official_probe import load_official_api

    _, _, Simulator, _ = load_official_api()
    function = Simulator.run_fc.__globals__["find_product_sparsity"]
    bits = (
        (masks[:, None].astype(np.uint32) >> np.arange(16, dtype=np.uint32))
        & 1
    ).astype(np.uint8)
    official_residual, official_parent = function(torch.from_numpy(bits))
    residual_bits = official_residual.numpy().astype(np.uint16)
    residual = np.sum(
        residual_bits << np.arange(16, dtype=np.uint16), axis=1
    ).astype(np.uint16)
    return residual, official_parent.numpy().astype(np.int16)


def deterministic_mapping_checks(
    rows_path: Path,
    tiles: tuple[int, ...],
    required: int,
) -> list[dict[str, Any]]:
    global _ROWS_FD, _TILES
    previous_fd = _ROWS_FD
    if previous_fd is not None:
        os.close(previous_fd)
    _ROWS_FD = os.open(rows_path, os.O_RDONLY)
    _TILES = tiles
    rng = np.random.default_rng(473)
    cases: list[tuple[int, int, int]] = []
    # Stratify the official-code parity checks across every frozen sample and
    # operator. Tile sizes, partitions and row starts rotate deterministically;
    # the three edge cases retain the first/last phase and short final tile.
    mandatory = []
    for sample in range(SAMPLES):
        for operator in range(OPERATORS):
            tile = int(tiles[(sample * OPERATORS + operator) % len(tiles)])
            partition = int((sample * 97 + operator * 53) % PARTITIONS)
            starts = list(range(0, ROWS_PER_PHASE, tile))
            start = int(starts[(sample * OPERATORS + operator) % len(starts)])
            phase = (sample * OPERATORS + operator) * PARTITIONS + partition
            mandatory.append((phase, tile, start))
    mandatory.extend(
        [
            (0, tiles[-1], 0),
            (PARTITIONS - 1, tiles[-1], 0),
            (
                PHASES - 1,
                tiles[0],
                (ROWS_PER_PHASE // tiles[0]) * tiles[0],
            ),
        ]
    )
    cases.extend(mandatory)
    while len(cases) < required:
        phase = int(rng.integers(0, PHASES))
        tile = int(tiles[int(rng.integers(0, len(tiles)))])
        starts = list(range(0, ROWS_PER_PHASE, tile))
        start = int(starts[int(rng.integers(0, len(starts)))])
        case = (phase, tile, start)
        if case not in cases:
            cases.append(case)

    checks = []
    for phase, tile, start in cases:
        phase_masks = read_phase(phase)
        masks = phase_masks[start : min(start + tile, ROWS_PER_PHASE)]
        residual, parent = cleanroom_subset(masks)
        official_residual, official_parent = official_find_product_sparsity(masks)
        residual_mismatches = int(np.count_nonzero(residual != official_residual))
        parent_mismatches = int(np.count_nonzero(parent != official_parent))
        checks.append(
            {
                "phase_index": phase,
                "sample": phase // (OPERATORS * PARTITIONS),
                "operator": (phase // PARTITIONS) % OPERATORS,
                "partition": phase % PARTITIONS,
                "row_tile": tile,
                "row_start": start,
                "rows": int(masks.size),
                "residual_mismatches": residual_mismatches,
                "parent_mismatches": parent_mismatches,
                "pass": residual_mismatches == 0 and parent_mismatches == 0,
            }
        )
    os.close(_ROWS_FD)
    _ROWS_FD = previous_fd
    return checks


def validate_frozen_semantics(
    contract: dict[str, Any],
    m468: dict[str, Any],
    m468_hammer: dict[str, Any],
    m41: dict[str, Any],
) -> dict[str, Any]:
    anchors = contract["frozen_semantic_anchors"]
    require(m468["status"] == anchors["m468_status"], "M468 status drift")
    require(
        m468["exact_lazy_pwp_generation"]["signed12_bound_pass"] is True,
        "M468 signed12 bound is not admitted",
    )
    require(
        int(m468["schedule"]["accumulator_bits"])
        == int(anchors["accumulator_bits"]),
        "M468 accumulator width drift",
    )
    require(
        m468_hammer["status"] == anchors["m468_hammer_status"],
        "M468 hammer status drift",
    )
    require(
        int(m468_hammer["score"]) == int(anchors["m468_hammer_score"]),
        "M468 hammer score drift",
    )
    require(m41["status"] == anchors["m41_status"], "M41 status drift")
    require(
        int(m41["accumulator_width"]["checkpoint_tight_signed_bits"])
        == int(anchors["accumulator_bits"]),
        "M41 accumulator width drift",
    )

    selected: dict[str, dict[str, int]] = {}
    for banks_text, expected in anchors["m468_128Bps_strong_zero"].items():
        banks = int(banks_text)
        candidates = [
            point
            for point in m468["points"]
            if point["mode"] == "strong_zero"
            and point["bandwidth_bytes_per_cycle"] == 128
            and point["fits_both_240k_gates"] is True
            and int(point["resident_block_banks"]) == banks
        ]
        require(candidates, f"missing M468 {banks}-bank 128-B/cycle anchor")
        best = min(candidates, key=lambda point: int(point["cycles"]))
        require(
            int(best["cycles"]) == int(expected["cycles"])
            and int(best["row_tile"]) == int(expected["row_tile"]),
            f"M468 {banks}-bank 128-B/cycle anchor drift",
        )
        selected[banks_text] = {
            "cycles": int(best["cycles"]),
            "row_tile": int(best["row_tile"]),
        }
    return {
        "status": "PASS_FROZEN_SEMANTIC_ANCHORS",
        "m468_status": m468["status"],
        "m468_hammer_status": m468_hammer["status"],
        "m468_hammer_score": int(m468_hammer["score"]),
        "m41_status": m41["status"],
        "accumulator_bits": int(anchors["accumulator_bits"]),
        "m468_128Bps_strong_zero": selected,
    }


def pipeline_cycles(preprocess: np.ndarray, work: np.ndarray, tail: int) -> int:
    preprocess = np.asarray(preprocess, dtype=np.int64)
    work = np.asarray(work, dtype=np.int64)
    require(preprocess.shape == work.shape, "pipeline shape mismatch")
    require(preprocess.size > 0, "empty pipeline")
    total = int(preprocess[0])
    if preprocess.size > 1:
        total += int(np.maximum(work[:-1], preprocess[1:]).sum())
        total += (preprocess.size - 1) * tail
    total += int(work[-1]) + tail
    return total


def capacity_breakdown(
    row_tile: int,
    block_banks: int,
    peak_live_rows: int,
    fixed_reserve: int,
) -> dict[str, Any]:
    tile_depth = depth64(row_tile)
    # The admitted physical point is deliberately row-indexed. Peak-live
    # compact allocation remains an offline diagnostic and cannot reduce the
    # capacity gate before a row-to-slot map/free-list is physicalized.
    scratch_depth = depth64(row_tile)
    half_slots = 2 if block_banks == 8 else 1
    logical = {
        "psum": bit_bytes(row_tile * block_banks * OUTPUT_LANES * 19),
        "active_bitmap": bit_bytes(row_tile),
        "psum_valid_bitmap": bit_bytes(row_tile * block_banks),
        "source_mask_pingpong": row_tile * 2 * 2,
        "descriptor32_pingpong": row_tile * 4 * 2,
        "weight_payload": half_slots * 6144,
        "one_block_parent_scratch_signed12_row_indexed": row_tile * 144,
        "fifo_control_reserve": fixed_reserve,
    }
    rounded = {
        "psum": block_banks * 13 * 18 * tile_depth,
        "active_bitmap": 18 * tile_depth,
        "psum_valid_bitmap": 18 * tile_depth,
        "source_mask_pingpong": 2 * 18 * tile_depth,
        "descriptor32_pingpong": 2 * 18 * tile_depth,
        "weight_payload": half_slots * 22 * 18 * 64,
        "one_block_parent_scratch_signed12_row_indexed": 8 * 18 * scratch_depth,
        "fifo_control_reserve": fixed_reserve,
    }
    for key, logical_bytes in logical.items():
        require(rounded[key] >= logical_bytes, f"macro undercounts {key}")
    return {
        "logical_items": logical,
        "macro_rounded_items": rounded,
        "logical_total_bytes": int(sum(logical.values())),
        "macro_rounded_total_bytes": int(sum(rounded.values())),
        "row_tile_macro_depth": tile_depth,
        "peak_live_rows_diagnostic_only": peak_live_rows,
        "parent_scratch_entries_charged": row_tile,
        "parent_scratch_macro_depth": scratch_depth,
        "parent_scratch_read_port_bytes_per_cycle": 144,
        "parent_scratch_write_port_bytes_per_cycle": 144,
        "parent_scratch_port_obligation": "1R1W",
        "cam_and_scheduler_physicalized": False,
    }


def data_cycles(byte_count: int, bandwidth: int | str) -> int:
    if bandwidth == "infinite":
        return 0
    return int(math.ceil(byte_count / int(bandwidth)))


def flatten_sample(array: np.ndarray, sample: int) -> np.ndarray:
    # Stored as sample, operator, row-chunk, partition, which is the frozen
    # M468 row-tile-major execution order inside one sample.
    return np.asarray(array[sample]).reshape(-1).astype(np.int64)


def build_points(
    task_arrays: dict[int, dict[str, np.ndarray]],
    contract: dict[str, Any],
    m468: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    axes = contract["dse_axes"]
    cycle = contract["cycle_model"]
    capacity_contract = contract["capacity_model"]
    budget = int(capacity_contract["budget_bytes"])
    fixed_reserve = int(
        capacity_contract["fixed_fifo_and_control_reserve_bytes"]
    )

    best_m468_zero: dict[tuple[int, str], dict[str, Any]] = {}
    for point in m468["points"]:
        if point["mode"] != "strong_zero" or not point["fits_both_240k_gates"]:
            continue
        key = (int(point["resident_block_banks"]), str(point["bandwidth_bytes_per_cycle"]))
        prior = best_m468_zero.get(key)
        if prior is None or point["cycles"] < prior["cycles"]:
            best_m468_zero[key] = point

    points: list[dict[str, Any]] = []
    for row_tile in axes["row_tile"]:
        arrays = task_arrays[int(row_tile)]
        peak_live = int(arrays["peak_live_rows"].max(initial=0))
        for banks in axes["resident_block_banks"]:
            capacity = capacity_breakdown(
                int(row_tile), int(banks), peak_live, fixed_reserve
            )
            fits_logical = capacity["logical_total_bytes"] <= budget
            fits_macro = capacity["macro_rounded_total_bytes"] <= budget
            fits_both = fits_logical and fits_macro
            passes = 2 if int(banks) == 4 else 1
            for bandwidth in axes["bandwidth_bytes_per_cycle"]:
                half_dma = (
                    data_cycles(cycle["weight_bytes_per_four_blocks"], bandwidth)
                    + int(cycle["dma_command_setup_cycles"])
                )
                weight_dma = half_dma if int(banks) == 4 else 2 * half_dma
                for cam_lanes in axes["cam_compare_lanes"]:
                    effective_cam = min(int(cam_lanes), int(row_tile))
                    bit_total_without_commit = 0
                    product_without_commit = {
                        mode: 0 for mode in axes["scratch_latency_mode"]
                    }
                    aggregate = {field: 0 for field in TASK_FIELDS}
                    nonempty_tasks = 0
                    task_count = 0
                    for sample in range(SAMPLES):
                        row_count = flatten_sample(arrays["row_count"], sample)
                        input_nnz = flatten_sample(arrays["input_nnz"], sample)
                        search_rows = flatten_sample(arrays["search_rows"], sample)
                        direct_issue = input_nnz
                        product_issue = flatten_sample(
                            arrays["product_issue_per_block"], sample
                        )
                        nonempty = input_nnz != 0
                        bit_frontend = (
                            (row_count + int(cycle["popcount_lanes"]) - 1)
                            // int(cycle["popcount_lanes"])
                            + 2
                        )
                        capture_cycles = bit_frontend - 2
                        search_sweeps = (
                            row_count + effective_cam - 1
                        ) // effective_cam
                        product_frontend = (
                            capture_cycles
                            + search_rows * search_sweeps
                            + 17 * capture_cycles
                            + 2
                        )
                        bit_preprocess = np.where(
                            nonempty,
                            np.maximum(bit_frontend, weight_dma),
                            bit_frontend,
                        )
                        product_preprocess = np.where(
                            nonempty,
                            np.maximum(product_frontend, weight_dma),
                            product_frontend,
                        )
                        bit_work = direct_issue * int(banks)
                        parent_edges = flatten_sample(
                            arrays["parent_rows"], sample
                        )
                        active_rows = flatten_sample(
                            arrays["active_rows"], sample
                        )
                        bit_total_without_commit += passes * pipeline_cycles(
                            bit_preprocess,
                            bit_work,
                            int(cycle["tail_cycles_per_pass"]),
                        )
                        for latency_mode in axes["scratch_latency_mode"]:
                            product_work_per_block = product_issue.copy()
                            if latency_mode == "unfused_sync_upper":
                                product_work_per_block = (
                                    product_work_per_block
                                    + parent_edges
                                    + active_rows
                                )
                            else:
                                require(
                                    latency_mode == "fused_forwarded_1r1w",
                                    "unknown scratch latency mode",
                                )
                            product_without_commit[latency_mode] += (
                                passes
                                * pipeline_cycles(
                                    product_preprocess,
                                    product_work_per_block * int(banks),
                                    int(cycle["tail_cycles_per_pass"]),
                                )
                            )
                        nonempty_tasks += int(np.count_nonzero(nonempty)) * passes
                        task_count += int(input_nnz.size) * passes
                        for field in TASK_FIELDS:
                            aggregate[field] += int(
                                flatten_sample(arrays[field], sample).sum()
                            )

                    commit_cycles = SAMPLES * int(cycle["commit_cycles_per_sample"])
                    bit_cycles = bit_total_without_commit + commit_cycles
                    weight_bytes = (
                        nonempty_tasks
                        * int(cycle["weight_bytes_per_four_blocks"])
                        * (2 if int(banks) == 8 else 1)
                    )
                    weight_dma_commands = nonempty_tasks * (
                        2 if int(banks) == 8 else 1
                    )
                    source_sram_bytes = aggregate["row_count"] * 2 * passes
                    descriptor_write_bytes = aggregate["row_count"] * 4 * passes
                    candidate_store_search_read_bytes = (
                        int(
                            (
                                arrays["search_rows"].astype(np.int64)
                                * arrays["row_count"].astype(np.int64)
                            ).sum()
                        )
                        * 2
                        * passes
                    )
                    descriptor_order_scan_read_bytes = (
                        aggregate["row_count"] * 4 * 17 * passes
                    )
                    scratch_read_bytes = (
                        aggregate["parent_rows"]
                        * OUTPUT_BLOCKS
                        * int(cycle["pwp_scratch_read_bytes_per_parent_row_per_output_block"])
                    )
                    scratch_write_bytes = (
                        aggregate["active_rows"]
                        * OUTPUT_BLOCKS
                        * int(cycle["pwp_scratch_write_bytes_per_active_row_per_output_block"])
                    )
                    key = (int(banks), str(bandwidth))
                    m468_zero = best_m468_zero.get(key)
                    for latency_mode in axes["scratch_latency_mode"]:
                        product_cycles_without_commit = product_without_commit[
                            latency_mode
                        ]
                        product_cycles = product_cycles_without_commit + commit_cycles
                        point = {
                            "row_tile": int(row_tile),
                            "resident_block_banks": int(banks),
                            "bandwidth_bytes_per_cycle": bandwidth,
                            "cam_compare_lanes": int(cam_lanes),
                            "effective_cam_compare_lanes": effective_cam,
                            "scratch_latency_mode": latency_mode,
                            "task_count_with_passes": task_count,
                            "nonempty_tasks_with_passes": nonempty_tasks,
                            "bit_cycles": int(bit_cycles),
                            "product_cycles": int(product_cycles),
                            "bit_cycles_without_commit": int(
                                bit_total_without_commit
                            ),
                            "product_cycles_without_commit": int(
                                product_cycles_without_commit
                            ),
                            "commit_cycles": commit_cycles,
                            "same_coordinate_product_vs_bit_speedup": (
                                bit_cycles / max(1, product_cycles)
                            ),
                            "best_same_budget_m468_zero_row_tile": (
                                m468_zero["row_tile"] if m468_zero else None
                            ),
                            "best_same_budget_m468_zero_cycles": (
                                m468_zero["cycles"] if m468_zero else None
                            ),
                            "speedup_vs_best_same_budget_m468_zero": (
                                m468_zero["cycles"] / max(1, product_cycles)
                                if m468_zero
                                else None
                            ),
                            "speedup_vs_m430_517041352_diagnostic": (
                                517041352 / max(1, product_cycles)
                            ),
                            "input_nnz": aggregate["input_nnz"],
                            "residual_nnz": aggregate["residual_nnz"],
                            "product_issue_per_block": aggregate[
                                "product_issue_per_block"
                            ],
                            "parent_edges": aggregate["parent_rows"],
                            "partial_parent_edges": aggregate[
                                "partial_parent_rows"
                            ],
                            "exact_parent_edges": aggregate["exact_parent_rows"],
                            "active_rows": aggregate["active_rows"],
                            "search_rows_with_passes": (
                                aggregate["search_rows"] * passes
                            ),
                            "matcher_mask_comparisons": (
                                int(
                                    (
                                        arrays["search_rows"].astype(np.int64)
                                        * arrays["row_count"].astype(np.int64)
                                    ).sum()
                                )
                                * passes
                            ),
                            "maximum_peak_live_rows_diagnostic": peak_live,
                            "maximum_refcount_diagnostic": int(
                                arrays["max_refcount"].max(initial=0)
                            ),
                            "future_parent_edges": aggregate[
                                "future_parent_rows"
                            ],
                            "maximum_parent_span": int(
                                arrays["maximum_parent_span"].max(initial=0)
                            ),
                            "weight_dram_bytes": int(weight_bytes),
                            "weight_dma_commands": int(weight_dma_commands),
                            "source_sram_bytes": int(source_sram_bytes),
                            "descriptor_write_bytes": int(
                                descriptor_write_bytes
                            ),
                            "candidate_store_search_read_bytes": int(
                                candidate_store_search_read_bytes
                            ),
                            "descriptor_order_scan_read_bytes": int(
                                descriptor_order_scan_read_bytes
                            ),
                            "traffic_scope": (
                                "Logical on-chip access bytes plus off-chip weight "
                                "payload; not physical SRAM energy or DRAM system energy."
                            ),
                            "parent_scratch_read_bytes": int(
                                scratch_read_bytes
                            ),
                            "parent_scratch_write_bytes": int(
                                scratch_write_bytes
                            ),
                            "capacity": capacity,
                            "fits_240k_logical": bool(fits_logical),
                            "fits_240k_macro_rounded": bool(fits_macro),
                            "fits_both_240k_gates": bool(fits_both),
                            "cam_area_unclosed": True,
                            "cam_and_scheduler_ppa_admitted": False,
                            "physical_scratch_1r1w_admitted": False,
                            "cpu_dse_nominated": False,
                            "performance_admitted": False,
                            "system_speedup": False,
                        }
                        points.append(point)

    by_coordinate: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for point in points:
        coordinate = (
            point["row_tile"],
            point["resident_block_banks"],
            str(point["bandwidth_bytes_per_cycle"]),
            point["cam_compare_lanes"],
        )
        by_coordinate.setdefault(coordinate, {})[
            point["scratch_latency_mode"]
        ] = point

    comparisons: list[dict[str, Any]] = []
    for coordinate, modes in sorted(by_coordinate.items()):
        fused = modes["fused_forwarded_1r1w"]
        upper = modes["unfused_sync_upper"]
        m468_fused = fused["speedup_vs_best_same_budget_m468_zero"]
        m468_upper = upper["speedup_vs_best_same_budget_m468_zero"]
        nominated = (
            fused["bandwidth_bytes_per_cycle"] == 128
            and fused["fits_both_240k_gates"]
            and fused["same_coordinate_product_vs_bit_speedup"] >= 1.75
            and m468_fused is not None
            and m468_fused >= 1.50
            and upper["same_coordinate_product_vs_bit_speedup"] >= 1.25
            and m468_upper is not None
            and m468_upper >= 1.10
        )
        fused["cpu_dse_nominated"] = bool(nominated)
        upper["supports_fused_nomination"] = bool(nominated)
        comparisons.append(
            {
                "row_tile": fused["row_tile"],
                "resident_block_banks": fused["resident_block_banks"],
                "bandwidth_bytes_per_cycle": fused[
                    "bandwidth_bytes_per_cycle"
                ],
                "cam_compare_lanes": fused["cam_compare_lanes"],
                "bit_cycles": fused["bit_cycles"],
                "fused_product_cycles": fused["product_cycles"],
                "unfused_sync_upper_cycles": upper["product_cycles"],
                "fused_same_coordinate_speedup": fused[
                    "same_coordinate_product_vs_bit_speedup"
                ],
                "unfused_same_coordinate_speedup": upper[
                    "same_coordinate_product_vs_bit_speedup"
                ],
                "best_same_budget_m468_zero_cycles": fused[
                    "best_same_budget_m468_zero_cycles"
                ],
                "fused_speedup_vs_m468_zero": m468_fused,
                "unfused_speedup_vs_m468_zero": m468_upper,
                "fits_both_240k_gates": fused["fits_both_240k_gates"],
                "cam_area_unclosed": True,
                "cpu_dse_nominated": bool(nominated),
                "performance_admitted": False,
            }
        )
    return points, comparisons


def operator_sample_summary(
    task_arrays: dict[int, dict[str, np.ndarray]]
) -> list[dict[str, Any]]:
    rows = []
    for tile, arrays in task_arrays.items():
        for sample in range(SAMPLES):
            for operator in range(OPERATORS):
                selector = (sample, operator)
                original = int(arrays["input_nnz"][selector].sum())
                product = int(
                    arrays["product_issue_per_block"][selector].sum()
                )
                rows.append(
                    {
                        "row_tile": tile,
                        "sample": sample,
                        "operator": operator,
                        "input_nnz": original,
                        "product_issue_per_block": product,
                        "issue_reduction": 1.0 - product / max(1, original),
                        "parent_rows": int(
                            arrays["parent_rows"][selector].sum()
                        ),
                        "maximum_peak_live_rows": int(
                            arrays["peak_live_rows"][selector].max(initial=0)
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def self_test() -> None:
    cases = [
        np.array([3, 3], dtype=np.uint16),
        np.array([1, 3, 7], dtype=np.uint16),
        np.array([3, 1], dtype=np.uint16),
        np.array([0, 5, 1, 4, 5], dtype=np.uint16),
        np.array([0, 3], dtype=np.uint16),
    ]
    for masks in cases:
        residual, parent = cleanroom_subset(masks)
        official_residual, official_parent = official_find_product_sparsity(masks)
        require(
            np.array_equal(residual, official_residual),
            "self-test official residual parity",
        )
        require(
            np.array_equal(parent, official_parent),
            "self-test official parent parity",
        )
        metrics = task_metrics(masks)
        require(metrics["reconstruction_mismatches"] == 0, "self-test reconstruct")
        require(metrics["topological_mismatches"] == 0, "self-test topology")
        for row in range(masks.size):
            reconstructed = int(residual[row])
            if parent[row] >= 0:
                reconstructed ^= int(masks[parent[row]])
            require(reconstructed == int(masks[row]), "self-test exactness")
    residual, parent = cleanroom_subset(np.array([3, 1], dtype=np.uint16))
    require(parent.tolist() == [1, -1], "future strict-subset parent self-test")
    require(residual.tolist() == [2, 1], "future residual self-test")
    residual, parent = cleanroom_subset(np.array([0, 3], dtype=np.uint16))
    require(parent.tolist() == [-1, -1], "zero-mask parent exclusion self-test")
    require(residual.tolist() == [0, 3], "zero-mask residual self-test")
    print("M473 synthetic self-test PASS cases=5 official_parity=5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-contract", type=Path, default=DEFAULT_EXECUTION_CONTRACT
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--limit-phases", type=int, default=None)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    execution_path = args.execution_contract.resolve()
    execution = strict_json(execution_path)
    require(
        execution["schema"]
        == "m473_h67_online_subset_live_pwp_execution_contract_v1",
        "execution schema mismatch",
    )
    preflight_path = ROOT / execution["preflight"]["path"]
    require(
        sha256_file(preflight_path) == execution["preflight"]["sha256"],
        "preflight SHA mismatch",
    )
    contract = strict_json(preflight_path)
    require(
        sha256_file(Path(__file__).resolve()) == execution["analyzer"]["sha256"],
        "analyzer SHA mismatch",
    )
    checked: dict[str, Any] = {
        "execution_contract": {
            "path": str(execution_path.relative_to(ROOT)),
            "sha256": sha256_file(execution_path),
        },
        "preflight": execution["preflight"],
        "analyzer": execution["analyzer"],
    }
    for key, item in contract["frozen_inputs"].items():
        if key == "prosperity_repo":
            repo = ROOT / item["path"]
            commit = git_stdout(repo, "rev-parse", "HEAD")
            dirty = git_stdout(repo, "status", "--porcelain")
            require(commit == item["commit"], "Prosperity commit mismatch")
            require(not item["must_be_clean"] or not dirty, "Prosperity dirty")
            checked[key] = {
                "path": item["path"],
                "commit": commit,
                "clean": not bool(dirty),
            }
            continue
        path = ROOT / item["path"]
        actual = sha256_file(path)
        require(actual == item["sha256"], f"{key} SHA mismatch")
        checked[key] = {"path": item["path"], "sha256": actual}

    rows_path = ROOT / contract["frozen_inputs"]["m410r2_rows"]["path"]
    require(
        rows_path.stat().st_size == PHASES * ROWS_PER_PHASE * BYTES_PER_ROW,
        "M410 fixed-width row bytes mismatch",
    )
    m468 = strict_json(
        ROOT / contract["frozen_inputs"]["m468r3_m469_result"]["path"]
    )
    m468_hammer = strict_json(
        ROOT / contract["frozen_inputs"]["m468r6_independent_hammer"]["path"]
    )
    m41 = strict_json(
        ROOT
        / contract["frozen_inputs"]["m41_accumulator_independent_audit"]["path"]
    )
    frozen_semantic_validation = validate_frozen_semantics(
        contract, m468, m468_hammer, m41
    )
    tiles = tuple(int(item) for item in contract["dse_axes"]["row_tile"])
    total_phases = PHASES if args.limit_phases is None else min(PHASES, args.limit_phases)
    task_arrays: dict[int, dict[str, np.ndarray]] = {}
    for tile in tiles:
        chunks = int(math.ceil(ROWS_PER_PHASE / tile))
        task_arrays[tile] = {
            field: np.zeros(
                (SAMPLES, OPERATORS, chunks, PARTITIONS), dtype=np.int32
            )
            for field in TASK_FIELDS
        }

    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(rows_path), tiles),
    ) as executor:
        for completed, (phase_index, payload) in enumerate(
            executor.map(worker_phase, range(total_phases), chunksize=args.chunksize),
            start=1,
        ):
            sample_operator, partition = divmod(phase_index, PARTITIONS)
            sample, operator = divmod(sample_operator, OPERATORS)
            for tile in tiles:
                for field in TASK_FIELDS:
                    task_arrays[tile][field][sample, operator, :, partition] = (
                        payload[tile][field]
                    )
            if completed % 1000 == 0 or completed == total_phases:
                print(f"M473 phases {completed}/{total_phases}", flush=True)

    full_run = total_phases == PHASES
    for tile, arrays in task_arrays.items():
        require(
            int(arrays["reconstruction_mismatches"].sum()) == 0,
            f"tile {tile} reconstruction mismatch",
        )
        require(
            int(arrays["topological_mismatches"].sum()) == 0,
            f"tile {tile} topology mismatch",
        )
        if full_run:
            expected_rows = PHASES * ROWS_PER_PHASE
            require(
                int(arrays["row_count"].sum()) == expected_rows,
                f"tile {tile} row population mismatch",
            )

    required_checks = int(
        contract["exact_subset_semantics"]["required_official_mapping_checks"]
    )
    mapping_checks = deterministic_mapping_checks(
        rows_path, tiles, required_checks if full_run else min(8, required_checks)
    )
    mapping_mismatches = int(
        sum(
            check["residual_mismatches"] + check["parent_mismatches"]
            for check in mapping_checks
        )
    )
    require(
        mapping_mismatches
        == contract["exact_subset_semantics"]["required_mapping_mismatches"],
        "official mapping validation failed",
    )

    if not full_run:
        print("M473 development subset PASS; no sealed result written")
        return 0

    points, comparisons = build_points(task_arrays, contract, m468)
    summaries = operator_sample_summary(task_arrays)
    nominations = [item for item in comparisons if item["cpu_dse_nominated"]]
    best_128 = min(
        (
            point
            for point in points
            if point["bandwidth_bytes_per_cycle"] == 128
            and point["fits_both_240k_gates"]
            and point["scratch_latency_mode"] == "fused_forwarded_1r1w"
        ),
        key=lambda point: point["product_cycles"],
    )
    best_128_upper = next(
        point
        for point in points
        if point["row_tile"] == best_128["row_tile"]
        and point["resident_block_banks"]
        == best_128["resident_block_banks"]
        and point["bandwidth_bytes_per_cycle"]
        == best_128["bandwidth_bytes_per_cycle"]
        and point["cam_compare_lanes"] == best_128["cam_compare_lanes"]
        and point["scratch_latency_mode"] == "unfused_sync_upper"
    )
    status = "PASS_M473_CPU_DSE_NOMINATED" if nominations else "PASS_M473_CPU_DSE_NO_GO"

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sidecar_payload = {}
    for tile, arrays in task_arrays.items():
        for field, array in arrays.items():
            sidecar_payload[f"tile{tile}_{field}"] = array
    sidecar_path = args.out / "m473_exact_task_sidecar.npz"
    np.savez_compressed(sidecar_path, **sidecar_payload)
    points_csv = args.out / "m473_cycle_traffic_capacity_points.csv"
    point_fields = [
        "row_tile",
        "resident_block_banks",
        "bandwidth_bytes_per_cycle",
        "cam_compare_lanes",
        "effective_cam_compare_lanes",
        "scratch_latency_mode",
        "bit_cycles",
        "product_cycles",
        "same_coordinate_product_vs_bit_speedup",
        "best_same_budget_m468_zero_cycles",
        "speedup_vs_best_same_budget_m468_zero",
        "speedup_vs_m430_517041352_diagnostic",
        "maximum_peak_live_rows_diagnostic",
        "parent_edges",
        "active_rows",
        "weight_dram_bytes",
        "weight_dma_commands",
        "parent_scratch_read_bytes",
        "parent_scratch_write_bytes",
        "candidate_store_search_read_bytes",
        "descriptor_order_scan_read_bytes",
        "traffic_scope",
        "fits_240k_logical",
        "fits_240k_macro_rounded",
        "fits_both_240k_gates",
        "cpu_dse_nominated",
        "performance_admitted",
    ]
    write_csv(points_csv, points, point_fields)
    comparisons_csv = args.out / "m473_materiality_comparisons.csv"
    write_csv(comparisons_csv, comparisons, list(comparisons[0].keys()))
    summary_csv = args.out / "m473_operator_sample_summary.csv"
    write_csv(summary_csv, summaries, list(summaries[0].keys()))

    report = {
        "schema": "m473_h67_online_subset_live_pwp_v1",
        "date": "2026-08-26",
        "status": status,
        "identity": checked,
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "population": {
            "samples": SAMPLES,
            "operators": OPERATORS,
            "partitions": PARTITIONS,
            "rows_per_phase": ROWS_PER_PHASE,
            "phases": PHASES,
            "source_rows_per_tile_axis": PHASES * ROWS_PER_PHASE,
            "row_tile_axes": list(tiles),
        },
        "official_mapping_validation": {
            "checks": mapping_checks,
            "checks_executed": len(mapping_checks),
            "mismatches": mapping_mismatches,
            "status": "PASS",
        },
        "frozen_semantic_validation": frozen_semantic_validation,
        "exactness": {
            "reconstruction_mismatches": 0,
            "topological_mismatches": 0,
            "accuracy_change": False,
            "checkpoint_change": False,
        },
        "best_128Bps_feasible_point": best_128,
        "matching_unfused_sync_upper_point": best_128_upper,
        "nominations": nominations,
        "nomination_count": len(nominations),
        "points": points,
        "operator_sample_summary": summaries,
        "output_files": {
            "task_sidecar": {
                "path": sidecar_path.name,
                "sha256": sha256_file(sidecar_path),
            },
            "points_csv": {
                "path": points_csv.name,
                "sha256": sha256_file(points_csv),
            },
            "comparisons_csv": {
                "path": comparisons_csv.name,
                "sha256": sha256_file(comparisons_csv),
            },
            "operator_sample_summary_csv": {
                "path": summary_csv.name,
                "sha256": sha256_file(summary_csv),
            },
        },
        "admission": {
            "cpu_dse_nominated": bool(nominations),
            "performance_admitted": False,
            "rtl_nominated": False,
            "synopsys": False,
            "energy": False,
            "ppa": False,
            "full_network": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "required_next_gate": (
            "Independent hammer, then M473 matcher/bucket/refcount/1R1W scratch RTL with Synopsys VCS; only a surviving point may enter DC/STA/power."
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    result_path = args.out / "m473_h67_online_subset_live_pwp_result_r1.json"
    result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    receipt = {
        "schema": "m473_h67_online_subset_live_pwp_receipt_v1",
        "status": status,
        "result": {
            "path": result_path.name,
            "sha256": sha256_file(result_path),
        },
        "best_128Bps_feasible_point": {
            key: best_128[key]
            for key in (
                "row_tile",
                "resident_block_banks",
                "cam_compare_lanes",
                "product_cycles",
                "bit_cycles",
                "same_coordinate_product_vs_bit_speedup",
                "best_same_budget_m468_zero_cycles",
                "speedup_vs_best_same_budget_m468_zero",
                "speedup_vs_m430_517041352_diagnostic",
                "maximum_peak_live_rows_diagnostic",
                "fits_both_240k_gates",
                "cpu_dse_nominated",
            )
        },
        "matching_unfused_sync_upper": {
            "product_cycles": best_128_upper["product_cycles"],
            "same_coordinate_product_vs_bit_speedup": best_128_upper[
                "same_coordinate_product_vs_bit_speedup"
            ],
            "speedup_vs_best_same_budget_m468_zero": best_128_upper[
                "speedup_vs_best_same_budget_m468_zero"
            ],
        },
        "official_mapping_mismatches": mapping_mismatches,
        "nomination_count": len(nominations),
        "admission": report["admission"],
    }
    receipt_path = args.out / "m473_h67_online_subset_live_pwp_receipt_r1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    readme = args.out / "README.md"
    readme.write_text(
        "# M473 online subset + live parent scratch\n\n"
        f"Status: `{status}`. Best feasible 128 B/cycle CPU point: "
        f"row_tile={best_128['row_tile']}, banks={best_128['resident_block_banks']}, "
        f"CAM={best_128['cam_compare_lanes']}, cycles={best_128['product_cycles']:,}, "
        f"same-coordinate product/bit={best_128['same_coordinate_product_vs_bit_speedup']:.6f}x, "
        f"vs best same-budget M468 zero={best_128['speedup_vs_best_same_budget_m468_zero']:.6f}x. "
        f"Matching unfused-sync upper={best_128_upper['product_cycles']:,} cycles.\n\n"
        "CPU DSE only. CAM/scheduler/1R1W scratch are not physicalized; performance, RTL, PPA, energy, system and headline remain false.\n"
    )
    sealed = [
        readme,
        result_path,
        receipt_path,
        sidecar_path,
        points_csv,
        comparisons_csv,
        summary_csv,
    ]
    sums_path = args.out / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in sealed)
    )
    (args.out / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256_file(sums_path)}  SHA256SUMS\n"
    )
    print(result_path)
    print(
        "M473 best",
        best_128["product_cycles"],
        f"same_bit={best_128['same_coordinate_product_vs_bit_speedup']:.6f}x",
        f"m468zero={best_128['speedup_vs_best_same_budget_m468_zero']:.6f}x",
        f"nominations={len(nominations)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
