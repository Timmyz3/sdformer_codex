#!/usr/bin/env python3
"""Fair exact row-tile and lazy-PWP DSE for four frozen H67 Conv3x3.

Only the sealed M410R2 original16 derivative is traversed.  The M40 payload is
neither an input nor opened.  Scheduling is sample -> operator -> output-row
tile -> 432 partitions, with the psum tile resident until its exact commit.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def popcount_u32(values, byte_popcount):
    values = np.asarray(values, dtype=np.uint32)
    return (byte_popcount[values & 255] +
            byte_popcount[(values >> 8) & 255] +
            byte_popcount[(values >> 16) & 255] +
            byte_popcount[(values >> 24) & 255]).astype(np.uint8)


def count_runs_u32(values, byte_popcount):
    values = np.asarray(values, dtype=np.uint32)
    starts = np.bitwise_and(values, np.bitwise_not(values << np.uint32(1)))
    return popcount_u32(starts, byte_popcount)


def bank_max_mod8_u16(values):
    values = np.asarray(values, dtype=np.uint16)
    both = np.bitwise_and(values & np.uint16(255), values >> np.uint16(8))
    return np.where(both != 0, 2, np.where(values != 0, 1, 0)).astype(np.uint8)


def ceil_div_array(values, divisor):
    return (values.astype(np.int64) + divisor - 1) // divisor


def data_cycles_array(byte_count, bandwidth):
    byte_count = np.asarray(byte_count, dtype=np.int64)
    if bandwidth == "infinite":
        return np.zeros(byte_count.shape, dtype=np.int64)
    return (byte_count + int(bandwidth) - 1) // int(bandwidth)


def data_cycles_scalar(byte_count, bandwidth):
    if bandwidth == "infinite":
        return 0
    return int(math.ceil(byte_count / float(bandwidth)))


def depth64(rows):
    return int(math.ceil(rows / 64.0) * 64)


def bit_bytes(bits):
    return int((int(bits) + 7) // 8)


def capacity_breakdown(row_tile, block_banks, mode, max_patterns,
                       fixed_reserve):
    depth = depth64(row_tile)
    half_slots = 2 if block_banks == 8 else 1
    logical = {
        "psum": bit_bytes(row_tile * block_banks * 96 * 19),
        "active_bitmap": bit_bytes(row_tile),
        "psum_valid_bitmap": bit_bytes(row_tile * block_banks),
        "source_pingpong": row_tile * 2 * 2,
        "descriptor_pingpong_including_row_tags": row_tile * 6 * 2,
        "config_pingpong": 192 if mode != "strong_zero" else 0,
        "weight_payload": half_slots * 6144,
        "stored_pwp_payload": 0,
        "lazy_center_buffer_serial": 0,
        "fifo_control_reserve": fixed_reserve,
    }
    if mode == "stored_pwp":
        logical["stored_pwp_payload"] = half_slots * max_patterns * 640
    elif mode == "lazy_pwp":
        logical["lazy_center_buffer_serial"] = max_patterns * 576

    # Conservative macro proxy: every memory width is sliced into 144-bit
    # banks and every depth is rounded upward to a 64-entry quantum.
    rounded = {
        "psum": block_banks * 13 * 18 * depth,
        "active_bitmap": 18 * depth,
        "psum_valid_bitmap": 18 * depth,  # 4/8 bits fit one 144b slice.
        "source_pingpong": 2 * 18 * depth,
        "descriptor_pingpong_including_row_tags": 2 * 18 * depth,
        # Each 96-byte config is one 768-bit record: six 144-bit slices.
        "config_pingpong": (2 * 6 * 18 * 64
                            if mode != "strong_zero" else 0),
        # One source row for four output blocks is 384 B = 3072 bits:
        # twenty-two 144-bit slices, with sixteen rows rounded to depth 64.
        "weight_payload": half_slots * 22 * 18 * 64,
        "stored_pwp_payload": 0,
        "lazy_center_buffer_serial": 0,
        "fifo_control_reserve": fixed_reserve,
    }
    if mode == "stored_pwp":
        # Stored record is the frozen padded 640 B = 5120 bits.
        rounded["stored_pwp_payload"] = half_slots * 36 * 18 * 64
    elif mode == "lazy_pwp":
        # Generated record is 4*96*signed12 = 576 B = 4608 bits.
        rounded["lazy_center_buffer_serial"] = 32 * 18 * 64

    # The overlap diagnostic needs two generated-center buffers for 8 banks.
    overlap_extra_logical = 0
    overlap_extra_rounded = 0
    if mode == "lazy_pwp" and block_banks == 8:
        overlap_extra_logical = max_patterns * 576
        overlap_extra_rounded = 32 * 18 * 64
    for name, logical_bytes in logical.items():
        require(rounded[name] >= logical_bytes,
                "macro-rounded capacity undercounts logical item: " + name)
    return {
        "logical_items": logical,
        "macro_rounded_items": rounded,
        "logical_total_bytes": int(sum(logical.values())),
        "macro_rounded_total_bytes": int(sum(rounded.values())),
        "overlap_diagnostic_extra_logical_bytes": overlap_extra_logical,
        "overlap_diagnostic_extra_macro_rounded_bytes": overlap_extra_rounded,
        "overlap_diagnostic_logical_total_bytes": int(
            sum(logical.values()) + overlap_extra_logical),
        "overlap_diagnostic_macro_rounded_total_bytes": int(
            sum(rounded.values()) + overlap_extra_rounded),
        "macro_depth_rows": depth,
        "macro_width_slice_bits": 144,
    }


def empty_point_accumulator():
    return {
        "task_count": 0,
        "task_passes": 0,
        "empty_task_passes": 0,
        "nonempty_task_passes": 0,
        "source_sram_bytes": 0,
        "source_dram_bytes": 0,
        "weight_dram_bytes": 0,
        "pwp_dram_bytes": 0,
        "config_dram_bytes": 0,
        "psum_sram_reads": 0,
        "psum_sram_writes": 0,
        "psum_sram_read_bytes": 0,
        "psum_sram_write_bytes": 0,
        "psum_dram_spill_bytes": 0,
        "pwp_sram_read_bytes": 0,
        "dma_commands": 0,
        "generator_weight_read_bytes": 0,
        "generator_signed_adds": 0,
        "generator_cache_write_bytes": 0,
        "generator_commands": 0,
        "generator_cycles": 0,
        "empty_gate_cycle_savings_vs_forced_payload": 0,
        "empty_gate_skipped_config_bytes": 0,
        "empty_gate_skipped_weight_bytes": 0,
        "empty_gate_skipped_dma_commands": 0,
        "cycles_without_commit": 0,
        "overlap_upper_bound_cycles_without_commit": 0,
    }


def add_scalar(point, key, value):
    point[key] += int(value)


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("\n".join(
        "{}  {}".format(sha256(output_dir / name), name)
        for name in sorted(names)) + "\n", encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--anchor-only", action="store_true",
                        help="Exact-SHA diagnostic: sweep only row3000, check "
                             "both frozen anchors, and write no result")
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M468R3/M469 overwrite")

    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m468r3_m469_h67_lazy_pwp_execution_contract_v1" and
            contract.get("status") == "FROZEN_EXACT_SHA_BEFORE_EXECUTION",
            "M468R3/M469 execution contract drift")
    root = args.contract.resolve().parents[1]
    inputs = {}
    identity = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M468R3/M469 input SHA drift: " + name)
        inputs[name] = path
        identity[name] = dict(spec)
    require(inputs["analyzer"].resolve() == source_path and
            identity["analyzer"]["sha256"] == source_start,
            "M468R3/M469 analyzer self-SHA drift")

    preflight = strict_json(inputs["preflight"])
    require(preflight["status"] == "FROZEN_PREFLIGHT_BEFORE_ANALYZER_EXECUTION",
            "preflight status drift")
    for name, spec in preflight["frozen_inputs"].items():
        require(name in inputs and inputs[name] == root / spec["path"] and
                identity[name]["sha256"] == spec["sha256"],
                "execution/preflight input mismatch: " + name)

    m430 = strict_json(inputs["m430_result"])
    catalog = strict_json(inputs["m430_catalog"])
    derivative = strict_json(inputs["m410r2_manifest"])
    require(m430["status"] ==
            "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY" and
            derivative["status"] ==
            "PASS_M410R2_CONTRACT_VISIBLE_FULL_RUNTIME_STIMULUS_EXPORT",
            "upstream status drift")
    require(derivative["output"]["rows"]["sha256"] ==
            identity["m410r2_rows"]["sha256"],
            "derivative row identity mismatch")

    phase_reference = []
    with inputs["m430_phase_csv"].open(
            "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase_reference.append({key: int(value)
                                    for key, value in row.items()})
    sched = preflight["schedule_contract"]
    cycle_model = preflight["cycle_model"]
    samples = sched["samples"]
    operators = sched["operators"]
    partitions = sched["partitions_per_operator"]
    rows_per_phase = sched["rows_per_phase"]
    phases = samples * operators * partitions
    require(len(phase_reference) == phases, "phase CSV extent drift")

    tile_sizes = tuple(preflight["dse"]["row_tile_sizes"])
    if args.anchor_only:
        tile_sizes = (rows_per_phase,)
    block_axes = tuple(preflight["dse"]["resident_block_banks"])
    bandwidths = tuple(preflight["dse"]["dram_bytes_per_cycle"])
    generator_ks = tuple(preflight["lazy_generator_contract"][
        "source_lanes_k"])
    budget = preflight["dse"]["sram_budget_bytes"]
    fixed_reserve = preflight["capacity_contract"]["fixed_reserve_bytes"]

    pop16 = np.fromiter((bin(value).count("1")
                         for value in range(1 << 16)),
                        dtype=np.uint8, count=1 << 16)
    pop8 = pop16[:256]

    # Prove that actual frozen INT8 weights generate exact signed PWP values.
    pwp_value_hash = hashlib.sha256()
    pwp_value_min = 32767
    pwp_value_max = -32768
    pwp_values_checked = 0
    center_population = np.zeros((operators, partitions, 32), dtype=np.uint8)
    for operator in range(operators):
        weights = np.fromfile(str(inputs["weight_o{}".format(operator)]),
                              dtype=np.int8)
        require(weights.size == partitions * 16 * 768,
                "INT8 weight geometry drift for operator {}".format(operator))
        weights = weights.reshape(partitions, 16, 768)
        for partition in range(partitions):
            centers = np.asarray([
                int(value, 16) for value in
                catalog["operators"][operator]["partitions"][partition]
                ["nested_patterns"]], dtype=np.uint16)
            require(centers.size == 32, "q32 center extent drift")
            masks = (((centers[:, None].astype(np.uint32) >>
                       np.arange(16, dtype=np.uint32)[None, :]) & 1)
                     .astype(np.int16))
            generated = np.matmul(
                masks, weights[partition].astype(np.int16)).astype(np.int16)
            pwp_value_min = min(pwp_value_min, int(generated.min()))
            pwp_value_max = max(pwp_value_max, int(generated.max()))
            require(int(generated.min()) >= -2048 and
                    int(generated.max()) <= 2032,
                    "generated PWP exceeds signed12 realizable bound")
            pwp_value_hash.update(generated.astype("<i2", copy=False).tobytes())
            pwp_values_checked += generated.size
            center_population[operator, partition] = pop16[centers]

    aggregate = {
        "source_rows": 0, "active_rows": 0, "pwp_rows": 0,
        "fallback_rows": 0, "correction_ops_per_block": 0,
        "bit_sparse_vector_ops_per_block": 0,
    }
    phase_mismatches = 0
    phase_sidecar = []
    g15_tile = {(tile, k): {
        "rows": 0, "empty_rows": 0, "direct_rows": 0,
        "parent_rows": 0, "direct_cycle_sum": 0,
        "parent_cycle_sum": 0, "selected_cycle_sum": 0,
    } for tile in tile_sizes for k in generator_ks}

    # One typed compact row per scheduled task.  This is intentionally built
    # during the only derivative traversal so M470 can model partition windows
    # and psum spills without reopening the 445 MiB original16 file.
    tasks_per_sample_operator = partitions * sum(
        int(math.ceil(rows_per_phase / float(tile))) for tile in tile_sizes)
    total_compact_tasks = samples * operators * tasks_per_sample_operator
    task_sidecar = {
        "sample": np.empty(total_compact_tasks, dtype=np.uint8),
        "operator": np.empty(total_compact_tasks, dtype=np.uint8),
        "partition": np.empty(total_compact_tasks, dtype=np.uint16),
        "row_tile_size": np.empty(total_compact_tasks, dtype=np.uint16),
        "row_start": np.empty(total_compact_tasks, dtype=np.uint16),
        "row_stop": np.empty(total_compact_tasks, dtype=np.uint16),
        "active_rows": np.empty(total_compact_tasks, dtype=np.uint16),
        "pwp_rows": np.empty(total_compact_tasks, dtype=np.uint16),
        "correction_ops_per_block": np.empty(total_compact_tasks,
                                              dtype=np.uint32),
        "strong_zero_pop_ops_per_block": np.empty(total_compact_tasks,
                                                   dtype=np.uint32),
        "early_matcher_cycles": np.empty(total_compact_tasks,
                                          dtype=np.uint16),
        "used_center_mask_u32": np.empty(total_compact_tasks,
                                         dtype=np.uint32),
        "used_center_runs": np.empty(total_compact_tasks, dtype=np.uint8),
        "used_center_population_sum": np.empty(total_compact_tasks,
                                                dtype=np.uint16),
    }
    for k in generator_ks:
        task_sidecar["generator_center_ceil_sum_k{}".format(k)] = np.empty(
            total_compact_tasks, dtype=np.uint16)
    task_cursor = 0

    raw = {}
    strong_stream_state = {}
    max_patterns_by_tile = {tile: 0 for tile in tile_sizes}
    for tile in tile_sizes:
        for banks in block_axes:
            for bandwidth in bandwidths:
                raw[(tile, banks, str(bandwidth), "strong_zero", 0)] = \
                    empty_point_accumulator()
                strong_stream_state[(tile, banks, str(bandwidth))] = {
                    "last_work": None,
                    "forced_last_work": None,
                }
                raw[(tile, banks, str(bandwidth), "stored_pwp", 0)] = \
                    empty_point_accumulator()
                for k in generator_ks:
                    raw[(tile, banks, str(bandwidth), "lazy_pwp", k)] = \
                        empty_point_accumulator()

    row_path = inputs["m410r2_rows"]
    with row_path.open("r", encoding="ascii") as rows_handle:
        for sample in range(samples):
            require(all(state["last_work"] is None and
                        state["forced_last_work"] is None
                        for state in strong_stream_state.values()),
                    "strong-zero stream crossed sample/commit boundary")
            for operator in range(operators):
                original_all = np.empty((partitions, rows_per_phase),
                                        dtype=np.uint16)
                active_all = np.empty_like(original_all, dtype=np.uint8)
                pwp_all = np.empty_like(original_all, dtype=np.uint8)
                correction_all = np.empty_like(original_all, dtype=np.uint8)
                population_all = np.empty_like(original_all, dtype=np.uint8)
                early_all = np.empty_like(original_all, dtype=np.uint8)
                best_all = np.empty_like(original_all, dtype=np.uint8)
                residual_population_all = np.empty_like(original_all,
                                                        dtype=np.uint8)
                original_bank_all = np.empty_like(original_all, dtype=np.uint8)
                residual_bank_all = np.empty_like(original_all, dtype=np.uint8)

                # One derivative read.  All M468R3/M469 and G15 statistics are
                # derived before this operator tensor is released.
                for partition in range(partitions):
                    originals = np.fromiter(
                        (int(rows_handle.readline(), 16) & 0xffff
                         for _ in range(rows_per_phase)),
                        dtype=np.uint16, count=rows_per_phase)
                    require(originals.size == rows_per_phase,
                            "premature derivative EOF")
                    centers = np.asarray([
                        int(value, 16) for value in
                        catalog["operators"][operator]["partitions"][partition]
                        ["nested_patterns"]], dtype=np.uint16)
                    unique, inverse = np.unique(originals, return_inverse=True)
                    pop_u = pop16[unique]
                    distances = pop16[np.bitwise_xor(
                        unique[:, None], centers[None, :])]
                    best_u = np.argmin(distances, axis=1).astype(np.uint8)
                    dist_u = distances[np.arange(unique.size), best_u]
                    active_u = unique != 0
                    pwp_u = active_u & ((1 + dist_u) < pop_u)
                    correction_u = np.where(
                        active_u, np.where(pwp_u, dist_u, pop_u), 0)
                    q16_exact_u = np.any(
                        unique[:, None] == centers[None, :16], axis=1)
                    early_u = ((pop_u >= 2) & ~q16_exact_u)

                    active = active_u[inverse]
                    pwp = pwp_u[inverse]
                    best = best_u[inverse]
                    population = pop_u[inverse]
                    correction = correction_u[inverse]
                    early = early_u[inverse]
                    residual = np.bitwise_xor(originals, centers[best])
                    residual_population = pop16[residual]
                    original_bank = bank_max_mod8_u16(originals)
                    residual_bank = bank_max_mod8_u16(residual)

                    original_all[partition] = originals
                    active_all[partition] = active
                    pwp_all[partition] = pwp
                    correction_all[partition] = correction
                    population_all[partition] = population
                    early_all[partition] = early
                    best_all[partition] = best
                    residual_population_all[partition] = residual_population
                    original_bank_all[partition] = original_bank
                    residual_bank_all[partition] = residual_bank

                    used_mask = np.uint32(0)
                    for center_id in np.unique(best[pwp]):
                        used_mask |= np.uint32(1) << np.uint32(center_id)
                    used_count = int(popcount_u32(
                        np.asarray([used_mask]), pop8)[0])
                    used_runs = int(count_runs_u32(
                        np.asarray([used_mask]), pop8)[0])
                    ref_index = ((sample * operators + operator) *
                                 partitions + partition)
                    actual = {
                        "sample": sample,
                        "operator": operator,
                        "partition": partition,
                        "active_rows": int(active.sum()),
                        "pwp_rows": int(pwp.sum()),
                        "fallback_rows": int(active.sum() - pwp.sum()),
                        "correction_ops_per_block": int(correction.sum()),
                        "used_pwp_patterns": used_count,
                        "used_center_runs": used_runs,
                        "early_matcher": int(rows_per_phase + early.sum() + 2),
                    }
                    ref = phase_reference[ref_index]
                    for key in actual:
                        if actual[key] != ref[key]:
                            phase_mismatches += 1

                    aggregate["source_rows"] += rows_per_phase
                    aggregate["active_rows"] += actual["active_rows"]
                    aggregate["pwp_rows"] += actual["pwp_rows"]
                    aggregate["fallback_rows"] += actual["fallback_rows"]
                    aggregate["correction_ops_per_block"] += actual[
                        "correction_ops_per_block"]
                    aggregate["bit_sparse_vector_ops_per_block"] += int(
                        population.sum())

                    side = {
                        "sample": sample, "operator": operator,
                        "partition": partition,
                        "source_rows": rows_per_phase,
                        "active_rows": actual["active_rows"],
                        "m430_use_pwp_rows": int(pwp.sum()),
                        "m430_fallback_rows": actual["fallback_rows"],
                        "m430_correction_ops_per_block": actual[
                            "correction_ops_per_block"],
                        "strong_zero_bit_sparse_ops_per_block": int(
                            population.sum()),
                        "m430_used_pwp_patterns": used_count,
                        "m430_used_center_runs": used_runs,
                        "m430_early_matcher_cycles": actual["early_matcher"],
                        "used_center_population_sum": int(
                            sum(int(center_population[
                                operator, partition, center_id])
                                for center_id in range(32)
                                if (int(used_mask) >> center_id) & 1)),
                    }
                    pop_hist = np.bincount(population, minlength=17)
                    residual_hist = np.bincount(residual_population, minlength=17)
                    center_hist = np.bincount(best[pwp], minlength=32)
                    original_bank_hist = np.bincount(original_bank, minlength=3)
                    residual_bank_hist = np.bincount(residual_bank, minlength=3)
                    for value in range(17):
                        side["original_pop{}".format(value)] = int(pop_hist[value])
                        side["residual_pop{}".format(value)] = int(
                            residual_hist[value])
                    for value in range(3):
                        side["original_maxbank{}".format(value)] = int(
                            original_bank_hist[value])
                        side["residual_maxbank{}".format(value)] = int(
                            residual_bank_hist[value])
                    for value in range(32):
                        side["m430_center{}".format(value)] = int(
                            center_hist[value])
                    for k in generator_ks:
                        direct_cycle = np.maximum(
                            ceil_div_array(population, k), original_bank)
                        parent_cycle = 1 + np.maximum(
                            ceil_div_array(residual_population, k),
                            residual_bank)
                        parent_route = ((population > k) &
                                        (parent_cycle < direct_cycle))
                        side["k{}_direct_rows".format(k)] = int(
                            np.count_nonzero((population != 0) & ~parent_route))
                        side["k{}_parent_rows".format(k)] = int(
                            parent_route.sum())
                        side["k{}_direct_cycle_sum".format(k)] = int(
                            direct_cycle.sum())
                        side["k{}_parent_cycle_sum".format(k)] = int(
                            parent_cycle[parent_route].sum())
                        side["k{}_selected_cycle_sum".format(k)] = int(
                            np.where(parent_route, parent_cycle,
                                     direct_cycle).sum())
                        used_center_ceil = sum(
                            int(math.ceil(int(center_population[
                                operator, partition, center_id]) / float(k)))
                            for center_id in range(32)
                            if (int(used_mask) >> center_id) & 1)
                        side["k{}_m430_generator_cycles_per_half".format(k)] = (
                            used_count * cycle_model[
                                "generator_center_setup_cycles"] +
                            4 * used_center_ceil +
                            (cycle_model["generator_half_flush_cycles"]
                             if used_count else 0))
                    phase_sidecar.append(side)

                # Required schedule: each output-row tile traverses all 432
                # partitions before its psum vectors are committed/reset.
                for tile in tile_sizes:
                    starts = np.arange(0, rows_per_phase, tile, dtype=np.int64)
                    chunks = starts.size
                    row_counts_one = np.minimum(
                        tile, rows_per_phase - starts).astype(np.int64)
                    shape = (partitions, chunks)
                    active_task = np.empty(shape, dtype=np.int64)
                    pwp_task = np.empty(shape, dtype=np.int64)
                    correction_task = np.empty(shape, dtype=np.int64)
                    pop_task = np.empty(shape, dtype=np.int64)
                    early_task = np.empty(shape, dtype=np.int64)
                    used_masks = np.zeros(shape, dtype=np.uint32)
                    gen_pop_sum = np.zeros(shape, dtype=np.int64)
                    gen_add_source_sum = np.zeros(shape, dtype=np.int64)
                    gen_ceil = {k: np.zeros(shape, dtype=np.int64)
                                for k in generator_ks}

                    for partition in range(partitions):
                        active_task[partition] = np.add.reduceat(
                            active_all[partition], starts)
                        pwp_task[partition] = np.add.reduceat(
                            pwp_all[partition], starts)
                        correction_task[partition] = np.add.reduceat(
                            correction_all[partition], starts)
                        pop_task[partition] = np.add.reduceat(
                            population_all[partition], starts)
                        early_task[partition] = np.add.reduceat(
                            early_all[partition], starts)
                        task_index = (np.arange(rows_per_phase,
                                                dtype=np.int64) // tile)
                        selected = pwp_all[partition] != 0
                        if np.any(selected):
                            np.bitwise_or.at(
                                used_masks[partition], task_index[selected],
                                np.left_shift(np.uint32(1),
                                              best_all[partition][selected]
                                              .astype(np.uint32)))
                        for center_id in range(32):
                            present = ((used_masks[partition] >>
                                        np.uint32(center_id)) & 1).astype(np.int64)
                            center_pop = int(center_population[
                                operator, partition, center_id])
                            gen_pop_sum[partition] += present * center_pop
                            gen_add_source_sum[partition] += present * max(
                                center_pop - 1, 0)
                            for k in generator_ks:
                                gen_ceil[k][partition] += present * int(
                                    math.ceil(center_pop / float(k)))

                        # G15 tile aggregate uses the same already resident
                        # row facts, never a second derivative traversal.
                        for k in generator_ks:
                            direct = np.maximum(
                                ceil_div_array(population_all[partition], k),
                                original_bank_all[partition])
                            parent = 1 + np.maximum(
                                ceil_div_array(
                                    residual_population_all[partition], k),
                                residual_bank_all[partition])
                            parent_route = ((population_all[partition] > k) &
                                            (parent < direct))
                            selected_cycle = np.where(parent_route,
                                                      parent, direct)
                            direct_count = ((population_all[partition] != 0) &
                                            ~parent_route).astype(np.int64)
                            for chunk, start in enumerate(starts):
                                stop = min(int(start) + tile, rows_per_phase)
                                g = g15_tile[(tile, k)]
                                g["rows"] += stop - int(start)
                                g["empty_rows"] += int(np.count_nonzero(
                                    population_all[partition, start:stop] == 0))
                                g["direct_rows"] += int(direct_count[
                                    start:stop].sum())
                                g["parent_rows"] += int(parent_route[
                                    start:stop].sum())
                                g["direct_cycle_sum"] += int(direct[
                                    start:stop].sum())
                                g["parent_cycle_sum"] += int(parent[
                                    start:stop][parent_route[start:stop]].sum())
                                g["selected_cycle_sum"] += int(selected_cycle[
                                    start:stop].sum())

                    # Flatten in row-tile-major, then partition order.
                    def ordered(array):
                        return array.T.reshape(-1)

                    active_v = ordered(active_task)
                    pwp_v = ordered(pwp_task)
                    correction_v = ordered(correction_task)
                    pop_v = ordered(pop_task)
                    early_v = ordered(early_task)
                    masks_v = ordered(used_masks)
                    used_v = popcount_u32(masks_v, pop8).astype(np.int64)
                    runs_v = count_runs_u32(masks_v, pop8).astype(np.int64)
                    gen_pop_v = ordered(gen_pop_sum)
                    gen_add_v = ordered(gen_add_source_sum)
                    row_v = np.tile(row_counts_one, partitions).reshape(
                        partitions, chunks).T.reshape(-1)
                    # The construction above is explicit rather than relying
                    # on a repeated scalar: it also covers the short tail tile.
                    row_v = np.repeat(row_counts_one, partitions)
                    nonempty = active_v != 0
                    tasks = active_v.size
                    max_patterns_by_tile[tile] = max(
                        max_patterns_by_tile[tile], int(used_v.max()))
                    matcher_v = row_v + early_v + 2

                    task_stop = task_cursor + tasks
                    task_slice = slice(task_cursor, task_stop)
                    task_sidecar["sample"][task_slice] = sample
                    task_sidecar["operator"][task_slice] = operator
                    task_sidecar["partition"][task_slice] = np.tile(
                        np.arange(partitions, dtype=np.uint16), chunks)
                    task_sidecar["row_tile_size"][task_slice] = tile
                    task_sidecar["row_start"][task_slice] = np.repeat(
                        starts.astype(np.uint16), partitions)
                    task_sidecar["row_stop"][task_slice] = np.repeat(
                        np.minimum(starts + tile, rows_per_phase)
                        .astype(np.uint16), partitions)
                    task_sidecar["active_rows"][task_slice] = active_v
                    task_sidecar["pwp_rows"][task_slice] = pwp_v
                    task_sidecar["correction_ops_per_block"][task_slice] = \
                        correction_v
                    task_sidecar[
                        "strong_zero_pop_ops_per_block"][task_slice] = pop_v
                    task_sidecar["early_matcher_cycles"][task_slice] = matcher_v
                    task_sidecar["used_center_mask_u32"][task_slice] = masks_v
                    task_sidecar["used_center_runs"][task_slice] = runs_v
                    task_sidecar[
                        "used_center_population_sum"][task_slice] = gen_pop_v
                    for k in generator_ks:
                        task_sidecar[
                            "generator_center_ceil_sum_k{}".format(k)
                        ][task_slice] = ordered(gen_ceil[k])
                    task_cursor = task_stop

                    for banks in block_axes:
                        passes = 1 if banks == 8 else 2
                        replay = (pwp_v + correction_v) * 4 + \
                            cycle_model["descriptor_sram_latency_cycles"]
                        matcher = matcher_v
                        scan = row_v + cycle_model[
                            "popcount_filter_pipeline_cycles"]
                        for bandwidth in bandwidths:
                            bw_key = str(bandwidth)
                            weight_half_dma = (
                                data_cycles_scalar(
                                    cycle_model["weight_bytes_per_four_blocks"],
                                    bandwidth) +
                                cycle_model["dma_command_setup_cycles"])
                            weight_full_dma = 2 * weight_half_dma
                            config_dma = (
                                data_cycles_scalar(
                                    cycle_model["elastic_config_bytes"],
                                    bandwidth) +
                                cycle_model["dma_command_setup_cycles"])

                            # Fair strong-zero with the identical exact-empty
                            # gate and naturally distinct four-block payloads.
                            point = raw[(tile, banks, bw_key,
                                         "strong_zero", 0)]
                            strong_weight_dma = (weight_full_dma
                                                 if banks == 8
                                                 else weight_half_dma)
                            preprocess = np.where(
                                nonempty, np.maximum(scan, strong_weight_dma),
                                scan)
                            forced_preprocess = np.maximum(
                                scan, strong_weight_dma)
                            strong_work = pop_v * banks
                            stream = strong_stream_state[(tile, banks, bw_key)]
                            if stream["last_work"] is None:
                                strong_pass = int(preprocess[0])
                                forced_strong_pass = int(forced_preprocess[0])
                            else:
                                strong_pass = max(
                                    int(stream["last_work"]),
                                    int(preprocess[0])) + cycle_model[
                                        "tail_cycles_per_pass"]
                                forced_strong_pass = max(
                                    int(stream["forced_last_work"]),
                                    int(forced_preprocess[0])) + cycle_model[
                                        "tail_cycles_per_pass"]
                            if tasks > 1:
                                strong_pass += int(np.maximum(
                                    strong_work[:-1], preprocess[1:]).sum())
                                forced_strong_pass += int(np.maximum(
                                    strong_work[:-1],
                                    forced_preprocess[1:]).sum())
                                strong_pass += ((tasks - 1) * cycle_model[
                                    "tail_cycles_per_pass"])
                                forced_strong_pass += ((tasks - 1) *
                                    cycle_model["tail_cycles_per_pass"])
                            stream["last_work"] = int(strong_work[-1])
                            stream["forced_last_work"] = int(strong_work[-1])
                            add_scalar(point, "cycles_without_commit",
                                       strong_pass * passes)
                            add_scalar(
                                point,
                                "empty_gate_cycle_savings_vs_forced_payload",
                                (forced_strong_pass - strong_pass) * passes)
                            add_scalar(point, "task_count", tasks)
                            add_scalar(point, "task_passes", tasks * passes)
                            add_scalar(point, "empty_task_passes",
                                       np.count_nonzero(~nonempty) * passes)
                            add_scalar(point, "nonempty_task_passes",
                                       np.count_nonzero(nonempty) * passes)
                            add_scalar(point, "source_sram_bytes",
                                       row_v.sum() * 2 * passes)
                            add_scalar(point, "weight_dram_bytes",
                                       np.count_nonzero(nonempty) * 6144 *
                                       2)
                            add_scalar(point, "dma_commands",
                                       np.count_nonzero(nonempty) * 2)
                            add_scalar(point, "empty_gate_skipped_weight_bytes",
                                       np.count_nonzero(~nonempty) * 6144 * 2)
                            add_scalar(point, "empty_gate_skipped_dma_commands",
                                       np.count_nonzero(~nonempty) * 2)
                            psum_access = int(active_v.sum()) * banks * passes
                            add_scalar(point, "psum_sram_reads", psum_access)
                            add_scalar(point, "psum_sram_writes", psum_access)

                            # Fair stored-PWP M430 mode.
                            point = raw[(tile, banks, bw_key,
                                         "stored_pwp", 0)]
                            payload_bytes = 6144 + used_v * 640
                            payload_dma = (data_cycles_array(
                                payload_bytes, bandwidth) +
                                (1 + runs_v) *
                                cycle_model["dma_command_setup_cycles"])
                            common = config_dma + matcher + 1
                            if banks == 8:
                                stored_nongated = (common + payload_dma +
                                                    np.maximum(replay,
                                                               payload_dma) +
                                                    replay + cycle_model[
                                                        "tail_cycles_per_pass"])
                                stored_cycles = np.where(
                                    nonempty,
                                    stored_nongated,
                                    scan + cycle_model[
                                        "tail_cycles_per_pass"])
                            else:
                                stored_nongated = passes * (
                                    common + payload_dma + replay +
                                    cycle_model["tail_cycles_per_pass"])
                                stored_cycles = np.where(
                                    nonempty,
                                    stored_nongated,
                                    passes * (scan + cycle_model[
                                        "tail_cycles_per_pass"]))
                            add_scalar(point, "cycles_without_commit",
                                       stored_cycles.sum())
                            add_scalar(
                                point,
                                "empty_gate_cycle_savings_vs_forced_payload",
                                (stored_nongated[~nonempty] -
                                 stored_cycles[~nonempty]).sum())
                            add_scalar(point, "task_count", tasks)
                            add_scalar(point, "task_passes", tasks * passes)
                            add_scalar(point, "empty_task_passes",
                                       np.count_nonzero(~nonempty) * passes)
                            add_scalar(point, "nonempty_task_passes",
                                       np.count_nonzero(nonempty) * passes)
                            add_scalar(point, "source_sram_bytes",
                                       row_v.sum() * 2 * passes)
                            add_scalar(point, "config_dram_bytes",
                                       np.count_nonzero(nonempty) * 96 * passes)
                            add_scalar(point, "weight_dram_bytes",
                                       np.count_nonzero(nonempty) * 6144 *
                                       2)
                            add_scalar(point, "pwp_dram_bytes",
                                       used_v[nonempty].sum() * 640 * 2)
                            add_scalar(point, "dma_commands",
                                       np.count_nonzero(nonempty) * passes +
                                       np.sum((1 + runs_v[nonempty]) * 2))
                            add_scalar(point, "empty_gate_skipped_config_bytes",
                                       np.count_nonzero(~nonempty) * 96 * passes)
                            add_scalar(point, "empty_gate_skipped_weight_bytes",
                                       np.count_nonzero(~nonempty) * 6144 * 2)
                            add_scalar(point, "empty_gate_skipped_dma_commands",
                                       np.count_nonzero(~nonempty) *
                                       (passes + 2))
                            add_scalar(point, "pwp_sram_read_bytes",
                                       pwp_v.sum() * 8 * 144)
                            add_scalar(point, "psum_sram_reads", psum_access)
                            add_scalar(point, "psum_sram_writes", psum_access)

                            # Exact lazy PWP; generation is the conservative
                            # serial headline schedule.  The overlap number is
                            # retained only as a non-headline upper bound.
                            for k in generator_ks:
                                point = raw[(tile, banks, bw_key,
                                             "lazy_pwp", k)]
                                gen_half = (used_v * cycle_model[
                                    "generator_center_setup_cycles"] +
                                    4 * ordered(gen_ceil[k]) +
                                    np.where(used_v != 0,
                                             cycle_model[
                                                 "generator_half_flush_cycles"],
                                             0))
                                lazy_common = config_dma + matcher + 1
                                if banks == 8:
                                    lazy_nongated = (
                                        lazy_common + 2 * weight_half_dma +
                                        2 * gen_half + 2 * replay +
                                        cycle_model["tail_cycles_per_pass"])
                                    lazy_cycles = np.where(
                                        nonempty,
                                        lazy_nongated,
                                        scan + cycle_model[
                                            "tail_cycles_per_pass"])
                                    lazy_overlap_nongated = (
                                        lazy_common + 2 * weight_half_dma +
                                        gen_half + np.maximum(replay, gen_half) +
                                        replay + cycle_model[
                                            "tail_cycles_per_pass"])
                                    lazy_overlap = np.where(
                                        nonempty,
                                        lazy_overlap_nongated,
                                        scan + cycle_model[
                                            "tail_cycles_per_pass"])
                                else:
                                    lazy_nongated = passes * (
                                        lazy_common + weight_half_dma + gen_half +
                                        replay + cycle_model[
                                            "tail_cycles_per_pass"])
                                    lazy_cycles = np.where(
                                        nonempty, lazy_nongated,
                                        passes * (scan + cycle_model[
                                            "tail_cycles_per_pass"]))
                                    lazy_overlap = lazy_cycles
                                add_scalar(point, "cycles_without_commit",
                                           lazy_cycles.sum())
                                add_scalar(
                                    point,
                                    "overlap_upper_bound_cycles_without_commit",
                                    lazy_overlap.sum())
                                add_scalar(
                                    point,
                                    "empty_gate_cycle_savings_vs_forced_payload",
                                    (lazy_nongated[~nonempty] -
                                     lazy_cycles[~nonempty]).sum())
                                add_scalar(point, "task_count", tasks)
                                add_scalar(point, "task_passes", tasks * passes)
                                add_scalar(point, "empty_task_passes",
                                           np.count_nonzero(~nonempty) * passes)
                                add_scalar(point, "nonempty_task_passes",
                                           np.count_nonzero(nonempty) * passes)
                                add_scalar(point, "source_sram_bytes",
                                           row_v.sum() * 2 * passes)
                                add_scalar(point, "config_dram_bytes",
                                           np.count_nonzero(nonempty) *
                                           96 * passes)
                                add_scalar(point, "weight_dram_bytes",
                                           np.count_nonzero(nonempty) * 6144 * 2)
                                add_scalar(point, "dma_commands",
                                           np.count_nonzero(nonempty) * passes +
                                           np.count_nonzero(nonempty) * 2)
                                add_scalar(
                                    point, "empty_gate_skipped_config_bytes",
                                    np.count_nonzero(~nonempty) * 96 * passes)
                                add_scalar(
                                    point, "empty_gate_skipped_weight_bytes",
                                    np.count_nonzero(~nonempty) * 6144 * 2)
                                add_scalar(
                                    point, "empty_gate_skipped_dma_commands",
                                    np.count_nonzero(~nonempty) * (passes + 2))
                                add_scalar(point, "generator_weight_read_bytes",
                                           gen_pop_v.sum() * 4 * 96 * 2)
                                add_scalar(point, "generator_signed_adds",
                                           gen_add_v.sum() * 4 * 96 * 2)
                                add_scalar(point, "generator_cache_write_bytes",
                                           used_v.sum() * 576 * 2)
                                add_scalar(point, "generator_commands",
                                           used_v.sum() * 2)
                                add_scalar(point, "generator_cycles",
                                           gen_half.sum() * 2)
                                add_scalar(point, "pwp_sram_read_bytes",
                                           pwp_v.sum() * 8 * 144)
                                add_scalar(point, "psum_sram_reads", psum_access)
                                add_scalar(point, "psum_sram_writes", psum_access)

                print("[M468R3/M469] sample={}/{} operator={}/{}".format(
                    sample + 1, samples, operator + 1, operators), flush=True)
            # Commit is the hard boundary: finish the pending final task in
            # every independent DSE stream and prohibit carry to next sample.
            for (tile, banks, bw_key), stream in strong_stream_state.items():
                require(stream["last_work"] is not None and
                        stream["forced_last_work"] is not None,
                        "empty strong-zero stream before sample commit")
                passes = 1 if banks == 8 else 2
                point = raw[(tile, banks, bw_key, "strong_zero", 0)]
                final_cycles = (stream["last_work"] +
                                cycle_model["tail_cycles_per_pass"])
                forced_final_cycles = (stream["forced_last_work"] +
                                       cycle_model["tail_cycles_per_pass"])
                add_scalar(point, "cycles_without_commit",
                           final_cycles * passes)
                add_scalar(
                    point, "empty_gate_cycle_savings_vs_forced_payload",
                    (forced_final_cycles - final_cycles) * passes)
                stream["last_work"] = None
                stream["forced_last_work"] = None
        require(rows_handle.readline() == "", "unexpected derivative trailing rows")

    require(task_cursor == total_compact_tasks,
            "M470 compact task sidecar extent mismatch")

    require(phase_mismatches == 0,
            "M410R2 derivative does not reproduce M430 phase ledger")
    for key, value in aggregate.items():
        require(value == m430["runtime_population"][key],
                "M430 aggregate mismatch: " + key)

    points = []
    point_index = {}
    for key in sorted(raw.keys(), key=lambda item: (
            item[0], item[1], item[2], item[3], item[4])):
        tile, banks, bw_key, mode, k = key
        bandwidth = "infinite" if bw_key == "infinite" else int(bw_key)
        point = dict(raw[key])
        commit_cycles = samples * sched["commit_cycles_per_sample"]
        cycles = point["cycles_without_commit"] + commit_cycles
        overlap_cycles = (point["overlap_upper_bound_cycles_without_commit"] +
                          commit_cycles if mode == "lazy_pwp" else cycles)
        point["cycles"] = int(cycles)
        point["overlap_upper_bound_cycles"] = int(overlap_cycles)
        point["commit_cycles"] = commit_cycles
        chunks = int(math.ceil(rows_per_phase / float(tile)))
        point["operator_row_tile_commits"] = samples * operators * chunks
        point["block_half_commit_events"] = (
            samples * operators * chunks * (1 if banks == 8 else 2))
        point["committed_accumulator_vectors"] = (
            samples * operators * rows_per_phase * 8)
        point["psum_reset_events"] = point["block_half_commit_events"]
        point["psum_sram_read_bytes"] = point["psum_sram_reads"] * 228
        point["psum_sram_write_bytes"] = point["psum_sram_writes"] * 228
        point["dram_bytes"] = (point["weight_dram_bytes"] +
                               point["pwp_dram_bytes"] +
                               point["config_dram_bytes"])
        point["row_tile"] = tile
        point["resident_block_banks"] = banks
        point["bandwidth_bytes_per_cycle"] = bandwidth
        point["mode"] = mode
        point["generator_source_lanes_k"] = k if mode == "lazy_pwp" else None
        if mode == "lazy_pwp":
            point["generator_peak_onchip_weight_read_bytes_per_cycle"] = 96 * k
            point["generator_independent_source_banks_or_ports"] = k
            point["generator_signed_preadder_proxy"] = 96 * (k - 1)
            point["generator_physical_product_slots"] = 96 * k
            point["generator_same_resource_across_k"] = False
        capacity = capacity_breakdown(
            tile, banks, mode, max_patterns_by_tile[tile], fixed_reserve)
        point["capacity"] = capacity
        point["fits_240k_logical"] = (
            capacity["logical_total_bytes"] <= budget)
        point["fits_240k_macro_rounded"] = (
            capacity["macro_rounded_total_bytes"] <= budget)
        point["fits_both_240k_gates"] = (
            point["fits_240k_logical"] and
            point["fits_240k_macro_rounded"])
        point["m40_payload_reads"] = 0
        point["scope"] = "four frozen H67 ep35 bottleneck Conv3x3 only"
        point["system_speedup"] = False
        points.append(point)
        point_index[(tile, banks, bw_key, mode, k)] = point

    anchor = point_index[(3000, 8, "32", "stored_pwp", 0)]
    anchor_zero = point_index[(3000, 8, "32", "strong_zero", 0)]
    print("[M468R3/M469] anchor stored actual={} expected={} zero actual={} "
          "expected={}".format(
              anchor["cycles"],
              m430["comparisons"]["m430_catalog_dual_cycles"],
              anchor_zero["cycles"],
              m430["comparisons"]["strong_zero_cycles"]), flush=True)
    require(anchor["cycles"] ==
            m430["comparisons"]["m430_catalog_dual_cycles"],
            "full-resident stored-PWP M430 anchor drift: actual={} expected={}".
            format(anchor["cycles"],
                   m430["comparisons"]["m430_catalog_dual_cycles"]))
    require(anchor_zero["cycles"] ==
            m430["comparisons"]["strong_zero_cycles"],
            "full-resident strong-zero anchor drift: actual={} expected={}".
            format(anchor_zero["cycles"],
                   m430["comparisons"]["strong_zero_cycles"]))
    if args.anchor_only:
        require(source_start == sha256(source_path),
                "analyzer changed during anchor-only diagnostic")
        print("M468R3_M469_ANCHOR_ONLY_PASS no_result_written=1", flush=True)
        return 0

    comparisons = []
    for banks in block_axes:
        for bandwidth in bandwidths:
            bw_key = str(bandwidth)
            eligible_zero = [point_index[(tile, banks, bw_key,
                                          "strong_zero", 0)]
                             for tile in tile_sizes
                             if point_index[(tile, banks, bw_key,
                                             "strong_zero", 0)]
                             ["fits_both_240k_gates"]]
            eligible_stored = [point_index[(tile, banks, bw_key,
                                            "stored_pwp", 0)]
                               for tile in tile_sizes
                               if point_index[(tile, banks, bw_key,
                                               "stored_pwp", 0)]
                               ["fits_both_240k_gates"]]
            require(eligible_zero,
                    "no same-budget strong-zero baseline for banks={} bw={}".format(
                        banks, bandwidth))
            best_zero = min(eligible_zero, key=lambda point: point["cycles"])
            best_stored = (min(eligible_stored,
                               key=lambda point: point["cycles"])
                           if eligible_stored else None)
            for tile in tile_sizes:
                bw_key = str(bandwidth)
                zero = point_index[(tile, banks, bw_key, "strong_zero", 0)]
                stored = point_index[(tile, banks, bw_key, "stored_pwp", 0)]
                stored_speed = zero["cycles"] / float(stored["cycles"])
                stored_best_budget_speed = (
                    best_zero["cycles"] / float(stored["cycles"]))
                stored_vs_best_stored = (
                    best_stored["cycles"] / float(stored["cycles"])
                    if best_stored else None)
                stored["speedup_vs_fair_same_tile_strong_zero"] = stored_speed
                stored["speedup_vs_best_same_budget_strong_zero"] = (
                    stored_best_budget_speed)
                stored["best_same_budget_strong_zero_row_tile"] = \
                    best_zero["row_tile"]
                stored["best_same_budget_strong_zero_cycles"] = \
                    best_zero["cycles"]
                stored["speedup_vs_original_m430_517041352_diagnostic"] = (
                    517041352 / float(stored["cycles"]))
                comparisons.append({
                    "row_tile": tile,
                    "resident_block_banks": banks,
                    "bandwidth_bytes_per_cycle": bandwidth,
                    "candidate": "stored_pwp",
                    "candidate_cycles": stored["cycles"],
                    "fair_strong_zero_cycles": zero["cycles"],
                    "speedup_vs_fair_strong_zero": stored_speed,
                    "best_same_budget_strong_zero_row_tile":
                        best_zero["row_tile"],
                    "best_same_budget_strong_zero_cycles":
                        best_zero["cycles"],
                    "speedup_vs_best_same_budget_strong_zero":
                        stored_best_budget_speed,
                    "best_same_budget_stored_pwp_exists":
                        best_stored is not None,
                    "best_same_budget_stored_pwp_row_tile":
                        (best_stored["row_tile"] if best_stored else None),
                    "best_same_budget_stored_pwp_cycles":
                        (best_stored["cycles"] if best_stored else None),
                    "speedup_vs_same_point_stored_pwp": 1.0,
                    "speedup_vs_best_same_budget_stored_pwp":
                        stored_vs_best_stored,
                    "fits_both_240k_gates": stored["fits_both_240k_gates"],
                    "material_vs_strong_zero_1p15": (
                        stored["fits_both_240k_gates"] and
                        stored_best_budget_speed >= 1.15),
                    "material_vs_stored_1p10": False,
                    "performance_admitted": False,
                })
                for k in generator_ks:
                    lazy = point_index[(tile, banks, bw_key,
                                        "lazy_pwp", k)]
                    vs_zero = zero["cycles"] / float(lazy["cycles"])
                    vs_stored = stored["cycles"] / float(lazy["cycles"])
                    vs_best_zero = (best_zero["cycles"] /
                                    float(lazy["cycles"]))
                    vs_best_stored = (best_stored["cycles"] /
                                      float(lazy["cycles"])
                                      if best_stored else None)
                    lazy["speedup_vs_fair_same_tile_strong_zero"] = vs_zero
                    lazy["speedup_vs_same_point_stored_pwp"] = vs_stored
                    lazy["speedup_vs_best_same_budget_strong_zero"] = \
                        vs_best_zero
                    lazy["speedup_vs_best_same_budget_stored_pwp"] = \
                        vs_best_stored
                    lazy["best_same_budget_strong_zero_row_tile"] = \
                        best_zero["row_tile"]
                    lazy["best_same_budget_strong_zero_cycles"] = \
                        best_zero["cycles"]
                    lazy["best_same_budget_stored_pwp_exists"] = \
                        best_stored is not None
                    lazy["best_same_budget_stored_pwp_row_tile"] = \
                        (best_stored["row_tile"] if best_stored else None)
                    lazy["best_same_budget_stored_pwp_cycles"] = \
                        (best_stored["cycles"] if best_stored else None)
                    lazy["speedup_vs_original_m430_517041352_diagnostic"] = (
                        517041352 / float(lazy["cycles"]))
                    lazy["overlap_upper_bound_speedup_vs_fair_strong_zero"] = (
                        zero["cycles"] /
                        float(lazy["overlap_upper_bound_cycles"]))
                    comparisons.append({
                        "row_tile": tile,
                        "resident_block_banks": banks,
                        "bandwidth_bytes_per_cycle": bandwidth,
                        "candidate": "lazy_pwp_k{}".format(k),
                        "candidate_cycles": lazy["cycles"],
                        "fair_strong_zero_cycles": zero["cycles"],
                        "speedup_vs_fair_strong_zero": vs_zero,
                        "best_same_budget_strong_zero_row_tile":
                            best_zero["row_tile"],
                        "best_same_budget_strong_zero_cycles":
                            best_zero["cycles"],
                        "speedup_vs_best_same_budget_strong_zero":
                            vs_best_zero,
                        "best_same_budget_stored_pwp_exists":
                            best_stored is not None,
                        "best_same_budget_stored_pwp_row_tile":
                            (best_stored["row_tile"]
                             if best_stored else None),
                        "best_same_budget_stored_pwp_cycles":
                            (best_stored["cycles"] if best_stored else None),
                        "speedup_vs_same_point_stored_pwp": vs_stored,
                        "speedup_vs_best_same_budget_stored_pwp":
                            vs_best_stored,
                        "fits_both_240k_gates": lazy["fits_both_240k_gates"],
                        "material_vs_strong_zero_1p15": (
                            lazy["fits_both_240k_gates"] and
                            vs_best_zero >= 1.15),
                        "material_vs_stored_1p10": (
                            lazy["fits_both_240k_gates"] and
                            best_stored is not None and
                            vs_best_stored >= 1.10),
                        "performance_admitted": False,
                    })

    result = {
        "schema": "m468r3_m469_h67_lazy_pwp_peer_budget_result_v1",
        "status": "PASS_EXACT_DERIVATIVE_FAIR_EMPTY_LAZY_PWP_CPU_DSE",
        "identity": identity,
        "paper_identity": preflight["paper_identity"],
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "schedule": preflight["schedule_contract"],
        "identity_reproduction": {
            "phase_field_mismatches": phase_mismatches,
            "aggregate": aggregate,
            "full_resident_m430_cycle_mismatches": 0,
            "m40_payload_reads": 0,
            "derivative_traversals": 1,
        },
        "exact_lazy_pwp_generation": {
            "values_checked": pwp_values_checked,
            "canonical_signed16_value_sha256": pwp_value_hash.hexdigest(),
            "minimum_value": pwp_value_min,
            "maximum_value": pwp_value_max,
            "signed12_bound_pass": True,
            "pwp_dram_bytes": 0,
            "serial_generation_is_headline_model": True,
            "overlap_is_nonheadline_upper_bound_only": True,
            "k_points_same_resource": False,
        },
        "points": points,
        "comparisons": comparisons,
        "materiality_rule": preflight["materiality_rule"],
        "claim_boundary": preflight["claim_boundary"],
        "performance_admitted": False,
        "independent_hammer_pending": True,
        "rtl_nominated": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    task_npz = args.output_dir / "m470_partition_window_task_compact_sidecar.npz"
    np.savez_compressed(str(task_npz), **task_sidecar)
    result["m470_partition_window_task_sidecar"] = {
        "path": task_npz.name,
        "sha256": sha256(task_npz),
        "rows": total_compact_tasks,
        "order": "sample,operator,row_tile_size,row_start,partition",
        "fields": sorted(task_sidecar.keys()),
        "derivative_rereads": 0,
        "m40_payload_reads": 0,
        "performance_claim": False
    }
    point_csv = args.output_dir / "m468r3_m469_cycle_traffic_capacity_points.csv"
    flat_rows = []
    for point in points:
        flat = {key: value for key, value in point.items()
                if key != "capacity"}
        flat["logical_sram_bytes"] = point["capacity"][
            "logical_total_bytes"]
        flat["macro_rounded_sram_bytes"] = point["capacity"][
            "macro_rounded_total_bytes"]
        flat_rows.append(flat)
    with point_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    compare_csv = args.output_dir / "m468r3_m469_materiality_comparisons.csv"
    with compare_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    phase_csv = args.output_dir / "m469_g15_phase_compact_sidecar.csv"
    with phase_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_sidecar[0].keys()))
        writer.writeheader()
        writer.writerows(phase_sidecar)

    tile_csv = args.output_dir / "m469_g15_rowtile_compact_sidecar.csv"
    tile_rows = []
    for tile in tile_sizes:
        for k in generator_ks:
            row = {"row_tile": tile, "source_lanes_k": k}
            row.update(g15_tile[(tile, k)])
            tile_rows.append(row)
    with tile_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tile_rows)

    result_path = args.output_dir / \
        "m468r3_m469_h67_lazy_pwp_peer_budget_result_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(source_start == sha256(source_path),
            "analyzer changed during execution")
    names = [point_csv.name, compare_csv.name, phase_csv.name,
             tile_csv.name, task_npz.name, result_path.name]
    manifest, seal = write_seal(args.output_dir, names)
    (args.output_dir / "RUN_COMPLETE.txt").write_text(
        "PASS_EXACT_DERIVATIVE_FAIR_EMPTY_LAZY_PWP_CPU_DSE\n",
        encoding="utf-8")
    print("M468R3_M469_PASS points={} anchor={} seal={}".format(
        len(points), anchor["cycles"], sha256(seal)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
