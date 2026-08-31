#!/usr/bin/env python3
"""M470 exact partition-window payload-stationary/psum-spill CPU DSE.

The primary workload ledger is the sealed M468R3/M469 compact sidecar.  The
contract-visible M410R2 original16 derivative is traversed once only to count
exact empty/nonempty row tiles at the capacity-selected row-tile sizes.  The
M40 one-shot payload is neither an input nor opened.
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


def data_cycles(byte_count, bandwidth):
    if bandwidth == "infinite":
        return 0
    return int(math.ceil(int(byte_count) / float(int(bandwidth))))


def depth64(rows):
    rows = int(rows)
    if rows <= 0:
        return 0
    return int(math.ceil(rows / 64.0) * 64)


def bit_bytes(bits):
    return int((int(bits) + 7) // 8)


def contiguous_runs(ids):
    ids = sorted(int(value) for value in ids)
    if not ids:
        return []
    runs = []
    start = ids[0]
    previous = ids[0]
    for value in ids[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("\n".join(
        "{}  {}".format(sha256(output_dir / name), name)
        for name in sorted(names)) + "\n", encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def capacity_breakdown(row_tile, block_banks, mode, max_active_partitions,
                       max_used_patterns, fixed_reserve):
    require(block_banks in (4, 8), "invalid block-bank axis")
    require(mode in ("strong_zero", "stored_pwp", "lazy_pwp"),
            "invalid M470 mode")
    row_tile = int(row_tile)
    halves_resident = 2 if block_banks == 8 else 1
    config_enabled = mode != "strong_zero" and max_active_partitions > 0
    pwp_enabled = mode != "strong_zero" and max_used_patterns > 0
    pwp_bytes_per_center = 640 if mode == "stored_pwp" else 576

    logical = {
        "psum": bit_bytes(row_tile * block_banks * 96 * 19),
        "active_bitmap": bit_bytes(row_tile),
        "psum_valid_bitmap": bit_bytes(row_tile * block_banks),
        "source_pingpong": row_tile * 2 * 2,
        "descriptor_pingpong_including_row_tags": row_tile * 6 * 2,
        "window_config": (max_active_partitions * 96
                          if config_enabled else 0),
        "window_weights": (max_active_partitions * 6144 * halves_resident),
        "window_stored_pwp": (max_used_patterns * pwp_bytes_per_center *
                              halves_resident
                              if mode == "stored_pwp" else 0),
        "window_lazy_generated_pwp": (max_used_patterns *
                                      pwp_bytes_per_center * halves_resident
                                      if mode == "lazy_pwp" else 0),
        "fifo_control_reserve": fixed_reserve,
    }

    depth = depth64(row_tile)
    rounded = {
        "psum": block_banks * 13 * 18 * depth,
        "active_bitmap": 18 * depth,
        "psum_valid_bitmap": 18 * depth,
        "source_pingpong": 2 * 18 * depth,
        "descriptor_pingpong_including_row_tags": 2 * 18 * depth,
        "window_config": (6 * 18 * depth64(max_active_partitions)
                          if config_enabled else 0),
        "window_weights": (halves_resident * 6 * 18 *
                           depth64(64 * max_active_partitions)),
        "window_stored_pwp": (halves_resident * 9 * 18 *
                              depth64(4 * max_used_patterns)
                              if mode == "stored_pwp" and pwp_enabled else 0),
        "window_lazy_generated_pwp": (halves_resident * 8 * 18 *
                                      depth64(4 * max_used_patterns)
                                      if mode == "lazy_pwp" and pwp_enabled
                                      else 0),
        "fifo_control_reserve": fixed_reserve,
    }
    for name in logical:
        require(rounded[name] >= logical[name],
                "macro-rounded item smaller than logical item: " + name)
    return {
        "row_tile": row_tile,
        "resident_block_banks": block_banks,
        "resident_four_block_payload_halves": halves_resident,
        "maximum_active_partitions_in_window": max_active_partitions,
        "maximum_referenced_centers_in_window": max_used_patterns,
        "logical_items": logical,
        "macro_rounded_items": rounded,
        "logical_total_bytes": int(sum(logical.values())),
        "macro_rounded_total_bytes": int(sum(rounded.values())),
        "macro_depth_rows": depth,
        "macro_width_slice_bits": 144,
        "macro_depth_quantum": 64,
        "every_macro_item_ge_logical": True,
    }


def choose_row_tile(block_banks, mode, window_stat, fixed_reserve, budget):
    selected = None
    for row_tile in range(3000, 0, -1):
        cap = capacity_breakdown(
            row_tile, block_banks, mode,
            window_stat["maximum_active_partitions_in_window"],
            window_stat["maximum_referenced_centers_in_window"],
            fixed_reserve)
        if (cap["logical_total_bytes"] <= budget and
                cap["macro_rounded_total_bytes"] <= budget):
            selected = cap
            break
    minimum = capacity_breakdown(
        1, block_banks, mode,
        window_stat["maximum_active_partitions_in_window"],
        window_stat["maximum_referenced_centers_in_window"],
        fixed_reserve)
    return selected, minimum


def payload_fill_ledger(phases, mode, bandwidth, setup_cycles):
    result = {
        "config_dram_bytes": 0,
        "weight_dram_bytes": 0,
        "pwp_dram_bytes": 0,
        "payload_fill_bytes": 0,
        "config_dma_commands": 0,
        "weight_dma_commands": 0,
        "pwp_dma_commands": 0,
        "payload_dma_commands": 0,
        "payload_fill_cycles": 0,
    }
    for phase in phases:
        if phase["active_rows"] == 0:
            continue
        if mode != "strong_zero":
            result["config_dram_bytes"] += 96
            result["config_dma_commands"] += 1
            result["payload_fill_cycles"] += (
                data_cycles(96, bandwidth) + setup_cycles)
        for unused_half in range(2):
            del unused_half
            result["weight_dram_bytes"] += 6144
            result["weight_dma_commands"] += 1
            result["payload_fill_cycles"] += (
                data_cycles(6144, bandwidth) + setup_cycles)
        if mode == "stored_pwp":
            for start, stop in phase["used_center_runs"]:
                run_bytes = (stop - start + 1) * 640
                for unused_half in range(2):
                    del unused_half
                    result["pwp_dram_bytes"] += run_bytes
                    result["pwp_dma_commands"] += 1
                    result["payload_fill_cycles"] += (
                        data_cycles(run_bytes, bandwidth) + setup_cycles)
    result["payload_fill_bytes"] = (
        result["config_dram_bytes"] + result["weight_dram_bytes"] +
        result["pwp_dram_bytes"])
    result["payload_dma_commands"] = (
        result["config_dma_commands"] + result["weight_dma_commands"] +
        result["pwp_dma_commands"])
    return result


def compute_point(mode, p_value, block_banks, row_tile, bandwidth, phases,
                  aggregate, nonempty_tasks, window_stat, cycle_contract,
                  lazy_k=None):
    passes = 1 if block_banks == 8 else 2
    chunks = int(math.ceil(3000 / float(row_tile)))
    task_count = len(phases) * chunks
    setup = cycle_contract["dma_command_setup_cycles"]
    payload = payload_fill_ledger(phases, mode, bandwidth, setup)

    boundaries = window_stat["total_operator_window_boundaries"]
    spill_one_direction = cycle_contract[
        "psum_bytes_per_direction_per_boundary"]
    spill_write_bytes = boundaries * spill_one_direction
    reload_read_bytes = boundaries * spill_one_direction
    spill_reload_cycles = boundaries * 2 * (
        data_cycles(spill_one_direction, bandwidth) + setup)
    spill_dma_commands = boundaries * 2

    source_sram_bytes = aggregate["source_rows"] * 2 * passes
    descriptor_bytes = aggregate["active_rows"] * 6 * passes
    task_drain_cycles = task_count * passes * cycle_contract[
        "task_drain_cycles"]
    final_commit_cycles = len(phases) // 432 * 24000
    internal_psum_bytes = aggregate["active_rows"] * 8 * 228

    matcher_cycles = 0
    popcount_cycles = 0
    task_fill_cycles = 0
    replay_or_issue_cycles = 0
    descriptor_latency_cycles = 0
    generator_cycles = 0
    generator_commands = 0
    generator_weight_read_bytes = 0
    generator_signed_adds = 0
    generator_cache_write_bytes = 0
    pwp_sram_read_bytes = 0
    weight_sram_read_bytes = 0

    if mode == "strong_zero":
        popcount_cycles = (aggregate["source_rows"] +
                           task_count * cycle_contract[
                               "popcount_filter_pipeline_cycles_per_task"])
        popcount_cycles *= passes
        replay_or_issue_cycles = (
            aggregate["bit_sparse_vector_ops_per_block"] * block_banks *
            passes)
        weight_sram_read_bytes = (
            aggregate["bit_sparse_vector_ops_per_block"] * 8 * 96)
    else:
        matcher_cycles = (aggregate["source_rows"] +
                          aggregate["early_extra"] + 2 * task_count) * passes
        task_fill_cycles = (task_count *
                            cycle_contract["config_select_cycles_per_task"] *
                            passes)
        replay_or_issue_cycles = (
            (aggregate["pwp_rows"] +
             aggregate["correction_ops_per_block"]) * block_banks * passes)
        descriptor_latency_cycles = (
            nonempty_tasks *
            cycle_contract["descriptor_sram_latency_cycles_per_nonempty_task"] *
            passes)
        pwp_sram_read_bytes = aggregate["pwp_rows"] * 8 * 144
        weight_sram_read_bytes = (
            aggregate["correction_ops_per_block"] * 8 * 96)
        if mode == "lazy_pwp":
            require(lazy_k in (1, 2, 4, 8), "missing lazy K")
            field = "k{}_m430_generator_cycles_per_half".format(lazy_k)
            generator_cycles = sum(phase[field] for phase in phases) * 2
            generator_commands = sum(
                phase["m430_used_pwp_patterns"] for phase in phases) * 2
            generator_weight_read_bytes = sum(
                phase["used_center_population_sum"] for phase in phases
            ) * 4 * 96 * 2
            generator_signed_adds = sum(
                phase["used_center_add_source_sum"] for phase in phases
            ) * 4 * 96 * 2
            generator_cache_write_bytes = sum(
                phase["m430_used_pwp_patterns"] for phase in phases
            ) * 576 * 2

    execution_cycles = (matcher_cycles + popcount_cycles + task_fill_cycles +
                        replay_or_issue_cycles + descriptor_latency_cycles +
                        task_drain_cycles)
    total_cycles = (payload["payload_fill_cycles"] + generator_cycles +
                    execution_cycles + spill_reload_cycles +
                    final_commit_cycles)
    point = {
        "mode": mode,
        "partition_window_p": p_value,
        "resident_block_banks": block_banks,
        "row_tile": row_tile,
        "bandwidth_bytes_per_cycle": bandwidth,
        "passes_per_task": passes,
        "row_tiles_per_phase": chunks,
        "task_count": task_count,
        "nonempty_task_count": nonempty_tasks,
        "empty_task_count": task_count - nonempty_tasks,
        "window_count": window_stat["total_operator_windows"],
        "operator_window_boundary_count": boundaries,
        "source_sram_bytes": source_sram_bytes,
        "descriptor_sram_read_bytes": descriptor_bytes,
        "descriptor_sram_write_bytes": descriptor_bytes,
        "matcher_cycles": matcher_cycles,
        "popcount_cycles": popcount_cycles,
        "matcher_or_popcount_cycles": matcher_cycles + popcount_cycles,
        "task_fill_cycles": task_fill_cycles,
        "replay_or_source_issue_cycles": replay_or_issue_cycles,
        "descriptor_latency_cycles": descriptor_latency_cycles,
        "task_drain_cycles": task_drain_cycles,
        "execution_cycles": execution_cycles,
        "weight_sram_read_bytes": weight_sram_read_bytes,
        "pwp_sram_read_bytes": pwp_sram_read_bytes,
        "internal_psum_sram_read_bytes": internal_psum_bytes,
        "internal_psum_sram_write_bytes": internal_psum_bytes,
        "psum_spill_write_bytes": spill_write_bytes,
        "psum_reload_read_bytes": reload_read_bytes,
        "spill_reload_sram_bytes": spill_write_bytes + reload_read_bytes,
        "spill_reload_dram_bytes": spill_write_bytes + reload_read_bytes,
        "spill_reload_cycles": spill_reload_cycles,
        "spill_dma_commands": spill_dma_commands,
        "final_commit_cycles": final_commit_cycles,
        "final_commit_sram_read_bytes": len(phases) // 432 * spill_one_direction,
        "generator_source_lanes_k": lazy_k,
        "generator_cycles": generator_cycles,
        "generator_commands": generator_commands,
        "generator_weight_read_bytes": generator_weight_read_bytes,
        "generator_signed_adds": generator_signed_adds,
        "generator_cache_write_bytes": generator_cache_write_bytes,
        "generator_same_resource_across_k": False if mode == "lazy_pwp" else None,
        "payload_reuse_row_tiles": chunks,
        "total_cycles": int(total_cycles),
        "m40_payload_reads": 0,
        "system_speedup": False,
        "performance_admitted": False,
    }
    point.update(payload)
    point["dma_commands"] = (payload["payload_dma_commands"] +
                             spill_dma_commands)
    point["dram_bytes"] = (payload["payload_fill_bytes"] +
                           spill_write_bytes + reload_read_bytes)
    return point


def flatten_point(point):
    result = {}
    for key, value in point.items():
        if key == "capacity":
            continue
        result[key] = value
    cap = point["capacity"]
    result["logical_sram_bytes"] = cap["logical_total_bytes"]
    result["macro_rounded_sram_bytes"] = cap[
        "macro_rounded_total_bytes"]
    return result


def csv_field_union(rows):
    require(rows, "cannot derive CSV fields from no rows")
    return sorted(set().union(*(set(row.keys()) for row in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M470 overwrite")

    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m470_h67_partition_window_payload_stationary_execution_contract_v1" and
            contract.get("status") ==
            "FROZEN_EXACT_SHA_BEFORE_EXECUTION",
            "M470 execution contract drift")
    root = args.contract.resolve().parents[1]
    inputs = {}
    identity = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M470 input SHA drift: " + name)
        inputs[name] = path
        identity[name] = dict(spec)
    require(inputs["analyzer"].resolve() == source_path and
            identity["analyzer"]["sha256"] == source_start,
            "M470 analyzer self-SHA drift")

    preflight = strict_json(inputs["preflight"])
    require(preflight["schema"] ==
            "m470_h67_partition_window_payload_stationary_preflight_contract_v1" and
            preflight["status"] ==
            "FROZEN_PREFLIGHT_WAITING_FOR_SEALED_M468R3_M469_SIDECAR",
            "M470 preflight drift")
    upstream = strict_json(inputs["m468r3_m469_result"])
    require(upstream["schema"] ==
            "m468r3_m469_h67_lazy_pwp_peer_budget_result_v1" and
            upstream["status"].startswith("PASS_") and
            upstream["identity_reproduction"]["m40_payload_reads"] == 0 and
            upstream["identity_reproduction"]["derivative_traversals"] == 1,
            "M468R3/M469 upstream not sealed PASS")
    task_sidecar_meta = upstream["m470_partition_window_task_sidecar"]
    require(task_sidecar_meta["sha256"] ==
            identity["m470_task_sidecar"]["sha256"] and
            task_sidecar_meta["derivative_rereads"] == 0 and
            task_sidecar_meta["m40_payload_reads"] == 0,
            "M470 task sidecar identity drift")
    derivative = strict_json(inputs["m410r2_manifest"])
    require(derivative["status"] ==
            "PASS_M410R2_CONTRACT_VISIBLE_FULL_RUNTIME_STIMULUS_EXPORT" and
            derivative["output"]["rows"]["sha256"] ==
            identity["m410r2_rows"]["sha256"],
            "M410R2 derivative identity drift")

    phase_rows = []
    with inputs["m469_phase_sidecar"].open(
            "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "empty M469 phase sidecar")
        for raw in reader:
            phase = {key: int(value) for key, value in raw.items()}
            phase_rows.append(phase)
    require(len(phase_rows) == 10 * 4 * 432,
            "M469 phase sidecar extent drift")

    catalog = strict_json(inputs["m430_catalog"])
    lazy_fields = [
        "k{}_m430_generator_cycles_per_half".format(k)
        for k in (1, 2, 4, 8)]
    lazy_supported = all(field in phase_rows[0] for field in lazy_fields)
    require(lazy_supported, "sealed M469 sidecar lacks exact lazy ledger")

    aggregate = {
        "source_rows": 0,
        "active_rows": 0,
        "pwp_rows": 0,
        "fallback_rows": 0,
        "correction_ops_per_block": 0,
        "bit_sparse_vector_ops_per_block": 0,
        "early_extra": 0,
    }
    for index, phase in enumerate(phase_rows):
        sample, rem = divmod(index, 4 * 432)
        operator, partition = divmod(rem, 432)
        require(phase["sample"] == sample and
                phase["operator"] == operator and
                phase["partition"] == partition,
                "M469 phase order drift")
        require(phase["source_rows"] == 3000,
                "M469 source rows drift")
        used_ids = [center for center in range(32)
                    if phase["m430_center{}".format(center)] > 0]
        runs = contiguous_runs(used_ids)
        require(len(used_ids) == phase["m430_used_pwp_patterns"] and
                len(runs) == phase["m430_used_center_runs"],
                "M469 center/run sidecar mismatch")
        patterns = catalog["operators"][operator]["partitions"][partition][
            "nested_patterns"]
        require(len(patterns) == 32, "M430 catalog extent drift")
        center_pops = [bin(int(patterns[value], 16)).count("1")
                       for value in used_ids]
        require(sum(center_pops) == phase["used_center_population_sum"],
                "M469 center population mismatch")
        phase["used_center_ids"] = used_ids
        phase["used_center_runs"] = runs
        phase["used_center_add_source_sum"] = sum(
            max(value - 1, 0) for value in center_pops)
        for k in (1, 2, 4, 8):
            expected_gen = (len(used_ids) +
                            4 * sum(int(math.ceil(value / float(k)))
                                    for value in center_pops) +
                            (2 if used_ids else 0))
            require(expected_gen == phase[
                "k{}_m430_generator_cycles_per_half".format(k)],
                "M469 generator ledger mismatch")
        early_extra = phase["m430_early_matcher_cycles"] - 3000 - 2
        require(early_extra >= 0, "negative matcher early extra")
        aggregate["source_rows"] += phase["source_rows"]
        aggregate["active_rows"] += phase["active_rows"]
        aggregate["pwp_rows"] += phase["m430_use_pwp_rows"]
        aggregate["fallback_rows"] += phase["m430_fallback_rows"]
        aggregate["correction_ops_per_block"] += phase[
            "m430_correction_ops_per_block"]
        aggregate["bit_sparse_vector_ops_per_block"] += phase[
            "strong_zero_bit_sparse_ops_per_block"]
        aggregate["early_extra"] += early_extra

    upstream_aggregate = upstream["identity_reproduction"]["aggregate"]
    for key in ("source_rows", "active_rows", "pwp_rows", "fallback_rows",
                "correction_ops_per_block",
                "bit_sparse_vector_ops_per_block"):
        require(aggregate[key] == upstream_aggregate[key],
                "M470 aggregate != M468R3: " + key)

    p_values = preflight["dse"]["partition_window_sizes"]
    block_axes = preflight["dse"]["resident_block_banks"]
    bandwidths = preflight["dse"]["dram_bytes_per_cycle"]
    budget = preflight["dse"]["sram_budget_bytes"]
    fixed_reserve = preflight["capacity_contract"][
        "fixed_fifo_control_reserve_bytes"]

    window_stats = {}
    for p_value in p_values:
        max_active = 0
        max_patterns = 0
        windows_total = 0
        for sample in range(10):
            for operator in range(4):
                base = (sample * 4 + operator) * 432
                operator_rows = phase_rows[base:base + 432]
                require(len(operator_rows) == 432,
                        "operator phase extent drift")
                for start in range(0, 432, p_value):
                    window = operator_rows[start:start + p_value]
                    active = sum(1 for phase in window
                                 if phase["active_rows"] != 0)
                    patterns = sum(phase["m430_used_pwp_patterns"]
                                   for phase in window
                                   if phase["active_rows"] != 0)
                    max_active = max(max_active, active)
                    max_patterns = max(max_patterns, patterns)
                    windows_total += 1
        windows_per_operator = int(math.ceil(432 / float(p_value)))
        require(windows_total == 40 * windows_per_operator,
                "window enumeration drift")
        window_stats[p_value] = {
            "partition_window_p": p_value,
            "windows_per_sample_operator": windows_per_operator,
            "total_operator_windows": windows_total,
            "boundaries_per_sample_operator": windows_per_operator - 1,
            "total_operator_window_boundaries":
                40 * (windows_per_operator - 1),
            "maximum_active_partitions_in_window": max_active,
            "maximum_referenced_centers_in_window": max_patterns,
        }

    modes = ["strong_zero", "stored_pwp"]
    if lazy_supported:
        modes.append("lazy_pwp")
    selections = {}
    infeasible = []
    needed_tiles = set()
    for mode in modes:
        for p_value in p_values:
            for banks in block_axes:
                selected, minimum = choose_row_tile(
                    banks, mode, window_stats[p_value], fixed_reserve, budget)
                selections[(mode, p_value, banks)] = selected
                if selected is None:
                    infeasible.append({
                        "mode": mode,
                        "partition_window_p": p_value,
                        "resident_block_banks": banks,
                        "reason": "NO_INTEGER_ROW_TILE_PASSES_BOTH_240K_GATES",
                        "row_tile_tested": 1,
                        "logical_sram_bytes_at_row_tile1": minimum[
                            "logical_total_bytes"],
                        "macro_rounded_sram_bytes_at_row_tile1": minimum[
                            "macro_rounded_total_bytes"],
                        "logical_fit_at_row_tile1": minimum[
                            "logical_total_bytes"] <= budget,
                        "macro_fit_at_row_tile1": minimum[
                            "macro_rounded_total_bytes"] <= budget,
                        "maximum_active_partitions_in_window": window_stats[
                            p_value]["maximum_active_partitions_in_window"],
                        "maximum_referenced_centers_in_window": window_stats[
                            p_value]["maximum_referenced_centers_in_window"],
                    })
                else:
                    needed_tiles.add(selected["row_tile"])

    require(needed_tiles, "no feasible M470 capacity point")
    nonempty_by_tile = {}
    task_sidecar_tiles = []
    with np.load(str(inputs["m470_task_sidecar"]),
                 allow_pickle=False) as sidecar:
        require(set(sidecar.files) == set(task_sidecar_meta["fields"]),
                "M470 task sidecar field drift")
        available_tiles = sorted(set(int(value) for value in
                                     np.unique(sidecar["row_tile_size"])))
        for tile in sorted(needed_tiles):
            if tile not in available_tiles:
                continue
            selected = sidecar["row_tile_size"] == tile
            expected_tasks = len(phase_rows) * int(math.ceil(
                3000 / float(tile)))
            require(int(np.count_nonzero(selected)) == expected_tasks,
                    "task sidecar tile extent drift")
            active_rows = sidecar["active_rows"][selected]
            require(int(active_rows.astype(np.int64).sum()) ==
                    aggregate["active_rows"],
                    "task sidecar active aggregate drift")
            require(int(sidecar["pwp_rows"][selected].astype(
                np.int64).sum()) == aggregate["pwp_rows"],
                "task sidecar PWP aggregate drift")
            require(int(sidecar["correction_ops_per_block"][selected].astype(
                np.int64).sum()) == aggregate["correction_ops_per_block"],
                "task sidecar correction aggregate drift")
            require(int(sidecar[
                "strong_zero_pop_ops_per_block"][selected].astype(
                    np.int64).sum()) ==
                    aggregate["bit_sparse_vector_ops_per_block"],
                    "task sidecar zero-pop aggregate drift")
            expected_matcher = (aggregate["source_rows"] +
                                aggregate["early_extra"] +
                                2 * expected_tasks)
            require(int(sidecar["early_matcher_cycles"][selected].astype(
                np.int64).sum()) == expected_matcher,
                "task sidecar matcher aggregate drift")
            nonempty_by_tile[tile] = int(np.count_nonzero(active_rows))
            task_sidecar_tiles.append(tile)

    missing_tiles = sorted(needed_tiles - set(task_sidecar_tiles))
    derivative_active_mismatches = 0
    derivative_traversals = 0
    if missing_tiles:
        derivative_traversals = 1
        for tile in missing_tiles:
            nonempty_by_tile[tile] = 0
        row_path = inputs["m410r2_rows"]
        with row_path.open("r", encoding="ascii") as rows_handle:
            for phase_index, phase in enumerate(phase_rows):
                originals = np.fromiter(
                    (int(rows_handle.readline(), 16) & 0xffff
                     for unused in range(3000)),
                    dtype=np.uint16, count=3000)
                require(originals.size == 3000,
                        "premature M410R2 derivative EOF")
                active = originals != 0
                if int(active.sum()) != phase["active_rows"]:
                    derivative_active_mismatches += 1
                for tile in missing_tiles:
                    starts = np.arange(0, 3000, tile, dtype=np.int64)
                    counts = np.add.reduceat(active.astype(np.uint16), starts)
                    nonempty_by_tile[tile] += int(np.count_nonzero(counts))
                if (phase_index + 1) % 432 == 0:
                    print("[M470] derivative phases={}/{}".format(
                        phase_index + 1, len(phase_rows)), flush=True)
            require(rows_handle.readline() == "",
                    "unexpected M410R2 derivative trailing rows")
    require(derivative_active_mismatches == 0,
            "M410R2 active rows != sealed M469 sidecar")

    cycle_contract = dict(preflight["cycle_and_traffic_contract"])
    cycle_contract["psum_bytes_per_direction_per_boundary"] = preflight[
        "schedule_contract"]["psum_bytes_per_direction_per_boundary"]
    points = []
    point_index = {}
    for mode in modes:
        k_axis = ([None] if mode != "lazy_pwp" else
                  preflight["dse"]["lazy_generator_source_lanes_k"])
        for p_value in p_values:
            for banks in block_axes:
                capacity = selections[(mode, p_value, banks)]
                if capacity is None:
                    continue
                tile = capacity["row_tile"]
                for bandwidth in bandwidths:
                    for lazy_k in k_axis:
                        point = compute_point(
                            mode, p_value, banks, tile, bandwidth, phase_rows,
                            aggregate, nonempty_by_tile[tile],
                            window_stats[p_value], cycle_contract,
                            lazy_k=lazy_k)
                        point["capacity"] = capacity
                        point["fits_240k_logical"] = (
                            capacity["logical_total_bytes"] <= budget)
                        point["fits_240k_macro_rounded"] = (
                            capacity["macro_rounded_total_bytes"] <= budget)
                        point["fits_both_240k_gates"] = True
                        points.append(point)
                        point_index[(mode, p_value, banks, str(bandwidth),
                                     lazy_k)] = point

    comparisons = []
    for point in points:
        if point["mode"] == "strong_zero":
            continue
        p_value = point["partition_window_p"]
        banks = point["resident_block_banks"]
        bandwidth = point["bandwidth_bytes_per_cycle"]
        lazy_k = point["generator_source_lanes_k"]
        strong_capacity = selections[("strong_zero", p_value, banks)]
        require(strong_capacity is not None,
                "candidate fit but strong-zero does not fit")
        optimized_zero = point_index[("strong_zero", p_value, banks,
                                      str(bandwidth), None)]
        same_tile_zero = compute_point(
            "strong_zero", p_value, banks, point["row_tile"], bandwidth,
            phase_rows, aggregate, nonempty_by_tile[point["row_tile"]],
            window_stats[p_value], cycle_contract)
        speed_same_tile = (same_tile_zero["total_cycles"] /
                           float(point["total_cycles"]))
        speed_same_resource = (optimized_zero["total_cycles"] /
                               float(point["total_cycles"]))
        point["speedup_vs_same_tile_strong_zero"] = speed_same_tile
        point["speedup_vs_same_resource_optimized_strong_zero"] = (
            speed_same_resource)
        point["same_resource_strong_zero_row_tile"] = optimized_zero[
            "row_tile"]
        comparison = {
            "candidate": (point["mode"] if lazy_k is None else
                          "lazy_pwp_k{}".format(lazy_k)),
            "partition_window_p": p_value,
            "resident_block_banks": banks,
            "bandwidth_bytes_per_cycle": bandwidth,
            "candidate_row_tile": point["row_tile"],
            "candidate_cycles": point["total_cycles"],
            "same_tile_strong_zero_cycles": same_tile_zero["total_cycles"],
            "same_tile_speedup": speed_same_tile,
            "same_resource_strong_zero_row_tile": optimized_zero["row_tile"],
            "same_resource_strong_zero_cycles": optimized_zero["total_cycles"],
            "same_resource_speedup": speed_same_resource,
            "candidate_fits_both_240k_gates": True,
            "same_resource_zero_fits_both_240k_gates": True,
            "material_1p15": speed_same_resource >= 1.15,
            "kill_below_1p10": speed_same_resource < 1.10,
            "performance_admitted": False,
        }
        comparisons.append(comparison)

    stored_128 = [row for row in comparisons
                  if row["candidate"] == "stored_pwp" and
                  row["bandwidth_bytes_per_cycle"] == 128]
    require(stored_128, "no feasible stored-PWP 128 B/cycle point")
    best = max(stored_128, key=lambda row: row["same_resource_speedup"])
    if best["same_resource_speedup"] >= preflight[
            "fairness_and_decision"]["nominate_threshold"]:
        decision = "NOMINATE_FOR_INDEPENDENT_HAMMER_NOT_PERFORMANCE_ADMISSION"
    elif best["same_resource_speedup"] < preflight[
            "fairness_and_decision"]["kill_threshold"]:
        decision = "KILL_M470_HARDWARE_PERFORMANCE_AXIS"
    else:
        decision = "HOLD_NONMATERIAL_1P10_TO_1P15"

    feasible_ranges = {}
    for mode in modes:
        feasible_ranges[mode] = {}
        for banks in block_axes:
            feasible_ranges[mode][str(banks)] = [
                p_value for p_value in p_values
                if selections[(mode, p_value, banks)] is not None]

    result = {
        "schema": "m470_h67_partition_window_payload_stationary_result_v1",
        "status": "PASS_M470_EXACT_PARTITION_WINDOW_PAYLOAD_STATIONARY_CPU_DSE",
        "identity": identity,
        "paper_identity": preflight["paper_identity"],
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "upstream_boundary": preflight["upstream_boundary"],
        "identity_reproduction": {
            "phase_rows": len(phase_rows),
            "aggregate": aggregate,
            "m468r3_aggregate_mismatches": 0,
            "m410r2_active_row_mismatches": derivative_active_mismatches,
            "m470_task_sidecar_tiles": task_sidecar_tiles,
            "m410r2_supplement_tiles": missing_tiles,
            "m410r2_derivative_traversals": derivative_traversals,
            "m40_payload_reads": 0,
        },
        "lazy_sidecar_support": {
            "supported": lazy_supported,
            "generator_source_lanes_k": [1, 2, 4, 8],
            "same_resource_across_k": False,
            "stored_pwp_is_primary": True,
        },
        "window_statistics": [window_stats[value] for value in p_values],
        "feasible_partition_window_range": feasible_ranges,
        "points": points,
        "infeasible_points": infeasible,
        "comparisons": comparisons,
        "decision": {
            "status": decision,
            "best_stored_pwp_128Bpc": best,
            "nominate_threshold": preflight[
                "fairness_and_decision"]["nominate_threshold"],
            "kill_threshold": preflight[
                "fairness_and_decision"]["kill_threshold"],
            "performance_admitted": False,
            "independent_hammer_pending": decision.startswith("NOMINATE"),
            "rtl_nominated": False,
        },
        "claim_boundary": preflight["claim_boundary"],
        "performance_admitted": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    point_path = args.output_dir / "m470_cycle_traffic_capacity_points.csv"
    point_rows = [flatten_point(point) for point in points]
    with point_path.open("w", encoding="utf-8", newline="") as handle:
        point_fields = csv_field_union(point_rows)
        writer = csv.DictWriter(handle, fieldnames=point_fields)
        writer.writeheader()
        writer.writerows(point_rows)

    compare_path = args.output_dir / "m470_materiality_comparisons.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    infeasible_path = args.output_dir / "m470_infeasible_capacity_points.csv"
    with infeasible_path.open("w", encoding="utf-8", newline="") as handle:
        if infeasible:
            writer = csv.DictWriter(handle,
                                    fieldnames=list(infeasible[0].keys()))
            writer.writeheader()
            writer.writerows(infeasible)
        else:
            handle.write("mode,partition_window_p,resident_block_banks,reason\n")

    result_path = args.output_dir / \
        "m470_h67_partition_window_payload_stationary_result_r1.json"
    result_path.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    require(source_start == sha256(source_path),
            "M470 analyzer changed during execution")
    names = [point_path.name, compare_path.name, infeasible_path.name,
             result_path.name]
    unused_manifest, seal = write_seal(args.output_dir, names)
    del unused_manifest
    (args.output_dir / "RUN_COMPLETE.txt").write_text(
        "PASS_M470_EXACT_PARTITION_WINDOW_PAYLOAD_STATIONARY_CPU_DSE\n",
        encoding="utf-8")
    print("M470_PASS points={} best={:.6f} decision={} seal={}".format(
        len(points), best["same_resource_speedup"], decision,
        sha256(seal)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
