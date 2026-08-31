#!/usr/bin/env python3
"""Independent fail-closed hammer for the sealed M468R3/M469 R6 CPU DSE."""

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


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_bandwidth(value):
    return "infinite" if str(value) == "infinite" else int(value)


def bw_cycles(byte_count, bandwidth):
    if bandwidth == "infinite":
        return np.zeros(np.asarray(byte_count).shape, dtype=np.int64)
    values = np.asarray(byte_count, dtype=np.int64)
    return (values + int(bandwidth) - 1) // int(bandwidth)


def bw_scalar(byte_count, bandwidth):
    if bandwidth == "infinite":
        return 0
    return int(math.ceil(int(byte_count) / float(bandwidth)))


def bit_bytes(bits):
    return (int(bits) + 7) // 8


def depth64(rows):
    return int(math.ceil(int(rows) / 64.0) * 64)


def capacity(row_tile, banks, mode, max_patterns, reserve):
    """Reviewer-owned dimensional reconstruction of both capacity gates."""
    depth = depth64(row_tile)
    halves = 2 if banks == 8 else 1
    logical = {
        "psum": bit_bytes(row_tile * banks * 96 * 19),
        "active_bitmap": bit_bytes(row_tile),
        "psum_valid_bitmap": bit_bytes(row_tile * banks),
        "source_pingpong": row_tile * 2 * 2,
        "descriptor_pingpong_including_row_tags": row_tile * 6 * 2,
        "config_pingpong": 192 if mode != "strong_zero" else 0,
        "weight_payload": halves * 6144,
        "stored_pwp_payload": (halves * max_patterns * 640
                               if mode == "stored_pwp" else 0),
        "lazy_center_buffer_serial": (max_patterns * 576
                                      if mode == "lazy_pwp" else 0),
        "fifo_control_reserve": reserve,
    }
    rounded = {
        "psum": banks * 13 * 18 * depth,
        "active_bitmap": 18 * depth,
        "psum_valid_bitmap": 18 * depth,
        "source_pingpong": 2 * 18 * depth,
        "descriptor_pingpong_including_row_tags": 2 * 18 * depth,
        "config_pingpong": 2 * 6 * 18 * 64 if mode != "strong_zero" else 0,
        "weight_payload": halves * 22 * 18 * 64,
        "stored_pwp_payload": (halves * 36 * 18 * 64
                               if mode == "stored_pwp" else 0),
        "lazy_center_buffer_serial": (32 * 18 * 64
                                      if mode == "lazy_pwp" else 0),
        "fifo_control_reserve": reserve,
    }
    require(all(rounded[name] >= logical[name] for name in logical),
            "macro proxy undercounts a logical item")
    return logical, rounded


def point_key(row):
    mode = row["mode"]
    k = (int(row["generator_source_lanes_k"])
         if row.get("generator_source_lanes_k") not in (None, "", "null")
         else 0)
    return (int(row["row_tile"]), int(row["resident_block_banks"]),
            str(row["bandwidth_bytes_per_cycle"]), mode, k)


def bitcounts_u32(values):
    lookup = np.fromiter((bin(int(i)).count("1") for i in range(256)),
                         dtype=np.uint8, count=256)
    values = np.asarray(values, dtype=np.uint32)
    return (lookup[values & 255] + lookup[(values >> 8) & 255] +
            lookup[(values >> 16) & 255] +
            lookup[(values >> 24) & 255]).astype(np.int64)


def streamed_strong_cycles(sample, work, preprocess, tail, passes):
    total = 0
    for sample_id in range(10):
        index = np.flatnonzero(sample == sample_id)
        require(index.size > 0, "missing sample in task sidecar")
        w = work[index]
        p = preprocess[index]
        total += int(p[0])
        if index.size > 1:
            total += int(np.maximum(w[:-1], p[1:]).sum())
            total += int(index.size - 1) * tail
        total += int(w[-1]) + tail
    return total * passes


def reset_sequence_strong_cycles(sample, operator, row_start, partition,
                                 work, preprocess, tail, passes):
    """Diagnostic alternative implied by stale preflight recovery wording."""
    total = 0
    boundary = ((np.arange(sample.size) == 0) |
                (sample != np.roll(sample, 1)) |
                (operator != np.roll(operator, 1)) |
                (row_start != np.roll(row_start, 1)))
    starts = np.flatnonzero(boundary)
    stops = np.r_[starts[1:], sample.size]
    for start, stop in zip(starts, stops):
        require(np.array_equal(partition[start:stop],
                               np.arange(stop - start, dtype=partition.dtype)),
                "partition sequence is not contiguous")
        w = work[start:stop]
        p = preprocess[start:stop]
        total += int(p[0])
        if w.size > 1:
            total += int(np.maximum(w[:-1], p[1:]).sum())
            total += int(w.size - 1) * tail
        total += int(w[-1]) + tail
    return total * passes


def recompute_point(arrays, mask, tile, banks, bandwidth, mode, k,
                    cycle_model, commit_cycles):
    sample = arrays["sample"][mask].astype(np.int64)
    operator = arrays["operator"][mask].astype(np.int64)
    partition = arrays["partition"][mask].astype(np.int64)
    row_start = arrays["row_start"][mask].astype(np.int64)
    row_stop = arrays["row_stop"][mask].astype(np.int64)
    row_count = row_stop - row_start
    active = arrays["active_rows"][mask].astype(np.int64)
    pwp = arrays["pwp_rows"][mask].astype(np.int64)
    correction = arrays["correction_ops_per_block"][mask].astype(np.int64)
    zero_work_per_block = arrays[
        "strong_zero_pop_ops_per_block"][mask].astype(np.int64)
    matcher = arrays["early_matcher_cycles"][mask].astype(np.int64)
    used_mask = arrays["used_center_mask_u32"][mask].astype(np.uint32)
    used = bitcounts_u32(used_mask)
    runs = arrays["used_center_runs"][mask].astype(np.int64)
    gen_pop = arrays["used_center_population_sum"][mask].astype(np.int64)
    nonempty = active != 0
    empty = ~nonempty
    passes = 1 if banks == 8 else 2
    tail = int(cycle_model["tail_cycles_per_pass"])
    scan = row_count + int(cycle_model["popcount_filter_pipeline_cycles"])
    replay = ((pwp + correction) * 4 +
              int(cycle_model["descriptor_sram_latency_cycles"]))
    config_dma = (bw_scalar(cycle_model["elastic_config_bytes"], bandwidth) +
                  int(cycle_model["dma_command_setup_cycles"]))
    weight_half = (bw_scalar(
        cycle_model["weight_bytes_per_four_blocks"], bandwidth) +
        int(cycle_model["dma_command_setup_cycles"]))
    point = {
        "task_count": int(sample.size),
        "task_passes": int(sample.size * passes),
        "empty_task_passes": int(empty.sum() * passes),
        "nonempty_task_passes": int(nonempty.sum() * passes),
        "source_sram_bytes": int(row_count.sum() * 2 * passes),
        "source_dram_bytes": 0,
        "weight_dram_bytes": int(nonempty.sum() * 6144 * 2),
        "pwp_dram_bytes": 0,
        "config_dram_bytes": 0,
        "psum_sram_reads": int(active.sum() * banks * passes),
        "psum_sram_writes": int(active.sum() * banks * passes),
        "psum_dram_spill_bytes": 0,
        "pwp_sram_read_bytes": 0,
        "dma_commands": 0,
        "generator_weight_read_bytes": 0,
        "generator_signed_adds": 0,
        "generator_cache_write_bytes": 0,
        "generator_commands": 0,
        "generator_cycles": 0,
        "empty_gate_skipped_config_bytes": 0,
        "empty_gate_skipped_weight_bytes": int(empty.sum() * 6144 * 2),
        "empty_gate_skipped_dma_commands": 0,
    }
    alternate_reset_cycles = None
    if mode == "strong_zero":
        weight_dma = 2 * weight_half if banks == 8 else weight_half
        preprocess = np.where(nonempty, np.maximum(scan, weight_dma), scan)
        forced = np.maximum(scan, weight_dma)
        work = zero_work_per_block * banks
        without_commit = streamed_strong_cycles(
            sample, work, preprocess, tail, passes)
        forced_without_commit = streamed_strong_cycles(
            sample, work, forced, tail, passes)
        alternate_reset_cycles = reset_sequence_strong_cycles(
            sample, operator, row_start, partition, work, preprocess,
            tail, passes) + commit_cycles
        point["dma_commands"] = int(nonempty.sum() * 2)
        point["empty_gate_skipped_dma_commands"] = int(empty.sum() * 2)
        empty_savings = forced_without_commit - without_commit
        overlap = without_commit
    elif mode == "stored_pwp":
        payload_bytes = 6144 + used * 640
        payload_dma = (bw_cycles(payload_bytes, bandwidth) +
                       (1 + runs) *
                       int(cycle_model["dma_command_setup_cycles"]))
        common = config_dma + matcher + 1
        if banks == 8:
            nongated = (common + payload_dma +
                        np.maximum(replay, payload_dma) + replay + tail)
            cycles = np.where(nonempty, nongated, scan + tail)
        else:
            nongated = passes * (common + payload_dma + replay + tail)
            cycles = np.where(nonempty, nongated, passes * (scan + tail))
        without_commit = int(cycles.sum())
        empty_savings = int((nongated[empty] - cycles[empty]).sum())
        overlap = without_commit
        point["config_dram_bytes"] = int(nonempty.sum() * 96 * passes)
        point["pwp_dram_bytes"] = int(used[nonempty].sum() * 640 * 2)
        point["dma_commands"] = int(nonempty.sum() * passes +
                                    ((1 + runs[nonempty]) * 2).sum())
        point["empty_gate_skipped_config_bytes"] = int(
            empty.sum() * 96 * passes)
        point["empty_gate_skipped_dma_commands"] = int(
            empty.sum() * (passes + 2))
        point["pwp_sram_read_bytes"] = int(pwp.sum() * 8 * 144)
    else:
        gen_ceil = arrays[
            "generator_center_ceil_sum_k{}".format(k)][mask].astype(np.int64)
        gen_half = (used * int(cycle_model["generator_center_setup_cycles"]) +
                    4 * gen_ceil +
                    np.where(used != 0,
                             int(cycle_model["generator_half_flush_cycles"]),
                             0))
        common = config_dma + matcher + 1
        if banks == 8:
            nongated = (common + 2 * weight_half + 2 * gen_half +
                        2 * replay + tail)
            cycles = np.where(nonempty, nongated, scan + tail)
            overlap_nongated = (common + 2 * weight_half + gen_half +
                                 np.maximum(replay, gen_half) + replay + tail)
            overlap_cycles = np.where(nonempty, overlap_nongated, scan + tail)
        else:
            nongated = passes * (common + weight_half + gen_half +
                                 replay + tail)
            cycles = np.where(nonempty, nongated, passes * (scan + tail))
            overlap_cycles = cycles
        without_commit = int(cycles.sum())
        overlap = int(overlap_cycles.sum())
        empty_savings = int((nongated[empty] - cycles[empty]).sum())
        point["config_dram_bytes"] = int(nonempty.sum() * 96 * passes)
        point["dma_commands"] = int(nonempty.sum() * (passes + 2))
        point["empty_gate_skipped_config_bytes"] = int(
            empty.sum() * 96 * passes)
        point["empty_gate_skipped_dma_commands"] = int(
            empty.sum() * (passes + 2))
        point["generator_weight_read_bytes"] = int(gen_pop.sum() * 4 * 96 * 2)
        point["generator_signed_adds"] = int(
            (gen_pop - used).sum() * 4 * 96 * 2)
        point["generator_cache_write_bytes"] = int(used.sum() * 576 * 2)
        point["generator_commands"] = int(used.sum() * 2)
        point["generator_cycles"] = int(gen_half.sum() * 2)
        point["pwp_sram_read_bytes"] = int(pwp.sum() * 8 * 144)
    point["cycles_without_commit"] = int(without_commit)
    point["overlap_upper_bound_cycles_without_commit"] = int(overlap)
    point["empty_gate_cycle_savings_vs_forced_payload"] = int(empty_savings)
    point["empty_gate_skipped_weight_bytes"] = int(empty.sum() * 6144 * 2)
    point["cycles"] = int(without_commit + commit_cycles)
    point["overlap_upper_bound_cycles"] = int(overlap + commit_cycles)
    point["commit_cycles"] = int(commit_cycles)
    point["psum_sram_read_bytes"] = point["psum_sram_reads"] * 228
    point["psum_sram_write_bytes"] = point["psum_sram_writes"] * 228
    point["dram_bytes"] = (point["weight_dram_bytes"] +
                           point["pwp_dram_bytes"] +
                           point["config_dram_bytes"])
    point["operator_row_tile_commits"] = int(
        10 * 4 * math.ceil(3000 / float(tile)))
    point["block_half_commit_events"] = int(
        point["operator_row_tile_commits"] * (1 if banks == 8 else 2))
    point["committed_accumulator_vectors"] = 10 * 4 * 3000 * 8
    point["psum_reset_events"] = point["block_half_commit_events"]
    return point, alternate_reset_cycles


def compare_integer_fields(actual, expected, fields, label):
    for field in fields:
        require(int(actual[field]) == int(expected[field]),
                "{} {} mismatch: {} vs {}".format(
                    label, field, actual[field], expected[field]))


def write_manifest(output_dir, names):
    manifest = output_dir / "M472_REVIEW_SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(output_dir / name), name)
        for name in sorted(names)), encoding="utf-8")
    seal = output_dir / "M472_REVIEW_SHA256SUMS.seal.sha256"
    seal.write_text("{}  M472_REVIEW_SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing to overwrite review")
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m472_m468r3_m469_r6_independent_hammer_contract_v1",
            "review contract schema drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    input_hashes = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        actual_hash = sha256(path)
        require(actual_hash == spec["sha256"], "input SHA drift: " + name)
        paths[name] = path
        input_hashes[name] = actual_hash

    execution = strict_json(paths["execution_contract"])
    preflight = strict_json(paths["preflight_contract"])
    producer = strict_json(paths["producer_result"])
    require(producer["status"] ==
            "PASS_EXACT_DERIVATIVE_FAIR_EMPTY_LAZY_PWP_CPU_DSE",
            "producer status drift")
    require(producer["performance_admitted"] is False and
            producer["rtl_nominated"] is False and
            producer["independent_hammer_pending"] is True,
            "producer fail-closed boundary drift")

    # Producer seal: exactly six relative payload entries, then one outer seal.
    manifest_rows = []
    for line in paths["producer_manifest"].read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name and not Path(name).is_absolute() and
                Path(name).name == name and name not in [x[1] for x in manifest_rows],
                "unsafe or duplicate producer manifest target")
        require(sha256(paths["producer_manifest"].parent / name) == digest,
                "producer manifest mismatch: " + name)
        manifest_rows.append((digest, name))
    require(len(manifest_rows) == 6, "producer manifest extent drift")
    seal_digest, seal_target = paths["producer_outer_seal"].read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(seal_target == "SHA256SUMS" and
            seal_digest == sha256(paths["producer_manifest"]),
            "producer outer seal mismatch")
    require(paths["producer_run_complete"].read_text(
        encoding="utf-8").strip() ==
        "PASS_EXACT_DERIVATIVE_FAIR_EMPTY_LAZY_PWP_CPU_DSE",
        "RUN_COMPLETE marker drift")

    # r1-r5 must remain marker-only fail-closed evidence.
    failure_chain = []
    expected_tokens = {
        "r1_abort": "not a PASS",
        "r2_failure": "no result JSON",
        "r3_failure": "No point in this directory is a PASS",
        "r4_failure": "No point in this directory is a PASS",
        "r5_failure": "No result or performance point was written",
    }
    for name, token in expected_tokens.items():
        marker = paths[name]
        require(token in marker.read_text(encoding="utf-8"),
                "prior failure marker semantic drift: " + name)
        directory_files = sorted(p.name for p in marker.parent.iterdir()
                                 if p.is_file())
        require(directory_files == [marker.name],
                "prior failure directory is no longer marker-only: " + name)
        failure_chain.append({"stage": name, "marker": marker.name,
                              "sha256": input_hashes[name],
                              "result_files": 0})

    point_rows = read_csv(paths["point_csv"])
    comparison_rows = read_csv(paths["comparison_csv"])
    phase_rows = read_csv(paths["phase_sidecar"])
    rowtile_rows = read_csv(paths["rowtile_sidecar"])
    require(len(point_rows) == 324 and len(producer["points"]) == 324,
            "point population must be 324")
    require(len(comparison_rows) == 270 and
            len(producer["comparisons"]) == 270,
            "comparison population must be 270")
    require(len(phase_rows) == 17280 and len(rowtile_rows) == 36,
            "phase or row-tile sidecar extent drift")
    csv_points = {point_key(row): row for row in point_rows}
    json_points = {point_key(row): row for row in producer["points"]}
    require(len(csv_points) == len(json_points) == 324 and
            set(csv_points) == set(json_points), "point key mismatch")

    with np.load(str(paths["task_sidecar"]), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required_fields = set(producer["m470_partition_window_task_sidecar"][
        "fields"])
    require(set(arrays) == required_fields and len(arrays) == 18,
            "task NPZ fields drift")
    lengths = {value.shape[0] for value in arrays.values()}
    require(lengths == {4250880}, "task NPZ extent drift")
    require(all(value.ndim == 1 and value.dtype.kind in "ui"
                for value in arrays.values()), "NPZ dtype is not typed integer")
    order = np.lexsort((arrays["partition"], arrays["row_start"],
                        arrays["row_tile_size"], arrays["operator"],
                        arrays["sample"]))
    require(np.array_equal(order, np.arange(order.size)),
            "task NPZ order drift")
    require(np.all(arrays["row_stop"] > arrays["row_start"]) and
            np.all(arrays["row_stop"] <= 3000), "invalid row extents")

    cycle_model = preflight["cycle_model"]
    budget = int(preflight["dse"]["sram_budget_bytes"])
    reserve = int(preflight["capacity_contract"]["fixed_reserve_bytes"])
    commit_cycles = 10 * int(preflight["schedule_contract"][
        "commit_cycles_per_sample"])
    int_fields = [
        "task_count", "task_passes", "empty_task_passes",
        "nonempty_task_passes", "source_sram_bytes", "source_dram_bytes",
        "weight_dram_bytes", "pwp_dram_bytes", "config_dram_bytes",
        "psum_sram_reads", "psum_sram_writes", "psum_sram_read_bytes",
        "psum_sram_write_bytes", "psum_dram_spill_bytes",
        "pwp_sram_read_bytes", "dma_commands",
        "generator_weight_read_bytes", "generator_signed_adds",
        "generator_cache_write_bytes", "generator_commands",
        "generator_cycles", "empty_gate_cycle_savings_vs_forced_payload",
        "empty_gate_skipped_config_bytes", "empty_gate_skipped_weight_bytes",
        "empty_gate_skipped_dma_commands", "cycles_without_commit",
        "overlap_upper_bound_cycles_without_commit", "cycles",
        "overlap_upper_bound_cycles", "commit_cycles",
        "operator_row_tile_commits", "block_half_commit_events",
        "committed_accumulator_vectors", "psum_reset_events", "dram_bytes",
    ]
    recomputed = {}
    reset_boundary_diagnostic = {}
    max_patterns = {}
    tiles = list(preflight["dse"]["row_tile_sizes"])
    for tile in tiles:
        tile_mask = arrays["row_tile_size"] == tile
        require(int(tile_mask.sum()) == 17280 * int(math.ceil(3000 / tile)),
                "task population mismatch for tile {}".format(tile))
        max_patterns[tile] = int(bitcounts_u32(
            arrays["used_center_mask_u32"][tile_mask]).max())
        for banks in preflight["dse"]["resident_block_banks"]:
            for bandwidth in preflight["dse"]["dram_bytes_per_cycle"]:
                bandwidth = to_bandwidth(bandwidth)
                modes = [("strong_zero", 0), ("stored_pwp", 0)] + [
                    ("lazy_pwp", int(k)) for k in
                    preflight["lazy_generator_contract"]["source_lanes_k"]]
                for mode, k in modes:
                    key = (tile, int(banks), str(bandwidth), mode, k)
                    independent, alt_reset = recompute_point(
                        arrays, tile_mask, tile, int(banks), bandwidth,
                        mode, k, cycle_model, commit_cycles)
                    producer_point = json_points[key]
                    compare_integer_fields(independent, producer_point,
                                           int_fields, str(key))
                    logical, rounded = capacity(
                        tile, int(banks), mode, max_patterns[tile], reserve)
                    cap = producer_point["capacity"]
                    require(logical == cap["logical_items"] and
                            rounded == cap["macro_rounded_items"],
                            "capacity item mismatch: {}".format(key))
                    logical_total = sum(logical.values())
                    rounded_total = sum(rounded.values())
                    require(logical_total == cap["logical_total_bytes"] and
                            rounded_total == cap["macro_rounded_total_bytes"],
                            "capacity total mismatch: {}".format(key))
                    fits_logical = logical_total <= budget
                    fits_rounded = rounded_total <= budget
                    require(fits_logical == producer_point["fits_240k_logical"] and
                            fits_rounded == producer_point[
                                "fits_240k_macro_rounded"] and
                            (fits_logical and fits_rounded) == producer_point[
                                "fits_both_240k_gates"],
                            "capacity gate mismatch: {}".format(key))
                    csv_point = csv_points[key]
                    compare_integer_fields(independent, csv_point,
                                           int_fields, "CSV " + str(key))
                    require(int(csv_point["logical_sram_bytes"]) ==
                            logical_total and
                            int(csv_point["macro_rounded_sram_bytes"]) ==
                            rounded_total, "flat capacity mismatch")
                    if mode == "lazy_pwp":
                        require(producer_point[
                            "generator_peak_onchip_weight_read_bytes_per_cycle"] ==
                            96 * k and producer_point[
                            "generator_independent_source_banks_or_ports"] == k and
                            producer_point["generator_signed_preadder_proxy"] ==
                            96 * (k - 1) and producer_point[
                            "generator_physical_product_slots"] == 96 * k and
                            producer_point["generator_same_resource_across_k"]
                            is False, "K resource ledger mismatch")
                    recomputed[key] = independent
                    if alt_reset is not None:
                        reset_boundary_diagnostic[key] = alt_reset

    # Independent anchors come from the typed task facts, not producer totals.
    stored_anchor_key = (3000, 8, "32", "stored_pwp", 0)
    zero_anchor_key = (3000, 8, "32", "strong_zero", 0)
    anchors = {
        "stored_pwp_actual": recomputed[stored_anchor_key]["cycles"],
        "stored_pwp_expected": int(execution["anchor_gate"]
                                    ["stored_pwp_row3000_banks8_bw32_cycles"]),
        "strong_zero_actual": recomputed[zero_anchor_key]["cycles"],
        "strong_zero_expected": int(execution["anchor_gate"]
                                     ["strong_zero_row3000_banks8_bw32_cycles"]),
    }
    require(anchors["stored_pwp_actual"] == anchors["stored_pwp_expected"] ==
            517041352 and anchors["strong_zero_actual"] ==
            anchors["strong_zero_expected"] == 742148386,
            "independent dual anchor failure")

    # Same-budget minima and comparison gates are selected independently.
    best_budget = []
    expected_comparisons = {}
    for banks in preflight["dse"]["resident_block_banks"]:
        for bandwidth_value in preflight["dse"]["dram_bytes_per_cycle"]:
            bandwidth = to_bandwidth(bandwidth_value)
            bw_key = str(bandwidth)
            zero_eligible = []
            stored_eligible = []
            lazy_eligible = []
            for tile in tiles:
                for mode, k in [("strong_zero", 0), ("stored_pwp", 0)]:
                    key = (tile, int(banks), bw_key, mode, k)
                    if json_points[key]["fits_both_240k_gates"]:
                        (zero_eligible if mode == "strong_zero" else
                         stored_eligible).append(key)
                for k in preflight["lazy_generator_contract"]["source_lanes_k"]:
                    key = (tile, int(banks), bw_key, "lazy_pwp", int(k))
                    if json_points[key]["fits_both_240k_gates"]:
                        lazy_eligible.append(key)
            require(zero_eligible, "missing same-budget zero baseline")
            best_zero_key = min(zero_eligible,
                                key=lambda key: recomputed[key]["cycles"])
            best_stored_key = (min(stored_eligible,
                                   key=lambda key: recomputed[key]["cycles"])
                               if stored_eligible else None)
            best_lazy_key = min(lazy_eligible,
                                key=lambda key: recomputed[key]["cycles"])
            best_budget.append({
                "resident_block_banks": int(banks),
                "bandwidth_bytes_per_cycle": bandwidth,
                "best_strong_zero_row_tile": best_zero_key[0],
                "best_strong_zero_cycles": recomputed[best_zero_key]["cycles"],
                "best_stored_pwp_exists": best_stored_key is not None,
                "best_stored_pwp_row_tile": (best_stored_key[0]
                                             if best_stored_key else None),
                "best_stored_pwp_cycles": (recomputed[best_stored_key]["cycles"]
                                           if best_stored_key else None),
                "best_lazy_row_tile": best_lazy_key[0],
                "best_lazy_k": best_lazy_key[4],
                "best_lazy_cycles": recomputed[best_lazy_key]["cycles"],
                "best_lazy_vs_best_zero": (
                    recomputed[best_zero_key]["cycles"] /
                    float(recomputed[best_lazy_key]["cycles"])),
                "best_lazy_vs_best_stored": (
                    recomputed[best_stored_key]["cycles"] /
                    float(recomputed[best_lazy_key]["cycles"])
                    if best_stored_key else None),
            })
            for tile in tiles:
                zero_key = (tile, int(banks), bw_key, "strong_zero", 0)
                stored_key = (tile, int(banks), bw_key, "stored_pwp", 0)
                for mode, k in [("stored_pwp", 0)] + [
                        ("lazy_pwp", int(v)) for v in
                        preflight["lazy_generator_contract"]["source_lanes_k"]]:
                    key = (tile, int(banks), bw_key, mode, k)
                    cand = recomputed[key]["cycles"]
                    best_zero_ratio = recomputed[best_zero_key]["cycles"] / cand
                    best_stored_ratio = (recomputed[best_stored_key]["cycles"] /
                                         cand if best_stored_key else None)
                    expected_comparisons[(tile, int(banks), bw_key,
                                          "stored_pwp" if mode == "stored_pwp"
                                          else "lazy_pwp_k{}".format(k))] = {
                        "candidate_cycles": cand,
                        "fair_strong_zero_cycles":
                            recomputed[zero_key]["cycles"],
                        "speedup_vs_fair_strong_zero":
                            recomputed[zero_key]["cycles"] / cand,
                        "speedup_vs_same_point_stored_pwp": (
                            1.0 if mode == "stored_pwp" else
                            recomputed[stored_key]["cycles"] / cand),
                        "best_same_budget_strong_zero_row_tile":
                            best_zero_key[0],
                        "best_same_budget_strong_zero_cycles":
                            recomputed[best_zero_key]["cycles"],
                        "speedup_vs_best_same_budget_strong_zero":
                            best_zero_ratio,
                        "best_same_budget_stored_pwp_exists":
                            best_stored_key is not None,
                        "best_same_budget_stored_pwp_row_tile": (
                            best_stored_key[0] if best_stored_key else None),
                        "best_same_budget_stored_pwp_cycles": (
                            recomputed[best_stored_key]["cycles"]
                            if best_stored_key else None),
                        "speedup_vs_best_same_budget_stored_pwp":
                            best_stored_ratio,
                        "fits_both_240k_gates":
                            json_points[key]["fits_both_240k_gates"],
                        "material_vs_strong_zero_1p15": (
                            json_points[key]["fits_both_240k_gates"] and
                            best_zero_ratio >= 1.15),
                        "material_vs_stored_1p10": (
                            mode == "lazy_pwp" and
                            json_points[key]["fits_both_240k_gates"] and
                            best_stored_key is not None and
                            best_stored_ratio >= 1.10),
                    }

    require(len(expected_comparisons) == len(comparison_rows) == 270,
            "comparison key population mismatch")
    dual_gate_points = []
    for row in comparison_rows:
        key = (int(row["row_tile"]), int(row["resident_block_banks"]),
               str(row["bandwidth_bytes_per_cycle"]), row["candidate"])
        require(key in expected_comparisons, "unknown comparison key")
        expected = expected_comparisons[key]
        for name in ["candidate_cycles", "fair_strong_zero_cycles",
                     "best_same_budget_strong_zero_row_tile",
                     "best_same_budget_strong_zero_cycles"]:
            require(int(row[name]) == int(expected[name]),
                    "comparison integer mismatch: {} {}".format(key, name))
        for name in ["speedup_vs_fair_strong_zero",
                     "speedup_vs_same_point_stored_pwp",
                     "speedup_vs_best_same_budget_strong_zero"]:
            require(math.isclose(float(row[name]), float(expected[name]),
                                 rel_tol=0, abs_tol=1e-12),
                    "comparison ratio mismatch: {} {}".format(key, name))
        nullable = ["best_same_budget_stored_pwp_row_tile",
                    "best_same_budget_stored_pwp_cycles",
                    "speedup_vs_best_same_budget_stored_pwp"]
        for name in nullable:
            value = row[name]
            if expected[name] is None:
                require(value == "", "comparison null mismatch")
            elif "speedup" in name:
                require(math.isclose(float(value), float(expected[name]),
                                     rel_tol=0, abs_tol=1e-12),
                        "comparison nullable ratio mismatch")
            else:
                require(int(value) == int(expected[name]),
                        "comparison nullable integer mismatch")
        for name in ["best_same_budget_stored_pwp_exists",
                     "fits_both_240k_gates",
                     "material_vs_strong_zero_1p15",
                     "material_vs_stored_1p10"]:
            require((row[name] == "True") == bool(expected[name]),
                    "comparison boolean mismatch: {} {}".format(key, name))
        require(row["performance_admitted"] == "False",
                "comparison illegally admits performance")
        if (expected["material_vs_strong_zero_1p15"] and
                expected["material_vs_stored_1p10"]):
            dual_gate_points.append(key)
    require(not dual_gate_points, "R6 unexpectedly passes both material gates")

    # Phase aggregate is independently summed from the producer-neutral CSV.
    phase_aggregate = {
        "source_rows": sum(int(row["source_rows"]) for row in phase_rows),
        "active_rows": sum(int(row["active_rows"]) for row in phase_rows),
        "pwp_rows": sum(int(row["m430_use_pwp_rows"]) for row in phase_rows),
        "fallback_rows": sum(int(row["m430_fallback_rows"])
                             for row in phase_rows),
        "correction_ops_per_block": sum(
            int(row["m430_correction_ops_per_block"]) for row in phase_rows),
        "bit_sparse_vector_ops_per_block": sum(
            int(row["strong_zero_bit_sparse_ops_per_block"])
            for row in phase_rows),
    }
    require(phase_aggregate == producer["identity_reproduction"]["aggregate"],
            "phase aggregate mismatch")
    require(producer["identity_reproduction"]["phase_field_mismatches"] == 0 and
            producer["identity_reproduction"]["m40_payload_reads"] == 0 and
            producer["m470_partition_window_task_sidecar"][
                "derivative_rereads"] == 0,
            "identity reproduction boundary drift")

    k_resources = [{
        "k": k,
        "peak_onchip_weight_read_bytes_per_cycle": 96 * k,
        "independent_source_banks_or_ports": k,
        "signed_preadder_proxy": 96 * (k - 1),
        "physical_product_slots": 96 * k,
        "same_resource_across_k": False,
    } for k in preflight["lazy_generator_contract"]["source_lanes_k"]]

    # The two findings constrain downstream claims but do not invalidate the
    # narrow, exactly reproduced CPU DSE result.
    findings = [
        {
            "severity": "P1",
            "title": "No same-budget candidate clears both material gates",
            "detail": "At 128 B/cycle the best 4-bank lazy point is row128/K8: "
                      "725,989,364 cycles, 1.036627x versus independently best "
                      "strong-zero and 1.201743x versus independently best stored-PWP. "
                      "The 8-bank axis has no dual-gate stored-PWP baseline. R6 is "
                      "therefore a CPU-DSE integrity PASS but a performance/RTL NO-GO."
        },
        {
            "severity": "P1",
            "title": "K8 is not a same-resource point and its physical cost is unclosed",
            "detail": "K8 assumes 768 B/cycle on-chip weight reads, eight independent "
                      "source banks/ports, 672 signed preadder proxies and 768 product "
                      "slots. These resources are disclosed but are outside the 240 KiB "
                      "capacity gate and have no RTL/DC/macro evidence."
        },
        {
            "severity": "P2",
            "title": "Strong-zero boundary wording is stale across recovery artifacts",
            "detail": "R4/R6 and the frozen anchors require one continuous overlap stream "
                      "inside each sample, while the preflight recovery prose says restart "
                      "per operator/row-tile/half. The implementation and anchors are "
                      "self-consistent, but the canonical schedule text should be repaired "
                      "before M470 reuses it."
        },
        {
            "severity": "P2",
            "title": "Old analyzer bodies are hash-addressed but not preserved beside r1-r5",
            "detail": "The marker-only failure directories correctly contain zero result "
                      "files, but the historical analyzer versions are represented only by "
                      "hashes in execution contracts. The claimed one-change recoveries "
                      "cannot be independently source-diffed from these directories alone."
        }
    ]
    score = 88
    review = {
        "schema": "m472_m468r3_m469_r6_independent_hammer_review_v1",
        "status": "PASS_CPU_DSE_INTEGRITY_NO_GO_PERFORMANCE_OR_RTL",
        "score": score,
        "severity_counts": {"P0": 0, "P1": 2, "P2": 2},
        "input_hashes": input_hashes,
        "producer_seal": {
            "manifest_entries": len(manifest_rows),
            "manifest_all_match": True,
            "outer_seal_match": True,
            "run_complete_match": True,
        },
        "failure_chain": failure_chain,
        "recomputation": {
            "points_recomputed": len(recomputed),
            "point_cycle_mismatches": 0,
            "point_traffic_mismatches": 0,
            "point_capacity_mismatches": 0,
            "comparison_rows_recomputed": len(expected_comparisons),
            "comparison_mismatches": 0,
            "task_npz_rows": 4250880,
            "phase_rows": 17280,
            "phase_aggregate": phase_aggregate,
            "anchors": anchors,
            "dual_gate_points": dual_gate_points,
        },
        "best_same_budget": best_budget,
        "k_resources": k_resources,
        "selected_128B_points": {
            "four_bank_best_zero": next(row for row in best_budget
                if row["resident_block_banks"] == 4 and
                row["bandwidth_bytes_per_cycle"] == 128),
            "eight_bank_best_zero": next(row for row in best_budget
                if row["resident_block_banks"] == 8 and
                row["bandwidth_bytes_per_cycle"] == 128),
        },
        "boundary_diagnostic": {
            "frozen_continuous_sample_zero_anchor": anchors[
                "strong_zero_actual"],
            "reset_each_operator_rowtile_zero_anchor":
                reset_boundary_diagnostic[zero_anchor_key],
            "performance_claim": False,
        },
        "findings": findings,
        "admission": {
            "sealed_r6_cpu_dse_integrity": "GO",
            "dual_anchor": "GO",
            "traffic_and_capacity_ledgers": "GO",
            "same_budget_fairness": "GO",
            "performance": "NO_GO_NO_DUAL_GATE_POINT",
            "rtl_nomination": "NO_GO",
            "k8_same_resource": "NO_GO",
            "physical_macro_or_synopsys": "NO_GO",
            "full_network_system_energy_headline": "NO_GO",
        },
        "allowed_claim": "R6 is an independently reproduced exact-arithmetic CPU DSE "
                         "over four frozen H67 ep35 bottleneck Conv3x3 operators; all "
                         "324 point cycles/traffic/capacities, 270 comparisons, both "
                         "frozen anchors and the producer dual seal close exactly.",
        "forbidden_claims": [
            "R6 admits a material performance point or RTL nomination.",
            "K1/K2/K4/K8 are same-resource alternatives.",
            "The 240 KiB gate includes K-dependent generator logic and ports.",
            "R6 is RTL-, Synopsys-, physical-macro-, energy-, full-network-, "
            "system- or DATE-headline evidence."
        ]
    }

    # Verify producer inputs did not move during this read-only hammer.
    for name, path in paths.items():
        require(sha256(path) == input_hashes[name],
                "producer input changed during hammer: " + name)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    review_json = args.output_dir / "m472_m468r3_m469_r6_independent_hammer_review.json"
    review_json.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    best_csv = args.output_dir / "m472_best_same_budget_recomputation.csv"
    with best_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_budget[0].keys()))
        writer.writeheader()
        writer.writerows(best_budget)
    review_md = args.output_dir / "README.md"
    review_md.write_text(
        "# M472 independent hammer of M468R3/M469 R6\n\n"
        "**Score: 88/100. Status: CPU-DSE integrity PASS; performance and RTL NO-GO.**\n\n"
        "The reviewer independently recomputed all 324 cycle/traffic/capacity "
        "points from the typed 4,250,880-row NPZ, all 270 comparison rows, both "
        "row3000/bank8/BW32 anchors, same-budget minima, K resource ledgers, "
        "r1-r5 marker-only failure chain, and the producer manifest plus outer seal.\n\n"
        "- Stored-PWP anchor: 517,041,352 cycles.\n"
        "- Strong-zero anchor: 742,148,386 cycles.\n"
        "- Best 4-bank/128 B-cycle lazy point: row128/K8, 725,989,364 "
        "cycles; 1.036627x versus best zero and 1.201743x versus best stored.\n"
        "- Best 8-bank/128 B-cycle lazy point: row64/K8, 736,160,660 "
        "cycles; 1.032859x versus best zero; no stored-PWP point passes both "
        "240 KiB capacity gates.\n"
        "- No point clears both the 1.15x zero and 1.10x stored gates.\n"
        "- K8 is explicitly not same-resource: 768 B/cycle, 8 source "
        "banks/ports, 672 preadder proxies and 768 product slots.\n\n"
        "See the JSON receipt for complete findings and claim boundaries.\n",
        encoding="utf-8")
    write_manifest(args.output_dir,
                   [review_json.name, best_csv.name, review_md.name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
