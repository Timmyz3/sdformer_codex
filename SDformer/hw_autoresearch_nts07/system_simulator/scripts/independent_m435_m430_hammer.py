#!/usr/bin/env python3
"""Independent M435 audit of M430 train catalog and M40 held-out replay.

This program intentionally does not import any M430, M423, M401, or M43
analyzer.  Packed convolution source words, catalog selection objectives,
static PWP codec data, cycle schedules, and traffic are reconstructed here.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


K = 16
PARTITIONS = 432
ROWS = 3000
OUTPUT_BLOCKS = 8
TRAIN_SAMPLES = 32
HELDOUT_SAMPLES = 10
POPCOUNT = np.asarray([bin(value).count("1") for value in range(1 << K)],
                      dtype=np.uint8)
OPTIONS = ("m338_q32", "dual_single_gain_q32", "dual_greedy_q32")
SAMPLED_PARTITIONS = tuple(range(0, PARTITIONS, 27))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs_hook(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(raw):
        raise RuntimeError("non-standard JSON token: " + raw)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def centers(catalog, operator, partition, count=32):
    return [int(value, 16) for value in
            catalog["operators"][operator]["partitions"][partition]
            ["nested_patterns"][:count]]


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    return 1 + sum(current != previous + 1
                   for previous, current in zip(ordered, ordered[1:]))


def unpack_conv_words(trace_dir, record):
    """Create the 3000 x 432 little-endian 16-bit convolution source words."""
    require(record["shape"] == [10, 1, 768, 15, 20],
            "source shape drift")
    packed_path = trace_dir / record["packed_file"]
    value_path = trace_dir / record["value_payload_file"]
    require(sha256(packed_path) == record["packed_file_sha256"],
            "packed payload SHA mismatch")
    require(sha256(value_path) == record["value_payload_sha256"],
            "value payload SHA mismatch")
    raw = np.frombuffer(packed_path.read_bytes(), dtype=np.uint8)
    plane_bytes = int(record["positive_plane_bytes"])
    require(raw.size == plane_bytes * 3, "packed plane extent drift")
    require(not np.any(raw[plane_bytes:2 * plane_bytes]),
            "frozen trace unexpectedly has negative source bits")
    source = np.unpackbits(raw[:plane_bytes], bitorder="little")
    source = source[:10 * 768 * 15 * 20].reshape(10, 768, 15, 20)
    padded = np.pad(source, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    taps = np.stack(
        [padded[:, :, ky:ky + 15, kx:kx + 20]
         for ky in range(3) for kx in range(3)], axis=2)
    feature_rows = np.ascontiguousarray(
        taps.transpose(0, 3, 4, 1, 2).reshape(ROWS, 768 * 9))
    packed_features = np.packbits(feature_rows, axis=1, bitorder="little")
    words = np.ascontiguousarray(packed_features).view("<u2")
    require(words.shape == (ROWS, PARTITIONS), "conv word extent drift")
    return words


def distance_matrix(values, catalog_centers):
    centers_np = np.asarray(catalog_centers, dtype=np.uint16)
    return POPCOUNT[np.bitwise_xor(centers_np[:, None], values[None, :])]


def analyze_hist(values, counts, catalog_centers, narrow_flags=None):
    values = np.asarray(values, dtype=np.uint16)
    counts = np.asarray(counts, dtype=np.int64)
    pops = POPCOUNT[values].astype(np.int16)
    matrix = distance_matrix(values, catalog_centers)
    best_id = matrix.argmin(axis=0)
    best_distance = matrix[best_id, np.arange(values.size)].astype(np.int16)
    nonzero = values != 0
    eligible = pops >= 2
    use_pwp = nonzero & (1 + best_distance < pops)
    fallback = nonzero & (~use_pwp)
    q16_exact = matrix[:16].min(axis=0) == 0
    correction = np.where(use_pwp, best_distance, pops).astype(np.int64)
    selected = np.asarray(catalog_centers, dtype=np.uint16)[best_id]
    plus = values & np.bitwise_not(selected)
    minus = selected & np.bitwise_not(values)
    reconstructed = (selected | plus) & np.bitwise_not(minus)
    reconstruction_mismatches = int(np.count_nonzero(
        reconstructed[use_pwp] != values[use_pwp]))
    used_ids = set(int(index) for index in np.unique(
        best_id[use_pwp & (counts > 0)]))
    result = {
        "source_rows": int(counts.sum()),
        "zero_rows": int(counts[~nonzero].sum()),
        "active_rows": int(counts[nonzero].sum()),
        "eligible_rows": int(counts[eligible].sum()),
        "pop1_rows": int(counts[pops == 1].sum()),
        "pwp_rows": int(counts[use_pwp].sum()),
        "exact_pwp_rows": int(counts[use_pwp & (best_distance == 0)].sum()),
        "fallback_rows": int(counts[fallback].sum()),
        "correction_ops_per_block": int(np.dot(counts, correction)),
        "bit_sparse_vector_ops_per_block":
            int(np.dot(counts, pops.astype(np.int64))),
        "candidate_vector_ops_per_block": int(np.dot(
            counts, np.where(use_pwp, 1 + best_distance, pops)
            .astype(np.int64))),
        "q32_early_extra_prefix_tasks":
            int(counts[eligible & (~q16_exact)].sum()),
        "used_pwp_patterns": len(used_ids),
        "used_center_runs": count_runs(used_ids),
        "reconstruction_mismatches": reconstruction_mismatches,
    }
    result["q32_early_matcher_cycles"] = (
        result["source_rows"] + result["q32_early_extra_prefix_tasks"] + 2)
    result["q32_reference_matcher_cycles"] = (
        result["source_rows"] + result["eligible_rows"] + 2)
    result["q32_early_saved_cycles"] = (
        result["eligible_rows"] - result["q32_early_extra_prefix_tasks"])
    if narrow_flags is not None:
        weights = counts[use_pwp]
        ids = best_id[use_pwp]
        result["narrow_block_descriptors_tile0"] = int(np.dot(
            weights, narrow_flags[ids, :4].sum(axis=1)))
        result["narrow_block_descriptors_tile1"] = int(np.dot(
            weights, narrow_flags[ids, 4:].sum(axis=1)))
    require(result["source_rows"] ==
            result["zero_rows"] + result["active_rows"],
            "source population conservation failure")
    require(result["active_rows"] ==
            result["pwp_rows"] + result["fallback_rows"],
            "active population conservation failure")
    require(reconstruction_mismatches == 0,
            "bit residual reconstruction mismatch")
    return result


def analyze_words(words, catalog_centers, narrow_flags=None):
    values, counts = np.unique(words, return_counts=True)
    return analyze_hist(values, counts, catalog_centers, narrow_flags)


def dual_objective(values, counts, catalog_centers):
    pops = POPCOUNT[values].astype(np.int16)
    distance = distance_matrix(values, catalog_centers).min(axis=0).astype(
        np.int16)
    units = np.where(1 + distance < pops, 1 + distance, pops)
    return int(np.dot(counts.astype(np.int64), units.astype(np.int64)))


def single_gain_tail(values, counts, q16, pool):
    base_distance = distance_matrix(values, q16).min(axis=0)
    pops = POPCOUNT[values].astype(np.int16)
    base_units = np.where(1 + base_distance < pops,
                          1 + base_distance, pops)
    pool_distance = distance_matrix(values, pool)
    scored = []
    for pool_id, center in enumerate(pool):
        candidate_distance = np.minimum(base_distance, pool_distance[pool_id])
        units = np.where(1 + candidate_distance < pops,
                         1 + candidate_distance, pops)
        gain = int(np.dot(counts.astype(np.int64),
                          (base_units - units).astype(np.int64)))
        exact = int(counts[values == center].sum())
        scored.append((-gain, -exact, pool_id, int(center)))
    scored.sort()
    return [row[-1] for row in scored[:16]]


def greedy_tail(values, counts, q16, pool):
    pool_distance = distance_matrix(values, pool)
    best_distance = distance_matrix(values, q16).min(axis=0)
    pops = POPCOUNT[values].astype(np.int16)
    selected = []
    selected_ids = set()
    for _ in range(16):
        best_key = None
        best_pool_id = None
        best_next = None
        for pool_id, center in enumerate(pool):
            if pool_id in selected_ids:
                continue
            candidate_distance = np.minimum(best_distance,
                                            pool_distance[pool_id])
            units = np.where(1 + candidate_distance < pops,
                             1 + candidate_distance, pops)
            objective = int(np.dot(counts.astype(np.int64),
                                   units.astype(np.int64)))
            key = (objective, pool_id, int(center))
            if best_key is None or key < best_key:
                best_key = key
                best_pool_id = pool_id
                best_next = candidate_distance
        require(best_pool_id is not None, "greedy pool exhausted")
        selected_ids.add(best_pool_id)
        selected.append(int(pool[best_pool_id]))
        best_distance = best_next
    return selected


def phase_front_cycles(phase, model, command_setup):
    config_data = int(math.ceil(
        model["elastic_config_bytes"] / model["dram_bytes_per_cycle"]))
    return config_data + command_setup + phase["q32_early_matcher_cycles"] + 1


def phase_tile_dma(phase, model, command_setup):
    tile_bytes = (model["weight_bytes_per_tile"] +
                  phase["used_pwp_patterns"] *
                  model["elastic_center_stride_bytes"])
    require(model["elastic_config_bytes"] + tile_bytes <=
            model["tile_slot_bytes"], "tile slot overflow")
    require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
            "tile DMA alignment failure")
    data = tile_bytes // model["dram_bytes_per_cycle"]
    commands = (1 + phase["used_center_runs"]) * command_setup
    return data + commands


def dual_sample(phases, model, command_setup, latency, timestamps=False):
    now = 0
    component = Counter()
    rows = []
    max_slot = 0
    for phase_index, phase in enumerate(phases):
        phase_start = now
        config_data = int(math.ceil(
            model["elastic_config_bytes"] / model["dram_bytes_per_cycle"]))
        now += phase_front_cycles(phase, model, command_setup)
        component.update({"config_data": config_data,
                          "config_command": command_setup,
                          "matcher": phase["q32_early_matcher_cycles"],
                          "bitmap_seal": 1})
        if phase["active_rows"] == 0:
            now += model["tail_cycles"]
            component["tail"] += model["tail_cycles"]
            continue
        tile_dma = phase_tile_dma(phase, model, command_setup)
        tile_bytes = (model["weight_bytes_per_tile"] +
                      phase["used_pwp_patterns"] *
                      model["elastic_center_stride_bytes"])
        max_slot = max(max_slot, model["elastic_config_bytes"] + tile_bytes)
        # Legal dual co-read: one PWP issue and d correction issues for each
        # of four output blocks per tile.  There is no narrow subtraction and
        # no replacement of persistent old_psum.
        work = 4 * (phase["correction_ops_per_block"] + phase["pwp_rows"])
        require(work >= phase["active_rows"], "dual service underflow")
        replay = work + latency
        now += tile_dma
        tile0_start = now
        tile0_end = tile0_start + replay
        tile1_dma_end = tile0_start + tile_dma
        tile1_start = max(tile0_end, tile1_dma_end)
        exposed = max(0, tile1_dma_end - tile0_end)
        now = tile1_start + replay + model["tail_cycles"]
        component["tile0_dma_data"] += (
            tile_bytes // model["dram_bytes_per_cycle"])
        component["tile0_dma_commands"] += (
            (1 + phase["used_center_runs"]) * command_setup)
        component["tile1_dma_exposed"] += exposed
        component["replay0"] += replay
        component["replay1"] += replay
        component["active_compute"] += 2 * work
        component["descriptor_sram_startup"] += 2 * latency
        component["tail"] += model["tail_cycles"]
        component["pwp_dram_physical_bytes"] += (
            phase["used_pwp_patterns"] *
            model["elastic_center_stride_bytes"] * 2)
        component["weight_dram_bytes"] += model["weight_bytes_per_tile"] * 2
        component["tile_dma_commands"] += 2 * (1 + phase["used_center_runs"])
        component["descriptor_reads_responses_bundles"] += (
            phase["active_rows"] * 2)
        component["pwp_output_block_issues"] += (
            phase["pwp_rows"] * OUTPUT_BLOCKS)
        component["pwp_logical_onchip_read_bytes"] += (
            phase["pwp_rows"] * OUTPUT_BLOCKS * 144)
        component["pwp_padded_signal_bytes"] += (
            phase["pwp_rows"] * OUTPUT_BLOCKS * 160)
        component["correction_output_block_issues"] += (
            phase["correction_ops_per_block"] * OUTPUT_BLOCKS)
        component["correction_onchip_read_bytes"] += (
            phase["correction_ops_per_block"] * OUTPUT_BLOCKS * 96)
        if timestamps:
            rows.append({
                "phase_index": phase_index,
                "phase_start": phase_start,
                "tile0_replay_start": tile0_start,
                "tile0_replay_end": tile0_end,
                "tile1_dma_end": tile1_dma_end,
                "exposed_tile1_dma": exposed,
                "tile1_replay_start": tile1_start,
                "phase_end": now,
            })
    now += model["commit_cycles_per_sample"]
    component["commit"] += model["commit_cycles_per_sample"]
    return int(now), component, rows, max_slot


def serial_sample(phases, model, command_setup, latency):
    now = 0
    for phase in phases:
        now += phase_front_cycles(phase, model, command_setup)
        if phase["active_rows"] == 0:
            now += model["tail_cycles"]
            continue
        tile_dma = phase_tile_dma(phase, model, command_setup)
        base = 4 * phase["correction_ops_per_block"] + 8 * phase["pwp_rows"]
        work0 = base - phase["narrow_block_descriptors_tile0"]
        work1 = base - phase["narrow_block_descriptors_tile1"]
        require(work0 >= phase["active_rows"] and
                work1 >= phase["active_rows"], "serial service underflow")
        now += tile_dma
        tile0_end = now + work0 + latency
        tile1_dma_end = now + tile_dma
        now = max(tile0_end, tile1_dma_end) + work1 + latency + \
            model["tail_cycles"]
    return int(now + model["commit_cycles_per_sample"])


def baseline_sample(phases, model, command_setup):
    preprocess = max(
        model["rows_per_phase"] + model["popcount_filter_pipeline_cycles"],
        model["weight_phase_bytes"] // model["dram_bytes_per_cycle"] +
        command_setup)
    now = preprocess
    for index, phase in enumerate(phases):
        compute = phase["bit_sparse_vector_ops_per_block"] * OUTPUT_BLOCKS
        next_preprocess = preprocess if index + 1 < len(phases) else 0
        now += max(compute, next_preprocess) + model["tail_cycles"]
    return int(now + model["commit_cycles_per_sample"])


def training_partition_cycles(words_by_sample, catalog_centers, model,
                              command_setup, latency):
    total = 0
    objective = 0
    phases = []
    for sample_words in words_by_sample:
        phase = analyze_words(sample_words, catalog_centers)
        phases.append(phase)
        objective += phase["candidate_vector_ops_per_block"]
        phase_cycles = phase_front_cycles(phase, model, command_setup)
        if phase["active_rows"] == 0:
            phase_cycles += model["tail_cycles"]
        else:
            tile_dma = phase_tile_dma(phase, model, command_setup)
            work = 4 * (phase["correction_ops_per_block"] +
                        phase["pwp_rows"])
            replay = work + latency
            phase_cycles += (tile_dma + max(replay, tile_dma) + replay +
                             model["tail_cycles"])
        total += phase_cycles
    return int(total), int(objective), phases


def build_static_codec(catalog, weight_paths, compare_rows=None):
    flags = []
    global_digest = hashlib.sha256()
    block_count = 0
    lanes = 0
    narrow_count = 0
    signed12_violations = 0
    wide_mismatches = 0
    narrow_mismatches = 0
    padding_nonzero = 0
    minimum_all = 1 << 30
    maximum_all = -(1 << 30)
    row_mismatches = 0
    compare_index = 0
    for operator in range(4):
        weights = np.fromfile(weight_paths[operator], dtype=np.int8)
        require(weights.size == 6912 * 768, "weight extent drift")
        weights = weights.reshape(6912, 768).astype(np.int16)
        op_flags = []
        for partition in range(PARTITIONS):
            center_words = centers(catalog, operator, partition)
            bits = np.asarray(
                [[(center >> bit) & 1 for bit in range(16)]
                 for center in center_words], dtype=np.int16)
            products = bits @ weights[partition * 16:(partition + 1) * 16]
            partition_flags = np.zeros((32, OUTPUT_BLOCKS), dtype=np.bool_)
            for center_id in range(32):
                for output_block in range(OUTPUT_BLOCKS):
                    vector = products[center_id,
                                      output_block * 96:(output_block + 1) * 96]
                    minimum = int(vector.min())
                    maximum = int(vector.max())
                    minimum_all = min(minimum_all, minimum)
                    maximum_all = max(maximum_all, maximum)
                    signed12_violations += int(np.count_nonzero(
                        (vector < -2048) | (vector > 2047)))
                    raw12 = vector.astype(np.int32) & 0xfff
                    low8 = (raw12 & 0xff).astype(np.uint8)
                    high_nibbles = ((raw12 >> 8) & 0xf).astype(np.uint8)
                    high4 = (high_nibbles[0::2] |
                             (high_nibbles[1::2] << 4)).astype(np.uint8)
                    padding = np.zeros(16, dtype=np.uint8)
                    unpack_high = np.empty(96, dtype=np.int32)
                    unpack_high[0::2] = high4 & 0xf
                    unpack_high[1::2] = high4 >> 4
                    wide_raw = (unpack_high << 8) | low8.astype(np.int32)
                    wide = np.where(wide_raw >= 2048,
                                    wide_raw - 4096, wide_raw)
                    wide_mismatches += int(np.count_nonzero(
                        wide != vector.astype(np.int32)))
                    narrow = minimum >= -128 and maximum <= 127
                    narrow_recon = np.where(low8.astype(np.int32) >= 128,
                                            low8.astype(np.int32) - 256,
                                            low8.astype(np.int32))
                    if narrow:
                        narrow_mismatches += int(np.count_nonzero(
                            narrow_recon != vector.astype(np.int32)))
                    padding_nonzero += int(np.count_nonzero(padding))
                    partition_flags[center_id, output_block] = narrow
                    narrow_count += int(narrow)
                    block_count += 1
                    lanes += 96
                    header = struct.pack("<HHBBH", operator, partition,
                                         center_id, output_block,
                                         center_words[center_id])
                    block_sha = hashlib.sha256(
                        header + low8.tobytes() + high4.tobytes() +
                        padding.tobytes() + bytes([int(narrow)])).hexdigest()
                    global_digest.update(bytes.fromhex(block_sha))
                    if compare_rows is not None:
                        row = compare_rows[compare_index]
                        observed = (operator, partition, center_id,
                                    output_block, minimum, maximum,
                                    int(narrow), block_sha)
                        expected = (int(row["operator"]),
                                    int(row["partition"]),
                                    int(row["center_id"]),
                                    int(row["output_block"]),
                                    int(row["minimum"]),
                                    int(row["maximum"]),
                                    int(row["narrow"]), row["codec_sha256"])
                        row_mismatches += int(observed != expected)
                        compare_index += 1
            op_flags.append(partition_flags)
        flags.append(op_flags)
    if compare_rows is not None:
        require(compare_index == len(compare_rows), "codec CSV extent drift")
    summary = {
        "blocks": block_count,
        "lanes": lanes,
        "narrow_blocks": narrow_count,
        "global_minimum": minimum_all,
        "global_maximum": maximum_all,
        "maximum_absolute": max(abs(minimum_all), abs(maximum_all)),
        "signed12_violations": signed12_violations,
        "wide_reconstruction_mismatches": wide_mismatches,
        "narrow_reconstruction_mismatches": narrow_mismatches,
        "nonzero_padding_bytes": padding_nonzero,
        "codec_global_sha256": global_digest.hexdigest(),
        "upstream_row_mismatches": row_mismatches,
    }
    return flags, summary


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_phases(phases):
    keys = (
        "source_rows", "zero_rows", "active_rows", "eligible_rows",
        "pop1_rows", "pwp_rows", "fallback_rows",
        "correction_ops_per_block", "bit_sparse_vector_ops_per_block",
        "candidate_vector_ops_per_block", "q32_early_extra_prefix_tasks",
        "q32_early_matcher_cycles", "q32_reference_matcher_cycles",
        "q32_early_saved_cycles", "used_pwp_patterns", "used_center_runs")
    return {key: sum(int(phase.get(key, 0)) for phase in phases)
            for key in keys}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha256(output_dir / name)}  {name}\n" for name in sorted(names)),
        encoding="utf-8")
    outer = output_dir / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha256(manifest), sha256(outer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing output overwrite")
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m435_m430_independent_hammer_contract_v1",
            "contract schema drift")
    hw_root = args.contract.resolve().parents[1]
    paths = {}
    for name, identity in contract["inputs"].items():
        path = hw_root / identity["path"]
        require(path.is_file(), "missing input " + name)
        require(sha256(path) == identity["sha256"],
                "input SHA drift: " + name)
        paths[name] = path
    source_sha = sha256(Path(__file__).resolve())
    docs359_before = sha256(paths["docs359"])

    # Verify both upstream double seals before reading payload evidence.
    seal_mismatches = 0
    for prefix in ("m430_train", "m430_heldout"):
        manifest_path = paths[prefix + "_manifest"]
        directory = manifest_path.parent
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            seal_mismatches += int(sha256(directory / name) != expected)
        expected, name = paths[prefix + "_seal"].read_text(
            encoding="utf-8").strip().split("  ", 1)
        seal_mismatches += int(name != "SHA256SUMS" or
                               sha256(manifest_path) != expected)
    require(seal_mismatches == 0, "upstream double seal mismatch")

    catalog = strict_json(paths["m430_catalog"])
    m77 = strict_json(paths["m77_catalog"])
    m338 = strict_json(paths["m338_catalog"])
    m423 = strict_json(paths["m423_catalog"])
    m73 = strict_json(paths["m73_manifest"])
    m40 = strict_json(paths["m40_manifest"])
    heldout_contract = strict_json(paths["m430_heldout_contract"])
    train_contract = strict_json(paths["m430_train_contract"])
    upstream_result = strict_json(paths["m430_heldout_result"])
    upstream_train = strict_json(paths["m430_train_audit"])
    semantic = strict_json(paths["m427r3_review"])
    model = heldout_contract["cycle_model"]
    command_setup = heldout_contract["decision_rule"][
        "dma_command_setup_cycles"]
    latency = heldout_contract["decision_rule"][
        "descriptor_sram_latency_cycles"]
    for key, value in model.items():
        if key in train_contract["cycle_model"]:
            require(train_contract["cycle_model"][key] == value,
                    "cycle model drift: " + key)
    require(train_contract["cycle_model"]["dma_command_setup_cycles"] ==
            command_setup and
            train_contract["cycle_model"]
            ["descriptor_sram_latency_cycles"] == latency,
            "cycle decision parameter drift")
    require(semantic["verdict"]["seed_fusion_rtl_as_specified"] ==
            "NO_GO" and
            semantic["verdict"]["m426_seed_fusion_1p695794x"] ==
            "REVOKED_DO_NOT_CITE",
            "M427r3 withdrawal red line drift")

    operators = m40["cohort"]["operators"]
    require(operators == m73["cohort"]["operators"] ==
            catalog["geometry"]["operators"], "operator identity drift")
    train_keys = set(record["sample_key"] for record in m73["records"])
    heldout_keys = set(record["sample_key"] for record in m40["records"])
    key_overlap = sorted(train_keys & heldout_keys)
    require(len(train_keys) == 32 and len(heldout_keys) == 10 and
            not key_overlap, "train/heldout key overlap")

    train_manifest_mtime = paths["m430_train_seal"].stat().st_mtime_ns
    marker_mtime = paths["m430_one_shot_marker"].stat().st_mtime_ns
    heldout_seal_mtime = paths["m430_heldout_seal"].stat().st_mtime_ns
    mtime_order = train_manifest_mtime < marker_mtime < heldout_seal_mtime
    require(mtime_order, "seal/marker/result mtime order drift")

    # Full catalog identity and membership checks.
    q16_mismatches = 0
    tail_pool_mismatches = 0
    duplicate_mismatches = 0
    selected_count = Counter()
    for op in range(4):
        for partition in range(PARTITIONS):
            current = centers(catalog, op, partition)
            q16 = [int(item["value_hex"], 16) for item in
                   m77["operators"][op]["partitions"][partition]["patterns"]]
            parent = centers(m338, op, partition, 128)
            q16_mismatches += int(current[:16] != q16 or parent[:16] != q16)
            pool = set(parent[16:128])
            tail_pool_mismatches += sum(value not in pool
                                        for value in current[16:])
            duplicate_mismatches += int(len(set(current)) != 32)
            selected_count[catalog["operators"][op]["partitions"][partition]
                           ["selected_train_option"]] += 1
    require(q16_mismatches == tail_pool_mismatches ==
            duplicate_mismatches == 0, "catalog nesting/membership failure")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    codec_rows = read_csv(paths["m430_static_codec"])
    weight_paths = [paths[f"weight_o{op}"] for op in range(4)]
    m430_flags, m430_codec = build_static_codec(
        catalog, weight_paths, codec_rows)
    m338_flags, m338_codec = build_static_codec(m338, weight_paths)
    require(m430_codec["signed12_violations"] == 0 and
            m430_codec["wide_reconstruction_mismatches"] == 0 and
            m430_codec["narrow_reconstruction_mismatches"] == 0 and
            m430_codec["nonzero_padding_bytes"] == 0 and
            m430_codec["upstream_row_mismatches"] == 0,
            "M430 codec reconstruction failure")
    for key in ("blocks", "lanes", "global_minimum", "global_maximum",
                "maximum_absolute", "signed12_violations",
                "wide_reconstruction_mismatches",
                "narrow_reconstruction_mismatches",
                "nonzero_padding_bytes", "codec_global_sha256"):
        require(m430_codec[key] == upstream_result["static_codec"][key],
                "codec summary mismatch: " + key)
    print("[M435] static codec reconstructed", flush=True)

    # Reconstruct all M73 words.  Validate selected catalog objective/cycles on
    # all 1728 partitions and regenerate all candidates on 64 stratified
    # partitions (16 per operator).
    train_dir = paths["m73_manifest"].parent
    option_rows = {(int(row["operator"]), int(row["partition"])): row
                   for row in read_csv(paths["m430_train_options"])}
    train_selected_cycles = 0
    train_strong_phases = [[] for _ in range(TRAIN_SAMPLES)]
    full_objective_mismatches = 0
    full_selected_cycle_mismatches = 0
    sampled_candidate_pattern_mismatches = 0
    sampled_candidate_objective_mismatches = 0
    sampled_candidate_cycle_mismatches = 0
    sampled_chosen_mismatches = 0
    train_payload_files = 0
    train_payload_bytes = 0
    for op in range(4):
        records = sorted((record for record in m73["records"]
                          if record["operator"] == operators[op]),
                         key=lambda record: int(record["sample_id"]))
        require([int(record["sample_id"]) for record in records] ==
                list(range(TRAIN_SAMPLES)), "train sample order drift")
        words_by_sample = []
        for record in records:
            words_by_sample.append(unpack_conv_words(train_dir, record))
            train_payload_files += 2
            train_payload_bytes += ((train_dir / record["packed_file"]).stat().st_size +
                                    (train_dir / record["value_payload_file"]).stat().st_size)
        words_by_sample = np.stack(words_by_sample, axis=0)
        for partition in range(PARTITIONS):
            current = centers(catalog, op, partition)
            selected_cycles, selected_objective, phases = \
                training_partition_cycles(words_by_sample[:, :, partition],
                                          current, model, command_setup,
                                          latency)
            train_selected_cycles += selected_cycles
            for sample, phase in enumerate(phases):
                train_strong_phases[sample].append(phase)
            entry = catalog["operators"][op]["partitions"][partition]
            chosen = entry["selected_train_option"]
            full_objective_mismatches += int(
                selected_objective !=
                int(entry["dual_issue_objective_by_option"][chosen]))
            full_selected_cycle_mismatches += int(
                selected_cycles !=
                int(entry["train_phase_cycles_by_option"][chosen]))
            csv_row = option_rows[(op, partition)]
            full_objective_mismatches += int(
                selected_objective != int(csv_row[chosen + "_objective"]))
            full_selected_cycle_mismatches += int(
                selected_cycles != int(csv_row[chosen + "_cycles"]))
            if partition in SAMPLED_PARTITIONS:
                flattened = words_by_sample[:, :, partition].reshape(-1)
                values, counts = np.unique(flattened, return_counts=True)
                parent = centers(m338, op, partition, 128)
                q16 = parent[:16]
                pool = parent[16:128]
                candidates = {
                    "m338_q32": q16 + pool[:16],
                    "dual_single_gain_q32": q16 +
                        single_gain_tail(values, counts, q16, pool),
                    "dual_greedy_q32": q16 +
                        greedy_tail(values, counts, q16, pool),
                }
                evaluation = {}
                for option in OPTIONS:
                    cycles_value, objective_value, _ = \
                        training_partition_cycles(
                            words_by_sample[:, :, partition],
                            candidates[option], model, command_setup, latency)
                    evaluation[option] = (cycles_value, objective_value)
                    sampled_candidate_objective_mismatches += int(
                        objective_value !=
                        int(entry["dual_issue_objective_by_option"][option]))
                    sampled_candidate_cycle_mismatches += int(
                        cycles_value !=
                        int(entry["train_phase_cycles_by_option"][option]))
                sampled_candidate_pattern_mismatches += int(
                    candidates[chosen] != current)
                independently_chosen = min(
                    OPTIONS, key=lambda option: (
                        evaluation[option][0], evaluation[option][1],
                        OPTIONS.index(option)))
                sampled_chosen_mismatches += int(independently_chosen != chosen)
        print(f"[M435] train operator {op + 1}/4 reconstructed", flush=True)
    train_selected_cycles += TRAIN_SAMPLES * model["commit_cycles_per_sample"]
    train_strong_cycles = sum(baseline_sample(
        phases, model, command_setup) for phases in train_strong_phases)
    require(train_selected_cycles == upstream_train["train_only_observation"]
            ["hybrid_selected_cycles"], "full train selected cycles mismatch")
    require(train_strong_cycles == upstream_train["train_only_observation"]
            ["strong_zero_elided_baseline_cycles"],
            "full train strong baseline mismatch")
    require(full_objective_mismatches ==
            full_selected_cycle_mismatches ==
            sampled_candidate_pattern_mismatches ==
            sampled_candidate_objective_mismatches ==
            sampled_candidate_cycle_mismatches ==
            sampled_chosen_mismatches == 0,
            "training selection/objective audit failure")

    # Reconstruct all M40 words and all three relevant catalog schedules.
    heldout_dir = paths["m40_manifest"].parent
    phase_csv = read_csv(paths["m430_phase_rows"])
    timestamp_csv = read_csv(paths["m430_timestamps"])
    require(len(phase_csv) == len(timestamp_csv) == 17280,
            "upstream heldout CSV extent drift")
    phase_csv_index = {(int(row["sample"]), int(row["operator"]),
                        int(row["partition"])): row for row in phase_csv}
    m430_by_sample = [[] for _ in range(HELDOUT_SAMPLES)]
    m423_by_sample = [[] for _ in range(HELDOUT_SAMPLES)]
    m338_by_sample = [[] for _ in range(HELDOUT_SAMPLES)]
    phase_row_mismatches = 0
    heldout_payload_files = 0
    heldout_payload_bytes = 0
    heldout_reconstruction_mismatches = 0
    records = sorted(m40["records"], key=lambda row: (
        int(row["sample_id"]), operators.index(row["operator"])))
    for record_index, record in enumerate(records):
        sample = int(record["sample_id"])
        op = operators.index(record["operator"])
        words = unpack_conv_words(heldout_dir, record)
        heldout_payload_files += 2
        heldout_payload_bytes += (
            (heldout_dir / record["packed_file"]).stat().st_size +
            (heldout_dir / record["value_payload_file"]).stat().st_size)
        for partition in range(PARTITIONS):
            values, counts = np.unique(words[:, partition], return_counts=True)
            current = analyze_hist(values, counts,
                                   centers(catalog, op, partition),
                                   m430_flags[op][partition])
            old423 = analyze_hist(values, counts,
                                  centers(m423, op, partition))
            old338 = analyze_hist(values, counts,
                                  centers(m338, op, partition),
                                  m338_flags[op][partition])
            m430_by_sample[sample].append(current)
            m423_by_sample[sample].append(old423)
            m338_by_sample[sample].append(old338)
            heldout_reconstruction_mismatches += (
                current["reconstruction_mismatches"] +
                old423["reconstruction_mismatches"] +
                old338["reconstruction_mismatches"])
            row = phase_csv_index[(sample, op, partition)]
            observed = (
                current["active_rows"], current["eligible_rows"],
                current["pwp_rows"], current["exact_pwp_rows"],
                current["fallback_rows"],
                current["correction_ops_per_block"],
                current["used_pwp_patterns"], current["used_center_runs"],
                current["q32_early_matcher_cycles"])
            expected = tuple(int(row[key]) for key in (
                "active_rows", "eligible_rows", "pwp_rows",
                "exact_pwp_rows", "fallback_rows",
                "correction_ops_per_block", "used_pwp_patterns",
                "used_center_runs", "early_matcher"))
            phase_row_mismatches += int(observed != expected)
        print(f"[M435] heldout record {record_index + 1}/40 reconstructed",
              flush=True)

    m430_cycles = 0
    m423_dual_cycles = 0
    m338_dual_cycles = 0
    m401_serial_cycles = 0
    strong_zero_cycles = 0
    m430_components = Counter()
    independent_timestamps = []
    maximum_slot = 0
    for sample in range(HELDOUT_SAMPLES):
        cycles_value, components_value, timestamps_value, slot = dual_sample(
            m430_by_sample[sample], model, command_setup, latency, True)
        m430_cycles += cycles_value
        m430_components.update(components_value)
        maximum_slot = max(maximum_slot, slot)
        for row in timestamps_value:
            independent_timestamps.append((sample, row))
        m423_dual_cycles += dual_sample(
            m423_by_sample[sample], model, command_setup, latency)[0]
        m338_dual_cycles += dual_sample(
            m338_by_sample[sample], model, command_setup, latency)[0]
        m401_serial_cycles += serial_sample(
            m338_by_sample[sample], model, command_setup, latency)
        strong_zero_cycles += baseline_sample(
            m338_by_sample[sample], model, command_setup)

    timestamp_mismatches = 0
    for (sample, observed), expected in zip(independent_timestamps,
                                            timestamp_csv):
        observed_tuple = (sample, observed["phase_index"],
                          observed["phase_start"],
                          observed["tile0_replay_start"],
                          observed["tile0_replay_end"],
                          observed["tile1_dma_end"],
                          observed["exposed_tile1_dma"],
                          observed["tile1_replay_start"],
                          observed["phase_end"])
        expected_tuple = tuple(int(expected[key]) for key in (
            "sample", "phase_index", "phase_start", "tile0_replay_start",
            "tile0_replay_end", "tile1_dma_end", "exposed_tile1_dma",
            "tile1_replay_start", "phase_end"))
        timestamp_mismatches += int(observed_tuple != expected_tuple)

    expected = contract["expected"]
    cycle_mismatches = sum((
        m430_cycles != expected["m430_dual_cycles"],
        strong_zero_cycles != expected["strong_zero_cycles"],
        m423_dual_cycles != expected["m423_dual_cycles"],
        m401_serial_cycles != expected["m401_serial_cycles"],
        m338_dual_cycles != upstream_result["comparisons"]
        ["m338_catalog_dual_cycles"],
    ))
    traffic_mismatches = int(
        m430_components["pwp_dram_physical_bytes"] !=
        expected["pwp_dram_physical_bytes"])
    component_mismatches = 0
    for key, value in upstream_result["component_ledger"].items():
        component_mismatches += int(m430_components[key] != value)
    require(cycle_mismatches == traffic_mismatches ==
            component_mismatches == phase_row_mismatches ==
            timestamp_mismatches == heldout_reconstruction_mismatches == 0,
            "heldout independent reconstruction failure")

    docs359_after = sha256(paths["docs359"])
    require(docs359_before == docs359_after ==
            contract["inputs"]["docs359"]["sha256"],
            "docs359 changed during audit")
    require(source_sha == sha256(Path(__file__).resolve()),
            "auditor changed during execution")

    result = {
        "schema": "m435_m430_independent_recomputation_v1",
        "status": "PASS_INDEPENDENT_FULL_HELDOUT_AND_FULL_SELECTED_TRAIN",
        "identity": {
            "contract": {"path": str(args.contract.resolve().relative_to(hw_root)),
                         "sha256": sha256(args.contract)},
            "auditor": {
                "path": str(Path(__file__).resolve().relative_to(hw_root)),
                "sha256": source_sha},
            "docs359_sha256_before": docs359_before,
            "docs359_sha256_after": docs359_after,
        },
        "split_and_seal": {
            "upstream_inner_outer_seal_mismatches": seal_mismatches,
            "train_unique_keys": len(train_keys),
            "heldout_unique_keys": len(heldout_keys),
            "train_heldout_key_overlap": len(key_overlap),
            "seal_marker_result_mtime_order": mtime_order,
            "one_shot_marker_exists": paths["m430_one_shot_marker"].is_file(),
            "upstream_reported_m40_payload_read_attempts":
                upstream_result["execution_gates"]["m40_payload_read_attempts"],
            "upstream_reported_completed_evaluations":
                upstream_result["execution_gates"]["completed_heldout_evaluations"],
            "qualification": "The marker, immutable hashes and monotonic mtimes corroborate pre-heldout sealing and one completed result; historical read-attempt count is process provenance, not independently observable from final files alone."
        },
        "catalog_identity": {
            "partitions_checked": 4 * PARTITIONS,
            "q16_prefix_mismatches": q16_mismatches,
            "tail_outside_m338_ids16_to127": tail_pool_mismatches,
            "duplicate_catalog_mismatches": duplicate_mismatches,
            "selected_partition_counts": dict(selected_count),
        },
        "training_recomputation": {
            "payload_files_rehashed": train_payload_files,
            "payload_bytes_rehashed": train_payload_bytes,
            "full_selected_partitions_checked": 4 * PARTITIONS,
            "full_selected_objective_mismatches": full_objective_mismatches,
            "full_selected_cycle_mismatches":
                full_selected_cycle_mismatches,
            "candidate_regeneration_partitions":
                4 * len(SAMPLED_PARTITIONS),
            "candidate_regeneration_partition_ids":
                list(SAMPLED_PARTITIONS),
            "sampled_candidate_pattern_mismatches":
                sampled_candidate_pattern_mismatches,
            "sampled_candidate_objective_mismatches":
                sampled_candidate_objective_mismatches,
            "sampled_candidate_cycle_mismatches":
                sampled_candidate_cycle_mismatches,
            "sampled_chosen_option_mismatches": sampled_chosen_mismatches,
            "selected_cycles": train_selected_cycles,
            "strong_zero_cycles": train_strong_cycles,
            "tie_break_rule": "minimum phase cycles, then objective, then fixed option order; greedy inner ties use M338 pool ID then value",
            "heldout_feedback_used_by_independent_train_rebuild": False,
        },
        "static_codec": {
            "m430": m430_codec,
            "m338_signed12_violations":
                m338_codec["signed12_violations"],
        },
        "heldout_recomputation": {
            "payload_files_rehashed": heldout_payload_files,
            "payload_bytes_rehashed": heldout_payload_bytes,
            "rows": sum(phase["source_rows"] for phases in m430_by_sample
                        for phase in phases),
            "phases": sum(len(phases) for phases in m430_by_sample),
            "bit_residual_reconstruction_mismatches":
                heldout_reconstruction_mismatches,
            "upstream_phase_row_mismatches": phase_row_mismatches,
            "upstream_timestamp_mismatches": timestamp_mismatches,
            "upstream_component_mismatches": component_mismatches,
            "maximum_slot_bytes": maximum_slot,
            "m430_population": aggregate_phases(
                [phase for phases in m430_by_sample for phase in phases]),
        },
        "cycle_recomputation": {
            "m430_legal_dual_cycles": m430_cycles,
            "strong_zero_cycles": strong_zero_cycles,
            "m423_catalog_legal_dual_cycles": m423_dual_cycles,
            "m338_catalog_legal_dual_cycles": m338_dual_cycles,
            "m401_serial_cycles": m401_serial_cycles,
            "m430_speedup_vs_strong_zero":
                strong_zero_cycles / float(m430_cycles),
            "m430_speedup_vs_m423_dual":
                m423_dual_cycles / float(m430_cycles),
            "mismatches": cycle_mismatches,
            "formula": "per output tile work=4*correction_ops_per_block+4*pwp_rows; no narrow subtraction and no seed/correction fusion",
        },
        "traffic_and_semantics": {
            "pwp_dram_physical_bytes":
                m430_components["pwp_dram_physical_bytes"],
            "pwp_logical_onchip_read_bytes":
                m430_components["pwp_logical_onchip_read_bytes"],
            "pwp_padded_signal_bytes":
                m430_components["pwp_padded_signal_bytes"],
            "correction_onchip_read_bytes":
                m430_components["correction_onchip_read_bytes"],
            "wide_logical_bytes_per_cycle": 144,
            "wide_physical_signal_bytes_per_cycle": 160,
            "strong_zero_shared_source_bytes_per_cycle": 96,
            "new_port_is_free_shared96_upgrade": False,
            "persistent_old_psum": "preserved: each exact delta PWP[p]+W*(x-p) is subsequently accumulated as old_psum += delta",
            "seed_first_correction_fusion": False,
            "m427r3_withdrawal_respected": True,
            "traffic_mismatches": traffic_mismatches,
        },
        "claim_boundary": {
            "scope": expected["scope"],
            "exact_arithmetic": True,
            "checkpoint_or_accuracy_changed": False,
            "go_full_population_stimulus_and_legal_dual_functional_rtl": True,
            "system_speedup": False,
            "rtl_measured_speedup": False,
            "resource_normalized_speedup": False,
            "physical_sram_or_interconnect": False,
            "power_or_energy": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
    }
    result_path = args.output_dir / "m435_independent_recomputation.json"
    write_json(result_path, result)

    review = {
        "schema": "m435_m430_independent_hammer_review_v1",
        "status": "PASS_WITH_RESOURCE_QUALIFICATION",
        "score": 93,
        "severity_counts": {"P0": 0, "P1": 1, "P2": 2},
        "decision": {
            "full_population_rtl_stimulus": "GO",
            "legal_dual_functional_rtl": "GO",
            "paper_speedup_or_headline": "NO_GO_PENDING_MATCHED_PORT_AREA_POWER",
        },
        "findings": [
            {
                "severity": "P1",
                "title": "The 1.435375x cycle point is resource-unmatched",
                "detail": "Wide PWP issues require 144 logical/160 padded bytes per cycle versus SHARED96's 96 bytes.  The cycle result is valid for the new port point but is not a free same-resource speedup; DC/PT and SRAM/interconnect normalization remain mandatory."
            },
            {
                "severity": "P2",
                "title": "One-shot attempt count is provenance rather than independently recoverable state",
                "detail": "Double seals, marker contents, hashes and mtime ordering corroborate the declared sequence.  Final files alone cannot prove that no unrecorded process read M40 before the marker; wording must stay 'sole completed replay under the recorded workflow'."
            },
            {
                "severity": "P2",
                "title": "Candidate generation was independently regenerated on 64/1728 partitions",
                "detail": "Every selected catalog objective/cycle and membership was recomputed on all 1728 partitions; deterministic single-gain/greedy candidate construction and tie-breaking were rebuilt on 64 stratified partitions.  This is strong but not a full 1728-partition independent regeneration of discarded candidates."
            }
        ],
        "verified": [
            "M77 q16 identity and M338 IDs16..127 tail membership on all 1728 partitions",
            "M73/M40 exact key overlap zero and both upstream double seals intact",
            "full selected training objectives/cycles on 165,888,000 source rows",
            "full 51,840,000-row heldout bit residual reconstruction and 17,280 phases",
            "M430 517,041,352; strong zero 742,148,386; M423 dual 527,837,132; M338 dual 530,606,660; M401 serial 641,790,704",
            "all M430 per-phase rows and timestamps match exactly",
            "PWP DRAM traffic 702,350,080 bytes and all component ledgers match",
            "M427r3 persistent-old-psum red line is preserved and seed fusion remains withdrawn"
        ],
        "scope": expected["scope"] +
                 "; not system/RTL/PPA/power/energy/DATE headline",
    }
    review_path = args.output_dir / "m435_m430_independent_hammer_review.json"
    write_json(review_path, review)
    markdown = """# M435 M430 独立打铁评审\n\n""" + \
        f"**评分：{review['score']}/100；结论：带资源限定通过。**\n\n" + \
        "- GO：生成 full-population RTL stimulus，并进入保留 persistent old_psum 的合法 dual co-read 功能 RTL。\n" + \
        "- NO-GO：把 1.435375x 写成同资源或系统加速；144/160 B/cycle 新端口必须做 DC/PT/存储与互联归一。\n" + \
        "- 独立重算：51.84M held-out rows、17,280 phases、逐 phase timestamp 全部 0 mismatch。\n" + \
        "- 精确周期：M430 517,041,352；strong-zero 742,148,386；M423+dual 527,837,132；M401 serial 641,790,704。\n" + \
        "- 语义红线：不做 seed-first-correction fusion；`old_psum += PWP + correction`。\n\n" + \
        "P1 是资源不匹配；P2 是 one-shot 历史只能由封存/marker/时间序辅证，以及废弃候选的独立生成只抽查 64/1728 partition。\n"
    markdown_path = args.output_dir / "m435_m430_independent_hammer_review.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    readme_path = args.output_dir / "README.md"
    readme_path.write_text(
        "M435 independently recomputes M430 without importing upstream analyzers.\n",
        encoding="utf-8")
    manifest_sha, seal_sha = seal(args.output_dir, [
        result_path.name, review_path.name, markdown_path.name,
        readme_path.name])
    print(json.dumps({
        "status": review["status"], "score": review["score"],
        "m430_cycles": m430_cycles,
        "m430_speedup_vs_strong_zero":
            strong_zero_cycles / float(m430_cycles),
        "phase_mismatches": phase_row_mismatches,
        "timestamp_mismatches": timestamp_mismatches,
        "pwp_dram_physical_bytes":
            m430_components["pwp_dram_physical_bytes"],
        "manifest_sha256": manifest_sha, "seal_sha256": seal_sha,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
