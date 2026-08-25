#!/usr/bin/env python3
"""Independent M78 reconstruction from frozen M41/M72/M40 artifacts.

This validator intentionally does not import the M78 producer, M76 simulator,
M72 analyzer, M43 analyzer, or any other production Python module.  INT8 PWP
ranges, packed-plane convolution geometry, Hamming assignment, block-local
escape, traffic, and all three cycle models are reimplemented locally.
"""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
M78_ANALYZER = HW / "system_simulator/scripts/analyze_m78_precision_elastic_pwp.py"
M78 = HW / (
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/"
    "m78_precision_elastic_pwp.json")
M76_R2 = HW / (
    "results/m76_phi_style_pattern_engine_cycle_sim_valid825_internal_dev_r2_20260823/"
    "m76_phi_style_pattern_engine_cycle_sim.json")
M72 = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_ROOT = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41 = M41_ROOT / "m41_h67_ep35_bottleneck_int8_bridge.json"
M40_ROOT = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
M40 = M40_ROOT / "m40_bottleneck_packed_source_manifest.json"
RECONSTRUCTION = HERE / "m78_independent_reconstruction.json"
REVIEW = HERE / "m78_precision_elastic_pwp_independent_hammer_review.json"
RECEIPT = HERE / "m78_precision_elastic_pwp_independent_hammer_validation_receipt.json"

EXPECTED_SHA = {
    "m78_analyzer": "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b",
    "m78": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m76_r2": "2b0c12addc4fae781a8aa1309145459b7da7cfae038c463445f701e25849890c",
    "m72": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m40": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
}
EXPECTED_WEIGHT_SHA = (
    "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
)

TIMESTEPS = 10
CHANNELS = 768
HEIGHT = 15
WIDTH = 20
ROWS = TIMESTEPS * HEIGHT * WIDTH
FEATURES = CHANNELS * 9
TILE_BITS = 256
TILES = (FEATURES + TILE_BITS - 1) // TILE_BITS
PARTITION_BITS = 16
PARTITIONS = FEATURES // PARTITION_BITS
PARTITIONS_PER_TILE = TILE_BITS // PARTITION_BITS
PATTERNS = 16
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
OUTPUT_CHANNELS = OUTPUT_BLOCKS * OUTPUT_LANES
WEIGHT_VECTOR_BYTES = OUTPUT_LANES
WIDTH_CAPS = (8, 9, 10, 11, 12)
DRAM_BYTES_PER_CYCLE = 32
MATCHER_FILL = 16
PACKER_FILL = 4
PACKER_UNITS = 8
COMPUTE_TAIL = 2
WEIGHT_PHASE_BYTES = PARTITION_BITS * OUTPUT_BLOCKS * WEIGHT_VECTOR_BYTES
PORTS = (
    ("WIDE_PRECISION_ELASTIC", 1, None),
    ("SHARED_96B", 1, 96),
    ("SHARED_32B", 3, 32),
)
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << PARTITION_BITS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs_hook(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + raw)))


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8")


def signed_width(minimum, maximum):
    for width in range(1, 33):
        if minimum >= -(1 << (width - 1)) and maximum <= (1 << (width - 1)) - 1:
            return max(8, width)
    raise ValueError("signed width exceeds 32 bits")


def unpack_histograms(record):
    """Decode an M40 support plane without production unpack helpers."""
    require(record["shape"] == [10, 1, 768, 15, 20], "M40 shape drift")
    packed = M40_ROOT / record["packed_file"]
    require(packed.is_file() and sha256_path(packed) == record["packed_file_sha256"],
            "M40 heldout plane SHA drift")
    raw = packed.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(plane_bytes == 288000 and len(raw) == 3 * plane_bytes,
            "M40 plane extent drift")
    positive = raw[:plane_bytes]
    negative = raw[plane_bytes:2 * plane_bytes]
    require(not any(negative), "unexpected negative heldout support plane")
    masks = [0] * (ROWS * TILES)
    for byte_index, original in enumerate(positive):
        byte = original
        if byte == 0:
            continue
        bit_base = byte_index * 8
        while byte:
            low = byte & -byte
            bit = low.bit_length() - 1
            flat = bit_base + bit
            require(flat < TIMESTEPS * CHANNELS * HEIGHT * WIDTH,
                    "nonzero packed tail bit")
            timestep_channel, spatial = divmod(flat, HEIGHT * WIDTH)
            timestep, channel = divmod(timestep_channel, CHANNELS)
            input_y, input_x = divmod(spatial, WIDTH)
            feature_base = channel * 9
            for kernel_y in range(3):
                output_y = input_y - kernel_y + 1
                if output_y < 0 or output_y >= HEIGHT:
                    continue
                for kernel_x in range(3):
                    output_x = input_x - kernel_x + 1
                    if output_x < 0 or output_x >= WIDTH:
                        continue
                    feature = feature_base + kernel_y * 3 + kernel_x
                    tile, tile_bit = divmod(feature, TILE_BITS)
                    row = (timestep * HEIGHT + output_y) * WIDTH + output_x
                    masks[row * TILES + tile] |= 1 << tile_bit
            byte ^= low
    histograms = [Counter() for _ in range(PARTITIONS)]
    for row in range(ROWS):
        row_base = row * TILES
        for tile in range(TILES):
            packed_tile = masks[row_base + tile]
            partition_base = tile * PARTITIONS_PER_TILE
            for subtile in range(PARTITIONS_PER_TILE):
                value = (packed_tile >> (subtile * PARTITION_BITS)) & 0xffff
                histograms[partition_base + subtile][value] += 1
    require(all(sum(histogram.values()) == ROWS for histogram in histograms),
            "per-partition row conservation drift")
    return histograms


def build_catalog(m72, m41):
    operator_names = [row["operator"] for row in m72["operators"]]
    require(len(operator_names) == 4 and len(m41["layers"]) == 4,
            "operator extent drift")
    widths = np.zeros((4, PARTITIONS, PATTERNS, OUTPUT_BLOCKS), dtype=np.uint8)
    minima = np.zeros_like(widths, dtype=np.int32)
    maxima = np.zeros_like(widths, dtype=np.int32)
    width_hist = Counter()
    center_max_hist = Counter()
    outliers = []
    weight_shas = []
    exact_twos_complement_mismatches = 0
    catalog_digest = hashlib.sha256()
    for op, operator in enumerate(m72["operators"]):
        layer = next(row for row in m41["layers"]
                     if row["operator"] == operator["operator"])
        weight_info = next(row for row in layer["payloads"] if row["role"] == "weight")
        weight_path = M41_ROOT / weight_info["file"]
        observed_sha = sha256_path(weight_path)
        require(observed_sha == EXPECTED_WEIGHT_SHA[op] == weight_info["sha256"],
                "M41 INT8 weight identity drift")
        weight_shas.append(observed_sha)
        weight = np.fromfile(str(weight_path), dtype=np.int8)
        require(weight.size == FEATURES * OUTPUT_CHANNELS,
                "M41 INT8 weight extent drift")
        weight = weight.reshape(FEATURES, OUTPUT_CHANNELS).astype(np.int32)
        require(len(operator["partitions"]) == PARTITIONS,
                "M72 partition extent drift")
        for partition, row in enumerate(operator["partitions"]):
            require(row["partition"] == partition and len(row["centers_hex"]) == PATTERNS,
                    "M72 center extent/order drift")
            centers = [int(raw, 16) for raw in row["centers_hex"]]
            require(len(set(centers)) == PATTERNS, "M72 centers are not unique")
            source = weight[
                partition * PARTITION_BITS:(partition + 1) * PARTITION_BITS]
            for pattern, center in enumerate(centers):
                indices = [bit for bit in range(PARTITION_BITS)
                           if center & (1 << bit)]
                pwp = source[indices].sum(axis=0, dtype=np.int32)
                center_widths = []
                for block in range(OUTPUT_BLOCKS):
                    values = pwp[block * OUTPUT_LANES:(block + 1) * OUTPUT_LANES]
                    minimum = int(values.min())
                    maximum = int(values.max())
                    width = signed_width(minimum, maximum)
                    widths[op, partition, pattern, block] = width
                    minima[op, partition, pattern, block] = minimum
                    maxima[op, partition, pattern, block] = maximum
                    width_hist[width] += 1
                    center_widths.append(width)
                    mask = (1 << width) - 1
                    encoded = np.bitwise_and(values.astype(np.int64), mask)
                    decoded = np.where(
                        encoded >= (1 << (width - 1)), encoded - (1 << width), encoded)
                    exact_twos_complement_mismatches += int(np.count_nonzero(
                        decoded != values.astype(np.int64)))
                    catalog_digest.update(canonical_bytes({
                        "operator": op,
                        "partition": partition,
                        "pattern": pattern,
                        "block": block,
                        "center": center,
                        "minimum": minimum,
                        "maximum": maximum,
                        "width": width,
                    }))
                    if width >= 12:
                        outliers.append({
                            "operator_index": op,
                            "operator": operator["operator"],
                            "partition": partition,
                            "pattern_index": pattern,
                            "center_hex": row["centers_hex"][pattern],
                            "output_block": block,
                            "minimum": minimum,
                            "maximum": maximum,
                            "required_signed_bits": width,
                        })
                center_max_hist[max(center_widths)] += 1
        print("[M78 INDEPENDENT WIDTH] operator={}/4".format(op + 1), flush=True)
    require(exact_twos_complement_mismatches == 0,
            "variable-width two's-complement round-trip failure")
    return {
        "widths": widths,
        "minima": minima,
        "maxima": maxima,
        "width_hist": width_hist,
        "center_max_hist": center_max_hist,
        "outliers": outliers,
        "weight_shas": weight_shas,
        "catalog_digest": catalog_digest.hexdigest(),
        "twos_complement_mismatches": exact_twos_complement_mismatches,
    }


def pwp_bytes(width):
    return width * OUTPUT_LANES // 8


def pwp_cycles(width, port_bytes):
    if port_bytes is None:
        return 1
    return int(math.ceil(pwp_bytes(width) / float(port_bytes)))


def evaluate_phase(histogram, centers, center_widths, outlier_key, phase_key):
    base = Counter()
    caps = dict((cap, Counter()) for cap in WIDTH_CAPS)
    cap_width_uses = dict((cap, Counter()) for cap in WIDTH_CAPS)
    escape_by_sample = dict((cap, Counter()) for cap in WIDTH_CAPS)
    algebraic_mask_mismatches = 0
    outlier_selected_rows = 0
    for value, count in histogram.items():
        population = POPCOUNT[value]
        base["partition_vectors"] += count
        base["baseline_ops_per_block"] += count * population
        if population >= 2:
            base["matcher_rows"] += count
        best_distance, best_center, best_index = min(
            (POPCOUNT[value ^ center], center, index)
            for index, center in enumerate(centers))
        beneficial = 1 + best_distance < population
        if beneficial:
            add_mask = value & (~best_center & 0xffff)
            remove_mask = best_center & (~value & 0xffff)
            reconstructed = (best_center | add_mask) & (~remove_mask & 0xffff)
            algebraic_mask_mismatches += int(
                reconstructed != value or (add_mask & remove_mask) != 0)
        if beneficial and phase_key == outlier_key[:2] and best_index == outlier_key[2]:
            outlier_selected_rows += count
        for cap in WIDTH_CAPS:
            row = caps[cap]
            any_eligible = False
            for block in range(OUTPUT_BLOCKS):
                width = int(center_widths[best_index, block])
                if beneficial and width <= cap:
                    any_eligible = True
                    row["correction_ops_all_blocks"] += count * best_distance
                    row["pwp_ops_all_blocks"] += count
                    row["pwp_read_bytes"] += count * pwp_bytes(width)
                    cap_width_uses[cap][width] += count
                else:
                    row["correction_ops_all_blocks"] += count * population
                    if beneficial:
                        row["escape_rows_all_blocks"] += count
                        escape_by_sample[cap][block] += count
            if beneficial and any_eligible:
                row["assignment_rows"] += count
    return (base, caps, cap_width_uses, escape_by_sample,
            algebraic_mask_mismatches, outlier_selected_rows)


def phase_payload_bytes(center_widths, cap):
    total = 0
    for pattern in range(PATTERNS):
        for block in range(OUTPUT_BLOCKS):
            width = int(center_widths[pattern, block])
            if width <= cap:
                total += pwp_bytes(width)
    return total


def replay_sample(phases, cap, port):
    port_name, weight_service, port_bytes = port
    baseline_load = int(math.ceil(WEIGHT_PHASE_BYTES / float(DRAM_BYTES_PER_CYCLE)))
    candidate_loads = [int(math.ceil(
        (WEIGHT_PHASE_BYTES + phase["payload_bytes"][cap]) /
        float(DRAM_BYTES_PER_CYCLE))) for phase in phases]
    dense_cycles = baseline_load
    baseline_cycles = baseline_load
    candidate_cycles = candidate_loads[0]
    binding = Counter()
    component = Counter()
    serial_candidate = candidate_loads[0]
    serial_baseline = baseline_load
    minimum_compute_margin = None
    for phase_index, phase in enumerate(phases):
        base = phase["base"]
        cap_row = phase["caps"][cap]
        dense_compute = (
            base["partition_vectors"] * PARTITION_BITS * OUTPUT_BLOCKS * weight_service)
        baseline_compute = (
            base["baseline_ops_per_block"] * OUTPUT_BLOCKS * weight_service)
        pwp_compute = sum(
            uses * pwp_cycles(width, port_bytes)
            for width, uses in phase["width_uses"][cap].items())
        candidate_compute = (
            cap_row["correction_ops_all_blocks"] * weight_service + pwp_compute)
        matcher = base["matcher_rows"] + MATCHER_FILL
        packer = int(math.ceil(cap_row["assignment_rows"] / float(PACKER_UNITS))) + PACKER_FILL
        next_baseline = baseline_load if phase_index + 1 < len(phases) else 0
        next_candidate = candidate_loads[phase_index + 1] if phase_index + 1 < len(phases) else 0
        dense_cycles += max(dense_compute, next_baseline) + COMPUTE_TAIL
        baseline_cycles += max(baseline_compute, next_baseline) + COMPUTE_TAIL
        candidates = (
            (candidate_compute, "compute"),
            (matcher, "matcher"),
            (packer, "packer"),
            (next_candidate, "dma"),
        )
        candidate_cycles += max(value for value, _ in candidates) + COMPUTE_TAIL
        winner = max(candidates)[1]
        binding[winner] += 1
        component["matcher_cycles"] += matcher
        component["packer_cycles"] += packer
        component["candidate_compute_cycles"] += candidate_compute
        serial_candidate += candidate_compute + matcher + packer + next_candidate + COMPUTE_TAIL
        serial_baseline += baseline_compute + next_baseline + COMPUTE_TAIL
        margin = candidate_compute - max(matcher, packer, next_candidate)
        minimum_compute_margin = margin if minimum_compute_margin is None else min(
            minimum_compute_margin, margin)
    return {
        "producer_projection": {
            "dense_cycles": dense_cycles,
            "bit_sparse_cycles": baseline_cycles,
            "candidate_cycles": candidate_cycles,
            "speedup_vs_dense": dense_cycles / float(candidate_cycles),
            "speedup_vs_bit_sparse": baseline_cycles / float(candidate_cycles),
            "binding_phases": dict(binding),
            "component_cycles": dict(component),
        },
        "fully_serial_candidate_cycles": serial_candidate,
        "fully_serial_bit_sparse_cycles": serial_baseline,
        "minimum_compute_margin_over_matcher_packer_dma": minimum_compute_margin,
        "candidate_prefetch_cycles": sum(candidate_loads),
        "candidate_prefetch_bytes": sum(
            WEIGHT_PHASE_BYTES + phase["payload_bytes"][cap] for phase in phases),
    }


def compare(left, right, label="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), label + " keys drift")
        for key in left:
            compare(left[key], right[key], label + "." + str(key))
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), label + " length drift")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, label + "[{}]".format(index))
    elif isinstance(left, float) or isinstance(right, float):
        require(abs(float(left) - float(right)) <=
                1e-12 * max(1.0, abs(float(right))), label + " float drift")
    else:
        require(left == right, label + " drift: {} != {}".format(left, right))


def reconstruct():
    paths = {
        "m78_analyzer": M78_ANALYZER,
        "m78": M78,
        "m76_r2": M76_R2,
        "m72": M72,
        "m41": M41,
        "m40": M40,
    }
    for name, path in paths.items():
        require(sha256_path(path) == EXPECTED_SHA[name], name + " SHA drift")
    m78 = strict_json(M78)
    m76 = strict_json(M76_R2)
    m72 = strict_json(M72)
    m41 = strict_json(M41)
    manifest = strict_json(M40)
    require(m72["split"]["heldout_samples_within_valid825"] == [5, 6, 7, 8, 9]
            and m72["split"]["train_catalog_eligible"] is False,
            "M72 split boundary drift")
    require(m78["admission"]["valid825_internal_only"] is True and
            all(m78["admission"][key] is False for key in (
                "independent_validation", "train_catalog", "accuracy",
                "rtl_or_synopsys_ppa", "full_network_or_system_speedup",
                "date_headline")), "M78 claim boundary widened")

    catalog = build_catalog(m72, m41)
    producer_precision = m78["pwp_precision"]
    independent_precision = {
        "output_block_entry_count": int(catalog["widths"].size),
        "minimum_width_floor_bits": 8,
        "width_histogram": dict((str(key), value)
                                for key, value in sorted(catalog["width_hist"].items())),
        "center_max_width_histogram": dict(
            (str(key), value) for key, value in sorted(catalog["center_max_hist"].items())),
        "required_12bit_outliers": catalog["outliers"],
        "pattern_table_bytes": 4 * PARTITIONS * PATTERNS * 2,
        "fixed12_pwp_payload_bytes": int(catalog["widths"].size) * 12 * OUTPUT_LANES // 8,
    }
    compare(independent_precision, producer_precision, "pwp_precision")
    require(len(catalog["outliers"]) == 1, "12-bit outlier is not unique")
    outlier = catalog["outliers"][0]
    outlier_key = (outlier["operator_index"], outlier["partition"],
                   outlier["pattern_index"], outlier["output_block"])

    operator_names = [row["operator"] for row in m72["operators"]]
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    records = [row for row in manifest["records"] if row["sample_id"] >= 5]
    require(len(records) == 20, "M40 heldout record extent drift")
    phases_by_sample = dict((sample, []) for sample in range(5, 10))
    aggregate_base = Counter()
    aggregate_caps = dict((cap, Counter()) for cap in WIDTH_CAPS)
    aggregate_width_uses = dict((cap, Counter()) for cap in WIDTH_CAPS)
    aggregate_escape_by_block = dict((cap, Counter()) for cap in WIDTH_CAPS)
    mask_mismatches = 0
    outlier_selected_rows_by_sample = Counter()
    phase_digest = hashlib.sha256()
    for record_number, record in enumerate(records):
        sample = record["sample_id"]
        op = operator_index[record["operator"]]
        histograms = unpack_histograms(record)
        for partition, histogram in enumerate(histograms):
            row = m72["operators"][op]["partitions"][partition]
            centers = [int(raw, 16) for raw in row["centers_hex"]]
            center_widths = catalog["widths"][op, partition]
            (base, caps, width_uses, escape_by_block,
             phase_mask_mismatches, outlier_rows) = evaluate_phase(
                 histogram, centers, center_widths, outlier_key,
                 (op, partition))
            mask_mismatches += phase_mask_mismatches
            outlier_selected_rows_by_sample[sample] += outlier_rows
            aggregate_base.update(base)
            payload_bytes = {}
            for cap in WIDTH_CAPS:
                aggregate_caps[cap].update(caps[cap])
                aggregate_width_uses[cap].update(width_uses[cap])
                aggregate_escape_by_block[cap].update(escape_by_block[cap])
                payload_bytes[cap] = phase_payload_bytes(center_widths, cap)
            phase = {
                "sample_id": sample,
                "operator_index": op,
                "partition": partition,
                "base": base,
                "caps": caps,
                "width_uses": width_uses,
                "payload_bytes": payload_bytes,
            }
            phases_by_sample[sample].append(phase)
            phase_digest.update(canonical_bytes({
                "sample": sample,
                "operator": op,
                "partition": partition,
                "base": dict(base),
                "caps": dict((str(cap), dict(caps[cap])) for cap in WIDTH_CAPS),
                "width_uses": dict((str(cap), dict(width_uses[cap]))
                                   for cap in WIDTH_CAPS),
                "payload_bytes": dict((str(cap), payload_bytes[cap])
                                      for cap in WIDTH_CAPS),
            }))
        print("[M78 INDEPENDENT TRACE] record={}/20 sample={} op={}".format(
            record_number + 1, sample, op), flush=True)
    require(mask_mismatches == 0, "PWP correction mask identity drift")
    require(all(len(phases_by_sample[sample]) == 4 * PARTITIONS
                for sample in range(5, 10)), "per-sample phase extent drift")
    require(aggregate_base["partition_vectors"] == 25920000 and
            aggregate_base["baseline_ops_per_block"] == 46432637 and
            aggregate_base["matcher_rows"] == 9932028,
            "heldout work conservation drift")

    total_entries = int(catalog["widths"].size)
    fixed12_bits = total_entries * 12 * OUTPUT_LANES
    independent_configs = []
    extended_configs = []
    cycle_digest = hashlib.sha256()
    for cap in WIDTH_CAPS:
        eligible_hist = dict((width, count)
                             for width, count in catalog["width_hist"].items()
                             if width <= cap)
        eligible_entries = sum(eligible_hist.values())
        elastic_bits = sum(width * count * OUTPUT_LANES
                           for width, count in eligible_hist.items())
        fixed_cap_bits = eligible_entries * cap * OUTPUT_LANES
        cap_row = aggregate_caps[cap]
        per_port = []
        extended_ports = []
        for port in PORTS:
            totals = Counter()
            per_sample = []
            serial_candidate = 0
            serial_baseline = 0
            prefetch_cycles = 0
            prefetch_bytes = 0
            minimum_margin = None
            for sample in range(5, 10):
                replay = replay_sample(phases_by_sample[sample], cap, port)
                projected = replay["producer_projection"]
                projected["sample_id"] = sample
                per_sample.append(projected)
                for key in ("dense_cycles", "bit_sparse_cycles", "candidate_cycles"):
                    totals[key] += projected[key]
                totals.update(projected["binding_phases"])
                serial_candidate += replay["fully_serial_candidate_cycles"]
                serial_baseline += replay["fully_serial_bit_sparse_cycles"]
                prefetch_cycles += replay["candidate_prefetch_cycles"]
                prefetch_bytes += replay["candidate_prefetch_bytes"]
                minimum_margin = (replay["minimum_compute_margin_over_matcher_packer_dma"]
                                  if minimum_margin is None else min(
                                      minimum_margin,
                                      replay["minimum_compute_margin_over_matcher_packer_dma"]))
            projected_config = {
                "port": port[0],
                "weight_vector_service_cycles": port[1],
                "candidate_cycles": totals["candidate_cycles"],
                "bit_sparse_cycles": totals["bit_sparse_cycles"],
                "dense_cycles": totals["dense_cycles"],
                "speedup_vs_bit_sparse": (
                    totals["bit_sparse_cycles"] / float(totals["candidate_cycles"])),
                "speedup_vs_dense": (
                    totals["dense_cycles"] / float(totals["candidate_cycles"])),
                "binding_phases": {
                    "compute": totals["compute"],
                    "matcher": totals["matcher"],
                    "packer": totals["packer"],
                    "dma": totals["dma"],
                },
                "per_sample": per_sample,
            }
            per_port.append(projected_config)
            extended_ports.append({
                "port": port[0],
                "fully_serial_candidate_cycles": serial_candidate,
                "fully_serial_bit_sparse_cycles": serial_baseline,
                "fully_serial_speedup_vs_bit_sparse": (
                    serial_baseline / float(serial_candidate)),
                "minimum_compute_margin_over_matcher_packer_dma": minimum_margin,
                "candidate_prefetch_cycles_hidden_or_initial": prefetch_cycles,
                "candidate_prefetch_bytes_modelled": prefetch_bytes,
            })
            cycle_digest.update(canonical_bytes({
                "cap": cap,
                "port": port[0],
                "projection": projected_config,
                "extended": extended_ports[-1],
            }))
        baseline_all_blocks = aggregate_base["baseline_ops_per_block"] * OUTPUT_BLOCKS
        candidate_ops = (cap_row["correction_ops_all_blocks"] +
                         cap_row["pwp_ops_all_blocks"])
        independent_config = {
            "signed_width_cap": cap,
            "eligible_output_block_entries": eligible_entries,
            "ineligible_output_block_entries": total_entries - eligible_entries,
            "eligible_fraction": eligible_entries / float(total_entries),
            "exact_elastic_pwp_payload_bytes": elastic_bits // 8,
            "fixed_cap_pwp_payload_bytes": fixed_cap_bits // 8,
            "fixed12_reference_payload_bytes": fixed12_bits // 8,
            "elastic_storage_reduction_vs_fixed12": (
                1.0 - elastic_bits / float(fixed12_bits)),
            "fixed_cap_storage_reduction_vs_fixed12": (
                1.0 - fixed_cap_bits / float(fixed12_bits)),
            "metadata_bytes_three_bits_per_entry": int(math.ceil(total_entries * 3 / 8.0)),
            "heldout": {
                "baseline_bit_sparse_vector_ops_all_blocks": baseline_all_blocks,
                "candidate_vector_ops_all_blocks": candidate_ops,
                "natural_vector_op_speedup_vs_bit_sparse": (
                    baseline_all_blocks / float(candidate_ops)),
                "pwp_ops_all_blocks": cap_row["pwp_ops_all_blocks"],
                "correction_ops_all_blocks": cap_row["correction_ops_all_blocks"],
                "block_local_escape_rows": cap_row["escape_rows_all_blocks"],
                "assignment_rows": cap_row["assignment_rows"],
                "pwp_uses_by_width": dict(
                    (str(width), count)
                    for width, count in sorted(aggregate_width_uses[cap].items())),
                "baseline_weight_sram_read_bytes": baseline_all_blocks * WEIGHT_VECTOR_BYTES,
                "candidate_correction_sram_read_bytes": (
                    cap_row["correction_ops_all_blocks"] * WEIGHT_VECTOR_BYTES),
                "candidate_pwp_sram_read_bytes": cap_row["pwp_read_bytes"],
            },
            "cycle_simulations": per_port,
        }
        independent_configs.append(independent_config)
        metadata_bytes = independent_config["metadata_bytes_three_bits_per_entry"]
        extended_configs.append({
            "signed_width_cap": cap,
            "elastic_payload_plus_width_metadata_bytes": elastic_bits // 8 + metadata_bytes,
            "storage_reduction_vs_fixed12_after_metadata": (
                1.0 - (elastic_bits // 8 + metadata_bytes) /
                float(fixed12_bits // 8)),
            "metadata_bytes_per_phase": PATTERNS * OUTPUT_BLOCKS * 3 // 8,
            "metadata_bytes_if_streamed_for_five_samples": 5 * metadata_bytes,
            "pattern_table_bytes_not_charged_to_phase_prefetch": (
                independent_precision["pattern_table_bytes"]),
            "escape_rows_by_output_block": dict(
                (str(block), count)
                for block, count in sorted(aggregate_escape_by_block[cap].items())),
            "ports": extended_ports,
        })
    compare(independent_configs, m78["configurations"], "configurations")

    independent_work = {
        "partition_vectors_per_output_block": aggregate_base["partition_vectors"],
        "baseline_bit_sparse_vector_ops_per_output_block": (
            aggregate_base["baseline_ops_per_block"]),
        "matcher_rows": aggregate_base["matcher_rows"],
    }
    compare(independent_work, m78["work_conservation"], "work_conservation")

    m76_shared32 = next(row for row in m76["configurations"]
                        if row["name"] == "SHARED_32B_PORT")
    m78_cap11 = next(row for row in independent_configs
                     if row["signed_width_cap"] == 11)
    m78_shared32 = next(row for row in m78_cap11["cycle_simulations"]
                        if row["port"] == "SHARED_32B")
    m78_cap12 = next(row for row in independent_configs
                     if row["signed_width_cap"] == 12)
    m78_cap12_shared32 = next(row for row in m78_cap12["cycle_simulations"]
                              if row["port"] == "SHARED_32B")
    require(m76_shared32["candidate_speedup_vs_bit_sparse"] ==
            1.2968616827300425 and
            m78_shared32["speedup_vs_bit_sparse"] == 1.4094065141412047,
            "M76-to-M78 comparison drift")
    require(sum(outlier_selected_rows_by_sample.values()) == 362 and
            m78_cap11["heldout"]["block_local_escape_rows"] == 362,
            "cap11 unique outlier escape use drift")

    return {
        "schema": "m78_precision_elastic_pwp_independent_reconstruction_v1",
        "status": "PASS_M78_INDEPENDENT_M41_M72_M40_RECONSTRUCTION",
        "identity_sha256": dict(EXPECTED_SHA, **{
            "weight_payloads": list(catalog["weight_shas"]),
            "independent_catalog_digest": catalog["catalog_digest"],
            "independent_phase_digest": phase_digest.hexdigest(),
            "independent_cycle_digest": cycle_digest.hexdigest(),
        }),
        "independence": {
            "production_m78_imported": False,
            "production_m76_m72_m43_modules_imported": False,
            "inputs_parsed_directly": ["M41 INT8 binaries", "M72 center JSON",
                                        "M40 heldout packed support planes"],
        },
        "population": {
            "samples": 5,
            "operators": 4,
            "partitions_per_operator": PARTITIONS,
            "phases": 5 * 4 * PARTITIONS,
            "output_blocks": OUTPUT_BLOCKS,
            "output_lanes": OUTPUT_LANES,
            "catalog_output_block_entries": total_entries,
        },
        "precision": dict(independent_precision, **{
            "all_catalog_values_exact_twos_complement_roundtrip_mismatches": (
                catalog["twos_complement_mismatches"]),
        }),
        "work_conservation": independent_work,
        "configurations": independent_configs,
        "extended_accounting": extended_configs,
        "block_local_escape_exactness": {
            "correction_mask_algebra_mismatches": mask_mismatches,
            "unique_12bit_catalog_entry": outlier,
            "cap11_outlier_selected_rows_total": sum(
                outlier_selected_rows_by_sample.values()),
            "cap11_outlier_selected_rows_by_sample": dict(
                (str(key), value)
                for key, value in sorted(outlier_selected_rows_by_sample.items())),
            "cap11_escape_rows_all_blocks": m78_cap11["heldout"]["block_local_escape_rows"],
            "interpretation": (
                "Eligible blocks use an exact signed center sum plus add/remove "
                "corrections; the sole width-12 block is absent from cap11 PWP "
                "storage and performs its exact baseline bit-sparse sum."),
        },
        "m76_r2_to_m78_cap11_shared32": {
            "m76_r2_speedup_vs_bit_sparse": (
                m76_shared32["candidate_speedup_vs_bit_sparse"]),
            "m78_cap11_speedup_vs_bit_sparse": (
                m78_shared32["speedup_vs_bit_sparse"]),
            "relative_speedup_improvement_fraction": (
                m78_shared32["speedup_vs_bit_sparse"] /
                m76_shared32["candidate_speedup_vs_bit_sparse"] - 1.0),
            "m76_r2_candidate_cycles": m76_shared32["candidate_cycles"],
            "m78_cap11_candidate_cycles": m78_shared32["candidate_cycles"],
            "candidate_cycle_reduction_fraction": (
                1.0 - m78_shared32["candidate_cycles"] /
                float(m76_shared32["candidate_cycles"])),
            "m78_cap12_speedup_vs_bit_sparse": (
                m78_cap12_shared32["speedup_vs_bit_sparse"]),
            "cap11_cycle_penalty_vs_cap12": (
                m78_shared32["candidate_cycles"] -
                m78_cap12_shared32["candidate_cycles"]),
            "cap11_payload_bytes_saved_vs_cap12": (
                m78_cap12["exact_elastic_pwp_payload_bytes"] -
                m78_cap11["exact_elastic_pwp_payload_bytes"]),
        },
        "producer_mismatches": {
            "precision": 0,
            "work": 0,
            "configurations_and_per_sample_cycles": 0,
            "heldout_storage_and_traffic": 0,
        },
    }


def validate_review(payload):
    if RECONSTRUCTION.exists():
        compare(strict_json(RECONSTRUCTION), payload, "stored_reconstruction")
    if REVIEW.exists():
        review = strict_json(REVIEW)
        require(review["status"] ==
                "M78_COUNTS_REPRODUCED_GO_INTERNAL_DSE_NO_GO_RTL_SYSTEM_DATE_HEADLINE",
                "review status drift")
        require(len(review["findings"]["p0"]) == 0 and
                len(review["findings"]["p1"]) == 7,
                "review finding count drift")
        require(review["scores"] == {
            "hardware_innovation": 53,
            "performance_advantage": 66,
            "evidence_quality": 79,
            "m78_scoped_milestone_quality": 86,
            "date_paper_completeness": 50,
        }, "review score drift")
    if RECEIPT.exists():
        receipt = strict_json(RECEIPT)
        require(receipt["status"] ==
                "PASS_M78_PRECISION_ELASTIC_PWP_INDEPENDENT_HAMMER_VALIDATION" and
                receipt["identity"]["review_sha256"] == sha256_path(REVIEW) and
                receipt["identity"]["reconstruction_sha256"] ==
                sha256_path(RECONSTRUCTION) and
                receipt["identity"]["validator_sha256"] == sha256_path(Path(__file__)),
                "validation receipt drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = reconstruct()
    validate_review(payload)
    if args.output is not None:
        require(not args.output.exists(), "refusing reconstruction overwrite")
        require(args.output.resolve().parent == HERE.resolve(),
                "output must stay in review directory")
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison = payload["m76_r2_to_m78_cap11_shared32"]
    print("PASS M78 independent: entries={} outliers=1 escape_uses={} "
          "shared32={:.9f}x previous={:.9f}x P0=0".format(
              payload["population"]["catalog_output_block_entries"],
              payload["block_local_escape_exactness"]["cap11_escape_rows_all_blocks"],
              comparison["m78_cap11_speedup_vs_bit_sparse"],
              comparison["m76_r2_speedup_vs_bit_sparse"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M78 independent: {}".format(error), flush=True)
        raise SystemExit(1)
