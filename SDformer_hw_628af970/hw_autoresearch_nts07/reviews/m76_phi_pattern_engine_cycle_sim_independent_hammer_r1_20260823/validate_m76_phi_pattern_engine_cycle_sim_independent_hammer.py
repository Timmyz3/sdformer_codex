#!/usr/bin/env python3
"""Independent M76 reconstruction from M72 JSON and M40 heldout planes.

This validator intentionally does not import the M76 producer simulator, M72
analyzer, M43 analyzer, or any other production Python module.  Geometry,
bit-plane unpacking, nearest-pattern arithmetic, phase service, traffic, and
the three port configurations are reimplemented locally.
"""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
M76 = HW / (
    "results/m76_phi_style_pattern_engine_cycle_sim_valid825_internal_dev_r1_20260823/"
    "m76_phi_style_pattern_engine_cycle_sim.json")
M72 = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M40_ROOT = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
M40 = M40_ROOT / "m40_bottleneck_packed_source_manifest.json"
RECONSTRUCTION = HERE / "m76_independent_reconstruction.json"
REVIEW = HERE / "m76_phi_pattern_engine_cycle_sim_independent_hammer_review.json"
VALIDATION_RECEIPT = HERE / "m76_phi_pattern_engine_cycle_sim_independent_hammer_validation_receipt.json"

EXPECTED_SHA = {
    "m76": "487e9cfe32011f7a1b47342d2251e2cd93a3f10cfa22f8c2f38e397e810d1853",
    "m72": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m40": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
}

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
WEIGHT_VECTOR_BYTES = 96
PWP_SIGNED_BITS = 12
PWP_VECTOR_BYTES = OUTPUT_LANES * PWP_SIGNED_BITS // 8
DRAM_BYTES_PER_CYCLE = 32
MATCHER_FILL = 16
PACKER_FILL = 4
PACKER_UNITS = 8
COMPUTE_TAIL = 2
WEIGHT_PHASE_BYTES = PARTITION_BITS * OUTPUT_BLOCKS * WEIGHT_VECTOR_BYTES
PWP_PHASE_BYTES = PATTERNS * OUTPUT_BLOCKS * PWP_VECTOR_BYTES
BASELINE_LOAD = (WEIGHT_PHASE_BYTES + DRAM_BYTES_PER_CYCLE - 1) // DRAM_BYTES_PER_CYCLE
CANDIDATE_LOAD = (
    WEIGHT_PHASE_BYTES + PWP_PHASE_BYTES + DRAM_BYTES_PER_CYCLE - 1
) // DRAM_BYTES_PER_CYCLE
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << PARTITION_BITS))
PORTS = (
    ("WIDE_96B_WEIGHT_144B_PWP", 1, 1),
    ("SHARED_96B_PORT", 1, 2),
    ("SHARED_32B_PORT", 3, 5),
)
WORK_FIELDS = (
    "partition_vectors", "nonzero_partition_vectors",
    "baseline_bit_sparse_vector_ops", "nearest_signed_vector_ops",
    "nearest_pwp_vector_ops", "nearest_correction_vector_ops",
    "exact_pattern_hits", "matcher_rows", "onehot_rows", "zero_rows",
)


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


def add_fields(target, source):
    for field in WORK_FIELDS:
        target[field] += source[field]


def unpack_histograms(record):
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
    require(not any(negative), "unexpected negative heldout plane")
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
    histograms = [dict() for _ in range(PARTITIONS)]
    for row in range(ROWS):
        row_base = row * TILES
        for tile in range(TILES):
            packed_tile = masks[row_base + tile]
            partition_base = tile * PARTITIONS_PER_TILE
            for subtile in range(PARTITIONS_PER_TILE):
                value = (packed_tile >> (subtile * PARTITION_BITS)) & 0xffff
                histogram = histograms[partition_base + subtile]
                histogram[value] = histogram.get(value, 0) + 1
    require(all(sum(histogram.values()) == ROWS for histogram in histograms),
            "per-partition vector conservation drift")
    return histograms


def evaluate(histogram, centers):
    center_set = frozenset(centers)
    metrics = Counter()
    used = set()
    for value, count in histogram.items():
        population = POPCOUNT[value]
        metrics["partition_vectors"] += count
        metrics["baseline_bit_sparse_vector_ops"] += count * population
        if value == 0:
            metrics["zero_rows"] += count
        elif population == 1:
            metrics["nonzero_partition_vectors"] += count
            metrics["onehot_rows"] += count
        else:
            metrics["nonzero_partition_vectors"] += count
            metrics["matcher_rows"] += count
        if value != 0 and value in center_set:
            metrics["exact_pattern_hits"] += count
        best_distance, best_center = min(
            (POPCOUNT[value ^ center], center) for center in centers)
        if 1 + best_distance < population:
            metrics["nearest_signed_vector_ops"] += count * (1 + best_distance)
            metrics["nearest_pwp_vector_ops"] += count
            metrics["nearest_correction_vector_ops"] += count * best_distance
            used.add(best_center)
        else:
            metrics["nearest_signed_vector_ops"] += count * population
            metrics["nearest_correction_vector_ops"] += count * population
    require(metrics["nearest_pwp_vector_ops"] +
            metrics["nearest_correction_vector_ops"] ==
            metrics["nearest_signed_vector_ops"],
            "PWP/correction conservation drift")
    require(metrics["zero_rows"] + metrics["onehot_rows"] +
            metrics["matcher_rows"] == metrics["partition_vectors"],
            "row class conservation drift")
    result = dict(metrics)
    for field in WORK_FIELDS:
        result.setdefault(field, 0)
    result["used_centers"] = len(used)
    return result


def phase_service(phase, weight_cycles, pwp_cycles, next_phase):
    dense_compute = (
        phase["partition_vectors"] * PARTITION_BITS * OUTPUT_BLOCKS * weight_cycles)
    baseline_compute = (
        phase["baseline_bit_sparse_vector_ops"] * OUTPUT_BLOCKS * weight_cycles)
    candidate_compute = OUTPUT_BLOCKS * (
        phase["nearest_correction_vector_ops"] * weight_cycles +
        phase["nearest_pwp_vector_ops"] * pwp_cycles)
    matcher = phase["matcher_rows"] + MATCHER_FILL
    packer = ((phase["nearest_pwp_vector_ops"] + PACKER_UNITS - 1) //
              PACKER_UNITS + PACKER_FILL)
    next_baseline = BASELINE_LOAD if next_phase else 0
    next_candidate = CANDIDATE_LOAD if next_phase else 0
    values = {
        "compute": candidate_compute,
        "matcher": matcher,
        "packer": packer,
        "dma": next_candidate,
    }
    maximum = max(values.values())
    winners = sorted(name for name, value in values.items() if value == maximum)
    return {
        "dense_compute": dense_compute,
        "baseline_compute": baseline_compute,
        "candidate_compute": candidate_compute,
        "matcher": matcher,
        "packer": packer,
        "next_baseline": next_baseline,
        "next_candidate": next_candidate,
        "dense_service": max(dense_compute, next_baseline) + COMPUTE_TAIL,
        "baseline_service": max(baseline_compute, next_baseline) + COMPUTE_TAIL,
        "candidate_service": maximum + COMPUTE_TAIL,
        "binding": winners[-1],
        "binding_tie_count": len(winners),
        "fully_serial_candidate_service": (
            candidate_compute + matcher + packer + next_candidate + COMPUTE_TAIL),
        "fully_serial_baseline_service": (
            baseline_compute + next_baseline + COMPUTE_TAIL),
    }


def nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[int(math.ceil(fraction * len(ordered))) - 1]


def almost_equal(left, right, label):
    require(abs(float(left) - float(right)) <= 1e-12 * max(1.0, abs(float(right))),
            label + " float drift")


def compare_config(independent, producer):
    integer_fields = (
        "dense_cycles", "bit_sparse_cycles", "candidate_cycles",
        "matcher_service_cycles", "packer_service_cycles",
        "candidate_compute_bound_phases", "candidate_matcher_bound_phases",
        "candidate_packer_bound_phases", "candidate_dma_bound_phases",
        "candidate_cycle_p50_nearest_rank", "candidate_cycle_p95_nearest_rank",
        "weight_vector_service_cycles", "pwp_vector_service_cycles",
    )
    float_fields = (
        "candidate_speedup_vs_dense", "candidate_speedup_vs_bit_sparse",
        "bit_sparse_speedup_vs_dense",
    )
    for field in integer_fields:
        require(independent[field] == producer[field],
                producer["name"] + " " + field + " drift")
    for field in float_fields:
        almost_equal(independent[field], producer[field],
                     producer["name"] + " " + field)
    require(len(independent["per_sample"]) == len(producer["per_sample"]) == 5,
            "per-sample configuration extent drift")
    for left, right in zip(independent["per_sample"], producer["per_sample"]):
        require(left["sample_id"] == right["sample_id"], "sample order drift")
        for field in (
                "dense_cycles", "bit_sparse_cycles", "candidate_cycles",
                "matcher_service_cycles", "packer_service_cycles",
                "candidate_compute_bound_phases", "candidate_matcher_bound_phases",
                "candidate_packer_bound_phases", "candidate_dma_bound_phases",
                "maximum_phase_cycles"):
            require(left[field] == right[field], "per-sample config integer drift")
        for field in (
                "candidate_speedup_vs_dense", "candidate_speedup_vs_bit_sparse",
                "bit_sparse_speedup_vs_dense"):
            almost_equal(left[field], right[field], "per-sample " + field)


def reconstruct():
    for name, path in (("m76", M76), ("m72", M72), ("m40", M40)):
        require(sha256_path(path) == EXPECTED_SHA[name], name + " root SHA drift")
    m76 = strict_json(M76)
    m72 = strict_json(M72)
    manifest = strict_json(M40)
    require(m72["status"] ==
            "PASS_M72_VALID825_INTERNAL_SCREEN_NOT_TRAIN_CATALOG_CYCLES_RTL_UNADMITTED",
            "M72 status drift")
    require(m76["status"] ==
            "PASS_M76_ISOLATED_PATTERN_ENGINE_CYCLE_SIM_VALID825_INTERNAL_RTL_SYSTEM_UNADMITTED",
            "M76 status drift")
    require(m72["split"]["heldout_samples_within_valid825"] == [5, 6, 7, 8, 9] and
            m72["split"]["train_catalog_eligible"] is False,
            "M72 internal split boundary drift")
    records = [row for row in manifest["records"] if row["sample_id"] >= 5]
    require(len(records) == 20, "M40 heldout record extent drift")
    operator_names = sorted(set(row["operator"] for row in manifest["records"]))
    require(len(operator_names) == 4 and
            [row["operator"] for row in m72["operators"]] == operator_names,
            "operator identity/order drift")
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    m72_partitions = []
    for operator_index_value, operator in enumerate(m72["operators"]):
        require(len(operator["partitions"]) == PARTITIONS,
                "M72 partition extent drift")
        partition_rows = []
        for partition, row in enumerate(operator["partitions"]):
            require(row["partition"] == partition and len(row["centers_hex"]) == 16,
                    "M72 partition/codebook identity drift")
            centers = [int(value, 16) for value in row["centers_hex"]]
            require(len(set(centers)) == 16 and
                    all(0 <= center < (1 << PARTITION_BITS) for center in centers),
                    "M72 center uniqueness/range drift")
            partition_rows.append((centers, row))
        m72_partitions.append(partition_rows)

    phases_by_sample = dict((sample, []) for sample in range(5, 10))
    sample_operator_work = {}
    partition_heldout = [[Counter() for _ in range(PARTITIONS)] for _ in range(4)]
    phase_work_digest = hashlib.sha256()
    total_work = Counter()
    for record_number, record in enumerate(records):
        sample = record["sample_id"]
        op = operator_index[record["operator"]]
        histograms = unpack_histograms(record)
        sample_op = Counter()
        for partition, histogram in enumerate(histograms):
            centers, _ = m72_partitions[op][partition]
            metrics = evaluate(histogram, centers)
            metrics["sample_id"] = sample
            metrics["operator_index"] = op
            metrics["operator"] = record["operator"]
            metrics["partition"] = partition
            phases_by_sample[sample].append(metrics)
            add_fields(sample_op, metrics)
            add_fields(partition_heldout[op][partition], metrics)
            add_fields(total_work, metrics)
            phase_work_digest.update(canonical_bytes({
                key: metrics[key] for key in (
                    "sample_id", "operator_index", "partition") + WORK_FIELDS
            }))
        sample_operator_work[(sample, op)] = sample_op
        print("[M76 INDEPENDENT] record={}/20 sample={} op={} phases={}".format(
            record_number + 1, sample, op, len(histograms)), flush=True)

    for op in range(4):
        operator_total = Counter()
        for partition in range(PARTITIONS):
            reconstructed = partition_heldout[op][partition]
            producer = m72_partitions[op][partition][1]["heldout"]
            for field in (
                    "partition_vectors", "nonzero_partition_vectors",
                    "baseline_bit_sparse_vector_ops", "nearest_signed_vector_ops",
                    "nearest_pwp_vector_ops", "nearest_correction_vector_ops",
                    "exact_pattern_hits"):
                require(reconstructed[field] == producer[field],
                        "M72 heldout partition reconstruction drift")
            add_fields(operator_total, reconstructed)
        producer_operator = m72["operators"][op]
        for field in (
                "partition_vectors", "nonzero_partition_vectors",
                "baseline_bit_sparse_vector_ops", "nearest_signed_vector_ops",
                "nearest_pwp_vector_ops", "nearest_correction_vector_ops",
                "exact_pattern_hits"):
            require(operator_total[field] == producer_operator[field],
                    "M72 operator reconstruction drift")

    require(total_work["partition_vectors"] == 25920000 and
            total_work["baseline_bit_sparse_vector_ops"] == 46432637 and
            total_work["nearest_signed_vector_ops"] == 30889399 and
            total_work["nearest_pwp_vector_ops"] == 7371217 and
            total_work["nearest_correction_vector_ops"] == 23518182 and
            total_work["matcher_rows"] == 9932028 and
            total_work["zero_rows"] == 12212045 and
            total_work["onehot_rows"] == 3775927,
            "all-heldout work conservation drift")

    phase_cycle_digest = hashlib.sha256()
    configurations = []
    for port_name, weight_cycles, pwp_cycles in PORTS:
        per_sample = []
        aggregate = Counter()
        serial_candidate_total = 0
        serial_baseline_total = 0
        tie_phases = 0
        min_compute_margin = None
        sample_operator_cycles = []
        for sample in range(5, 10):
            phases = phases_by_sample[sample]
            require(len(phases) == 4 * PARTITIONS, "sample phase extent drift")
            sample_totals = Counter()
            sample_totals["bit_sparse_cycles"] += BASELINE_LOAD
            sample_totals["candidate_cycles"] += CANDIDATE_LOAD
            sample_serial_candidate = CANDIDATE_LOAD
            sample_serial_baseline = BASELINE_LOAD
            op_totals = [Counter() for _ in range(4)]
            maximum_phase = 0
            for phase_index, phase in enumerate(phases):
                service = phase_service(
                    phase, weight_cycles, pwp_cycles,
                    phase_index + 1 < len(phases))
                sample_totals["dense_cycles"] += service["dense_service"]
                sample_totals["bit_sparse_cycles"] += service["baseline_service"]
                sample_totals["candidate_cycles"] += service["candidate_service"]
                sample_totals["matcher_service_cycles"] += service["matcher"]
                sample_totals["packer_service_cycles"] += service["packer"]
                sample_totals["candidate_{}_bound_phases".format(
                    service["binding"])] += 1
                sample_serial_candidate += service["fully_serial_candidate_service"]
                sample_serial_baseline += service["fully_serial_baseline_service"]
                tie_phases += int(service["binding_tie_count"] != 1)
                margin = service["candidate_compute"] - max(
                    service["matcher"], service["packer"], service["next_candidate"])
                min_compute_margin = margin if min_compute_margin is None else min(
                    min_compute_margin, margin)
                maximum_phase = max(maximum_phase, service["candidate_service"])
                op_counter = op_totals[phase["operator_index"]]
                op_counter["dense_cycles"] += service["dense_service"]
                op_counter["bit_sparse_cycles"] += service["baseline_service"]
                op_counter["candidate_cycles"] += service["candidate_service"]
                op_counter["matcher_cycles"] += service["matcher"]
                op_counter["packer_cycles"] += service["packer"]
                op_counter["phases"] += 1
                phase_cycle_digest.update(canonical_bytes({
                    "sample": sample,
                    "operator": phase["operator_index"],
                    "partition": phase["partition"],
                    "port": port_name,
                    "dense_compute": service["dense_compute"],
                    "bit_sparse_compute": service["baseline_compute"],
                    "candidate_compute": service["candidate_compute"],
                    "matcher": service["matcher"],
                    "packer": service["packer"],
                    "next_candidate": service["next_candidate"],
                    "candidate_service": service["candidate_service"],
                    "binding": service["binding"],
                }))
            sample_totals["maximum_phase_cycles"] = maximum_phase
            sample_totals["sample_id"] = sample
            sample_totals["candidate_speedup_vs_dense"] = (
                sample_totals["dense_cycles"] / float(sample_totals["candidate_cycles"]))
            sample_totals["candidate_speedup_vs_bit_sparse"] = (
                sample_totals["bit_sparse_cycles"] /
                float(sample_totals["candidate_cycles"]))
            sample_totals["bit_sparse_speedup_vs_dense"] = (
                sample_totals["dense_cycles"] /
                float(sample_totals["bit_sparse_cycles"]))
            for field in (
                    "candidate_compute_bound_phases",
                    "candidate_matcher_bound_phases",
                    "candidate_packer_bound_phases",
                    "candidate_dma_bound_phases"):
                sample_totals.setdefault(field, 0)
            per_sample.append(dict(sample_totals))
            for field in (
                    "dense_cycles", "bit_sparse_cycles", "candidate_cycles",
                    "matcher_service_cycles", "packer_service_cycles",
                    "candidate_compute_bound_phases", "candidate_matcher_bound_phases",
                    "candidate_packer_bound_phases", "candidate_dma_bound_phases"):
                aggregate[field] += sample_totals[field]
            serial_candidate_total += sample_serial_candidate
            serial_baseline_total += sample_serial_baseline
            for op, counters in enumerate(op_totals):
                row = dict(counters)
                row.update({"sample_id": sample, "operator_index": op})
                sample_operator_cycles.append(row)
        config = dict(aggregate)
        config.update({
            "name": port_name,
            "weight_vector_service_cycles": weight_cycles,
            "pwp_vector_service_cycles": pwp_cycles,
            "candidate_speedup_vs_dense": (
                aggregate["dense_cycles"] / float(aggregate["candidate_cycles"])),
            "candidate_speedup_vs_bit_sparse": (
                aggregate["bit_sparse_cycles"] / float(aggregate["candidate_cycles"])),
            "bit_sparse_speedup_vs_dense": (
                aggregate["dense_cycles"] / float(aggregate["bit_sparse_cycles"])),
            "candidate_cycle_p50_nearest_rank": nearest_rank(
                [row["candidate_cycles"] for row in per_sample], 0.50),
            "candidate_cycle_p95_nearest_rank": nearest_rank(
                [row["candidate_cycles"] for row in per_sample], 0.95),
            "per_sample": per_sample,
            "per_sample_operator": sample_operator_cycles,
            "fully_serial_candidate_cycles": serial_candidate_total,
            "fully_serial_bit_sparse_cycles": serial_baseline_total,
            "fully_serial_candidate_speedup_vs_bit_sparse": (
                serial_baseline_total / float(serial_candidate_total)),
            "producer_bit_sparse_over_fully_serial_candidate": (
                aggregate["bit_sparse_cycles"] / float(serial_candidate_total)),
            "binding_tie_phases": tie_phases,
            "minimum_compute_margin_over_matcher_packer_dma": min_compute_margin,
        })
        configurations.append(config)

    producer_configs = dict((row["name"], row) for row in m76["configurations"])
    require(set(producer_configs) == set(row[0] for row in PORTS),
            "M76 configuration set drift")
    for config in configurations:
        compare_config(config, producer_configs[config["name"]])

    traffic = {
        "baseline_weight_sram_read_bytes": (
            total_work["baseline_bit_sparse_vector_ops"] * OUTPUT_BLOCKS *
            WEIGHT_VECTOR_BYTES),
        "candidate_weight_correction_sram_read_bytes": (
            total_work["nearest_correction_vector_ops"] * OUTPUT_BLOCKS *
            WEIGHT_VECTOR_BYTES),
        "candidate_pwp_sram_read_bytes": (
            total_work["nearest_pwp_vector_ops"] * OUTPUT_BLOCKS *
            PWP_VECTOR_BYTES),
        "baseline_weight_dram_prefetch_bytes": (
            5 * 4 * PARTITIONS * WEIGHT_PHASE_BYTES),
        "candidate_weight_plus_pwp_dram_prefetch_bytes": (
            5 * 4 * PARTITIONS * (WEIGHT_PHASE_BYTES + PWP_PHASE_BYTES)),
    }
    require(traffic == m76["traffic_all_five_samples"], "M76 traffic drift")
    work_map = {
        "partition_vectors": total_work["partition_vectors"],
        "baseline_bit_sparse_vector_ops_per_output_block":
            total_work["baseline_bit_sparse_vector_ops"],
        "candidate_vector_ops_per_output_block":
            total_work["nearest_signed_vector_ops"],
        "pwp_vector_ops_per_output_block": total_work["nearest_pwp_vector_ops"],
        "correction_vector_ops_per_output_block":
            total_work["nearest_correction_vector_ops"],
        "matcher_rows": total_work["matcher_rows"],
        "zero_rows": total_work["zero_rows"],
        "onehot_rows": total_work["onehot_rows"],
    }
    require(work_map == m76["work_conservation"], "M76 work map drift")
    require(m76["pipeline"] == {
        "matcher": "16-comparator 1-D systolic, one input row per cycle after 16-cycle fill",
        "zero_onehot_prefilter": True,
        "packer_units": 8,
        "weight_phase_bytes": 12288,
        "pwp_phase_bytes": 18432,
        "combined_double_buffer_bytes": 61440,
        "dram_bytes_per_cycle": 32,
        "weight_phase_load_cycles": 384,
        "candidate_phase_load_cycles": 960,
    }, "M76 pipeline constants drift")
    require(m76["admission"]["valid825_internal_only"] is True and
            all(m76["admission"][name] is False for name in (
                "independent_validation", "train_catalog", "accuracy",
                "rtl_cycle_evidence", "synopsys_ppa",
                "full_network_or_system_speedup", "date_headline")),
            "M76 claim boundary widened")

    sample_operator_payload = []
    for sample in range(5, 10):
        for op in range(4):
            row = {field: sample_operator_work[(sample, op)][field]
                   for field in WORK_FIELDS}
            row.update({
                "sample_id": sample,
                "operator_index": op,
                "operator": operator_names[op],
            })
            sample_operator_payload.append(row)
    baseline_sram = traffic["baseline_weight_sram_read_bytes"]
    candidate_sram = (traffic["candidate_weight_correction_sram_read_bytes"] +
                      traffic["candidate_pwp_sram_read_bytes"])
    payload = {
        "schema": "m76_independent_reconstruction_v1",
        "status": "PASS_M76_INDEPENDENT_M72_M40_PHASE_RECONSTRUCTION",
        "identity": {
            "m76_sha256": EXPECTED_SHA["m76"],
            "m72_sha256": EXPECTED_SHA["m72"],
            "m40_manifest_sha256": EXPECTED_SHA["m40"],
            "production_simulator_imported": False,
        },
        "population": {
            "samples": 5,
            "operators": 4,
            "partitions_per_operator": 432,
            "phases": 8640,
            "output_blocks": OUTPUT_BLOCKS,
            "output_lanes": OUTPUT_LANES,
        },
        "work_conservation": work_map,
        "sample_operator_work": sample_operator_payload,
        "phase_work_sha256": phase_work_digest.hexdigest(),
        "phase_cycle_all_three_ports_sha256": phase_cycle_digest.hexdigest(),
        "configurations": configurations,
        "traffic": dict(traffic, **{
            "candidate_total_sram_read_bytes": candidate_sram,
            "candidate_sram_read_reduction_fraction": (
                (baseline_sram - candidate_sram) / float(baseline_sram)),
            "candidate_over_baseline_dram_prefetch": (
                traffic["candidate_weight_plus_pwp_dram_prefetch_bytes"] /
                float(traffic["baseline_weight_dram_prefetch_bytes"])),
        }),
        "buffer_accounting": {
            "one_weight_phase_bytes": WEIGHT_PHASE_BYTES,
            "one_pwp_phase_bytes": PWP_PHASE_BYTES,
            "weight_double_buffer_bytes": 2 * WEIGHT_PHASE_BYTES,
            "pwp_double_buffer_bytes": 2 * PWP_PHASE_BYTES,
            "combined_payload_double_buffer_bytes": 2 * (
                WEIGHT_PHASE_BYTES + PWP_PHASE_BYTES),
            "pattern_table_bytes_excluded_from_payload_double_buffer":
                m72["hardware_capacity"]["pattern_table_bytes"],
            "all_pwp_catalog_bytes_not_resident_in_double_buffer":
                m72["hardware_capacity"]["all_pwp_bytes_bit_tight"],
        },
        "audit_observations": {
            "output_block_factor_applied_to_dense_sparse_candidate_and_sram": True,
            "assignment_matcher_and_packer_shared_across_output_blocks": True,
            "pwp_vector_bytes": PWP_VECTOR_BYTES,
            "weight_vector_bytes": WEIGHT_VECTOR_BYTES,
            "pwp_over_weight_vector_byte_ratio": 1.5,
            "prefetch_phase_count": 5 * 4 * PARTITIONS,
            "every_sample_replays_all_prefetch_phases": True,
            "producer_dense_initial_prefetch_cycles_omitted": 5 * BASELINE_LOAD,
            "producer_candidate_initial_prefetch_cycles_included": 5 * CANDIDATE_LOAD,
            "all_candidate_phases_compute_bound_under_perfect_overlap": all(
                row["candidate_compute_bound_phases"] == 8640
                for row in configurations),
            "matcher_packer_dma_overlap_proved_by_rtl_or_queue_schedule": False,
            "valid825_internal_not_independent_or_train": True,
        },
    }
    return payload


def validate_review(payload):
    if not REVIEW.exists():
        return
    review = strict_json(REVIEW)
    require(review["status"] ==
            "M76_COUNTS_REPRODUCED_GO_INTERNAL_DSE_NO_GO_RTL_SYSTEM_DATE_HEADLINE",
            "review status drift")
    require(len(review["findings"]["p0"]) == 0 and
            len(review["findings"]["p1"]) == 6,
            "review P0/P1 count drift")
    scores = review["scores"]
    require(scores["hardware_innovation"] == 48 and
            scores["performance_advantage"] == 62 and
            scores["evidence_quality"] == 72,
            "review score drift")
    require(review["claim_boundary"]["dense_13p42x_date_headline"] is False,
            "review admitted dense headline")
    if RECONSTRUCTION.exists():
        stored = strict_json(RECONSTRUCTION)
        require(stored == payload, "stored independent reconstruction drift")
    if VALIDATION_RECEIPT.exists():
        receipt = strict_json(VALIDATION_RECEIPT)
        require(receipt["status"] ==
                "PASS_M76_PHI_PATTERN_ENGINE_INDEPENDENT_HAMMER_VALIDATION" and
                receipt["identity"]["review_sha256"] == sha256_path(REVIEW) and
                receipt["identity"]["reconstruction_sha256"] ==
                sha256_path(RECONSTRUCTION) and
                receipt["identity"]["validator_sha256"] ==
                sha256_path(Path(__file__)),
                "validation receipt identity drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = reconstruct()
    validate_review(payload)
    if args.output is not None:
        require(not args.output.exists(), "refusing reconstruction overwrite")
        require(args.output.resolve().parent == HERE.resolve(),
                "reconstruction output must stay in review directory")
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wide = payload["configurations"][0]
    print("PASS M76 independent: phases=8640 work={} wide_cycles={} "
          "wide_vs_sparse={:.9f}x dense={:.9f}x P0=0 P1=6".format(
              payload["work_conservation"]["candidate_vector_ops_per_output_block"],
              wide["candidate_cycles"], wide["candidate_speedup_vs_bit_sparse"],
              wide["candidate_speedup_vs_dense"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M76 independent: {}".format(error), flush=True)
        raise SystemExit(1)
