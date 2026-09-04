#!/usr/bin/env python3
"""Cycle-simulate a standalone Phi-style 96-lane pattern engine.

The simulator replays heldout samples 5--9 independently.  It compares dense,
bit-sparse, and exact one-PWP-plus-signed-correction execution on the same 96
output lanes.  Pattern assignment is shared across eight output blocks; each
partition phase overlaps its compute, one-row/cycle systolic matcher, eight-way
assignment packer, and double-buffered next-partition DMA.  PWP and correction
SRAM service widths are explicit DSE parameters.

This is an isolated-module cycle estimate on a valid825-internal screen.  It is
not full-network/system performance, RTL timing, accuracy, or a DATE headline.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M72_ANALYZER = HW / (
    "system_simulator/scripts/analyze_m72_phi_kmeans_k16q16_heldout.py")
M72_RESULT = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
EXPECTED_M72_RESULT_SHA256 = (
    "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133")
OUTPUT_BLOCKS = 8
WEIGHT_VECTOR_BYTES = 96
PWP_VECTOR_BYTES = 144
PATTERNS = 16
PARTITION_BITS = 16
DRAM_BYTES_PER_CYCLE = 32
WEIGHT_PHASE_BYTES = PARTITION_BITS * OUTPUT_BLOCKS * WEIGHT_VECTOR_BYTES
PWP_PHASE_BYTES = PATTERNS * OUTPUT_BLOCKS * PWP_VECTOR_BYTES
MATCHER_PIPELINE_CYCLES = 16
PACKER_PIPELINE_CYCLES = 4
COMPUTE_TAIL_CYCLES = 2
PORTS = (
    {"name": "WIDE_96B_WEIGHT_144B_PWP", "weight_cycles": 1, "pwp_cycles": 1},
    {"name": "SHARED_96B_PORT", "weight_cycles": 1, "pwp_cycles": 2},
    {"name": "SHARED_32B_PORT", "weight_cycles": 3, "pwp_cycles": 5},
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_m72():
    spec = importlib.util.spec_from_file_location("m76_m72", str(M72_ANALYZER))
    require(spec is not None and spec.loader is not None, "cannot import M72")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_per_sample_histograms(m72, m43, manifest, operator_names):
    histograms = defaultdict(Counter)
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    mask16 = (1 << PARTITION_BITS) - 1
    heldout_records = [row for row in manifest["records"] if row["sample_id"] >= 5]
    require(len(heldout_records) == 20, "M76 heldout record extent drift")
    for record_index, record in enumerate(heldout_records):
        sample = record["sample_id"]
        op = operator_index[record["operator"]]
        masks = m43.unpack_record_masks(m72.MANIFEST_PATH.parent, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // PARTITION_BITS)
                for subtile in range(m43.TILE_BITS // PARTITION_BITS):
                    value = (value256 >> (subtile * PARTITION_BITS)) & mask16
                    histograms[(sample, op, partition_base + subtile)][value] += 1
        print("[M76 HIST] {}/20 sample={} operator={}".format(
            record_index + 1, sample, record["operator"]), flush=True)
    return histograms


def nearest_metrics(m72, counter, centers):
    result = m72.evaluate(counter, centers)
    matcher_rows = sum(
        count for value, count in counter.items() if m72.POPCOUNT[value] >= 2)
    onehot_rows = sum(
        count for value, count in counter.items() if m72.POPCOUNT[value] == 1)
    zero_rows = counter.get(0, 0)
    require(zero_rows + onehot_rows + matcher_rows ==
            result["partition_vectors"], "M76 row-class conservation failure")
    result["matcher_rows"] = matcher_rows
    result["onehot_rows"] = onehot_rows
    result["zero_rows"] = zero_rows
    return result


def replay_sample(phases, port):
    baseline_cycle = math.ceil(WEIGHT_PHASE_BYTES / DRAM_BYTES_PER_CYCLE)
    candidate_cycle = math.ceil(
        (WEIGHT_PHASE_BYTES + PWP_PHASE_BYTES) / DRAM_BYTES_PER_CYCLE)
    dense_cycles = baseline_cycles = candidate_cycles = 0
    matcher_cycles = packer_cycles = 0
    # All three same-scope paths pay the first phase prefetch.  Earlier M76 r1
    # omitted this 384-cycle charge from dense only; the effect was tiny but
    # made the comparison asymmetric.
    dense_cycles += baseline_cycle
    baseline_cycles += baseline_cycle
    candidate_cycles += candidate_cycle
    phase_rows = []
    for phase_index, phase in enumerate(phases):
        dense_compute = (
            phase["partition_vectors"] * PARTITION_BITS * OUTPUT_BLOCKS *
            port["weight_cycles"])
        baseline_compute = (
            phase["baseline_bit_sparse_vector_ops"] * OUTPUT_BLOCKS *
            port["weight_cycles"])
        candidate_compute = OUTPUT_BLOCKS * (
            phase["nearest_correction_vector_ops"] * port["weight_cycles"]
            + phase["nearest_pwp_vector_ops"] * port["pwp_cycles"])
        matcher = phase["matcher_rows"] + MATCHER_PIPELINE_CYCLES
        packer = (math.ceil(phase["nearest_pwp_vector_ops"] / 8.0)
                  + PACKER_PIPELINE_CYCLES)
        next_baseline_load = baseline_cycle if phase_index + 1 < len(phases) else 0
        next_candidate_load = candidate_cycle if phase_index + 1 < len(phases) else 0
        dense_service = max(dense_compute, next_baseline_load) + COMPUTE_TAIL_CYCLES
        baseline_service = max(baseline_compute, next_baseline_load) + COMPUTE_TAIL_CYCLES
        candidate_service = max(
            candidate_compute, matcher, packer, next_candidate_load
        ) + COMPUTE_TAIL_CYCLES
        dense_cycles += dense_service
        baseline_cycles += baseline_service
        candidate_cycles += candidate_service
        matcher_cycles += matcher
        packer_cycles += packer
        phase_rows.append({
            "phase_index": phase_index,
            "dense_compute_cycles": dense_compute,
            "bit_sparse_compute_cycles": baseline_compute,
            "candidate_compute_cycles": candidate_compute,
            "matcher_cycles": matcher,
            "packer_cycles": packer,
            "next_candidate_load_cycles": next_candidate_load,
            "candidate_phase_service_cycles": candidate_service,
            "candidate_binding": max(
                (candidate_compute, "compute"),
                (matcher, "matcher"),
                (packer, "packer"),
                (next_candidate_load, "dma"))[1],
        })
    return {
        "dense_cycles": dense_cycles,
        "bit_sparse_cycles": baseline_cycles,
        "candidate_cycles": candidate_cycles,
        "candidate_speedup_vs_dense": dense_cycles / float(candidate_cycles),
        "candidate_speedup_vs_bit_sparse": baseline_cycles / float(candidate_cycles),
        "bit_sparse_speedup_vs_dense": dense_cycles / float(baseline_cycles),
        "matcher_service_cycles": matcher_cycles,
        "packer_service_cycles": packer_cycles,
        "candidate_compute_bound_phases": sum(
            row["candidate_binding"] == "compute" for row in phase_rows),
        "candidate_matcher_bound_phases": sum(
            row["candidate_binding"] == "matcher" for row in phase_rows),
        "candidate_packer_bound_phases": sum(
            row["candidate_binding"] == "packer" for row in phase_rows),
        "candidate_dma_bound_phases": sum(
            row["candidate_binding"] == "dma" for row in phase_rows),
        "maximum_phase_cycles": max(
            row["candidate_phase_service_cycles"] for row in phase_rows),
    }


def nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1,
                              int(math.ceil(fraction * len(ordered))) - 1))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M76 output overwrite")
    analyzer_start_sha = sha256(Path(__file__).resolve())
    m72_start_sha = sha256(M72_ANALYZER)
    require(sha256(M72_RESULT) == EXPECTED_M72_RESULT_SHA256,
            "M76 M72 result identity drift")
    m72 = load_m72()
    result = m72.strict_json(M72_RESULT)
    require(result["split"]["train_catalog_eligible"] is False,
            "M76 requires internal-screen identity")
    manifest = m72.strict_json(m72.MANIFEST_PATH)
    operator_names = sorted(set(row["operator"] for row in manifest["records"]))
    m43 = m72.load_m43()
    histograms = collect_per_sample_histograms(
        m72, m43, manifest, operator_names)

    sample_phases = defaultdict(list)
    conservation = Counter()
    for sample in range(5, 10):
        for op, operator in enumerate(operator_names):
            partitions = result["operators"][op]["partitions"]
            for partition, row in enumerate(partitions):
                centers = [int(value, 16) for value in row["centers_hex"]]
                metrics = nearest_metrics(
                    m72, histograms[(sample, op, partition)], centers)
                metrics.update({
                    "sample_id": sample,
                    "operator": operator,
                    "operator_index": op,
                    "partition": partition,
                })
                sample_phases[sample].append(metrics)
                conservation.update(metrics)
    require(conservation["partition_vectors"] == 25920000,
            "M76 partition-vector conservation drift")
    require(conservation["baseline_bit_sparse_vector_ops"] == 46432637,
            "M76 baseline conservation drift")
    require(conservation["nearest_signed_vector_ops"] == 30889399,
            "M76 candidate conservation drift")

    configurations = []
    for port in PORTS:
        per_sample = []
        totals = Counter()
        for sample in range(5, 10):
            replay = replay_sample(sample_phases[sample], port)
            replay["sample_id"] = sample
            per_sample.append(replay)
            for key in ("dense_cycles", "bit_sparse_cycles", "candidate_cycles",
                        "matcher_service_cycles", "packer_service_cycles",
                        "candidate_compute_bound_phases",
                        "candidate_matcher_bound_phases",
                        "candidate_packer_bound_phases",
                        "candidate_dma_bound_phases"):
                totals[key] += replay[key]
        configuration = {
            "name": port["name"],
            "weight_vector_service_cycles": port["weight_cycles"],
            "pwp_vector_service_cycles": port["pwp_cycles"],
            "dense_cycles": totals["dense_cycles"],
            "bit_sparse_cycles": totals["bit_sparse_cycles"],
            "candidate_cycles": totals["candidate_cycles"],
            "candidate_speedup_vs_dense": (
                totals["dense_cycles"] / totals["candidate_cycles"]),
            "candidate_speedup_vs_bit_sparse": (
                totals["bit_sparse_cycles"] / totals["candidate_cycles"]),
            "bit_sparse_speedup_vs_dense": (
                totals["dense_cycles"] / totals["bit_sparse_cycles"]),
            "matcher_service_cycles": totals["matcher_service_cycles"],
            "packer_service_cycles": totals["packer_service_cycles"],
            "candidate_compute_bound_phases":
                totals["candidate_compute_bound_phases"],
            "candidate_matcher_bound_phases":
                totals["candidate_matcher_bound_phases"],
            "candidate_packer_bound_phases":
                totals["candidate_packer_bound_phases"],
            "candidate_dma_bound_phases":
                totals["candidate_dma_bound_phases"],
            "candidate_cycle_p50_nearest_rank": nearest_rank(
                [row["candidate_cycles"] for row in per_sample], 0.50),
            "candidate_cycle_p95_nearest_rank": nearest_rank(
                [row["candidate_cycles"] for row in per_sample], 0.95),
            "per_sample": per_sample,
        }
        configurations.append(configuration)
        print("[M76] {} dense={:.6f}x bit_sparse={:.6f}x cycles={}".format(
            configuration["name"], configuration["candidate_speedup_vs_dense"],
            configuration["candidate_speedup_vs_bit_sparse"],
            configuration["candidate_cycles"]), flush=True)

    require(sha256(Path(__file__).resolve()) == analyzer_start_sha,
            "M76 analyzer source changed during execution")
    require(sha256(M72_ANALYZER) == m72_start_sha,
            "M76 M72 dependency changed during execution")
    payload = {
        "schema": "m76_phi_style_pattern_engine_cycle_simulator_valid825_internal_v1",
        "status": "PASS_M76_ISOLATED_PATTERN_ENGINE_CYCLE_SIM_VALID825_INTERNAL_RTL_SYSTEM_UNADMITTED",
        "identity": {
            "analyzer_start_end_sha256": analyzer_start_sha,
            "m72_analyzer_start_end_sha256": m72_start_sha,
            "m72_result_sha256": sha256(M72_RESULT),
        },
        "scope": {
            "operators": operator_names,
            "samples": [5, 6, 7, 8, 9],
            "output_lanes": 96,
            "output_blocks_replayed_sequentially": OUTPUT_BLOCKS,
            "partition_bits": PARTITION_BITS,
            "patterns_per_partition": PATTERNS,
            "pattern_assignment_shared_across_output_blocks": True,
        },
        "pipeline": {
            "matcher": "16-comparator 1-D systolic, one input row per cycle after 16-cycle fill",
            "zero_onehot_prefilter": True,
            "packer_units": 8,
            "weight_phase_bytes": WEIGHT_PHASE_BYTES,
            "pwp_phase_bytes": PWP_PHASE_BYTES,
            "combined_double_buffer_bytes": 2 * (
                WEIGHT_PHASE_BYTES + PWP_PHASE_BYTES),
            "dram_bytes_per_cycle": DRAM_BYTES_PER_CYCLE,
            "weight_phase_load_cycles": math.ceil(
                WEIGHT_PHASE_BYTES / DRAM_BYTES_PER_CYCLE),
            "candidate_phase_load_cycles": math.ceil(
                (WEIGHT_PHASE_BYTES + PWP_PHASE_BYTES) /
                DRAM_BYTES_PER_CYCLE),
        },
        "work_conservation": {
            "partition_vectors": conservation["partition_vectors"],
            "baseline_bit_sparse_vector_ops_per_output_block":
                conservation["baseline_bit_sparse_vector_ops"],
            "candidate_vector_ops_per_output_block":
                conservation["nearest_signed_vector_ops"],
            "pwp_vector_ops_per_output_block":
                conservation["nearest_pwp_vector_ops"],
            "correction_vector_ops_per_output_block":
                conservation["nearest_correction_vector_ops"],
            "matcher_rows": conservation["matcher_rows"],
            "zero_rows": conservation["zero_rows"],
            "onehot_rows": conservation["onehot_rows"],
        },
        "traffic_all_five_samples": {
            "baseline_weight_sram_read_bytes": (
                conservation["baseline_bit_sparse_vector_ops"] *
                OUTPUT_BLOCKS * WEIGHT_VECTOR_BYTES),
            "candidate_weight_correction_sram_read_bytes": (
                conservation["nearest_correction_vector_ops"] *
                OUTPUT_BLOCKS * WEIGHT_VECTOR_BYTES),
            "candidate_pwp_sram_read_bytes": (
                conservation["nearest_pwp_vector_ops"] *
                OUTPUT_BLOCKS * PWP_VECTOR_BYTES),
            "baseline_weight_dram_prefetch_bytes": (
                5 * 4 * m72.PARTITIONS * WEIGHT_PHASE_BYTES),
            "candidate_weight_plus_pwp_dram_prefetch_bytes": (
                5 * 4 * m72.PARTITIONS *
                (WEIGHT_PHASE_BYTES + PWP_PHASE_BYTES)),
        },
        "configurations": configurations,
        "admission": {
            "same_scope_dense_and_bit_sparse_baselines": True,
            "isolated_module_cycle_simulator_estimate": True,
            "valid825_internal_only": True,
            "independent_validation": False,
            "train_catalog": False,
            "accuracy": False,
            "rtl_cycle_evidence": False,
            "synopsys_ppa": False,
            "full_network_or_system_speedup": False,
            "date_headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M76 configurations={} output={}".format(
        len(configurations), args.output), flush=True)


if __name__ == "__main__":
    main()
