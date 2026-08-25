#!/usr/bin/env python3
"""DSE a precision-elastic PWP bank with exact INT8 checkpoint weights.

The M72 Hamming assignments are replayed on heldout samples 5--9.  For every
operator/partition/pattern/output-block, the selected 16-source INT8 weights
are summed exactly and the minimum signed PWP width is derived.  Width caps
8--12 use a block-local exact bit-sparse escape when a PWP does not fit.

This is an isolated-module, valid825-internal cycle/traffic DSE.  It does not
admit accuracy, RTL timing/PPA, full-network performance, or a paper headline.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M72_ANALYZER = HW / "system_simulator/scripts/analyze_m72_phi_kmeans_k16q16_heldout.py"
M72_RESULT = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41_RESULT = M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json"
EXPECTED_SHA256 = {
    "m72_result": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
}
EXPECTED_WEIGHT_SHA256 = (
    "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
)
PARTITION_BITS = 16
PARTITIONS = 432
PATTERNS = 16
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
OUTPUT_CHANNELS = OUTPUT_BLOCKS * OUTPUT_LANES
WEIGHT_VECTOR_BYTES = OUTPUT_LANES
WEIGHT_PHASE_BYTES = PARTITION_BITS * OUTPUT_BLOCKS * WEIGHT_VECTOR_BYTES
DRAM_BYTES_PER_CYCLE = 32
MATCHER_PIPELINE_CYCLES = 16
PACKER_PIPELINE_CYCLES = 4
COMPUTE_TAIL_CYCLES = 2
WIDTH_CAPS = (8, 9, 10, 11, 12)
PORTS = (
    {"name": "WIDE_PRECISION_ELASTIC", "weight_cycles": 1, "pwp_port_bytes": None},
    {"name": "SHARED_96B", "weight_cycles": 1, "pwp_port_bytes": 96},
    {"name": "SHARED_32B", "weight_cycles": 3, "pwp_port_bytes": 32},
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


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_m72():
    spec = importlib.util.spec_from_file_location("m78_m72", str(M72_ANALYZER))
    require(spec is not None and spec.loader is not None, "cannot import M72")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signed_width(minimum, maximum):
    for width in range(1, 33):
        if minimum >= -(1 << (width - 1)) and maximum <= (1 << (width - 1)) - 1:
            return max(8, width)
    raise ValueError("PWP sum exceeds signed32")


def build_width_catalog(m72_result, m41_result):
    layers = m41_result["layers"]
    require(len(layers) == 4, "M78 M41 layer extent drift")
    width_catalog = []
    width_hist = Counter()
    center_max_hist = Counter()
    outliers = []
    observed_weight_sha = []
    for op, operator in enumerate(m72_result["operators"]):
        payload = next(item for item in layers if item["operator"] == operator["operator"])
        weight_info = next(item for item in payload["payloads"] if item["role"] == "weight")
        weight_path = M41_DIR / weight_info["file"]
        weight_sha = sha256(weight_path)
        require(weight_sha == EXPECTED_WEIGHT_SHA256[op] == weight_info["sha256"],
                "M78 M41 weight identity drift op{}".format(op))
        observed_weight_sha.append(weight_sha)
        weights = np.fromfile(str(weight_path), dtype=np.int8)
        require(weights.size == 6912 * OUTPUT_CHANNELS,
                "M78 INT8 weight extent drift")
        weights = weights.reshape(6912, OUTPUT_CHANNELS).astype(np.int32)
        op_catalog = []
        for partition, row in enumerate(operator["partitions"]):
            require(row["partition"] == partition, "M78 M72 partition order drift")
            source = weights[partition * PARTITION_BITS:(partition + 1) * PARTITION_BITS]
            center_rows = []
            for pattern_index, center_hex in enumerate(row["centers_hex"]):
                center = int(center_hex, 16)
                indices = [bit for bit in range(PARTITION_BITS) if center & (1 << bit)]
                pwp = source[indices].sum(axis=0, dtype=np.int32)
                block_rows = []
                for block in range(OUTPUT_BLOCKS):
                    values = pwp[block * OUTPUT_LANES:(block + 1) * OUTPUT_LANES]
                    minimum = int(values.min())
                    maximum = int(values.max())
                    width = signed_width(minimum, maximum)
                    block_rows.append({
                        "width": width,
                        "minimum": minimum,
                        "maximum": maximum,
                    })
                    width_hist[width] += 1
                    if width >= 12:
                        outliers.append({
                            "operator_index": op,
                            "operator": operator["operator"],
                            "partition": partition,
                            "pattern_index": pattern_index,
                            "center_hex": center_hex,
                            "output_block": block,
                            "minimum": minimum,
                            "maximum": maximum,
                            "required_signed_bits": width,
                        })
                center_max_hist[max(item["width"] for item in block_rows)] += 1
                center_rows.append({"center": center, "blocks": block_rows})
            op_catalog.append(center_rows)
        width_catalog.append(op_catalog)
        print("[M78 WIDTH] operator={}/4 entries={}".format(
            op + 1, PARTITIONS * PATTERNS * OUTPUT_BLOCKS), flush=True)
    return width_catalog, width_hist, center_max_hist, outliers, observed_weight_sha


def collect_per_sample_histograms(m72, m43, manifest, operator_names):
    histograms = defaultdict(Counter)
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    mask16 = (1 << PARTITION_BITS) - 1
    records = [row for row in manifest["records"] if row["sample_id"] >= 5]
    require(len(records) == 20, "M78 heldout record extent drift")
    for record_index, record in enumerate(records):
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
        print("[M78 HIST] {}/20 sample={} operator={}".format(
            record_index + 1, sample, record["operator"]), flush=True)
    return histograms


def pwp_bytes(width):
    return width * OUTPUT_LANES // 8


def pwp_service_cycles(width, port):
    if port["pwp_port_bytes"] is None:
        return 1
    return int(math.ceil(pwp_bytes(width) / float(port["pwp_port_bytes"])))


def phase_metrics(counter, centers, center_widths):
    base = {
        "partition_vectors": 0,
        "baseline_ops_per_block": 0,
        "matcher_rows": 0,
    }
    caps = {}
    for cap in WIDTH_CAPS:
        caps[cap] = {
            "correction_ops_all_blocks": 0,
            "pwp_ops_all_blocks": 0,
            "pwp_read_bytes": 0,
            "escape_rows_all_blocks": 0,
            "assignment_rows": 0,
            "pwp_uses_by_width": Counter(),
        }
    for value, count in counter.items():
        pop = m72_popcount(value)
        base["partition_vectors"] += count
        base["baseline_ops_per_block"] += count * pop
        if pop >= 2:
            base["matcher_rows"] += count
        best_distance, best_center, best_index = min(
            (m72_popcount(value ^ center), center, index)
            for index, center in enumerate(centers))
        beneficial = 1 + best_distance < pop
        for cap in WIDTH_CAPS:
            row = caps[cap]
            any_eligible = False
            for block in range(OUTPUT_BLOCKS):
                width = center_widths[best_index][block]["width"]
                if beneficial and width <= cap:
                    any_eligible = True
                    row["correction_ops_all_blocks"] += count * best_distance
                    row["pwp_ops_all_blocks"] += count
                    row["pwp_read_bytes"] += count * pwp_bytes(width)
                    row["pwp_uses_by_width"][width] += count
                else:
                    row["correction_ops_all_blocks"] += count * pop
                    if beneficial:
                        row["escape_rows_all_blocks"] += count
            if beneficial and any_eligible:
                row["assignment_rows"] += count
    return base, caps


def m72_popcount(value):
    return bin(int(value)).count("1")


def phase_pwp_payload_bytes(center_widths, cap):
    bits = sum(
        block["width"] * OUTPUT_LANES
        for center in center_widths for block in center["blocks"]
        if block["width"] <= cap)
    require(bits % 8 == 0, "M78 PWP phase payload not byte aligned")
    return bits // 8


def replay_sample(phases, cap, port):
    weight_load = int(math.ceil(WEIGHT_PHASE_BYTES / float(DRAM_BYTES_PER_CYCLE)))
    candidate_loads = [int(math.ceil(
        (WEIGHT_PHASE_BYTES + phase["pwp_payload_bytes"][cap]) /
        float(DRAM_BYTES_PER_CYCLE))) for phase in phases]
    dense_cycles = weight_load
    baseline_cycles = weight_load
    candidate_cycles = candidate_loads[0]
    bindings = Counter()
    component = Counter()
    for phase_index, phase in enumerate(phases):
        base = phase["base"]
        row = phase["caps"][cap]
        dense_compute = (base["partition_vectors"] * PARTITION_BITS *
                         OUTPUT_BLOCKS * port["weight_cycles"])
        baseline_compute = (base["baseline_ops_per_block"] * OUTPUT_BLOCKS *
                            port["weight_cycles"])
        pwp_compute = sum(
            uses * pwp_service_cycles(width, port)
            for width, uses in row["pwp_uses_by_width"].items())
        candidate_compute = (row["correction_ops_all_blocks"] *
                             port["weight_cycles"] + pwp_compute)
        matcher = base["matcher_rows"] + MATCHER_PIPELINE_CYCLES
        packer = int(math.ceil(row["assignment_rows"] / 8.0)) + PACKER_PIPELINE_CYCLES
        next_weight = weight_load if phase_index + 1 < len(phases) else 0
        next_candidate = candidate_loads[phase_index + 1] if phase_index + 1 < len(phases) else 0
        dense_cycles += max(dense_compute, next_weight) + COMPUTE_TAIL_CYCLES
        baseline_cycles += max(baseline_compute, next_weight) + COMPUTE_TAIL_CYCLES
        candidates = (
            (candidate_compute, "compute"),
            (matcher, "matcher"),
            (packer, "packer"),
            (next_candidate, "dma"),
        )
        candidate_cycles += max(item[0] for item in candidates) + COMPUTE_TAIL_CYCLES
        binding = max(candidates)[1]
        bindings[binding] += 1
        component["matcher_cycles"] += matcher
        component["packer_cycles"] += packer
        component["candidate_compute_cycles"] += candidate_compute
    return {
        "dense_cycles": dense_cycles,
        "bit_sparse_cycles": baseline_cycles,
        "candidate_cycles": candidate_cycles,
        "speedup_vs_dense": dense_cycles / float(candidate_cycles),
        "speedup_vs_bit_sparse": baseline_cycles / float(candidate_cycles),
        "binding_phases": dict(bindings),
        "component_cycles": dict(component),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M78 output overwrite")
    analyzer_start_sha = sha256(Path(__file__).resolve())
    require(sha256(M72_RESULT) == EXPECTED_SHA256["m72_result"],
            "M78 M72 result identity drift")
    require(sha256(M41_RESULT) == EXPECTED_SHA256["m41_result"],
            "M78 M41 result identity drift")
    m72 = load_m72()
    m72_result = strict_json(M72_RESULT)
    m41_result = strict_json(M41_RESULT)
    require(m72_result["split"]["train_catalog_eligible"] is False,
            "M78 requires valid825-internal M72 identity")
    width_catalog, width_hist, center_max_hist, outliers, weight_shas = (
        build_width_catalog(m72_result, m41_result))

    manifest = strict_json(m72.MANIFEST_PATH)
    operator_names = [item["operator"] for item in m72_result["operators"]]
    m43 = m72.load_m43()
    histograms = collect_per_sample_histograms(m72, m43, manifest, operator_names)
    sample_phases = defaultdict(list)
    aggregate_base = Counter()
    aggregate_caps = dict((cap, Counter()) for cap in WIDTH_CAPS)
    aggregate_width_uses = dict((cap, Counter()) for cap in WIDTH_CAPS)
    for sample in range(5, 10):
        for op in range(4):
            for partition in range(PARTITIONS):
                centers = [item["center"] for item in width_catalog[op][partition]]
                widths = [item["blocks"] for item in width_catalog[op][partition]]
                base, caps = phase_metrics(
                    histograms[(sample, op, partition)], centers, widths)
                aggregate_base.update(base)
                payload_bytes = {}
                for cap in WIDTH_CAPS:
                    for key, value in caps[cap].items():
                        if key != "pwp_uses_by_width":
                            aggregate_caps[cap][key] += value
                    aggregate_width_uses[cap].update(caps[cap]["pwp_uses_by_width"])
                    payload_bytes[cap] = phase_pwp_payload_bytes(
                        width_catalog[op][partition], cap)
                sample_phases[sample].append({
                    "base": base,
                    "caps": caps,
                    "pwp_payload_bytes": payload_bytes,
                })
    require(aggregate_base["partition_vectors"] == 25920000,
            "M78 heldout population conservation drift")
    require(aggregate_base["baseline_ops_per_block"] == 46432637,
            "M78 bit-sparse work conservation drift")

    total_entries = 4 * PARTITIONS * PATTERNS * OUTPUT_BLOCKS
    fixed12_bits = total_entries * 12 * OUTPUT_LANES
    configurations = []
    for cap in WIDTH_CAPS:
        eligible_hist = dict((width, count) for width, count in width_hist.items()
                             if width <= cap)
        eligible_entries = sum(eligible_hist.values())
        elastic_bits = sum(width * count * OUTPUT_LANES
                           for width, count in eligible_hist.items())
        fixed_cap_bits = eligible_entries * cap * OUTPUT_LANES
        cap_row = aggregate_caps[cap]
        per_port = []
        for port in PORTS:
            totals = Counter()
            per_sample = []
            for sample in range(5, 10):
                replay = replay_sample(sample_phases[sample], cap, port)
                replay["sample_id"] = sample
                per_sample.append(replay)
                for key in ("dense_cycles", "bit_sparse_cycles", "candidate_cycles"):
                    totals[key] += replay[key]
                totals.update(replay["binding_phases"])
            per_port.append({
                "port": port["name"],
                "weight_vector_service_cycles": port["weight_cycles"],
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
            })
        baseline_all_blocks = aggregate_base["baseline_ops_per_block"] * OUTPUT_BLOCKS
        candidate_ops = (cap_row["correction_ops_all_blocks"] +
                         cap_row["pwp_ops_all_blocks"])
        configurations.append({
            "signed_width_cap": cap,
            "eligible_output_block_entries": eligible_entries,
            "ineligible_output_block_entries": total_entries - eligible_entries,
            "eligible_fraction": eligible_entries / float(total_entries),
            "exact_elastic_pwp_payload_bytes": elastic_bits // 8,
            "fixed_cap_pwp_payload_bytes": fixed_cap_bits // 8,
            "fixed12_reference_payload_bytes": fixed12_bits // 8,
            "elastic_storage_reduction_vs_fixed12": 1.0 - elastic_bits / float(fixed12_bits),
            "fixed_cap_storage_reduction_vs_fixed12": 1.0 - fixed_cap_bits / float(fixed12_bits),
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
                "pwp_uses_by_width": dict(sorted(aggregate_width_uses[cap].items())),
                "baseline_weight_sram_read_bytes": (
                    baseline_all_blocks * WEIGHT_VECTOR_BYTES),
                "candidate_correction_sram_read_bytes": (
                    cap_row["correction_ops_all_blocks"] * WEIGHT_VECTOR_BYTES),
                "candidate_pwp_sram_read_bytes": cap_row["pwp_read_bytes"],
            },
            "cycle_simulations": per_port,
        })
        wide = per_port[0]
        print("[M78 CAP{}] eligible={:.6f} storage_save={:.6f} wide_vs_bs={:.6f}x".format(
            cap, eligible_entries / float(total_entries),
            1.0 - elastic_bits / float(fixed12_bits),
            wide["speedup_vs_bit_sparse"]), flush=True)

    require(sha256(Path(__file__).resolve()) == analyzer_start_sha,
            "M78 analyzer source changed during execution")
    payload = {
        "schema": "m78_precision_elastic_pwp_cycle_dse_valid825_internal_v1",
        "status": "PASS_M78_EXACT_INT8_PWP_WIDTH_AND_BLOCK_ESCAPE_DSE_INTERNAL_ONLY",
        "identity": {
            "analyzer_start_end_sha256": analyzer_start_sha,
            "m72_result_sha256": sha256(M72_RESULT),
            "m41_result_sha256": sha256(M41_RESULT),
            "weight_payload_sha256": weight_shas,
        },
        "scope": {
            "operators": operator_names,
            "heldout_samples": [5, 6, 7, 8, 9],
            "output_lanes": OUTPUT_LANES,
            "output_blocks": OUTPUT_BLOCKS,
            "precision_caps": list(WIDTH_CAPS),
            "escape": "selected PWP is shared; each output block independently falls back to exact bit-sparse weights if its PWP range exceeds the cap",
        },
        "pwp_precision": {
            "output_block_entry_count": total_entries,
            "minimum_width_floor_bits": 8,
            "width_histogram": dict(sorted(width_hist.items())),
            "center_max_width_histogram": dict(sorted(center_max_hist.items())),
            "required_12bit_outliers": outliers,
            "pattern_table_bytes": 4 * PARTITIONS * PATTERNS * 2,
            "fixed12_pwp_payload_bytes": fixed12_bits // 8,
        },
        "work_conservation": {
            "partition_vectors_per_output_block": aggregate_base["partition_vectors"],
            "baseline_bit_sparse_vector_ops_per_output_block": (
                aggregate_base["baseline_ops_per_block"]),
            "matcher_rows": aggregate_base["matcher_rows"],
        },
        "configurations": configurations,
        "admission": {
            "exact_checkpoint_int8_pwp_ranges": True,
            "same_scope_dense_and_bit_sparse_cycle_baselines": True,
            "isolated_module_cycle_simulator_estimate": True,
            "valid825_internal_only": True,
            "independent_validation": False,
            "train_catalog": False,
            "accuracy": False,
            "rtl_or_synopsys_ppa": False,
            "full_network_or_system_speedup": False,
            "date_headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M78 configurations={} output={}".format(
        len(configurations), args.output), flush=True)


if __name__ == "__main__":
    main()
