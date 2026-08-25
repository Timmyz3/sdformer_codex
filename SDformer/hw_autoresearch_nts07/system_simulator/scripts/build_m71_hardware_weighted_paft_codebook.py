#!/usr/bin/env python3
"""Freeze a train-only k16/q16 pattern catalog for hardware-weighted PAFT.

The catalog is derived only from H67 samples 0--4.  It is deliberately small
enough for a 16-way matcher and uses an implicit zero pattern.  The emitted
PWP capacity uses the tight per-partition INT8 bound: a sum of sixteen signed
INT8 weights fits exactly in signed 12 bits.  No accuracy or cycle claim is
made by this builder.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M43_PATH = HW / "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py"
MANIFEST_PATH = HW / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
EXPECTED_SHA256 = {
    "m43": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
}
CALIBRATION_SAMPLES = frozenset(range(5))
PARTITION_BITS = 16
PATTERNS_PER_PARTITION = 16
FEATURES = 768 * 3 * 3
PARTITIONS = FEATURES // PARTITION_BITS
OUTPUT_LANES = 96
OUTPUT_BLOCKS = 8
PWP_SIGNED_BITS = 12


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


def load_m43():
    spec = importlib.util.spec_from_file_location("m71_m43", M43_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M43")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(m43, manifest, operator_names):
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    histogram = defaultdict(Counter)
    records = [row for row in manifest["records"]
               if row["sample_id"] in CALIBRATION_SAMPLES]
    require(len(records) == 20, "M71 calibration extent drift")
    mask16 = (1 << PARTITION_BITS) - 1
    for record_index, record in enumerate(records):
        op = operator_index[record["operator"]]
        masks = m43.unpack_record_masks(MANIFEST_PATH.parent, record)
        for row in range(m43.ROWS):
            row_offset = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[row_offset + tile]
                partition_base = tile * (m43.TILE_BITS // PARTITION_BITS)
                for subtile in range(m43.TILE_BITS // PARTITION_BITS):
                    value = (value256 >> (subtile * PARTITION_BITS)) & mask16
                    histogram[(op, partition_base + subtile)][value] += 1
        print("[M71 CAL] {}/20 sample={} operator={}".format(
            record_index + 1, record["sample_id"], record["operator"]),
            flush=True)
    return histogram


def build(output):
    require(not output.exists(), "refusing M71 codebook overwrite")
    for name, path in (("m43", M43_PATH), ("manifest", MANIFEST_PATH)):
        require(path.is_file() and sha256(path) == EXPECTED_SHA256[name],
                "M71 input SHA drift: {}".format(name))
    manifest = strict_json(MANIFEST_PATH)
    require(len(manifest["records"]) == 40, "M71 manifest population drift")
    operator_names = sorted(set(row["operator"] for row in manifest["records"]))
    require(len(operator_names) == 4, "M71 operator population drift")
    m43 = load_m43()
    histogram = collect(m43, manifest, operator_names)

    operators = []
    total_entries = 0
    total_vectors = 0
    baseline_ops = 0
    exact_fallback_ops = 0
    exact_hits = 0
    for op, name in enumerate(operator_names):
        partitions = []
        op_entries = 0
        op_vectors = 0
        op_baseline = 0
        op_exact = 0
        op_hits = 0
        for partition in range(PARTITIONS):
            counter = histogram[(op, partition)]
            ranked = sorted(
                ((count, value) for value, count in counter.items() if value != 0),
                key=lambda item: (-item[0], item[1]))
            selected = ranked[:PATTERNS_PER_PARTITION]
            pattern_set = frozenset(value for _, value in selected)
            vectors = sum(counter.values())
            local_baseline = sum(count * bin(value).count("1")
                                 for value, count in counter.items())
            local_hits = sum(count for value, count in counter.items()
                             if value != 0 and value in pattern_set)
            local_exact = sum(
                count if value != 0 and value in pattern_set
                else count * bin(value).count("1")
                for value, count in counter.items())
            partitions.append({
                "partition": partition,
                "patterns": [
                    {"value_hex": "{:04x}".format(value),
                     "calibration_count": count}
                    for count, value in selected
                ],
                "calibration_vectors": vectors,
                "calibration_baseline_bit_sparse_vector_ops": local_baseline,
                "calibration_exact_match_fallback_vector_ops": local_exact,
                "calibration_exact_pattern_hits": local_hits,
            })
            op_entries += len(selected)
            op_vectors += vectors
            op_baseline += local_baseline
            op_exact += local_exact
            op_hits += local_hits
        operators.append({
            "operator": name,
            "partitions": partitions,
            "codebook_entries": op_entries,
            "calibration_vectors": op_vectors,
            "calibration_baseline_bit_sparse_vector_ops": op_baseline,
            "calibration_exact_match_fallback_vector_ops": op_exact,
            "calibration_exact_match_speedup": op_baseline / op_exact,
            "calibration_exact_pattern_hit_fraction": op_hits / op_vectors,
        })
        total_entries += op_entries
        total_vectors += op_vectors
        baseline_ops += op_baseline
        exact_fallback_ops += op_exact
        exact_hits += op_hits

    pwp_vector_bytes = OUTPUT_LANES * PWP_SIGNED_BITS // 8
    payload = {
        "schema": "m71_h67_k16_q16_train_only_paft_codebook_v1",
        "status": "PASS_M71_TRAIN_ONLY_CODEBOOK_ACCURACY_CYCLES_RTL_UNADMITTED",
        "identity": {
            "builder_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": EXPECTED_SHA256,
        },
        "split": {
            "catalog_samples": sorted(CALIBRATION_SAMPLES),
            "heldout_samples_excluded": list(range(5, 10)),
            "test_or_validation_data_used": False,
        },
        "format": {
            "input_feature_order": "I_KY_KX",
            "partition_bits": PARTITION_BITS,
            "partitions_per_operator": PARTITIONS,
            "maximum_explicit_patterns_per_partition": PATTERNS_PER_PARTITION,
            "implicit_zero_pattern": True,
            "runtime_selection": "nearest signed Hamming correction with zero fallback",
        },
        "int8_pwp_bound": {
            "weight_range": [-128, 127],
            "terms_per_pwp": PARTITION_BITS,
            "exact_sum_range": [-2048, 2032],
            "required_signed_bits": PWP_SIGNED_BITS,
            "output_lanes_per_vector": OUTPUT_LANES,
            "pwp_vector_bytes_bit_tight": pwp_vector_bytes,
        },
        "hardware_capacity": {
            "total_codebook_entries": total_entries,
            "pattern_table_bytes": total_entries * PARTITION_BITS // 8,
            "all_pwp_bytes_bit_tight": (
                total_entries * OUTPUT_BLOCKS * pwp_vector_bytes),
            "one_partition_one_output_block_pwp_working_set_bytes": (
                PATTERNS_PER_PARTITION * pwp_vector_bytes),
            "matcher_candidates": PATTERNS_PER_PARTITION,
            "matcher_xor_bits_per_input": (
                PATTERNS_PER_PARTITION * PARTITION_BITS),
        },
        "calibration_observation_only": {
            "partition_vectors": total_vectors,
            "baseline_bit_sparse_vector_ops": baseline_ops,
            "exact_match_fallback_vector_ops": exact_fallback_ops,
            "exact_match_speedup": baseline_ops / exact_fallback_ops,
            "exact_pattern_hit_fraction": exact_hits / total_vectors,
        },
        "training_contract": {
            "loss_proxy": "output-fanout-weighted nearest-codeword L1/Hamming distance",
            "catalog_frozen_during_candidate_run": True,
            "minimum_candidate_epochs": 5,
            "required_accuracy_check": "standard valid825 plus frozen checkpoint identity",
            "promotion_compute_gate": "heldout nearest signed vector-op speedup >= 3.0x",
            "promotion_strong_baseline_gate": "cycle model speedup vs M53 >= 1.5x",
        },
        "admission": {
            "train_only_catalog": True,
            "bit_tight_pwp_bound": True,
            "paft_checkpoint": False,
            "valid825_accuracy": False,
            "heldout_pattern_speedup": False,
            "cycle_accurate_speedup": False,
            "matcher_rtl_synopsys": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "operators": operators,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M71 entries={} codebook_kib={:.3f} pwp_mib={:.3f} exact={:.6f}x".format(
        total_entries, total_entries * 2 / 1024.0,
        payload["hardware_capacity"]["all_pwp_bytes_bit_tight"] / float(1 << 20),
        baseline_ops / exact_fallback_ops))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
