#!/usr/bin/env python3
"""Held-out Phi-like exact pattern/correction DSE on the M40 H67 cohort.

Samples 0--4 calibrate per-operator, per-K-partition binary pattern tables.
Samples 5--9 are never used for pattern selection.  Runtime arithmetic remains
exact:

    W*x = PWP[p] + W*(x-p),  PWP[p] = W*p.

The result counts 96-lane vector additions: one for a nonzero PWP plus one for
each signed correction bit.  It is a compute opportunity oracle.  PWP traffic,
matcher cycles, buffer ports and RTL/PPA are intentionally not admitted.
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
PARTITION_BITS = (16, 32, 64)
PATTERN_COUNTS = (8, 16, 32, 64, 128)
CALIBRATION_SAMPLES = frozenset(range(5))
HELDOUT_SAMPLES = frozenset(range(5, 10))
OUTPUT_VECTOR_BYTES_SIGNED19 = 228
OUTPUT_BLOCKS = 8


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
    spec = importlib.util.spec_from_file_location("m70_m43", M43_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M43")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_histograms(m43, manifest):
    calibration = defaultdict(Counter)
    heldout = defaultdict(Counter)
    operator_names = sorted(set(row["operator"] for row in manifest["records"]))
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    require(len(operator_names) == 4, "M70 operator population drift")
    for record_index, record in enumerate(manifest["records"]):
        sample_id = record["sample_id"]
        require(sample_id in CALIBRATION_SAMPLES or sample_id in HELDOUT_SAMPLES,
                "M70 sample outside split")
        destination = calibration if sample_id in CALIBRATION_SAMPLES else heldout
        op = operator_index[record["operator"]]
        masks = m43.unpack_record_masks(MANIFEST_PATH.parent, record)
        for row in range(m43.ROWS):
            row_offset = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[row_offset + tile]
                for width in PARTITION_BITS:
                    subtiles = m43.TILE_BITS // width
                    mask = (1 << width) - 1
                    base_partition = tile * subtiles
                    for subtile in range(subtiles):
                        value = (value256 >> (subtile * width)) & mask
                        destination[(op, width, base_partition + subtile)][value] += 1
        print("[M70 HIST] {}/40 split={} sample={} operator={}".format(
            record_index + 1,
            "calibration" if sample_id in CALIBRATION_SAMPLES else "heldout",
            sample_id, record["operator"]), flush=True)
    return calibration, heldout, operator_names


def evaluate(calibration, heldout, operator_names):
    results = []
    for width in PARTITION_BITS:
        partitions_per_operator = 6912 // width
        for q in PATTERN_COUNTS:
            baseline_vector_ops = 0
            exact_fallback_vector_ops = 0
            nearest_signed_vector_ops = 0
            nearest_pwp_ops = 0
            nearest_correction_ops = 0
            exact_pattern_hits = 0
            heldout_vectors = 0
            codebook_entries = 0
            used_entries = 0
            operator_rows = []
            for op, operator in enumerate(operator_names):
                op_baseline = op_exact = op_nearest = 0
                op_pwp = op_correction = op_hits = op_vectors = 0
                op_entries = op_used = 0
                for partition in range(partitions_per_operator):
                    key = (op, width, partition)
                    calibration_counter = calibration[key]
                    heldout_counter = heldout[key]
                    # Zero is represented by an implicit no-PWP choice.  Stable
                    # count/value ordering makes the calibration deterministic.
                    ranked = sorted(
                        ((count, value) for value, count in calibration_counter.items()
                         if value != 0), key=lambda item: (-item[0], item[1]))
                    patterns = [value for _, value in ranked[:q]]
                    pattern_set = frozenset(patterns)
                    op_entries += len(patterns)
                    used = set()
                    for value, count in heldout_counter.items():
                        # The Synopsys host is pinned to Python 3.6, where
                        # int.bit_count() is unavailable.  The masks are only
                        # k<=64 bits wide, so this exact fallback is cheap and
                        # keeps the DSE runnable in the evidence environment.
                        pop = bin(value).count("1")
                        op_baseline += count * pop
                        op_vectors += count
                        if value != 0 and value in pattern_set:
                            op_exact += count
                            op_hits += count
                        else:
                            op_exact += count * pop

                        best_cost = pop
                        best_pattern = 0
                        best_hamming = pop
                        for pattern in patterns:
                            hamming = bin(value ^ pattern).count("1")
                            cost = 1 + hamming
                            if (cost, hamming, pattern) < (
                                    best_cost, best_hamming, best_pattern):
                                best_cost = cost
                                best_pattern = pattern
                                best_hamming = hamming
                        op_nearest += count * best_cost
                        op_correction += count * best_hamming
                        if best_pattern != 0:
                            op_pwp += count
                            used.add(best_pattern)
                    op_used += len(used)
                require(op_pwp + op_correction == op_nearest,
                        "M70 PWP/correction conservation failure")
                operator_rows.append({
                    "operator": operator,
                    "heldout_baseline_bit_sparse_vector_ops": op_baseline,
                    "exact_match_fallback_vector_ops": op_exact,
                    "nearest_signed_vector_ops": op_nearest,
                    "nearest_pwp_vector_ops": op_pwp,
                    "nearest_correction_vector_ops": op_correction,
                    "exact_pattern_hit_vectors": op_hits,
                    "heldout_partition_vectors": op_vectors,
                    "codebook_entries": op_entries,
                    "heldout_used_codebook_entries": op_used,
                })
                baseline_vector_ops += op_baseline
                exact_fallback_vector_ops += op_exact
                nearest_signed_vector_ops += op_nearest
                nearest_pwp_ops += op_pwp
                nearest_correction_ops += op_correction
                exact_pattern_hits += op_hits
                heldout_vectors += op_vectors
                codebook_entries += op_entries
                used_entries += op_used
            require(nearest_pwp_ops + nearest_correction_ops ==
                    nearest_signed_vector_ops, "M70 aggregate conservation failure")
            results.append({
                "partition_bits": width,
                "maximum_patterns_per_partition": q,
                "heldout_baseline_bit_sparse_vector_ops": baseline_vector_ops,
                "exact_match_fallback_vector_ops": exact_fallback_vector_ops,
                "nearest_signed_vector_ops": nearest_signed_vector_ops,
                "nearest_pwp_vector_ops": nearest_pwp_ops,
                "nearest_correction_vector_ops": nearest_correction_ops,
                "exact_match_fallback_speedup": (
                    baseline_vector_ops / exact_fallback_vector_ops),
                "nearest_signed_speedup": (
                    baseline_vector_ops / nearest_signed_vector_ops),
                "exact_pattern_hit_fraction": (
                    exact_pattern_hits / heldout_vectors),
                "codebook_entries": codebook_entries,
                "heldout_used_codebook_entries": used_entries,
                "pattern_table_bytes": codebook_entries * width // 8,
                "all_codebook_pwp_bytes_signed19": (
                    codebook_entries * OUTPUT_BLOCKS *
                    OUTPUT_VECTOR_BYTES_SIGNED19),
                "operators": operator_rows,
            })
            print("[M70 DSE] k={} q={} exact={:.6f}x nearest={:.6f}x pwp_mib={:.3f}".format(
                width, q, results[-1]["exact_match_fallback_speedup"],
                results[-1]["nearest_signed_speedup"],
                results[-1]["all_codebook_pwp_bytes_signed19"] / (1 << 20)),
                flush=True)
    return results


def build(output):
    for name, path in (("m43", M43_PATH), ("manifest", MANIFEST_PATH)):
        require(path.is_file() and sha256(path) == EXPECTED_SHA256[name],
                "M70 input SHA drift: {}".format(name))
    require(not output.exists(), "refusing M70 result overwrite")
    manifest = strict_json(MANIFEST_PATH)
    require(len(manifest["records"]) == 40, "M70 manifest extent drift")
    m43 = load_m43()
    calibration, heldout, operators = collect_histograms(m43, manifest)
    configurations = evaluate(calibration, heldout, operators)
    best = min(configurations, key=lambda row: (
        -row["nearest_signed_speedup"],
        row["all_codebook_pwp_bytes_signed19"],
        row["partition_bits"], row["maximum_patterns_per_partition"]))
    payload = {
        "schema": "m70_phi_pattern_heldout_dse_v1",
        "status": "PASS_M70_HELDOUT_EXACT_PATTERN_VECTOR_OP_DSE_CYCLES_RTL_MEMORY_UNADMITTED",
        "identity": {
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": EXPECTED_SHA256,
        },
        "population": {
            "calibration_samples": sorted(CALIBRATION_SAMPLES),
            "heldout_samples": sorted(HELDOUT_SAMPLES),
            "operators": operators,
            "records": 40,
        },
        "arithmetic_contract": {
            "identity": "W*x = PWP[p] + W*(x-p)",
            "runtime_values": "PWP signed19 vector plus signed +1/-1 correction",
            "zero_pattern": "implicit and costs zero PWP vector operations",
            "pattern_selection": "top-q nonzero exact patterns on calibration split only",
            "nearest_selection": "minimize one-PWP-plus-Hamming correction vector operations",
        },
        "configurations": configurations,
        "best_vector_op_configuration": best,
        "admission": {
            "calibration_heldout_split": True,
            "exact_arithmetic_identity": True,
            "heldout_vector_operation_speedup": True,
            "cycle_accurate_speedup": False,
            "pwp_memory_feasible": False,
            "matcher_rtl_synopsys": False,
            "accuracy_change": False,
            "paft_training_gain": False,
            "full_network_or_system_speedup": False,
            "date_headline": False,
        },
        "promotion_gate": {
            "minimum_nearest_signed_vector_op_speedup_vs_bit_sparse": 3.0,
            "maximum_all_codebook_pwp_bytes": 268435456,
            "passes_compute_gate": best["nearest_signed_speedup"] >= 3.0,
            "passes_unpruned_pwp_capacity_gate":
                best["all_codebook_pwp_bytes_signed19"] <= 268435456,
            "rtl_allowed": False,
            "next": (
                "if compute gate passes, build used-PWP prefetch and packed-L2 cycle model; "
                "otherwise require hardware-weighted PAFT before RTL"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M70 best k={} q={} nearest={:.6f}x pwp_mib={:.3f}".format(
        best["partition_bits"], best["maximum_patterns_per_partition"],
        best["nearest_signed_speedup"],
        best["all_codebook_pwp_bytes_signed19"] / (1 << 20)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
