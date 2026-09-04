#!/usr/bin/env python3
"""Deterministic Phi-style Hamming k-means k16/q16 valid825-internal screen.

Samples 0--4 form per-operator/per-partition binary centers and samples 5--9
are an internal holdout.  Both sides come from the local valid825 population,
so this result is deliberately ineligible as a PAFT training catalog or final
validation result.  Following Phi, all-zero and one-hot rows are excluded from
clustering; runtime retains the zero/bit-sparse fallback whenever
one-PWP-plus-signed-Hamming is not cheaper.
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
M70_PATH = HW / (
    "results/m70_phi_pattern_heldout_dse_dev_r1_20260823/"
    "m70_phi_pattern_heldout_dse.json")
EXPECTED_SHA256 = {
    "m43": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
}
CALIBRATION_SAMPLES = frozenset(range(5))
HELDOUT_SAMPLES = frozenset(range(5, 10))
K = 16
Q = 16
PARTITIONS = 6912 // K
OUTPUT_LANES = 96
OUTPUT_BLOCKS = 8
PWP_SIGNED_BITS = 12
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << K))


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
    spec = importlib.util.spec_from_file_location("m72_m43", M43_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M43")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_histograms(m43, manifest, operator_names):
    calibration = defaultdict(Counter)
    heldout = defaultdict(Counter)
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    mask16 = (1 << K) - 1
    for record_index, record in enumerate(manifest["records"]):
        sample_id = record["sample_id"]
        target = calibration if sample_id in CALIBRATION_SAMPLES else heldout
        require(sample_id in CALIBRATION_SAMPLES or sample_id in HELDOUT_SAMPLES,
                "M72 sample outside frozen split")
        op = operator_index[record["operator"]]
        masks = m43.unpack_record_masks(MANIFEST_PATH.parent, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // K)
                for subtile in range(m43.TILE_BITS // K):
                    value = (value256 >> (subtile * K)) & mask16
                    target[(op, partition_base + subtile)][value] += 1
        print("[M72 HIST] {}/40 split={} sample={} operator={}".format(
            record_index + 1,
            "calibration" if sample_id in CALIBRATION_SAMPLES else "heldout",
            sample_id, record["operator"]), flush=True)
    return calibration, heldout


def farthest_fill(values, counts, centers, target_count):
    centers = list(dict.fromkeys(centers))
    available = [value for value in values if value not in frozenset(centers)]
    while len(centers) < target_count and available:
        if centers:
            chosen = max(
                available,
                key=lambda value: (
                    counts[value] * min(POPCOUNT[value ^ center] for center in centers),
                    min(POPCOUNT[value ^ center] for center in centers),
                    counts[value], -value))
        else:
            chosen = max(available, key=lambda value: (counts[value], -value))
        centers.append(chosen)
        available.remove(chosen)
    return centers


def hamming_kmeans(counter, iterations=20):
    # Phi filters all-zero and one-hot rows before Hamming k-means.
    values = sorted(value for value in counter
                    if value != 0 and POPCOUNT[value] != 1)
    require(values, "M72 empty filtered partition")
    if len(values) <= Q:
        return farthest_fill(values, counter, values, Q), 0
    centers = farthest_fill(values, counter, [], Q)
    completed = 0
    for iteration in range(iterations):
        totals = [0] * Q
        ones = [[0] * K for _ in range(Q)]
        for value in values:
            index = min(range(Q), key=lambda candidate: (
                POPCOUNT[value ^ centers[candidate]], candidate))
            count = counter[value]
            totals[index] += count
            for bit in range(K):
                if value & (1 << bit):
                    ones[index][bit] += count
        updated = []
        for index in range(Q):
            if totals[index] == 0:
                updated.append(centers[index])
                continue
            center = 0
            for bit in range(K):
                # Strict majority makes the 0.5 tie deterministic at zero.
                if 2 * ones[index][bit] > totals[index]:
                    center |= 1 << bit
            updated.append(center)
        updated = farthest_fill(values, counter,
                                [value for value in updated if value != 0], Q)
        require(len(updated) == Q, "M72 center refill failure")
        completed = iteration + 1
        if updated == centers:
            break
        centers = updated
    return centers, completed


def evaluate(counter, centers):
    baseline = candidate = pwp = correction = vectors = nonzero = exact = 0
    used = set()
    center_set = frozenset(centers)
    for value, count in counter.items():
        pop = POPCOUNT[value]
        baseline += count * pop
        vectors += count
        if value != 0:
            nonzero += count
        if value != 0 and value in center_set:
            exact += count
        best_hamming, best_center = min(
            (POPCOUNT[value ^ center], center) for center in centers)
        if 1 + best_hamming < pop:
            candidate += count * (1 + best_hamming)
            pwp += count
            correction += count * best_hamming
            used.add(best_center)
        else:
            candidate += count * pop
            correction += count * pop
    require(pwp + correction == candidate,
            "M72 PWP/correction conservation failure")
    return {
        "partition_vectors": vectors,
        "nonzero_partition_vectors": nonzero,
        "baseline_bit_sparse_vector_ops": baseline,
        "nearest_signed_vector_ops": candidate,
        "nearest_pwp_vector_ops": pwp,
        "nearest_correction_vector_ops": correction,
        "exact_pattern_hits": exact,
        "used_centers": len(used),
    }


def build(output):
    require(not output.exists(), "refusing M72 output overwrite")
    for name, path in (("m43", M43_PATH), ("manifest", MANIFEST_PATH)):
        require(path.is_file() and sha256(path) == EXPECTED_SHA256[name],
                "M72 input SHA drift: {}".format(name))
    require(M70_PATH.is_file(), "M72 top-frequency reference missing")
    manifest = strict_json(MANIFEST_PATH)
    operator_names = sorted(set(row["operator"] for row in manifest["records"]))
    require(len(manifest["records"]) == 40 and len(operator_names) == 4,
            "M72 population drift")
    m43 = load_m43()
    calibration, heldout = collect_histograms(m43, manifest, operator_names)

    operator_rows = []
    aggregate = Counter()
    total_entries = 0
    for op, operator in enumerate(operator_names):
        partitions = []
        op_aggregate = Counter()
        iteration_counts = []
        for partition in range(PARTITIONS):
            centers, iterations = hamming_kmeans(calibration[(op, partition)])
            require(len(centers) == Q and len(set(centers)) == Q,
                    "M72 codebook extent/uniqueness failure")
            calibration_result = evaluate(calibration[(op, partition)], centers)
            heldout_result = evaluate(heldout[(op, partition)], centers)
            partitions.append({
                "partition": partition,
                "centers_hex": ["{:04x}".format(value) for value in centers],
                "lloyd_iterations": iterations,
                "calibration": calibration_result,
                "heldout": heldout_result,
            })
            iteration_counts.append(iterations)
            total_entries += len(centers)
            for key, value in heldout_result.items():
                if key != "used_centers":
                    op_aggregate[key] += value
            if (partition + 1) % 108 == 0:
                print("[M72 KMEANS] operator={}/4 partition={}/432".format(
                    op + 1, partition + 1), flush=True)
        op_payload = dict(op_aggregate)
        op_payload.update({
            "operator": operator,
            "nearest_signed_speedup": (
                op_aggregate["baseline_bit_sparse_vector_ops"] /
                op_aggregate["nearest_signed_vector_ops"]),
            "lloyd_iterations_maximum": max(iteration_counts),
            "partitions": partitions,
        })
        operator_rows.append(op_payload)
        aggregate.update(op_aggregate)

    top = strict_json(M70_PATH)
    top_q16 = next(row for row in top["configurations"]
                   if row["partition_bits"] == K and
                   row["maximum_patterns_per_partition"] == Q)
    speedup = (aggregate["baseline_bit_sparse_vector_ops"] /
               aggregate["nearest_signed_vector_ops"])
    pwp_vector_bytes = OUTPUT_LANES * PWP_SIGNED_BITS // 8
    payload = {
        "schema": "m72_phi_hamming_kmeans_k16q16_valid825_internal_screen_v1",
        "status": "PASS_M72_VALID825_INTERNAL_SCREEN_NOT_TRAIN_CATALOG_CYCLES_RTL_UNADMITTED",
        "identity": {
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": EXPECTED_SHA256,
            "top_frequency_reference_sha256": sha256(M70_PATH),
        },
        "split": {
            "source_population": "local_DSEC_valid825_first_ten_samples",
            "calibration_samples_within_valid825": sorted(CALIBRATION_SAMPLES),
            "heldout_samples_within_valid825": sorted(HELDOUT_SAMPLES),
            "validation_or_test_data_used_for_centers": True,
            "train_catalog_eligible": False,
        },
        "calibration_algorithm": {
            "method": "per-operator per-partition weighted binary Lloyd k-means",
            "distance": "Hamming",
            "initialization": "deterministic count-weighted farthest-first",
            "filtered_rows": ["all_zero", "one_hot"],
            "center_update": "weighted bit majority with tie-to-zero",
            "partition_bits": K,
            "patterns_per_partition": Q,
        },
        "heldout": {
            "partition_vectors": aggregate["partition_vectors"],
            "nonzero_partition_vectors": aggregate["nonzero_partition_vectors"],
            "baseline_bit_sparse_vector_ops": aggregate["baseline_bit_sparse_vector_ops"],
            "nearest_signed_vector_ops": aggregate["nearest_signed_vector_ops"],
            "nearest_pwp_vector_ops": aggregate["nearest_pwp_vector_ops"],
            "nearest_correction_vector_ops": aggregate["nearest_correction_vector_ops"],
            "nearest_signed_speedup": speedup,
            "nonzero_matcher_only_ceiling_vs_bit_sparse": (
                aggregate["baseline_bit_sparse_vector_ops"] /
                aggregate["nonzero_partition_vectors"]),
            "top_frequency_q16_speedup": top_q16["nearest_signed_speedup"],
            "kmeans_over_top_frequency_candidate_cycle_reduction": (
                top_q16["nearest_signed_vector_ops"] /
                aggregate["nearest_signed_vector_ops"]),
        },
        "hardware_capacity": {
            "codebook_entries": total_entries,
            "pattern_table_bytes": total_entries * K // 8,
            "pwp_signed_bits": PWP_SIGNED_BITS,
            "pwp_vector_bytes": pwp_vector_bytes,
            "all_pwp_bytes_bit_tight": (
                total_entries * OUTPUT_BLOCKS * pwp_vector_bytes),
            "one_partition_one_output_block_pwp_working_set_bytes": (
                Q * pwp_vector_bytes),
        },
        "promotion_gate": {
            "minimum_heldout_vector_op_speedup": 3.0,
            "passes_natural_compute_gate": speedup >= 3.0,
            "paft_allowed": False,
            "rtl_allowed": False,
            "next": (
                "capture a disjoint training-only calibration cohort, rebuild "
                "the k-means catalog, then run five-epoch PAFT; this valid825 "
                "internal screen is never a training input"),
        },
        "admission": {
            "valid825_internal_vector_operation_screen": True,
            "independent_validation_speedup": False,
            "train_catalog": False,
            "cycle_accurate_speedup": False,
            "pwp_dram_traffic_charged": False,
            "matcher_packer_rtl_synopsys": False,
            "accuracy_change": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "operators": operator_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M72 kmeans={:.6f}x top={:.6f}x pwp_mib={:.3f}".format(
        speedup, top_q16["nearest_signed_speedup"],
        payload["hardware_capacity"]["all_pwp_bytes_bit_tight"] / float(1 << 20)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
