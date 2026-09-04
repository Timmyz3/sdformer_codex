#!/usr/bin/env python3
"""Build the H67 M77 PAFT catalog only from an admitted M73 train trace."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M43_PATH = HW / "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py"
EXPECTED = {
    "m43": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "train_list": "919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10",
    "valid_list": "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0",
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "forward_config": "86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc",
}
OPERATORS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
REVOKED_CATALOG_SHA256 = (
    "142e32f0d988721ce9edf25d4dcf3883d82f2604f2aee9c755cde87b2ef70cdd",
)
K = 16
Q = 16
PARTITIONS = 432
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
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_m43():
    spec = importlib.util.spec_from_file_location("m77_m43", str(M43_PATH))
    require(spec is not None and spec.loader is not None, "cannot load M43")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def farthest_fill(values, counts, centers):
    centers = list(dict.fromkeys(centers))
    available = [value for value in values if value not in frozenset(centers)]
    while len(centers) < Q and available:
        if centers:
            chosen = max(available, key=lambda value: (
                counts[value] * min(POPCOUNT[value ^ center]
                                    for center in centers),
                min(POPCOUNT[value ^ center] for center in centers),
                counts[value], -value))
        else:
            chosen = max(available, key=lambda value: (counts[value], -value))
        centers.append(chosen)
        available.remove(chosen)
    return centers


def hamming_kmeans(counter, iterations=20):
    values = sorted(value for value in counter
                    if value != 0 and POPCOUNT[value] != 1)
    require(len(values) >= Q, "M77 filtered partition has fewer than q16 values")
    centers = farthest_fill(values, counter, [])
    require(len(centers) == Q, "M77 initialization extent failure")
    completed = 0
    for iteration in range(iterations):
        totals = [0] * Q
        ones = [[0] * K for _ in range(Q)]
        for value in values:
            cluster = min(range(Q), key=lambda index: (
                POPCOUNT[value ^ centers[index]], index))
            count = counter[value]
            totals[cluster] += count
            for bit in range(K):
                if value & (1 << bit):
                    ones[cluster][bit] += count
        updated = []
        for cluster in range(Q):
            if totals[cluster] == 0:
                updated.append(centers[cluster])
                continue
            center = 0
            for bit in range(K):
                if 2 * ones[cluster][bit] > totals[cluster]:
                    center |= 1 << bit
            if center != 0:
                updated.append(center)
        updated = farthest_fill(values, counter, updated)
        require(len(updated) == Q and len(set(updated)) == Q,
                "M77 center refill/uniqueness failure")
        completed = iteration + 1
        if updated == centers:
            break
        centers = updated
    require(all(value != 0 for value in centers), "M77 zero explicit center")
    return centers, completed


def evaluate(counter, centers):
    totals = Counter()
    center_set = frozenset(centers)
    for value, count in counter.items():
        pop = POPCOUNT[value]
        distance = min(POPCOUNT[value ^ center] for center in centers)
        candidate = min(pop, 1 + distance)
        totals["partition_vectors"] += count
        totals["baseline_bit_sparse_vector_ops"] += count * pop
        totals["nearest_signed_vector_ops"] += count * candidate
        if value in center_set and value != 0:
            totals["exact_pattern_hits"] += count
    return dict(totals)


def collect_histograms(m43, manifest, manifest_path):
    histograms = defaultdict(Counter)
    operator_index = dict((name, index) for index, name in enumerate(OPERATORS))
    mask16 = (1 << K) - 1
    records = manifest["records"]
    require(len(records) == 128, "M77 expected 32 samples times four operators")
    seen = Counter()
    for index, record in enumerate(records):
        sample = int(record["sample_id"])
        operator = str(record["operator"])
        require(operator in operator_index, "M77 unexpected operator")
        seen[(sample, operator)] += 1
        masks = m43.unpack_record_masks(manifest_path.parent, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // K)
                for subtile in range(m43.TILE_BITS // K):
                    value = (value256 >> (subtile * K)) & mask16
                    histograms[(operator_index[operator],
                                partition_base + subtile)][value] += 1
        print("[M77 HIST] {}/128 sample={} operator={}".format(
            index + 1, sample, operator), flush=True)
    require(len(seen) == 128 and all(count == 1 for count in seen.values()),
            "M77 sample/operator uniqueness failure")
    return histograms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-manifest", required=True, type=Path)
    parser.add_argument("--output-catalog", required=True, type=Path)
    parser.add_argument("--output-admission-contract", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_catalog.exists(), "refusing M77 catalog overwrite")
    require(not args.output_admission_contract.exists(),
            "refusing M77 contract overwrite")
    require(sha256(M43_PATH) == EXPECTED["m43"], "M77 M43 identity drift")
    source_sha = sha256(Path(__file__).resolve())
    manifest_path = args.train_trace_manifest.resolve()
    manifest = strict_json(manifest_path)
    require(manifest.get("schema") ==
            "m73_h67_ep35_train_calibration_packed_source_trace_v1",
            "M77 requires M73 train-only trace schema")
    require(manifest.get("status") ==
            "PASS_M73_DSEC_TRAIN_ONLY_S32_ALL18_SEQUENCES_EXACT_H67_EP35_FOUR_BOTTLENECK_TRACE",
            "M77 M73 status mismatch")
    identity = manifest["identity"]
    split = manifest["split_audit"]
    cohort = manifest["cohort"]
    require(identity["train_sequence_list_sha256"] == EXPECTED["train_list"]
            and identity["valid825_sequence_list_sha256"] == EXPECTED["valid_list"]
            and identity["checkpoint_sha256"] == EXPECTED["checkpoint"]
            and identity["config_sha256"] == EXPECTED["forward_config"]
            and identity["paft_forward_base_config_sha256"] ==
                EXPECTED["forward_config"],
            "M77 trace identity mismatch")
    require(split["role"] == "DSEC_TRAIN_ONLY_PAFT_CALIBRATION"
            and split["full_train_valid825_key_overlap"] == 0
            and split["selected_valid825_key_overlap"] == 0
            and split["selected_samples"] == 32
            and split["selected_sequences"] == 18,
            "M77 train/validation isolation failure")
    require(tuple(cohort["operators"]) == OPERATORS
            and cohort["records"] == 128,
            "M77 operator/cohort extent mismatch")

    m43 = load_m43()
    histograms = collect_histograms(m43, manifest, manifest_path)
    operators = []
    aggregate = Counter()
    for op, name in enumerate(OPERATORS):
        partitions = []
        op_total = Counter()
        maximum_iterations = 0
        for partition in range(PARTITIONS):
            counter = histograms[(op, partition)]
            centers, iterations = hamming_kmeans(counter)
            observation = evaluate(counter, centers)
            partitions.append({
                "partition": partition,
                "patterns": [{
                    "value_hex": "{:04x}".format(center),
                    "calibration_count": int(counter.get(center, 0)),
                } for center in centers],
                "lloyd_iterations": iterations,
                "calibration": observation,
            })
            maximum_iterations = max(maximum_iterations, iterations)
            op_total.update(observation)
            if (partition + 1) % 108 == 0:
                print("[M77 KMEANS] operator={}/4 partition={}/432".format(
                    op + 1, partition + 1), flush=True)
        op_payload = dict(op_total)
        op_payload.update({
            "operator": name,
            "partitions": partitions,
            "lloyd_iterations_maximum": maximum_iterations,
            "calibration_nearest_signed_speedup": (
                op_total["baseline_bit_sparse_vector_ops"] /
                op_total["nearest_signed_vector_ops"]),
        })
        operators.append(op_payload)
        aggregate.update(op_total)

    trace_sha = sha256(manifest_path)
    catalog = {
        "schema": "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1",
        "status": "PASS_M77_TRAIN_ONLY_KMEANS_PAFT_CATALOG_ACCURACY_CYCLES_UNADMITTED",
        "identity": {
            "builder_start_end_sha256": source_sha,
            "train_trace_manifest_sha256": trace_sha,
            "train_sequence_list_sha256": EXPECTED["train_list"],
            "valid825_sequence_list_sha256": EXPECTED["valid_list"],
            "checkpoint_sha256": EXPECTED["checkpoint"],
            "forward_base_config_sha256": EXPECTED["forward_config"],
        },
        "split": {
            "role": "DSEC_TRAIN_ONLY_PAFT_CALIBRATION",
            "train_catalog_eligible": True,
            "test_or_validation_data_used": False,
            "selected_train_samples": 32,
            "selected_train_sequences": 18,
            "train_valid825_key_overlap": 0,
        },
        "format": {
            "partition_bits": K,
            "partitions_per_operator": PARTITIONS,
            "maximum_explicit_patterns_per_partition": Q,
            "implicit_zero_pattern": True,
            "runtime_selection": "minimum of bit-sparse zero fallback and one PWP plus signed Hamming corrections",
        },
        "calibration_algorithm": {
            "method": "per-operator per-partition weighted binary Lloyd k-means",
            "distance": "Hamming",
            "initialization": "deterministic count-weighted farthest-first",
            "filtered_rows": ["all_zero", "one_hot"],
            "center_update": "weighted bit majority with tie-to-zero",
        },
        "calibration_observation_only": {
            "partition_vectors": aggregate["partition_vectors"],
            "baseline_bit_sparse_vector_ops": aggregate[
                "baseline_bit_sparse_vector_ops"],
            "nearest_signed_vector_ops": aggregate["nearest_signed_vector_ops"],
            "nearest_signed_speedup": (
                aggregate["baseline_bit_sparse_vector_ops"] /
                aggregate["nearest_signed_vector_ops"]),
            "exact_pattern_hits": aggregate["exact_pattern_hits"],
        },
        "hardware_capacity": {
            "codebook_entries": len(OPERATORS) * PARTITIONS * Q,
            "pattern_table_bytes": len(OPERATORS) * PARTITIONS * Q * 2,
            "all_signed12_pwp_bytes": (
                len(OPERATORS) * PARTITIONS * Q * 8 * 144),
        },
        "admission": {
            "train_only_catalog": True,
            "paft_checkpoint": False,
            "valid825_accuracy": False,
            "independent_validation_speedup": False,
            "cycle_accurate_speedup": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "operators": operators,
    }
    require(sha256(Path(__file__).resolve()) == source_sha,
            "M77 builder source changed during run")
    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.output_catalog.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    catalog_sha = sha256(args.output_catalog)
    contract = {
        "schema": "m77_pattern_paft_catalog_admission_contract_v1",
        "status": "PASS_M77_TRAIN_ONLY_CATALOG_ADMISSION_NO_ACCURACY_OR_SPEEDUP",
        "unit_test_only": False,
        "train_only_admitted": True,
        "catalog_sha256": catalog_sha,
        "train_trace_manifest_sha256": trace_sha,
        "train_sequence_list_sha256": EXPECTED["train_list"],
        "valid825_sequence_list_sha256": EXPECTED["valid_list"],
        "train_valid825_key_overlap": 0,
        "checkpoint_sha256": EXPECTED["checkpoint"],
        "forward_base_config_sha256": EXPECTED["forward_config"],
        "operator_names": list(OPERATORS),
        "revoked_catalog_sha256": list(REVOKED_CATALOG_SHA256),
        "claim_boundary": {
            "paft_training_input": True,
            "accuracy": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    args.output_admission_contract.parent.mkdir(parents=True, exist_ok=True)
    args.output_admission_contract.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M77 catalog={} contract={} calibration_speedup={:.6f}x".format(
        catalog_sha, sha256(args.output_admission_contract),
        catalog["calibration_observation_only"]["nearest_signed_speedup"]),
        flush=True)


if __name__ == "__main__":
    main()
