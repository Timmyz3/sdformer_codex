#!/usr/bin/env python3
"""Independent M40 -> M70/M71 recomputation and fail-closed audit.

This validator intentionally does not import the M43, M70, M71, or PAFT
production modules.  It reconstructs Conv3x3 feature masks directly from the
frozen M40 little-endian positive planes, freezes codebooks from samples 0--4,
and evaluates samples 5--9.
"""

from __future__ import print_function

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
TRACE = HW / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822")
MANIFEST = TRACE / "m40_bottleneck_packed_source_manifest.json"
M70_RESULT = HW / (
    "results/m70_phi_pattern_heldout_dse_dev_r1_20260823/"
    "m70_phi_pattern_heldout_dse.json")
M71_RESULT = HW / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json")
M72_RESULT = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M70_SOURCE = HW / (
    "system_simulator/scripts/analyze_m70_phi_pattern_heldout_dse.py")
M71_SOURCE = HW / (
    "system_simulator/scripts/build_m71_hardware_weighted_paft_codebook.py")
M72_SOURCE = HW / (
    "system_simulator/scripts/analyze_m72_phi_kmeans_k16q16_heldout.py")
VALID825_LIST = REPO / (
    "data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv")

EXPECTED_MANIFEST_SHA256 = (
    "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3")
EXPECTED_VALID825_LIST_SHA256 = (
    "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0")
T, C, H, W = 10, 768, 15, 20
ROWS = T * H * W
FEATURES = C * 3 * 3
WIDTHS = (16, 32, 64)
Q_VALUES = (8, 16, 32, 64, 128)
CALIBRATION = frozenset(range(5))
HELDOUT = frozenset(range(5, 10))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


POPCOUNT16 = np.asarray([bin(value).count("1") for value in range(1 << 16)],
                        dtype=np.uint8)


def popcount(values, width):
    values = np.asarray(values, dtype=np.uint64)
    result = np.zeros(values.shape, dtype=np.uint16)
    for shift in range(0, width, 16):
        result += POPCOUNT16[((values >> shift) & 0xffff).astype(np.uint16)]
    return result


def reconstruct_feature_bytes(record):
    path = TRACE / record["packed_file"]
    require(path.is_file(), "missing packed plane: " + str(path))
    require(sha256(path) == record["packed_file_sha256"],
            "packed-plane SHA drift")
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    plane_bytes = int(record["positive_plane_bytes"])
    require(raw.size == 3 * plane_bytes, "packed-plane extent drift")
    require(not np.any(raw[plane_bytes:2 * plane_bytes]),
            "negative plane is not zero")
    positive = np.unpackbits(raw[:plane_bytes], bitorder="little")
    require(positive.size == T * C * H * W, "positive bit extent drift")
    require(int(positive.sum()) == int(record["positive_count"]),
            "manifest positive count mismatch")
    source = positive.reshape(T, C, H, W)
    padded = np.pad(source, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    taps = np.empty((T, C, 9, H, W), dtype=np.uint8)
    tap = 0
    for kernel_y in range(3):
        for kernel_x in range(3):
            taps[:, :, tap, :, :] = padded[
                :, :, kernel_y:kernel_y + H, kernel_x:kernel_x + W]
            tap += 1
    # [T,C,KY*KX,H,W] -> rows [T,H,W], feature [I,KY,KX].
    feature_bits = taps.transpose(0, 3, 4, 1, 2).reshape(ROWS, FEATURES)
    return np.packbits(feature_bits, axis=1, bitorder="little")


def collect_histograms(manifest, operator_names):
    operator_index = dict((name, index) for index, name in enumerate(operator_names))
    histograms = {
        "calibration": defaultdict(Counter),
        "heldout": defaultdict(Counter),
    }
    split_records = Counter()
    for record_index, record in enumerate(manifest["records"]):
        sample = int(record["sample_id"])
        require(sample in CALIBRATION or sample in HELDOUT,
                "sample outside frozen split")
        split = "calibration" if sample in CALIBRATION else "heldout"
        split_records[split] += 1
        op = operator_index[record["operator"]]
        feature_bytes = reconstruct_feature_bytes(record)
        require(feature_bytes.shape == (ROWS, FEATURES // 8),
                "feature byte geometry mismatch")
        for width, dtype in ((16, "<u2"), (32, "<u4"), (64, "<u8")):
            vectors = feature_bytes.view(dtype)
            require(vectors.shape == (ROWS, FEATURES // width),
                    "partition geometry mismatch")
            for partition in range(vectors.shape[1]):
                values, counts = np.unique(vectors[:, partition],
                                           return_counts=True)
                counter = histograms[split][(op, width, partition)]
                counter.update(dict((int(value), int(count))
                                    for value, count in zip(values, counts)))
        print("[INDEPENDENT MASK] {}/40 split={} sample={} op={}".format(
            record_index + 1, split, sample, record["operator"]), flush=True)
    require(split_records == Counter({"calibration": 20, "heldout": 20}),
            "record split extent mismatch")
    return histograms


def choose_patterns(counter, q):
    ranked = sorted(((count, value) for value, count in counter.items()
                     if value != 0), key=lambda item: (-item[0], item[1]))
    return [value for _, value in ranked[:q]]


def evaluate_partition(calibration, heldout, width, q):
    patterns = choose_patterns(calibration, q)
    held_values = np.asarray(sorted(heldout), dtype=np.uint64)
    counts = np.asarray([heldout[int(value)] for value in held_values],
                        dtype=np.int64)
    populations = popcount(held_values, width).astype(np.int64)
    baseline = int(np.dot(counts, populations))
    pattern_set = frozenset(patterns)
    exact_hit_mask = np.asarray(
        [int(value) != 0 and int(value) in pattern_set for value in held_values],
        dtype=np.bool_)
    exact_hits = int(counts[exact_hit_mask].sum())
    exact_cost = populations.copy()
    exact_cost[exact_hit_mask] = 1
    exact = int(np.dot(counts, exact_cost))

    best_cost = populations.copy()
    best_hamming = populations.copy()
    best_pattern = np.zeros(held_values.shape, dtype=np.uint64)
    for pattern in patterns:
        hamming = popcount(held_values ^ np.uint64(pattern), width).astype(np.int64)
        cost = hamming + 1
        update = ((cost < best_cost) |
                  ((cost == best_cost) & (hamming < best_hamming)) |
                  ((cost == best_cost) & (hamming == best_hamming) &
                   (np.uint64(pattern) < best_pattern)))
        best_cost[update] = cost[update]
        best_hamming[update] = hamming[update]
        best_pattern[update] = np.uint64(pattern)
    nearest = int(np.dot(counts, best_cost))
    corrections = int(np.dot(counts, best_hamming))
    pwp = int(counts[best_pattern != 0].sum())
    require(pwp + corrections == nearest, "PWP/correction conservation failure")
    used = len(set(int(value) for value in best_pattern if int(value) != 0))
    return {
        "patterns": patterns,
        "baseline": baseline,
        "exact": exact,
        "exact_hits": exact_hits,
        "nearest": nearest,
        "pwp": pwp,
        "corrections": corrections,
        "vectors": int(counts.sum()),
        "used": used,
    }


def recompute_m70(histograms, operator_names):
    output = {}
    for width in WIDTHS:
        partitions = FEATURES // width
        for q in Q_VALUES:
            aggregate = Counter()
            operator_rows = []
            for op, operator in enumerate(operator_names):
                local = Counter()
                entries = 0
                for partition in range(partitions):
                    result = evaluate_partition(
                        histograms["calibration"][(op, width, partition)],
                        histograms["heldout"][(op, width, partition)], width, q)
                    entries += len(result["patterns"])
                    for key in ("baseline", "exact", "exact_hits", "nearest",
                                "pwp", "corrections", "vectors", "used"):
                        local[key] += result[key]
                local["entries"] = entries
                aggregate.update(local)
                operator_rows.append((operator, local))
            output[(width, q)] = (aggregate, operator_rows)
            print("[INDEPENDENT DSE] k={} q={} baseline={} nearest={} speedup={:.12f}".format(
                width, q, aggregate["baseline"], aggregate["nearest"],
                aggregate["baseline"] / float(aggregate["nearest"])), flush=True)
    return output


def compare_m70(recomputed, payload, operator_names):
    mismatch = []
    configurations = dict(
        ((int(row["partition_bits"]), int(row["maximum_patterns_per_partition"])), row)
        for row in payload["configurations"])
    require(set(configurations) == set(recomputed), "M70 configuration extent mismatch")
    mapping = {
        "heldout_baseline_bit_sparse_vector_ops": "baseline",
        "exact_match_fallback_vector_ops": "exact",
        "nearest_signed_vector_ops": "nearest",
        "nearest_pwp_vector_ops": "pwp",
        "nearest_correction_vector_ops": "corrections",
        "exact_pattern_hit_vectors": "exact_hits",
        "heldout_partition_vectors": "vectors",
        "codebook_entries": "entries",
        "heldout_used_codebook_entries": "used",
    }
    for key, (aggregate, operator_rows) in recomputed.items():
        row = configurations[key]
        for json_key, local_key in mapping.items():
            if json_key in row and int(row[json_key]) != int(aggregate[local_key]):
                mismatch.append("M70 {} aggregate {}".format(key, json_key))
        expected_table_bytes = aggregate["entries"] * key[0] // 8
        if int(row["pattern_table_bytes"]) != expected_table_bytes:
            mismatch.append("M70 {} pattern bytes".format(key))
        expected_pwp = aggregate["entries"] * 8 * 228
        if int(row["all_codebook_pwp_bytes_signed19"]) != expected_pwp:
            mismatch.append("M70 {} PWP bytes".format(key))
        rows_by_name = dict((item["operator"], item) for item in row["operators"])
        for operator, local in operator_rows:
            json_row = rows_by_name[operator]
            for json_key, local_key in mapping.items():
                if json_key in json_row and int(json_row[json_key]) != int(local[local_key]):
                    mismatch.append("M70 {} {} {}".format(key, operator, json_key))
    require(not mismatch, "; ".join(mismatch[:20]))
    best = min(configurations.values(), key=lambda row: (
        -float(row["nearest_signed_speedup"]),
        int(row["all_codebook_pwp_bytes_signed19"]),
        int(row["partition_bits"]), int(row["maximum_patterns_per_partition"])))
    require(best == payload["best_vector_op_configuration"],
            "M70 best configuration mismatch")
    return configurations


def compare_m71(histograms, payload, operator_names):
    require(payload["split"]["catalog_samples"] == [0, 1, 2, 3, 4],
            "M71 catalog split mismatch")
    require(payload["split"]["heldout_samples_excluded"] == [5, 6, 7, 8, 9],
            "M71 heldout exclusion mismatch")
    operators = dict((row["operator"], row) for row in payload["operators"])
    totals = Counter()
    for op, operator in enumerate(operator_names):
        row = operators[operator]
        require(len(row["partitions"]) == 432, "M71 partition extent mismatch")
        local = Counter()
        for partition, partition_row in enumerate(row["partitions"]):
            counter = histograms["calibration"][(op, 16, partition)]
            patterns = choose_patterns(counter, 16)
            observed = [(int(item["value_hex"], 16), int(item["calibration_count"]))
                        for item in partition_row["patterns"]]
            expected = [(value, int(counter[value])) for value in patterns]
            require(observed == expected,
                    "M71 catalog mismatch op={} partition={}".format(op, partition))
            vectors = sum(counter.values())
            baseline = sum(count * bin(value).count("1")
                           for value, count in counter.items())
            pattern_set = frozenset(patterns)
            hits = sum(count for value, count in counter.items()
                       if value != 0 and value in pattern_set)
            exact = sum(count if value != 0 and value in pattern_set
                        else count * bin(value).count("1")
                        for value, count in counter.items())
            require(int(partition_row["calibration_vectors"]) == vectors,
                    "M71 vector count mismatch")
            require(int(partition_row["calibration_baseline_bit_sparse_vector_ops"]) == baseline,
                    "M71 baseline count mismatch")
            require(int(partition_row["calibration_exact_match_fallback_vector_ops"]) == exact,
                    "M71 exact count mismatch")
            require(int(partition_row["calibration_exact_pattern_hits"]) == hits,
                    "M71 hit count mismatch")
            local.update({"entries": len(patterns), "vectors": vectors,
                          "baseline": baseline, "exact": exact, "hits": hits})
        require(int(row["codebook_entries"]) == local["entries"],
                "M71 operator entry mismatch")
        totals.update(local)
    capacity = payload["hardware_capacity"]
    require(totals["entries"] == 27648 == int(capacity["total_codebook_entries"]),
            "M71 total entry mismatch")
    require(int(capacity["pattern_table_bytes"]) == 27648 * 2,
            "M71 pattern capacity mismatch")
    require(payload["int8_pwp_bound"]["exact_sum_range"] == [-2048, 2032],
            "M71 PWP range mismatch")
    require(int(payload["int8_pwp_bound"]["required_signed_bits"]) == 12,
            "M71 PWP signed width mismatch")
    require(int(payload["int8_pwp_bound"]["pwp_vector_bytes_bit_tight"]) == 144,
            "M71 PWP vector byte mismatch")
    require(int(capacity["all_pwp_bytes_bit_tight"]) == 27648 * 8 * 144,
            "M71 all-PWP capacity mismatch")
    require(int(capacity["one_partition_one_output_block_pwp_working_set_bytes"]) == 16 * 144,
            "M71 working-set mismatch")
    observation = payload["calibration_observation_only"]
    require(int(observation["partition_vectors"]) == totals["vectors"],
            "M71 observation vector mismatch")
    require(int(observation["baseline_bit_sparse_vector_ops"]) == totals["baseline"],
            "M71 observation baseline mismatch")
    require(int(observation["exact_match_fallback_vector_ops"]) == totals["exact"],
            "M71 observation exact mismatch")
    return totals


def independent_lloyd(counter, q=16, iterations=20):
    """Independent deterministic weighted binary Lloyd implementation."""
    values = sorted(value for value in counter
                    if value != 0 and bin(value).count("1") != 1)
    require(len(values) >= q, "M72 filtered population smaller than q")

    def refill(current):
        centers = list(dict.fromkeys(current))
        available = [value for value in values if value not in set(centers)]
        while len(centers) < q:
            require(available, "M72 independent refill exhausted")
            if centers:
                def score(value):
                    distance = min(bin(value ^ center).count("1")
                                   for center in centers)
                    return (counter[value] * distance, distance,
                            counter[value], -value)
                chosen = max(available, key=score)
            else:
                chosen = max(available,
                             key=lambda value: (counter[value], -value))
            centers.append(chosen)
            available.remove(chosen)
        return centers

    centers = refill([])
    completed = 0
    for iteration in range(iterations):
        total = [0] * q
        one_counts = [[0] * 16 for _ in range(q)]
        for value in values:
            assignment = min(
                range(q),
                key=lambda index: (bin(value ^ centers[index]).count("1"),
                                   index))
            count = counter[value]
            total[assignment] += count
            for bit in range(16):
                one_counts[assignment][bit] += count * ((value >> bit) & 1)
        updated = []
        for index in range(q):
            if total[index] == 0:
                updated.append(centers[index])
                continue
            center = sum((1 << bit) for bit in range(16)
                         if 2 * one_counts[index][bit] > total[index])
            if center != 0:
                updated.append(center)
        updated = refill(updated)
        completed = iteration + 1
        if updated == centers:
            break
        centers = updated
    return centers, completed


def evaluate_centers(counter, centers):
    result = Counter()
    center_set = frozenset(centers)
    used = set()
    for value, count in counter.items():
        population = bin(value).count("1")
        result["partition_vectors"] += count
        result["nonzero_partition_vectors"] += count if value != 0 else 0
        result["baseline_bit_sparse_vector_ops"] += count * population
        result["exact_pattern_hits"] += (
            count if value != 0 and value in center_set else 0)
        hamming, center = min((bin(value ^ candidate).count("1"), candidate)
                              for candidate in centers)
        if 1 + hamming < population:
            result["nearest_signed_vector_ops"] += count * (1 + hamming)
            result["nearest_pwp_vector_ops"] += count
            result["nearest_correction_vector_ops"] += count * hamming
            used.add(center)
        else:
            result["nearest_signed_vector_ops"] += count * population
            result["nearest_correction_vector_ops"] += count * population
    result["used_centers"] = len(used)
    require(result["nearest_pwp_vector_ops"] +
            result["nearest_correction_vector_ops"] ==
            result["nearest_signed_vector_ops"],
            "M72 independent conservation failure")
    return result


def compare_m72(histograms, payload, operator_names):
    require(payload["identity"]["analyzer_sha256"] == sha256(M72_SOURCE),
            "M72 source identity mismatch")
    require(payload["split"]["calibration_samples_within_valid825"] ==
            [0, 1, 2, 3, 4],
            "M72 calibration split mismatch")
    require(payload["split"]["heldout_samples_within_valid825"] ==
            [5, 6, 7, 8, 9],
            "M72 heldout split mismatch")
    require(payload["split"]["validation_or_test_data_used_for_centers"] is True,
            "M72 validation usage must be explicit")
    require(payload["split"]["train_catalog_eligible"] is False,
            "M72 valid825 screen must not be train-catalog eligible")
    require(payload["promotion_gate"]["paft_allowed"] is False,
            "M72 valid825 screen must block PAFT")
    rows = dict((row["operator"], row) for row in payload["operators"])
    aggregate = Counter()
    for op, operator in enumerate(operator_names):
        row = rows[operator]
        require(len(row["partitions"]) == 432, "M72 partition extent mismatch")
        local = Counter()
        for partition, partition_row in enumerate(row["partitions"]):
            centers, completed = independent_lloyd(
                histograms["calibration"][(op, 16, partition)])
            observed = [int(value, 16) for value in partition_row["centers_hex"]]
            require(observed == centers,
                    "M72 center mismatch op={} partition={}".format(op, partition))
            require(int(partition_row["lloyd_iterations"]) == completed,
                    "M72 Lloyd iteration mismatch")
            heldout = evaluate_centers(
                histograms["heldout"][(op, 16, partition)], centers)
            for key, value in heldout.items():
                require(int(partition_row["heldout"][key]) == value,
                        "M72 heldout mismatch op={} partition={} key={}".format(
                            op, partition, key))
                if key != "used_centers":
                    local[key] += value
        for key, value in local.items():
            require(int(row[key]) == value,
                    "M72 operator aggregate mismatch {} {}".format(operator, key))
        aggregate.update(local)
    for key, value in aggregate.items():
        if key in payload["heldout"]:
            require(int(payload["heldout"][key]) == value,
                    "M72 aggregate mismatch " + key)
    require(int(payload["hardware_capacity"]["codebook_entries"]) == 27648,
            "M72 entry capacity mismatch")
    require(int(payload["hardware_capacity"]["all_pwp_bytes_bit_tight"]) ==
            27648 * 8 * 144, "M72 PWP capacity mismatch")
    return aggregate


def main():
    require(sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256,
            "frozen M40 manifest SHA drift")
    manifest = strict_json(MANIFEST)
    m70 = strict_json(M70_RESULT)
    m71 = strict_json(M71_RESULT)
    m72 = strict_json(M72_RESULT)
    require(len(manifest["records"]) == 40, "M40 record extent mismatch")
    operator_names = sorted(set(row["operator"] for row in manifest["records"]))
    require(len(operator_names) == 4, "operator extent mismatch")
    require(set(row["sample_id"] for row in manifest["records"]
                if row["sample_id"] in CALIBRATION) == CALIBRATION,
            "calibration identity mismatch")
    require(set(row["sample_id"] for row in manifest["records"]
                if row["sample_id"] in HELDOUT) == HELDOUT,
            "heldout identity mismatch")
    require(CALIBRATION.isdisjoint(HELDOUT), "split leakage")
    require(len(set(manifest["cohort"]["sample_keys"])) == 10,
            "sample key duplication")
    require(sha256(VALID825_LIST) == EXPECTED_VALID825_LIST_SHA256,
            "valid825 list identity drift")
    valid825_keys = [row.strip() for row in
                     VALID825_LIST.read_text(encoding="utf-8").splitlines()
                     if row.strip()]
    require(len(valid825_keys) == 825, "valid825 population drift")
    require(valid825_keys[:10] == manifest["cohort"]["sample_keys"],
            "M40 cohort is not the first ten valid825 samples")
    require(len(set(key.rsplit("_", 1)[0]
                    for key in manifest["cohort"]["sample_keys"])) == 1,
            "expected same-sequence cohort changed")
    require(m70["identity"]["analyzer_sha256"] == sha256(M70_SOURCE),
            "M70 source identity mismatch")
    require(m71["identity"]["builder_sha256"] == sha256(M71_SOURCE),
            "M71 source identity mismatch")
    histograms = collect_histograms(manifest, operator_names)
    recomputed = recompute_m70(histograms, operator_names)
    configurations = compare_m70(recomputed, m70, operator_names)
    m71_totals = compare_m71(histograms, m71, operator_names)
    m72_totals = compare_m72(histograms, m72, operator_names)

    best = configurations[(16, 128)]
    q16 = configurations[(16, 16)]
    heldout_partition_vectors = sum(
        int(row["heldout_partition_vectors"]) for row in q16["operators"])
    require(heldout_partition_vectors == 25920000,
            "heldout partition population mismatch")
    # M70's counters omit the eight 96-lane output blocks.  Ratios are
    # invariant, but physical 96-lane vector additions are 8x.  A matcher
    # assignment can be shared across the eight blocks if the schedule keeps
    # the assignment live, so charge one assignment per input partition here.
    q16_physical_baseline = int(q16["heldout_baseline_bit_sparse_vector_ops"]) * 8
    q16_physical_candidate = int(q16["nearest_signed_vector_ops"]) * 8
    best_physical_candidate = int(best["nearest_signed_vector_ops"]) * 8
    q16_serial_matcher = q16_physical_candidate + heldout_partition_vectors
    best_serial_matcher = best_physical_candidate + heldout_partition_vectors
    q16_baseline_bytes = int(q16["heldout_baseline_bit_sparse_vector_ops"]) * 96
    q16_candidate_bytes = (int(q16["nearest_correction_vector_ops"]) * 96 +
                           int(q16["nearest_pwp_vector_ops"]) * 144)
    best_candidate_bytes_12 = (int(best["nearest_correction_vector_ops"]) * 96 +
                               int(best["nearest_pwp_vector_ops"]) * 144)
    print("[INDEPENDENT CAPACITY] pwp_range=[-2048,2032] bits=12 "
          "all_pwp={} working_set={}".format(
              m71["hardware_capacity"]["all_pwp_bytes_bit_tight"],
              m71["hardware_capacity"]["one_partition_one_output_block_pwp_working_set_bytes"]))
    print("[INDEPENDENT PHYSICAL_OPS] baseline={} q16={} q128={} "
          "m70_counters_are_per_output_block_equivalents=true".format(
              q16_physical_baseline, q16_physical_candidate,
              best_physical_candidate))
    print("[INDEPENDENT FAIRNESS] q16 one_shared_matcher_cycle_speedup={:.12f} "
          "bit_tight_byte_speedup={:.12f}".format(
              q16_physical_baseline / float(q16_serial_matcher),
              q16_baseline_bytes / float(q16_candidate_bytes)))
    print("[INDEPENDENT FAIRNESS] q128 one_shared_matcher_cycle_speedup={:.12f} "
          "hypothetical_12b_byte_speedup={:.12f}".format(
              q16_physical_baseline / float(best_serial_matcher),
              q16_baseline_bytes / float(best_candidate_bytes_12)))
    print("[INDEPENDENT SPLIT] record_disjoint=true sample_key_disjoint=true "
          "sequence_disjoint=false sequence=zurich_city_09_a")
    print("[INDEPENDENT LEAKAGE] m40_cohort_is_valid825_first10=true "
          "m71_catalog_uses_valid825_first5=true "
          "m71_declared_test_or_validation_data_used={}".format(
              str(m71["split"]["test_or_validation_data_used"]).lower()))
    print("[INDEPENDENT M71] calibration_entries={} vectors={} baseline={} exact={}".format(
        m71_totals["entries"], m71_totals["vectors"],
        m71_totals["baseline"], m71_totals["exact"]))
    print("[INDEPENDENT M72] baseline={} nearest={} speedup={:.12f}".format(
        m72_totals["baseline_bit_sparse_vector_ops"],
        m72_totals["nearest_signed_vector_ops"],
        m72_totals["baseline_bit_sparse_vector_ops"] /
        float(m72_totals["nearest_signed_vector_ops"])))
    print("PASS_M70_M71_M72_INDEPENDENT_RECOMPUTE_ZERO_MISMATCH")


if __name__ == "__main__":
    main()
