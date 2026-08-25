#!/usr/bin/env python3
"""Independent M74 hammer: direct M40 plane decode, no production imports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
M40_DIR = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
MANIFEST = M40_DIR / "m40_bottleneck_packed_source_manifest.json"
M72_RESULT = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M74_RESULT = HW / (
    "results/m74_dual_pwp_signed_decomposition_valid825_internal_dev_r1_20260823/"
    "m74_dual_pwp_signed_decomposition.json")

T, C, H, W = 10, 768, 15, 20
K, Q, PARTITIONS = 16, 16, 432
ROWS = T * H * W
BEAMS = (1, 2, 4, 16)
PWP_VECTOR_BYTES = 144
MASK16 = (1 << 16) - 1


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

    def hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=hook, parse_constant=reject)


def bit_truth_table():
    rows = []
    mismatches = 0
    for x in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                plus_residual = x - a - b
                minus_residual = x - a + b
                plus_ok = a + b + plus_residual == x
                minus_ok = a - b + minus_residual == x
                mismatches += int(not plus_ok) + int(not minus_ok)
                rows.append({
                    "x": x, "a": a, "b": b,
                    "plus_residual": plus_residual,
                    "plus_l1": abs(plus_residual),
                    "plus_identity_ok": plus_ok,
                    "minus_residual": minus_residual,
                    "minus_l1": abs(minus_residual),
                    "minus_identity_ok": minus_ok,
                })
    require(mismatches == 0, "one-bit signed identity failure")
    return rows


def decode_partition_vectors(record, popcount):
    require(record["shape"] == [T, 1, C, H, W], "M40 shape drift")
    path = M40_DIR / record["packed_file"]
    require(path.is_file() and sha256(path) == record["packed_file_sha256"],
            "M40 packed file identity drift")
    raw = path.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes, "M40 packed extent drift")
    positive = np.frombuffer(raw[:plane_bytes], dtype=np.uint8)
    negative = np.frombuffer(raw[plane_bytes:2 * plane_bytes], dtype=np.uint8)
    require(not np.any(negative), "heldout trace unexpectedly has negative support")
    bits = np.unpackbits(positive, bitorder="little")[:T * C * H * W]
    bits = bits.reshape(T, C, H, W)
    require(int(bits.sum()) == record["positive_count"] == record["nonzero_count"],
            "M40 positive population mismatch")

    padded = np.pad(bits, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # Independent im2col.  The final feature order is C, KY, KX, matching
    # feature=(cin*3+ky)*3+kx without calling M43/M72 code.
    patches = np.empty((T, H, W, C, 3, 3), dtype=np.uint8)
    for ky in range(3):
        for kx in range(3):
            patches[:, :, :, :, ky, kx] = padded[
                :, :, ky:ky + H, kx:kx + W].transpose(0, 2, 3, 1)
    feature_rows = patches.reshape(ROWS, C * 9)
    packed = np.packbits(feature_rows, axis=1, bitorder="little")
    values = np.ascontiguousarray(packed).view("<u2").reshape(ROWS, PARTITIONS)

    vertical = np.array([2 if y in (0, H - 1) else 3 for y in range(H)],
                        dtype=np.int64)
    horizontal = np.array([2 if x in (0, W - 1) else 3 for x in range(W)],
                          dtype=np.int64)
    fanout = vertical[:, None] * horizontal[None, :]
    independently_expanded = int((bits * fanout[None, None, :, :]).sum())
    partition_popcount = int(popcount[values].sum())
    require(partition_popcount == independently_expanded,
            "im2col source conservation failure")
    return values, {
        "sample_id": record["sample_id"],
        "operator": record["operator"],
        "input_positive_bits": int(bits.sum()),
        "expanded_feature_bits": partition_popcount,
        "packed_file_sha256": record["packed_file_sha256"],
    }


def weighted_sum(counts, values):
    return int(np.sum(counts * values.astype(np.int64), dtype=np.int64))


def direct_selected_bit_check(x, a, b, mode, expected_l1):
    l1 = 0
    for bit in range(K):
        xb = (x >> bit) & 1
        ab = (a >> bit) & 1
        bb = (b >> bit) & 1
        base = ab + bb if mode == 0 else ab - bb
        residual = xb - base
        require(base + residual == xb, "selected-pair bit identity mismatch")
        l1 += abs(residual)
    require(l1 == expected_l1, "selected-pair L1 mismatch")


def evaluate_partition(values, centers, popcount):
    unique, counts = np.unique(values, return_counts=True)
    unique = unique.astype(np.uint16)
    counts = counts.astype(np.int64)
    centers = np.asarray(centers, dtype=np.uint16)
    require(len(centers) == Q and len(set(int(x) for x in centers)) == Q,
            "M72 center extent/uniqueness drift")

    baseline = popcount[unique].astype(np.int64)
    distances = popcount[np.bitwise_xor(unique[:, None], centers[None, :])]
    # Stable sort gives Hamming distance then original center index.
    ranked = np.argsort(distances, axis=1, kind="stable")
    single_raw = 1 + distances[np.arange(len(unique)), ranked[:, 0]].astype(np.int64)
    single_candidate = np.minimum(baseline, single_raw)
    result = {}
    best_key = np.full(len(unique), np.iinfo(np.int64).max, dtype=np.int64)

    for slot in range(Q):
        first_index = ranked[:, slot].astype(np.int64)
        first = centers[first_index]
        for second_index, second_scalar in enumerate(centers):
            second = np.uint16(second_scalar)

            # Direct coefficient-class L1 accounting for x-a-b.
            plus_zero = np.bitwise_not(np.bitwise_or(first, second)) & MASK16
            plus_one = np.bitwise_xor(first, second)
            plus_two = np.bitwise_and(first, second)
            not_x = np.bitwise_not(unique) & MASK16
            plus_l1 = (
                popcount[np.bitwise_and(unique, plus_zero)].astype(np.int64)
                + popcount[np.bitwise_and(not_x, plus_one)].astype(np.int64)
                + popcount[np.bitwise_and(unique, plus_two)].astype(np.int64)
                + 2 * popcount[np.bitwise_and(not_x, plus_two)].astype(np.int64))

            # Direct coefficient-class L1 accounting for x-a+b.
            minus_positive = np.bitwise_and(first, np.bitwise_not(second) & MASK16)
            minus_negative = np.bitwise_and(np.bitwise_not(first) & MASK16, second)
            minus_zero = np.bitwise_not(
                np.bitwise_or(minus_positive, minus_negative)) & MASK16
            minus_l1 = (
                popcount[np.bitwise_and(not_x, minus_positive)].astype(np.int64)
                + popcount[np.bitwise_and(not_x, minus_negative)].astype(np.int64)
                + popcount[np.bitwise_and(unique, minus_zero)].astype(np.int64)
                + 2 * popcount[np.bitwise_and(unique, minus_negative)].astype(np.int64))

            plus_cost = 2 + plus_l1
            minus_cost = 2 + minus_l1
            plus_key = (plus_cost * 512 + first_index * 16 + second_index)
            minus_key = (minus_cost * 512 + 256
                         + first_index * 16 + second_index)
            best_key = np.minimum(best_key, plus_key)
            best_key = np.minimum(best_key, minus_key)

        beam = slot + 1
        if beam not in BEAMS:
            continue
        pair_cost = best_key // 512
        pair_mode = (best_key // 256) & 1
        pair_first = (best_key // 16) & 15
        pair_second = best_key & 15
        candidate = np.minimum(np.minimum(baseline, single_raw), pair_cost)
        dual = ((candidate == pair_cost) & (pair_cost < baseline)
                & (pair_cost < single_raw))
        single = ((~dual) & (candidate == single_raw) & (single_raw < baseline))
        fallback = ~(dual | single)

        row = Counter()
        row["partition_vectors"] = int(counts.sum())
        row["baseline_bit_sparse_vector_ops"] = weighted_sum(counts, baseline)
        row["single_pwp_vector_ops"] = weighted_sum(counts, single_candidate)
        row["dual_candidate_vector_ops"] = weighted_sum(counts, candidate)
        row["dual_selected_vectors"] = int(counts[dual].sum())
        row["single_selected_vectors"] = int(counts[single].sum())
        row["bit_sparse_fallback_vectors"] = int(counts[fallback].sum())
        row["dual_pwp_reads"] = 2 * row["dual_selected_vectors"]
        row["single_pwp_reads"] = row["single_selected_vectors"]
        row["dual_correction_vector_ops"] = weighted_sum(
            counts[dual], pair_cost[dual] - 2)
        row["single_correction_vector_ops"] = weighted_sum(
            counts[single], single_raw[single] - 1)
        row["bit_sparse_fallback_vector_ops"] = weighted_sum(
            counts[fallback], baseline[fallback])
        row["dual_plus_selected_vectors"] = int(counts[dual & (pair_mode == 0)].sum())
        row["dual_minus_selected_vectors"] = int(counts[dual & (pair_mode == 1)].sum())
        row["dual_same_center_selected_vectors"] = int(
            counts[dual & (pair_first == pair_second)].sum())

        selected_unique = np.flatnonzero(dual)
        for index in selected_unique:
            mode = int(pair_mode[index])
            a = int(centers[int(pair_first[index])])
            b = int(centers[int(pair_second[index])])
            direct_selected_bit_check(
                int(unique[index]), a, b, mode, int(pair_cost[index]) - 2)
        row["selected_unique_pattern_identity_checks"] = len(selected_unique)
        row["selected_unique_pattern_bit_checks"] = len(selected_unique) * K

        require(row["dual_pwp_reads"] == 2 * row["dual_selected_vectors"],
                "dual PWP read conservation failure")
        require(
            row["single_pwp_reads"] + row["single_correction_vector_ops"]
            + row["dual_pwp_reads"] + row["dual_correction_vector_ops"]
            + row["bit_sparse_fallback_vector_ops"]
            == row["dual_candidate_vector_ops"],
            "candidate operation conservation failure")

        by_distance_all = Counter()
        by_distance_dual = Counter()
        nearest = distances[np.arange(len(unique)), ranked[:, 0]].astype(np.int64)
        for distance in range(K + 1):
            at_distance = nearest == distance
            by_distance_all[str(distance)] = int(counts[at_distance].sum())
            by_distance_dual[str(distance)] = int(counts[at_distance & dual].sum())
        result[beam] = {
            "metrics": row,
            "nearest_distance_all_vectors": by_distance_all,
            "nearest_distance_dual_vectors": by_distance_dual,
        }
    return result


def add_counter(destination, source):
    for key, value in source.items():
        destination[key] += int(value)


def compare_production(independent, production):
    integer_fields = (
        "partition_vectors", "baseline_bit_sparse_vector_ops",
        "single_pwp_vector_ops", "dual_candidate_vector_ops",
        "single_selected_vectors", "dual_selected_vectors",
        "dual_plus_selected_vectors", "dual_minus_selected_vectors",
        "bit_sparse_fallback_vectors", "single_pwp_reads", "dual_pwp_reads",
        "single_correction_vector_ops", "dual_correction_vector_ops",
        "bit_sparse_fallback_vector_ops", "matcher_hamming_comparisons")
    mismatches = []
    for beam in BEAMS:
        got = independent[str(beam)]
        ref = next(row for row in production["configurations"] if row["beam"] == beam)
        for field in integer_fields:
            if got[field] != ref[field]:
                mismatches.append({"beam": beam, "field": field,
                                   "independent": got[field], "production": ref[field]})
        for got_op in got["operators"]:
            ref_op = next(row for row in ref["operators"]
                          if row["operator"] == got_op["operator"])
            for field in ("baseline_bit_sparse_vector_ops", "single_pwp_vector_ops",
                          "dual_candidate_vector_ops"):
                if got_op[field] != ref_op[field]:
                    mismatches.append({"beam": beam, "operator": got_op["operator"],
                                       "field": field, "independent": got_op[field],
                                       "production": ref_op[field]})
    return mismatches


def threshold_gates(all_hist, dual_hist):
    total = sum(all_hist.values())
    selected = sum(dual_hist.values())
    rows = []
    for threshold in range(K + 1):
        eligible = sum(value for key, value in all_hist.items()
                       if int(key) >= threshold)
        captured = sum(value for key, value in dual_hist.items()
                       if int(key) >= threshold)
        rows.append({
            "minimum_nearest_hamming": threshold,
            "eligible_vectors": eligible,
            "eligible_fraction": eligible / float(total),
            "captured_dual_vectors": captured,
            "dual_capture_fraction": captured / float(selected) if selected else 1.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing output overwrite")

    manifest = strict_json(MANIFEST)
    m72 = strict_json(M72_RESULT)
    m74 = strict_json(M74_RESULT)
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1",
            "M40 schema drift")
    require(len(manifest["records"]) == 40, "M40 population drift")
    operators = sorted(set(row["operator"] for row in manifest["records"]))
    require([row["operator"] for row in m72["operators"]] == operators,
            "M72 operator order drift")

    popcount = np.array([int(value).bit_count() for value in range(1 << K)],
                        dtype=np.uint8)
    values_by_operator = defaultdict(list)
    decode_audit = []
    for record in manifest["records"]:
        if record["sample_id"] < 5:
            continue
        values, audit = decode_partition_vectors(record, popcount)
        values_by_operator[record["operator"]].append(values)
        decode_audit.append(audit)
        print("[independent decode] sample={} operator={} expanded={}".format(
            record["sample_id"], record["operator"], audit["expanded_feature_bits"]),
              flush=True)
    require(all(len(values_by_operator[name]) == 5 for name in operators),
            "heldout sample/operator extent drift")

    aggregates = dict((beam, Counter()) for beam in BEAMS)
    operator_metrics = dict((beam, []) for beam in BEAMS)
    aggregate_all_hist = dict((beam, Counter()) for beam in BEAMS)
    aggregate_dual_hist = dict((beam, Counter()) for beam in BEAMS)

    for op_index, operator in enumerate(operators):
        values = np.concatenate(values_by_operator[operator], axis=0)
        op_totals = dict((beam, Counter()) for beam in BEAMS)
        for partition in range(PARTITIONS):
            center_row = m72["operators"][op_index]["partitions"][partition]
            require(center_row["partition"] == partition, "M72 partition order drift")
            centers = [int(value, 16) for value in center_row["centers_hex"]]
            rows = evaluate_partition(values[:, partition], centers, popcount)
            for beam in BEAMS:
                add_counter(op_totals[beam], rows[beam]["metrics"])
                aggregate_all_hist[beam].update(rows[beam]["nearest_distance_all_vectors"])
                aggregate_dual_hist[beam].update(rows[beam]["nearest_distance_dual_vectors"])
        for beam in BEAMS:
            row = op_totals[beam]
            operator_metrics[beam].append({
                "operator": operator,
                "baseline_bit_sparse_vector_ops": row["baseline_bit_sparse_vector_ops"],
                "single_pwp_vector_ops": row["single_pwp_vector_ops"],
                "dual_candidate_vector_ops": row["dual_candidate_vector_ops"],
                "dual_candidate_speedup": (
                    row["baseline_bit_sparse_vector_ops"] /
                    float(row["dual_candidate_vector_ops"])),
                "dual_selected_vectors": row["dual_selected_vectors"],
                "dual_selected_fraction": (
                    row["dual_selected_vectors"] / float(row["partition_vectors"])),
            })
            add_counter(aggregates[beam], row)
        print("[independent evaluate] operator={}/4 {}".format(op_index + 1, operator),
              flush=True)

    configurations = {}
    for beam in BEAMS:
        row = aggregates[beam]
        matcher = row["partition_vectors"] * (Q + 2 * beam * Q)
        row["matcher_hamming_comparisons"] = matcher
        pwp_reads = row["single_pwp_reads"] + row["dual_pwp_reads"]
        vector_saving = (row["baseline_bit_sparse_vector_ops"]
                         - row["dual_candidate_vector_ops"])
        marginal_pair_saving = (row["single_pwp_vector_ops"]
                                - row["dual_candidate_vector_ops"])
        second_stage = row["partition_vectors"] * 2 * beam * Q
        raw = dict(row)
        raw.update({
            "beam": beam,
            "dual_candidate_speedup": (
                row["baseline_bit_sparse_vector_ops"] /
                float(row["dual_candidate_vector_ops"])),
            "dual_over_single_candidate_cycle_reduction": (
                row["single_pwp_vector_ops"] /
                float(row["dual_candidate_vector_ops"])),
            "pair_fraction": row["dual_selected_vectors"] / float(row["partition_vectors"]),
            "dual_plus_fraction_within_pairs": (
                row["dual_plus_selected_vectors"] /
                float(row["dual_selected_vectors"])),
            "dual_minus_fraction_within_pairs": (
                row["dual_minus_selected_vectors"] /
                float(row["dual_selected_vectors"])),
            "dual_only_pwp_read_bytes": row["dual_pwp_reads"] * PWP_VECTOR_BYTES,
            "all_selected_pwp_read_bytes": pwp_reads * PWP_VECTOR_BYTES,
            "pwp_read_bytes_per_partition_vector": (
                pwp_reads * PWP_VECTOR_BYTES / float(row["partition_vectors"])),
            "matcher_break_even_comparisons_per_cycle": matcher / float(vector_saving),
            "matcher_32way_conditional_speedup": (
                row["baseline_bit_sparse_vector_ops"] /
                float(row["dual_candidate_vector_ops"] +
                      (matcher + 31) // 32)),
            "matcher_serial_conditional_speedup": (
                row["baseline_bit_sparse_vector_ops"] /
                float(row["dual_candidate_vector_ops"] + matcher)),
            "second_stage_hamming_comparisons": second_stage,
            "second_stage_comparisons_per_marginal_pair_op_saved": (
                second_stage / float(marginal_pair_saving)),
            "nearest_hamming_threshold_gates": threshold_gates(
                aggregate_all_hist[beam], aggregate_dual_hist[beam]),
            "operators": operator_metrics[beam],
        })
        configurations[str(beam)] = raw

    mismatches = compare_production(configurations, m74)
    require(not mismatches, "independent/production mismatch: {}".format(mismatches[:3]))
    require(configurations["16"]["baseline_bit_sparse_vector_ops"]
            == m72["heldout"]["baseline_bit_sparse_vector_ops"] == 46432637,
            "M72 heldout baseline mismatch")
    require(configurations["16"]["single_pwp_vector_ops"]
            == m72["heldout"]["nearest_signed_vector_ops"] == 30889399,
            "M72 single-PWP mismatch")

    payload = {
        "schema": "m74_dual_pwp_independent_hammer_reconstruction_v1",
        "status": "PASS_EXACT_RECONSTRUCTION_NO_PRODUCTION_IMPORT",
        "identity": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "manifest_sha256": sha256(MANIFEST),
            "m72_result_sha256": sha256(M72_RESULT),
            "m74_result_sha256": sha256(M74_RESULT),
            "production_analyzer_imported": False,
        },
        "population": {
            "heldout_samples": [5, 6, 7, 8, 9],
            "operators": operators,
            "records_decoded": len(decode_audit),
            "rows_per_record": ROWS,
            "partitions_per_row": PARTITIONS,
            "partition_vectors": configurations["16"]["partition_vectors"],
            "source_manifest_declared_identity": "H67_EP35_S10",
            "m72_split_label": m72["split"]["source_population"],
        },
        "decode_audit": decode_audit,
        "arithmetic_bit_truth_table": bit_truth_table(),
        "configurations": configurations,
        "production_comparison": {
            "integer_and_operator_mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "reported_speedups": [
                configurations[str(beam)]["dual_candidate_speedup"] for beam in BEAMS],
            "expected_reported_speedups": [
                1.5923286763822577, 1.597795772692031,
                1.5991595504147487, 1.59972581919019],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS independent M74 beams={}".format(
        ",".join("{}:{:.9f}".format(b, configurations[str(b)][
            "dual_candidate_speedup"]) for b in BEAMS)), flush=True)


if __name__ == "__main__":
    main()
