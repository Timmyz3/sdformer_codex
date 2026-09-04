#!/usr/bin/env python3
"""Screen exact dual-PWP signed decomposition on the M72 internal holdout.

For a binary input x and codebook patterns a/b, the engine may compute either
W*a + W*b or W*a - W*b, followed by exact signed unit-weight corrections.  Two
PWP reads are charged explicitly.  A small beam of the best single-pattern
matches limits the second-stage search to beam*Q pairs instead of Q^2.

This is a valid825-internal mechanism screen, not a train catalog, independent
validation result, cycle model, or paper speedup.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
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
MASK16 = (1 << 16) - 1
BEAMS = (1, 2, 4, 16)


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
    spec = importlib.util.spec_from_file_location("m74_m72", str(M72_ANALYZER))
    require(spec is not None and spec.loader is not None, "cannot import M72")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plus_residual_l1(x, a, b, popcount):
    overlap = a & b
    union = a | b
    return (
        popcount[x & overlap]
        + popcount[x & (MASK16 ^ union)]
        + popcount[(MASK16 ^ x) & (a ^ b)]
        + 2 * popcount[(MASK16 ^ x) & overlap]
    )


def minus_residual_l1(x, a, b, popcount):
    positive = a & (MASK16 ^ b)
    negative = b & (MASK16 ^ a)
    nonzero = positive | negative
    return (
        popcount[x & (MASK16 ^ nonzero)]
        + 2 * popcount[x & negative]
        + popcount[(MASK16 ^ x) & nonzero]
    )


def evaluate(counter, centers, popcount, beam):
    totals = Counter()
    for x, count in counter.items():
        baseline = popcount[x]
        ranked = sorted(
            ((popcount[x ^ center], index, center)
             for index, center in enumerate(centers)))
        single = 1 + ranked[0][0]
        pair_cost = None
        pair_mode = None
        pair_first = None
        pair_second = None
        for _distance, first_index, first in ranked[:beam]:
            for second_index, second in enumerate(centers):
                plus = 2 + plus_residual_l1(x, first, second, popcount)
                minus = 2 + minus_residual_l1(x, first, second, popcount)
                for cost, mode in ((plus, "plus"), (minus, "minus")):
                    key = (cost, 0 if mode == "plus" else 1,
                           first_index, second_index)
                    current = (pair_cost,
                               0 if pair_mode == "plus" else 1,
                               pair_first, pair_second) if pair_cost is not None else None
                    if current is None or key < current:
                        pair_cost = cost
                        pair_mode = mode
                        pair_first = first_index
                        pair_second = second_index
        require(pair_cost is not None, "M74 empty pair search")
        candidate = min(baseline, single, pair_cost)
        totals["partition_vectors"] += count
        totals["baseline_bit_sparse_vector_ops"] += count * baseline
        totals["single_pwp_vector_ops"] += count * min(baseline, single)
        totals["dual_candidate_vector_ops"] += count * candidate
        totals["matcher_hamming_comparisons"] += count * (
            len(centers) + 2 * beam * len(centers))
        if candidate == pair_cost and pair_cost < baseline and pair_cost < single:
            totals["dual_selected_vectors"] += count
            totals["dual_pwp_reads"] += count * 2
            totals["dual_correction_vector_ops"] += count * (pair_cost - 2)
            totals["dual_{}_selected_vectors".format(pair_mode)] += count
        elif candidate == single and single < baseline:
            totals["single_selected_vectors"] += count
            totals["single_pwp_reads"] += count
            totals["single_correction_vector_ops"] += count * (single - 1)
        else:
            totals["bit_sparse_fallback_vectors"] += count
            totals["bit_sparse_fallback_vector_ops"] += count * baseline
    require(
        totals["single_pwp_reads"]
        + totals["single_correction_vector_ops"]
        + totals["dual_pwp_reads"]
        + totals["dual_correction_vector_ops"]
        + totals["bit_sparse_fallback_vector_ops"]
        == totals["dual_candidate_vector_ops"],
        "M74 candidate operation conservation failure")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M74 output overwrite")
    require(M72_RESULT.is_file() and sha256(M72_RESULT) == EXPECTED_M72_RESULT_SHA256,
            "M74 M72 result identity drift")
    m72 = load_m72()
    result = m72.strict_json(M72_RESULT)
    require(result["split"]["train_catalog_eligible"] is False,
            "M74 requires the explicitly internal-only M72 identity")
    manifest = m72.strict_json(m72.MANIFEST_PATH)
    operator_names = sorted(set(row["operator"] for row in manifest["records"]))
    m43 = m72.load_m43()
    _calibration, heldout = m72.collect_histograms(
        m43, manifest, operator_names)
    rows = []
    for beam in BEAMS:
        aggregate = Counter()
        operator_rows = []
        for op, operator in enumerate(operator_names):
            op_total = Counter()
            centers_by_partition = result["operators"][op]["partitions"]
            require(len(centers_by_partition) == m72.PARTITIONS,
                    "M74 partition extent drift")
            for partition, center_row in enumerate(centers_by_partition):
                require(center_row["partition"] == partition,
                        "M74 partition order drift")
                centers = [int(value, 16) for value in center_row["centers_hex"]]
                totals = evaluate(
                    heldout[(op, partition)], centers, m72.POPCOUNT, beam)
                op_total.update(totals)
            operator_rows.append({
                "operator": operator,
                "baseline_bit_sparse_vector_ops":
                    op_total["baseline_bit_sparse_vector_ops"],
                "single_pwp_vector_ops": op_total["single_pwp_vector_ops"],
                "dual_candidate_vector_ops": op_total["dual_candidate_vector_ops"],
                "dual_candidate_speedup": (
                    op_total["baseline_bit_sparse_vector_ops"] /
                    op_total["dual_candidate_vector_ops"]),
                "dual_selected_fraction": (
                    op_total["dual_selected_vectors"] /
                    float(op_total["partition_vectors"])),
            })
            aggregate.update(op_total)
        rows.append({
            "beam": beam,
            "first_stage_patterns": m72.Q,
            "second_stage_signed_pair_candidates_per_vector": 2 * beam * m72.Q,
            "matcher_hamming_comparisons": aggregate["matcher_hamming_comparisons"],
            "partition_vectors": aggregate["partition_vectors"],
            "baseline_bit_sparse_vector_ops": aggregate["baseline_bit_sparse_vector_ops"],
            "single_pwp_vector_ops": aggregate["single_pwp_vector_ops"],
            "dual_candidate_vector_ops": aggregate["dual_candidate_vector_ops"],
            "single_pwp_speedup": (
                aggregate["baseline_bit_sparse_vector_ops"] /
                aggregate["single_pwp_vector_ops"]),
            "dual_candidate_speedup": (
                aggregate["baseline_bit_sparse_vector_ops"] /
                aggregate["dual_candidate_vector_ops"]),
            "dual_over_single_candidate_cycle_reduction": (
                aggregate["single_pwp_vector_ops"] /
                aggregate["dual_candidate_vector_ops"]),
            "single_selected_vectors": aggregate["single_selected_vectors"],
            "dual_selected_vectors": aggregate["dual_selected_vectors"],
            "dual_plus_selected_vectors": aggregate["dual_plus_selected_vectors"],
            "dual_minus_selected_vectors": aggregate["dual_minus_selected_vectors"],
            "bit_sparse_fallback_vectors": aggregate["bit_sparse_fallback_vectors"],
            "single_pwp_reads": aggregate["single_pwp_reads"],
            "dual_pwp_reads": aggregate["dual_pwp_reads"],
            "single_correction_vector_ops": aggregate["single_correction_vector_ops"],
            "dual_correction_vector_ops": aggregate["dual_correction_vector_ops"],
            "bit_sparse_fallback_vector_ops":
                aggregate["bit_sparse_fallback_vector_ops"],
            "operators": operator_rows,
        })
        print("[M74] beam={} single={:.6f}x dual={:.6f}x pair_fraction={:.6f}".format(
            beam, rows[-1]["single_pwp_speedup"],
            rows[-1]["dual_candidate_speedup"],
            aggregate["dual_selected_vectors"] /
            float(aggregate["partition_vectors"])), flush=True)

    best = min(rows, key=lambda row: row["dual_candidate_vector_ops"])
    payload = {
        "schema": "m74_dual_pwp_signed_decomposition_valid825_internal_screen_v1",
        "status": "PASS_M74_VALID825_INTERNAL_DUAL_PWP_SCREEN_CYCLES_RTL_UNADMITTED",
        "identity": {
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "m72_result_sha256": sha256(M72_RESULT),
        },
        "arithmetic_identity": {
            "plus": "W*x = PWP[a] + PWP[b] + W*(x-a-b)",
            "minus": "W*x = PWP[a] - PWP[b] + W*(x-a+b)",
            "correction_cost": "L1 integer residual, so coefficient magnitude two costs two unit vector operations",
            "pwp_reads_charged": 2,
        },
        "best": best,
        "configurations": rows,
        "promotion_gate": {
            "minimum_internal_vector_op_speedup": 2.0,
            "passes_internal_vector_op_gate": best["dual_candidate_speedup"] >= 2.0,
            "train_trace_rerun_required": True,
            "paft_required_before_rtl": True,
            "cycle_model_required_before_rtl": True,
        },
        "admission": {
            "valid825_internal_vector_operation_screen": True,
            "train_catalog": False,
            "independent_validation": False,
            "matcher_cycles_charged": False,
            "pwp_sram_dram_traffic_charged": False,
            "cycle_accurate_speedup": False,
            "rtl": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M74 best_beam={} speedup={:.6f}x".format(
        best["beam"], best["dual_candidate_speedup"]), flush=True)


if __name__ == "__main__":
    main()
