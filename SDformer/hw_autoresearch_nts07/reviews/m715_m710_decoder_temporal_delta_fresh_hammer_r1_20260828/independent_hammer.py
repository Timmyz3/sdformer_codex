#!/usr/bin/env python3
"""Receipt-blind M710 recompute from canonical M699 packed payloads.

This file intentionally does not import or execute the M710 author analyzer.
It reconstructs all source counts and K3/S2/P1/OP1 legal-tap work directly
from the 120 little-bit-first C-order payload records.  Author CSV/summary is
read only after the independent population has been computed, for comparison.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from collections import defaultdict

import numpy as np


MODULES = {
    0: ("D0", 1536, 384, 15, 20),
    1: ("D1", 770, 192, 30, 40),
    2: ("D2", 386, 96, 60, 80),
    3: ("D3", 194, 96, 120, 160),
}
SEQUENCES = ("interlaken_01_a", "thun_01_b", "zurich_city_12_a")
THETA = 0.9999954104423523
THETA_U32 = 1065353139
THETA_HEX = "b3ff7f3f"
THETA_SHA256 = "5df16d346190fdd928ee71a5c3e1dbeaf4d9b71985167bd7eccbdf1d87cc3721"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_manifest(root, manifest_name="SHA256SUMS", seal_name="SHA256SUMS.seal.sha256"):
    manifest_path = os.path.join(root, manifest_name)
    seal_path = os.path.join(root, seal_name)
    members = []
    mismatches = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            expected, rel = line.split("  ", 1)
            actual = sha256_file(os.path.join(root, rel))
            members.append(rel)
            if actual != expected:
                mismatches.append({"path": rel, "expected": expected, "actual": actual})
    seal_lines = [x.rstrip("\n") for x in open(seal_path, "r") if x.strip()]
    seal_ok = False
    seal_expected = None
    if len(seal_lines) == 1:
        seal_expected, seal_rel = seal_lines[0].split("  ", 1)
        seal_ok = seal_rel == manifest_name and seal_expected == sha256_file(manifest_path)
    return {
        "members": len(members),
        "member_mismatches": mismatches,
        "members_ok": not mismatches,
        "manifest_sha256": sha256_file(manifest_path),
        "seal_file_sha256": sha256_file(seal_path),
        "seal_expected_manifest_sha256": seal_expected,
        "seal_ok": seal_ok,
    }


def tap_partition(spatial_counts):
    """Return source counts in 4-, 6-, and 9-legal-tap spatial classes."""
    n4 = int(spatial_counts[0, 0])
    n6 = int(spatial_counts[0, 1:].sum(dtype=np.int64))
    n6 += int(spatial_counts[1:, 0].sum(dtype=np.int64))
    n9 = int(spatial_counts[1:, 1:].sum(dtype=np.int64))
    return n4, n6, n9


def legal_events(n4, n6, n9):
    return 4 * n4 + 6 * n6 + 9 * n9


def exhaustive_geometry_checks():
    checks = []
    for module_index in sorted(MODULES):
        module, cin, cout, height, width = MODULES[module_index]
        out_h = (height - 1) * 2 - 2 + (3 - 1) + 1 + 1
        out_w = (width - 1) * 2 - 2 + (3 - 1) + 1 + 1
        histogram = defaultdict(int)
        category_match = True
        for y in range(height):
            for x in range(width):
                taps = 0
                for ky in range(3):
                    oy = 2 * y - 1 + ky
                    for kx in range(3):
                        ox = 2 * x - 1 + kx
                        if 0 <= oy < out_h and 0 <= ox < out_w:
                            taps += 1
                histogram[taps] += 1
                expected = 4 if (y == 0 and x == 0) else (6 if (y == 0 or x == 0) else 9)
                category_match = category_match and taps == expected
        expected_histogram = {
            4: 1,
            6: height + width - 2,
            9: (height - 1) * (width - 1),
        }
        checks.append({
            "module": module,
            "input_hw": [height, width],
            "derived_output_hw": [out_h, out_w],
            "histogram": {str(k): int(histogram[k]) for k in sorted(histogram)},
            "expected_histogram": {str(k): int(expected_histogram[k]) for k in sorted(expected_histogram)},
            "only_4_6_9": set(histogram) == {4, 6, 9},
            "coordinate_category_match": category_match,
            "histogram_match": dict(histogram) == expected_histogram,
        })
    return checks


def ratio(delta, full):
    return float(delta) / float(full) if full else None


def aggregate(rows, key):
    buckets = defaultdict(lambda: {
        "records": 0,
        "full_active_sources": 0,
        "delta_initial_active_sources": 0,
        "delta_transition_sources": 0,
        "delta_sources": 0,
        "full_active_legal_tap_events": 0,
        "delta_initial_plus_xor_legal_tap_events": 0,
        "full_product_work": 0,
        "delta_product_work": 0,
    })
    for row in rows:
        k = row[key] if isinstance(key, str) else tuple(row[x] for x in key)
        b = buckets[k]
        b["records"] += 1
        for name in list(b):
            if name != "records":
                b[name] += int(row[name])
    output = []
    for k in sorted(buckets, key=lambda x: str(x)):
        b = dict(buckets[k])
        if isinstance(key, str):
            b[key] = k
        else:
            for name, value in zip(key, k):
                b[name] = value
        b["delta_over_full_product_work"] = ratio(b["delta_product_work"], b["full_product_work"])
        output.append(b)
    return output


def compare_author_rows(author_csv, rows):
    independent = {(int(x["global_sample_id"]), x["module"]): x for x in rows}
    mismatches = []
    checked = 0
    integer_fields = (
        "full_active_sources", "delta_initial_active_sources", "delta_transition_sources",
        "delta_sources", "full_active_legal_tap_events",
        "delta_initial_plus_xor_legal_tap_events", "full_product_work", "delta_product_work",
    )
    with open(author_csv, "r") as f:
        for author in csv.DictReader(f):
            key = (int(author["global_sample_id"]), author["module"])
            mine = independent.get(key)
            if mine is None:
                mismatches.append({"key": key, "problem": "missing independent row"})
                continue
            checked += 1
            for field in integer_fields:
                if int(author[field]) != int(mine[field]):
                    mismatches.append({
                        "key": key,
                        "field": field,
                        "author": int(author[field]),
                        "independent": int(mine[field]),
                    })
            author_ratio = float(author["delta_over_full_product_work"])
            if not math.isclose(author_ratio, mine["delta_over_full_product_work"], rel_tol=0.0, abs_tol=1e-15):
                mismatches.append({
                    "key": key,
                    "field": "delta_over_full_product_work",
                    "author": author_ratio,
                    "independent": mine["delta_over_full_product_work"],
                })
    return {"records_checked": checked, "mismatches": mismatches, "all_match": checked == 120 and not mismatches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo_root)
    hroot = os.path.join(repo, "hw_autoresearch_nts07")
    payload_root = os.path.join(hroot, "system_handoff", "outgoing", "m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828")
    m705_root = os.path.join(hroot, "reviews", "m705_m699_multisequence_decoder_payload_fresh_result_hammer_r1_20260828")
    author_result = os.path.join(hroot, "results", "m710_h67_decoder_temporal_delta_legal_tap_product_work_r1_20260828")
    author_handoff = os.path.join(hroot, "reviews", "m710_decoder_temporal_delta_legal_tap_product_work_author_handoff_r1_20260828")
    m699_contract = os.path.join(hroot, "contracts", "m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json")
    m705_review = os.path.join(m705_root, "review.json")
    m710_contract = os.path.join(hroot, "contracts", "m710_decoder_temporal_delta_legal_tap_product_work_contract_r1_20260828.json")
    docs359 = os.path.join(hroot, "docs", "359_DATE终局冻结_20260813.md")

    seals = {
        "m699": verify_manifest(payload_root),
        "m705": verify_manifest(m705_root),
        "m710_result": verify_manifest(author_result),
        "m710_author_handoff": verify_manifest(author_handoff),
    }
    geometry_checks = exhaustive_geometry_checks()

    manifest_path = os.path.join(payload_root, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    canonical_binding = {
        "m699_contract_sha256": sha256_file(m699_contract),
        "m699_contract_matches_manifest": sha256_file(m699_contract) == manifest["identity"]["contract"]["sha256"],
        "m699_manifest_sha256": sha256_file(manifest_path),
        "m699_manifest_matches_member_manifest": sha256_file(manifest_path) == "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0",
        "m705_review_sha256": sha256_file(m705_review),
        "m705_review_matches_m710_contract": sha256_file(m705_review) == "6af48fb271254ef20f6baa1e435acfe51fdf38b457fe9782d6cac0b0e2883bd3",
        "m710_contract_sha256": sha256_file(m710_contract),
        "m710_contract_matches_author_identity": sha256_file(m710_contract) == "9234a517c4fab185a4ae2d0a2b5bc76f41181125510ca35da03fbe0dda4e5132",
    }

    population_errors = []
    if len(manifest.get("records", [])) != 120:
        population_errors.append("record count is not 120")
    lattice = set()
    rows = []
    d1_checks = []

    for rec in manifest["records"]:
        sample = int(rec["global_sample_id"])
        module_index = int(rec["module_index"])
        if module_index not in MODULES:
            population_errors.append("unknown module index {}".format(module_index))
            continue
        module, cin, cout, height, width = MODULES[module_index]
        key = (rec["sequence"], int(rec["sequence_sample_id"]), module_index)
        if key in lattice:
            population_errors.append("duplicate lattice key {}".format(key))
        lattice.add(key)
        expected_shape = [10, 1, cin, height, width]
        if rec["input_shape"] != expected_shape:
            population_errors.append("shape mismatch {} {}".format(key, rec["input_shape"]))
        expected_route = "EXACT_SCALED_BINARY_BITPACK" if module == "D1" else "EXACT_BINARY_BITPACK"
        if rec["route"] != expected_route:
            population_errors.append("route mismatch {} {}".format(key, rec["route"]))

        packed_path = os.path.join(payload_root, rec["relative_path"])
        packed_sha = sha256_file(packed_path)
        stats = rec["statistics"]["scaled_binary_audit"] if module == "D1" else rec["statistics"]
        if packed_sha != stats["packed_sha256"]:
            population_errors.append("payload sha mismatch {}".format(key))
        element_count = int(np.prod(np.asarray(expected_shape, dtype=np.int64)))
        packed = np.fromfile(packed_path, dtype=np.uint8)
        if packed.size != (element_count + 7) // 8:
            population_errors.append("payload size mismatch {}".format(key))
        bits = np.unpackbits(packed, bitorder="little")[:element_count].reshape(expected_shape)

        full_spatial = bits.sum(axis=(0, 1, 2), dtype=np.int64)
        initial_spatial = bits[0].sum(axis=(0, 1), dtype=np.int64)
        transition_spatial = np.logical_xor(bits[1:], bits[:-1]).sum(axis=(0, 1, 2), dtype=np.int64)
        delta_spatial = initial_spatial + transition_spatial
        full4, full6, full9 = tap_partition(full_spatial)
        init4, init6, init9 = tap_partition(initial_spatial)
        trans4, trans6, trans9 = tap_partition(transition_spatial)
        delta4, delta6, delta9 = tap_partition(delta_spatial)

        row = {
            "global_sample_id": sample,
            "sequence": rec["sequence"],
            "sequence_sample_id": int(rec["sequence_sample_id"]),
            "module": module,
            "module_index": module_index,
            "route": rec["route"],
            "cin": cin,
            "cout": cout,
            "height": height,
            "width": width,
            "packed_sha256": packed_sha,
            "full_n4": full4,
            "full_n6": full6,
            "full_n9": full9,
            "initial_n4": init4,
            "initial_n6": init6,
            "initial_n9": init9,
            "transition_n4": trans4,
            "transition_n6": trans6,
            "transition_n9": trans9,
            "delta_n4": delta4,
            "delta_n6": delta6,
            "delta_n9": delta9,
            "full_active_sources": int(full_spatial.sum(dtype=np.int64)),
            "delta_initial_active_sources": int(initial_spatial.sum(dtype=np.int64)),
            "delta_transition_sources": int(transition_spatial.sum(dtype=np.int64)),
            "delta_sources": int(delta_spatial.sum(dtype=np.int64)),
            "full_active_legal_tap_events": legal_events(full4, full6, full9),
            "delta_initial_plus_xor_legal_tap_events": legal_events(delta4, delta6, delta9),
        }
        row["full_product_work"] = row["full_active_legal_tap_events"] * cout
        row["delta_product_work"] = row["delta_initial_plus_xor_legal_tap_events"] * cout
        row["delta_over_full_product_work"] = ratio(row["delta_product_work"], row["full_product_work"])
        row["conservation"] = {
            "delta_equals_initial_plus_transition": row["delta_sources"] == row["delta_initial_active_sources"] + row["delta_transition_sources"],
            "delta_partition_equals_initial_plus_transition": (delta4, delta6, delta9) == (init4 + trans4, init6 + trans6, init9 + trans9),
            "full_source_partition": row["full_active_sources"] == full4 + full6 + full9,
            "delta_source_partition": row["delta_sources"] == delta4 + delta6 + delta9,
            "full_legal_formula": row["full_active_legal_tap_events"] == 4 * full4 + 6 * full6 + 9 * full9,
            "delta_legal_formula": row["delta_initial_plus_xor_legal_tap_events"] == 4 * delta4 + 6 * delta6 + 9 * delta9,
            "full_product_formula": row["full_product_work"] == row["full_active_legal_tap_events"] * cout,
            "delta_product_formula": row["delta_product_work"] == row["delta_initial_plus_xor_legal_tap_events"] * cout,
        }
        rows.append(row)

        if module == "D1":
            raw = rec["statistics"]["raw"]
            d1_checks.append({
                "global_sample_id": sample,
                "route_exact_scaled": rec["route"] == "EXACT_SCALED_BINARY_BITPACK",
                "not_coerced": rec["coerced"] is False,
                "not_rounded": rec["rounded"] is False,
                "not_thresholded": rec["thresholded"] is False,
                "theta_gate_pass": stats["theta_gate_pass"] is True,
                "other_finite_zero": int(stats["other_finite_count"]) == 0,
                "nonfinite_zero": int(stats["nonfinite_count"]) == 0,
                "theta_count_matches_bits": int(stats["theta_count"]) == row["full_active_sources"],
                "raw_nonbinary_matches_theta": int(raw["nonbinary_finite_count"]) == int(stats["theta_count"]),
                "raw_sha_bound": raw["content_sha256"] == rec["raw_fp32_content_sha256"] == stats["raw_content_sha256"],
            })

    expected_lattice = set((seq, sid, midx) for seq in SEQUENCES for sid in range(10) for midx in range(4))
    if lattice != expected_lattice:
        population_errors.append("record lattice differs; missing={} extra={}".format(sorted(expected_lattice - lattice), sorted(lattice - expected_lattice)))

    rows.sort(key=lambda x: (x["global_sample_id"], x["module_index"]))
    per_module = aggregate(rows, "module")
    per_sequence = aggregate(rows, "sequence")
    per_sample = aggregate(rows, "global_sample_id")
    overall = aggregate(rows, ("route",))  # ignored except to reuse exact sum logic
    totals = {
        "records": len(rows),
        "full_active_sources": sum(x["full_active_sources"] for x in rows),
        "delta_initial_active_sources": sum(x["delta_initial_active_sources"] for x in rows),
        "delta_transition_sources": sum(x["delta_transition_sources"] for x in rows),
        "delta_sources": sum(x["delta_sources"] for x in rows),
        "full_active_legal_tap_events": sum(x["full_active_legal_tap_events"] for x in rows),
        "delta_initial_plus_xor_legal_tap_events": sum(x["delta_initial_plus_xor_legal_tap_events"] for x in rows),
        "full_product_work": sum(x["full_product_work"] for x in rows),
        "delta_product_work": sum(x["delta_product_work"] for x in rows),
    }
    totals["delta_over_full_product_work"] = ratio(totals["delta_product_work"], totals["full_product_work"])
    totals["delta_product_work_change_fraction"] = totals["delta_over_full_product_work"] - 1.0

    theta_bytes = struct.pack("<f", THETA)
    theta_identity = {
        "manifest_value": manifest["d1_runtime_threshold_identity"]["value"],
        "independent_float32_uint32": struct.unpack("<I", theta_bytes)[0],
        "independent_float32_le_hex": theta_bytes.hex(),
        "independent_float32_sha256": hashlib.sha256(theta_bytes).hexdigest(),
        "expected_uint32": THETA_U32,
        "expected_hex": THETA_HEX,
        "expected_sha256": THETA_SHA256,
        "records": len(d1_checks),
        "all_record_checks_pass": all(all(v for k, v in x.items() if k != "global_sample_id") for x in d1_checks),
        "folded_weight_deployment_admitted": False,
        "decoder_numeric_equivalence_admitted": False,
    }
    theta_identity["scalar_identity_pass"] = (
        theta_identity["manifest_value"] == THETA
        and theta_identity["independent_float32_uint32"] == THETA_U32
        and theta_identity["independent_float32_le_hex"] == THETA_HEX
        and theta_identity["independent_float32_sha256"] == THETA_SHA256
    )

    author_compare = compare_author_rows(os.path.join(author_result, "per_record.csv"), rows)
    with open(os.path.join(author_result, "summary.json"), "r") as f:
        author_summary = json.load(f)
    author_total = author_summary["overall_ratio_of_sums"]
    author_summary_compare = {
        "full_product_work_match": int(author_total["full_product_work"]) == totals["full_product_work"],
        "delta_product_work_match": int(author_total["delta_product_work"]) == totals["delta_product_work"],
        "full_legal_tap_events_match": int(author_total["full_active_legal_tap_events"]) == totals["full_active_legal_tap_events"],
        "delta_legal_tap_events_match": int(author_total["delta_initial_plus_xor_legal_tap_events"]) == totals["delta_initial_plus_xor_legal_tap_events"],
        "ratio_match": math.isclose(float(author_total["delta_over_full_product_work"]), totals["delta_over_full_product_work"], rel_tol=0.0, abs_tol=1e-15),
    }

    all_conservation = all(all(x["conservation"].values()) for x in rows)
    all_seals = all(x["members_ok"] and x["seal_ok"] for x in seals.values())
    gate_pass = totals["delta_over_full_product_work"] < 0.70
    all_checks = (
        not population_errors
        and all(v for k, v in canonical_binding.items() if k.endswith("matches_manifest") or k.endswith("matches_member_manifest") or k.endswith("matches_m710_contract") or k.endswith("matches_author_identity"))
        and all(x["only_4_6_9"] and x["coordinate_category_match"] and x["histogram_match"] for x in geometry_checks)
        and all_conservation
        and all_seals
        and theta_identity["scalar_identity_pass"]
        and theta_identity["all_record_checks_pass"]
        and author_compare["all_match"]
        and all(author_summary_compare.values())
        and sha256_file(docs359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    )

    output = {
        "schema": "m715_m710_decoder_temporal_delta_receipt_blind_recompute_v1",
        "method": {
            "author_analyzer_imported_or_executed": False,
            "core_input": "M699 120 packed payload records",
            "unpack": "numpy.unpackbits little bit order then [T=10,B=1,C,H,W] C-order reshape",
            "legal_geometry": "K3/S2/P1/OP1: y_out=2*y-1+ky, x_out=2*x-1+kx; legal multiplicity 4 top-left, 6 one top/left boundary, 9 otherwise",
            "full": "sum_t active * legal_taps * Cout",
            "delta": "active_t0 * legal_taps * Cout + sum_t>=1 XOR(active_t,active_t-1) * legal_taps * Cout",
            "aggregate": "ratio of sums",
        },
        "identity": {
            "m699_manifest_sha256": sha256_file(manifest_path),
            "docs359_sha256": sha256_file(docs359),
            "seals": seals,
            "canonical_binding": canonical_binding,
            "population_errors": population_errors,
            "record_lattice_complete": lattice == expected_lattice,
        },
        "geometry_checks": geometry_checks,
        "d1_theta_mask_identity": theta_identity,
        "d1_record_checks": d1_checks,
        "overall": totals,
        "per_module": per_module,
        "per_sequence": per_sequence,
        "per_sample": per_sample,
        "distribution": {
            "per_record_ratio_min": min(x["delta_over_full_product_work"] for x in rows),
            "per_record_ratio_max": max(x["delta_over_full_product_work"] for x in rows),
            "per_sample_ratio_min": min(x["delta_over_full_product_work"] for x in per_sample),
            "per_sample_ratio_max": max(x["delta_over_full_product_work"] for x in per_sample),
            "per_module_ratio_min": min(x["delta_over_full_product_work"] for x in per_module),
            "per_module_ratio_max": max(x["delta_over_full_product_work"] for x in per_module),
        },
        "conservation": {
            "all_120_records_pass": all_conservation,
            "ratio_of_sums_used": True,
        },
        "author_comparison_after_independent_recompute": {
            "per_record": author_compare,
            "summary": author_summary_compare,
        },
        "gate": {
            "strict_maximum_delta_over_full": 0.70,
            "actual": totals["delta_over_full_product_work"],
            "pass": gate_pass,
            "decision": "GO_STATE_IDENTITY_ONLY" if gate_pass else "KILL_N2_NO_RTL",
        },
        "claim_boundary": {
            "product_work_regression": True,
            "cycles": False,
            "speedup": False,
            "system_speedup": False,
            "accuracy": False,
            "numeric_bridge": False,
            "rtl": False,
            "eda": False,
            "energy": False,
            "ppa": False,
            "date_headline": False,
        },
        "all_audit_checks_pass": all_checks,
        "records": rows,
        "unused_debug_route_aggregate": overall,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "all_audit_checks_pass": all_checks,
        "decision": output["gate"]["decision"],
        "records": len(rows),
        "full_product_work": totals["full_product_work"],
        "delta_product_work": totals["delta_product_work"],
        "delta_over_full": totals["delta_over_full_product_work"],
        "per_record_author_match": author_compare["all_match"],
    }, sort_keys=True))
    return 0 if all_checks else 2


if __name__ == "__main__":
    raise SystemExit(main())
