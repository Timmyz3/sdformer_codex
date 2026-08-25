#!/usr/bin/env python3
"""Narrow independent review of the M81 r2 13-bit descriptor fix.

The producer modules are not imported.  PWP widths and both variant start-word
streams are rebuilt directly from M41 INT8 weights and M72 centers.  All
16-bit descriptors are encoded/decoded locally and r1/r2 capacity invariance
is checked independently.
"""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SCRIPT = HW / "system_simulator/scripts/analyze_m81_interleaved_word_pwp_buffer.py"
R2 = HW / (
    "results/m81_interleaved_word_pwp_buffer_valid825_internal_dev_r2_20260823/"
    "m81_interleaved_word_pwp_buffer.json")
R1 = HW / (
    "results/m81_interleaved_word_pwp_buffer_valid825_internal_dev_r1_20260823/"
    "m81_interleaved_word_pwp_buffer.json")
M72 = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_ROOT = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41 = M41_ROOT / "m41_h67_ep35_bottleneck_int8_bridge.json"
R1_RECONSTRUCTION = HW / (
    "reviews/m81_interleaved_word_pwp_buffer_independent_hammer_r1_20260823/"
    "m81_independent_reconstruction.json")
R1_REVIEW = HW / (
    "reviews/m81_interleaved_word_pwp_buffer_independent_hammer_r1_20260823/"
    "m81_interleaved_word_pwp_buffer_independent_hammer_review.json")
RECONSTRUCTION = HERE / "m81_r2_independent_reconstruction.json"
REVIEW = HERE / "m81_interleaved_word_pwp_buffer_independent_hammer_r2_review.json"
RECEIPT = HERE / "m81_interleaved_word_pwp_buffer_independent_hammer_r2_validation_receipt.json"

EXPECTED_SHA = {
    "script": "4aa7b2608d307cecc913c5edabaa4c7237b5b6cc39a818d3e90e8baa665e0aba",
    "r2": "515e023421a2650077b61fb620b06428786eb180dd3d36d05eccec1c8d2fabad",
    "r1": "ab0ea4e87a58ee3945ff7ab36dc0507573fd8bf1ddba1958bbdd63a415a6b381",
    "m72": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "r1_reconstruction": "6c853a5bc08df6cf79ec30333bd1cfcbdc6549406b932b9584afe5c1fc62ea75",
    "r1_review": "842f50ed1527d32f94d79d677f50493a3aee2cdc994377df1a334b0c38eb1b3d",
}
EXPECTED_WEIGHT_SHA = (
    "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
)

FEATURES = 6912
PARTITION_BITS = 16
PARTITIONS = 432
PATTERNS = 16
BLOCKS = 8
LANES = 96
CHANNELS = BLOCKS * LANES
PHASES = 4 * PARTITIONS
ENTRIES = PATTERNS * BLOCKS
CAP = 11
HEADER_BYTES = 48
OFFSET_TABLE_BYTES = (PHASES + 1) * 4
WORD_BYTES = 4
BANKS = 8
PORT_BYTES = 32
DESCRIPTOR_BYTES_PER_PHASE = ENTRIES * 2
FIXED12_PHASE_BYTES = ENTRIES * 144
WEIGHT_CATALOG_BYTES = PHASES * 12288


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + raw)))


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8")


def align(value, quantum):
    return ((value + quantum - 1) // quantum) * quantum


def signed_width(minimum, maximum):
    for width in range(1, 33):
        if minimum >= -(1 << (width - 1)) and maximum <= (1 << (width - 1)) - 1:
            return max(8, width)
    raise ValueError("signed width exceeds 32 bits")


def compare(left, right, label="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), label + " key drift")
        for key in left:
            compare(left[key], right[key], label + "." + str(key))
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), label + " length drift")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, label + "[{}]".format(index))
    elif isinstance(left, float) or isinstance(right, float):
        require(abs(float(left) - float(right)) <=
                1e-12 * max(1.0, abs(float(right))), label + " float drift")
    else:
        require(left == right, label + " drift: {} != {}".format(left, right))


def encode_descriptor(escape, width_class, start_word):
    require(escape in (0, 1), "escape range")
    require(0 <= width_class < 4, "class range")
    require(0 <= start_word < (1 << 13), "start range")
    encoded = (escape << 15) | (width_class << 13) | start_word
    require(0 <= encoded < (1 << 16), "descriptor width overflow")
    decoded = {
        "escape": (encoded >> 15) & 1,
        "width_class": (encoded >> 13) & 3,
        "start_word": encoded & 0x1fff,
    }
    require(decoded == {
        "escape": escape, "width_class": width_class,
        "start_word": start_word}, "descriptor round-trip drift")
    return encoded


def build_widths(m72, m41):
    widths = np.zeros((4, PARTITIONS, PATTERNS, BLOCKS), dtype=np.uint8)
    weight_shas = []
    digest = hashlib.sha256()
    for op, operator in enumerate(m72["operators"]):
        layer = next(row for row in m41["layers"]
                     if row["operator"] == operator["operator"])
        info = next(row for row in layer["payloads"] if row["role"] == "weight")
        path = M41_ROOT / info["file"]
        observed = sha256_path(path)
        require(observed == EXPECTED_WEIGHT_SHA[op] == info["sha256"],
                "weight identity drift")
        weight_shas.append(observed)
        weight = np.fromfile(str(path), dtype=np.int8)
        require(weight.size == FEATURES * CHANNELS, "weight extent drift")
        weight = weight.reshape(FEATURES, CHANNELS).astype(np.int32)
        for partition, row in enumerate(operator["partitions"]):
            require(row["partition"] == partition, "partition order drift")
            source = weight[
                partition * PARTITION_BITS:(partition + 1) * PARTITION_BITS]
            for pattern, raw_center in enumerate(row["centers_hex"]):
                center = int(raw_center, 16)
                indices = [bit for bit in range(PARTITION_BITS)
                           if center & (1 << bit)]
                pwp = source[indices].sum(axis=0, dtype=np.int32)
                for block in range(BLOCKS):
                    values = pwp[block * LANES:(block + 1) * LANES]
                    width = signed_width(int(values.min()), int(values.max()))
                    widths[op, partition, pattern, block] = width
                    digest.update(canonical_bytes({
                        "op": op, "partition": partition, "pattern": pattern,
                        "block": block, "width": width,
                    }))
        print("[M81 R2 WIDTH] operator={}/4".format(op + 1), flush=True)
    return widths, weight_shas, digest.hexdigest()


def variant_capacity(payloads, catalog_payload, name):
    maximum = max(payloads)
    maximum_words = align(maximum, WORD_BYTES) // WORD_BYTES
    exact_rows = int(math.ceil(maximum_words / float(BANKS)))
    envelopes = []
    for quantum in (1, 16, 64):
        rows = align(exact_rows, quantum)
        one_buffer = rows * BANKS * WORD_BYTES
        double_total = 2 * one_buffer + 2 * DESCRIPTOR_BYTES_PER_PHASE
        envelopes.append({
            "bank_depth_row_quantum": quantum,
            "rows_per_bank": rows,
            "one_payload_buffer_bytes": one_buffer,
            "two_payload_buffers_plus_two_descriptor_tables_bytes": double_total,
            "reduction_vs_fixed12_double_payload": (
                1.0 - double_total / float(2 * FIXED12_PHASE_BYTES)),
        })
    records = [align(HEADER_BYTES + payload, PORT_BYTES) for payload in payloads]
    addressable = sum(records) + OFFSET_TABLE_BYTES
    five_sample = 5 * (WEIGHT_CATALOG_BYTES + sum(records)) + 5 * OFFSET_TABLE_BYTES
    fixed_five = 5 * (WEIGHT_CATALOG_BYTES + PHASES * FIXED12_PHASE_BYTES)
    return {
        "name": name,
        "catalog_pwp_payload_bytes": catalog_payload,
        "phase_payload_bytes_minimum": min(payloads),
        "phase_payload_bytes_maximum": maximum,
        "phase_record_alignment_padding_bytes": (
            sum(records) - PHASES * HEADER_BYTES - sum(payloads)),
        "addressable_catalog_bytes_including_headers_offsets_alignment": addressable,
        "addressable_catalog_reduction_vs_fixed12": (
            1.0 - addressable / float(PHASES * FIXED12_PHASE_BYTES)),
        "five_sample_weight_plus_addressable_pwp_bytes": five_sample,
        "five_sample_prefetch_reduction_vs_fixed12": (
            1.0 - five_sample / float(fixed_five)),
        "maximum_payload_words": maximum_words,
        "exact_rows_per_32bit_bank": exact_rows,
        "capacity_envelopes": envelopes,
    }


def reconstruct():
    paths = {
        "script": SCRIPT, "r2": R2, "r1": R1, "m72": M72, "m41": M41,
        "r1_reconstruction": R1_RECONSTRUCTION, "r1_review": R1_REVIEW,
    }
    for name, path in paths.items():
        require(sha256_path(path) == EXPECTED_SHA[name], name + " SHA drift")
    r2 = strict_json(R2)
    r1 = strict_json(R1)
    m72 = strict_json(M72)
    m41 = strict_json(M41)
    r1_reconstruction = strict_json(R1_RECONSTRUCTION)
    r1_review = strict_json(R1_REVIEW)
    require(r1_review["findings"]["p0"][0]["id"] ==
            "M81-P0-01-ALIGNED-DESCRIPTOR-OVERFLOW",
            "r1 P0 identity drift")
    widths, weight_shas, width_digest = build_widths(m72, m41)

    packed_payloads = []
    aligned_payloads = []
    packed_catalog = aligned_catalog = 0
    packed_maximum = aligned_maximum = 0
    descriptor_checks = Counter()
    previous_overflow_now_pass = []
    phase_digest = hashlib.sha256()
    for op in range(4):
        for partition in range(PARTITIONS):
            packed_bytes = aligned_bytes = 0
            entries = []
            for pattern in range(PATTERNS):
                for block in range(BLOCKS):
                    width = int(widths[op, partition, pattern, block])
                    if width > CAP:
                        encode_descriptor(1, 0, 0)
                        descriptor_checks["escape_roundtrips"] += 2
                        entries.append({
                            "pattern": pattern, "block": block,
                            "width": width, "escape": True,
                        })
                        continue
                    width_class = width - 8
                    packed_start = packed_bytes // WORD_BYTES
                    aligned_start = aligned_bytes // WORD_BYTES
                    packed_maximum = max(packed_maximum, packed_start)
                    aligned_maximum = max(aligned_maximum, aligned_start)
                    encode_descriptor(0, width_class, packed_start)
                    encode_descriptor(0, width_class, aligned_start)
                    descriptor_checks["eligible_roundtrips"] += 2
                    if aligned_start >= 4096:
                        previous_overflow_now_pass.append({
                            "operator_index": op,
                            "partition": partition,
                            "pattern": pattern,
                            "output_block": block,
                            "width": width,
                            "start_word": aligned_start,
                        })
                    entry_bytes = width * LANES // 8
                    entries.append({
                        "pattern": pattern, "block": block,
                        "width": width, "escape": False,
                        "packed_start_word": packed_start,
                        "aligned_start_word": aligned_start,
                    })
                    packed_bytes += entry_bytes
                    aligned_bytes += align(entry_bytes, PORT_BYTES)
            packed_payloads.append(packed_bytes)
            aligned_payloads.append(aligned_bytes)
            packed_catalog += packed_bytes
            aligned_catalog += aligned_bytes
            phase_digest.update(canonical_bytes({
                "op": op, "partition": partition,
                "packed_payload_bytes": packed_bytes,
                "aligned_payload_bytes": aligned_bytes,
                "entries": entries,
            }))
        print("[M81 R2 PHASE] operator={}/4".format(op + 1), flush=True)
    require(packed_maximum == 3648 and aligned_maximum == 4152,
            "variant maximum start drift")
    require(packed_maximum < 8192 and aligned_maximum < 8192,
            "13-bit descriptor capacity failure")
    require(len(previous_overflow_now_pass) == 8,
            "r1 overflow regression population drift")

    variants = [
        variant_capacity(packed_payloads, packed_catalog,
                         "WORD_PACKED_8X32_INTERLEAVED"),
        variant_capacity(aligned_payloads, aligned_catalog,
                         "ENTRY_ALIGNED_32B"),
    ]
    compare(variants, r2["variants"], "r2_variants")
    compare(r1["variants"], r2["variants"], "r1_r2_capacity_invariance")
    compare(r1["start_word_mod8_histogram"], r2["start_word_mod8_histogram"],
            "r1_r2_start_histogram")
    require(r2["architecture"]["local_descriptor"] == {
        "entries_per_phase": 128,
        "bits_per_entry": 16,
        "fields": "escape[15], width_class[14:13], start_word[12:0]",
        "bytes_per_phase": 256,
        "maximum_observed_word_packed_start_word": 3648,
        "maximum_observed_entry_aligned_start_word": 4152,
    }, "r2 descriptor contract drift")
    require(r1_reconstruction["bank_mapping"]["bank_conflicts"] == 0 and
            r1_reconstruction["bank_mapping"]["maximum_reads_of_one_bank_in_one_cycle"] == 1,
            "r1 independently sealed bank mapping drift")
    require(all(not r2["admission"][key] for key in (
        "multi_pwp_concurrency", "zero_bubble_or_finite_queue_rtl",
        "dc_sta_formality_saif_ptpx", "accuracy",
        "full_network_or_system_speedup", "date_headline")),
        "r2 claim boundary widened")

    return {
        "schema": "m81_r2_descriptor_fix_independent_reconstruction_v1",
        "status": "PASS_M81_R2_13BIT_DESCRIPTOR_P0_CLOSED",
        "identity_sha256": dict(EXPECTED_SHA, **{
            "weight_payloads": weight_shas,
            "independent_width_digest": width_digest,
            "independent_phase_descriptor_digest": phase_digest.hexdigest(),
        }),
        "independence": {
            "production_modules_imported": False,
            "direct_inputs": ["M41 INT8 payloads", "M72 center JSON"],
            "r1_sealed_bank_mapping_used_only_for_out_of_scope_regression": True,
        },
        "descriptor_fix": {
            "bits_per_entry": 16,
            "fields": "escape[15], width_class[14:13], start_word[12:0]",
            "field_bits_sum": 16,
            "maximum_encodable_start_word": 8191,
            "word_packed_maximum_start_word": packed_maximum,
            "word_packed_capacity_margin_codes": 8191 - packed_maximum,
            "entry_aligned_maximum_start_word": aligned_maximum,
            "entry_aligned_capacity_margin_codes": 8191 - aligned_maximum,
            "eligible_descriptor_roundtrips": descriptor_checks["eligible_roundtrips"],
            "escape_descriptor_roundtrips": descriptor_checks["escape_roundtrips"],
            "descriptor_roundtrip_mismatches": 0,
            "r1_overflow_entries_now_representable": previous_overflow_now_pass,
            "r1_p0_closed": True,
        },
        "capacity_invariance": {
            "descriptor_bytes_per_phase_r1": 256,
            "descriptor_bytes_per_phase_r2": 256,
            "descriptor_byte_delta": 0,
            "word_packed_catalog_payload_bytes": packed_catalog,
            "entry_aligned_catalog_payload_bytes": aligned_catalog,
            "r1_r2_variant_field_mismatches": 0,
            "word_packed_exact_double_buffer_bytes": 29952,
            "word_packed_exact_reduction_vs_fixed12": 0.1875,
            "word_packed_464_double_buffer_bytes": 30208,
            "word_packed_464_reduction_vs_fixed12": 0.18055555555555558,
            "word_packed_512_double_buffer_bytes": 33280,
            "word_packed_512_reduction_vs_fixed12": 0.09722222222222221,
        },
        "variants": variants,
        "p0_reassessment": {
            "r1_p0_count": 1,
            "r2_p0_count": 0,
            "new_p0_found": False,
            "remaining_evidence_gaps_are_p1": True,
        },
        "claim_boundary_regression": {
            "bank_mapping_identity_from_r1_reconstruction_intact": True,
            "capacity_and_traffic_intact": True,
            "zero_bubble_rtl": False,
            "synopsys_ppa": False,
            "system_or_accuracy": False,
            "date_headline": False,
        },
    }


def validate_review(payload):
    if RECONSTRUCTION.exists():
        compare(strict_json(RECONSTRUCTION), payload, "stored_reconstruction")
    if REVIEW.exists():
        review = strict_json(REVIEW)
        require(review["status"] ==
                "M81_R2_DESCRIPTOR_P0_CLOSED_GO_BOTH_LAYOUT_DSE_RTL_MACRO_UNADMITTED",
                "review status drift")
        require(len(review["findings"]["p0"]) == 0 and
                len(review["findings"]["p1"]) == 6,
                "review finding count drift")
        require(review["scores"] == {
            "hardware_innovation": 61,
            "capacity_advantage": 78,
            "performance_advantage": 54,
            "evidence_quality": 83,
            "m81_scoped_milestone_quality": 88,
            "date_paper_completeness": 48,
        }, "review scores drift")
    if RECEIPT.exists():
        receipt = strict_json(RECEIPT)
        require(receipt["status"] ==
                "PASS_M81_R2_DESCRIPTOR_FIX_INDEPENDENT_HAMMER" and
                receipt["identity"]["validator_sha256"] == sha256_path(Path(__file__)) and
                receipt["identity"]["reconstruction_sha256"] ==
                sha256_path(RECONSTRUCTION) and
                receipt["identity"]["review_sha256"] == sha256_path(REVIEW),
                "receipt identity drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = reconstruct()
    validate_review(payload)
    if args.output is not None:
        require(not args.output.exists(), "refusing output overwrite")
        require(args.output.resolve().parent == HERE.resolve(),
                "output must stay in r2 review directory")
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fix = payload["descriptor_fix"]
    print("PASS M81 r2 independent: packed_max={} aligned_max={} "
          "roundtrip_mismatch=0 byte_delta=0 P0=0 P1=6".format(
              fix["word_packed_maximum_start_word"],
              fix["entry_aligned_maximum_start_word"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M81 r2 independent: {}".format(error), flush=True)
        raise SystemExit(1)
