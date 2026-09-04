#!/usr/bin/env python3
"""Independent reconstruction and architecture alternatives for M80 r1."""

import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
REPO = HW.parent
M80_SCRIPT = HW / (
    "system_simulator/scripts/analyze_m80_addressable_phase_record_pwp.py")
M80_RESULT = HW / (
    "results/m80_addressable_phase_record_pwp_valid825_internal_dev_r1_20260823/"
    "m80_addressable_phase_record_pwp.json")
M72_RESULT = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41_RESULT = M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json"

TARGET_M80_SCRIPT_SHA = (
    "d367f9f73e6b12a956c1a1983f3f9710cf5af8a0c8f4f6b7ba039a65da188a12")
TARGET_M80_RESULT_SHA = (
    "dec76e2afa2b91420df514157a8ba9ca0f10ccae03004c84cee2e82b9d72a7da")
TARGET_M72_RESULT_SHA = (
    "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133")
TARGET_M41_RESULT_SHA = (
    "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb")
TARGET_WEIGHT_SHA = (
    "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
)

PHASES = 4 * 432
ENTRIES = 16 * 8
WIDTHS = (8, 9, 10, 11)
DMA_BYTES = 32
HEADER_BYTES = 48
FIXED12_PHASE_BYTES = ENTRIES * 12 * 96 // 8
WEIGHT_PHASE_BYTES = 16 * 8 * 96


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
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def signed_width(minimum, maximum):
    for width in range(1, 33):
        if (minimum >= -(1 << (width - 1)) and
                maximum <= (1 << (width - 1)) - 1):
            return max(8, width)
    raise AssertionError("signed width exceeds 32")


def nearest_rank(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def pack_header_lsb_first(codes):
    bits = []
    for code in codes:
        bits.extend((code >> index) & 1 for index in range(3))
    output = bytearray(HEADER_BYTES)
    for index, bit in enumerate(bits):
        output[index // 8] |= bit << (index % 8)
    return bytes(output)


def pack_header_msb_first(codes):
    bits = []
    for code in codes:
        bits.extend((code >> index) & 1 for index in (2, 1, 0))
    output = bytearray(HEADER_BYTES)
    for index, bit in enumerate(bits):
        output[index // 8] |= bit << (7 - index % 8)
    return bytes(output)


def phase_catalog():
    m72 = strict_json(M72_RESULT)
    m41 = strict_json(M41_RESULT)
    require(len(m72["operators"]) == len(m41["layers"]) == 4,
            "operator extent drift")
    phase_rows = []
    widths_global = dict((width, 0) for width in range(8, 13))
    maxima = dict((width, {"entries": -1}) for width in WIDTHS)
    offsets = [0]
    payload_sizes = []
    record_sizes = []
    aligned_entry_payload_sizes = []
    maximum_word_offset = 0
    ambiguous_header_phases = 0
    weight_shas = []

    for operator_index, operator in enumerate(m72["operators"]):
        layer = next(row for row in m41["layers"]
                     if row["operator"] == operator["operator"])
        weight_info = next(row for row in layer["payloads"]
                           if row["role"] == "weight")
        weight_path = M41_DIR / weight_info["file"]
        observed_sha = sha256(weight_path)
        require(observed_sha == weight_info["sha256"] ==
                TARGET_WEIGHT_SHA[operator_index], "weight identity drift")
        weight_shas.append(observed_sha)
        weights = np.fromfile(str(weight_path), dtype=np.int8)
        require(weights.size == 6912 * 768, "weight extent drift")
        weights = weights.reshape(6912, 768).astype(np.int32)

        for partition, partition_row in enumerate(operator["partitions"]):
            require(partition_row["partition"] == partition,
                    "partition ordering drift")
            source = weights[partition * 16:(partition + 1) * 16]
            counts = dict((width, 0) for width in WIDTHS)
            codes = []
            payload_bytes = 0
            aligned_entry_payload_bytes = 0
            escape_entries = 0
            word_offset = 0
            for center_hex in partition_row["centers_hex"]:
                center = int(center_hex, 16)
                indices = [bit for bit in range(16)
                           if center & (1 << bit)]
                pwp = source[indices].sum(axis=0, dtype=np.int32)
                for block in range(8):
                    values = pwp[block * 96:(block + 1) * 96]
                    width = signed_width(int(values.min()), int(values.max()))
                    widths_global[width] += 1
                    if width <= 11:
                        counts[width] += 1
                        codes.append(width - 8)
                        entry_bytes = width * 96 // 8
                        require(entry_bytes % 4 == 0,
                                "entry not 32-bit-word aligned")
                        maximum_word_offset = max(maximum_word_offset,
                                                  word_offset)
                        word_offset += entry_bytes // 4
                        payload_bytes += entry_bytes
                        aligned_entry_payload_bytes += int(math.ceil(
                            entry_bytes / float(DMA_BYTES))) * DMA_BYTES
                    else:
                        require(width == 12, "unexpected width above cap")
                        codes.append(4)
                        escape_entries += 1
            require(len(codes) == ENTRIES, "phase code extent drift")
            if pack_header_lsb_first(codes) != pack_header_msb_first(codes):
                ambiguous_header_phases += 1
            raw_bytes = HEADER_BYTES + payload_bytes
            aligned_bytes = int(math.ceil(
                raw_bytes / float(DMA_BYTES))) * DMA_BYTES
            padding_bytes = aligned_bytes - raw_bytes
            for width in WIDTHS:
                if counts[width] > maxima[width]["entries"]:
                    maxima[width] = {
                        "entries": counts[width],
                        "operator_index": operator_index,
                        "partition": partition,
                    }
            phase_rows.append({
                "operator_index": operator_index,
                "partition": partition,
                "record_offset_bytes": offsets[-1],
                "width_class_entries": dict(
                    (str(width), counts[width]) for width in WIDTHS),
                "escape_entries": escape_entries,
                "header_bytes": HEADER_BYTES,
                "pwp_payload_bytes": payload_bytes,
                "raw_record_bytes": raw_bytes,
                "dma_aligned_record_bytes": aligned_bytes,
                "dma_padding_bytes": padding_bytes,
                "record_dma_beats": aligned_bytes // DMA_BYTES,
            })
            offsets.append(offsets[-1] + aligned_bytes)
            payload_sizes.append(payload_bytes)
            record_sizes.append(aligned_bytes)
            aligned_entry_payload_sizes.append(aligned_entry_payload_bytes)

    require(len(phase_rows) == PHASES, "phase extent drift")
    rows_serialized = json.dumps(
        phase_rows, sort_keys=True, separators=(",", ":")) + "\n"
    phase_rows_sha = hashlib.sha256(rows_serialized.encode("utf-8")).hexdigest()
    offset_le = b"".join(struct.pack("<I", value) for value in offsets)
    offset_be = b"".join(struct.pack(">I", value) for value in offsets)
    return {
        "phase_rows": phase_rows,
        "phase_rows_sha256": phase_rows_sha,
        "width_histogram": widths_global,
        "maximum_class_entries_with_witness": maxima,
        "offsets": offsets,
        "offset_table_little_endian_sha256": hashlib.sha256(offset_le).hexdigest(),
        "offset_table_big_endian_sha256": hashlib.sha256(offset_be).hexdigest(),
        "offset_endianness_changes_bytes": offset_le != offset_be,
        "payload_sizes": payload_sizes,
        "record_sizes": record_sizes,
        "aligned_entry_payload_sizes": aligned_entry_payload_sizes,
        "maximum_word_offset": maximum_word_offset,
        "ambiguous_header_phases": ambiguous_header_phases,
        "weight_shas": weight_shas,
    }


def verify_m80(catalog):
    result = strict_json(M80_RESULT)
    phase_rows = catalog["phase_rows"]
    payload_total = sum(row["pwp_payload_bytes"] for row in phase_rows)
    header_total = sum(row["header_bytes"] for row in phase_rows)
    padding_total = sum(row["dma_padding_bytes"] for row in phase_rows)
    record_total = sum(row["dma_aligned_record_bytes"] for row in phase_rows)
    escape_total = sum(row["escape_entries"] for row in phase_rows)
    offset_bytes = (PHASES + 1) * 4
    fixed_payload = PHASES * FIXED12_PHASE_BYTES
    addressable = record_total + offset_bytes
    capacity = result["catalog_capacity"]
    expected = {
        "elastic_pwp_payload_bytes": payload_total,
        "width_header_bytes": header_total,
        "phase_alignment_padding_bytes": padding_total,
        "phase_offset_table_bytes": offset_bytes,
        "addressable_elastic_catalog_bytes": addressable,
        "fixed12_reference_pwp_bytes": fixed_payload,
        "unique_escape_entries": escape_total,
    }
    for key, value in expected.items():
        require(capacity[key] == value, "M80 capacity mismatch: " + key)
    require(catalog["phase_rows_sha256"] == result["phase_rows_sha256"],
            "phase-row SHA mismatch")

    stats = result["record_statistics"]
    expected_stats = {
        "payload_bytes_minimum": min(catalog["payload_sizes"]),
        "payload_bytes_p50_nearest_rank": nearest_rank(
            catalog["payload_sizes"], 0.50),
        "payload_bytes_p95_nearest_rank": nearest_rank(
            catalog["payload_sizes"], 0.95),
        "payload_bytes_maximum": max(catalog["payload_sizes"]),
        "aligned_record_bytes_minimum": min(catalog["record_sizes"]),
        "aligned_record_bytes_p50_nearest_rank": nearest_rank(
            catalog["record_sizes"], 0.50),
        "aligned_record_bytes_p95_nearest_rank": nearest_rank(
            catalog["record_sizes"], 0.95),
        "aligned_record_bytes_maximum": max(catalog["record_sizes"]),
    }
    for key, value in expected_stats.items():
        require(stats[key] == value, "M80 record statistic mismatch: " + key)

    maxima = catalog["maximum_class_entries_with_witness"]
    staging = result["staging_capacity"]
    require(staging["maximum_entries_by_width_class"] == dict(
        (str(width), maxima[width]["entries"]) for width in WIDTHS),
        "class maximum mismatch")
    bank_bytes = dict((str(width), maxima[width]["entries"] * width * 12)
                      for width in WIDTHS)
    require(staging["single_buffer_bytes_by_width_class"] == bank_bytes,
            "class bank byte mismatch")
    class_single = sum(bank_bytes.values())
    descriptor_single = int(math.ceil(ENTRIES * 10 / 8.0))
    class_double = 2 * (class_single + descriptor_single)
    fixed_double = 2 * FIXED12_PHASE_BYTES
    require(staging["elastic_class_banks_single_buffer_bytes"] == class_single,
            "class single buffer mismatch")
    require(staging["elastic_class_banks_plus_descriptor_double_buffer_bytes"] ==
            class_double, "class double buffer mismatch")
    require(staging["fixed12_reference_double_buffer_bytes"] == fixed_double,
            "fixed double buffer mismatch")

    weight_catalog = PHASES * WEIGHT_PHASE_BYTES
    fixed_five = 5 * (weight_catalog + fixed_payload)
    elastic_five = 5 * (weight_catalog + addressable)
    traffic = result["traffic"]
    require(traffic["fixed12_weight_plus_pwp_five_sample_bytes"] == fixed_five,
            "fixed traffic mismatch")
    require(traffic["elastic_weight_plus_addressable_pwp_five_sample_bytes"] ==
            elastic_five, "elastic traffic mismatch")
    require(result["parser_throughput"]["minimum_phase_record_dma_cycles"] ==
            min(catalog["record_sizes"]) // DMA_BYTES,
            "minimum DMA cycle mismatch")
    require(ENTRIES <= min(catalog["record_sizes"]) // DMA_BYTES,
            "header parser not hidden by record DMA")
    return {
        "capacity_reconstruction_exact": True,
        "phase_rows_sha256_exact": True,
        "record_statistics_exact": True,
        "traffic_reconstruction_exact": True,
        "parser_envelope_arithmetic_exact": True,
        "payload_bytes": payload_total,
        "header_bytes": header_total,
        "padding_bytes": padding_total,
        "offset_table_bytes": offset_bytes,
        "addressable_catalog_bytes": addressable,
        "storage_reduction_vs_fixed12": 1.0 - addressable / float(fixed_payload),
        "five_sample_traffic_reduction_vs_fixed12": 1.0 - (
            elastic_five / float(fixed_five)),
        "dedicated_class_single_buffer_bytes": class_single,
        "dedicated_class_plus_descriptor_double_buffer_bytes": class_double,
        "dedicated_class_double_buffer_reduction_vs_fixed12": 1.0 - (
            class_double / float(fixed_double)),
        "fixed12_double_buffer_bytes": fixed_double,
    }


def alternatives(catalog):
    fixed_double = 2 * FIXED12_PHASE_BYTES
    maximum_payload = max(catalog["payload_sizes"])
    maximum_payload_phase = catalog["payload_sizes"].index(maximum_payload)
    unified_payload_single = int(math.ceil(
        maximum_payload / float(DMA_BYTES))) * DMA_BYTES
    descriptor_single = ENTRIES * 2
    unified_single = unified_payload_single + descriptor_single
    unified_double = 2 * unified_single
    require(catalog["maximum_word_offset"] < (1 << 12),
            "12-bit word offset insufficient")

    maximum_entry32 = max(catalog["aligned_entry_payload_sizes"])
    maximum_entry32_phase = catalog["aligned_entry_payload_sizes"].index(
        maximum_entry32)
    entry32_single = maximum_entry32 + descriptor_single
    entry32_double = 2 * entry32_single
    global_entry32_payload = sum(catalog["aligned_entry_payload_sizes"])
    # A 48-byte phase header needs 16 bytes of front padding so every following
    # variable-size entry record begins on a 32-byte boundary.
    entry32_catalog = global_entry32_payload + PHASES * 64 + (PHASES + 1) * 4
    fixed_payload = PHASES * FIXED12_PHASE_BYTES
    weight_catalog = PHASES * WEIGHT_PHASE_BYTES
    fixed_five = 5 * (weight_catalog + fixed_payload)
    entry32_five = 5 * (weight_catalog + entry32_catalog)
    return {
        "unified_32bit_word_interleaved": {
            "descriptor_layout": {
                "escape_bits": 1,
                "width_class_bits": 2,
                "reserved_bits": 1,
                "payload_word_offset_bits": 12,
                "offset_unit_bytes": 4,
            },
            "maximum_observed_word_offset": catalog["maximum_word_offset"],
            "maximum_payload_bytes": maximum_payload,
            "maximum_payload_witness": {
                "operator_index": maximum_payload_phase // 432,
                "partition": maximum_payload_phase % 432,
            },
            "payload_buffer_single_bytes_32B_aligned": unified_payload_single,
            "descriptor_single_bytes": descriptor_single,
            "single_buffer_total_bytes": unified_single,
            "double_buffer_total_bytes": unified_double,
            "double_buffer_reduction_vs_fixed12": 1.0 - (
                unified_double / float(fixed_double)),
            "shared32_service_cycles_by_width": dict(
                (str(width), int(math.ceil(width * 12 / 32.0)))
                for width in WIDTHS),
            "catalog_storage_and_prefetch_unchanged_from_m80": True,
            "finite_bank_conflict_and_backpressure_rtl_proven": False,
        },
        "per_entry_32B_aligned_record": {
            "record_bytes_by_width": dict(
                (str(width), int(math.ceil(width * 12 / 32.0)) * 32)
                for width in WIDTHS),
            "maximum_aligned_payload_bytes": maximum_entry32,
            "maximum_payload_witness": {
                "operator_index": maximum_entry32_phase // 432,
                "partition": maximum_entry32_phase % 432,
            },
            "descriptor_single_bytes": descriptor_single,
            "single_buffer_total_bytes": entry32_single,
            "double_buffer_total_bytes": entry32_double,
            "double_buffer_reduction_vs_fixed12": 1.0 - (
                entry32_double / float(fixed_double)),
            "global_aligned_entry_payload_bytes": global_entry32_payload,
            "global_padding_over_packed_payload_bytes": (
                global_entry32_payload - sum(catalog["payload_sizes"])),
            "addressable_catalog_bytes": entry32_catalog,
            "catalog_storage_reduction_vs_fixed12": 1.0 - (
                entry32_catalog / float(fixed_payload)),
            "five_sample_prefetch_reduction_vs_fixed12": 1.0 - (
                entry32_five / float(fixed_five)),
        },
    }


def main():
    require(sha256(M80_SCRIPT) == TARGET_M80_SCRIPT_SHA,
            "M80 analyzer drift")
    require(sha256(M80_RESULT) == TARGET_M80_RESULT_SHA,
            "M80 result drift")
    require(sha256(M72_RESULT) == TARGET_M72_RESULT_SHA, "M72 result drift")
    require(sha256(M41_RESULT) == TARGET_M41_RESULT_SHA, "M41 result drift")
    catalog = phase_catalog()
    reconstruction = verify_m80(catalog)
    alternative_rows = alternatives(catalog)
    maxima = catalog["maximum_class_entries_with_witness"]
    payload = {
        "schema": "m80_addressable_phase_record_independent_hammer_oracle_v1",
        "status": "PASS_M80_ARITHMETIC_WITH_FORMAT_SPECIFICATION_P0",
        "identity": {
            "m80_analyzer_sha256": TARGET_M80_SCRIPT_SHA,
            "m80_result_sha256": TARGET_M80_RESULT_SHA,
            "m72_result_sha256": TARGET_M72_RESULT_SHA,
            "m41_result_sha256": TARGET_M41_RESULT_SHA,
            "weight_payload_sha256": catalog["weight_shas"],
        },
        "reconstruction": reconstruction,
        "width_histogram": dict(
            (str(width), catalog["width_histogram"][width])
            for width in range(8, 13)),
        "dedicated_bank_peak_witnesses": dict(
            (str(width), maxima[width]) for width in WIDTHS),
        "sum_of_independent_class_peaks": sum(
            maxima[width]["entries"] for width in WIDTHS),
        "entries_in_any_one_phase": ENTRIES,
        "independent_peaks_are_from_four_distinct_phases": len(set(
            (maxima[width]["operator_index"], maxima[width]["partition"])
            for width in WIDTHS)) == 4,
        "format_ambiguity": {
            "header_bit_order_unspecified": True,
            "ambiguous_header_phases": catalog["ambiguous_header_phases"],
            "offset_table_endianness_unspecified": True,
            "little_endian_offset_table_sha256": (
                catalog["offset_table_little_endian_sha256"]),
            "big_endian_offset_table_sha256": (
                catalog["offset_table_big_endian_sha256"]),
            "offset_endianness_changes_bytes": (
                catalog["offset_endianness_changes_bytes"]),
            "signed_payload_bit_order_and_twos_complement_serialization_unspecified": True,
            "round_trip_parser_vector_exists": False,
        },
        "alternatives": alternative_rows,
        "claim_boundary": {
            "arithmetic_and_capacity_envelope": True,
            "uniquely_serializable_byte_format": False,
            "finite_queue_or_bank_conflict_cycles": False,
            "sram_macro_or_ppa": False,
            "accuracy_or_system_speedup": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
