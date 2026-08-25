#!/usr/bin/env python3
"""DSE an addressable unified 8x32-bit-bank PWP staging buffer.

M80 showed that separately sizing four width-class banks by independent phase
maxima over-allocates capacity.  M81 instead concatenates every eligible PWP
in 32-bit words.  Eight interleaved 32-bit banks return eight consecutive words
per cycle; per-bank row addresses straddle at most two adjacent rows and a
barrel reorder restores logical order.  A 16-bit local descriptor stores
    escape, width class, and a 13-bit start-word offset for each of 128 entries.

Two record formats are compared: word-packed (no per-entry padding) and a
simpler 32-byte-entry-aligned format.  This remains a layout/capacity DSE, not
RTL, macro, timing, energy, accuracy, or system performance evidence.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M78_ANALYZER = HW / "system_simulator/scripts/analyze_m78_precision_elastic_pwp.py"
M78_RESULT = HW / (
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/"
    "m78_precision_elastic_pwp.json")
M80_RESULT = HW / (
    "results/m80_addressable_phase_record_pwp_valid825_internal_dev_r1_20260823/"
    "m80_addressable_phase_record_pwp.json")
EXPECTED_SHA256 = {
    "m78_analyzer": "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b",
    "m78_result": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m80_result": "dec76e2afa2b91420df514157a8ba9ca0f10ccae03004c84cee2e82b9d72a7da",
}
CAP = 11
PHASES = 1728
ENTRIES = 128
HEADER_BYTES = 48
OFFSET_TABLE_BYTES = (PHASES + 1) * 4
WORD_BYTES = 4
BANKS = 8
PORT_BYTES = BANKS * WORD_BYTES
LOCAL_DESCRIPTOR_BYTES = ENTRIES * 2
FIXED12_PHASE_BYTES = ENTRIES * 144
WEIGHT_CATALOG_BYTES = PHASES * 12288
WIDTHS = (8, 9, 10, 11)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_m78():
    spec = importlib.util.spec_from_file_location("m81_m78", str(M78_ANALYZER))
    require(spec is not None and spec.loader is not None, "cannot import M78")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def align(value, quantum):
    return int(math.ceil(value / float(quantum))) * quantum


def variant_capacity(phase_payloads, catalog_payload, name):
    max_payload = max(phase_payloads)
    max_words = align(max_payload, WORD_BYTES) // WORD_BYTES
    exact_rows = int(math.ceil(max_words / float(BANKS)))
    envelopes = []
    for row_quantum in (1, 16, 64):
        rows = align(exact_rows, row_quantum)
        one_buffer = rows * BANKS * WORD_BYTES
        double_total = 2 * one_buffer + 2 * LOCAL_DESCRIPTOR_BYTES
        envelopes.append({
            "bank_depth_row_quantum": row_quantum,
            "rows_per_bank": rows,
            "one_payload_buffer_bytes": one_buffer,
            "two_payload_buffers_plus_two_descriptor_tables_bytes": double_total,
            "reduction_vs_fixed12_double_payload": 1.0 - (
                double_total / float(2 * FIXED12_PHASE_BYTES)),
        })
    record_bytes = [align(HEADER_BYTES + value, PORT_BYTES)
                    for value in phase_payloads]
    addressable_catalog = sum(record_bytes) + OFFSET_TABLE_BYTES
    five_sample = 5 * (WEIGHT_CATALOG_BYTES + sum(record_bytes)) + 5 * OFFSET_TABLE_BYTES
    fixed_five = 5 * (WEIGHT_CATALOG_BYTES + PHASES * FIXED12_PHASE_BYTES)
    return {
        "name": name,
        "catalog_pwp_payload_bytes": catalog_payload,
        "phase_payload_bytes_minimum": min(phase_payloads),
        "phase_payload_bytes_maximum": max_payload,
        "phase_record_alignment_padding_bytes": (
            sum(record_bytes) - PHASES * HEADER_BYTES - sum(phase_payloads)),
        "addressable_catalog_bytes_including_headers_offsets_alignment": (
            addressable_catalog),
        "addressable_catalog_reduction_vs_fixed12": 1.0 - (
            addressable_catalog / float(PHASES * FIXED12_PHASE_BYTES)),
        "five_sample_weight_plus_addressable_pwp_bytes": five_sample,
        "five_sample_prefetch_reduction_vs_fixed12": 1.0 - (
            five_sample / float(fixed_five)),
        "maximum_payload_words": max_words,
        "exact_rows_per_32bit_bank": exact_rows,
        "capacity_envelopes": envelopes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M81 output overwrite")
    analyzer_start_sha = sha256(Path(__file__).resolve())
    for name, path in (("m78_analyzer", M78_ANALYZER),
                       ("m78_result", M78_RESULT),
                       ("m80_result", M80_RESULT)):
        require(sha256(path) == EXPECTED_SHA256[name],
                "M81 input identity drift: " + name)
    m78 = load_m78()
    m72_result = m78.strict_json(m78.M72_RESULT)
    m41_result = m78.strict_json(m78.M41_RESULT)
    width_catalog, width_hist, center_max_hist, outliers, weight_shas = (
        m78.build_width_catalog(m72_result, m41_result))
    require(len(outliers) == 1, "M81 unique escape drift")

    packed_phase_payloads = []
    aligned_phase_payloads = []
    packed_catalog = aligned_catalog = 0
    start_word_mod8_hist = Counter()
    maximum_start_word = 0
    maximum_aligned_start_word = 0
    escape_entries = 0
    for op in range(4):
        for partition in range(432):
            packed_bytes = aligned_bytes = 0
            for pattern in range(16):
                for block in range(8):
                    width = width_catalog[op][partition][pattern]["blocks"][block]["width"]
                    if width > CAP:
                        escape_entries += 1
                        continue
                    require(width in WIDTHS, "M81 unexpected eligible width")
                    require(packed_bytes % WORD_BYTES == 0,
                            "M81 packed entry is not word aligned")
                    start_word = packed_bytes // WORD_BYTES
                    require(start_word < (1 << 13),
                            "M81 13-bit local word offset overflow")
                    maximum_start_word = max(maximum_start_word, start_word)
                    start_word_mod8_hist[start_word % BANKS] += 1
                    entry_bytes = width * 96 // 8
                    aligned_start_word = aligned_bytes // WORD_BYTES
                    require(aligned_start_word < (1 << 13),
                            "M81 aligned 13-bit local word offset overflow")
                    maximum_aligned_start_word = max(
                        maximum_aligned_start_word, aligned_start_word)
                    packed_bytes += entry_bytes
                    aligned_bytes += align(entry_bytes, PORT_BYTES)
            packed_phase_payloads.append(packed_bytes)
            aligned_phase_payloads.append(aligned_bytes)
            packed_catalog += packed_bytes
            aligned_catalog += aligned_bytes
    require(len(packed_phase_payloads) == PHASES,
            "M81 phase extent drift")
    require(packed_catalog == 23776068 and escape_entries == 1,
            "M81 cap11 payload/escape drift")

    variants = [
        variant_capacity(packed_phase_payloads, packed_catalog,
                         "WORD_PACKED_8X32_INTERLEAVED"),
        variant_capacity(aligned_phase_payloads, aligned_catalog,
                         "ENTRY_ALIGNED_32B"),
    ]
    require(variants[0]["capacity_envelopes"][0]
            ["reduction_vs_fixed12_double_payload"] > 0.0,
            "M81 unified packed staging failed to beat fixed12")

    require(sha256(Path(__file__).resolve()) == analyzer_start_sha,
            "M81 analyzer source changed during execution")
    payload = {
        "schema": "m81_unified_interleaved_word_pwp_buffer_dse_v1",
        "status": "PASS_M81_UNIFIED_BUFFER_CAPACITY_DSE_RTL_MACRO_UNADMITTED",
        "identity": {
            "analyzer_start_end_sha256": analyzer_start_sha,
            "m78_analyzer_sha256": sha256(M78_ANALYZER),
            "m78_result_sha256": sha256(M78_RESULT),
            "m80_result_sha256": sha256(M80_RESULT),
            "weight_payload_sha256": weight_shas,
        },
        "architecture": {
            "payload_banks": 8,
            "bits_per_bank_word": 32,
            "aggregate_read_bits_per_cycle": 256,
            "logical_word_mapping": "bank=word_offset mod 8; row=floor(word_offset/8)",
            "cross_row_read": "for a start modulo 8, each bank reads either base row or base+1; one read per bank and a barrel reorder emits eight consecutive logical words",
            "bank_conflicts_per_single_pwp_read": 0,
            "local_descriptor": {
                "entries_per_phase": ENTRIES,
                "bits_per_entry": 16,
                "fields": "escape[15], width_class[14:13], start_word[12:0]",
                "bytes_per_phase": LOCAL_DESCRIPTOR_BYTES,
                "maximum_observed_word_packed_start_word": maximum_start_word,
                "maximum_observed_entry_aligned_start_word": (
                    maximum_aligned_start_word),
            },
            "pwp_service_cycles": {"signed8": 3, "signed9": 4, "signed10": 4, "signed11": 5},
            "descriptor_lookup_throughput": "one per cycle is sufficient because every PWP occupies at least three payload cycles",
            "zero_bubble_streaming_rtl_proven": False,
        },
        "start_word_mod8_histogram": dict(sorted(start_word_mod8_hist.items())),
        "variants": variants,
        "comparison_to_m80_width_class_max_banks": {
            "m80_double_buffer_bytes": 63632,
            "m80_reduction_vs_fixed12": -0.7261284722222223,
            "m81_word_packed_exact_double_buffer_bytes": (
                variants[0]["capacity_envelopes"][0]
                ["two_payload_buffers_plus_two_descriptor_tables_bytes"]),
            "independent_width_class_maxima_rejected": True,
        },
        "admission": {
            "concrete_word_address_and_local_descriptor_format": True,
            "single_pwp_bank_conflict_free_mapping_by_construction": True,
            "catalog_header_alignment_offset_and_descriptor_capacity_charged": True,
            "macro_depth_rounding_envelopes": True,
            "multi_pwp_concurrency": False,
            "zero_bubble_or_finite_queue_rtl": False,
            "dc_sta_formality_saif_ptpx": False,
            "accuracy": False,
            "full_network_or_system_speedup": False,
            "date_headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    best = variants[0]["capacity_envelopes"][0]
    print("PASS M81 packed_catalog_save={:.6f} staging_save={:.6f} aligned_catalog_save={:.6f}".format(
        variants[0]["addressable_catalog_reduction_vs_fixed12"],
        best["reduction_vs_fixed12_double_payload"],
        variants[1]["addressable_catalog_reduction_vs_fixed12"]),
        flush=True)


if __name__ == "__main__":
    main()
