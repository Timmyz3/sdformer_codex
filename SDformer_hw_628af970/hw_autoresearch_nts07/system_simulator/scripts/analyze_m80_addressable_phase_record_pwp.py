#!/usr/bin/env python3
"""Make M78 precision-elastic PWPs byte-addressable by phase records.

Each of the 1,728 operator/partition phases is stored as a 48-byte width header
(3 bits for each of 16 patterns x 8 output blocks), followed by the eligible
8--11-bit PWP vectors in deterministic pattern-major/block-major order.  The
record is padded once to a 32-byte DMA boundary.  A prefetch parser scatters
payloads into four width-class staging banks and builds a 10-bit local
{escape,class,rank} table, avoiding a global per-PWP pointer table.

This closes an address-format DSE only.  It does not admit SRAM macros, RTL
cycles/PPA, accuracy, full-network performance, or a paper headline.
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
EXPECTED_M78_ANALYZER_SHA256 = (
    "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b")
EXPECTED_M78_RESULT_SHA256 = (
    "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc")
CAP = 11
PHASES = 4 * 432
ENTRIES_PER_PHASE = 16 * 8
HEADER_BITS_PER_ENTRY = 3
HEADER_BYTES = ENTRIES_PER_PHASE * HEADER_BITS_PER_ENTRY // 8
LOCAL_DESCRIPTOR_BITS = 10
LOCAL_DESCRIPTOR_BYTES = int(math.ceil(
    ENTRIES_PER_PHASE * LOCAL_DESCRIPTOR_BITS / 8.0))
DMA_BYTES = 32
WEIGHT_PHASE_BYTES = 16 * 8 * 96
FIXED12_PWP_PHASE_BYTES = 16 * 8 * 144
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
    spec = importlib.util.spec_from_file_location("m80_m78", str(M78_ANALYZER))
    require(spec is not None and spec.loader is not None, "cannot import M78")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nearest_rank(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M80 output overwrite")
    analyzer_start_sha = sha256(Path(__file__).resolve())
    require(sha256(M78_ANALYZER) == EXPECTED_M78_ANALYZER_SHA256,
            "M80 M78 analyzer identity drift")
    require(sha256(M78_RESULT) == EXPECTED_M78_RESULT_SHA256,
            "M80 M78 result identity drift")
    m78 = load_m78()
    m72_result = m78.strict_json(m78.M72_RESULT)
    m41_result = m78.strict_json(m78.M41_RESULT)
    width_catalog, width_hist, center_max_hist, outliers, weight_shas = (
        m78.build_width_catalog(m72_result, m41_result))
    require(len(outliers) == 1 and outliers[0]["required_signed_bits"] == 12,
            "M80 expected the frozen unique 12-bit outlier")

    phase_rows = []
    total = Counter()
    maximum_class_entries = Counter()
    record_sizes = []
    payload_sizes = []
    offset = 0
    for op in range(4):
        for partition in range(432):
            class_counts = Counter()
            payload_bytes = 0
            escape_entries = 0
            for pattern in range(16):
                for block in range(8):
                    width = width_catalog[op][partition][pattern]["blocks"][block]["width"]
                    if width <= CAP:
                        class_counts[width] += 1
                        payload_bytes += width * 96 // 8
                    else:
                        escape_entries += 1
            require(sum(class_counts.values()) + escape_entries == ENTRIES_PER_PHASE,
                    "M80 phase entry conservation failure")
            for width in WIDTHS:
                maximum_class_entries[width] = max(
                    maximum_class_entries[width], class_counts[width])
            raw_bytes = HEADER_BYTES + payload_bytes
            aligned_bytes = int(math.ceil(raw_bytes / float(DMA_BYTES))) * DMA_BYTES
            padding_bytes = aligned_bytes - raw_bytes
            phase_rows.append({
                "operator_index": op,
                "partition": partition,
                "record_offset_bytes": offset,
                "width_class_entries": dict((str(width), class_counts[width])
                                             for width in WIDTHS),
                "escape_entries": escape_entries,
                "header_bytes": HEADER_BYTES,
                "pwp_payload_bytes": payload_bytes,
                "raw_record_bytes": raw_bytes,
                "dma_aligned_record_bytes": aligned_bytes,
                "dma_padding_bytes": padding_bytes,
                "record_dma_beats": aligned_bytes // DMA_BYTES,
            })
            offset += aligned_bytes
            total["pwp_payload_bytes"] += payload_bytes
            total["header_bytes"] += HEADER_BYTES
            total["padding_bytes"] += padding_bytes
            total["aligned_record_bytes"] += aligned_bytes
            total["escape_entries"] += escape_entries
            record_sizes.append(aligned_bytes)
            payload_sizes.append(payload_bytes)
    require(len(phase_rows) == PHASES, "M80 phase extent drift")
    require(total["pwp_payload_bytes"] == 23776068,
            "M80 cap11 elastic payload drift")
    require(total["escape_entries"] == 1,
            "M80 cap11 escape-entry drift")

    # One 32-bit offset for every phase boundary, including the terminal end.
    random_access_offset_table_bytes = (PHASES + 1) * 4
    fixed12_payload_bytes = PHASES * FIXED12_PWP_PHASE_BYTES
    addressable_catalog_bytes = (
        total["aligned_record_bytes"] + random_access_offset_table_bytes)
    addressable_reduction = 1.0 - addressable_catalog_bytes / float(
        fixed12_payload_bytes)

    class_bank_payload_bytes = dict(
        (str(width), maximum_class_entries[width] * width * 96 // 8)
        for width in WIDTHS)
    single_buffer_class_bank_bytes = sum(class_bank_payload_bytes.values())
    fixed12_single_buffer_bytes = FIXED12_PWP_PHASE_BYTES
    local_descriptor_double_buffer_bytes = 2 * LOCAL_DESCRIPTOR_BYTES
    elastic_staging_double_buffer_bytes = (
        2 * single_buffer_class_bank_bytes + local_descriptor_double_buffer_bytes)
    fixed12_staging_double_buffer_bytes = 2 * fixed12_single_buffer_bytes

    weight_catalog_bytes = PHASES * WEIGHT_PHASE_BYTES
    fixed12_five_sample_prefetch_bytes = 5 * (
        weight_catalog_bytes + fixed12_payload_bytes)
    elastic_five_sample_prefetch_bytes = 5 * (
        weight_catalog_bytes + total["aligned_record_bytes"])
    # Conservatively reload the small random-access table for every sample.
    elastic_with_offsets_five_sample_bytes = (
        elastic_five_sample_prefetch_bytes +
        5 * random_access_offset_table_bytes)

    parser_cycles_per_phase = ENTRIES_PER_PHASE
    minimum_record_dma_cycles = min(record_sizes) // DMA_BYTES
    require(parser_cycles_per_phase <= minimum_record_dma_cycles,
            "M80 one-entry/cycle header parser cannot hide under phase DMA")

    require(sha256(Path(__file__).resolve()) == analyzer_start_sha,
            "M80 analyzer source changed during execution")
    payload = {
        "schema": "m80_addressable_phase_record_precision_elastic_pwp_v1",
        "status": "PASS_M80_CAP11_PHASE_RECORD_LAYOUT_DSE_RTL_MACRO_UNADMITTED",
        "identity": {
            "analyzer_start_end_sha256": analyzer_start_sha,
            "m78_analyzer_sha256": sha256(M78_ANALYZER),
            "m78_result_sha256": sha256(M78_RESULT),
            "weight_payload_sha256": weight_shas,
        },
        "format": {
            "logical_phase_order": "operator-major then partition-major",
            "entry_order_within_phase": "pattern-major then output-block-major",
            "phase_count": PHASES,
            "entries_per_phase": ENTRIES_PER_PHASE,
            "header": {
                "bits_per_entry": HEADER_BITS_PER_ENTRY,
                "bytes_per_phase": HEADER_BYTES,
                "codes": {"0": "signed8", "1": "signed9", "2": "signed10", "3": "signed11", "4": "bit_sparse_escape"},
            },
            "payload": "eligible PWP vectors concatenate in entry order; each vector has exactly 96*width bits and is 32-bit-word aligned",
            "phase_record_alignment_bytes": DMA_BYTES,
            "random_access_phase_offset_table": {
                "entries": PHASES + 1,
                "bits_per_entry": 32,
                "bytes": random_access_offset_table_bytes,
            },
            "prefetch_parser": "one header entry per cycle; scatter to width-class staging bank and emit 10-bit {escape,class,local_rank}",
            "local_descriptor_bits_per_entry": LOCAL_DESCRIPTOR_BITS,
            "local_descriptor_bytes_per_phase": LOCAL_DESCRIPTOR_BYTES,
        },
        "catalog_capacity": {
            "fixed12_reference_pwp_bytes": fixed12_payload_bytes,
            "elastic_pwp_payload_bytes": total["pwp_payload_bytes"],
            "width_header_bytes": total["header_bytes"],
            "phase_alignment_padding_bytes": total["padding_bytes"],
            "phase_offset_table_bytes": random_access_offset_table_bytes,
            "addressable_elastic_catalog_bytes": addressable_catalog_bytes,
            "addressable_storage_reduction_vs_fixed12": addressable_reduction,
            "unique_escape_entries": total["escape_entries"],
        },
        "staging_capacity": {
            "maximum_entries_by_width_class": dict(
                (str(width), maximum_class_entries[width]) for width in WIDTHS),
            "single_buffer_bytes_by_width_class": class_bank_payload_bytes,
            "elastic_class_banks_single_buffer_bytes": single_buffer_class_bank_bytes,
            "local_descriptor_single_buffer_bytes": LOCAL_DESCRIPTOR_BYTES,
            "elastic_class_banks_plus_descriptor_double_buffer_bytes": (
                elastic_staging_double_buffer_bytes),
            "fixed12_reference_double_buffer_bytes": fixed12_staging_double_buffer_bytes,
            "elastic_double_buffer_reduction_vs_fixed12": 1.0 - (
                elastic_staging_double_buffer_bytes /
                float(fixed12_staging_double_buffer_bytes)),
            "bank_word_bits": dict((str(width), width * 96) for width in WIDTHS),
            "shared32_words_per_entry": dict((str(width), width * 3) for width in WIDTHS),
            "shared32_beats_per_entry": {"8": 3, "9": 4, "10": 4, "11": 5},
        },
        "traffic": {
            "bit_sparse_weight_only_five_sample_bytes": 5 * weight_catalog_bytes,
            "fixed12_weight_plus_pwp_five_sample_bytes": (
                fixed12_five_sample_prefetch_bytes),
            "elastic_weight_plus_addressable_pwp_five_sample_bytes": (
                elastic_with_offsets_five_sample_bytes),
            "elastic_prefetch_reduction_vs_fixed12": 1.0 - (
                elastic_with_offsets_five_sample_bytes /
                float(fixed12_five_sample_prefetch_bytes)),
            "elastic_prefetch_ratio_vs_bit_sparse_weight_only": (
                elastic_with_offsets_five_sample_bytes /
                float(5 * weight_catalog_bytes)),
        },
        "parser_throughput": {
            "header_entries_per_cycle": 1,
            "cycles_per_phase": parser_cycles_per_phase,
            "minimum_phase_record_dma_cycles": minimum_record_dma_cycles,
            "parser_hidden_under_record_dma_for_every_phase": True,
            "runtime_local_descriptor_lookup_per_cycle": 1,
            "minimum_pwp_shared32_service_cycles": 3,
            "lookup_throughput_below_pwp_port_demand": True,
            "finite_fifo_and_backpressure_rtl_proven": False,
        },
        "record_statistics": {
            "payload_bytes_minimum": min(payload_sizes),
            "payload_bytes_p50_nearest_rank": nearest_rank(payload_sizes, 0.50),
            "payload_bytes_p95_nearest_rank": nearest_rank(payload_sizes, 0.95),
            "payload_bytes_maximum": max(payload_sizes),
            "aligned_record_bytes_minimum": min(record_sizes),
            "aligned_record_bytes_p50_nearest_rank": nearest_rank(record_sizes, 0.50),
            "aligned_record_bytes_p95_nearest_rank": nearest_rank(record_sizes, 0.95),
            "aligned_record_bytes_maximum": max(record_sizes),
        },
        "phase_rows_sha256": hashlib.sha256(
            (json.dumps(phase_rows, sort_keys=True, separators=(",", ":")) + "\n")
            .encode("utf-8")).hexdigest(),
        "admission": {
            "byte_addressable_phase_record_format": True,
            "metadata_header_and_alignment_charged": True,
            "phase_offset_table_charged": True,
            "width_class_staging_capacity_envelope": True,
            "rtl_or_finite_queue_cycles": False,
            "sram_macro_feasibility_or_ppa": False,
            "accuracy": False,
            "full_network_or_system_speedup": False,
            "date_headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M80 addressable_save={:.6f} traffic_save={:.6f} staging_save={:.6f}".format(
        addressable_reduction,
        payload["traffic"]["elastic_prefetch_reduction_vs_fixed12"],
        payload["staging_capacity"]["elastic_double_buffer_reduction_vs_fixed12"]),
        flush=True)


if __name__ == "__main__":
    main()
