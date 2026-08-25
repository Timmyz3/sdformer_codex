#!/usr/bin/env python3
"""Independent M81 phase-layout and interleaved-bank reconstruction.

No M81, M80, M78, or M72 production Python module is imported.  Widths are
rebuilt directly from frozen M41 INT8 weights and M72 center JSON.  Both M81
record variants, descriptor offsets, catalog traffic, capacity envelopes, and
every logical 32-bit bank read are reconstructed locally.
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
M81_ANALYZER = HW / "system_simulator/scripts/analyze_m81_interleaved_word_pwp_buffer.py"
M81 = HW / (
    "results/m81_interleaved_word_pwp_buffer_valid825_internal_dev_r1_20260823/"
    "m81_interleaved_word_pwp_buffer.json")
M80 = HW / (
    "results/m80_addressable_phase_record_pwp_valid825_internal_dev_r1_20260823/"
    "m80_addressable_phase_record_pwp.json")
M78 = HW / (
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/"
    "m78_precision_elastic_pwp.json")
M72 = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_ROOT = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41 = M41_ROOT / "m41_h67_ep35_bottleneck_int8_bridge.json"
RECONSTRUCTION = HERE / "m81_independent_reconstruction.json"
REVIEW = HERE / "m81_interleaved_word_pwp_buffer_independent_hammer_review.json"
RECEIPT = HERE / "m81_interleaved_word_pwp_buffer_independent_hammer_validation_receipt.json"

EXPECTED_SHA = {
    "m81_analyzer": "a9a951e845599a6ff2c7a0ea86c2cb44260c83c88fdb968c0f358010583ce94e",
    "m81": "ab0ea4e87a58ee3945ff7ab36dc0507573fd8bf1ddba1958bbdd63a415a6b381",
    "m80": "dec76e2afa2b91420df514157a8ba9ca0f10ccae03004c84cee2e82b9d72a7da",
    "m78": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m72": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
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
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
OUTPUT_CHANNELS = OUTPUT_BLOCKS * OUTPUT_LANES
PHASES = 4 * PARTITIONS
ENTRIES = PATTERNS * OUTPUT_BLOCKS
CAP = 11
WIDTHS = (8, 9, 10, 11)
HEADER_BYTES = ENTRIES * 3 // 8
OFFSET_TABLE_BYTES = (PHASES + 1) * 4
WORD_BYTES = 4
BANKS = 8
PORT_BYTES = BANKS * WORD_BYTES
LOCAL_DESCRIPTOR_BYTES = ENTRIES * 2
FIXED12_PHASE_BYTES = ENTRIES * 144
WEIGHT_PHASE_BYTES = PARTITION_BITS * OUTPUT_BLOCKS * OUTPUT_LANES
WEIGHT_CATALOG_BYTES = PHASES * WEIGHT_PHASE_BYTES


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
        require(set(left) == set(right), label + " keys drift")
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


def build_widths(m72, m41):
    widths = np.zeros((4, PARTITIONS, PATTERNS, OUTPUT_BLOCKS), dtype=np.uint8)
    width_hist = Counter()
    outliers = []
    weight_shas = []
    digest = hashlib.sha256()
    for op, operator in enumerate(m72["operators"]):
        layer = next(row for row in m41["layers"]
                     if row["operator"] == operator["operator"])
        weight_info = next(row for row in layer["payloads"] if row["role"] == "weight")
        weight_path = M41_ROOT / weight_info["file"]
        observed = sha256_path(weight_path)
        require(observed == EXPECTED_WEIGHT_SHA[op] == weight_info["sha256"],
                "M41 weight identity drift")
        weight_shas.append(observed)
        weight = np.fromfile(str(weight_path), dtype=np.int8)
        require(weight.size == FEATURES * OUTPUT_CHANNELS, "weight extent drift")
        weight = weight.reshape(FEATURES, OUTPUT_CHANNELS).astype(np.int32)
        require(len(operator["partitions"]) == PARTITIONS, "partition extent drift")
        for partition, row in enumerate(operator["partitions"]):
            require(row["partition"] == partition and
                    len(row["centers_hex"]) == PATTERNS,
                    "center order/extent drift")
            source = weight[
                partition * PARTITION_BITS:(partition + 1) * PARTITION_BITS]
            for pattern, raw_center in enumerate(row["centers_hex"]):
                center = int(raw_center, 16)
                indices = [bit for bit in range(PARTITION_BITS)
                           if center & (1 << bit)]
                pwp = source[indices].sum(axis=0, dtype=np.int32)
                for block in range(OUTPUT_BLOCKS):
                    values = pwp[block * OUTPUT_LANES:(block + 1) * OUTPUT_LANES]
                    minimum = int(values.min())
                    maximum = int(values.max())
                    width = signed_width(minimum, maximum)
                    widths[op, partition, pattern, block] = width
                    width_hist[width] += 1
                    digest.update(canonical_bytes({
                        "op": op, "partition": partition, "pattern": pattern,
                        "block": block, "width": width,
                        "minimum": minimum, "maximum": maximum,
                    }))
                    if width > CAP:
                        outliers.append({
                            "operator_index": op,
                            "operator": operator["operator"],
                            "partition": partition,
                            "pattern": pattern,
                            "block": block,
                            "width": width,
                            "minimum": minimum,
                            "maximum": maximum,
                        })
        print("[M81 INDEPENDENT WIDTH] operator={}/4".format(op + 1), flush=True)
    return widths, width_hist, outliers, weight_shas, digest.hexdigest()


def map_entry_reads(start_word, word_count):
    """Return and verify every bank/row request for one logical PWP stream."""
    rows = []
    bank_conflicts = 0
    maximum_reads_one_bank = 0
    maximum_row = 0
    cross_row_beats = 0
    same_row_beats = 0
    logical_words = []
    service_beats = (word_count + BANKS - 1) // BANKS
    for beat in range(service_beats):
        first = start_word + beat * BANKS
        valid = min(BANKS, word_count - beat * BANKS)
        offsets = list(range(first, first + valid))
        requests = [(offset % BANKS, offset // BANKS, offset) for offset in offsets]
        per_bank = Counter(bank for bank, _, _ in requests)
        maximum_reads_one_bank = max(
            [maximum_reads_one_bank] + list(per_bank.values()))
        bank_conflicts += sum(max(0, count - 1) for count in per_bank.values())
        beat_rows = sorted(set(row for _, row, _ in requests))
        require(len(beat_rows) <= 2, "more than two adjacent rows in one beat")
        if len(beat_rows) == 2:
            require(beat_rows[1] == beat_rows[0] + 1,
                    "cross-row beat is not adjacent")
            cross_row_beats += 1
        else:
            same_row_beats += 1
        maximum_row = max([maximum_row] + [row for _, row, _ in requests])
        logical_words.extend(offset for _, _, offset in requests)
        rows.append(requests)
    require(logical_words == list(range(start_word, start_word + word_count)),
            "barrel logical-word coverage drift")
    return {
        "service_beats": service_beats,
        "bank_conflicts": bank_conflicts,
        "maximum_reads_one_bank": maximum_reads_one_bank,
        "cross_row_beats": cross_row_beats,
        "same_row_beats": same_row_beats,
        "maximum_row": maximum_row,
        "requests": rows,
    }


def variant_capacity(phase_payloads, catalog_payload, name):
    maximum = max(phase_payloads)
    maximum_words = align(maximum, WORD_BYTES) // WORD_BYTES
    exact_rows = int(math.ceil(maximum_words / float(BANKS)))
    envelopes = []
    for quantum in (1, 16, 64):
        rows = align(exact_rows, quantum)
        one_buffer = rows * BANKS * WORD_BYTES
        double_total = 2 * one_buffer + 2 * LOCAL_DESCRIPTOR_BYTES
        envelopes.append({
            "bank_depth_row_quantum": quantum,
            "rows_per_bank": rows,
            "one_payload_buffer_bytes": one_buffer,
            "two_payload_buffers_plus_two_descriptor_tables_bytes": double_total,
            "reduction_vs_fixed12_double_payload": (
                1.0 - double_total / float(2 * FIXED12_PHASE_BYTES)),
        })
    records = [align(HEADER_BYTES + payload, PORT_BYTES)
               for payload in phase_payloads]
    addressable = sum(records) + OFFSET_TABLE_BYTES
    five_sample = 5 * (WEIGHT_CATALOG_BYTES + sum(records)) + 5 * OFFSET_TABLE_BYTES
    fixed_five = 5 * (WEIGHT_CATALOG_BYTES + PHASES * FIXED12_PHASE_BYTES)
    return {
        "name": name,
        "catalog_pwp_payload_bytes": catalog_payload,
        "phase_payload_bytes_minimum": min(phase_payloads),
        "phase_payload_bytes_maximum": maximum,
        "phase_record_alignment_padding_bytes": (
            sum(records) - PHASES * HEADER_BYTES - sum(phase_payloads)),
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
    for name, path in {
            "m81_analyzer": M81_ANALYZER, "m81": M81, "m80": M80,
            "m78": M78, "m72": M72, "m41": M41}.items():
        require(sha256_path(path) == EXPECTED_SHA[name], name + " SHA drift")
    producer = strict_json(M81)
    m80 = strict_json(M80)
    m78 = strict_json(M78)
    m72 = strict_json(M72)
    m41 = strict_json(M41)
    require(m78["admission"]["valid825_internal_only"] is True,
            "M78 scope identity drift")
    widths, width_hist, outliers, weight_shas, catalog_digest = build_widths(m72, m41)
    require(width_hist == Counter({8: 52248, 9: 128893, 10: 37144, 11: 2898, 12: 1}),
            "width histogram drift")
    require(len(outliers) == 1, "escape extent drift")

    packed_payloads = []
    aligned_payloads = []
    packed_catalog = 0
    aligned_catalog = 0
    start_mod_hist = Counter()
    packed_max_start = -1
    aligned_max_start = -1
    aligned_overflows = []
    escape_entries = 0
    bank_totals = Counter()
    maximum_bank_row = 0
    phase_digest = hashlib.sha256()
    bank_digest = hashlib.sha256()
    for op in range(4):
        for partition in range(PARTITIONS):
            packed_bytes = 0
            aligned_bytes = 0
            phase_entries = []
            for pattern in range(PATTERNS):
                for block in range(OUTPUT_BLOCKS):
                    width = int(widths[op, partition, pattern, block])
                    if width > CAP:
                        escape_entries += 1
                        phase_entries.append({
                            "pattern": pattern, "block": block,
                            "width": width, "escape": True,
                        })
                        continue
                    entry_bytes = width * OUTPUT_LANES // 8
                    word_count = entry_bytes // WORD_BYTES
                    require(entry_bytes % WORD_BYTES == 0,
                            "entry is not 32-bit-word aligned")
                    packed_start = packed_bytes // WORD_BYTES
                    aligned_start = aligned_bytes // WORD_BYTES
                    packed_max_start = max(packed_max_start, packed_start)
                    aligned_max_start = max(aligned_max_start, aligned_start)
                    require(packed_start < (1 << 12),
                            "packed descriptor start overflow")
                    if aligned_start >= (1 << 12):
                        aligned_overflows.append({
                            "operator_index": op,
                            "partition": partition,
                            "pattern": pattern,
                            "output_block": block,
                            "width": width,
                            "start_word": aligned_start,
                        })
                    start_mod_hist[packed_start % BANKS] += 1
                    mapping = map_entry_reads(packed_start, word_count)
                    bank_totals["service_beats"] += mapping["service_beats"]
                    bank_totals["bank_conflicts"] += mapping["bank_conflicts"]
                    bank_totals["cross_row_beats"] += mapping["cross_row_beats"]
                    bank_totals["same_row_beats"] += mapping["same_row_beats"]
                    bank_totals["logical_words"] += word_count
                    bank_totals["read_slots"] += mapping["service_beats"] * BANKS
                    bank_totals["maximum_reads_one_bank"] = max(
                        bank_totals["maximum_reads_one_bank"],
                        mapping["maximum_reads_one_bank"])
                    maximum_bank_row = max(maximum_bank_row, mapping["maximum_row"])
                    bank_digest.update(canonical_bytes({
                        "op": op, "partition": partition, "pattern": pattern,
                        "block": block, "width": width,
                        "packed_start_word": packed_start,
                        "word_count": word_count,
                        "requests": mapping["requests"],
                    }))
                    phase_entries.append({
                        "pattern": pattern,
                        "block": block,
                        "width": width,
                        "escape": False,
                        "packed_start_word": packed_start,
                        "aligned_start_word": aligned_start,
                        "entry_words": word_count,
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
                "entries": phase_entries,
            }))
        print("[M81 INDEPENDENT PHASE] operator={}/4".format(op + 1), flush=True)
    require(len(packed_payloads) == PHASES and escape_entries == 1,
            "phase or escape conservation drift")
    require(packed_catalog == 23776068, "cap11 packed catalog drift")
    require(bank_totals["bank_conflicts"] == 0 and
            bank_totals["maximum_reads_one_bank"] == 1,
            "single-PWP bank conflict")

    variants = [
        variant_capacity(packed_payloads, packed_catalog,
                         "WORD_PACKED_8X32_INTERLEAVED"),
        variant_capacity(aligned_payloads, aligned_catalog,
                         "ENTRY_ALIGNED_32B"),
    ]
    compare(variants, producer["variants"], "variants")
    compare(dict((str(key), value) for key, value in sorted(start_mod_hist.items())),
            producer["start_word_mod8_histogram"], "start_word_mod8_histogram")
    require(packed_max_start ==
            producer["architecture"]["local_descriptor"]["maximum_observed_start_word"]
            == 3648, "packed maximum start-word drift")
    require(len(aligned_overflows) == 8 and aligned_max_start == 4152,
            "aligned descriptor overflow counterexample drift")
    require(variants[0]["exact_rows_per_32bit_bank"] == 460 and
            maximum_bank_row < 460, "packed bank depth coverage drift")

    producer_comparison = producer["comparison_to_m80_width_class_max_banks"]
    expected_comparison = {
        "m80_double_buffer_bytes": 63632,
        "m80_reduction_vs_fixed12": -0.7261284722222223,
        "m81_word_packed_exact_double_buffer_bytes": 29952,
        "independent_width_class_maxima_rejected": True,
    }
    compare(expected_comparison, producer_comparison, "m80_comparison")
    require(m80["staging_capacity"]
            ["elastic_class_banks_plus_descriptor_double_buffer_bytes"] == 63632,
            "M80 source comparison drift")

    exact = variants[0]["capacity_envelopes"][0]
    rounded16 = variants[0]["capacity_envelopes"][1]
    rounded64 = variants[0]["capacity_envelopes"][2]
    return {
        "schema": "m81_interleaved_word_pwp_buffer_independent_reconstruction_v1",
        "status": "PASS_M81_PACKED_RECONSTRUCTION_WITH_ALIGNED_DESCRIPTOR_P0",
        "identity_sha256": dict(EXPECTED_SHA, **{
            "weight_payloads": weight_shas,
            "independent_width_catalog_digest": catalog_digest,
            "independent_phase_layout_digest": phase_digest.hexdigest(),
            "independent_bank_request_digest": bank_digest.hexdigest(),
        }),
        "independence": {
            "production_modules_imported": False,
            "width_source": "direct M41 INT8 sum using M72 center JSON",
            "layout_and_bank_mapping_reimplemented": True,
        },
        "population": {
            "phases": PHASES,
            "entries_per_phase": ENTRIES,
            "eligible_entries": int(np.count_nonzero(widths <= CAP)),
            "escape_entries": escape_entries,
            "packed_catalog_payload_bytes": packed_catalog,
            "aligned_catalog_payload_bytes": aligned_catalog,
        },
        "descriptor_capacity": {
            "descriptor_bits_per_entry": 16,
            "packed_contract": "escape[15], width_class[14:13], reserved[12], start_word[11:0]",
            "packed_maximum_start_word": packed_max_start,
            "packed_12bit_start_capacity_pass": packed_max_start < 4096,
            "packed_unused_start_codes_above_maximum": 4096 - 1 - packed_max_start,
            "aligned_maximum_start_word": aligned_max_start,
            "aligned_required_start_bits": aligned_max_start.bit_length(),
            "aligned_12bit_start_capacity_pass": aligned_max_start < 4096,
            "aligned_overflow_entry_count": len(aligned_overflows),
            "aligned_overflow_entries": aligned_overflows,
            "zero_byte_cost_repair": (
                "Use escape[15], width_class[14:13], start_word[12:0]; "
                "the current reserved bit is sufficient to make both variants addressable."),
        },
        "bank_mapping": {
            "banks": BANKS,
            "bits_per_bank_word": 32,
            "logical_words": bank_totals["logical_words"],
            "service_beats": bank_totals["service_beats"],
            "read_slots": bank_totals["read_slots"],
            "logical_word_slot_utilization": (
                bank_totals["logical_words"] / float(bank_totals["read_slots"])),
            "bank_conflicts": bank_totals["bank_conflicts"],
            "maximum_reads_of_one_bank_in_one_cycle": (
                bank_totals["maximum_reads_one_bank"]),
            "same_row_beats": bank_totals["same_row_beats"],
            "cross_adjacent_row_beats": bank_totals["cross_row_beats"],
            "cross_row_fraction": (
                bank_totals["cross_row_beats"] /
                float(bank_totals["service_beats"])),
            "maximum_observed_bank_row": maximum_bank_row,
            "exact_460_row_depth_covers_all_reads": maximum_bank_row < 460,
            "single_cycle_interpretation": (
                "For one PWP stream, every beat addresses each bank at most "
                "once. Independent bank row-address inputs can therefore read "
                "base or base+1 in one cycle; a shared row address cannot."),
            "zero_bubble_rtl_proved": False,
        },
        "catalog_and_traffic": {
            "width_headers_bytes": PHASES * HEADER_BYTES,
            "phase_offset_table_bytes": OFFSET_TABLE_BYTES,
            "packed_phase_alignment_padding_bytes": (
                variants[0]["phase_record_alignment_padding_bytes"]),
            "packed_addressable_catalog_bytes": (
                variants[0]["addressable_catalog_bytes_including_headers_offsets_alignment"]),
            "catalog_overhead_beyond_pwp_payload_bytes": (
                variants[0]["addressable_catalog_bytes_including_headers_offsets_alignment"]
                - packed_catalog),
            "five_sample_weight_plus_addressable_pwp_bytes": (
                variants[0]["five_sample_weight_plus_addressable_pwp_bytes"]),
            "five_sample_prefetch_reduction_vs_fixed12": (
                variants[0]["five_sample_prefetch_reduction_vs_fixed12"]),
        },
        "variants": variants,
        "m80_to_m81_capacity": {
            "fixed12_double_payload_bytes": 2 * FIXED12_PHASE_BYTES,
            "m80_four_independent_width_class_bank_bytes": 63632,
            "m80_reduction_vs_fixed12": -0.7261284722222223,
            "m81_exact_460_rows_bytes": exact[
                "two_payload_buffers_plus_two_descriptor_tables_bytes"],
            "m81_exact_reduction_vs_fixed12": exact[
                "reduction_vs_fixed12_double_payload"],
            "m81_reduction_vs_m80_fraction": 1.0 - 29952 / 63632.0,
            "m81_464_rows_bytes": rounded16[
                "two_payload_buffers_plus_two_descriptor_tables_bytes"],
            "m81_464_rows_reduction_vs_fixed12": rounded16[
                "reduction_vs_fixed12_double_payload"],
            "m81_512_rows_bytes": rounded64[
                "two_payload_buffers_plus_two_descriptor_tables_bytes"],
            "m81_512_rows_reduction_vs_fixed12": rounded64[
                "reduction_vs_fixed12_double_payload"],
        },
        "producer_mismatches": {
            "packed_phase_payloads_and_catalog": 0,
            "aligned_phase_payloads_and_catalog": 0,
            "start_mod_histogram": 0,
            "capacity_envelopes": 0,
            "traffic": 0,
            "aligned_descriptor_contract_counterexample": 8,
        },
    }


def validate_review(payload):
    if RECONSTRUCTION.exists():
        compare(strict_json(RECONSTRUCTION), payload, "stored_reconstruction")
    if REVIEW.exists():
        review = strict_json(REVIEW)
        require(review["status"] ==
                "M81_PACKED_GO_ALIGNED_NO_GO_DESCRIPTOR_P0_RTL_MACRO_UNADMITTED",
                "review status drift")
        require(len(review["findings"]["p0"]) == 1 and
                len(review["findings"]["p1"]) == 6,
                "review finding count drift")
        require(review["scores"] == {
            "hardware_innovation": 61,
            "capacity_advantage": 78,
            "performance_advantage": 54,
            "evidence_quality": 78,
            "m81_scoped_milestone_quality": 82,
            "date_paper_completeness": 47,
        }, "review scores drift")
    if RECEIPT.exists():
        receipt = strict_json(RECEIPT)
        require(receipt["status"] ==
                "PASS_M81_INTERLEAVED_WORD_PWP_BUFFER_INDEPENDENT_HAMMER" and
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
                "output must stay in review directory")
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bank = payload["bank_mapping"]
    descriptor = payload["descriptor_capacity"]
    print("PASS M81 independent: phases=1728 packed_max={} conflicts={} "
          "cross_row={:.6f} aligned_overflows={} P0=1".format(
              descriptor["packed_maximum_start_word"], bank["bank_conflicts"],
              bank["cross_row_fraction"],
              descriptor["aligned_overflow_entry_count"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M81 independent: {}".format(error), flush=True)
        raise SystemExit(1)
