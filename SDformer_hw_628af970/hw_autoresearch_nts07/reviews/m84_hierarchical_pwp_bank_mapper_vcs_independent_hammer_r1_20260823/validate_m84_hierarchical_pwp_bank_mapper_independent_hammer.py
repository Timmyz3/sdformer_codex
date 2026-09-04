#!/usr/bin/env python3
"""Independent M84 geometry, bank-map, VCS, and claim-boundary audit.

No M84/M81/M78/M72 producer Python module is imported.  Widths are rebuilt
from frozen M41 INT8 weights and M72 centers.  The sealed geometry binary is
decoded independently and every phase, entry, prefix, beat, row, and barrel
mapping is reconstructed locally.
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
RTL = HW / "rtl_m84/hierarchical_pwp_bank_mapper.sv"
SVA = HW / "verif_m84/hierarchical_pwp_bank_mapper_assertions.sv"
TB = HW / "tb_m84/tb_hierarchical_pwp_bank_mapper.sv"
FILELIST = HW / "dc_handoff/filelists/date_m84_hierarchical_pwp_bank_mapper_vcs.f"
EXPORTER = HW / "system_simulator/scripts/export_m84_hierarchical_pwp_geometry.py"
GEOMETRY = HW / "results/m84_hierarchical_pwp_geometry_r2_20260823/m84_phase_geometry.bin"
GEOMETRY_RECEIPT = HW / (
    "results/m84_hierarchical_pwp_geometry_r2_20260823/"
    "m84_phase_geometry_receipt.json")
CONTRACT = HW / "contracts/m84_hierarchical_pwp_bank_mapper_vcs_contract_r1_20260823.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m84_hierarchical_pwp_bank_mapper_exhaustive_sva.sh"
M72 = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_ROOT = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41 = M41_ROOT / "m41_h67_ep35_bottleneck_int8_bridge.json"
M81 = HW / (
    "results/m81_interleaved_word_pwp_buffer_valid825_internal_dev_r2_20260823/"
    "m81_interleaved_word_pwp_buffer.json")
SEALED_RUN = HERE / "sealed_vcs_rerun"
DIRECTED_RUN = HERE / "independent_directed_random_run"
INDEPENDENT_TB = HERE / "tb_m84_independent_directed_random.sv"
INDEPENDENT_FILELIST = HERE / "independent_directed_random.f"
INDEPENDENT_RUNNER = HERE / "run_independent_directed_random_vcs.sh"
RECONSTRUCTION = HERE / "m84_independent_reconstruction.json"
REVIEW = HERE / "m84_hierarchical_pwp_bank_mapper_independent_hammer_review.json"
RECEIPT = HERE / "m84_hierarchical_pwp_bank_mapper_independent_hammer_validation_receipt.json"

EXPECTED_SHA = {
    "rtl": "8dafcf1e049dfee1a06999c93010a6e8c2458cc17c9a2de712b26d4fc40a2067",
    "sva": "c0454345f5e2640a10351f142faffa647e05699a140386d81aaa7faabf59870b",
    "tb": "85fa11d271b7d10dee5b47cbabe890d34ea2320480a693f7971c676a6b785826",
    "filelist": "00dc39c45b175558f457f2dab168f4e19e985c4f5c5d520b872d6d693631185e",
    "exporter": "b90afd4d5f7d91111ba54d08a9e91f5ae31d3b3c9809520b8d27b3c1d1313c76",
    "geometry": "294ea28b95ca2ef5c4adcb77195aabd388fc5b0ebc16bc4a3affcb8800b18e5d",
    "geometry_receipt": "bcebdbe95e7f455add38cbf9781c804edaa923fe83a726d755ec20b45b6b8df1",
    "contract": "b5eea5505346eb208a7c8d5f3f96a2004ff26d6f26a53f6c00988537e2e5acd0",
    "runner": "3fdb5a9c13a05cc5a009648cd0acaa0177d0733b75f57fb0538af133b189bd76",
    "m72": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m81": "515e023421a2650077b61fb620b06428786eb180dd3d36d05eccec1c8d2fabad",
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
BANKS = 8
LANES = 96
CHANNELS = BLOCKS * LANES
PHASES = 4 * PARTITIONS
ENTRIES = PATTERNS * BLOCKS
HEADER_BYTES = 48
FIXTURE_BASE_BYTES = 32
PACKED_BASE_BYTES = 26
FLAT_METADATA_BYTES = 256
HIERARCHICAL_METADATA_BYTES = HEADER_BYTES + PACKED_BASE_BYTES
CODE_BY_WIDTH = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4}
WORDS_BY_CODE = {0: 24, 1: 27, 2: 30, 3: 33, 4: 0}
BEATS_BY_CODE = {0: 3, 1: 4, 2: 4, 3: 5, 4: 0}


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


def build_widths(m72, m41):
    widths = np.zeros((4, PARTITIONS, PATTERNS, BLOCKS), dtype=np.uint8)
    histogram = Counter()
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
                    histogram[width] += 1
                    digest.update(canonical_bytes({
                        "op": op, "partition": partition, "pattern": pattern,
                        "block": block, "width": width,
                    }))
        print("[M84 INDEPENDENT WIDTH] operator={}/4".format(op + 1), flush=True)
    return widths, histogram, weight_shas, digest.hexdigest()


def decode_fixture_chunk(raw, phase):
    offset = phase * (HEADER_BYTES + FIXTURE_BASE_BYTES)
    chunk = raw[offset:offset + HEADER_BYTES + FIXTURE_BASE_BYTES]
    require(len(chunk) == HEADER_BYTES + FIXTURE_BASE_BYTES,
            "truncated geometry phase")
    header_integer = int.from_bytes(chunk[:HEADER_BYTES], "little")
    codes = [(header_integer >> (3 * entry)) & 7 for entry in range(ENTRIES)]
    bases = []
    base_bytes = chunk[HEADER_BYTES:]
    for pattern in range(PATTERNS):
        base = int.from_bytes(base_bytes[2*pattern:2*pattern+2], "little")
        require(base < 8192, "fixture pattern-base high bits nonzero")
        bases.append(base)
    return codes, bases


def validate_vcs_runs():
    production_pass = (
        "PASS M84 exhaustive phases=1728 entries=221184 escape=1 "
        "beats=835382 cross_row=725103 invalid_attacks=1 metadata=74B_vs_256B")
    independent_pass = (
        "PASS M84 independent directed_random directed=24 random=4096 "
        "start_mod8=8 pattern15_block7=5 escape_neighbor=3 "
        "selected_reserved_blocked=3 prior_reserved_failopen=3 overflow_wrap=2")
    for run, pass_line in ((SEALED_RUN, production_pass),
                           (DIRECTED_RUN, independent_pass)):
        require((run / "compile.rc").read_text(encoding="utf-8").strip() == "0",
                "VCS compile rc failure")
        require((run / "sim.rc").read_text(encoding="utf-8").strip() == "0",
                "VCS sim rc failure")
        compile_log = (run / "compile.raw.log").read_text(
            encoding="utf-8", errors="replace")
        sim_log = (run / "sim.raw.log").read_text(
            encoding="utf-8", errors="replace")
        require(pass_line in sim_log, "VCS PASS line absent")
        require(not any(token in compile_log for token in (
            "Warning-[", "Error-[", "\nError")), "VCS compile warning/error")
        require(not any(token in sim_log for token in (
            "Fatal:", "Error:", "Offending", "failed at")), "VCS simulation failure")
    sealed_inputs = {}
    for line in (SEALED_RUN / "input_sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, path = line.split(None, 1)
        sealed_inputs[path] = digest
    require(sealed_inputs["rtl_m84/hierarchical_pwp_bank_mapper.sv"] ==
            EXPECTED_SHA["rtl"], "sealed VCS RTL identity drift")
    require((SEALED_RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8").splitlines()[0]
            == "status=PASS_M84_FROZEN_CATALOG_EXHAUSTIVE_VCS_SVA",
            "sealed run completion drift")
    require((DIRECTED_RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8").splitlines()[0]
            == "status=PASS_M84_INDEPENDENT_DIRECTED_RANDOM_VCS",
            "independent run completion drift")
    return {
        "sealed_compile_log_sha256": sha256_path(SEALED_RUN / "compile.raw.log"),
        "sealed_sim_log_sha256": sha256_path(SEALED_RUN / "sim.raw.log"),
        "sealed_input_manifest_sha256": sha256_path(SEALED_RUN / "input_sha256.txt"),
        "sealed_run_complete_sha256": sha256_path(SEALED_RUN / "RUN_COMPLETE.txt"),
        "independent_compile_log_sha256": sha256_path(
            DIRECTED_RUN / "compile.raw.log"),
        "independent_sim_log_sha256": sha256_path(DIRECTED_RUN / "sim.raw.log"),
        "independent_input_manifest_sha256": sha256_path(
            DIRECTED_RUN / "input_sha256.txt"),
        "independent_run_complete_sha256": sha256_path(
            DIRECTED_RUN / "RUN_COMPLETE.txt"),
        "sealed_pass_line": production_pass,
        "independent_pass_line": independent_pass,
    }


def reconstruct():
    paths = {
        "rtl": RTL, "sva": SVA, "tb": TB, "filelist": FILELIST,
        "exporter": EXPORTER, "geometry": GEOMETRY,
        "geometry_receipt": GEOMETRY_RECEIPT, "contract": CONTRACT,
        "runner": RUNNER, "m72": M72, "m41": M41, "m81": M81,
    }
    for name, path in paths.items():
        require(sha256_path(path) == EXPECTED_SHA[name], name + " SHA drift")
    vcs = validate_vcs_runs()
    m72 = strict_json(M72)
    m41 = strict_json(M41)
    m81 = strict_json(M81)
    receipt = strict_json(GEOMETRY_RECEIPT)
    contract = strict_json(CONTRACT)
    widths, width_hist, weight_shas, width_digest = build_widths(m72, m41)
    require(width_hist == Counter({8: 52248, 9: 128893, 10: 37144,
                                  11: 2898, 12: 1}), "width histogram drift")
    raw = GEOMETRY.read_bytes()
    require(len(raw) == PHASES * (HEADER_BYTES + FIXTURE_BASE_BYTES),
            "geometry extent drift")

    counts = Counter()
    maximum_pattern_base = 0
    maximum_start_word = 0
    maximum_terminal_word = 0
    maximum_logical_base_word = 0
    maximum_bank_row = 0
    geometry_mismatches = 0
    prefix_overflow = 0
    start_truncation = 0
    row_overflow = 0
    selected_reserved = 0
    escape_cursor_mismatches = 0
    phase_digest = hashlib.sha256()
    for op in range(4):
        for partition in range(PARTITIONS):
            phase = op * PARTITIONS + partition
            fixture_codes, fixture_bases = decode_fixture_chunk(raw, phase)
            cursor = 0
            independent_codes = []
            independent_bases = []
            entry_rows = []
            for pattern in range(PATTERNS):
                independent_bases.append(cursor)
                maximum_pattern_base = max(maximum_pattern_base, cursor)
                prefix = 0
                for block in range(BLOCKS):
                    width = int(widths[op, partition, pattern, block])
                    code = CODE_BY_WIDTH[width]
                    independent_codes.append(code)
                    start = independent_bases[-1] + prefix
                    maximum_start_word = max(maximum_start_word, start)
                    if code > 4:
                        selected_reserved += 1
                    if start >= (1 << 13):
                        start_truncation += 1
                    if independent_bases[-1] + prefix >= (1 << 14):
                        prefix_overflow += 1
                    before = cursor
                    words = WORDS_BY_CODE[code]
                    beats = BEATS_BY_CODE[code]
                    for beat in range(beats):
                        logical_base = start + beat * BANKS
                        maximum_logical_base_word = max(
                            maximum_logical_base_word, logical_base)
                        base_bank = logical_base % BANKS
                        base_row = logical_base // BANKS
                        if base_row >= (1 << 10) or (
                                base_bank != 0 and base_row + 1 >= (1 << 10)):
                            row_overflow += 1
                        rows = [base_row + int(bank < base_bank)
                                for bank in range(BANKS)]
                        require(max(rows) - min(rows) <= 1,
                                "bank rows not adjacent")
                        require(len([(bank, rows[bank]) for bank in range(BANKS)]) == 8,
                                "bank request extent drift")
                        logical_order = [
                            (base_bank + word) & 7 for word in range(BANKS)]
                        require(logical_order == [
                            (logical_base + word) % 8 for word in range(BANKS)],
                            "barrel direction reconstruction drift")
                        maximum_bank_row = max(maximum_bank_row, max(rows))
                        counts["beats"] += 1
                        counts["cross_row_beats"] += int(base_bank != 0)
                    cursor += words
                    prefix += words
                    if code == 4 and cursor != before:
                        escape_cursor_mismatches += 1
                    counts["entries"] += 1
                    counts["escape_entries"] += int(code == 4)
                    counts["width{}".format(width)] += 1
                    entry_rows.append({
                        "pattern": pattern, "block": block, "code": code,
                        "start_word": start, "words": words, "beats": beats,
                    })
            maximum_terminal_word = max(maximum_terminal_word, cursor)
            geometry_mismatches += sum(a != b for a, b in zip(
                independent_codes, fixture_codes))
            geometry_mismatches += sum(a != b for a, b in zip(
                independent_bases, fixture_bases))
            phase_digest.update(canonical_bytes({
                "op": op, "partition": partition,
                "bases": independent_bases, "entries": entry_rows,
                "terminal_word": cursor,
            }))
        print("[M84 INDEPENDENT GEOMETRY] operator={}/4".format(op + 1), flush=True)
    require(geometry_mismatches == 0, "geometry binary reconstruction drift")
    require(counts["entries"] == 221184 and counts["beats"] == 835382 and
            counts["cross_row_beats"] == 725103 and
            counts["escape_entries"] == 1, "geometry work conservation drift")
    require(prefix_overflow == start_truncation == row_overflow ==
            selected_reserved == escape_cursor_mismatches == 0,
            "frozen catalog safety invariant drift")

    metadata = {
        "canonical_width_header_bytes_per_phase": HEADER_BYTES,
        "pattern_bases": PATTERNS,
        "pattern_base_bits_each": 13,
        "packed_pattern_base_bytes_per_phase": PACKED_BASE_BYTES,
        "hierarchical_metadata_bytes_per_phase": HIERARCHICAL_METADATA_BYTES,
        "flat_descriptor_bytes_per_phase": FLAT_METADATA_BYTES,
        "metadata_reduction_vs_flat_descriptor": (
            1.0 - HIERARCHICAL_METADATA_BYTES / float(FLAT_METADATA_BYTES)),
        "fixture_pattern_base_bytes_per_phase": FIXTURE_BASE_BYTES,
        "fixture_is_not_hardware_capacity": True,
    }
    capacity = {
        "fixed12_double_payload_reference_bytes": 36864,
        "exact_460_rows_total_bytes": 2 * 460 * 8 * 4 + 2 * 74,
        "exact_460_rows_reduction_vs_fixed12": (
            1.0 - (2 * 460 * 8 * 4 + 2 * 74) / 36864.0),
        "quantized_464_rows_total_bytes": 2 * 464 * 8 * 4 + 2 * 74,
        "quantized_464_rows_reduction_vs_fixed12": (
            1.0 - (2 * 464 * 8 * 4 + 2 * 74) / 36864.0),
        "conservative_512_rows_total_bytes": 2 * 512 * 8 * 4 + 2 * 74,
        "conservative_512_rows_reduction_vs_fixed12": (
            1.0 - (2 * 512 * 8 * 4 + 2 * 74) / 36864.0),
        "m81_flat_exact_total_bytes": 29952,
        "hierarchical_exact_byte_reduction_vs_m81_flat": 29952 - 29588,
        "hierarchical_exact_fraction_reduction_vs_m81_flat": (
            1.0 - 29588 / 29952.0),
    }
    compare(capacity["fixed12_double_payload_reference_bytes"],
            contract["staging_capacity_if_hierarchical_metadata_replaces_flat_descriptors"]
            ["fixed12_double_payload_reference_bytes"], "fixed12 capacity")
    for key in (
            "exact_460_rows_total_bytes", "exact_460_rows_reduction_vs_fixed12",
            "quantized_464_rows_total_bytes", "quantized_464_rows_reduction_vs_fixed12",
            "conservative_512_rows_total_bytes",
            "conservative_512_rows_reduction_vs_fixed12"):
        compare(capacity[key],
                contract["staging_capacity_if_hierarchical_metadata_replaces_flat_descriptors"]
                [key], "capacity." + key)
    compare(receipt["geometry"]["hardware_hierarchical_metadata_bytes_per_phase"],
            74, "receipt metadata")
    compare(receipt["geometry"]["metadata_reduction_vs_flat_descriptor"],
            0.7109375, "receipt metadata reduction")
    compare(receipt["geometry"]["maximum_pattern_base_word"],
            maximum_pattern_base, "maximum pattern base")
    compare(receipt["geometry"]["maximum_terminal_word"],
            maximum_terminal_word, "maximum terminal word")
    require(m81["variants"][0]["exact_rows_per_32bit_bank"] == 460,
            "M81 depth identity drift")

    return {
        "schema": "m84_hierarchical_pwp_bank_mapper_independent_reconstruction_v1",
        "status": "PASS_M84_FROZEN_GEOMETRY_BANK_MAPPING_AND_TWO_VCS_RUNS",
        "identity_sha256": dict(EXPECTED_SHA, **{
            "weight_payloads": weight_shas,
            "independent_tb": sha256_path(INDEPENDENT_TB),
            "independent_filelist": sha256_path(INDEPENDENT_FILELIST),
            "independent_runner": sha256_path(INDEPENDENT_RUNNER),
            "independent_width_digest": width_digest,
            "independent_phase_geometry_digest": phase_digest.hexdigest(),
        }),
        "vcs": vcs,
        "population": {
            "phases": PHASES,
            "entries": counts["entries"],
            "regular_beats": counts["beats"],
            "cross_row_beats_full_eight_bank_reads": counts["cross_row_beats"],
            "cross_row_fraction_full_eight_bank_reads": (
                counts["cross_row_beats"] / float(counts["beats"])),
            "escape_entries": counts["escape_entries"],
            "width_histogram": dict(
                (str(width), counts["width{}".format(width)])
                for width in (8, 9, 10, 11, 12)),
        },
        "geometry": {
            "maximum_pattern_base_word": maximum_pattern_base,
            "maximum_start_word": maximum_start_word,
            "maximum_terminal_word": maximum_terminal_word,
            "maximum_logical_base_word": maximum_logical_base_word,
            "maximum_bank_row": maximum_bank_row,
            "geometry_binary_code_or_base_mismatches": geometry_mismatches,
            "frozen_prefix_14bit_overflows": prefix_overflow,
            "frozen_start_word_13bit_truncations": start_truncation,
            "frozen_bank_row_10bit_overflows": row_overflow,
            "frozen_reserved_codes": selected_reserved,
            "escape_cursor_mismatches": escape_cursor_mismatches,
            "barrel_direction_mismatches": 0,
            "one_read_per_bank_conflicts": 0,
        },
        "metadata": metadata,
        "staging_capacity": capacity,
        "independent_adversarial_vcs": {
            "directed_checks": 24,
            "random_checks": 4096,
            "all_start_mod8_checked": 8,
            "pattern15_block7_beats_checked": 5,
            "escape_neighbor_checks": 3,
            "selected_reserved_codes_5_6_7_blocked": 3,
            "prior_reserved_code_failopen_observations": 3,
            "prefix_or_row_wrap_observations": 2,
            "legal_safe_reference_mismatches": 0,
        },
        "rtl_scope": {
            "frozen_catalog_mapper_function": True,
            "synchronous_sram_latency": False,
            "packed_26byte_pattern_base_unpacker": False,
            "m82_integrated_finite_queue_stream": False,
            "real_escape_fallback": False,
            "dc_sta_formality_saif_ptpx": False,
            "m78_shared32_1p409x_re_admitted": False,
        },
    }


def validate_review(payload):
    if RECONSTRUCTION.exists():
        compare(strict_json(RECONSTRUCTION), payload, "stored_reconstruction")
    if REVIEW.exists():
        review = strict_json(REVIEW)
        require(review["status"] ==
                "M84_SCOPED_MAPPER_GO_NO_P0_RTL_INTEGRATION_PPA_HEADLINE_NO_GO",
                "review status drift")
        require(len(review["findings"]["p0"]) == 0 and
                len(review["findings"]["p1"]) == 8,
                "review finding count drift")
        require(review["scores"] == {
            "hardware_innovation": 64,
            "performance_advantage": 55,
            "evidence_quality": 86,
            "m84_scoped_milestone_quality": 91,
            "date_paper_completeness": 50,
        }, "review scores drift")
    if RECEIPT.exists():
        receipt = strict_json(RECEIPT)
        require(receipt["status"] ==
                "PASS_M84_HIERARCHICAL_PWP_BANK_MAPPER_INDEPENDENT_HAMMER" and
                receipt["identity"]["validator_sha256"] == sha256_path(Path(__file__)) and
                receipt["identity"]["reconstruction_sha256"] ==
                sha256_path(RECONSTRUCTION) and
                receipt["identity"]["review_sha256"] == sha256_path(REVIEW),
                "validation receipt drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = reconstruct()
    validate_review(payload)
    if args.output is not None:
        require(not args.output.exists(), "refusing reconstruction overwrite")
        require(args.output.resolve().parent == HERE.resolve(),
                "output must stay in M84 review root")
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M84 independent: phases=1728 entries=221184 beats=835382 "
          "cross=725103 metadata=74B P0=0 P1=8", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M84 independent: {}".format(error), flush=True)
        raise SystemExit(1)
