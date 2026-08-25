#!/usr/bin/env python3
"""Export frozen cap11 headers and per-pattern bases for exhaustive M84 VCS."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M78_ANALYZER = HW / "system_simulator/scripts/analyze_m78_precision_elastic_pwp.py"
M78_RESULT = HW / (
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/"
    "m78_precision_elastic_pwp.json")
M81_RESULT = HW / (
    "results/m81_interleaved_word_pwp_buffer_valid825_internal_dev_r2_20260823/"
    "m81_interleaved_word_pwp_buffer.json")
EXPECTED_SHA256 = {
    "m78_analyzer": "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b",
    "m78_result": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m81_result": "515e023421a2650077b61fb620b06428786eb180dd3d36d05eccec1c8d2fabad",
}
CODE_BY_WIDTH = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4}
PHASES = 1728
PATTERNS = 16
BLOCKS = 8
HEADER_BYTES = 48
PATTERN_BASE_PACKED_BYTES = PATTERNS * 13 // 8
PATTERN_BASE_FIXTURE_BYTES = PATTERNS * 2
WORDS_BY_WIDTH = {8: 24, 9: 27, 10: 30, 11: 33, 12: 0}


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
    spec = importlib.util.spec_from_file_location("m84_m78", str(M78_ANALYZER))
    require(spec is not None and spec.loader is not None, "cannot import M78")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_header(codes):
    require(len(codes) == PATTERNS * BLOCKS, "M84 header extent drift")
    packed = sum(code << (3 * index) for index, code in enumerate(codes))
    return packed.to_bytes(HEADER_BYTES, "little")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M84 output overwrite")
    start_sha = sha256(Path(__file__).resolve())
    for name, path in (("m78_analyzer", M78_ANALYZER),
                       ("m78_result", M78_RESULT),
                       ("m81_result", M81_RESULT)):
        require(sha256(path) == EXPECTED_SHA256[name],
                "M84 input identity drift: " + name)

    m78 = load_m78()
    m72 = m78.strict_json(m78.M72_RESULT)
    m41 = m78.strict_json(m78.M41_RESULT)
    catalog, _, _, outliers, weight_shas = m78.build_width_catalog(m72, m41)
    require(len(outliers) == 1, "M84 escape extent drift")
    args.output_dir.mkdir(parents=True)
    binary_path = args.output_dir / "m84_phase_geometry.bin"
    counts = Counter()
    maximum_pattern_base = 0
    maximum_terminal_word = 0
    with binary_path.open("wb") as handle:
        for op in range(4):
            for partition in range(432):
                cursor = 0
                bases = []
                codes = []
                for pattern in range(PATTERNS):
                    bases.append(cursor)
                    maximum_pattern_base = max(maximum_pattern_base, cursor)
                    for block in range(BLOCKS):
                        width = catalog[op][partition][pattern]["blocks"][block]["width"]
                        require(width in CODE_BY_WIDTH, "M84 unexpected width")
                        codes.append(CODE_BY_WIDTH[width])
                        counts["width{}".format(width)] += 1
                        cursor += WORDS_BY_WIDTH[width]
                require(cursor < (1 << 13), "M84 terminal word overflow")
                maximum_terminal_word = max(maximum_terminal_word, cursor)
                handle.write(encode_header(codes))
                for base in bases:
                    handle.write(struct.pack("<H", base))
                counts["phases"] += 1
                counts["entries"] += len(codes)

    require(counts["phases"] == PHASES and
            counts["entries"] == PHASES * PATTERNS * BLOCKS,
            "M84 phase/entry conservation drift")
    require(counts["width12"] == 1, "M84 unique escape drift")
    require(binary_path.stat().st_size ==
            PHASES * (HEADER_BYTES + PATTERN_BASE_FIXTURE_BYTES),
            "M84 geometry binary extent drift")
    require(sha256(Path(__file__).resolve()) == start_sha,
            "M84 exporter source changed during run")
    receipt = {
        "schema": "m84_hierarchical_pwp_geometry_export_v1",
        "status": "PASS_M84_FROZEN_CAP11_HIERARCHICAL_GEOMETRY",
        "identity": {
            "exporter_start_end_sha256": start_sha,
            "m78_analyzer_sha256": sha256(M78_ANALYZER),
            "m78_result_sha256": sha256(M78_RESULT),
            "m81_result_sha256": sha256(M81_RESULT),
            "weight_payload_sha256": weight_shas,
        },
        "geometry": {
            "phases": counts["phases"],
            "entries": counts["entries"],
            "width_histogram": {str(width): counts["width{}".format(width)]
                                for width in CODE_BY_WIDTH},
            "header_bytes_per_phase": HEADER_BYTES,
            "pattern_base_packed_bytes_per_phase": PATTERN_BASE_PACKED_BYTES,
            "fixture_uint16_pattern_base_bytes_per_phase": (
                PATTERN_BASE_FIXTURE_BYTES),
            "hardware_hierarchical_metadata_bytes_per_phase": (
                HEADER_BYTES + PATTERN_BASE_PACKED_BYTES),
            "flat_descriptor_bytes_per_phase": 256,
            "metadata_reduction_vs_flat_descriptor": 1.0 -
                (HEADER_BYTES + PATTERN_BASE_PACKED_BYTES) / 256.0,
            "maximum_pattern_base_word": maximum_pattern_base,
            "maximum_terminal_word": maximum_terminal_word,
        },
        "file": {
            "name": binary_path.name,
            "bytes": binary_path.stat().st_size,
            "sha256": sha256(binary_path),
        },
        "admission": {
            "frozen_catalog_geometry": True,
            "vcs_exhaustive_mapping": False,
            "sram_macro_ppa": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    receipt_path = args.output_dir / "m84_phase_geometry_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print("PASS M84 geometry phases={} entries={} escape={} sha={}".format(
        counts["phases"], counts["entries"], counts["width12"],
        receipt["file"]["sha256"]), flush=True)


if __name__ == "__main__":
    main()
