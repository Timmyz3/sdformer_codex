#!/usr/bin/env python3
"""Repack the frozen M84 uint16 fixture into the canonical 74-byte RTL image."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M84_GEOMETRY = HW / (
    "results/m84_hierarchical_pwp_geometry_r2_20260823/"
    "m84_phase_geometry.bin")
M84_RECEIPT = HW / (
    "results/m84_hierarchical_pwp_geometry_r2_20260823/"
    "m84_phase_geometry_receipt.json")
EXPECTED = {
    "geometry": "294ea28b95ca2ef5c4adcb77195aabd388fc5b0ebc16bc4a3affcb8800b18e5d",
    "receipt": "bcebdbe95e7f455add38cbf9781c804edaa923fe83a726d755ec20b45b6b8df1",
}
PHASES = 1728
HEADER_BYTES = 48
FIXTURE_BASE_BYTES = 32
PACKED_BASE_BYTES = 26


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def codes_from_header(header):
    value = int.from_bytes(header, "little")
    return [(value >> (3 * entry)) & 7 for entry in range(128)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M85 repack overwrite")
    source_sha = sha256(Path(__file__).resolve())
    require(sha256(M84_GEOMETRY) == EXPECTED["geometry"],
            "M85 M84 geometry identity drift")
    require(sha256(M84_RECEIPT) == EXPECTED["receipt"],
            "M85 M84 receipt identity drift")
    raw = M84_GEOMETRY.read_bytes()
    require(len(raw) == PHASES * (HEADER_BYTES + FIXTURE_BASE_BYTES),
            "M85 M84 fixture extent drift")
    args.output_dir.mkdir(parents=True)
    output = bytearray()
    max_terminal = 0
    escape_count = 0
    for phase in range(PHASES):
        begin = phase * (HEADER_BYTES + FIXTURE_BASE_BYTES)
        row = raw[begin:begin + HEADER_BYTES + FIXTURE_BASE_BYTES]
        header = row[:HEADER_BYTES]
        bases = [int.from_bytes(row[HEADER_BYTES + 2*pattern:
                                    HEADER_BYTES + 2*pattern + 2], "little")
                 for pattern in range(16)]
        codes = codes_from_header(header)
        cursor = 0
        for pattern in range(16):
            require(bases[pattern] == cursor,
                    "M85 pattern-base reconstruction mismatch")
            for block in range(8):
                code = codes[pattern*8 + block]
                require(code <= 4, "M85 reserved frozen code")
                if code == 4:
                    escape_count += 1
                else:
                    cursor += 24 + 3 * code
        require(cursor < 8192, "M85 phase terminal overflow")
        max_terminal = max(max_terminal, cursor)
        packed_bases = sum(base << (13 * pattern)
                           for pattern, base in enumerate(bases))
        output.extend(header)
        output.extend(packed_bases.to_bytes(PACKED_BASE_BYTES, "little"))
    require(escape_count == 1, "M85 escape extent drift")
    require(len(output) == PHASES * (HEADER_BYTES + PACKED_BASE_BYTES),
            "M85 packed metadata extent drift")
    args.output_dir.joinpath("m85_phase_metadata_74b.bin").write_bytes(output)
    require(sha256(Path(__file__).resolve()) == source_sha,
            "M85 repacker source changed during run")
    output_path = args.output_dir / "m85_phase_metadata_74b.bin"
    receipt = {
        "schema": "m85_canonical_74b_phase_metadata_repack_v1",
        "status": "PASS_M85_CANONICAL_48B_HEADER_PLUS_26B_BASES",
        "identity": {
            "repacker_start_end_sha256": source_sha,
            "m84_geometry_sha256": sha256(M84_GEOMETRY),
            "m84_receipt_sha256": sha256(M84_RECEIPT),
        },
        "geometry": {
            "phases": PHASES,
            "bytes_per_phase": HEADER_BYTES + PACKED_BASE_BYTES,
            "header_bytes_per_phase": HEADER_BYTES,
            "packed_base_bytes_per_phase": PACKED_BASE_BYTES,
            "escape_entries": escape_count,
            "maximum_terminal_word": max_terminal,
        },
        "file": {
            "name": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "admission": {
            "canonical_packed_metadata": True,
            "rtl_decode": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    args.output_dir.joinpath("m85_phase_metadata_74b_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M85 metadata phases={} bytes={} sha={}".format(
        PHASES, output_path.stat().st_size, receipt["file"]["sha256"]),
        flush=True)


if __name__ == "__main__":
    main()
