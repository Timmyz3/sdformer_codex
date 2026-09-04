#!/usr/bin/env python3
"""Export and round-trip the canonical M83 cap11 PWP phase records.

Canonical serialization:
* phase order: operator-major, then partition-major;
* entry order: pattern-major, then output-block-major;
* header: 128 3-bit codes packed LSB-first into 48 bytes;
* eligible payload: lane-major signed two's-complement, lane 0 in the least
  significant bits, byte stream little-endian;
* width-12 code is an exact bit-sparse escape and has no payload;
* every phase record is zero padded once to a 32-byte boundary;
* the 1,729 phase-boundary offsets are uint32 little-endian.

The exporter re-decodes every header and all 221,183 stored PWP vectors and
compares all 96 lanes to exact checkpoint INT8 sums before admitting PASS.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M78_ANALYZER = HW / "system_simulator/scripts/analyze_m78_precision_elastic_pwp.py"
M78_RESULT = HW / (
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/"
    "m78_precision_elastic_pwp.json")
M81_R2_RESULT = HW / (
    "results/m81_interleaved_word_pwp_buffer_valid825_internal_dev_r2_20260823/"
    "m81_interleaved_word_pwp_buffer.json")
EXPECTED_SHA256 = {
    "m78_analyzer": "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b",
    "m78_result": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m81_r2_result": "515e023421a2650077b61fb620b06428786eb180dd3d36d05eccec1c8d2fabad",
}
CODE_BY_WIDTH = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4}
WIDTH_BY_CODE = dict((value, key) for key, value in CODE_BY_WIDTH.items())
PHASES = 1728
ENTRIES = 128
LANES = 96
HEADER_BYTES = 48
ALIGNMENT = 32


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
    spec = importlib.util.spec_from_file_location("m83_m78", str(M78_ANALYZER))
    require(spec is not None and spec.loader is not None, "cannot import M78")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_header(codes):
    require(len(codes) == ENTRIES, "M83 header entry extent drift")
    packed = 0
    for index, code in enumerate(codes):
        require(0 <= code < 8, "M83 header code exceeds three bits")
        packed |= code << (3 * index)
    return packed.to_bytes(HEADER_BYTES, byteorder="little", signed=False)


def decode_header(payload):
    require(len(payload) == HEADER_BYTES, "M83 header byte extent drift")
    packed = int.from_bytes(payload, byteorder="little", signed=False)
    return [(packed >> (3 * index)) & 7 for index in range(ENTRIES)]


def encode_lanes(values, width):
    require(len(values) == LANES, "M83 PWP lane extent drift")
    mask = (1 << width) - 1
    packed = 0
    for lane, value in enumerate(values):
        integer = int(value)
        require(-(1 << (width - 1)) <= integer < (1 << (width - 1)),
                "M83 signed PWP does not fit frozen width")
        packed |= (integer & mask) << (lane * width)
    return packed.to_bytes(width * LANES // 8,
                           byteorder="little", signed=False)


def decode_lanes(payload, width):
    require(len(payload) == width * LANES // 8,
            "M83 encoded PWP byte extent drift")
    packed = int.from_bytes(payload, byteorder="little", signed=False)
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    values = []
    for lane in range(LANES):
        value = (packed >> (lane * width)) & mask
        if value & sign:
            value -= 1 << width
        values.append(value)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M83 output-dir overwrite")
    analyzer_start_sha = sha256(Path(__file__).resolve())
    for name, path in (("m78_analyzer", M78_ANALYZER),
                       ("m78_result", M78_RESULT),
                       ("m81_r2_result", M81_R2_RESULT)):
        require(sha256(path) == EXPECTED_SHA256[name],
                "M83 input identity drift: " + name)
    m78 = load_m78()
    m72_result = m78.strict_json(m78.M72_RESULT)
    m41_result = m78.strict_json(m78.M41_RESULT)
    width_catalog, width_hist, center_max_hist, outliers, weight_shas = (
        m78.build_width_catalog(m72_result, m41_result))
    require(len(outliers) == 1, "M83 unique width12 escape drift")

    args.output_dir.mkdir(parents=True)
    records_path = args.output_dir / "m83_cap11_phase_records.bin"
    offsets_path = args.output_dir / "m83_cap11_phase_offsets_u32le.bin"
    receipt_path = args.output_dir / "m83_canonical_cap11_pwp_records_receipt.json"
    status_path = args.output_dir / "RUN_INCOMPLETE_DO_NOT_CITE.txt"
    status_path.write_text("M83 export in progress; do not cite\n", encoding="utf-8")

    layers = m41_result["layers"]
    weights_by_operator = []
    for op, operator in enumerate(m72_result["operators"]):
        layer = next(item for item in layers
                     if item["operator"] == operator["operator"])
        info = next(item for item in layer["payloads"] if item["role"] == "weight")
        path = m78.M41_DIR / info["file"]
        require(sha256(path) == weight_shas[op] == info["sha256"],
                "M83 weight identity drift")
        weights = np.fromfile(str(path), dtype=np.int8)
        require(weights.size == 6912 * 768, "M83 weight extent drift")
        weights_by_operator.append(weights.reshape(6912, 768).astype(np.int32))

    offsets = [0]
    totals = Counter()
    phase_lengths = []
    with records_path.open("wb") as records:
        for op, operator in enumerate(m72_result["operators"]):
            weights = weights_by_operator[op]
            for partition, partition_row in enumerate(operator["partitions"]):
                source = weights[partition * 16:(partition + 1) * 16]
                codes = []
                encoded_entries = []
                exact_entries = []
                for pattern_index, center_hex in enumerate(partition_row["centers_hex"]):
                    center = int(center_hex, 16)
                    indices = [bit for bit in range(16) if center & (1 << bit)]
                    pwp = source[indices].sum(axis=0, dtype=np.int32)
                    for block in range(8):
                        values = pwp[block * LANES:(block + 1) * LANES]
                        width = width_catalog[op][partition][pattern_index]["blocks"][block]["width"]
                        codes.append(CODE_BY_WIDTH[width])
                        exact_entries.append((width, [int(item) for item in values]))
                        if width <= 11:
                            encoded_entries.append(encode_lanes(values, width))
                            totals["stored_entries"] += 1
                            totals["payload_bytes"] += width * LANES // 8
                        else:
                            encoded_entries.append(b"")
                            totals["escape_entries"] += 1
                header = encode_header(codes)
                raw = header + b"".join(encoded_entries)
                padding = (-len(raw)) % ALIGNMENT
                record = raw + bytes(padding)

                decoded_codes = decode_header(record[:HEADER_BYTES])
                require(decoded_codes == codes,
                        "M83 canonical header round-trip mismatch")
                cursor = HEADER_BYTES
                for entry, (width, expected) in enumerate(exact_entries):
                    decoded_width = WIDTH_BY_CODE.get(decoded_codes[entry])
                    require(decoded_width == width,
                            "M83 header width round-trip mismatch")
                    if width <= 11:
                        count = width * LANES // 8
                        observed = decode_lanes(record[cursor:cursor + count], width)
                        require(observed == expected,
                                "M83 signed lane round-trip mismatch")
                        cursor += count
                        totals["roundtrip_lanes"] += LANES
                    else:
                        totals["roundtrip_escapes"] += 1
                require(cursor == len(raw), "M83 record payload cursor mismatch")
                require(not any(record[cursor:]), "M83 record padding is nonzero")
                records.write(record)
                offsets.append(offsets[-1] + len(record))
                phase_lengths.append(len(record))
                totals["phases"] += 1
                totals["header_bytes"] += HEADER_BYTES
                totals["padding_bytes"] += padding
            print("[M83 EXPORT] operator={}/4 bytes={}".format(
                op + 1, offsets[-1]), flush=True)

    require(totals["phases"] == PHASES and len(offsets) == PHASES + 1,
            "M83 phase/offset extent drift")
    require(totals["stored_entries"] == 221183 and
            totals["escape_entries"] == 1,
            "M83 stored/escape entry conservation drift")
    require(totals["payload_bytes"] == 23776068,
            "M83 payload byte conservation drift")
    require(offsets[-1] == records_path.stat().st_size,
            "M83 terminal offset/file size mismatch")
    with offsets_path.open("wb") as handle:
        for offset in offsets:
            require(0 <= offset < (1 << 32), "M83 uint32 offset overflow")
            handle.write(struct.pack("<I", offset))
    require(offsets_path.stat().st_size == (PHASES + 1) * 4,
            "M83 offset table extent drift")
    require(sha256(Path(__file__).resolve()) == analyzer_start_sha,
            "M83 exporter source changed during execution")

    receipt = {
        "schema": "m83_canonical_cap11_pwp_phase_record_export_receipt_v1",
        "status": "PASS_M83_CANONICAL_BINARY_ALL_ENTRIES_ROUNDTRIP",
        "identity": {
            "exporter_start_end_sha256": analyzer_start_sha,
            "m78_analyzer_sha256": sha256(M78_ANALYZER),
            "m78_result_sha256": sha256(M78_RESULT),
            "m81_r2_result_sha256": sha256(M81_R2_RESULT),
            "weight_payload_sha256": weight_shas,
        },
        "canonical_serialization": {
            "phase_order": "operator-major then partition-major",
            "entry_order": "pattern-major then output-block-major",
            "header": "128 three-bit codes, entry0 at bit0, LSB-first, 48 bytes",
            "header_codes": {"0": "signed8", "1": "signed9", "2": "signed10", "3": "signed11", "4": "bit_sparse_escape"},
            "payload": "lane-major signed two's-complement, lane0 least-significant, little-endian bytes",
            "phase_padding": "zero bytes to the next 32-byte boundary",
            "offsets": "1729 uint32 little-endian phase-boundary byte offsets",
        },
        "files": {
            "phase_records": {
                "file": records_path.name,
                "bytes": records_path.stat().st_size,
                "sha256": sha256(records_path),
            },
            "phase_offsets": {
                "file": offsets_path.name,
                "bytes": offsets_path.stat().st_size,
                "sha256": sha256(offsets_path),
            },
        },
        "conservation": {
            "phases": totals["phases"],
            "stored_pwp_entries": totals["stored_entries"],
            "escape_entries": totals["escape_entries"],
            "roundtrip_signed_lanes": totals["roundtrip_lanes"],
            "roundtrip_escapes": totals["roundtrip_escapes"],
            "payload_bytes": totals["payload_bytes"],
            "header_bytes": totals["header_bytes"],
            "padding_bytes": totals["padding_bytes"],
            "record_bytes": records_path.stat().st_size,
            "record_length_minimum": min(phase_lengths),
            "record_length_maximum": max(phase_lengths),
            "numeric_or_serialization_mismatches": 0,
        },
        "admission": {
            "canonical_byte_serialization": True,
            "all_entry_roundtrip": True,
            "binary_payload_sha": True,
            "rtl_reader_or_bank_integration": False,
            "sram_macro_ppa_or_energy": False,
            "accuracy": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    status_path.write_text(
        "status=PASS_M83_CANONICAL_BINARY_ALL_ENTRIES_ROUNDTRIP\n"
        "system_speedup=false\nheadline=false\n",
        encoding="utf-8")
    status_path.rename(args.output_dir / "RUN_COMPLETE.txt")
    print("PASS M83 records={} offsets={} lanes={}".format(
        receipt["files"]["phase_records"]["sha256"],
        receipt["files"]["phase_offsets"]["sha256"],
        totals["roundtrip_lanes"]), flush=True)


if __name__ == "__main__":
    main()
