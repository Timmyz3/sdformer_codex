#!/usr/bin/env python3
"""Independently decode all M83 records without importing M78/M83 code."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


EXPECTED = {
    "records_sha256": "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d",
    "offsets_sha256": "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c",
    "m72_result_sha256": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41_result_sha256": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "weight_sha256": (
        "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
        "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
        "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
        "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
    ),
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_width(minimum, maximum):
    for width in range(1, 33):
        if (minimum >= -(1 << (width - 1)) and
                maximum <= (1 << (width - 1)) - 1):
            return max(8, width)
    raise AssertionError("PWP exceeds signed32")


def decode_cross_byte_header_code(header, index):
    bit = 3 * index
    byte = bit // 8
    shift = bit % 8
    raw = header[byte]
    if shift > 5:
        raw |= header[byte + 1] << 8
    return (raw >> shift) & 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    hw = args.repo_root.resolve() / "hw_autoresearch_nts07"
    output = hw / "results/m83_canonical_cap11_pwp_records_r1_20260823"
    records_path = output / "m83_cap11_phase_records.bin"
    offsets_path = output / "m83_cap11_phase_offsets_u32le.bin"
    m72_path = hw / (
        "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
        "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
    m41_dir = hw / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
    m41_path = m41_dir / "m41_h67_ep35_bottleneck_int8_bridge.json"
    require(sha256(records_path) == EXPECTED["records_sha256"],
            "record SHA mismatch")
    require(sha256(offsets_path) == EXPECTED["offsets_sha256"],
            "offset SHA mismatch")
    require(sha256(m72_path) == EXPECTED["m72_result_sha256"],
            "M72 identity mismatch")
    require(sha256(m41_path) == EXPECTED["m41_result_sha256"],
            "M41 identity mismatch")

    records = records_path.read_bytes()
    offset_blob = offsets_path.read_bytes()
    require(len(records) == 23884000, "record extent mismatch")
    require(len(offset_blob) == 6916, "offset extent mismatch")
    offsets = list(struct.unpack("<1729I", offset_blob))
    require(offsets[0] == 0 and offsets[-1] == len(records),
            "offset boundary mismatch")
    require(all(left < right for left, right in zip(offsets, offsets[1:])),
            "offsets not strictly monotonic")
    require(all(offset % 32 == 0 for offset in offsets),
            "phase boundary not 32-byte aligned")
    lengths = [right - left for left, right in zip(offsets, offsets[1:])]
    require(min(lengths) == 12832 and max(lengths) == 14784,
            "phase record length extrema drift")

    m72 = json.loads(m72_path.read_text(encoding="utf-8"))
    m41 = json.loads(m41_path.read_text(encoding="utf-8"))
    weights = []
    weight_shas = []
    for operator_index, operator in enumerate(m72["operators"]):
        layer = next(row for row in m41["layers"]
                     if row["operator"] == operator["operator"])
        info = next(row for row in layer["payloads"]
                    if row["role"] == "weight")
        path = m41_dir / info["file"]
        observed_sha = sha256(path)
        require(observed_sha == info["sha256"] ==
                EXPECTED["weight_sha256"][operator_index],
                "weight identity mismatch")
        weight_shas.append(observed_sha)
        array = np.fromfile(str(path), dtype=np.int8)
        require(array.size == 6912 * 768, "weight extent mismatch")
        weights.append(array.reshape(6912, 768).astype(np.int32))

    histogram = dict((width, 0) for width in range(8, 13))
    negative = dict((width, 0) for width in range(8, 12))
    positive = dict((width, 0) for width in range(8, 12))
    minima = dict((width, None) for width in range(8, 12))
    maxima = dict((width, None) for width in range(8, 12))
    phase = 0
    stored_entries = 0
    roundtrip_lanes = 0
    payload_bytes = 0
    padding_bytes = 0
    cross_byte_codes = 0
    escapes = []
    code_by_width = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4}

    for operator_index, operator in enumerate(m72["operators"]):
        weight = weights[operator_index]
        for partition, row in enumerate(operator["partitions"]):
            record = records[offsets[phase]:offsets[phase + 1]]
            require(len(record) % 32 == 0, "record alignment drift")
            header = record[:48]
            cursor = 48
            entry = 0
            source = weight[partition * 16:(partition + 1) * 16]
            for pattern_index, center_hex in enumerate(row["centers_hex"]):
                center = int(center_hex, 16)
                indices = [bit for bit in range(16)
                           if center & (1 << bit)]
                pwp = source[indices].sum(axis=0, dtype=np.int32)
                for block in range(8):
                    expected = pwp[block * 96:(block + 1) * 96]
                    width = signed_width(int(expected.min()),
                                         int(expected.max()))
                    code = decode_cross_byte_header_code(header, entry)
                    require(code == code_by_width[width],
                            "header width mismatch")
                    require(code not in (5, 6, 7), "reserved code observed")
                    if (3 * entry) % 8 > 5:
                        cross_byte_codes += 1
                    histogram[width] += 1
                    if width <= 11:
                        count = width * 12
                        blob = record[cursor:cursor + count]
                        require(len(blob) == count, "payload truncation")
                        packed = int.from_bytes(blob, "little", signed=False)
                        mask = (1 << width) - 1
                        sign = 1 << (width - 1)
                        observed = []
                        for lane in range(96):
                            value = (packed >> (lane * width)) & mask
                            if value & sign:
                                value -= 1 << width
                            observed.append(value)
                        expected_list = [int(value) for value in expected]
                        require(observed == expected_list,
                                "signed lane mismatch")
                        negative[width] += sum(value < 0 for value in observed)
                        positive[width] += sum(value > 0 for value in observed)
                        low, high = min(observed), max(observed)
                        minima[width] = low if minima[width] is None else min(
                            minima[width], low)
                        maxima[width] = high if maxima[width] is None else max(
                            maxima[width], high)
                        cursor += count
                        payload_bytes += count
                        stored_entries += 1
                        roundtrip_lanes += 96
                    else:
                        escapes.append({
                            "operator_index": operator_index,
                            "partition": partition,
                            "pattern_index": pattern_index,
                            "output_block": block,
                            "minimum": int(expected.min()),
                            "maximum": int(expected.max()),
                            "header_entry": entry,
                            "header_code_crosses_byte": (3 * entry) % 8 > 5,
                        })
                    entry += 1
            require(entry == 128, "entry extent mismatch")
            require(not any(record[cursor:]), "nonzero phase padding")
            padding_bytes += len(record) - cursor
            phase += 1

    require(phase == 1728 and stored_entries == 221183,
            "phase/stored-entry conservation mismatch")
    require(roundtrip_lanes == 21233568, "lane conservation mismatch")
    require(histogram == {8: 52248, 9: 128893, 10: 37144,
                          11: 2898, 12: 1}, "width histogram mismatch")
    require(payload_bytes == 23776068 and padding_bytes == 24988,
            "payload/padding conservation mismatch")
    require(len(escapes) == 1, "unique escape mismatch")
    require(all(negative[width] > 0 and positive[width] > 0
                for width in range(8, 12)), "signed-domain coverage missing")

    result = {
        "schema": "m83_remote_independent_full_decode_v1",
        "status": "PASS_REMOTE_INDEPENDENT_FULL_DECODE",
        "records": {"bytes": len(records), "sha256": sha256(records_path)},
        "offsets": {
            "bytes": len(offset_blob),
            "sha256": sha256(offsets_path),
            "entries": len(offsets),
            "first": offsets[0],
            "last": offsets[-1],
            "strictly_monotonic": True,
            "all_phase_boundaries_32B_aligned": True,
            "minimum_delta": min(lengths),
            "maximum_delta": max(lengths),
        },
        "decode": {
            "phases": phase,
            "stored_entries": stored_entries,
            "roundtrip_lanes": roundtrip_lanes,
            "cross_byte_header_codes_checked": cross_byte_codes,
            "width_histogram": histogram,
            "payload_bytes": payload_bytes,
            "zero_padding_bytes": padding_bytes,
            "mismatches": 0,
        },
        "signed_domain": {
            "negative_lane_counts": negative,
            "positive_lane_counts": positive,
            "minimum_by_width": minima,
            "maximum_by_width": maxima,
        },
        "unique_escape": escapes[0],
        "weight_payload_sha256": weight_shas,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
