#!/usr/bin/env python3
"""Build the exact M430 static-codec population for M433 VCS replay."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


EXPECTED = {
    "catalog": "3ff522ff2296a021b005ca5733d846cc169560c125c8713c814b22a14d372f78",
    "m430_result": "6cf413e93d8159d9516ad048eaa26c741e49c2c9a3b330fb1d6dd20ba64dab2a",
    "m430_seal": "462501b849f42f1a0690d2fe8dbe3dc226e83ae05dea86f7cb0396d60e9faf7e",
    "weight_o0": "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "weight_o1": "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "weight_o2": "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "weight_o3": "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def packed_integer(values, width):
    result = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        result |= (int(value) & mask) << (lane * width)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.hardware_root.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing M442 stimulus overwrite")

    paths = {
        "catalog": root / "results/m430a_trainonly_dualaware_q32_catalog_r1_20260826/m430_trainonly_dualaware_q32_catalog_r1.json",
        "m430_result": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/m430b_h67_dualaware_q32_heldout_r1.json",
        "m430_seal": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/SHA256SUMS.seal.sha256",
        "weight_o0": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o0_weight_i_ky_kx_o_s8.bin",
        "weight_o1": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o1_weight_i_ky_kx_o_s8.bin",
        "weight_o2": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o2_weight_i_ky_kx_o_s8.bin",
        "weight_o3": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o3_weight_i_ky_kx_o_s8.bin",
        "docs359": root / "docs/359_DATE终局冻结_20260813.md",
    }
    identities = {}
    for name, path in paths.items():
        actual = sha256(path)
        require(actual == EXPECTED[name], "M442 input SHA drift: " + name)
        identities[name] = {
            "path": str(path.relative_to(root)), "sha256": actual}

    catalog = strict_json(paths["catalog"])
    m430 = strict_json(paths["m430_result"])
    require(catalog["status"] ==
            "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT",
            "M442 catalog status drift")
    require(m430["status"] ==
            "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY" and
            m430["execution_gates"]["population_or_exact_reconstruction_mismatches"] == 0,
            "M442 M430 admission drift")
    geometry = catalog["geometry"]
    require(len(catalog["operators"]) == 4 and
            geometry["partitions_per_operator"] == 432 and
            geometry["q_capacity"] == 32 and
            geometry["output_blocks"] == 8 and
            geometry["shared_lanes"] == 96,
            "M442 geometry drift")

    output_dir.mkdir(parents=True, exist_ok=False)
    stimulus = output_dir / "m442_m430_static_codec_population.hex"
    global_digest = hashlib.sha256()
    blocks = lanes = narrow_blocks = signed12_violations = 0
    original_payload_mismatches = narrow_reconstruction_mismatches = 0
    global_minimum = 1 << 30
    global_maximum = -(1 << 30)

    with stimulus.open("w", encoding="ascii") as handle:
        for operator in range(4):
            weights = np.fromfile(paths[f"weight_o{operator}"], dtype=np.int8)
            require(weights.size == 6912 * 768, "M442 weight extent drift")
            weights = weights.reshape(6912, 768).astype(np.int16)
            for partition in range(432):
                centers = [int(value, 16) for value in
                           catalog["operators"][operator]["partitions"]
                           [partition]["nested_patterns"][:32]]
                bits = np.asarray([
                    [(center >> bit) & 1 for bit in range(16)]
                    for center in centers], dtype=np.int16)
                values = bits @ weights[partition * 16:(partition + 1) * 16]
                require(values.shape == (32, 768), "M442 PWP shape drift")
                for center_id in range(32):
                    for output_block in range(8):
                        vector = values[center_id,
                                        output_block * 96:(output_block + 1) * 96]
                        minimum = int(vector.min())
                        maximum = int(vector.max())
                        global_minimum = min(global_minimum, minimum)
                        global_maximum = max(global_maximum, maximum)
                        signed12_violations += int(np.count_nonzero(
                            (vector < -2048) | (vector > 2047)))
                        narrow = minimum >= -128 and maximum <= 127
                        raw12 = vector.astype(np.int32) & 0xfff
                        low = packed_integer(raw12 & 0xff, 8)
                        high_full = packed_integer(raw12 >> 8, 4)
                        high = 0 if narrow else high_full
                        expected = packed_integer(raw12, 12)
                        reconstructed = np.asarray([
                            (((high >> (lane * 4)) & 0xf) << 8) |
                            ((low >> (lane * 8)) & 0xff)
                            for lane in range(96)], dtype=np.int32)
                        if narrow:
                            reconstructed = np.asarray([
                                value - 256 if value >= 128 else value
                                for value in reconstructed], dtype=np.int32) & 0xfff
                        original_payload_mismatches += int(np.count_nonzero(
                            reconstructed != raw12))
                        if narrow:
                            signed8 = np.where((raw12 & 0xff) >= 128,
                                               (raw12 & 0xff) - 256,
                                               raw12 & 0xff)
                            narrow_reconstruction_mismatches += int(
                                np.count_nonzero(signed8 != vector.astype(np.int32)))
                        header = struct.pack("<HHBBH", operator, partition,
                                             center_id, output_block,
                                             centers[center_id])
                        low_bytes = low.to_bytes(96, "little")
                        # M430's frozen codec identity hashes the physical
                        # low8/high4 representation before the narrow block
                        # elects to suppress its redundant high side.
                        high_bytes = high_full.to_bytes(48, "little")
                        block_digest = hashlib.sha256(
                            header + low_bytes + high_bytes + bytes(16) +
                            bytes([int(narrow)])).digest()
                        global_digest.update(block_digest)
                        tag = blocks
                        tile = operator & 1
                        handle.write(
                            f"{tag:06x} {tile:d} {center_id:02x} "
                            f"{output_block:x} {int(narrow):d} "
                            f"{low:0192x} {high:0128x} {expected:0288x}\n")
                        blocks += 1
                        lanes += 96
                        narrow_blocks += int(narrow)

    expected_codec = m430["static_codec"]
    actual_codec_sha = global_digest.hexdigest()
    require(blocks == expected_codec["blocks"] == 442368 and
            lanes == expected_codec["lanes"] == 42467328 and
            narrow_blocks == expected_codec["narrow_blocks"] == 70503 and
            signed12_violations == 0 and original_payload_mismatches == 0 and
            narrow_reconstruction_mismatches == 0 and
            global_minimum == expected_codec["global_minimum"] and
            global_maximum == expected_codec["global_maximum"] and
            actual_codec_sha == expected_codec["codec_global_sha256"],
            "M442 full static codec population gate failed")

    receipt = {
        "schema": "m442_m430_static_codec_vcs_stimulus_receipt_v1",
        "status": "PASS_M442_M430_FULL_STATIC_CODEC_STIMULUS",
        "identity": identities,
        "population": {
            "blocks": blocks, "lanes": lanes,
            "narrow_blocks": narrow_blocks,
            "wide_blocks": blocks - narrow_blocks,
            "global_minimum": global_minimum,
            "global_maximum": global_maximum,
            "codec_global_sha256": actual_codec_sha,
            "stimulus_sha256": sha256(stimulus),
            "stimulus_bytes": stimulus.stat().st_size,
        },
        "verification": {
            "signed12_violations": signed12_violations,
            "original_payload_mismatches": original_payload_mismatches,
            "narrow_reconstruction_mismatches": narrow_reconstruction_mismatches,
            "matches_m430_static_codec_identity": True,
        },
        "claim_boundary": {
            "full_static_codec_population": True,
            "runtime_issue_population": False,
            "runtime_cycles": False,
            "system_speedup": False,
            "power": False,
            "ppa": False,
            "date_headline": False,
        },
    }
    receipt_path = output_dir / "m442_m430_static_codec_stimulus_receipt_r1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text(
        f"{sha256(stimulus)}  {stimulus.name}\n"
        f"{sha256(receipt_path)}  {receipt_path.name}\n", encoding="utf-8")
    (output_dir / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")
    print("PASS_M442_M430_FULL_STATIC_CODEC_STIMULUS "
          f"blocks={blocks} lanes={lanes} narrow={narrow_blocks} "
          f"wide={blocks-narrow_blocks} mismatches=0")


if __name__ == "__main__":
    main()
