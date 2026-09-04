#!/usr/bin/env python3
"""Export all frozen H67 q32/O4 exact PWP blocks for Synopsys VCS.

Each memh word is 1281 bits, LSB first by field:
  [767:0]    low8, lane i in byte i
  [1279:768] high sidecar, lane i in nibble i; bits [511:384] padding
  [1280]     trusted narrow bitmap value
The line index deterministically decodes as operator/partition/center/block.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_BLOCKS = 4 * 432 * 32 * 8
EXPECTED_LANES = EXPECTED_BLOCKS * 96


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

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs)


def pack_little(values, bits):
    word = 0
    mask = (1 << bits) - 1
    for index, value in enumerate(values):
        word |= (int(value) & mask) << (index * bits)
    return word


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m401-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M408 overwrite")

    contract = strict_json(args.m401_contract)
    require(contract["schema"] ==
            "m401_h67_q32_elastic_pwp_full_replay_contract_v1",
            "M401 contract schema drift")
    hw_root = args.m401_contract.resolve().parents[1]
    inputs = contract["inputs"]
    catalog_path = hw_root / inputs["m338_catalog"]["path"]
    require(sha256(catalog_path) == inputs["m338_catalog"]["sha256"],
            "catalog SHA drift")
    weight_paths = []
    for operator in range(4):
        entry = inputs[f"weight_o{operator}"]
        path = hw_root / entry["path"]
        require(sha256(path) == entry["sha256"],
                f"weight_o{operator} SHA drift")
        weight_paths.append(path)
    docs = hw_root / inputs["docs359"]["path"]
    require(sha256(docs) == inputs["docs359"]["sha256"],
            "docs359 SHA drift")

    catalog = strict_json(catalog_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stimulus_path = args.output_dir / "m408_h67_q32_static_codec_1281.memh"
    stimulus_digest = hashlib.sha256()
    codec_digest = hashlib.sha256()
    blocks = lanes = narrow_blocks = 0
    signed12_violations = wide_mismatches = narrow_mismatches = 0
    padding_nonzero = 0
    minimum = 1 << 30
    maximum = -(1 << 30)

    with stimulus_path.open("wb") as output:
        for operator in range(4):
            weights = np.fromfile(weight_paths[operator], dtype=np.int8)
            require(weights.size == 6912 * 768,
                    "weight extent drift")
            weights = weights.reshape(6912, 768).astype(np.int16)
            for partition in range(432):
                centers = [int(value, 16) for value in
                           catalog["operators"][operator]["partitions"]
                           [partition]["nested_patterns"][:32]]
                center_bits = np.asarray([
                    [(center >> bit) & 1 for bit in range(16)]
                    for center in centers], dtype=np.int16)
                values = center_bits @ weights[
                    partition * 16:(partition + 1) * 16]
                require(values.shape == (32, 768), "PWP shape drift")
                for center_id in range(32):
                    for output_block in range(8):
                        vector = values[center_id,
                                        output_block*96:(output_block+1)*96]
                        vector32 = vector.astype(np.int32)
                        local_min = int(vector32.min())
                        local_max = int(vector32.max())
                        minimum = min(minimum, local_min)
                        maximum = max(maximum, local_max)
                        signed12_violations += int(np.count_nonzero(
                            (vector32 < -2048) | (vector32 > 2047)))
                        raw12 = vector32 & 0xfff
                        low = raw12 & 0xff
                        high = (raw12 >> 8) & 0xf
                        narrow = local_min >= -128 and local_max <= 127
                        wide_raw = (high << 8) | low
                        wide_recon = np.where(wide_raw >= 2048,
                                              wide_raw - 4096, wide_raw)
                        wide_mismatches += int(np.count_nonzero(
                            wide_recon != vector32))
                        if narrow:
                            narrow_recon = np.where(low >= 128,
                                                    low - 256, low)
                            narrow_mismatches += int(np.count_nonzero(
                                narrow_recon != vector32))
                        low_word = pack_little(low, 8)
                        high_word = pack_little(high, 4)
                        physical = low_word | (high_word << 768)
                        physical |= int(narrow) << 1280
                        padding_nonzero += int(
                            ((physical >> 1152) & ((1 << 128) - 1)) != 0)
                        line = f"{physical:0321x}\n".encode("ascii")
                        output.write(line)
                        stimulus_digest.update(line)
                        identity = (operator.to_bytes(2, "little") +
                                    partition.to_bytes(2, "little") +
                                    center_id.to_bytes(1, "little") +
                                    output_block.to_bytes(1, "little"))
                        codec_digest.update(identity)
                        codec_digest.update(low.astype(np.uint8).tobytes())
                        codec_digest.update(high.astype(np.uint8).tobytes())
                        codec_digest.update(bytes([int(narrow)]))
                        blocks += 1
                        lanes += 96
                        narrow_blocks += int(narrow)

    require(blocks == EXPECTED_BLOCKS and lanes == EXPECTED_LANES,
            "static extent drift")
    require(narrow_blocks == 112167 and minimum == -1089
            and maximum == 1059, "frozen M401 codec population drift")
    require(signed12_violations == 0 and wide_mismatches == 0
            and narrow_mismatches == 0 and padding_nonzero == 0,
            "codec exactness failure")
    require(stimulus_digest.hexdigest() == sha256(stimulus_path),
            "streaming stimulus digest mismatch")

    manifest = {
        "schema": "m408_h67_q32_static_codec_vcs_stimulus_v1",
        "status": "PASS_M408_DETERMINISTIC_STATIC_STIMULUS_EXPORT",
        "identity": {
            "m401_contract": {
                "path": str(args.m401_contract.resolve().relative_to(hw_root)),
                "sha256": sha256(args.m401_contract)},
            "catalog": inputs["m338_catalog"],
            "weights": {f"o{operator}": inputs[f"weight_o{operator}"]
                        for operator in range(4)},
            "docs359": inputs["docs359"],
        },
        "layout": {
            "word_bits": 1281,
            "hex_digits_per_record": 321,
            "low_bits": [0, 767],
            "high_sidecar_bits": [768, 1279],
            "high_payload_bits_within_sidecar": [0, 383],
            "zero_padding_bits_within_sidecar": [384, 511],
            "narrow_bit": 1280,
            "index_order": "operator,partition,center_id,global_output_block"
        },
        "population": {
            "blocks": blocks,
            "lanes": lanes,
            "narrow_blocks": narrow_blocks,
            "wide_blocks": blocks - narrow_blocks,
            "expected_accepted_contributions":
                narrow_blocks + 2 * (blocks - narrow_blocks),
            "global_minimum": minimum,
            "global_maximum": maximum,
            "signed12_violations": signed12_violations,
            "wide_reconstruction_mismatches": wide_mismatches,
            "narrow_reconstruction_mismatches": narrow_mismatches,
            "nonzero_padding_bits": padding_nonzero,
        },
        "output": {
            "path": stimulus_path.name,
            "bytes": stimulus_path.stat().st_size,
            "sha256": stimulus_digest.hexdigest(),
            "codec_identity_sha256": codec_digest.hexdigest(),
        },
        "claim_boundary": {
            "derived_static_stimulus": True,
            "vcs_executed": False,
            "rtl_measured_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    manifest_path = args.output_dir / "m408_h67_q32_static_codec_vcs_stimulus_r1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2,
                                        sort_keys=True) + "\n",
                             encoding="utf-8")
    print("M408_EXPORT_PASS blocks={} narrow={} contributions={} sha256={}".
          format(blocks, narrow_blocks,
                 manifest["population"]["expected_accepted_contributions"],
                 stimulus_digest.hexdigest()), flush=True)


if __name__ == "__main__":
    main()
