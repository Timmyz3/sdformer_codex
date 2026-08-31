#!/usr/bin/env python3
"""M1105D additive decoder-only source and identity preflight.

This program freezes call order, address regions, transaction/timing schema and
the exact D1 scaled-binary representation.  It deliberately does not enumerate
the production transaction population, run a cycle model, or calculate a
speedup.  A different-author hammer must release a later production runner.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct


SCHEMA = "m1105d_decoder_only_address_timed_source_contract_v1"
THETA_WORD = 1065353139
THETA_HEX_LE = "b3ff7f3f"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
MODULES = {
    0: ("sttmultires_unet.decoders.0.deconv.0", (10, 1, 1536, 15, 20), "EXACT_BINARY_BITPACK"),
    1: ("sttmultires_unet.decoders.1.deconv.0", (10, 1, 770, 30, 40), "EXACT_SCALED_BINARY_BITPACK"),
    2: ("sttmultires_unet.decoders.2.deconv.0", (10, 1, 386, 60, 80), "EXACT_BINARY_BITPACK"),
    3: ("sttmultires_unet.decoders.3.deconv.0", (10, 1, 194, 120, 160), "EXACT_BINARY_BITPACK"),
}


class Failure(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise Failure(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("non-finite JSON: " + token)))


def safe_member(name):
    member = PurePosixPath(name)
    require(not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == name, "unsafe member: " + name)
    return member


def verify_sealed_directory(directory):
    directory = Path(directory).resolve()
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "missing directory seals")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "bad SHA256SUMS line")
        require(fields[1] not in expected, "duplicate sealed member")
        expected[fields[1]] = fields[0]
    for name, digest in expected.items():
        path = directory.joinpath(*safe_member(name).parts)
        require(path.is_file() and not path.is_symlink(), "missing sealed member: " + name)
        require(sha256(path) == digest, "sealed member mismatch: " + name)
    outer_fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(outer_fields == [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    return {"members": len(expected), "manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}


def d1_scaled_binary_raw_sha(bitpack, elements, theta_word):
    """Recreate canonical little-endian FP32 bytes as 0 or exact theta word."""
    require(theta_word == THETA_WORD, "D1 theta word drift")
    digest = hashlib.sha256()
    seen = 0
    with Path(bitpack).open("rb") as handle:
        while True:
            block = handle.read(1 << 18)
            if not block:
                break
            remaining = elements - seen
            take = min(remaining, len(block) * 8)
            words = bytearray(take * 4)
            for index in range(take):
                bit = (block[index >> 3] >> (index & 7)) & 1
                struct.pack_into("<I", words, index * 4, theta_word if bit else 0)
            digest.update(words)
            seen += take
    require(seen == elements, "D1 packed payload element underflow")
    require(Path(bitpack).stat().st_size == (elements + 7) // 8,
            "D1 packed payload byte count drift")
    return digest.hexdigest()


def call_address_regions(global_call_index, module_index):
    stride = 1 << 32
    return {
        "input_descriptor_base": (1 << 60) + global_call_index * stride,
        "weight_base": (2 << 60) + module_index * stride,
        "psum_base": (3 << 60) + global_call_index * stride,
        "output_commit_base": (4 << 60) + global_call_index * stride,
        "control_descriptor_base": (5 << 60) + global_call_index * stride,
        "per_call_region_bytes": stride,
    }


def build(repo_root, contract_path):
    repo_root = Path(repo_root).resolve()
    hw = repo_root / "hw_autoresearch_nts07"
    contract = strict_json(contract_path)
    require(contract.get("schema") == SCHEMA, "contract schema drift")
    require(contract.get("launch_production") is False, "production launch forbidden")
    require(contract.get("system_speedup_admitted") is False, "speedup must be false")
    require("m700" not in json.dumps(contract, sort_keys=True).lower(),
            "M700 cannot enter an ours source contract")
    require(sha256(hw / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA256,
            "docs359 drift")
    source_row = contract["inputs"]["source"]
    require(sha256(Path(__file__).resolve()) == source_row["sha256"],
            "M1105D source identity drift")
    m705_root = hw / "reviews/m705_m699_multisequence_decoder_payload_fresh_result_hammer_r1_20260828"
    require(sha256(m705_root / "review.json") ==
            contract["inputs"]["m705_review_json_sha256"], "M705 review drift")
    require(sha256(m705_root / "SHA256SUMS.seal.sha256") ==
            contract["inputs"]["m705_outer_seal_file_sha256"], "M705 seal drift")
    payload_root = hw / contract["inputs"]["m699_directory"]
    seal = verify_sealed_directory(payload_root)
    manifest_path = payload_root / "manifest.json"
    require(sha256(manifest_path) == contract["inputs"]["m699_manifest_sha256"],
            "M699 manifest drift")
    manifest = strict_json(manifest_path)
    require(manifest.get("schema") == "m699_h67_ep35_multisequence_decoder_payload_v1",
            "M699 schema drift")
    threshold = manifest["d1_runtime_threshold_identity"]
    require(threshold["ieee754_uint32"] == THETA_WORD and
            threshold["ieee754_le_hex"] == THETA_HEX_LE,
            "D1 theta identity drift")
    require(struct.pack("<I", THETA_WORD).hex() == THETA_HEX_LE,
            "host-independent theta packing failure")
    rows = manifest.get("records")
    require(isinstance(rows, list) and len(rows) == 120, "M699 scope must be 120 calls")
    sequences = contract["population"]["sequence_order"]
    require(sequences == ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"],
            "sequence order drift")
    calls = []
    d1_miters = []
    packed_total = 0
    for ordinal, row in enumerate(rows):
        module_index = ordinal % 4
        global_sample = ordinal // 4
        expected_sequence = sequences[global_sample // 10]
        expected_name, expected_shape, expected_route = MODULES[module_index]
        require(row["global_call_index"] == ordinal, "global call ordinal drift")
        require(row["global_sample_id"] == global_sample, "global sample order drift")
        require(row["module_index"] == module_index, "D0-D3 order drift")
        require(row["sequence"] == expected_sequence and
                row["sequence_sample_id"] == global_sample % 10,
                "three-sequence cohort order drift")
        require(row["name"] == expected_name and tuple(row["input_shape"]) == expected_shape,
                "module identity/shape drift")
        require(row["route"] == expected_route, "route drift")
        stats = row["statistics"]["scaled_binary_audit"] if module_index == 1 else row["statistics"]
        payload = payload_root.joinpath(*safe_member(row["relative_path"]).parts)
        require(payload.is_file() and not payload.is_symlink(), "payload missing")
        require(sha256(payload) == stats["packed_sha256"], "payload hash drift")
        require(payload.stat().st_size == stats["packed_bytes"], "payload size drift")
        packed_total += payload.stat().st_size
        if module_index == 1:
            reconstructed = d1_scaled_binary_raw_sha(payload, stats["elements"], THETA_WORD)
            require(reconstructed == row["raw_fp32_content_sha256"],
                    "D1 exact scaled-binary raw-content miter mismatch")
            d1_miters.append({"global_call_index": ordinal,
                              "packed_sha256": stats["packed_sha256"],
                              "reconstructed_raw_fp32_sha256": reconstructed,
                              "expected_raw_fp32_sha256": row["raw_fp32_content_sha256"],
                              "mismatch": False})
        calls.append({
            "global_ordinal": ordinal,
            "global_sample_id": global_sample,
            "sequence_ordinal": global_sample // 10,
            "sequence": row["sequence"],
            "sequence_sample_id": row["sequence_sample_id"],
            "module_ordinal": module_index,
            "module": row["name"],
            "route": row["route"],
            "input_shape": row["input_shape"],
            "payload_relative_path": row["relative_path"],
            "payload_sha256": stats["packed_sha256"],
            "numeric_source": ({"encoding": "bit_times_exact_theta_word",
                                "theta_word": THETA_WORD,
                                "theta_le_hex": THETA_HEX_LE,
                                "weight_folding": False}
                               if module_index == 1 else
                               {"encoding": "exact_binary", "theta_word": None,
                                "weight_folding": False}),
            "address_regions": call_address_regions(ordinal, module_index),
        })
    require(packed_total == 261090000, "packed population byte drift")
    require(len(d1_miters) == 30 and all(not row["mismatch"] for row in d1_miters),
            "D1 miter population drift")
    return {
        "schema": "m1105d_decoder_only_source_preflight_receipt_v1",
        "status": "PASS_SOURCE_AND_FULL_IDENTITY_PREFLIGHT__PRODUCTION_NOT_RELEASED",
        "contract_sha256": sha256(contract_path),
        "input_identity": {"m699_manifest_sha256": sha256(manifest_path),
            "sealed_directory": seal,
                           "m705_review_sha256": sha256(m705_root / "review.json"),
                           "checkpoint_sha256": manifest["identity"]["core_inputs"]["checkpoint"]["sha256"],
                           "final_checkpoint_rebind_required_if_changed": True},
        "population": {"sequences": 3, "samples": 30, "calls": 120,
                       "packed_bytes": packed_total,
                       "global_ordinals_contiguous": True,
                       "per_sample_module_order": ["D0", "D1", "D2", "D3"]},
        "d1_exact_scaled_binary_miter": {
            "theta_word": THETA_WORD, "theta_le_hex": THETA_HEX_LE,
            "calls_checked": len(d1_miters), "mismatches": 0,
            "folded_weights": False, "coerced_to_one": False,
            "records": d1_miters},
        "common_resource_schedule_schema": contract["common_resource_schedule_schema"],
        "transaction_event_schema": contract["transaction_event_schema"],
        "calls": calls,
        "release": {"production_run_allowed": False,
                    "requires_different_author_hammer": True,
                    "production_cycles": None, "speedup": None,
                    "system_speedup_admitted": False},
        "claim_boundary": {"source": True, "identity_preflight": True,
                           "global_call_order": True, "address_schema": True,
                           "timing_schema": True, "d1_input_numeric_miter": True,
                           "production_transactions": False, "cycles": False,
                           "traffic": False, "speedup": False,
                           "system_speedup": False, "ours_performance": False,
                           "rtl": False, "eda": False, "energy": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repo_root, args.contract)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
