#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1105Dr2 fixed-trust decoder identity/address/timing source preflight.

This source derives every canonical path from its own fixed location, accepts
zero arguments, writes no output and cannot run production.  Its sole contract
is pinned by file, sidecar and outer-seal SHA.  A sealed author receipt and a
later independent hammer bind this successor source SHA without a hash cycle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
from typing import Any


SOURCE = Path(__file__).absolute()
HERE = SOURCE.parent
HW = HERE.parent.parent
REPO = HW.parent
CONTRACT = HW / "contracts/m1105dr2_decoder_only_address_timed_source_contract_r2_20260830.json"
CONTRACT_SHA256 = "cdbae0362d3ea093dbcb318aa2efad04e70677f8d984a9908cda44b0de3b80a4"
CONTRACT_SIDECAR_SHA256 = "37cdc8aa6b0c31103affa46f1aea80f073689540b16a40ea0eec68904a0fb4fe"
CONTRACT_OUTER_SHA256 = "4f95a616e16530bc30f94b68235247f7c7abe1b32956fc981412b3b1576193d3"
CONTRACT_LEAF_COUNT = 136
CONTRACT_LEAF_DIGEST_SHA256 = "a4551d23ed3298206e4f1e1c2a36a943f1fbf66f46e1b0b49f6610dc5160a9de"
PAYLOAD_ROOT = HW / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
M699_MANIFEST = PAYLOAD_ROOT / "manifest.json"
M705 = HW / "reviews/m705_m699_multisequence_decoder_payload_fresh_result_hammer_r1_20260828"
M1106D = HW / "reviews/m1106d_m1105d_decoder_source_contract_independent_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m1105dr2_decoder_only_address_timed_source_contract_v2"
THETA_WORD = 1065353139
THETA_HEX_LE = "b3ff7f3f"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M699_MANIFEST_SHA256 = "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0"
M699_OUTER_SHA256 = "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c"
M705_REVIEW_SHA256 = "6af48fb271254ef20f6baa1e435acfe51fdf38b457fe9782d6cac0b0e2883bd3"
M705_OUTER_SHA256 = "26781f5de30c6b6283c955144bbdac9c2b094aac3c19962b37016a57a6d24ff7"
M1106D_OUTER_SHA256 = "eb5fc732c83c533f4637f87e0727dfaa57019014f14cb43423f26fc736ff1132"
CHECKPOINT_SHA256 = "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
SEQUENCES = ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"]
MODULES = {
    0: ("sttmultires_unet.decoders.0.deconv.0", (10, 1, 1536, 15, 20), "EXACT_BINARY_BITPACK"),
    1: ("sttmultires_unet.decoders.1.deconv.0", (10, 1, 770, 30, 40), "EXACT_SCALED_BINARY_BITPACK"),
    2: ("sttmultires_unet.decoders.2.deconv.0", (10, 1, 386, 60, 80), "EXACT_BINARY_BITPACK"),
    3: ("sttmultires_unet.decoders.3.deconv.0", (10, 1, 194, 120, 160), "EXACT_BINARY_BITPACK"),
}

EXPECTED_RESOURCE = {
    "lanes": 96,
    "accumulator_bits": 24,
    "clock_ns": 3.0,
    "external_bytes_per_cycle": 192,
    "onchip_sram_bytes_macro_rounded": 245760,
    "macro_round_bytes": 128,
    "partitions": {"weight_bytes": 13824, "psum_bytes": 221184,
                   "descriptor_control_bytes": 8192,
                   "reserved_unallocated_bytes": 2560},
    "ports": {
        "weight": {"banks": 8, "mode": "1R1W", "row_bytes": 16,
                   "read_latency_cycles": 4, "initiation_interval": 1,
                   "outstanding_per_bank": 8},
        "psum": {"banks": 6, "mode": "1RW", "row_bytes": 48,
                 "read_latency_cycles": 2, "write_latency_cycles": 1,
                 "initiation_interval": 1, "outstanding_per_bank": 8},
        "external": {"banks": 1, "mode": "1RW", "row_bytes": 192,
                     "read_latency_cycles": 32, "write_latency_cycles": 3,
                     "initiation_interval": 1, "outstanding_per_bank": 16},
        "compute": {"contexts": 1, "row_bytes": 288,
                    "latency_cycles": 1, "initiation_interval": 1},
    },
    "address_regions": {
        "input_descriptor": "0x1_000000000000000 + global_call_ordinal * 2^32",
        "weight": "0x2_000000000000000 + module_ordinal * 2^32",
        "psum": "0x3_000000000000000 + global_call_ordinal * 2^32",
        "output_commit": "0x4_000000000000000 + global_call_ordinal * 2^32",
        "control_descriptor": "0x5_000000000000000 + global_call_ordinal * 2^32",
    },
}
EXPECTED_TRANSACTION_SCHEMA = {
    "required_identity_fields": ["global_call_ordinal", "global_transaction_ordinal",
        "sequence_ordinal", "sequence_sample_id", "module_ordinal", "timestep",
        "phase", "destination", "output_block", "configuration"],
    "required_address_fields": ["kind", "base_address", "byte_width", "bank_ids",
                                "address_offsets"],
    "required_dependency_fields": ["dependency_tokens", "produces_token"],
    "required_time_fields": ["earliest_issue_cycle", "dependency_ready_cycle",
        "issue_cycle", "return_cycle", "commit_cycle", "stall_class"],
    "allowed_kinds": ["input_descriptor_read", "weight_read", "psum_read", "compute",
                      "psum_write", "output_commit"],
    "time_policy": "All production timestamps remain absent until a separately hammered runner consumes every frozen call under the identical resource.",
}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("non-finite JSON: " + token)))


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "non-regular canonical member: " + str(path))
    require(sha256(path) == expected, "canonical identity drift: " + str(path))


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    require(not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == name, "unsafe member: " + name)
    return member


def verify_double_contract() -> dict[str, str]:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    verify_regular(CONTRACT, CONTRACT_SHA256)
    verify_regular(side, CONTRACT_SIDECAR_SHA256)
    verify_regular(outer, CONTRACT_OUTER_SHA256)
    require(side.read_text(encoding="utf-8").split() ==
            [CONTRACT_SHA256, CONTRACT.name], "canonical contract sidecar drift")
    require(outer.read_text(encoding="utf-8").split() ==
            [CONTRACT_SIDECAR_SHA256, side.name], "canonical contract outer drift")
    return {"contract_sha256": CONTRACT_SHA256,
            "contract_sidecar_sha256": CONTRACT_SIDECAR_SHA256,
            "contract_outer_seal_file_sha256": CONTRACT_OUTER_SHA256}


def verify_sealed_directory(directory: Path, expected_outer: str) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory absent/symlink")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64 and fields[1] not in expected,
                "bad/duplicate sealed member")
        expected[fields[1]] = fields[0]
    for name, digest in expected.items():
        verify_regular(directory.joinpath(*safe_member(name).parts), digest)
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS",
                                                       "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "sealed directory coverage drift")
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "sealed directory outer drift")
    return {"members": len(expected), "manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": expected_outer}


def contract_leaves(value: Any) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    def walk(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                walk(item[key], path + (key,))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, path + (str(index),))
        else:
            leaves.append({"path": "/".join(path), "type": type(item).__name__,
                           "value": item})
    walk(value, ())
    return leaves


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    leaves = contract_leaves(contract)
    leaf_digest = hashlib.sha256(json.dumps(
        leaves, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    require(len(leaves) == CONTRACT_LEAF_COUNT and
            leaf_digest == CONTRACT_LEAF_DIGEST_SHA256,
            "canonical contract leaf projection drift")
    require(set(contract) == {"schema", "status", "date", "objective", "trust_root",
            "inputs", "population", "d1_numeric_contract",
            "common_resource_schedule_schema", "transaction_event_schema", "release",
            "claim_boundary"}, "canonical contract top-level field drift")
    require(contract["schema"] == SCHEMA and
            contract["status"] ==
                "M1105DR2_CANONICAL_SOURCE_CONTRACT__SOURCE_SHA_BOUND_BY_AUTHOR_RECEIPT",
            "contract schema/status drift")
    require(contract["trust_root"] == {
        "canonical_contract_path": "contracts/m1105dr2_decoder_only_address_timed_source_contract_r2_20260830.json",
        "canonical_source_path": "system_simulator/scripts/build_m1105dr2_decoder_only_address_timed_source.py",
        "source_sha_in_contract": False,
        "source_and_contract_bound_by_sealed_author_receipt": True,
        "caller_repo_root_allowed": False,
        "caller_contract_path_allowed": False,
        "caller_output_path_allowed": False,
        "caller_environment_override_allowed": False,
    }, "contract trust-root fields drift")
    require(contract["inputs"] == {
        "m699_directory": "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828",
        "m699_manifest_sha256": M699_MANIFEST_SHA256,
        "m699_outer_seal_file_sha256": M699_OUTER_SHA256,
        "m705_review_json_sha256": M705_REVIEW_SHA256,
        "m705_outer_seal_file_sha256": M705_OUTER_SHA256,
        "m1106d_stop_outer_seal_file_sha256": M1106D_OUTER_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }, "contract input fields drift")
    population = contract["population"]
    require(population == {
        "checkpoint": "H67_ep35", "checkpoint_sha256": CHECKPOINT_SHA256,
        "sequence_order": SEQUENCES, "samples_per_sequence": 10,
        "calls_per_sample": 4, "expected_calls": 120,
        "expected_packed_bytes": 261090000,
        "final_checkpoint_rebind_required_if_changed": True,
        "final_checkpoint_rebind_scope": ["payload_activity", "D1_theta_identity",
            "weight_identity", "numeric_miters", "transaction_population", "cycles",
            "traffic", "energy", "system_table"],
    }, "contract population/rebind fields drift")
    require(contract["d1_numeric_contract"] == {
        "encoding": "EXACT_SCALED_BINARY_BITPACK",
        "allowed_values": ["FP32_WORD_0", "FP32_WORD_1065353139"],
        "theta_word_uint32": THETA_WORD,
        "theta_ieee754_le_hex": THETA_HEX_LE,
        "weight_folding_allowed": False,
        "coercion_to_binary_one_allowed": False,
        "miter": "Stream unpack little-bit-order payload and reconstruct canonical little-endian FP32 words 0 or theta; SHA256 must equal each frozen raw_fp32_content_sha256.",
    }, "contract D1 fields drift")
    require(contract["common_resource_schedule_schema"] == EXPECTED_RESOURCE and
            sum(EXPECTED_RESOURCE["partitions"].values()) == 245760,
            "contract resource fields drift")
    require(contract["transaction_event_schema"] == EXPECTED_TRANSACTION_SCHEMA,
            "contract transaction fields drift")
    require(contract["release"] == {
        "source_and_identity_preflight_authorized": True,
        "small_numeric_miter_authorized": True,
        "production_run_allowed": False,
        "production_replay_authorized": False,
        "different_author_source_hammer_required": True,
        "different_author_runner_and_launch_hammer_required": True,
    }, "contract release fields drift")
    boundary = contract["claim_boundary"]
    require(boundary["source"] is True and boundary["identity_preflight"] is True and
            boundary["global_call_order"] is True and
            boundary["address_and_timing_schema"] is True and
            all(boundary[key] is False for key in ("external_opportunity_result_admitted",
                "production_transactions", "cycles", "traffic", "speedup",
                "system_speedup", "ours_performance", "rtl", "eda", "energy", "ppa")),
            "contract claim boundary drift")
    require("m700" not in json.dumps(contract, sort_keys=True).lower(),
            "external M700 result cannot enter canonical contract")
    return {"leaf_count": len(leaves), "leaf_digest_sha256": leaf_digest}


def d1_scaled_binary_raw_sha(bitpack: Path, elements: int,
                             theta_word: int) -> str:
    require(theta_word == THETA_WORD, "D1 theta word drift")
    digest = hashlib.sha256()
    seen = 0
    with bitpack.open("rb") as stream:
        while True:
            block = stream.read(1 << 18)
            if not block:
                break
            take = min(elements - seen, len(block) * 8)
            words = bytearray(take * 4)
            for index in range(take):
                bit = (block[index >> 3] >> (index & 7)) & 1
                struct.pack_into("<I", words, index * 4,
                                 theta_word if bit else 0)
            digest.update(words)
            seen += take
    require(seen == elements and bitpack.stat().st_size == (elements + 7) // 8,
            "D1 packed population drift")
    return digest.hexdigest()


def call_address_regions(global_call_index: int, module_index: int) -> dict[str, int]:
    stride = 1 << 32
    result = {
        "input_descriptor_base": (1 << 60) + global_call_index * stride,
        "weight_base": (2 << 60) + module_index * stride,
        "psum_base": (3 << 60) + global_call_index * stride,
        "output_commit_base": (4 << 60) + global_call_index * stride,
        "control_descriptor_base": (5 << 60) + global_call_index * stride,
        "per_call_region_bytes": stride,
    }
    require(len({value >> 60 for key, value in result.items()
                 if key.endswith("_base")}) == 5, "address region overlap")
    return result


def build_canonical() -> dict[str, Any]:
    require(SOURCE == HERE / "build_m1105dr2_decoder_only_address_timed_source.py" and
            SOURCE.is_file() and not SOURCE.is_symlink() and
            HW.name == "hw_autoresearch_nts07" and REPO.name == "SDformer",
            "source-derived canonical root drift")
    contract_identity = verify_double_contract()
    contract = strict_json(CONTRACT)
    contract_projection = validate_contract(contract)
    verify_regular(DOCS359, DOCS359_SHA256)
    m1106d_seal = verify_sealed_directory(M1106D, M1106D_OUTER_SHA256)
    m1106d_review = strict_json(M1106D / "review.json")
    require(m1106d_review["status"] ==
            "STOP_M1106D_CALLER_CONTRACT_FORGERY__NO_PRODUCTION_RUNNER",
            "M1106D STOP status drift")
    m705_seal = verify_sealed_directory(M705, M705_OUTER_SHA256)
    verify_regular(M705 / "review.json", M705_REVIEW_SHA256)
    payload_seal = verify_sealed_directory(PAYLOAD_ROOT, M699_OUTER_SHA256)
    verify_regular(M699_MANIFEST, M699_MANIFEST_SHA256)
    manifest = strict_json(M699_MANIFEST)
    require(manifest["schema"] == "m699_h67_ep35_multisequence_decoder_payload_v1" and
            manifest["identity"]["core_inputs"]["checkpoint"]["sha256"] ==
                CHECKPOINT_SHA256,
            "M699 schema/checkpoint drift")
    threshold = manifest["d1_runtime_threshold_identity"]
    require(threshold["ieee754_uint32"] == THETA_WORD and
            threshold["ieee754_le_hex"] == THETA_HEX_LE and
            struct.pack("<I", THETA_WORD).hex() == THETA_HEX_LE,
            "D1 theta identity drift")
    rows = manifest["records"]
    require(type(rows) is list and len(rows) == 120, "canonical 120-call scope drift")
    calls: list[dict[str, Any]] = []
    d1_records: list[dict[str, Any]] = []
    packed_total = 0
    for ordinal, row in enumerate(rows):
        module = ordinal % 4
        sample = ordinal // 4
        expected_name, expected_shape, expected_route = MODULES[module]
        require(row["global_call_index"] == ordinal and
                row["global_sample_id"] == sample and
                row["module_index"] == module and
                row["sequence"] == SEQUENCES[sample // 10] and
                row["sequence_sample_id"] == sample % 10 and
                row["name"] == expected_name and
                tuple(row["input_shape"]) == expected_shape and
                row["route"] == expected_route,
                "call order/module/sequence drift")
        stats = (row["statistics"]["scaled_binary_audit"] if module == 1
                 else row["statistics"])
        payload = PAYLOAD_ROOT.joinpath(*safe_member(row["relative_path"]).parts)
        verify_regular(payload, stats["packed_sha256"])
        require(payload.stat().st_size == stats["packed_bytes"], "payload byte drift")
        packed_total += payload.stat().st_size
        if module == 1:
            reconstructed = d1_scaled_binary_raw_sha(payload, stats["elements"],
                                                       THETA_WORD)
            require(reconstructed == row["raw_fp32_content_sha256"],
                    "D1 exact scaled-binary miter mismatch")
            d1_records.append({"global_call_index": ordinal,
                "packed_sha256": stats["packed_sha256"],
                "reconstructed_raw_fp32_sha256": reconstructed,
                "expected_raw_fp32_sha256": row["raw_fp32_content_sha256"],
                "mismatch": False})
        calls.append({
            "global_ordinal": ordinal, "global_sample_id": sample,
            "sequence_ordinal": sample // 10, "sequence": row["sequence"],
            "sequence_sample_id": row["sequence_sample_id"],
            "module_ordinal": module, "module": row["name"],
            "route": row["route"], "input_shape": row["input_shape"],
            "payload_relative_path": row["relative_path"],
            "payload_sha256": stats["packed_sha256"],
            "numeric_source": ({"encoding": "bit_times_exact_theta_word",
                "theta_word": THETA_WORD, "theta_le_hex": THETA_HEX_LE,
                "weight_folding": False, "coerced_to_one": False}
                if module == 1 else {"encoding": "exact_binary", "theta_word": None,
                    "weight_folding": False, "coerced_to_one": False}),
            "address_regions": call_address_regions(ordinal, module),
        })
    require(packed_total == 261090000 and len(d1_records) == 30 and
            all(not row["mismatch"] for row in d1_records),
            "population/D1 miter count drift")
    return {
        "schema": "m1105dr2_decoder_only_source_preflight_receipt_v2",
        "status": "PASS_M1105DR2_FIXED_TRUST_SOURCE_PREFLIGHT__PRODUCTION_NOT_RELEASED",
        "trust_root": {
            "source_derived_repo": str(REPO), "source_derived_hw": str(HW),
            "canonical_payload": str(PAYLOAD_ROOT), **contract_identity,
            "contract_leaf_count": contract_projection["leaf_count"],
            "contract_leaf_digest_sha256": contract_projection["leaf_digest_sha256"],
            "source_sha_bound_by_sealed_author_receipt": True,
            "m1106d_stop": m1106d_seal,
        },
        "input_identity": {"m699_manifest_sha256": M699_MANIFEST_SHA256,
            "payload_seal": payload_seal, "m705_seal": m705_seal,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "final_checkpoint_rebind_required_if_changed": True,
            "final_checkpoint_rebind_scope": contract["population"][
                "final_checkpoint_rebind_scope"]},
        "population": {"sequences": 3, "samples": 30, "calls": 120,
            "packed_bytes": packed_total, "global_ordinals_contiguous": True,
            "per_sample_module_order": ["D0", "D1", "D2", "D3"]},
        "d1_exact_scaled_binary_miter": {"theta_word": THETA_WORD,
            "theta_le_hex": THETA_HEX_LE, "calls_checked": len(d1_records),
            "mismatches": 0, "folded_weights": False,
            "coerced_to_one": False, "records": d1_records},
        "common_resource_schedule_schema": EXPECTED_RESOURCE,
        "transaction_event_schema": EXPECTED_TRANSACTION_SCHEMA,
        "calls": calls,
        "external_baseline_rejection": {"m700_admitted": False,
            "ours_cycles_from_external_artifact": False},
        "release": {"production_run_allowed": False,
            "requires_different_author_source_hammer": True,
            "requires_different_author_runner_and_launch_hammer": True,
            "production_cycles": None, "speedup": None,
            "system_speedup_admitted": False},
        "claim_boundary": {"source": True, "identity_preflight": True,
            "global_call_order": True, "address_schema": True,
            "timing_schema": True, "d1_input_numeric_miter": True,
            "production_transactions": False, "cycles": False, "traffic": False,
            "speedup": False, "system_speedup": False, "ours_performance": False,
            "rtl": False, "eda": False, "energy": False, "ppa": False},
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    require(arguments == [], "M1105Dr2 accepts zero arguments")
    result = build_canonical()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
