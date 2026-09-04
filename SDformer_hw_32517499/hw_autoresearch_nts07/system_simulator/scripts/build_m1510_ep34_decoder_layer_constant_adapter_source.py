#!/usr/bin/env python3
"""Read-only layer-constant adapter for the sealed M1458 ep34 decoder capture.

The frozen M1323 ordered-population audit is retained exactly.  For payloads,
the frozen M1321 two-plane audit is also retained, with one narrow semantic
projection: D0 and D1 are both checked as a single positive finite layer
constant, while D2 and D3 remain exact {+0,+1}.  The projection is necessary
because M1321's older D0={0,1} assumption is false for the final ep34 capture.

Every retained decoder payload is then independently decompressed as a stream.
The derived nonzero FP32 word must be unique within the call, stable across all
thirty calls of its layer, and equal to the expected final-capture word.  No
bitplane schedule, cycle count, traffic, speedup, energy, PPA, or paper result
is produced by this source-only adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Mapping, Sequence
import zlib


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
SOURCE = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1510_ep34_decoder_layer_constant_adapter_source.py"
CONTRACT = HW / "contracts/m1510_ep34_decoder_layer_constant_adapter_source_contract_r1_20260831.json"
M1323_SOURCE = HERE / "build_m1323_ep34_decoder_capture_adapter_source.py"
M1323_SHA256 = "0481e39372ffe19cd3cff8d5053c9eae8326de4fb5ac61bd9e42527a3ad3a12a"
M1501_SOURCE = HW / "scripts/hammer_m1501_m1458_motion_ep34_live93_capture_result_safe_audit_source.py"
M1501_SHA256 = "0c271bba3dfa57940b0ebe5a2ddf980d15f058b5ea25244aec5ead77d8146c83"
CAPTURE_ROOT = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
CAPTURE_MANIFEST_SHA256 = "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e"
CAPTURE_OUTER_SHA256 = "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed"
ORDERED_SHA256 = "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1510_ep34_decoder_layer_constant_adapter_source_audit_r1_v1"
STATUS = "PASS_M1510_SOURCE_ONLY_LAYER_CONSTANT_AUDIT__NO_BITPLANE_NO_CYCLES"
SOURCE_STATUS = "SOURCE_ONLY__SEALED_M1458_LAYER_CONSTANT_ADAPTER__NO_EDA_NO_GPU_NO_REMOTE"
EXPECTED_WORDS = {
    0: 0x3F7FFD6B,
    1: 0x3F7FFFA0,
    2: 0x3F800000,
    3: 0x3F800000,
}
EXPECTED_CALLS_PER_LAYER = 30
CLAIM_BOUNDARY = {
    "source_only": True,
    "read_only": True,
    "sealed_capture_input": True,
    "normalized_payload_written": False,
    "bitplane": False,
    "cycles": False,
    "traffic": False,
    "speedup": False,
    "system_speedup": False,
    "energy": False,
    "ppa": False,
    "table_a": False,
    "paper_result": False,
}


class M1510Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1510Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise M1510Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    regular_exact(path, digest, name + " after import")
    return module


M1323 = load_exact("m1510_frozen_m1323", M1323_SOURCE, M1323_SHA256)
M1321 = M1323.M1321
M1501 = load_exact("m1510_frozen_m1501", M1501_SOURCE, M1501_SHA256)


def lowercase_sha(value: Any, label: str) -> str:
    require(type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            label + " is not lowercase SHA256")
    return value


def stream_word_census(compressed_path: Path, shape: Sequence[int],
                       expected_raw_sha256: str,
                       expected_compressed_sha256: str) -> dict[str, Any]:
    """Stream one zlib FP32 payload and derive its exact word census."""
    elements = M1321.product(tuple(shape))
    raw_digest = hashlib.sha256()
    compressed_digest = hashlib.sha256()
    decompressor = zlib.decompressobj()
    carry = b""
    words = 0
    zero_count = 0
    positive_count = 0
    negative_count = 0
    nonfinite_count = 0
    positive_words: set[int] = set()

    def consume(payload: bytes) -> None:
        nonlocal carry, words, zero_count, positive_count
        nonlocal negative_count, nonfinite_count
        raw_digest.update(payload)
        block = carry + payload
        usable = len(block) - len(block) % 4
        for (word,) in struct.iter_unpack("<I", block[:usable]):
            require(words < elements, "FP32 payload exceeds declared shape")
            if word == 0:
                zero_count += 1
            elif ((word >> 23) & 0xFF) == 0xFF:
                nonfinite_count += 1
            elif word & 0x80000000:
                negative_count += 1
            else:
                positive_count += 1
                positive_words.add(word)
            words += 1
        carry = block[usable:]

    with Path(compressed_path).open("rb") as compressed:
        for chunk in iter(lambda: compressed.read(1 << 20), b""):
            compressed_digest.update(chunk)
            consume(decompressor.decompress(chunk))
    consume(decompressor.flush())
    require(decompressor.eof and not decompressor.unused_data and
            not decompressor.unconsumed_tail,
            "compressed FP32 stream is truncated or has trailing data")
    require(not carry and words == elements,
            "FP32 decompressed extent differs from shape")
    require(raw_digest.hexdigest() == lowercase_sha(
        expected_raw_sha256, "raw FP32 SHA"), "raw FP32 SHA mismatch")
    require(compressed_digest.hexdigest() == lowercase_sha(
        expected_compressed_sha256, "compressed payload SHA"),
        "compressed payload SHA mismatch")
    require(nonfinite_count == 0, "nonfinite decoder word observed")
    require(negative_count == 0, "negative decoder word observed")
    require(positive_count > 0 and len(positive_words) == 1,
            "call does not contain exactly one unique positive finite nonzero word")
    word = next(iter(positive_words))
    return {
        "elements": elements,
        "zero_count": zero_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "nonfinite_count": nonfinite_count,
        "positive_word_uint32": word,
        "positive_word_ieee754_le_hex": struct.pack("<I", word).hex(),
        "positive_word_float32": struct.unpack("<f", struct.pack("<I", word))[0],
        "raw_fp32_sha256": raw_digest.hexdigest(),
        "compressed_sha256": compressed_digest.hexdigest(),
    }


def audit_call_payload(root: Path, call: Mapping[str, Any]) -> dict[str, Any]:
    """Apply exact M1321 planes plus the M1510 final-layer word contract."""
    ordinal = call.get("module_ordinal")
    require(type(ordinal) is int and ordinal in range(4),
            "module ordinal is not exact integer 0..3")
    try:
        compressed = M1321.safe_member(root, call.get("compressed_fp32"),
                                       "compressed FP32")
        support = M1321.safe_member(root, call.get("support_sign"), "support/sign")
    except M1321.AdapterError as error:
        raise M1510Error(str(error)) from error

    # M1321's D1 rule is precisely the needed single-positive-finite-word rule.
    # Project D0 onto that rule rather than weakening or rewriting the predecessor.
    semantic_ordinal = 1 if ordinal in (0, 1) else ordinal
    try:
        planes = M1323.audit_two_plane_payload(
            compressed, support, call.get("shape"), semantic_ordinal,
            call.get("raw_fp32_sha256"), call.get("compressed_sha256"),
            call.get("support_sign_sha256"))
    except M1323.M1323Error as error:
        raise M1510Error(str(error)) from error
    require(planes["positive_plane_bytes"] == call.get("positive_plane_bytes") and
            planes["negative_plane_bytes"] == call.get("negative_plane_bytes"),
            "recorded support-plane extent drift")

    census = stream_word_census(
        compressed, call.get("shape"), call.get("raw_fp32_sha256"),
        call.get("compressed_sha256"))
    expected = EXPECTED_WORDS[ordinal]
    require(census["positive_word_uint32"] == expected,
            "layer positive word differs from exact expected ep34 word")
    if ordinal in (2, 3):
        require(census["positive_word_uint32"] == M1321.ONE_WORD,
                "D2/D3 positive word is not exact ONE")
    return {
        "global_call_ordinal": call.get("global_call_ordinal"),
        "global_order": call.get("global_order"),
        "global_sample_id": call.get("global_sample_id"),
        "sequence": call.get("sequence"),
        "sample_key": call.get("sample_key"),
        "source_sha256": call.get("source_sha256"),
        "module_ordinal": ordinal,
        "module": call.get("module"),
        "shape": list(call.get("shape")),
        "compressed_fp32": call.get("compressed_fp32"),
        "support_sign": call.get("support_sign"),
        "raw_fp32_sha256": census["raw_fp32_sha256"],
        "compressed_sha256": census["compressed_sha256"],
        "support_sign_sha256": sha256(support),
        "positive_plane_bytes": planes["positive_plane_bytes"],
        "negative_plane_bytes": planes["negative_plane_bytes"],
        "semantic_projection": ("D0_TO_SINGLE_POSITIVE_FINITE_WORD"
                                if ordinal == 0 else "IDENTITY"),
        **{key: census[key] for key in (
            "elements", "zero_count", "positive_count", "negative_count",
            "nonfinite_count", "positive_word_uint32",
            "positive_word_ieee754_le_hex", "positive_word_float32")},
    }


def summarize_layers(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(type(calls) is list and len(calls) == 120,
            "decoder audited call population is not 120")
    output = []
    for ordinal in range(4):
        rows = [row for row in calls if row.get("module_ordinal") == ordinal]
        require(len(rows) == EXPECTED_CALLS_PER_LAYER,
                "layer call population is not thirty")
        words = {row.get("positive_word_uint32") for row in rows}
        require(len(words) == 1, "layer positive word drifts across calls")
        word = next(iter(words))
        require(type(word) is int and word == EXPECTED_WORDS[ordinal],
                "layer positive word differs from exact expected ep34 word")
        require([row.get("global_sample_id") for row in rows] == list(range(10, 40)),
                "layer sample population/order drift")
        output.append({
            "module_ordinal": ordinal,
            "module": M1323.MODULES[ordinal],
            "calls": EXPECTED_CALLS_PER_LAYER,
            "word_uint32": word,
            "word_hex": "0x{:08x}".format(word),
            "ieee754_le_hex": struct.pack("<I", word).hex(),
            "float32": struct.unpack("<f", struct.pack("<I", word))[0],
            "all_calls_same_word": True,
            "exact_one_required": ordinal in (2, 3),
        })
    return output


def ordered_calls(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    ordered_path = root / "unified_ordered_records.jsonl"
    regular_exact(ordered_path, ORDERED_SHA256, "M1458 ordered JSONL")
    records = [M1323.strict_json_text(line) for line in
               ordered_path.read_text(encoding="utf-8").splitlines()]
    calls, identity = M1323.decoder_rows_from_ordered(records)
    return calls, identity, sha256(ordered_path)


def validate_capture_seal(root: Path) -> dict[str, Any]:
    require(root.resolve() == CAPTURE_ROOT.resolve(),
            "capture root is not exact local M1458 result")
    regular_exact(root / "SHA256SUMS", CAPTURE_MANIFEST_SHA256,
                  "M1458 recursive manifest")
    regular_exact(root / "SHA256SUMS.seal.sha256", CAPTURE_OUTER_SHA256,
                  "M1458 outer seal")
    require((root / "SHA256SUMS.seal.sha256").read_text().split() ==
            [CAPTURE_MANIFEST_SHA256, "SHA256SUMS"],
            "M1458 outer seal content drift")
    try:
        result = M1501.validate_result(root)
    except Exception as error:
        raise M1510Error("exact M1501 sealed-result validation failed") from error
    require(result.get("status") ==
            "PASS_M1501_M1458_EP34_LIVE93_CAPTURE_RESULT",
            "M1501 sealed-result status drift")
    return result


def audit_capture(root: Path = CAPTURE_ROOT) -> dict[str, Any]:
    root = Path(root)
    seal = validate_capture_seal(root)
    calls, ordered_identity, ordered_sha = ordered_calls(root)
    audited = [audit_call_payload(root, call) for call in calls]
    layers = summarize_layers(audited)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "capture_root": str(root),
        "capture_seal": {
            "m1501_status": seal["status"],
            "sha256sums_sha256": CAPTURE_MANIFEST_SHA256,
            "outer_seal_sha256": CAPTURE_OUTER_SHA256,
            "ordered_jsonl_sha256": ordered_sha,
        },
        "ordered_identity": ordered_identity,
        "population": {"samples": 30, "calls": 120, "modules": 4,
                       "global_sample_ids": [10, 39]},
        "layer_scale_words": layers,
        "calls_schema": {
            "schema": "m1510_ep34_decoder_layer_constant_call_r1",
            "rows": 120,
            "ordered_by": ["global_sample_id", "module_ordinal"],
            "row_keys": sorted(audited[0]),
        },
        "calls": audited,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def validate_source_policy() -> dict[str, Any]:
    regular_exact(M1323_SOURCE, M1323_SHA256, "frozen M1323 source")
    regular_exact(M1501_SOURCE, M1501_SHA256, "frozen M1501 source")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    policy = M1323.strict_json(CONTRACT)
    require(policy.get("schema") == SCHEMA and
            policy.get("status") == SOURCE_STATUS,
            "M1510 contract schema/status drift")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "M1510 source/test identity drift")
    require(policy.get("claim_boundary") == CLAIM_BOUNDARY,
            "M1510 claim boundary drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--audit-capture", action="store_true")
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.source_self_check:
        validate_source_policy()
        print("PASS_M1510_SOURCE_SELF_CHECK__NO_CAPTURE_READ_NO_EDA_NO_GPU_NO_REMOTE")
        return 0
    result = audit_capture(args.capture_root)
    print(json.dumps({key: value for key, value in result.items() if key != "calls"},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1510Error as error:
        print("M1510_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
