#!/usr/bin/env python3
"""Source-only adapter from the future M1249 capture to decoder replay rows.

This module deliberately has no production writer and no cycle-simulator entry
point.  It validates the part of the final-checkpoint capture that a later,
separately reviewed M785 successor will consume:

* decoder samples are exactly global sample 10..39, in D0,D1,D2,D3 order;
* every retained support/sign file is two little-bit-order planes, not a legacy
  one-plane decoder bitpack;
* D0/D2/D3 are bit-exact {+0, +1}; D1 is bit-exact {+0, theta}, with one
  positive finite FP32 theta word shared by all thirty D1 calls;
* the positive plane reconstructs the nonzero locations and the negative plane
  is identically zero; and
* a future checkpoint extractor can bind the exact four ConvTranspose weight
  identities through a strict, checkpoint-keyed interface.

Reading an unhammered capture through :func:`audit_capture` never promotes it:
all performance/production fields in the returned boundary remain false.  A
future successor must bind the actual M1249 result seal and independent result
hammer before using these rows for a replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence
import zlib


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
DEFAULT_CAPTURE_ROOT = HW / (
    "results/m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830")

MODULES = (
    "sttmultires_unet.decoders.0.deconv.0",
    "sttmultires_unet.decoders.1.deconv.0",
    "sttmultires_unet.decoders.2.deconv.0",
    "sttmultires_unet.decoders.3.deconv.0",
)
SHAPES = (
    (10, 1, 1536, 15, 20),
    (10, 1, 770, 30, 40),
    (10, 1, 386, 60, 80),
    (10, 1, 194, 120, 160),
)
WEIGHT_SHAPES = (
    (1536, 384, 3, 3),
    (770, 192, 3, 3),
    (386, 96, 3, 3),
    (194, 96, 3, 3),
)
ZERO_WORD = 0x00000000
ONE_WORD = 0x3F800000
EXPECTED_SAMPLES = tuple(range(10, 40))
EXPECTED_CALLS = 120
EXPECTED_ORDERED_ROWS = 9880


class AdapterError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AdapterError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lowercase_sha(value: Any, label: str) -> str:
    require(type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            label + " is not lowercase SHA256")
    return value


def strict_json(path: Path) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AdapterError("nonfinite JSON token: " + token)))


def strict_json_text(text: str) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AdapterError("nonfinite JSON token: " + token)))


def regular(path: Path, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise AdapterError("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")


def safe_member(root: Path, relative: Any, label: str) -> Path:
    require(type(relative) is str, label + " path must be string")
    member = PurePosixPath(relative)
    require(member.parts and not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == relative, label + " path is unsafe")
    cursor = Path(root)
    for part in member.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), label + " path contains symlink")
    regular(cursor, label)
    return cursor


def product(values: Sequence[int]) -> int:
    require(values and all(type(value) is int and value > 0 for value in values),
            "shape must contain positive exact integers")
    return math.prod(values)


def bit_at(payload: mmap.mmap, index: int) -> int:
    return (payload[index >> 3] >> (index & 7)) & 1


def positive_finite_word(word: int) -> bool:
    return (word != ZERO_WORD and not (word & 0x80000000) and
            ((word >> 23) & 0xFF) != 0xFF)


def _check_padding(payload: mmap.mmap, plane_offset: int, elements: int,
                   label: str) -> None:
    remainder = elements & 7
    if remainder:
        final = payload[plane_offset + elements // 8]
        require(final >> remainder == 0, label + " has nonzero padding bits")


def audit_two_plane_payload(compressed_path: Path, support_path: Path,
                            shape: Sequence[int], module_ordinal: int,
                            expected_raw_sha256: str | None = None,
                            expected_compressed_sha256: str | None = None,
                            expected_support_sha256: str | None = None) -> dict[str, Any]:
    """Stream one FP32 tensor and prove exact support/sign-plane semantics."""
    require(module_ordinal in range(4), "module ordinal out of range")
    elements = product(tuple(shape))
    plane_bytes = (elements + 7) // 8
    regular(compressed_path, "compressed FP32 payload")
    regular(support_path, "support/sign payload")
    require(support_path.stat().st_size == 2 * plane_bytes,
            "support/sign payload is not exactly positive||negative planes")
    if expected_compressed_sha256 is not None:
        require(sha256(compressed_path) == lowercase_sha(
            expected_compressed_sha256, "compressed payload SHA"),
            "compressed payload SHA drift")
    if expected_support_sha256 is not None:
        require(sha256(support_path) == lowercase_sha(
            expected_support_sha256, "support payload SHA"),
            "support payload SHA drift")

    raw_digest = hashlib.sha256()
    raw_bytes = 0
    words_seen = 0
    active = 0
    theta_words: set[int] = set()
    decompressor = zlib.decompressobj()
    carry = b""
    with support_path.open("rb") as support_stream:
        with mmap.mmap(support_stream.fileno(), 0, access=mmap.ACCESS_READ) as planes:
            _check_padding(planes, 0, elements, "positive plane")
            _check_padding(planes, plane_bytes, elements, "negative plane")

            def consume(payload: bytes) -> None:
                nonlocal carry, raw_bytes, words_seen, active
                raw_digest.update(payload)
                raw_bytes += len(payload)
                block = carry + payload
                usable = len(block) - (len(block) % 4)
                for (word,) in struct.iter_unpack("<I", block[:usable]):
                    require(words_seen < elements, "FP32 payload exceeds shape")
                    positive = bit_at(planes, words_seen)
                    negative = bit_at(planes, plane_bytes * 8 + words_seen)
                    require(negative == 0, "decoder negative plane is nonzero")
                    if word == ZERO_WORD:
                        require(positive == 0, "zero FP32 word has positive support")
                    else:
                        require(positive == 1, "nonzero FP32 word lacks positive support")
                        active += 1
                        if module_ordinal == 1:
                            require(positive_finite_word(word),
                                    "D1 theta word is not positive finite")
                            theta_words.add(word)
                        else:
                            require(word == ONE_WORD,
                                    "D0/D2/D3 are not exact {+0,+1}")
                    words_seen += 1
                carry = block[usable:]

            with compressed_path.open("rb") as compressed:
                for chunk in iter(lambda: compressed.read(1 << 20), b""):
                    consume(decompressor.decompress(chunk))
            consume(decompressor.flush())
            require(decompressor.eof and not decompressor.unused_data and
                    not decompressor.unconsumed_tail,
                    "compressed FP32 stream is truncated or has trailing data")
            require(not carry and words_seen == elements and raw_bytes == elements * 4,
                    "FP32 decompressed extent differs from shape")

    raw_sha = raw_digest.hexdigest()
    if expected_raw_sha256 is not None:
        require(raw_sha == lowercase_sha(expected_raw_sha256, "raw FP32 SHA"),
                "raw FP32 SHA mismatch")
    require(module_ordinal != 1 or len(theta_words) <= 1,
            "one D1 call contains multiple nonzero theta words")
    return {
        "module_ordinal": module_ordinal,
        "shape": list(shape),
        "elements": elements,
        "active": active,
        "positive_plane_bytes": plane_bytes,
        "negative_plane_bytes": plane_bytes,
        "negative_count": 0,
        "raw_fp32_sha256": raw_sha,
        "positive_plane_sha256": _plane_sha256(support_path, 0, plane_bytes),
        "negative_plane_sha256": _plane_sha256(support_path, plane_bytes, plane_bytes),
        "theta_word_uint32": (next(iter(theta_words)) if theta_words else None),
    }


def _plane_sha256(path: Path, offset: int, count: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = count
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            require(block, "support plane truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def decoder_rows_from_ordered(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    per_sample: dict[int, list[Mapping[str, Any]]] = {sample: [] for sample in EXPECTED_SAMPLES}
    total = 0
    for row in records:
        total += 1
        sample = row.get("global_sample_id")
        require(type(sample) is int and 0 <= sample < 40,
                "ordered sample id is not exact 0..39 int")
        if sample in per_sample and row.get("category") == "decoder_convtranspose":
            per_sample[sample].append(row)
    require(total == EXPECTED_ORDERED_ROWS, "ordered population is not 9880")
    for sample in EXPECTED_SAMPLES:
        rows = per_sample[sample]
        require(len(rows) == 4, "decoder sample does not contain four calls")
        require(all(type(row.get("global_order")) is int and
                    row["global_order"] >= 0 for row in rows),
                "decoder global order is not nonnegative exact int")
        rows = sorted(rows, key=lambda row: row["global_order"])
        require([row.get("name") for row in rows] == list(MODULES),
                "decoder module order drift")
        for module_ordinal, row in enumerate(rows):
            require(row.get("input", {}).get("shape") == list(SHAPES[module_ordinal]),
                    "decoder input shape drift")
            payload = row.get("payload")
            require(type(payload) is dict and payload.get("retained") is True,
                    "decoder payload not retained")
            selected.append({
                "global_call_ordinal": len(selected),
                "global_sample_id": sample,
                "sequence": row.get("sequence"),
                "sample_key": row.get("sample_key"),
                "source_sha256": row.get("source_sha256"),
                "module_ordinal": module_ordinal,
                "module": MODULES[module_ordinal],
                "shape": list(SHAPES[module_ordinal]),
                "compressed_fp32": payload.get("compressed_fp32"),
                "compressed_sha256": payload.get("compressed_sha256"),
                "support_sign": payload.get("support_sign"),
                "support_sign_sha256": payload.get("support_sign_sha256"),
                "raw_fp32_sha256": payload.get("raw_fp32_sha256"),
                "positive_plane_bytes": payload.get("positive_plane_bytes"),
                "negative_plane_bytes": payload.get("negative_plane_bytes"),
            })
    require(len(selected) == EXPECTED_CALLS, "decoder call population is not 120")
    return selected


def validate_weight_identities(rows: Any, checkpoint_sha256: str) -> list[dict[str, Any]]:
    """Validate, but do not create, a future four-weight checkpoint export."""
    checkpoint_sha256 = lowercase_sha(checkpoint_sha256, "checkpoint SHA")
    require(type(rows) is list and len(rows) == 4, "weight identity population must be four")
    output = []
    for ordinal, row in enumerate(rows):
        require(type(row) is dict and set(row) == {
            "module_ordinal", "module", "checkpoint_sha256", "weight", "bias"},
            "weight identity keys drift")
        require(row["module_ordinal"] == ordinal and row["module"] == MODULES[ordinal] and
                row["checkpoint_sha256"] == checkpoint_sha256,
                "weight module/checkpoint identity drift")
        require(row["bias"] is None, "final decoder bias is not absent")
        weight = row["weight"]
        require(type(weight) is dict and set(weight) == {
            "shape", "dtype", "layout", "byte_order", "content_bytes", "content_sha256"},
            "weight fields drift")
        shape = WEIGHT_SHAPES[ordinal]
        require(weight["shape"] == list(shape) and weight["dtype"] == "torch.float32" and
                weight["layout"] == "C_ORDER_CONTIGUOUS" and
                weight["byte_order"] == "little" and
                weight["content_bytes"] == product(shape) * 4,
                "weight geometry/encoding drift")
        lowercase_sha(weight["content_sha256"], "weight content SHA")
        output.append(dict(row))
    return output


def audit_capture(capture_root: Path = DEFAULT_CAPTURE_ROOT,
                  weight_identities: Any | None = None,
                  checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Read-only source audit.  It intentionally does not verify/admit a result seal."""
    root = Path(capture_root)
    require(root.is_dir() and not root.is_symlink(), "capture root missing/symlink")
    ordered_path = root / "unified_ordered_records.jsonl"
    regular(ordered_path, "ordered records")
    ordered = [strict_json_text(line) for line in ordered_path.read_text(
        encoding="utf-8").splitlines()]
    calls = decoder_rows_from_ordered(ordered)
    theta_words = set()
    audited = []
    for call in calls:
        compressed = safe_member(root, call["compressed_fp32"], "compressed FP32")
        support = safe_member(root, call["support_sign"], "support/sign")
        result = audit_two_plane_payload(
            compressed, support, call["shape"], call["module_ordinal"],
            call["raw_fp32_sha256"], call["compressed_sha256"],
            call["support_sign_sha256"])
        require(result["positive_plane_bytes"] == call["positive_plane_bytes"] and
                result["negative_plane_bytes"] == call["negative_plane_bytes"],
                "recorded support-plane extent drift")
        if result["theta_word_uint32"] is not None:
            theta_words.add(result["theta_word_uint32"])
        audited.append({**call, **result})
    require(len(theta_words) == 1, "D1 theta is not stable across thirty calls")
    weights = None
    if weight_identities is not None or checkpoint_sha256 is not None:
        require(weight_identities is not None and checkpoint_sha256 is not None,
                "weights and checkpoint SHA must be supplied together")
        weights = validate_weight_identities(weight_identities, checkpoint_sha256)
    return {
        "schema": "m1321_ep34_decoder_capture_adapter_source_audit_r1",
        "status": "PASS_SOURCE_AUDIT__ACTUAL_RESULT_HAMMER_AND_SUCCESSOR_REQUIRED",
        "capture_root": str(root),
        "population": {"samples": 30, "calls": 120, "modules": 4,
                       "global_sample_ids": [10, 39]},
        "d1": {"calls": 30, "theta_word_uint32": next(iter(theta_words)),
               "theta_ieee754_le_hex": struct.pack("<I", next(iter(theta_words))).hex(),
               "negative_count": 0, "coerced_to_one": False,
               "weight_folding": False},
        "calls": audited,
        "weight_identities": weights,
        "claim_boundary": {
            "source_only": True, "read_only": True, "capture_result_hammered": False,
            "normalized_payload_written": False, "production_replay": False,
            "cycles": False, "traffic": False, "speedup": False,
            "system_speedup": False, "energy": False, "ppa": False,
            "table_a": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="store_true")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    require(args.source_audit, "only --source-audit is available")
    result = audit_capture(args.capture_root)
    # Do not dump 120 large row records by default; keep this CLI diagnostic.
    print(json.dumps({key: value for key, value in result.items() if key != "calls"},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as error:
        print("M1321_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
