#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive successor to rejected M1335; read-only final M1327 result hammer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence
import zlib


SOURCE_FILE = Path(__file__).resolve()
ROOT = SOURCE_FILE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
M1335_SOURCE = HW / "scripts/hammer_m1335_m1327_final_ep34_capture_result_source.py"
M1335_SOURCE_SHA256 = "05f97bf187d63cca5a378c0b7f5f39dd12ecdb63cc3524b5b585d2d296d9b77e"
M1335_TEST = HW / "tests/test_hammer_m1335_m1327_final_ep34_capture_result_source.py"
M1335_TEST_SHA256 = "cdda667f926a2a6a53d0cca252c8ca31ba6a279e20ce81d05495c04d121f80ca"
M1335_CONTRACT = HW / "contracts/m1335_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
M1335_CONTRACT_SHA256 = "fa2ab8f5033d51453588c79031555c629aaaa974b39ddb2de70cf0424dc768bc"
M1336_FAIL = HW / "reviews/m1336_m1335_m1327_final_ep34_capture_result_hammer_source_blind_review_r1_20260831"
M1336_FAIL_REVIEW_SHA256 = "fce98fcc13180dd7d8664556dcda71b627539220382ed9dd23968317bce7c0dc"
M1336_FAIL_MANIFEST_SHA256 = "2da8afd66cb3c1bab2d6043e9c5bcfc0e627b4ebb70fbd1a8fdf15aaa6ea9b6f"
M1336_FAIL_OUTER_FILE_SHA256 = "b1548f14e42ed148003947c4cd8e62668976ce2ae8a135c58437d60d1fbde353"
CANONICAL_RESULT = HW / "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"
SOURCE_CONTRACT = HW / "contracts/m1338_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
TEST = HW / "tests/test_hammer_m1338_m1327_final_ep34_capture_result_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SCHEMA = "m1338_m1327_final_ep34_capture_result_hammer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1335_REJECTED__CONTENT_DERIVED_EXTENTS__NO_CAPTURE"
PASS_TOKEN = "PASS_M1338_SOURCE_SELF_CHECK__FIXTURES_ONLY_NO_CANONICAL_RESULT"
ATTENTION_KEYS = {"q_shape", "k_shape", "q_bits_packed", "k_bits_packed", "gate_q17"}


class M1338Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1338Error(message)


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
        raise M1338Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(), label + " not regular")
    require(sha256(path) == digest, label + " SHA drift")


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_exact("m1338_sealed_m1335", M1335_SOURCE, M1335_SOURCE_SHA256)


def strict_json(path: Path) -> dict[str, Any]:
    value = OLD.strict_json(path)
    require(type(value) is dict, "JSON root is not object")
    return value


def verify_failed_predecessor() -> dict[str, Any]:
    regular_exact(M1335_TEST, M1335_TEST_SHA256, "M1335 test")
    regular_exact(M1335_CONTRACT, M1335_CONTRACT_SHA256, "M1335 contract")
    rows, seal = OLD.OLD.verify_recursive_seal(M1336_FAIL)
    require(seal == {"manifest_sha256": M1336_FAIL_MANIFEST_SHA256,
                     "outer_file_sha256": M1336_FAIL_OUTER_FILE_SHA256},
            "M1336 final blind seal drift")
    require(rows.get("review.json") == M1336_FAIL_REVIEW_SHA256,
            "M1336 final blind review member drift")
    review = strict_json(M1336_FAIL / "review.json")
    require(review.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED" and
            review.get("authorization", {}).get("additive_successor_source_authoring") is True and
            review.get("authorization", {}).get("production_result_hammer") is False and
            review.get("false_negative_count") == 6 and
            review.get("accepted_attack_count") == 9,
            "M1336 final blind authority drift")
    return review


def canonical_absent(path: Path = CANONICAL_RESULT) -> None:
    require(not os.path.lexists(str(path)),
            "canonical namespace residue exists, including possible broken symlink")


def canonical_directory(path: Path = CANONICAL_RESULT) -> None:
    require(os.path.lexists(str(path)), "canonical result absent")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1338Error("canonical result disappeared") from error
    require(stat.S_ISDIR(mode) and not path.is_symlink(),
            "canonical result is not a real directory")


def _tail_zero(packed: Any, elements: int, label: str) -> None:
    remainder = elements & 7
    if remainder:
        require((int(packed[-1]) >> remainder) == 0, label + " has nonzero padding bits")


def validate_one_retained_payload(root: Path, seal_rows: dict[str, str],
                                  row: dict[str, Any]) -> None:
    import numpy as np
    meta = row["input"]
    payload = row["payload"]
    require(meta.get("dtype") == "torch.float32", "retained input dtype is not torch.float32")
    elements = meta.get("elements")
    byte_count = meta.get("bytes")
    require(type(elements) is int and elements > 0 and type(byte_count) is int and
            byte_count == elements * 4, "retained input FP32 byte extent drift")
    plane_bytes = (elements + 7) // 8
    require(payload.get("positive_plane_bytes") == plane_bytes and
            payload.get("negative_plane_bytes") == plane_bytes,
            "retained support plane extent is not derived from elements")

    for path_key, sha_key in (("compressed_fp32", "compressed_sha256"),
                              ("support_sign", "support_sign_sha256")):
        relative = payload[path_key]
        member = OLD.OLD.safe_member(root, relative)
        record_sha = payload[sha_key]
        require(seal_rows.get(relative) == record_sha == sha256(member),
                "retained record/seal/actual SHA mismatch: " + relative)

    support_path = OLD.OLD.safe_member(root, payload["support_sign"])
    support = support_path.read_bytes()
    require(len(support) == 2 * plane_bytes,
            "retained support is not exact positive||negative extent")
    positive = np.frombuffer(support[:plane_bytes], dtype=np.uint8)
    negative = np.frombuffer(support[plane_bytes:], dtype=np.uint8)
    _tail_zero(positive, elements, "positive support plane")
    _tail_zero(negative, elements, "negative support plane")
    positive_bits = np.unpackbits(positive, bitorder="little")[:elements]
    negative_bits = np.unpackbits(negative, bitorder="little")[:elements]
    require(not bool(np.any(np.logical_and(positive_bits, negative_bits))),
            "positive and negative support planes overlap")

    compressed_path = OLD.OLD.safe_member(root, payload["compressed_fp32"])
    decompressor = zlib.decompressobj()
    digest = hashlib.sha256()
    carry = b""
    words = 0
    raw_bytes = 0
    active = positive_count = negative_count = nonfinite = 0

    def consume(block: bytes) -> None:
        nonlocal carry, words, raw_bytes, active, positive_count, negative_count, nonfinite
        digest.update(block)
        raw_bytes += len(block)
        joined = carry + block
        usable = len(joined) - len(joined) % 4
        if usable:
            values = np.frombuffer(joined[:usable], dtype="<f4")
            require(words + values.size <= elements, "retained raw FP32 exceeds elements")
            expected_positive = np.greater(values, 0)
            expected_negative = np.less(values, 0)
            span = slice(words, words + values.size)
            require(np.array_equal(positive_bits[span], expected_positive) and
                    np.array_equal(negative_bits[span], expected_negative),
                    "retained support signs differ from raw FP32")
            active += int(np.count_nonzero(np.not_equal(values, 0)))
            positive_count += int(np.count_nonzero(expected_positive))
            negative_count += int(np.count_nonzero(expected_negative))
            nonfinite += int(np.count_nonzero(np.logical_not(np.isfinite(values))))
            words += int(values.size)
        carry = joined[usable:]

    try:
        with compressed_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                consume(decompressor.decompress(chunk))
        consume(decompressor.flush())
    except M1338Error:
        raise
    except Exception as error:
        raise M1338Error("retained compressed payload is not valid zlib") from error
    require(decompressor.eof and not decompressor.unused_data and
            not decompressor.unconsumed_tail,
            "retained zlib stream is truncated or has trailing data")
    require(not carry and words == elements and raw_bytes == byte_count,
            "retained raw FP32 length differs from input elements/bytes")
    require(digest.hexdigest() == payload["raw_fp32_sha256"],
            "retained raw_fp32_sha256 mismatch")
    require((active, positive_count, negative_count, nonfinite) ==
            (meta["active"], meta["positive"], meta["negative"], meta["nonfinite"]),
            "retained raw FP32 statistics differ from input record")


def validate_retained_payloads(root: Path, seal_rows: dict[str, str],
                               ordered: list[dict[str, Any]]) -> int:
    retained = 0
    for row in ordered:
        if row["payload"].get("retained") is not True:
            continue
        validate_one_retained_payload(root, seal_rows, row)
        retained += 1
    require(retained == 320, "retained payload population is not 320")
    return retained


def validate_attention_npz(payload: Path, row: dict[str, Any]) -> None:
    import numpy as np
    windows = row["windows_captured"]
    heads = row["heads"]
    spatial = row["spatial_tokens"]
    lanes = row["lanes"]
    with np.load(payload, allow_pickle=False) as data:
        require(set(data.files) == ATTENTION_KEYS, "attention NPZ member set is not exact")
        q_shape = data["q_shape"]
        k_shape = data["k_shape"]
        q_bits = data["q_bits_packed"]
        k_bits = data["k_bits_packed"]
        gate = data["gate_q17"]
        expected_shape = [2, windows, heads, spatial, lanes]
        require(q_shape.dtype == np.dtype("int32") and k_shape.dtype == np.dtype("int32") and
                q_shape.ndim == 1 and k_shape.ndim == 1 and
                q_shape.tolist() == expected_shape and k_shape.tolist() == expected_shape,
                "attention q/k shape metadata drift")
        elements = math.prod(expected_shape)
        require(q_bits.dtype == np.dtype("uint8") and k_bits.dtype == np.dtype("uint8") and
                q_bits.ndim == 1 and k_bits.ndim == 1 and
                q_bits.size == (elements + 7) // 8 and
                k_bits.size == (elements + 7) // 8,
                "attention packed q/k dtype or extent drift")
        _tail_zero(q_bits, elements, "attention q_bits")
        _tail_zero(k_bits, elements, "attention k_bits")
        require(gate.dtype == np.dtype("uint16") and
                gate.shape == (windows, heads, row["temporal_tokens"]) and
                gate.size > 0 and int(gate.max()) <= 256,
                "attention gate dtype/shape/range drift")


def validate_attention_exact_archive(root: Path) -> int:
    manifest = strict_json(root / "attention_qk/manifest.json")
    records = manifest["records"]
    require(type(records) is list and len(records) == 480,
            "attention population is not 480")
    for row in records:
        safe_name = row["name"].replace(".", "_").replace("/", "_")
        relative = "attention_qk/sample{}_{}.npz".format(row["sample_id"], safe_name)
        validate_attention_npz(OLD.OLD.safe_member(root, relative), row)
    return len(records)


def validate_result(root: Path) -> dict[str, Any]:
    canonical_directory(root)
    try:
        inherited = OLD.validate_result(root)
    except Exception as error:
        raise M1338Error("M1335 retained validation boundary failed") from error
    seal_rows, seal = OLD.OLD.verify_recursive_seal(root)
    ordered = [OLD.OLD.strict_text(line) for line in
               (root / "unified_ordered_records.jsonl").read_text(encoding="utf-8").splitlines()]
    retained = validate_retained_payloads(root, seal_rows, ordered)
    attention = validate_attention_exact_archive(root)
    return {"status": "PASS_M1338_M1327_EP34_CAPTURE_RESULT",
            "seal": seal, "population": {**inherited["population"], "retained": retained,
                                           "attention": attention},
            "identity": inherited["identity"],
            "claim_boundary": {"capture_only": True, "paper_result": False}}


def validate_source_policy() -> dict[str, Any]:
    verify_failed_predecessor()
    OLD.validate_runtime_identity()
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "source policy schema/status drift")
    require(policy.get("source") == {"path": str(SOURCE_FILE.relative_to(ROOT)),
                                     "sha256": sha256(SOURCE_FILE)} and
            policy.get("test") == {"path": str(TEST.relative_to(ROOT)),
                                   "sha256": sha256(TEST)},
            "source/test policy identity drift")
    require(policy.get("predecessor_m1335") == "FAIL_DO_NOT_CITE" and
            policy.get("actual_result_seal_prefilled") is False and
            policy.get("production_authorized") is False,
            "source-only predecessor/result boundary drift")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--validate-canonical-result", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.source_self_check:
        validate_source_policy()
        canonical_absent()
        print(PASS_TOKEN)
        return 0
    canonical_directory()
    print(json.dumps(validate_result(CANONICAL_RESULT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1338Error as error:
        print("M1338_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
