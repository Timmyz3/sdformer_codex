#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author M1335 source hammer; disposable fixtures only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zlib

import numpy as np


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/hammer_m1335_m1327_final_ep34_capture_result_source.py"
TEST = HW / "tests/test_hammer_m1335_m1327_final_ep34_capture_result_source.py"
CONTRACT = HW / "contracts/m1335_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1335_m1327_final_ep34_capture_result_hammer_source_author_r1_20260831"
CANONICAL = HW / "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"
OUTPUT = Path(__file__).resolve().parent / "hammer_output.json"

EXPECTED = {
    SOURCE: "05f97bf187d63cca5a378c0b7f5f39dd12ecdb63cc3524b5b585d2d296d9b77e",
    TEST: "cdda667f926a2a6a53d0cca252c8ca31ba6a279e20ce81d05495c04d121f80ca",
    CONTRACT: "fa2ab8f5033d51453588c79031555c629aaaa974b39ddb2de70cf0424dc768bc",
    HW / "docs/359_DATE终局冻结_20260813.md":
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1336_target", SOURCE)
H = load("m1336_author_fixture", TEST)


def clone(base: Path, parent: Path, name: str) -> Path:
    root = parent / name
    shutil.copytree(base, root)
    return root


def ordered_rows(root: Path) -> list[dict]:
    return [json.loads(line) for line in
            (root / "unified_ordered_records.jsonl").read_text().splitlines()]


def write_ordered(root: Path, rows: list[dict]) -> None:
    (root / "unified_ordered_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")


def reseal(root: Path) -> None:
    H.B.seal(root)


def first_scalar_retained(rows: list[dict]) -> dict:
    return next(row for row in rows if row["payload"].get("retained") is True and
                row["input"]["elements"] == 1)


def rewrite_payload(root: Path, *, raw: bytes | None = None,
                    compressed: bytes | None = None, support: bytes | None = None,
                    planes: tuple[int, int] | None = None,
                    input_update: dict | None = None) -> None:
    rows = ordered_rows(root)
    row = first_scalar_retained(rows)
    payload = row["payload"]
    if input_update:
        row["input"].update(input_update)
    if raw is not None:
        compressed_bytes = zlib.compress(raw) if compressed is None else compressed
        compressed_path = root / payload["compressed_fp32"]
        compressed_path.write_bytes(compressed_bytes)
        payload["raw_fp32_sha256"] = hashlib.sha256(raw).hexdigest()
        payload["compressed_sha256"] = sha(compressed_path)
    elif compressed is not None:
        compressed_path = root / payload["compressed_fp32"]
        compressed_path.write_bytes(compressed)
        payload["compressed_sha256"] = sha(compressed_path)
    if support is not None:
        support_path = root / payload["support_sign"]
        support_path.write_bytes(support)
        payload["support_sign_sha256"] = sha(support_path)
    if planes is not None:
        payload["positive_plane_bytes"], payload["negative_plane_bytes"] = planes
    write_ordered(root, rows)
    reseal(root)


def rewrite_attention(root: Path, mode: str) -> None:
    manifest_path = root / "attention_qk/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row = manifest["records"][0]
    payload = root / "attention_qk" / Path(row["file"]).name
    with np.load(payload, allow_pickle=False) as data:
        values = {name: data[name] for name in data.files}
    if mode == "extra_member":
        values["invented_payload"] = np.array([1], dtype=np.uint8)
    elif mode == "nonzero_tail":
        row["lanes"] = 7
        values["q_shape"] = np.array([2, 1, 1, 1, 7], dtype=np.int32)
        values["k_shape"] = np.array([2, 1, 1, 1, 7], dtype=np.int32)
        values["q_bits_packed"] = np.array([0, 0b11000000], dtype=np.uint8)
        values["k_bits_packed"] = np.array([0, 0b11000000], dtype=np.uint8)
        row["q_active_bits"] = 0
        row["k_active_bits"] = 0
    else:
        raise AssertionError("unknown attention mutation")
    np.savez_compressed(payload, **values)
    row["sha256"] = sha(payload)
    H.write_json(manifest_path, manifest)
    reseal(root)


def accepted(root: Path, label: str, result: list[str]) -> None:
    M.validate_result(root)
    result.append(label)


def main() -> None:
    need(not os.path.lexists(str(CANONICAL)), "forbidden canonical residue before review")
    for path, digest in EXPECTED.items():
        need(sha(path) == digest, "identity drift: " + str(path))
    M.validate_runtime_identity()
    predecessor = M.verify_failed_predecessor()
    need(predecessor["false_negative_count"] == 5, "M1334 FAIL authority drift")
    author_rows, author_seal = M.OLD.verify_recursive_seal(AUTHOR)
    need(author_rows.get("receipt.json") ==
         "23719f7f3a7d53706be2aaca6940c4d42c22d1b494472c2611ead2a03801ff50",
         "author receipt drift")
    need(author_rows.get("review.json") ==
         "be4b410881248ba7276e1a20ba6dfb6c6af818a4726b590e29a3748e1a46f05f",
         "author review drift")
    need(author_seal == {
        "manifest_sha256": "79f328edec60229794454a5c8d7dd0ac57535100ead5914cdf6277dec40d5741",
        "outer_file_sha256": "caba41a3e760d5fff199f3cdf3745249e4caa8a5e569f96bb3114675f2d14537"},
        "author recursive seal drift")

    fixture = H.StrongFixture()
    passed_attacks: list[str] = []
    try:
        baseline = M.validate_result(fixture.root)
        need(baseline["population"]["retained"] == 320 and
             baseline["population"]["attention"] == 480, "author fixture failed")
        with tempfile.TemporaryDirectory(prefix="m1336_cases_") as td_raw:
            td = Path(td_raw)

            root = clone(fixture.root, td, "raw_length")
            rewrite_payload(root, raw=b"abc")
            accepted(root, "raw_length_not_equal_input_bytes", passed_attacks)

            root = clone(fixture.root, td, "plane_extent")
            rewrite_payload(root, support=b"\0\0\0\0", planes=(2, 2))
            accepted(root, "plane_extent_not_ceil_elements_over_8", passed_attacks)

            root = clone(fixture.root, td, "zlib_trailing")
            raw = b"\0\0\0\0"
            rewrite_payload(root, raw=raw, compressed=zlib.compress(raw) + b"TRAILING",
                            support=b"\0\0", planes=(1, 1))
            accepted(root, "zlib_trailing_garbage", passed_attacks)

            root = clone(fixture.root, td, "raw_stats")
            rewrite_payload(root, raw=struct.pack("<f", 1.0), support=b"\x01\x00",
                            planes=(1, 1), input_update={"active": 0, "positive": 0,
                            "negative": 0, "nonfinite": 0})
            accepted(root, "raw_values_disagree_with_input_statistics", passed_attacks)

            root = clone(fixture.root, td, "support_semantics")
            rewrite_payload(root, raw=struct.pack("<f", 1.0), support=b"\x00\x01",
                            planes=(1, 1), input_update={"active": 1, "positive": 1,
                            "negative": 0, "nonfinite": 0})
            accepted(root, "support_sign_disagrees_with_raw", passed_attacks)

            root = clone(fixture.root, td, "support_padding")
            rewrite_payload(root, raw=struct.pack("<f", 0.0), support=b"\xfe\x00",
                            planes=(1, 1), input_update={"active": 0, "positive": 0,
                            "negative": 0, "nonfinite": 0})
            accepted(root, "support_nonzero_padding_bits", passed_attacks)

            root = clone(fixture.root, td, "dtype_label")
            rewrite_payload(root, raw=struct.pack("<f", 0.0), support=b"\0\0",
                            planes=(1, 1), input_update={"dtype": "torch.float16"})
            accepted(root, "fp32_payload_with_float16_input_label", passed_attacks)

            root = clone(fixture.root, td, "attention_extra")
            rewrite_attention(root, "extra_member")
            accepted(root, "attention_invented_npz_member", passed_attacks)

            root = clone(fixture.root, td, "attention_padding")
            rewrite_attention(root, "nonzero_tail")
            accepted(root, "attention_nonzero_packbits_tail", passed_attacks)
    finally:
        fixture.close()

    need(len(passed_attacks) == 9, "expected nine independently accepted attacks")
    need(not os.path.lexists(str(CANONICAL)), "forbidden canonical residue after review")
    OUTPUT.write_text(json.dumps({
        "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "author_new_tests": "18/18 PASS",
        "inherited_tests": "13/13 PASS",
        "source_self_check": "PASS",
        "independently_accepted_attacks": passed_attacks,
        "accepted_attack_count": len(passed_attacks),
        "false_negative_groups": 6,
        "canonical_lexically_absent_before_and_after": True,
        "canonical_read_or_created": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("M1336_FAIL_M1335__9_ACCEPTED_ATTACKS__6_FALSE_NEGATIVE_GROUPS")


if __name__ == "__main__":
    main()
