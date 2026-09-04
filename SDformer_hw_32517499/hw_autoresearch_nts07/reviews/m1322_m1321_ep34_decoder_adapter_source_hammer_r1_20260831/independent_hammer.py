#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author blind hammer for sealed M1321.

This hammer is deliberately source-only.  It never contacts the remote host,
uses a GPU, reads the future production result, or launches a replay.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import struct
import subprocess
import sys
import tempfile
from typing import Any
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = HW / "system_simulator/scripts/build_m1321_ep34_decoder_capture_adapter_source.py"
TEST = HW / "system_simulator/tests/test_m1321_ep34_decoder_capture_adapter_source.py"
CONTRACT = HW / "contracts/m1321_ep34_decoder_capture_adapter_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1321_ep34_decoder_capture_adapter_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    "source": "52fb82ab1e4262d6ce838f28a443ce82c6deba00678f9c65fb8227ac30702d85",
    "test": "e704d6420929fdc225f0d9a809f379d17606adca333068b9614b35be00a88edf",
    "contract": "4dde544db5b8f32facbe5fdb10c8adb52d6abb19ca65c4dca7f3b2cce9f06f5c",
    "author_review": "6c4b14e5698580465821f288503aa65b4a64ca4447ee4c1c6563bfaabfd45e1b",
    "author_manifest": "fc661b29e410cd7ce1b9474685a86ff5061800f6aa1f3662d26e76dc368f2200",
    "author_outer_file": "989f16cd2b9739ef22578e435a4abf6f0cf6854760fff41f38185ac19db23497",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64, "malformed manifest row")
        relative = PurePosixPath(parts[1])
        require(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
                "unsafe manifest member")
        require(parts[1] not in rows, "duplicate manifest member")
        rows[parts[1]] = parts[0]
    return rows


def verify_double_seal(root: Path, manifest_sha: str, outer_file_sha: str) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "author root missing/symlink")
    require(sha256(manifest) == manifest_sha, "author manifest SHA drift")
    require(sha256(outer) == outer_file_sha, "author outer-file SHA drift")
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
            "author outer seal content drift")
    rows = parse_manifest(manifest)
    actual = sorted(str(path.relative_to(root).as_posix()) for path in root.rglob("*")
                    if path.is_file() and path.name not in {
                        "SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(rows) == actual, "author recursive member population drift")
    for relative, digest in rows.items():
        member = root / relative
        require(member.is_file() and not member.is_symlink(), "author member missing/symlink")
        require(sha256(member) == digest, "author member SHA drift: " + relative)
    return rows


def load_source():
    spec = importlib.util.spec_from_file_location("m1322_sealed_m1321", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import sealed source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_adapter_error(fn, label: str) -> str:
    try:
        fn()
    except Exception as error:
        require(error.__class__.__name__ == "AdapterError",
                label + " raised wrong exception: " + error.__class__.__name__)
        return str(error)
    raise HammerError(label + " was accepted")


def write_payload(root: Path, words: list[int], stem: str) -> dict[str, Any]:
    raw = b"".join(struct.pack("<I", word) for word in words)
    plane_bytes = (len(words) + 7) // 8
    positive = bytearray(plane_bytes)
    negative = bytearray(plane_bytes)
    for index, word in enumerate(words):
        if word != 0:
            target = negative if word & 0x80000000 else positive
            target[index >> 3] |= 1 << (index & 7)
    compressed = root / (stem + ".fp32.zlib")
    support = root / (stem + ".support_sign.le.bitpack")
    compressed.write_bytes(zlib.compress(raw))
    support.write_bytes(bytes(positive + negative))
    return {
        "compressed": compressed,
        "support": support,
        "raw_sha": hashlib.sha256(raw).hexdigest(),
        "compressed_sha": sha256(compressed),
        "support_sha": sha256(support),
        "plane_bytes": plane_bytes,
    }


def ordered_rows(module, shapes: tuple[tuple[int, ...], ...]) -> list[dict[str, Any]]:
    rows = []
    order = 0
    for sample in range(40):
        for local in range(247):
            if local < 4:
                ordinal = local
                payload = {
                    "retained": True,
                    "compressed_fp32": "payloads/d%d.fp32.zlib" % ordinal,
                    "compressed_sha256": "1" * 64,
                    "support_sign": "payloads/d%d.support_sign.le.bitpack" % ordinal,
                    "support_sign_sha256": "2" * 64,
                    "raw_fp32_sha256": "3" * 64,
                    "positive_plane_bytes": 2,
                    "negative_plane_bytes": 2,
                }
                row = {
                    "global_order": order, "global_sample_id": sample,
                    "category": "decoder_convtranspose", "name": module.MODULES[ordinal],
                    "sequence": "seq_%d" % ((sample - 10) // 10),
                    "sample_key": "sample_%02d" % sample,
                    "source_sha256": "4" * 64,
                    "input": {"shape": list(shapes[ordinal])}, "payload": payload,
                }
            else:
                row = {
                    "global_order": order, "global_sample_id": sample,
                    "category": "atlif", "name": "other.%d" % local,
                    "sequence": "seq", "sample_key": "sample_%02d" % sample,
                    "source_sha256": "4" * 64,
                    "input": {"shape": [1]},
                    "payload": {"retained": False},
                }
            rows.append(row)
            order += 1
    require(len(rows) == 9880, "synthetic ordered population construction failed")
    return rows


def weight_rows(module, checkpoint: str) -> list[dict[str, Any]]:
    rows = []
    for ordinal, shape in enumerate(module.WEIGHT_SHAPES):
        rows.append({
            "module_ordinal": ordinal, "module": module.MODULES[ordinal],
            "checkpoint_sha256": checkpoint,
            "weight": {
                "shape": list(shape), "dtype": "torch.float32",
                "layout": "C_ORDER_CONTIGUOUS", "byte_order": "little",
                "content_bytes": module.product(shape) * 4,
                "content_sha256": ("%x" % (ordinal + 1)) * 64,
            },
            "bias": None,
        })
    return rows


def main() -> int:
    for label, path in (("source", SOURCE), ("test", TEST), ("contract", CONTRACT),
                        ("docs359", DOCS359)):
        require(path.is_file() and not path.is_symlink(), label + " missing/symlink")
        require(sha256(path) == EXPECTED[label], label + " SHA drift")
    author_rows = verify_double_seal(AUTHOR, EXPECTED["author_manifest"],
                                     EXPECTED["author_outer_file"])
    require(author_rows.get("review.json") == EXPECTED["author_review"],
            "author review member mismatch")

    baseline = subprocess.run([str(PYTHON), "-I", str(TEST)], cwd=ROOT,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
    require(baseline.returncode == 0 and "Ran 8 tests" in baseline.stdout and
            baseline.stdout.rstrip().endswith("OK"), "author tests did not pass 8/8")
    module = load_source()
    passed: list[str] = []
    findings: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="m1322_blind_") as temp_name:
        temp = Path(temp_name)
        exact = write_payload(temp, [0, module.ONE_WORD, 0, module.ONE_WORD, 0,
                                     0, 0, 0, 0], "exact")
        result = module.audit_two_plane_payload(
            exact["compressed"], exact["support"], (1, 1, 1, 1, 9), 0,
            exact["raw_sha"], exact["compressed_sha"], exact["support_sha"])
        require(result["active"] == 2 and result["negative_count"] == 0,
                "exact two-plane positive control failed")
        passed.append("two_plane_positive_control")

        shifted = write_payload(temp, [0, module.ONE_WORD, 0], "shifted")
        attack = bytearray(shifted["support"].read_bytes())
        attack[0] = 0b00000100
        shifted["support"].write_bytes(attack)
        expect_adapter_error(lambda: module.audit_two_plane_payload(
            shifted["compressed"], shifted["support"], (1, 1, 1, 1, 3), 0,
            shifted["raw_sha"]), "shifted positive plane")
        passed.append("support_plane_bit_offset_rejected")

        negative = write_payload(temp, [0, 0xBF800000], "negative")
        expect_adapter_error(lambda: module.audit_two_plane_payload(
            negative["compressed"], negative["support"], (1, 1, 1, 1, 2), 0,
            negative["raw_sha"]), "negative decoder plane")
        passed.append("negative_plane_rejected")

        padded = write_payload(temp, [module.ONE_WORD], "padding")
        bits = bytearray(padded["support"].read_bytes()); bits[0] |= 0x80
        padded["support"].write_bytes(bits)
        expect_adapter_error(lambda: module.audit_two_plane_payload(
            padded["compressed"], padded["support"], (1, 1, 1, 1, 1), 0,
            padded["raw_sha"]), "nonzero support padding")
        passed.append("padding_rejected")

        trail = write_payload(temp, [module.ONE_WORD], "trailing")
        trail["compressed"].write_bytes(trail["compressed"].read_bytes() + b"junk")
        expect_adapter_error(lambda: module.audit_two_plane_payload(
            trail["compressed"], trail["support"], (1, 1, 1, 1, 1), 0,
            trail["raw_sha"]), "zlib trailing bytes")
        passed.append("zlib_trailing_rejected")

        raw = write_payload(temp, [module.ONE_WORD], "rawsha")
        expect_adapter_error(lambda: module.audit_two_plane_payload(
            raw["compressed"], raw["support"], (1, 1, 1, 1, 1), 0,
            "0" * 64), "raw FP32 SHA drift")
        passed.append("raw_sha_rejected")

        theta = write_payload(temp, [0, 0x3F400000, 0, 0x3F400000], "theta")
        theta_result = module.audit_two_plane_payload(
            theta["compressed"], theta["support"], (1, 1, 1, 1, 4), 1,
            theta["raw_sha"], theta["compressed_sha"], theta["support_sha"])
        require(theta_result["theta_word_uint32"] == 0x3F400000,
                "dynamic D1 theta control failed")
        multi = write_payload(temp, [0x3F400000, 0x3F000000], "multi_theta")
        expect_adapter_error(lambda: module.audit_two_plane_payload(
            multi["compressed"], multi["support"], (1, 1, 1, 1, 2), 1,
            multi["raw_sha"]), "multiple D1 theta words")
        all_zero = write_payload(temp, [0, 0, 0], "zero_theta")
        zero_result = module.audit_two_plane_payload(
            all_zero["compressed"], all_zero["support"], (1, 1, 1, 1, 3), 1,
            all_zero["raw_sha"])
        require(zero_result["theta_word_uint32"] is None,
                "all-zero D1 call was assigned a fake theta")
        passed.extend(["d1_dynamic_theta_exact", "d1_multi_theta_rejected",
                       "d1_all_zero_call_not_coerced"])

    shapes = ((1, 1, 1, 1, 9),) * 4
    rows = ordered_rows(module, shapes)
    old_shapes = module.SHAPES
    module.SHAPES = shapes
    try:
        selected = module.decoder_rows_from_ordered(rows)
        require(len(selected) == 120 and selected[0]["global_sample_id"] == 10 and
                selected[-1]["global_sample_id"] == 39,
                "global sample 10..39 / 120 call positive control failed")
        passed.append("global_samples_10_39_and_120_calls")
        bad_type = copy.deepcopy(rows); bad_type[10 * 247]["global_sample_id"] = True
        expect_adapter_error(lambda: module.decoder_rows_from_ordered(bad_type),
                             "boolean global sample id")
        passed.append("sample_bool_rejected")
        duplicate_key = '{"a":1,"a":2}'
        expect_adapter_error(lambda: module.strict_json_text(duplicate_key),
                             "duplicate JSON key")
        passed.append("duplicate_json_key_rejected")

        duplicate_order = copy.deepcopy(rows)
        duplicate_order[10 * 247 + 1]["global_order"] = duplicate_order[10 * 247]["global_order"]
        try:
            module.decoder_rows_from_ordered(duplicate_order)
        except module.AdapterError:
            passed.append("duplicate_selected_global_order_rejected")
        else:
            findings.append({
                "id": "F1_DUPLICATE_GLOBAL_ORDER_ACCEPTED",
                "severity": "P0_EXACT_GRAPH",
                "evidence": "D1 in global sample 10 was assigned D0's global_order; adapter returned 120 calls",
                "required_repair": "Require the entire 9880-row stream global_order to be exact int and equal to file ordinal 0..9879 (therefore unique).",
            })

        duplicate_ignored = copy.deepcopy(rows)
        duplicate_ignored[5] = copy.deepcopy(duplicate_ignored[4])
        try:
            module.decoder_rows_from_ordered(duplicate_ignored)
        except module.AdapterError:
            passed.append("duplicate_ignored_jsonl_row_rejected")
        else:
            findings.append({
                "id": "F2_DUPLICATE_IGNORED_JSONL_ROW_ACCEPTED",
                "severity": "P1_ORDERED_POPULATION",
                "evidence": "One ignored ordered row was replaced by a duplicate while total remained 9880; projection passed",
                "required_repair": "The same exact global_order==file-ordinal invariant closes duplicate/replacement aliases before projection.",
            })
    finally:
        module.SHAPES = old_shapes

    checkpoint = "a" * 64
    weights = weight_rows(module, checkpoint)
    require(len(module.validate_weight_identities(weights, checkpoint)) == 4,
            "four-weight identity positive control failed")
    passed.append("four_weight_identity_positive_control")
    wrong_checkpoint = copy.deepcopy(weights); wrong_checkpoint[2]["checkpoint_sha256"] = "b" * 64
    expect_adapter_error(lambda: module.validate_weight_identities(wrong_checkpoint, checkpoint),
                         "weight checkpoint mismatch")
    extra_key = copy.deepcopy(weights); extra_key[0]["extra"] = 1
    expect_adapter_error(lambda: module.validate_weight_identities(extra_key, checkpoint),
                         "weight key drift")
    passed.extend(["weight_checkpoint_mismatch_rejected", "weight_key_drift_rejected"])
    bool_ordinal = copy.deepcopy(weights); bool_ordinal[1]["module_ordinal"] = True
    try:
        module.validate_weight_identities(bool_ordinal, checkpoint)
    except module.AdapterError:
        passed.append("weight_bool_ordinal_rejected")
    else:
        findings.append({
            "id": "F3_BOOLEAN_WEIGHT_ORDINAL_ACCEPTED",
            "severity": "P0_EXACT_IDENTITY",
            "evidence": "module_ordinal=True was accepted as exact ordinal 1 because bool == 1",
            "required_repair": "Require type(module_ordinal) is int before exact ordinal equality.",
        })

    expect_adapter_error(lambda: module.main([]), "CLI without source-audit")
    passed.append("cli_default_inert")
    cli = subprocess.run([str(PYTHON), "-I", str(SOURCE), "--production-replay"],
                         cwd=ROOT, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    require(cli.returncode != 0 and "unrecognized arguments" in cli.stdout,
            "production-replay CLI surface was unexpectedly accepted")
    passed.append("cli_production_mode_absent")

    claim_keys = {
        "source_only", "read_only", "capture_result_hammered",
        "normalized_payload_written", "production_replay", "cycles", "traffic",
        "speedup", "system_speedup", "energy", "ppa", "table_a",
    }
    source_text = SOURCE.read_text(encoding="utf-8")
    require('"source_only": True' in source_text and
            all(('"%s": False' % key) in source_text for key in
                claim_keys - {"source_only", "read_only"}) and
            '"read_only": True' in source_text,
            "source claim boundary is not fail-closed")
    passed.append("claim_boundary_static_fail_closed")

    status = ("FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED" if findings else
              "PASS_M1322_M1321_SOURCE_HAMMER__SUCCESSOR_AUTHORING_ALLOWED")
    output = {
        "schema": "m1322_m1321_ep34_decoder_adapter_source_hammer_r1_v1",
        "status": status,
        "source_authority": {
            "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
            "test_path": str(TEST.relative_to(ROOT)), "test_sha256": sha256(TEST),
            "contract_path": str(CONTRACT.relative_to(ROOT)), "contract_sha256": sha256(CONTRACT),
            "author_review_path": str((AUTHOR / "review.json").relative_to(ROOT)),
            "author_review_sha256": sha256(AUTHOR / "review.json"),
            "author_manifest_sha256": sha256(AUTHOR / "SHA256SUMS"),
            "author_outer_file_sha256": sha256(AUTHOR / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha256(DOCS359),
        },
        "independence": {"different_author": True},
        "author_tests": {"passed": True, "count": 8, "output": baseline.stdout},
        "blind_checks_passed": passed,
        "findings": findings,
        "authorization": {
            "m1321_citable": False if findings else True,
            "production_replay": False,
            "remote_access": False,
            "gpu": False,
            "additive_successor_required": bool(findings),
        },
        "claim_boundary": {
            "source_only": True, "actual_capture_read": False,
            "actual_result_hammer_bound": False, "normalized_payload_written": False,
            "production_replay": False, "cycles": False, "traffic": False,
            "speedup": False, "system_speedup": False, "energy": False,
            "ppa": False, "table_a": False, "paper_citable_performance": False,
        },
    }
    (OUT / "hammer_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "author_test_output.txt").write_text(baseline.stdout, encoding="utf-8")
    return 2 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
