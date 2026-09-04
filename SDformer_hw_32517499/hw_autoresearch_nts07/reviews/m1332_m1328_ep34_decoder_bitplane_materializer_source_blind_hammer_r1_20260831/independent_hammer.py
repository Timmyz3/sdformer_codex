#!/usr/bin/env python3
"""Different-author, source-only hammer for M1328.

This never opens an M1327 capture and never calls execute_once with a valid
release.  All writer, seal, and mutation checks use private temporary roots.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1328_ep34_decoder_bitplane_materializer_source.py"
TEST = HW / "system_simulator/tests/test_m1328_ep34_decoder_bitplane_materializer_source.py"
CONTRACT = HW / "contracts/m1328_ep34_decoder_bitplane_materializer_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1328_ep34_decoder_bitplane_materializer_source_author_r1_20260831"
EXPECTED = {
    "source": "abf81c781046d66a223f9b616e9d3ffa2a876cca62691883fa4279dbb460af43",
    "test": "524638e0823242efde38b8209f9c139a4ea0935d713d3fb89d389166afed1864",
    "contract": "4b446ef0f1e491ce3340443f062bdd141a62a5c44a4dfa0e1b3dd12683409dbb",
    "author_manifest": "178b8f6a3eb5f165bcd0212e386ba2b59fb2ef5cabe28ea3acba77f542190dab",
    "author_outer_file": "5f5c575beec8134f1c56fd9decce0504f7014dc0c37f5d47f5a8fb17f7c20715",
    "author_receipt": "f685a5c88c55a2aaa3fce568d44c1f1dca143eed6a891fb80381e25f7565fa0f",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, label: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(label)
    checks.append(label)


def reject(callable_, label: str, checks: list[str]) -> None:
    try:
        callable_()
    except BaseException:
        checks.append(label)
        return
    raise AssertionError("mutation accepted: " + label)


def load_source():
    spec = importlib.util.spec_from_file_location("m1332_m1328", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_audit(m, theta: int = 0x3F400000):
    calls = []
    for ordinal in range(120):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        shape = list(m.M1323.SHAPES[module])
        plane_bytes = (math.prod(shape) + 7) // 8
        positive = bytes([ordinal & 0xFF]) * plane_bytes
        negative = bytes(plane_bytes)
        calls.append({
            "global_call_ordinal": ordinal,
            "global_order": sample * 247 + module,
            "global_sample_id": sample,
            "sequence": "seq_%d" % ((sample - 10) // 10),
            "sample_key": "sample_%02d.npy" % sample,
            "source_sha256": "1" * 64,
            "module_ordinal": module,
            "module": m.M1323.MODULES[module],
            "shape": shape,
            "elements": math.prod(shape),
            "support_sign": "payloads/source_%03d.support_sign.le.bitpack" % ordinal,
            "support_sign_sha256": hashlib.sha256(positive + negative).hexdigest(),
            "raw_fp32_sha256": "3" * 64,
            "positive_plane_bytes": plane_bytes,
            "negative_plane_bytes": plane_bytes,
            "positive_plane_sha256": hashlib.sha256(positive).hexdigest(),
            "negative_plane_sha256": hashlib.sha256(negative).hexdigest(),
            "negative_count": 0,
        })
    return {
        "calls": calls,
        "d1": {"theta_word_uint32": theta,
               "theta_ieee754_le_hex": struct.pack("<I", theta).hex()},
        "ordered_jsonl_sha256": "6" * 64,
        "ordered_identity": {"ordered_rows": 9880},
    }


def authority(m):
    return {
        "release": {
            "capture_result": {
                "path": str(m.M1327_CAPTURE.relative_to(m.ROOT)),
                "manifest_sha256": "7" * 64,
                "outer_file_sha256": "8" * 64,
                "capture_manifest_sha256": "9" * 64,
                "admission_sha256": "a" * 64,
            },
            "capture_result_hammer": {
                "path": "hw_autoresearch_nts07/reviews/future",
                "manifest_sha256": "b" * 64,
                "outer_file_sha256": "c" * 64,
                "review_sha256": "d" * 64,
            },
        },
        "result_hammer": {"identity": {
            "epoch": 34,
            "checkpoint_sha256": "e" * 64,
            "config_sha256": "f" * 64,
            "profile_sha256": "0" * 64,
        }},
    }


def author_seal(checks: list[str]) -> None:
    check(sha(AUTHOR / "SHA256SUMS") == EXPECTED["author_manifest"],
          "author manifest digest exact", checks)
    check(sha(AUTHOR / "SHA256SUMS.seal.sha256") == EXPECTED["author_outer_file"],
          "author outer-seal file digest exact", checks)
    outer = (AUTHOR / "SHA256SUMS.seal.sha256").read_text().split()
    check(outer == [EXPECTED["author_manifest"], "SHA256SUMS"],
          "author outer seal content exact", checks)
    rows = {}
    for line in (AUTHOR / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        check(relative not in rows, "author manifest member unique: " + relative, checks)
        member = AUTHOR / relative
        check(member.is_file() and not member.is_symlink() and sha(member) == digest,
              "author member exact: " + relative, checks)
        rows[relative] = digest
    actual = {p.relative_to(AUTHOR).as_posix() for p in AUTHOR.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    check(actual == set(rows), "author recursive population exact", checks)
    check(rows.get("receipt.json") == EXPECTED["author_receipt"],
          "author receipt member pinned", checks)


def release_shape(m):
    auth = authority(m)
    return {
        "schema": m.RELEASE_SCHEMA,
        "status": m.RELEASE_STATUS,
        "contract_path": str(m.FUTURE_RELEASE.relative_to(m.ROOT)),
        "release_identity": {
            "source_path": str(m.SOURCE_FILE.relative_to(m.ROOT)),
            "source_sha256": m.sha256(m.SOURCE_FILE),
            "test_path": str(m.TEST.relative_to(m.ROOT)),
            "test_sha256": m.sha256(m.TEST),
            "source_contract_path": str(m.SOURCE_CONTRACT.relative_to(m.ROOT)),
            "source_contract_sha256": m.sha256(m.SOURCE_CONTRACT),
        },
        "source_hammer": {"path": "future", "manifest_sha256": "1" * 64,
                          "outer_file_sha256": "2" * 64, "review_sha256": "3" * 64},
        "capture_result": auth["release"]["capture_result"],
        "capture_result_hammer": auth["release"]["capture_result_hammer"],
        "one_shot": {"attempt_marker": str(m.ATTEMPT.relative_to(m.ROOT)),
                     "automatic_retry": False, "maximum_materializations": 1},
        "output": {"path": str(m.OUTPUT.relative_to(m.ROOT)),
                   "atomic_no_replace": True, "recursive_double_seal": True},
        "claim_boundary": {"bitplane_materialization": True,
                           "decoder_replay": False, "cycles": False,
                           "traffic": False, "speedup": False,
                           "system_speedup": False, "energy": False,
                           "rtl": False, "eda": False, "ppa": False},
    }


def main() -> int:
    checks: list[str] = []
    for label, path in (("source", SOURCE), ("test", TEST), ("contract", CONTRACT)):
        check(sha(path) == EXPECTED[label], label + " digest exact", checks)
    author_seal(checks)
    m = load_source()

    policy = m.validate_source_policy()
    check(policy["actual_m1327_result"] == {
        "present": False, "sha256_predeclared": False,
        "result_hammer_present": False}, "actual result SHA is not prefilled", checks)
    check(policy["production_authorized"] is False,
          "source contract cannot authorize production", checks)
    check(not m.FUTURE_RELEASE.exists() and not m.M1327_CAPTURE.exists() and
          not m.OUTPUT.exists() and not m.ATTEMPT.exists(),
          "canonical result/release/output/attempt absent", checks)
    check(not any(m.OUTPUT.parent.glob(m.WORK_PREFIX + "*")),
          "canonical work namespace absent", checks)
    m.verify_m1324_hammer(); checks.append("M1323/M1324 chain exact")
    m.verify_m1111dr2_template(); checks.append("M1111DR2/M1105DR2/M1115D chain exact")

    # Identity mutations are rejected before any materialization.
    with mock.patch.object(m, "M1323_SOURCE_SHA256", "0" * 64):
        reject(m.verify_m1324_hammer, "M1323 identity mutation rejected", checks)
    changed_entry = dict(m.M1324_HAMMER_ENTRY); changed_entry["review_sha256"] = "0" * 64
    with mock.patch.object(m, "M1324_HAMMER_ENTRY", changed_entry):
        reject(m.verify_m1324_hammer, "M1324 review mutation rejected", checks)
    with mock.patch.object(m, "M1111DR2_RUNNER_SHA256", "0" * 64):
        reject(m.verify_m1111dr2_template, "M1111DR2 identity mutation rejected", checks)
    with mock.patch.object(m, "M1105DR2_SOURCE_SHA256", "0" * 64):
        reject(m.verify_m1111dr2_template, "M1105DR2 identity mutation rejected", checks)
    changed_hammer = dict(m.M1115D_ENTRY); changed_hammer["review_sha256"] = "0" * 64
    with mock.patch.object(m, "M1115D_ENTRY", changed_hammer):
        reject(m.verify_m1111dr2_template, "M1115D identity mutation rejected", checks)

    audit = synthetic_audit(m)
    manifest = m.build_output_manifest(audit, authority(m))
    check(len(manifest["records"]) == 120 and
          len({r["positive_output"] for r in manifest["records"]}) == 120 and
          len({r["negative_output"] for r in manifest["records"]}) == 120,
          "samples 10..39 yield 120 calls and 240 unique outputs", checks)
    check([r["global_sample_id"] for r in manifest["records"][::4]] == list(range(10, 40)),
          "global samples 10..39 contiguous", checks)
    check(all(r["numeric_encoding"]["theta_word_uint32"] == 0x3F400000
              for r in manifest["records"] if r["module_ordinal"] == 1) and
          all(r["numeric_encoding"]["kind"] == "exact_binary"
              for r in manifest["records"] if r["module_ordinal"] != 1),
          "dynamic D1 theta retained and other modules exact binary", checks)
    check(manifest["weight_identity"] == {
        "present": False, "required_before_decoder_replay": True} and
          manifest["claim_boundary"]["decoder_replay"] is False,
          "missing weight identity blocks decoder-replay admission", checks)

    for field, value, label in (
            ("global_sample_id", 9, "sample outside 10..39 rejected"),
            ("global_call_ordinal", 5, "duplicate/misordered call rejected"),
            ("module_ordinal", 2, "module order mutation rejected"),
            ("negative_count", 1, "nonzero negative plane rejected"),
            ("positive_plane_bytes", 1, "plane extent mutation rejected")):
        mutant = synthetic_audit(m)
        mutant["calls"][0][field] = value
        reject(lambda mutant=mutant: m.build_output_manifest(mutant, authority(m)), label, checks)
    short = synthetic_audit(m); short["calls"].pop()
    reject(lambda: m.build_output_manifest(short, authority(m)), "119 calls rejected", checks)
    duplicate = synthetic_audit(m); duplicate["calls"].append(copy.deepcopy(duplicate["calls"][-1]))
    reject(lambda: m.build_output_manifest(duplicate, authority(m)), "121 calls rejected", checks)
    zero_theta = synthetic_audit(m, 0)
    reject(lambda: m.build_output_manifest(zero_theta, authority(m)),
           "non-positive dynamic theta rejected", checks)

    base_release = release_shape(m)
    m.validate_release_shape(base_release, m.FUTURE_RELEASE)
    checks.append("future release exact source-only boundary accepted")
    for path, value, label in (
            (("claim_boundary", "decoder_replay"), True,
             "decoder-replay claim without weight identity rejected"),
            (("claim_boundary", "speedup"), True, "speedup claim rejected"),
            (("one_shot", "automatic_retry"), True, "automatic retry rejected"),
            (("output", "atomic_no_replace"), False, "replace publication rejected")):
        mutant = copy.deepcopy(base_release)
        mutant[path[0]][path[1]] = value
        reject(lambda mutant=mutant: m.validate_release_shape(mutant, m.FUTURE_RELEASE),
               label, checks)

    # Missing canonical result/release fails before attempt/output/work creation.
    reject(lambda: m.execute_once(m.FUTURE_RELEASE),
           "missing canonical release/result fails closed", checks)
    check(not m.OUTPUT.exists() and not m.ATTEMPT.exists() and
          not any(m.OUTPUT.parent.glob(m.WORK_PREFIX + "*")),
          "missing canonical result leaves no namespace residue", checks)

    with tempfile.TemporaryDirectory(prefix="m1332_m1328_") as directory:
        root = Path(directory)
        attempt = root / "attempt"
        with mock.patch.object(m, "ATTEMPT", attempt):
            m.consume_attempt()
            reject(m.consume_attempt, "O_EXCL attempt refuses second consume", checks)

        # Positive/negative offsets are exact and swapped planes fail hashes.
        positive = b"\xa5\x01"; negative = b"\x00\x00"
        support = root / "support.bin"; support.write_bytes(positive + negative)
        p = root / "positive.bin"; n = root / "negative.bin"
        m.copy_plane_exclusive(support, 0, 2, p, hashlib.sha256(positive).hexdigest())
        m.copy_plane_exclusive(support, 2, 2, n, hashlib.sha256(negative).hexdigest())
        check(p.read_bytes() == positive and n.read_bytes() == negative,
              "positive and negative planes preserve offset/order", checks)
        reject(lambda: m.copy_plane_exclusive(support, 0, 2, p,
                                               hashlib.sha256(positive).hexdigest()),
               "O_EXCL plane refuses overwrite", checks)
        reject(lambda: m.copy_plane_exclusive(support, 2, 2, root / "swapped.bin",
                                               hashlib.sha256(positive).hexdigest()),
               "positive/negative plane swap rejected by digest", checks)

        staging = root / "staging"; (staging / "payloads").mkdir(parents=True)
        for index in range(240):
            m.write_exclusive(staging / "payloads" / ("p%03d.bin" % index), b"x")
        minimal = {"population": {"calls": 120}, "records": [{} for _ in range(120)]}
        m.write_exclusive(staging / "manifest.json",
                          (json.dumps(minimal) + "\n").encode(), 0o400)
        m.write_exclusive(staging / "RUN_COMPLETE.txt", b"PASS\n", 0o400)
        seal = m.seal_staging(staging)
        check(seal["members"] == 242, "recursive double seal covers 240+2 members", checks)
        (staging / "payloads/p000.bin").chmod(0o600)
        (staging / "payloads/p000.bin").write_bytes(b"changed")
        reject(lambda: m.verify_materialized_seal(staging),
               "recursive member mutation rejected", checks)

        # Extra files and symlinks cannot hide outside the recursive population.
        extra = root / "extra_staging"; (extra / "payloads").mkdir(parents=True)
        for index in range(240):
            m.write_exclusive(extra / "payloads" / ("p%03d.bin" % index), b"x")
        m.write_exclusive(extra / "manifest.json", (json.dumps(minimal) + "\n").encode())
        m.write_exclusive(extra / "RUN_COMPLETE.txt", b"PASS\n")
        m.write_exclusive(extra / "unexpected.bin", b"x")
        reject(lambda: m.seal_staging(extra), "extra recursive member rejected", checks)
        link = root / "link_staging"; (link / "payloads").mkdir(parents=True)
        for index in range(240):
            m.write_exclusive(link / "payloads" / ("p%03d.bin" % index), b"x")
        m.write_exclusive(link / "manifest.json", (json.dumps(minimal) + "\n").encode())
        m.write_exclusive(link / "RUN_COMPLETE.txt", b"PASS\n")
        os.symlink(link / "payloads/p000.bin", link / "payloads/alias.bin")
        reject(lambda: m.seal_staging(link), "recursive symlink rejected", checks)

        src = root / "publish_source"; src.mkdir()
        dst = root / "publish_destination"; dst.mkdir()
        (dst / "sentinel").write_text("keep")
        reject(lambda: m.rename_noreplace(src, dst),
               "rename no-replace collision rejected", checks)
        check((dst / "sentinel").read_text() == "keep" and src.exists(),
              "rename collision preserves source and destination", checks)

    result = {
        "schema": "m1332_m1328_ep34_decoder_bitplane_materializer_source_blind_hammer_output_r1_v1",
        "status": "PASS_SOURCE_ONLY__RELEASE_AUTHORING_ONLY__NO_MATERIALIZATION_OR_REPLAY",
        "checks_passed": len(checks),
        "checks": checks,
        "canonical_access": {"capture_opened": False, "release_present": False,
                             "attempt_created": False, "output_created": False,
                             "work_created": False, "materialization_run": False,
                             "decoder_replay_run": False},
        "claim_boundary": {"source_only": True, "production_materialization": False,
                           "decoder_replay": False, "cycles": False,
                           "traffic": False, "speedup": False, "energy": False,
                           "rtl": False, "eda": False, "ppa": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
