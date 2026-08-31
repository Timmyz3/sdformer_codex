#!/usr/bin/env python3
"""Source-only ep34 decoder positive-plane materializer.

The future one-shot copies, without normalization, the positive half of each
sealed M1458 decoder support/sign payload.  M1510 supplies the exact four
layer scale words: D0/D1 are ``bit_times_layer_constant`` and D2/D3 are exact
binary.  The negative half is revalidated as all-zero and represented only by
its SHA in the output manifest.  M1517 independent source hammer and M1518
release are required before the inert production hook may be called.

This source's CLI is deliberately limited to source self-check.  It creates no
attempt, staging directory, output plane, manifest, or seal.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE = Path(__file__).resolve()
HERE = SOURCE.parent
HW = HERE.parent.parent
ROOT = HW.parent
TEST = HW / "system_simulator/tests/test_m1516_ep34_decoder_positive_plane_materializer_source.py"
CONTRACT = HW / "contracts/m1516_ep34_decoder_positive_plane_materializer_source_contract_r1_20260831.json"
M1510_SOURCE = HERE / "build_m1510_ep34_decoder_layer_constant_adapter_source.py"
M1510_SOURCE_SHA256 = "051b61d5cf8a7b164096da229601afb2ca8867d3b878e491bd7279148e5793aa"
M1510_CONTRACT = HW / "contracts/m1510_ep34_decoder_layer_constant_adapter_source_contract_r1_20260831.json"
M1510_CONTRACT_SHA256 = "88203261b26abee15ec57430e46cef7b4225f53fbb67abe9d18fc87c82d1abd7"
M1512 = HW / "reviews/m1512_m1501_m1458_ep34_capture_source_result_independent_hammer_r1_20260831"
M1512_PINS = (
    "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    "2af7a59b6a4df07dc6047c0d48c52b7798b7f0803e31e290b2ad842e6c154b81",
    "ccbcd7bf1b99fd944062a6fb220d7ec719d96da91c190697db125cbd4ad58f7c",
)
M1513 = HW / "reviews/m1513_m1512_m1458_ep34_production_provenance_addendum_r1_20260831"
M1513_PINS = (
    "1eb36a76fac29d5d15607dbb4ee3f9a434c4b0686843acac11f18116b48c7aaa",
    "966ba95baf00f698b6ca1fb8613afbfb78e40d2a70223f0a72bd4a87dcea04fa",
    "dc19cacbbb5ecae7f0327fd17b310be79a3b144937be7f289c25eb6f64794832",
)
CAPTURE = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
CAPTURE_MANIFEST_SHA256 = "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e"
CAPTURE_OUTER_SHA256 = "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
OUTPUT = HW / "results/m1516_ep34_decoder_positive_planes_s30_c120_r1_20260831"
ATTEMPT = HW / "results/.m1516_ep34_decoder_positive_planes_s30_c120_r1_20260831.attempt_consumed"
WORK_PREFIX = ".m1516_ep34_decoder_positive_planes_work."
FUTURE_RELEASE = HW / "contracts/m1518_ep34_decoder_positive_plane_materializer_production_release_r1_20260831.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1516_ep34_decoder_positive_plane_materializer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1517_HAMMER_AND_M1518_RELEASE_REQUIRED__NO_PRODUCTION"
OUTPUT_SCHEMA = "m1516_ep34_decoder_positive_plane_materialization_r1_v1"
OUTPUT_STATUS = "MATERIALIZATION_COMPLETE__ADDRESS_TIMED_REPLAY_NOT_RUN"
RELEASE_SCHEMA = "m1518_ep34_decoder_positive_plane_materializer_production_release_r1_v1"
RELEASE_STATUS = "M1517_SOURCE_HAMMER_BOUND__ONE_M1516_MATERIALIZATION"
HAMMER_SCHEMA = "m1517_m1516_ep34_decoder_positive_plane_materializer_source_hammer_r1_v1"
HAMMER_STATUS = "PASS_M1517_M1516_SOURCE_HAMMER__M1518_RELEASE_AUTHORING_ONLY"
ATTEMPT_TOKEN = "M1516_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
RUN_TOKEN = "PASS_M1516_EP34_DECODER_POSITIVE_PLANE_MATERIALIZATION\n"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
EXPECTED_CALLS = 120
EXPECTED_PAYLOAD_FILES = 120
EXPECTED_SEALED_MEMBERS = 122
EXPECTED_SCALE_WORDS = (0x3F7FFD6B, 0x3F7FFFA0, 0x3F800000, 0x3F800000)
CLAIM_BOUNDARY = {
    "source_only": True,
    "production": False,
    "positive_plane_materialization": False,
    "negative_plane_output": False,
    "weight_folding": False,
    "normalization": False,
    "coercion": False,
    "address_timed_replay": False,
    "cycles": False,
    "traffic": False,
    "speedup": False,
    "system_speedup": False,
    "energy": False,
    "rtl": False,
    "eda": False,
    "ppa": False,
    "table_a": False,
}


class M1516Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1516Error(message)


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


def regular(path: Path, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise M1516Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")


def regular_exact(path: Path, digest: str, label: str) -> None:
    regular(path, label)
    require(sha256(path) == lowercase_sha(digest, label + " SHA"),
            label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1516Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def safe_member(root: Path, relative: Any, label: str) -> Path:
    require(type(relative) is str, label + " relative path is not string")
    member = PurePosixPath(relative)
    require(member.parts and not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == relative,
            label + " relative path unsafe")
    cursor = Path(root)
    for part in member.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), label + " path contains symlink")
    regular(cursor, label)
    return cursor


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    regular_exact(path, digest, name + " after import")
    return module


M1510 = load_exact("m1516_frozen_m1510", M1510_SOURCE, M1510_SOURCE_SHA256)


def verify_sealed_review(root: Path, pins: tuple[str, str, str],
                         expected_status: str) -> dict[str, Any]:
    review_sha, manifest_sha, outer_sha = pins
    regular_exact(root / "review.json", review_sha, root.name + " review")
    regular_exact(root / MANIFEST, manifest_sha, root.name + " manifest")
    regular_exact(root / OUTER, outer_sha, root.name + " outer")
    require((root / OUTER).read_text().split() == [manifest_sha, MANIFEST],
            root.name + " outer content drift")
    members: set[str] = set()
    prefix = root.relative_to(ROOT).as_posix() + "/"
    for line in (root / MANIFEST).read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, root.name + " manifest row malformed")
        digest, name = fields
        name = name.lstrip("*")
        if name.startswith(prefix):
            name = name[len(prefix):]
        require(name not in members, root.name + " duplicate manifest member")
        regular_exact(safe_member(root, name, root.name + " member"), digest,
                      root.name + " member")
        members.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.relative_to(root).as_posix() not in
              {MANIFEST, OUTER}}
    require(actual == members, root.name + " sealed population drift")
    value = strict_json(root / "review.json")
    require(value.get("status") == expected_status, root.name + " status drift")
    return value


def verify_authorities() -> dict[str, Any]:
    regular_exact(M1510_CONTRACT, M1510_CONTRACT_SHA256, "M1510 contract")
    M1510.validate_source_policy()
    m1512 = verify_sealed_review(
        M1512, M1512_PINS,
        "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT")
    m1513 = verify_sealed_review(
        M1513, M1513_PINS,
        "PASS_M1513_COMPLETE_M1458_EP34_PRODUCTION_PROVENANCE")
    require(m1512.get("verification", {}).get("identity", {}).get(
        "checkpoint_sha256") == CHECKPOINT_SHA256 and
            m1513.get("capture_binding", {}).get("checkpoint_sha256") ==
            CHECKPOINT_SHA256,
            "M1512/M1513 checkpoint identity drift")
    for value in (m1512, m1513):
        bindings = value.get("bindings", {})
        require(bindings.get("result_manifest_sha256") == CAPTURE_MANIFEST_SHA256 and
                bindings.get("result_outer_file_sha256") == CAPTURE_OUTER_SHA256,
                "M1512/M1513 M1458 capture seal drift")
    require(m1513.get("bindings", {}).get("m1512_review_sha256") == M1512_PINS[0] and
            m1513.get("bindings", {}).get("m1512_manifest_sha256") == M1512_PINS[1] and
            m1513.get("bindings", {}).get("m1512_outer_file_sha256") == M1512_PINS[2],
            "M1513 does not exact-bind M1512")
    return {"m1512": m1512["status"], "m1513": m1513["status"]}


def output_plane_name(global_call_ordinal: int, global_sample_id: int,
                      module_ordinal: int) -> str:
    require(type(global_call_ordinal) is int and 0 <= global_call_ordinal < 120 and
            type(global_sample_id) is int and 10 <= global_sample_id < 40 and
            type(module_ordinal) is int and 0 <= module_ordinal < 4,
            "output plane ordinal identity invalid")
    return "payloads/c{:03d}_s{:02d}_d{}.positive.le.bitpack".format(
        global_call_ordinal, global_sample_id, module_ordinal)


def plane_digest(path: Path, offset: int, count: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        remaining = count
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            require(block, "support/sign plane truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def audit_support_planes(capture_root: Path, call: Mapping[str, Any]) -> dict[str, Any]:
    support = safe_member(capture_root, call.get("support_sign"), "support/sign payload")
    expected_support = lowercase_sha(call.get("support_sign_sha256"), "support SHA")
    require(sha256(support) == expected_support, "support/sign payload SHA drift")
    shape = call.get("shape")
    require(type(shape) is list and shape and
            all(type(value) is int and value > 0 for value in shape),
            "support shape invalid")
    elements = math.prod(shape)
    plane_bytes = (elements + 7) // 8
    require(call.get("positive_plane_bytes") == plane_bytes and
            call.get("negative_plane_bytes") == plane_bytes and
            support.stat().st_size == 2 * plane_bytes,
            "support/sign extent drift")
    payload = support.read_bytes()
    positive = payload[:plane_bytes]
    negative = payload[plane_bytes:]
    remainder = elements & 7
    if remainder:
        require(positive[-1] >> remainder == 0 and negative[-1] >> remainder == 0,
                "support/sign padding bits are nonzero")
    require(not any(negative), "negative support plane is not all zero")
    return {
        "source_support_sign_sha256": expected_support,
        "positive_plane_sha256": hashlib.sha256(positive).hexdigest(),
        "negative_zero_plane_sha256": hashlib.sha256(negative).hexdigest(),
        "plane_bytes": plane_bytes,
        "elements": elements,
    }


def enrich_audit(audit: Mapping[str, Any], capture_root: Path) -> dict[str, Any]:
    calls = audit.get("calls")
    require(type(calls) is list and len(calls) == EXPECTED_CALLS,
            "M1510 call population is not 120")
    enriched = []
    for call in calls:
        enriched.append({**dict(call), **audit_support_planes(capture_root, call)})
    return {**dict(audit), "calls": enriched}


def build_output_manifest(audit: Mapping[str, Any]) -> dict[str, Any]:
    calls = audit.get("calls")
    layers = audit.get("layer_scale_words")
    require(type(calls) is list and len(calls) == EXPECTED_CALLS,
            "M1510 call population is not 120")
    require(type(layers) is list and len(layers) == 4,
            "M1510 layer scale population is not four")
    scale_words = []
    for ordinal, layer in enumerate(layers):
        require(type(layer) is dict and layer.get("module_ordinal") == ordinal and
                layer.get("word_uint32") == EXPECTED_SCALE_WORDS[ordinal] and
                layer.get("calls") == 30 and
                layer.get("all_calls_same_word") is True,
                "M1510 layer scale word drift")
        scale_words.append(layer["word_uint32"])

    records = []
    output_paths: set[str] = set()
    source_paths: set[str] = set()
    global_orders: set[int] = set()
    for ordinal, call in enumerate(calls):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        require(call.get("global_call_ordinal") == ordinal and
                call.get("global_sample_id") == sample and
                call.get("module_ordinal") == module and
                call.get("module") == M1510.M1323.MODULES[module] and
                call.get("shape") == list(M1510.M1323.SHAPES[module]),
                "M1510 30x4 call identity/order drift")
        require(call.get("positive_word_uint32") == scale_words[module] and
                call.get("negative_count") == 0 and
                call.get("nonfinite_count") == 0,
                "M1510 per-call scale/sign semantic drift")
        plane_bytes = (math.prod(call["shape"]) + 7) // 8
        require(call.get("plane_bytes", call.get("positive_plane_bytes")) == plane_bytes and
                call.get("positive_plane_bytes") == plane_bytes and
                call.get("negative_plane_bytes") == plane_bytes,
                "M1510 plane extent drift")
        for key in ("source_support_sign_sha256", "positive_plane_sha256",
                    "negative_zero_plane_sha256"):
            lowercase_sha(call.get(key), key)
        source = call.get("support_sign")
        global_order = call.get("global_order")
        require(type(source) is str and source not in source_paths and
                type(global_order) is int and global_order not in global_orders,
                "source support/global order duplicate")
        source_paths.add(source); global_orders.add(global_order)
        output = output_plane_name(ordinal, sample, module)
        require(output not in output_paths, "positive output path duplicate")
        output_paths.add(output)
        kind = "bit_times_layer_constant" if module in (0, 1) else "exact_binary"
        records.append({
            "global_call_ordinal": ordinal,
            "capture_global_order": global_order,
            "global_sample_id": sample,
            "replay_sample_ordinal": sample - 10,
            "sequence": call.get("sequence"),
            "sample_key": call.get("sample_key"),
            "module_ordinal": module,
            "module": call.get("module"),
            "shape": call.get("shape"),
            "elements": math.prod(call["shape"]),
            "plane_bytes": plane_bytes,
            "source_support_sign": source,
            "source_support_sign_sha256": call["source_support_sign_sha256"],
            "source_positive_plane_sha256": call["positive_plane_sha256"],
            "source_negative_zero_plane_sha256": call["negative_zero_plane_sha256"],
            "positive_output": output,
            "positive_output_sha256": call["positive_plane_sha256"],
            "layer_scale_word_uint32": scale_words[module],
            "layer_scale_word_hex": "0x{:08x}".format(scale_words[module]),
            "numeric_encoding": kind,
            "negative_plane_output": None,
            "negative_plane_all_zero": True,
            "weight_folding": False,
            "normalized": False,
            "coerced": False,
        })
    require(len(output_paths) == len(source_paths) == len(global_orders) == EXPECTED_CALLS,
            "120-call unique identity population drift")
    seal = audit.get("capture_seal", {})
    require(seal.get("sha256sums_sha256") == CAPTURE_MANIFEST_SHA256 and
            seal.get("outer_seal_sha256") == CAPTURE_OUTER_SHA256,
            "M1510 capture seal identity drift")
    return {
        "schema": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "capture": {
            "path": str(CAPTURE.relative_to(ROOT)),
            "sha256sums_sha256": CAPTURE_MANIFEST_SHA256,
            "outer_seal_sha256": CAPTURE_OUTER_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "m1510_schema": audit.get("schema"),
            "m1510_status": audit.get("status"),
        },
        "population": {
            "samples": 30, "calls": 120, "modules": 4,
            "global_sample_ids": [10, 39], "positive_plane_files": 120,
            "negative_plane_files": 0, "global_call_ordinals_contiguous": True,
        },
        "layer_scale_words": [dict(layer) for layer in layers],
        "records": records,
        "claim_boundary": {
            "positive_plane_materialization": True,
            "address_timed_replay": False,
            "cycles": False, "traffic": False, "speedup": False,
            "system_speedup": False, "energy": False, "rtl": False,
            "eda": False, "ppa": False, "table_a": False,
        },
    }


def write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "exclusive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_positive_plane_exclusive(source: Path, destination: Path,
                                  elements: int, plane_bytes: int,
                                  expected_support_sha256: str,
                                  expected_positive_sha256: str,
                                  expected_negative_sha256: str) -> None:
    regular(source, "source support/sign payload")
    require(type(elements) is int and elements > 0 and
            type(plane_bytes) is int and plane_bytes == (elements + 7) // 8 and
            source.stat().st_size == 2 * plane_bytes,
            "source support/sign plane geometry drift")
    require(sha256(source) == lowercase_sha(expected_support_sha256, "support SHA"),
            "source support/sign SHA drift")
    with source.open("rb") as stream:
        positive = stream.read(plane_bytes)
        negative = stream.read(plane_bytes)
        require(not stream.read(1), "source support/sign trailing bytes")
    remainder = elements & 7
    if remainder:
        require(positive[-1] >> remainder == 0 and negative[-1] >> remainder == 0,
                "source support/sign padding bits are nonzero")
    require(not any(negative), "source negative plane is not all zero")
    require(hashlib.sha256(positive).hexdigest() == lowercase_sha(
        expected_positive_sha256, "positive SHA") and
            hashlib.sha256(negative).hexdigest() == lowercase_sha(
                expected_negative_sha256, "negative SHA"),
            "source positive/negative plane SHA drift")
    write_exclusive(destination, positive, 0o400)
    require(destination.stat().st_size == plane_bytes and
            sha256(destination) == expected_positive_sha256,
            "materialized positive plane SHA/extent drift")


def fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def payload_files(root: Path) -> list[Path]:
    output = []
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), "output seal refuses symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in {MANIFEST, OUTER}:
            output.append(path)
    return output


def verify_materialized_seal(root: Path) -> dict[str, Any]:
    manifest = root / MANIFEST
    outer = root / OUTER
    regular(manifest, "output manifest seal")
    regular(outer, "output outer seal")
    require(outer.read_text().split() == [sha256(manifest), MANIFEST],
            "output outer seal drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "output seal row malformed")
        digest, relative = fields
        relative = relative.lstrip("*")
        require(relative not in rows, "duplicate output seal member")
        member = safe_member(root, relative, "output sealed member")
        require(sha256(member) == lowercase_sha(digest, "output member SHA"),
                "output sealed member drift")
        rows[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in payload_files(root)}
    require(actual == set(rows) and len(rows) == EXPECTED_SEALED_MEMBERS,
            "output recursive seal population drift")
    value = strict_json(root / "manifest.json")
    records = value.get("records")
    require(value.get("schema") == OUTPUT_SCHEMA and
            value.get("population", {}).get("calls") == 120 and
            value.get("population", {}).get("positive_plane_files") == 120 and
            type(records) is list and len(records) == 120,
            "output materialization manifest population drift")
    record_paths = [row.get("positive_output") for row in records]
    require(len(set(record_paths)) == 120 and
            set(record_paths) == {name for name in rows if name.startswith("payloads/")},
            "output manifest/seal positive-plane identity drift")
    require(all(rows[row["positive_output"]] == row.get("positive_output_sha256")
                for row in records),
            "output manifest positive SHA differs from recursive seal")
    return {"manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer), "members": len(rows)}


def seal_staging(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink() and
            not (root / MANIFEST).exists() and not (root / OUTER).exists(),
            "bad staging seal target")
    members = payload_files(root)
    require(len(members) == EXPECTED_SEALED_MEMBERS,
            "staging member population is not 122")
    fsync_dir(root / "payloads")
    lines = [sha256(path) + "  " + path.relative_to(root).as_posix()
             for path in members]
    write_exclusive(root / MANIFEST, ("\n".join(lines) + "\n").encode(), 0o400)
    write_exclusive(root / OUTER,
                    (sha256(root / MANIFEST) + "  " + MANIFEST + "\n").encode(), 0o400)
    fsync_dir(root)
    return verify_materialized_seal(root)


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    require(function is not None, "renameat2 unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise M1516Error("atomic output no-replace collision")
        raise OSError(code, os.strerror(code), str(destination))


def namespace_fresh(output: Path = OUTPUT, attempt: Path = ATTEMPT,
                    work_prefix: str = WORK_PREFIX) -> None:
    require(not os.path.lexists(str(output)) and not os.path.lexists(str(attempt)) and
            not any(output.parent.glob(work_prefix + "*")),
            "M1516 materialization namespace is not fresh")


def consume_attempt(attempt: Path = ATTEMPT) -> None:
    write_exclusive(attempt, ATTEMPT_TOKEN.encode("ascii"), 0o400)


def materialize_prepared_once(capture_root: Path, audit: Mapping[str, Any],
                              output: Path, attempt: Path,
                              work_prefix: str = WORK_PREFIX) -> Path:
    """Future one-shot primitive; failures intentionally preserve staging."""
    manifest = build_output_manifest(audit)
    namespace_fresh(output, attempt, work_prefix)
    consume_attempt(attempt)
    staging = output.parent / (work_prefix + str(os.getpid()) + "." + str(time.time_ns()))
    staging.mkdir(mode=0o700)
    (staging / "payloads").mkdir(mode=0o700)
    try:
        for call, record in zip(audit["calls"], manifest["records"]):
            source = safe_member(capture_root, call["support_sign"], "capture support/sign")
            destination = staging.joinpath(
                *PurePosixPath(record["positive_output"]).parts)
            copy_positive_plane_exclusive(
                source, destination, record["elements"], record["plane_bytes"],
                record["source_support_sign_sha256"],
                record["source_positive_plane_sha256"],
                record["source_negative_zero_plane_sha256"])
        write_exclusive(staging / "manifest.json",
                        (json.dumps(manifest, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n").encode(), 0o400)
        write_exclusive(staging / "RUN_COMPLETE.txt", RUN_TOKEN.encode(), 0o400)
        seal_staging(staging)
        rename_noreplace(staging, output)
        fsync_dir(output.parent)
        verify_materialized_seal(output)
    except BaseException:
        # The consumed attempt and exclusive stage remain for failure forensics.
        raise
    return output


RELEASE_KEYS = {"schema", "status", "source_identity", "m1517_source_hammer",
                "authority", "one_shot", "output", "claim_boundary"}


def validate_release_shape(release: Any) -> None:
    require(type(release) is dict and set(release) == RELEASE_KEYS,
            "M1518 release key set drift")
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS,
            "M1518 release schema/status drift")
    require(release.get("source_identity") == {
        "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "test_path": str(TEST.relative_to(ROOT)), "test_sha256": sha256(TEST),
        "contract_path": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT)},
            "M1518 source identity drift")
    require(release.get("authority") == {
        "m1510_source_sha256": M1510_SOURCE_SHA256,
        "m1510_contract_sha256": M1510_CONTRACT_SHA256,
        "m1512_review_manifest_outer": list(M1512_PINS),
        "m1513_review_manifest_outer": list(M1513_PINS),
        "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        "capture_outer_sha256": CAPTURE_OUTER_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256},
            "M1518 authority drift")
    require(release.get("one_shot") == {
        "attempt_marker": str(ATTEMPT.relative_to(ROOT)),
        "automatic_retry": False, "maximum_materializations": 1,
        "failure_stage_preserved": True},
            "M1518 one-shot policy drift")
    require(release.get("output") == {
        "path": str(OUTPUT.relative_to(ROOT)), "positive_plane_files": 120,
        "negative_plane_files": 0, "atomic_no_replace": True,
        "recursive_double_seal": True},
            "M1518 output policy drift")
    require(release.get("claim_boundary") == {
        "positive_plane_materialization": True,
        "address_timed_replay": False, "cycles": False, "traffic": False,
        "speedup": False, "system_speedup": False, "energy": False,
        "rtl": False, "eda": False, "ppa": False, "table_a": False},
            "M1518 claim boundary drift")


def verify_m1517_hammer(entry: Any) -> dict[str, Any]:
    require(type(entry) is dict and set(entry) == {
        "path", "review_sha256", "manifest_sha256", "outer_file_sha256"},
            "M1517 hammer entry drift")
    relative = PurePosixPath(entry["path"])
    require(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
            "M1517 hammer path unsafe")
    root = ROOT.joinpath(*relative.parts)
    require(root.parent == HW / "reviews", "M1517 hammer not directly under reviews")
    review = verify_sealed_review(root, (
        lowercase_sha(entry["review_sha256"], "M1517 review SHA"),
        lowercase_sha(entry["manifest_sha256"], "M1517 manifest SHA"),
        lowercase_sha(entry["outer_file_sha256"], "M1517 outer SHA")), HAMMER_STATUS)
    require(review.get("schema") == HAMMER_SCHEMA and
            review.get("source_identity") == {
                "source_sha256": sha256(SOURCE), "test_sha256": sha256(TEST),
                "contract_sha256": sha256(CONTRACT)} and
            review.get("authorization") == {
                "m1518_release_authoring": True, "production_materialization": False},
            "M1517 hammer authority drift")
    return review


def execute_once(release_path: Path) -> Path:
    """Future M1518 hook; deliberately unreachable from this source's CLI."""
    require(Path(release_path).resolve() == FUTURE_RELEASE.resolve(),
            "only canonical M1518 release path allowed")
    regular(release_path, "M1518 release")
    release = strict_json(release_path)
    validate_release_shape(release)
    verify_m1517_hammer(release["m1517_source_hammer"])
    verify_authorities()
    audit = M1510.audit_capture(CAPTURE)
    enriched = enrich_audit(audit, CAPTURE)
    return materialize_prepared_once(CAPTURE, enriched, OUTPUT, ATTEMPT)


def validate_source_policy() -> dict[str, Any]:
    regular_exact(M1510_SOURCE, M1510_SOURCE_SHA256, "M1510 source")
    regular_exact(M1510_CONTRACT, M1510_CONTRACT_SHA256, "M1510 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    policy = strict_json(CONTRACT)
    require(policy.get("schema") == SCHEMA and
            policy.get("status") == SOURCE_STATUS,
            "M1516 source policy schema/status drift")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "M1516 source/test identity drift")
    require(policy.get("production_authorized") is False and
            policy.get("future_release") == str(FUTURE_RELEASE.relative_to(ROOT)) and
            policy.get("claim_boundary") == CLAIM_BOUNDARY,
            "M1516 production/future/claim boundary drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    require(args.source_self_check,
            "M1516 is source-only; production materialization CLI is forbidden")
    validate_source_policy()
    verify_authorities()
    require(not os.path.lexists(str(OUTPUT)) and not os.path.lexists(str(ATTEMPT)) and
            not any(OUTPUT.parent.glob(WORK_PREFIX + "*")),
            "M1516 production namespace already exists")
    require(not FUTURE_RELEASE.exists(), "future M1518 release already exists")
    print("PASS_M1516_SOURCE_SELF_CHECK__NO_CAPTURE_READ_NO_MATERIALIZATION")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1516Error as error:
        print("M1516_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
