#!/usr/bin/env python3
"""Source-only M1328 ep34 decoder bitplane materializer.

This is an inert successor over the independently hammered M1323 adapter.  A
future, separately authored release must bind an *actual* successful M1327
capture and its different-author result hammer.  No M1327 result digest is
guessed here.

After that future gate, one materialization may split every retained decoder
``positive||negative`` support file into two exact little-bit-order planes.
The output keeps global samples 10..39 and all 120 D0,D1,D2,D3 calls, carries
the observed D1 theta word dynamically, consumes an O_EXCL attempt, writes all
files with exclusive creation, recursively seals the staging directory, and
atomically publishes it without replacement.  It does not run the decoder
simulator and admits no cycle, traffic, speedup, energy, RTL, EDA, or PPA claim.
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
from typing import Any, Iterable, Mapping, Sequence


SOURCE_FILE = Path(__file__).resolve()
HERE = SOURCE_FILE.parent
HW = HERE.parent.parent
ROOT = HW.parent
M1323_SOURCE = HERE / "build_m1323_ep34_decoder_capture_adapter_source.py"
M1323_SOURCE_SHA256 = "0481e39372ffe19cd3cff8d5053c9eae8326de4fb5ac61bd9e42527a3ad3a12a"
M1323_TEST = HW / "system_simulator/tests/test_m1323_ep34_decoder_capture_adapter_source.py"
M1323_TEST_SHA256 = "c29980f357ea0e0a9b2e11650239b706f6c4e18892b4975925db164a72439487"
M1323_CONTRACT = HW / "contracts/m1323_ep34_decoder_capture_adapter_source_contract_r1_20260831.json"
M1323_CONTRACT_SHA256 = "e4df50fed6068b0f384693044705b30f595d41d70dce78e738cb36a98e24cecc"
M1324_HAMMER = HW / "reviews/m1324_m1323_ep34_decoder_adapter_source_hammer_r1_20260831"
M1324_HAMMER_ENTRY = {
    "path": str(M1324_HAMMER.relative_to(ROOT)),
    "manifest_sha256": "bec10a857db964f94919aaa20d2aa603b7b0521b427164f225cda8d54b730a4f",
    "outer_file_sha256": "3e3bdb0d13089de323fd6c2c723ae263014cd9a3005ff7fabcfe34bede20e4ea",
    "review_sha256": "79683ae29e70bd8272073c28ecbe26290c91201524d82a72a83f7ffc8ac719a2",
}
M1324_SCHEMA = "m1324_m1323_ep34_decoder_adapter_source_hammer_review_r1_v1"
M1324_STATUS = "PASS_M1324_M1323_SOURCE_HAMMER__ACTUAL_RESULT_SUCCESSOR_ALLOWED"
M1111DR2_RUNNER = HERE / "run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py"
M1111DR2_RUNNER_SHA256 = "1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746"
M1105DR2_SOURCE = HERE / "build_m1105dr2_decoder_only_address_timed_source.py"
M1105DR2_SOURCE_SHA256 = "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4"
M1115D_HAMMER = HW / "reviews/m1115d_m1111dr2_decoder_runner_final_independent_hammer_r1_20260830"
M1115D_ENTRY = {
    "manifest_sha256": "d40d8656180860598aca111b8061513d61a43a307482ff77f3ed6a1c1fede863",
    "outer_file_sha256": "1b13a418984866c6bd9a4088523488d2c6c3b8cfdce1eab79d04b8d6028d9fc3",
    "review_sha256": "732d43860b4594e84f267537ad7ceebd2924e3cf2f6cdd3d2cf2171cfe481e08",
}
M1327_CAPTURE = HW / "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"
SOURCE_CONTRACT = HW / "contracts/m1328_ep34_decoder_bitplane_materializer_source_contract_r1_20260831.json"
TEST = HW / "system_simulator/tests/test_m1328_ep34_decoder_bitplane_materializer_source.py"
FUTURE_RELEASE = HW / "contracts/m1328_ep34_decoder_bitplane_materializer_production_release_r1_20260831.json"
OUTPUT = HW / "results/m1328_ep34_decoder_bitplanes_s30_c120_r1_20260831"
ATTEMPT = HW / "results/.m1328_ep34_decoder_bitplanes_s30_c120_r1_20260831.attempt_consumed"
WORK_PREFIX = ".m1328_ep34_decoder_bitplanes_work."
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SOURCE_SCHEMA = "m1328_ep34_decoder_bitplane_materializer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__ACTUAL_M1327_RESULT_HAMMER_AND_RELEASE_REQUIRED__NO_PRODUCTION"
RELEASE_SCHEMA = "m1328_ep34_decoder_bitplane_materializer_production_release_r1_v1"
RELEASE_STATUS = "ACTUAL_M1327_RESULT_AND_TWO_HAMMERS_BOUND__ONE_M1328_MATERIALIZATION"
SOURCE_HAMMER_SCHEMA = "m1329_m1328_decoder_bitplane_materializer_source_hammer_r1_v1"
SOURCE_HAMMER_STATUS = "PASS_M1329_M1328_SOURCE_HAMMER__RELEASE_AUTHORING_ONLY"
RESULT_HAMMER_SCHEMA = "m1330_m1327_capture_result_hammer_for_decoder_materialization_r1_v1"
RESULT_HAMMER_STATUS = "PASS_M1330_ACTUAL_M1327_CAPTURE__M1328_MATERIALIZATION_RELEASE_ALLOWED"
ATTEMPT_TOKEN = "M1328_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
PASS_TOKEN = "PASS_M1328_SOURCE_SELF_CHECK__NO_CAPTURE_NO_MATERIALIZATION"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
EXPECTED_CALLS = 120
EXPECTED_FILES = 240


class M1328Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1328Error(message)


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
        raise M1328Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")


def regular_exact(path: Path, expected: str, label: str) -> None:
    regular(path, label)
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1328Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root is not object: " + str(path))
    return value


def safe_member(root: Path, relative: Any, label: str) -> Path:
    require(type(relative) is str, label + " relative path is not string")
    member = PurePosixPath(relative)
    require(member.parts and not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == relative, label + " relative path unsafe")
    cursor = Path(root)
    for part in member.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), label + " path contains symlink")
    regular(cursor, label)
    return cursor


def _load_m1323():
    regular_exact(M1323_SOURCE, M1323_SOURCE_SHA256, "sealed M1323 source")
    regular_exact(M1323_TEST, M1323_TEST_SHA256, "sealed M1323 test")
    regular_exact(M1323_CONTRACT, M1323_CONTRACT_SHA256, "sealed M1323 contract")
    spec = importlib.util.spec_from_file_location("m1328_sealed_m1323", M1323_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load sealed M1323")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1323 = _load_m1323()


def verify_double_sealed_directory(root: Path, manifest_sha: str,
                                   outer_file_sha: str) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "sealed root missing/symlink")
    manifest = root / MANIFEST
    outer = root / OUTER
    regular_exact(manifest, lowercase_sha(manifest_sha, "manifest SHA"), "manifest")
    regular_exact(outer, lowercase_sha(outer_file_sha, "outer SHA"), "outer seal")
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, MANIFEST],
            "outer seal content mismatch")
    members = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "malformed manifest row")
        relative = fields[1].lstrip("*")
        require(relative not in members, "duplicate manifest member")
        member = safe_member(root, relative, "sealed member")
        require(sha256(member) == fields[0], "sealed member SHA drift: " + relative)
        members[relative] = fields[0]
    population = list(root.rglob("*"))
    require(all(not path.is_symlink() for path in population),
            "recursive sealed population contains symlink")
    actual = {path.relative_to(root).as_posix() for path in population
              if path.is_file() and path.relative_to(root).as_posix() not in
              {MANIFEST, OUTER}}
    require(actual == set(members), "recursive sealed population mismatch")
    return members


def verify_m1324_hammer() -> dict[str, Any]:
    rows = verify_double_sealed_directory(
        M1324_HAMMER, M1324_HAMMER_ENTRY["manifest_sha256"],
        M1324_HAMMER_ENTRY["outer_file_sha256"])
    require(rows.get("review.json") == M1324_HAMMER_ENTRY["review_sha256"],
            "M1324 review member mismatch")
    review = strict_json(M1324_HAMMER / "review.json")
    require(review.get("schema") == M1324_SCHEMA and review.get("status") == M1324_STATUS,
            "M1324 schema/status mismatch")
    authority = review.get("source_authority", {})
    require(authority.get("source_sha256") == M1323_SOURCE_SHA256 and
            authority.get("test_sha256") == M1323_TEST_SHA256 and
            authority.get("contract_sha256") == M1323_CONTRACT_SHA256 and
            review.get("authorization") == {
                "source_audit_citable": True,
                "actual_result_successor_authoring": True,
                "production_replay": False, "remote_access": False, "gpu": False},
            "M1324 authority mismatch")
    return review


def verify_m1111dr2_template() -> dict[str, Any]:
    """Bind the only permitted writer/atomic-publication implementation template."""
    regular_exact(M1111DR2_RUNNER, M1111DR2_RUNNER_SHA256, "M1111DR2 runner template")
    regular_exact(M1105DR2_SOURCE, M1105DR2_SOURCE_SHA256, "M1105DR2 source template")
    rows = verify_double_sealed_directory(
        M1115D_HAMMER, M1115D_ENTRY["manifest_sha256"], M1115D_ENTRY["outer_file_sha256"])
    require(rows.get("review.json") == M1115D_ENTRY["review_sha256"],
            "M1115D review member mismatch")
    review = strict_json(M1115D_HAMMER / "review.json")
    require(review.get("status") ==
            "PASS_M1115D_M1111DR2_FINAL_RUNNER_HAMMER__ONE_EXTERNAL_ROOT_LAUNCH_AUTHORIZED" and
            review.get("identity", {}).get("runner_sha256") == M1111DR2_RUNNER_SHA256 and
            review.get("authorization", {}).get("automatic_retry") is False,
            "M1111DR2 template hammer authority drift")
    return review


CAPTURE_ENTRY_KEYS = {
    "path", "manifest_sha256", "outer_file_sha256",
    "capture_manifest_sha256", "admission_sha256",
}
HAMMER_ENTRY_KEYS = {"path", "manifest_sha256", "outer_file_sha256", "review_sha256"}
IDENTITY_KEYS = {"epoch", "checkpoint_sha256", "config_sha256", "profile_sha256"}


def validate_capture_entry(entry: Any) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    require(type(entry) is dict and set(entry) == CAPTURE_ENTRY_KEYS,
            "actual capture entry keys drift")
    require(entry["path"] == str(M1327_CAPTURE.relative_to(ROOT)),
            "only canonical M1327 capture may be materialized")
    for key in CAPTURE_ENTRY_KEYS - {"path"}:
        lowercase_sha(entry[key], "capture " + key)
    rows = verify_double_sealed_directory(
        M1327_CAPTURE, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("manifest.json") == entry["capture_manifest_sha256"] and
            rows.get("m1227_admission.json") == entry["admission_sha256"],
            "capture manifest/admission member mismatch")
    manifest = strict_json(M1327_CAPTURE / "manifest.json")
    admission = strict_json(M1327_CAPTURE / "m1227_admission.json")
    require(manifest.get("schema") ==
            "m1227_motion_final_checkpoint_unified_hardware_capture_r1_v1" and
            manifest.get("status") ==
            "CAPTURE_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "M1327 capture manifest schema/status drift")
    require(admission == {
        "schema": "m1227_final_capture_admission_r1_v1", "status": "PASS",
        "ordered": 9880, "attention": 480, "payload_files": 640,
        "execution": 7360, "operator_rows": 79, "atlif_live_rows": 93,
        "atlif_static": 105, "dead_sn_v": admission.get("dead_sn_v"),
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False,
                           "energy": False, "ppa": False}},
            "M1327 admission fields drift")
    require(type(admission["dead_sn_v"]) is list and len(admission["dead_sn_v"]) == 12,
            "M1327 dead sn_v population drift")
    return rows, manifest, admission


def _verify_hammer_entry(entry: Any, schema: str, status: str,
                         review_label: str) -> tuple[dict[str, Any], dict[str, str]]:
    require(type(entry) is dict and set(entry) == HAMMER_ENTRY_KEYS,
            review_label + " entry keys drift")
    for key in HAMMER_ENTRY_KEYS - {"path"}:
        lowercase_sha(entry[key], review_label + " " + key)
    relative = PurePosixPath(entry["path"])
    require(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
            review_label + " path unsafe")
    root = ROOT.joinpath(*relative.parts)
    require(root.parent == HW / "reviews", review_label + " must be directly under reviews")
    rows = verify_double_sealed_directory(
        root, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            review_label + " review member mismatch")
    review = strict_json(root / "review.json")
    require(review.get("schema") == schema and review.get("status") == status,
            review_label + " schema/status mismatch")
    return review, rows


def verify_source_hammer(entry: Any) -> dict[str, Any]:
    review, _rows = _verify_hammer_entry(
        entry, SOURCE_HAMMER_SCHEMA, SOURCE_HAMMER_STATUS, "source hammer")
    require(review.get("source_authority") == {
        "source_path": str(SOURCE_FILE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE_FILE),
        "test_path": str(TEST.relative_to(ROOT)), "test_sha256": sha256(TEST),
        "contract_path": str(SOURCE_CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(SOURCE_CONTRACT)},
            "source hammer cross-identity mismatch")
    require(review.get("independence") == {"different_author": True} and
            review.get("authorization") == {
                "production_release_authoring": True,
                "production_materialization": False},
            "source hammer authorization mismatch")
    return review


def verify_result_hammer(entry: Any, capture_entry: dict[str, Any],
                         capture_manifest: dict[str, Any]) -> dict[str, Any]:
    review, _rows = _verify_hammer_entry(
        entry, RESULT_HAMMER_SCHEMA, RESULT_HAMMER_STATUS, "result hammer")
    require(review.get("capture_result") == capture_entry,
            "result hammer does not cross-bind actual capture")
    identity = review.get("identity")
    require(type(identity) is dict and set(identity) == IDENTITY_KEYS and
            type(identity["epoch"]) is int and identity["epoch"] == 34,
            "result hammer ep34 identity mismatch")
    for key in IDENTITY_KEYS - {"epoch"}:
        lowercase_sha(identity[key], "result hammer " + key)
    frozen = capture_manifest.get("m1227_runtime_contract", {}).get(
        "final_selection_identity", {})
    require(frozen.get("epoch") == 34 and
            frozen.get("checkpoint_sha256") == identity["checkpoint_sha256"] and
            frozen.get("config_sha256") == identity["config_sha256"] and
            frozen.get("profile_sha256") == identity["profile_sha256"],
            "capture manifest and result-hammer identity mismatch")
    require(review.get("independence") == {"different_author": True} and
            review.get("authorization") == {
                "m1328_materialization_release_authoring": True,
                "production_replay": False, "gpu": False, "eda": False},
            "result hammer authorization mismatch")
    return review


RELEASE_KEYS = {
    "schema", "status", "contract_path", "release_identity", "source_hammer",
    "capture_result", "capture_result_hammer", "one_shot", "output",
    "claim_boundary",
}


def validate_release_shape(release: Any, release_path: Path) -> None:
    require(type(release) is dict and set(release) == RELEASE_KEYS,
            "production release keys drift")
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == RELEASE_STATUS,
            "source-only/nonproduction release cannot materialize")
    require(release.get("contract_path") == str(release_path.relative_to(ROOT)),
            "release path identity mismatch")
    require(release.get("release_identity") == {
        "source_path": str(SOURCE_FILE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE_FILE),
        "test_path": str(TEST.relative_to(ROOT)), "test_sha256": sha256(TEST),
        "source_contract_path": str(SOURCE_CONTRACT.relative_to(ROOT)),
        "source_contract_sha256": sha256(SOURCE_CONTRACT)},
            "release source/test/contract identity mismatch")
    require(release.get("one_shot") == {
        "attempt_marker": str(ATTEMPT.relative_to(ROOT)),
        "automatic_retry": False, "maximum_materializations": 1},
            "one-shot policy mismatch")
    require(release.get("output") == {
        "path": str(OUTPUT.relative_to(ROOT)), "atomic_no_replace": True,
        "recursive_double_seal": True}, "output policy mismatch")
    require(release.get("claim_boundary") == {
        "bitplane_materialization": True, "decoder_replay": False,
        "cycles": False, "traffic": False, "speedup": False,
        "system_speedup": False, "energy": False, "rtl": False,
        "eda": False, "ppa": False}, "release claim boundary mismatch")


def validate_production_release(release_path: Path) -> dict[str, Any]:
    require(release_path.resolve() == FUTURE_RELEASE,
            "only canonical future M1328 release path allowed")
    regular(release_path, "production release")
    release = strict_json(release_path)
    validate_release_shape(release, release_path)
    verify_m1324_hammer()
    verify_m1111dr2_template()
    verify_source_hammer(release["source_hammer"])
    _rows, manifest, _admission = validate_capture_entry(release["capture_result"])
    result_review = verify_result_hammer(
        release["capture_result_hammer"], release["capture_result"], manifest)
    audit = M1323.audit_capture(M1327_CAPTURE)
    require(audit["claim_boundary"]["source_only"] is True and
            audit["claim_boundary"]["production_replay"] is False and
            audit["population"] == {"samples": 30, "calls": 120, "modules": 4,
                                    "global_sample_ids": [10, 39]},
            "M1323 actual-capture audit boundary/population drift")
    return {"release": release, "capture_manifest": manifest,
            "result_hammer": result_review, "audit": audit}


def output_plane_names(global_call_ordinal: int, global_sample_id: int,
                       module_ordinal: int) -> tuple[str, str]:
    require(type(global_call_ordinal) is int and 0 <= global_call_ordinal < 120 and
            type(global_sample_id) is int and 10 <= global_sample_id < 40 and
            type(module_ordinal) is int and 0 <= module_ordinal < 4,
            "output plane ordinal identity invalid")
    stem = "payloads/c{:03d}_s{:02d}_d{}".format(
        global_call_ordinal, global_sample_id, module_ordinal)
    return stem + ".positive.le.bitpack", stem + ".negative.le.bitpack"


def build_output_manifest(audit: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    calls = audit.get("calls")
    require(type(calls) is list and len(calls) == EXPECTED_CALLS,
            "audited decoder calls are not 120")
    theta = audit.get("d1", {}).get("theta_word_uint32")
    require(type(theta) is int and M1323.M1321.positive_finite_word(theta),
            "dynamic D1 theta is not positive finite FP32 word")
    records = []
    positive_paths = set()
    negative_paths = set()
    for ordinal, call in enumerate(calls):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        require(call.get("global_call_ordinal") == ordinal and
                call.get("global_sample_id") == sample and
                call.get("module_ordinal") == module and
                call.get("module") == M1323.MODULES[module] and
                call.get("shape") == list(M1323.SHAPES[module]),
                "audited 30x4 call identity/order drift")
        elements = math.prod(call["shape"])
        plane_bytes = (elements + 7) // 8
        require(call["positive_plane_bytes"] == plane_bytes and
                call["negative_plane_bytes"] == plane_bytes and
                call["negative_count"] == 0,
                "audited plane extent/sign drift")
        positive, negative = output_plane_names(ordinal, sample, module)
        require(positive not in positive_paths and negative not in negative_paths,
                "output bitplane path alias")
        positive_paths.add(positive); negative_paths.add(negative)
        records.append({
            "global_call_ordinal": ordinal, "capture_global_order": call["global_order"],
            "global_sample_id": sample, "replay_sample_ordinal": sample - 10,
            "sequence": call["sequence"], "sequence_sample_id": (sample - 10) % 10,
            "module_ordinal": module, "module": call["module"],
            "shape": call["shape"], "elements": elements, "plane_bytes": plane_bytes,
            "source_support_sign": call["support_sign"],
            "source_support_sign_sha256": call["support_sign_sha256"],
            "source_raw_fp32_sha256": call["raw_fp32_sha256"],
            "positive_output": positive,
            "positive_sha256": call["positive_plane_sha256"],
            "negative_output": negative,
            "negative_sha256": call["negative_plane_sha256"],
            "negative_count": 0,
            "numeric_encoding": ({"kind": "bit_times_dynamic_theta",
                                  "theta_word_uint32": theta,
                                  "theta_ieee754_le_hex": audit["d1"]["theta_ieee754_le_hex"],
                                  "weight_folding": False, "coerced_to_one": False}
                                 if module == 1 else
                                 {"kind": "exact_binary", "theta_word_uint32": None,
                                  "weight_folding": False, "coerced_to_one": False}),
        })
    require(len(positive_paths) == EXPECTED_CALLS and len(negative_paths) == EXPECTED_CALLS,
            "output bitplane population not unique 120+120")
    identity = authority["result_hammer"]["identity"]
    return {
        "schema": "m1328_ep34_decoder_bitplane_materialization_r1_v1",
        "status": "MATERIALIZATION_COMPLETE__DECODER_REPLAY_NOT_RUN__NO_PERFORMANCE_CLAIM",
        "capture_authority": {
            "capture_result": authority["release"]["capture_result"],
            "capture_result_hammer": authority["release"]["capture_result_hammer"],
            "epoch": identity["epoch"], "checkpoint_sha256": identity["checkpoint_sha256"],
            "config_sha256": identity["config_sha256"],
            "profile_sha256": identity["profile_sha256"],
            "ordered_jsonl_sha256": audit["ordered_jsonl_sha256"],
            "ordered_identity": audit["ordered_identity"],
        },
        "population": {"global_samples": [10, 39], "samples": 30,
                       "calls": 120, "modules_per_sample": 4,
                       "bitplane_files": 240, "negative_nonzero": 0,
                       "global_call_ordinals_contiguous": True},
        "d1_dynamic_theta": {"word_uint32": theta,
                             "ieee754_le_hex": audit["d1"]["theta_ieee754_le_hex"],
                             "calls": 30, "weight_folding": False,
                             "coerced_to_one": False},
        "weight_identity": {"present": False,
                            "required_before_decoder_replay": True},
        "records": records,
        "claim_boundary": {"bitplane_materialization": True,
                           "decoder_replay": False, "cycles": False,
                           "traffic": False, "speedup": False,
                           "system_speedup": False, "energy": False,
                           "rtl": False, "eda": False, "ppa": False},
    }


def write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_plane_exclusive(source: Path, offset: int, count: int,
                         destination: Path, expected_sha: str) -> None:
    regular(source, "source support/sign payload")
    require(type(offset) is int and offset >= 0 and type(count) is int and count > 0 and
            source.stat().st_size == 2 * count and offset in (0, count),
            "source support/sign plane geometry drift")
    descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            stream.seek(offset)
            remaining = count
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                require(block, "source support/sign plane truncated")
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                remaining -= len(block)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(digest.hexdigest() == lowercase_sha(expected_sha, "plane SHA") and
            destination.stat().st_size == count,
            "materialized bitplane SHA/extent mismatch")


def fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def payload_files(root: Path) -> list[Path]:
    output = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "output seal refuses symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in {MANIFEST, OUTER}:
            output.append(path)
    return output


def seal_staging(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink() and
            not (root / MANIFEST).exists() and not (root / OUTER).exists(),
            "bad staging seal target")
    members = payload_files(root)
    require(len(members) == EXPECTED_FILES + 2, "staging member population is not 242")
    fsync_dir(root / "payloads")
    lines = [sha256(path) + "  " + path.relative_to(root).as_posix()
             for path in members]
    write_exclusive(root / MANIFEST, ("\n".join(lines) + "\n").encode(), 0o400)
    write_exclusive(root / OUTER,
                    (sha256(root / MANIFEST) + "  " + MANIFEST + "\n").encode(), 0o400)
    fsync_dir(root)
    return verify_materialized_seal(root)


def verify_materialized_seal(root: Path) -> dict[str, Any]:
    manifest = root / MANIFEST; outer = root / OUTER
    regular(manifest, "output manifest seal"); regular(outer, "output outer seal")
    require(outer.read_text(encoding="utf-8").split() == [sha256(manifest), MANIFEST],
            "output outer seal drift")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        require(relative not in rows, "duplicate output seal member")
        member = safe_member(root, relative, "output sealed member")
        require(sha256(member) == digest, "output sealed member drift")
        rows[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in payload_files(root)}
    require(actual == set(rows) and len(rows) == EXPECTED_FILES + 2,
            "output recursive seal coverage drift")
    output_manifest = strict_json(root / "manifest.json")
    require(output_manifest["population"]["calls"] == 120 and
            len(output_manifest["records"]) == 120,
            "output materialization manifest population drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer), "members": len(rows)}


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
            raise M1328Error("atomic output no-replace collision")
        raise OSError(code, os.strerror(code), str(destination))


def namespace_fresh() -> None:
    require(not os.path.lexists(str(OUTPUT)) and not os.path.lexists(str(ATTEMPT)) and
            not any(OUTPUT.parent.glob(WORK_PREFIX + "*")),
            "M1328 materialization namespace is not fresh")


def consume_attempt() -> None:
    write_exclusive(ATTEMPT, ATTEMPT_TOKEN.encode("ascii"), 0o400)


def execute_once(release_path: Path) -> Path:
    """Future release hook.  Not reachable from this source's CLI."""
    authority = validate_production_release(release_path)
    output_manifest = build_output_manifest(authority["audit"], authority)
    namespace_fresh()
    consume_attempt()
    staging = OUTPUT.parent / (WORK_PREFIX + str(os.getpid()) + "." + str(time.time_ns()))
    staging.mkdir(mode=0o700)
    (staging / "payloads").mkdir(mode=0o700)
    try:
        for call, record in zip(authority["audit"]["calls"], output_manifest["records"]):
            source = safe_member(M1327_CAPTURE, call["support_sign"], "capture support/sign")
            count = call["positive_plane_bytes"]
            positive = staging.joinpath(*PurePosixPath(record["positive_output"]).parts)
            negative = staging.joinpath(*PurePosixPath(record["negative_output"]).parts)
            copy_plane_exclusive(source, 0, count, positive, record["positive_sha256"])
            copy_plane_exclusive(source, count, count, negative, record["negative_sha256"])
        write_exclusive(staging / "manifest.json",
                        (json.dumps(output_manifest, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n").encode(), 0o400)
        write_exclusive(staging / "RUN_COMPLETE.txt",
                        b"PASS_M1328_EP34_DECODER_BITPLANE_MATERIALIZATION\n", 0o400)
        seal_staging(staging)
        rename_noreplace(staging, OUTPUT)
        fsync_dir(OUTPUT.parent)
        verify_materialized_seal(OUTPUT)
    except BaseException:
        # Preserve the exclusive staging directory for one-shot failure forensics.
        raise
    return OUTPUT


def validate_source_policy() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "M1328 source policy mismatch")
    require(policy.get("source") == {
        "path": str(SOURCE_FILE.relative_to(ROOT)), "sha256": sha256(SOURCE_FILE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)} and
            policy.get("actual_m1327_result") == {
                "present": False, "sha256_predeclared": False,
                "result_hammer_present": False},
            "M1328 source/test/future-result policy mismatch")
    require(policy.get("production_authorized") is False,
            "source policy cannot authorize materialization")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    require(args.source_self_check, "M1328 is source-only; materialization CLI is forbidden")
    validate_source_policy()
    verify_m1324_hammer()
    verify_m1111dr2_template()
    require(not FUTURE_RELEASE.exists(), "future production release already exists")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1328Error as error:
        print("M1328_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
