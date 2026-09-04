#!/usr/bin/env python3
"""Additive, fail-closed successor to the rejected M1331 result hammer.

The canonical M1327 result is read only with ``--validate-canonical-result``.
The source self-check never reads or creates it.  M1331 remains immutable and
FAIL_DO_NOT_CITE; this successor binds that failure and closes its four P0s.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Sequence


SOURCE_FILE = Path(__file__).resolve()
ROOT = SOURCE_FILE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
OLD_SOURCE = HW / "scripts/hammer_m1331_m1327_final_ep34_capture_result_source.py"
OLD_SOURCE_SHA256 = "44297a2225be726d56b5769ef536458148933f489e1ea8c318dde779afbff5b1"
OLD_TEST = HW / "tests/test_hammer_m1331_m1327_final_ep34_capture_result_source.py"
OLD_TEST_SHA256 = "a443885767d955a79962e0ee2509fecc9aa0cc6e15601029beb39a05a180679a"
OLD_CONTRACT = HW / "contracts/m1331_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
OLD_CONTRACT_SHA256 = "57a779d27f8bdec7afae7f8a72aa8142badfb3dc49bd72fbc56f965cce3d145a"
FAIL_REVIEW = HW / "reviews/m1332_m1331_m1327_capture_result_hammer_source_blind_review_r1_20260831"
FAIL_MANIFEST_SHA256 = "785efc5bbde4c7e0bce6889781ef4c2859a1c7bbfd1f4f2c3c2ebf2def36e63a"
FAIL_OUTER_FILE_SHA256 = "aa73823c29230532480abdfcd85a6924fc2d492ff7055691cccb47b6a2a1e201"
FAIL_REVIEW_SHA256 = "ee618b7a3f7150e8a2ae127aa7b9d94c05b8ed6d9169ac0305ac7208ff872d9c"
M1323_SOURCE = HW / "system_simulator/scripts/build_m1323_ep34_decoder_capture_adapter_source.py"
M1323_SOURCE_SHA256 = "0481e39372ffe19cd3cff8d5053c9eae8326de4fb5ac61bd9e42527a3ad3a12a"
M1323_TEST = HW / "system_simulator/tests/test_m1323_ep34_decoder_capture_adapter_source.py"
M1323_TEST_SHA256 = "c29980f357ea0e0a9b2e11650239b706f6c4e18892b4975925db164a72439487"
M1323_CONTRACT = HW / "contracts/m1323_ep34_decoder_capture_adapter_source_contract_r1_20260831.json"
M1323_CONTRACT_SHA256 = "e4df50fed6068b0f384693044705b30f595d41d70dce78e738cb36a98e24cecc"
CANONICAL_RESULT = HW / "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"
SOURCE_CONTRACT = HW / "contracts/m1333_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
TEST = HW / "tests/test_hammer_m1333_m1327_final_ep34_capture_result_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SCHEMA = "m1333_m1327_final_ep34_capture_result_hammer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1331_REJECTED__CANONICAL_RESULT_MUST_PREEXIST__NO_CAPTURE"
PASS_TOKEN = "PASS_M1333_SOURCE_SELF_CHECK__FIXTURES_ONLY_NO_CANONICAL_RESULT"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"


class M1333Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1333Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise M1333Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")


def regular_exact(path: Path, digest: str, label: str) -> None:
    regular(path, label)
    require(sha256(path) == digest, label + " SHA drift")


def strict_text(raw: str) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              M1333Error("nonfinite JSON: " + token)))
    except (ValueError, TypeError) as error:
        raise M1333Error("invalid JSON") from error


def strict_json(path: Path) -> dict[str, Any]:
    regular(path, str(path))
    value = strict_text(path.read_text(encoding="utf-8"))
    require(type(value) is dict, "JSON root is not object")
    return value


def strict_file(path: Path) -> Any:
    regular(path, str(path))
    return strict_text(path.read_text(encoding="utf-8"))


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_exact("m1333_sealed_m1331", OLD_SOURCE, OLD_SOURCE_SHA256)
M1323 = load_exact("m1333_sealed_m1323", M1323_SOURCE, M1323_SOURCE_SHA256)
M1227 = OLD.M1227


def recursive_population(root: Path) -> tuple[set[str], set[str]]:
    """Return regular files/directories while rejecting every symlink, broken too."""
    files: set[str] = set()
    directories: set[str] = set()
    stack = [Path(root)]
    while stack:
        parent = stack.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                relative = Path(entry.path).relative_to(root).as_posix()
                require(not entry.is_symlink(), "recursive population contains symlink: " + relative)
                if entry.is_dir(follow_symlinks=False):
                    directories.add(relative)
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files.add(relative)
                else:
                    raise M1333Error("recursive population contains special member: " + relative)
    return files, directories


def safe_member(root: Path, relative: str) -> Path:
    require(type(relative) is str, "sealed member path not string")
    pure = PurePosixPath(relative)
    require(pure.parts and not pure.is_absolute() and ".." not in pure.parts and
            pure.as_posix() == relative and relative not in {MANIFEST, OUTER},
            "unsafe sealed member path")
    cursor = Path(root)
    for part in pure.parts:
        cursor = cursor / part
        require(os.path.lexists(str(cursor)) and not cursor.is_symlink(),
                "missing/symlink sealed component")
    regular(cursor, "sealed member")
    return cursor


def verify_recursive_seal(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    require(Path(root).is_dir() and not Path(root).is_symlink(), "result root invalid")
    files, _directories = recursive_population(root)
    manifest = Path(root) / MANIFEST
    outer = Path(root) / OUTER
    regular(manifest, "result manifest seal")
    regular(outer, "result outer seal")
    require(outer.read_text(encoding="ascii").split() == [sha256(manifest), MANIFEST],
            "outer seal mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "seal row malformed")
        relative = fields[1].lstrip("*")
        require(relative not in rows, "duplicate sealed member")
        member = safe_member(root, relative)
        require(sha256(member) == fields[0], "sealed member SHA mismatch")
        rows[relative] = fields[0]
    require(files - {MANIFEST, OUTER} == set(rows), "recursive sealed population mismatch")
    return rows, {"manifest_sha256": sha256(manifest),
                  "outer_file_sha256": sha256(outer)}


def verify_failed_predecessor() -> dict[str, Any]:
    regular_exact(OLD_TEST, OLD_TEST_SHA256, "M1331 test")
    regular_exact(OLD_CONTRACT, OLD_CONTRACT_SHA256, "M1331 contract")
    rows, seal = verify_recursive_seal(FAIL_REVIEW)
    require(seal == {"manifest_sha256": FAIL_MANIFEST_SHA256,
                     "outer_file_sha256": FAIL_OUTER_FILE_SHA256},
            "M1332 failure seal identity drift")
    require(rows.get("review.json") == FAIL_REVIEW_SHA256,
            "M1332 failure review member drift")
    review = strict_json(FAIL_REVIEW / "review.json")
    require(review.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED" and
            review.get("authorization", {}).get("additive_successor_source_authoring") is True and
            review.get("authorization", {}).get("production_result_hammer") is False,
            "M1332 failure authority drift")
    return review


def validate_checkpoint_load_audit(identity: dict[str, Any]) -> None:
    require("checkpoint_load_audit" in identity and
            type(identity["checkpoint_load_audit"]) is dict,
            "checkpoint_load_audit missing/not object")
    audit = identity["checkpoint_load_audit"]
    for key in ("missing_count", "unexpected_count"):
        require(key in audit and type(audit[key]) is int and audit[key] == 0,
                "checkpoint_load_audit " + key + " missing/type/value drift")


def validate_identity(manifest: dict[str, Any]) -> None:
    require(manifest.get("schema") ==
            "m1227_motion_final_checkpoint_unified_hardware_capture_r1_v1" and
            manifest.get("status") ==
            "CAPTURE_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "M1227 manifest schema/status drift")
    identity = manifest.get("identity")
    require(type(identity) is dict, "identity missing/not object")
    require(identity.get("contract_sha256") == OLD.RUNTIME_SHA256,
            "runtime contract identity drift")
    validate_checkpoint_load_audit(identity)
    require(identity.get("module_counts") ==
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "module counts drift")
    selected = identity.get("selection", {}).get("selected", {})
    require(selected.get("candidate_id") == "resume_ep34" and
            type(selected.get("epoch")) is int and selected["epoch"] == 34,
            "ep34 selection drift")
    require(selected.get("checkpoint", {}).get("sha256") == OLD.CHECKPOINT_SHA256 and
            selected.get("configuration", {}).get("sha256") == OLD.CONFIG_SHA256 and
            selected.get("profile", {}).get("sha256") == OLD.PROFILE_SHA256,
            "selected artifact SHA drift")
    require(selected["profile"].get("samples") == 825 and
            selected["profile"].get("module_counts") ==
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "selected profile drift")
    final = manifest.get("m1227_runtime_contract", {}).get("final_selection_identity", {})
    require(final == {"epoch": 34, "checkpoint_sha256": OLD.CHECKPOINT_SHA256,
                      "config_sha256": OLD.CONFIG_SHA256,
                      "profile_sha256": OLD.PROFILE_SHA256,
                      "selection_sha256": OLD.SELECTION_SHA256},
            "final selection identity drift")


def validate_attention(root: Path, rows: dict[str, str]) -> dict[str, int]:
    manifest = strict_json(root / "attention_qk/manifest.json")
    records = manifest.get("records")
    require(type(records) is list, "attention records missing/not list")
    try:
        audit = M1227.audit_attention_population(records, samples=40)
    except Exception as error:
        raise M1333Error("attention 40x12 Cartesian audit") from error
    import numpy as np
    for row in records:
        require(type(row) is dict and type(row.get("sample_id")) is int and
                type(row.get("name")) is str and type(row.get("file")) is str and
                type(row.get("sha256")) is str and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
                "attention record identity/path/SHA malformed")
        safe_name = row["name"].replace(".", "_").replace("/", "_")
        expected_name = "sample{}_{}.npz".format(row["sample_id"], safe_name)
        require(Path(row["file"]).name == expected_name,
                "attention record filename identity drift")
        relative = "attention_qk/" + expected_name
        payload = safe_member(root, relative)
        require(rows.get(relative) == row["sha256"] and sha256(payload) == row["sha256"],
                "attention NPZ seal/record SHA mismatch")
        try:
            with np.load(payload, allow_pickle=False) as data:
                require({"q_bits_packed", "k_bits_packed", "gate_q17"} <= set(data.files) and
                        data["q_bits_packed"].size > 0 and
                        data["k_bits_packed"].size > 0 and
                        data["gate_q17"].size > 0,
                        "attention NPZ content incomplete")
        except M1333Error:
            raise
        except Exception as error:
            raise M1333Error("attention NPZ unreadable") from error
    return audit


def validate_result(root: Path) -> dict[str, Any]:
    rows, seal = verify_recursive_seal(root)
    required = {"manifest.json", "m1227_admission.json",
                "unified_ordered_records.jsonl", "attention_qk/manifest.json",
                "execution_trace.json", "operator_runtime.json",
                "atlif_activity.json", "RUN_COMPLETE.txt"}
    require(required <= set(rows), "required sealed members missing")
    manifest = strict_json(root / "manifest.json")
    validate_identity(manifest)
    admission = strict_json(root / "m1227_admission.json")
    require(admission == {"schema": "m1227_final_capture_admission_r1_v1",
             "status": "PASS", "ordered": 9880, "attention": 480,
             "payload_files": 640, "execution": 7360, "operator_rows": 79,
             "atlif_live_rows": 93, "atlif_static": 105,
             "dead_sn_v": list(M1227.DEAD_SN_V),
             "claim_boundary": {"capture_only": True, "paper_result": False,
                                "cycles": False, "speedup": False,
                                "energy": False, "ppa": False}},
            "M1227 admission drift")
    runtime = manifest.get("m1227_runtime_contract", {})
    require(runtime.get("static_modules") == 259 and runtime.get("static_atlif") == 105 and
            runtime.get("live_modules_per_sample") == 247 and runtime.get("live_atlif") == 93 and
            runtime.get("dead_sn_v") == list(M1227.DEAD_SN_V) and
            runtime.get("dead_calls_per_sample") == 0 and
            runtime.get("ordered_records") == 9880 and
            runtime.get("attention_records") == 480 and runtime.get("payload_files") == 640,
            "M1227 runtime admission drift")
    observed = manifest.get("cohort", {}).get("samples")
    expected = OLD.expected_cohort()
    require(type(observed) is list and len(observed) == 40 and
            [{key: row[key] for key in expected[0]} for row in observed] == expected,
            "cohort order/SHA drift")

    ordered_path = root / "unified_ordered_records.jsonl"
    ordered = [strict_text(line) for line in ordered_path.read_text(encoding="utf-8").splitlines()]
    try:
        decoder_calls, ordered_identity = M1323.decoder_rows_from_ordered(ordered)
    except Exception as error:
        raise M1333Error("M1323 full ordered/global-order/frozen-247 audit") from error
    require(len(decoder_calls) == 120 and ordered_identity["ordered_rows"] == 9880 and
            ordered_identity["all_sample_sequences_equal"] is True,
            "M1323 ordered projection drift")
    attention_audit = validate_attention(root, rows)
    try:
        payloads = M1227.validate_payload_population(root)
    except Exception as error:
        raise M1333Error("payload population audit") from error
    require(len(payloads) == 640, "payload population is not 640")

    execution = strict_file(root / "execution_trace.json")
    operators = strict_file(root / "operator_runtime.json")
    atlif = strict_file(root / "atlif_activity.json")
    require(type(execution) is list and len(execution) == 7360, "execution population drift")
    require(type(operators) is list and len(operators) == 79 and
            len({row["name"] for row in operators}) == 79 and
            all(type(row.get("calls")) is int and row["calls"] == 40 for row in operators),
            "operator population drift")
    require(type(atlif) is list and len(atlif) == 93 and
            len({row["name"] for row in atlif}) == 93 and
            all(type(row.get("calls")) is int and row["calls"] == 40 for row in atlif) and
            not ({row["name"] for row in atlif} & set(M1227.DEAD_SN_V)),
            "ATLIF population drift")
    require((root / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n",
            "completion token drift")
    require(manifest.get("claim_boundary") == {
        "capture_only": True, "accuracy": False, "cycles": False,
        "speedup": False, "system_speedup": False, "energy": False,
        "rtl": False, "ppa": False, "fresh_result_hammer_required": True},
        "claim boundary drift")
    return {"status": "PASS_M1333_M1327_EP34_CAPTURE_RESULT",
            "seal": seal,
            "population": {"ordered": 9880, "attention": attention_audit["records"],
                           "payload": 640, "execution": 7360,
                           "operator": 79, "atlif": 93},
            "identity": {"checkpoint_sha256": OLD.CHECKPOINT_SHA256,
                         "config_sha256": OLD.CONFIG_SHA256,
                         "profile_sha256": OLD.PROFILE_SHA256},
            "claim_boundary": {"capture_only": True, "paper_result": False}}


def validate_source_policy() -> dict[str, Any]:
    verify_failed_predecessor()
    regular_exact(M1323_TEST, M1323_TEST_SHA256, "M1323 test")
    regular_exact(M1323_CONTRACT, M1323_CONTRACT_SHA256, "M1323 contract")
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "source policy schema/status drift")
    require(policy.get("source") == {
        "path": str(SOURCE_FILE.relative_to(ROOT)), "sha256": sha256(SOURCE_FILE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "source/test policy identity drift")
    require(policy.get("actual_result_seal_prefilled") is False and
            policy.get("production_authorized") is False and
            policy.get("predecessor_m1331") == "FAIL_DO_NOT_CITE",
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
        require(not CANONICAL_RESULT.exists(),
                "source self-check refuses an already-present canonical result")
        print(PASS_TOKEN)
        return 0
    require(CANONICAL_RESULT.exists(), "canonical M1327 result does not yet exist")
    print(json.dumps(validate_result(CANONICAL_RESULT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1333Error as error:
        print("M1333_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
