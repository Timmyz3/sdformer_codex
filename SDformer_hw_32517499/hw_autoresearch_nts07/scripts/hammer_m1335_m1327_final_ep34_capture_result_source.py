#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive successor to rejected M1333; read-only final M1327 result hammer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Sequence
import zlib


SOURCE_FILE = Path(__file__).resolve()
ROOT = SOURCE_FILE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
M1333_SOURCE = HW / "scripts/hammer_m1333_m1327_final_ep34_capture_result_source.py"
M1333_SOURCE_SHA256 = "7522be99557b23c6be7feee3b3b69b2d1825118d724bb7b2379a7a24aee3bc52"
M1333_TEST = HW / "tests/test_hammer_m1333_m1327_final_ep34_capture_result_source.py"
M1333_TEST_SHA256 = "9bc86e030d8e6d09daf9cca04fdd93ad3419244d1a60ebbc03432bcfff69422d"
M1333_CONTRACT = HW / "contracts/m1333_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
M1333_CONTRACT_SHA256 = "9323e431bb75d534e465cbfb87d81892b5e875c71c646b1cb509527e928120b8"
M1334_FAIL = HW / "reviews/m1334_m1333_m1327_final_ep34_capture_result_hammer_source_blind_review_r1_20260831"
# Filled only from the completed different-author recursive seal.
M1334_FAIL_REVIEW_SHA256 = "07934ce56d2168cf820da3454b5086b263ad6056b70aafeb7dc638a50ec6191e"
M1334_FAIL_MANIFEST_SHA256 = "39e519447f1f27aae551da1df3431c135198d6d6d8be8ac55b748e740426f586"
M1334_FAIL_OUTER_FILE_SHA256 = "55472f06537893e9473b0fa7fd203a9e931b4f8a92db817aeee06d0dfa3761a2"
CANONICAL_RESULT = HW / "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"
SOURCE_CONTRACT = HW / "contracts/m1335_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
TEST = HW / "tests/test_hammer_m1335_m1327_final_ep34_capture_result_source.py"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
NUMPY_VERSION = "2.1.2"
NUMPY_INIT_SHA256 = "39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SCHEMA = "m1335_m1327_final_ep34_capture_result_hammer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1333_REJECTED__PINNED_PYTHON_NUMPY__NO_CAPTURE"
PASS_TOKEN = "PASS_M1335_SOURCE_SELF_CHECK__FIXTURES_ONLY_NO_CANONICAL_RESULT"


class M1335Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1335Error(message)


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
        raise M1335Error("missing " + label) from error
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


OLD = load_exact("m1335_sealed_m1333", M1333_SOURCE, M1333_SOURCE_SHA256)
M1323 = OLD.M1323
M1227 = OLD.M1227


def strict_json(path: Path) -> dict[str, Any]:
    value = OLD.strict_file(path)
    require(type(value) is dict, "JSON root is not object")
    return value


def verify_failed_predecessor() -> dict[str, Any]:
    regular_exact(M1333_TEST, M1333_TEST_SHA256, "M1333 test")
    regular_exact(M1333_CONTRACT, M1333_CONTRACT_SHA256, "M1333 contract")
    rows, seal = OLD.verify_recursive_seal(M1334_FAIL)
    require(seal == {"manifest_sha256": M1334_FAIL_MANIFEST_SHA256,
                     "outer_file_sha256": M1334_FAIL_OUTER_FILE_SHA256},
            "M1334 final blind seal drift")
    require(rows.get("review.json") == M1334_FAIL_REVIEW_SHA256,
            "M1334 final blind review member drift")
    review = strict_json(M1334_FAIL / "review.json")
    require(review.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED" and
            review.get("authorization", {}).get("additive_successor_source_authoring") is True and
            review.get("authorization", {}).get("production_result_hammer") is False and
            review.get("false_negative_count") == 5,
            "M1334 final blind authority drift")
    return review


def canonical_absent(path: Path = CANONICAL_RESULT) -> None:
    require(not os.path.lexists(str(path)),
            "canonical namespace residue exists, including possible broken symlink")


def canonical_directory(path: Path = CANONICAL_RESULT) -> None:
    require(os.path.lexists(str(path)), "canonical result absent")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1335Error("canonical result disappeared") from error
    require(stat.S_ISDIR(mode) and not path.is_symlink(),
            "canonical result is not a real directory")


def validate_retained_payloads(root: Path, seal_rows: dict[str, str],
                               ordered: list[dict[str, Any]]) -> int:
    retained = 0
    for row in ordered:
        payload = row["payload"]
        if payload.get("retained") is not True:
            continue
        retained += 1
        for path_key, sha_key in (("compressed_fp32", "compressed_sha256"),
                                  ("support_sign", "support_sign_sha256")):
            relative = payload[path_key]
            member = OLD.safe_member(root, relative)
            record_sha = payload[sha_key]
            require(seal_rows.get(relative) == record_sha == sha256(member),
                    "retained record/seal/actual SHA mismatch: " + relative)
        compressed = OLD.safe_member(root, payload["compressed_fp32"])
        try:
            raw = zlib.decompress(compressed.read_bytes())
        except Exception as error:
            raise M1335Error("retained compressed payload is not valid zlib") from error
        require(hashlib.sha256(raw).hexdigest() == payload["raw_fp32_sha256"],
                "retained raw_fp32_sha256 mismatch")
        support = OLD.safe_member(root, payload["support_sign"])
        require(support.stat().st_size ==
                payload["positive_plane_bytes"] + payload["negative_plane_bytes"],
                "retained support two-plane extent mismatch")
    require(retained == 320, "retained payload population is not 320")
    return retained


def frozen_operator_rows() -> list[dict[str, str]]:
    M1323.regular_exact(M1323.OPERATOR_AUTHORITY, M1323.OPERATOR_AUTHORITY_SHA256,
                        "operator authority")
    with M1323.OPERATOR_AUTHORITY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 79, "operator authority population drift")
    return rows


def frozen_atlif_rows() -> list[dict[str, str]]:
    M1323.regular_exact(M1323.ATLIF_AUTHORITY, M1323.ATLIF_AUTHORITY_SHA256,
                        "ATLIF authority")
    with M1323.ATLIF_AUTHORITY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 93, "ATLIF authority population drift")
    return rows


def validate_runtime_identities(root: Path) -> None:
    operators = OLD.strict_file(root / "operator_runtime.json")
    operator_authority = frozen_operator_rows()
    require(type(operators) is list and len(operators) == 79, "operator rows invalid")
    require([(row.get("name"), row.get("operator"), row.get("scope")) for row in operators] ==
            [(row["name"], row["operator"], row["scope"]) for row in operator_authority],
            "operator frozen identity/order drift")
    require(all(type(row.get("calls")) is int and row["calls"] == 40
                for row in operators), "operator calls not exact 40")

    atlif = OLD.strict_file(root / "atlif_activity.json")
    atlif_authority = frozen_atlif_rows()
    require(type(atlif) is list and len(atlif) == 93, "ATLIF rows invalid")
    require([(row.get("name"), row.get("output_mode"), row.get("threshold_mode"))
             for row in atlif] ==
            [(row["name"], row["output_mode"], row["threshold_mode"])
             for row in atlif_authority], "ATLIF frozen identity/order drift")
    require(all(type(row.get("calls")) is int and row["calls"] == 40 for row in atlif),
            "ATLIF calls not exact 40")
    require(not ({row["name"] for row in atlif} & set(M1227.DEAD_SN_V)),
            "dead ATLIF identity present")


def exact_int(row: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = row.get(key)
    require(type(value) is int and value >= minimum,
            "attention record " + key + " not exact integer")
    return value


def validate_attention_geometry(root: Path, seal_rows: dict[str, str]) -> int:
    import numpy as np
    manifest = strict_json(root / "attention_qk/manifest.json")
    records = manifest.get("records")
    require(type(records) is list, "attention records missing/not list")
    try:
        M1227.audit_attention_population(records, samples=40)
    except Exception as error:
        raise M1335Error("attention Cartesian identity drift") from error
    for row in records:
        sample = exact_int(row, "sample_id")
        require(type(row.get("name")) is str and row["name"] in M1227.ATTENTION_ALIASES,
                "attention alias drift")
        windows = exact_int(row, "windows_captured", 1)
        heads = exact_int(row, "heads", 1)
        spatial = exact_int(row, "spatial_tokens", 1)
        temporal = exact_int(row, "temporal_tokens", 1)
        lanes = exact_int(row, "lanes", 1)
        require(temporal == 2 * spatial, "attention temporal/spatial geometry drift")
        for key in ("q_active_bits", "k_active_bits", "gate_nonzero",
                    "gate_min", "gate_max"):
            exact_int(row, key)
        require(row["gate_min"] <= row["gate_max"] <= 256,
                "attention gate record range drift")
        safe_name = row["name"].replace(".", "_").replace("/", "_")
        relative = "attention_qk/sample{}_{}.npz".format(sample, safe_name)
        require(Path(row.get("file", "")).name == Path(relative).name,
                "attention basename drift")
        payload = OLD.safe_member(root, relative)
        require(type(row.get("sha256")) is str and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None and
                seal_rows.get(relative) == row["sha256"] == sha256(payload),
                "attention record/seal/actual SHA mismatch")
        try:
            with np.load(payload, allow_pickle=False) as data:
                required = {"q_shape", "k_shape", "q_bits_packed",
                            "k_bits_packed", "gate_q17"}
                require(required <= set(data.files), "attention NPZ required keys missing")
                q_shape = data["q_shape"]; k_shape = data["k_shape"]
                q_bits = data["q_bits_packed"]; k_bits = data["k_bits_packed"]
                gate = data["gate_q17"]
                expected_shape = [2, windows, heads, spatial, lanes]
                require(q_shape.dtype == np.dtype("int32") and
                        k_shape.dtype == np.dtype("int32") and
                        q_shape.ndim == 1 and k_shape.ndim == 1 and
                        q_shape.tolist() == expected_shape and
                        k_shape.tolist() == expected_shape,
                        "attention q/k shape metadata drift")
                q_elements = math.prod(expected_shape)
                require(q_bits.dtype == np.dtype("uint8") and q_bits.ndim == 1 and
                        q_bits.size == (q_elements + 7) // 8 and
                        k_bits.dtype == np.dtype("uint8") and k_bits.ndim == 1 and
                        k_bits.size == (q_elements + 7) // 8,
                        "attention packed q/k dtype or extent drift")
                require(gate.dtype == np.dtype("uint16") and
                        gate.shape == (windows, heads, temporal) and
                        gate.size > 0 and int(gate.max()) <= 256,
                        "attention gate dtype/shape/range drift")
                q_active = int(np.unpackbits(q_bits, bitorder="little")[:q_elements].sum())
                k_active = int(np.unpackbits(k_bits, bitorder="little")[:q_elements].sum())
                require(q_active == row["q_active_bits"] and
                        k_active == row["k_active_bits"] and
                        int(np.count_nonzero(gate)) == row["gate_nonzero"] and
                        int(gate.min()) == row["gate_min"] and
                        int(gate.max()) == row["gate_max"],
                        "attention Q/K/gate record statistic drift")
        except M1335Error:
            raise
        except Exception as error:
            raise M1335Error("attention NPZ unreadable") from error
    require(len(records) == 480, "attention population is not 480")
    return len(records)


def validate_result(root: Path) -> dict[str, Any]:
    canonical_directory(root)
    try:
        OLD.validate_result(root)
    except Exception as error:
        raise M1335Error("M1333 retained validation boundary failed") from error
    seal_rows, seal = OLD.verify_recursive_seal(root)
    ordered = [OLD.strict_text(line) for line in
               (root / "unified_ordered_records.jsonl").read_text(encoding="utf-8").splitlines()]
    retained = validate_retained_payloads(root, seal_rows, ordered)
    validate_runtime_identities(root)
    attention = validate_attention_geometry(root, seal_rows)
    return {"status": "PASS_M1335_M1327_EP34_CAPTURE_RESULT",
            "seal": seal, "population": {"ordered": 9880, "retained": retained,
                                           "attention": attention, "payload": 640,
                                           "execution": 7360, "operator": 79,
                                           "atlif": 93},
            "identity": {"checkpoint_sha256": OLD.OLD.CHECKPOINT_SHA256,
                         "config_sha256": OLD.OLD.CONFIG_SHA256,
                         "profile_sha256": OLD.OLD.PROFILE_SHA256},
            "claim_boundary": {"capture_only": True, "paper_result": False}}


def validate_runtime_identity() -> None:
    regular_exact(PYTHON, PYTHON_SHA256, "pinned Python 3.10")
    require(Path(sys.executable).resolve() == PYTHON.resolve() and
            sys.version_info[:2] == (3, 10), "wrong Python runtime")
    import numpy as np
    require(np.__version__ == NUMPY_VERSION, "NumPy version drift")
    regular_exact(Path(np.__file__), NUMPY_INIT_SHA256, "pinned NumPy init")


def validate_source_policy() -> dict[str, Any]:
    verify_failed_predecessor()
    validate_runtime_identity()
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "source policy schema/status drift")
    require(policy.get("source") == {"path": str(SOURCE_FILE.relative_to(ROOT)),
                                     "sha256": sha256(SOURCE_FILE)} and
            policy.get("test") == {"path": str(TEST.relative_to(ROOT)),
                                   "sha256": sha256(TEST)},
            "source/test policy identity drift")
    require(policy.get("predecessor_m1333") == "FAIL_DO_NOT_CITE" and
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
    except M1335Error as error:
        print("M1335_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
