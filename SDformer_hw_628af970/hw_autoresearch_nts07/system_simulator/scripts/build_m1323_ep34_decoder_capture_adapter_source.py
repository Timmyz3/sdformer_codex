#!/usr/bin/env python3
"""Fail-closed additive successor to the sealed M1321 decoder adapter.

M1321 correctly audited decoder values and the two-plane support encoding, but
an independent M1322 hammer found three projection-boundary holes: ``True``
could alias weight ordinal 1, decoder global orders need not be unique, and an
unselected ordered row could be duplicated/replaced while preserving the row
count.  This successor keeps M1321 immutable and closes those holes *before*
projecting the 120 decoder calls:

* every one of the 9,880 JSONL rows has exact ``global_order == file ordinal``;
* every sample contains the exact frozen 247 live module identities once;
* all row, input-statistic, payload, and cohort identities are checked; and
* integer protocol fields reject ``bool`` explicitly.

This remains a read-only source audit.  It does not normalize payloads, export
weights, run the decoder simulator, or admit cycles/traffic/speedup/Table-A.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M1321_SOURCE = HERE / "build_m1321_ep34_decoder_capture_adapter_source.py"
M1321_SOURCE_SHA256 = "52fb82ab1e4262d6ce838f28a443ce82c6deba00678f9c65fb8227ac30702d85"
M1313_CONTRACT = HW / (
    "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json")
M1313_CONTRACT_SHA256 = "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda"
OPERATOR_AUTHORITY = HW / (
    "results/h67_ep35_full_network_ordered_trace_s10_20260821/operator_runtime.csv")
OPERATOR_AUTHORITY_SHA256 = "915a67dda7b72d619573a73908ad8726b509741b01c16896aeec4460bf37467c"
DEPENDENCY_AUTHORITY = HW / (
    "results/h67_ep35_dependency_dag_s1_20260822/dependency_events.jsonl")
DEPENDENCY_AUTHORITY_SHA256 = "e1d2007195a036eedcee1e49d960955b3508ffe590ba3d075a3877a501a62f6b"
ATLIF_AUTHORITY = HW / (
    "results/h67_ep35_full_network_ordered_trace_s10_20260821/atlif_activity.csv")
ATLIF_AUTHORITY_SHA256 = "c40c568635b759e433b816f74c472a79c6080250540f65495e8bb57468e2e1ad"

EXPECTED_COUNTS = {
    "c1_conv3x3": 4, "decoder_convtranspose": 4, "atlif": 93,
    "fc1": 12, "fc2": 12, "patch_embed": 8, "batch_norm": 78,
    "qkv": 24, "attention": 12,
}
EXPECTED_NAME_SHA256 = {
    "c1_conv3x3": "cfe6ae229feafd35419fb9254b26dfe1e076af41a12fbb0ecc595b5baab4a1b7",
    "decoder_convtranspose": "ec47a3f9a063a46e19b956da7a3fc75b342580adb876574d2dd2d8bbb8ba67aa",
    "atlif": "f2dfcedab9ebe77b30b32d84bc38a2b1ea6511b0b3b359feb81a118ad2de252e",
    "fc1": "b344f10b3dcdfcd31e6f2d718473853374323e36f94850be5b3ad45055daa6c6",
    "fc2": "3bce98386df9b931cb2a975bf407e375408f664f0df98864688be7bbca094e9a",
    "patch_embed": "0757d609b1cc9d1a3ade8dea2c4b3c10add918b5b21859c28d3f6a4656f8757b",
    "batch_norm": "36689b3a9e08d50197ade07da7149964957cf42952da15ac73080e18d2679d5f",
    "qkv": "fa3c0c984643b6ada590d8074a692aa522231285a303de132bbaf575cebdba6d",
    "attention": "fe44105314114567a21cea4ab08bc9019e6e70b27c1b1121981d075fccf18bbf",
}
ROW_KEYS = {
    "global_order", "global_sample_id", "cohort", "sequence", "sample_key",
    "source_sha256", "category", "name", "input", "payload",
}
INPUT_KEYS = {
    "shape", "stride", "dtype", "elements", "bytes", "active", "positive",
    "negative", "nonfinite",
}
RETAINED_PAYLOAD_KEYS = {
    "retained", "raw_fp32_sha256", "compressed_fp32", "compressed_sha256",
    "support_sign", "support_sign_sha256", "positive_plane_bytes",
    "negative_plane_bytes",
}
NONRETAINED_PAYLOAD = {"retained": False, "reason": "ordered statistics only"}
C1_TARGETS = tuple(
    "sttmultires_unet.resblocks.{}.conv{}.0".format(block, conv)
    for block in range(2) for conv in range(1, 3)
)


class M1323Error(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1323Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected_sha: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1323Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected_sha, label + " SHA drift")


def _load_m1321():
    regular_exact(M1321_SOURCE, M1321_SOURCE_SHA256, "sealed M1321 source")
    spec = importlib.util.spec_from_file_location("m1323_sealed_m1321", M1321_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load sealed M1321")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1321 = _load_m1321()
DEFAULT_CAPTURE_ROOT = M1321.DEFAULT_CAPTURE_ROOT
MODULES = M1321.MODULES
SHAPES = M1321.SHAPES
WEIGHT_SHAPES = M1321.WEIGHT_SHAPES
EXPECTED_SAMPLES = M1321.EXPECTED_SAMPLES
EXPECTED_CALLS = M1321.EXPECTED_CALLS
EXPECTED_ORDERED_ROWS = M1321.EXPECTED_ORDERED_ROWS
ONE_WORD = M1321.ONE_WORD


def strict_json(path: Path) -> Any:
    try:
        return M1321.strict_json(path)
    except M1321.AdapterError as error:
        raise M1323Error(str(error)) from error


def strict_json_text(text: str) -> Any:
    try:
        return M1321.strict_json_text(text)
    except M1321.AdapterError as error:
        raise M1323Error(str(error)) from error


def lowercase_sha(value: Any, label: str) -> str:
    try:
        return M1321.lowercase_sha(value, label)
    except M1321.AdapterError as error:
        raise M1323Error(str(error)) from error


def inventory_digest(names: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def frozen_inventory_names() -> dict[str, tuple[str, ...]]:
    """Reconstruct and cross-hash the exact frozen 247 live module names."""
    regular_exact(OPERATOR_AUTHORITY, OPERATOR_AUTHORITY_SHA256, "operator authority")
    regular_exact(DEPENDENCY_AUTHORITY, DEPENDENCY_AUTHORITY_SHA256,
                  "dependency authority")
    regular_exact(ATLIF_AUTHORITY, ATLIF_AUTHORITY_SHA256, "ATLIF authority")
    with OPERATOR_AUTHORITY.open(newline="", encoding="utf-8") as stream:
        operators = list(csv.DictReader(stream))
    fc1 = sorted(set(row["name"] for row in operators if row["name"].endswith(".mlp.fc1")))
    fc2 = sorted(set(row["name"] for row in operators if row["name"].endswith(".mlp.fc2")))
    qkv = sorted(set(
        row["name"] for row in operators
        if row["name"].endswith(".attn.linear_q") or
        row["name"].endswith(".attn.linear_k")
    ))
    patch = sorted(set(row["name"] for row in operators if ".patch_embed." in row["name"]))
    batch_norm: set[str] = set()
    for line in DEPENDENCY_AUTHORITY.read_text(encoding="utf-8").splitlines():
        row = strict_json_text(line)
        if row.get("module_type") in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d"):
            batch_norm.add(row["name"])
    with ATLIF_AUTHORITY.open(newline="", encoding="utf-8") as stream:
        atlif = sorted(set(row["name"] for row in csv.DictReader(stream)))
    inventory = {
        "c1_conv3x3": tuple(sorted(C1_TARGETS)),
        "decoder_convtranspose": tuple(sorted(MODULES)),
        "atlif": tuple(atlif),
        "fc1": tuple(fc1), "fc2": tuple(fc2), "patch_embed": tuple(patch),
        "batch_norm": tuple(sorted(batch_norm)), "qkv": tuple(qkv),
        "attention": tuple(sorted(set(name.rsplit(".", 1)[0] for name in qkv))),
    }
    require(set(inventory) == set(EXPECTED_COUNTS), "frozen category population drift")
    for category, names in inventory.items():
        require(len(names) == EXPECTED_COUNTS[category],
                "frozen inventory count drift: " + category)
        require(inventory_digest(names) == EXPECTED_NAME_SHA256[category],
                "frozen inventory identity drift: " + category)
    require(sum(map(len, inventory.values())) == 247, "frozen live inventory is not 247")
    return inventory


def expected_cohort() -> dict[int, dict[str, Any]]:
    """Bind sample identity to the exact M1313 launch consumed by M1320."""
    regular_exact(M1313_CONTRACT, M1313_CONTRACT_SHA256, "M1313 launch contract")
    contract = strict_json(M1313_CONTRACT)
    rows = contract.get("cohort", {}).get("samples")
    require(type(rows) is list and len(rows) == 40, "M1313 cohort is not forty samples")
    output: dict[int, dict[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        require(type(row) is dict and type(row.get("global_sample_id")) is int and
                row["global_sample_id"] == ordinal, "M1313 sample ordinal drift")
        identity = {
            "cohort": row.get("cohort"), "sequence": row.get("sequence"),
            "sample_key": row.get("sample_key"), "source_sha256": row.get("sha256"),
        }
        require(all(type(identity[key]) is str and identity[key]
                    for key in ("cohort", "sequence", "sample_key")),
                "M1313 sample string identity invalid")
        lowercase_sha(identity["source_sha256"], "M1313 source SHA")
        output[ordinal] = identity
    return output


def _exact_int(value: Any, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, label + " is not exact integer")
    return value


def _validate_input(value: Any) -> None:
    require(type(value) is dict and set(value) == INPUT_KEYS, "input-statistic keys drift")
    shape = value["shape"]
    stride = value["stride"]
    require(type(shape) is list and shape and
            all(type(item) is int and item > 0 for item in shape), "input shape invalid")
    require(type(stride) is list and len(stride) == len(shape) and
            all(type(item) is int and item >= 0 for item in stride), "input stride invalid")
    elements = _exact_int(value["elements"], "input elements", 1)
    require(elements == math.prod(shape), "input element count differs from shape")
    byte_count = _exact_int(value["bytes"], "input bytes", 1)
    require(byte_count % elements == 0, "input bytes are not element aligned")
    require(type(value["dtype"]) is str and value["dtype"].startswith("torch."),
            "input dtype invalid")
    active = _exact_int(value["active"], "input active")
    positive = _exact_int(value["positive"], "input positive")
    negative = _exact_int(value["negative"], "input negative")
    nonfinite = _exact_int(value["nonfinite"], "input nonfinite")
    require(max(active, positive, negative, nonfinite) <= elements and
            positive + negative <= active + nonfinite,
            "input statistic range inconsistent")


def _validate_payload(value: Any, retained: bool, sample: int,
                      global_order: int, module_name: str) -> None:
    if not retained:
        require(value == NONRETAINED_PAYLOAD, "unretained payload identity drift")
        return
    require(type(value) is dict and set(value) == RETAINED_PAYLOAD_KEYS and
            value.get("retained") is True, "retained payload keys drift")
    for key in ("raw_fp32_sha256", "compressed_sha256", "support_sign_sha256"):
        lowercase_sha(value[key], key)
    for key in ("compressed_fp32", "support_sign"):
        require(type(value[key]) is str and value[key].startswith("payloads/") and
                ".." not in Path(value[key]).parts, key + " path invalid")
    stem = "s{:02d}_o{:05d}_{}".format(
        sample, global_order, hashlib.sha256(module_name.encode()).hexdigest()[:12])
    require(value["compressed_fp32"] == "payloads/{}.fp32.zlib".format(stem) and
            value["support_sign"] ==
            "payloads/{}.support_sign.le.bitpack".format(stem),
            "retained payload path is not exact call identity")
    _exact_int(value["positive_plane_bytes"], "positive plane bytes", 1)
    _exact_int(value["negative_plane_bytes"], "negative plane bytes", 1)
    require(value["positive_plane_bytes"] == value["negative_plane_bytes"],
            "support plane extents differ")


def decoder_rows_from_ordered(
        records: Iterable[Mapping[str, Any]],
        frozen_inventory: Mapping[str, Sequence[str]] | None = None,
        cohort: Mapping[int, Mapping[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate all 9,880 rows, then project the 120 decoder calls."""
    inventory = dict(frozen_inventory_names() if frozen_inventory is None else frozen_inventory)
    samples = dict(expected_cohort() if cohort is None else cohort)
    require(set(inventory) == set(EXPECTED_COUNTS) and set(samples) == set(range(40)),
            "ordered audit authority population drift")
    frozen_pairs = set()
    for category, names in inventory.items():
        require(type(category) is str and len(names) == EXPECTED_COUNTS[category] and
                len(set(names)) == len(names), "ordered inventory malformed: " + category)
        require(inventory_digest(names) == EXPECTED_NAME_SHA256[category],
                "ordered inventory hash mismatch: " + category)
        frozen_pairs.update((category, name) for name in names)
    require(len(frozen_pairs) == 247, "ordered frozen identity population is not 247")

    rows = []
    retained_compressed_paths: set[str] = set()
    retained_support_paths: set[str] = set()
    retained_path_pairs: set[tuple[str, str]] = set()
    per_sample: dict[int, list[Mapping[str, Any]]] = {sample: [] for sample in range(40)}
    sequence_hashes = []
    for file_ordinal, row in enumerate(records):
        require(file_ordinal < EXPECTED_ORDERED_ROWS, "ordered population exceeds 9880")
        require(type(row) is dict and set(row) == ROW_KEYS, "ordered row keys drift")
        require(type(row.get("global_order")) is int and
                row["global_order"] == file_ordinal,
                "global_order is not exact file ordinal")
        sample = file_ordinal // 247
        require(type(row.get("global_sample_id")) is int and
                row["global_sample_id"] == sample,
                "ordered sample id is not exact contiguous 40x247")
        category = row.get("category")
        name = row.get("name")
        require(type(category) is str and type(name) is str and
                (category, name) in frozen_pairs,
                "ordered module/category identity is not frozen")
        identity = samples[sample]
        require(all(row.get(key) == identity[key]
                    for key in ("cohort", "sequence", "sample_key", "source_sha256")),
                "ordered sample identity differs from M1313 cohort")
        _validate_input(row.get("input"))
        retained = category in {"c1_conv3x3", "decoder_convtranspose"}
        _validate_payload(row.get("payload"), retained, sample, file_ordinal, name)
        if retained:
            compressed_path = row["payload"]["compressed_fp32"]
            support_path = row["payload"]["support_sign"]
            require(compressed_path not in retained_compressed_paths and
                    support_path not in retained_support_paths and
                    (compressed_path, support_path) not in retained_path_pairs,
                    "retained payload path alias across calls")
            retained_compressed_paths.add(compressed_path)
            retained_support_paths.add(support_path)
            retained_path_pairs.add((compressed_path, support_path))
        rows.append(row)
        per_sample[sample].append(row)
    require(len(rows) == EXPECTED_ORDERED_ROWS, "ordered population is not 9880")
    require(len(retained_compressed_paths) == 320 and
            len(retained_support_paths) == 320 and len(retained_path_pairs) == 320,
            "retained payload path population is not exact 40x8")

    selected = []
    reference_sequence: list[tuple[str, str]] | None = None
    for sample in range(40):
        sample_rows = per_sample[sample]
        require(len(sample_rows) == 247, "sample ordered population is not 247")
        observed = [(row["category"], row["name"]) for row in sample_rows]
        require(len(set(observed)) == 247 and set(observed) == frozen_pairs,
                "sample module identities are duplicated/missing/replaced")
        if reference_sequence is None:
            reference_sequence = observed
        require(observed == reference_sequence, "per-sample module execution order drift")
        sequence_hashes.append(hashlib.sha256(json.dumps(
            observed, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest())
        decoder = [row for row in sample_rows if row["category"] == "decoder_convtranspose"]
        require([row["name"] for row in decoder] == list(MODULES),
                "decoder module order drift")
        if sample not in EXPECTED_SAMPLES:
            continue
        for module_ordinal, row in enumerate(decoder):
            require(row["input"]["shape"] == list(SHAPES[module_ordinal]),
                    "decoder input shape drift")
            payload = row["payload"]
            selected.append({
                "global_call_ordinal": len(selected), "global_order": row["global_order"],
                "global_sample_id": sample, "sequence": row["sequence"],
                "sample_key": row["sample_key"], "source_sha256": row["source_sha256"],
                "module_ordinal": module_ordinal, "module": MODULES[module_ordinal],
                "shape": list(SHAPES[module_ordinal]),
                "compressed_fp32": payload["compressed_fp32"],
                "compressed_sha256": payload["compressed_sha256"],
                "support_sign": payload["support_sign"],
                "support_sign_sha256": payload["support_sign_sha256"],
                "raw_fp32_sha256": payload["raw_fp32_sha256"],
                "positive_plane_bytes": payload["positive_plane_bytes"],
                "negative_plane_bytes": payload["negative_plane_bytes"],
            })
    require(len(selected) == EXPECTED_CALLS, "decoder call population is not 120")
    cohort_rows = [{"global_sample_id": sample, **dict(samples[sample])} for sample in range(40)]
    audit_identity = {
        "ordered_rows": len(rows), "samples": 40, "live_modules_per_sample": 247,
        "unique_retained_payload_pairs": len(retained_path_pairs),
        "module_sequence_sha256": sequence_hashes[0],
        "all_sample_sequences_equal": len(set(sequence_hashes)) == 1,
        "frozen_inventory_sha256": hashlib.sha256(json.dumps(
            {key: list(inventory[key]) for key in sorted(inventory)},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "m1313_cohort_sha256": hashlib.sha256(json.dumps(
            cohort_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return selected, audit_identity


def audit_two_plane_payload(*args, **kwargs):
    if len(args) >= 4:
        ordinal = args[3]
    else:
        ordinal = kwargs.get("module_ordinal")
    require(type(ordinal) is int and 0 <= ordinal < 4,
            "module ordinal is not exact integer 0..3")
    try:
        return M1321.audit_two_plane_payload(*args, **kwargs)
    except M1321.AdapterError as error:
        raise M1323Error(str(error)) from error


def validate_weight_identities(rows: Any, checkpoint_sha256: str) -> list[dict[str, Any]]:
    require(type(rows) is list and len(rows) == 4, "weight identity population must be four")
    for ordinal, row in enumerate(rows):
        require(type(row) is dict and type(row.get("module_ordinal")) is int and
                row["module_ordinal"] == ordinal,
                "weight module ordinal is not exact integer")
    try:
        return M1321.validate_weight_identities(rows, checkpoint_sha256)
    except M1321.AdapterError as error:
        raise M1323Error(str(error)) from error


def audit_capture(capture_root: Path = DEFAULT_CAPTURE_ROOT,
                  weight_identities: Any | None = None,
                  checkpoint_sha256: str | None = None) -> dict[str, Any]:
    root = Path(capture_root)
    require(root.is_dir() and not root.is_symlink(), "capture root missing/symlink")
    ordered_path = root / "unified_ordered_records.jsonl"
    try:
        mode = ordered_path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1323Error("missing ordered records") from error
    require(stat.S_ISREG(mode) and not ordered_path.is_symlink(),
            "ordered records must be regular non-symlink")
    ordered = [strict_json_text(line) for line in ordered_path.read_text(
        encoding="utf-8").splitlines()]
    calls, ordered_identity = decoder_rows_from_ordered(ordered)
    theta_words = set()
    audited = []
    for call in calls:
        try:
            compressed = M1321.safe_member(root, call["compressed_fp32"], "compressed FP32")
            support = M1321.safe_member(root, call["support_sign"], "support/sign")
        except M1321.AdapterError as error:
            raise M1323Error(str(error)) from error
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
        "schema": "m1323_ep34_decoder_capture_adapter_source_audit_r1",
        "status": "PASS_SOURCE_AUDIT__ACTUAL_RESULT_HAMMER_AND_SUCCESSOR_REQUIRED",
        "capture_root": str(root), "ordered_identity": ordered_identity,
        "ordered_jsonl_sha256": sha256(ordered_path),
        "population": {"samples": 30, "calls": 120, "modules": 4,
                       "global_sample_ids": [10, 39]},
        "d1": {"calls": 30, "theta_word_uint32": next(iter(theta_words)),
               "theta_ieee754_le_hex": struct.pack("<I", next(iter(theta_words))).hex(),
               "negative_count": 0, "coerced_to_one": False,
               "weight_folding": False},
        "calls": audited, "weight_identities": weights,
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
    print(json.dumps({key: value for key, value in result.items() if key != "calls"},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1323Error as error:
        print("M1323_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
