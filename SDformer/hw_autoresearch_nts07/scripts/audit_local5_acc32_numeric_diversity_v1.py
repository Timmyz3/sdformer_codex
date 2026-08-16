#!/usr/bin/env python3
"""Independently audit Local5 K/weight/Acc32 numeric diversity and equality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


TOKENS = 450
LANES = 32
OUT_DIM = 32
TRUSTED_EXPECTED_SOURCES = {
    "local5_erep_numeric_window_expected.py":
        "3fcee148205321bbf0365cd8542080251d7554ed802b6d1bbb25c49d1788d63c",
    "local5_erep_formal_canary_expected.py":
        "c2de13cff523ece31b2ada35b2557d89be3f79344c05e4071104e8b5357de861",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def read_acc32(path: Path, count: int) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    if len(rows) != count or any(
        len(row) != 8 or any(character not in "0123456789abcdef" for character in row)
        for row in rows
    ):
        raise ValueError(f"{path.name} is not exact Acc32 memh")
    return np.asarray([int(row, 16) for row in rows], dtype=np.uint32).view(np.int32)


def distribution(values: np.ndarray) -> dict[str, int]:
    values64 = values.astype(np.int64, copy=False)
    if values64.size == 0:
        raise ValueError("cannot summarize an empty numeric array")
    return {
        "count": int(values.size),
        "zero": int(np.count_nonzero(values64 == 0)),
        "nonzero": int(np.count_nonzero(values64 != 0)),
        "positive": int(np.count_nonzero(values64 > 0)),
        "negative": int(np.count_nonzero(values64 < 0)),
        "minimum": int(values64.min()),
        "maximum": int(values64.max()),
        "sum": int(values64.sum()),
        "sum_abs": int(np.abs(values64).sum()),
    }


def read_inputs(path: Path, heads: int) -> dict[str, Any]:
    rows = path.read_text(encoding="ascii").splitlines()
    if len(rows) != heads * TOKENS:
        raise ValueError("combined input row count differs")
    q_words: list[int] = []
    k_words: list[int] = []
    masks: list[int] = []
    k_nonzero_by_head = [0] * heads
    seen: set[tuple[int, int, int, int]] = set()
    q_nonzero_by_head = [0] * heads
    q_and_k_bits_by_head = [0] * heads
    q_xor_k_bits_by_head = [0] * heads
    neither_q_nor_k_bits_by_head = [0] * heads
    valid_candidate_bits_by_head = [0] * heads
    for row in rows:
        fields = row.split()
        if len(fields) != 11:
            raise ValueError("combined input row width differs")
        head, plane, y, x = map(int, fields[:4])
        source = (plane * 15 + y) * 15 + x
        if not (
            0 <= head < heads and 0 <= plane < 2
            and 0 <= y < 15 and 0 <= x < 15 and 0 <= source < TOKENS
        ):
            raise ValueError("combined input identity is outside contract")
        coordinate = (head, plane, y, x)
        if coordinate in seen:
            raise ValueError("combined input identity is duplicated")
        seen.add(coordinate)
        if (
            len(fields[4]) != 8
            or any(len(value) != 8 for value in fields[5:10])
            or len(fields[10]) != 2
            or any(
                character not in "0123456789abcdef"
                for value in fields[4:11]
                for character in value
            )
        ):
            raise ValueError("combined input hex width/alphabet differs")
        q_value = int(fields[4], 16)
        q_words.append(q_value)
        q_nonzero_by_head[head] += int(q_value != 0)
        parsed_k = [int(value, 16) for value in fields[5:10]]
        k_words.extend(parsed_k)
        k_nonzero_by_head[head] += sum(value != 0 for value in parsed_k)
        mask = int(fields[10], 16)
        if mask & ~0x1F:
            raise ValueError("combined input valid mask exceeds five candidates")
        masks.append(mask)
        for candidate, k_value in enumerate(parsed_k):
            if not ((mask >> candidate) & 1):
                continue
            q_and_k_bits_by_head[head] += (q_value & k_value).bit_count()
            q_xor_k_bits_by_head[head] += (q_value ^ k_value).bit_count()
            neither_q_nor_k_bits_by_head[head] += (
                (~(q_value | k_value)) & 0xFFFFFFFF
            ).bit_count()
            valid_candidate_bits_by_head[head] += 32
    expected_coordinates = {
        (head, plane, y, x)
        for head in range(heads) for plane in range(2)
        for y in range(15) for x in range(15)
    }
    if seen != expected_coordinates:
        raise ValueError("combined input coordinate set differs")
    return {
        "rows": len(rows),
        "q_words": len(q_words),
        "q_nonzero": sum(value != 0 for value in q_words),
        "q_nonzero_by_input_head": q_nonzero_by_head,
        "k_words": len(k_words),
        "k_nonzero": sum(value != 0 for value in k_words),
        "k_nonzero_by_input_head": k_nonzero_by_head,
        "valid_mask_nonzero": sum(value != 0 for value in masks),
        "valid_candidate_bits": sum(valid_candidate_bits_by_head),
        "q_and_k_bits": sum(q_and_k_bits_by_head),
        "q_and_k_bits_by_input_head": q_and_k_bits_by_head,
        "q_xor_k_bits": sum(q_xor_k_bits_by_head),
        "q_xor_k_bits_by_input_head": q_xor_k_bits_by_head,
        "neither_q_nor_k_bits": sum(neither_q_nor_k_bits_by_head),
        "neither_q_nor_k_bits_by_input_head": neither_q_nor_k_bits_by_head,
    }


def read_weights(path: Path, heads: int) -> dict[str, int]:
    rows = path.read_text(encoding="ascii").splitlines()
    expected = heads * heads * LANES * OUT_DIM
    if len(rows) != expected:
        raise ValueError("projection weight row count differs")
    values: list[int] = []
    seen: set[tuple[int, int, int, int]] = set()
    for row in rows:
        fields = row.split()
        if len(fields) != 5:
            raise ValueError("projection weight row width differs")
        identity = tuple(map(int, fields[:4]))
        if identity in seen:
            raise ValueError("projection weight identity is duplicated")
        seen.add(identity)
        if len(fields[4]) != 2 or any(
            character not in "0123456789abcdef" for character in fields[4]
        ):
            raise ValueError("projection weight is not exact lowercase 8-bit hex")
        values.append(signed(int(fields[4], 16), 8))
    expected_coordinates = {
        (head, tile, lane, out)
        for tile in range(heads)
        for head in range(heads)
        for lane in range(LANES)
        for out in range(OUT_DIM)
    }
    if seen != expected_coordinates:
        raise ValueError("projection weight coordinate set differs")
    return distribution(np.asarray(values, dtype=np.int16))


def verify_expected_receipt(
    receipt_path: Path,
    task_plan_path: Path,
    expected_path: Path,
    identity: dict[str, int],
    release_manifest_path: Path,
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    if (
        receipt.get("schema") != "local5_erep_numeric_window_expected_v1"
        or receipt.get("status") != "PASS_NUMERIC_WINDOW_EXPECTED_NOT_G0"
        or receipt.get("formal_g0") != "DENY"
        or receipt.get("identity") != identity
        or receipt.get("task_plan_sha256") != sha256(task_plan_path)
        or receipt.get("software_expected_sha256") != sha256(expected_path)
        or receipt.get("expected_shape")
        != [identity["heads"], TOKENS, OUT_DIM]
        or receipt.get("expected_scalar_count")
        != identity["heads"] * TOKENS * OUT_DIM
    ):
        raise ValueError("software expected receipt semantics differ")
    source_rows = receipt.get("source_bindings")
    release_manifest = read_json(release_manifest_path)
    release_sources = {
        row["path"]: row["sha256"]
        for row in release_manifest.get("source_bindings", [])
        if isinstance(row, dict) and set(row) >= {"path", "sha256"}
    }
    if not isinstance(source_rows, list) or len(source_rows) != 2:
        raise ValueError("software expected source bindings differ")
    verified_sources = []
    for row in source_rows:
        if not isinstance(row, dict) or set(row) != {"file", "sha256"}:
            raise ValueError("software expected source binding schema differs")
        source_name = Path(row["file"]).name
        path = Path(row["file"]).resolve(strict=True)
        trusted_path = (Path(__file__).resolve().parents[1] / "scripts" / source_name)
        trusted_sha = TRUSTED_EXPECTED_SOURCES.get(source_name)
        if (
            trusted_sha is None
            or path != trusted_path.resolve(strict=True)
            or row["sha256"] != trusted_sha
            or sha256(path) != trusted_sha
            or release_sources.get(f"scripts/{source_name}") != trusted_sha
        ):
            raise ValueError("software expected source binding SHA differs")
        verified_sources.append({"file": str(path), "sha256": row["sha256"]})
    if {Path(row["file"]).name for row in verified_sources} != set(
        TRUSTED_EXPECTED_SOURCES
    ):
        raise ValueError("software expected source binding whitelist differs")
    return {
        "receipt_sha256": sha256(receipt_path),
        "task_plan_sha256": sha256(task_plan_path),
        "software_expected_sha256": sha256(expected_path),
        "source_bindings": verified_sources,
        "release_manifest_sha256": sha256(release_manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-q-and-k", action="store_true")
    parser.add_argument("--require-all-heads-active", action="store_true")
    args = parser.parse_args()
    canary = args.canary_dir.resolve()
    vectors = args.vector_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")

    complete_path = canary / "complete.json"
    complete = read_json(complete_path)
    identity = complete.get("identity")
    if (
        complete.get("status")
        != "PASS_SEALED_PARAMETERIZED_IDENTITY_PHASE_CANARY_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or not isinstance(identity, dict)
        or set(identity) != {"sample", "stage", "block", "window", "heads"}
    ):
        raise ValueError("canary complete identity/status differs")
    heads = int(identity["heads"])
    expected_count = heads * TOKENS * OUT_DIM
    baseline = read_acc32(canary / "baseline_actual.memh", expected_count)
    candidate = read_acc32(canary / "candidate_actual.memh", expected_count)
    expected_path = vectors / "software_expected/software_expected.npz"
    with np.load(expected_path, allow_pickle=False) as handle:
        if set(handle.files) != {"schema_version", "expected_acc32"}:
            raise ValueError("software expected NPZ schema differs")
        schema_version = np.array(handle["schema_version"], copy=True)
        expected = np.array(handle["expected_acc32"], copy=True)
    if schema_version.dtype != np.uint16 or schema_version.tolist() != [1]:
        raise ValueError("software expected schema version differs")
    if expected.dtype != np.int32 or expected.shape != (expected_count,):
        raise ValueError("software expected Acc32 shape/dtype differs")
    if not np.array_equal(baseline, candidate) or not np.array_equal(candidate, expected):
        raise ValueError("baseline/candidate/software Acc32 differ")
    acc32 = distribution(expected)
    acc32_by_output_tile = [
        distribution(expected.reshape(heads, TOKENS, OUT_DIM)[tile])
        for tile in range(heads)
    ]
    input_stats = read_inputs(vectors / "vectors/combined_head_inputs.txt", heads)
    weight_stats = read_weights(vectors / "vectors/projection_weights.txt", heads)
    if (
        input_stats["k_nonzero"] == 0
        or weight_stats["positive"] == 0
        or weight_stats["negative"] == 0
        or acc32["nonzero"] == 0
        or acc32["positive"] == 0
        or acc32["negative"] == 0
    ):
        raise ValueError("numeric diversity admission failed")
    if args.require_q_and_k and input_stats["q_and_k_bits"] == 0:
        raise ValueError("Q-and-K branch coverage admission failed")
    if args.require_all_heads_active and (
        any(value == 0 for value in input_stats["k_nonzero_by_input_head"])
        or any(value == 0 for value in input_stats["q_and_k_bits_by_input_head"])
    ):
        raise ValueError("all-head K/Q-and-K coverage admission failed")

    output.mkdir(parents=True)
    source_dir = output / "source"
    source_dir.mkdir()
    source_copy = source_dir / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), source_copy)
    task_plan = vectors / "software_expected/task_plan.json"
    expected_receipt = vectors / "software_expected/software_expected_receipt.json"
    vector_manifest = vectors / "vectors/manifest.json"
    task_plan_value = read_json(task_plan)
    vector_manifest_value = read_json(vector_manifest)
    expected_identity = {
        name: task_plan_value.get(name)
        for name in ("sample", "stage", "block", "window", "heads")
    }
    if (
        expected_identity != identity
        or vector_manifest_value.get("identity")
        != {**identity, "tokens": TOKENS, "out_dim": OUT_DIM}
        or vector_manifest_value.get("task_plan_sha256") != sha256(task_plan)
        or vector_manifest_value.get("files", {}).get(
            "combined_head_inputs", {}
        ).get("sha256") != sha256(vectors / "vectors/combined_head_inputs.txt")
        or vector_manifest_value.get("files", {}).get(
            "projection_weights", {}
        ).get("sha256") != sha256(vectors / "vectors/projection_weights.txt")
    ):
        raise ValueError("task plan/vector manifest identity or binding differs")
    release_binding = complete.get("external_bindings", {}).get(
        "baseline_release_manifest", {}
    )
    release_manifest_path = Path(release_binding.get("path", ""))
    if (
        not release_manifest_path.is_file()
        or sha256(release_manifest_path) != release_binding.get("sha256")
    ):
        raise ValueError("baseline release manifest binding differs")
    receipt_verification = verify_expected_receipt(
        expected_receipt, task_plan, expected_path, identity,
        release_manifest_path,
    )
    bindings = {
        "canary_complete": complete_path,
        "baseline_acc32": canary / "baseline_actual.memh",
        "candidate_acc32": canary / "candidate_actual.memh",
        "software_expected": expected_path,
        "software_expected_receipt": expected_receipt,
        "task_plan": task_plan,
        "vector_manifest": vector_manifest,
        "combined_inputs": vectors / "vectors/combined_head_inputs.txt",
        "projection_weights": vectors / "vectors/projection_weights.txt",
        "archived_auditor": source_copy,
    }
    report = {
        "schema": "local5_acc32_numeric_diversity_audit_v4",
        "status": "PASS_NONDEGENERATE_ACC32_MITER_NOT_G0",
        "evidence": "[rtl]+[独立软件数值审计]",
        "formal_g0": "DENY",
        "identity": identity,
        "admission": {
            "require_q_and_k": args.require_q_and_k,
            "require_all_heads_active": args.require_all_heads_active,
        },
        "input": input_stats,
        "weight_int8": weight_stats,
        "acc32": {
            **acc32, "mismatch": 0,
            "by_output_tile": acc32_by_output_tile,
        },
        "expected_receipt_verification": receipt_verification,
        "bindings": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in bindings.items()
        },
        "boundary": [
            f"single real H{heads} window; not full formal corpus",
            "proves nonzero signed Acc32 equality, not overflow exhaustiveness",
            "formal G0, full encoder, and ASIC PPA remain unavailable",
        ],
    }
    report_path = output / "numeric_diversity_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete_out = {
        "schema": "local5_acc32_numeric_diversity_complete_v2",
        "status": report["status"],
        "formal_g0": "DENY",
        "identity": identity,
        "report_sha256": sha256(report_path),
        "archived_auditor_sha256": sha256(source_copy),
    }
    temporary = output / f"complete.json.tmp.{os.getpid()}"
    temporary.write_text(
        json.dumps(complete_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "complete.json")
    print(json.dumps({
        "status": report["status"], "identity": identity,
        "input": input_stats, "weight_int8": weight_stats, "acc32": acc32,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
