#!/usr/bin/env python3
"""不导入项目 oracle，独立重算并封存 Local5 identity-service table 包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "local5_erep_identity_service_v4"
DEFAULT_SEED = 20260810
TOKENS = 450
LANES = 32
OUT_DIM = 32
ARTIFACT_FILES = {
    "identity_service_tables": "identity_service_tables.npz",
    "relation_delay": "relation_delay.memh",
    "weight_delay": "weight_delay.memh",
    "final_delay": "final_delay.memh",
}
SOURCE_FILES = {
    "generator": "source/generate_local5_identity_service_tables_v4.py",
    "identity_oracle": "source/local5_erep_identity_service_v4.py",
    "independent_verifier": "source/verify_local5_identity_service_tables_v4.py",
}
RUNTIME_ORDER = {
    "relation": "output_tile,input_head,source_id; identity excludes output_tile",
    "weight": "output_tile,input_head,lane,out",
    "final": "output_tile,source_id,out",
}
FLAT_INDEX_CONTRACT = {
    "relation": "input_head*450+source_id",
    "relation_runtime_reuse": "output_tile does not enter the flat index",
    "weight": "(((output_tile*heads+input_head)*32+lane)*32)+out",
    "final": "((output_tile*450+source_id)*32)+out",
}
HANDSHAKE_CONTRACT = {
    "request_accept": "posedge(request_valid&&request_ready)",
    "response_available": "first posedge response_valid after request_accept",
    "response_latency": "response_available-request_accept=delay+1",
    "response_accept": "posedge(response_valid&&response_ready), not earlier than response_available",
    "response_stability": "response_valid and payload remain stable from available through accept",
    "outstanding": "at most one relation request and one weight request",
    "final_request": "first posedge tile_result_valid for each ordered final item",
    "final_accept": "exactly delay+1 cycles after final_request while valid remains asserted",
}
BOUNDARY = [
    "delay is identity-derived 0..3; registered response latency is delay+1",
    "tables are verification inputs, not DUT state or an architecture result",
    "numeric DUT and formal G0 are unchanged",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def length_prefix(parts: Iterable[bytes]) -> bytes:
    framed = bytearray()
    for part in parts:
        if not isinstance(part, bytes):
            raise TypeError("hash frame component is not bytes")
        framed.extend(len(part).to_bytes(8, "big"))
        framed.extend(part)
    return bytes(framed)


def strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def load_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    required = {
        "schema", "sample", "stage", "block", "window", "heads", "out_dim",
        "tasks", "source_manifest_sha256", "source_payload_sha256",
        "projection_contract_sha256", "projection_payload_sha256", "task_sha256",
    }
    if not required.issubset(plan) or plan.get("schema") != "local5_projection_task_plan_v1":
        raise ValueError("task plan schema is invalid")
    heads = strict_int(plan["heads"], "heads")
    if heads < 1 or heads > 32 or strict_int(plan["out_dim"], "out_dim") != OUT_DIM:
        raise ValueError("task plan heads/out_dim is invalid")
    for name in ("sample", "stage", "block", "window"):
        strict_int(plan[name], name)
    for name in (
        "source_manifest_sha256", "source_payload_sha256",
        "projection_contract_sha256", "projection_payload_sha256", "task_sha256",
    ):
        if not is_sha256(plan[name]):
            raise ValueError(f"task plan {name} is not SHA-256")
    tasks = plan["tasks"]
    if not isinstance(tasks, list) or len(tasks) != heads * heads:
        raise ValueError("task plan task count is invalid")
    groups: list[int] = []
    for head in range(heads):
        row = tasks[head]
        if not isinstance(row, dict) or set(row) != {"input_group_index", "output_tile"}:
            raise ValueError("task row schema is invalid")
        groups.append(strict_int(row["input_group_index"], "input_group_index"))
    expected = [
        {"input_group_index": groups[head], "output_tile": tile}
        for tile in range(heads)
        for head in range(heads)
    ]
    if tasks != expected or len(set(groups)) != heads or canonical_sha(tasks) != plan["task_sha256"]:
        raise ValueError("task plan tile-major order is invalid")
    return plan


def sample_identity(plan: dict[str, Any]) -> str:
    return (
        f"local5/profile100/{plan['source_manifest_sha256']}"
        f"/sample/{strict_int(plan['sample'], 'sample'):03d}"
    )


def make_identity(plan: dict[str, Any], **coordinates: int) -> dict[str, Any]:
    return {
        "sample": sample_identity(plan),
        "stage": plan["stage"],
        "block": plan["block"],
        "window": plan["window"],
        **coordinates,
    }


def transaction(kind: str, identity: dict[str, Any]) -> dict[str, Any]:
    identity_json = canonical_bytes(identity)
    material = length_prefix(
        (
            b"schema", SCHEMA.encode("utf-8"), b"seed",
            canonical_bytes(DEFAULT_SEED), b"kind", kind.encode("utf-8"),
            b"identity", identity_json,
        )
    )
    digest = hashlib.sha256(material).hexdigest()
    delay = int.from_bytes(bytes.fromhex(digest)[:8], "big") % 4
    record = {
        "schema": SCHEMA,
        "seed": DEFAULT_SEED,
        "kind": kind,
        "identity": identity,
        "delay": delay,
    }
    return {
        "kind": kind,
        "identity": identity,
        "identity_key": canonical_bytes({"kind": kind, "identity": identity}),
        "digest": digest,
        "delay": delay,
        "record": canonical_bytes(record),
    }


def expected_transactions(plan: dict[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    heads = strict_int(plan["heads"], "heads")
    relation_unique = tuple(
        transaction("relation", make_identity(plan, input_head=head, source_id=source))
        for head in range(heads) for source in range(TOKENS)
    )
    relation_runtime = tuple(
        transaction("relation", make_identity(plan, input_head=head, source_id=source))
        for _tile in range(heads) for head in range(heads) for source in range(TOKENS)
    )
    weights = tuple(
        transaction(
            "weight", make_identity(
                plan, input_head=head, output_tile=tile, lane=lane, out=out
            )
        )
        for tile in range(heads) for head in range(heads)
        for lane in range(LANES) for out in range(OUT_DIM)
    )
    finals = tuple(
        transaction(
            "final", make_identity(plan, output_tile=tile, source_id=source, out=out)
        )
        for tile in range(heads) for source in range(TOKENS) for out in range(OUT_DIM)
    )
    return {
        "relation_unique": relation_unique,
        "relation_runtime": relation_runtime,
        "weight_runtime": weights,
        "final_runtime": finals,
    }


def ledger_hash(domain: bytes, records: Iterable[bytes]) -> str:
    rows = tuple(records)
    material = length_prefix(
        (
            b"local5-erep-ledger-v4", domain, SCHEMA.encode("utf-8"),
            canonical_bytes(DEFAULT_SEED), canonical_bytes(len(rows)), *rows,
        )
    )
    return hashlib.sha256(material).hexdigest()


def ledger_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    groups: dict[bytes, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["identity_key"]].append(row)
    multiplicities = []
    for key in sorted(groups):
        group = groups[key]
        first = group[0]
        multiplicities.append(
            {
                "kind": first["kind"],
                "identity": first["identity"],
                "transaction_digest": first["digest"],
                "delay": first["delay"],
                "multiplicity": len(group),
            }
        )
    histogram = Counter(row["multiplicity"] for row in multiplicities)
    records = tuple(row["record"] for row in rows)
    return {
        "transaction_count": len(rows),
        "identity_count": len(multiplicities),
        "ordered_ledger_digest": ledger_hash(b"ordered", records),
        "unordered_multiset_digest": ledger_hash(b"multiset", sorted(records)),
        "identity_multiplicity_sha256": canonical_sha(multiplicities),
        "multiplicity_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
    }


def read_memh(path: Path, count: int) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    if len(rows) != count or any(len(row) != 1 or row not in "0123" for row in rows):
        raise ValueError(f"{path.name} content/count is invalid")
    return np.asarray([int(row, 16) for row in rows], dtype=np.uint8)


def expected_artifact_rows(heads: int, root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, file_name in ARTIFACT_FILES.items():
        path = root / file_name
        row: dict[str, Any] = {"file": file_name, "sha256": sha256(path)}
        if name != "identity_service_tables":
            row.update(
                entries={
                    "relation_delay": heads * TOKENS,
                    "weight_delay": heads * heads * LANES * OUT_DIM,
                    "final_delay": heads * TOKENS * OUT_DIM,
                }[name],
                width=2,
            )
        result[name] = row
    return result


def expected_npz_members(heads: int) -> dict[str, dict[str, Any]]:
    relation = heads * TOKENS
    weight = heads * heads * LANES * OUT_DIM
    final = heads * TOKENS * OUT_DIM
    return {
        "schema_version": {"shape": [1], "dtype": "uint16"},
        "relation_delay": {"shape": [relation], "dtype": "uint8"},
        "relation_digest": {"shape": [relation, 32], "dtype": "uint8"},
        "weight_delay": {"shape": [weight], "dtype": "uint8"},
        "weight_digest": {"shape": [weight, 32], "dtype": "uint8"},
        "final_delay": {"shape": [final], "dtype": "uint8"},
        "final_digest": {"shape": [final, 32], "dtype": "uint8"},
    }


def verify_core(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    producer_path = root / "producer_complete.json"
    manifest = read_json(manifest_path)
    producer = read_json(producer_path)
    expected_files = {
        "manifest.json", "producer_complete.json", "task_plan.json",
        *ARTIFACT_FILES.values(), *SOURCE_FILES.values(),
    }
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    allowed_files = expected_files | {"verification_receipt.json"}
    if not expected_files.issubset(observed_files) or not observed_files.issubset(
        allowed_files
    ):
        raise ValueError("package file set contains missing or unregistered files")
    required_manifest_keys = {
        "schema", "status", "evidence", "formal_g0", "identity_schema", "seed",
        "sample_identity", "task_plan", "task_plan_sha256", "identity",
        "runtime_order", "flat_index_contract", "handshake_contract", "runtime_counts",
        "ledger_summary", "artifacts", "npz_members", "source_bindings", "generator",
        "generator_sha256", "boundary",
    }
    if set(manifest) != required_manifest_keys:
        raise ValueError("manifest key set is not exact")
    if (
        manifest["schema"] != "local5_identity_service_tables_v4"
        or manifest["status"] != "PASS_IDENTITY_SERVICE_TABLES_NOT_G0"
        or manifest["evidence"] != "[软件确定性服务合同]"
        or manifest["formal_g0"] != "DENY"
        or manifest["identity_schema"] != SCHEMA
        or manifest["seed"] != DEFAULT_SEED
        or manifest["runtime_order"] != RUNTIME_ORDER
        or manifest["flat_index_contract"] != FLAT_INDEX_CONTRACT
        or manifest["handshake_contract"] != HANDSHAKE_CONTRACT
        or manifest["boundary"] != BOUNDARY
    ):
        raise ValueError("manifest frozen metadata differs")
    plan_path = root / manifest["task_plan"]
    if plan_path.parent != root or plan_path.name != "task_plan.json":
        raise ValueError("task plan is not package-local")
    if not plan_path.is_file() or manifest["task_plan_sha256"] != sha256(plan_path):
        raise ValueError("task plan binding is invalid")
    plan = load_plan(plan_path)
    heads = strict_int(plan["heads"], "heads")
    identity_expected = {
        key: plan[key] for key in ("sample", "stage", "block", "window", "heads")
    }
    if manifest["sample_identity"] != sample_identity(plan) or manifest["identity"] != identity_expected:
        raise ValueError("window identity differs")
    runtime_counts = {
        "relation_lookup": heads * heads * TOKENS,
        "relation_unique_identity": heads * TOKENS,
        "weight": heads * heads * LANES * OUT_DIM,
        "final": heads * TOKENS * OUT_DIM,
    }
    if manifest["runtime_counts"] != runtime_counts:
        raise ValueError("runtime counts differ")

    expected_artifacts = expected_artifact_rows(heads, root)
    if manifest["artifacts"] != expected_artifacts:
        raise ValueError("artifact set/metadata differs")
    if manifest["npz_members"] != expected_npz_members(heads):
        raise ValueError("NPZ manifest schema differs")
    source_bindings = manifest["source_bindings"]
    if not isinstance(source_bindings, list) or len(source_bindings) != 3:
        raise ValueError("source binding count differs")
    observed_sources: dict[str, str] = {}
    for row in source_bindings:
        if not isinstance(row, dict) or set(row) != {"role", "file", "sha256"}:
            raise ValueError("source binding row schema differs")
        role = row["role"]
        if role in observed_sources or SOURCE_FILES.get(role) != row["file"]:
            raise ValueError("source binding role/path differs")
        path = root / row["file"]
        if not path.is_file() or row["sha256"] != sha256(path):
            raise ValueError("source binding SHA differs")
        observed_sources[role] = row["sha256"]
    if set(observed_sources) != set(SOURCE_FILES):
        raise ValueError("source binding roles differ")
    if observed_sources["independent_verifier"] != sha256(Path(__file__).resolve()):
        raise ValueError("executing verifier is not the package-bound verifier")
    if (
        manifest["generator"] != SOURCE_FILES["generator"]
        or manifest["generator_sha256"] != observed_sources["generator"]
    ):
        raise ValueError("generator role binding differs")

    transactions = expected_transactions(plan)
    expected_delays = {
        "relation": np.asarray(
            [row["delay"] for row in transactions["relation_unique"]], dtype=np.uint8
        ),
        "weight": np.asarray(
            [row["delay"] for row in transactions["weight_runtime"]], dtype=np.uint8
        ),
        "final": np.asarray(
            [row["delay"] for row in transactions["final_runtime"]], dtype=np.uint8
        ),
    }
    expected_digests = {
        "relation": np.frombuffer(
            b"".join(bytes.fromhex(row["digest"]) for row in transactions["relation_unique"]),
            dtype=np.uint8,
        ).reshape(heads * TOKENS, 32),
        "weight": np.frombuffer(
            b"".join(bytes.fromhex(row["digest"]) for row in transactions["weight_runtime"]),
            dtype=np.uint8,
        ).reshape(heads * heads * LANES * OUT_DIM, 32),
        "final": np.frombuffer(
            b"".join(bytes.fromhex(row["digest"]) for row in transactions["final_runtime"]),
            dtype=np.uint8,
        ).reshape(heads * TOKENS * OUT_DIM, 32),
    }
    for name, expected in expected_delays.items():
        if not np.array_equal(read_memh(root / f"{name}_delay.memh", expected.size), expected):
            raise ValueError(f"{name} delay differs from independent oracle")
    with np.load(root / "identity_service_tables.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected_npz_members(heads)):
            raise ValueError("NPZ member set differs")
        if archive["schema_version"].dtype != np.uint16 or not np.array_equal(
            archive["schema_version"], np.asarray([4], dtype=np.uint16)
        ):
            raise ValueError("NPZ schema_version differs")
        for name in expected_delays:
            if archive[f"{name}_delay"].dtype != np.uint8 or not np.array_equal(
                archive[f"{name}_delay"], expected_delays[name]
            ):
                raise ValueError(f"NPZ {name}_delay differs")
            if archive[f"{name}_digest"].dtype != np.uint8 or not np.array_equal(
                archive[f"{name}_digest"], expected_digests[name]
            ):
                raise ValueError(f"NPZ {name}_digest differs")
    expected_ledgers = {name: ledger_summary(rows) for name, rows in transactions.items()}
    if manifest["ledger_summary"] != expected_ledgers:
        raise ValueError("ledger summary differs from independent oracle")

    expected_producer = {
        "schema": "local5_identity_service_tables_producer_complete_v1",
        "status": "PASS_PRODUCER_COMPLETE_AWAITING_INDEPENDENT_VERIFY",
        "formal_g0": "DENY",
        "manifest_sha256": sha256(manifest_path),
        "artifact_sha256": {
            name: row["sha256"] for name, row in sorted(expected_artifacts.items())
        },
        "task_plan_sha256": sha256(plan_path),
        "source_sha256": dict(sorted(observed_sources.items())),
    }
    if producer != expected_producer:
        raise ValueError("producer complete receipt differs")
    return {
        "root": root,
        "heads": heads,
        "manifest_sha256": sha256(manifest_path),
        "producer_complete_sha256": sha256(producer_path),
        "artifact_sha256": expected_producer["artifact_sha256"],
        "source_sha256": expected_producer["source_sha256"],
        "task_plan_sha256": sha256(plan_path),
    }


def expected_verification_receipt(core: dict[str, Any]) -> dict[str, Any]:
    verifier_path = Path(__file__).resolve()
    return {
        "schema": "local5_identity_service_tables_independent_verify_v1",
        "status": "PASS_INDEPENDENT_VERIFY_NOT_G0",
        "evidence": "[独立软件重算]",
        "formal_g0": "DENY",
        "manifest_sha256": core["manifest_sha256"],
        "producer_complete_sha256": core["producer_complete_sha256"],
        "task_plan_sha256": core["task_plan_sha256"],
        "artifact_sha256": core["artifact_sha256"],
        "source_sha256": core["source_sha256"],
        "verifier_sha256": sha256(verifier_path),
        "independence": "no import from generator or local5_erep_identity_service_v4",
    }


def write_verification_receipt(root: Path) -> dict[str, Any]:
    core = verify_core(root)
    path = core["root"] / "verification_receipt.json"
    if path.exists():
        raise ValueError("verification receipt already exists")
    receipt = expected_verification_receipt(core)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return receipt


def verify_package(root: Path) -> dict[str, Any]:
    core = verify_core(root)
    receipt_path = core["root"] / "verification_receipt.json"
    receipt = read_json(receipt_path)
    if receipt != expected_verification_receipt(core):
        raise ValueError("independent verification receipt differs")
    return {
        "status": "PASS_IDENTITY_SERVICE_TABLES_VERIFIED_NOT_G0",
        "formal_g0": "DENY",
        "heads": core["heads"],
        "manifest_sha256": core["manifest_sha256"],
        "verification_receipt_sha256": sha256(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write-receipt", "verify"))
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    report = (
        write_verification_receipt(args.package_dir)
        if args.mode == "write-receipt"
        else verify_package(args.package_dir)
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
