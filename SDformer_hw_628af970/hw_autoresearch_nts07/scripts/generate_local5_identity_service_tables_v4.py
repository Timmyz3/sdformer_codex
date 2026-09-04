#!/usr/bin/env python3
"""Generate identity-derived service-delay tables for a Local5 numeric window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scripts.local5_erep_identity_service_v4 import (
        DEFAULT_SEED,
        SCHEMA,
        Transaction,
        ledger_digests,
        make_transaction,
    )
except ModuleNotFoundError:
    from local5_erep_identity_service_v4 import (
        DEFAULT_SEED,
        SCHEMA,
        Transaction,
        ledger_digests,
        make_transaction,
    )


TOKENS = 450
LANES = 32
OUT_DIM = 32


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"task plan {name} is not a non-negative integer")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "sample",
        "stage",
        "block",
        "window",
        "heads",
        "out_dim",
        "tasks",
        "source_manifest_sha256",
        "source_payload_sha256",
        "projection_contract_sha256",
        "projection_payload_sha256",
        "task_sha256",
    }
    if (
        not isinstance(plan, dict)
        or not required.issubset(plan)
        or plan.get("schema") != "local5_projection_task_plan_v1"
    ):
        raise ValueError("task plan schema is not a Local5 projection plan")
    heads = strict_nonnegative_int(plan["heads"], "heads")
    out_dim = strict_nonnegative_int(plan["out_dim"], "out_dim")
    if heads < 1 or heads > 32 or out_dim != OUT_DIM:
        raise ValueError("task plan heads/out_dim is outside the frozen contract")
    tasks = plan["tasks"]
    base_groups: list[int] = []
    if isinstance(tasks, list) and len(tasks) == heads * heads:
        for head in range(heads):
            row = tasks[head]
            if not isinstance(row, dict) or set(row) != {
                "input_group_index", "output_tile"
            }:
                raise ValueError("task plan task row schema is not exact")
            base_groups.append(
                strict_nonnegative_int(row["input_group_index"], "input_group_index")
            )
    expected_tasks = [
        {"input_group_index": base_groups[head], "output_tile": tile}
        for tile in range(heads)
        for head in range(heads)
    ] if base_groups else []
    if (
        len(set(base_groups)) != heads
        or tasks != expected_tasks
        or canonical_sha(tasks) != plan["task_sha256"]
    ):
        raise ValueError("task plan is not exact tile-major HxH order")
    for field in ("sample", "stage", "block", "window"):
        strict_nonnegative_int(plan[field], field)
    for field in (
        "source_manifest_sha256",
        "source_payload_sha256",
        "projection_contract_sha256",
        "projection_payload_sha256",
    ):
        value = plan[field]
        if not is_sha256(value):
            raise ValueError(f"task plan {field} is not SHA-256")
    return plan


def sample_identity(plan: dict[str, Any]) -> str:
    return (
        "local5/profile100/"
        f"{plan['source_manifest_sha256']}/sample/{int(plan['sample']):03d}"
    )


def relation_transaction(
    plan: dict[str, Any], input_head: int, source_id: int, seed: int
) -> Transaction:
    return make_transaction(
        "relation",
        {
            "sample": sample_identity(plan),
            "stage": int(plan["stage"]),
            "block": int(plan["block"]),
            "window": int(plan["window"]),
            "input_head": input_head,
            "source_id": source_id,
        },
        seed=seed,
    )


def weight_transaction(
    plan: dict[str, Any], input_head: int, output_tile: int,
    lane: int, out: int, seed: int
) -> Transaction:
    return make_transaction(
        "weight",
        {
            "sample": sample_identity(plan),
            "stage": int(plan["stage"]),
            "block": int(plan["block"]),
            "window": int(plan["window"]),
            "input_head": input_head,
            "output_tile": output_tile,
            "lane": lane,
            "out": out,
        },
        seed=seed,
    )


def final_transaction(
    plan: dict[str, Any], output_tile: int, source_id: int, out: int, seed: int
) -> Transaction:
    return make_transaction(
        "final",
        {
            "sample": sample_identity(plan),
            "stage": int(plan["stage"]),
            "block": int(plan["block"]),
            "window": int(plan["window"]),
            "output_tile": output_tile,
            "source_id": source_id,
            "out": out,
        },
        seed=seed,
    )


def transaction_arrays(
    rows: Iterable[Transaction],
) -> tuple[np.ndarray, np.ndarray]:
    items = tuple(rows)
    delays = np.asarray([row.delay for row in items], dtype=np.uint8)
    digests = np.frombuffer(
        b"".join(bytes.fromhex(row.digest) for row in items), dtype=np.uint8
    ).reshape(len(items), 32).copy()
    return delays, digests


def compact_ledger(rows: Iterable[Transaction]) -> dict[str, Any]:
    digest = ledger_digests(rows)
    multiplicities = [row.as_dict() for row in digest.identity_multiplicities]
    histogram = Counter(row["multiplicity"] for row in multiplicities)
    return {
        "transaction_count": digest.transaction_count,
        "identity_count": digest.identity_count,
        "ordered_ledger_digest": digest.ordered_digest,
        "unordered_multiset_digest": digest.multiset_digest,
        "identity_multiplicity_sha256": canonical_sha(multiplicities),
        "multiplicity_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
    }


def build_tables(plan: dict[str, Any], seed: int) -> dict[str, Any]:
    if seed != DEFAULT_SEED:
        raise ValueError(f"seed must equal frozen DEFAULT_SEED={DEFAULT_SEED}")
    heads = int(plan["heads"])
    relation_unique = tuple(
        relation_transaction(plan, head, source, seed)
        for head in range(heads)
        for source in range(TOKENS)
    )
    weights = tuple(
        weight_transaction(plan, head, tile, lane, out, seed)
        for tile in range(heads)
        for head in range(heads)
        for lane in range(LANES)
        for out in range(OUT_DIM)
    )
    finals = tuple(
        final_transaction(plan, tile, source, out, seed)
        for tile in range(heads)
        for source in range(TOKENS)
        for out in range(OUT_DIM)
    )
    relation_runtime = tuple(
        relation_transaction(plan, head, source, seed)
        for _tile in range(heads)
        for head in range(heads)
        for source in range(TOKENS)
    )
    relation_delay, relation_digest = transaction_arrays(relation_unique)
    weight_delay, weight_digest = transaction_arrays(weights)
    final_delay, final_digest = transaction_arrays(finals)
    return {
        "relation_delay": relation_delay,
        "relation_digest": relation_digest,
        "weight_delay": weight_delay,
        "weight_digest": weight_digest,
        "final_delay": final_delay,
        "final_digest": final_digest,
        "ledger_summary": {
            "relation_unique": compact_ledger(relation_unique),
            "relation_runtime": compact_ledger(relation_runtime),
            "weight_runtime": compact_ledger(weights),
            "final_runtime": compact_ledger(finals),
        },
    }


def write_memh(path: Path, values: np.ndarray) -> None:
    if values.dtype != np.uint8 or values.ndim != 1 or np.any(values > 3):
        raise ValueError("service-delay memh values must be uint8 in 0..3")
    path.write_text("".join(f"{int(value):x}\n" for value in values), encoding="ascii")


def write_tables(plan_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    plan = load_plan(plan_path)
    tables = build_tables(plan, seed)
    if output_dir.exists():
        raise ValueError("output directory already exists; use a new immutable path")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f"{output_dir.name}.staging.{os.getpid()}")
    if staging.exists():
        raise ValueError("staging directory already exists")
    staging.mkdir()
    source_dir = staging / "source"
    source_dir.mkdir()
    archived_task_plan = staging / "task_plan.json"
    shutil.copy2(plan_path, archived_task_plan)
    source_paths = {
        "generator": Path(__file__).resolve(),
        "identity_oracle": (
            Path(__file__).resolve().parent / "local5_erep_identity_service_v4.py"
        ),
        "independent_verifier": (
            Path(__file__).resolve().parent
            / "verify_local5_identity_service_tables_v4.py"
        ),
    }
    for source in source_paths.values():
        shutil.copy2(source, source_dir / source.name)
    npz_path = staging / "identity_service_tables.npz"
    np.savez(
        npz_path,
        schema_version=np.asarray([4], dtype=np.uint16),
        relation_delay=tables["relation_delay"],
        relation_digest=tables["relation_digest"],
        weight_delay=tables["weight_delay"],
        weight_digest=tables["weight_digest"],
        final_delay=tables["final_delay"],
        final_digest=tables["final_digest"],
    )
    artifacts: dict[str, Any] = {
        "identity_service_tables": {
            "file": npz_path.name,
            "sha256": sha256(npz_path),
        }
    }
    for name in ("relation", "weight", "final"):
        path = staging / f"{name}_delay.memh"
        values = tables[f"{name}_delay"]
        write_memh(path, values)
        artifacts[f"{name}_delay"] = {
            "file": path.name,
            "entries": int(values.size),
            "width": 2,
            "sha256": sha256(path),
        }
    manifest = {
        "schema": "local5_identity_service_tables_v4",
        "status": "PASS_IDENTITY_SERVICE_TABLES_NOT_G0",
        "evidence": "[软件确定性服务合同]",
        "formal_g0": "DENY",
        "identity_schema": SCHEMA,
        "seed": seed,
        "sample_identity": sample_identity(plan),
        "task_plan": archived_task_plan.name,
        "task_plan_sha256": sha256(archived_task_plan),
        "identity": {
            key: int(plan[key])
            for key in ("sample", "stage", "block", "window", "heads")
        },
        "runtime_order": {
            "relation": "output_tile,input_head,source_id; identity excludes output_tile",
            "weight": "output_tile,input_head,lane,out",
            "final": "output_tile,source_id,out",
        },
        "flat_index_contract": {
            "relation": "input_head*450+source_id",
            "relation_runtime_reuse": "output_tile does not enter the flat index",
            "weight": "(((output_tile*heads+input_head)*32+lane)*32)+out",
            "final": "((output_tile*450+source_id)*32)+out",
        },
        "handshake_contract": {
            "request_accept": "posedge(request_valid&&request_ready)",
            "response_available": "first posedge response_valid after request_accept",
            "response_latency": "response_available-request_accept=delay+1",
            "response_accept": "posedge(response_valid&&response_ready), not earlier than response_available",
            "response_stability": "response_valid and payload remain stable from available through accept",
            "outstanding": "at most one relation request and one weight request",
            "final_request": "first posedge tile_result_valid for each ordered final item",
            "final_accept": "exactly delay+1 cycles after final_request while valid remains asserted",
        },
        "runtime_counts": {
            "relation_lookup": int(plan["heads"]) ** 2 * TOKENS,
            "relation_unique_identity": int(plan["heads"]) * TOKENS,
            "weight": int(plan["heads"]) ** 2 * LANES * OUT_DIM,
            "final": int(plan["heads"]) * TOKENS * OUT_DIM,
        },
        "ledger_summary": tables["ledger_summary"],
        "artifacts": artifacts,
        "npz_members": {
            "schema_version": {"shape": [1], "dtype": "uint16"},
            "relation_delay": {
                "shape": [int(tables["relation_delay"].size)], "dtype": "uint8"
            },
            "relation_digest": {
                "shape": list(tables["relation_digest"].shape), "dtype": "uint8"
            },
            "weight_delay": {
                "shape": [int(tables["weight_delay"].size)], "dtype": "uint8"
            },
            "weight_digest": {
                "shape": list(tables["weight_digest"].shape), "dtype": "uint8"
            },
            "final_delay": {
                "shape": [int(tables["final_delay"].size)], "dtype": "uint8"
            },
            "final_digest": {
                "shape": list(tables["final_digest"].shape), "dtype": "uint8"
            },
        },
        "source_bindings": [
            {
                "role": role,
                "file": f"source/{source.name}",
                "sha256": sha256(source_dir / source.name),
            }
            for role, source in source_paths.items()
        ],
        "generator": "source/generate_local5_identity_service_tables_v4.py",
        "generator_sha256": sha256(
            source_dir / "generate_local5_identity_service_tables_v4.py"
        ),
        "boundary": [
            "delay is identity-derived 0..3; registered response latency is delay+1",
            "tables are verification inputs, not DUT state or an architecture result",
            "numeric DUT and formal G0 are unchanged",
        ],
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete = {
        "schema": "local5_identity_service_tables_producer_complete_v1",
        "status": "PASS_PRODUCER_COMPLETE_AWAITING_INDEPENDENT_VERIFY",
        "formal_g0": "DENY",
        "manifest_sha256": sha256(manifest_path),
        "artifact_sha256": {
            name: row["sha256"] for name, row in sorted(artifacts.items())
        },
        "task_plan_sha256": sha256(archived_task_plan),
        "source_sha256": {
            row["role"]: row["sha256"] for row in manifest["source_bindings"]
        },
    }
    (staging / "producer_complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = write_tables(args.task_plan.resolve(), args.output_dir.resolve(), args.seed)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
