#!/usr/bin/env python3
"""Audit whether chunk-granular Local/Motion selection justifies partial-output state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid boolean token: {value}")
    return value == "True"


def analyze_identity(label: str, tile_dir: Path) -> dict[str, Any]:
    records_path = tile_dir / "tile_records.csv"
    packed_path = tile_dir / "packed_tiles.npz"
    manifest_path = tile_dir / "manifest.json"
    for path in (records_path, packed_path, manifest_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing admitted tile evidence: {path}")
    records = list(csv.DictReader(records_path.open(encoding="utf-8")))
    if not records:
        raise ValueError(f"empty tile population: {records_path}")

    local_sources = 0
    row_hybrid_sources = 0
    chunk_hybrid_sources = 0
    local_products = 0
    row_hybrid_products = 0
    chunk_hybrid_products = 0
    chunk_motion_records = 0
    row_motion_records = 0
    physical_rows: dict[tuple[str, ...], tuple[int, int]] = {}
    per_sample_rows: dict[str, set[tuple[str, ...]]] = defaultdict(set)

    for row in records:
        current = int(row["tile_current_count"])
        transition = int(row["tile_positive_count"]) + int(row["tile_negative_count"])
        fanout = int(row["output_channel_fanout"])
        chunks = int(row["chunks_per_row"])
        state_valid = as_bool(row["state_valid"])
        row_motion = as_bool(row["row_use_motion"])
        if row_motion and not state_valid:
            raise ValueError("row selected Motion without valid temporal state")
        selected_row = transition if row_motion else current
        selected_chunk = min(current, transition) if state_valid else current
        local_sources += current
        row_hybrid_sources += selected_row
        chunk_hybrid_sources += selected_chunk
        local_products += current * fanout
        row_hybrid_products += selected_row * fanout
        chunk_hybrid_products += selected_chunk * fanout
        row_motion_records += int(row_motion)
        chunk_motion_records += int(state_valid and transition < current)

        physical_key = (
            row["sample_id"], row["sequence_key"], row["name"], row["operator"],
            row["operator_call_index"], row["row_id"], row["weight_group"],
        )
        geometry = (fanout, chunks)
        if physical_key in physical_rows and physical_rows[physical_key] != geometry:
            raise ValueError(f"physical row geometry changed: {physical_key}")
        physical_rows[physical_key] = geometry
        per_sample_rows[row["sample_id"]].add(physical_key)

    def saving(reference: int, candidate: int) -> float:
        return (reference - candidate) / reference if reference else 0.0

    state_by_sample: dict[str, dict[str, int]] = {}
    for sample, keys in per_sample_rows.items():
        row_bits = sum(physical_rows[key][0] * 32 for key in keys)
        chunk_bits = sum(
            physical_rows[key][0] * physical_rows[key][1] * 32 for key in keys
        )
        state_by_sample[sample] = {
            "row_destination_state_bits": row_bits,
            "chunk_partial_destination_state_bits": chunk_bits,
            "extra_chunk_partial_state_bits": chunk_bits - row_bits,
            "row_direction_bits": len(keys),
            "chunk_direction_bits": sum(physical_rows[key][1] for key in keys),
        }

    return {
        "identity": label,
        "tile_dir": str(tile_dir.resolve()),
        "records": len(records),
        "tile_records_sha256": sha256(records_path),
        "packed_tiles_sha256": sha256(packed_path),
        "tile_manifest_sha256": sha256(manifest_path),
        "source_work": {
            "local": local_sources,
            "row_hybrid": row_hybrid_sources,
            "chunk_hybrid": chunk_hybrid_sources,
            "row_saving_vs_local": saving(local_sources, row_hybrid_sources),
            "chunk_saving_vs_local": saving(local_sources, chunk_hybrid_sources),
            "chunk_incremental_saving_vs_row": saving(
                row_hybrid_sources, chunk_hybrid_sources
            ),
        },
        "product_work": {
            "local": local_products,
            "row_hybrid": row_hybrid_products,
            "chunk_hybrid": chunk_hybrid_products,
            "row_saving_vs_local": saving(local_products, row_hybrid_products),
            "chunk_saving_vs_local": saving(local_products, chunk_hybrid_products),
            "chunk_incremental_saving_vs_row": saving(
                row_hybrid_products, chunk_hybrid_products
            ),
        },
        "selection": {
            "row_motion_records": row_motion_records,
            "chunk_motion_records": chunk_motion_records,
        },
        "state_by_sample": state_by_sample,
        "max_state": {
            name: max(values[name] for values in state_by_sample.values())
            for name in next(iter(state_by_sample.values()))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"),
        required=True,
    )
    parser.add_argument("--survival-gain", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identities = [
        analyze_identity(label, Path(directory)) for label, directory in args.identity
    ]
    max_incremental = max(
        item["product_work"]["chunk_incremental_saving_vs_row"]
        for item in identities
    )
    survives = max_incremental >= args.survival_gain
    payload = {
        "schema": "m6_motion_granularity_audit_v1",
        "status": (
            "PASS_CHUNK_MOTION_SURVIVES"
            if survives else "PASS_CHUNK_MOTION_REJECTED_BELOW_GATE"
        ),
        "survival_gain": args.survival_gain,
        "max_incremental_product_saving": max_incremental,
        "decision": (
            "prototype chunk-partial destination state"
            if survives
            else "retain row-granular Local/Motion selection; do not add chunk-partial state"
        ),
        "claim_boundary": (
            "Exact source/product counts from admitted tile records. Chunk selection is a "
            "non-implemented upper bound and requires per-chunk previous-output partial sums; "
            "this is not an RTL speedup or an energy result."
        ),
        "identities": identities,
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "motion_granularity.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# M6 Local/Motion granularity audit\n\n",
        "| identity | row product saving | chunk product saving | chunk vs row | max extra partial state |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for item in identities:
        product = item["product_work"]
        lines.append(
            f"| {item['identity']} | {product['row_saving_vs_local']:.6%} | "
            f"{product['chunk_saving_vs_local']:.6%} | "
            f"{product['chunk_incremental_saving_vs_row']:.6%} | "
            f"{item['max_state']['extra_chunk_partial_state_bits']} bit |\n"
        )
    lines.append(f"\nDecision: `{payload['decision']}`.\n")
    (args.output / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    print(f"PASS: {payload['status']} -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
