#!/usr/bin/env python3
"""Build content-addressed VCS vectors from admitted real checkpoint bitmaps.

The output is deliberately limited to activation bitmaps and record identities.
The M2B VCS test supplies a deterministic INT8 weight function, so this proves
the banked issue/accumulation machinery on real density/order patterns but does
not claim checkpoint-weight or network-output equivalence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validator() -> tuple[Any, Path]:
    path = Path(__file__).with_name("build_dual_line_tile_memory_trace.py")
    spec = importlib.util.spec_from_file_location("dual_line_tile_memory_trace", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def select_indices(count: int, requested: int) -> list[int]:
    if count <= 0 or requested <= 0:
        raise ValueError("record count and requested count must be positive")
    if requested >= count:
        return list(range(count))
    # Endpoint-inclusive uniform sampling covers all operator/time regions and
    # is deterministic across Python/numpy versions after integer conversion.
    indices = np.linspace(0, count - 1, num=requested, dtype=np.int64).tolist()
    if len(set(indices)) != requested:
        raise ValueError("uniform record sampling unexpectedly produced duplicates")
    return [int(index) for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True)
    parser.add_argument("--per-identity", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validator, validator_path = load_validator()
    cases: list[dict[str, Any]] = []
    identities: dict[str, Any] = {}
    tile_bits: int | None = None
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        manifest, records, current, _previous = validator.validate(directory)
        identity_tile_bits = int(manifest["tile_bits"])
        if tile_bits is None:
            tile_bits = identity_tile_bits
        elif tile_bits != identity_tile_bits:
            raise ValueError("all identities must use the same tile width")
        indices = select_indices(len(records), args.per_identity)
        popcounts = []
        for record_id in indices:
            row = records[record_id]
            valid_bits = int(row["valid_bits"])
            packed = bytes(current[record_id])
            value = int.from_bytes(packed, byteorder="little", signed=False)
            if valid_bits < identity_tile_bits:
                value &= (1 << valid_bits) - 1
            count = int(row["tile_current_count"])
            if value.bit_count() != count:
                raise ValueError(f"masked bitmap count mismatch: {label}:{record_id}")
            popcounts.append(count)
            cases.append({
                "case_id": len(cases),
                "identity": label,
                "record_id": record_id,
                "valid_bits": valid_bits,
                "popcount": count,
                "bitmap": value,
            })
        identities[label] = {
            "source_directory": str(directory),
            "source_manifest_sha256": sha256(directory / "manifest.json"),
            "source_packed_tiles_sha256": sha256(directory / "packed_tiles.npz"),
            "available_records": len(records),
            "selected_records": len(indices),
            "selected_record_id_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "popcount": {
                "min": min(popcounts),
                "max": max(popcounts),
                "mean": float(np.mean(popcounts)),
                "p50": float(np.percentile(popcounts, 50)),
                "p95": float(np.percentile(popcounts, 95)),
            },
        }

    assert tile_bits is not None
    hex_digits = (tile_bits + 3) // 4
    hex_path = args.output_dir / "current_tiles.hex"
    hex_path.write_text(
        "".join(f"{case['bitmap']:0{hex_digits}x}\n" for case in cases), encoding="ascii"
    )
    index_path = args.output_dir / "case_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "identity", "record_id", "valid_bits", "popcount"]
        )
        writer.writeheader()
        writer.writerows({key: case[key] for key in writer.fieldnames} for case in cases)

    payload = {
        "schema": "m2b_real_tile_vcs_vectors_v1",
        "status": "PASS_REAL_BITMAPS_DETERMINISTIC_WEIGHT_MITER_NOT_CHECKPOINT_WEIGHT_ORACLE",
        "tile_bits": tile_bits,
        "cases": len(cases),
        "selection": "endpoint_inclusive_uniform_record_id",
        "identities": identities,
        "sha256": {
            hex_path.name: sha256(hex_path),
            index_path.name: sha256(index_path),
            Path(__file__).name: sha256(Path(__file__)),
            validator_path.name: sha256(validator_path),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: wrote {len(cases)} real-tile VCS vectors to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
