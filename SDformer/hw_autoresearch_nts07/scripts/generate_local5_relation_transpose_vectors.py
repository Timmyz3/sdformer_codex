#!/usr/bin/env python3
"""生成T450 Local5 Python/RTL relation-transpose共用向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from profile_local5_hardware_features import source_descriptor_trace


HEIGHT = 15
WIDTH = 15
PLANES = 2
TOKENS = HEIGHT * WIDTH
TOTAL = PLANES * TOKENS
LANES = 32
GATE_W = 9


def write_memh(path: Path, values: list[int], width: int) -> None:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text(
        "".join(f"{value & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plane(plane: int) -> dict[str, list[int] | list[list[int]]]:
    neighbor = torch.zeros((TOKENS, 5), dtype=torch.long)
    valid = torch.zeros((TOKENS, 5), dtype=torch.bool)
    for destination in range(TOKENS):
        y, x = divmod(destination, WIDTH)
        coordinates = (
            (y, x),
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
        )
        for role, (source_y, source_x) in enumerate(coordinates):
            geometric = (
                0 <= source_y < HEIGHT
                and 0 <= source_x < WIDTH
            )
            runtime_valid = (
                role == 0
                or (destination * 7 + role * 11 + plane * 13) % 19
                != 0
            )
            valid[destination, role] = geometric and runtime_valid
            neighbor[destination, role] = (
                source_y * WIDTH + source_x
                if geometric
                else destination
            )

    source_k = torch.zeros((TOKENS, LANES), dtype=torch.bool)
    for source in range(TOKENS):
        for lane in range(LANES):
            source_k[source, lane] = (
                (source * 17 + lane * 29 + plane * 31) % 11
            ) in (0, 1, 4)
    k_candidates = source_k[neighbor]
    gates = torch.zeros((TOKENS, 5), dtype=torch.long)
    for destination in range(TOKENS):
        for role in range(5):
            if valid[destination, role]:
                gates[destination, role] = (
                    1
                    + (
                        plane * TOKENS * 5
                        + destination * 5
                        + role
                    )
                    * 73
                    % 256
                )

    descriptor = source_descriptor_trace(
        k_candidates,
        gates,
        valid,
        neighbor,
    )
    input_valid = []
    input_k = []
    input_gates = []
    for destination in range(TOKENS):
        mask = 0
        packed_gate = 0
        for role in range(5):
            mask |= int(valid[destination, role]) << role
            packed_gate |= int(gates[destination, role]) << (
                role * GATE_W
            )
        input_valid.append(mask)
        k_bitmap = 0
        for lane in range(LANES):
            k_bitmap |= int(source_k[destination, lane]) << lane
        input_k.append(k_bitmap)
        input_gates.append(packed_gate)
    return {
        "input_valid": input_valid,
        "input_k": input_k,
        "input_gates": input_gates,
        "expected_k": descriptor["source_k_bitmap"],
        "expected_gates": [
            sum(
                int(gate) << (role * GATE_W)
                for role, gate in enumerate(row)
            )
            for row in descriptor["incoming_gates"]
        ],
        "expected_mask": descriptor["incoming_valid_mask"],
    }


def generate(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[int]] = {
        "input_valid": [],
        "input_k": [],
        "input_gates": [],
        "expected_k": [],
        "expected_gates": [],
        "expected_mask": [],
    }
    for plane in range(PLANES):
        row = build_plane(plane)
        for name in combined:
            combined[name].extend(int(value) for value in row[name])
    widths = {
        "input_valid": 5,
        "input_k": LANES,
        "input_gates": 5 * GATE_W,
        "expected_k": LANES,
        "expected_gates": 5 * GATE_W,
        "expected_mask": 5,
    }
    artifacts = {}
    for name, values in combined.items():
        path = output_dir / f"{name}.memh"
        write_memh(path, values, widths[name])
        artifacts[name] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "entries": len(values),
            "width": widths[name],
        }
    manifest = {
        "schema": "local5_relation_transpose_python_vectors_v1",
        "height": HEIGHT,
        "width": WIDTH,
        "planes": PLANES,
        "tokens_per_plane": TOKENS,
        "total_descriptors": TOTAL,
        "lanes": LANES,
        "gate_width": GATE_W,
        "runtime_invalid": True,
        "python_reference": "source_descriptor_trace",
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    value = generate(args.output_dir)
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
