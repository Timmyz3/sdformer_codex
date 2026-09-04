#!/usr/bin/env python3
"""Generate checkpoint-bound two-channel projection vectors for H67 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
EXPECTED_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
LANES = 32
TOKENS = 450


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_bits(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape)
    bits = np.unpackbits(packed, bitorder="little")
    if bits.size < count:
        raise ValueError("packed bit payload is shorter than the declared shape")
    return bits[:count].reshape(shape).astype(np.bool_)


def expected_names() -> list[str]:
    return [
        f"S{stage}.B{block}.attn"
        for stage, depth in EXPECTED_BLOCKS.items()
        for block in range(depth)
    ]


def parse_base_vectors(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().split()
        if header != ["138", str(TOKENS)]:
            raise ValueError(f"unexpected H67 base-vector header: {header}")
        for expected_row in range(138):
            fields = handle.readline().split()
            if len(fields) != 6 or int(fields[0]) != expected_row:
                raise ValueError(f"invalid base-vector row header {expected_row}")
            vectors = []
            for token in range(TOKENS):
                values = handle.readline().split()
                if len(values) != 4:
                    raise ValueError(
                        f"invalid base-vector payload row={expected_row} token={token}"
                    )
                vectors.append(
                    {
                        "q": int(values[0], 16),
                        "k": int(values[1], 16),
                        "peer": int(values[2], 16),
                        "gate": int(values[3]),
                    }
                )
            rows.append(
                {
                    "row": int(fields[0]),
                    "stage": int(fields[1]),
                    "block": int(fields[2]),
                    "head": int(fields[3]),
                    "vectors": vectors,
                }
            )
        if handle.read().strip():
            raise ValueError("base-vector file has trailing data")
    return rows


def bitmap32(bits: np.ndarray) -> int:
    value = 0
    for lane in np.flatnonzero(bits):
        value |= 1 << int(lane)
    return value


def generate(
    trace_manifest_path: Path,
    base_vector_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    trace_manifest_path = trace_manifest_path.resolve()
    base_vector_path = base_vector_path.resolve()
    output_dir = output_dir.resolve()
    manifest = json.loads(trace_manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if [str(record.get("name", "")) for record in records] != expected_names():
        raise ValueError("trace manifest is not ordered all12 H67 coverage")
    base_rows = parse_base_vectors(base_vector_path)

    output_rows: list[dict[str, Any]] = []
    record_receipts: list[dict[str, Any]] = []
    base_index = 0
    for record in records:
        name = str(record["name"])
        stage = int(name.split(".")[0][1:])
        block = int(name.split(".")[1][1:])
        source = Path(record["file"])
        if sha256_file(source) != record.get("sha256"):
            raise ValueError(f"trace SHA mismatch: {source}")
        with np.load(source) as payload:
            k_shape = tuple(int(value) for value in payload["k_shape"])
            k_bits = unpack_bits(payload["k_bits_packed"], k_shape)
            gate = payload["gate_q17"].astype(np.int64)
            weight = payload["projection_weight_int8"].astype(np.int64)
            bias = payload["projection_bias_acc_int64"].astype(np.int64)

        time_steps, windows, heads, spatial_tokens, lanes = k_shape
        dim = heads * lanes
        if (
            time_steps != 2
            or windows < 1
            or heads != EXPECTED_HEADS[stage]
            or spatial_tokens * 2 != TOKENS
            or lanes != LANES
            or gate.shape != (windows, heads, TOKENS)
            or weight.shape != (dim, dim)
            or bias.shape != (dim,)
        ):
            raise ValueError(f"unsupported H67 record shape: {name}")

        channel0 = (stage * 13 + block * 31) % dim
        channel1 = (channel0 + dim // 2) % dim
        k_rows = k_bits[:, 0].transpose(1, 0, 2, 3).reshape(
            heads, TOKENS, LANES
        )
        gate_rows = gate[0]
        record_expected = np.zeros((heads, 2), dtype=np.int64)
        for head in range(heads):
            base = base_rows[base_index]
            if (base["stage"], base["block"], base["head"]) != (
                stage,
                block,
                head,
            ):
                raise ValueError(f"base-vector identity mismatch at row {base_index}")
            expected_k = [bitmap32(bits) for bits in k_rows[head]]
            expected_gate = [int(value) for value in gate_rows[head]]
            if [item["k"] for item in base["vectors"]] != expected_k:
                raise ValueError(f"base-vector K mismatch at row {base_index}")
            if [item["gate"] for item in base["vectors"]] != expected_gate:
                raise ValueError(f"base-vector gate mismatch at row {base_index}")

            input_base = head * LANES
            selected_weights = weight[
                [channel0, channel1], input_base : input_base + LANES
            ]
            expected = np.zeros(2, dtype=np.int64)
            for token in range(TOKENS):
                active = np.flatnonzero(k_rows[head, token])
                if active.size:
                    expected += int(gate_rows[head, token]) * selected_weights[
                        :, active
                    ].sum(axis=1, dtype=np.int64)
            if expected.min(initial=0) < -(1 << 31) or expected.max(
                initial=0
            ) >= (1 << 31):
                raise ValueError(f"projection2 Acc32 overflow at row {base_index}")
            record_expected[head] = expected
            output_rows.append(
                {
                    "row": base_index,
                    "stage": stage,
                    "block": block,
                    "head": head,
                    "channels": [channel0, channel1],
                    "expected": [int(expected[0]), int(expected[1])],
                    "weight0": [int(value) for value in selected_weights[0]],
                    "weight1": [int(value) for value in selected_weights[1]],
                }
            )
            base_index += 1

        total_from_rows = record_expected.sum(axis=0, dtype=np.int64)
        selected_full_weights = weight[[channel0, channel1]].reshape(
            2, heads, LANES
        )
        direct_total = np.asarray(
            [
                np.sum(
                    gate_rows[:, :, None]
                    * k_rows.astype(np.int64)
                    * selected_full_weights[channel, :, None, :],
                    dtype=np.int64,
                )
                for channel in range(2)
            ],
            dtype=np.int64,
        )
        if not np.array_equal(total_from_rows, direct_total):
            raise ValueError(f"per-head/full-record projection mismatch: {name}")
        record_receipts.append(
            {
                "name": name,
                "source": str(source.resolve()),
                "source_sha256": record["sha256"],
                "heads": heads,
                "channels": [channel0, channel1],
                "pre_bias_acc32_sum": [
                    int(total_from_rows[0]),
                    int(total_from_rows[1]),
                ],
                "bias_acc32": [int(bias[channel0]), int(bias[channel1])],
            }
        )

    if base_index != 138 or len(output_rows) != 138:
        raise ValueError("generated row coverage is not 138")
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / "h67_real_weight_projection2.txt"
    with vector_path.open("w", encoding="ascii") as handle:
        handle.write(f"{len(output_rows)} 2\n")
        for row in output_rows:
            handle.write(
                f"{row['row']} {row['stage']} {row['block']} {row['head']} "
                f"{row['expected'][0]} {row['expected'][1]}\n"
            )
            for weight0, weight1 in zip(row["weight0"], row["weight1"]):
                handle.write(f"{weight0} {weight1}\n")

    result = {
        "schema": "h67_real_weight_projection2_vectors_v1",
        "status": "PASS",
        "scope": (
            "checkpoint-derived INT8 two-output-channel pre-bias Acc32 per "
            "head-row; numeric integration boundary, not full projection throughput"
        ),
        "rows": len(output_rows),
        "channels_per_row": 2,
        "acc32_expected_values": len(output_rows) * 2,
        "trace_manifest": str(trace_manifest_path),
        "trace_manifest_sha256": sha256_file(trace_manifest_path),
        "base_vector": str(base_vector_path),
        "base_vector_sha256": sha256_file(base_vector_path),
        "run_context": manifest.get("run_context", {}),
        "records": record_receipts,
        "vector": str(vector_path),
        "vector_sha256": sha256_file(vector_path),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": [
            "Weights are checkpoint-derived INT8 codes under the frozen trace contract.",
            "Results are pre-bias and cover two deterministic output channels per block.",
            "This vector set does not establish full projection, full encoder, or ASIC PPA.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--base-vector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.trace_manifest, args.base_vector, args.output_dir)
    print(
        f"PASS H67 real-weight projection2 vectors rows={result['rows']} "
        f"acc32={result['acc32_expected_values']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
