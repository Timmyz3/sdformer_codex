#!/usr/bin/env python3
"""Generate batched all-output checkpoint-weight vectors for H67 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts import generate_h67_real_weight_projection2_vectors as base
except ImportError:
    import generate_h67_real_weight_projection2_vectors as base


BATCH_CHANNELS = 16
MAX_DIM = max(base.EXPECTED_HEADS.values()) * base.LANES
BATCHES = MAX_DIM // BATCH_CHANNELS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def expected_valid_scalars() -> int:
    return sum(
        base.EXPECTED_BLOCKS[stage]
        * base.EXPECTED_HEADS[stage]
        * (base.EXPECTED_HEADS[stage] * base.LANES)
        for stage in range(4)
    )


def generate(
    trace_manifest_path: Path,
    base_vector_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    trace_manifest_path = trace_manifest_path.resolve()
    base_vector_path = base_vector_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")

    trace_manifest = json.loads(trace_manifest_path.read_text(encoding="utf-8"))
    records = trace_manifest.get("records", [])
    if [str(record.get("name", "")) for record in records] != base.expected_names():
        raise ValueError("trace manifest is not ordered all12 H67 coverage")
    base_rows = base.parse_base_vectors(base_vector_path)

    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
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
            k_bits = base.unpack_bits(payload["k_bits_packed"], k_shape)
            gate = payload["gate_q17"].astype(np.int64)
            weight = payload["projection_weight_int8"].astype(np.int64)
            bias = payload["projection_bias_acc_int64"].astype(np.int64)

        time_steps, windows, heads, spatial_tokens, lanes = k_shape
        dim = heads * lanes
        if (
            time_steps != 2
            or windows < 1
            or heads != base.EXPECTED_HEADS[stage]
            or spatial_tokens * 2 != base.TOKENS
            or lanes != base.LANES
            or gate.shape != (windows, heads, base.TOKENS)
            or weight.shape != (dim, dim)
            or bias.shape != (dim,)
        ):
            raise ValueError(f"unsupported H67 record shape: {name}")

        k_rows = k_bits[:, 0].transpose(1, 0, 2, 3).reshape(
            heads, base.TOKENS, base.LANES
        )
        gate_rows = gate[0]
        lane_gate_sum = np.einsum(
            "ht,htl->hl",
            gate_rows,
            k_rows.astype(np.int64),
            dtype=np.int64,
            optimize=True,
        )
        partials = np.zeros((heads, dim), dtype=np.int64)
        for head in range(heads):
            input_base = head * base.LANES
            partials[head] = (
                weight[:, input_base : input_base + base.LANES]
                @ lane_gate_sum[head]
            )
            if partials[head].min(initial=0) < -(1 << 31) or partials[
                head
            ].max(initial=0) >= (1 << 31):
                raise ValueError(f"Acc32 overflow at {name} head={head}")

            source_row = base_rows[base_index]
            if (source_row["stage"], source_row["block"], source_row["head"]) != (
                stage,
                block,
                head,
            ):
                raise ValueError(f"base-vector identity mismatch at row {base_index}")
            expected_k = [base.bitmap32(bits) for bits in k_rows[head]]
            expected_gate = [int(value) for value in gate_rows[head]]
            if [item["k"] for item in source_row["vectors"]] != expected_k:
                raise ValueError(f"base-vector K mismatch at row {base_index}")
            if [item["gate"] for item in source_row["vectors"]] != expected_gate:
                raise ValueError(f"base-vector gate mismatch at row {base_index}")
            rows.append(
                {
                    "row": base_index,
                    "stage": stage,
                    "block": block,
                    "head": head,
                    "dim": dim,
                    "expected": partials[head],
                    "weight": weight[:, input_base : input_base + base.LANES],
                }
            )
            base_index += 1

        direct_total = np.einsum(
            "ht,htl,chl->c",
            gate_rows,
            k_rows.astype(np.int64),
            weight.reshape(dim, heads, base.LANES),
            dtype=np.int64,
            optimize=True,
        )
        if not np.array_equal(partials.sum(axis=0, dtype=np.int64), direct_total):
            raise ValueError(f"cross-head full-output mismatch: {name}")
        receipts.append(
            {
                "name": name,
                "source": str(source.resolve()),
                "source_sha256": record["sha256"],
                "heads": heads,
                "output_channels": dim,
                "partial_acc32_sha256": array_sha256(partials),
                "cross_head_acc32_sha256": array_sha256(direct_total),
                "bias_acc32_sha256": array_sha256(bias),
            }
        )

    if base_index != 138 or len(rows) != 138:
        raise ValueError("generated row coverage is not 138")

    output_dir.mkdir(parents=True)
    vectors = []
    valid_scalars = 0
    for batch in range(BATCHES):
        channel_base = batch * BATCH_CHANNELS
        vector_path = output_dir / f"batch_{batch:02d}.txt"
        batch_valid = 0
        with vector_path.open("w", encoding="ascii") as handle:
            handle.write(f"{len(rows)} {BATCH_CHANNELS} {batch}\n")
            for row in rows:
                valid = max(0, min(BATCH_CHANNELS, row["dim"] - channel_base))
                batch_valid += valid
                handle.write(
                    f"{row['row']} {row['stage']} {row['block']} "
                    f"{row['head']} {valid}\n"
                )
                expected = np.zeros(BATCH_CHANNELS, dtype=np.int64)
                weights = np.zeros(
                    (BATCH_CHANNELS, base.LANES), dtype=np.int64
                )
                if valid:
                    expected[:valid] = row["expected"][
                        channel_base : channel_base + valid
                    ]
                    weights[:valid] = row["weight"][
                        channel_base : channel_base + valid
                    ]
                handle.write(" ".join(str(int(value)) for value in expected) + "\n")
                for lane in range(base.LANES):
                    handle.write(
                        " ".join(
                            str(int(weights[channel, lane]))
                            for channel in range(BATCH_CHANNELS)
                        )
                        + "\n"
                    )
        valid_scalars += batch_valid
        vectors.append(
            {
                "batch": batch,
                "channel_base": channel_base,
                "path": str(vector_path),
                "sha256": sha256_file(vector_path),
                "valid_acc32_values": batch_valid,
            }
        )

    if valid_scalars != expected_valid_scalars():
        raise ValueError(
            f"full-output scalar coverage mismatch: {valid_scalars}"
        )
    result = {
        "schema": "h67_real_weight_projection_all_vectors_v1",
        "status": "PASS",
        "scope": (
            "checkpoint-derived INT8 all-output-channel pre-bias partial Acc32 "
            "per head-row, batched 16 channels at a time"
        ),
        "rows": len(rows),
        "batch_channels": BATCH_CHANNELS,
        "batches": BATCHES,
        "valid_acc32_values_per_design": valid_scalars,
        "trace_manifest": str(trace_manifest_path),
        "trace_manifest_sha256": sha256_file(trace_manifest_path),
        "base_vector": str(base_vector_path),
        "base_vector_sha256": sha256_file(base_vector_path),
        "run_context": trace_manifest.get("run_context", {}),
        "records": receipts,
        "vectors": vectors,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": [
            "This covers sample0/window0 all12 ep35 only, not multiple samples.",
            "Values are pre-bias partial Acc32; cross-head equality is checked by the generator.",
            "The checker is a numeric sidecar, not a throughput projection backend.",
            "This does not establish a full block, full encoder, or ASIC PPA result.",
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
        "PASS H67 real-weight all-output vectors "
        f"rows={result['rows']} batches={result['batches']} "
        f"acc32={result['valid_acc32_values_per_design']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
