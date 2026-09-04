#!/usr/bin/env python3
"""Generate checkpoint-bound H67 T450 score/Shiftmax row vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


LANES = 32
LUT_Q8 = (256, 245, 234, 224, 215, 205, 196, 188,
          181, 173, 165, 158, 152, 145, 139, 133)
NAME_RE = re.compile(r"^S(?P<stage>\d+)\.B(?P<block>\d+)\.attn$")
EXPECTED_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
EXPECTED_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
REPO = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_bits(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape)
    bits = np.unpackbits(packed, bitorder="little")
    if bits.size < count:
        raise ValueError(f"packed bit count {bits.size} < {count}")
    return bits[:count].reshape(shape).astype(np.bool_)


def bitmap32(bits: np.ndarray) -> int:
    if bits.shape != (LANES,):
        raise ValueError(f"expected {LANES} lanes, got {bits.shape}")
    value = 0
    for lane in np.flatnonzero(bits):
        value |= 1 << int(lane)
    return value


def round_even_silence(count: int, integer_base: int) -> int:
    quotient, remainder = divmod(count, 16)
    if remainder > 8 or (remainder == 8 and (integer_base + quotient) & 1):
        quotient += 1
    return quotient


def score_q7(q: int, current_k: int, peer_k: int) -> int:
    mask = (1 << LANES) - 1
    overlap = (q & current_k).bit_count()
    same_zero = ((~q) & (~current_k) & mask).bit_count()
    motion = (current_k ^ peer_k).bit_count()
    integer_base = 4 * overlap + motion
    return integer_base + round_even_silence(same_zero, integer_base)


def exp2_q8(delta_q7: int) -> int:
    if delta_q7 >= 0:
        return 256
    absolute = -delta_q7
    integer_shift = min(absolute >> 7, 8)
    fraction = absolute & 127
    fraction_index = min((fraction + 7) // 8, 15)
    return LUT_Q8[fraction_index] >> integer_shift


def round_shift_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    quotient = value >> shift
    remainder = value - (quotient << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return quotient


def row_gate_codes(q: list[int], current: list[int], peer: list[int]) -> list[int]:
    tokens = len(q)
    scores = [score_q7(q[i], current[i], peer[i]) for i in range(tokens)]
    row_max = max(scores)
    exponentials = [exp2_q8(score - row_max) for score in scores]
    row_sum = sum(exponentials)
    denominator_shift = max(row_sum - 1, 0).bit_length()
    return [
        min(round_shift_even(exp_value * tokens * 128, denominator_shift), 256)
        for exp_value in exponentials
    ]


def validate_run_context(manifest: dict[str, Any], expected_tokens: int) -> dict[str, Any]:
    context = manifest.get("run_context")
    if not isinstance(context, dict) or not context:
        raise ValueError("trace manifest is not bound to a run_context")
    identity = context.get("artifact_identity", {})
    for key in ("config_sha256", "checkpoint_sha256", "checkpoint_path"):
        if not identity.get(key):
            raise ValueError(f"trace run_context missing artifact_identity.{key}")
    config_path = Path(identity["config_path"])
    checkpoint_path = Path(identity["checkpoint_path"])
    if file_sha256(config_path) != identity["config_sha256"]:
        raise ValueError("bound config SHA256 no longer matches the file")
    if file_sha256(checkpoint_path) != identity["checkpoint_sha256"]:
        raise ValueError("bound checkpoint SHA256 no longer matches the file")
    protocol = context.get("eval_protocol", {})
    if protocol.get("resolution") != [480, 640]:
        raise ValueError(f"expected full resolution 480x640, got {protocol}")
    if protocol.get("window_size") != [2, 15, 15]:
        raise ValueError(f"expected window 2x15x15, got {protocol}")
    if protocol.get("tokens_per_window") != expected_tokens:
        raise ValueError("trace token count differs from bound eval protocol")
    if protocol.get("bn_policy") != "no_running":
        raise ValueError("checkpoint-bound H67 trace requires BN no_running")
    counts = context.get("module_counts", {})
    if counts.get("ATLIFTernaryPSN") != 105 or counts.get("ShiftmaxAttention") != 12:
        raise ValueError(f"unexpected H67 module counts: {counts}")
    load = context.get("checkpoint_load_audit", {})
    required_zero = (
        "missing_count",
        "unexpected_count",
        "overlay_missing_count",
        "overlay_unexpected_count",
    )
    if any(int(load.get(key, -1)) != 0 for key in required_zero):
        raise ValueError(f"checkpoint load audit is not exact: {load}")
    if int(load.get("checkpoint_overlay_keys", -1)) != 210:
        raise ValueError(f"unexpected checkpoint overlay count: {load}")
    if int(load.get("model_overlay_keys", -1)) != 210:
        raise ValueError(f"unexpected model overlay count: {load}")
    source_sha = context.get("source_sha256", {})
    source_paths = {
        "profiler": REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
        "trace_writer": REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_bit_trace.py",
    }
    for name, path in source_paths.items():
        if source_sha.get(name) != file_sha256(path):
            raise ValueError(f"bound source SHA256 mismatch: {name}")
    return context


def parse_record(record: dict[str, Any], expected_tokens: int) -> tuple[dict[str, Any], list[dict[str, int]]]:
    match = NAME_RE.fullmatch(str(record.get("name", "")))
    if match is None:
        raise ValueError(f"invalid attention name: {record.get('name')}")
    stage = int(match.group("stage"))
    block = int(match.group("block"))
    source = Path(record["file"])
    digest = file_sha256(source)
    if digest != record.get("sha256"):
        raise ValueError(f"trace SHA256 mismatch: {source}")
    with np.load(source) as payload:
        q_shape = tuple(int(value) for value in payload["q_shape"])
        k_shape = tuple(int(value) for value in payload["k_shape"])
        q_bits = unpack_bits(payload["q_bits_packed"], q_shape)
        k_bits = unpack_bits(payload["k_bits_packed"], k_shape)
        gates = payload["gate_q17"].astype(np.int64)
    if q_shape != k_shape or len(q_shape) != 5:
        raise ValueError(f"Q/K trace shape mismatch: {q_shape} {k_shape}")
    time_steps, windows, heads, spatial_tokens, lanes = q_shape
    if time_steps != 2 or windows < 1 or lanes != LANES:
        raise ValueError(f"unsupported H67 trace shape: {q_shape}")
    if 2 * spatial_tokens != expected_tokens:
        raise ValueError(f"expected T{expected_tokens}, got T{2 * spatial_tokens}")
    if heads != EXPECTED_HEADS.get(stage):
        raise ValueError(f"unexpected S{stage} head count {heads}")
    if gates.shape != (windows, heads, expected_tokens):
        raise ValueError(f"gate trace shape mismatch: {gates.shape}")

    rows: list[dict[str, Any]] = []
    for head in range(heads):
        q_row: list[int] = []
        current_row: list[int] = []
        peer_row: list[int] = []
        for time_index in range(2):
            peer_index = 1 - time_index
            for token in range(spatial_tokens):
                q_row.append(bitmap32(q_bits[time_index, 0, head, token]))
                current_row.append(bitmap32(k_bits[time_index, 0, head, token]))
                peer_row.append(bitmap32(k_bits[peer_index, 0, head, token]))
        reference = row_gate_codes(q_row, current_row, peer_row)
        recorded = [int(value) for value in gates[0, head]]
        if reference != recorded:
            mismatch = next(
                index for index, (lhs, rhs) in enumerate(zip(reference, recorded))
                if lhs != rhs
            )
            raise ValueError(
                f"independent gate mismatch {record['name']} head={head} "
                f"token={mismatch}: reference={reference[mismatch]} "
                f"trace={recorded[mismatch]}"
            )
        vectors = [
            {
                "q": q_row[index],
                "current_k": current_row[index],
                "peer_k": peer_row[index],
                "gate": recorded[index],
            }
            for index in range(expected_tokens)
        ]
        rows.append(
            {
                "stage": stage,
                "block": block,
                "head": head,
                "expected_outputs": sum(value != 0 for value in current_row),
                "expected_folded": sum(value == 0 for value in current_row),
                "vectors": vectors,
            }
        )
    summary = {
        "name": record["name"],
        "source": str(source),
        "source_sha256": digest,
        "stage": stage,
        "block": block,
        "heads": heads,
        "rows": len(rows),
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tokens", type=int, default=450)
    parser.add_argument("--allow-unbound", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="tooling test only: accept an ordered subset instead of all12",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    expected_names = [
        f"S{stage}.B{block}.attn"
        for stage, depth in EXPECTED_BLOCKS.items()
        for block in range(depth)
    ]
    names = [str(record.get("name")) for record in records]
    if not args.allow_partial and names != expected_names:
        raise ValueError(f"all12 attention coverage/order mismatch: {names}")
    if args.allow_partial and (
        not names
        or len(names) != len(set(names))
        or any(name not in expected_names for name in names)
        or names != sorted(names, key=expected_names.index)
    ):
        raise ValueError(f"partial attention coverage/order mismatch: {names}")
    context = (
        manifest.get("run_context", {})
        if args.allow_unbound
        else validate_run_context(manifest, args.expected_tokens)
    )

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        summary, record_rows = parse_record(record, args.expected_tokens)
        summaries.append(summary)
        rows.extend(record_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / "h67_checkpoint_rows.txt"
    with vector_path.open("w", encoding="ascii") as handle:
        handle.write(f"{len(rows)} {args.expected_tokens}\n")
        for row_index, row in enumerate(rows):
            handle.write(
                f"{row_index} {row['stage']} {row['block']} {row['head']} "
                f"{row['expected_outputs']} {row['expected_folded']}\n"
            )
            for vector in row["vectors"]:
                handle.write(
                    f"{vector['q']:08x} {vector['current_k']:08x} "
                    f"{vector['peer_k']:08x} {vector['gate']}\n"
                )

    output = {
        "schema": "h67_checkpoint_t450_score_shiftmax_vectors_v1",
        "scope": "checkpoint_bound_qk_score_scs_shiftmax_not_projection_or_full_network",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.manifest),
        "run_context": context,
        "tokens_per_row": args.expected_tokens,
        "row_count": len(rows),
        "token_vector_count": len(rows) * args.expected_tokens,
        "expected_active_outputs": sum(int(row["expected_outputs"]) for row in rows),
        "expected_folded_tokens": sum(int(row["expected_folded"]) for row in rows),
        "records": summaries,
        "vector_file": str(vector_path.resolve()),
        "vector_sha256": file_sha256(vector_path),
        "independent_reference_matches_trace": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
