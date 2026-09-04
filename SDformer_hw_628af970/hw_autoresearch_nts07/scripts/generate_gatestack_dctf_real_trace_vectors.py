#!/usr/bin/env python3
"""将H67真实S0-S3 trace生成DCTF96 projection-only RTL回放向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from generate_gatestack_h67_stage3_trace import write_memh
from generate_gatestack_real_trace_vectors import (
    LANES,
    TOKENS,
    build_terms,
    signed_to_bits,
    unpack_bits,
)


EXPECTED_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
EXPECTED_ALL12_NAMES = {
    *(f"S0.B{block}.attn" for block in range(2)),
    *(f"S1.B{block}.attn" for block in range(2)),
    *(f"S2.B{block}.attn" for block in range(6)),
    *(f"S3.B{block}.attn" for block in range(2)),
}
METADATA_SCHEMA_VERSION = 1
MAX_DESTINATIONS_PER_TERM = 255


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_term_stream(
    term_rows: list[list[dict[str, Any]]],
) -> dict[str, list[int]]:
    head_offsets = [0]
    token_offsets = [0]
    gates: list[int] = []
    lanes: list[int] = []
    counts: list[int] = []
    tokens: list[int] = []
    for terms in term_rows:
        for term in terms:
            term_tokens = [int(token) for token in term["tokens"]]
            if not term_tokens or len(term_tokens) > MAX_DESTINATIONS_PER_TERM:
                raise ValueError("term destination count超出DCTF接口范围")
            gates.append(int(term["gate"]))
            lanes.append(int(term["lane"]))
            counts.append(len(term_tokens))
            tokens.extend(term_tokens)
            token_offsets.append(len(tokens))
        head_offsets.append(len(gates))
    return {
        "head_offsets": head_offsets,
        "token_offsets": token_offsets,
        "gates": gates,
        "lanes": lanes,
        "counts": counts,
        "tokens": tokens,
    }


def split_terms_by_destination_limit(
    terms: list[dict[str, Any]], limit: int = MAX_DESTINATIONS_PER_TERM
) -> list[dict[str, Any]]:
    if limit <= 0 or limit > 255:
        raise ValueError("term destination limit必须在1..255")
    result: list[dict[str, Any]] = []
    for term in terms:
        tokens = [int(token) for token in term["tokens"]]
        for offset in range(0, len(tokens), limit):
            result.append(
                {
                    "gate": int(term["gate"]),
                    "lane": int(term["lane"]),
                    "tokens": tokens[offset : offset + limit],
                }
            )
    return result


def projection_reference(
    k_rows: np.ndarray,
    gate_rows: np.ndarray,
    weight: np.ndarray,
    bias_acc: np.ndarray,
) -> np.ndarray:
    heads, tokens, lanes = k_rows.shape
    dim = heads * lanes
    if tokens <= 0 or lanes != LANES:
        raise ValueError("K trace布局不符合动态token/32 lane合同")
    if gate_rows.shape != (heads, tokens):
        raise ValueError("gate trace布局错误")
    if weight.shape != (dim, dim) or bias_acc.shape != (dim,):
        raise ValueError("projection weight/bias维度错误")

    expected = np.zeros((tokens, dim), dtype=np.int64)
    for head in range(heads):
        input_base = head * LANES
        for token in range(tokens):
            active_lanes = np.flatnonzero(k_rows[head, token])
            if active_lanes.size:
                input_channels = input_base + active_lanes
                expected[token] += (
                    int(gate_rows[head, token])
                    * weight[:, input_channels].sum(axis=1, dtype=np.int64)
                )
    expected += bias_acc[None, :]
    if expected.min(initial=0) < -(1 << 31) or expected.max(initial=0) >= (1 << 31):
        raise ValueError("projection整数金参考超出acc32")
    return expected


def record_vector_name(record: dict[str, Any], *, disambiguate_block: bool) -> str:
    match = re.fullmatch(r"S(?P<stage>\d+)\.B(?P<block>\d+)\.attn", str(record["name"]))
    if match is None:
        raise ValueError(f"无法解析attention记录名: {record['name']}")
    stage = int(match.group("stage"))
    block = int(match.group("block"))
    return f"s{stage}_b{block}" if disambiguate_block else f"s{stage}"


def generate_record(
    record: dict[str, Any],
    output_root: Path,
    *,
    window: int = 0,
    vector_name: str | None = None,
) -> dict[str, Any]:
    stage = int(str(record["name"]).split(".")[0][1:])
    if stage not in EXPECTED_HEADS:
        raise ValueError(f"不支持的stage: {record['name']}")
    source = Path(record["file"])
    if sha256_file(source) != str(record["sha256"]):
        raise ValueError(f"源trace SHA256不匹配: {source}")

    with np.load(source) as payload:
        k_shape = tuple(int(value) for value in payload["k_shape"])
        k_bits = unpack_bits(payload["k_bits_packed"], k_shape)
        gate = payload["gate_q17"].astype(np.int64)
        weight = payload["projection_weight_int8"].astype(np.int64)
        weight_exp = payload["projection_weight_scale_exp2"].astype(np.int64)
        bias_acc = payload["projection_bias_acc_int64"].astype(np.int64)

    _, windows, heads, spatial_tokens, lanes = k_shape
    if window < 0 or window >= windows:
        raise ValueError(f"window越界: {window}/{windows}")
    if heads != EXPECTED_HEADS[stage]:
        raise ValueError(f"S{stage} head数应为{EXPECTED_HEADS[stage]}，实际为{heads}")
    temporal_tokens = spatial_tokens * 2
    if temporal_tokens <= 0 or lanes != LANES:
        raise ValueError(f"trace布局不支持: {record['name']} {k_shape}")
    if gate.shape != (windows, heads, temporal_tokens):
        raise ValueError(f"gate布局不支持: {record['name']} {gate.shape}")

    dim = heads * LANES
    k_rows = k_bits[:, window].transpose(1, 0, 2, 3).reshape(
        heads, temporal_tokens, LANES
    )
    gate_rows = gate[window]
    term_rows = [
        split_terms_by_destination_limit(build_terms(k_rows[head], gate_rows[head]))
        for head in range(heads)
    ]
    stream = flatten_term_stream(term_rows)
    expected = projection_reference(k_rows, gate_rows, weight, bias_acc)

    vector_dir = output_root / (vector_name or f"s{stage}")
    vector_dir.mkdir(parents=True, exist_ok=True)
    total_terms = len(stream["gates"])
    total_events = len(stream["tokens"])
    supertiles = heads // 3
    metadata = [
        METADATA_SCHEMA_VERSION,
        stage,
        heads,
        supertiles,
        dim,
        total_terms,
        total_events,
        temporal_tokens,
        LANES,
    ]
    write_memh(vector_dir / "metadata.memh", metadata, 32)
    write_memh(vector_dir / "head_term_offsets.memh", stream["head_offsets"], 32)
    write_memh(vector_dir / "term_token_offsets.memh", stream["token_offsets"], 32)
    write_memh(vector_dir / "term_gate_codes.memh", stream["gates"], 9)
    write_memh(vector_dir / "term_lane_ids.memh", stream["lanes"], 5)
    write_memh(vector_dir / "term_destination_counts.memh", stream["counts"], 8)
    token_id_width = max(1, (temporal_tokens - 1).bit_length())
    write_memh(
        vector_dir / "term_tokens.memh", stream["tokens"], token_id_width
    )
    write_memh(
        vector_dir / "projection_weights_int8.memh",
        [signed_to_bits(value, 8) for value in weight.reshape(-1)],
        8,
    )
    write_memh(
        vector_dir / "projection_weight_scale_exp2.memh",
        [signed_to_bits(value, 16) for value in weight_exp.reshape(-1)],
        16,
    )
    write_memh(
        vector_dir / "projection_bias_acc32.memh",
        [signed_to_bits(value, 32) for value in bias_acc],
        32,
    )
    write_memh(
        vector_dir / "expected_output_acc32.memh",
        [signed_to_bits(value, 32) for value in expected.reshape(-1)],
        32,
    )

    files = {}
    for path in sorted(vector_dir.glob("*.memh")):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    rows = [
        {
            "head": head,
            "terms": len(terms),
            "events": sum(len(term["tokens"]) for term in terms),
        }
        for head, terms in enumerate(term_rows)
    ]
    result = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "name": record["name"],
        "sample_id": int(record["sample_id"]),
        "stage": stage,
        "window": window,
        "source": str(source),
        "source_sha256": record["sha256"],
        "quantization_contract": record["quantization_contract"],
        "vector_dir": str(vector_dir),
        "heads": heads,
        "logical_supertiles": supertiles,
        "dim": dim,
        "tokens": temporal_tokens,
        "token_id_width": token_id_width,
        "lanes": LANES,
        "terms_per_full_input_replay": total_terms,
        "events_per_full_input_replay": total_events,
        "expected_replayed_heads": heads * supertiles,
        "expected_issued_terms": total_terms * supertiles,
        "expected_physical_weight_requests": total_terms * supertiles * 3,
        "expected_bias_requests": temporal_tokens * supertiles * 3,
        "expected_final_beats": temporal_tokens * heads,
        "expected_final_checks": temporal_tokens * dim,
        "expected_output_min": int(expected.min(initial=0)),
        "expected_output_max": int(expected.max(initial=0)),
        "rows": rows,
        "files": files,
    }
    (vector_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_records = manifest["records"]
    names = [str(record["name"]) for record in source_records]
    if len(names) != len(set(names)):
        raise ValueError("输入manifest存在重复attention record")
    if len(names) == 12 and set(names) != EXPECTED_ALL12_NAMES:
        raise ValueError("all12 manifest的stage/block集合不完整")
    stage_counts: dict[int, int] = {}
    for source_record in source_records:
        stage = int(str(source_record["name"]).split(".")[0][1:])
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    records = [
        generate_record(
            record,
            args.output_root,
            window=args.window,
            vector_name=record_vector_name(
                record,
                disambiguate_block=stage_counts[
                    int(str(record["name"]).split(".")[0][1:])
                ]
                > 1,
            ),
        )
        for record in source_records
    ]
    if sorted({record["stage"] for record in records}) != [0, 1, 2, 3]:
        raise ValueError("输入manifest必须覆盖S0-S3")
    temporal_token_counts = sorted({int(record["tokens"]) for record in records})
    if len(temporal_token_counts) != 1:
        raise ValueError(f"attention records的token数不一致: {temporal_token_counts}")
    result = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "说明": "H67真实all-attention DCTF96 projection-only RTL回放向量",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "source_run_context": manifest.get("run_context", {}),
        "window": args.window,
        "temporal_tokens": temporal_token_counts[0],
        "token_id_width": max(int(record["token_id_width"]) for record in records),
        "records": records,
        "限制": [
            "Q/K/gate来自H67真实推理trace",
            "INT8权重与acc32 bias沿用源manifest候选量化合同",
            "本向量只验证projection execution slice，不代表完整encoder",
        ],
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
