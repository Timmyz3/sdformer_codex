#!/usr/bin/env python3
"""将H67真实位级trace转换为GateStack RTL向量与整数金参考。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from generate_gatestack_h67_stage3_trace import (
    PAYLOAD_TAG_BASE,
    serialize_ipd,
    serialize_raw,
    words_from_payload,
    write_memh,
)


TOKENS = 162
LANES = 32
WORDS_PER_HEAD = 104


def unpack_bits(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape)
    return np.unpackbits(packed, bitorder="little")[:count].reshape(shape).astype(bool)


def build_terms(k_row: np.ndarray, gate_row: np.ndarray) -> list[dict[str, Any]]:
    terms = []
    for gate_code in sorted(int(value) for value in np.unique(gate_row)):
        gate_mask = gate_row == gate_code
        for lane in range(k_row.shape[1]):
            tokens = np.flatnonzero(gate_mask & k_row[:, lane]).astype(int).tolist()
            if tokens:
                terms.append({"gate": gate_code, "lane": lane, "tokens": tokens})
    return terms


def signed_to_bits(value: int, width: int) -> int:
    return int(value) & ((1 << width) - 1)


def generate_record(
    record: dict[str, Any], output_root: Path, *, window: int = 0
) -> dict[str, Any]:
    source = Path(record["file"])
    with np.load(source) as payload:
        k_shape = tuple(int(value) for value in payload["k_shape"])
        k_bits = unpack_bits(payload["k_bits_packed"], k_shape)
        gate = payload["gate_q17"].astype(np.int64)
        weight = payload["projection_weight_int8"].astype(np.int64)
        weight_exp = payload["projection_weight_scale_exp2"].astype(np.int64)
        bias_acc = payload["projection_bias_acc_int64"].astype(np.int64)
    _, windows, heads, spatial_tokens, lanes = k_shape
    if window >= windows or spatial_tokens * 2 != TOKENS or lanes != LANES:
        raise ValueError(f"trace布局不支持: {record['name']} {k_shape}")
    dim = heads * lanes
    if weight.shape != (dim, dim) or bias_acc.shape != (dim,):
        raise ValueError("weight/bias维度错误")
    k_rows = k_bits[:, window].transpose(1, 0, 2, 3).reshape(heads, TOKENS, LANES)
    gate_rows = gate[window]
    term_rows = [build_terms(k_rows[head], gate_rows[head]) for head in range(heads)]

    expected = np.zeros((TOKENS, dim), dtype=np.int64)
    for head in range(heads):
        input_base = head * LANES
        for token in range(TOKENS):
            active_lanes = np.flatnonzero(k_rows[head, token])
            if active_lanes.size:
                input_channels = input_base + active_lanes
                expected[token] += int(gate_rows[head, token]) * weight[:, input_channels].sum(axis=1)
    expected += bias_acc[None, :]
    if expected.min(initial=0) < -(1 << 31) or expected.max(initial=0) >= (1 << 31):
        raise ValueError("整数金参考超出ACC_W=32")

    stage = int(record["name"].split(".")[0][1:])
    summaries = {}
    for force_raw, suffix in ((False, "capacity"), (True, "rawonly")):
        vector_dir = output_root / f"real_sample{record['sample_id']}_s{stage}_b0_{suffix}"
        vector_dir.mkdir(parents=True, exist_ok=True)
        payload_words = [0] * (heads * WORDS_PER_HEAD)
        payload_bits = []
        payload_modes = []
        payload_word_counts = []
        term_counts = []
        event_counts = []
        rows = []
        for head, terms in enumerate(term_rows):
            ipd_payload, ipd_bits = serialize_ipd(terms, PAYLOAD_TAG_BASE + head)
            raw_payload, raw_bits = serialize_raw(terms)
            mode_is_csr = ipd_bits <= raw_bits and not force_raw
            encoded, bits = (
                (ipd_payload, ipd_bits) if mode_is_csr else (raw_payload, raw_bits)
            )
            words = words_from_payload(encoded)
            for index, word in enumerate(words):
                payload_words[head * WORDS_PER_HEAD + index] = word
            events = sum(len(term["tokens"]) for term in terms)
            payload_bits.append(bits)
            payload_modes.append(int(mode_is_csr))
            payload_word_counts.append(len(words))
            term_counts.append(len(terms))
            event_counts.append(events)
            rows.append(
                {
                    "head": head,
                    "mode": "IPD32W" if mode_is_csr else "RAW41",
                    "terms": len(terms),
                    "events": events,
                    "payload_bits": bits,
                    "payload_words": len(words),
                }
            )
        write_memh(vector_dir / "payload_words.memh", payload_words, 64)
        write_memh(vector_dir / "payload_bits.memh", payload_bits, 16)
        write_memh(vector_dir / "payload_modes.memh", payload_modes, 1)
        write_memh(vector_dir / "payload_word_counts.memh", payload_word_counts, 8)
        write_memh(vector_dir / "term_counts.memh", term_counts, 8)
        write_memh(vector_dir / "event_counts.memh", event_counts, 13)
        write_memh(
            vector_dir / "projection_weights_int8.memh",
            [signed_to_bits(value, 8) for value in weight.reshape(-1)],
            8,
        )
        write_memh(
            vector_dir / "projection_weight_scale_exp2.memh",
            [signed_to_bits(value, 16) for value in weight_exp],
            16,
        )
        write_memh(
            vector_dir / "projection_bias_acc.memh",
            [signed_to_bits(value, 32) for value in bias_acc],
            32,
        )
        write_memh(
            vector_dir / "expected_output_acc32.memh",
            [signed_to_bits(value, 32) for value in expected.reshape(-1)],
            32,
        )
        csr_heads = sum(payload_modes)
        raw_heads = heads - csr_heads
        nonempty_csr = sum(
            mode and terms != 0
            for mode, terms in zip(payload_modes, term_counts)
        )
        full_words = sum(payload_word_counts)
        warm_words = sum(
            row["payload_words"]
            if row["mode"] == "RAW41"
            else max(row["payload_words"] - (2 + (row["terms"] + 1) // 2), 0)
            for row in rows
        )
        summaries[suffix] = {
            "vector_dir": str(vector_dir),
            "force_raw": force_raw,
            "heads": heads,
            "output_tiles": heads,
            "dim": dim,
            "csr_heads": csr_heads,
            "raw_heads": raw_heads,
            "nonempty_csr_heads": nonempty_csr,
            "events": sum(event_counts),
            "terms": sum(term_counts),
            "expected_projection_terms": heads
            * sum(
                row["terms"] if row["mode"] == "IPD32W" else row["events"]
                for row in rows
            ),
            "expected_slot_replays_resident": heads
            + (heads - 1) * (nonempty_csr + raw_heads),
            "expected_slot_replays_no_residency": heads * heads,
            "expected_cache_hits": csr_heads * (heads - 1),
            "expected_cache_releases": csr_heads,
            "payload_words_cold": full_words,
            "payload_words_all_tiles_resident": full_words + (heads - 1) * warm_words,
            "payload_words_all_tiles_no_residency": heads * full_words,
            "expected_output_min": int(expected.min(initial=0)),
            "expected_output_max": int(expected.max(initial=0)),
            "rows": rows,
        }
    return {
        "name": record["name"],
        "sample_id": int(record["sample_id"]),
        "stage": stage,
        "window": window,
        "source": str(source),
        "quantization_contract": record["quantization_contract"],
        "modes": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        generate_record(record, args.output_root, window=args.window)
        for record in manifest["records"]
    ]
    result = {
        "schema_version": 1,
        "evidence": "[H67真实Q/K/gate]+[checkpoint权重]+[候选dyadic INT8]",
        "source_manifest": str(args.manifest),
        "records": records,
        "limits": [
            "Q/K/gate来自真实H67推理",
            "权重INT8与bias accumulator是候选量化合同，尚未通过valid825",
            "向量用于projection execution slice，不代表完整encoder",
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
