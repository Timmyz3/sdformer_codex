#!/usr/bin/env python3
"""生成四stage FADC24/RAW容量安全的同顶层RTL向量。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from gatestack_fadc24_reference import RAW_HEAD_BITS, serialize_fadc24
from generate_gatestack_h67_stage3_trace import (
    PAYLOAD_TAG_BASE,
    serialize_raw,
    words_from_payload,
    write_memh,
)
from generate_gatestack_real_trace_vectors import (
    LANES,
    TOKENS,
    WORDS_PER_HEAD,
    build_terms,
    unpack_bits,
)


COPY_FILES = (
    "projection_weights_int8.memh",
    "projection_weight_scale_exp2.memh",
    "projection_bias_acc.memh",
    "expected_output_acc32.memh",
)


def generate_record(
    record: dict[str, Any], *, source_vectors: Path, output_root: Path, window: int
) -> dict[str, Any]:
    with np.load(record["file"]) as payload:
        shape = tuple(int(value) for value in payload["k_shape"])
        k_bits = unpack_bits(payload["k_bits_packed"], shape)
        gate = payload["gate_q17"].astype(np.int64)
    _, windows, heads, spatial_tokens, lanes = shape
    if window >= windows or spatial_tokens * 2 != TOKENS or lanes != LANES:
        raise ValueError(f"trace布局不支持: {record['name']} {shape}")
    stage = int(record["name"].split(".")[0][1:])
    k_rows = k_bits[:, window].transpose(1, 0, 2, 3).reshape(heads, TOKENS, LANES)
    term_rows = [build_terms(k_rows[head], gate[window, head]) for head in range(heads)]
    vector_dir = output_root / f"fadc24_real_sample{record['sample_id']}_s{stage}_b0"
    vector_dir.mkdir(parents=True, exist_ok=True)

    payload_words = [0] * (heads * WORDS_PER_HEAD)
    payload_bits = []
    payload_modes = []
    payload_word_counts = []
    term_counts = []
    event_counts = []
    rows = []
    for head, terms in enumerate(term_rows):
        fadc_payload = serialize_fadc24(terms, tag=PAYLOAD_TAG_BASE + head)
        raw_payload, raw_bits = serialize_raw(terms)
        fadc_bits = len(fadc_payload) * 8
        mode_is_fadc = fadc_bits <= RAW_HEAD_BITS
        encoded = fadc_payload if mode_is_fadc else raw_payload
        bits = fadc_bits if mode_is_fadc else raw_bits
        words = words_from_payload(encoded)
        for index, word in enumerate(words):
            payload_words[head * WORDS_PER_HEAD + index] = word
        events = sum(len(term["tokens"]) for term in terms)
        payload_bits.append(bits)
        payload_modes.append(int(mode_is_fadc))
        payload_word_counts.append(len(words))
        term_counts.append(len(terms))
        event_counts.append(events)
        rows.append(
            {
                "head": head,
                "mode": "FADC24" if mode_is_fadc else "RAW41",
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
    source_dir = source_vectors / f"real_sample{record['sample_id']}_s{stage}_b0_capacity"
    for filename in COPY_FILES:
        shutil.copyfile(source_dir / filename, vector_dir / filename)

    result = {
        "name": record["name"],
        "sample_id": int(record["sample_id"]),
        "stage": stage,
        "window": window,
        "vector_dir": str(vector_dir),
        "heads": heads,
        "output_tiles": heads,
        "fadc_heads": sum(payload_modes),
        "raw_fallback_heads": heads - sum(payload_modes),
        "events": sum(event_counts),
        "ideal_terms": sum(term_counts),
        "expected_projection_terms": heads * sum(
            row["terms"] if row["mode"] == "FADC24" else row["events"]
            for row in rows
        ),
        "expected_slot_replays": heads * heads,
        "payload_words_all_tiles": heads * sum(payload_word_counts),
        "rows": rows,
    }
    (vector_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-vectors", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        generate_record(
            record,
            source_vectors=args.source_vectors,
            output_root=args.output_root,
            window=args.window,
        )
        for record in manifest["records"]
    ]
    result = {
        "schema_version": 1,
        "format": "FADC24 with exact RAW41 fallback",
        "source_manifest": str(args.manifest),
        "records": records,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
