#!/usr/bin/env python3
"""从H67真实位级trace生成FADC24 leaf decoder自检向量。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gatestack_fadc24_reference import (
    LANES,
    SLOT_BYTES,
    TOKENS,
    deserialize_fadc24,
    serialize_fadc24,
)
from generate_gatestack_h67_stage3_trace import words_from_payload, write_memh
from generate_gatestack_real_trace_vectors import build_terms, unpack_bits


def generate(
    manifest: dict,
    *,
    stage: int,
    head: int,
    window: int,
    output_dir: Path,
) -> dict:
    record = next(
        item for item in manifest["records"] if item["name"].startswith(f"S{stage}.")
    )
    with np.load(record["file"]) as source:
        shape = tuple(int(value) for value in source["k_shape"])
        k_bits = unpack_bits(source["k_bits_packed"], shape)
        gate = source["gate_q17"].astype(np.int64)
    _, windows, heads, spatial_tokens, lanes = shape
    if not (0 <= window < windows and 0 <= head < heads):
        raise ValueError("window/head越界")
    if spatial_tokens * 2 != TOKENS or lanes != LANES:
        raise ValueError("trace布局不支持")

    k_rows = k_bits[:, window].transpose(1, 0, 2, 3).reshape(heads, TOKENS, LANES)
    terms = build_terms(k_rows[head], gate[window, head])
    tag = 0xFA00_0000 | (stage << 8) | head
    payload = serialize_fadc24(terms, tag=tag)
    decoded = deserialize_fadc24(payload)
    if decoded["terms"] != terms:
        raise AssertionError("FADC24往返不等价")
    if len(payload) > SLOT_BYTES:
        raise ValueError("所选head的FADC24 payload无法装入物理槽")

    words = words_from_payload(payload)
    padded_words = words + [0] * (SLOT_BYTES // 8 - len(words))
    corrupted_padding = bytearray(payload)
    destination_cursor = 16 + 3 * len(terms)
    first_bitmap_offset = None
    for term in terms:
        if len(term["tokens"]) > 21:
            first_bitmap_offset = destination_cursor
            break
        destination_cursor += len(term["tokens"])
    if first_bitmap_offset is None:
        raise ValueError("所选head没有bitmap term，无法生成padding错误向量")
    corrupted_padding[first_bitmap_offset + 20] |= 1 << 2
    corrupted_words = words_from_payload(bytes(corrupted_padding))
    corrupted_words += [0] * (SLOT_BYTES // 8 - len(corrupted_words))
    events = [
        {"gate": term["gate"], "lane": term["lane"], "token": token}
        for term in terms
        for token in term["tokens"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_memh(output_dir / "payload_words.memh", padded_words, 64)
    write_memh(
        output_dir / "payload_words_bad_bitmap_padding.memh", corrupted_words, 64
    )
    write_memh(output_dir / "term_gate.memh", [term["gate"] for term in terms], 9)
    write_memh(output_dir / "term_lane.memh", [term["lane"] for term in terms], 5)
    write_memh(
        output_dir / "term_destination_count.memh",
        [len(term["tokens"]) for term in terms],
        8,
    )
    write_memh(
        output_dir / "event_gate.memh", [event["gate"] for event in events], 9
    )
    write_memh(
        output_dir / "event_lane.memh", [event["lane"] for event in events], 5
    )
    write_memh(
        output_dir / "event_token.memh", [event["token"] for event in events], 8
    )
    result = {
        "schema_version": 1,
        "source": record["file"],
        "stage": stage,
        "head": head,
        "window": window,
        "tag": tag,
        "payload_bytes": len(payload),
        "payload_bits": len(payload) * 8,
        "payload_words": len(words),
        "slot_words": SLOT_BYTES // 8,
        "terms": len(terms),
        "events": len(events),
        "bitmap_terms": decoded["bitmap_terms"],
        "max_fanout": max((len(term["tokens"]) for term in terms), default=0),
        "negative_vector": {
            "file": "payload_words_bad_bitmap_padding.memh",
            "fault": "首个bitmap term的token162 padding bit置1",
        },
        "vector_dir": str(output_dir),
        "contract": "真实K/gate；FADC24 encode/decode往返等价",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        stage=args.stage,
        head=args.head,
        window=args.window,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
