#!/usr/bin/env python3
"""生成四stage全部45个head的完整C0 builder RTL向量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gatestack_fadc24_reference import _record_rows, serialize_fadc24
from generate_gatestack_h67_stage3_trace import (
    serialize_ipd,
    serialize_raw,
    words_from_payload,
    write_memh,
)
from generate_gatestack_typed_serializer_vectors import reconstruct_raw_records


SLOT_BYTES = 832
FORMAT = {"RAW41": 0, "IPD32W": 1, "FADC24": 2}
REASON = {
    "ipd_fit": 0,
    "fadc_ipd_class": 1,
    "fadc_ipd_capacity": 2,
    "raw_fadc_capacity": 3,
    "raw_metadata_overflow": 4,
}


def select_format(terms: list[dict[str, Any]]) -> tuple[str, str]:
    active_classes = len({int(term["gate"]) for term in terms})
    events = sum(len(term["tokens"]) for term in terms)
    ipd_bytes = 16 + 8 * ((len(terms) + 1) // 2) + events
    fadc_bytes = 16 + 3 * len(terms) + sum(
        min(len(term["tokens"]), 21) for term in terms
    )
    if active_classes > 4:
        return "RAW41", "raw_metadata_overflow"
    if ipd_bytes <= SLOT_BYTES:
        return "IPD32W", "ipd_fit"
    if fadc_bytes <= SLOT_BYTES:
        return "FADC24", "fadc_ipd_capacity"
    return "RAW41", "raw_fadc_capacity"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.trace_manifest.read_text(encoding="utf-8"))
    records = sorted(source["records"], key=lambda row: int(row["stage"]))
    raw_records: list[int] = []
    expected_words: list[int] = []
    tags: list[int] = []
    stages: list[int] = []
    heads: list[int] = []
    formats: list[int] = []
    reasons: list[int] = []
    payload_bits_rows: list[int] = []
    word_counts: list[int] = []
    word_offsets: list[int] = []
    term_counts: list[int] = []
    event_counts: list[int] = []
    rows: list[dict[str, Any]] = []

    for record in records:
        stage = int(record["stage"])
        source_record = {"name": record["name"], "file": record["source"]}
        term_rows = _record_rows(source_record, int(record["window"]))
        for head, terms in enumerate(term_rows):
            tag = 0x6A00_0000 | (stage << 16) | head
            format_name, reason_name = select_format(terms)
            if format_name == "IPD32W":
                payload, payload_bits = serialize_ipd(terms, tag)
            elif format_name == "FADC24":
                payload = serialize_fadc24(terms, tag=tag)
                payload_bits = len(payload) * 8
            else:
                payload, payload_bits = serialize_raw(terms)
            words = words_from_payload(payload)
            events = sum(len(term["tokens"]) for term in terms)
            active_classes = len({int(term["gate"]) for term in terms})
            active_tokens = len(
                {int(token) for term in terms for token in term["tokens"]}
            )
            bitmap_terms = sum(len(term["tokens"]) > 21 for term in terms)
            word_offset = len(expected_words)
            raw_offset = len(raw_records)
            raw_records.extend(reconstruct_raw_records(terms))
            expected_words.extend(words)
            tags.append(tag)
            stages.append(stage)
            heads.append(head)
            formats.append(FORMAT[format_name])
            reasons.append(REASON[reason_name])
            payload_bits_rows.append(payload_bits)
            word_counts.append(len(words))
            word_offsets.append(word_offset)
            term_counts.append(len(terms))
            event_counts.append(events)
            rows.append(
                {
                    "index": len(rows),
                    "stage": stage,
                    "head": head,
                    "tag": tag,
                    "format": format_name,
                    "reason": reason_name,
                    "terms": len(terms),
                    "events": events,
                    "active_classes": active_classes,
                    "active_tokens": active_tokens,
                    "bitmap_terms": bitmap_terms,
                    "payload_bits": payload_bits,
                    "word_count": len(words),
                    "raw_offset": raw_offset,
                    "word_offset": word_offset,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_memh(args.output_root / "raw_records.memh", raw_records, 41)
    write_memh(args.output_root / "expected_words.memh", expected_words, 64)
    write_memh(args.output_root / "tags.memh", tags, 32)
    write_memh(args.output_root / "stages.memh", stages, 2)
    write_memh(args.output_root / "heads.memh", heads, 5)
    write_memh(args.output_root / "formats.memh", formats, 2)
    write_memh(args.output_root / "reasons.memh", reasons, 3)
    write_memh(args.output_root / "payload_bits.memh", payload_bits_rows, 16)
    write_memh(args.output_root / "word_counts.memh", word_counts, 8)
    write_memh(args.output_root / "word_offsets.memh", word_offsets, 16)
    write_memh(args.output_root / "term_counts.memh", term_counts, 8)
    write_memh(args.output_root / "event_counts.memh", event_counts, 13)
    manifest = {
        "schema_version": 1,
        "source_manifest": str(args.trace_manifest.resolve()),
        "head_count": len(rows),
        "raw_record_count": len(raw_records),
        "expected_word_count": len(expected_words),
        "format_counts": {
            name: sum(row["format"] == name for row in rows) for name in FORMAT
        },
        "rows": rows,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
