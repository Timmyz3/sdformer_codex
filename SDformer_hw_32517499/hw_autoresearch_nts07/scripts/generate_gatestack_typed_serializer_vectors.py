#!/usr/bin/env python3
"""从H67真实位级trace生成三格式serializer逐word金参考。"""

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


TOKENS = 162
LANES = 32
FORMAT = {"RAW41": 0, "IPD32W": 1, "FADC24": 2}


def reconstruct_raw_records(terms: list[dict[str, Any]]) -> list[int]:
    gates = [0] * TOKENS
    k_bits = [0] * TOKENS
    for term in terms:
        gate = int(term["gate"])
        lane = int(term["lane"])
        for token_value in term["tokens"]:
            token = int(token_value)
            if gates[token] not in (0, gate):
                raise ValueError("同一token出现不同gate")
            gates[token] = gate
            k_bits[token] |= 1 << lane
    return [k_bits[token] | (gates[token] << LANES) for token in range(TOKENS)]


def write_case(
    output_root: Path,
    *,
    name: str,
    terms: list[dict[str, Any]],
    format_name: str,
    tag: int,
) -> dict[str, Any]:
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    events = sum(len(term["tokens"]) for term in terms)
    active_classes = len({int(term["gate"]) for term in terms})
    active_tokens = len({int(token) for term in terms for token in term["tokens"]})
    bitmap_terms = sum(len(term["tokens"]) > 21 for term in terms)
    fadc_destination_bytes = sum(min(len(term["tokens"]), 21) for term in terms)

    if format_name == "IPD32W":
        payload, payload_bits = serialize_ipd(terms, tag)
    elif format_name == "FADC24":
        payload = serialize_fadc24(terms, tag=tag)
        payload_bits = len(payload) * 8
    elif format_name == "RAW41":
        payload, payload_bits = serialize_raw(terms)
    else:
        raise ValueError(format_name)

    descriptor_values = []
    destination_values = []
    for term in terms:
        count = len(term["tokens"])
        descriptor_values.append(
            int(term["gate"]) | (int(term["lane"]) << 9) | (count << 14)
        )
        destination_values.extend(int(token) for token in term["tokens"])

    write_memh(output_dir / "descriptors.memh", descriptor_values or [0], 24)
    write_memh(output_dir / "destinations.memh", destination_values or [0], 8)
    write_memh(output_dir / "raw_records.memh", reconstruct_raw_records(terms), 41)
    expected_words = words_from_payload(payload)
    write_memh(output_dir / "expected_words.memh", expected_words, 64)

    result = {
        "name": name,
        "format_name": format_name,
        "format": FORMAT[format_name],
        "tag": tag,
        "active_classes": active_classes,
        "active_tokens": active_tokens,
        "term_count": len(terms),
        "event_count": events,
        "bitmap_term_count": bitmap_terms,
        "fadc_destination_bytes": fadc_destination_bytes,
        "payload_bits": payload_bits,
        "word_count": len(expected_words),
        "descriptor_count": len(descriptor_values),
        "destination_count": len(destination_values),
        "sha256_inputs": "generated from source trace; top-level provenance recorded in manifest",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.trace_manifest.read_text(encoding="utf-8"))
    by_stage: dict[int, list[list[dict[str, Any]]]] = {}
    for record in manifest["records"]:
        source_record = {"name": record["name"], "file": record["source"]}
        by_stage[int(record["stage"])] = _record_rows(
            source_record, int(record["window"])
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        write_case(
            args.output_root,
            name="ipd_s0_h0",
            terms=by_stage[0][0],
            format_name="IPD32W",
            tag=0x6900_0000,
        ),
        write_case(
            args.output_root,
            name="fadc_s3_h4",
            terms=by_stage[3][4],
            format_name="FADC24",
            tag=0x6903_0004,
        ),
        write_case(
            args.output_root,
            name="raw_s0_h0",
            terms=by_stage[0][0],
            format_name="RAW41",
            tag=0x69F0_0000,
        ),
        write_case(
            args.output_root,
            name="ipd_empty_s1_h0",
            terms=by_stage[1][0],
            format_name="IPD32W",
            tag=0x6901_0000,
        ),
        write_case(
            args.output_root,
            name="raw_class_overflow_synth",
            terms=[
                {"gate": gate, "lane": 0, "tokens": [gate - 1]}
                for gate in range(1, 6)
            ],
            format_name="RAW41",
            tag=0x69FF_0001,
        ),
    ]
    result = {
        "schema_version": 1,
        "source_manifest": str(args.trace_manifest.resolve()),
        "cases": cases,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
