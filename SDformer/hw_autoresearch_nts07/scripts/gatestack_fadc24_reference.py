#!/usr/bin/env python3
"""GateStack FADC24无损格式金参考与H67真实trace容量分析。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from generate_gatestack_real_trace_vectors import build_terms, unpack_bits


MAGIC = 0x4641
VERSION = 1
HEADER_BYTES = 16
DESCRIPTOR_BYTES = 3
BITMAP_BYTES = 21
TOKENS = 162
LANES = 32
RAW_HEAD_BITS = TOKENS * (LANES + 9)
SLOT_BYTES = (RAW_HEAD_BITS + 63) // 64 * 8


def _validate_terms(terms: list[dict[str, Any]]) -> None:
    seen: set[tuple[int, int]] = set()
    for term in terms:
        gate = int(term["gate"])
        lane = int(term["lane"])
        tokens = [int(token) for token in term["tokens"]]
        if not 0 <= gate < (1 << 9):
            raise ValueError("gate超出9 bit")
        if not 0 <= lane < LANES:
            raise ValueError("lane超出5 bit")
        if not 0 < len(tokens) < (1 << 8):
            raise ValueError("destination_count超出8 bit")
        if tokens != sorted(set(tokens)) or not all(0 <= token < TOKENS for token in tokens):
            raise ValueError("token必须是有序、唯一且位于[0,161]")
        key = (gate, lane)
        if key in seen:
            raise ValueError("gate/lane term重复")
        seen.add(key)


def _bitmap_bytes(tokens: list[int]) -> bytes:
    value = 0
    for token in tokens:
        value |= 1 << token
    return value.to_bytes(BITMAP_BYTES, "little")


def serialize_fadc24(terms: list[dict[str, Any]], *, tag: int) -> bytes:
    """编码24-bit描述符，并逐term选择token list或162-bit bitmap。"""

    _validate_terms(terms)
    event_count = sum(len(term["tokens"]) for term in terms)
    if len(terms) > 255 or event_count > 8191:
        raise ValueError("header计数器溢出")

    destination_payload = bytearray()
    descriptors = bytearray()
    bitmap_terms = 0
    for term in terms:
        tokens = [int(token) for token in term["tokens"]]
        bitmap_mode = len(tokens) > BITMAP_BYTES
        bitmap_terms += int(bitmap_mode)
        descriptor = (
            int(term["gate"])
            | (int(term["lane"]) << 9)
            | (len(tokens) << 14)
            | (int(bitmap_mode) << 22)
        )
        descriptors.extend(descriptor.to_bytes(DESCRIPTOR_BYTES, "little"))
        destination_payload.extend(_bitmap_bytes(tokens) if bitmap_mode else bytes(tokens))

    destination_offset = HEADER_BYTES + len(descriptors)
    payload_bytes = destination_offset + len(destination_payload)
    word0 = MAGIC | (VERSION << 16) | ((tag & 0xFFFF_FFFF) << 32)
    word1 = (
        payload_bytes
        | (len(terms) << 16)
        | (event_count << 24)
        | (bitmap_terms << 37)
        | (destination_offset << 45)
    )
    output = bytearray(word0.to_bytes(8, "little"))
    output.extend(word1.to_bytes(8, "little"))
    output.extend(descriptors)
    output.extend(destination_payload)
    if len(output) != payload_bytes:
        raise AssertionError("FADC24 payload长度内部错误")
    return bytes(output)


def deserialize_fadc24(payload: bytes) -> dict[str, Any]:
    if len(payload) < HEADER_BYTES:
        raise ValueError("payload短于header")
    word0 = int.from_bytes(payload[:8], "little")
    word1 = int.from_bytes(payload[8:16], "little")
    if word0 & 0xFFFF != MAGIC or (word0 >> 16) & 0xFF != VERSION:
        raise ValueError("magic/version错误")
    payload_bytes = word1 & 0xFFFF
    term_count = (word1 >> 16) & 0xFF
    event_count = (word1 >> 24) & 0x1FFF
    bitmap_terms = (word1 >> 37) & 0xFF
    destination_offset = (word1 >> 45) & 0x7FF
    if payload_bytes != len(payload):
        raise ValueError("payload_bytes错误")
    if destination_offset != HEADER_BYTES + DESCRIPTOR_BYTES * term_count:
        raise ValueError("destination_offset错误")

    descriptors = []
    for index in range(term_count):
        base = HEADER_BYTES + index * DESCRIPTOR_BYTES
        value = int.from_bytes(payload[base : base + DESCRIPTOR_BYTES], "little")
        if value >> 23:
            raise ValueError("descriptor保留位非零")
        descriptors.append(
            {
                "gate": value & 0x1FF,
                "lane": (value >> 9) & 0x1F,
                "count": (value >> 14) & 0xFF,
                "bitmap_mode": bool((value >> 22) & 1),
            }
        )

    cursor = destination_offset
    terms = []
    for descriptor in descriptors:
        if descriptor["count"] == 0:
            raise ValueError("零长度term")
        if descriptor["bitmap_mode"]:
            end = cursor + BITMAP_BYTES
            if end > len(payload):
                raise ValueError("bitmap越界")
            bitmap = int.from_bytes(payload[cursor:end], "little")
            if bitmap >> TOKENS:
                raise ValueError("bitmap padding非零")
            tokens = [token for token in range(TOKENS) if (bitmap >> token) & 1]
        else:
            end = cursor + descriptor["count"]
            if end > len(payload):
                raise ValueError("token list越界")
            tokens = list(payload[cursor:end])
        cursor = end
        if len(tokens) != descriptor["count"]:
            raise ValueError("destination_count与解码结果不一致")
        terms.append(
            {
                "gate": descriptor["gate"],
                "lane": descriptor["lane"],
                "tokens": tokens,
            }
        )
    if cursor != len(payload):
        raise ValueError("payload尾部存在未消费字节")
    if sum(len(term["tokens"]) for term in terms) != event_count:
        raise ValueError("event_count错误")
    if sum(descriptor["bitmap_mode"] for descriptor in descriptors) != bitmap_terms:
        raise ValueError("bitmap_terms错误")
    _validate_terms(terms)
    return {
        "tag": (word0 >> 32) & 0xFFFF_FFFF,
        "terms": terms,
        "payload_bytes": payload_bytes,
        "bitmap_terms": bitmap_terms,
        "event_count": event_count,
    }


def _record_rows(record: dict[str, Any], window: int) -> list[list[dict[str, Any]]]:
    with np.load(record["file"]) as payload:
        shape = tuple(int(value) for value in payload["k_shape"])
        k_bits = unpack_bits(payload["k_bits_packed"], shape)
        gate = payload["gate_q17"].astype(np.int64)
    _, windows, heads, spatial_tokens, lanes = shape
    if window >= windows or spatial_tokens * 2 != TOKENS or lanes != LANES:
        raise ValueError(f"trace布局不支持: {record['name']} {shape}")
    k_rows = k_bits[:, window].transpose(1, 0, 2, 3).reshape(heads, TOKENS, LANES)
    return [build_terms(k_rows[head], gate[window, head]) for head in range(heads)]


def analyze_manifest(manifest: dict[str, Any], *, window: int) -> dict[str, Any]:
    records = []
    for record in manifest["records"]:
        stage = int(record["name"].split(".")[0][1:])
        rows = []
        for head, terms in enumerate(_record_rows(record, window)):
            encoded = serialize_fadc24(terms, tag=(stage << 16) | head)
            decoded = deserialize_fadc24(encoded)
            if decoded["terms"] != terms:
                raise AssertionError("FADC24往返不等价")
            events = sum(len(term["tokens"]) for term in terms)
            ipd32w_bytes = 16 + 8 * math.ceil(len(terms) / 2) + events
            raw_words = math.ceil(RAW_HEAD_BITS / 64)
            rows.append(
                {
                    "head": head,
                    "terms": len(terms),
                    "events": events,
                    "max_fanout": max((len(term["tokens"]) for term in terms), default=0),
                    "bitmap_terms": decoded["bitmap_terms"],
                    "ipd32w_bytes": ipd32w_bytes,
                    "fadc24_bytes": len(encoded),
                    "fadc24_words": math.ceil(len(encoded) / 8),
                    "slot_words": raw_words,
                    "ipd32w_fits": ipd32w_bytes <= SLOT_BYTES,
                    "fadc24_fits": len(encoded) <= SLOT_BYTES,
                }
            )
        raw_fallbacks = sum(not row["ipd32w_fits"] for row in rows)
        fadc_fallbacks = sum(not row["fadc24_fits"] for row in rows)
        current_terms = sum(
            row["terms"] if row["ipd32w_fits"] else row["events"] for row in rows
        )
        fadc_terms = sum(
            row["terms"] if row["fadc24_fits"] else row["events"] for row in rows
        )
        records.append(
            {
                "name": record["name"],
                "stage": stage,
                "heads": len(rows),
                "events": sum(row["events"] for row in rows),
                "ideal_terms": sum(row["terms"] for row in rows),
                "current_executed_terms_per_output_tile": current_terms,
                "fadc_executed_terms_per_output_tile": fadc_terms,
                "ipd32w_raw_fallbacks": raw_fallbacks,
                "fadc24_raw_fallbacks": fadc_fallbacks,
                "fadc_vs_current_term_reduction": (
                    0.0 if current_terms == 0 else 1.0 - fadc_terms / current_terms
                ),
                "rows": rows,
            }
        )
    return {
        "schema_version": 1,
        "format": {
            "name": "FADC24",
            "header_bytes": HEADER_BYTES,
            "descriptor_bytes": DESCRIPTOR_BYTES,
            "bitmap_bytes": BITMAP_BYTES,
            "slot_bytes": SLOT_BYTES,
            "selection": "每term在8-bit token list与162-bit bitmap之间选择较短者",
        },
        "window": window,
        "records": records,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GateStack FADC24真实Trace容量分析",
        "",
        "## 结论",
        "",
        "FADC24使用24-bit term描述符，并按term在8-bit token list与162-bit bitmap（21 byte）之间选择较短表示。该格式不改变gate、K或projection数值语义。",
        "",
        "| Stage | Head数 | 事件数 | 理想term数 | IPD32W RAW fallback | FADC24 RAW fallback | 当前每输出tile执行term | FADC24每输出tile执行term | 额外term减少 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in result["records"]:
        lines.append(
            f"| S{record['stage']} | {record['heads']} | {record['events']} | "
            f"{record['ideal_terms']} | {record['ipd32w_raw_fallbacks']} | "
            f"{record['fadc24_raw_fallbacks']} | "
            f"{record['current_executed_terms_per_output_tile']} | "
            f"{record['fadc_executed_terms_per_output_tile']} | "
            f"{record['fadc_vs_current_term_reduction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 溢出Head明细",
            "",
            "| Stage | Head | term | event | 最大fanout | bitmap term | IPD32W byte | FADC24 byte | 槽byte |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    found = False
    for record in result["records"]:
        for row in record["rows"]:
            if row["ipd32w_fits"]:
                continue
            found = True
            lines.append(
                f"| S{record['stage']} | {row['head']} | {row['terms']} | "
                f"{row['events']} | {row['max_fanout']} | {row['bitmap_terms']} | "
                f"{row['ipd32w_bytes']} | {row['fadc24_bytes']} | {SLOT_BYTES} |"
            )
    if not found:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 这是基于一个真实样本、四个stage首个block首个window的格式金参考与容量结果。",
            "- 已执行逐head encode/decode往返等价检查，但尚未实现FADC24 RTL decoder。",
            "- FADC24是否改善周期、面积和EDP，必须由同顶层RTL回放和目标库综合决定。",
            "- 24-bit非字对齐descriptor会增加字节选择与移位控制，不能仅凭payload缩小宣称硬件获益。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = analyze_manifest(manifest, window=args.window)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
