#!/usr/bin/env python3
"""GateStack IPD24字节流的可逆序列化金参考。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/gatestack_ipd_format_reference_20260715.json"
MAGIC = 0x4753
VERSION = 1
HEADER_BYTES = 16
DESCRIPTOR_BYTES = 3
RAW_HEAD_BITS = 6642


def build_terms(k_head: np.ndarray, gate_head: np.ndarray) -> list[dict[str, Any]]:
    if k_head.ndim != 2 or gate_head.shape != (k_head.shape[0],):
        raise ValueError("单head输入形状不匹配")
    if k_head.shape[0] > 256 or k_head.shape[1] > 32:
        raise ValueError("IPD24只支持token<=256且lane<=32")
    active_tokens = np.flatnonzero(k_head.any(axis=1))
    terms: list[dict[str, Any]] = []
    for gate in np.unique(gate_head[active_tokens]):
        if not 0 <= int(gate) <= 256:
            raise ValueError("gate_code超出9 bit")
        gate_tokens = active_tokens[gate_head[active_tokens] == gate]
        for lane in np.flatnonzero(k_head[gate_tokens].any(axis=0)):
            destinations = gate_tokens[k_head[gate_tokens, lane]]
            if destinations.size > 255:
                raise ValueError("单term事件数超出8 bit")
            terms.append(
                {
                    "gate_code": int(gate),
                    "lane": int(lane),
                    "destinations": destinations.astype(np.uint8).tolist(),
                }
            )
    return terms


def serialize_ipd24(
    k_head: np.ndarray,
    gate_head: np.ndarray,
    *,
    tag: int,
    class_slots: int = 4,
) -> bytes | None:
    terms = build_terms(k_head, gate_head)
    active_tokens = np.flatnonzero(k_head.any(axis=1))
    active_classes = np.unique(gate_head[active_tokens])
    if active_classes.size > class_slots:
        return None
    event_count = sum(len(term["destinations"]) for term in terms)
    if len(terms) > 255 or event_count > 8191:
        raise ValueError("header计数器溢出")
    token_offset = HEADER_BYTES + DESCRIPTOR_BYTES * len(terms)
    payload_bytes = token_offset + event_count
    payload_bits = payload_bytes * 8
    if payload_bits > RAW_HEAD_BITS:
        return None

    word0 = MAGIC | (VERSION << 16) | (1 << 20) | ((tag & 0xFFFF_FFFF) << 32)
    word1 = (
        payload_bits
        | (len(terms) << 13)
        | (event_count << 21)
        | (int(active_classes.size) << 34)
        | (int(active_tokens.size) << 37)
        | (token_offset << 45)
    )
    output = bytearray(word0.to_bytes(8, "little"))
    output.extend(word1.to_bytes(8, "little"))
    for term in terms:
        descriptor = (
            term["gate_code"]
            | (term["lane"] << 9)
            | (len(term["destinations"]) << 14)
        )
        output.extend(descriptor.to_bytes(3, "little"))
    for term in terms:
        output.extend(term["destinations"])
    if len(output) != payload_bytes:
        raise AssertionError("IPD24 payload长度内部错误")
    return bytes(output)


def serialize_ipd32w(
    k_head: np.ndarray,
    gate_head: np.ndarray,
    *,
    tag: int,
    class_slots: int = 4,
) -> bytes | None:
    """主线格式：每个64-bit word容纳两个32-bit隐式前缀descriptor。"""

    terms = build_terms(k_head, gate_head)
    active_tokens = np.flatnonzero(k_head.any(axis=1))
    active_classes = np.unique(gate_head[active_tokens])
    if active_classes.size > class_slots:
        return None
    event_count = sum(len(term["destinations"]) for term in terms)
    descriptor_words = (len(terms) + 1) // 2
    token_offset = HEADER_BYTES + 8 * descriptor_words
    payload_bytes = token_offset + event_count
    payload_bits = payload_bytes * 8
    if payload_bits > RAW_HEAD_BITS:
        return None

    word0 = MAGIC | (VERSION << 16) | (1 << 20) | ((tag & 0xFFFF_FFFF) << 32)
    word1 = (
        payload_bits
        | (len(terms) << 13)
        | (event_count << 21)
        | (int(active_classes.size) << 34)
        | (int(active_tokens.size) << 37)
        | (token_offset << 45)
    )
    output = bytearray(word0.to_bytes(8, "little"))
    output.extend(word1.to_bytes(8, "little"))
    for base in range(0, len(terms), 2):
        packed_word = 0
        for way in range(2):
            if base + way >= len(terms):
                continue
            term = terms[base + way]
            descriptor = (
                term["gate_code"]
                | (term["lane"] << 9)
                | (len(term["destinations"]) << 14)
            )
            packed_word |= descriptor << (way * 32)
        output.extend(packed_word.to_bytes(8, "little"))
    for term in terms:
        output.extend(term["destinations"])
    if len(output) != payload_bytes:
        raise AssertionError("IPD32W payload长度内部错误")
    return bytes(output)


def parse_ipd24(payload: bytes, *, tokens: int, lanes: int) -> dict[str, Any]:
    if len(payload) < HEADER_BYTES:
        raise ValueError("payload短于header")
    word0 = int.from_bytes(payload[:8], "little")
    word1 = int.from_bytes(payload[8:16], "little")
    if (word0 & 0xFFFF) != MAGIC or ((word0 >> 16) & 0xF) != VERSION:
        raise ValueError("magic/version错误")
    if ((word0 >> 20) & 1) != 1 or ((word0 >> 21) & 0x7FF) != 0:
        raise ValueError("mode/reserved错误")
    if (word1 >> 55) != 0:
        raise ValueError("header reserved非零")
    payload_bits = word1 & 0x1FFF
    term_count = (word1 >> 13) & 0xFF
    event_count = (word1 >> 21) & 0x1FFF
    active_classes = (word1 >> 34) & 0x7
    active_tokens = (word1 >> 37) & 0xFF
    token_offset = (word1 >> 45) & 0x3FF
    if payload_bits != len(payload) * 8:
        raise ValueError("payload_bits与输入长度不一致")
    if token_offset != HEADER_BYTES + DESCRIPTOR_BYTES * term_count:
        raise ValueError("token_offset错误")
    if token_offset + event_count != len(payload):
        raise ValueError("event_count错误")

    terms: list[dict[str, Any]] = []
    event_cursor = token_offset
    for index in range(term_count):
        base = HEADER_BYTES + index * DESCRIPTOR_BYTES
        descriptor = int.from_bytes(payload[base : base + 3], "little")
        if descriptor >> 22:
            raise ValueError("descriptor reserved非零")
        gate_code = descriptor & 0x1FF
        lane = (descriptor >> 9) & 0x1F
        count = (descriptor >> 14) & 0xFF
        if lane >= lanes or event_cursor + count > len(payload):
            raise ValueError("descriptor范围错误")
        destinations = list(payload[event_cursor : event_cursor + count])
        if any(token >= tokens for token in destinations):
            raise ValueError("token id越界")
        if len(set(destinations)) != len(destinations):
            raise ValueError("单term token重复")
        terms.append(
            {
                "gate_code": gate_code,
                "lane": lane,
                "destinations": destinations,
            }
        )
        event_cursor += count
    if event_cursor != len(payload):
        raise ValueError("隐式前缀未消费完整token列表")
    if len({term["gate_code"] for term in terms}) != active_classes:
        raise ValueError("active_classes错误")
    if len({token for term in terms for token in term["destinations"]}) != active_tokens:
        raise ValueError("active_tokens错误")
    return {
        "tag": (word0 >> 32) & 0xFFFF_FFFF,
        "payload_bits": payload_bits,
        "term_count": term_count,
        "event_count": event_count,
        "active_classes": active_classes,
        "active_tokens": active_tokens,
        "token_offset": token_offset,
        "terms": terms,
    }


def parse_ipd32w(payload: bytes, *, tokens: int, lanes: int) -> dict[str, Any]:
    if len(payload) < HEADER_BYTES:
        raise ValueError("payload短于header")
    word0 = int.from_bytes(payload[:8], "little")
    word1 = int.from_bytes(payload[8:16], "little")
    if (word0 & 0xFFFF) != MAGIC or ((word0 >> 16) & 0xF) != VERSION:
        raise ValueError("magic/version错误")
    if ((word0 >> 20) & 1) != 1 or ((word0 >> 21) & 0x7FF) != 0:
        raise ValueError("mode/reserved错误")
    if (word1 >> 55) != 0:
        raise ValueError("header reserved非零")
    payload_bits = word1 & 0x1FFF
    term_count = (word1 >> 13) & 0xFF
    event_count = (word1 >> 21) & 0x1FFF
    active_classes = (word1 >> 34) & 0x7
    active_tokens = (word1 >> 37) & 0xFF
    token_offset = (word1 >> 45) & 0x3FF
    expected_offset = HEADER_BYTES + 8 * ((term_count + 1) // 2)
    if payload_bits != len(payload) * 8:
        raise ValueError("payload_bits与输入长度不一致")
    if token_offset != expected_offset:
        raise ValueError("token_offset错误")
    if token_offset + event_count != len(payload):
        raise ValueError("event_count错误")

    terms: list[dict[str, Any]] = []
    for index in range(term_count):
        base = HEADER_BYTES + (index // 2) * 8 + (index % 2) * 4
        descriptor = int.from_bytes(payload[base : base + 4], "little")
        if descriptor >> 22:
            raise ValueError("descriptor reserved非零")
        terms.append(
            {
                "gate_code": descriptor & 0x1FF,
                "lane": (descriptor >> 9) & 0x1F,
                "event_count": (descriptor >> 14) & 0xFF,
            }
        )
    if term_count & 1:
        padding = int.from_bytes(payload[token_offset - 4 : token_offset], "little")
        if padding != 0:
            raise ValueError("奇数descriptor padding非零")

    event_cursor = token_offset
    parsed_terms: list[dict[str, Any]] = []
    for term in terms:
        count = term.pop("event_count")
        if term["lane"] >= lanes or event_cursor + count > len(payload):
            raise ValueError("descriptor范围错误")
        destinations = list(payload[event_cursor : event_cursor + count])
        if any(token >= tokens for token in destinations):
            raise ValueError("token id越界")
        if len(set(destinations)) != len(destinations):
            raise ValueError("单term token重复")
        term["destinations"] = destinations
        parsed_terms.append(term)
        event_cursor += count
    if event_cursor != len(payload):
        raise ValueError("隐式前缀未消费完整token列表")
    if len({term["gate_code"] for term in parsed_terms}) != active_classes:
        raise ValueError("active_classes错误")
    if len({token for term in parsed_terms for token in term["destinations"]}) != active_tokens:
        raise ValueError("active_tokens错误")
    return {
        "tag": (word0 >> 32) & 0xFFFF_FFFF,
        "payload_bits": payload_bits,
        "term_count": term_count,
        "event_count": event_count,
        "active_classes": active_classes,
        "active_tokens": active_tokens,
        "token_offset": token_offset,
        "terms": parsed_terms,
    }


def reconstruct(
    parsed: dict[str, Any], *, tokens: int, lanes: int
) -> tuple[np.ndarray, np.ndarray]:
    k_head = np.zeros((tokens, lanes), dtype=bool)
    gated = np.zeros((tokens, lanes), dtype=np.int16)
    for term in parsed["terms"]:
        for token in term["destinations"]:
            if k_head[token, term["lane"]]:
                raise ValueError("跨term出现重复token/lane")
            k_head[token, term["lane"]] = True
            gated[token, term["lane"]] = term["gate_code"]
    return k_head, gated


def run_trials(seed: int = 20260715, trials: int = 500) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    csr = 0
    raw_class = 0
    raw_capacity = 0
    compared_events = 0
    for trial in range(trials):
        density = rng.uniform(0.005, 0.55)
        k_head = rng.random((162, 32)) < density
        class_count = int(rng.integers(1, 7))
        codebook = rng.choice(257, size=class_count, replace=False)
        gate_head = codebook[rng.integers(0, class_count, size=162)].astype(np.int16)
        active_classes = np.unique(gate_head[k_head.any(axis=1)]).size
        payload = serialize_ipd32w(k_head, gate_head, tag=trial)
        if payload is None:
            if active_classes > 4:
                raw_class += 1
            else:
                raw_capacity += 1
            continue
        parsed = parse_ipd32w(payload, tokens=162, lanes=32)
        reconstructed_k, reconstructed_gated = reconstruct(
            parsed, tokens=162, lanes=32
        )
        expected_gated = k_head.astype(np.int16) * gate_head[:, None]
        if not np.array_equal(reconstructed_k, k_head):
            raise AssertionError("K event重建不一致")
        if not np.array_equal(reconstructed_gated, expected_gated):
            raise AssertionError("gated K重建不一致")
        if parsed["tag"] != trial:
            raise AssertionError("tag重建不一致")
        csr += 1
        compared_events += int(k_head.sum())
    return {
        "seed": seed,
        "trials": trials,
        "csr_trials": csr,
        "raw_class_trials": raw_class,
        "raw_capacity_trials": raw_capacity,
        "compared_csr_events": compared_events,
        "mismatches": 0,
    }


def main() -> int:
    result = run_trials()
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
