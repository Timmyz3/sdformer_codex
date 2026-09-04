#!/usr/bin/env python3
"""生成H67 stage3真实有序统计塑形的GateStack RTL回放向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_VECTOR_DIR = ROOT / "tb_hitflow/vectors/gatestack_h67_stage3_sample0_b0"
DEFAULT_RESULT = ROOT / "results/gatestack_h67_stage3_trace_20260716/manifest.json"

TOKENS = 162
LANES = 32
HEADS = 24
RAW_HEAD_BITS = TOKENS * (LANES + 9)
WORDS_PER_HEAD = (RAW_HEAD_BITS + 63) // 64
PAYLOAD_TAG_BASE = 0x6800_0000


def descriptor(gate: int, lane: int, count: int) -> int:
    return gate | (lane << 9) | (count << 14)


def distribute_counts(terms: int, events: int, max_fanout: int) -> list[int]:
    if terms == 0:
        if events != 0 or max_fanout != 0:
            raise ValueError("空head统计不一致")
        return []
    if not (terms <= events <= terms * max_fanout):
        raise ValueError("term/event/max_fanout统计不可构造")
    counts = [1] * terms
    counts[0] = max_fanout
    remaining = events - sum(counts)
    for index in range(1, terms):
        add = min(remaining, max_fanout - 1)
        counts[index] += add
        remaining -= add
    if remaining != 0 or max(counts) != max_fanout:
        raise ValueError("fanout分配失败")
    return counts


def build_terms(
    term_count: int,
    event_count: int,
    class_count: int,
    max_fanout: int,
) -> list[dict[str, Any]]:
    if term_count == 0:
        return []
    if class_count <= 0 or term_count > class_count * LANES:
        raise ValueError("class/term统计超出GateStack表示范围")
    counts = distribute_counts(term_count, event_count, max_fanout)
    pairs = [(class_id, lane) for lane in range(LANES) for class_id in range(class_count)]
    assigned = [(*pairs[index], counts[index]) for index in range(term_count)]
    class_pool_sizes = [0] * class_count
    for class_id, _, count in assigned:
        class_pool_sizes[class_id] = max(class_pool_sizes[class_id], count)
    if sum(class_pool_sizes) > TOKENS:
        raise ValueError("确定性token池超过162")
    class_pool_starts: list[int] = []
    cursor = 0
    for size in class_pool_sizes:
        class_pool_starts.append(cursor)
        cursor += size
    terms = []
    for class_id, lane, count in assigned:
        start = class_pool_starts[class_id]
        terms.append(
            {
                "gate": 64 * (class_id + 1),
                "lane": lane,
                "tokens": list(range(start, start + count)),
            }
        )
    return terms


def serialize_ipd(terms: list[dict[str, Any]], tag: int) -> tuple[bytes, int]:
    event_count = sum(len(term["tokens"]) for term in terms)
    classes = len({term["gate"] for term in terms})
    active_tokens = len({token for term in terms for token in term["tokens"]})
    descriptor_words = (len(terms) + 1) // 2
    token_offset = 16 + 8 * descriptor_words
    payload_bytes = token_offset + event_count
    payload_bits = payload_bytes * 8
    word0 = 0x4753 | (1 << 16) | (1 << 20) | ((tag & 0xFFFF_FFFF) << 32)
    word1 = (
        payload_bits
        | (len(terms) << 13)
        | (event_count << 21)
        | (classes << 34)
        | (active_tokens << 37)
        | (token_offset << 45)
    )
    payload = bytearray(word0.to_bytes(8, "little"))
    payload.extend(word1.to_bytes(8, "little"))
    for base in range(0, len(terms), 2):
        packed = 0
        for way in range(2):
            if base + way < len(terms):
                term = terms[base + way]
                packed |= descriptor(term["gate"], term["lane"], len(term["tokens"])) << (32 * way)
        payload.extend(packed.to_bytes(8, "little"))
    for term in terms:
        payload.extend(bytes(term["tokens"]))
    if len(payload) != payload_bytes:
        raise AssertionError("IPD payload长度错误")
    return bytes(payload), payload_bits


def serialize_raw(terms: list[dict[str, Any]]) -> tuple[bytes, int]:
    k_bits = [0] * TOKENS
    gates = [0] * TOKENS
    for term in terms:
        for token in term["tokens"]:
            if gates[token] not in (0, term["gate"]):
                raise ValueError("同一token出现不同gate")
            gates[token] = term["gate"]
            k_bits[token] |= 1 << term["lane"]
    packed = 0
    for token in range(TOKENS):
        packed |= (k_bits[token] | (gates[token] << LANES)) << (token * 41)
    return packed.to_bytes(WORDS_PER_HEAD * 8, "little"), RAW_HEAD_BITS


def words_from_payload(payload: bytes) -> list[int]:
    padded = payload + bytes((-len(payload)) % 8)
    return [int.from_bytes(padded[index : index + 8], "little") for index in range(0, len(padded), 8)]


def select_record(profile: dict[str, Any]) -> dict[str, Any]:
    for record in profile["summary"]["h60_records"]:
        if (
            int(record["stage"]) == 3
            and int(record["sample_id"]) == 0
            and str(record["name"]) == "S3.B0.attn"
        ):
            return record
    raise ValueError("未找到H67 sample0 S3.B0.attn")


def write_memh(path: Path, values: list[int], width: int) -> None:
    digits = (width + 3) // 4
    path.write_text("".join(f"{value & ((1 << width) - 1):0{digits}x}\n" for value in values), encoding="ascii")


def generate(
    profile_path: Path,
    vector_dir: Path,
    result_path: Path,
    *,
    force_raw: bool = False,
) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    record = select_record(profile)
    fields = {
        "terms": "projection_gate_class_channel_terms_deploy_ordered_trace",
        "events": "projection_baseline_active_lanes_ordered_trace",
        "classes": "projection_active_gate_classes_deploy_ordered_trace",
        "max_fanout": "projection_gate_class_channel_max_fanout_deploy_ordered_trace",
    }
    traces = {name: decode_count_trace(record[field])[:HEADS] for name, field in fields.items()}
    if any(len(values) != HEADS for values in traces.values()):
        raise ValueError("ordered trace不足24个head")

    payload_words = [0] * (HEADS * WORDS_PER_HEAD)
    payload_bits: list[int] = []
    payload_modes: list[int] = []
    payload_word_counts: list[int] = []
    expected_gate_sum = [0] * TOKENS
    heads: list[dict[str, Any]] = []
    for head in range(HEADS):
        terms = build_terms(
            traces["terms"][head],
            traces["events"][head],
            traces["classes"][head],
            traces["max_fanout"][head],
        )
        ipd_payload, ipd_bits = serialize_ipd(terms, PAYLOAD_TAG_BASE + head)
        mode_is_csr = ipd_bits <= RAW_HEAD_BITS and not force_raw
        payload, bits = (ipd_payload, ipd_bits) if mode_is_csr else serialize_raw(terms)
        words = words_from_payload(payload)
        if len(words) > WORDS_PER_HEAD:
            raise ValueError("payload超过物理head slot")
        for index, word in enumerate(words):
            payload_words[head * WORDS_PER_HEAD + index] = word
        for term in terms:
            for token in term["tokens"]:
                expected_gate_sum[token] += term["gate"]
        payload_bits.append(bits)
        payload_modes.append(int(mode_is_csr))
        payload_word_counts.append(len(words))
        heads.append(
            {
                "head": head,
                "term_count": traces["terms"][head],
                "event_count": traces["events"][head],
                "class_count": traces["classes"][head],
                "max_fanout": traces["max_fanout"][head],
                "mode": "IPD32W" if mode_is_csr else "RAW41",
                "payload_bits": bits,
                "word_count": len(words),
            }
        )

    vector_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_memh(vector_dir / "payload_words.memh", payload_words, 64)
    write_memh(vector_dir / "payload_bits.memh", payload_bits, 16)
    write_memh(vector_dir / "payload_modes.memh", payload_modes, 1)
    write_memh(vector_dir / "payload_word_counts.memh", payload_word_counts, 8)
    write_memh(vector_dir / "term_counts.memh", traces["terms"], 8)
    write_memh(vector_dir / "event_counts.memh", traces["events"], 13)
    write_memh(vector_dir / "expected_gate_sum.memh", expected_gate_sum, 32)

    csr_heads = sum(payload_modes)
    raw_heads = HEADS - csr_heads
    nonempty_csr_heads = sum(
        row["mode"] == "IPD32W" and row["term_count"] != 0 for row in heads
    )
    full_payload_words_per_tile = sum(row["word_count"] for row in heads)
    warm_payload_words_per_tile = 0
    for row in heads:
        if row["mode"] == "RAW41":
            warm_payload_words_per_tile += row["word_count"]
        else:
            token_start_word = 2 + (row["term_count"] + 1) // 2
            warm_payload_words_per_tile += max(row["word_count"] - token_start_word, 0)
    no_residency_payload_words = HEADS * full_payload_words_per_tile
    residency_payload_words = full_payload_words_per_tile + (HEADS - 1) * warm_payload_words_per_tile
    with profile_path.open("rb") as handle:
        source_sha256 = hashlib.sha256(handle.read()).hexdigest()
    manifest = {
        "schema_version": 1,
        "evidence": "[prof ordered trace]+[确定性trace-shaped payload]",
        "baseline_mode": "RAW41_ONLY" if force_raw else "CAPACITY_SAFE",
        "source": str(profile_path),
        "source_sha256": source_sha256,
        "selection": {"sample_id": 0, "stage": 3, "block": "S3.B0.attn", "window": 0},
        "dimensions": {"tokens": TOKENS, "lanes": LANES, "heads": HEADS, "output_tiles": HEADS},
        "totals": {
            "terms": sum(traces["terms"]),
            "events": sum(traces["events"]),
            "csr_heads": csr_heads,
            "raw_heads": raw_heads,
            "empty_csr_heads": csr_heads - nonempty_csr_heads,
            "expected_cache_hits_after_first_tile": csr_heads * (HEADS - 1),
            "expected_slot_replays_all_tiles": HEADS
            + (HEADS - 1) * (nonempty_csr_heads + raw_heads),
            "expected_projection_terms_all_tiles": HEADS
            * (
                sum(row["term_count"] for row in heads if row["mode"] == "IPD32W")
                + sum(row["event_count"] for row in heads if row["mode"] == "RAW41")
            ),
            "full_payload_words_per_cold_tile": full_payload_words_per_tile,
            "payload_words_per_warm_tile": warm_payload_words_per_tile,
            "no_residency_payload_words_all_tiles": no_residency_payload_words,
            "residency_payload_words_all_tiles": residency_payload_words,
            "payload_word_reduction_ratio": 1.0
            - residency_payload_words / no_residency_payload_words,
        },
        "heads": heads,
        "limits": [
            "有序term/event/class/max-fanout来自真实profile",
            "profile不含完整gate/lane/token payload，数值内容为确定性可逆构造",
            "不得称为网络逐bit trace或目标库功耗证据",
        ],
    }
    result_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--force-raw", action="store_true")
    args = parser.parse_args()
    manifest = generate(
        args.profile,
        args.vector_dir,
        args.result,
        force_raw=args.force_raw,
    )
    print(json.dumps(manifest["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
