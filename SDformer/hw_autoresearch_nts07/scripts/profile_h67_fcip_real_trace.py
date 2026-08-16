#!/usr/bin/env python3
"""从H67真实位级trace重建FCIP所需的逐行关系。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/h67_real_bit_trace_20260717/manifest.json"
DEFAULT_OUT = ROOT / "results/h67_fcip_real_trace_profile_20260730"
SEGMENT_TOKENS = 64


def unpack_bits(
    packed: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    count = int(np.prod(shape))
    bits = np.unpackbits(packed, bitorder="little", count=count)
    return bits.reshape(shape).astype(bool)


def rne_div16(numerator: np.ndarray) -> np.ndarray:
    quotient = numerator // 16
    remainder = numerator % 16
    increment = (remainder > 8) | (
        (remainder == 8) & ((quotient & 1) != 0)
    )
    return quotient + increment.astype(quotient.dtype)


def reconstruct_h67_score_class(
    q_bits: np.ndarray,
    k_bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回[B,H,2N] H67 score class与[B,H,2N,L] K事件。"""

    if q_bits.shape != k_bits.shape or q_bits.ndim != 5:
        raise ValueError("Q/K必须共享[T=2,B,H,N,L]布局")
    if q_bits.shape[0] != 2:
        raise ValueError("H67 temporal pair要求T=2")
    lanes = int(q_bits.shape[-1])
    q_count = q_bits.sum(axis=-1, dtype=np.int32)
    k_count = k_bits.sum(axis=-1, dtype=np.int32)
    overlap = (q_bits & k_bits).sum(axis=-1, dtype=np.int32)
    same_zero = lanes - q_count - k_count + overlap
    motion = (k_bits[0] ^ k_bits[1]).sum(axis=-1, dtype=np.int32)
    numerator = 64 * overlap + same_zero + 16 * motion[None, ...]
    score = rne_div16(numerator)
    row_score = score.transpose(1, 2, 0, 3).reshape(
        q_bits.shape[1],
        q_bits.shape[2],
        -1,
    )
    row_k = k_bits.transpose(1, 2, 0, 3, 4).reshape(
        q_bits.shape[1],
        q_bits.shape[2],
        -1,
        lanes,
    )
    return row_score.astype(np.int16), row_k


def _bitmap_words(bits: np.ndarray, segment_tokens: int) -> list[int]:
    words: list[int] = []
    for start in range(0, bits.size, segment_tokens):
        word = 0
        for offset in np.flatnonzero(bits[start : start + segment_tokens]):
            word |= 1 << int(offset)
        words.append(word)
    return words


def build_row_relation(
    score_class: np.ndarray,
    k_event: np.ndarray,
    gate_q17: np.ndarray,
    *,
    segment_tokens: int = SEGMENT_TOKENS,
) -> dict[str, Any]:
    tokens, lanes = k_event.shape
    if score_class.shape != (tokens,) or gate_q17.shape != (tokens,):
        raise ValueError("逐行score/gate长度必须等于token数")
    active_token = k_event.any(axis=1)
    active_classes = np.unique(score_class[active_token])
    kzero_classes = np.unique(score_class[~active_token])
    class_to_gate: dict[int, int] = {}
    for class_id in np.unique(score_class):
        class_gates = np.unique(gate_q17[score_class == class_id])
        if class_gates.size != 1:
            raise ValueError(
                f"同一row内score class {int(class_id)}映射多个gate"
            )
        class_to_gate[int(class_id)] = int(class_gates[0])

    class_words: dict[str, list[int]] = {}
    for class_id in active_classes:
        class_words[str(int(class_id))] = _bitmap_words(
            (score_class == class_id) & active_token,
            segment_tokens,
        )
    k_words = [
        _bitmap_words(k_event[:, lane], segment_tokens)
        for lane in range(lanes)
    ]

    final_gate_groups: dict[str, list[int]] = {}
    for class_id in active_classes:
        gate = class_to_gate[int(class_id)]
        if gate == 0:
            continue
        final_gate_groups.setdefault(str(gate), []).append(int(class_id))

    class_lane_terms = 0
    class_lane_segments = 0
    for class_id in active_classes:
        class_mask = score_class == class_id
        for lane in range(lanes):
            relation = class_mask & k_event[:, lane]
            if relation.any():
                class_lane_terms += 1
                class_lane_segments += sum(
                    word != 0 for word in _bitmap_words(relation, segment_tokens)
                )

    final_gate_lane_terms = 0
    final_gate_lane_segments = 0
    for gate_text, classes in final_gate_groups.items():
        gate = int(gate_text)
        gate_mask = np.isin(score_class, classes) & (gate_q17 == gate)
        for lane in range(lanes):
            relation = gate_mask & k_event[:, lane]
            if relation.any():
                final_gate_lane_terms += 1
                final_gate_lane_segments += sum(
                    word != 0 for word in _bitmap_words(relation, segment_tokens)
                )

    return {
        "tokens": tokens,
        "lanes": lanes,
        "segments": (tokens + segment_tokens - 1) // segment_tokens,
        "active_tokens": int(active_token.sum()),
        "active_lane_events": int(k_event.sum()),
        "active_nonzero_gate_lane_events": int(
            (
                k_event
                & gate_q17.astype(bool).reshape(tokens, 1)
            ).sum()
        ),
        "active_gatezero_tokens": int(
            (active_token & (gate_q17 == 0)).sum()
        ),
        "all_score_classes": int(np.unique(score_class).size),
        "active_score_classes": int(active_classes.size),
        "kzero_score_classes": int(kzero_classes.size),
        "active_final_gates": len(final_gate_groups),
        "class_lane_terms": class_lane_terms,
        "class_lane_segments": class_lane_segments,
        "final_gate_lane_terms": final_gate_lane_terms,
        "final_gate_lane_segments": final_gate_lane_segments,
        "max_classes_per_final_gate": max(
            (len(classes) for classes in final_gate_groups.values()),
            default=0,
        ),
        "class_to_gate": {
            str(class_id): class_to_gate[class_id]
            for class_id in map(int, active_classes)
        },
        "class_words": class_words,
        "k_words": k_words,
        "final_gate_groups": final_gate_groups,
    }


def load_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(record["file"])
    with np.load(path, allow_pickle=False) as payload:
        q_shape = tuple(int(v) for v in payload["q_shape"])
        k_shape = tuple(int(v) for v in payload["k_shape"])
        q_bits = unpack_bits(payload["q_bits_packed"], q_shape)
        k_bits = unpack_bits(payload["k_bits_packed"], k_shape)
        gate_q17 = payload["gate_q17"].astype(np.int16)
    score_class, row_k = reconstruct_h67_score_class(q_bits, k_bits)
    if score_class.shape != gate_q17.shape:
        raise ValueError(
            f"{record['name']}重建score与gate shape不一致："
            f"{score_class.shape} vs {gate_q17.shape}"
        )
    rows = []
    for window in range(score_class.shape[0]):
        for head in range(score_class.shape[1]):
            relation = build_row_relation(
                score_class[window, head],
                row_k[window, head],
                gate_q17[window, head],
            )
            relation.update(
                {
                    "name": record["name"],
                    "sample_key": record["sample_key"],
                    "window": window,
                    "head": head,
                }
            )
            rows.append(relation)
    return rows


def summarize(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def profile(
    manifest: dict[str, Any],
    *,
    source_manifest: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in manifest["records"]:
        rows.extend(load_record(record))
    token_counts = sorted({int(row["tokens"]) for row in rows})
    sample_ids = sorted(
        {int(record["sample_id"]) for record in manifest["records"]}
    )
    record_names = [str(record["name"]) for record in manifest["records"]]
    first_block_only = bool(manifest.get("first_block_only", False))
    scope = (
        f"{len(sample_ids)} sample、{len(record_names)} block-window record、"
        f"{len(rows)} head-row、token={token_counts}"
    )
    metrics = (
        "active_tokens",
        "active_lane_events",
        "active_nonzero_gate_lane_events",
        "active_gatezero_tokens",
        "all_score_classes",
        "active_score_classes",
        "kzero_score_classes",
        "active_final_gates",
        "class_lane_terms",
        "class_lane_segments",
        "final_gate_lane_terms",
        "final_gate_lane_segments",
        "max_classes_per_final_gate",
    )
    return {
        "schema": "h67_fcip_real_trace_profile_v1",
        "evidence": (
            f"[真实网络bit trace] {scope}；"
            "不是多样本总体分布、RTL周期或PPA"
        ),
        "source_manifest": str(source_manifest.resolve()),
        "source_scope": {
            "sample_ids": sample_ids,
            "records": record_names,
            "first_block_only": first_block_only,
            "token_counts": token_counts,
        },
        "rows": rows,
        "row_count": len(rows),
        "summaries": {
            metric: summarize([int(row[metric]) for row in rows])
            for metric in metrics
        },
        "invariants": {
            "class_to_gate_single_valued_per_row": True,
            "class_lane_segment_ge_term": all(
                row["class_lane_segments"] >= row["class_lane_terms"]
                for row in rows
            ),
            "final_gate_lane_segment_ge_term": all(
                row["final_gate_lane_segments"]
                >= row["final_gate_lane_terms"]
                for row in rows
            ),
        },
        "limits": [
            f"只覆盖{scope}。",
            "单样本结果不能代替多样本 mean/p95/p99。"
            if len(sample_ids) == 1
            else "样本集合由source manifest限定。",
            "本报告重建关系，不估计周期、面积或功耗。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# H67 FCIP 真实位级关系画像",
        "",
        f"- 逐行数：{report['row_count']}",
        f"- 证据：{report['evidence']}",
        "",
        "| 指标 | mean | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in report["summaries"].items():
        lines.append(
            f"| {name} | {values['mean']:.3f} | {values['p95']:.3f} | "
            f"{values['p99']:.3f} | {values['max']} |"
        )
    lines += [
        "",
        "## 合同",
        "",
        "- H67 score class按RTL部署式重建：",
        "  `RNE((64*overlap + same_zero + 16*motion)/16)`。",
        "- 每个row内验证同一score class只映射一个Q1.7 final gate。",
        "- 关系同时保留score-class平面、K-lane平面和final-gate alias。",
        "",
        "## 边界",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = profile(manifest, source_manifest=args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
