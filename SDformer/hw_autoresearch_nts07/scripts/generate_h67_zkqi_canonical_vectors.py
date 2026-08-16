#!/usr/bin/env python3
"""从全量count trace构造可实现canonical Q/K并生成ZKQI RTL replay向量。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.generate_h67_checkpoint_row_vectors import row_gate_codes, score_q7
    from scripts.profile_h67_zkqi_multisample_ordered import (
        DEFAULT_PROFILE,
        block_identity,
        decode_record,
        decode_trace,
        receipt,
        validate_profile_contract,
    )
except ModuleNotFoundError:
    from generate_h67_checkpoint_row_vectors import row_gate_codes, score_q7
    from profile_h67_zkqi_multisample_ordered import (
        DEFAULT_PROFILE,
        block_identity,
        decode_record,
        decode_trace,
        receipt,
        validate_profile_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tb_h67/vectors/h67_zkqi_canonical_multisample_20260809"
MASK32 = (1 << 32) - 1


def take_lowest(mask: int, count: int) -> int:
    if count < 0 or count > mask.bit_count():
        raise ValueError("canonical bit选择数量不可实现")
    result = 0
    remaining = mask & MASK32
    for _ in range(count):
        bit = remaining & -remaining
        result |= bit
        remaining ^= bit
    return result


def rotate_left32(value: int, shift: int) -> int:
    amount = shift & 31
    if amount == 0:
        return value & MASK32
    return ((value << amount) | (value >> (32 - amount))) & MASK32


def canonical_pair(
    q0_count: int,
    q1_count: int,
    k0_count: int,
    k1_count: int,
    overlap0: int,
    overlap1: int,
    motion: int,
    *,
    rotate: int = 0,
) -> tuple[int, int, int, int]:
    intersection_twice = k0_count + k1_count - motion
    if intersection_twice & 1:
        raise ValueError("K count/motion奇偶不可实现")
    intersection = intersection_twice // 2
    if (
        intersection < 0
        or intersection > min(k0_count, k1_count)
        or k0_count + k1_count - intersection > 32
    ):
        raise ValueError("K count/motion集合关系不可实现")

    k0 = (1 << k0_count) - 1 if k0_count else 0
    shared = take_lowest(k0, intersection)
    k1_only = take_lowest((~k0) & MASK32, k1_count - intersection)
    k1 = shared | k1_only

    def make_q(q_count: int, k_value: int, overlap: int) -> int:
        if overlap > min(q_count, k_value.bit_count()):
            raise ValueError("Q/K overlap超过集合大小")
        q_only = q_count - overlap
        if q_only > 32 - k_value.bit_count():
            raise ValueError("Q-only集合没有足够lane")
        return take_lowest(k_value, overlap) | take_lowest(
            (~k_value) & MASK32, q_only
        )

    q0 = make_q(q0_count, k0, overlap0)
    q1 = make_q(q1_count, k1, overlap1)
    q0, q1, k0, k1 = (
        rotate_left32(value, rotate) for value in (q0, q1, k0, k1)
    )

    observed = (
        q0.bit_count(), q1.bit_count(), k0.bit_count(), k1.bit_count(),
        (q0 & k0).bit_count(), (q1 & k1).bit_count(), (k0 ^ k1).bit_count(),
    )
    expected = (
        q0_count, q1_count, k0_count, k1_count,
        overlap0, overlap1, motion,
    )
    if observed != expected:
        raise RuntimeError(f"canonical pair构造不守恒: {observed} != {expected}")
    return q0, q1, k0, k1


def row_index(flat_index: int, heads: int) -> tuple[int, int]:
    return flat_index // heads, flat_index % heads


def add_selection(
    selected: dict[tuple[int, str, int, int], set[str]],
    sample: int,
    name: str,
    window: int,
    head: int,
    reason: str,
) -> None:
    selected[(sample, name, window, head)].add(reason)


def select_rows(records: list[dict[str, Any]]) -> dict[tuple[int, str, int, int], set[str]]:
    selected: dict[tuple[int, str, int, int], set[str]] = defaultdict(set)
    for record in records:
        sample = int(record["sample_id"])
        stage, block, name = block_identity(record)
        metrics, _ = decode_record(record)
        windows, heads = metrics["active_pairs"].shape
        density = metrics["active_pairs"].reshape(-1)

        hashed_window = (sample * 37 + stage * 11 + block * 17) % windows
        hashed_head = (sample * 13 + block * 7 + stage) % heads
        add_selection(
            selected, sample, name, hashed_window, hashed_head,
            "sample_block_hash",
        )

        ordered = np.argsort(density, kind="stable")
        for reason, index in (
            ("record_density_min", int(ordered[0])),
            ("record_density_median", int(ordered[len(ordered) // 2])),
            ("record_density_max", int(ordered[-1])),
        ):
            window, head = row_index(index, heads)
            add_selection(selected, sample, name, window, head, reason)

        slower = np.flatnonzero(
            metrics["ttb_cycles"].reshape(-1)
            > metrics["baseline_cycles"].reshape(-1)
        )
        if slower.size:
            window, head = row_index(int(slower[0]), heads)
            add_selection(
                selected, sample, name, window, head, "record_first_ttb_slow"
            )

        # 每个block的每个head至少由一个跨sample canonical row覆盖。
        for head in range(heads):
            if sample == head % 100:
                window = (head * 19 + stage * 5 + block * 3) % windows
                add_selection(
                    selected, sample, name, window, head, "block_head_coverage"
                )
    return selected


def build_row(
    record: dict[str, Any],
    metrics: dict[str, np.ndarray],
    checks: dict[str, Any],
    window: int,
    head: int,
) -> dict[str, Any]:
    sample = int(record["sample_id"])
    stage, block, name = block_identity(record)
    q_count = checks["q_count"][:, window, head]
    k_count = checks["k_count"][:, window, head]
    # decode_record不返回overlap/motion，按原trace单独解码会浪费；调用方注入缓存。
    overlap = checks["overlap"][:, window, head]
    motion = checks["motion"][window, head]
    score = metrics["scores"][:, window, head]

    q0: list[int] = []
    q1: list[int] = []
    k0: list[int] = []
    k1: list[int] = []
    for pair in range(q_count.shape[1]):
        rotate = (
            sample * 7 + stage * 13 + block * 17 + window * 3
            + head * 5 + pair * 11
        ) & 31
        values = canonical_pair(
            int(q_count[0, pair]), int(q_count[1, pair]),
            int(k_count[0, pair]), int(k_count[1, pair]),
            int(overlap[0, pair]), int(overlap[1, pair]),
            int(motion[pair]), rotate=rotate,
        )
        cq0, cq1, ck0, ck1 = values
        if (
            score_q7(cq0, ck0, ck1) != int(score[0, pair])
            or score_q7(cq1, ck1, ck0) != int(score[1, pair])
        ):
            raise RuntimeError("canonical bit向量没有保持H67 score")
        q0.append(cq0)
        q1.append(cq1)
        k0.append(ck0)
        k1.append(ck1)

    q = q0 + q1
    current = k0 + k1
    peer = k1 + k0
    gates = row_gate_codes(q, current, peer)
    return {
        "sample": sample,
        "name": name,
        "stage": stage,
        "block": block,
        "window": window,
        "head": head,
        "active_pairs": int(metrics["active_pairs"][window, head]),
        "expected_outputs": int(metrics["outputs"][window, head]),
        "expected_folded": 450 - int(metrics["outputs"][window, head]),
        "baseline_cycles": int(metrics["baseline_cycles"][window, head]),
        "ttb_cycles": int(metrics["ttb_cycles"][window, head]),
        "baseline_read_bits": int(metrics["baseline_read_bits"][window, head]),
        "candidate_read_bits": int(metrics["candidate_read_bits"][window, head]),
        "vectors": list(zip(q, current, peer, gates)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    records = profile.get("summary", {}).get("h60_records") or []
    contract = validate_profile_contract(profile, records)
    selected = select_rows(records)
    by_record: dict[tuple[int, str], list[tuple[int, int, set[str]]]] = defaultdict(list)
    for (sample, name, window, head), reasons in selected.items():
        by_record[(sample, name)].append((window, head, reasons))

    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        key = (int(record["sample_id"]), str(record["name"]))
        requests = by_record.get(key)
        if not requests:
            continue
        metrics, checks = decode_record(record)
        # 保持解码边界清楚；这两个字段只用于canonical bit构造。
        checks["overlap"] = decode_trace(record["pair_overlap_ordered_trace"])
        checks["motion"] = decode_trace(record["pair_motion_ordered_trace"])
        for window, head, reasons in sorted(requests):
            row = build_row(record, metrics, checks, window, head)
            row["selection_reasons"] = sorted(reasons)
            reason_counts.update(reasons)
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["sample"], row["stage"], row["block"], row["window"], row["head"]
        )
    )
    if not rows:
        raise ValueError("canonical selection为空")
    sample_ids = {row["sample"] for row in rows}
    block_names = {row["name"] for row in rows}
    head_coverage: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        head_coverage[row["name"]].add(row["head"])
    if sample_ids != set(range(100)) or len(block_names) != 12:
        raise ValueError("canonical selection没有覆盖100 sample/all12")
    for record in records[:12]:
        name = str(record["name"])
        if head_coverage[name] != set(range(int(record["num_heads"]))):
            raise ValueError(f"canonical selection没有覆盖{name}全部head")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / "h67_canonical_rows.txt"
    with vector_path.open("w", encoding="ascii") as handle:
        handle.write(f"{len(rows)} 450\n")
        for row_id, row in enumerate(rows):
            handle.write(
                f"{row_id} {row['stage']} {row['block']} {row['head']} "
                f"{row['expected_outputs']} {row['expected_folded']}\n"
            )
            for q, current, peer, gate in row["vectors"]:
                handle.write(f"{q:08x} {current:08x} {peer:08x} {gate}\n")

    row_receipts = []
    for row_id, row in enumerate(rows):
        row_receipts.append(
            {
                "row": row_id,
                **{key: row[key] for key in (
                    "sample", "name", "stage", "block", "window", "head",
                    "active_pairs", "expected_outputs", "expected_folded",
                    "baseline_cycles", "ttb_cycles", "baseline_read_bits",
                    "candidate_read_bits", "selection_reasons",
                )},
            }
        )
    manifest = {
        "schema": "h67_zkqi_canonical_multisample_vectors_v1",
        "status": "PASS",
        "evidence_level": "[prof构造向量]",
        "scope": "canonical control-state replay；不恢复原始bit身份，不代表真实toggle/SAIF",
        "profile_contract": contract,
        "selection_policy": {
            "per_sample_block": [
                "固定hash window/head", "active-pair最小", "中位", "最大",
                "首个TTB慢行（若存在）",
            ],
            "coverage_guard": "100 sample、12 block、每block全部head",
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "coverage": {
            "rows": len(rows),
            "samples": len(sample_ids),
            "blocks": len(block_names),
            "tokens": len(rows) * 450,
            "active_outputs": sum(row["expected_outputs"] for row in rows),
            "ttb_slow_rows": sum(row["ttb_cycles"] > row["baseline_cycles"] for row in rows),
            "active_pair_min": min(row["active_pairs"] for row in rows),
            "active_pair_max": max(row["active_pairs"] for row in rows),
        },
        "canonical_invariants": [
            "Q0/Q1 count", "K0/K1 count", "Q0&K0/Q1&K1 overlap",
            "K0^K1 motion count", "H67 Q7 score", "zero-K类别", "gated-K输出数",
        ],
        "non_invariants": [
            "原始lane身份", "Q0/Q1 temporal overlap", "真实门级切换", "SAIF/功耗/能量",
        ],
        "source_profile": receipt(args.profile),
        "vector_file": receipt(vector_path),
        "rows": row_receipts,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS canonical rows={len(rows)} samples={len(sample_ids)} "
        f"slow={manifest['coverage']['ttb_slow_rows']} vector={vector_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
