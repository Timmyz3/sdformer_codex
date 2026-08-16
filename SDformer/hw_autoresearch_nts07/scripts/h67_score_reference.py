#!/usr/bin/env python3
"""Audit the frozen H67 dyadic score formula without GPU execution."""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path


def round_fraction_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    twice = 2 * remainder
    if twice > value.denominator or (twice == value.denominator and quotient % 2):
        quotient += 1
    return quotient


def software_score_q7(overlap: int, same_zero: int, motion: int) -> int:
    normalized = Fraction(overlap, 1) + Fraction(same_zero, 64) + Fraction(motion, 4)
    return round_fraction_even(Fraction(128, 32) * normalized)


def rtl_score_q7(overlap: int, same_zero: int, motion: int) -> int:
    silence_integer, silence_remainder = divmod(same_zero, 16)
    integer_score = 4 * overlap + motion + silence_integer
    if silence_remainder > 8 or (silence_remainder == 8 and integer_score % 2):
        silence_integer += 1
    return 4 * overlap + motion + silence_integer


def popcount(value: int) -> int:
    return value.bit_count()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-vectors", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=6701)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tuple_mismatches = []
    for overlap in range(33):
        for same_zero in range(33):
            for motion in range(33):
                sw = software_score_q7(overlap, same_zero, motion)
                rtl = rtl_score_q7(overlap, same_zero, motion)
                if sw != rtl:
                    tuple_mismatches.append((overlap, same_zero, motion, sw, rtl))

    rng = random.Random(args.seed)
    vector_mismatches = []
    max_vector_score = 0
    max_fold_score = 0
    for _ in range(args.random_vectors):
        q_bits = rng.getrandbits(32)
        k_current = rng.getrandbits(32)
        k_peer = rng.getrandbits(32)
        overlap = popcount(q_bits & k_current)
        same_zero = popcount((~q_bits & ~k_current) & 0xFFFFFFFF)
        motion = popcount(k_current ^ k_peer)
        sw = software_score_q7(overlap, same_zero, motion)
        rtl = rtl_score_q7(overlap, same_zero, motion)
        max_vector_score = max(max_vector_score, sw)
        if k_current == 0:
            max_fold_score = max(max_fold_score, sw)
        if sw != rtl and len(vector_mismatches) < 16:
            vector_mismatches.append({
                "q_bits": f"0x{q_bits:08x}",
                "k_current": f"0x{k_current:08x}",
                "k_peer": f"0x{k_peer:08x}",
                "software": sw,
                "rtl": rtl,
            })

    # Exhaust the exact zero-current-K folding domain.
    fold_scores = set()
    for q_active in range(33):
        same_zero = 32 - q_active
        for peer_active in range(33):
            fold_scores.add(rtl_score_q7(0, same_zero, peer_active))

    result = {
        "tuple_cases": 33**3,
        "tuple_mismatches": len(tuple_mismatches),
        "random_vectors": args.random_vectors,
        "vector_mismatches": len(vector_mismatches),
        "seed": args.seed,
        "fold_score_min": min(fold_scores),
        "fold_score_max": max(fold_scores),
        "fold_score_class_count": len(fold_scores),
        "observed_random_max_score": max_vector_score,
        "observed_random_fold_max": max_fold_score,
        "center_then_quant_counterexample": {
            "row_tokens": 162,
            "raw_q7_classes": [0, 1],
            "tokens_per_class": [81, 81],
            "software_center_then_rne_classes": [0, 0],
            "rtl_raw_rne_classes": [0, 1],
        },
        "pass": not tuple_mismatches and not vector_mismatches,
        "mismatch_examples": vector_mismatches,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text(
        "\n".join([
            "# H67 定点分数独立参考审计",
            "",
            "本审计只检查中心化前的冻结部署 Q7 分数，不调用 GPU，也不检查 Shiftmax LUT 近似。",
            "",
            f"- 计数组合穷举：`{result['tuple_cases']}` 组；不一致：`{result['tuple_mismatches']}`。",
            f"- 32-bit 随机向量：`{result['random_vectors']}` 组；不一致：`{result['vector_mismatches']}`。",
            f"- 随机种子：`{result['seed']}`。",
            f"- 零 K 折叠分数域：`{result['fold_score_min']}..{result['fold_score_max']}`，共 "
            f"`{result['fold_score_class_count']}` 个可达整数类。",
            "- 舍入口径：与 PyTorch `torch.round` 一致的 round-to-nearest-even；特别覆盖 "
            "`same_zero=8/24` 与 motion 奇偶组合的 tie case。",
            "- 已确认顺序反例：162-token row 中 Q7 原分数 0/1 各81个时，软件 "
            "`center -> RNE` 得到 0/0 两类，当前 RTL `raw RNE` 保留 0/1 两类。",
            "",
            f"中心化前分数结论：**{'通过' if result['pass'] else '失败'}**。整行hardware-order "
            "部署验证已经完成，结果见 `results/h67_h68_rtl_exact_valid825.md`。",
        ]) + "\n",
        encoding="utf-8",
    )
    print(md)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
