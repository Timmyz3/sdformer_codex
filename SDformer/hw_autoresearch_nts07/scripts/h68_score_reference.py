#!/usr/bin/env python3
"""Independent pre-centering score audit for H68 dyadic deployment."""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

from h67_score_reference import round_fraction_even


def software_score_q7(overlap: int, same_zero: int) -> int:
    return round_fraction_even(Fraction(4 * overlap, 1) + Fraction(same_zero, 16))


def rtl_score_q7(overlap: int, same_zero: int) -> int:
    quotient, remainder = divmod(same_zero, 16)
    integer_score = 4 * overlap + quotient
    if remainder > 8 or (remainder == 8 and integer_score % 2):
        integer_score += 1
    return integer_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-vectors", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=6801)
    args = parser.parse_args()

    tuple_mismatches = 0
    for overlap in range(33):
        for same_zero in range(33):
            tuple_mismatches += software_score_q7(overlap, same_zero) != rtl_score_q7(overlap, same_zero)

    rng = random.Random(args.seed)
    vector_mismatches = 0
    for _ in range(args.random_vectors):
        q_bits = rng.getrandbits(32)
        k_bits = rng.getrandbits(32)
        overlap = (q_bits & k_bits).bit_count()
        same_zero = ((~q_bits & ~k_bits) & 0xFFFFFFFF).bit_count()
        vector_mismatches += software_score_q7(overlap, same_zero) != rtl_score_q7(overlap, same_zero)

    fold_scores = {
        rtl_score_q7(0, 32 - q_active)
        for q_active in range(33)
    }
    result = {
        "pass": tuple_mismatches == 0 and vector_mismatches == 0,
        "tuple_cases": 33 * 33,
        "tuple_mismatches": tuple_mismatches,
        "random_vectors": args.random_vectors,
        "vector_mismatches": vector_mismatches,
        "fold_score_classes": sorted(fold_scores),
        "fold_score_class_count": len(fold_scores),
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text("\n".join([
        "# H68 中心化前定点分数审计", "",
        f"- 计数组合：`{result['tuple_cases']}`；不一致：`{tuple_mismatches}`。",
        f"- 随机向量：`{args.random_vectors}`；不一致：`{vector_mismatches}`。",
        f"- 零 K 可达分数类：`{sorted(fold_scores)}`，共 `{len(fold_scores)}` 类。",
        "- 本脚本只覆盖分数前端；整行hardware-order部署验证已经完成，见 `results/h67_h68_rtl_exact_valid825.md`。", "",
        f"结论：**{'通过' if result['pass'] else '失败'}**。",
    ]) + "\n", encoding="utf-8")
    print(md)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
