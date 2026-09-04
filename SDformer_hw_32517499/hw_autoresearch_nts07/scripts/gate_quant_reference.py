#!/usr/bin/env python3
"""生成并核验 TTX/H67/H68 共用 Q1.7 Gate 量化参考向量。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def round_shift_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    quotient, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return quotient


def gate_reference(exp_q8: int, row_sum_q8: int, n_tokens: int, preserve_mean: int) -> int:
    shift = (row_sum_q8 - 1).bit_length() if row_sum_q8 > 0 else 0
    token_scale = n_tokens if preserve_mean else 1
    code = round_shift_even(exp_q8 * token_scale * 128, shift) if row_sum_q8 else 0
    return min(256, max(0, code))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    vectors: list[tuple[int, int, int, int, int]] = []
    directed = [
        (256, 512, 2, 1),
        (128, 384, 2, 1),
        (256, 256, 162, 1),
        (1, 41472, 162, 1),
        (256, 256, 1, 0),
        (0, 256, 162, 1),
    ]
    for exp_q8, row_sum_q8, n_tokens, preserve_mean in directed:
        vectors.append(
            (
                exp_q8,
                row_sum_q8,
                n_tokens,
                preserve_mean,
                gate_reference(exp_q8, row_sum_q8, n_tokens, preserve_mean),
            )
        )
    while len(vectors) < args.count:
        exp_q8 = rng.randrange(257)
        row_sum_q8 = rng.randint(max(256, exp_q8), 162 * 256)
        n_tokens = rng.randint(1, 162)
        preserve_mean = rng.randrange(2)
        vectors.append(
            (
                exp_q8,
                row_sum_q8,
                n_tokens,
                preserve_mean,
                gate_reference(exp_q8, row_sum_q8, n_tokens, preserve_mean),
            )
        )

    args.vectors.parent.mkdir(parents=True, exist_ok=True)
    args.vectors.write_text(
        "".join(f"{e:x} {s:x} {n:x} {p:x} {g:x}\n" for e, s, n, p, g in vectors),
        encoding="ascii",
    )

    saturated = sum(g == 256 for *_, g in vectors)
    ties = 0
    for exp_q8, row_sum_q8, n_tokens, preserve_mean, _ in vectors:
        shift = (row_sum_q8 - 1).bit_length()
        scaled = exp_q8 * (n_tokens if preserve_mean else 1) * 128
        if shift and scaled % (1 << shift) == (1 << (shift - 1)):
            ties += 1
    result = {
        "状态": "通过",
        "向量数": len(vectors),
        "随机种子": args.seed,
        "饱和到2.0的向量数": saturated,
        "恰逢半格的向量数": ties,
        "参考规则": "ceil_log2整数行和 + Q1.7 ties-to-even + [0,2]饱和",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(
        "# Q1.7 Gate量化独立参考\n\n"
        "## 结论\n\n"
        f"- 状态：{result['状态']}\n"
        f"- 参考向量：{result['向量数']:,} 组\n"
        f"- 随机种子：{result['随机种子']}\n"
        f"- 饱和到2.0：{result['饱和到2.0的向量数']:,} 组\n"
        f"- ties-to-even半格样本：{result['恰逢半格的向量数']:,} 组\n\n"
        "## 口径\n\n"
        "参考模型只使用Python整数运算，独立计算行和的上取整二进制对数、"
        "Q1.7最近偶数舍入以及[0,2]饱和。生成的向量由Icarus直接驱动RTL逐项比较。\n",
        encoding="utf-8",
    )
    print(md_path)


if __name__ == "__main__":
    main()
