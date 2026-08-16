#!/usr/bin/env python3
"""TTX ZAF-Shiftmax 与 FGK 的纯 Python 随机等价性参考模型。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


FRACTION_LUT = (256, 245, 234, 224, 215, 205, 196, 188, 181, 173, 165, 158, 152, 145, 139, 133)


def popcount(value: int) -> int:
    return value.bit_count()


def score_q7(q_bits: int, k_bits: int, head_dim: int = 32, alpha0_q8: int = 5) -> int:
    q_active = popcount(q_bits)
    k_active = popcount(k_bits)
    overlap = popcount(q_bits & k_bits)
    same_zero = head_dim - q_active - k_active + overlap
    numerator_q8 = overlap * 256 + same_zero * alpha0_q8
    shift = 8 + (head_dim.bit_length() - 1) - 7
    return (numerator_q8 + (1 << (shift - 1))) >> shift


def zero_k_class_score_q7(q_active: int, head_dim: int = 32, alpha0_q8: int = 5) -> int:
    numerator_q8 = (head_dim - q_active) * alpha0_q8
    shift = 8 + (head_dim.bit_length() - 1) - 7
    return (numerator_q8 + (1 << (shift - 1))) >> shift


def exp2_q8(delta_q7: int) -> int:
    if delta_q7 >= 0:
        return 256
    abs_delta = -delta_q7
    integer_shift = min(8, abs_delta >> 7)
    fraction_index = (abs_delta >> 3) & 0xF
    if abs_delta & 0x7:
        fraction_index = min(15, fraction_index + 1)
    return FRACTION_LUT[fraction_index] >> integer_shift


def ceil_log2(value: int) -> int:
    return max(0, (max(1, value) - 1).bit_length())


def dense_row(q_row: list[int], k_row: list[int], preserve_mean: bool = True) -> tuple[list[int], int]:
    scores = [score_q7(q_bits, k_bits) for q_bits, k_bits in zip(q_row, k_row, strict=True)]
    row_max = max(scores)
    numerators = [exp2_q8(score - row_max) for score in scores]
    row_sum = sum(numerators)
    shift = ceil_log2(row_sum)
    token_scale = len(scores) if preserve_mean else 1
    gates = [min(255, (numerator * 255 * token_scale) >> shift) for numerator in numerators]
    return gates, row_sum


def folded_row(q_row: list[int], k_row: list[int], head_dim: int = 32) -> tuple[dict[int, int], int, int, int]:
    scores = [score_q7(q_bits, k_bits, head_dim) for q_bits, k_bits in zip(q_row, k_row, strict=True)]
    row_max = max(scores)
    histogram = [0] * (head_dim + 1)
    active_entries: list[tuple[int, int]] = []
    for token_idx, (q_bits, k_bits, score) in enumerate(zip(q_row, k_row, scores, strict=True)):
        if k_bits == 0:
            histogram[popcount(q_bits)] += 1
        else:
            active_entries.append((token_idx, score))

    row_sum = sum(exp2_q8(score - row_max) for _, score in active_entries)
    nonempty_classes = 0
    for q_active, multiplicity in enumerate(histogram):
        if multiplicity:
            nonempty_classes += 1
            class_score = zero_k_class_score_q7(q_active, head_dim)
            row_sum += multiplicity * exp2_q8(class_score - row_max)

    shift = ceil_log2(row_sum)
    gates = {
        token_idx: min(255, (exp2_q8(score - row_max) * 255 * len(scores)) >> shift)
        for token_idx, score in active_entries
    }
    return gates, row_sum, len(active_entries), nonempty_classes


def random_sparse_bits(rng: random.Random, head_dim: int, probability: float) -> int:
    bits = 0
    for bit_idx in range(head_dim):
        if rng.random() < probability:
            bits |= 1 << bit_idx
    return bits


def run_reference(rows: int, tokens: int, seed: int) -> dict[str, float | int]:
    rng = random.Random(seed)
    total_active_entries = 0
    total_fold_classes = 0
    total_folded_tokens = 0
    total_dense_exp = 0
    total_fold_exp = 0

    for _ in range(rows):
        q_probability = rng.uniform(0.0003, 0.03)
        k_probability = rng.uniform(0.001, 0.04)
        q_row = [random_sparse_bits(rng, 32, q_probability) for _ in range(tokens)]
        k_row = [random_sparse_bits(rng, 32, k_probability) for _ in range(tokens)]

        dense_gates, dense_sum = dense_row(q_row, k_row)
        folded_gates, folded_sum, active_entries, fold_classes = folded_row(q_row, k_row)
        if dense_sum != folded_sum:
            raise AssertionError(f"Shiftmax 分母不一致: dense={dense_sum}, folded={folded_sum}")
        for token_idx, folded_gate in folded_gates.items():
            if dense_gates[token_idx] != folded_gate:
                raise AssertionError(
                    f"token {token_idx} gate 不一致: dense={dense_gates[token_idx]}, folded={folded_gate}"
                )

        weights = [rng.randint(-127, 127) for _ in range(32)]
        k_bits = random_sparse_bits(rng, 32, k_probability)
        gate = rng.randint(0, 255)
        threshold = rng.randint(1, 255)
        dense_accum = sum(weights[idx] * ((k_bits >> idx) & 1) * gate * threshold for idx in range(32))
        factorized_accum = sum(weights[idx] for idx in range(32) if (k_bits >> idx) & 1) * gate * threshold
        if dense_accum != factorized_accum:
            raise AssertionError("FGK late-gate 代数等价性失败")

        total_active_entries += active_entries
        total_fold_classes += fold_classes
        total_folded_tokens += tokens - active_entries
        total_dense_exp += 2 * tokens
        total_fold_exp += 2 * active_entries + fold_classes

    total_tokens = rows * tokens
    return {
        "rows": rows,
        "tokens_per_row": tokens,
        "seed": seed,
        "dense_vs_folded_mismatches": 0,
        "fgk_mismatches": 0,
        "k_zero_token_ratio": total_folded_tokens / total_tokens,
        "average_active_entries": total_active_entries / rows,
        "average_fold_classes": total_fold_classes / rows,
        "dense_exp_transactions": total_dense_exp,
        "zaf_exp_transactions": total_fold_exp,
        "exp_transaction_reduction": 1.0 - total_fold_exp / total_dense_exp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--tokens", type=int, default=162)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_reference(args.rows, args.tokens, args.seed)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
