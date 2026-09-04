#!/usr/bin/env python3
"""生成 Local5 masked Shiftmax 的独立整数金参考向量。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


EXP2_LUT = (256, 245, 234, 224, 215, 205, 196, 188,
            181, 173, 165, 158, 152, 145, 139, 133)


def rne_pow2(value: int, shift: int) -> int:
    quotient = value >> shift
    remainder = value - (quotient << shift)
    half = 1 << (shift - 1)
    return quotient + int(
        remainder > half or (remainder == half and (quotient & 1))
    )


def score_q7(q_bits: int, k_bits: int) -> int:
    overlap = (q_bits & k_bits).bit_count()
    same_zero = ((~q_bits) & (~k_bits) & 0xFFFF_FFFF).bit_count()
    return rne_pow2(overlap * 256 + same_zero * 4, 6)


def masked_shiftmax_q17(scores: list[int], valid: list[int]) -> list[int]:
    if not any(valid):
        return [0] * len(scores)
    row_max = max(score for score, is_valid in zip(scores, valid) if is_valid)
    exp_values: list[int] = []
    for score, is_valid in zip(scores, valid):
        if not is_valid:
            exp_values.append(0)
            continue
        abs_delta = row_max - score
        integer_shift = min(abs_delta >> 7, 8)
        frac_index = min(((abs_delta & 127) + 7) >> 3, 15)
        exp_values.append(EXP2_LUT[frac_index] >> integer_shift)
    denominator_shift = max(0, (sum(exp_values) - 1).bit_length())
    gates = []
    for exp_value, is_valid in zip(exp_values, valid):
        if not is_valid:
            gates.append(0)
            continue
        scaled = exp_value * 128
        gate = rne_pow2(scaled, denominator_shift)
        gates.append(min(gate, 256))
    return gates


def make_vectors(count: int, seed: int) -> list[tuple[int, list[int], int, list[int], list[int]]]:
    rng = random.Random(seed)
    vectors = []

    # Directed ties-to-even case: q=0, popcount(k)=24 gives numerator 32,
    # exactly half of the Q7 divisor. Correct score is 0, not half-up 1.
    directed = [
        (0, [0x00FF_FFFF] * 5, 0b1_1111),
        (0, [0] * 5, 0b1_1111),
        (0xAAAA_AAAA, [0x5555_5555] * 5, 0b1_0101),
        (0xFFFF_FFFF, [0, 1, 2, 4, 8], 0b0_0000),
    ]
    for q_bits, k_bits, valid_mask in directed:
        valid = [(valid_mask >> index) & 1 for index in range(5)]
        scores = [score_q7(q_bits, value) for value in k_bits]
        gates = masked_shiftmax_q17(scores, valid)
        vectors.append((q_bits, k_bits, valid_mask, scores, gates))

    while len(vectors) < count:
        q_bits = rng.getrandbits(32)
        k_bits = [rng.getrandbits(32) for _ in range(5)]
        valid_mask = rng.randrange(32)
        valid = [(valid_mask >> index) & 1 for index in range(5)]
        scores = [score_q7(q_bits, value) for value in k_bits]
        gates = masked_shiftmax_q17(scores, valid)
        vectors.append((q_bits, k_bits, valid_mask, scores, gates))
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x66D5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as handle:
        for q_bits, k_bits, valid_mask, scores, gates in make_vectors(
            args.count, args.seed
        ):
            fields = [f"{q_bits:08x}"]
            fields.extend(f"{value:08x}" for value in k_bits)
            fields.append(f"{valid_mask:02x}")
            fields.extend(f"{value & 0xFFFF:04x}" for value in scores)
            fields.extend(f"{value:03x}" for value in gates)
            handle.write(" ".join(fields) + "\n")
    print(args.output, args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
