#!/usr/bin/env python3
"""Generate an independent Local5 Q/K-to-Acc32 integer oracle."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


HEIGHT = 3
WIDTH = 6
TIME_PLANES = 2
HEAD_DIM = 32
OUT_DIM = 2
ROLES = 5
MASK32 = 0xFFFF_FFFF
EXP2_LUT = (
    256, 245, 234, 224, 215, 205, 196, 188,
    181, 173, 165, 158, 152, 145, 139, 133,
)


def rne_pow2(value: int, shift: int) -> int:
    quotient = value >> shift
    remainder = value - (quotient << shift)
    half = 1 << (shift - 1)
    return quotient + int(
        remainder > half or (remainder == half and (quotient & 1))
    )


def score_q7(q_bits: int, k_bits: int) -> int:
    overlap = (q_bits & k_bits).bit_count()
    same_zero = ((~q_bits) & (~k_bits) & MASK32).bit_count()
    return rne_pow2(overlap * 256 + same_zero * 4, 6)


def shiftmax5_q17(scores: list[int], valid_mask: int) -> list[int]:
    valid = [(valid_mask >> role) & 1 for role in range(ROLES)]
    if not any(valid):
        return [0] * ROLES
    row_max = max(score for score, enabled in zip(scores, valid) if enabled)
    exp_values = []
    for score, enabled in zip(scores, valid):
        if not enabled:
            exp_values.append(0)
            continue
        delta = row_max - score
        integer_shift = min(delta >> 7, 8)
        fraction_index = min(((delta & 127) + 7) >> 3, 15)
        exp_values.append(EXP2_LUT[fraction_index] >> integer_shift)
    denominator_shift = max(0, (sum(exp_values) - 1).bit_length())
    return [
        min(rne_pow2(exp_value * 128, denominator_shift), 256)
        if enabled else 0
        for exp_value, enabled in zip(exp_values, valid)
    ]


def rotate_left_one(value: int) -> int:
    return ((value << 1) | (value >> 31)) & MASK32


def source_k(plane: int, y: int, x: int) -> int:
    base = (0x1357_9BDF ^ (plane << 29)) & MASK32
    index = y * WIDTH + x + 1
    return rotate_left_one(base) ^ ((0x0001_0101 * index) & MASK32)


def query_bits(plane: int, y: int, x: int) -> int:
    index = plane * HEIGHT * WIDTH + y * WIDTH + x
    return (0xA5C3_5A3C ^ ((0x0102_0408 * index) & MASK32)) & MASK32


def weight_value(lane: int, out: int) -> int:
    return (((lane + 1) * 37 + (out + 1) * 53 + lane * out * 11) % 127) - 63


def role_source(y: int, x: int, role: int) -> tuple[int, int]:
    if role == 1:
        return y - 1, x
    if role == 2:
        return y + 1, x
    if role == 3:
        return y, x - 1
    if role == 4:
        return y, x + 1
    return y, x


def build_values(
    seed: int | None,
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    source_values = [
        [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        for _ in range(TIME_PLANES)
    ]
    query_values = [
        [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        for _ in range(TIME_PLANES)
    ]
    rng = random.Random(seed) if seed is not None else None
    for plane in range(TIME_PLANES):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if rng is None:
                    source_values[plane][y][x] = source_k(plane, y, x)
                    query_values[plane][y][x] = query_bits(plane, y, x)
                else:
                    source_values[plane][y][x] = rng.getrandbits(HEAD_DIM)
                    query_values[plane][y][x] = rng.getrandbits(HEAD_DIM)
    if rng is not None:
        # Keep directed extremes inside the randomized, boundary-aware tile.
        source_values[0][0][0] = 0
        source_values[0][0][1] = MASK32
        query_values[0][0][0] = 0
        query_values[0][0][1] = MASK32
    return source_values, query_values


def build_destination(
    plane: int,
    y: int,
    x: int,
    source_values: list[list[list[int]]],
    query_values: list[list[list[int]]],
) -> dict[str, object]:
    k_values = [0] * ROLES
    valid_mask = 0
    for role in range(ROLES):
        source_y, source_x = role_source(y, x, role)
        if 0 <= source_y < HEIGHT and 0 <= source_x < WIDTH:
            valid_mask |= 1 << role
            k_values[role] = source_values[plane][source_y][source_x]

    if plane == 1:
        if (y, x) == (1, 1):
            valid_mask &= ~(1 << 0)
        if (y, x) == (2, 1):
            valid_mask &= ~(1 << 1)
        if (y, x) == (1, 2):
            valid_mask &= ~(1 << 3)

    q_value = query_values[plane][y][x]
    scores = [score_q7(q_value, k_value) for k_value in k_values]
    gates = shiftmax5_q17(scores, valid_mask)
    return {
        "q": q_value,
        "k": k_values,
        "valid_mask": valid_mask,
        "gates": gates,
    }


def build_oracle(
    seed: int | None,
) -> tuple[list[dict[str, object]], list[list[list[list[int]]]], int, int]:
    source_values, query_values = build_values(seed)
    destinations: list[dict[str, object]] = []
    relation: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = (
        defaultdict(list)
    )

    for plane in range(TIME_PLANES):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                destination = build_destination(
                    plane, y, x, source_values, query_values
                )
                destinations.append(destination)
                valid_mask = int(destination["valid_mask"])
                gates = list(destination["gates"])
                for role in range(ROLES):
                    if not ((valid_mask >> role) & 1):
                        continue
                    source_y, source_x = role_source(y, x, role)
                    relation[(plane, source_y, source_x)].append(
                        (plane, y, x, gates[role])
                    )

    accumulators = [
        [
            [[0 for _ in range(OUT_DIM)] for _ in range(WIDTH)]
            for _ in range(HEIGHT)
        ]
        for _ in range(TIME_PLANES)
    ]
    term_count = 0
    update_count = 0
    for plane in range(TIME_PLANES):
        for source_y in range(HEIGHT):
            for source_x in range(WIDTH):
                source_bits = source_values[plane][source_y][source_x]
                relations = relation[(plane, source_y, source_x)]
                for lane in range(HEAD_DIM):
                    if not ((source_bits >> lane) & 1):
                        continue
                    by_gate: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
                    for dest_plane, dest_y, dest_x, gate in relations:
                        if gate != 0:
                            by_gate[gate].append((dest_plane, dest_y, dest_x))
                    for gate, destinations_for_term in sorted(by_gate.items()):
                        term_count += 1
                        for dest_plane, dest_y, dest_x in destinations_for_term:
                            update_count += 1
                            for out in range(OUT_DIM):
                                accumulators[dest_plane][dest_y][dest_x][out] += (
                                    gate * weight_value(lane, out)
                                )
    return destinations, accumulators, term_count, update_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--seed", type=lambda value: int(value, 0))
    args = parser.parse_args()

    destinations, accumulators, terms, updates = build_oracle(args.seed)
    if len(destinations) != HEIGHT * WIDTH * TIME_PLANES:
        raise AssertionError("destination count mismatch")
    if args.seed is None and (terms != 1498 or updates != 2332):
        raise AssertionError(
            f"independent work ledger mismatch terms={terms} updates={updates}"
        )

    args.inputs.parent.mkdir(parents=True, exist_ok=True)
    args.expected.parent.mkdir(parents=True, exist_ok=True)
    with args.inputs.open("w", encoding="ascii") as handle:
        index = 0
        for plane in range(TIME_PLANES):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    destination = destinations[index]
                    index += 1
                    fields = [str(plane), str(y), str(x), f"{int(destination['q']):08x}"]
                    fields.extend(f"{value:08x}" for value in destination["k"])
                    fields.append(f"{int(destination['valid_mask']):02x}")
                    handle.write(" ".join(fields) + "\n")

    with args.expected.open("w", encoding="ascii") as handle:
        for plane in range(TIME_PLANES):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    for out in range(OUT_DIM):
                        handle.write(
                            f"{plane} {y} {x} {out} "
                            f"{accumulators[plane][y][x][out]}\n"
                        )

    print(
        "PASS Local5 independent fullchain oracle "
        f"inputs={len(destinations)} acc32={TIME_PLANES * HEIGHT * WIDTH * OUT_DIM} "
        f"terms={terms} updates={updates} seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
