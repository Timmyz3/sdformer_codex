#!/usr/bin/env python3
"""Generate a synthetic full-depth Local5 T450 full-chain oracle."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from generate_local5_fullchain_oracle import score_q7, shiftmax5_q17


HEIGHT = 15
WIDTH = 15
TIME_PLANES = 2
HEAD_DIM = 32
ROLES = 5


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


def build(
    seed: int,
    out_dim: int,
) -> tuple[list[dict[str, object]], list[list[list[list[int]]]], int, int]:
    rng = random.Random(seed)
    source_values = [
        [
            [rng.getrandbits(HEAD_DIM) for _ in range(WIDTH)]
            for _ in range(HEIGHT)
        ]
        for _ in range(TIME_PLANES)
    ]
    query_values = [
        [
            [rng.getrandbits(HEAD_DIM) for _ in range(WIDTH)]
            for _ in range(HEIGHT)
        ]
        for _ in range(TIME_PLANES)
    ]
    source_values[0][0][0] = 0
    source_values[0][0][1] = 0xFFFF_FFFF
    query_values[0][0][0] = 0
    query_values[0][0][1] = 0xFFFF_FFFF

    inputs: list[dict[str, object]] = []
    relation: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = (
        defaultdict(list)
    )
    for plane in range(TIME_PLANES):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                valid_mask = 0
                k_values = [0] * ROLES
                for role in range(ROLES):
                    source_y, source_x = role_source(y, x, role)
                    if 0 <= source_y < HEIGHT and 0 <= source_x < WIDTH:
                        valid_mask |= 1 << role
                        k_values[role] = source_values[plane][source_y][source_x]

                # Exercise runtime invalid candidates beyond geometric borders.
                if (plane + y + x) % 29 == 0:
                    valid_mask &= ~(1 << 0)
                if (3 * plane + 5 * y + x) % 37 == 0:
                    valid_mask &= ~(1 << 4)
                if valid_mask == 0:
                    valid_mask = 1
                    k_values[0] = source_values[plane][y][x]

                q_value = query_values[plane][y][x]
                scores = [score_q7(q_value, k_value) for k_value in k_values]
                gates = shiftmax5_q17(scores, valid_mask)
                inputs.append(
                    {
                        "q": q_value,
                        "k": k_values,
                        "valid_mask": valid_mask,
                    }
                )
                for role, gate in enumerate(gates):
                    if not ((valid_mask >> role) & 1):
                        continue
                    source_y, source_x = role_source(y, x, role)
                    relation[(plane, source_y, source_x)].append(
                        (plane, y, x, gate)
                    )

    accumulators = [
        [
            [[0 for _ in range(out_dim)] for _ in range(WIDTH)]
            for _ in range(HEIGHT)
        ]
        for _ in range(TIME_PLANES)
    ]
    terms = 0
    updates = 0
    for plane in range(TIME_PLANES):
        for source_y in range(HEIGHT):
            for source_x in range(WIDTH):
                source_bits = source_values[plane][source_y][source_x]
                source_relations = relation[(plane, source_y, source_x)]
                for lane in range(HEAD_DIM):
                    if not ((source_bits >> lane) & 1):
                        continue
                    by_gate: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
                    for dest_plane, dest_y, dest_x, gate in source_relations:
                        if gate != 0:
                            by_gate[gate].append((dest_plane, dest_y, dest_x))
                    for gate, destinations in sorted(by_gate.items()):
                        terms += 1
                        for dest_plane, dest_y, dest_x in destinations:
                            updates += 1
                            for out in range(out_dim):
                                accumulators[dest_plane][dest_y][dest_x][out] += (
                                    gate * weight_value(lane, out)
                                )
    return inputs, accumulators, terms, updates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x4505_2026)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--out-dim", type=int, default=2)
    args = parser.parse_args()
    if args.out_dim < 2 or args.out_dim > 256 or args.out_dim & (args.out_dim - 1):
        raise ValueError("out-dim必须是[2,256]范围内的2次幂")
    weight_columns = {
        tuple(weight_value(lane, out) for lane in range(HEAD_DIM))
        for out in range(args.out_dim)
    }
    if len(weight_columns) != args.out_dim:
        raise ValueError("输出维度权重列不唯一，无法观测地址混叠")

    inputs, expected, terms, updates = build(args.seed, args.out_dim)
    total = HEIGHT * WIDTH * TIME_PLANES
    if len(inputs) != total:
        raise AssertionError("T450 input count mismatch")
    args.inputs.parent.mkdir(parents=True, exist_ok=True)
    args.expected.parent.mkdir(parents=True, exist_ok=True)

    with args.inputs.open("w", encoding="ascii") as handle:
        index = 0
        for plane in range(TIME_PLANES):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    item = inputs[index]
                    index += 1
                    fields = [str(plane), str(y), str(x), f"{int(item['q']):08x}"]
                    fields.extend(f"{value:08x}" for value in item["k"])
                    fields.append(f"{int(item['valid_mask']):02x}")
                    handle.write(" ".join(fields) + "\n")

    with args.expected.open("w", encoding="ascii") as handle:
        for plane in range(TIME_PLANES):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    for out in range(args.out_dim):
                        handle.write(
                            f"{plane} {y} {x} {out} "
                            f"{expected[plane][y][x][out]}\n"
                        )

    print(
        "PASS Local5 synthetic T450 fullchain oracle "
        f"seed={args.seed} out_dim={args.out_dim} "
        f"inputs={total} acc32={total * args.out_dim} "
        f"terms={terms} updates={updates}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
