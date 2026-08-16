#!/usr/bin/env python3
"""生成 Local5 T450 双窗口独立整数金参考。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from generate_local5_masked_integer_vectors import masked_shiftmax_q17, score_q7


HEAD_DIM = 32
OUT_DIM = 4
MAX_DEST = 450


def weight(lane: int, out_idx: int) -> int:
    return ((lane * 7 + out_idx * 11) % 15) - 7


def make_destination(
    rng: random.Random,
    tag: int,
    dest: int,
    mask: int,
    last: int,
    directed: str | None = None,
) -> tuple[int, int, int, list[int], int, int]:
    if directed == "ones":
        q_bits = 0xFFFF_FFFF
        k_bits = [0xFFFF_FFFF] * 5
    elif directed == "zeros":
        q_bits = 0
        k_bits = [0, 0x00FF_FFFF, 0xAAAA_AAAA, 0x5555_5555, 1]
    else:
        q_bits = rng.getrandbits(32)
        k_bits = [rng.getrandbits(32) for _ in range(5)]
    return tag, dest, q_bits, k_bits, mask, last


def accumulate(
    destinations: list[tuple[int, int, int, list[int], int, int]],
) -> list[list[int]]:
    acc = [[0 for _ in range(OUT_DIM)] for _ in range(MAX_DEST)]
    for _, dest, q_bits, k_bits, valid_mask, _ in destinations:
        valid = [(valid_mask >> cand) & 1 for cand in range(5)]
        scores = [score_q7(q_bits, value) for value in k_bits]
        gates = masked_shiftmax_q17(scores, valid)
        for cand, gate in enumerate(gates):
            if not valid[cand] or gate == 0:
                continue
            for lane in range(HEAD_DIM):
                if (k_bits[cand] >> lane) & 1:
                    for out_idx in range(OUT_DIM):
                        acc[dest][out_idx] += gate * weight(lane, out_idx)
    return acc


def build_runs(seed: int) -> list[list[tuple[int, int, int, list[int], int, int]]]:
    rng = random.Random(seed)
    run0 = [
        make_destination(rng, 0x1000, 449, 0b00001, 0, "ones"),
        make_destination(rng, 0x1001, 0, 0b11111, 0, "zeros"),
        make_destination(rng, 0x1002, 225, 0b10111, 0),
        make_destination(rng, 0x1003, 449, 0b11111, 0),
        make_destination(rng, 0x1004, 17, 0b01101, 0),
        make_destination(rng, 0x1005, 448, 0b11011, 1),
    ]
    run1 = [
        make_destination(rng, 0x2000, 1, 0b11111, 0),
        make_destination(rng, 0x2001, 449, 0b00111, 0, "zeros"),
        make_destination(rng, 0x2002, 225, 0b10001, 1),
    ]
    return [run0, run1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x4505)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = build_runs(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as handle:
        handle.write(f"{HEAD_DIM} {OUT_DIM} {MAX_DEST} {len(runs)}\n")
        for lane in range(HEAD_DIM):
            for out_idx in range(OUT_DIM):
                handle.write(f"{lane} {out_idx} {weight(lane, out_idx)}\n")
        for destinations in runs:
            handle.write(f"{len(destinations)}\n")
            for tag, dest, q_bits, k_bits, mask, last in destinations:
                fields = [f"{tag:04x}", str(dest), f"{q_bits:08x}"]
                fields.extend(f"{value:08x}" for value in k_bits)
                fields.extend((f"{mask:02x}", str(last)))
                handle.write(" ".join(fields) + "\n")
            golden = accumulate(destinations)
            for dest in range(MAX_DEST):
                for out_idx in range(OUT_DIM):
                    handle.write(f"{dest} {out_idx} {golden[dest][out_idx]}\n")
    print(args.output, len(runs), MAX_DEST * OUT_DIM * len(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
