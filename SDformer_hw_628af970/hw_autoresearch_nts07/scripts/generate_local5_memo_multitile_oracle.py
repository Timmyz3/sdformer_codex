#!/usr/bin/env python3
"""生成 Local5 T450 三头三输出 tile 的 relation-memo 独立金参考。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from generate_local5_fullchain_oracle import score_q7, shiftmax5_q17


HEIGHT = 15
WIDTH = 15
TIME_PLANES = 2
HEAD_DIM = 32
HEADS = 3
OUTPUT_TILES = 3
OUT_DIM = 32
ROLES = 5


def weight_value(head: int, tile: int, lane: int, out: int) -> int:
    value = (
        (head + 1) * 29
        + (tile + 1) * 43
        + (lane + 1) * 37
        + (out + 1) * 53
        + lane * out * 11
    )
    return value % 127 - 63


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


def candidate_mask(plane: int, y: int, x: int) -> int:
    mask = 0
    for role in range(ROLES):
        sy, sx = role_source(y, x, role)
        if 0 <= sy < HEIGHT and 0 <= sx < WIDTH:
            mask |= 1 << role
    if (plane + 3 * y + 5 * x) % 37 == 0:
        mask &= ~(1 << 0)
    if (7 * plane + y + 11 * x) % 53 == 0:
        mask &= ~(1 << 4)
    return mask or 1


def sparse_sources(head: int) -> dict[tuple[int, int, int], int]:
    if head == 0:
        coords = [
            (0, 3, 3),
            (0, 5, 9),
            (0, 10, 4),
            (1, 4, 11),
            (1, 8, 6),
            (1, 11, 10),
        ]
    elif head == 2:
        coords = [
            (0, 2, 7),
            (0, 4, 4),
            (0, 7, 12),
            (0, 12, 8),
            (1, 3, 10),
            (1, 6, 2),
            (1, 9, 9),
            (1, 12, 4),
        ]
    else:
        return {}
    return {
        coord: 1 << ((index * 7 + head * 3) % HEAD_DIM)
        for index, coord in enumerate(coords)
    }


def build_head(head: int, seed: int) -> tuple[list[dict[str, object]], dict]:
    rng = random.Random(seed + head * 0x10001)
    sparse = sparse_sources(head)
    sources = []
    queries = []
    for plane in range(TIME_PLANES):
        source_plane = []
        query_plane = []
        for y in range(HEIGHT):
            source_row = []
            query_row = []
            for x in range(WIDTH):
                if head == 1:
                    source = 0xFFFF_FFFF
                else:
                    source = sparse.get((plane, y, x), 0)
                source_row.append(source)
                query_row.append(rng.getrandbits(HEAD_DIM))
            source_plane.append(source_row)
            query_plane.append(query_row)
        sources.append(source_plane)
        queries.append(query_plane)

    inputs: list[dict[str, object]] = []
    relation: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = (
        defaultdict(list)
    )
    for plane in range(TIME_PLANES):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                mask = candidate_mask(plane, y, x)
                k_values = [0] * ROLES
                for role in range(ROLES):
                    sy, sx = role_source(y, x, role)
                    # Runtime invalid-mask suppresses score/gate participation,
                    # but the self K payload remains the source event consumed
                    # after relation transpose.
                    if 0 <= sy < HEIGHT and 0 <= sx < WIDTH:
                        k_values[role] = sources[plane][sy][sx]
                q_value = queries[plane][y][x]
                gates = shiftmax5_q17(
                    [score_q7(q_value, value) for value in k_values],
                    mask,
                )
                inputs.append(
                    {"q": q_value, "k": k_values, "valid_mask": mask}
                )
                for role, gate in enumerate(gates):
                    if not ((mask >> role) & 1) or gate == 0:
                        continue
                    sy, sx = role_source(y, x, role)
                    relation[(plane, sy, sx)].append((plane, y, x, gate))

    active_records = 0
    term_count = 0
    update_count = 0
    per_tile = []
    for tile in range(OUTPUT_TILES):
        acc = [
            [
                [[0 for _ in range(OUT_DIM)] for _ in range(WIDTH)]
                for _ in range(HEIGHT)
            ]
            for _ in range(TIME_PLANES)
        ]
        for plane in range(TIME_PLANES):
            for sy in range(HEIGHT):
                for sx in range(WIDTH):
                    source = sources[plane][sy][sx]
                    if source and relation[(plane, sy, sx)] and tile == 0:
                        active_records += 1
                    for lane in range(HEAD_DIM):
                        if not ((source >> lane) & 1):
                            continue
                        by_gate: dict[int, list[tuple[int, int, int]]] = (
                            defaultdict(list)
                        )
                        for dest_plane, dy, dx, gate in relation[(plane, sy, sx)]:
                            by_gate[gate].append((dest_plane, dy, dx))
                        for gate, destinations in by_gate.items():
                            if tile == 0:
                                term_count += 1
                                update_count += len(destinations)
                            for dest_plane, dy, dx in destinations:
                                for out in range(OUT_DIM):
                                    acc[dest_plane][dy][dx][out] += (
                                        gate * weight_value(head, tile, lane, out)
                                    )
        per_tile.append(acc)

    return inputs, {
        "expected": per_tile,
        "active_records": active_records,
        "terms": term_count,
        "updates": update_count,
        "service_cycles": 15 + term_count,
        "resident": 15 + term_count < 450,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x5A052026)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    combined = [
        [
            [
                [[0 for _ in range(OUT_DIM)] for _ in range(WIDTH)]
                for _ in range(HEIGHT)
            ]
            for _ in range(TIME_PLANES)
        ]
        for _ in range(OUTPUT_TILES)
    ]
    metadata = {"seed": args.seed, "heads": []}
    for head in range(HEADS):
        inputs, stats = build_head(head, args.seed)
        input_path = args.out_dir / f"head{head}_inputs.txt"
        with input_path.open("w", encoding="ascii") as handle:
            index = 0
            for plane in range(TIME_PLANES):
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        item = inputs[index]
                        index += 1
                        fields = [
                            str(plane),
                            str(y),
                            str(x),
                            f"{int(item['q']):08x}",
                        ]
                        fields.extend(f"{value:08x}" for value in item["k"])
                        fields.append(f"{int(item['valid_mask']):02x}")
                        handle.write(" ".join(fields) + "\n")
        for tile in range(OUTPUT_TILES):
            for plane in range(TIME_PLANES):
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        for out in range(OUT_DIM):
                            combined[tile][plane][y][x][out] += stats["expected"][
                                tile
                            ][plane][y][x][out]
        metadata["heads"].append(
            {key: value for key, value in stats.items() if key != "expected"}
        )

    expected_path = args.out_dir / "expected_all_tiles.txt"
    with expected_path.open("w", encoding="ascii") as handle:
        for tile in range(OUTPUT_TILES):
            for plane in range(TIME_PLANES):
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        for out in range(OUT_DIM):
                            handle.write(
                                f"{tile} {plane} {y} {x} {out} "
                                f"{combined[tile][plane][y][x][out]}\n"
                            )
    metadata["resident_heads"] = [
        index for index, item in enumerate(metadata["heads"]) if item["resident"]
    ]
    metadata["expected_token_requests"] = 2250
    metadata["expected_memo_hits"] = 4
    metadata["expected_fallbacks"] = 2
    metadata["expected_replay_records"] = 2 * sum(
        metadata["heads"][index]["active_records"] for index in (0, 2)
    )
    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if metadata["resident_heads"] != [0, 2]:
        raise AssertionError(f"驻留构造失败: {metadata['resident_heads']}")
    print(
        "PASS Local5 memo multi-tile oracle "
        f"resident={metadata['resident_heads']} "
        f"records={[item['active_records'] for item in metadata['heads']]} "
        f"terms={[item['terms'] for item in metadata['heads']]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
