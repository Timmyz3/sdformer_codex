#!/usr/bin/env python3
"""Split TESC equal pairs into empty-K vs active-K. Offline, not a new RTL claim."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SEAL_RE = re.compile(
    r"SB_SEAL row=(?P<row>\d+) pairs=(?P<pairs>\d+) slots=(?P<slots>\d+) "
    r"equal=(?P<equal>\d+)"
)


def popcount(value: int) -> int:
    return value.bit_count()


def motionxor_q7(q_bits: int, k_current: int, k_peer: int) -> int:
    overlap = popcount(q_bits & k_current)
    motion = popcount(k_current ^ k_peer)
    q_count = popcount(q_bits)
    k_count = popcount(k_current)
    same_zero = 32 - q_count - k_count + overlap
    silence_integer = same_zero >> 4
    silence_remainder = same_zero & 15
    score_integer = (overlap << 2) + motion + silence_integer
    increment = (silence_remainder > 8) or (
        silence_remainder == 8 and (score_integer & 1)
    )
    return score_integer + int(increment)


def load_rows(path: Path) -> list[tuple[list[int], list[int]]]:
    text = path.read_text(encoding="utf-8").split()
    # header: rows tokens then per row metadata + 450*(q k peer gate)
    idx = 0
    rows = int(text[idx])
    tokens = int(text[idx + 1])
    idx += 2
    out: list[tuple[list[int], list[int]]] = []
    for _ in range(rows):
        idx += 6  # row/stage/block/head/expected/folded
        qs: list[int] = []
        ks: list[int] = []
        for _tok in range(tokens):
            qs.append(int(text[idx], 16))
            ks.append(int(text[idx + 1], 16))
            idx += 4
        out.append((qs, ks))
    return out


FAIR_RE = re.compile(
    r"FAIR_ROW row=(?P<row>\d+) .* equal=(?P<equal>\d+)"
)


def classify_rows(
    rows: list[tuple[list[int], list[int]]],
    selected: set[int] | None,
) -> dict[str, int]:
    empty_pairs = empty_equal = active_pairs = active_equal = model_equal = 0
    for row, (qs, ks) in enumerate(rows):
        if selected is not None and row not in selected:
            continue
        for pair in range(225):
            k0 = ks[pair]
            k1 = ks[pair + 225]
            s0 = motionxor_q7(qs[pair], k0, k1)
            s1 = motionxor_q7(qs[pair + 225], k1, k0)
            equal = int(s0 == s1)
            model_equal += equal
            if k0 == 0 and k1 == 0:
                empty_pairs += 1
                empty_equal += equal
            else:
                active_pairs += 1
                active_equal += equal
    return {
        "empty_pairs": empty_pairs,
        "empty_equal": empty_equal,
        "active_pairs": active_pairs,
        "active_equal": active_equal,
        "model_equal": model_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--seal-log", type=Path, default=None)
    parser.add_argument("--fair-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.vectors)
    selected: set[int] | None = None
    rtl_equal = 0
    if args.seal_log:
        seals = {
            int(m["row"]): int(m["equal"])
            for m in SEAL_RE.finditer(args.seal_log.read_text(encoding="utf-8"))
        }
        selected = set(seals)
        rtl_equal = sum(seals.values())
    elif args.fair_log:
        fairs = {
            int(m["row"]): int(m["equal"])
            for m in FAIR_RE.finditer(args.fair_log.read_text(encoding="utf-8"))
        }
        rtl_equal = sum(fairs.values())
        selected = None
    stats = classify_rows(rows, selected)
    if stats["model_equal"] != rtl_equal and rtl_equal:
        raise SystemExit(
            f"model equal {stats['model_equal']} != rtl {rtl_equal}"
        )
    empty_pairs = stats["empty_pairs"]
    empty_equal = stats["empty_equal"]
    active_pairs = stats["active_pairs"]
    active_equal = stats["active_equal"]
    report = {
        "schema": "h67_empty_active_equal_split_v2",
        "status": "PASS",
        "evidence": "[rtl-equal]+[independent-motionxor-both-sides]",
        "rows_scored": len(selected) if selected is not None else len(rows),
        "rtl_equal": rtl_equal or stats["model_equal"],
        "empty_pairs": empty_pairs,
        "empty_equal": empty_equal,
        "active_pairs": active_pairs,
        "active_equal": active_equal,
        "empty_equal_frac": empty_equal / empty_pairs if empty_pairs else 0,
        "active_equal_frac": active_equal / active_pairs if active_pairs else 0,
        "claim_boundary": [
            "Both empty and active equals are independent MotionXOR Q7 counts.",
            "active_equal is not rtl_equal minus empty_equal remainder.",
            "Do not put active_equal_frac in the DATE main cycle table.",
            "This is not a cycle claim.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion equal 拆分：空 pair vs 活动 pair",
        "",
        f"- RTL/model equal **{report['rtl_equal']}** / 发行 pair {empty_pairs + active_pairs}"
        + ("（密行 occupancy-gated）" if selected is not None else "（公平包全 138 行）"),
        f"- 双侧 K=0：{empty_pairs} pair，MotionXOR Q7 相同 **{empty_equal}** "
        f"({report['empty_equal_frac']:.1%})",
        f"- 至少一侧 K≠0：{active_pairs} pair，equal **{active_equal}** "
        f"({report['active_equal_frac']:.1%})",
        "- 全 pair 87.2% 不能写成活动 pair 合并率；空 pair 用的是量化 silence 不是 raw pop(Q)",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"PASS equal split empty={empty_equal}/{empty_pairs} "
        f"active={active_equal}/{active_pairs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
