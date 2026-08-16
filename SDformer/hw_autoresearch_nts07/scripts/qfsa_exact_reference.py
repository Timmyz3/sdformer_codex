#!/usr/bin/env python3
"""QFSA self-anchor与四方向exact residual的整数参考。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/qfsa_exact_reference_20260730"


def lane_raw(q_bit: int, k_bit: int) -> int:
    if q_bit and k_bit:
        return 64
    if not q_bit and not k_bit:
        return 1
    return 0


def alpha_xnor_raw16(q: int, k: int, lanes: int = 32) -> int:
    return sum(
        lane_raw((q >> lane) & 1, (k >> lane) & 1)
        for lane in range(lanes)
    )


def rne_div16(raw16: int) -> int:
    quotient, remainder = divmod(raw16, 16)
    return quotient + int(
        remainder > 8 or (remainder == 8 and (quotient & 1))
    )


def residual_raw16(
    q: int,
    k_anchor: int,
    k_target: int,
    lanes: int = 32,
) -> int:
    value = alpha_xnor_raw16(q, k_anchor, lanes)
    changed = k_anchor ^ k_target
    for lane in range(lanes):
        if (changed >> lane) & 1:
            q_bit = (q >> lane) & 1
            old_k = (k_anchor >> lane) & 1
            new_k = (k_target >> lane) & 1
            value += lane_raw(q_bit, new_k) - lane_raw(q_bit, old_k)
    return value


def direct_scores(q: int, candidates: list[int]) -> list[int]:
    return [rne_div16(alpha_xnor_raw16(q, k)) for k in candidates]


def qfsa_scores(q: int, candidates: list[int]) -> list[int]:
    if len(candidates) != 5:
        raise ValueError("QFSA requires self plus four directions")
    anchor = candidates[0]
    return [
        rne_div16(residual_raw16(q, anchor, target))
        for target in candidates
    ]


def finite_width_contract(
    lanes: int = 32,
    wave_width: int = 4,
) -> dict[str, int]:
    """给出任意事件顺序下的保守且可达位宽边界。"""

    if lanes <= 0 or wave_width <= 0:
        raise ValueError("lanes和wave_width必须为正")
    accumulator_max = 64 * lanes
    reducer_abs_max = 64 * wave_width
    return {
        "accumulator_min_raw16": 0,
        "accumulator_max_raw16": accumulator_max,
        "accumulator_signed_width": accumulator_max.bit_length() + 1,
        "wave_reducer_abs_max": reducer_abs_max,
        "wave_reducer_signed_width": reducer_abs_max.bit_length() + 1,
    }


def run_random(cases: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    mismatches = 0
    compared_scores = 0
    max_raw = 0
    for _ in range(cases):
        q = rng.getrandbits(32)
        candidates = [rng.getrandbits(32) for _ in range(5)]
        direct = direct_scores(q, candidates)
        qfsa = qfsa_scores(q, candidates)
        mismatches += sum(a != b for a, b in zip(direct, qfsa))
        compared_scores += 5
        max_raw = max(
            max_raw,
            *(alpha_xnor_raw16(q, k) for k in candidates),
        )
    return {
        "schema": "qfsa_exact_reference_v1",
        "cases": cases,
        "seed": seed,
        "compared_scores": compared_scores,
        "mismatches": mismatches,
        "max_raw16": max_raw,
        "score_contract": "RNE(raw16/16), raw16=64*n11+n00",
        "finite_width_contract": finite_width_contract(),
        "evidence": "[整数参考]，非RTL/PPA",
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# QFSA 整数等价参考",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| random cases | {report['cases']} |",
            f"| compared scores | {report['compared_scores']} |",
            f"| mismatch | {report['mismatches']} |",
            f"| max raw16 | {report['max_raw16']} |",
            f"| accumulator raw16范围 | [{report['finite_width_contract']['accumulator_min_raw16']}, {report['finite_width_contract']['accumulator_max_raw16']}] |",
            f"| accumulator signed位宽 | {report['finite_width_contract']['accumulator_signed_width']} bit |",
            f"| W4 reducer绝对值上界 | {report['finite_width_contract']['wave_reducer_abs_max']} |",
            f"| W4 reducer signed位宽 | {report['finite_width_contract']['wave_reducer_signed_width']} bit |",
            "",
            "比较路径：",
            "",
            "1. 五个候选分别 direct 计算 `raw16=64*n11+n00`，再 `RNE(raw16/16)`；",
            "2. self 候选计算一次 anchor raw，其余四方向只在 changed K lane 上累加",
            "   exact signed residual，最后每候选各执行一次相同 RNE。",
            "3. 对任意 residual 事件顺序，负事件不超过 anchor 中已有匹配项，",
            "   正事件不超过尚未匹配项，因此目标 raw accumulator 始终位于",
            "   `[0,2048]`；W4 单拍同方向归约位于 `[-256,256]`。",
            "",
            "该结果只证明整数算术等价，不证明 tagged compactor、cost router、",
            "ready/valid、Shiftmax5、周期、面积或功耗。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_random(args.cases, args.seed)
    if report["mismatches"]:
        raise RuntimeError("QFSA整数参考发现不等价")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
