#!/usr/bin/env python3
"""H67 Motion-Delta 与 Local5 RCSD 的无界整数 bit-exact 参考。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "results/dual_line_delta_reference_20260726.json"
)
MASK32 = (1 << 32) - 1


def rne_div16(value: int) -> int:
    """非负整数除 16，round-to-nearest ties-to-even。"""

    if value < 0:
        raise ValueError("RNE 输入必须非负")
    quotient, remainder = divmod(value, 16)
    if remainder > 8 or (remainder == 8 and (quotient & 1)):
        quotient += 1
    return quotient


def popcount(value: int) -> int:
    return (value & MASK32).bit_count()


def axnor_raw16(q_bits: int, k_bits: int) -> int:
    """返回 Local5/H67 alpha-XNOR Q7 舍入前的 1/16 numerator。"""

    q = q_bits & MASK32
    k = k_bits & MASK32
    overlap = popcount(q & k)
    q_count = popcount(q)
    k_count = popcount(k)
    return 65 * overlap + 32 - q_count - k_count


def local5_direct_q7(q_bits: int, k_bits: int) -> int:
    return rne_div16(axnor_raw16(q_bits, k_bits))


def local5_rcsd_q7(
    q_bits: int, self_k_bits: int, neighbor_k_bits: int
) -> tuple[int, dict[str, int]]:
    q = q_bits & MASK32
    self_k = self_k_bits & MASK32
    neighbor_k = neighbor_k_bits & MASK32
    up = ((~self_k) & neighbor_k) & MASK32
    down = (self_k & (~neighbor_k)) & MASK32
    q_up = popcount(q & up)
    q_down = popcount(q & down)
    up_count = popcount(up)
    down_count = popcount(down)
    delta = 65 * (q_up - q_down) - (up_count - down_count)
    raw = axnor_raw16(q, self_k) + delta
    return rne_div16(raw), {
        "up": up_count,
        "down": down_count,
        "q_up": q_up,
        "q_down": q_down,
        "delta_raw16": delta,
        "final_raw16": raw,
    }


def h67_direct_q7(
    q_bits: int, k_bits: int, peer_k_bits: int
) -> int:
    motion = popcount(k_bits ^ peer_k_bits)
    return rne_div16(axnor_raw16(q_bits, k_bits) + 16 * motion)


def h67_motion_delta_q7(
    q0_bits: int,
    k0_bits: int,
    q1_bits: int,
    k1_bits: int,
) -> tuple[int, dict[str, int]]:
    q0 = q0_bits & MASK32
    k0 = k0_bits & MASK32
    q1 = q1_bits & MASK32
    k1 = k1_bits & MASK32
    motion = popcount(k0 ^ k1)
    raw0 = axnor_raw16(q0, k0) + 16 * motion
    update_mask = (q0 ^ q1) | (k0 ^ k1)
    delta = 0
    pending = update_mask
    while pending:
        lane_bit = pending & -pending
        old_q = bool(q0 & lane_bit)
        old_k = bool(k0 & lane_bit)
        new_q = bool(q1 & lane_bit)
        new_k = bool(k1 & lane_bit)
        old_raw = 64 if old_q and old_k else (1 if not old_q and not old_k else 0)
        new_raw = 64 if new_q and new_k else (1 if not new_q and not new_k else 0)
        delta += new_raw - old_raw
        pending ^= lane_bit
    return rne_div16(raw0 + delta), {
        "motion": motion,
        "updates": popcount(update_mask),
        "delta_raw16": delta,
        "final_raw16": raw0 + delta,
    }


def exhaustive_single_active_lane() -> None:
    for q in (0, 1):
        for k0 in (0, 1):
            for k1 in (0, 1):
                local_delta, _ = local5_rcsd_q7(q, k0, k1)
                local_direct = local5_direct_q7(q, k1)
                if local_delta != local_direct:
                    raise AssertionError(
                        (
                            "Local5 single-active-lane",
                            q,
                            k0,
                            k1,
                            local_delta,
                            local_direct,
                        )
                    )
    for q0 in (0, 1):
        for k0 in (0, 1):
            for q1 in (0, 1):
                for k1 in (0, 1):
                    h67_delta, meta = h67_motion_delta_q7(q0, k0, q1, k1)
                    h67_direct_raw = (
                        axnor_raw16(q1, k1) + 16 * popcount(k0 ^ k1)
                    )
                    h67_direct = rne_div16(h67_direct_raw)
                    if (
                        meta["final_raw16"] != h67_direct_raw
                        or h67_delta != h67_direct
                    ):
                        raise AssertionError(
                            (
                                "H67 single-active-lane",
                                q0,
                                k0,
                                q1,
                                k1,
                                meta["final_raw16"],
                                h67_direct_raw,
                                h67_delta,
                                h67_direct,
                            )
                        )


def run_random(seed: int, vectors: int) -> dict[str, Any]:
    rng = random.Random(seed)
    local5_mismatches: list[dict[str, Any]] = []
    local5_raw_mismatches: list[dict[str, Any]] = []
    h67_mismatches: list[dict[str, Any]] = []
    h67_raw_mismatches: list[dict[str, Any]] = []
    local5_delta_hist = [0] * 33
    h67_update_hist = [0] * 33
    local5_raw_min = 1 << 60
    local5_raw_max = -(1 << 60)
    h67_raw_min = 1 << 60
    h67_raw_max = -(1 << 60)

    for _ in range(vectors):
        q0 = rng.getrandbits(32)
        k0 = rng.getrandbits(32)
        q1 = rng.getrandbits(32)
        k1 = rng.getrandbits(32)

        local5_delta, local5_meta = local5_rcsd_q7(q0, k0, k1)
        local5_direct_raw = axnor_raw16(q0, k1)
        local5_direct = rne_div16(local5_direct_raw)
        local5_delta_hist[local5_meta["up"] + local5_meta["down"]] += 1
        local5_raw_min = min(local5_raw_min, local5_meta["delta_raw16"])
        local5_raw_max = max(local5_raw_max, local5_meta["delta_raw16"])
        if (
            local5_meta["final_raw16"] != local5_direct_raw
            and len(local5_raw_mismatches) < 16
        ):
            local5_raw_mismatches.append(
                {
                    "q": f"0x{q0:08x}",
                    "self_k": f"0x{k0:08x}",
                    "neighbor_k": f"0x{k1:08x}",
                    "delta_raw": local5_meta["final_raw16"],
                    "direct_raw": local5_direct_raw,
                }
            )
        if local5_delta != local5_direct and len(local5_mismatches) < 16:
            local5_mismatches.append(
                {
                    "q": f"0x{q0:08x}",
                    "self_k": f"0x{k0:08x}",
                    "neighbor_k": f"0x{k1:08x}",
                    "delta": local5_delta,
                    "direct": local5_direct,
                }
            )

        h67_delta, h67_meta = h67_motion_delta_q7(q0, k0, q1, k1)
        h67_direct_raw = (
            axnor_raw16(q1, k1) + 16 * popcount(k0 ^ k1)
        )
        h67_direct = rne_div16(h67_direct_raw)
        h67_update_hist[h67_meta["updates"]] += 1
        h67_raw_min = min(h67_raw_min, h67_meta["delta_raw16"])
        h67_raw_max = max(h67_raw_max, h67_meta["delta_raw16"])
        if (
            h67_meta["final_raw16"] != h67_direct_raw
            and len(h67_raw_mismatches) < 16
        ):
            h67_raw_mismatches.append(
                {
                    "q0": f"0x{q0:08x}",
                    "k0": f"0x{k0:08x}",
                    "q1": f"0x{q1:08x}",
                    "k1": f"0x{k1:08x}",
                    "delta_raw": h67_meta["final_raw16"],
                    "direct_raw": h67_direct_raw,
                }
            )
        if h67_delta != h67_direct and len(h67_mismatches) < 16:
            h67_mismatches.append(
                {
                    "q0": f"0x{q0:08x}",
                    "k0": f"0x{k0:08x}",
                    "q1": f"0x{q1:08x}",
                    "k1": f"0x{k1:08x}",
                    "delta": h67_delta,
                    "direct": h67_direct,
                }
            )

    return {
        "vectors": vectors,
        "seed": seed,
        "local5": {
            "raw_mismatches": len(local5_raw_mismatches),
            "raw_mismatch_examples": local5_raw_mismatches,
            "mismatches": len(local5_mismatches),
            "mismatch_examples": local5_mismatches,
            "random_delta_count_histogram": local5_delta_hist,
            "observed_delta_raw16_min": local5_raw_min,
            "observed_delta_raw16_max": local5_raw_max,
        },
        "h67_motion": {
            "raw_mismatches": len(h67_raw_mismatches),
            "raw_mismatch_examples": h67_raw_mismatches,
            "mismatches": len(h67_mismatches),
            "mismatch_examples": h67_mismatches,
            "random_update_count_histogram": h67_update_hist,
            "observed_delta_raw16_min": h67_raw_min,
            "observed_delta_raw16_max": h67_raw_max,
        },
        "pass": (
            not local5_raw_mismatches
            and not local5_mismatches
            and not h67_raw_mismatches
            and not h67_mismatches
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# H67 Motion-Delta 与 Local5 RCSD 整数等价参考",
            "",
            "本报告只证明 score Q7 的无界整数等价，不包含 Shiftmax、队列、"
            "周期、面积或功耗。",
            "",
            f"- 随机向量：`{result['vectors']}`；种子：`{result['seed']}`。",
            "- 单活动 lane、其余 31 lane 为零的全组合：Local5 为 8 种，"
            "H67 为 `Q0/K0/Q1/K1` 全 16 种，均通过。",
            f"- Local5 RCSD 不一致：`{result['local5']['mismatches']}`。",
            f"- Local5 RCSD raw16 不一致："
            f"`{result['local5']['raw_mismatches']}`。",
            f"- H67 Motion-Delta raw16 不一致："
            f"`{result['h67_motion']['raw_mismatches']}`。",
            f"- H67 Motion-Delta 不一致：`{result['h67_motion']['mismatches']}`。",
            f"- 总结：**{'PASS' if result['pass'] else 'FAIL'}**。",
            "",
            "## Local5 RCSD",
            "",
            "```text",
            "A0 = 65*n11_self + 32 - q1 - k1_self",
            "U  = (~Kself) & Kneighbor",
            "D  = Kself & (~Kneighbor)",
            "Delta = 65*(|Q&U|-|Q&D|) - (|U|-|D|)",
            "score_q7 = RNE((A0 + Delta) / 16)",
            "```",
            "",
            "RNE 只在最终邻居 score 上执行一次，不能从已经舍入的 self score "
            "继续累加舍入后的 delta。",
            "",
            "## H67 Motion-Delta",
            "",
            "```text",
            "A0 = AXNOR_raw16(Q0,K0) + 16*popcount(K0 XOR K1)",
            "M = (Q0 XOR Q1) OR (K0 XOR K1)",
            "Delta = sum_{lane in M}(AXNOR_raw16_lane(Q1,K1)",
            "                         - AXNOR_raw16_lane(Q0,K0))",
            "score1_q7 = RNE((A0 + Delta) / 16)",
            "```",
            "",
            "Motion 项对两个时间片相同，因此不进入 delta；`K0 XOR K1` 仍可"
            "同时作为 Motion-popcount 与 update-mask 的输入。",
            "",
            "## 证据边界",
            "",
            "- `[推导+随机整数]`，不是部署 trace bit-exact；",
            "- Local5 在 G0/G1 后仍需对真实 Q/K 全向量复跑；",
            "- H67 仍需把该增量路径与冻结 hardware-order score trace 对齐。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=675005)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exhaustive_single_active_lane()
    result = run_random(args.seed, args.vectors)
    result["schema"] = "dual_line_delta_reference_v2"
    result["single_active_lane_exhaustive"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
