#!/usr/bin/env python3
"""证明Local5 hardware-order Shiftmax的每个有效候选gate至少为1 LSB。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LUT = (256, 245, 234, 224, 215, 205, 196, 188, 181, 173, 165, 158, 152, 145, 139, 133)


def exp_weight(abs_delta_q7: int) -> int:
    integer_shift = min(8, abs_delta_q7 >> 7)
    frac_q7 = abs_delta_q7 & 0x7F
    frac_index = min(15, (frac_q7 + 7) >> 3)
    return LUT[frac_index] >> integer_shift


def round_to_nearest_even(numerator: int, shift: int) -> int:
    quotient = numerator >> shift
    remainder = numerator - (quotient << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return quotient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/local5_geometry_ahead_gate_proof_20260803"),
    )
    args = parser.parse_args()

    weights = [exp_weight(delta) for delta in range(513)]
    min_weight = min(weights)
    max_weight = max(weights)
    max_row_sum = 5 * max_weight
    denominator = 1 << (max_row_sum - 1).bit_length()
    denominator_shift = denominator.bit_length() - 1
    min_gate = round_to_nearest_even(min_weight << 7, denominator_shift)
    if min_weight != 16 or denominator != 2048 or min_gate != 1:
        raise AssertionError("Local5有效gate下界与RTL合同不一致")

    result = {
        "schema": "local5_valid_gate_nonzero_proof_v1",
        "score_q7_range": [-256, 256],
        "max_abs_delta_q7": 512,
        "enumerated_delta_count": len(weights),
        "min_exp_q8": min_weight,
        "max_exp_q8": max_weight,
        "max_five_candidate_row_sum_q8": max_row_sum,
        "ceil_pow2_denominator": denominator,
        "minimum_valid_gate_q17": min_gate,
        "conclusion": "每个valid Local5候选gate至少为Q1.7的1 LSB",
        "contract_files": [
            "rtl_local5/local5_shiftmax5_q17.sv",
            "neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "proof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "PROOF.md").write_text(
        "\n".join(
            [
                "# Local5 有效 gate 非零证明",
                "",
                "## 结论",
                "",
                "在当前 Q7 score、16 项 Q8 exp2 LUT、五候选 ceil-pow2 分母和 Q1.7 RNE 合同下，每个 `valid=1` 的 Local5 候选 gate 至少为 `1` LSB，不会量化为零。",
                "",
                "## 边界推导",
                "",
                "- 有效 score 范围为 `[-256,256]`，故 `abs(score-row_max) <= 512`。",
                f"- 穷举 513 个可能 delta 后，`exp_q8` 下界为 `{min_weight}`，上界为 `{max_weight}`。",
                f"- 五候选 row sum 上界为 `{max_row_sum}`，ceil-pow2 分母上界为 `{denominator}`。",
                f"- 最小 gate 为 RNE(`{min_weight}*128/{denominator}`) = `{min_gate}`。",
                "",
                "## 对架构的约束",
                "",
                "因此只要候选在几何/invalid-candidate mask 中有效，它就必然产生非零 gate。Accumulator 目标地址可在 score/gate payload 返回前，仅由 source 坐标和固定 Local5 stencil 精确生成；这不适用于更大候选数、不同 score clamp 或不同 gate 量化格式。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
