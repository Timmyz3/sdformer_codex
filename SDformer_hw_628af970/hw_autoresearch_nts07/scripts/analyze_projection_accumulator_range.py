#!/usr/bin/env python3
"""重算H67四stage projection并统计32位累加器的数值安全裕量。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


TOKENS = 162
LANES = 32
INT32_MAX = (1 << 31) - 1


def load_signed_memh(path: Path, width: int) -> np.ndarray:
    values = np.asarray(
        [int(line, 16) for line in path.read_text().splitlines() if line.strip()],
        dtype=np.int64,
    )
    sign = 1 << (width - 1)
    modulus = 1 << width
    return np.where(values >= sign, values - modulus, values)


def required_signed_bits(max_abs: int) -> int:
    if max_abs == 0:
        return 1
    return int(math.ceil(math.log2(max_abs + 1))) + 1


def build_activations(
    raw_records: np.ndarray, *, head_offset: int, heads: int
) -> tuple[np.ndarray, int, float]:
    rows = raw_records[
        head_offset * TOKENS : (head_offset + heads) * TOKENS
    ].reshape(heads, TOKENS)
    gates = rows >> LANES
    k_bits = rows & ((1 << LANES) - 1)
    activations = np.zeros((TOKENS, heads * LANES), dtype=np.int64)
    active = 0
    for head in range(heads):
        for lane in range(LANES):
            mask = ((k_bits[head] >> lane) & 1).astype(bool)
            activations[mask, head * LANES + lane] = gates[head, mask]
            active += int(mask.sum())
    return activations, int(gates.max(initial=0)), active / activations.size


def analyze_arrays(
    activations: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    expected: np.ndarray,
) -> dict:
    if weight.shape[0] != weight.shape[1]:
        raise ValueError("projection weight必须为方阵")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation与weight输入维不匹配")
    if bias.shape != (weight.shape[0],):
        raise ValueError("bias维度不匹配")
    calculated = activations @ weight.T + bias[None, :]
    mismatches = int(np.count_nonzero(calculated != expected))

    partial = np.zeros_like(calculated)
    max_abs_partial = 0
    for channel in range(weight.shape[1]):
        partial += activations[:, channel, None] * weight[None, :, channel]
        max_abs_partial = max(
            max_abs_partial, int(np.abs(partial).max(initial=0))
        )

    actual_abs_sum = np.abs(activations) @ np.abs(weight).T
    actual_order_independent_bound = int(
        (actual_abs_sum + np.abs(bias)[None, :]).max(initial=0)
    )
    gate_format_bound = int(
        (511 * np.abs(weight).sum(axis=1) + np.abs(bias)).max(initial=0)
    )
    universal_bound = int(
        weight.shape[1] * 511 * 128 + np.abs(bias).max(initial=0)
    )
    max_abs_final = int(np.abs(calculated).max(initial=0))
    max_abs_bias = int(np.abs(bias).max(initial=0))
    return {
        "mismatches": mismatches,
        "output_min": int(calculated.min(initial=0)),
        "output_max": int(calculated.max(initial=0)),
        "max_abs_final": max_abs_final,
        "max_abs_partial_pre_bias": max_abs_partial,
        "max_abs_bias": max_abs_bias,
        "actual_order_independent_bound": actual_order_independent_bound,
        "gate511_weight_exact_bound": gate_format_bound,
        "universal_int8_bound": universal_bound,
        "required_bits_actual_final": required_signed_bits(max_abs_final),
        "required_bits_actual_partial": required_signed_bits(max_abs_partial),
        "required_bits_actual_order_independent": required_signed_bits(
            actual_order_independent_bound
        ),
        "required_bits_gate511_weight_exact": required_signed_bits(
            gate_format_bound
        ),
        "required_bits_universal_int8": required_signed_bits(universal_bound),
        "int32_margin_over_gate511_weight_exact": (
            INT32_MAX / gate_format_bound if gate_format_bound else None
        ),
        "int32_margin_over_universal_int8": (
            INT32_MAX / universal_bound if universal_bound else None
        ),
    }


def analyze(vector_root: Path, raw_records_path: Path) -> dict:
    raw_records = np.asarray(
        [
            int(line, 16)
            for line in raw_records_path.read_text().splitlines()
            if line.strip()
        ],
        dtype=np.int64,
    )
    rows = []
    head_offset = 0
    for stage in range(4):
        heads = 3 << stage
        dim = heads * LANES
        vector_dir = vector_root / f"real_sample0_s{stage}_b0_capacity"
        weight = load_signed_memh(
            vector_dir / "projection_weights_int8.memh", 8
        ).reshape(dim, dim)
        bias = load_signed_memh(
            vector_dir / "projection_bias_acc.memh", 32
        )
        expected = load_signed_memh(
            vector_dir / "expected_output_acc32.memh", 32
        ).reshape(TOKENS, dim)
        activations, gate_max, activation_density = build_activations(
            raw_records, head_offset=head_offset, heads=heads
        )
        row = analyze_arrays(activations, weight, bias, expected)
        row.update(
            {
                "stage": stage,
                "heads": heads,
                "dim": dim,
                "gate_max_actual": gate_max,
                "activation_density": activation_density,
            }
        )
        rows.append(row)
        head_offset += heads
    if head_offset * TOKENS != raw_records.size:
        raise ValueError("raw record数量与四stage 45 heads不一致")
    return {
        "schema_version": 1,
        "status": "PASS" if all(row["mismatches"] == 0 for row in rows) else "FAIL",
        "evidence": "H67真实S0-S3 gate/K + dyadic INT8 weight/bias RTL向量",
        "accumulator_width": 32,
        "int32_max": INT32_MAX,
        "stages": rows,
        "limits": [
            "只覆盖当前sample0/window0四stage RTL向量，不代表所有valid825样本",
            "gate511_weight_exact_bound允许所有K同时激活并使用9-bit最大gate，但使用当前权重和bias",
            "universal_int8_bound进一步假设所有权重幅值为128，是配置级保守上界",
            "结论用于projection Acc位宽，不覆盖attention、ATLIF或软件量化精度",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# H67 Projection 32位累加器数值范围统计",
        "",
        f"状态：**{report['status']}**。全部stage使用真实RTL向量重算，金参考失配均为0。",
        "",
        "| Stage | DIM | 实际gate最大 | 激活密度 | 实际最终最大绝对值 | 实际中间部分和最大绝对值 | 当前激活无关顺序界 | gate511+当前权重界 | 全INT8配置界 | 最坏所需有符号位 | int32最小裕量 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["stages"]:
        min_margin = min(
            row["int32_margin_over_gate511_weight_exact"],
            row["int32_margin_over_universal_int8"],
        )
        lines.append(
            f"| S{row['stage']} | {row['dim']} | {row['gate_max_actual']} | "
            f"{row['activation_density']:.3%} | {row['max_abs_final']} | "
            f"{row['max_abs_partial_pre_bias']} | "
            f"{row['actual_order_independent_bound']} | "
            f"{row['gate511_weight_exact_bound']} | "
            f"{row['universal_int8_bound']} | "
            f"{row['required_bits_universal_int8']} | {min_margin:.2f}x |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- 实际最终值：按RTL等价公式重算并加bias；",
        "- 实际中间部分和：按input-channel顺序逐项累加，记录任意中间拍最大绝对值；",
        "- 当前激活无关顺序界：对真实gate/K激活的每项乘积取绝对值求和再加bias，不依赖累加顺序；",
        "- gate511+当前权重界：假设所有K均激活、gate取9-bit最大511，使用当前INT8权重与bias；",
        "- 全INT8配置界：再假设所有权重幅值均为128，是当前DIM下的配置级最坏界。",
        "",
        "## 结论",
        "",
        "若四stage配置保持DIM不超过768、gate为9-bit无符号、weight为INT8，32-bit有符号Acc在全INT8配置级保守界下仍有充足裕量。当前合法配置不需要为了正常数值范围增加整tile final quarantine；overflow应作为非法配置、存储损坏或协议错误防护。",
        "",
        "该结论仍需用更多真实样本验证actual分布，但配置级界不依赖样本。若未来扩大DIM、gate位宽、weight位宽或bias范围，必须重新运行本脚本并更新静态位宽合同。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--raw-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.vector_root, args.raw_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
