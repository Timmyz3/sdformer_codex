#!/usr/bin/env python3
"""汇总 profile100、FADC 容量上下界和精确 bit trace 的证据边界。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("分母必须为正")
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile100", type=Path, required=True)
    parser.add_argument("--fadc-bounds", type=Path, required=True)
    parser.add_argument("--bit-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.profile100.read_text(encoding="utf-8"))
    h67 = profile["models"]["H67"]
    ordered = h67["binary_temporal_pairs"]
    fadc = json.loads(args.fadc_bounds.read_text(encoding="utf-8"))
    bit_trace = json.loads(args.bit_trace.read_text(encoding="utf-8"))

    baseline_work = int(ordered["projection_baseline_active_lanes"])
    canonical_work = int(ordered["projection_gate_group_terms_g1"])
    aggregate = {
        "samples": int(h67["samples"]),
        "ordered_trace": bool(h67["ordered_trace"]),
        "pair_total": int(ordered["pair_total"]),
        "pair_empty_rate": ratio(
            int(ordered["pair_empty"]), int(ordered["pair_total"])
        ),
        "pair_motion_zero_rate": ratio(
            int(ordered["pair_motion_zero"]), int(ordered["pair_total"])
        ),
        "token_total": int(ordered["token_total"]),
        "token_kzero_rate": ratio(
            int(ordered["token_kzero"]), int(ordered["token_total"])
        ),
        "row_total": int(ordered["row_total"]),
        "baseline_projection_work": baseline_work,
        "canonical_g1_work": canonical_work,
        "canonical_work_reduction": 1.0 - canonical_work / baseline_work,
    }

    stage_rows = []
    for row in fadc["records"]:
        stage_rows.append(
            {
                "stage": int(row["stage"]),
                "head_instances": int(row["head_instances"]),
                "ipd_raw_fallback_rate": float(row["ipd32w_raw_fallback_rate"]),
                "fadc_guaranteed_fit_rate": float(
                    row["fadc24_guaranteed_fit_rate"]
                ),
                "fadc_ambiguous_rate": float(row["fadc24_ambiguous_rate"]),
                "fadc_impossible_fit_rate": float(
                    row["fadc24_impossible_fit_rate"]
                ),
                "fadc_best_case_term_reduction": float(
                    row["fadc24_best_case_term_reduction"]
                ),
                "fadc_worst_case_term_reduction": float(
                    row["fadc24_worst_case_term_reduction"]
                ),
            }
        )

    trace_records = bit_trace["records"]
    exact = {
        "sample_limit": int(bit_trace["sample_limit"]),
        "windows_per_call": int(bit_trace["windows_per_call"]),
        "first_block_only": bool(bit_trace["first_block_only"]),
        "records": len(trace_records),
        "captured_windows": sum(int(row["windows_captured"]) for row in trace_records),
        "heads": sum(int(row["heads"]) for row in trace_records),
        "attention_blocks": sorted(row["name"] for row in trace_records),
    }

    result = {
        "schema_version": 1,
        "status": "PASS_WITH_SCOPE_LIMIT",
        "evidence": {
            "aggregate_sparsity": "[prof] profile100 ordered trace",
            "format_capacity": "[prof]+[上下界]，缺逐term fanout消歧",
            "builder_cycles": "[rtl] sample0/B0/window0",
        },
        "aggregate": aggregate,
        "fadc_stage_bounds": stage_rows,
        "exact_bit_trace": exact,
        "architecture_implications": [
            "K-zero与pair-empty足以支持metadata-first和clock/data gating",
            "canonical G1显著减少投影work，但系统收益必须经过replay/multicast/AccTile闭环",
            "Stage 0的IPD RAW fallback最高，Stage 3的FADC容量长尾最高，不能使用全stage单格式",
            "当前精确Builder周期仅单sample、单block、单window，不足以报告数据集分布",
        ],
        "minimum_expansion": {
            "valid_frames": 100,
            "attention_blocks": 12,
            "windows_per_block_min": 4,
            "required_statistics": [
                "IPD/FADC/RAW比例",
                "fanout/term/payload/BPB/fallback的P50/P95/P99/max",
                "C0/C1周期、阻塞、重叠和背压分布",
                "按stage/block/运动强度分层",
            ],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# GateStack profile100 与精确 bit trace 证据分层",
        "",
        "## 1. 可直接使用的 profile100 结论",
        "",
        f"H67 ordered profile 覆盖 {aggregate['samples']} 个样本、"
        f"{aggregate['row_total']:,} 行、{aggregate['pair_total']:,} 个时间 pair。",
        "",
        "| 指标 | 数值 | 证据 |",
        "|---|---:|---|",
        f"| pair empty | {aggregate['pair_empty_rate']:.2%} | `[prof]` |",
        f"| pair motion-zero | {aggregate['pair_motion_zero_rate']:.2%} | `[prof]` |",
        f"| token K-zero | {aggregate['token_kzero_rate']:.2%} | `[prof]` |",
        f"| canonical G1 work 减少 | {aggregate['canonical_work_reduction']:.2%} | `[prof]` |",
        "",
        "这些统计足以支持 metadata-first、K-zero gating、canonical term 复用和稀疏功耗门控，但不能直接推出 C1 的端到端加速。",
        "",
        "## 2. FADC 容量上下界",
        "",
        "| Stage | Head实例 | IPD RAW fallback | FADC保证可装入 | ambiguous | impossible | term减少下界~上界 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| S{row['stage']} | {row['head_instances']:,} | "
            f"{row['ipd_raw_fallback_rate']:.3%} | "
            f"{row['fadc_guaranteed_fit_rate']:.3%} | "
            f"{row['fadc_ambiguous_rate']:.3%} | "
            f"{row['fadc_impossible_fit_rate']:.3%} | "
            f"{row['fadc_worst_case_term_reduction']:.2%}~"
            f"{row['fadc_best_case_term_reduction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Stage 0 的 IPD fallback 压力最大；Stage 3 的 FADC impossible 长尾最大。结论仍是逐 head 容量选择 IPD/FADC/RAW，而不是全 stage 固定单格式。ambiguous 项必须由扩展逐 term bit trace 消歧。",
            "",
            "## 3. 当前精确 RTL trace 边界",
            "",
            f"当前仅 {exact['sample_limit']} 个样本、{exact['captured_windows']} 个 window、"
            f"{exact['heads']} 个 head，且 `first_block_only={str(exact['first_block_only']).lower()}`。",
            "因此 1.403x 只能标为 `sample0/B0/window0` Builder bundle 的 RTL 结果，不能标成 profile100、projection slice 或 full encoder 结果。",
            "",
            "## 4. 最小扩展门槛",
            "",
            "- 至少 100 个有效帧；",
            "- 覆盖全部 12 个 attention block；",
            "- 每个 block 至少 4 个 window，并对运动强度分层；",
            "- 报告格式比例、fanout/term/payload/BPB/fallback 与 C0/C1 周期的 P50/P95/P99/max；",
            "- 与 profile100 aggregate 交叉核对，发现采样偏差时继续扩充；",
            "- 当前 GPU 满载，未启动扩展 bit trace，避免干扰训练。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
