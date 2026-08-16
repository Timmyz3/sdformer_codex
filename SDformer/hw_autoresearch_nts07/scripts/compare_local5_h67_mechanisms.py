#!/usr/bin/env python3
"""把 Local5 空间差分与 H67 时间差分 profile 归一到同一硬件决策口径。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL5 = ROOT / (
    "hw_autoresearch_nts07/results/"
    "local5_hardware_profile_preG0_profile100_20260726/"
    "local5_hardware_features.json"
)
DEFAULT_H67 = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "h67_ep19_ttb_delta_cycle_v2_profile100_20260713/"
    "nts11_hardware_p0_profile.json"
)
DEFAULT_OUTPUT = ROOT / (
    "hw_autoresearch_nts07/results/"
    "local5_h67_dual_profile_decision_20260726"
)


def ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def string_list_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def sum_histograms(*histograms: list[int]) -> list[int]:
    length = max((len(hist) for hist in histograms), default=0)
    result = [0] * length
    for hist in histograms:
        for index, value in enumerate(hist):
            result[index] += int(value)
    return result


def percentile_from_histogram(histogram: list[int], percentile: float) -> int:
    total = sum(histogram)
    if not total:
        return 0
    target = max(1, int(total * percentile + 0.999999999))
    accumulated = 0
    for index, count in enumerate(histogram):
        accumulated += count
        if accumulated >= target:
            return index
    return len(histogram) - 1


def coverage_le(histogram: list[int], width: int, *, exclude_zero: bool) -> float:
    start = 1 if exclude_zero else 0
    denominator = sum(histogram[start:])
    return ratio(sum(histogram[start : width + 1]), denominator)


def local5_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    delta_histogram = sum_histograms(
        summary["up_delta_histogram"],
        summary["down_delta_histogram"],
        summary["left_delta_histogram"],
        summary["right_delta_histogram"],
    )
    directional_edges = int(summary["directional_valid_edges"])
    self_edges = int(summary["valid_edges"]) - directional_edges
    channels = int(round(summary["direct_neighbor_lane_work"] / directional_edges))
    direct_all_score_lane_work = int(summary["valid_edges"]) * channels
    selected_delta_lanes = self_edges * channels + int(
        summary["directional_delta_lane_sum"]
    )
    selected_lane_reduction = 1.0 - ratio(
        selected_delta_lanes, direct_all_score_lane_work
    )
    structural_selected_lane_reduction_max = ratio(
        directional_edges, int(summary["valid_edges"])
    )

    return {
        "samples": int(payload["samples"]),
        "channels": channels,
        "valid_edges": int(summary["valid_edges"]),
        "self_edges": self_edges,
        "directional_edges": directional_edges,
        "delta_lane_density": float(summary["delta_lane_density"]),
        "exact_k_edge_ratio": float(summary["delta_zero_edge_ratio"]),
        "delta_count_p50": percentile_from_histogram(delta_histogram, 0.50),
        "delta_count_p95": percentile_from_histogram(delta_histogram, 0.95),
        "delta_count_p99": percentile_from_histogram(delta_histogram, 0.99),
        "changed_edge_coverage_le2": coverage_le(
            delta_histogram, 2, exclude_zero=True
        ),
        "changed_edge_coverage_le4": coverage_le(
            delta_histogram, 4, exclude_zero=True
        ),
        "changed_edge_coverage_le8": coverage_le(
            delta_histogram, 8, exclude_zero=True
        ),
        "direct_all_score_lane_work": direct_all_score_lane_work,
        "selected_delta_lanes": selected_delta_lanes,
        "selected_lane_reduction_ideal": selected_lane_reduction,
        "structural_selected_lane_reduction_max": (
            structural_selected_lane_reduction_max
        ),
        "fraction_of_structural_lane_reduction": ratio(
            selected_lane_reduction,
            structural_selected_lane_reduction_max,
        ),
        "topology_k_read_reduction": float(
            summary["topology_k_read_reduction"]
        ),
        "active_k_read_reduction": float(summary["active_k_read_reduction"]),
        "gate_cardinality_mean_pre_g0": float(
            summary["gate_cardinality_mean"]
        ),
        "gate_cardinality_p95_pre_g0": int(
            summary["gate_cardinality_p95"]
        ),
        "offset_term_ratio_pre_g0": float(summary["offset_term_ratio"]),
        "mfep_term_ratio_pre_g0": float(summary["mfep_term_ratio"]),
        "mfep_term_count_reduction_pre_g0": 1.0
        - float(summary["mfep_term_ratio"]),
        "by_stage": payload["by_stage"],
        "evidence_boundary": payload["evidence_boundary"],
    }


def h67_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    delta = summary["delta_ttx"]
    pairs = summary["binary_temporal_pairs"]
    bundles = {
        int(item["token_bundle"]): item
        for item in summary["token_time_bundles"]
    }
    changed = int(delta["delta_changed_token_heads"])
    lane_sum = int(delta["qk_temporal_update_elements"])
    stage_update_density = {
        f"S{item['group']}": float(item["qk_temporal_update_density"])
        for item in summary["h60_by_stage"]
    }

    return {
        "samples": int(payload["samples"]),
        "temporal_pairs": int(delta["delta_token_heads"]),
        "update_zero_ratio": float(delta["delta_zero_update_token_ratio"]),
        "motion_zero_ratio": float(pairs["pair_motion_zero_ratio"]),
        "pair_empty_ratio": float(pairs["pair_empty_ratio"]),
        "score_equal_ratio": float(pairs["pair_score_equal_h67_ratio"]),
        "kzero_same_class_ratio": float(
            pairs["pair_kzero_same_class_h67_ratio"]
        ),
        "qk_update_lane_density": float(
            delta["qk_temporal_update_density"]
        ),
        "changed_pair_mean_lanes": ratio(lane_sum, changed),
        "changed_pair_coverage_le2": ratio(
            int(delta["delta_active_le2"]), changed
        ),
        "changed_pair_coverage_le4": ratio(
            int(delta["delta_active_le4"]), changed
        ),
        "changed_pair_coverage_le8": ratio(
            int(delta["delta_active_le8"]), changed
        ),
        "changed_run_mean_length": float(
            delta["delta_mean_changed_run_length"]
        ),
        "full_t2_compare_reduction": float(
            delta["full_t2_ideal_compare_reduction"]
        ),
        "structural_compare_reduction_max": 0.5,
        "fraction_of_structural_compare_reduction": ratio(
            float(delta["full_t2_ideal_compare_reduction"]), 0.5
        ),
        "ttb4_empty_ratio": float(bundles[4]["empty_ratio"]),
        "ttb8_empty_ratio": float(bundles[8]["empty_ratio"]),
        "kzero_token_ratio": float(pairs["token_kzero_ratio"]),
        "k_temporal_source_read_reduction": float(
            pairs["k_temporal_exact_reuse_ratio"]
        ),
        "final_gate_term_ratio": float(
            pairs["projection_gate_class_channel_ratio_deploy"]
        ),
        "final_gate_term_count_reduction": 1.0
        - float(pairs["projection_gate_class_channel_ratio_deploy"]),
        "gate_classes_per_row_mean": float(
            pairs["row_active_projection_gate_classes_mean_deploy"]
        ),
        "stage_update_density": stage_update_density,
        "sample_key_sha256": string_list_sha256(
            [
                str(record["sample_key"])
                for record in summary["sample_records"]
            ]
        ),
    }


def mechanism_decisions(
    local5: dict[str, Any], h67: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {
            "机制": "32-lane 差分检测与 4/8-lane 压缩候选",
            "Local5": (
                f"pre-G0：XOR={pct(local5['delta_lane_density'])}，"
                f"变化边单拍 <=4/8 覆盖 "
                f"{pct(local5['changed_edge_coverage_le4'])}/"
                f"{pct(local5['changed_edge_coverage_le8'])}"
            ),
            "H67 Motion": (
                f"Q/K 时间更新={pct(h67['qk_update_lane_density'])}，"
                f"变化 pair 单拍 <=4/8 覆盖 "
                f"{pct(h67['changed_pair_coverage_le4'])}/"
                f"{pct(h67['changed_pair_coverage_le8'])}"
            ),
            "决策": "候选；必须补多拍、fallback、burst/FIFO 和同约束 PPA",
        },
        {
            "机制": "Prosperity 式 exact/partial residual reuse",
            "Local5": (
                f"主推：拓扑固定 self anchor，exact-K="
                f"{pct(local5['exact_k_edge_ratio'])}（pre-G0）"
            ),
            "H67 Motion": (
                f"主推：T=2 共驻留，完全无更新="
                f"{pct(h67['update_zero_ratio'])}"
            ),
            "决策": "直接借 exact reuse 原理，改为空间/时间静态锚定",
        },
        {
            "机制": "Bishop TTB 打包与 bundle gating",
            "Local5": "可改为带 halo/方向 mask 的 STT；尚缺 ordered bundle profile",
            "H67 Motion": (
                f"直接适用：TTB4/8 empty="
                f"{pct(h67['ttb4_empty_ratio'])}/"
                f"{pct(h67['ttb8_empty_ratio'])}"
            ),
            "决策": "H67 先做 TTB4 cycle model；Local5 补 STT ordered profile 后再定",
        },
        {
            "机制": "FireFly-T 多 lane decoder / 蝶形 zero skipper",
            "Local5": "适合作为空间 delta index compactor",
            "H67 Motion": "适合作为时间更新 lane compactor",
            "决策": "先做 4/8-lane 多拍 DSE，再和 priority encoder 做同约束 PPA",
        },
        {
            "机制": "Phi 式 pattern + residual",
            "Local5": "可能与 RCSD 重复，需先测小 codebook residual",
            "H67 Motion": "时间 XOR 已提供天然 residual，额外 codebook 优先级低",
            "决策": "只做软件 profile 对照，不进入首版 RTL",
        },
        {
            "机制": "SCS / gate-class 后端折叠",
            "Local5": (
                f"暂定：MFEP term 比={pct(local5['mfep_term_ratio_pre_g0'])}，"
                "受 G0/G1 数值合同影响"
            ),
            "H67 Motion": (
                f"已证：final-gate term-count 减少="
                f"{pct(h67['final_gate_term_count_reduction'])}"
            ),
            "决策": "这是 term-count 压缩；H67 保留，Local5 G0/G1 后复跑",
        },
        {
            "机制": "SpAtten 式级联 issue，改为 exact pair coalescing",
            "Local5": "邻域 degree 固定，需等 G0 后按 gate/multiplicity 决定",
            "H67 Motion": (
                f"双时间 score 相同={pct(h67['score_equal_ratio'])}，"
                f"双 K-zero 同 class={pct(h67['kzero_same_class_ratio'])}"
            ),
            "决策": "只合并 SCS class-count commit；需证明相对现有 SCS 的增量收益",
        },
        {
            "机制": "H67 Motion 专属 bypass 与 stage-aware gating",
            "Local5": "不适用 Motion-popcount；空间方向可另做 stage-aware DSE",
            "H67 Motion": (
                f"motion-zero={pct(h67['motion_zero_ratio'])}，"
                f"K source-read reuse={pct(h67['k_temporal_source_read_reduction'])}，"
                f"changed-run mean={h67['changed_run_mean_length']:.4f}"
            ),
            "决策": (
                "联合评估 Motion-popcount bypass、ordered-run issue 和 "
                "per-stage width/power gating"
            ),
        },
    ]


def render_markdown(result: dict[str, Any]) -> str:
    local5 = result["local5"]
    h67 = result["h67_motion"]
    cohort = result["cohort_audit"]
    rows = result["mechanism_decisions"]
    lines = [
        "# Local5 与 H67 Motion 各自基线归一化 Profile 和机制决策",
        "",
        f"- Local5 样本数：`{local5['samples']}`",
        f"- H67 样本数：`{h67['samples']}`",
        "- 证据边界：本报告是 `[prof]` 工作量统计与 `[模型]` 收益上限，"
        "不是 RTL cycle、DC PPA 或端到端 FPS。",
        "- Local5 是 pre-G0 探索 profile。除固定 738-edge 拓扑和纯拓扑读取"
        "模型外，Q/K、K-XOR、exact/subset、gate 和 MFEP 均须在 G0/G1 后复跑。",
        f"- Cohort 审计：Local5/H67 的 100 个 ordered sample key "
        f"{'逐项一致' if cohort['exact_match'] else '不一致'}，SHA256 为 "
        f"`{cohort['local5_sample_key_sha256']}`。",
        "",
        "## 1. 结论",
        "",
        "1. **必须补 Local5 profile，现已完成 profile100。** "
        "否则无法判断 Local5 的五邻域是否只是把 162 个 score 膨胀成 738 条边。",
        "2. **H67 Motion 同样能用 exact residual、TTB 和多 lane compactor。** "
        "这些机制不是 Local5 专属；两条线的差分前端和累加资源具有共享潜力。",
        "3. **短期不应仅因 Local5 AEE 更好就全量换线。** H67 的数值和 RTL "
        "边界更成熟，且 profile 支持一个低迁移成本的 Motion-Delta 前端。",
        "4. **建议双线分工：** H67 Motion-Delta 作为近期可实现性能基线；"
        "Local5 RCSD 作为高收益候选。Local5 MFEP 必须等 G0/G1 后复跑。",
        "",
        "## 2. 同口径 Profile",
        "",
        "| 指标 | Local5 空间邻域 | H67 Motion 时间对 |",
        "|---|---:|---:|",
        (
            "| 完全不变工作单元 | "
            f"exact-K edge {pct(local5['exact_k_edge_ratio'])} | "
            f"update-zero pair {pct(h67['update_zero_ratio'])} |"
        ),
        (
            "| delta lane density | "
            f"{pct(local5['delta_lane_density'])} | "
            f"{pct(h67['qk_update_lane_density'])} |"
        ),
        (
            "| 变化项中 <=4 lane | "
            f"{pct(local5['changed_edge_coverage_le4'])} | "
            f"{pct(h67['changed_pair_coverage_le4'])} |"
        ),
        (
            "| 变化项中 <=8 lane | "
            f"{pct(local5['changed_edge_coverage_le8'])} | "
            f"{pct(h67['changed_pair_coverage_le8'])} |"
        ),
        (
            "| 假设 detector/metadata 免费时的 selected-lane 减少 | "
            f"{pct(local5['selected_lane_reduction_ideal'])} | "
            f"{pct(h67['full_t2_compare_reduction'])} |"
        ),
        (
            "| 各自结构上限与达到比例 | "
            f"{pct(local5['structural_selected_lane_reduction_max'])} / "
            f"{pct(local5['fraction_of_structural_lane_reduction'])} | "
            f"{pct(h67['structural_compare_reduction_max'])} / "
            f"{pct(h67['fraction_of_structural_compare_reduction'])} |"
        ),
        (
            "| 输入读取/打包机会 | "
            f"source-resident K-bit 读取减少 "
            f"{pct(local5['topology_k_read_reduction'])} | "
            f"TTB4 empty {pct(h67['ttb4_empty_ratio'])} |"
        ),
        (
            "| 后端 term 机会 | "
            f"MFEP term-count 减少 "
            f"{pct(local5['mfep_term_count_reduction_pre_g0'])}"
            "（预修复） | "
            f"final-gate term-count 减少 "
            f"{pct(h67['final_gate_term_count_reduction'])} |"
        ),
        (
            "| pair-class 合并机会 | "
            "待 G0 后按 Local5 gate/multiplicity 重算 | "
            f"score 相同 {pct(h67['score_equal_ratio'])}，"
            f"双 K-zero 同 class {pct(h67['kzero_same_class_ratio'])} |"
        ),
        (
            "| 额外 Motion 机会 | 不适用 | "
            f"motion-zero {pct(h67['motion_zero_ratio'])}，"
            f"K source-read reuse {pct(h67['k_temporal_source_read_reduction'])}，"
            f"changed-run mean {h67['changed_run_mean_length']:.4f} |"
        ),
        "",
        "Local5 的 `76.6%` 是下式的 selected update-lane 数量模型：",
        "",
        "```text",
        "baseline = 738 条有效边 × 32 lane",
        "RCSD     = 162 条 self anchor × 32 lane + 四方向 K-XOR 活动 lane",
        "```",
        "",
        "该模型假设 XOR detector、metadata、索引压缩器、余数携带和 direct "
        "fallback 均免费，而且不认为 Local5/H67 的单 lane 算术等成本。"
        "Local5 和 H67 的结构上限分别是 `78.05%` 和 `50%`；二者已达到各自"
        "上限的约 `98.11%` 和 `97.50%`，不能用 `76.6% > 48.7%` 排序架构。",
        "",
        "H67 的 stage update density 为："
        + "、".join(
            f"`{stage}={pct(value)}`"
            for stage, value in h67["stage_update_density"].items()
        )
        + "。因此固定全局 compactor 宽度可能过配，必须把 per-stage "
        "width/power gating 纳入 DSE。",
        "",
        "## 3. 可直接借用并修改的机制",
        "",
        "| 机制 | Local5 | H67 Motion | 决策 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['机制']} | {row['Local5']} | "
            f"{row['H67 Motion']} | {row['决策']} |"
        )
    lines.extend(
        [
            "",
            "借用不是问题，但论文和实现必须满足两点：",
            "",
            "- 清楚标注来源：TTB 来自 Bishop，exact/partial residual 思想来自 "
            "Prosperity，多 lane sparse decoder 参考 FireFly-T，蝶形压缩网络参考"
            "复旦 ISSCC 工作；",
            "- 贡献写成面向本 workload 的改造和组合收益，不把已有机制改名后"
            "宣称首次提出。",
            "",
            "## 4. 待验证的共享硬件骨架候选",
            "",
            "```text",
            "Packed operand SRAM / line buffer",
            "          |",
            "          v",
            "Static-anchor delta detector",
            "  Local5: self K -> N/S/E/W K-XOR",
            "  H67:    {Q0,K0} -> {Q1,K1} temporal XOR",
            "          |",
            "          +--> zero/exact bypass",
            "          |",
            "          +--> 32-to-4/8 lane compactor",
            "          |          |",
            "          |          +--> sparse delta issue",
            "          |",
            "          +--> dense direct fallback",
            "                     |",
            "                     v",
            "shared reduction tree + RNE/remainder state",
            "                     |",
            "          Shiftmax5(Local5) / SCS(H67)",
            "                     |",
            "             gate/term projection backend",
            "```",
            "",
            "当前只确认 bitmap detector、set-bit extractor 和部分累加/fallback "
            "控制具有共享潜力；"
            "Local5 使用空间 line buffer 和 4 个方向 delta，H67 使用 temporal-pair "
            "共驻留和 Motion-XOR mask。reduction tree、RNE/remainder 是否物理共享"
            "要等 opcode、端口、吞吐和同约束 PPA 后决定。",
            "",
            "## 5. 实施优先级与晋级门槛",
            "",
            "| 优先级 | 工作 | 晋级条件 |",
            "|---:|---|---|",
            "| P0 | 修 Local5 mask、Shiftmax x2、score RNE 数值合同 | "
            "Python/SV 全向量零失配，复跑 valid825 |",
            "| P1 | H67 Motion-Delta cycle model；收益为正后再做 compactor 叶 RTL | "
            "真实 trace bit-exact，报告多拍/fallback/burst，含 compactor 开销后周期下降 |",
            "| P1 | Local5 RCSD 无界整数/定点参考 | "
            "direct 与 delta 全向量 score_q7 零失配 |",
            "| P2 | Local5 ordered STT profile | "
            "给出 bundle empty、burst、FIFO、line-buffer 端口冲突 |",
            "| P2 | Local5 MFEP 复跑 | "
            "G0/G1 后 term 减少仍显著，且 accumulator 逐项零失配 |",
            "| P3 | 两条线统一 substrate 的同约束综合 | "
            "direct-only、H67 delta、Local5 delta 同 SDC/SRAM 规则比较 |",
            "",
            "## 6. 主线建议",
            "",
            "- **现在可以立即做 H67 Motion-Delta 与 SCS class-count coalescing "
            "的 reference/cycle model。** 只有模型包含 detector、compactor、"
            "多拍和 fallback 后仍有净收益，才进入叶 RTL。",
            "- **Local5 继续作为潜在主线。** pre-G0 RCSD 已接近其自身"
            " selected-lane 结构上限，但必须先关闭数值 P0；不能与 H67 的"
            " `48.7%` 直接排序，MFEP 暂不能成为论文实测贡献。",
            "- **不先做异构双核。** 当前两条 workload 都更支持一个可切换的"
            " direct/delta 同构核；是否复制 dense/sparse core 应由综合后的 EDP "
            "决定，而不是由借鉴 Bishop 本身决定。",
            "",
            "## 7. 证据文件",
            "",
            "- Local5 profile100："
            "`results/local5_hardware_profile_preG0_profile100_20260726/"
            "local5_hardware_features.json`",
            "- H67 ordered profile100："
            "`neuron_experiments/H9_bipolar_self_attention/results/"
            "h67_ep19_ttb_delta_cycle_v2_profile100_20260713/"
            "nts11_hardware_p0_profile.json`",
            "- Local5/H67 语义与切线审计："
            "`docs/150_Local5与H67硬件切线审计及架构创新候选_20260726.md`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local5", type=Path, default=DEFAULT_LOCAL5)
    parser.add_argument("--h67", type=Path, default=DEFAULT_H67)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.local5.open("r", encoding="utf-8") as handle:
        local5_payload = json.load(handle)
    with args.h67.open("r", encoding="utf-8") as handle:
        h67_payload = json.load(handle)

    local5 = local5_metrics(local5_payload)
    h67 = h67_metrics(h67_payload)
    result = {
        "schema": "local5_h67_dual_profile_decision_v1",
        "local5_source": str(args.local5),
        "h67_source": str(args.h67),
        "local5": local5,
        "h67_motion": h67,
        "cohort_audit": {
            "local5_sample_key_sha256": local5_payload["cohort"][
                "sample_key_sha256"
            ],
            "h67_sample_key_sha256": h67["sample_key_sha256"],
            "exact_match": (
                local5_payload["cohort"]["sample_key_sha256"]
                == h67["sample_key_sha256"]
                and int(local5_payload["cohort"]["count"])
                == int(h67["samples"])
            ),
        },
        "mechanism_decisions": mechanism_decisions(local5, h67),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "local5_h67_dual_profile_decision.json"
    md_path = args.output_dir / "local5_h67_dual_profile_decision.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"[dual-profile] wrote {json_path}")
    print(f"[dual-profile] wrote {md_path}")


if __name__ == "__main__":
    main()
