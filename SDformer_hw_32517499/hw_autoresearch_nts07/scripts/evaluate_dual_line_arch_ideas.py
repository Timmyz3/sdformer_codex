#!/usr/bin/env python3
"""基于真实 profile 对 Motion/Local5 架构 idea 做证据分级筛选。

本脚本只读取已经生成的 profile，不运行 GPU 推理。输出是架构决策证据，
不是 RTL cycle、目标工艺 PPA 或端到端 FPS。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / (
    "hw_autoresearch_nts07/results/"
    "local5_h67_dual_profile_decision_20260726/"
    "local5_h67_dual_profile_decision.json"
)
DEFAULT_OUTPUT = ROOT / (
    "hw_autoresearch_nts07/results/"
    "dual_line_arch_idea_screen_20260726"
)
DEFAULT_MOTION_DETAIL = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "h67_ep19_ttb_delta_cycle_v2_profile100_20260713/"
    "nts11_hardware_p0_profile.json"
)
DEFAULT_LOCAL5_DETAIL = ROOT / (
    "hw_autoresearch_nts07/results/"
    "local5_hardware_profile_preG0_profile100_20260726/"
    "local5_hardware_features.json"
)


def combined_exact_or_sparse_coverage(
    exact_ratio: float, changed_coverage: float
) -> float:
    """返回 ZERO/exact 或变化项落入稀疏宽度时的总覆盖率。"""
    return exact_ratio + (1.0 - exact_ratio) * changed_coverage


def pct(value: float | None) -> str:
    if value is None:
        return "待测"
    return f"{100.0 * value:.4f}%"


def build_result(
    profile: dict[str, Any],
    motion_detail: dict[str, Any] | None = None,
    local5_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    motion = profile["h67_motion"]
    local5 = profile["local5"]

    motion_tare4_coverage = combined_exact_or_sparse_coverage(
        float(motion["update_zero_ratio"]),
        float(motion["changed_pair_coverage_le4"]),
    )
    local5_tare4_coverage = combined_exact_or_sparse_coverage(
        float(local5["exact_k_edge_ratio"]),
        float(local5["changed_edge_coverage_le4"]),
    )

    absolute_work: dict[str, Any] = {"available": False}
    if motion_detail is not None and local5_detail is not None:
        motion_abs = motion_detail["summary"]["binary_temporal_pairs"]
        local5_abs = local5_detail["summary"]
        motion_active_k_reads = int(motion_abs["k_temporal_union_reads"])
        local5_active_k_reads = round(
            int(local5_abs["query_major_active_k_lane_reads"])
            * (1.0 - float(local5_abs["active_k_read_reduction"]))
        )
        motion_terms = int(
            motion_abs["projection_gate_class_channel_terms_deploy"]
        )
        local5_terms = int(local5_abs["mfep_multicast_terms"])
        absolute_work = {
            "available": True,
            "motion_active_k_reads": motion_active_k_reads,
            "local5_active_k_reads_pre_g0": local5_active_k_reads,
            "local5_over_motion_active_k_reads": (
                local5_active_k_reads / motion_active_k_reads
            ),
            "motion_projection_terms": motion_terms,
            "local5_projection_terms_pre_g0": local5_terms,
            "local5_over_motion_projection_terms": (
                local5_terms / motion_terms
            ),
            "local5_valid_edges": int(local5_abs["valid_edges"]),
            "motion_token_total": int(motion_abs["token_total"]),
            "local5_over_motion_raw_items": (
                int(local5_abs["valid_edges"]) / int(motion_abs["token_total"])
            ),
        }

    ideas = [
        {
            "id": "I1",
            "name": "静态锚定精确残差执行",
            "sources": ["Prosperity", "Bishop", "FireFly-T"],
            "motion": {
                "evidence": "profile100+RTL",
                "opportunity": motion_tare4_coverage,
                "status": "已完成单 pair TARE-4；缺 T0/T1 原子 packet",
            },
            "local5": {
                "evidence": "pre-G0 profile100+合成RTL",
                "opportunity": local5_tare4_coverage,
                "status": "已完成单 edge；缺 ANCHOR_LOAD/PROBE row context",
            },
            "localization": (
                "以时间 peer 或 self-stencil 作为免费静态 anchor 候选，"
                "取消 Prosperity 的在线 TCAM 搜索；ZERO/LIST4/DIRECT "
                "通过 anchor+完整 residual 精确恢复 target。当前 RTL 分别实例化 "
                "raw32 和 delta4，只共享选择/RNE；真正共享 reduction tree 尚待实现。"
            ),
            "validation": (
                "与 Direct32、Direct32x2、zero-only、Prosperity-like matcher "
                "在同 SRAM/SDC 下比较 cycles、Fmax、area、SAIF、EDP。"
            ),
        },
        {
            "id": "I2",
            "name": "语义化 Bundle 与 metadata-first issue",
            "sources": ["Bishop", "SpAtten"],
            "motion": {
                "evidence": "profile100",
                "opportunity": float(motion["ttb4_empty_ratio"]),
                "status": "TTB4 统计已完成；尚未成为统一 score/SCS/term descriptor",
            },
            "local5": {
                "evidence": "固定拓扑+待测",
                "opportunity": None,
                "status": "STT 尚缺 post-G0 ordered halo/burst profile",
            },
            "localization": (
                "Motion 使用 Temporal-Affinity Bundle，携带 empty、motion-zero、"
                "delta width、K-zero 和 class multiplicity；Local5 使用 "
                "Stencil-Time Tile，携带 row/halo、方向 valid、anchor 和 delta。"
            ),
            "validation": (
                "比较无 bundle、TTB/STT、Bishop-like 双核三种执行；计入 metadata "
                "bit、FIFO、路由、fallback 和 p95/p99 stall。"
            ),
        },
        {
            "id": "I3",
            "name": "格式自适应精确残差包",
            "sources": ["Phi", "FireFly-T", "复旦蝶形 zero-skip"],
            "motion": {
                "evidence": "profile100",
                "opportunity": float(motion["changed_pair_coverage_le8"]),
                "status": "LIST4 已有；LIST8/bitmap/butterfly 未实现",
            },
            "local5": {
                "evidence": "pre-G0 profile100",
                "opportunity": float(local5["changed_edge_coverage_le8"]),
                "status": "LIST4 已有；四方向联合 pack 与冲突模型未实现",
            },
            "localization": (
                "不用 Phi 的学习 codebook，以网络语义给出的静态 anchor 作为 "
                "Level-1；anchor 本身不等于 target，必须叠加完整 Level-2 residual "
                "才零误差。"
                "Level-2 residual 在 ZERO/LIST4/LIST8/BITMAP32/DIRECT 中按块选格式。"
            ),
            "validation": (
                "linear priority、segmented prefix、butterfly compactor 同约束综合；"
                "报告编码 bit、解码 cycles、toggle、fallback 和净 SRAM/NoC 能量。"
            ),
        },
        {
            "id": "I4",
            "name": "归一化结果驱动的免物化投影",
            "sources": ["FLAT", "LoAS", "Prosperity"],
            "motion": {
                "evidence": "profile100+子系统RTL",
                "opportunity": float(motion["final_gate_term_count_reduction"]),
                "status": "SCS/NMF/DCTF 分离存在；缺统一顶层和 overflow fallback",
            },
            "local5": {
                "evidence": "pre-G0 profile100",
                "opportunity": float(local5["mfep_term_count_reduction_pre_g0"]),
                "status": "MFEP 仅 profile；数值合同和 multiplicity RTL 未闭合",
            },
            "localization": (
                "Motion 将 active {gate,K,token} 重编码为 gate-lane destination "
                "term；Local5 将有向 edge 重排为 multiplicity-aware source term，"
                "均不物化完整 gated-K 数值张量。"
            ),
            "validation": (
                "与 materialized gated-K、token-major、source-major 三个基线比较 "
                "SRAM bytes、端口冲突、cycles、energy；必须接通无损 fallback。"
            ),
        },
        {
            "id": "I5",
            "name": "源驻留与拓扑/时间多播",
            "sources": ["FLAT", "LoAS", "Bishop"],
            "motion": {
                "evidence": "profile100",
                "opportunity": float(motion["k_temporal_source_read_reduction"]),
                "status": "时间 K 精确读复用只有约一成，不宜单独主打",
            },
            "local5": {
                "evidence": "固定拓扑推导",
                "opportunity": float(local5["topology_k_read_reduction"]),
                "status": "理论读取减少明确；缺 line buffer/halo/SRAM 端口 RTL",
            },
            "localization": (
                "Motion 采用 temporal-pair 共驻留；Local5 采用三行 K line buffer，"
                "每个 source K 读取一次并向最多五个 destination 多播。"
            ),
            "validation": (
                "真实 SRAM macro 下比较 query-major 与 source-stationary；计入 halo、"
                "banking、边界、跨窗口换行和 line-buffer leakage。"
            ),
        },
        {
            "id": "I6",
            "name": "精确级联 issue 与块级功耗门控",
            "sources": ["SpAtten", "Bishop"],
            "motion": {
                "evidence": "profile100",
                "opportunity": float(motion["motion_zero_ratio"]),
                "status": "条件已统计；尚未形成统一 cascade scheduler/SAIF 消融",
            },
            "local5": {
                "evidence": "pre-G0 profile100",
                "opportunity": float(local5["exact_k_edge_ratio"]),
                "status": "exact-K bypass 已进 TARE；方向/degree gate 尚未实现",
            },
            "localization": (
                "不做 token/head pruning，按 exact 条件依次关闭 payload fetch、"
                "delta、Motion、class commit 或 projection；阈值按 stage/block "
                "descriptor 配置。"
            ),
            "validation": (
                "逐级消融 L0-L3，使用 post-layout 或 mapped SAIF 分账 clock/data "
                "gating，不能用 skip ratio 直接代替功耗收益。"
            ),
        },
    ]

    return {
        "schema": "dual-line-architecture-idea-screen-v1",
        "evidence_boundary": (
            "profile/推导/现有RTL筛选；不是目标工艺PPA。Local5除固定拓扑读取"
            "机会外均为pre-G0，必须在数值合同修复后复跑。"
        ),
        "profile_source": str(DEFAULT_PROFILE.relative_to(ROOT)),
        "derived": {
            "motion_tare4_zero_or_list4_coverage": motion_tare4_coverage,
            "local5_tare4_exact_or_list4_coverage_pre_g0": (
                local5_tare4_coverage
            ),
            "absolute_work": absolute_work,
        },
        "ideas": ideas,
        "mainline": {
            "current": "H67 Motion",
            "reason": (
                "已有冻结算法、profile100、ordered DSE、真实位级 trace、TARE、"
                "SCS/NMF/DCTF 子系统证据；Local5 仍受 G0/G1 和 row-context 阻塞。"
            ),
            "challenger": "Local5",
            "challenger_upside": (
                "规则 stencil 提供更高的局部复用比例、静态 self anchor 和"
                "multiplicity-aware edge-to-term 融合；但当前绝对 active-K "
                "与 projection term 均高于 Motion，不能据此宣称 PPA 上限更高。"
            ),
            "switch_gates": [
                "修复并冻结 Local5 mask/RNE/Shiftmax5 合同，valid825 AEE 仍优于 H67",
                "post-G0 profile100 保持 exact-K>=80%、源读取减少>=70%",
                "post-G0 Local5 active-K read/byte 绝对量不高于 Motion",
                "post-G0 Local5 projection term/cycle 绝对量不高于 Motion",
                "ANCHOR_LOAD/PROBE+Shiftmax5+MFEP 完整 RTL bit-exact",
                "同约束 DC/SAIF 下 Local5 子系统 EDP 至少优于 Motion 10%",
                "full-encoder Amdahl 后端到端 energy/frame 或 throughput 改善>=8%",
            ],
            "current_switch_decision": {
                "pass": False,
                "failed_or_pending": [
                    "Local5 mask/RNE/Shiftmax5 合同未冻结",
                    "Local5 只有 pre-G0 profile",
                    "Local5 完整 row-context/MFEP RTL 未实现",
                    (
                        "pre-G0 Local5 absolute active-K reads 高于 Motion"
                        if absolute_work.get(
                            "local5_over_motion_active_k_reads", 0.0
                        )
                        > 1.0
                        else "absolute active-K reads 待 post-G0 核验"
                    ),
                    (
                        "pre-G0 Local5 absolute projection terms 高于 Motion"
                        if absolute_work.get(
                            "local5_over_motion_projection_terms", 0.0
                        )
                        > 1.0
                        else "absolute projection terms 待 post-G0 核验"
                    ),
                    "同约束 DC/STA/SAIF 与系统 Amdahl 未完成",
                ],
            },
        },
        "screening_limitations": [
            "当前输出是 profile 机会与主线门控筛选，不是 cycle/PPA 排名器",
            "尚无逐 sample bootstrap 95% 置信区间",
            "尚未统一计算 detector/FIFO/SRAM/backpressure 成本",
            "Local5 数字为 pre-G0，不能用于最终主线宣称",
        ],
        "open_source_audit": {
            "Prosperity": {
                "repository": "https://github.com/dubcyfor3/Prosperity",
                "commit": "6ee1c6f1cb419fcf942f2eda63db84ca28248f4b",
                "license": "MIT",
                "available": (
                    "官方 cycle-accurate Python/CUDA simulator、Eyeriss/PTB/"
                    "SATO/MINT/LoAS 基线、CACTI buffer 接口、DSE 和统计对象"
                ),
                "reuse": (
                    "复用 component Stats、compute/preprocess/memory overlap、"
                    "DSE/消融方法；不直接复用其 activation subset kernel 和功耗常数。"
                ),
            },
            "Bishop": {
                "repository": None,
                "available": (
                    "未检索到官方仓库；论文使用自建 analytic cycle-accurate "
                    "simulator、STONNE/SIGMA、CACTI、DRAMsim3 和 28nm DC。"
                ),
                "reuse": (
                    "使用 STONNE 作为通用 sparse projection 基线；TTB/stratifier "
                    "需要按本项目 exact semantics 自建 trace simulator。"
                ),
            },
            "Phi": {
                "repository": None,
                "available": (
                    "未检索到官方仓库；论文描述自建 simulator、SystemVerilog、"
                    "28nm DC、CACTI 和 DRAMsim3。"
                ),
                "reuse": (
                    "按论文重建 pattern+residual、packer 和 conflict-free baseline；"
                    "不声称复现其未公开 simulator。"
                ),
            },
            "STONNE": {
                "repository": "https://github.com/stonne-simulator/stonne",
                "available": "开源 cycle-level dense/sparse GEMM simulator",
                "reuse": (
                    "只用于 projection/SpGEMM 通用基线；TARE、SCS、Shiftmax5、"
                    "term-atomic 控制仍由项目自建 ordered-event simulator 评估。"
                ),
            },
        },
        "date_gaps": [
            "冻结一条算法主线及部署数值合同",
            "完成 score->normalization->term->projection->Acc 单一顶层",
            "实现所有 overflow/dense/malformed/flush 无损 fallback",
            "多样本多窗口 ordered trace 的 mean/p50/p95/p99 与置信区间",
            "真实 SRAM latency、端口、bank conflict、反压和跨 tile 生命周期",
            "Direct/Prosperity-like/Phi-like/Bishop-like/本架构公平基线",
            "相同 SDC/PVT/SRAM macro 下 DC/STA/SAIF 与等价检查",
            "area、Fmax、power、energy/frame、EDP、面积归一吞吐主表",
            "full-encoder Amdahl 与至少一个端到端 FPS/energy 结果",
            "完整 RTL 回归、覆盖率、长反压、overflow、epoch 回绕和 LEC",
        ],
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Motion 与 Local5 架构 Idea 证据筛选",
        "",
        f"- 证据边界：{result['evidence_boundary']}",
        f"- 当前主线：**{result['mainline']['current']}**",
        f"- 挑战者：**{result['mainline']['challenger']}**",
        "",
        "## Profile 派生结论",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        (
            "| Motion ZERO 或 LIST4 总覆盖 | "
            f"{pct(result['derived']['motion_tare4_zero_or_list4_coverage'])} |"
        ),
        (
            "| Local5 exact 或 LIST4 总覆盖（pre-G0） | "
            f"{pct(result['derived']['local5_tare4_exact_or_list4_coverage_pre_g0'])} |"
        ),
    ]
    absolute = result["derived"]["absolute_work"]
    if absolute["available"]:
        lines.extend(
            [
                (
                    "| Motion absolute active-K reads | "
                    f"{absolute['motion_active_k_reads']:,} |"
                ),
                (
                    "| Local5 absolute active-K reads（pre-G0） | "
                    f"{absolute['local5_active_k_reads_pre_g0']:,} "
                    f"（Motion 的 {absolute['local5_over_motion_active_k_reads']:.3f}x） |"
                ),
                (
                    "| Motion projection terms | "
                    f"{absolute['motion_projection_terms']:,} |"
                ),
                (
                    "| Local5 projection terms（pre-G0） | "
                    f"{absolute['local5_projection_terms_pre_g0']:,} "
                    f"（Motion 的 {absolute['local5_over_motion_projection_terms']:.3f}x） |"
                ),
                (
                    "| Local5 valid edges / Motion tokens | "
                    f"{absolute['local5_over_motion_raw_items']:.3f}x |"
                ),
            ]
        )
    lines.extend(
        [
        "",
        "## Idea 矩阵",
        "",
        "| ID | 本土化架构 idea | 来源 | Motion | Local5 |",
        "|---|---|---|---|---|",
        ]
    )
    for idea in result["ideas"]:
        motion = idea["motion"]
        local5 = idea["local5"]
        lines.append(
            f"| {idea['id']} | {idea['name']} | "
            f"{'/'.join(idea['sources'])} | "
            f"{pct(motion['opportunity'])}；{motion['status']} | "
            f"{pct(local5['opportunity'])}；{local5['status']} |"
        )

    lines.extend(
        [
            "",
            "## 本土化定义与验证",
            "",
        ]
    )
    for idea in result["ideas"]:
        lines.extend(
            [
                f"### {idea['id']} {idea['name']}",
                "",
                f"- 本土化：{idea['localization']}",
                f"- 有效性验证：{idea['validation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 主线判定",
            "",
            f"- 当前选择 Motion：{result['mainline']['reason']}",
            f"- Local5 潜力边界：{result['mainline']['challenger_upside']}",
            (
                "- 当前切线结果："
                f"**{'PASS' if result['mainline']['current_switch_decision']['pass'] else 'FAIL'}**"
            ),
            "- Local5 切换门槛：",
        ]
    )
    lines.extend(
        f"  {index}. {item}"
        for index, item in enumerate(
            result["mainline"]["switch_gates"], start=1
        )
    )
    lines.extend(["", "当前失败或待闭合项："])
    lines.extend(
        f"  {index}. {item}"
        for index, item in enumerate(
            result["mainline"]["current_switch_decision"][
                "failed_or_pending"
            ],
            start=1,
        )
    )
    lines.extend(["", "## 筛选器边界", ""])
    lines.extend(f"- {item}" for item in result["screening_limitations"])

    lines.extend(["", "## 开源评估器审计", ""])
    for name, item in result["open_source_audit"].items():
        repo = item.get("repository") or "未发现官方仓库"
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 仓库：{repo}",
                f"- 可用内容：{item['available']}",
                f"- 本项目用法：{item['reuse']}",
                "",
            ]
        )

    lines.extend(["## 距离 DATE 的硬件缺口", ""])
    lines.extend(
        f"{index}. {item}"
        for index, item in enumerate(result["date_gaps"], start=1)
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--motion-detail", type=Path, default=DEFAULT_MOTION_DETAIL
    )
    parser.add_argument(
        "--local5-detail", type=Path, default=DEFAULT_LOCAL5_DETAIL
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    motion_detail = json.loads(
        args.motion_detail.read_text(encoding="utf-8")
    )
    local5_detail = json.loads(
        args.local5_detail.read_text(encoding="utf-8")
    )
    result = build_result(profile, motion_detail, local5_detail)
    result["profile_source"] = str(args.profile.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "idea_screen.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "idea_screen.md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(args.output_dir / "idea_screen.md")


if __name__ == "__main__":
    main()
