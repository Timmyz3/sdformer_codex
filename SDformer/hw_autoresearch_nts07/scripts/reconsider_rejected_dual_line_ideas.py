#!/usr/bin/env python3
"""重新评估 Motion/Local5 过去否定或搁置的硬件候选。

本脚本只读取已经生成的 profile、周期代理和静态分辨率账本，不运行 GPU，
不输出目标工艺 PPA。所有结果用于决定是否重开候选，不等价于 RTL 或芯片收益。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HW_ROOT = ROOT / "hw_autoresearch_nts07"

DEFAULT_DUAL_PROFILE = (
    HW_ROOT
    / "results/local5_h67_dual_profile_decision_20260726"
    / "local5_h67_dual_profile_decision.json"
)
DEFAULT_MOTION_PROFILE = (
    HW_ROOT / "results/profile100_compact_arch_stats_20260714.json"
)
DEFAULT_LOCAL5_PROFILE = (
    HW_ROOT
    / "results/local5_hardware_profile_preG0_profile100_20260726"
    / "local5_hardware_features.json"
)
DEFAULT_PPDI_PROFILE = (
    HW_ROOT / "results/gatestack_ppdi_profile_20260722/统计结果.json"
)
DEFAULT_ADAPTIVE_FORMAT = (
    HW_ROOT / "results/gatestack_adaptive_csr_fulltop_20260718/report.json"
)
DEFAULT_RESOLUTION = (
    HW_ROOT / "results/resolution_tile_term_ledger_20260728/ledger.json"
)
DEFAULT_OUTPUT = (
    HW_ROOT / "results/dual_line_reopened_ideas_20260728"
)


def combined_exact_sparse_coverage(
    exact_ratio: float, changed_coverage: float
) -> float:
    return exact_ratio + (1.0 - exact_ratio) * changed_coverage


def histogram_quantile(histogram: list[int], quantile: float) -> int:
    total = sum(histogram)
    if total <= 0:
        return 0
    target = math.ceil(total * quantile)
    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running >= target:
            return value
    return len(histogram) - 1


def histogram_max(histogram: list[int]) -> int:
    return max(
        (value for value, count in enumerate(histogram) if count),
        default=0,
    )


def ideal_dual_destination_commands(histogram: list[int]) -> int:
    return sum(((fanout + 1) // 2) * count for fanout, count in enumerate(histogram))


def destination_encoding_cost(
    histogram: list[int],
    *,
    tokens: int,
    destination_id_bits: int,
    format_tag_bits: int = 1,
) -> dict[str, Any]:
    terms = sum(histogram)
    deliveries = sum(
        fanout * count for fanout, count in enumerate(histogram)
    )
    list_bits = deliveries * destination_id_bits
    bitmap_bits = terms * tokens
    adaptive_payload_bits = sum(
        min(fanout * destination_id_bits, tokens) * count
        for fanout, count in enumerate(histogram)
    )
    adaptive_bits = adaptive_payload_bits + terms * format_tag_bits
    bitmap_terms = sum(
        count
        for fanout, count in enumerate(histogram)
        if fanout * destination_id_bits > tokens
    )
    return {
        "tokens": tokens,
        "destination_id_bits": destination_id_bits,
        "terms": terms,
        "deliveries": deliveries,
        "list_bits": list_bits,
        "bitmap_bits": bitmap_bits,
        "adaptive_bits_including_tag": adaptive_bits,
        "adaptive_reduction_vs_list": (
            1.0 - adaptive_bits / list_bits if list_bits else 0.0
        ),
        "adaptive_reduction_vs_bitmap": (
            1.0 - adaptive_bits / bitmap_bits if bitmap_bits else 0.0
        ),
        "bitmap_selected_term_ratio": (
            bitmap_terms / terms if terms else 0.0
        ),
        "list_bitmap_break_even_fanout": (
            tokens / destination_id_bits
        ),
    }


def validate_inputs(
    dual_profile: dict[str, Any],
    motion_profile: dict[str, Any],
    local5_profile: dict[str, Any],
    ppdi_profile: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "dual_schema": dual_profile.get("schema")
        == "local5_h67_dual_profile_decision_v1",
        "motion_schema": motion_profile.get("schema_version") == 1,
        "local5_schema": local5_profile.get("schema")
        == "local5_hardware_features_v1",
        "ppdi_schema": ppdi_profile.get("schema_version") == 1,
        "resolution_schema": resolution.get("schema_version") == 1,
        "local5_config_hash_present": bool(
            local5_profile.get("config_sha256")
        ),
        "local5_checkpoint_hash_present": bool(
            local5_profile.get("checkpoint_sha256")
        ),
        "local5_cohort_hash_present": bool(
            local5_profile.get("cohort", {}).get("sample_key_sha256")
        ),
        "dual_cohort_exact_match": bool(
            dual_profile.get("cohort_audit", {}).get("exact_match")
        ),
        "local5_cohort_matches_dual": (
            local5_profile.get("cohort", {}).get("sample_key_sha256")
            == dual_profile.get("cohort_audit", {}).get(
                "local5_sample_key_sha256"
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "输入证据 schema/provenance 检查失败：" + ", ".join(failed)
        )
    return {"checks": checks, "all_passed": True}


def build_result(
    dual_profile: dict[str, Any],
    motion_profile: dict[str, Any],
    local5_profile: dict[str, Any],
    ppdi_profile: dict[str, Any],
    adaptive_format: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    input_audit = validate_inputs(
        dual_profile,
        motion_profile,
        local5_profile,
        ppdi_profile,
        resolution,
    )
    motion = dual_profile["h67_motion"]
    local5 = dual_profile["local5"]
    local_summary = local5_profile["summary"]
    motion_summary = motion_profile["models"]["H67"][
        "binary_temporal_pairs"
    ]
    fanout_histogram = local_summary["mfep_fanout_histogram"]

    motion_tare4 = combined_exact_sparse_coverage(
        float(motion["update_zero_ratio"]),
        float(motion["changed_pair_coverage_le4"]),
    )
    local5_tare4 = combined_exact_sparse_coverage(
        float(local5["exact_k_edge_ratio"]),
        float(local5["changed_edge_coverage_le4"]),
    )

    local5_mpet_terms = int(local_summary["mfep_multicast_terms"])
    local5_destination_terms = int(
        local_summary["destination_gate_lane_groups"]
    )
    local5_deliveries = sum(
        fanout * count for fanout, count in enumerate(fanout_histogram)
    )
    local5_exploded_commands = int(
        local_summary["naive_active_edge_products"]
    )
    local5_ideal_pair_commands = ideal_dual_destination_commands(
        fanout_histogram
    )

    resolutions = {row["name"]: row for row in resolution["cases"]}
    crop_w9 = resolutions["crop-w9"]
    full_w15 = resolutions["full-w15"]
    crop_w9_id_bits = max(
        1, math.ceil(math.log2(int(crop_w9["tokens_per_row"])))
    )
    full_w15_id_bits = max(
        1, math.ceil(math.log2(int(full_w15["tokens_per_row"])))
    )
    w9_encoding = destination_encoding_cost(
        fanout_histogram,
        tokens=int(crop_w9["tokens_per_row"]),
        destination_id_bits=crop_w9_id_bits,
    )
    # 这是把 w9 fanout 分布代入 w15 容量的敏感性模型，不是 w15 profile。
    w15_sensitivity = destination_encoding_cost(
        fanout_histogram,
        tokens=int(full_w15["tokens_per_row"]),
        destination_id_bits=full_w15_id_bits,
    )

    ppdi_sample = ppdi_profile["sample0_window0"]
    adaptive_trace = adaptive_format["trace_bundle"]

    reopened = [
        {
            "id": "R1",
            "name": "ARST：锚点驻留语义 Tile",
            "old_idea": "TTB/STT + Prosperity residual + exact cascade issue",
            "motion_support": {
                "pair_empty_ratio": float(motion["pair_empty_ratio"]),
                "update_zero_ratio": float(motion["update_zero_ratio"]),
                "ttb4_empty_ratio": float(motion["ttb4_empty_ratio"]),
                "zero_or_list4_coverage": motion_tare4,
            },
            "local5_support": {
                "exact_k_ratio_pre_g0": float(
                    local5["exact_k_edge_ratio"]
                ),
                "source_read_reduction": float(
                    local5["topology_k_read_reduction"]
                ),
                "exact_or_list4_coverage_pre_g0": local5_tare4,
            },
            "decision": "DEFER_AFTER_ET3",
            "localization": (
                "不复制 Bishop dense/sparse 双核。Tile 持有网络语义给出的静态"
                " anchor、payload ownership、ZERO/LIST4/REPLAY mode 和 term "
                "commit/retire 生命周期；Motion anchor 是时间 peer，Local5 "
                "anchor 是 self stencil。"
            ),
            "missing": [
                "可执行 residency controller",
                "真实 payload fetch/hold/release 计数",
                "多窗口反压与 SRAM latency",
                "同约束 PPA",
            ],
        },
        {
            "id": "R2",
            "name": "ET3：精确 Set/Multiset Tile-to-Term Transduction",
            "old_idea": "NMF/MFEP/term flow",
            "motion_support": {
                "term_reduction_vs_active_lanes": float(
                    motion["final_gate_term_count_reduction"]
                ),
                "terms": int(
                    motion_summary[
                        "projection_gate_class_channel_terms_deploy"
                    ]
                ),
            },
            "local5_support": {
                "term_reduction_vs_active_edge_products_pre_g0": (
                    1.0 - float(local5["mfep_term_ratio_pre_g0"])
                ),
                "term_reduction_vs_per_destination_pre_g0": (
                    1.0 - local5_mpet_terms / local5_destination_terms
                ),
                "fanout_mean_pre_g0": (
                    local5_deliveries / local5_mpet_terms
                ),
                "fanout_p95_pre_g0": histogram_quantile(
                    fanout_histogram, 0.95
                ),
                "fanout_p99_pre_g0": histogram_quantile(
                    fanout_histogram, 0.99
                ),
                "fanout_max_pre_g0": histogram_max(fanout_histogram),
            },
            "decision": "RTL_PROTOTYPE_DONE_PROFILE_PENDING",
            "localization": (
                "把 Motion 的 set term 与 Local5 的 multiplicity-plane term "
                "统一为 exact typed IR；transducer 在 tile retire 前完成完整 key "
                "聚合，不物化 gated-K tensor。"
            ),
            "missing": [
                "post-G0/full-resolution Local5 trace",
                "post-G0/full-resolution ordered destination trace 回放",
                "真实目录容量/segment/fallback DSE",
                "同 SRAM/SDC 的 PPA 与端到端周期",
            ],
        },
        {
            "id": "R3",
            "name": "CATF：基数自适应 Set/Multiset Term Fabric",
            "old_idea": "AdaptiveCSR + PPDI + DCTF + butterfly multicast",
            "motion_support": {
                "sample0_ppdi_command_reduction": float(
                    ppdi_sample["command_reduction"]
                ),
                "adaptive_format_sample0_speedup_vs_gatestack": float(
                    adaptive_trace["speedup_vs_gatestack"]
                ),
            },
            "local5_support": {
                "ideal_unconstrained_pair_command_reduction_pre_g0": (
                    1.0
                    - local5_ideal_pair_commands / local5_deliveries
                ),
                "w9_encoding": w9_encoding,
                "w15_sensitivity_using_w9_fanout": w15_sensitivity,
            },
            "decision": "FOLD_INTO_ET3_ABLATION",
            "localization": (
                "每个 exact term 在 short-list、segmented bitmap 和 parity-paired "
                "delivery 间选择；选择只改变表示和发射，不改变 term 数学语义。"
                "蝶形网络仅作为高 fanout multicast/compaction 的实现候选。"
            ),
            "missing": [
                "Local5 parity-aware fanout profile",
                "Motion 多样本 PPDI profile",
                "格式 tag、decoder、SRAM 对齐成本",
                "central/prefix/butterfly 同约束 PPA",
            ],
        },
    ]

    held_or_rejected = [
        {
            "id": "H1",
            "name": "PHEA/Bishop 式 dense+sparse 双核",
            "decision": "KEEP_REJECTED",
            "reason": (
                "两线均已有超过九成的 exact-or-LIST4 覆盖，当前证据支持单一"
                " replay-capable 核；没有证据证明复制 dense core、双 FIFO 和"
                " stratifier 的面积/空闲功耗能被回收。"
            ),
            "reopen_gate": (
                "ordered trace 显示单核 p99 队列失稳，且双核在同 SRAM/SDC 下"
                " EDP 至少改善 15%。"
            ),
        },
        {
            "id": "H2",
            "name": "Phi 式学习 codebook",
            "decision": "KEEP_REJECTED",
            "reason": (
                "Motion/Local5 已有网络语义提供的 exact anchor；额外 pattern "
                "matcher、codebook SRAM 和 residual decode 目前没有增量收益证据。"
            ),
            "reopen_gate": (
                "post-G0 trace 证明静态 anchor 的 residual p95 明显恶化，而小"
                " codebook 能在计入 matcher 后净降 EDP。"
            ),
        },
        {
            "id": "H3",
            "name": "独立蝶形网络贡献",
            "decision": "KEEP_AS_MICROARCH_DSE",
            "reason": (
                "固定 Local5 五点 stencil 不需要通用路由；Motion LIST4/8 也可"
                "用 segmented prefix。蝶形只能作为 CATF compactor/multicast "
                "的物理实现消融，不能单独列贡献。"
            ),
            "reopen_gate": (
                "w15 或高 fanout term 下 central/prefix Fmax、wire power 或"
                " arbitration 明显劣化，蝶形 EDP 至少改善 10%。"
            ),
        },
    ]

    return {
        "schema": "dual-line-rejected-idea-reconsideration-v1",
        "evidence_boundary": (
            "Motion 为 profile100；Local5 为 pre-G0 profile100；w15 编码仅为"
            " w9 fanout 敏感性外推。Local5 数字是离线 unique 得到的理想"
            " MPET 聚合机会，不是在线目录或 RTL 收益。编码数字是逻辑 payload"
            " 上界，不是 SRAM 流量、面积或功耗。"
        ),
        "input_audit": input_audit,
        "sources": {
            "dual_profile": str(DEFAULT_DUAL_PROFILE),
            "motion_profile": str(DEFAULT_MOTION_PROFILE),
            "local5_profile": str(DEFAULT_LOCAL5_PROFILE),
            "ppdi_profile": str(DEFAULT_PPDI_PROFILE),
            "adaptive_format": str(DEFAULT_ADAPTIVE_FORMAT),
            "resolution": str(DEFAULT_RESOLUTION),
        },
        "derived": {
            "motion_zero_or_list4_coverage": motion_tare4,
            "local5_exact_or_list4_coverage_pre_g0": local5_tare4,
            "local5_mpet_terms_pre_g0": local5_mpet_terms,
            "local5_per_destination_terms_pre_g0": (
                local5_destination_terms
            ),
            "local5_mpet_deliveries_pre_g0": local5_deliveries,
            "local5_exploded_commands_pre_g0": local5_exploded_commands,
            "local5_native_multiset_command_reduction_pre_g0": (
                1.0 - local5_deliveries / local5_exploded_commands
            ),
            "local5_mpet_product_compute_reduction_pre_g0": (
                1.0 - local5_mpet_terms / local5_exploded_commands
            ),
            "local5_mpet_fanout_mean_pre_g0": (
                local5_deliveries / local5_mpet_terms
            ),
            "local5_mpet_fanout_p95_pre_g0": histogram_quantile(
                fanout_histogram, 0.95
            ),
            "local5_mpet_fanout_max_pre_g0": histogram_max(
                fanout_histogram
            ),
            "local5_ideal_pair_commands_pre_g0": (
                local5_ideal_pair_commands
            ),
            "local5_ideal_pair_command_reduction_pre_g0": (
                1.0 - local5_ideal_pair_commands / local5_deliveries
            ),
            "local5_w9_destination_encoding": w9_encoding,
            "local5_w15_sensitivity_using_w9_fanout": w15_sensitivity,
        },
        "reopened": reopened,
        "held_or_rejected": held_or_rejected,
        "proposed_architecture": {
            "name": "ET3 原生 Set/Multiset 端到端切片",
            "description": (
                "用有限 key directory 将 Motion set 或 Local5 bounded multiset"
                " 转为 segmented typed term，并由原生 multiplicity-aware"
                " bank executor 执行；目录 overflow 走无损 fallback。"
            ),
            "date_contributions_if_implemented": [
                (
                    "C1：保持 set/multiset 代数语义的在线有界 exact "
                    "tile-to-term transduction，不物化 gated-K tensor。"
                ),
            ],
            "current_status": (
                "小规模 RTL 切片已实现有限目录、segmented destination、"
                "无损 fallback、原生 multiplicity executor 和 dense 整数"
                " scoreboard，并通过 Icarus/Verilator/Yosys；尚无 post-G0/"
                "全分辨率 ordered trace、目标工艺 PPA 或端到端收益。"
            ),
        },
        "next_iteration": [
            (
                "导出 Local5 post-G0/full-resolution ordered destination trace，"
                "同时扩展 Motion 多样本 ordered trace。"
            ),
            (
                "用真实 trace 扫 KEY_CAP、SEG_DEPTH、fallback 深度，报告"
                " overflow、mean/p95/p99 周期和 SRAM 流量。"
            ),
            (
                "把真实尺寸参数与 SRAM latency/backpressure 接入 ET3，完成"
                " dense per-edge、MFEP+EXPLODE、ET3 三方 bit-exact 回放。"
            ),
            (
                "在同一 SRAM macro、SDC 和活动 trace 下比较 dense、"
                "EXPLODE 与 ET3 的 DC/STA/SAIF。"
            ),
            (
                "切片通过独立 DATE 复审后，再决定是否恢复 ARST 或 CATF。"
            ),
        ],
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def render_markdown(result: dict[str, Any]) -> str:
    derived = result["derived"]
    w9 = derived["local5_w9_destination_encoding"]
    w15 = derived["local5_w15_sensitivity_using_w9_fanout"]
    lines = [
        "# Motion/Local5 被否定 Idea 重开审计与架构重组",
        "",
        f"- 证据边界：{result['evidence_boundary']}",
        f"- 新候选总架构：**{result['proposed_architecture']['name']}**",
        "",
        "## 1. 新证据",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
        (
            "| Motion ZERO 或 LIST4 覆盖 | "
            f"{pct(derived['motion_zero_or_list4_coverage'])} | "
            "单一精确核有较高覆盖 |"
        ),
        (
            "| Local5 exact 或 LIST4 覆盖（pre-G0） | "
            f"{pct(derived['local5_exact_or_list4_coverage_pre_g0'])} | "
            "不需要先复制 dense core |"
        ),
        (
        "| Local5 理想 MPET / 逐 destination term（pre-G0） | "
            f"{derived['local5_mpet_terms_pre_g0']:,} / "
            f"{derived['local5_per_destination_terms_pre_g0']:,} | "
            f"term 数减少 "
            f"{pct(1.0 - derived['local5_mpet_terms_pre_g0'] / derived['local5_per_destination_terms_pre_g0'])} |"
        ),
        (
            "| Local5 原生 multiplicity destination command 减少"
            "（pre-G0） | "
            f"{pct(derived['local5_native_multiset_command_reduction_pre_g0'])} | "
            "相对把 multiplicity 展开为重复 edge command；离线上界 |"
        ),
        (
            "| Local5 理想 MPET product compute 减少（pre-G0） | "
            f"{pct(derived['local5_mpet_product_compute_reduction_pre_g0'])} | "
            "相对逐 active-edge product；未计目录/fallback 开销 |"
        ),
        (
        "| Local5 理想 MPET fanout mean/p95/max（pre-G0） | "
            f"{derived['local5_mpet_fanout_mean_pre_g0']:.2f}/"
            f"{derived['local5_mpet_fanout_p95_pre_g0']}/"
            f"{derived['local5_mpet_fanout_max_pre_g0']} | "
            "支持跨 destination product multicast |"
        ),
        (
            "| Local5 理想双目的 command 减少（pre-G0） | "
            f"{pct(derived['local5_ideal_pair_command_reduction_pre_g0'])} | "
            "无奇偶约束上界，不是 PPDI 实测 |"
        ),
        (
        "| w9 理想逻辑 payload 相对全 list | "
            f"{pct(w9['adaptive_reduction_vs_list'])} | "
            "含每 term 1-bit format tag 的静态模型 |"
        ),
        (
        "| w9 理想逻辑 payload 相对全 bitmap | "
            f"{pct(w9['adaptive_reduction_vs_bitmap'])} | "
            f"仅 {pct(w9['bitmap_selected_term_ratio'])} term 选择 bitmap |"
        ),
        (
            "| w15 敏感性：相对全 list | "
            f"{pct(w15['adaptive_reduction_vs_list'])} | "
            "沿用 w9 fanout，不是 w15 profile |"
        ),
        "",
        "## 2. 重开候选",
        "",
        "| ID | 新架构 | 旧来源组合 | 决策 |",
        "|---|---|---|---|",
    ]
    for item in result["reopened"]:
        lines.append(
            f"| {item['id']} | {item['name']} | "
            f"{item['old_idea']} | {item['decision']} |"
        )
    for item in result["reopened"]:
        lines.extend(
            [
                "",
                f"### {item['id']} {item['name']}",
                "",
                f"- 本土化：{item['localization']}",
                f"- 缺口：{'；'.join(item['missing'])}。",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. 继续否定或只做物理消融的候选",
            "",
            "| ID | 候选 | 决策 | 原因 | 重开门槛 |",
            "|---|---|---|---|---|",
        ]
    )
    for item in result["held_or_rejected"]:
        lines.append(
            f"| {item['id']} | {item['name']} | {item['decision']} | "
            f"{item['reason']} | {item['reopen_gate']} |"
        )

    architecture = result["proposed_architecture"]
    lines.extend(
        [
            "",
            "## 4. 架构重组",
            "",
            f"**{architecture['name']}**：{architecture['description']}",
            "",
            "只有实现并量化后才可列为 DATE 贡献：",
            "",
        ]
    )
    lines.extend(
        f"{index}. {item}"
        for index, item in enumerate(
            architecture["date_contributions_if_implemented"], start=1
        )
    )
    lines.extend(
        [
            "",
            f"当前状态：{architecture['current_status']}",
            "",
            "## 5. 下一轮实施",
            "",
        ]
    )
    lines.extend(
        f"{index}. {item}"
        for index, item in enumerate(result["next_iteration"], start=1)
    )
    lines.append("")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-profile", type=Path, default=DEFAULT_DUAL_PROFILE)
    parser.add_argument(
        "--motion-profile", type=Path, default=DEFAULT_MOTION_PROFILE
    )
    parser.add_argument(
        "--local5-profile", type=Path, default=DEFAULT_LOCAL5_PROFILE
    )
    parser.add_argument("--ppdi-profile", type=Path, default=DEFAULT_PPDI_PROFILE)
    parser.add_argument(
        "--adaptive-format", type=Path, default=DEFAULT_ADAPTIVE_FORMAT
    )
    parser.add_argument("--resolution", type=Path, default=DEFAULT_RESOLUTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = build_result(
        load_json(args.dual_profile),
        load_json(args.motion_profile),
        load_json(args.local5_profile),
        load_json(args.ppdi_profile),
        load_json(args.adaptive_format),
        load_json(args.resolution),
    )
    result["sources"] = {
        "dual_profile": str(args.dual_profile.resolve()),
        "motion_profile": str(args.motion_profile.resolve()),
        "local5_profile": str(args.local5_profile.resolve()),
        "ppdi_profile": str(args.ppdi_profile.resolve()),
        "adaptive_format": str(args.adaptive_format.resolve()),
        "resolution": str(args.resolution.resolve()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reconsideration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "reconsideration.md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(args.output_dir / "reconsideration.md")


if __name__ == "__main__":
    main()
