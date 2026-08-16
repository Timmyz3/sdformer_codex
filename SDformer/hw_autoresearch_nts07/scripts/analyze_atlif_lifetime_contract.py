#!/usr/bin/env python3
"""从ATLIF调用表生成固定部署的事件生命周期和路由合同。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEAD_SUFFIX = ".attn.attn_sn.spiking_neuron"


def classify(name: str) -> tuple[str, str, str]:
    if name.endswith(DEAD_SUFFIX):
        return "dead_debug", "删除", "仅return_attention调试返回"
    if name.endswith(".attn.proj_sn.spiking_neuron"):
        return "dual_consumer_fanout", "Forward或Resident", "同一event同时供linear_q和linear_k"
    if name.endswith(".attn.sn_q.spiking_neuron") or name.endswith(".attn.sn_k.spiking_neuron"):
        return "temporal_pair_assembly", "Forward或Resident", "Q/K需按head和窗口对齐后送TESSA"
    known_single = (
        ".mlp.sn1.spiking_neuron",
        ".mlp.sn2.spiking_neuron",
        ".downsample.sn.spiking_neuron",
        ".patch_embed.head.sn.spiking_neuron",
        ".patch_embed.proj.sn.spiking_neuron",
        ".residual_encoding.resblocks.0.sn1.spiking_neuron",
        ".residual_encoding.resblocks.0.sn2.spiking_neuron",
        ".residual_encoding.resblocks.1.sn1.spiking_neuron",
        ".residual_encoding.resblocks.1.sn2.spiking_neuron",
        ".resblocks.0.sn1.spiking_neuron",
        ".resblocks.0.sn2.spiking_neuron",
        ".resblocks.1.sn1.spiking_neuron",
        ".resblocks.1.sn2.spiking_neuron",
        ".decoders.0.sn.spiking_neuron",
        ".decoders.1.sn.spiking_neuron",
        ".decoders.2.sn.spiking_neuron",
        ".decoders.3.sn.spiking_neuron",
        ".preds.0.sn.spiking_neuron",
        ".preds.1.sn.spiking_neuron",
        ".preds.2.sn.spiking_neuron",
        ".preds.3.sn.spiking_neuron",
    )
    if name.endswith(known_single):
        return "single_immediate_consumer", "Forward优先", "输出直接进入相邻Linear/Conv或下一stage"
    raise ValueError(f"未分类ATLIF路径: {name}")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"空ATLIF活动表: {path}")
    return rows


def analyze(path: Path, model: str) -> dict[str, Any]:
    rows = load_rows(path)
    modules: Counter[str] = Counter()
    elements: Counter[str] = Counter()
    active: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for row in rows:
        calls = int(row["calls"])
        if calls <= 0:
            raise ValueError(f"calls必须为正: {row['name']}")
        if int(row["elements"]) % calls:
            raise ValueError(f"elements不能按calls整除: {row['name']}")
        category, route, reason = classify(row["name"])
        per_frame_elements = int(row["elements"]) // calls
        per_frame_active = int(round(int(row["active"]) / calls))
        modules[category] += 1
        elements[category] += per_frame_elements
        active[category] += per_frame_active
        details.append({
            "name": row["name"],
            "category": category,
            "route_candidate": route,
            "reason": reason,
            "elements_per_frame": per_frame_elements,
            "active_per_frame": per_frame_active,
            "activity": float(row["activity"]),
        })

    live_categories = (
        "single_immediate_consumer",
        "dual_consumer_fanout",
        "temporal_pair_assembly",
    )
    live_elements = sum(elements[key] for key in live_categories)
    called_elements = live_elements + elements["dead_debug"]
    live_modules = sum(modules[key] for key in live_categories)
    if len(rows) != 93 or live_modules != 81 or modules["dead_debug"] != 12:
        raise ValueError(
            f"部署图数量不符: called={len(rows)}, live={live_modules}, dead={modules['dead_debug']}"
        )
    return {
        "model": model,
        "source": str(path),
        "called_modules": len(rows),
        "live_modules": live_modules,
        "dead_modules": modules["dead_debug"],
        "called_output_elements_per_frame": called_elements,
        "live_output_elements_per_frame": live_elements,
        "categories": [
            {
                "category": key,
                "modules": modules[key],
                "elements_per_frame": elements[key],
                "element_fraction_of_live": elements[key] / live_elements if key in live_categories else 0.0,
                "active_per_frame": active[key],
                "activity": active[key] / elements[key] if elements[key] else 0.0,
            }
            for key in (*live_categories, "dead_debug")
        ],
        "static_forward_eligible_upper_bound": {
            "elements_per_frame": elements["single_immediate_consumer"],
            "fraction_of_live": elements["single_immediate_consumer"] / live_elements,
            "warning": "仅表示单消费者且代码相邻；未计端口、排队、tile顺序和反压，不能当作真实bypass率。",
        },
        "long_lived_binary_event_outputs": 0,
        "precision_boundary": (
            "Swin block两次ADD、MS ResBlock ADD和S0-S2长skip保存的是多位算子输出/identity，"
            "不是ATLIF二值输出；必须进入RPI或更高层存储。"
        ),
        "details": details,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# H67/H68 ATLIF事件生命周期与静态路由合同",
        "",
        "**状态**：代码静态合同加旧profile100元素数；不是有序trace的实际bypass率  ",
        "**目的**：区分event可直接转发、需短期同步和多位残差/长skip，约束LR-HTT设计",
        "",
        "## 1. 结论",
        "",
        "- H67和H68固定部署均为93个动态调用，其中12个`attn_sn`仅服务调试返回，可删除；功能活跃81个。",
        "- 81个活跃ATLIF输出均为局部消费者，没有任何ATLIF二值输出本身承担S0-S2长skip。",
        "- Swin block的两次ADD、MS ResBlock ADD和encoder-decoder skip保存的是多位算子输出或identity，不能放进1-bit event bank。",
        "- 静态单消费者输出只说明具备direct-forward资格；能否绕过SRAM仍取决于producer-consumer时序、端口和反压。",
        "",
    ]
    for section_index, model in enumerate(result["models"], start=2):
        lines += [
            f"## {section_index}. {model['model']}分类统计",
            "",
            f"活跃event输出为`{model['live_output_elements_per_frame']:,}`元素/帧。静态单消费者上界为"
            f"`{model['static_forward_eligible_upper_bound']['elements_per_frame']:,}`元素/帧，"
            f"占`{model['static_forward_eligible_upper_bound']['fraction_of_live']:.2%}`。这不是实际bypass率。",
            "",
            "| 类别 | 模块数 | 元素/帧 | 活跃比例 | 占活跃输出 | 路由含义 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        meanings = {
            "single_immediate_consumer": "优先直接转发，stall时写HTT",
            "dual_consumer_fanout": "广播给Q/K或短期驻留到第二消费者完成",
            "temporal_pair_assembly": "Q/K按head/window同步，进入TESSA pair bank",
            "dead_debug": "固定部署删除",
        }
        for row in model["categories"]:
            fraction = row["element_fraction_of_live"]
            lines.append(
                f"| `{row['category']}` | {row['modules']} | {row['elements_per_frame']:,} | "
                f"{row['activity']:.2%} | {fraction:.2%} | {meanings[row['category']]} |"
            )
        lines.append("")

    lines += [
        "## 4. 对LR-HTT的直接约束",
        "",
        "1. event router至少需要`Forward/Resident`两种动作；`Spill`不能因为ATLIF输出是1-bit就写event bank，spill对象主要是多位ADD和长skip。",
        "2. `proj_sn`需要一写双读或一次广播，不能按单消费者处理；第二消费者完成前必须保留有效tag。",
        "3. `sn_q/sn_k`需要按`{stage, block, head, window, token, time}`对齐，缺一侧时写pair assembly bank，不能丢弃silent/K-zero记录。",
        "4. 其余单消费者点可采用弹性直通FIFO；FIFO满时必须无损退化为HTT resident，不允许覆盖或重排残差边界。",
        "5. 有序profile完成后，应以实际`forward/resident`计数替换80.13%左右的静态上界，并报告p50/p95/p99连续stall和bank事务。",
        "",
        "## 5. 架构收益口径",
        "",
        "静态合同只证明LR-HTT存在较大的候选空间，不能直接证明减少80%的SRAM事务。论文可引用的收益必须来自有序trace或RTL计数器：",
        "",
        "```text",
        "forward_ratio = direct_forward_elements / live_event_elements",
        "resident_ratio = event_bank_written_elements / live_event_elements",
        "fanout_reuse = avoided_second_read_elements / proj_sn_elements",
        "pair_assembly_stall = q_or_k_wait_cycles / pair_issue_cycles",
        "RPI_traffic = multi_bit_residual_and_skip_bits_read_write",
        "```",
        "",
        "只有`forward_ratio`真实达到40%以上且系统EDP相对局部fusion基线改善12%以上，LR-HTT才保留为主贡献。",
        "",
        "若把80.13%静态单消费者输出全部理想直通，剩余fanout/pair事件写回再读出的1-bit打包流量约为0.784 GB/s@30FPS，相对全物化3.945 GB/s减少3.161 GB/s。该值只作为上界，不能进入论文结果。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h67", type=Path, required=True)
    parser.add_argument("--h68", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "models": [analyze(args.h67, "H67"), analyze(args.h68, "H68")],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.md, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
