#!/usr/bin/env python3
"""用 H67 ordered trace 审计 projection 相序模型及基线敏感性。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_PROFILE = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_JSON = ROOT / "results/projection_phase_model_audit_20260715.json"
DEFAULT_MD = ROOT / "results/projection_phase_model_audit_20260715.md"


def evaluate(
    profile: dict[str, Any],
    *,
    class_slots: int,
    multicast_width: int,
    output_lanes: int,
    product_engines: int,
    tokens: int,
    head_dim: int,
) -> dict[str, Any]:
    directory_cells = class_slots * head_dim
    totals = {
        "direct_backend": 0,
        "candidate_backend": 0,
        "legacy_direct_buffered": 0,
        "legacy_candidate_no_scan": 0,
        "current_g1_scan_lower_bound": 0,
        "streaming_direct": 0,
        "dual_context_candidate": 0,
        "full_overlap_candidate": 0,
    }
    rows = 0
    overflow_rows = 0
    stage_totals: dict[int, dict[str, int]] = {}
    for record in profile["summary"]["h60_records"]:
        active = decode_count_trace(
            record["projection_baseline_active_lanes_ordered_trace"]
        )
        terms = decode_count_trace(
            record["projection_gate_class_channel_terms_deploy_ordered_trace"]
        )
        classes = decode_count_trace(
            record["projection_active_gate_classes_deploy_ordered_trace"]
        )
        delivery = decode_count_trace(
            record[f"projection_gate_multicast_delivery_m{multicast_width}_ordered_trace"]
        )
        output_channels = int(record["num_heads"]) * int(record["head_dim"])
        chunks = math.ceil(output_channels / output_lanes)
        stage = int(record["stage"])
        stage_row = stage_totals.setdefault(
            stage,
            {"rows": 0, "direct_backend": 0, "candidate_backend": 0, "scan_candidate": 0},
        )
        for active_row, term_row, class_row, delivery_row in zip(
            active, terms, classes, delivery
        ):
            direct = math.ceil(active_row / product_engines) * chunks
            overflow = class_row > class_slots
            if overflow:
                candidate = direct
                overflow_rows += 1
            else:
                product = math.ceil(term_row / product_engines) * chunks
                multicast = delivery_row * chunks
                candidate = max(product, multicast)
            drain_lower_bound = max(directory_cells, candidate)

            # bias=tokens/组是跨 head、output-tile 总量的等价分摊，不代表当前
            # 单 tile RTL 已实现完整多 head 生命周期。
            totals["direct_backend"] += direct
            totals["candidate_backend"] += candidate
            totals["legacy_direct_buffered"] += tokens + direct + tokens + 2
            totals["legacy_candidate_no_scan"] += tokens + candidate + tokens + 2
            totals["current_g1_scan_lower_bound"] += (
                tokens + drain_lower_bound + tokens + 2
            )
            totals["streaming_direct"] += max(tokens, direct) + tokens + 2
            totals["dual_context_candidate"] += (
                max(tokens, drain_lower_bound) + tokens + 2
            )
            totals["full_overlap_candidate"] += max(
                tokens, drain_lower_bound, tokens
            ) + 2
            rows += 1
            stage_row["rows"] += 1
            stage_row["direct_backend"] += direct
            stage_row["candidate_backend"] += candidate
            stage_row["scan_candidate"] += tokens + drain_lower_bound + tokens + 2

    streaming_direct = totals["streaming_direct"]
    comparisons = {
        "legacy_claim": (
            totals["legacy_direct_buffered"] / totals["legacy_candidate_no_scan"]
        ),
        "scan_candidate_vs_buffered_direct": (
            totals["legacy_direct_buffered"] / totals["current_g1_scan_lower_bound"]
        ),
        "scan_candidate_vs_streaming_direct": (
            streaming_direct / totals["current_g1_scan_lower_bound"]
        ),
        "dual_context_vs_streaming_direct": (
            streaming_direct / totals["dual_context_candidate"]
        ),
        "full_overlap_upper_vs_streaming_direct": (
            streaming_direct / totals["full_overlap_candidate"]
        ),
    }
    return {
        "rows": rows,
        "overflow_rows": overflow_rows,
        "overflow_ratio": overflow_rows / rows,
        "parameters": {
            "tokens": tokens,
            "head_dim": head_dim,
            "class_slots": class_slots,
            "directory_cells": directory_cells,
            "multicast_width": multicast_width,
            "output_lanes": output_lanes,
            "product_engines": product_engines,
        },
        "totals": totals,
        "cycles_per_group": {key: value / rows for key, value in totals.items()},
        "comparisons": comparisons,
        "stage_totals": stage_totals,
        "model_contract": {
            "legacy_direct_buffered": "token建流 + direct backend + bias + 控制",
            "legacy_candidate_no_scan": "token建表 + DSE backend + bias + 控制；旧1.51x口径",
            "current_g1_scan_lower_bound": "旧候选再加入max(128项目录扫描,DSE backend)",
            "streaming_direct": "max(token建流,direct backend) + bias + 控制",
            "dual_context_candidate": "max(本组建表,前组目录/backend) + bias + 控制",
            "full_overlap_candidate": "建表、目录/backend、bias三相完全重叠的乐观上界",
        },
    }


def render_md(result: dict[str, Any]) -> str:
    model = result["model"]
    totals = model["totals"]
    per = model["cycles_per_group"]
    comp = model["comparisons"]
    lines = [
        "# Projection 相序模型口径审计",
        "",
        f"输入：`{result['profile']}`，配置：`S=4, M=4, L=32, P=1`。",
        "",
        "## 1. 结果",
        "",
        "| 口径 | 总周期 | 周期/输入head组 | 相对可流式direct |",
        "|---|---:|---:|---:|",
        f"| 旧 buffered direct | {totals['legacy_direct_buffered']} | {per['legacy_direct_buffered']:.2f} | - |",
        f"| 旧 NMF，不计目录扫描 | {totals['legacy_candidate_no_scan']} | {per['legacy_candidate_no_scan']:.2f} | {totals['streaming_direct']/totals['legacy_candidate_no_scan']:.3f}x |",
        f"| 当前 G1 扫描下界 | {totals['current_g1_scan_lower_bound']} | {per['current_g1_scan_lower_bound']:.2f} | **{comp['scan_candidate_vs_streaming_direct']:.3f}x** |",
        f"| 可流式 direct | {totals['streaming_direct']} | {per['streaming_direct']:.2f} | 1.000x |",
        f"| 双 context NMF 候选 | {totals['dual_context_candidate']} | {per['dual_context_candidate']:.2f} | **{comp['dual_context_vs_streaming_direct']:.3f}x** |",
        f"| 三相全重叠上界 | {totals['full_overlap_candidate']} | {per['full_overlap_candidate']:.2f} | {comp['full_overlap_upper_vs_streaming_direct']:.3f}x |",
        "",
        f"旧报告的 `1.510x` 可复算为 `{comp['legacy_claim']:.3f}x`，但它同时采用 buffered direct 且忽略目录扫描。加入 128 项扫描并改用可流式 direct 后，保守模型只剩 **{comp['scan_candidate_vs_streaming_direct']:.3f}x**。",
        "",
        "## 2. 架构指导",
        "",
        f"- `S=4` overflow 为 `{model['overflow_rows']}/{model['rows']}`（{model['overflow_ratio']:.6%}），模型按 direct fallback，当前 RTL 尚未实现；",
        f"- 双 context 将模型点恢复到 `{comp['dual_context_vs_streaming_direct']:.3f}x`，因此比先扩 G>1 或双核更值得实现和测量；",
        "- 目录扫描固定 128 项与平均 backend 137.26 周期同量级，不能再默认完全免费；",
        "- 当前 RTL 只覆盖一个输入 head 和一个 output tile；完整 C 输出需要跨 head 保持 accumulator，并在全部输入 head 后对各 output tile 提交 bias；",
        "- `bias=tokens/输入head组` 只是完整 window 总量的等价分摊，不证明当前控制器生命周期正确。",
        "",
        "## 3. 证据边界",
        "",
        "- 所有数字均为 ordered trace 驱动的周期模型，不是 RTL cycle replay；",
        "- product 与 multicast 仍按完全重叠处理；",
        "- 未计 weight SRAM 延迟、跨 head/tile 切换、fallback FIFO、输出 requant；",
        "- 论文可引用 workload 计数，不能把本表写成 ASIC speedup。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    model = evaluate(
        profile,
        class_slots=4,
        multicast_width=4,
        output_lanes=32,
        product_engines=1,
        tokens=162,
        head_dim=32,
    )
    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "model": model,
        "evidence": "[prof ordered trace]+[模型敏感性]",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
