#!/usr/bin/env python3
"""把TERM-CSR两遍compaction纳入GateStack完整窗口周期模型。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_gatestack_csr_storage import classify_head_slot
from analyze_gatestack_compactor_profile import (
    compactor_cycles_by_row,
    reconstruct_row_k_counts,
)
from analyze_hit_flow_ordered_profiles import decode_count_trace
from model_gatestack_full_projection import overlap_two_contexts


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_PROFILE = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_JSON = ROOT / "results/gatestack_csr_full_projection_model_20260715.json"
DEFAULT_MD = ROOT / "results/gatestack_csr_full_projection_model_20260715.md"


def csr_commit_cycles(
    *,
    mode: str,
    active_lanes: int,
    class_terms: int,
    tokens: int,
    event_compactor_width: int,
    exact_event_cycles: int | None = None,
    scan_cycles: int | None = None,
) -> int:
    if event_compactor_width <= 0:
        raise ValueError("event_compactor_width必须为正")
    if mode == "TERM_CSR":
        # 第二遍顺序读RAW；OBI/prefix按有效term计；R路提取活动K事件。
        event_cycles = (
            math.ceil(active_lanes / event_compactor_width)
            if exact_event_cycles is None
            else exact_event_cycles
        )
        raw_scan_cycles = tokens if scan_cycles is None else scan_cycles
        return raw_scan_cycles + class_terms + event_cycles
    if mode in ("RAW_CLASS_OVERFLOW", "RAW_CAPACITY_OVERFLOW"):
        # 41-bit scratch到固定RAW slot，按一token/周期复制。
        return tokens
    raise ValueError(f"未知mode: {mode}")


def two_scratch_prepare_cycles(capture_cycles: int, commits: list[int]) -> int:
    """两个scratch使head h提交与head h+1捕获重叠。"""

    if capture_cycles <= 0:
        raise ValueError("capture_cycles必须为正")
    if not commits:
        return 0
    total = capture_cycles
    for commit in commits[:-1]:
        total += max(commit, capture_cycles)
    return total + commits[-1]


def _delivery_cycles(cycles: int, efficiency: float) -> int:
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("delivery_efficiency必须位于(0,1]")
    return math.ceil(cycles / efficiency)


def csr_replay_frontend_cycles(
    *,
    mode: str,
    class_terms: int,
    header_words: int = 2,
    descriptors_per_word: int = 2,
) -> int:
    """IPD32W token列表之前不可隐藏的header/descriptor顺序读延迟。"""

    if header_words < 0 or descriptors_per_word <= 0:
        raise ValueError("非法replay前端参数")
    if mode == "TERM_CSR":
        return header_words + math.ceil(class_terms / descriptors_per_word)
    if mode in ("RAW_CLASS_OVERFLOW", "RAW_CAPACITY_OVERFLOW"):
        return 0
    raise ValueError(f"未知mode: {mode}")


def resident_replay_frontend_cycles(
    *, mode: str, class_terms: int, descriptor_cache_terms: int
) -> int:
    if descriptor_cache_terms < 0:
        raise ValueError("descriptor cache深度不能为负")
    if (
        mode == "TERM_CSR"
        and descriptor_cache_terms > 0
        and class_terms <= descriptor_cache_terms
    ):
        return 0
    return csr_replay_frontend_cycles(mode=mode, class_terms=class_terms)


def evaluate(
    profile: dict[str, Any],
    *,
    tokens: int = 162,
    head_dim: int = 32,
    class_slots: int = 4,
    output_lanes: int = 32,
    product_engines: int = 1,
    multicast_width: int = 4,
    accumulator_banks: int = 4,
    event_compactor_width: int = 4,
    delivery_efficiency: float = 0.85,
    pipeline_fill: int = 4,
    active_token_skip: bool = False,
    descriptor_cache_terms: int = 0,
) -> dict[str, Any]:
    totals = {
        "windows": 0,
        "head_rows": 0,
        "csr_rows": 0,
        "raw_class_rows": 0,
        "raw_capacity_rows": 0,
        "direct_prepare": 0,
        "csr_prepare": 0,
        "direct_single": 0,
        "csr_single": 0,
        "direct_dual": 0,
        "csr_dual": 0,
        "direct_terms_all_tiles": 0,
        "selected_terms_all_tiles": 0,
        "descriptor_cached_rows": 0,
    }
    stage_totals: dict[int, dict[str, int]] = {}

    for record in profile["summary"]["h60_records"]:
        heads = int(record["num_heads"])
        output_channels = heads * int(record["head_dim"])
        output_tiles = math.ceil(output_channels / output_lanes)
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
        compact = compactor_cycles_by_row(record, event_compactor_width)
        row_k_counts = reconstruct_row_k_counts(record)
        active_tokens = (
            (row_k_counts != 0).sum(axis=2).reshape(-1).astype(int).tolist()
        )
        if not (
            len(active)
            == len(terms)
            == len(classes)
            == len(delivery)
            == len(compact)
            == len(active_tokens)
        ):
            raise ValueError("ordered trace长度不一致")
        if len(active) % heads:
            raise ValueError("ordered trace不能按窗口head数整组")
        stage = int(record["stage"])
        stage_row = stage_totals.setdefault(
            stage,
            {
                "windows": 0,
                "head_rows": 0,
                "csr_rows": 0,
                "raw_rows": 0,
                "direct_dual": 0,
                "csr_dual": 0,
                "descriptor_cached_rows": 0,
            },
        )
        direct_prepares: list[int] = []
        csr_prepares: list[int] = []
        direct_executes: list[int] = []
        csr_executes: list[int] = []
        tile_tail = math.ceil(tokens / accumulator_banks) * 2 + 2

        for base in range(0, len(active), heads):
            commits = []
            direct_head_replay = 0
            csr_head_replay = 0
            window_direct_terms = 0
            window_selected_terms = 0
            for offset in range(heads):
                index = base + offset
                representation = classify_head_slot(
                    active_lanes=active[index],
                    class_terms=terms[index],
                    active_classes=classes[index],
                    tokens=tokens,
                    head_dim=head_dim,
                    class_slots=class_slots,
                )
                mode = str(representation["mode"])
                commits.append(
                    csr_commit_cycles(
                        mode=mode,
                        active_lanes=active[index],
                        class_terms=terms[index],
                        tokens=tokens,
                        event_compactor_width=event_compactor_width,
                        exact_event_cycles=compact[index],
                        scan_cycles=active_tokens[index] if active_token_skip else tokens,
                    )
                )
                if mode == "TERM_CSR":
                    totals["csr_rows"] += 1
                    stage_row["csr_rows"] += 1
                    selected = terms[index]
                    product = math.ceil(terms[index] / product_engines)
                    issue = terms[index]
                    deliver = _delivery_cycles(
                        delivery[index], delivery_efficiency
                    )
                    cache_hit = (
                        descriptor_cache_terms > 0
                        and terms[index] <= descriptor_cache_terms
                    )
                    if cache_hit:
                        totals["descriptor_cached_rows"] += 1
                        stage_row["descriptor_cached_rows"] += 1
                    replay = resident_replay_frontend_cycles(
                        mode=mode,
                        class_terms=terms[index],
                        descriptor_cache_terms=descriptor_cache_terms,
                    ) + max(product, issue, deliver)
                    if terms[index] or delivery[index]:
                        replay += pipeline_fill
                else:
                    if mode == "RAW_CLASS_OVERFLOW":
                        totals["raw_class_rows"] += 1
                    else:
                        totals["raw_capacity_rows"] += 1
                    stage_row["raw_rows"] += 1
                    selected = active[index]
                    product = math.ceil(active[index] / product_engines)
                    deliver = _delivery_cycles(
                        math.ceil(active[index] / multicast_width),
                        delivery_efficiency,
                    )
                    replay = max(product, deliver)
                    if active[index]:
                        replay += pipeline_fill
                direct_product = math.ceil(active[index] / product_engines)
                direct_delivery = _delivery_cycles(
                    math.ceil(active[index] / multicast_width), delivery_efficiency
                )
                direct_replay = max(direct_product, direct_delivery)
                if active[index]:
                    direct_replay += pipeline_fill
                direct_head_replay += direct_replay
                csr_head_replay += replay
                window_direct_terms += active[index]
                window_selected_terms += selected

            direct_prepare = heads * tokens
            csr_prepare = two_scratch_prepare_cycles(tokens, commits)
            direct_execute = output_tiles * (direct_head_replay + tile_tail)
            csr_execute = output_tiles * (csr_head_replay + tile_tail)
            direct_prepares.append(direct_prepare)
            csr_prepares.append(csr_prepare)
            direct_executes.append(direct_execute)
            csr_executes.append(csr_execute)
            totals["windows"] += 1
            totals["head_rows"] += heads
            totals["direct_prepare"] += direct_prepare
            totals["csr_prepare"] += csr_prepare
            totals["direct_terms_all_tiles"] += window_direct_terms * output_tiles
            totals["selected_terms_all_tiles"] += window_selected_terms * output_tiles
            stage_row["windows"] += 1
            stage_row["head_rows"] += heads

        direct_single = sum(
            prepare + execute
            for prepare, execute in zip(direct_prepares, direct_executes)
        )
        csr_single = sum(
            prepare + execute
            for prepare, execute in zip(csr_prepares, csr_executes)
        )
        direct_dual = overlap_two_contexts(direct_prepares, direct_executes)
        csr_dual = overlap_two_contexts(csr_prepares, csr_executes)
        totals["direct_single"] += direct_single
        totals["csr_single"] += csr_single
        totals["direct_dual"] += direct_dual
        totals["csr_dual"] += csr_dual
        stage_row["direct_dual"] += direct_dual
        stage_row["csr_dual"] += csr_dual

    hybrid_direct_stages = [
        stage
        for stage, row in stage_totals.items()
        if row["direct_dual"] < row["csr_dual"]
    ]
    hybrid_dual = sum(
        row["direct_dual"] if stage in hybrid_direct_stages else row["csr_dual"]
        for stage, row in stage_totals.items()
    )
    totals["hybrid_dual"] = hybrid_dual
    return {
        "parameters": {
            "tokens": tokens,
            "head_dim": head_dim,
            "class_slots": class_slots,
            "output_lanes": output_lanes,
            "product_engines": product_engines,
            "multicast_width": multicast_width,
            "accumulator_banks": accumulator_banks,
            "event_compactor_width": event_compactor_width,
            "delivery_efficiency": delivery_efficiency,
            "pipeline_fill": pipeline_fill,
            "active_token_skip": active_token_skip,
            "descriptor_cache_terms": descriptor_cache_terms,
        },
        "totals": totals,
        "ratios": {
            "csr_rows": totals["csr_rows"] / totals["head_rows"],
            "raw_rows": (
                totals["raw_class_rows"] + totals["raw_capacity_rows"]
            )
            / totals["head_rows"],
            "prepare_overhead_vs_direct": totals["csr_prepare"] / totals["direct_prepare"],
            "selected_term_reduction": 1.0
            - totals["selected_terms_all_tiles"] / totals["direct_terms_all_tiles"],
            "descriptor_cache_hit_within_csr": (
                totals["descriptor_cached_rows"] / totals["csr_rows"]
                if totals["csr_rows"]
                else 0.0
            ),
        },
        "speedups": {
            "csr_single_vs_direct_single": totals["direct_single"] / totals["csr_single"],
            "csr_dual_vs_direct_dual": totals["direct_dual"] / totals["csr_dual"],
            "hybrid_dual_vs_direct_dual": totals["direct_dual"] / hybrid_dual,
        },
        "hybrid_direct_stages": hybrid_direct_stages,
        "stage_totals": stage_totals,
        "model_limits": [
            "active-token skip按既有逐token K-count精确计数；随机地址scratch读延迟尚未计入",
            "R路event提取使用既有逐token K-count精确求和，但packed-slot写bank冲突尚未计入",
            "TERM-CSR descriptor和token list按顺序SRAM吞吐抽象，未计真实宏延迟",
            "IPD32W已逐head计入2+ceil(term/2)个不可隐藏header/descriptor读拍",
            "descriptor cache命中head假设compaction旁路写入，不增加既有term枚举周期",
            "direct与CSR均允许双window context，基线不故意串行化",
            "结果为[prof]+[模型]，不是RTL或DC结果",
        ],
    }


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# GateStack TERM-CSR完整窗口周期模型",
        "",
        f"输入：`{result['profile']}`。所有结果为 `[prof]+[模型]`。",
        "",
        "## 1. Compactor并行度敏感性",
        "",
        "| R | token扫描 | delivery效率 | CSR比例 | prepare/direct | 单context | 全CSR双context | 分stage混合 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in result["models"]:
        p = model["parameters"]
        lines.append(
            f"| {p['event_compactor_width']} | {'活动token' if p['active_token_skip'] else '全162'} | "
            f"{p['delivery_efficiency']:.0%} | "
            f"{model['ratios']['csr_rows']:.3%} | {model['ratios']['prepare_overhead_vs_direct']:.3f}x | "
            f"{model['speedups']['csr_single_vs_direct_single']:.3f}x | "
            f"{model['speedups']['csr_dual_vs_direct_dual']:.3f}x | "
            f"**{model['speedups']['hybrid_dual_vs_direct_dual']:.3f}x** |"
        )
    lines += [
        "",
        "## 2. 有界 Descriptor Residency DSE（R=2、活动token）",
        "",
        "| cache深度 | CSR内命中 | 双context speedup | 分stage混合 |",
        "|---:|---:|---:|---:|",
    ]
    for model in result["cache_models"]:
        lines.append(
            f"| {model['parameters']['descriptor_cache_terms']} | "
            f"{model['ratios']['descriptor_cache_hit_within_csr']:.4%} | "
            f"{model['speedups']['csr_dual_vs_direct_dual']:.3f}x | "
            f"{model['speedups']['hybrid_dual_vs_direct_dual']:.3f}x |"
        )
    chosen = next(
        model
        for model in result["cache_models"]
        if model["parameters"]["descriptor_cache_terms"] == 80
    )
    lines += [
        "",
        "## 3. 默认R=2、活动token、Depth=80、delivery=85%",
        "",
        f"- 完整窗口：{chosen['totals']['windows']}；head row：{chosen['totals']['head_rows']}；",
        f"- TERM-CSR：{chosen['ratios']['csr_rows']:.4%}；RAW：{chosen['ratios']['raw_rows']:.4%}；",
        f"- CSR内descriptor cache命中：{chosen['ratios']['descriptor_cache_hit_within_csr']:.4%}；",
        f"- CSR准备周期是direct捕获的{chosen['ratios']['prepare_overhead_vs_direct']:.3f}倍；",
        f"- 跨output-tile加权后term减少{chosen['ratios']['selected_term_reduction']:.4%}；",
        f"- 双context相对公平direct为{chosen['speedups']['csr_dual_vs_direct_dual']:.3f}x；",
        f"- 编译期旁路stage {chosen['hybrid_direct_stages']} 后为{chosen['speedups']['hybrid_dual_vs_direct_dual']:.3f}x；",
        "",
        "### 分stage",
        "",
        "| Stage | CSR比例 | RAW rows | 双context speedup |",
        "|---|---:|---:|---:|",
    ]
    for stage, row in chosen["stage_totals"].items():
        lines.append(
            f"| {stage} | {row['csr_rows']/row['head_rows']:.3%} | {row['raw_rows']} | "
            f"{row['direct_dual']/row['csr_dual']:.3f}x |"
        )
    lines += [
        "",
        "## 4. 限制",
        "",
    ]
    lines.extend(f"- {item}；" for item in chosen["model_limits"])
    lines += [
        "",
        "如果RTL packed-slot写bank冲突或随机scratch读延迟显著，必须按真实stall重算；不得沿用本表结果。",
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
    models = []
    for width in (1, 2, 4, 8):
        models.append(
            evaluate(
                profile,
                event_compactor_width=width,
                delivery_efficiency=0.85,
            )
        )
    for width in (1, 2, 4, 8):
        models.append(
            evaluate(
                profile,
                event_compactor_width=width,
                delivery_efficiency=0.85,
                active_token_skip=True,
            )
        )
    cache_models = [
        evaluate(
            profile,
            event_compactor_width=2,
            delivery_efficiency=0.85,
            active_token_skip=True,
            descriptor_cache_terms=depth,
        )
        for depth in (32, 64, 80, 96)
    ]
    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "models": models,
        "cache_models": cache_models,
        "evidence": "[prof ordered trace]+[TERM-CSR完整窗口模型]",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
