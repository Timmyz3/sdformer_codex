#!/usr/bin/env python3
"""评估VL-GS-TTB双词表重叠与Local5有限队列的真实控制开销。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

try:
    from scripts.model_vl_gs_ttb_dual_line import (
        DEFAULT_OUT as OLD_OUT,
        distribution,
        load_local_rows,
        load_motion,
    )
except ModuleNotFoundError:
    from model_vl_gs_ttb_dual_line import (
        DEFAULT_OUT as OLD_OUT,
        distribution,
        load_local_rows,
        load_motion,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/vl_gs_ttb_overlap_fifo_20260801"


def motion_context_work(
    active_classes: Iterable[int], terms: Iterable[int], slots: int
) -> list[tuple[int, int, bool]]:
    """返回(active header周期, body周期, raw fallback)。"""
    work = []
    for classes, body in zip(active_classes, terms):
        classes = int(classes)
        body = int(body)
        if classes < 0 or body < 0 or classes > body:
            raise ValueError("Motion context计数不守恒")
        if body == 0:
            continue
        raw = classes > slots
        # fast header: mode/count一拍，随后每个slot一拍；raw只发模式。
        header = 1 if raw else 1 + classes
        work.append((header, body, raw))
    return work


def two_bank_flowshop(work: list[tuple[int, int, bool]]) -> dict[str, int | float]:
    """两个词表bank下，header构建与body消费的有界flow-shop。"""
    if not work:
        return {
            "contexts": 0,
            "serialized_cycles": 0,
            "dual_bank_cycles": 0,
            "header_cycles": 0,
            "body_cycles": 0,
            "hidden_header_cycles": 0,
            "header_hidden_ratio": 0.0,
            "speedup_vs_serialized_vl": 1.0,
            "header_engine_wait_bank_cycles": 0,
            "body_engine_wait_header_cycles": 0,
        }
    header_finish: list[int] = []
    body_finish: list[int] = []
    previous_header_finish = 0
    previous_body_finish = 0
    header_wait_bank = 0
    body_wait_header = 0
    for index, (header, body, _raw) in enumerate(work):
        bank_free = body_finish[index - 2] if index >= 2 else 0
        header_start = max(previous_header_finish, bank_free)
        header_wait_bank += max(0, bank_free - previous_header_finish)
        current_header_finish = header_start + header
        body_start = max(current_header_finish, previous_body_finish)
        body_wait_header += max(0, current_header_finish - previous_body_finish)
        current_body_finish = body_start + body
        header_finish.append(current_header_finish)
        body_finish.append(current_body_finish)
        previous_header_finish = current_header_finish
        previous_body_finish = current_body_finish
    header_cycles = sum(item[0] for item in work)
    body_cycles = sum(item[1] for item in work)
    serialized = header_cycles + body_cycles
    dual = body_finish[-1]
    hidden = serialized - dual
    return {
        "contexts": len(work),
        "serialized_cycles": serialized,
        "dual_bank_cycles": dual,
        "header_cycles": header_cycles,
        "body_cycles": body_cycles,
        "hidden_header_cycles": hidden,
        "header_hidden_ratio": hidden / header_cycles if header_cycles else 0.0,
        "speedup_vs_serialized_vl": serialized / dual if dual else 1.0,
        "header_engine_wait_bank_cycles": header_wait_bank,
        "body_engine_wait_header_cycles": body_wait_header,
    }


def evaluate_motion_overlap() -> dict[str, object]:
    _records, traces = load_motion()
    policies = []
    for slots in (2, 4, 6, 8):
        all_classes, all_terms = traces[-1]
        aggregate = two_bank_flowshop(
            motion_context_work(all_classes, all_terms, slots)
        )
        sample_results = [
            two_bank_flowshop(motion_context_work(classes, terms, slots))
            for sample, (classes, terms) in traces.items()
            if sample >= 0
        ]
        aggregate.update(
            {
                "slots": slots,
                "sample_speedup": distribution(
                    float(row["speedup_vs_serialized_vl"])
                    for row in sample_results
                ),
                "sample_header_hidden_ratio": distribution(
                    float(row["header_hidden_ratio"])
                    for row in sample_results
                ),
            }
        )
        policies.append(aggregate)
    return {
        "evidence": "[prof-ordered]+[bounded two-bank cycle model]",
        "cycle_definition": (
            "fast header=1个mode/count周期+每active class一周期；raw header=1周期；"
            "body=每term一周期；两个物理slot-table bank交替占用"
        ),
        "policies": policies,
        "claim_boundary": (
            "仅比较串行VL与双bank VL，未把raw 9-bit链路当成同等物理基线"
        ),
    }


def classify_first_bind(
    rows: list[dict[str, int]], slots: int
) -> list[dict[str, int | str]]:
    tables: dict[int, list[int]] = {}
    events: list[dict[str, int | str]] = []
    for sequence, row in enumerate(rows):
        set_id = int(row["lane"])
        gate = int(row["gate"])
        table = tables.setdefault(set_id, [])
        if gate in table:
            kind = "hit"
            slot = table.index(gate)
        elif len(table) < slots:
            kind = "fill"
            slot = len(table)
            table.append(gate)
        else:
            kind = "bypass"
            slot = 0
        events.append(
            {
                "sequence": sequence,
                "kind": kind,
                "set": set_id,
                "slot": slot,
                "gate": gate,
            }
        )
    return events


def ready_always(_cycle: int) -> bool:
    return True


def ready_3_of_4(cycle: int) -> bool:
    return cycle % 4 != 3


def ready_burst_8_4(cycle: int) -> bool:
    return cycle % 12 < 8


READY_PATTERNS: dict[str, Callable[[int], bool]] = {
    "always": ready_always,
    "ready_3_of_4": ready_3_of_4,
    "burst_8_ready_4_stall": ready_burst_8_4,
}


def simulate_local_fifo(
    events: list[dict[str, int | str]],
    depth: int,
    *,
    commit_forward: bool,
    elastic_output: bool,
    downstream_ready: Callable[[int], bool] = ready_always,
) -> dict[str, int | float | bool]:
    """逐周期模拟update/primary/exception三个有界队列与单输出decoder。"""
    if depth < 1:
        raise ValueError("FIFO深度必须为正")
    updates: deque[dict[str, int | str]] = deque()
    primaries: deque[dict[str, int | str]] = deque()
    exceptions: deque[dict[str, int | str]] = deque()
    committed: set[tuple[int, int]] = set()
    output_valid = False
    output_sequence = -1
    producer = 0
    retired = 0
    cycles = 0
    producer_stalls = 0
    fill_head_blocks = 0
    max_update = 0
    max_primary = 0
    max_exception = 0
    max_exception_lead = 0
    timeout = max(100, len(events) * 50)

    while retired < len(events):
        if cycles >= timeout:
            raise RuntimeError("Local FIFO模型未收敛")
        ready = downstream_ready(cycles)
        output_fire = output_valid and ready
        if output_fire:
            retired += 1

        output_space = (not output_valid) or (elastic_output and output_fire)
        update_head = updates[0] if updates else None
        update_key = (
            (int(update_head["set"]), int(update_head["slot"]))
            if update_head is not None
            else None
        )
        primary_head = primaries[0] if primaries else None
        primary_eligible = False
        if primary_head is not None and output_space:
            kind = str(primary_head["kind"])
            if kind == "bypass":
                primary_eligible = bool(
                    exceptions
                    and int(exceptions[0]["sequence"])
                    == int(primary_head["sequence"])
                )
            else:
                key = (int(primary_head["set"]), int(primary_head["slot"]))
                primary_eligible = key in committed or (
                    commit_forward and update_key == key
                )
                if not primary_eligible and kind == "fill":
                    fill_head_blocks += 1

        pop_update = update_head is not None
        pop_primary = primary_head is not None and primary_eligible
        pop_exception = pop_primary and str(primary_head["kind"]) == "bypass"

        if pop_update:
            committed.add(update_key)  # type: ignore[arg-type]
            updates.popleft()
        if pop_primary:
            item = primaries.popleft()
            if pop_exception:
                exceptions.popleft()
            output_valid = True
            output_sequence = int(item["sequence"])
        elif output_fire:
            output_valid = False
            output_sequence = -1

        if producer < len(events):
            event = events[producer]
            kind = str(event["kind"])
            needs_update = kind == "fill"
            needs_exception = kind == "bypass"
            can_push = (
                len(primaries) < depth
                and (not needs_update or len(updates) < depth)
                and (not needs_exception or len(exceptions) < depth)
            )
            if can_push:
                primaries.append(event)
                if needs_update:
                    updates.append(event)
                if needs_exception:
                    exceptions.append(event)
                producer += 1
            else:
                producer_stalls += 1

        max_update = max(max_update, len(updates))
        max_primary = max(max_primary, len(primaries))
        max_exception = max(max_exception, len(exceptions))
        if exceptions:
            max_exception_lead = max(
                max_exception_lead,
                int(exceptions[-1]["sequence"])
                - (output_sequence if output_valid else retired - 1),
            )
        cycles += 1

    kinds = [str(event["kind"]) for event in events]
    return {
        "depth": depth,
        "commit_forward": commit_forward,
        "elastic_output": elastic_output,
        "terms": len(events),
        "fills": kinds.count("fill"),
        "hits": kinds.count("hit"),
        "bypasses": kinds.count("bypass"),
        "cycles": cycles,
        # 注册式入口FIFO首项先入队，末项再从输出寄存器退休。
        "ideal_cycles": len(events) + 2,
        "cycle_overhead_vs_elastic_raw": cycles - (len(events) + 2),
        "term_per_cycle": len(events) / cycles,
        "producer_stalls": producer_stalls,
        "fill_head_block_cycles": fill_head_blocks,
        "max_update_fifo": max_update,
        "max_primary_fifo": max_primary,
        "max_exception_fifo": max_exception,
        "max_exception_sequence_lead": max_exception_lead,
    }


def evaluate_local_fifo() -> dict[str, object]:
    rows = load_local_rows()
    policies = []
    for slots in (4, 6):
        events = classify_first_bind(rows, slots)
        for pattern_name, ready_function in READY_PATTERNS.items():
            for depth in (1, 2, 4, 8):
                for name, commit_forward, elastic_output in (
                    ("registered", False, False),
                    ("elastic_only", False, True),
                    ("atomic_bind_issue", True, True),
                ):
                    row = simulate_local_fifo(
                        events,
                        depth,
                        commit_forward=commit_forward,
                        elastic_output=elastic_output,
                        downstream_ready=ready_function,
                    )
                    row.update(
                        {
                            "slots": slots,
                            "decoder": name,
                            "ready_pattern": pattern_name,
                        }
                    )
                    policies.append(row)
    return {
        "evidence": "[rtl-directed 1498-term trace]+[bounded FIFO cycle model]",
        "policies": policies,
        "claim_boundary": (
            "ready_3_of_4与8/4 burst是敏感性分析，不是实测projection反压"
        ),
    }


def build_report() -> dict[str, object]:
    return {
        "schema": "vl_gs_ttb_overlap_fifo_v1",
        "architecture": {
            "motion": "DVCO：Dual-Vocabulary Context Overlap",
            "local5": "ABIC：Atomic Bind-and-Issue Coalescing",
            "shared": "有界词表生命周期与显式控制/数据平面解耦",
        },
        "motion": evaluate_motion_overlap(),
        "local5": evaluate_local_fifo(),
        "source_model": str(OLD_OUT / "report.json"),
    }


def select_local(
    report: dict[str, object], slots: int, pattern: str, decoder: str
) -> list[dict[str, object]]:
    return [
        row
        for row in report["local5"]["policies"]  # type: ignore[index]
        if row["slots"] == slots
        and row["ready_pattern"] == pattern
        and row["decoder"] == decoder
    ]


def render_markdown(report: dict[str, object]) -> str:
    motion = report["motion"]
    local = report["local5"]
    lines = [
        "# VL-GS-TTB 重叠与有限队列成本模型",
        "",
        "> 日期：2026-08-01  ",
        "> 证据等级：`[prof-ordered]+[bounded-model]` 与",
        "> `[rtl-directed]+[bounded-model]`；不是 RTL 集成结果或 PPA。",
        "",
        "## 1. 本轮回答的问题",
        "",
        "1. Motion 的 eager header 是否会成为串行控制开销；",
        "2. 两个物理词表 bank 能隐藏多少 header 周期；",
        "3. Local5 S4 的 288 个 bypass 在有限 exception FIFO 下是否失控；",
        "4. first-bind 的 update-before-primary 是否引入逐 fill 气泡。",
        "",
        "## 2. Motion：DVCO 双词表上下文重叠",
        "",
        "```text",
        "SCS/header builder -> bank A/B交替commit",
        "                         |",
        "projection body  <- bank A/B交替consume",
        "```",
        "",
        "模型严格限制为两个物理 bank：构建 context i 时，只有当同一 bank 上的",
        "context i-2 已被 body 消费完才能覆盖。fast header 为一个 mode/count 周期",
        "加每个 active class 一周期，raw fallback header 为一周期。",
        "",
        "| slots | context | header周期 | 串行VL周期 | 双bank周期 | header隐藏 | 加速 | sample p95加速 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in motion["policies"]:
        lines.append(
            f"| {row['slots']} | {row['contexts']} | {row['header_cycles']} | "
            f"{row['serialized_cycles']} | {row['dual_bank_cycles']} | "
            f"{row['header_hidden_ratio']:.4%} | "
            f"{row['speedup_vs_serialized_vl']:.4f}x | "
            f"{row['sample_speedup']['p95']:.4f}x |"
        )
    lines += [
        "",
        "这里的加速只相对未重叠的 VL-GS-TTB，不与原始 9-bit raw 链路混淆。",
        "",
        "## 3. Local5：有限队列与 ABIC",
        "",
        "三种消费端：",
        "",
        "- registered：现有一项输出寄存器，退休和再装载不能同拍；",
        "- elastic-only：输出可同拍退休/再装载，但 update 下一拍才可引用；",
        "- ABIC：在 elastic 基础上，将同拍 commit 的 slot 转发给对应 primary。",
        "",
        "### 3.1 无下游反压",
        "",
        "| S | decoder | D | cycles | producer stall | fill block | max P/U/E | E最大领先 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slots in (4, 6):
        for decoder in ("registered", "elastic_only", "atomic_bind_issue"):
            for row in select_local(report, slots, "always", decoder):
                if row["depth"] not in (1, 4):
                    continue
                lines.append(
                    f"| {slots} | {decoder} | {row['depth']} | {row['cycles']} | "
                    f"{row['producer_stalls']} | {row['fill_head_block_cycles']} | "
                    f"{row['max_primary_fifo']}/{row['max_update_fifo']}/"
                    f"{row['max_exception_fifo']} | "
                    f"{row['max_exception_sequence_lead']} |"
                )
    lines += [
        "",
        "### 3.2 突发反压敏感性（8拍ready/4拍stall）",
        "",
        "| S | decoder | D | cycles | producer stall | max P/U/E |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for slots in (4, 6):
        for decoder in ("elastic_only", "atomic_bind_issue"):
            for row in select_local(
                report, slots, "burst_8_ready_4_stall", decoder
            ):
                if row["depth"] not in (1, 4, 8):
                    continue
                lines.append(
                    f"| {slots} | {decoder} | {row['depth']} | {row['cycles']} | "
                    f"{row['producer_stalls']} | {row['max_primary_fifo']}/"
                    f"{row['max_update_fifo']}/{row['max_exception_fifo']} |"
                )
    lines += [
        "",
        "## 4. 证据边界与晋级规则",
        "",
        f"- Motion：{motion['claim_boundary']}；",
        f"- Local5：{local['claim_boundary']}；",
        "- 若 ABIC 只修复现有 decoder 的吞吐缺陷而无系统 EDP，不能单列贡献；",
        "- 若 DVCO 几乎完全隐藏 header，下一步实现双 bank 协议与覆盖保护；",
        "- Local5 参数冻结仍需 fullres 多样本和真实 backend ready trace；",
        "- 所有周期均不包含 SRAM macro latency、CDC、物理布线和时钟门控。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
