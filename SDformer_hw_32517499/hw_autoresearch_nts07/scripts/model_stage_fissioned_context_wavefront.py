#!/usr/bin/env python3
"""用Motion ordered profile评估分裂生命周期三阶段context wavefront。"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import deque
from pathlib import Path
from typing import Any

try:
    from .analyze_hit_flow_ordered_profiles import decode_count_trace
    from .model_motion_ecgb_ordered_profile import (
        finalization_cycles,
        group_payload_bits,
        pingpong_cycles,
    )
except ImportError:
    from analyze_hit_flow_ordered_profiles import decode_count_trace
    from model_motion_ecgb_ordered_profile import (
        finalization_cycles,
        group_payload_bits,
        pingpong_cycles,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_OUT = ROOT / "results/stage_fissioned_context_wavefront_20260801"


def bounded_three_stage_cycles(
    header: list[int],
    build: list[int],
    execute: list[int],
    depth: int,
) -> dict[str, Any]:
    """事件驱动的三阶段credit流水；每级单server、两条FIFO同depth。"""

    if not (header and len(header) == len(build) == len(execute)):
        raise ValueError("三阶段duration长度不一致")
    if depth <= 0:
        raise ValueError("FIFO depth必须为正")
    durations = (header, build, execute)
    queues = (deque(), deque())
    reserved = [0, 0]
    busy = [False, False, False]
    events: list[tuple[int, int, int]] = []
    next_job = 0
    finished = 0
    makespan = 0
    max_reserved = [0, 0]
    starts = [0, 0, 0]

    def start_ready(now: int) -> None:
        nonlocal next_job
        changed = True
        while changed:
            changed = False
            if not busy[2] and queues[1]:
                job = queues[1].popleft()
                reserved[1] -= 1
                busy[2] = True
                starts[2] += 1
                heapq.heappush(events, (now + durations[2][job], 2, job))
                changed = True
            if not busy[1] and queues[0] and reserved[1] < depth:
                job = queues[0].popleft()
                reserved[0] -= 1
                reserved[1] += 1
                max_reserved[1] = max(max_reserved[1], reserved[1])
                busy[1] = True
                starts[1] += 1
                heapq.heappush(events, (now + durations[1][job], 1, job))
                changed = True
            if not busy[0] and next_job < len(header) and reserved[0] < depth:
                job = next_job
                next_job += 1
                reserved[0] += 1
                max_reserved[0] = max(max_reserved[0], reserved[0])
                busy[0] = True
                starts[0] += 1
                heapq.heappush(events, (now + durations[0][job], 0, job))
                changed = True

    start_ready(0)
    while finished < len(header):
        if not events:
            raise RuntimeError("有限流水死锁")
        now = events[0][0]
        completed = []
        while events and events[0][0] == now:
            completed.append(heapq.heappop(events))
        for _, stage, job in completed:
            busy[stage] = False
            if stage == 0:
                queues[0].append(job)
            elif stage == 1:
                queues[1].append(job)
            else:
                finished += 1
                makespan = now
        start_ready(now)
    return {
        "cycles": makespan,
        "jobs": len(header),
        "starts": starts,
        "max_reserved": max_reserved,
    }


def evaluate(profile_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("samples") != 100 or not profile.get("ordered_trace"):
        raise ValueError("需要Motion ordered profile100")
    records = profile["summary"]["h60_records"]
    rows = []
    for group_windows in (1, 4, 8):
        totals = {
            depth: {"cycles": 0, "jobs": 0, "max_reserved": [0, 0]}
            for depth in (1, 2, 4)
        }
        serial_cycles = 0
        two_stage_cycles = 0
        max_payload = 0
        for record in records:
            prefix = "projection_gate_group_"
            classes = decode_count_trace(
                record[f"{prefix}active_classes_g{group_windows}_ordered_trace"]
            )
            terms = decode_count_trace(
                record[f"{prefix}terms_g{group_windows}_ordered_trace"]
            )
            active = decode_count_trace(
                record[f"{prefix}active_lanes_g{group_windows}_ordered_trace"]
            )
            windows = decode_count_trace(
                record[f"{prefix}window_count_g{group_windows}_ordered_trace"]
            )
            delivery = decode_count_trace(
                record[
                    f"{prefix}delivery_g{group_windows}_m4_ordered_trace"
                ]
            )
            if len({len(classes), len(terms), len(active), len(windows), len(delivery)}) != 1:
                raise ValueError("ordered数组长度不一致")
            tokens = int(record["tokens"])
            heads = int(record["num_heads"])
            dim = heads * int(record["head_dim"])
            # S4 whole-context fallback只有mode header，不做fast/raw部分混合。
            header = [1 + count if count <= 4 else 1 for count in classes]
            build = [tokens * count for count in windows]
            execute = [
                heads * max(term, deliver)
                + finalization_cycles(tokens, count, "ibf_pipelined")
                for term, deliver, count in zip(terms, delivery, windows)
            ]
            serial_cycles += sum(header) + sum(build) + sum(execute)
            two_stage_cycles += pingpong_cycles(
                [h + b for h, b in zip(header, build)], execute
            )
            for depth in totals:
                result = bounded_three_stage_cycles(
                    header, build, execute, depth
                )
                totals[depth]["cycles"] += result["cycles"]
                totals[depth]["jobs"] += result["jobs"]
                totals[depth]["max_reserved"] = [
                    max(totals[depth]["max_reserved"][index], value)
                    for index, value in enumerate(result["max_reserved"])
                ]
            max_payload = max(
                max_payload,
                max(
                    group_payload_bits(
                        terms=term,
                        active_lanes=lane_count,
                        dim=dim,
                        tokens=tokens,
                        windows=count,
                    )
                    for term, lane_count, count in zip(terms, active, windows)
                ),
            )
        depth_rows = []
        for depth, result in totals.items():
            depth_rows.append(
                {
                    "depth": depth,
                    **result,
                    "speedup_vs_serial": serial_cycles / result["cycles"],
                    "speedup_vs_two_stage": two_stage_cycles / result["cycles"],
                    "term_fifo_storage_upper_bits": depth * max_payload,
                }
            )
        rows.append(
            {
                "group_windows": group_windows,
                "serial_cycles": serial_cycles,
                "two_stage_cycles": two_stage_cycles,
                "max_term_payload_bits": max_payload,
                "depths": depth_rows,
            }
        )
    return {
        "schema": "stage_fissioned_context_wavefront_v1",
        "profile": str(profile_path.resolve()),
        "evidence": "H67 crop/W9 ordered profile100 + finite credit pipeline model",
        "architecture": (
            "vocabulary header、term directory、product-only accumulator在提交点"
            "转移所有权并提前释放上游bank"
        ),
        "groups": rows,
        "decision": {
            "status": "NO_GO_AS_MAIN_CONTRIBUTION",
            "reason": (
                "depth1收益不足0.5%；depth4的2.8%-3.7%收益依赖67-501Kbit"
                "term FIFO上界，主要是队列容量而非生命周期分裂"
            ),
            "rtl": "不实现独立SFCW RTL；只保留现有提交边界和credit纪律",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SFCW分裂生命周期Context Wavefront有界模型",
        "",
        "> 日期：2026-08-01  ",
        "> 证据：`[prof-ordered]+[bounded-model]`；输入是旧crop/W9 T=162，",
        "> 不是fullres、RTL、DC或功耗结果。",
        "",
        "## 1. 架构定义",
        "",
        "SFCW（Stage-Fissioned Context Wavefront）把context生命周期拆成三个",
        "显式所有权域：",
        "",
        "```text",
        "Vocabulary bank --commit--> Term-directory bank --commit--> Product-only Acc",
        "     header build              token scan/build             product+IBF final",
        "```",
        "",
        "下游接收提交后，上游bank可立即复用，不再让一个monolithic context slot",
        "从header一直占到final。两条FIFO均使用credit；模型在启动任务时预留输出",
        "credit，因此不会用无限队列隐藏反压。",
        "",
        "## 2. Ordered结果",
        "",
        "`two-stage`是header+build绑定后与execute做现有双buffer ping-pong；SFCW",
        "把header和build进一步解耦。",
        "",
        "| G | two-stage周期 | FIFO深度 | SFCW周期 | 相对two-stage | term FIFO上界(bit) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        for row in group["depths"]:
            lines.append(
                f"| {group['group_windows']} | {group['two_stage_cycles']} | "
                f"{row['depth']} | {row['cycles']} | "
                f"{row['speedup_vs_two_stage']:.4f}x | "
                f"{row['term_fifo_storage_upper_bits']} |"
            )
    lines += [
        "",
        "## 3. 判定规则",
        "",
        "- 若depth=1已接近depth=4，说明收益来自生命周期拆分而不是堆FIFO；",
        "- 若相对two-stage不足3%，SFCW只保留为控制组织，不列主贡献；",
        "- term FIFO位数按该G观测到的最大窄目录乘深度，是保守上界，不含宏对齐；",
        "- 只有fullres T450仍保持收益且SRAM可接受，才进入RTL。",
        "",
        "## 4. 本轮冻结与DATE复审",
        "",
        "SFCW作为独立主贡献判定为`NO-GO`：depth=1在G1/G4/G8仅带来约",
        "0.42%/0.15%/0.09%，说明单纯拆分所有权几乎没有吞吐收益；depth=4的",
        "2.8%-3.7%需要明显增加term FIFO，收益来源更接近常规队列扩容。",
        "",
        "因此本轮不写SFCW RTL，也不把它列入DATE贡献。保留的只有两个工程合同：",
        "提交后提前释放上游bank、所有队列使用credit反压。这些合同可服务DVCO、",
        "ECGB和IBF，但不独立宣称新颖性。",
        "",
        "按DATE标准，本负结果避免了继续扩充低收益控制逻辑，但没有提高论文创新",
        "评分。下一候选必须直接减少product/weight/Acc活动或数据搬运，而不能只靠",
        "更深队列隐藏等待。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = evaluate(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
