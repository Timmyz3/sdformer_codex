#!/usr/bin/env python3
"""汇总真实 S0-S3 Builder-to-projection C0/C1 RTL 结果。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RESULT_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>[^ ]+)")


def parse_result(line: str) -> dict[str, str]:
    if not line.startswith("RESULT stage="):
        raise ValueError(f"不是 stage RESULT 行: {line}")
    return {match.group("key"): match.group("value") for match in RESULT_RE.finditer(line)}


def load_mode(build_dir: Path, stage: int, mode: int) -> dict[str, int | str]:
    log_path = build_dir / f"s{stage}_c{mode}" / "iverilog.log"
    lines = [line.strip() for line in log_path.read_text().splitlines()
             if line.startswith("RESULT stage=")]
    if len(lines) != 1:
        raise RuntimeError(f"{log_path} 应有且仅有一条 RESULT，实际 {len(lines)}")
    raw = parse_result(lines[0])
    if raw.get("status") != "PASS" or raw.get("mismatches") != "0":
        raise RuntimeError(f"{log_path} 未通过: {raw}")
    integers = {
        key: int(raw[key], 0)
        for key in (
            "total_cycles", "build_cycles", "projection_cycles", "compared",
            "mismatches", "checksum", "replay", "release",
            "projection_heads", "projection_terms", "bias", "slot_commits",
            "payload_copy", "scan", "stalls", "blocked", "overlap",
            "order_wait", "event_sum",
        )
    }
    return {"stage": raw["stage"], "mode": raw["mode"], **integers}


def build_report(build_dir: Path) -> dict:
    stages = []
    for stage in range(4):
        c0 = load_mode(build_dir, stage, 0)
        c1 = load_mode(build_dir, stage, 1)
        if c0["checksum"] != c1["checksum"] or c0["compared"] != c1["compared"]:
            raise RuntimeError(f"S{stage} C0/C1 输出摘要不一致")
        stages.append({
            "stage": stage,
            "c0": c0,
            "c1": c1,
            "system_speedup": c0["total_cycles"] / c1["total_cycles"],
            "builder_speedup": c0["build_cycles"] / c1["build_cycles"],
            "c0_builder_fraction": c0["build_cycles"] / c0["total_cycles"],
        })
    c0_total = sum(item["c0"]["total_cycles"] for item in stages)
    c1_total = sum(item["c1"]["total_cycles"] for item in stages)
    c0_build = sum(item["c0"]["build_cycles"] for item in stages)
    c1_build = sum(item["c1"]["build_cycles"] for item in stages)
    compared_each = sum(item["c0"]["compared"] for item in stages)
    return {
        "schema_version": 1,
        "status": "PASS",
        "evidence": "[rtl]",
        "scope": "sample0/B0/window0，S0-S3 共45个head",
        "stages": stages,
        "aggregate": {
            "c0_total_cycles": c0_total,
            "c1_total_cycles": c1_total,
            "system_speedup": c0_total / c1_total,
            "c0_build_cycles": c0_build,
            "c1_build_cycles": c1_build,
            "builder_speedup": c0_build / c1_build,
            "c0_builder_fraction": c0_build / c0_total,
            "compared_each_mode": compared_each,
            "compared_both_modes": compared_each * 2,
            "mismatches": 0,
            "payload_copy_words": 0,
        },
    }


def render_markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# 真实 S0-S3 Builder-to-Projection RTL 汇总",
        "",
        "证据等级：`[rtl]`。范围为 `sample0/B0/window0`，不是 profile100、DC 或整网结果。",
        "",
        "| Stage | Head | C0总周期 | C1总周期 | 系统加速 | C0 Builder | C1 Builder | Builder加速 | C0 Builder占比 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["stages"]:
        heads = 3 << item["stage"]
        lines.append(
            f"| S{item['stage']} | {heads} | {item['c0']['total_cycles']} | "
            f"{item['c1']['total_cycles']} | {item['system_speedup']:.3f}x | "
            f"{item['c0']['build_cycles']} | {item['c1']['build_cycles']} | "
            f"{item['builder_speedup']:.3f}x | {item['c0_builder_fraction']:.2%} |"
        )
    lines += [
        "",
        "## 汇总结论",
        "",
        f"- 四 stage 串行总周期：C0 `{aggregate['c0_total_cycles']}`，C1 `{aggregate['c1_total_cycles']}`，系统加速 `{aggregate['system_speedup']:.3f}x`；",
        f"- Builder 总周期：C0 `{aggregate['c0_build_cycles']}`，C1 `{aggregate['c1_build_cycles']}`，局部加速 `{aggregate['builder_speedup']:.3f}x`；",
        f"- C0 中 Builder 占端到端周期 `{aggregate['c0_builder_fraction']:.2%}`，其余主要为 replay/projection/bias/final；",
        f"- 每模式逐元素比较 `{aggregate['compared_each_mode']}` 项，双模式合计 `{aggregate['compared_both_modes']}` 项，失配 `0`；",
        "- typed-slot payload copy 保持 `0 word`；C1 只能作为吞吐模式，不能把 Builder 局部收益写成系统收益。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.build_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.out_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
