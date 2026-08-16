#!/usr/bin/env python3
"""汇总GateStack、no-residency与RAW41-only同顶层RTL消融。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRACE_RE = re.compile(
    r"slot=(?P<slot_replays>\d+)/(?P<slot_releases>\d+) "
    r"cache=(?P<cache_hits>\d+)/(?P<cache_releases>\d+) "
    r"proj=(?P<projection_heads>\d+)/(?P<projection_terms>\d+).*"
    r"final=(?P<finals>\d+) mismatch=(?P<mismatches>\d+)"
)
CYCLE_RE = re.compile(r"group_cycles=(?P<cycles>\d+)")
ERROR_RE = re.compile(
    r"error=(?P<done_error>\d+) protocol=(?P<protocol_errors>\d+)"
)


def parse_log(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    trace = TRACE_RE.search(text)
    cycles = CYCLE_RE.search(text)
    errors = ERROR_RE.search(text)
    if trace is None or cycles is None or errors is None:
        raise ValueError(f"无法解析RTL日志: {path}")
    result = {key: int(value) for key, value in trace.groupdict().items()}
    result.update({key: int(value) for key, value in errors.groupdict().items()})
    result["cycles"] = int(cycles.group("cycles"))
    if (
        result["mismatches"] != 0
        or result["done_error"] != 0
        or result["protocol_errors"] != 0
    ):
        raise ValueError(f"RTL结果不等价: {path}")
    return result


def load_payload_words(manifest: Path, field: str) -> int:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return int(data["totals"][field])


def load_yosys_structure(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))["design"]
    types = data["num_cells_by_type"]
    return {
        "generic_cells": int(data["num_cells"]),
        "logical_memories": int(types.get("$mem_v2", 0)),
        "generic_muls": int(types.get("$mul", 0)),
        "generic_muxes": int(types.get("$mux", 0)),
    }


def summarize(root: Path) -> dict:
    build = root / "build_hitflow/gatestack_p0_baselines"
    rows = {
        mode: parse_log(build / mode / "verilator.log")
        for mode in ("gatestack", "no_residency", "raw_only")
    }
    capacity_manifest = root / "results/gatestack_h67_stage3_trace_20260716/manifest.json"
    raw_manifest = root / "results/gatestack_h67_stage3_trace_rawonly_20260717/manifest.json"
    rows["gatestack"]["payload_words"] = load_payload_words(
        capacity_manifest, "residency_payload_words_all_tiles"
    )
    rows["no_residency"]["payload_words"] = load_payload_words(
        capacity_manifest, "no_residency_payload_words_all_tiles"
    )
    rows["raw_only"]["payload_words"] = load_payload_words(
        raw_manifest, "no_residency_payload_words_all_tiles"
    )
    full_structure = load_yosys_structure(
        root / "dc_handoff/runs/yosys_structure/gatestack_single_context_execution_top/stat.json"
    )
    no_residency_structure = load_yosys_structure(
        build / "yosys_no_residency/stat.json"
    )
    rows["gatestack"].update(full_structure)
    rows["no_residency"].update(no_residency_structure)
    # RAW-only is currently a runtime-path baseline through the same top. It is
    # not yet the physically stripped direct engine required for an area table.
    rows["raw_only"].update(full_structure)
    raw_cycles = rows["raw_only"]["cycles"]
    raw_words = rows["raw_only"]["payload_words"]
    raw_terms = rows["raw_only"]["projection_terms"]
    for row in rows.values():
        row["speedup_vs_raw"] = raw_cycles / row["cycles"]
        row["payload_reduction_vs_raw"] = 1.0 - row["payload_words"] / raw_words
        row["term_reduction_vs_raw"] = 1.0 - row["projection_terms"] / raw_terms
    return {
        "status": "PASS",
        "evidence": "[RTL trace-shaped workload]",
        "rows": rows,
        "limits": [
            "三种模式使用同一single-context执行顶层、相同T162/H24/O24/L32和相同构造数值语义",
            "当前payload由真实ordered统计塑形，并非真实网络bit trace",
            "cycle不含完整encoder、外存、bias/requant和目标库时序",
            "payload word是逻辑64-bit传输计数，不是功耗",
            "RAW-only当前复用完整顶层，只能作周期/流量基线，不能作物理面积基线",
            "Yosys generic cell与logical memory只作结构审计，不是目标库面积",
        ],
    }


def write_markdown(path: Path, result: dict) -> None:
    labels = {
        "gatestack": "GateStack完整机制",
        "no_residency": "IPD无驻留",
        "raw_only": "RAW41-only运行路径",
    }
    lines = [
        "# GateStack P0公平基线RTL消融",
        "",
        "## 结论",
        "",
        "三种模式复用同一single-context执行顶层、相同计算后端和相同trace-shaped数值语义，整数输出均为零mismatch。",
        "",
        "| 模式 | 周期 | 相对RAW加速 | payload words | 相对RAW减少 | projection terms | 相对RAW减少 | slot replay | cache hit | Yosys generic cell | logical memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("raw_only", "no_residency", "gatestack"):
        row = result["rows"][mode]
        lines.append(
            f"| {labels[mode]} | {row['cycles']} | {row['speedup_vs_raw']:.3f}x | "
            f"{row['payload_words']} | {row['payload_reduction_vs_raw']:.2%} | "
            f"{row['projection_terms']} | {row['term_reduction_vs_raw']:.2%} | "
            f"{row['slot_replays']} | {row['cache_hits']} | "
            f"{row['generic_cells']} | {row['logical_memories']} |"
        )
    lines.extend(
        [
            "",
            "## 审稿边界",
            "",
            "- 该表首次形成同顶层、同规模、同输出语义的RTL周期基线，可用于验证机制方向。",
            "- 它仍不是目标库PPA，也不允许把payload减少等同于节能。",
            "- RAW-only当前仍经过完整顶层；其Yosys结构数不能用于宣称Direct基线面积，后续需实现物理裁剪版。",
            "- no-residency编译期删除descriptor cache/auto-fill后，结构由12个降至9个logical memory；generic cell只作趋势。",
            "- 下一轮必须将真实四stage bit trace送入同一消融矩阵，替换当前统计塑形主表。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", result)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
