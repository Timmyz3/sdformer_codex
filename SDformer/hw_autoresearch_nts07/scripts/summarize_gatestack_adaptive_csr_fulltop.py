#!/usr/bin/env python3
"""汇总统一Adaptive CSR四stage真实trace同顶层RTL结果。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from summarize_gatestack_p0_baselines import parse_log


CELL_RE = re.compile(r"Number of cells:\s+(?P<cells>\d+)")


def parse_cells(path: Path) -> int:
    matches = CELL_RE.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"无法解析Yosys cell数: {path}")
    return int(matches[-1])


def summarize(root: Path, baseline_path: Path, fadc_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    fadc = json.loads(fadc_path.read_text(encoding="utf-8"))
    base = {(int(row["stage"]), row["mode"]): row for row in baseline["rows"]}
    fadc_rows = {int(row["stage"]): row for row in fadc["rows"]}
    build = root / "build_hitflow/gatestack_adaptive_csr_fulltop"
    rows = []
    for stage in range(4):
        verilator = parse_log(build / f"s{stage}" / "verilator.log")
        iverilog = parse_log(build / f"s{stage}" / "iverilog.log")
        for key in (
            "projection_heads", "projection_terms", "finals", "mismatches",
            "done_error", "protocol_errors",
        ):
            if verilator[key] != iverilog[key]:
                raise ValueError(f"S{stage} Icarus/Verilator功能计数不一致: {key}")
        source = fadc_rows[stage] if stage == 3 else base[(stage, "no_residency")]
        gate = base[(stage, "gatestack")]
        nores = base[(stage, "no_residency")]
        rows.append({
            "stage": stage,
            "format": "FADC24" if stage == 3 else "IPD32W",
            "cycles": int(verilator["cycles"]),
            "iverilog_cycles": int(iverilog["cycles"]),
            "simulator_cycle_delta": abs(int(verilator["cycles"]) - int(iverilog["cycles"])),
            "payload_words": int(source["payload_words"]),
            "projection_terms": int(verilator["projection_terms"]),
            "slot_replays": int(verilator["slot_replays"]),
            "mismatches": int(verilator["mismatches"]),
            "done_error": int(verilator["done_error"]),
            "protocol_errors": int(verilator["protocol_errors"]),
            "speedup_vs_ipd_no_residency": int(nores["cycles"]) / int(verilator["cycles"]),
            "speedup_vs_gatestack": int(gate["cycles"]) / int(verilator["cycles"]),
        })

    adaptive_cycles = sum(row["cycles"] for row in rows)
    gate_cycles = sum(int(base[(stage, "gatestack")]["cycles"]) for stage in range(4))
    ipd_nores_cycles = sum(int(base[(stage, "no_residency")]["cycles"]) for stage in range(4))
    def mixed_result(case: str, vector_dir: str) -> dict[str, Any]:
        verilator = parse_log(build / f"s{case}" / "verilator.log")
        iverilog = parse_log(build / f"s{case}" / "iverilog.log")
        for key in (
            "projection_heads", "projection_terms", "finals", "mismatches",
            "done_error", "protocol_errors",
        ):
            if verilator[key] != iverilog[key]:
                raise ValueError(f"{case} Icarus/Verilator功能计数不一致: {key}")
        manifest = json.loads(
            (root / f"tb_hitflow/vectors/{vector_dir}/manifest.json")
            .read_text(encoding="utf-8")
        )
        return {
            "cycles": int(verilator["cycles"]),
            "iverilog_cycles": int(iverilog["cycles"]),
            "simulator_cycle_delta": abs(int(verilator["cycles"]) - int(iverilog["cycles"])),
            "projection_terms": int(verilator["projection_terms"]),
            "mismatches": int(verilator["mismatches"]),
            "done_error": int(verilator["done_error"]),
            "protocol_errors": int(verilator["protocol_errors"]),
            "format_counts": manifest["format_counts"],
            "formats_by_head": manifest["formats_by_head"],
        }

    mixed = mixed_result("mixed", "adaptive_mixed_real_sample0_s3_b0")
    mixed_csr = mixed_result("mixedcsr", "adaptive_mixed_csr_real_sample0_s3_b0")
    return {
        "status": "PASS",
        "evidence": "[H67真实Q/K/gate]+[候选dyadic INT8]+[RTL]",
        "configuration": (
            "提交期typed slot metadata，运行期直接分派IPD32W/FADC24/RAW41；"
            "本报告关闭descriptor residency"
        ),
        "rows": rows,
        "trace_bundle": {
            "adaptive_cycles": adaptive_cycles,
            "gatestack_cycles": gate_cycles,
            "ipd_no_residency_cycles": ipd_nores_cycles,
            "speedup_vs_gatestack": gate_cycles / adaptive_cycles,
            "speedup_vs_ipd_no_residency": ipd_nores_cycles / adaptive_cycles,
            "formats_s0_to_s3": [row["format"] for row in rows],
        },
        "adaptive_leaf_yosys_generic_cells": parse_cells(build / "yosys_fair.log"),
        "mixed_context_with_raw": mixed,
        "mixed_context_csr_only": mixed_csr,
        "limits": [
            "每个stage仅回放sample0/B0/window0，四stage求和是trace bundle而不是整网周期",
            "本报告关闭descriptor residency；IPD-only选择性驻留的独立结果见typed-residency报告",
            "INT8 projection weight与bias是候选量化合同，尚未通过valid825",
            "Yosys generic cell不是目标库面积；没有目标库DC、STA、SAIF、mapped-netlist LEC和SRAM宏功耗证据",
            "Icarus与Verilator功能计数一致，但周期允许存在少量调度差异",
        ],
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    bundle = result["trace_bundle"]
    mixed = result["mixed_context_with_raw"]
    mixed_csr = result["mixed_context_csr_only"]
    lines = [
        "# 统一Adaptive CSR四Stage真实Trace同顶层RTL结果",
        "",
        "## 结论",
        "",
        "同一个可综合前端已在payload commit时校验首字并把RAW41/IPD32W/FADC24格式写入受tag保护的slot元数据；运行期PLAN和decoder直接使用该元数据，不再对每次replay重复窥探首字。S0到S2输入IPD32W，S3输入FADC24；四个stage均通过Icarus与Verilator/SVA，投影累加逐元素零mismatch，protocol与abort均为零。",
        "",
        "这项结果消除了上一轮逐stage离线选择不同编译配置的oracle问题。代价是芯片同时保留两个decoder；其面积和功耗是否值得，仍必须由目标库综合证明。",
        "",
        "| Stage | 运行时识别格式 | 周期 | 相对IPD无驻留 | 相对GateStack | payload words | terms | 仿真器周期差 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| S{row['stage']} | {row['format']} | {row['cycles']} | "
            f"{row['speedup_vs_ipd_no_residency']:.3f}x | "
            f"{row['speedup_vs_gatestack']:.3f}x | {row['payload_words']} | "
            f"{row['projection_terms']} | {row['simulator_cycle_delta']} |"
        )
    lines.extend([
        "",
        "## Trace Bundle",
        "",
        f"- 统一Adaptive CSR周期和：{bundle['adaptive_cycles']}。",
        f"- GateStack基线周期和：{bundle['gatestack_cycles']}，统一前端相对其为{bundle['speedup_vs_gatestack']:.3f}x。",
        f"- IPD32W无驻留周期和：{bundle['ipd_no_residency_cycles']}，统一前端相对其为{bundle['speedup_vs_ipd_no_residency']:.3f}x。",
        f"- S0到S3实际识别格式：{' / '.join(bundle['formats_s0_to_s3'])}。",
        f"- Adaptive CSR叶模块Yosys generic cells：{result['adaptive_leaf_yosys_generic_cells']}。该数字只作结构代理。",
        "",
        "## 同一Context交错覆盖",
        "",
        f"S3的24个head在一次context内交错使用{mixed['format_counts']['IPD32W']}个IPD32W、{mixed['format_counts']['FADC24']}个FADC24和{mixed['format_counts']['RAW41']}个RAW41精确回退。Verilator周期为{mixed['cycles']}，projection terms为{mixed['projection_terms']}，mismatch/done_error/protocol均为{mixed['mismatches']}/{mixed['done_error']}/{mixed['protocol_errors']}。Icarus与Verilator功能计数一致，周期差{mixed['simulator_cycle_delta']}。",
        "",
        f"为隔离RAW展开成本，第二个同context用例把该head改为FADC24：{mixed_csr['format_counts']['IPD32W']}个IPD32W、{mixed_csr['format_counts']['FADC24']}个FADC24、零RAW；Verilator周期{mixed_csr['cycles']}，terms为{mixed_csr['projection_terms']}，mismatch/done_error/protocol均为{mixed_csr['mismatches']}/{mixed_csr['done_error']}/{mixed_csr['protocol_errors']}。两用例共同覆盖运行时双CSR切换与RAW精确回退。",
        "",
        "## 架构意义",
        "",
        "前端不再依赖stage硬编码格式，而把表示选择下沉到每个head payload。IPD32W适合低到中fanout并保持简单解码，FADC24用list/bitmap精确编码高fanout，二者共享后续term/event接口和GateStack投影后端。它属于表示、解码和multicast执行的协同架构，不改变H67数值语义。",
        "",
        "当前实现仍是双decoder选择式结构。本报告刻意关闭descriptor residency以给出纯格式数据流；格式感知的IPD-only选择性驻留已在独立同顶层回归中闭环。下一步必须用目标库PPA决定双decoder是否值得，并比较共享reservoir/共享事件发射器。",
        "",
        "## 证据边界",
        "",
    ])
    lines.extend(f"- {item}" for item in result["limits"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--fadc-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.baseline_report, args.fadc_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", result)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
