#!/usr/bin/env python3
"""静态审计GateStack探索SDC与DC前置环境。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SDC = ROOT / "dc_handoff/constraints/gatestack_single_context_500mhz.sdc"
FILELIST = ROOT / "rtl_hitflow/filelist_single_context_execution.f"
OUTPUT = ROOT / "dc_handoff/runs/gatestack_constraint_audit.json"


def main() -> int:
    text = SDC.read_text(encoding="utf-8")
    rtl_files = [line.strip() for line in FILELIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = {
        "主时钟端口为clk_core": "[get_ports clk_core]" in text and "create_clock" in text,
        "500MHz探索周期": "-period 2.000" in text,
        "setup不确定度": "set_clock_uncertainty -setup 0.200" in text,
        "hold不确定度": "set_clock_uncertainty -hold 0.050" in text,
        "输入延迟": "set_input_delay" in text,
        "输出延迟": "set_output_delay" in text,
        "输入转换": "set_input_transition" in text,
        "输出负载": "set_load" in text,
        "最大扇出": "set_max_fanout 32" in text,
        "同步复位未误设false-path": "set_false_path" not in text,
        "filelist非空": bool(rtl_files),
        "filelist全部存在": all((ROOT / item).is_file() for item in rtl_files),
        "顶层在filelist": "rtl_hitflow/gatestack_single_context_execution_top.sv" in rtl_files,
    }
    static_status = "通过" if all(checks.values()) else "失败"
    tools = {name: shutil.which(name) for name in ("dc_shell", "fm_shell", "vcd2saif", "yosys")}
    result = {
        "状态": static_status,
        "约束文件": str(SDC),
        "filelist": str(FILELIST),
        "RTL文件数": len(rtl_files),
        "检查": checks,
        "工具": tools,
        "DC可执行": bool(tools["dc_shell"]),
        "Formality可执行": bool(tools["fm_shell"]),
        "SAIF转换可执行": bool(tools["vcd2saif"]),
        "说明": "静态通过只表示DC输入结构完整；无目标库、PVT和宏模型时不能签核WNS/面积/功耗。",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = OUTPUT.with_suffix(".md")
    lines = [
        "# GateStack DC约束与环境静态审计",
        "",
        f"- 静态约束状态：{static_status}",
        f"- RTL filelist：{len(rtl_files)} 个文件",
        f"- dc_shell：{'可用' if tools['dc_shell'] else '缺失'}",
        f"- fm_shell：{'可用' if tools['fm_shell'] else '缺失'}",
        f"- vcd2saif：{'可用' if tools['vcd2saif'] else '缺失'}",
        f"- yosys：{'可用' if tools['yosys'] else '缺失'}",
        "",
        "## 检查项",
        "",
    ]
    lines.extend(f"- {name}：{'通过' if passed else '失败'}" for name, passed in checks.items())
    lines.extend([
        "",
        "## 结论边界",
        "",
        "本审计不替代DC的 `check_design`、`check_timing`、无约束路径、目标库映射、SAIF注释覆盖率和Formality。当前环境只能完成交付准备与开放结构综合。",
        "",
    ])
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(markdown)
    return 0 if static_status == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
