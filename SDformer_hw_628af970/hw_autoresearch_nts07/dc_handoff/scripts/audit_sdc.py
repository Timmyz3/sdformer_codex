#!/usr/bin/env python3
"""静态审计H67/H68探索SDC的必需约束。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SDC = ROOT / "dc_handoff/constraints/h67_h68_500mhz.sdc"
OUTPUT = ROOT / "dc_handoff/runs/constraint_audit.json"


def main() -> int:
    text = SDC.read_text(encoding="utf-8")
    required = {
        "主时钟": "create_clock",
        "setup不确定度": "set_clock_uncertainty -setup",
        "hold不确定度": "set_clock_uncertainty -hold",
        "输入延迟": "set_input_delay",
        "输出延迟": "set_output_delay",
        "输入转换": "set_input_transition",
        "输出负载": "set_load",
        "最大扇出": "set_max_fanout",
    }
    checks = {name: token in text for name, token in required.items()}
    checks["同步复位未错误设置false path"] = "set_false_path" not in text
    checks["同步复位纳入输入时序"] = (
        "remove_from_collection [all_inputs] [get_ports clk]" in text
        and "[get_ports {clk rst_n}]" not in text
    )
    status = "通过" if all(checks.values()) else "失败"
    result = {
        "状态": status,
        "约束文件": str(SDC),
        "检查": checks,
        "说明": "本检查仅验证约束覆盖结构；真实单位、工艺角和无约束路径必须由DC报告复核。",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = OUTPUT.with_suffix(".md")
    md.write_text(
        "# H67/H68 SDC静态审计\n\n"
        f"- 状态：{status}\n"
        + "".join(f"- {name}：{'通过' if passed else '失败'}\n" for name, passed in checks.items())
        + "\n该检查不能替代DC的 `check_timing`、无约束路径和PVT报告。\n",
        encoding="utf-8",
    )
    print(md)
    return 0 if status == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
