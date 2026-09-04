#!/usr/bin/env python3
"""检查一次DC运行是否生成完整交接工件，不替代QoR人工签核。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    always_required = [
        args.run_dir / "dc.log",
        args.run_dir / "dc_run_manifest.json",
        args.run_dir / "netlist" / f"{args.design}_mapped.v",
        args.run_dir / "netlist" / f"{args.design}_mapped.sdc",
        args.run_dir / "netlist" / f"{args.design}.ddc",
        args.run_dir / "netlist" / f"{args.design}.svf",
        args.run_dir / "reports" / "qor.rpt",
        args.run_dir / "reports" / "area.rpt",
        args.run_dir / "reports" / "power_scope.rpt",
        args.run_dir / "reports" / "references.rpt",
        args.run_dir / "reports" / "timing_setup.rpt",
        args.run_dir / "reports" / "timing_hold.rpt",
        args.run_dir / "reports" / "timing_unconstrained.rpt",
        args.run_dir / "reports" / "constraint_violators.rpt",
        args.run_dir / "reports" / "clock_gating.rpt",
        args.run_dir / "reports" / "check_design_postcompile.rpt",
        args.run_dir / "reports" / "check_timing_postcompile.rpt",
    ]
    power_scope_path = args.run_dir / "reports" / "power_scope.rpt"
    power_scope = (
        power_scope_path.read_text(encoding="utf-8", errors="replace")
        if power_scope_path.is_file()
        else ""
    )
    saif_power = "scope=SAIF_ANNOTATED_EXPLORATORY" in power_scope
    power_paths = [
        args.run_dir / "reports" / "power.rpt",
        args.run_dir / "reports" / "power_hierarchy.rpt",
    ]
    required = always_required + (power_paths if saif_power else [])
    checks = {str(path): path.is_file() and path.stat().st_size > 0 for path in required}
    stale_default_power = (
        not saif_power and any(path.exists() for path in power_paths)
    )
    passed = all(checks.values())
    passed = passed and not stale_default_power
    result = {
        "状态": "工件齐全" if passed else "工件缺失",
        "设计": args.design,
        "运行目录": str(args.run_dir),
        "工件": checks,
        "功耗口径": "SAIF活动" if saif_power else "未运行功耗",
        "无SAIF遗留功耗报告": stale_default_power,
        "边界": "该检查只验证文件存在；WNS/TNS、无约束路径、违例、面积和功耗仍必须逐项签核。",
    }
    json_path = args.run_dir / "工件审计.json"
    md_path = args.run_dir / "工件审计.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Design Compiler工件审计\n\n"
        f"- 设计：`{args.design}`\n"
        f"- 状态：**{result['状态']}**\n"
        f"- 必需工件：`{sum(checks.values())}/{len(checks)}`。\n\n"
        "本审计只检查文件完整性，不代表时序、面积、功耗或等价验证通过。\n",
        encoding="utf-8",
    )
    print(md_path)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
