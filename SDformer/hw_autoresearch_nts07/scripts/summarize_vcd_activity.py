#!/usr/bin/env python3
"""流式汇总 VCD 的已知值翻转，输出 JSON 与中文 Markdown。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


VAR_RE = re.compile(r"^\$var\s+\S+\s+(\d+)\s+(\S+)\s+(.+?)\s+\$end$")


def bit_toggles(old: str, new: str, width: int) -> int:
    old_bits = old.lower().zfill(width)[-width:]
    new_bits = new.lower().zfill(width)[-width:]
    return sum(
        left in "01" and right in "01" and left != right
        for left, right in zip(old_bits, new_bits)
    )


def parse_vcd(path: Path) -> dict[str, object]:
    scopes: list[str] = []
    signals: dict[str, tuple[int, str]] = {}
    previous: dict[str, str] = {}
    toggles: Counter[str] = Counter()
    updates: Counter[str] = Counter()
    current_time = 0
    first_time: int | None = None
    last_time = 0
    in_header = True

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if in_header:
                if line.startswith("$scope "):
                    fields = line.split()
                    scopes.append(fields[2])
                elif line == "$upscope $end":
                    if scopes:
                        scopes.pop()
                elif line.startswith("$var "):
                    match = VAR_RE.match(line)
                    if match:
                        width, identifier, reference = match.groups()
                        signals[identifier] = (
                            int(width), ".".join([*scopes, reference])
                        )
                elif line == "$enddefinitions $end":
                    in_header = False
                continue

            if line.startswith("#"):
                current_time = int(line[1:])
                if first_time is None:
                    first_time = current_time
                last_time = current_time
                continue
            if line[0] in "01xXzZ":
                value, identifier = line[0], line[1:]
            elif line[0] in "bB":
                fields = line[1:].split()
                if len(fields) != 2:
                    continue
                value, identifier = fields
            else:
                continue
            if identifier not in signals:
                continue
            width, name = signals[identifier]
            if identifier in previous:
                toggles[name] += bit_toggles(previous[identifier], value, width)
            previous[identifier] = value
            updates[name] += 1

    hierarchy_toggles: Counter[str] = Counter()
    for name, count in toggles.items():
        parts = name.split(".")
        if len(parts) >= 4:
            group = ".".join(parts[:3])
        elif len(parts) >= 3:
            group = ".".join(parts[:2])
        else:
            group = name
        hierarchy_toggles[group] += count

    ranked_signals = [
        {"signal": name, "toggles": count, "updates": updates[name]}
        for name, count in toggles.most_common(30)
    ]
    ranked_hierarchy = [
        {"hierarchy": name, "toggles": count}
        for name, count in hierarchy_toggles.most_common()
    ]
    return {
        "source": str(path),
        "file_bytes": path.stat().st_size,
        "timescale_ticks": {
            "first": first_time or 0,
            "last": last_time,
            "span": last_time - (first_time or 0),
        },
        "declared_variables": len(signals),
        "variables_with_updates": len(previous),
        "total_known_bit_toggles": sum(toggles.values()),
        "ranked_hierarchy": ranked_hierarchy,
        "ranked_signals": ranked_signals,
    }


def write_markdown(result: dict[str, object], path: Path) -> None:
    ticks = result["timescale_ticks"]
    assert isinstance(ticks, dict)
    hierarchy = result["ranked_hierarchy"]
    signals = result["ranked_signals"]
    assert isinstance(hierarchy, list)
    assert isinstance(signals, list)
    lines = [
        "# GateStack T162 RTL 切换活动审计",
        "",
        "## 1. 证据边界",
        "",
        "本报告来自 Icarus RTL 仿真的 VCD 已知值翻转统计，不是 SAIF 功耗、门级功耗或目标工艺 PPA。未知态到已知态不计入 bit toggle。",
        "",
        "## 2. 总览",
        "",
        f"- VCD 文件：`{result['source']}`",
        f"- 文件大小：{int(result['file_bytes']) / (1024 * 1024):.2f} MiB",
        f"- 仿真 tick 范围：{ticks['first']} 至 {ticks['last']}，跨度 {ticks['span']}",
        f"- 声明变量数：{result['declared_variables']}",
        f"- 有更新变量数：{result['variables_with_updates']}",
        f"- 已知值 bit toggle 总数：{result['total_known_bit_toggles']}",
        "",
        "## 3. 层次翻转分布",
        "",
        "| 层次 | bit toggle |",
        "|---|---:|",
    ]
    for item in hierarchy[:20]:
        lines.append(f"| `{item['hierarchy']}` | {item['toggles']} |")
    lines.extend([
        "",
        "## 4. 高翻转信号",
        "",
        "| 信号 | bit toggle | value update |",
        "|---|---:|---:|",
    ])
    for item in signals[:20]:
        lines.append(
            f"| `{item['signal']}` | {item['toggles']} | {item['updates']} |"
        )
    lines.extend([
        "",
        "## 5. 使用方式",
        "",
        "目标库环境可用后，应使用同一 workload 的门级或至少综合网表活动重新生成 SAIF，并在报告中注明层次映射、时钟周期、复位区间和注释覆盖率。本文件只能用于检查 VCD 非空、主要层次有活动以及后续功耗流程输入可重复。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vcd", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = parse_vcd(args.vcd)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(result, args.markdown)


if __name__ == "__main__":
    main()
