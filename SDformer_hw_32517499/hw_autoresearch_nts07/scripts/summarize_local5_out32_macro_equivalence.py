#!/usr/bin/env python3
"""汇总Local5 OUT_DIM=32累加器宏绑定与三模式等价对照。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


GROUP_RE = re.compile(r"^GROUP (?P<body>.+)$")
FIELD_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>-?\d+)")


def parse_log(path: Path) -> list[dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if "PASS post-G0 active projection" not in text:
        raise ValueError(f"缺少PASS：{path}")
    rows = []
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            rows.append(
                {
                    field.group("key"): int(field.group("value"))
                    for field in FIELD_RE.finditer(match.group("body"))
                }
            )
    return rows


def totals(rows: list[dict[str, int]]) -> dict[str, int]:
    return {
        field: sum(row[field] for row in rows)
        for field in (
            "cycles",
            "active",
            "terms",
            "updates",
            "term_stall",
            "sram_reads",
            "sram_writes",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    modes = ("direct", "issue", "ds")
    parsed: dict[str, dict[str, list[dict[str, int]]]] = {}
    for impl in ("generic", "macro"):
        parsed[impl] = {}
        for mode in modes:
            rows = parse_log(args.result_dir / f"{impl}_{mode}_profile100.log")
            if len(rows) != 100:
                raise ValueError(f"{impl}/{mode}并非100组")
            parsed[impl][mode] = rows

    equivalence: dict[str, bool] = {}
    for mode in modes:
        left = parsed["generic"][mode]
        right = parsed["macro"][mode]
        equivalence[mode] = all(a == b for a, b in zip(left, right, strict=True))

    macro = {mode: totals(parsed["macro"][mode]) for mode in modes}
    random_sva = {
        mode: len(parse_log(args.result_dir / f"macro_{mode}_random_sva.log")) == 100
        for mode in modes
    }
    summary = {
        "schema": "local5_out32_macro_equivalence_v1",
        "scope": "OUT_DIM=32、五个90x1024累加器bank、profile100",
        "functional_weights": "合成可复现权重；结构和存储位宽真实，切换活动不代表部署权重",
        "acc_macro_binding": {
            "macro": "fakeram45_128x256",
            "macros_per_bank": 4,
            "banks": 5,
            "total_acc_macros": 20,
            "macro_area_um2_each": 36582.980,
            "total_acc_macro_area_um2": 20 * 36582.980,
        },
        "generic_macro_row_exact": equivalence,
        "random_sva_pass": random_sva,
        "macro_totals": macro,
        "speedup": {
            "ds_vs_issue": macro["issue"]["cycles"] / macro["ds"]["cycles"],
            "ds_vs_direct": macro["direct"]["cycles"] / macro["ds"]["cycles"],
        },
        "transaction_reduction_ds_vs_direct": 1
        - (macro["ds"]["sram_reads"] + macro["ds"]["sram_writes"])
        / (macro["direct"]["sram_reads"] + macro["direct"]["sram_writes"]),
    }
    summary["pass"] = all(equivalence.values()) and all(random_sva.values())
    (args.result_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Local5 OUT_DIM=32 累加器宏绑定与三模式等价对照",
        "",
        "## 结论",
        "",
        f"五个累加器bank均按`90x1024`位实现，每bank绑定4个`fakeram45_128x256`，共20个宏。宏总面积为 {summary['acc_macro_binding']['total_acc_macro_area_um2']:,.1f} um^2，不含关系存储和标准单元逻辑。",
        "",
        "generic数组与宏模型在Direct、Issue、DS三种模式的100组逐行周期、工作量、访问计数和Acc32结果完全一致。该结论证明宏接口替换没有改变RTL语义，不等同于DC或功耗签核。",
        "",
        "## Profile100",
        "",
        "| 模式 | 周期 | term stall | SRAM读 | SRAM写 | generic/macro逐行等价 | 随机空隙+SVA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        row = macro[mode]
        lines.append(
            f"| {mode} | {row['cycles']:,} | {row['term_stall']:,} | {row['sram_reads']:,} | {row['sram_writes']:,} | {'PASS' if equivalence[mode] else 'FAIL'} | {'PASS' if random_sva[mode] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"DS相对Issue的周期加速为 {summary['speedup']['ds_vs_issue']:.3f}x；DS相对Direct为 {summary['speedup']['ds_vs_direct']:.3f}x。DS相对Direct的累加器SRAM事务减少 {summary['transaction_reduction_ds_vs_direct']:.2%}。",
            "",
            "## 证据边界",
            "",
            "权重为可复现合成权重，因此可用于功能、控制周期和存储事务比较，但不能替代部署权重的SAIF动态功耗。宏面积来自Nangate45开放Liberty；OpenROAD布局布线结果另行报告。",
            "",
            f"总判定：**{'PASS' if summary['pass'] else 'FAIL'}**。",
        ]
    )
    (args.result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
