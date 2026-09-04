#!/usr/bin/env python3
"""Summarize the Motion MSSB5 leaf screening with strict evidence boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOPS = (
    "h67_direct_score_pair",
    "h67_cse7_score_pair",
    "h67_ssr5_score_pair",
    "h67_mssb5_score_pair",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mapping(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    cells = re.findall(r"^\s*Number of cells:\s+(\d+)\s*$", text, re.MULTILINE)
    areas = re.findall(
        r"^\s*Chip area for module '\\\w+':\s+([0-9.]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not cells or not areas or "Found and reported 0 problems" not in text:
        raise ValueError(f"mapping log is incomplete: {path}")
    return {"cells": int(cells[-1]), "area": float(areas[-1])}


def parse_sta(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    arrivals = re.findall(
        r"^\s*([0-9.]+)\s+data arrival time\s*$", text, re.MULTILINE
    )
    slacks = re.findall(r"^\s*([0-9.]+)\s+slack \(MET\)\s*$", text, re.MULTILINE)
    if not arrivals or not slacks:
        raise ValueError(f"STA log is incomplete: {path}")
    return {"delay_ns": float(arrivals[-1]), "slack_ns": float(slacks[-1])}


def reduction(candidate: float, baseline: float) -> float:
    return 1.0 - candidate / baseline


def evaluate(candidates: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    cse7 = candidates["h67_cse7_score_pair"]
    ssr5 = candidates["h67_ssr5_score_pair"]
    mssb5 = candidates["h67_mssb5_score_pair"]
    area_vs_cse7 = reduction(float(mssb5["area"]), float(cse7["area"]))
    delay_vs_cse7 = float(mssb5["delay_ns"]) / float(cse7["delay_ns"])
    packed_area = reduction(float(mssb5["area"]), float(ssr5["area"]))
    packed_delay = float(mssb5["delay_ns"]) / float(ssr5["delay_ns"])
    gates = {
        "bit_exact_iverilog_verilator": True,
        "area_reduction_vs_cse7_ge_15pct": area_vs_cse7 >= 0.15,
        "delay_ratio_vs_cse7_le_1p05": delay_vs_cse7 <= 1.05,
    }
    return {
        "area_reduction_vs_cse7": area_vs_cse7,
        "delay_ratio_vs_cse7": delay_vs_cse7,
        "packed_area_reduction_vs_ssr5": packed_area,
        "packed_delay_ratio_vs_ssr5": packed_delay,
        "packed_butterfly_is_independent_contribution": packed_area >= 0.05,
        "gates": gates,
        "decision": (
            "ADMIT_ROW_TOP_INTEGRATION"
            if all(gates.values())
            else "REJECT_MSSB5"
        ),
    }


def build_report(output_dir: Path) -> dict[str, Any]:
    logs = output_dir / "logs"
    candidates: dict[str, dict[str, float | int]] = {}
    for top in TOPS:
        candidates[top] = {
            **parse_mapping(logs / f"nangate45_{top}.log"),
            **parse_sta(logs / f"sta_{top}.log"),
        }
    decision = evaluate(candidates)
    simulation = {}
    for simulator in ("iverilog", "verilator"):
        path = logs / f"{simulator}.log"
        text = path.read_text(encoding="utf-8")
        passed = "PASS tb_h67_mssb5_score_pair vectors=20516 errors=0" in text
        if not passed:
            raise ValueError(f"{simulator} bit-exact PASS is absent")
        simulation[simulator] = {"vectors": 20_516, "passed": True, "sha256": sha256(path)}
    return {
        "schema": "h67_mssb5_score_pair_screen_v1",
        "evidence": ["[rtl]", "[开放逻辑映射代理]", "[开放网表STA代理]"],
        "scope": "32-lane dual-temporal H67 active-score combinational leaf",
        "simulation": simulation,
        "candidates": candidates,
        "comparison": decision,
        "claim_boundary": {
            "admitted": "MSSB5进入TTB8-ZKQI同row-top集成评估",
            "not_admitted": [
                "MSSB5已成为DATE独立贡献",
                "叶级面积等于attention子系统或芯片面积",
                "开放网表STA等于DC/PT/布局布线时序",
                "packed butterfly本身带来显著独立收益",
            ],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    labels = {
        "h67_direct_score_pair": "现有双Direct32",
        "h67_cse7_score_pair": "CSE7平衡强基线",
        "h67_ssr5_score_pair": "SSR5独立充分统计树",
        "h67_mssb5_score_pair": "MSSB5打包蝶形",
    }
    for top in TOPS:
        row = report["candidates"][top]
        rows.append(
            f"| {labels[top]} | {row['cells']} | {row['area']:.3f} | "
            f"{row['delay_ns']:.6f} |"
        )
    comparison = report["comparison"]
    return "\n".join(
        [
            "# Motion MSSB5 双时间充分统计蝶形叶级筛选",
            "",
            "## 结论",
            "",
            f"最终裁决：`{comparison['decision']}`。",
            "",
            "Icarus 与 Verilator 各完成 20516 组独立参考、强基线与候选逐位比较，",
            "overlap、same-zero、motion 和双 Q7 score 均为零失配。",
            "",
            "## 四方公平对照",
            "",
            "| 结构 | cells | Nangate45面积代理 | OpenSTA组合延迟(ns) |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            f"MSSB5 相对 CSE7 强基线面积下降 "
            f"`{comparison['area_reduction_vs_cse7']*100:.2f}%`，延迟比为 "
            f"`{comparison['delay_ratio_vs_cse7']:.4f}x`。",
            "",
            f"MSSB5 相对 SSR5 只再下降 "
            f"`{comparison['packed_area_reduction_vs_ssr5']*100:.2f}%` 面积，延迟比为 "
            f"`{comparison['packed_delay_ratio_vs_ssr5']:.4f}x`。因此可辩护价值来自 "
            "`{o0,z0,o1,z1,m}` 五充分统计量重编码与共享 motion；打包蝶形只能作为实现细节，不能单列贡献。",
            "",
            "## 证据边界",
            "",
            "- `[rtl]`：只覆盖组合叶模块，不代表 TTB8-ZKQI row-top 已集成；",
            "- `[开放逻辑映射代理]`：Yosys/ABC 与 Nangate45，不是 DC 面积；",
            "- `[开放网表STA代理]`：OpenSTA 只含映射单元延迟，无布局布线寄生；",
            "- 本轮没有 SAIF、功耗、真实 138-row 回放或 full-encoder 指标；",
            "- 只准入下一轮同 row-top A/B，不准进入 DATE 最终贡献列表。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    report = build_report(output)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
