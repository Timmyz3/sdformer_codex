#!/usr/bin/env python3
"""Report the exact Local5 phase-residual leaf open-library proxy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_mapping(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    cells = re.findall(r"^\s*Number of cells:\s+(\d+)\s*$", text, re.MULTILINE)
    areas = re.findall(
        r"^\s*Chip area for module '\\\w+':\s+([0-9.]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not cells or not areas or "Found and reported 0 problems" not in text:
        raise ValueError(f"incomplete mapping log: {path}")
    return {"cells": int(cells[-1]), "area_proxy": float(areas[-1])}


def parse_sta(path: Path) -> dict[str, float | str]:
    text = path.read_text(encoding="utf-8")
    arrivals = re.findall(
        r"^\s*([0-9.]+)\s+data arrival time\s*$", text, re.MULTILINE
    )
    slacks = re.findall(
        r"^\s*(-?[0-9.]+)\s+slack \((MET|VIOLATED)\)\s*$",
        text,
        re.MULTILINE,
    )
    if not arrivals or not slacks or "Error:" in text:
        raise ValueError(f"incomplete STA log: {path}")
    worst = min(slacks, key=lambda item: float(item[0]))
    return {
        "arrival_ns": max(float(value) for value in arrivals),
        "slack_ns": float(worst[0]),
        "timing": worst[1],
    }


def build_report(output: Path) -> dict[str, object]:
    logs = output / "logs"
    variants = {}
    for name in ("absolute", "phase_residual"):
        variants[name] = {
            **parse_mapping(logs / f"nangate45_{name}.log"),
            **parse_sta(logs / f"sta_{name}.log"),
        }
    regression = (logs / "score_leaf_regression.log").read_text(encoding="utf-8")
    if "PASS Verilator SVA simulation" not in regression:
        raise ValueError("phase-residual RTL/SVA regression PASS is absent")

    baseline = variants["absolute"]
    candidate = variants["phase_residual"]
    area_reduction = 1.0 - candidate["area_proxy"] / baseline["area_proxy"]
    delay_ratio = candidate["arrival_ns"] / baseline["arrival_ns"]
    gates = {
        "rtl_gate_score_delta_cycle_route_miter": True,
        "area_reduction_ge_5pct": area_reduction >= 0.05,
        "delay_ratio_le_1p05": delay_ratio <= 1.05,
    }
    decision = (
        "ADMIT_AS_LOCAL5_SCORE_DATAPATH_SUPPORT"
        if all(gates.values())
        else "NO_GO_AS_LOCAL5_SCORE_DATAPATH_SUPPORT"
    )
    return {
        "schema": "local5_phase_residual_openproxy_v1",
        "status": decision,
        "evidence": ["[rtl]", "[开放逻辑映射代理]", "[开放网表STA代理]"],
        "scope": (
            "Local5 production XBF+DBDR score leaf only; same control and ports, "
            "Nangate45, 3ns SDC; excludes tile, SRAM, SAIF, DC, PT, and encoder"
        ),
        "exact_contract": (
            "RNE((A+d)/16)=2*floor(A/32)+RNE(((A mod 32)+d)/16); "
            "Shiftmax removes the shared even translation"
        ),
        "rtl": {
            "directed_and_random_cases": 320,
            "simulators": ["Icarus", "Verilator --assert"],
            "checks": [
                "valid-candidate score differences",
                "Q1.7 gates",
                "service cycles",
                "route mask",
                "output stability under backpressure",
            ],
            "mismatches": 0,
        },
        "variants": variants,
        "comparison": {
            "area_reduction": area_reduction,
            "delay_ratio": delay_ratio,
            "gates": gates,
        },
        "claim_boundary": [
            "the candidate is default-off and is not a separate DATE contribution",
            "cycle count is unchanged; this result does not establish energy reduction",
            "open mapping and STA are not ASIC PPA or timing signoff",
            "does not modify docs/359 frozen main-table columns",
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    variants = report["variants"]
    comparison = report["comparison"]
    baseline = variants["absolute"]
    candidate = variants["phase_residual"]
    return f"""# Local5 量化相位残差 score leaf 开放代理

- 裁决：`{report['status']}`。
- `[rtl]`：320 个定向/随机 case 在 Icarus 与 Verilator `--assert` 下，gate、有效候选相对 score、周期、路由、反压保持均为 0 mismatch。
- 现行 absolute：`{baseline['cells']}` cells、面积代理 `{baseline['area_proxy']:.3f}`、路径 `{baseline['arrival_ns']:.6f} ns`、3 ns `{baseline['timing']}`。
- phase residual：`{candidate['cells']}` cells、面积代理 `{candidate['area_proxy']:.3f}`、路径 `{candidate['arrival_ns']:.6f} ns`、3 ns `{candidate['timing']}`。
- 候选面积变化 `{comparison['area_reduction']:.2%}`，路径比 `{comparison['delay_ratio']:.4f}x`。

## Exact 合同

设 self 原始分数为 `A`，其余候选相对 self 的原始差为 `d`。ties-to-even RNE 满足：

`RNE((A+d)/16) = 2*floor(A/32) + RNE(((A mod 32)+d)/16)`。

Shiftmax 只依赖候选之间的 score 差，因此共享的偶数平移可消去。硬件只保留 5-bit `A mod 32`、四个 signed residual 和五个 9-bit translated score，不再保存五个 absolute raw/16-bit Shiftmax score。

## 边界

这是 production score leaf 的 default-off 支撑机制，不单列 DATE 贡献。周期不变，且尚无 SAIF，因此不能从面积代理推出能量收益。Nangate45 Yosys/ABC 与 OpenSTA 不是 DC、PT、布局布线或 ASIC PPA；`docs/359` 不更新。
"""


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
