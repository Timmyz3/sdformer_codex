#!/usr/bin/env python3
"""Seal the Local5 topology scheduler open-library comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local5_relation_scheduler_openproxy_20260814"
PLATFORM_LIB = ROOT / "third_party/OpenROAD-flow-scripts/flow/platforms/nangate45/lib"


def parse_yosys(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    cells = re.findall(r"Number of cells:\s+(\d+)", text)
    areas = re.findall(r"Chip area for module.*?:\s+([0-9.]+)", text)
    if not cells or not areas or "Found and reported 0 problems" not in text:
        raise ValueError(f"incomplete Yosys log: {path}")
    return {"cells": int(cells[-1]), "area_proxy": float(areas[-1])}


def parse_sta(path: Path) -> dict[str, float | str]:
    text = path.read_text()
    arrivals = re.findall(r"^\s*([0-9.]+)\s+data arrival time$", text, re.M)
    slacks = re.findall(r"^\s*(-?[0-9.]+)\s+slack \((MET|VIOLATED)\)$", text, re.M)
    if not arrivals or not slacks or "Error:" in text:
        raise ValueError(f"incomplete STA log: {path}")
    worst_slack = min(slacks, key=lambda item: float(item[0]))
    return {
        "arrival_ns": max(float(value) for value in arrivals),
        "slack_ns": float(worst_slack[0]),
        "timing": worst_slack[1],
    }


def parse_macro_area(path: Path) -> float:
    text = path.read_text()
    match = re.search(r"\barea\s*:\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"macro area missing: {path}")
    return float(match.group(1))


def main() -> None:
    schedulers = {}
    for short, top in (
        ("fcsr", "qfit_fcsr_scheduler_openproxy"),
        ("dynamic", "qfit_dynamic_scheduler_openproxy"),
    ):
        schedulers[short] = {
            **parse_yosys(OUT / f"{top}_yosys.log"),
            **parse_sta(OUT / f"{top}_sta.log"),
        }
    schedulers["banked_dynamic"] = {
        **parse_yosys(
            OUT / "qfit_banked_dynamic_flop_scheduler_openproxy_yosys.log"
        ),
        **parse_sta(
            OUT / "qfit_banked_dynamic_flop_scheduler_openproxy_sta.log"
        ),
    }

    fcsr = schedulers["fcsr"]
    dynamic = schedulers["dynamic"]
    banked = schedulers["banked_dynamic"]
    report = {
        "schema": "local5_relation_scheduler_openproxy_v1",
        "status": "ADMIT_AS_TOPOLOGY_SPECIALIZATION_PROXY",
        "evidence": ["[开放逻辑映射代理]", "[开放网表STA代理]"],
        "scope": (
            "scheduler-control-only, same 15x15x2 ports, Nangate45, 3ns SDC; "
            "excludes relation/K memories, score, TCFM5, routing, SAIF, and SRAM macros"
        ),
        "schedulers": schedulers,
        "fcsr_vs_dynamic": {
            "area_ratio": dynamic["area_proxy"] / fcsr["area_proxy"],
            "delay_ratio": dynamic["arrival_ns"] / fcsr["arrival_ns"],
            "fcsr_area_reduction": 1.0 - fcsr["area_proxy"] / dynamic["area_proxy"],
            "fcsr_delay_reduction": 1.0 - fcsr["arrival_ns"] / dynamic["arrival_ns"],
        },
        "fcsr_vs_banked_dynamic": {
            "area_ratio": banked["area_proxy"] / fcsr["area_proxy"],
            "delay_ratio": banked["arrival_ns"] / fcsr["arrival_ns"],
            "fcsr_area_reduction": 1.0 - fcsr["area_proxy"] / banked["area_proxy"],
            "fcsr_delay_reduction": 1.0 - fcsr["arrival_ns"] / banked["arrival_ns"],
        },
        "banked_dynamic_contract": {
            "bank_map": "bank=(x+2*y) mod 5",
            "counter_storage": "5x9x(3-bit count + 3-bit entry generation) + 3x3-bit row generation = 279 bit",
            "functional_evidence": "8 Icarus seeds + 3 Verilator --assert seeds + sparse two-generation gap + 100-group production tile; cycle-exact against MODE_DYNAMIC/FCSR",
            "note": "strong topology-aware dynamic baseline; not a proposed contribution",
        },
        "shared_payload_macro_proxy": {
            "binding": "1xfakeram45_64x32 + 5xfakeram45_64x15",
            "logical_payload_bits": 3735,
            "physical_macro_bits": 64 * (32 + 5 * 15),
            "area_proxy": parse_macro_area(PLATFORM_LIB / "fakeram45_64x32.lib")
            + 5.0 * parse_macro_area(PLATFORM_LIB / "fakeram45_64x15.lib"),
            "note": "identical for FCSR and Dynamic; open FakeRAM45, not target SRAM",
        },
        "cycle_context_rtl": {
            "qsilent_t450": 183379,
            "qsilent_dynamic": 155791,
            "qsilent_banked_dynamic": 155791,
            "qsilent_fcsr": 155791,
            "dynamic_to_fcsr_speedup": 1.0,
        },
        "claim_boundary": [
            "FCSR wins because fixed five-neighbor raster topology gives closed-form retirement; Dynamic tracks per-source consumers",
            "the primary structural comparison is against the topology-aware five-bank Dynamic baseline; the original flat Dynamic is sensitivity only",
            "the comparison is not DC, place-and-route, SAIF, PTPX, SRAM PPA, or full encoder",
            "the 3ns Dynamic violation is an open-library structural warning, not signoff timing",
            "does not modify docs/359 frozen main-table columns",
        ],
    }
    if fcsr["timing"] != "MET" or dynamic["timing"] != "VIOLATED":
        raise ValueError("unexpected 3ns timing classification")
    if banked["timing"] != "VIOLATED":
        raise ValueError("unexpected banked-Dynamic 3ns timing classification")
    if report["fcsr_vs_banked_dynamic"]["area_ratio"] < 2.0:
        raise ValueError("topology specialization no longer has an area-proxy margin")

    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    delta = report["fcsr_vs_dynamic"]
    banked_delta = report["fcsr_vs_banked_dynamic"]
    cycles = report["cycle_context_rtl"]
    macro = report["shared_payload_macro_proxy"]
    fcsr_total = fcsr["area_proxy"] + macro["area_proxy"]
    dynamic_total = dynamic["area_proxy"] + macro["area_proxy"]
    banked_total = banked["area_proxy"] + macro["area_proxy"]
    (OUT / "report.md").write_text(f"""# Local5 FCSR 与 Dynamic 调度器物理代理对照

- 裁决：`{report['status']}`。
- 边界：{report['scope']}。
- FCSR：`{fcsr['cells']}` cells、面积代理 `{fcsr['area_proxy']:.3f}`、路径 `{fcsr['arrival_ns']:.6f} ns`、3 ns `{fcsr['timing']}`。
- 强 Banked-Dynamic：`{banked['cells']}` cells、面积代理 `{banked['area_proxy']:.3f}`、路径 `{banked['arrival_ns']:.6f} ns`、3 ns `{banked['timing']}`。
- Banked-Dynamic/FCSR：面积代理 `{banked_delta['area_ratio']:.2f}x`、路径 `{banked_delta['delay_ratio']:.2f}x`；FCSR 分别降低 `{banked_delta['fcsr_area_reduction']:.2%}` 和 `{banked_delta['fcsr_delay_reduction']:.2%}`。
- 原 flat Dynamic 敏感性：`{dynamic['cells']}` cells、面积代理 `{dynamic['area_proxy']:.3f}`、路径 `{dynamic['arrival_ns']:.6f} ns`；其 `{delta['area_ratio']:.2f}x` 面积倍率不再作为主要对照。
- 共享 payload 宏规则：`{macro['binding']}`，物理 `{macro['physical_macro_bits']}` bit，开放宏面积 `{macro['area_proxy']:.3f}`；计入相同宏后 Banked-Dynamic/FCSR 总面积代理为 `{banked_total / fcsr_total:.2f}x`，flat Dynamic/FCSR 为 `{dynamic_total / fcsr_total:.2f}x`。
- 同一 100-group Query-Silent RTL：T450 `{cycles['qsilent_t450']}`、active-filtered flat Dynamic `{cycles['qsilent_dynamic']}`、Banked-Dynamic `{cycles['qsilent_banked_dynamic']}`、FCSR `{cycles['qsilent_fcsr']}`；两种 Dynamic 与 FCSR 逐组周期完全相同。

## 架构解释

强 Dynamic 已使用 `bank=(x+2y) mod 5` 将同一 destination 的五个候选映射到五个无冲突 counter bank，并用 epoch 消除循环行清零；它在 8 个 Icarus seed 与 3 个 Verilator `--assert` seed 下和原 Dynamic 逐周期一致。即使如此，它仍需运行时维护 consumer count、比较最后消费者并排序退休事件。FCSR 则把固定五邻域和 raster 输入序编译成 north、末行 west 和最终 self 三类闭式边界。两者吞吐相同，因此该结果只支持 Local5 统一数据流中的 closed-form retirement，不产生新的调度贡献名。

## 证据边界

本表只比较调度控制，不含双方相同的 3735-bit relation/K payload memory、score、TCFM5、布线寄生和动态功耗。Banked-Dynamic 的 279-bit counter/generation 状态已映射为寄存器；双方共同 payload 仅作相同 FakeRAM45 宏敏感性相加。它不是 DC/STA 签核或 ASIC PPA；3 ns 违例只表明开放映射中的运行时 completion 路径仍深。`docs/359` 不更新。
""")


if __name__ == "__main__":
    main()
