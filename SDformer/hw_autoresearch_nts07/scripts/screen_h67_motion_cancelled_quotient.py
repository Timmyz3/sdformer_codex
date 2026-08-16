#!/usr/bin/env python3
"""Prove and screen motion-cancelled equality at the RQTB boundary."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/h67_motion_cancelled_quotient_screen_20260814"
RTL = ROOT / "tb_h67/h67_motion_cancelled_quotient_openproxy.sv"
MSSB5 = ROOT / "rtl_h67/h67_mssb5_score_pair.sv"
LIB = ROOT / "third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"


def finalize(overlap: int, same_zero: int, motion: int) -> int:
    integer = 4 * overlap + motion + same_zero // 16
    remainder = same_zero % 16
    increment = remainder > 8 or (remainder == 8 and integer % 2 == 1)
    return integer + int(increment)


def reduced(overlap: int, same_zero: int, motion_parity: int) -> int:
    base = 4 * overlap + same_zero // 16
    remainder = same_zero % 16
    increment = remainder > 8 or (
        remainder == 8 and (base + motion_parity) % 2 == 1
    )
    return base + int(increment)


def map_top(top: str) -> dict[str, float | int]:
    log = OUT / f"{top}.log"
    command = (
        f"read_liberty -lib {LIB}; read_verilog -sv {MSSB5} {RTL}; "
        f"hierarchy -check -top {top}; proc; flatten; opt; techmap; opt; "
        f"dfflibmap -liberty {LIB}; abc -D 3000 -liberty {LIB}; "
        "clean; check -assert; stat -liberty " + str(LIB)
    )
    subprocess.run(
        ["yosys", "-l", str(log), "-p", command],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    text = log.read_text()
    cells = re.findall(r"Number of cells:\s+(\d+)", text)
    areas = re.findall(r"Chip area for module.*?:\s+([0-9.]+)", text)
    if not cells or not areas or "Found and reported 0 problems" not in text:
        raise ValueError(f"incomplete mapping log: {log}")
    return {"cells": int(cells[-1]), "area_proxy": float(areas[-1])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    legal_stats = [
        (overlap, silence)
        for overlap in range(33)
        for silence in range(33 - overlap)
    ]
    cases = 0
    mismatches = 0
    for overlap0, silence0 in legal_stats:
        for overlap1, silence1 in legal_stats:
            for motion in range(33):
                score0 = finalize(overlap0, silence0, motion)
                score1 = finalize(overlap1, silence1, motion)
                exact_equal = score0 == score1
                candidate_equal = (
                    reduced(overlap0, silence0, motion & 1)
                    == reduced(overlap1, silence1, motion & 1)
                )
                mismatches += exact_equal != candidate_equal
                cases += 1
    if mismatches:
        raise ValueError(f"motion-cancelled equality mismatch: {mismatches}")

    mapping = {
        "baseline": map_top("h67_score_finalize_pair_baseline"),
        "candidate": map_top("h67_motion_cancelled_quotient_candidate"),
        "mssb5_baseline": map_top("h67_mssb5_quotient_baseline_openproxy"),
        "mssb5_candidate": map_top("h67_mssb5_motion_cancelled_openproxy"),
    }
    area_ratio = (
        mapping["candidate"]["area_proxy"]
        / mapping["baseline"]["area_proxy"]
    )
    full_area_ratio = (
        mapping["mssb5_candidate"]["area_proxy"]
        / mapping["mssb5_baseline"]["area_proxy"]
    )
    status = (
        "ADMIT_AS_RQTB_ALGEBRAIC_EXPLANATION_ONLY"
        if full_area_ratio >= 0.95
        else "CONDITIONAL_MICROARCH_SUPPORT"
    )
    report = {
        "schema": "h67_motion_cancelled_quotient_screen_v1",
        "status": status,
        "theorem": (
            "score equality cancels the shared full motion term; only motion parity "
            "is needed for round-to-even half ties"
        ),
        "exhaustive_count_domain": {
            "legal_stat_pairs": len(legal_stats),
            "motions": 33,
            "cases": cases,
            "mismatches": mismatches,
        },
        "open_mapping_proxy": mapping,
        "candidate_over_baseline_area": area_ratio,
        "mssb5_candidate_over_baseline_area": full_area_ratio,
        "claim_boundary": [
            "count-domain proof covers overlap+same_zero<=32 and motion 0..32",
            "mapping covers only score finalization/equality, not five popcount trees or row backend",
            "this is not TARE lane-delta execution, but its implementation gain may still be ordinary CSE",
            "does not modify frozen Motion cycles or docs/359",
        ],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    baseline = mapping["baseline"]
    candidate = mapping["candidate"]
    mssb5_baseline = mapping["mssb5_baseline"]
    mssb5_candidate = mapping["mssb5_candidate"]
    verdict = (
        "等价性成立，但强综合基线已吸收该代数 CSE，不晋级新机制。"
        if status == "ADMIT_AS_RQTB_ALGEBRAIC_EXPLANATION_ONLY"
        else "等价性成立且存在条件微结构余量，仍需行顶层证明。"
    )
    (OUT / "report.md").write_text(
        f"""# Motion 共享运动项抵消的 Quotient 判定筛选

- 裁决：`{status}`。
- 穷举：`{cases}` 个合法计数组合，mismatch=`{mismatches}`。
- 定理：两个时间 score 的完整 motion 项相同，等值判定中整体抵消；RNE 半数 tie 只需 `motion[0]`。
- baseline finalizer：`{baseline['cells']}` cells、面积代理 `{baseline['area_proxy']:.3f}`。
- candidate：`{candidate['cells']}` cells、面积代理 `{candidate['area_proxy']:.3f}`，相对 baseline `{area_ratio:.4f}x`。
- 完整 MSSB5 score-pair baseline：`{mssb5_baseline['cells']}` cells、面积代理 `{mssb5_baseline['area_proxy']:.3f}`。
- 完整 MSSB5 + motion-cancelled finalizer：`{mssb5_candidate['cells']}` cells、面积代理 `{mssb5_candidate['area_proxy']:.3f}`，相对 `{full_area_ratio:.4f}x`。

## 结论

{verdict}

该开放映射只覆盖计数后的 finalizer/equality，不含五棵统计树、slot FIFO、SCS、Shiftmax、gated-K 或 full row。不得改写 `112589/94891/34099/28001`，不得称 DC/PPA。
"""
    )


if __name__ == "__main__":
    main()
