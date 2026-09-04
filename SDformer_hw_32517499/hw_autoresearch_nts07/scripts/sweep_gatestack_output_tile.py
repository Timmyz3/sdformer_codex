#!/usr/bin/env python3
"""联合 ordered-profile 周期模型与通用综合扫描 GateStack 输出 tile 宽度。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_gatestack_full_projection import DEFAULT_PROFILE, evaluate  # noqa: E402


RTL_FILES = [
    "rtl_hitflow/gatestack_resident_replay_joiner.sv",
    "rtl_hitflow/gatestack_ipd32w_replay_decoder.sv",
    "rtl_hitflow/gatestack_raw41_replay_decoder.sv",
    "rtl_hitflow/gatestack_raw_tail_retimer.sv",
    "rtl_hitflow/gatestack_raw_issue_adapter.sv",
    "rtl_hitflow/gatestack_replay_mux.sv",
    "rtl_hitflow/gatestack_term_fork.sv",
    "rtl_hitflow/gatestack_destination_bitmap_assembler.sv",
    "rtl_hitflow/gatestack_decoupled_product_engine.sv",
    "rtl_hitflow/gatestack_product_bitmap_join.sv",
    "rtl_hitflow/hitflow_segmented_multicast.sv",
    "rtl_hitflow/gatestack_tdr_multicast_backend.sv",
    "rtl_hitflow/hitflow_banked_accumulator.sv",
    "rtl_hitflow/gatestack_multihead_tile_projection_top.sv",
    "rtl_hitflow/gatestack_routed_multihead_tile_projection_top.sv",
    "rtl_hitflow/gatestack_multihead_decoder_projection_top.sv",
]
TOP = "gatestack_multihead_decoder_projection_top"


def synthesize(out_tile: int, workdir: Path) -> dict[str, int]:
    workdir.mkdir(parents=True, exist_ok=True)
    stat_path = workdir / f"out_tile_{out_tile}_stat.json"
    rtl = " ".join(RTL_FILES)
    command = (
        f"read_verilog -sv {rtl}; "
        f"chparam -set OUT_TILE {out_tile} {TOP}; "
        f"hierarchy -check -top {TOP}; proc; opt; check; "
        f"tee -o {stat_path} stat -json"
    )
    completed = subprocess.run(
        ["yosys", "-q", "-p", command],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    (workdir / f"out_tile_{out_tile}_yosys.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"OUT_TILE={out_tile} Yosys失败，见 {workdir}"
        )
    stat = json.loads(stat_path.read_text(encoding="utf-8"))
    design = stat["design"]
    cells = design.get("num_cells_by_type", {})
    return {
        "generic_cells": int(design["num_cells"]),
        "memory_bits": int(design["num_memory_bits"]),
        "multipliers": int(cells.get("$mul", 0)),
        "register_cells": sum(
            int(value)
            for name, value in cells.items()
            if name.startswith("$dff") or name.startswith("$sdff")
        ),
    }


def render_chinese(result: dict[str, Any]) -> str:
    rows = result["rows"]
    baseline = next(row for row in rows if row["out_tile"] == 32)
    lines = [
        "# GateStack 输出 Tile 宽度联合 DSE",
        "",
        "本报告把完整窗口 ordered-profile 周期模型与实际三 decoder 顶层的参数化 Yosys 通用综合放在同一口径下。周期属于 `[prof]+[模型]`，单元数属于 `[通用综合]`，均不是目标库 DC/PPA。",
        "",
        "## 结果",
        "",
        "| OUT_TILE | 相对32-lane完整窗口周期 | OBI双context周期 | p99执行周期 | 通用单元 | memory bits | `$mul` |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['out_tile']} | {row['cycle_ratio_vs_32']:.3f}x | "
            f"{row['obi_dual_cycles']:,} | {row['obi_p99']:.0f} | "
            f"{row['generic_cells']:,} | {row['memory_bits']:,} | "
            f"{row['multipliers']} |"
        )
    lines += [
        "",
        "## 决策",
        "",
        f"- `OUT_TILE=8` 的完整窗口周期是 32-lane 的 {rows[0]['cycle_ratio_vs_32']:.3f} 倍；它不是与模型等价的低面积实现，而是把每个逻辑输出 tile 再切成四个物理 tile；",
        f"- `OUT_TILE=16` 的完整窗口周期是 32-lane 的 {rows[1]['cycle_ratio_vs_32']:.3f} 倍；",
        f"- `OUT_TILE=32` 与 H67 `head_dim=32` 及原模型的 `output_lanes=32` 一致，通用综合为 {baseline['generic_cells']:,} cells、{baseline['memory_bits']:,} memory bits、{baseline['multipliers']} 个 `$mul`；",
        "- 后续功能 RTL 和周期模型默认冻结 `OUT_TILE=32`。若 DC 表明 32-lane 无法满足频率/面积，再把 8/16-lane 作为物理折叠实现，并显式重算 tile 数、bias 尾相、权重带宽和帧率；",
        "- 通用单元、memory bits 和 `$mul` 只能用于结构趋势。最终选择仍需同一目标库、同一时钟约束、SRAM 宏和 SAIF 下的面积/功耗/时序结果。",
        "",
        "## 接口影响",
        "",
        "1. `output_tile_id` 表示 32-channel 逻辑 tile；",
        "2. 若内部采用 8/16-lane 物理折叠，必须另设 `output_subtile_id`，不能复用逻辑 tile 标识；",
        "3. bias、weight response、final output 和 tile done 都必须携带逻辑 tile 标识；",
        "4. descriptor payload tag 跨 tile 保持不变，execution tag 对每次 `{tile, head}` replay 唯一。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "gatestack_output_tile_dse_20260716",
    )
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for out_tile in (8, 16, 32):
        model = evaluate(
            profile,
            output_lanes=out_tile,
            delivery_efficiency=0.85,
        )
        row = {
            "out_tile": out_tile,
            "obi_dual_cycles": model["totals"]["obi_dual"],
            "obi_p99": model["window_execute_percentiles"]["obi_p99"],
            **synthesize(out_tile, args.out_dir / "yosys"),
        }
        rows.append(row)
    baseline = next(row for row in rows if row["out_tile"] == 32)
    for row in rows:
        row["cycle_ratio_vs_32"] = (
            row["obi_dual_cycles"] / baseline["obi_dual_cycles"]
        )

    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "evidence": "[prof ordered trace]+[模型]+[Yosys通用综合]",
        "rows": rows,
    }
    (args.out_dir / "output_tile_dse.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "output_tile_dse.md").write_text(
        render_chinese(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
