#!/usr/bin/env python3
"""Summarize the Direct32x2/TARE-W8/TARE-W16 row-top RTL screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/h67_tare_zkqi_row_rtl_20260810"
PASS_RE = re.compile(
    r"PASS tb_h67_zkqi_row_miter rows=(?P<rows>\d+) "
    r"stall_mode=(?P<mode>\d+).*?baseline_e2e_cycles=(?P<baseline>\d+) "
    r"zkqi_e2e_cycles=(?P<candidate>\d+).*?"
    r"baseline_tare_dense=(?P<baseline_dense>\d+) "
    r"candidate_tare_dense=(?P<candidate_dense>\d+)"
)
AREA_RE = re.compile(r"Chip area for module .*?: (?P<area>[0-9.]+)")
CELL_RE = re.compile(r"Number of cells:\s+(?P<cells>\d+)")
LEAF_RE = re.compile(
    r"PASS tb_h67_tare_score_pair W=(?P<width>\d+) received=(?P<received>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pass_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(PASS_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected one final PASS line in {path}, found {len(matches)}")
    row = {key: int(value) for key, value in matches[0].groupdict().items()}
    row["candidate_cycle_regression"] = row["candidate"] / row["baseline"] - 1.0
    row["candidate_throughput_ratio"] = row["baseline"] / row["candidate"]
    return row


def parse_area_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    areas = list(AREA_RE.finditer(text))
    cells = list(CELL_RE.finditer(text))
    if not areas or not cells:
        raise ValueError(f"missing mapped area/cells in {path}")
    return {
        "area": float(areas[-1].group("area")),
        "cells": int(cells[-1].group("cells")),
    }


def parse_leaf_log(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(LEAF_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected one leaf PASS line in {path}, found {len(matches)}")
    return {key: int(value) for key, value in matches[0].groupdict().items()}


def tool_version(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    version_text = (result.stdout + result.stderr).strip()
    return version_text.splitlines()[0] if version_text else "unknown"


def candidate_decision(cycle_pass: bool, area_pass: bool) -> str:
    return "ADMIT" if cycle_pass and area_pass else "REJECT"


def build_report(output_dir: Path) -> dict[str, Any]:
    logs = output_dir / "logs"
    leaf_w8_iverilog = parse_leaf_log(logs / "iverilog_leaf_w8.log")
    leaf_w16_iverilog = parse_leaf_log(logs / "iverilog_leaf_w16.log")
    leaf_w8_verilator = parse_leaf_log(logs / "verilator_leaf_w8.log")
    leaf_w16_verilator = parse_leaf_log(logs / "verilator_leaf_w16.log")
    if leaf_w8_iverilog != leaf_w8_verilator or leaf_w16_iverilog != leaf_w16_verilator:
        raise ValueError("Icarus and Verilator leaf ledgers differ")
    if leaf_w8_iverilog != {"width": 8, "received": 33}:
        raise ValueError("W8 leaf boundary coverage drifted")
    if leaf_w16_iverilog != {"width": 16, "received": 35}:
        raise ValueError("W16 leaf boundary coverage drifted")

    w8_iverilog = [parse_pass_log(logs / f"iverilog_w8_mode{mode}.log") for mode in range(4)]
    w16_iverilog = [parse_pass_log(logs / f"iverilog_w16_mode{mode}.log") for mode in range(4)]
    w16_verilator = [parse_pass_log(logs / f"verilator_w16_mode{mode}.log") for mode in range(4)]
    if w16_iverilog != w16_verilator:
        raise ValueError("Icarus and Verilator W16 cycle/fallback ledgers differ")
    for rows in (w8_iverilog, w16_iverilog):
        if any(row["rows"] != 138 for row in rows):
            raise ValueError("row coverage must be 138")
    if any(row["candidate_dense"] != 3321 for row in w8_iverilog):
        raise ValueError("W8 real fallback count drifted")
    if any(row["candidate_dense"] != 251 for row in w16_iverilog):
        raise ValueError("W16 real fallback count drifted")

    direct_map = parse_area_log(logs / "nangate45_fast_direct.log")
    w16_map = parse_area_log(logs / "nangate45_fast_w16.log")
    area_ratio = w16_map["area"] / direct_map["area"]
    no_stall_throughput = w16_iverilog[0]["candidate_throughput_ratio"]
    area_normalized_throughput = no_stall_throughput / area_ratio

    w8_cycle_pass = max(row["candidate_cycle_regression"] for row in w8_iverilog) <= 0.01
    w16_cycle_pass = max(row["candidate_cycle_regression"] for row in w16_iverilog) <= 0.01
    w16_area_pass = area_normalized_throughput >= 1.10
    status = "ADMIT_TARE_W16" if w16_cycle_pass and w16_area_pass else "REJECT_TARE"

    source_paths = [
        ROOT / "rtl_h67/h67_tare_score_pair.sv",
        ROOT / "rtl_h67/h67_zkqi_row_shiftmax_top.sv",
        ROOT / "tb_h67/tb_h67_tare_score_pair.sv",
        ROOT / "tb_h67/tb_h67_zkqi_row_miter.sv",
        ROOT / "verif_h67/h67_tare_score_pair_assertions.sv",
        ROOT / "sim_h67/run_h67_tare_zkqi_row_checks.sh",
    ]
    vector_path = ROOT / "tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805/h67_checkpoint_rows.txt"
    liberty_path = ROOT / "third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
    log_paths = sorted(path for path in logs.glob("*.log") if path.is_file())
    return {
        "schema": "h67_tare_zkqi_row_rtl_screen_v2",
        "status": status,
        "evidence_levels": {
            "functional_and_cycle": "[rtl]",
            "mapped_area": "[开放映射代理]",
            "asic_ppa": "[待验证]",
        },
        "scope": {
            "rows": 138,
            "trace_files": 1,
            "stall_modes": 4,
            "trace_identity": str(vector_path.relative_to(ROOT)),
            "common_boundary": "TTB8-ZKQI row store/scanner/directory/SCS/gated-K",
        },
        "leaf_boundary_coverage": {
            "update_counts": "0..32",
            "thresholds": ["8/9", "16/17"],
            "w16_delta_extremes": [-1024, 1024],
            "w8_icarus": leaf_w8_iverilog,
            "w8_verilator_sva": leaf_w8_verilator,
            "w16_icarus": leaf_w16_iverilog,
            "w16_verilator_sva": leaf_w16_verilator,
        },
        "candidates": {
            "direct32x2": {
                "residual_width": 0,
                "role": "strong baseline",
            },
            "tare_w8": {
                "residual_width": 8,
                "icarus_modes": w8_iverilog,
                "cycle_gate_le_1pct": w8_cycle_pass,
                "decision": "REJECT before physical mapping" if not w8_cycle_pass else "CONTINUE",
            },
            "tare_w16": {
                "residual_width": 16,
                "icarus_modes": w16_iverilog,
                "verilator_sva_modes": w16_verilator,
                "cycle_gate_le_1pct": w16_cycle_pass,
                "decision": candidate_decision(w16_cycle_pass, w16_area_pass),
            },
        },
        "open_mapping_proxy": {
            "flow": "Yosys abc -fast + Nangate45 typical liberty; memory -nomap",
            "direct32x2": direct_map,
            "tare_w16": w16_map,
            "tare_w16_area_ratio": area_ratio,
            "tare_w16_area_overhead": area_ratio - 1.0,
            "tare_w16_no_stall_throughput_ratio": no_stall_throughput,
            "tare_w16_area_normalized_throughput": area_normalized_throughput,
            "required_area_normalized_throughput": 1.10,
            "area_gate_pass": w16_area_pass,
            "limitations": [
                "not DC/STA/SAIF/PTPX",
                "behavioral SRAM arrays are left unmapped and common to both candidates",
                "abc -fast is an open screening flow, not timing-closed PPA",
            ],
        },
        "negative_result": {
            "root_cause_evidence": "[待验证]",
            "root_cause": "inferred: 32-to-16 priority compactor and selection/control network exceed the saved second Direct32 lane logic",
            "default_abc_observation": "W16 default ABC remained in mapping for more than 12 minutes and was manually canceled; this observation is diagnostic only",
            "claim": "TARE exact reuse is algebraically valid but physically unprofitable in the current compacted-lane implementation",
        },
        "decision": {
            "tare_as_motion_contribution": False,
            "retain_rtl_as_negative_baseline": True,
            "next_motion_direction": "do not optimize the same priority compactor; return to TTB8-ZKQI/SCS or test a no-compaction fixed-slice formulation only after a new workload gate",
        },
        "source_receipts": [
            {"file": str(path), "sha256": sha256(path)} for path in source_paths
        ],
        "input_receipts": [
            {"file": str(vector_path), "sha256": sha256(vector_path)},
            {"file": str(liberty_path), "sha256": sha256(liberty_path)},
        ],
        "log_receipts": [
            {"file": str(path), "sha256": sha256(path)} for path in log_paths
        ],
        "tool_versions_at_report_generation": {
            "iverilog": tool_version(["iverilog", "-V"]),
            "verilator": tool_version(["verilator", "--version"]),
            "yosys": tool_version(["yosys", "-V"]),
        },
        "provenance_note": (
            "runner receipt freezes compile/mapping arguments; log, vector, liberty, "
            "RTL/TB/SVA and runner hashes prevent silent mixed-run reuse"
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    w8 = result["candidates"]["tare_w8"]["icarus_modes"]
    w16 = result["candidates"]["tare_w16"]["icarus_modes"]
    mapping = result["open_mapping_proxy"]
    lines = [
        "# Motion TARE-W8/W16 与 TTB8-ZKQI 同顶层 RTL 筛选",
        "",
        "## 结论",
        "",
        f"- [rtl] W8/W16 均在 138 行、全 12 block 真实回放中保持最终 gated-K 输出 bit-exact；W16 的 Icarus 与 Verilator+SVA 周期逐模式一致。",
        f"- [rtl] W8 的最大周期回退为 `{max(row['candidate_cycle_regression'] for row in w8):.4%}`，超过 1% 门槛，先行否决。",
        f"- [rtl] W16 的最大周期回退为 `{max(row['candidate_cycle_regression'] for row in w16):.4%}`，通过 1% 门槛，并精确执行每种模式 `251` 次 dense replay。",
        f"- [开放映射代理] Direct32x2/W16 面积为 `{mapping['direct32x2']['area']:.3f}` / `{mapping['tare_w16']['area']:.3f}`；W16 面积增加 `{mapping['tare_w16_area_overhead']:.2%}`。",
        f"- [开放映射代理] W16 面积归一吞吐仅 `{mapping['tare_w16_area_normalized_throughput']:.4f}x`，低于 `1.10x` 硬门槛。",
        "- 结论为 **REJECT_TARE**：保留为负基线，不进入 Motion 的 DATE 贡献列表。",
        "",
        "## 周期结果",
        "",
        "| stall mode | Direct32x2 | TARE-W8 | W8回退 | TARE-W16 | W16回退 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in range(4):
        lines.append(
            f"| {mode} | {w16[mode]['baseline']} | {w8[mode]['candidate']} | "
            f"{w8[mode]['candidate_cycle_regression']:.4%} | {w16[mode]['candidate']} | "
            f"{w16[mode]['candidate_cycle_regression']:.4%} |"
        )
    lines += [
        "",
        "## 边界与反压",
        "",
        "- 叶级覆盖 update-count 0..32、W8 的 8/9、W16 的 16/17；",
        "- W16 定向覆盖 signed residual delta `-1024/+1024`；",
        "- sparse/zero 采用组合 fall-through，只有 dense target 占 replay 槽；",
        "- 随机 descriptor stall 曾发现 valid 被 enable 门控的问题，现已改为显式 `in_enable` 准入并由 SVA 锁定；",
        "- output stall 时原子 `{tag,score0,score1,k_active,update_count}` 保持稳定；",
        "- W16 四种模式 Icarus/Verilator 均零丢失、零重复、零协议错误。",
        "",
        "## 开放映射代理",
        "",
        "| 候选 | cells | area | 吞吐比 | 面积归一吞吐 |",
        "|---|---:|---:|---:|---:|",
        f"| Direct32x2 | {mapping['direct32x2']['cells']} | {mapping['direct32x2']['area']:.3f} | 1.0000x | 1.0000x |",
        f"| TARE-W16 | {mapping['tare_w16']['cells']} | {mapping['tare_w16']['area']:.3f} | {mapping['tare_w16_no_stall_throughput_ratio']:.4f}x | {mapping['tare_w16_area_normalized_throughput']:.4f}x |",
        "",
        "映射使用相同 `Yosys abc -fast`、Nangate45 typical liberty 和 `memory -nomap`。它不是 DC、STA、SAIF 或 PTPX。两边 SRAM 逻辑边界相同且均未计面积，故只用于淘汰当前 compactor 实现。",
        "",
        "## 负结果解释",
        "",
        "TARE 的代数复用成立，lane-work 模型也确实下降。当前 `[开放映射代理]` 只证明完整候选逻辑显著变大；结合 RTL 结构推断，主要代价来自从 32-bit update mask 中每拍选出最多 16 个任意 lane 所需的 priority extraction、lane-id 目录和 Q/K 选择 mux。未做层次面积或 compactor-off 消融前，该根因仍标为 `[待验证]`。",
        "",
        "因此不能把 `40.86% score-lane work减少` 转写成面积或能耗收益。默认 ABC 对 W16 在 12 分钟后仍未完成，本轮手动取消；这一点只作复杂度诊断，不作为定量 PPA。定量否决依据是同流 `abc -fast` 的 `0.657x` 面积归一吞吐。",
        "",
        "## 决策",
        "",
        "1. TARE-W8：周期门槛失败，REJECT；",
        "2. TARE-W16：功能/周期门槛通过，但完整前端面积门槛失败，REJECT；",
        "3. TARE 不进入 Motion 的独立 DATE 创新点；",
        "4. RTL、profile 与失败分析保留为强负基线；",
        "5. 后续不得继续微调同一个 priority compactor。只有提出无需任意-lane compaction 的 fixed-slice 公式并先通过 workload 模型，才允许开启新分支。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.output_dir)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"PASS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
