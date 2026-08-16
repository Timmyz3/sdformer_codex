#!/usr/bin/env python3
"""汇总RQTB2S、pair-bitmap ZK bypass与TTB8-ZKQI三方RTL对照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_provenance import sha256_file
    from scripts.report_h67_zkqi_row_miter import (
        EXPECTED_OUTPUTS,
        EXPECTED_ROWS,
        distribution,
        parse_area,
        parse_generic_cells,
        parse_log,
    )
except ModuleNotFoundError:
    from evidence_provenance import sha256_file
    from report_h67_zkqi_row_miter import (
        EXPECTED_OUTPUTS,
        EXPECTED_ROWS,
        distribution,
        parse_area,
        parse_generic_cells,
        parse_log,
    )


def receipts(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def load_family(
    iverilog_logs: list[Path], verilator_logs: list[Path], bundle_skip: int
) -> dict[int, tuple[list[dict[str, int]], dict[str, int]]]:
    result: dict[int, tuple[list[dict[str, int]], dict[str, int]]] = {}
    for mode in range(4):
        iv = parse_log(iverilog_logs[mode])
        vl = parse_log(verilator_logs[mode])
        if iv != vl:
            raise ValueError(f"双仿真器逐行不一致: bundle_skip={bundle_skip}, mode={mode}")
        rows, final = iv
        if final.get("bundle_skip") != bundle_skip or final.get("stall_mode") != mode:
            raise ValueError(f"bundle/mode receipt错误: bundle_skip={bundle_skip}, mode={mode}")
        if final.get("outputs") != EXPECTED_OUTPUTS:
            raise ValueError("gated-K输出覆盖数不是20841")
        if any(row.get("bundle_skip") != bundle_skip for row in rows):
            raise ValueError("逐行bundle_skip receipt漂移")
        result[mode] = iv
    return result


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    pair = load_family(args.iverilog_pair, args.verilator_pair, 0)
    ttb8 = load_family(args.iverilog_ttb8, args.verilator_ttb8, 1)

    baseline_keys = {
        "row", "stage", "block", "head", "active_pairs", "outputs",
        "baseline_preload", "baseline_cycles", "baseline_e2e_cycles",
        "baseline_slots", "baseline_read_bits",
    }
    candidate_work_keys = {
        "row", "stage", "block", "head", "active_pairs", "outputs",
        "zkqi_preload", "zkqi_slots", "seeded",
        "zkqi_read_bits",
    }
    for mode in range(4):
        pair_rows, pair_final = pair[mode]
        ttb_rows, ttb_final = ttb8[mode]
        for pair_row, ttb_row in zip(pair_rows, ttb_rows):
            if any(pair_row[key] != ttb_row[key] for key in baseline_keys):
                raise ValueError(f"三方baseline受候选污染: mode={mode}, row={pair_row['row']}")
            if any(pair_row[key] != ttb_row[key] for key in candidate_work_keys):
                raise ValueError(f"pair/TTB8工作量合同不一致: mode={mode}, row={pair_row['row']}")
        for key in ("baseline_cycles", "baseline_read_bits", "outputs"):
            if pair_final[key] != ttb_final[key]:
                raise ValueError(f"三方final baseline不一致: mode={mode}, key={key}")

    modes: dict[str, Any] = {}
    for mode in range(4):
        pair_rows, pair_final = pair[mode]
        ttb_rows, ttb_final = ttb8[mode]
        baseline_cycles = pair_final["baseline_cycles"]
        pair_cycles = pair_final["zkqi_cycles"]
        ttb_cycles = ttb_final["zkqi_cycles"]
        baseline_e2e = pair_final["baseline_e2e_cycles"]
        pair_e2e = pair_final["zkqi_e2e_cycles"]
        ttb_e2e = ttb_final["zkqi_e2e_cycles"]
        modes[str(mode)] = {
            "baseline_cycles": baseline_cycles,
            "pair_bitmap_cycles": pair_cycles,
            "ttb8_zkqi_cycles": ttb_cycles,
            "pair_vs_baseline_speedup": baseline_cycles / pair_cycles,
            "ttb8_vs_pair_speedup": pair_cycles / ttb_cycles,
            "ttb8_vs_baseline_speedup": baseline_cycles / ttb_cycles,
            "baseline_e2e_cycles": baseline_e2e,
            "pair_bitmap_e2e_cycles": pair_e2e,
            "ttb8_zkqi_e2e_cycles": ttb_e2e,
            "pair_vs_baseline_e2e_speedup": baseline_e2e / pair_e2e,
            "ttb8_vs_pair_e2e_speedup": pair_e2e / ttb_e2e,
            "ttb8_vs_baseline_e2e_speedup": baseline_e2e / ttb_e2e,
            "baseline_distribution": distribution([row["baseline_cycles"] for row in pair_rows]),
            "pair_bitmap_distribution": distribution([row["zkqi_cycles"] for row in pair_rows]),
            "ttb8_zkqi_distribution": distribution([row["zkqi_cycles"] for row in ttb_rows]),
            "pair_faster_equal_slower_rows": [
                sum(row["zkqi_cycles"] < row["baseline_cycles"] for row in pair_rows),
                sum(row["zkqi_cycles"] == row["baseline_cycles"] for row in pair_rows),
                sum(row["zkqi_cycles"] > row["baseline_cycles"] for row in pair_rows),
            ],
            "ttb8_faster_equal_slower_rows": [
                sum(t["zkqi_cycles"] < p["baseline_cycles"] for p, t in zip(pair_rows, ttb_rows)),
                sum(t["zkqi_cycles"] == p["baseline_cycles"] for p, t in zip(pair_rows, ttb_rows)),
                sum(t["zkqi_cycles"] > p["baseline_cycles"] for p, t in zip(pair_rows, ttb_rows)),
            ],
        }

    primary_pair_rows = pair[0][0]
    primary_ttb_rows = ttb8[0][0]
    stage_summary: dict[str, Any] = {}
    for stage in sorted({row["stage"] for row in primary_pair_rows}):
        p_rows = [row for row in primary_pair_rows if row["stage"] == stage]
        t_rows = [row for row in primary_ttb_rows if row["stage"] == stage]
        base_cycles = sum(row["baseline_cycles"] for row in p_rows)
        pair_cycles = sum(row["zkqi_cycles"] for row in p_rows)
        ttb_cycles = sum(row["zkqi_cycles"] for row in t_rows)
        base_e2e = sum(row["baseline_e2e_cycles"] for row in p_rows)
        pair_e2e = sum(row["zkqi_e2e_cycles"] for row in p_rows)
        ttb_e2e = sum(row["zkqi_e2e_cycles"] for row in t_rows)
        stage_summary[str(stage)] = {
            "rows": len(p_rows),
            "active_pair_ratio": sum(row["active_pairs"] for row in p_rows) / (225 * len(p_rows)),
            "baseline_cycles": base_cycles,
            "pair_bitmap_cycles": pair_cycles,
            "ttb8_zkqi_cycles": ttb_cycles,
            "pair_vs_baseline_speedup": base_cycles / pair_cycles,
            "ttb8_vs_pair_speedup": pair_cycles / ttb_cycles,
            "ttb8_vs_baseline_speedup": base_cycles / ttb_cycles,
            "baseline_e2e_cycles": base_e2e,
            "pair_bitmap_e2e_cycles": pair_e2e,
            "ttb8_zkqi_e2e_cycles": ttb_e2e,
            "ttb8_vs_baseline_e2e_speedup": base_e2e / ttb_e2e,
        }

    area = {
        "baseline": parse_area(args.map_baseline),
        "pair_bitmap": parse_area(args.map_pair),
        "ttb8_zkqi": parse_area(args.map_ttb8),
    }
    generic = {
        "baseline": parse_generic_cells(args.yosys_baseline),
        "pair_bitmap": parse_generic_cells(args.yosys_pair),
        "ttb8_zkqi": parse_generic_cells(args.yosys_ttb8),
    }
    primary = modes["0"]
    pair_area_ratio = area["pair_bitmap"] / area["baseline"]
    ttb_area_ratio = area["ttb8_zkqi"] / area["baseline"]
    read_base = pair[0][1]["baseline_read_bits"]
    read_sparse = pair[0][1]["zkqi_read_bits"]

    all_paths = [
        *args.iverilog_pair, *args.verilator_pair,
        *args.iverilog_ttb8, *args.verilator_ttb8,
        args.map_baseline, args.map_pair, args.map_ttb8,
        args.yosys_baseline, args.yosys_pair, args.yosys_ttb8,
        args.vector, *args.sources,
    ]
    return {
        "schema": "h67_zkqi_threeway_strong_baseline_v1",
        "status": "PASS",
        "evidence_level": "[rtl]+[open-map代理]",
        "scope": (
            "H67 ep30 sample0/window0、全12 attention block、138条真实fullres T450 head-row；"
            "三方共享row-SRAM、双score lane、weighted-SCS与gated-K backend"
        ),
        "candidates": {
            "baseline": "RQTB2S，225 pair全部Q/K读取与score",
            "pair_bitmap": "225-bit active bitmap逐pair扫描 + three-class exact zero-K seed",
            "ttb8_zkqi": "29x8 active mask分层跳扫 + 单槽skid + 同一three-class exact seed",
        },
        "coverage": {
            "rows": EXPECTED_ROWS,
            "tokens": EXPECTED_ROWS * 450,
            "outputs_checked_per_family_simulator_mode": EXPECTED_OUTPUTS,
            "simulators": ["Icarus Verilog", "Verilator+SVA"],
            "stall_modes": 4,
            "all_gate_k_token_exact": True,
            "baseline_independent_of_candidate": True,
        },
        "decomposition": {
            "row_read_bit_reduction_shared_by_pair_and_ttb8": 1.0 - read_sparse / read_base,
            "zero_k_direct_injection_no_stall_cycle_speedup": primary["pair_vs_baseline_speedup"],
            "ttb8_hierarchical_scan_incremental_speedup": primary["ttb8_vs_pair_speedup"],
            "combined_ttb8_zkqi_speedup": primary["ttb8_vs_baseline_speedup"],
            "combined_ttb8_zkqi_preload_inclusive_speedup":
                primary["ttb8_vs_baseline_e2e_speedup"],
            "interpretation": (
                "zero-K直注入在无反压下主要减少Q/K读与score工作；"
                "可见周期收益来自TTB8跨空pair的分层跳扫"
            ),
        },
        "cycles": {"modes": modes, "stage_primary_no_stall": stage_summary},
        "storage_model": {
            "pair_bitmap_extra_bits": 225 + 3 * 9,
            "ttb8_zkqi_extra_bits": 29 * 8 + (5 + 8) + 3 * 9,
            "common_memory_excluded_from_these_extra_bits": True,
        },
        "open_logic_mapping_proxy": {
            "library": "NangateOpenCellLibrary_typical.lib",
            "constraint": "无SDC，所有$mem_v2未计面积，只比较逻辑趋势",
            "logic_area": area,
            "generic_yosys": generic,
            "pair_area_overhead_ratio": pair_area_ratio - 1.0,
            "ttb8_area_overhead_ratio": ttb_area_ratio - 1.0,
            "pair_area_normalized_throughput": primary["pair_vs_baseline_speedup"] / pair_area_ratio,
            "ttb8_area_normalized_throughput": primary["ttb8_vs_baseline_speedup"] / ttb_area_ratio,
        },
        "negative_results": [
            "普通pair-bitmap zero-K bypass在无反压下不减少周期，证明只做zero-K检测/跳读不足以形成吞吐贡献。",
            "pair-bitmap与TTB8都把行读bit减少约45%，该数字不能单独归因于TTB8。",
            "当前仍只有sample0/window0；preload已纳入周期，但本表仍没有SRAM宏、SDC、功耗或EDP。",
            "TTB8的增量价值是分层跳扫，不是bitmap、skid或三计数器本身的新颖性。",
        ],
        "claim_boundary": [
            "可声称：三方强基线隔离了exact zero-K work gating与TTB8 hierarchical skipping的独立作用。",
            "可声称：三方在当前138条真实行和四种反压下均bit-exact，baseline计数不受候选完成时刻污染。",
            "不可声称：该子机制单独构成DATE完整架构、已获得目标ASIC PPA或full-encoder加速。",
        ],
        "source_receipts": receipts(all_paths),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    primary = result["cycles"]["modes"]["0"]
    mapping = result["open_logic_mapping_proxy"]
    dec = result["decomposition"]
    lines = [
        "# Motion ZKQI三方强基线RTL与机制分解",
        "",
        "## 结论",
        "",
        "- 状态：**PASS，但仍是attention行级子机制，不是DATE完整架构或ASIC PPA**。",
        f"- [rtl] RQTB2S / pair-bitmap / TTB8-ZKQI无反压周期：`{primary['baseline_cycles']}` / `{primary['pair_bitmap_cycles']}` / `{primary['ttb8_zkqi_cycles']}`。",
        f"- [rtl] 纳入每行225拍预加载后，三方周期为 `{primary['baseline_e2e_cycles']}` / `{primary['pair_bitmap_e2e_cycles']}` / `{primary['ttb8_zkqi_e2e_cycles']}`，TTB8端到端边界加速 `{primary['ttb8_vs_baseline_e2e_speedup']:.3f}x`。",
        f"- [rtl] 普通pair-bitmap相对基线为 `{primary['pair_vs_baseline_speedup']:.3f}x`；TTB8相对pair-bitmap增量为 `{primary['ttb8_vs_pair_speedup']:.3f}x`。",
        f"- [rtl] pair-bitmap与TTB8都减少 `{dec['row_read_bit_reduction_shared_by_pair_and_ttb8']:.2%}` 行读bit；因此该流量收益属于exact zero-K gating，不属于TTB8独占。",
        f"- [open-map代理] pair-bitmap/TTB8面积归一吞吐为 `{mapping['pair_area_normalized_throughput']:.3f}x` / `{mapping['ttb8_area_normalized_throughput']:.3f}x`。",
        "",
        "## 三方定义",
        "",
    ]
    lines.extend(f"- {key}：{value}。" for key, value in result["candidates"].items())
    lines += [
        "",
        "三方共享相同row-SRAM、两个H67 score lane、weighted-SCS和gated-K发射端；pair-bitmap与TTB8使用完全相同的three-class zero-K seed语义。",
        "",
        "## 四种反压结果",
        "",
        "| 模式 | RQTB2S执行 | pair执行 | TTB8执行 | TTB8执行加速 | RQTB2S含预载 | pair含预载 | TTB8含预载 | TTB8端到端加速 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in result["cycles"]["modes"].items():
        lines.append(
            f"| {mode} | {row['baseline_cycles']} | {row['pair_bitmap_cycles']} | "
            f"{row['ttb8_zkqi_cycles']} | {row['ttb8_vs_baseline_speedup']:.3f}x | "
            f"{row['baseline_e2e_cycles']} | {row['pair_bitmap_e2e_cycles']} | "
            f"{row['ttb8_zkqi_e2e_cycles']} | {row['ttb8_vs_baseline_e2e_speedup']:.3f}x |"
        )
    lines += [
        "",
        "## 开放逻辑映射",
        "",
        "| 候选 | 逻辑面积 | generic cell | 相对基线面积 | 面积归一吞吐 |",
        "|---|---:|---:|---:|---:|",
        f"| RQTB2S | {mapping['logic_area']['baseline']:.3f} | {mapping['generic_yosys']['baseline']['cells']} | 0 | 1.000x |",
        f"| pair-bitmap | {mapping['logic_area']['pair_bitmap']:.3f} | {mapping['generic_yosys']['pair_bitmap']['cells']} | {mapping['pair_area_overhead_ratio']:.2%} | {mapping['pair_area_normalized_throughput']:.3f}x |",
        f"| TTB8-ZKQI | {mapping['logic_area']['ttb8_zkqi']:.3f} | {mapping['generic_yosys']['ttb8_zkqi']['cells']} | {mapping['ttb8_area_overhead_ratio']:.2%} | {mapping['ttb8_area_normalized_throughput']:.3f}x |",
        "",
        "memory保留为`$mem_v2`且未计面积，该表不能当作ASIC PPA。",
        "",
        "## 负结果与论文边界",
        "",
    ]
    lines.extend(f"- {item}" for item in result["negative_results"])
    lines += ["", "## 可用表述", ""]
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iverilog-pair", type=Path, nargs=4, required=True)
    parser.add_argument("--verilator-pair", type=Path, nargs=4, required=True)
    parser.add_argument("--iverilog-ttb8", type=Path, nargs=4, required=True)
    parser.add_argument("--verilator-ttb8", type=Path, nargs=4, required=True)
    parser.add_argument("--map-baseline", type=Path, required=True)
    parser.add_argument("--map-pair", type=Path, required=True)
    parser.add_argument("--map-ttb8", type=Path, required=True)
    parser.add_argument("--yosys-baseline", type=Path, required=True)
    parser.add_argument("--yosys-pair", type=Path, required=True)
    parser.add_argument("--yosys-ttb8", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", result)
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
