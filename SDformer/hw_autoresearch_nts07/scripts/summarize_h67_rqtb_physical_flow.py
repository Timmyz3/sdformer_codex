#!/usr/bin/env python3
"""汇总H67 T450 Fixed-TTB32与RQTB16/32同约束RTL证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROW_RE = re.compile(r"^RQTB_ROW (?P<body>.+)$")
FINAL_RE = re.compile(r"^PASS H67 RQTB physical flow (?P<body>.+)$")
AREA_RE = re.compile(
    r"Chip area for module '\\h67_temporal_slot_shiftmax_sync_k_top':\s+([0-9.]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_values(body: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in body.split():
        key, value = field.split("=", 1)
        result[key] = int(value)
    return result


def parse_log(path: Path) -> tuple[list[dict[str, int]], dict[str, int], list[list[int]]]:
    rows: list[dict[str, int]] = []
    final: dict[str, int] | None = None
    occupancy: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if match:
            rows.append(key_values(match.group("body")))
            continue
        match = FINAL_RE.match(line)
        if match:
            final = key_values(match.group("body"))
            continue
        if line.startswith("RQTB_OCC "):
            values = line.split("=", 1)[1]
            occupancy.append([int(value) for value in values.split(",")])
    if final is None or len(occupancy) != 2:
        raise ValueError(f"RTL日志缺少final或occupancy: {path}")
    return rows, final, occupancy


def percentile(values: list[float | int], percent: float) -> float:
    if not values:
        raise ValueError("percentile输入为空")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float | int]) -> dict[str, float | int]:
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def speedup_distribution(
    baseline_cycles: list[int], candidate_cycles: list[int]
) -> dict[str, float | int]:
    if len(baseline_cycles) != len(candidate_cycles) or not baseline_cycles:
        raise ValueError("逐行周期列表必须非空且长度一致")
    if any(value <= 0 for value in baseline_cycles + candidate_cycles):
        raise ValueError("逐行周期必须为正数")
    return distribution(
        [baseline / candidate for baseline, candidate in zip(baseline_cycles, candidate_cycles)]
    )


def occupancy_distribution(histogram: list[int]) -> dict[str, float | int]:
    samples = sum(histogram)
    weighted = sum(index * count for index, count in enumerate(histogram))
    expanded: list[int] = []
    for index, count in enumerate(histogram):
        expanded.extend([index] * count)
    result = distribution(expanded)
    result["mean"] = weighted / samples
    result["samples"] = samples
    return result


def align_occupancy_histogram(
    histogram: list[int], expected_cycles: int, windows: int
) -> tuple[list[int], int]:
    aligned = list(histogram)
    adjustment = expected_cycles - sum(aligned)
    if adjustment < -1 or adjustment > windows:
        raise ValueError(
            f"occupancy采样数与周期不一致: samples={sum(aligned)}, cycles={expected_cycles}"
        )
    if adjustment < 0 and aligned[0] < -adjustment:
        raise ValueError("occupancy边界修正要求额外样本位于zero bin")
    aligned[0] += adjustment
    return aligned, adjustment


def parse_area(path: Path) -> float:
    matches = AREA_RE.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"Nangate45日志缺面积: {path}")
    return float(matches[-1])


def source_receipt(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in paths
    ]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows, final, occupancy = parse_log(args.verilator_log)
    icarus_rows, icarus_final, _ = parse_log(args.icarus_log)
    assert_rows, assert_final, _ = parse_log(args.assert_log)
    vector_manifest = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    activity = json.loads(args.activity.read_text(encoding="utf-8"))
    fixed_stat = json.loads(args.fixed_yosys.read_text(encoding="utf-8"))
    rqtb_stat = json.loads(args.rqtb_yosys.read_text(encoding="utf-8"))

    if len(rows) != 138 or final.get("rows") != 138:
        raise ValueError(f"必须覆盖138个真实head-row，实际{len(rows)}")
    if len(assert_rows) != 138 or assert_final != final:
        raise ValueError("SVA全量日志与主日志不一致")
    if len(icarus_rows) != 1 or icarus_final.get("rows") != 1:
        raise ValueError("Icarus交叉模拟必须覆盖一个真实T450 row")
    common_keys = [
        "row", "stage", "block", "head", "active", "equal",
        "fixed_cycles", "rqtb_cycles", "fixed_slots", "rqtb_slots",
        "fixed_desc", "rqtb_desc", "fixed_exp", "rqtb_exp",
    ]
    if any(icarus_rows[0][key] != rows[0][key] for key in common_keys):
        raise ValueError("Icarus与Verilator首行结果不一致")
    if vector_manifest.get("row_count") != 138 or vector_manifest.get("tokens_per_row") != 450:
        raise ValueError("vector manifest不是all12真实T450合同")
    if activity.get("status") != "PASS":
        raise ValueError("VCD活动统计未通过")

    for row in rows:
        if row["fixed_slots"] != 450:
            raise ValueError(f"Fixed-TTB slot不等于450: row={row['row']}")
        if row["rqtb_slots"] != 450 - row["equal"]:
            raise ValueError(f"RQTB slot与可逆商不一致: row={row['row']}")
        if row["fixed_desc"] != row["fixed_slots"] or row["rqtb_desc"] != row["rqtb_slots"]:
            raise ValueError(f"slot/descriptor边界不一致: row={row['row']}")
        if row["rqtb_exp"] > row["fixed_exp"]:
            raise ValueError(f"RQTB exp事务反增: row={row['row']}")

    fixed_cycles = [row["fixed_cycles"] for row in rows]
    rqtb_cycles = [row["rqtb_cycles"] for row in rows]
    fixed_total = sum(fixed_cycles)
    rqtb_total = sum(rqtb_cycles)
    fixed_slots = sum(row["fixed_slots"] for row in rows)
    rqtb_slots = sum(row["rqtb_slots"] for row in rows)
    fixed_exp = sum(row["fixed_exp"] for row in rows)
    rqtb_exp = sum(row["rqtb_exp"] for row in rows)
    fixed_histogram, fixed_boundary_adjustment = align_occupancy_histogram(
        occupancy[0], fixed_total, len(rows)
    )
    rqtb_histogram, rqtb_boundary_adjustment = align_occupancy_histogram(
        occupancy[1], rqtb_total, len(rows)
    )
    fixed_occupancy = occupancy_distribution(fixed_histogram)
    rqtb_occupancy = occupancy_distribution(rqtb_histogram)
    fixed_area = parse_area(args.fixed_mapping_log)
    rqtb_area = parse_area(args.rqtb_mapping_log)
    speedup = fixed_total / rqtb_total
    area_ratio = rqtb_area / fixed_area

    source_paths = [
        args.verilator_log,
        args.assert_log,
        args.icarus_log,
        args.vector_manifest,
        args.activity,
        args.fixed_yosys,
        args.rqtb_yosys,
        args.fixed_mapping_log,
        args.rqtb_mapping_log,
        Path(__file__),
    ] + args.rtl_source + args.verification_source

    result = {
        "schema": "h67_rqtb_physical_flow_t450_v1",
        "status": "PASS",
        "evidence_level": "[rtl]+[open-map代理]",
        "scope": (
            "H67 ep30 sample0/window0 all12的138个真实head-row；Fixed-TTB32与"
            "RQTB16/32共享同一16-bit FIFO容量、weighted-SCS/Shiftmax和同步双bank K存储"
        ),
        "identity": {
            "config_sha256": vector_manifest["run_context"]["artifact_identity"]["config_sha256"],
            "checkpoint_sha256": vector_manifest["run_context"]["artifact_identity"]["checkpoint_sha256"],
            "vector_manifest": str(args.vector_manifest.resolve()),
            "vector_manifest_sha256": sha256(args.vector_manifest),
            "vector_sha256": vector_manifest["vector_sha256"],
        },
        "coverage": {
            "rows": len(rows),
            "tokens": 138 * 450,
            "pairs": 138 * 225,
            "gated_k_outputs_checked": final["checked"],
            "synthetic_acc32_checksum_values_checked": 138 * 32,
            "synthetic_acc32_checksum_mismatch": final["acc32_mismatch"],
            "icarus_real_rows": 1,
            "verilator_real_rows": 138,
            "sva_real_rows": 138,
            "deterministic_periodic_descriptor_backpressure": True,
            "deterministic_periodic_output_backpressure": True,
        },
        "performance": {
            "fixed_cycles_total": fixed_total,
            "rqtb_cycles_total": rqtb_total,
            "speedup": speedup,
            "cycle_reduction_ratio": 1.0 - rqtb_total / fixed_total,
            "fixed_cycle_distribution": distribution(fixed_cycles),
            "rqtb_cycle_distribution": distribution(rqtb_cycles),
            "per_row_speedup_distribution": speedup_distribution(
                fixed_cycles, rqtb_cycles
            ),
            "rqtb_faster_rows": sum(r < f for f, r in zip(fixed_cycles, rqtb_cycles)),
        },
        "work_and_storage": {
            "fixed_slots": fixed_slots,
            "rqtb_slots": rqtb_slots,
            "slot_reduction_ratio": 1.0 - rqtb_slots / fixed_slots,
            "fixed_exp_transactions": fixed_exp,
            "rqtb_exp_transactions": rqtb_exp,
            "exp_reduction_ratio": 1.0 - rqtb_exp / fixed_exp,
            "k_read_transactions_both": final["checked"],
            "k_read_bits_both": final["checked"] * 32,
            "fixed_fifo_occupancy": fixed_occupancy,
            "rqtb_fifo_occupancy": rqtb_occupancy,
            "occupancy_boundary_zero_samples_adjusted": {
                "fixed": fixed_boundary_adjustment,
                "rqtb": rqtb_boundary_adjustment,
            },
        },
        "activity_proxy": activity,
        "open_mapping_proxy": {
            "library": "NangateOpenCellLibrary_typical.lib",
            "constraint": "无SDC、未计10个$mem_v2面积",
            "fixed_logic_area": fixed_area,
            "rqtb_logic_area": rqtb_area,
            "area_overhead_ratio": area_ratio - 1.0,
            "area_normalized_throughput": speedup / area_ratio,
            "fixed_generic_cells": fixed_stat["design"]["num_cells"],
            "rqtb_generic_cells": rqtb_stat["design"]["num_cells"],
            "unmapped_memories_each": fixed_stat["design"]["num_cells_by_type"]["$mem_v2"],
        },
        "negative_results": [
            "原始Icarus首跑因TB在valid=0时采样数据相关ready而跳过pair；修正为valid保持到正沿握手后关闭。",
            "RQTB不减少K读事务或K读bit；公平双bank active-mask gating下两者均为每个active token读取32 bit。",
            "当前活动量仅为单个真实row、排除跨设计共享alias后的VCD位翻转代理，不能称为SAIF功耗或能量。",
        ],
        "claim_boundary": [
            "可声称：在该真实all12样本的138个T450 head-row上，RQTB gated-K输出与synthetic Acc32 checksum一致，并获得同约束RTL周期和slot/exp事务收益。",
            "不可声称：多样本部署分布、目标工艺频率/面积/功耗、DC/PTPX签核或full encoder端到端加速。",
            "Nangate45数字只是不含memory面积的开放无约束逻辑映射代理。",
        ],
        "rows": rows,
        "source_receipts": source_receipt(source_paths),
    }
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    perf = result["performance"]
    work = result["work_and_storage"]
    mapping = result["open_mapping_proxy"]
    activity = result["activity_proxy"]
    lines = [
        "# Motion RQTB全分辨率T450物理流RTL报告",
        "",
        "## 结论",
        "",
        "- 状态：**PASS**；证据等级：**[rtl]+[open-map代理]**。",
        f"- 真实覆盖：138个head-row、62,100个token、{result['coverage']['gated_k_outputs_checked']:,}个gated-K输出、4,416个synthetic Acc32 checksum，零失配。",
        f"- Fixed-TTB32→RQTB16/32总周期：{perf['fixed_cycles_total']:,}→{perf['rqtb_cycles_total']:,}，加速{perf['speedup']:.3f}x，周期减少{perf['cycle_reduction_ratio']:.2%}。",
        f"- slot：{work['fixed_slots']:,}→{work['rqtb_slots']:,}，减少{work['slot_reduction_ratio']:.2%}；exp事务减少{work['exp_reduction_ratio']:.2%}。",
        f"- 单行VCD位翻转代理：{activity['bit_toggles']['fixed']:,}→{activity['bit_toggles']['rqtb']:,}，减少{activity['rqtb_reduction_ratio']:.2%}。这不是功耗。",
        f"- 开放无约束逻辑映射面积：{mapping['fixed_logic_area']:.3f}→{mapping['rqtb_logic_area']:.3f}（+{mapping['area_overhead_ratio']:.2%}）；面积归一吞吐代理{mapping['area_normalized_throughput']:.3f}x。",
        "",
        "## 公平边界",
        "",
        "| 项目 | Fixed-TTB32 | RQTB16/32 |",
        "|---|---|---|",
        "| score前端 | 同一Motion-XOR Q7 | 同一Motion-XOR Q7 |",
        "| slot FIFO | 16 bit × 32，pair内原子写入 | 16 bit × 32，pair内原子写入 |",
        "| K存储 | K0/K1同步双bank，一拍读 | 相同 |",
        "| 归一化 | weighted-SCS + Q1.7 Shiftmax | 相同 |",
        "| 唯一区别 | 每pair固定两条slot | score相等时一条slot，否则两条 |",
        "",
        "## 分布",
        "",
        "| 指标 | mean | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|",
        f"| Fixed周期/row | {perf['fixed_cycle_distribution']['mean']:.2f} | {perf['fixed_cycle_distribution']['p95']:.2f} | {perf['fixed_cycle_distribution']['p99']:.2f} | {perf['fixed_cycle_distribution']['max']} |",
        f"| RQTB周期/row | {perf['rqtb_cycle_distribution']['mean']:.2f} | {perf['rqtb_cycle_distribution']['p95']:.2f} | {perf['rqtb_cycle_distribution']['p99']:.2f} | {perf['rqtb_cycle_distribution']['max']} |",
        f"| Fixed FIFO占用 | {work['fixed_fifo_occupancy']['mean']:.2f} | {work['fixed_fifo_occupancy']['p95']:.2f} | {work['fixed_fifo_occupancy']['p99']:.2f} | {work['fixed_fifo_occupancy']['max']} |",
        f"| RQTB FIFO占用 | {work['rqtb_fifo_occupancy']['mean']:.2f} | {work['rqtb_fifo_occupancy']['p95']:.2f} | {work['rqtb_fifo_occupancy']['p99']:.2f} | {work['rqtb_fifo_occupancy']['max']} |",
        "",
        "## 负结果与边界",
        "",
    ]
    lines.extend(f"- {item}" for item in result["negative_results"])
    lines.extend(["", "## 论文允许表述", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verilator-log", type=Path, required=True)
    parser.add_argument("--assert-log", type=Path, required=True)
    parser.add_argument("--icarus-log", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--fixed-yosys", type=Path, required=True)
    parser.add_argument("--rqtb-yosys", type=Path, required=True)
    parser.add_argument("--fixed-mapping-log", type=Path, required=True)
    parser.add_argument("--rqtb-mapping-log", type=Path, required=True)
    parser.add_argument("--rtl-source", type=Path, action="append", default=[])
    parser.add_argument("--verification-source", type=Path, action="append", default=[])
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
