#!/usr/bin/env python3
"""汇总跨100 sample canonical Q/K的Motion ZKQI双仿真器差分回放。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.profile_h67_zkqi_multisample_ordered import receipt
    from scripts.report_h67_zkqi_row_miter import distribution, parse_log
except ModuleNotFoundError:
    from profile_h67_zkqi_multisample_ordered import receipt
    from report_h67_zkqi_row_miter import distribution, parse_log


def verify_mode(
    manifest_rows: list[dict[str, Any]],
    iverilog_log: Path,
    verilator_log: Path,
    mode: int,
) -> tuple[list[dict[str, int]], dict[str, int]]:
    expected_rows = len(manifest_rows)
    iverilog = parse_log(iverilog_log, expected_rows=expected_rows)
    verilator = parse_log(verilator_log, expected_rows=expected_rows)
    if iverilog != verilator:
        raise ValueError(f"mode={mode}: Icarus/Verilator逐行或final不一致")
    rows, final = iverilog
    if (
        final.get("rows") != expected_rows
        or final.get("stall_mode") != mode
        or final.get("bundle_skip") != 1
    ):
        raise ValueError(f"mode={mode}: final receipt漂移")
    invariant_keys = (
        "active_pairs", "outputs", "baseline_preload", "zkqi_preload",
        "baseline_slots", "zkqi_slots", "seeded", "baseline_read_bits",
        "zkqi_read_bits",
    )
    for expected, observed in zip(manifest_rows, rows):
        if (
            observed["row"] != expected["row"]
            or observed["stage"] != expected["stage"]
            or observed["block"] != expected["block"]
            or observed["head"] != expected["head"]
        ):
            raise ValueError(f"mode={mode}, row={observed['row']}: 身份漂移")
        model_fields = {
            "active_pairs": expected["active_pairs"],
            "outputs": expected["expected_outputs"],
            "baseline_preload": 225,
            "zkqi_preload": 225,
            "baseline_read_bits": expected["baseline_read_bits"],
            "zkqi_read_bits": expected["candidate_read_bits"],
            "seeded": 2 * (225 - expected["active_pairs"]),
        }
        for key, value in model_fields.items():
            if observed[key] != value:
                raise ValueError(
                    f"mode={mode}, row={observed['row']}: {key}="
                    f"{observed[key]} != {value}"
                )
        if mode == 0:
            if (
                observed["baseline_cycles"] != expected["baseline_cycles"]
                or observed["zkqi_cycles"] != expected["ttb_cycles"]
            ):
                raise ValueError(
                    f"row={observed['row']}: 无反压周期模型/RTL不一致"
                )
        if any(observed[key] < 0 for key in invariant_keys):
            raise ValueError(f"mode={mode}, row={observed['row']}: 负工作计数")
    return rows, final


def stage_summary(rows: list[dict[str, int]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        grouped[row["stage"]].append(row)
    result = {}
    for stage, values in sorted(grouped.items()):
        baseline = sum(row["baseline_e2e_cycles"] for row in values)
        candidate = sum(row["zkqi_e2e_cycles"] for row in values)
        result[str(stage)] = {
            "rows": len(values),
            "active_pair_ratio": sum(row["active_pairs"] for row in values)
            / (225 * len(values)),
            "baseline_e2e_cycles": baseline,
            "ttb8_e2e_cycles": candidate,
            "speedup": baseline / candidate,
        }
    return result


def render_md(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    mode0 = report["modes"]["0"]
    mode3 = report["modes"]["3"]
    lines = [
        "# Motion跨100 Sample Canonical Q/K RTL差分回放",
        "",
        "## 结论",
        "",
        "- 状态：**PASS**；证据等级：`[rtl-canonical]`。",
        f"- 覆盖`{coverage['rows']}`条row、`{coverage['tokens']}`个token、"
        f"`{coverage['samples']}`个sample、`{coverage['blocks']}`个block及每个block全部head。",
        f"- active pair覆盖`{coverage['active_pair_min']}..{coverage['active_pair_max']}`，"
        f"并包含`{coverage['ttb_slow_rows']}`条全活动模型慢行。",
        "- Icarus与Verilator+SVA在无反压和固定重反压两种模式下逐行、逐输出和最终账本完全一致。",
        f"- 无反压`{mode0['model_cycle_mismatches']}`个周期失配；RQTB2S/TTB8含preload总周期="
        f"`{mode0['baseline_e2e_cycles']}/{mode0['ttb8_e2e_cycles']}`，加速`{mode0['speedup']:.4f}x`。",
        "",
        "Canonical向量保持count/overlap/motion/score和zero-K控制语义，但不恢复原始lane身份；因此它扩大RTL控制状态覆盖，不是原始数据集逐bit replay，也不能用于真实SAIF。",
        "",
        "## 1. 构造与覆盖合同",
        "",
        "每个sample/block选择固定hash行、active-pair最小/中位/最大行及首个TTB慢行，并额外保证每个block全部head至少被覆盖。",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| row | {coverage['rows']} |",
        f"| token | {coverage['tokens']} |",
        f"| active output | {coverage['active_outputs']} |",
        f"| sample/block | {coverage['samples']}/{coverage['blocks']} |",
        f"| active pair范围 | {coverage['active_pair_min']}..{coverage['active_pair_max']} |",
        f"| TTB模型慢行 | {coverage['ttb_slow_rows']} |",
        "",
        "## 2. 双仿真器与反压",
        "",
        "| 模式 | RQTB2S含preload周期 | TTB8含preload周期 | 加速 | 输出数 | 双仿真器 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| 0 无反压 | {mode0['baseline_e2e_cycles']} | {mode0['ttb8_e2e_cycles']} | {mode0['speedup']:.4f}x | {mode0['outputs']} | PASS |",
        f"| 3 固定重反压 | {mode3['baseline_e2e_cycles']} | {mode3['ttb8_e2e_cycles']} | {mode3['speedup']:.4f}x | {mode3['outputs']} | PASS |",
        "",
        "反压模式只用于协议、稳定性和工作量不变性验证；不能把该模式的周期当作部署吞吐。",
        "",
        "## 3. 无反压分stage",
        "",
        "| Stage | rows | active pair | RQTB2S周期 | TTB8周期 | 加速 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, row in report["stage_mode0"].items():
        lines.append(
            f"| {stage} | {row['rows']} | {row['active_pair_ratio']:.2%} | "
            f"{row['baseline_e2e_cycles']} | {row['ttb8_e2e_cycles']} | {row['speedup']:.4f}x |"
        )
    lines += [
        "",
        "## 4. 证据提升与边界",
        "",
        "本轮将逐bit差分回放从sample0/window0的138条真实row扩展为跨100 sample状态分布的5570条canonical row。它证明count模型覆盖到的稀疏、稠密、全活动和反压状态可被真实RTL执行，并且无反压周期公式继续零残差。",
        "",
        "仍然不能宣称：",
        "",
        "- 5570条是原始bit trace；",
        "- canonical lane模式代表真实toggle、SAIF或功耗；",
        "- row周期等于encoder级FPS；",
        "- count profile的全部672000行已经逐bit RTL replay。",
        "",
        "## 5. 下一步",
        "",
        "模型跨100 sample控制状态的可信度已补强。下一轮可以进入B4/B8/B16/B32粗粒度TTB DSE，但只有在同一canonical replay、固定5 ns开放映射和mask选择器代价下仍有收益，才允许扩RTL。目标工艺SAIF仍等待预提交raw Q/K抓取。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--iverilog-mode0", type=Path, required=True)
    parser.add_argument("--verilator-mode0", type=Path, required=True)
    parser.add_argument("--iverilog-mode3", type=Path, required=True)
    parser.add_argument("--verilator-mode3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = manifest["rows"]
    mode0_rows, mode0_final = verify_mode(
        manifest_rows, args.iverilog_mode0, args.verilator_mode0, 0
    )
    mode3_rows, mode3_final = verify_mode(
        manifest_rows, args.iverilog_mode3, args.verilator_mode3, 3
    )
    invariant = (
        "row", "stage", "block", "head", "active_pairs", "outputs",
        "baseline_preload", "zkqi_preload", "baseline_slots", "zkqi_slots",
        "seeded", "baseline_read_bits", "zkqi_read_bits",
    )
    for primary, stressed in zip(mode0_rows, mode3_rows):
        if any(primary[key] != stressed[key] for key in invariant):
            raise ValueError(f"row={primary['row']}: 反压改变工作量合同")

    modes = {}
    for mode, rows, final in (
        (0, mode0_rows, mode0_final), (3, mode3_rows, mode3_final)
    ):
        baseline = final["baseline_e2e_cycles"]
        candidate = final["zkqi_e2e_cycles"]
        modes[str(mode)] = {
            "baseline_cycles": final["baseline_cycles"],
            "ttb8_cycles": final["zkqi_cycles"],
            "baseline_e2e_cycles": baseline,
            "ttb8_e2e_cycles": candidate,
            "speedup": baseline / candidate,
            "outputs": final["outputs"],
            "model_cycle_mismatches": 0 if mode == 0 else None,
            "baseline_distribution": distribution(
                [row["baseline_e2e_cycles"] for row in rows]
            ),
            "ttb8_distribution": distribution(
                [row["zkqi_e2e_cycles"] for row in rows]
            ),
        }
    report = {
        "schema": "h67_zkqi_canonical_multisample_rtl_replay_v1",
        "status": "PASS",
        "evidence_level": "[rtl-canonical]",
        "scope": "canonical control-state differential replay；非原始bit trace/SAIF",
        "coverage": manifest["coverage"],
        "canonical_invariants": manifest["canonical_invariants"],
        "non_invariants": manifest["non_invariants"],
        "selection_policy": manifest["selection_policy"],
        "modes": modes,
        "stage_mode0": stage_summary(mode0_rows),
        "verification": {
            "simulators": ["Icarus", "Verilator+SVA"],
            "modes": [0, 3],
            "逐行双仿真器一致": True,
            "反压工作量不变": True,
            "无反压模型周期零残差": True,
        },
        "source_receipts": {
            "manifest": receipt(args.manifest),
            "iverilog_mode0": receipt(args.iverilog_mode0),
            "verilator_mode0": receipt(args.verilator_mode0),
            "iverilog_mode3": receipt(args.iverilog_mode3),
            "verilator_mode3": receipt(args.verilator_mode3),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_md(report), encoding="utf-8")
    print(
        f"PASS canonical RTL rows={report['coverage']['rows']} "
        f"mode0={modes['0']['speedup']:.6f} mode3={modes['3']['speedup']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
