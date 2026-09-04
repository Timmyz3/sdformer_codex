#!/usr/bin/env python3
"""汇总Local5 GASR-2C单bank真实RTL回放、分布与公平边界。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


GROUP_RE = re.compile(
    r"GROUP group=(?P<group>\d+) direct_cycles=(?P<direct_cycles>\d+) "
    r"gasr_cycles=(?P<gasr_cycles>\d+) updates=(?P<updates>\d+) "
    r"direct_reads=(?P<direct_reads>\d+) direct_writes=(?P<direct_writes>\d+) "
    r"gasr_reads=(?P<gasr_reads>\d+) gasr_writes=(?P<gasr_writes>\d+) "
    r"gasr_hits=(?P<gasr_hits>\d+) gasr_misses=(?P<gasr_misses>\d+)"
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def parse_log(path: Path) -> list[dict[str, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = GROUP_RE.search(line)
        if match:
            rows.append({key: int(value) for key, value in match.groupdict().items()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tb_qfit/vectors/local5_gasr_singlebank_postg0_100/manifest.json"),
    )
    parser.add_argument(
        "--deterministic-log",
        type=Path,
        default=Path(
            "results/local5_gasr2c_singlebank_postg0_rtl_20260803/deterministic.log"
        ),
    )
    parser.add_argument(
        "--random-log",
        type=Path,
        default=Path(
            "results/local5_gasr2c_singlebank_postg0_rtl_20260803/random_gaps.log"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/local5_gasr2c_singlebank_postg0_rtl_20260803"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = parse_log(args.deterministic_log)
    random_rows = parse_log(args.random_log)
    if len(rows) != manifest["groups"] or len(random_rows) != manifest["groups"]:
        raise ValueError("deterministic/random日志未覆盖全部组")
    if "PASS GASR2C singlebank" not in args.deterministic_log.read_text():
        raise ValueError("deterministic日志无PASS")
    if "PASS GASR2C singlebank" not in args.random_log.read_text():
        raise ValueError("random日志无PASS")

    for row, meta in zip(rows, manifest["rows"], strict=True):
        if row["group"] != meta["vector_group"] or row["updates"] != meta["updates"]:
            raise AssertionError("RTL日志与向量manifest错位")
        row.update({"sources": meta["sources"], "stage": meta["stage"]})

    direct_cycles = sum(row["direct_cycles"] for row in rows)
    gasr_cycles = sum(row["gasr_cycles"] for row in rows)
    direct_transactions = sum(
        row["direct_reads"] + row["direct_writes"] for row in rows
    )
    gasr_transactions = sum(
        row["gasr_reads"] + row["gasr_writes"] for row in rows
    )
    nonempty = [row for row in rows if row["updates"]]
    speedups = [row["direct_cycles"] / row["gasr_cycles"] for row in nonempty]

    # Threshold 2 is the analytical reuse break-even: at least two updates per
    # prepared source. It is reported as a prospective mode selector, not RTL.
    threshold = 2.0
    hybrid_cycles = 0
    selected = 0
    selected_regressions = 0
    for row in rows:
        reuse = row["updates"] / row["sources"] if row["sources"] else 0.0
        use_gasr = reuse >= threshold
        hybrid_cycles += row["gasr_cycles"] if use_gasr else row["direct_cycles"]
        selected += int(use_gasr)
        selected_regressions += int(use_gasr and row["gasr_cycles"] > row["direct_cycles"])
    oracle_cycles = sum(min(row["direct_cycles"], row["gasr_cycles"]) for row in rows)

    stage_rows = []
    for stage in sorted({row["stage"] for row in rows}):
        subset = [row for row in rows if row["stage"] == stage]
        dcy = sum(row["direct_cycles"] for row in subset)
        gcy = sum(row["gasr_cycles"] for row in subset)
        hcy = sum(
            row["gasr_cycles"]
            if row["sources"] and row["updates"] / row["sources"] >= threshold
            else row["direct_cycles"]
            for row in subset
        )
        stage_rows.append(
            {
                "stage": stage,
                "groups": len(subset),
                "nonempty_groups": sum(bool(row["updates"]) for row in subset),
                "direct_cycles": dcy,
                "gasr_cycles": gcy,
                "gasr_speedup": dcy / gcy,
                "prospective_hybrid_cycles": hcy,
                "prospective_hybrid_speedup": dcy / hcy,
            }
        )

    summary = {
        "schema": "local5_gasr2c_singlebank_rtl_summary_v1",
        "evidence": "本机RTL，qualified post-G0 profile100，单颜色bank",
        "groups": len(rows),
        "nonempty_groups": len(nonempty),
        "updates": sum(row["updates"] for row in rows),
        "acc32_crosscheck": manifest["full_acc32_crosscheck"],
        "deterministic": {
            "direct_cycles": direct_cycles,
            "gasr_cycles": gasr_cycles,
            "aggregate_speedup": direct_cycles / gasr_cycles,
            "cycle_reduction": 1 - gasr_cycles / direct_cycles,
            "direct_sram_transactions": direct_transactions,
            "gasr_sram_transactions": gasr_transactions,
            "transaction_reduction": 1 - gasr_transactions / direct_transactions,
            "nonempty_win_equal_loss": {
                "win": sum(row["direct_cycles"] > row["gasr_cycles"] for row in nonempty),
                "equal": sum(row["direct_cycles"] == row["gasr_cycles"] for row in nonempty),
                "loss": sum(row["direct_cycles"] < row["gasr_cycles"] for row in nonempty),
            },
            "per_nonempty_group_speedup": {
                "mean": float(np.mean(speedups)),
                "p0": percentile(speedups, 0),
                "p25": percentile(speedups, 25),
                "p50": percentile(speedups, 50),
                "p75": percentile(speedups, 75),
                "p95": percentile(speedups, 95),
                "max": percentile(speedups, 100),
            },
        },
        "random_gap_correctness": {
            "groups": len(random_rows),
            "acc32": "PASS",
            "note": "随机输入空泡仅用于握手压力，不用于性能比较",
        },
        "prospective_reuse_stratifier": {
            "evidence": "同一100组RTL日志的post-hoc模型，尚无选择器RTL",
            "rule": "updates / active_target_sources >= 2选择GASR，否则direct-1RW",
            "selected_gasr_groups": selected,
            "selected_regressions": selected_regressions,
            "cycles": hybrid_cycles,
            "speedup_vs_direct": direct_cycles / hybrid_cycles,
            "oracle_cycles": oracle_cycles,
            "oracle_speedup_vs_direct": direct_cycles / oracle_cycles,
        },
        "per_stage": stage_rows,
        "fairness": {
            "memory": "两者使用同一1RW同步存储合同、同DEPTH/OUT_DIM/ACC_W",
            "baseline": "direct首触写入，复访同端口read-modify-write，lazy-zero",
            "candidate": "GASR双槽寄存器驻留，单端口后台回写/预取，lazy-zero",
            "cycle_boundary": "run_start释放至flush_done，不含结果readback",
            "transaction_boundary": "执行期SRAM read+write，不含结果readback",
        },
        "not_proven": [
            "五bank并行顶层周期",
            "SRAM macro绑定后的OpenROAD PPA",
            "真实外部SRAM可变延迟/反压",
            "复用强度选择器RTL与独立留出集泛化",
            "Motion主线收益",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    det = summary["deterministic"]
    dist = det["per_nonempty_group_speedup"]
    wins = det["nonempty_win_equal_loss"]
    prospective = summary["prospective_reuse_stratifier"]
    lines = [
        "# Local5 GASR-2C 单颜色 bank 真实 RTL 评估",
        "",
        "## 结论",
        "",
        f"在本机 qualified post-G0 profile100 的 bank0 回放中，direct-1RW 为 {direct_cycles:,} 周期，GASR-2C 为 {gasr_cycles:,} 周期，聚合加速 {direct_cycles / gasr_cycles:.3f}x；执行期单端口 SRAM 事务由 {direct_transactions:,} 降到 {gasr_transactions:,}，下降 {1 - gasr_transactions / direct_transactions:.2%}。两条路径均与原全投影 Acc32 金向量逐项一致（{manifest['full_acc32_crosscheck']}）。",
        "",
        "该结果允许把单 bank 机制从 `[模型]` 升为 `[rtl]`，但不等于五 bank 顶层或物理 PPA 已成立。",
        "",
        "## 公平口径",
        "",
        "- 两者使用同一个同步单端口 1RW 存储合同、相同深度90、OUT_DIM=2、Acc32。",
        "- direct 基线也使用 lazy-zero：首触直接写，复访执行同端口精确 RMW。",
        "- 周期从 `run_start` 释放计到 `flush_done`，不含结果读回；事务仅统计执行期 backing SRAM 读写。",
        "- deterministic 流用于性能；随机输入空泡流只验证握手稳定和 bit-exact，不拿来计算加速。",
        "",
        "## 分布而非只报总数",
        "",
        f"100组中55组非空；逐组 win/equal/loss={wins['win']}/{wins['equal']}/{wins['loss']}。非空组加速 p50={dist['p50']:.3f}x、p95={dist['p95']:.3f}x、最差={dist['p0']:.3f}x。固定 prepare/activate 开销会使极低复用组退化，因此不能宣称GASR对每个窗口都更快。",
        "",
        "| Stage | 非空/总组 | direct周期 | GASR周期 | GASR加速 | 前瞻混合加速 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row['stage']} | {row['nonempty_groups']}/{row['groups']} | {row['direct_cycles']:,} | {row['gasr_cycles']:,} | {row['gasr_speedup']:.3f}x | {row['prospective_hybrid_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## 下一项本土化机制：复用强度分层",
            "",
            f"同一组日志的前瞻模型使用 `updates/active_target_sources >= 2` 选择GASR，否则走direct-1RW：选择GASR {prospective['selected_gasr_groups']}组，选中组退化 {prospective['selected_regressions']}组，估算总周期 {prospective['cycles']:,}，相对direct为 {prospective['speedup_vs_direct']:.3f}x，接近逐组oracle的 {prospective['oracle_speedup_vs_direct']:.3f}x。",
            "",
            "该阈值来自每个已准备source至少两次更新的复用盈亏点，且计数可由K-popcount和有效目标数在projection前确定；它类似Bishop的密度分层思想，但不复制dense/sparse双核，而是在同一个颜色bank和同一个SRAM上切换direct-RMW与source-resident模式。当前仍是同cohort post-hoc模型，必须实现选择器RTL并用留出trace验证后才可列为贡献。",
            "",
            "## 尚未完成",
            "",
            "- 五bank并行集成与真实100组顶层回放；",
            "- Acc SRAM macro绑定后的OpenROAD面积、时序和活动功耗代理；",
            "- 可变SRAM延迟及真正下游反压；",
            "- 复用强度选择器RTL和独立留出集；",
            "- Motion线对应机制与公平对照。",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
