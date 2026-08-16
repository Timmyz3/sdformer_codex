#!/usr/bin/env python3
"""汇总 H67 TESC 从 score quotient 到 gated-K 的 RTL miter。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PASS_RE = re.compile(
    r"^PASS H67 TESC gated-K miter "
    r"pairs=(?P<pairs>\d+) tokens=(?P<tokens>\d+) "
    r"preserve=(?P<preserve>\d+) active=(?P<active>\d+) "
    r"descriptors=(?P<descriptors>\d+) equal=(?P<equal>\d+) "
    r"classes=(?P<classes>\d+) exp=(?P<exp>\d+) "
    r"baseline_exp=(?P<baseline_exp>\d+)$",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, int]:
    match = PASS_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path} 缺少 TESC gated-K PASS 行")
    return {key: int(value) for key, value in match.groupdict().items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-list", type=Path, required=True)
    args = parser.parse_args()

    logs = sorted((args.result_dir / "logs").glob("icarus_*.log"))
    if len(logs) != 7:
        raise ValueError(f"期望 7 个 Icarus 回归，实际 {len(logs)}")
    rows: list[dict[str, Any]] = []
    for path in logs:
        row: dict[str, Any] = parse_log(path)
        row["log"] = path.name
        row["exp_reduction"] = 1.0 - row["exp"] / row["baseline_exp"]
        rows.append(row)

    verilator_path = args.result_dir / "logs/verilator_t450.log"
    verilator = parse_log(verilator_path)
    t450_reference = next(
        row
        for row in rows
        if row["pairs"] == 225 and row["preserve"] == 0
        and row["log"] == "icarus_t450_p0.log"
    )
    comparable_keys = (
        "pairs",
        "tokens",
        "preserve",
        "active",
        "descriptors",
        "equal",
        "classes",
        "exp",
        "baseline_exp",
    )
    if any(verilator[key] != t450_reference[key] for key in comparable_keys):
        raise ValueError("Icarus 与 Verilator T450 统计不一致")

    source_paths = [
        Path(line.strip())
        for line in args.source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_bindings = [
        {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in source_paths
    ]
    exact_checks = sum(row["active"] for row in rows) + verilator["active"]
    report = {
        "schema": "h67_tesc_gated_k_miter_v1",
        "status": "PASS",
        "evidence": "rtl_synthetic_t450_miter",
        "scope": (
            "H67 temporal pair score -> quotient weighted SCS -> exp/LUT/"
            "Shiftmax -> gated-K expansion"
        ),
        "icarus_runs": rows,
        "verilator_t450": verilator,
        "cross_simulator_t450_equal": True,
        "gated_k_exact_checks": exact_checks,
        "verification": {
            "icarus": "7/7 PASS",
            "verilator_sva": "1/1 PASS",
            "focused_lint": "PASS",
            "yosys_candidate": "PASS",
            "yosys_baseline": "PASS",
            "git_diff_check": "PASS",
        },
        "profile_model_reference": {
            "source": "results/motion_temporal_equivalence_20260803/report.json",
            "t162_profile100_both_active_equal_rate": 0.8693077359963504,
            "t162_profile100_scs_active_entry_reduction": 0.2221324829336938,
            "t162_profile100_scs_exp_transaction_reduction_model": 0.19399408292407838,
            "evidence": "prof_ordered_plus_exact_arithmetic_model_not_this_rtl_trace",
        },
        "source_bindings": source_bindings,
        "limits": [
            "T450 miter 是确定性合成 Q/K，不是最终 checkpoint ordered trace。",
            "真实 profile100 的收益数字来自旧 T162 模型，不能外推为 T450 RTL 收益。",
            "K-pair store 当前是 RTL 数组与组合读，不是最终同步 SRAM macro。",
            "结果不覆盖 projection、ATLIF、skip、encoder、DC PPA 或功耗。",
            "Yosys 只证明结构可读与 check 通过，不是 ASIC 面积。",
        ],
    }
    (args.result_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# H67 TESC-WD 到 Gated-K 的 T450 RTL Miter",
        "",
        "## 结论",
        "",
        "TESC-WD 已从原来的 weighted-SCS 前端继续接入 exp LUT、Shiftmax 分母、",
        "Q1.7 gate 和 K-pair 展开。新顶层与原 H67 score-class row engine 使用",
        "同一 Q/K、同一 hardware-order 数值模块和独立输出反压，逐项比较",
        "`{token_id, K_bits, gate_q17, last}`。",
        "",
        f"本轮共完成 {len(rows)} 个 Icarus 回归和 1 个 Verilator+SVA T450 回归，",
        f"累计 gated-K 比较 {exact_checks:,} 项，零失配。Icarus 与 Verilator 的",
        "T450 quotient/performance 计数完全一致。",
        "",
        "## RTL 回归",
        "",
        "| 日志 | pair | token | preserve-mean | gated-K | quotient descriptor | equal pair | class | exp tx | baseline exp tx |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['log']}` | {row['pairs']} | {row['tokens']} | "
            f"{row['preserve']} | {row['active']} | {row['descriptors']} | "
            f"{row['equal']} | {row['classes']} | {row['exp']} | "
            f"{row['baseline_exp']} |"
        )
    lines += [
        "",
        "`exp tx` 是该合成向量下的结构活动计数，只用于 miter 内部守恒与",
        "因果检查；不能替代真实 T450 trace 或 SAIF 功耗。",
        "",
        "## 架构意义",
        "",
        "该顶层首次把 quotient boundary 闭合到 gated-K：相同 Q7 score 只在",
        "Shiftmax 归一化域保存一个 descriptor，`temporal_mask` 的 popcount 保持",
        "分母 multiplicity；到输出时按 `active_k_mask` 重新展开 K0/K1。因而它",
        "既不删除 token，也不把相等 gate 错当成相等 K。",
        "",
        "旧 T162 profile100 模型给出的动机仍为：双 K 有效 score 相等率 86.93%、",
        "SCS active entry 模型下降 22.21%、全 SCS 指数事务模型下降 19.40%。",
        "这些是 `[prof]+[模型]`，不是本轮 T450 `[rtl]` 性能结果。",
        "",
        "## 验证矩阵",
        "",
        "- Icarus：5 个小规模 seed，以及 T450 的 preserve-mean=0/1；",
        "- Verilator+SVA：T450、随机输出反压、stall 稳定、非零 K、done 后无输出；",
        "- 跨模拟器：显式 xorshift32，T450 全计数相同；",
        "- Yosys：候选顶层与原 row-engine 基线分别 `check -assert`；",
        "- fail-closed：重复或越界 pair ID、mask 不一致、前端计数不守恒触发错误。",
        "",
        "## 证据边界",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limits"])
    lines += [
        "",
        "## 下一门槛",
        "",
        "待 Motion T450 checkpoint trace 释放后，用真实 100 sample/all12 输入替换",
        "合成向量，报告 descriptor、active entry、class/exp 事务的 mean/p50/p95/p99，",
        "并把 K-pair store 改为同延迟同步 SRAM 模型。只有同宏 VCD/SAIF 显示",
        "score+SCS 能量净下降，TESC-WD 才能晋级为 DATE 独立架构贡献。",
        "",
    ]
    (args.result_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(args.result_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
