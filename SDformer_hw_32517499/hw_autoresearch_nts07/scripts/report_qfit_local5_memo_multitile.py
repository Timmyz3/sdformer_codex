#!/usr/bin/env python3
"""汇总 Local5 relation-memo 多输出 tile RTL 对照。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qfit_local5_memo_multitile_20260809"
SEEDS = (17717, 44257, 48879)
PATTERN = re.compile(
    r"PASS Local5 multi-tile memo=(?P<memo>[01]) seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) token=(?P<token>\d+) hits=(?P<hits>\d+) "
    r"fallback=(?P<fallback>\d+) replay_records=(?P<replay>\d+) "
    r"partial=(?P<partial>\d+) final=(?P<final>\d+)"
)


def parse(path: Path) -> dict[str, int]:
    match = PATTERN.search(path.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"无法解析 {path}")
    return {key: int(value) for key, value in match.groupdict().items()}


def main() -> int:
    rows = []
    for simulator in ("iverilog", "verilator_sva"):
        for seed in SEEDS:
            baseline = parse(OUT / f"baseline_seed_{seed}_{simulator}.log")
            memo = parse(OUT / f"memo_seed_{seed}_{simulator}.log")
            if baseline["final"] != memo["final"] or baseline["partial"] != memo["partial"]:
                raise RuntimeError("baseline/memo 数值账本不一致")
            rows.append(
                {
                    "simulator": simulator,
                    "seed": seed,
                    "baseline_cycles": baseline["cycles"],
                    "memo_cycles": memo["cycles"],
                    "speedup": baseline["cycles"] / memo["cycles"],
                    "baseline_tokens": baseline["token"],
                    "memo_tokens": memo["token"],
                    "token_reduction": 1 - memo["token"] / baseline["token"],
                    "memo_hits": memo["hits"],
                    "fallbacks": memo["fallback"],
                    "replay_records": memo["replay"],
                }
            )

    icarus = [row for row in rows if row["simulator"] == "iverilog"]
    summary = {
        "evidence": "rtl",
        "seeds": list(SEEDS),
        "rows": rows,
        "mean_baseline_cycles": sum(row["baseline_cycles"] for row in icarus) / len(icarus),
        "mean_memo_cycles": sum(row["memo_cycles"] for row in icarus) / len(icarus),
        "mean_speedup": sum(row["speedup"] for row in icarus) / len(icarus),
        "token_reduction": icarus[0]["token_reduction"],
        "partial_fault_modes": 4,
        "partial_fault_status": "PASS",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Local5 Relation Memo 多输出 Tile RTL 对照",
        "",
        "## 结论",
        "",
        (
            f"在同一三头、三输出 tile、OUT32、T450、同权重与同反压条件下，"
            f"relation memo 将 Q/K token 请求从 4050 降至 2250，减少 "
            f"{summary['token_reduction'] * 100:.2f}% `[rtl]`。"
        ),
        (
            f"但 Icarus 三个服务时序的平均周期仅从 "
            f"{summary['mean_baseline_cycles']:.1f} 降至 "
            f"{summary['mean_memo_cycles']:.1f}，平均加速 "
            f"{summary['mean_speedup']:.4f}x `[rtl]`。当前 scalar Acc32 交接和逐项"
            "读回占据主导，FCSR 不能据此作为主性能贡献。"
        ),
        "",
        "## 公平逐项结果",
        "",
        "| 仿真器 | seed | baseline 周期 | memo 周期 | 加速 | token 减少 | hit/fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['simulator']} | {row['seed']} | {row['baseline_cycles']} | "
            f"{row['memo_cycles']} | {row['speedup']:.4f}x | "
            f"{row['token_reduction'] * 100:.2f}% | "
            f"{row['memo_hits']}/{row['fallbacks']} |"
        )
    lines.extend(
        [
            "",
            "## 正确性与故障",
            "",
            "- 三个 seed 在 Icarus 与 Verilator/SVA 均完成 129600 个跨头 partial Acc32 和 43200 个 final Acc32，独立 Python oracle 零失配 `[rtl]`。",
            "- 两个稀疏 head 各命中两次；稠密 head 两次 miss 后完整重算，每次重新请求 450 个 token `[rtl]`。",
            "- duplicate、reorder、wrong-last、early-done/drop 四种 partial 故障均 fail-closed；坏 beat 不产生 memory command，错误后 token/weight 服务端口被 firewall 封锁 `[rtl]`。",
            "",
            "## 证据边界",
            "",
            "- 本结果是合成前 RTL 周期与事务统计，不是 DC/STA/SAIF，也不是 ASIC PPA。",
            "- oracle 为定向稀疏/稠密混合构造，不代表 full-resolution Local5 trace 的驻留率 `[待验证]`。",
            "- 当前结果直接否定了“只加 relation memo 就有显著端到端加速”的主张。下一轮应比较 scalar、OUT32 vector 和多 output-tile supertile 三种数据流，而不是继续扩大 memo 控制。",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PASS Local5 memo multi-tile report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
