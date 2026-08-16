#!/usr/bin/env python3
"""Build a fail-closed evidence report for the Local5 rolling-FCSR sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


GROUP_RE = re.compile(
    r"^GROUP .* group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"active=(?P<active>\d+).* terms=(?P<terms>\d+) "
    r"updates=(?P<updates>\d+)"
)
PASS_RE = re.compile(
    r"^PASS post-G0 active projection .* groups=(?P<groups>\d+) "
    r"total_cycles=(?P<cycles>\d+) descriptors=(?P<descriptors>\d+)"
)
BAD_RE = re.compile(r"%Error|Assertion failed|MISMATCH|\$fatal|\bFAIL\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path, groups: int) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    passes: list[tuple[int, int, int]] = []
    text = path.read_text(encoding="utf-8", errors="strict")
    if BAD_RE.search(text):
        raise ValueError(f"bad terminal marker in {path}")
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            values = {key: int(value) for key, value in match.groupdict().items()}
            group = values.pop("group")
            if group in rows:
                raise ValueError(f"duplicate group {group} in {path}")
            rows[group] = values
        match = PASS_RE.match(line)
        if match:
            passes.append(tuple(int(value) for value in match.groups()))
    if sorted(rows) != list(range(groups)):
        raise ValueError(f"non-contiguous group population in {path}")
    expected = (
        groups,
        sum(row["cycles"] for row in rows.values()),
        sum(row["active"] for row in rows.values()),
    )
    if passes != [expected]:
        raise ValueError(f"terminal PASS mismatch in {path}: {passes} != {expected}")
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def comparison(
    baseline: dict[int, dict[str, int]], candidate: dict[int, dict[str, int]]
) -> dict[str, object]:
    baseline_cycles = sum(row["cycles"] for row in baseline.values())
    candidate_cycles = sum(row["cycles"] for row in candidate.values())
    ratios = [
        baseline[group]["cycles"] / candidate[group]["cycles"]
        for group in sorted(baseline)
    ]
    return {
        "cycles": candidate_cycles,
        "aggregate_speedup": baseline_cycles / candidate_cycles,
        "cycle_reduction": 1.0 - candidate_cycles / baseline_cycles,
        "per_group": {
            "min": min(ratios),
            "p50": percentile(ratios, 0.50),
            "p95": percentile(ratios, 0.95),
            "max": max(ratios),
            "wins": sum(value > 1.0 for value in ratios),
            "ties": sum(value == 1.0 for value in ratios),
            "losses": sum(value < 1.0 for value in ratios),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/local5_rolling_sidecar_rtl_20260814"),
    )
    parser.add_argument("--groups", type=int, default=100)
    args = parser.parse_args()
    result_dir = args.result_dir
    logs = {
        "t450_materialize": result_dir / "tcfm5_t450_g100_iverilog.log",
        "rolling_unfiltered": result_dir / "tcfm5_rolling_g100_iverilog.log",
        "rolling_read_filter": result_dir / "tcfm5_active_g100_iverilog.log",
        "rolling_event_filter_icarus": result_dir
        / "tcfm5_active_sched_g100_iverilog.log",
        "rolling_event_filter_verilator": result_dir
        / "tcfm5_active_sched_g100_verilator_assert.log",
        "dynamic_frontier": result_dir / "tcfm5_dynamic_g100_iverilog.log",
        "nonblocking_stripe": result_dir / "tcfm5_stripe_g100_iverilog.log",
    }
    rows = {name: parse_log(path, args.groups) for name, path in logs.items()}
    if rows["rolling_event_filter_icarus"] != rows["rolling_event_filter_verilator"]:
        raise ValueError("Icarus/Verilator per-group ledger mismatch")

    baseline = rows["t450_materialize"]
    conservation_fields = ("active", "terms", "updates")
    conservation = {
        field: sum(row[field] for row in baseline.values())
        for field in conservation_fields
    }
    for name, ledger in rows.items():
        for field, expected in conservation.items():
            actual = sum(row[field] for row in ledger.values())
            if actual != expected:
                raise ValueError(f"{name} violates {field}: {actual} != {expected}")

    random_logs = sorted(result_dir.glob("tcfm5_active_sched_bp8_seed*_verilator_assert.log"))
    if len(random_logs) != 8:
        raise ValueError("expected exactly eight random-gap logs")
    random_cycles = []
    for path in random_logs:
        ledger = parse_log(path, 8)
        if sum(row["active"] for row in ledger.values()) != 753:
            raise ValueError(f"random-gap descriptor conservation failed: {path}")
        random_cycles.append(sum(row["cycles"] for row in ledger.values()))

    baseline_state_bits = 450 * (32 + 5 * (9 + 1))
    candidate_state_bits = 3 * 15 * (32 + 5 * (9 + 1)) + 3 * 15
    final_rows = rows["rolling_event_filter_icarus"]
    report = {
        "schema": "local5_active_rolling_fcsr_sidecar_rtl_v2",
        "status": "ADMIT_AS_208_PRODUCTION_INTEGRATION",
        "evidence": "[rtl]",
        "scope": (
            "100 sample-disjoint/stage-weighted real checkpoint-weight groups; "
            "post-score relation-to-Acc32; OUT_DIM=2 tile; not encoder"
        ),
        "architectural_delta": {
            "storage_object": "T450 sealed relation/K image -> three-row active frontier",
            "schedule_boundary": "seal-then-enumerate -> topology-ordered rolling retirement",
            "exact_filter": "K==0 source is removed before FCSR pending-event allocation",
        },
        "strong_baselines": {
            "t450_materialize": {
                "cycles": sum(row["cycles"] for row in baseline.values()),
                "state_bits": baseline_state_bits,
            },
            "rolling_unfiltered": comparison(baseline, rows["rolling_unfiltered"]),
            "rolling_read_filter": comparison(baseline, rows["rolling_read_filter"]),
            "rolling_event_filter": comparison(baseline, final_rows),
            "dynamic_frontier": comparison(baseline, rows["dynamic_frontier"]),
            "nonblocking_stripe": comparison(baseline, rows["nonblocking_stripe"]),
        },
        "increment_vs_existing_fcsr": {
            "speedup": sum(row["cycles"] for row in rows["rolling_unfiltered"].values())
            / sum(row["cycles"] for row in final_rows.values()),
            "cycle_reduction": 1.0
            - sum(row["cycles"] for row in final_rows.values())
            / sum(row["cycles"] for row in rows["rolling_unfiltered"].values()),
            "novel_architecture": False,
        },
        "state": {
            "baseline_bits": baseline_state_bits,
            "candidate_bits": candidate_state_bits,
            "ratio": candidate_state_bits / baseline_state_bits,
            "reduction": 1.0 - candidate_state_bits / baseline_state_bits,
            "accounting": "six 45-entry rings plus one 45-bit K-active ring",
        },
        "conservation": {
            "groups": args.groups,
            "descriptors": conservation["active"],
            "terms": conservation["terms"],
            "updates": conservation["updates"],
            "acc32_mismatch": 0,
            "cross_simulator_per_group_exact": True,
        },
        "random_gap_verilator_assert": {
            "seeds": 8,
            "groups_per_seed": 8,
            "descriptors_per_seed": 753,
            "cycle_min": min(random_cycles),
            "cycle_max": max(random_cycles),
            "status": "PASS",
        },
        "claim_boundary": [
            "sidecar RTL; production module is unchanged",
            "score and Shiftmax5 are excluded",
            "OUT_DIM=2 projection tile, not a full encoder",
            "no SRAM macro energy, DC, STA, SAIF, or ASIC PPA",
            "extends the existing inverse-stencil/TCFM5 Local5 dataflow; not a separate contribution",
            "three-row FCSR is inherited from docs/208; this package closes production integration and exact event filtering",
        ],
        "sha256": {str(path.relative_to(result_dir)): sha256(path) for path in logs.values()},
    }
    report["sha256"].update(
        {str(path.relative_to(result_dir)): sha256(path) for path in random_logs}
    )
    report_path = result_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    final = report["strong_baselines"]["rolling_event_filter"]
    distribution = final["per_group"]
    markdown = f"""# Local5 Active-Rolling FCSR 旁路 RTL 收口

- 裁决：`{report['status']}`，证据 `{report['evidence']}`。
- 边界：{report['scope']}。
- 强基线：全 T450 relation/K 物化后遍历，`{report['strong_baselines']['t450_materialize']['cycles']}` cycles。
- 候选：三行 active frontier 随拓扑退休，`{final['cycles']}` cycles，`{final['aggregate_speedup']:.4f}x`，周期 `{final['cycle_reduction']:.2%}`。
- 状态：`{baseline_state_bits} -> {candidate_state_bits}` bit，降低 `{report['state']['reduction']:.2%}`。
- 逐组 min/p50/p95/max：`{distribution['min']:.4f}/{distribution['p50']:.4f}/{distribution['p95']:.4f}/{distribution['max']:.4f}x`；win/tie/loss=`{distribution['wins']}/{distribution['ties']}/{distribution['losses']}`。
- 守恒：descriptor `{conservation['active']}`，term `{conservation['terms']}`，update `{conservation['updates']}`，Acc32 mismatch `0`；Icarus/Verilator 逐组账本一致。
- 随机间隙：8 seeds x 8 groups，`Verilator --assert` 全 PASS。

## 增量拆分

| 版本 | cycles | 相对 T450 |
|---|---:|---:|
| 三行 rolling，无 K-zero filter | {report['strong_baselines']['rolling_unfiltered']['cycles']} | {report['strong_baselines']['rolling_unfiltered']['aggregate_speedup']:.4f}x |
| 三行 rolling，read-boundary filter | {report['strong_baselines']['rolling_read_filter']['cycles']} | {report['strong_baselines']['rolling_read_filter']['aggregate_speedup']:.4f}x |
| 三行 rolling，event-boundary filter | {final['cycles']} | {final['aggregate_speedup']:.4f}x |
| Dynamic Frontier | {report['strong_baselines']['dynamic_frontier']['cycles']} | {report['strong_baselines']['dynamic_frontier']['aggregate_speedup']:.4f}x |
| Nonblocking Stripe | {report['strong_baselines']['nonblocking_stripe']['cycles']} | {report['strong_baselines']['nonblocking_stripe']['aggregate_speedup']:.4f}x |

三行 FCSR 的存储对象与调度边界来自 `docs/208`，本轮只完成真实 TCFM5/Acc32 生产边界接通，并把 exact K-zero filter 前移到 retirement-event 分配前。相对旧 FCSR 的增量为 `{report['increment_vs_existing_fcsr']['speedup']:.4f}x`，属于工程闭环和实现优化，不是新架构贡献。当前仍不得写成 encoder 加速、ASIC PPA 或封存主表更新。
"""
    (result_dir / "report.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
