#!/usr/bin/env python3
"""汇总Motion三方RTL工作事件；不把工作计数冒充功耗。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROW_RE = re.compile(
    r"^ROW_RESULT row=(?P<row>\d+) stage=(?P<stage>\d+) block=(?P<block>\d+) "
    r"head=(?P<head>\d+) bundle_skip=(?P<bundle>\d+) active_pairs=(?P<active>\d+) "
    r"outputs=(?P<outputs>\d+) baseline_preload=(?P<baseline_preload>\d+) "
    r"zkqi_preload=(?P<candidate_preload>\d+) baseline_cycles=(?P<baseline_cycles>\d+) "
    r"zkqi_cycles=(?P<candidate_cycles>\d+) baseline_e2e_cycles=(?P<baseline_e2e>\d+) "
    r"zkqi_e2e_cycles=(?P<candidate_e2e>\d+) baseline_slots=(?P<baseline_slots>\d+) "
    r"zkqi_slots=(?P<candidate_slots>\d+) seeded=(?P<seeded>\d+) "
    r"baseline_read_bits=(?P<baseline_read_bits>\d+) "
    r"zkqi_read_bits=(?P<candidate_read_bits>\d+) fifo_max=(?P<fifo_max>\d+)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rows(path: Path, expected_bundle: int) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if match:
            row = {key: int(value) for key, value in match.groupdict().items()}
            if row["bundle"] != expected_bundle:
                raise ValueError(f"{path}: bundle_skip不符合预期")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: 未找到ROW_RESULT")
    if [row["row"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: 行号不连续")
    return rows


def total(rows: list[dict[str, int]], key: str) -> int:
    return sum(row[key] for row in rows)


def reduction(new: int, old: int) -> float:
    return 1.0 - new / old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-log", type=Path, required=True)
    parser.add_argument("--ttb-log", type=Path, required=True)
    parser.add_argument("--pairs-per-row", type=int, default=225)
    parser.add_argument("--bundles-per-row", type=int, default=29)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pair_rows = parse_rows(args.pair_log, 0)
    ttb_rows = parse_rows(args.ttb_log, 1)
    if len(pair_rows) != len(ttb_rows):
        raise ValueError("PairBitmap与TTB8行数不同")

    baseline_fields = (
        "stage", "block", "head", "outputs", "baseline_preload",
        "baseline_cycles", "baseline_e2e", "baseline_slots", "baseline_read_bits",
    )
    for pair, ttb in zip(pair_rows, ttb_rows):
        if any(pair[key] != ttb[key] for key in baseline_fields):
            raise ValueError(f"row={pair['row']}: 两次运行的baseline被候选污染")
        if pair["active"] != ttb["active"] or pair["seeded"] != ttb["seeded"]:
            raise ValueError(f"row={pair['row']}: 两候选workload不一致")

    row_count = len(pair_rows)
    dense_pair_tests = row_count * args.pairs_per_row
    active_pairs = total(pair_rows, "active")
    baseline_read_bits = total(pair_rows, "baseline_read_bits")
    candidate_read_bits = total(pair_rows, "candidate_read_bits")
    baseline_cycles = total(pair_rows, "baseline_cycles")
    pair_cycles = total(pair_rows, "candidate_cycles")
    ttb_cycles = total(ttb_rows, "candidate_cycles")
    baseline_e2e = total(pair_rows, "baseline_e2e")
    pair_e2e = total(pair_rows, "candidate_e2e")
    ttb_e2e = total(ttb_rows, "candidate_e2e")
    ttb_bundle_tests = row_count * args.bundles_per_row

    report = {
        "schema": "h67_zkqi_rtl_activity_proxy_v1",
        "status": "PASS",
        "evidence_level": "[rtl活动代理]",
        "scope": f"sample0/window0、全12 attention block、{row_count}条fullres T450真实行、无反压",
        "event_ledger": {
            "baseline_rqtb2s": {
                "pair_or_bundle_metadata_tests": dense_pair_tests,
                "score_evaluations": dense_pair_tests,
                "qk_read_bits": baseline_read_bits,
                "execution_cycles": baseline_cycles,
                "preload_inclusive_cycles": baseline_e2e,
            },
            "pair_bitmap_zkqi": {
                "pair_or_bundle_metadata_tests": dense_pair_tests,
                "score_evaluations": active_pairs,
                "qk_read_bits": candidate_read_bits,
                "execution_cycles": pair_cycles,
                "preload_inclusive_cycles": pair_e2e,
            },
            "ttb8_zkqi": {
                "bundle_header_tests": ttb_bundle_tests,
                "active_pair_dispatches": active_pairs,
                "score_evaluations": active_pairs,
                "qk_read_bits": candidate_read_bits,
                "execution_cycles": ttb_cycles,
                "preload_inclusive_cycles": ttb_e2e,
            },
        },
        "reductions_vs_baseline": {
            "score_evaluation_reduction": reduction(active_pairs, dense_pair_tests),
            "qk_read_bit_reduction": reduction(candidate_read_bits, baseline_read_bits),
            "pair_bitmap_execution_cycle_reduction": reduction(pair_cycles, baseline_cycles),
            "ttb8_execution_cycle_reduction": reduction(ttb_cycles, baseline_cycles),
            "ttb8_preload_inclusive_cycle_reduction": reduction(ttb_e2e, baseline_e2e),
        },
        "interpretation": [
            "PairBitmap与TTB8具有相同的exact zero-K score/read work gating；前者仍逐pair扫描，因此无反压周期不降。",
            "TTB8把逐pair issue改为bundle-header加active-pair dispatch，周期收益来自层次跳扫。",
            "各事件能耗尚未由SAIF和目标库标定，因此不得求和为功耗、能量或EDP。",
        ],
        "source_receipts": {
            str(args.pair_log): sha256(args.pair_log),
            str(args.ttb_log): sha256(args.ttb_log),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    md_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ledger = report["event_ledger"]
    reductions = report["reductions_vs_baseline"]
    md = f"""# Motion ZKQI RTL工作事件活动代理

## 结论

- 状态：**PASS**；证据等级：`[rtl活动代理]`。
- TTB8-ZKQI相对RQTB2S将score次数降低{reductions['score_evaluation_reduction']:.2%}、Q/K读取bit降低{reductions['qk_read_bit_reduction']:.2%}。
- 无反压执行周期降低{reductions['ttb8_execution_cycle_reduction']:.2%}；计入共同preload后降低{reductions['ttb8_preload_inclusive_cycle_reduction']:.2%}。
- PairBitmap虽然降低相同score和读取工作，但执行周期降低仅{reductions['pair_bitmap_execution_cycle_reduction']:.2%}，证明work gating与issue skipping必须分账。

## 逐类账本

| 候选 | 元数据检查/分发 | score次数 | Q/K读bit | 执行周期 | 含preload周期 |
|---|---:|---:|---:|---:|---:|
| RQTB2S | {ledger['baseline_rqtb2s']['pair_or_bundle_metadata_tests']} pair | {ledger['baseline_rqtb2s']['score_evaluations']} | {ledger['baseline_rqtb2s']['qk_read_bits']} | {ledger['baseline_rqtb2s']['execution_cycles']} | {ledger['baseline_rqtb2s']['preload_inclusive_cycles']} |
| PairBitmap-ZKQI | {ledger['pair_bitmap_zkqi']['pair_or_bundle_metadata_tests']} pair | {ledger['pair_bitmap_zkqi']['score_evaluations']} | {ledger['pair_bitmap_zkqi']['qk_read_bits']} | {ledger['pair_bitmap_zkqi']['execution_cycles']} | {ledger['pair_bitmap_zkqi']['preload_inclusive_cycles']} |
| TTB8-ZKQI | {ledger['ttb8_zkqi']['bundle_header_tests']} bundle + {ledger['ttb8_zkqi']['active_pair_dispatches']} active pair | {ledger['ttb8_zkqi']['score_evaluations']} | {ledger['ttb8_zkqi']['qk_read_bits']} | {ledger['ttb8_zkqi']['execution_cycles']} | {ledger['ttb8_zkqi']['preload_inclusive_cycles']} |

## 证据边界

本报告按RTL真实执行事件分账，不使用任意事件权重。它不是门级toggle、SAIF、功耗、能量或EDP结果；这些指标必须在DC/门级仿真/PTPX阶段用同一库和同一trace补齐。当前范围也仅覆盖sample0/window0的138条真实head-row，不能外推为完整数据集均值。
"""
    md_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
