#!/usr/bin/env python3
"""评估RQTB时间对共享gate-product并向偶/奇Acc端口多播的精确收益。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.generate_h67_checkpoint_row_vectors import score_q7
except ModuleNotFoundError:  # 直接执行scripts内文件时，父目录不在sys.path。
    from generate_h67_checkpoint_row_vectors import score_q7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("分布不能为空")

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def analyze_row(vectors: list[tuple[int, int, int, int]]) -> dict[str, int | float]:
    if len(vectors) % 2:
        raise ValueError("时间token数必须为偶数")
    spatial_tokens = len(vectors) // 2
    baseline_commands = 0
    paired_commands = 0
    saved_commands = 0
    equal_pairs = 0
    equal_gate_mismatch = 0
    equal_pairs_both_active_tokens = 0

    for token in range(spatial_tokens):
        first = vectors[token]
        second = vectors[token + spatial_tokens]
        first_score = score_q7(first[0], first[1], first[2])
        second_score = score_q7(second[0], second[1], second[2])
        scalar = first[1].bit_count() + second[1].bit_count()
        baseline_commands += scalar
        if first_score == second_score:
            equal_pairs += 1
            if first[3] != second[3]:
                equal_gate_mismatch += 1
            paired = (first[1] | second[1]).bit_count()
            paired_commands += paired
            saved_commands += (first[1] & second[1]).bit_count()
            if first[1] and second[1]:
                equal_pairs_both_active_tokens += 1
        else:
            paired_commands += scalar

    return {
        "baseline_commands": baseline_commands,
        "paired_commands": paired_commands,
        "saved_commands": saved_commands,
        "command_reduction_ratio": (
            saved_commands / baseline_commands if baseline_commands else 0.0
        ),
        "equal_pairs": equal_pairs,
        "equal_gate_mismatch": equal_gate_mismatch,
        "equal_pairs_both_active_tokens": equal_pairs_both_active_tokens,
    }


def accumulate(target: dict[str, int], row: dict[str, int | float]) -> None:
    for key in (
        "baseline_commands",
        "paired_commands",
        "saved_commands",
        "equal_pairs",
        "equal_gate_mismatch",
        "equal_pairs_both_active_tokens",
    ):
        target[key] += int(row[key])


def build_report(vector_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(vector_path) != manifest.get("vector_sha256"):
        raise ValueError("vector SHA-256与manifest不一致")
    expected_rows = int(manifest["row_count"])
    expected_tokens = int(manifest["tokens_per_row"])
    if expected_tokens != 450:
        raise ValueError("该统计合同只接受T450")

    total = defaultdict(int)
    by_stage: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    row_reductions: list[float] = []
    with vector_path.open(encoding="utf-8") as handle:
        rows, tokens = (int(value) for value in handle.readline().split())
        if rows != expected_rows or tokens != expected_tokens:
            raise ValueError("vector头与manifest不一致")
        for expected_row in range(rows):
            header = [int(value) for value in handle.readline().split()]
            if len(header) != 6 or header[0] != expected_row:
                raise ValueError(f"row header非法: expected={expected_row}, actual={header}")
            stage = header[1]
            vectors: list[tuple[int, int, int, int]] = []
            for _ in range(tokens):
                q_hex, k_hex, peer_hex, gate_text = handle.readline().split()
                vectors.append(
                    (int(q_hex, 16), int(k_hex, 16), int(peer_hex, 16), int(gate_text))
                )
            row = analyze_row(vectors)
            if row["equal_gate_mismatch"] != 0:
                raise ValueError(f"score相等但gate不相等: row={expected_row}")
            accumulate(total, row)
            accumulate(by_stage[stage], row)
            row_reductions.append(float(row["command_reduction_ratio"]))
        if handle.readline():
            raise ValueError("vector文件存在尾随记录")

    baseline = total["baseline_commands"]
    paired = total["paired_commands"]
    reduction = 1.0 - paired / baseline
    stage_rows = {}
    for stage, values in sorted(by_stage.items()):
        stage_rows[str(stage)] = {
            **dict(values),
            "command_reduction_ratio": (
                values["saved_commands"] / values["baseline_commands"]
                if values["baseline_commands"]
                else 0.0
            ),
        }

    return {
        "schema": "h67_rqtb_temporal_pair_multicast_profile_v1",
        "status": "PASS_PROFILE_REJECT_RTL",
        "evidence_level": "[prof-sample0/all12]",
        "mechanism": (
            "仅在RQTB score相等的时间对内共享一次gate-product；因225为空间奇数，"
            "time0/time1原始token ID奇偶相反，可向偶/奇Acc端口无冲突提交"
        ),
        "coverage": {
            "rows": expected_rows,
            "tokens": expected_rows * expected_tokens,
            "pairs": expected_rows * expected_tokens // 2,
            "stages": sorted(by_stage),
        },
        "total": {
            **dict(total),
            "command_reduction_ratio": reduction,
            "ideal_command_speedup": baseline / paired,
        },
        "by_stage": stage_rows,
        "row_command_reduction_distribution": distribution(row_reductions),
        "admission": {
            "minimum_command_reduction_ratio": 0.15,
            "measured_command_reduction_ratio": reduction,
            "pass": reduction >= 0.15,
            "decision": "REJECT_RTL",
            "reason": "score相等不代表同lane的K事件重合；当前仅减少约3%的投影命令。",
        },
        "boundaries": [
            "这是H67 ep30 sample0/window0 all12的精确命令计数，不是多样本分布。",
            "没有计入pairing控制、双写端口、weight响应、反压或时钟周期，因此只是收益上界。",
            "该机制不同于跨整个gate-term任意配对目的的PPDI；不得把两者收益相加。",
            "未过15%准入线，不扩RTL、不列为DATE独立贡献。",
        ],
        "provenance": {
            "vector": {
                "path": str(vector_path.resolve()),
                "sha256": sha256(vector_path),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__)),
            },
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    total = report["total"]
    lines = [
        "# Motion RQTB时间对偶奇多播筛选",
        "",
        "## 结论",
        "",
        "该候选不晋级RTL。RQTB的score等价性很高，但同一lane的两时刻K事件重合不足，精确偶奇双目的多播只获得较小命令收益。",
        "",
        f"- 标量命令：{total['baseline_commands']:,}；候选命令：{total['paired_commands']:,}。",
        f"- 命令减少：{total['command_reduction_ratio']:.2%}；理想命令加速：{total['ideal_command_speedup']:.3f}x。",
        f"- 可配对lane命令：{total['saved_commands']:,}。",
        f"- score相等但gate不相等：{total['equal_gate_mismatch']}。",
        "- 准入线：命令减少至少15%；本轮未通过。",
        "",
        "## Stage分解",
        "",
        "| Stage | 标量命令 | 候选命令 | 节省 | 降低 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for stage, row in report["by_stage"].items():
        lines.append(
            f"| S{stage} | {row['baseline_commands']:,} | {row['paired_commands']:,} | "
            f"{row['saved_commands']:,} | {row['command_reduction_ratio']:.2%} |"
        )
    lines += ["", "## 证据边界", ""]
    lines.extend(f"- {item}" for item in report["boundaries"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.vectors, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", report)
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
