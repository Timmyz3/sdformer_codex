#!/usr/bin/env python3
"""Report and seal the Local5 score/Shiftmax5-to-Acc RTL comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


GROUP_RE = re.compile(
    r"GROUP backend=(?P<backend>\d+) latency=(?P<latency>\d+) "
    r"group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct_rows>\d+) active=(?P<active>\d+) "
    r"memory_wait=(?P<memory_wait>\d+) terms=(?P<terms>\d+) "
    r"updates=(?P<updates>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rows(path: Path) -> list[dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if "PASS Local5 score-to-projection" not in text:
        raise ValueError(f"{path}缺少PASS")
    rows = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in GROUP_RE.finditer(text)
    ]
    if not rows:
        raise ValueError(f"{path}没有GROUP记录")
    return rows


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "total": float(array.sum()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def read_lines(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines()]


def tool_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--postscore-report", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()

    vector_manifest_path = args.vector_dir / "manifest.json"
    vector_manifest = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    if vector_manifest.get("schema") != "local5_score_projection_vectors_v1":
        raise ValueError("vector manifest schema错误")
    group_count = int(vector_manifest["selection"]["groups"])
    if group_count != 100:
        raise ValueError("正式报告必须覆盖100组")
    postscore = json.loads(args.postscore_report.read_text(encoding="utf-8"))

    configurations: dict[str, dict[str, object]] = {}
    rows_by_key: dict[str, list[dict[str, int]]] = {}
    for backend, backend_id in (("tcfm5", 0), ("linear5", 1)):
        for latency in (1, 2):
            key = f"{backend}_l{latency}"
            rows = parse_rows(args.result_dir / f"{key}_verilator.log")
            if len(rows) != group_count:
                raise ValueError(f"{key} groups={len(rows)} expected={group_count}")
            if any(
                row["backend"] != backend_id
                or row["latency"] != latency
                or row["group"] != index
                or row["score_rows"] != 450
                for index, row in enumerate(rows)
            ):
                raise ValueError(f"{key} identity/score row不一致")
            rows_by_key[key] = rows
            configurations[key] = {
                "cycles": stats([row["cycles"] for row in rows]),
                "score_service_cycles": stats(
                    [row["score_service"] for row in rows]
                ),
                "score_direct_rows": stats(
                    [row["score_direct_rows"] for row in rows]
                ),
                "active_sources": stats([row["active"] for row in rows]),
                "terms": stats([row["terms"] for row in rows]),
                "updates": stats([row["updates"] for row in rows]),
            }

    speedups: dict[str, object] = {}
    stage_results: dict[str, object] = {}
    selection = vector_manifest["selection"]["rows"]
    for latency in (1, 2):
        t_rows = rows_by_key[f"tcfm5_l{latency}"]
        l_rows = rows_by_key[f"linear5_l{latency}"]
        if any(
            t[field] != l[field]
            for t, l in zip(t_rows, l_rows, strict=True)
            for field in (
                "score_rows", "score_service", "score_direct_rows",
                "active", "terms", "updates",
            )
        ):
            raise ValueError(f"L{latency}公平前端/工作量不一致")
        t_total = sum(row["cycles"] for row in t_rows)
        l_total = sum(row["cycles"] for row in l_rows)
        per_group = [
            lrow["cycles"] / trow["cycles"]
            for trow, lrow in zip(t_rows, l_rows, strict=True)
        ]
        speedups[f"l{latency}"] = {
            "ratio_of_totals": l_total / t_total,
            "per_group": stats(per_group),
        }
        stages = {}
        for stage in range(4):
            indices = [
                index
                for index, metadata in enumerate(selection)
                if int(metadata["stage"]) == stage
            ]
            stages[str(stage)] = {
                "groups": len(indices),
                "ratio_of_totals": (
                    sum(l_rows[index]["cycles"] for index in indices)
                    / sum(t_rows[index]["cycles"] for index in indices)
                ),
                "per_group": stats([per_group[index] for index in indices]),
            }
        stage_results[f"l{latency}"] = stages

    postscore_comparison: dict[str, object] = {}
    for latency in (1, 2):
        suffix = f"l{latency}"
        t_integrated = configurations[f"tcfm5_{suffix}"]["cycles"]["total"]
        l_integrated = configurations[f"linear5_{suffix}"]["cycles"]["total"]
        t_post = float(postscore["configurations"][f"tcfm5_{suffix}"]["cycles"]["total"])
        l_post = float(postscore["configurations"][f"linear5_{suffix}"]["cycles"]["total"])
        t_common = t_integrated - t_post
        l_common = l_integrated - l_post
        if t_common != l_common:
            raise ValueError(
                f"{suffix}公共score前端差额不相等: {t_common} != {l_common}"
            )
        post_speedup = l_post / t_post
        integrated_speedup = l_integrated / t_integrated
        postscore_comparison[suffix] = {
            "common_frontend_cycles": t_common,
            "common_frontend_fraction_of_tcfm5": t_common / t_integrated,
            "postscore_speedup": post_speedup,
            "integrated_speedup": integrated_speedup,
            "speedup_retention": integrated_speedup / post_speedup,
        }

    expected_acc_path = args.vector_dir / vector_manifest["artifacts"]["expected_acc"]["file"]
    expected_acc = read_lines(expected_acc_path)
    actual_acc: dict[str, object] = {}
    random_stress: dict[str, object] = {}
    for key in rows_by_key:
        actual_path = args.result_dir / f"{key}_actual_acc32.memh"
        actual = read_lines(actual_path)
        if actual != expected_acc:
            mismatch = next(
                index
                for index, pair in enumerate(zip(actual, expected_acc))
                if pair[0] != pair[1]
            )
            raise ValueError(f"{key} Acc32 mismatch at {mismatch}")
        actual_acc[key] = {
            "entries": len(actual),
            "sha256": sha256(actual_path),
            "zero_mismatch": True,
        }
        stress_path = args.result_dir / f"{key}_random_stress_verilator.log"
        stress_rows = parse_rows(stress_path)
        if len(stress_rows) != 8:
            raise ValueError(f"{key} random stress不是8组")
        random_stress[key] = {
            "groups": len(stress_rows),
            "random_input_gaps": True,
            "random_read_gaps": True,
            "headline_cycles": False,
            "log": stress_path.name,
            "log_sha256": sha256(stress_path),
        }

    source_dir = args.result_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_bindings: list[dict[str, str]] = []
    for source in args.source:
        target = source_dir / source.name
        shutil.copy2(source, target)
        source_bindings.append(
            {"path": str(source.resolve()), "sha256": sha256(source)}
        )

    report = {
        "schema": "local5_score_projection_rtl_report_v1",
        "status": "PASS",
        "evidence": "[rtl]+[profile-qualified-trace]+[real-checkpoint-int8]",
        "scope": (
            "raw Q/K through alpha-XNOR Q7, masked Shiftmax5 Q1.7, "
            "inverse-stencil relation build, source-major terms and Acc32"
        ),
        "groups": group_count,
        "score_gate_row_checks": group_count * 450 * 4,
        "score_gate_scalar_checks": group_count * 450 * 5 * 2 * 4,
        "acc32_checks": len(expected_acc) * 4,
        "configurations": configurations,
        "speedups": speedups,
        "stage_results": stage_results,
        "postscore_comparison": postscore_comparison,
        "actual_acc32": actual_acc,
        "random_stress": random_stress,
        "fairness": [
            "same raw Q/K, score leaf, Shiftmax5, relation SRAM/frontier, term builder, real checkpoint weights and five Acc banks",
            "only destination-to-bank mapping and exact conflict replay differ",
            "headline cycles include score/Shiftmax5, relation build, source-major execution and backend drain, but exclude result readback",
        ],
        "limits": [
            "one outcome-independent qualified T450 group per sample; not every deployment window",
            "two real output channels per group; full 32-channel parameter stress is in a separate package",
            "S0/S1 contain only 4/9 selected groups and do not support a stable stage-level benefit claim",
            "pre-bias/pre-BN/pre-requant/pre-residual and not cross-head/full-encoder output",
            "no foundry PPA or SRAM macro signoff",
        ],
        "vector_manifest": str(vector_manifest_path.resolve()),
        "vector_manifest_sha256": sha256(vector_manifest_path),
        "postscore_report": str(args.postscore_report.resolve()),
        "postscore_report_sha256": sha256(args.postscore_report),
        "source_bindings": source_bindings,
        "execution_receipt": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "iverilog": tool_version(["iverilog", "-V"]),
            "verilator": tool_version(["verilator", "--version"]),
            "yosys": tool_version(["yosys", "-V"]),
        },
    }
    report_json_path = args.result_dir / "report.json"
    report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    l1 = postscore_comparison["l1"]
    l2 = postscore_comparison["l2"]
    markdown = f"""# Local5 score/Shiftmax5 到 Acc32 公平 RTL 主表

> 证据：`[rtl]+[profile-qualified-trace]+[real-checkpoint-int8]`  
> 状态：PASS，但不是 full encoder 或 ASIC PPA。

## 结果

| 合同 | TCFM5 cycle | Linear5 cycle | integrated speedup | post-score speedup | retention |
|---|---:|---:|---:|---:|---:|
| SRAM L1 | {int(configurations['tcfm5_l1']['cycles']['total']):,} | {int(configurations['linear5_l1']['cycles']['total']):,} | {l1['integrated_speedup']:.4f}x | {l1['postscore_speedup']:.4f}x | {l1['speedup_retention']:.2%} |
| SRAM L2 | {int(configurations['tcfm5_l2']['cycles']['total']):,} | {int(configurations['linear5_l2']['cycles']['total']):,} | {l2['integrated_speedup']:.4f}x | {l2['postscore_speedup']:.4f}x | {l2['speedup_retention']:.2%} |

公共 score/Shiftmax5 前端在 L1/L2 分别增加 `{int(l1['common_frontend_cycles']):,}` / `{int(l2['common_frontend_cycles']):,}` cycle；因此必须使用 integrated speedup，而不能继续把 post-score 1.49x 当系统主数字。

## 正确性

- 四配置共检查 {group_count * 450 * 4:,} 行 score/gate，即 {group_count * 450 * 5 * 2 * 4:,} 个标量；
- 四配置共检查 {len(expected_acc) * 4:,} 个 Acc32；
- 所有配置均零失配；
- 四配置各通过8组随机输入/读回 gap 压力；压力周期不进入主结果；
- 100 sample 各一组，stage 配额保持 `{vector_manifest['selection']['stage_counts']}`。

## 边界

- 当前到 pre-bias/pre-BN/pre-requant/pre-residual；
- 每组仅两个真实输出通道，32-channel 参数压力另有封存包；
- 不是所有 window、cross-head、full encoder 或 ASIC PPA。
"""
    (args.result_dir / "report.md").write_text(markdown, encoding="utf-8")

    seal_files = sorted(
        path
        for path in args.result_dir.rglob("*")
        if path.is_file() and path.name != "complete.json"
    )
    complete = {
        "schema": "local5_score_projection_rtl_complete_v1",
        "status": "SEALED",
        "files": {
            str(path.relative_to(args.result_dir)): sha256(path)
            for path in seal_files
        },
    }
    (args.result_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "SEALED",
        "l1_speedup": l1["integrated_speedup"],
        "l2_speedup": l2["integrated_speedup"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
