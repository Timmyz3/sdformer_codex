#!/usr/bin/env python3
"""汇总Local5 issue-side与descriptor-synchronized GASR单变量消融。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


GROUP_RE = re.compile(r"^GROUP (?P<body>.+)$")
FIELD_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>-?\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_log(path: Path) -> list[dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if "PASS post-G0 active projection" not in text:
        raise ValueError(f"缺少PASS：{path}")
    rows = []
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            rows.append(
                {
                    field.group("key"): int(field.group("value"))
                    for field in FIELD_RE.finditer(match.group("body"))
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--vector-manifest",
        type=Path,
        default=Path("tb_qfit/vectors/local5_active_projection_postg0_100/manifest.json"),
    )
    parser.add_argument("--tracked-file", action="append", type=Path, default=[])
    args = parser.parse_args()

    issue = parse_log(args.result_dir / "issue_profile100.log")
    ds = parse_log(args.result_dir / "ds_profile100.log")
    issue_random = parse_log(args.result_dir / "issue_random_sva.log")
    ds_random = parse_log(args.result_dir / "ds_random_sva.log")
    manifest = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    metadata = manifest["selection"]["rows"]
    if not (len(issue) == len(ds) == len(issue_random) == len(ds_random) == 100):
        raise ValueError("四份日志必须各覆盖100组")

    rows = []
    for left, right, meta in zip(issue, ds, metadata, strict=True):
        if left["group"] != right["group"] or left["group"] != meta["vector_group_index"]:
            raise AssertionError("组号错位")
        for field in ("active", "terms", "updates", "sram_reads", "sram_writes"):
            if left[field] != right[field]:
                raise AssertionError(f"组{left['group']}的{field}不一致")
        rows.append(
            {
                "group": left["group"],
                "stage": meta["stage"],
                "issue_cycles": left["cycles"],
                "ds_cycles": right["cycles"],
                "active": left["active"],
                "terms": left["terms"],
                "updates": left["updates"],
                "sram_reads": left["sram_reads"],
                "sram_writes": left["sram_writes"],
            }
        )

    issue_cycles = sum(row["issue_cycles"] for row in rows)
    ds_cycles = sum(row["ds_cycles"] for row in rows)
    stage_rows = []
    for stage in range(4):
        subset = [row for row in rows if row["stage"] == stage]
        issue_stage = sum(row["issue_cycles"] for row in subset)
        ds_stage = sum(row["ds_cycles"] for row in subset)
        stage_rows.append(
            {
                "stage": stage,
                "groups": len(subset),
                "issue_cycles": issue_stage,
                "ds_cycles": ds_stage,
                "ds_speedup": issue_stage / ds_stage,
            }
        )

    checks = {
        "bit_exact_and_equal_work": True,
        "identical_sram_transactions": True,
        "aggregate_speedup_ge_1p02": issue_cycles / ds_cycles >= 1.02,
        "all_stages_non_regression": all(row["ds_speedup"] >= 1 for row in stage_rows),
        "random_sva": len(issue_random) == 100 and len(ds_random) == 100,
        "hashes_recorded": bool(args.tracked_file),
    }
    file_hashes = {str(path): sha256(path) for path in args.tracked_file}
    file_hashes[str(args.vector_manifest)] = sha256(args.vector_manifest)
    summary = {
        "schema": "local5_ds_gasr_sync_ablation_v1",
        "evidence": "同版本同接口同role集合，仅GEOMETRY_SYNC_MODE不同的本机RTL消融",
        "issue_cycles": issue_cycles,
        "ds_cycles": ds_cycles,
        "ds_speedup": issue_cycles / ds_cycles,
        "cycle_reduction": 1 - ds_cycles / issue_cycles,
        "descriptors": sum(row["active"] for row in rows),
        "terms": sum(row["terms"] for row in rows),
        "updates": sum(row["updates"] for row in rows),
        "sram_reads_each": sum(row["sram_reads"] for row in rows),
        "sram_writes_each": sum(row["sram_writes"] for row in rows),
        "stage": stage_rows,
        "review_gate": checks,
        "review_gate_pass": all(checks.values()),
        "hashes": file_hashes,
        "tool_versions": (args.result_dir / "tool_versions.txt").read_text(encoding="utf-8").splitlines(),
    }
    (args.result_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Local5 DS-GASR Prepare 同步点单变量消融",
        "",
        "## 结论",
        "",
        f"在同一份RTL、同一FIFO2、同一五bank GASR和同一profile100下，只切换`GEOMETRY_SYNC_MODE`：issue-side为 {issue_cycles:,} 周期，descriptor-synchronized为 {ds_cycles:,} 周期，DS加速 {issue_cycles / ds_cycles:.3f}x，周期缩短 {1 - ds_cycles / issue_cycles:.2%}。",
        "",
        f"两边均处理 {summary['descriptors']:,} descriptors、{summary['terms']:,} terms和{summary['updates']:,} updates；每边SRAM读/写均为 {summary['sram_reads_each']:,}/{summary['sram_writes_each']:,}。因此周期差异不是工作量或访问集合变化造成。",
        "",
        "## 四 Stage",
        "",
        "| Stage | issue周期 | DS周期 | DS加速 |",
        "|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row['stage']} | {row['issue_cycles']:,} | {row['ds_cycles']:,} | {row['ds_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## 第八轮评审门槛",
            "",
            "| 门槛 | 结果 |",
            "|---|---:|",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"总判定：**{'PASS' if all(checks.values()) else 'FAIL'}**。",
            "",
            "## 证据边界",
            "",
            "该结果证明descriptor同步点相对issue-side同步点的RTL周期因果；不证明标准单元面积、频率、功耗或EDP。下一阶段仍需部署位宽与SRAM宏感知物理评估。",
        ]
    )
    (args.result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
