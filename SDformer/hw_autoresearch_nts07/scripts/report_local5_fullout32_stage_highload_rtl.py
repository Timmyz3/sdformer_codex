#!/usr/bin/env python3
"""封存 Local5 四 stage 高负载组的完整32输出通道 RTL 回放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="ascii") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["selection"]["rows"]
    if manifest["shape"]["out_dim"] != 32 or len(rows) != 4:
        raise ValueError("fullout32包必须为4组、OUT_DIM=32")
    if {int(row["stage"]) for row in rows} != {0, 1, 2, 3}:
        raise ValueError("fullout32包必须覆盖四个stage")
    if any(len(row["projection_output_channels"]) != 32 for row in rows):
        raise ValueError("存在非完整32输出通道")
    for name, artifact in manifest["artifacts"].items():
        path = args.vector_dir / artifact["file"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"vector artifact {name} SHA失配")

    expected = args.vector_dir / manifest["artifacts"]["expected_acc"]["file"]
    expected_values = 4 * 450 * 32
    if count_lines(expected) != expected_values:
        raise ValueError("expected Acc32数量不守恒")
    configurations: dict[str, dict[str, object]] = {}
    actual_hashes: set[str] = set()
    for latency in (1, 2):
        for backend in ("tcfm5", "linear5"):
            key = f"{backend}_l{latency}"
            log = args.result_dir / f"{key}_verilator.log"
            actual = args.result_dir / f"{key}_actual_acc32.memh"
            if "PASS post-G0 active projection" not in log.read_text(encoding="utf-8"):
                raise ValueError(f"{key} RTL未通过")
            if count_lines(actual) != expected_values:
                raise ValueError(f"{key} actual Acc32数量不守恒")
            actual_digest = sha256(actual)
            actual_hashes.add(actual_digest)
            configurations[key] = {
                "actual_values": expected_values,
                "actual_sha256": actual_digest,
                "log": log.name,
                "log_sha256": sha256(log),
            }
    expected_digest = sha256(expected)
    if actual_hashes != {expected_digest}:
        raise ValueError("四配置actual Acc32与expected不一致")

    receipt = args.result_dir / "source_sha256.txt"
    source_dir = args.result_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, str] = {}
    for line in receipt.read_text(encoding="utf-8").splitlines():
        expected_source_digest, raw_path = line.split(maxsplit=1)
        source = Path(raw_path).resolve()
        if sha256(source) != expected_source_digest:
            raise ValueError(f"source receipt SHA失配: {source}")
        destination = source_dir / source.name
        shutil.copyfile(source, destination)
        snapshots[str(destination.relative_to(args.result_dir))] = sha256(destination)

    result = {
        "schema": "local5_fullout32_stage_highload_rtl_v1",
        "status": "RTL_REPLAY_COMPLETE",
        "evidence": "[rtl]+[selected-highload]+[full-output-tile]",
        "scope": "four stage-wise maximum-term groups, post-score relation-to-Acc, OUT_DIM=32",
        "selection": [
            {
                key: row[key]
                for key in (
                    "input_group_index", "sample", "stage", "block", "window",
                    "head", "active_sources", "terms", "updates",
                    "projection_output_tile", "projection_output_channels",
                )
            }
            for row in rows
        ],
        "vector_manifest": str(manifest_path.resolve()),
        "vector_manifest_sha256": sha256(manifest_path),
        "expected_acc32": {
            "values": expected_values,
            "sha256": expected_digest,
        },
        "configurations": configurations,
        "limits": [
            "四组按每stage最大product-term选择，是定向高负载正确性压力，不是无偏性能样本。",
            "每组只回放一个output tile的32通道，不是该stage全部output tile或跨head最终输出。",
            "作用域仍从post-score relation到Acc，不含score/Shiftmax5、bias/BN/requant/residual。",
            "该包不提供ASIC PPA或full encoder结论。",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.result_dir / "report.json"
    report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Local5 四 Stage 高负载完整输出通道 RTL 回放",
        "",
        "## 结论",
        "",
        "四个 stage 各选 full-profile 中 product-term 最大的一个 group，回放一个完整 "
        "32-channel output tile。TCFM5/Linear5 × L1/L2 四配置各比较 "
        f"`{expected_values:,}` 个 Acc32，全部与 expected 相同；合计 "
        f"`{expected_values * 4:,}` 次比较零失配。",
        "",
        "证据为 **[rtl]+[selected-highload]+[full-output-tile]**，不作为性能总体估计。",
        "",
        "| Stage | sample/block/window/head | active | term | update |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in result["selection"]:
        lines.append(
            f"| S{row['stage']} | {row['sample']}/{row['block']}/{row['window']}/{row['head']} | "
            f"{row['active_sources']} | {row['terms']} | {row['updates']} |"
        )
    lines += ["", "## 边界", ""]
    lines.extend(f"- {item}" for item in result["limits"])
    report_md = args.result_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    package_files = {
        path.name: sha256(path)
        for path in sorted(args.result_dir.iterdir())
        if path.is_file() and path.name != "complete.json"
    }
    package_files.update(snapshots)
    complete = {
        "schema": "local5_fullout32_stage_highload_rtl_package_v1",
        "status": "SEALED",
        "evidence": result["evidence"],
        "package_files": package_files,
    }
    (args.result_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
