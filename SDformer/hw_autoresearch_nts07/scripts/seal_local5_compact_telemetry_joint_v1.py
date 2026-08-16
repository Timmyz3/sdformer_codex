#!/usr/bin/env python3
"""只读封存 Local5 compact telemetry 分包、生成器和 execution chain。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOPOLOGY = {
    (0, 0, 3), (0, 1, 3), (1, 0, 6), (1, 1, 6),
    (2, 0, 12), (2, 1, 12), (2, 2, 12), (2, 3, 12),
    (2, 4, 12), (2, 5, 12), (3, 0, 24), (3, 1, 24),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 不是 JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_preledger(root: Path, source_sha: str) -> dict[str, Any]:
    complete_path = root / "complete.json"
    ledger_path = root / "compact_telemetry_preledger.json"
    complete = read_json(complete_path)
    ledger = read_json(ledger_path)
    rows = ledger.get("rows")
    if (
        complete.get("schema") != "local5_compact_telemetry_preledger_complete_v1"
        or complete.get("status") != "PASS_COMPACT_TELEMETRY_PRELEDGER_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or complete.get("ledger_sha256") != sha256(ledger_path)
        or complete.get("source_sha256") != source_sha
        or ledger.get("schema") != "local5_compact_telemetry_preledger_v1"
        or ledger.get("status") != complete.get("status")
        or ledger.get("formal_g0") != "DENY"
        or ledger.get("source", {}).get("sha256") != source_sha
        or not isinstance(rows, list)
    ):
        raise ValueError(f"{root.name} complete/ledger/source 绑定不一致")
    identities = [row.get("identity") for row in rows]
    if any(not isinstance(identity, dict) for identity in identities):
        raise ValueError(f"{root.name} identity 缺失")
    samples = sorted({int(identity["sample"]) for identity in identities})
    topology_by_sample: dict[int, set[tuple[int, int, int]]] = {}
    for identity in identities:
        sample = int(identity["sample"])
        topology_by_sample.setdefault(sample, set()).add(
            (int(identity["stage"]), int(identity["block"]), int(identity["heads"]))
        )
    if (
        len(rows) != len(samples) * 12
        or any(topology != EXPECTED_TOPOLOGY for topology in topology_by_sample.values())
        or complete.get("sample_count") != len(samples)
        or complete.get("window_count") != len(rows)
    ):
        raise ValueError(f"{root.name} sample/topology 覆盖不完整")
    if any(
        ".failed" in str(value) or ".staging" in str(value)
        for value in (ledger.get("parent_batch"), ledger.get("source", {}).get("path"))
    ):
        raise ValueError(f"{root.name} 吸收失败/staging 路径")
    final_acc32 = sum(
        int(row["identity"]["heads"]) * 450 * 32 for row in rows
    )
    return {
        "root": str(root),
        "complete_sha256": sha256(complete_path),
        "ledger_sha256": sha256(ledger_path),
        "parent_batch": ledger["parent_batch"],
        "parent_batch_sha256": ledger["parent_batch_sha256"],
        "sample_first": samples[0],
        "sample_last": samples[-1],
        "samples": len(samples),
        "windows": len(rows),
        "final_acc32": final_acc32,
        "cycles": sum(int(row["telemetry"]["cycles"]) for row in rows),
        "frontend_cycles": sum(
            int(row["telemetry"]["frontend_cycles"]) for row in rows
        ),
    }


def validate_execution_chain(root: Path, expected_samples: set[int]) -> dict[str, Any]:
    complete_path = root / "complete.json"
    report_path = root / "execution_chain_audit.json"
    manifest_path = root / "manifest.json"
    complete = read_json(complete_path)
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    if (
        complete.get("schema") != "local5_numeric_execution_chain_complete_v2"
        or complete.get("status") != "PASS_SEALED_EXECUTION_CHAIN_WITH_GAPS_NOT_G0"
        or complete.get("report_sha256") != sha256(report_path)
        or complete.get("manifest_sha256") != sha256(manifest_path)
        or report.get("schema") != "local5_numeric_execution_chain_audit_v2"
        or report.get("status") != "PASS_EXECUTION_CHAIN_WITH_EXPLICIT_GAPS_NOT_G0"
        or manifest.get("schema") != "local5_numeric_execution_chain_manifest_v2"
    ):
        raise ValueError("execution-chain 封存绑定不一致")
    indexed = {int(row["sample"]): row for row in report.get("rows", [])}
    for sample in expected_samples:
        row = indexed.get(sample)
        selected = row.get("selected_proven_execution") if row else None
        if (
            not isinstance(selected, dict)
            or row.get("origin_class") != "PROVEN_RUN_OR_RESUME_RECEIPT"
            or selected.get("execution") not in {"RUN", "RESUME_INCOMPLETE_SHARD"}
        ):
            raise ValueError(f"sample{sample} 缺少可证 RUN/RESUME 来源")
    return {
        "root": str(root),
        "complete_sha256": sha256(complete_path),
        "report_sha256": sha256(report_path),
        "manifest_sha256": sha256(manifest_path),
        "bound_samples": len(expected_samples),
        "run": sum(
            indexed[sample]["selected_proven_execution"]["execution"] == "RUN"
            for sample in expected_samples
        ),
        "resume": sum(
            indexed[sample]["selected_proven_execution"]["execution"]
            == "RESUME_INCOMPLETE_SHARD"
            for sample in expected_samples
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, nargs="+", required=True)
    parser.add_argument("--execution-chain", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    generator = ROOT / "scripts/build_local5_compact_telemetry_preledger_v1.py"
    generator_test = ROOT / "scripts/test_build_local5_compact_telemetry_preledger_v1.py"
    generator_sha = sha256(generator)
    package_rows = [
        validate_preledger(package.resolve(), generator_sha)
        for package in args.packages
    ]
    package_rows.sort(key=lambda row: row["sample_first"])
    expected = set(range(15, 79))
    observed: set[int] = set()
    for row in package_rows:
        current = set(range(row["sample_first"], row["sample_last"] + 1))
        if observed & current:
            raise ValueError("compact telemetry package sample 重叠")
        observed |= current
    if observed != expected:
        raise ValueError("compact telemetry package 未精确覆盖 sample15-78")
    chain = validate_execution_chain(args.execution_chain.resolve(), expected)

    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    source_dir = staging / "source"
    source_dir.mkdir(parents=True)
    files_to_snapshot = [Path(__file__).resolve(), generator, generator_test]
    snapshots = {}
    for source in files_to_snapshot:
        destination = source_dir / source.name
        shutil.copyfile(source, destination)
        snapshots[f"source/{source.name}"] = sha256(destination)

    totals = {
        "samples": sum(row["samples"] for row in package_rows),
        "windows": sum(row["windows"] for row in package_rows),
        "final_acc32": sum(row["final_acc32"] for row in package_rows),
        "mismatch": 0,
        "cycles": sum(row["cycles"] for row in package_rows),
        "frontend_cycles": sum(row["frontend_cycles"] for row in package_rows),
    }
    manifest = {
        "schema": "local5_compact_telemetry_joint_sidecar_v1",
        "status": "PASS_SEALED_COMPACT_TELEMETRY_JOINT_NOT_G0",
        "formal_g0": "DENY",
        "evidence": "[rtl汇总遥测]+[父级数值证据引用]+[执行来源绑定]",
        "packages": package_rows,
        "execution_chain": chain,
        "source_snapshots": snapshots,
        "totals": totals,
        "boundary": [
            "本 sidecar 只读绑定既有四个 preledger，不修改或升级父包证据。",
            "cycle/frontend 是验证回归遥测，不是架构性能或能量证据。",
            "这不是 462600-phase ledger、formal proof、full encoder 或 ASIC PPA。",
            "脚本、SHA、execution chain 和 sidecar 是验证封存，不是 DATE 架构创新。",
        ],
    }
    manifest_path = staging / "joint_manifest.json"
    write_json(manifest_path, manifest)
    report = "\n".join(
        [
            "# Local5 Compact Telemetry 联合封存",
            "",
            "> 状态：PASS；证据边界为 RTL 汇总遥测，不是 phase ledger 或 formal proof。",
            "",
            f"- sample：`{totals['samples']}`（15-78）。",
            f"- canonical block-window：`{totals['windows']}`。",
            f"- final Acc32：`{totals['final_acc32']}`，mismatch=0。",
            f"- execution 来源：RUN `{chain['run']}`，RESUME `{chain['resume']}`。",
            f"- 验证 cycle 汇总：`{totals['cycles']}`，不得作部署性能。",
            "- 生成器与测试已在 sidecar 内快照。",
            "",
        ]
    )
    (staging / "report.md").write_text(report, encoding="utf-8")
    complete = {
        "schema": "local5_compact_telemetry_joint_complete_v1",
        "status": manifest["status"],
        "formal_g0": "DENY",
        "manifest_sha256": sha256(manifest_path),
        "report_sha256": sha256(staging / "report.md"),
        "source_snapshots": snapshots,
        "totals": totals,
    }
    write_json(staging / "complete.json", complete)
    os.replace(staging, output)
    print(json.dumps(complete, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
