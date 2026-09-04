#!/usr/bin/env python3
"""汇总 Local5 B0v/B2v 公共向量边界的公平 RTL 对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/qfit_local5_vector_fair_baseline_20260810"
PASS_RE = re.compile(
    r"PASS Local5 multi-tile memo=(?P<memo>\d+) inplace=(?P<inplace>\d+) "
    r"acc_backend=(?P<backend>\d+) tx_service=(?P<tx>\d+) "
    r"seed=(?P<seed>\d+) cycles=(?P<cycles>\d+) token=(?P<token>\d+) "
    r"token_delay_sum=(?P<token_delay>\d+) "
    r"weight_delay_sum=(?P<weight_delay>\d+) "
    r"result_service=(?P<result_service>\d+).*?"
    r"partial=(?P<partial>\d+) final=(?P<final>\d+) "
    r"child_results=(?P<child_results>\d+).*?"
    r"vector=(?P<vector>\d+) "
    r"token_service_hash=(?P<token_hash>[0-9a-fA-F]+) "
    r"weight_service_hash=(?P<weight_hash>[0-9a-fA-F]+) "
    r"result_service_hash=(?P<result_hash>[0-9a-fA-F]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, int | str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(PASS_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"{path} expected one vector PASS ledger, found {len(matches)}")
    result: dict[str, int | str] = {}
    for key, value in matches[0].groupdict().items():
        result[key] = value.lower() if key.endswith("hash") else int(value)
    return result


def verify_sha_manifest(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, raw_path = line.split(maxsplit=1)
        source = Path(raw_path)
        if sha256(source) != expected:
            raise ValueError(f"stale source receipt: {source}")
        count += 1
    if count == 0:
        raise ValueError("empty source receipt")
    return count


def build_report(output: Path) -> dict[str, Any]:
    candidates = {"b0v_materialize": 0, "b2v_resident": 1}
    seeds = (17717, 44257, 48879)
    rows: dict[str, list[dict[str, int | str]]] = {}
    for candidate, inplace in candidates.items():
        runs = []
        for seed in seeds:
            iv = parse_log(output / f"{candidate}_seed_{seed}_iverilog.log")
            vl = parse_log(output / f"{candidate}_seed_{seed}_verilator_sva.log")
            if iv != vl:
                raise ValueError(f"Icarus/Verilator mismatch: {candidate}/{seed}")
            identity = {"memo": 0, "inplace": inplace, "backend": 1,
                        "tx": 1, "vector": 1, "seed": seed}
            if any(iv[key] != value for key, value in identity.items()):
                raise ValueError(f"candidate identity mismatch: {candidate}/{seed}")
            runs.append(iv)
        rows[candidate] = runs

    for b0v, b2v in zip(rows["b0v_materialize"], rows["b2v_resident"], strict=True):
        for key in ("seed", "token", "token_delay", "weight_delay",
                    "result_service", "token_hash", "weight_hash", "result_hash"):
            if b0v[key] != b2v[key]:
                raise ValueError(f"service identity mismatch: {key}")

    speedups = [
        int(b0v["cycles"]) / int(b2v["cycles"])
        for b0v, b2v in zip(rows["b0v_materialize"], rows["b2v_resident"], strict=True)
    ]
    required = 1.20
    cycle_gate = min(speedups) >= required
    source_manifest = output / "source_sha256.txt"
    tool_versions = output / "tool_versions.txt"
    oracle_meta = ROOT / "build_qfit/local5_vector_fair_baseline/oracle/metadata.json"
    for path in (source_manifest, tool_versions, oracle_meta):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "schema": "qfit_local5_vector_fair_baseline_v1",
        "status": "PASS_VECTOR_RESIDENCY" if cycle_gate else "REJECT_VECTOR_RESIDENCY",
        "evidence": {
            "functional_cycle": "[rtl]",
            "workload": "[rtl] 定向T450三头三输出tile，非真实fullres多样本",
            "physical": "[待验证]，因周期门槛失败而停止晋级",
        },
        "contract": {
            "common_boundary": "450x1024 Acc32 vector -> shared 32-lane serializer",
            "b0v": "per-head vector materialization into five legal 1RW cross-head banks",
            "b2v": "cross-head residency in the original five legal 1RW TCFM5 banks",
            "service_audit": "per-transaction token/weight/result identity+index+delay hash",
        },
        "candidates": {
            name: {"runs": runs, "mean_cycles": mean(int(r["cycles"]) for r in runs)}
            for name, runs in rows.items()
        },
        "comparison": {
            "b2v_over_b0v_per_seed": speedups,
            "b2v_over_b0v_min": min(speedups),
            "b2v_over_b0v_mean": mean(speedups),
            "required_each_seed": required,
            "cycle_gate_pass": cycle_gate,
            "physical_promotion": False if not cycle_gate else "pending",
        },
        "interpretation": [
            "旧scalar-B0的约1.94x不能作为DATE强基线收益",
            "公共向量边界下跨头驻留的定向周期收益低于事前1.20x门槛",
            "按冻结规则停止memory-inclusive OpenROAD晋级，不事后降低阈值",
        ],
        "receipts": [
            {"file": str(path), "sha256": sha256(path)}
            for path in sorted(output.glob("*.log"))
        ],
        "artifact_receipts": [
            {"file": str(path), "sha256": sha256(path)}
            for path in (source_manifest, tool_versions, oracle_meta)
        ],
        "source_receipt_entries_verified": verify_sha_manifest(source_manifest),
    }


def render_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    lines = [
        "# Local5 B0v/B2v 公共向量边界公平对照",
        "",
        f"结论：`{report['status']}`。",
        "",
        "| 候选 | mean 周期 |",
        "|---|---:|",
    ]
    for name, item in report["candidates"].items():
        lines.append(f"| {name} | {item['mean_cycles']:.1f} |")
    lines += [
        "",
        "B2v/B0v per-seed："
        + ", ".join(f"`{value:.4f}x`" for value in comparison["b2v_over_b0v_per_seed"])
        + f"；min=`{comparison['b2v_over_b0v_min']:.4f}x`，"
          f"事前门槛=`{comparison['required_each_seed']:.2f}x`。",
        "",
        "两者共享 `450x1024 Acc32 vector -> 32路 serializer` 边界，逐事务",
        "token/weight/result 的 identity、index、delay hash 完全一致，Acc32 零失配。",
        "",
        "由于周期门槛失败，本轮按事前规则停止开放物理晋级。该负结果说明旧",
        "scalar-B0 的约 `1.94x` 主要来自重复标量读出，不能作为正文主收益。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.output_dir)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(f"PASS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
