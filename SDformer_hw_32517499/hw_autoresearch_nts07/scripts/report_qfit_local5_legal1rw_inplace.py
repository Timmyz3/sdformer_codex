#!/usr/bin/env python3
"""汇总 Local5 合法 1RW TCFM5 的 B0-B3 同服务轨迹结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/qfit_local5_legal1rw_inplace_20260810"
PASS_RE = re.compile(
    r"PASS Local5 multi-tile memo=(?P<memo>\d+) inplace=(?P<inplace>\d+) "
    r"acc_backend=(?P<backend>\d+) tx_service=(?P<tx>\d+) seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) token=(?P<token>\d+) "
    r"token_delay_sum=(?P<token_delay>\d+) "
    r"weight_delay_sum=(?P<weight_delay>\d+) "
    r"result_service=(?P<result_service>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(PASS_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"{path} expected one PASS ledger, found {len(matches)}")
    return {key: int(value) for key, value in matches[0].groupdict().items()}


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
    candidates = {
        "b0_scalar_recompute": (0, 0),
        "b1_scalar_memo": (1, 0),
        "b2_inplace_recompute": (0, 1),
        "b3_inplace_memo": (1, 1),
    }
    seeds = (17717, 44257, 48879)
    rows: dict[str, list[dict[str, int]]] = {}
    for candidate, (memo, inplace) in candidates.items():
        parsed = []
        for seed in seeds:
            iv = parse_log(output / f"{candidate}_seed_{seed}_iverilog.log")
            vl = parse_log(output / f"{candidate}_seed_{seed}_verilator_sva.log")
            if iv != vl:
                raise ValueError(f"Icarus/Verilator ledger mismatch: {candidate}/{seed}")
            expected_identity = {"memo": memo, "inplace": inplace, "backend": 1, "tx": 1}
            if any(iv[key] != value for key, value in expected_identity.items()):
                raise ValueError(f"candidate identity mismatch: {candidate}/{seed}")
            if iv["seed"] != seed:
                raise ValueError(f"seed mismatch: {candidate}/{seed}")
            parsed.append(iv)
        rows[candidate] = parsed

    for left, right in (("b0_scalar_recompute", "b2_inplace_recompute"),
                        ("b1_scalar_memo", "b3_inplace_memo")):
        for lhs, rhs in zip(rows[left], rows[right], strict=True):
            for key in ("seed", "token", "token_delay", "weight_delay", "result_service"):
                if lhs[key] != rhs[key]:
                    raise ValueError(f"transaction service mismatch {left}/{right}: {key}")
    reference = rows["b0_scalar_recompute"]
    for candidate in rows:
        for ref, row in zip(reference, rows[candidate], strict=True):
            for key in ("seed", "weight_delay", "result_service"):
                if ref[key] != row[key]:
                    raise ValueError(f"common service mismatch b0/{candidate}: {key}")

    b2_speedups = [
        b0["cycles"] / b2["cycles"]
        for b0, b2 in zip(rows["b0_scalar_recompute"], rows["b2_inplace_recompute"], strict=True)
    ]
    b3_speedups = [
        b1["cycles"] / b3["cycles"]
        for b1, b3 in zip(rows["b1_scalar_memo"], rows["b3_inplace_memo"], strict=True)
    ]
    cycle_gate = min(b2_speedups) >= 1.5
    log_paths = sorted(output.glob("*.log"))
    artifact_paths = [
        output / "source_sha256.txt",
        output / "tool_versions.txt",
        ROOT / "build_qfit/local5_legal1rw_inplace/oracle/metadata.json",
    ]
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_receipt_entries = verify_sha_manifest(artifact_paths[0])
    return {
        "schema": "qfit_local5_legal1rw_inplace_v1",
        "status": "PASS_CYCLE_GATE" if cycle_gate else "REJECT_CYCLE_GATE",
        "evidence": {
            "functional_cycle": "[rtl]",
            "workload": "[rtl] 定向T450三头三输出tile，非真实fullres多样本",
            "asic_ppa": "[待验证]",
        },
        "contract": {
            "accumulator": "five identical synchronous single-port 1RW banks",
            "clear": "lazy-zero validity reset on first head",
            "residency": "run_accumulate preserves backing validity across later heads",
            "service": "nth token/weight response and nth final-result stall are transaction-indexed",
        },
        "candidates": {
            name: {
                "runs": values,
                "mean_cycles": mean(row["cycles"] for row in values),
            }
            for name, values in rows.items()
        },
        "comparisons": {
            "b2_over_b0_per_seed": b2_speedups,
            "b2_over_b0_min": min(b2_speedups),
            "b2_over_b0_mean": mean(b2_speedups),
            "b3_over_b1_per_seed": b3_speedups,
            "b3_over_b1_min": min(b3_speedups),
            "b3_over_b1_mean": mean(b3_speedups),
            "required_b2_over_b0_each_seed": 1.5,
            "cycle_gate_pass": cycle_gate,
        },
        "limitations": [
            "当前 oracle 是定向 T450 数值工作负载，不是 joint-head fullres profile100",
            "尚未绑定 fakeram45 20 个 128x256 宏执行 OpenROAD",
            "尚无 DC/STA/SAIF/PTPX 或 EDP",
        ],
        "receipts": [
            {"file": str(path), "sha256": sha256(path)} for path in log_paths
        ],
        "artifact_receipts": [
            {"file": str(path), "sha256": sha256(path)} for path in artifact_paths
        ],
        "source_receipt_entries_verified": source_receipt_entries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local5 合法 1RW TCFM5 B0-B3 RTL 报告",
        "",
        f"结论：`{report['status']}`。",
        "",
        "| 候选 | mean周期 |",
        "|---|---:|",
    ]
    for name, item in report["candidates"].items():
        lines.append(f"| {name} | {item['mean_cycles']:.1f} |")
    comparisons = report["comparisons"]
    lines += [
        "",
        "| 对照 | per-seed加速 | min | mean |",
        "|---|---|---:|---:|",
        "| B2/B0 | "
        + ", ".join(f"{value:.4f}x" for value in comparisons["b2_over_b0_per_seed"])
        + f" | {comparisons['b2_over_b0_min']:.4f}x | {comparisons['b2_over_b0_mean']:.4f}x |",
        "| B3/B1 | "
        + ", ".join(f"{value:.4f}x" for value in comparisons["b3_over_b1_per_seed"])
        + f" | {comparisons['b3_over_b1_min']:.4f}x | {comparisons['b3_over_b1_mean']:.4f}x |",
        "",
        "四候选使用相同单端口 1RW 合同。B0/B2、B1/B3 的第 n 个 token/weight",
        "事务服务延迟逐项相同；四候选的 weight 与最终结果服务账本相同。",
        "",
        "当前证据为 `[rtl]` 定向回放，不能替代真实 joint-head fullres 多样本和 ASIC PPA。",
        "工具版本、oracle、RTL/TB/SVA/runner 和全部运行日志均有 SHA-256 回执。",
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
    (args.output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"PASS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
