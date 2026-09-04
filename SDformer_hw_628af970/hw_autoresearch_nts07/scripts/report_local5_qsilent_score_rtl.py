#!/usr/bin/env python3
"""Seal Local5 Q-silent exact score RTL evidence against the residual baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

GROUP_RE = re.compile(
    r"GROUP backend=(?P<backend>\d+) latency=(?P<latency>\d+) "
    r"group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct_rows>\d+) active=(?P<active>\d+) "
    r"memory_wait=(?P<memory_wait>\d+) terms=(?P<terms>\d+) "
    r"updates=(?P<updates>\d+)"
)
MITER_RE = re.compile(
    r"QSILENT_MITER checked=(?P<checked>\d+) q0_rows=(?P<q0_rows>\d+) "
    r"base_cycles=(?P<base_cycles>\d+) fast_cycles=(?P<fast_cycles>\d+)"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()

    vector_manifest = json.loads((args.vector_dir / "manifest.json").read_text())
    if vector_manifest.get("schema") != "local5_score_projection_vectors_v1":
        raise ValueError("vector manifest schema错误")
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    expected_acc = [
        line.strip().lower()
        for line in (args.vector_dir / vector_manifest["artifacts"]["expected_acc"]["file"]).read_text().splitlines()
    ]

    miter_text = (args.result_dir / "qsilent_miter_verilator.log").read_text()
    if "PASS tb_qfit_local5_qsilent_score_leaf" not in miter_text:
        raise ValueError("miter未通过")
    miter = MITER_RE.search(miter_text)
    if miter is None:
        raise ValueError("miter缺少汇总行")

    keys = {
        "tcfm5_l1": (0, 1),
        "linear5_l1": (1, 1),
        "tcfm5_l2": (0, 2),
    }
    configs: dict[str, dict[str, object]] = {}
    for key, (backend_id, latency) in keys.items():
        rows = parse_rows(args.result_dir / f"{key}_verilator.log")
        if len(rows) != 100:
            raise ValueError(f"{key} groups={len(rows)}")
        actual = [
            line.strip().lower()
            for line in (args.result_dir / f"{key}_actual_acc32.memh").read_text().splitlines()
        ]
        if actual != expected_acc:
            raise ValueError(f"{key} Acc32 mismatch")
        total = sum(row["cycles"] for row in rows)
        service = sum(row["score_service"] for row in rows)
        base_total = float(baseline["configurations"][key]["cycles"]["total"])
        base_service = float(baseline["configurations"][key]["score_service_cycles"]["total"])
        configs[key] = {
            "cycles": total,
            "score_service_cycles": service,
            "baseline_cycles": base_total,
            "baseline_score_service_cycles": base_service,
            "cycle_reduction": 1.0 - total / base_total,
            "service_reduction": 1.0 - service / base_service if base_service else 0.0,
            "speedup_vs_residual": base_total / total,
            "acc32_entries": len(actual),
            "zero_mismatch": True,
        }

    t_l1 = float(configs["tcfm5_l1"]["cycles"])
    l_l1 = float(configs["linear5_l1"]["cycles"])
    report = {
        "schema": "local5_qsilent_score_rtl_report_v1",
        "status": "PASS",
        "evidence": "[rtl]+[profile-qualified-trace]+[real-checkpoint-int8]",
        "innovation": (
            "Query-silent exact score cascade: Q==0 collapses AXNOR to "
            "32-popcount(K) and skips residual XOR walking, preserving "
            "hardware-order Shiftmax5/Acc32."
        ),
        "miter": {key: int(value) for key, value in miter.groupdict().items()},
        "configurations": configs,
        "tcfm5_vs_linear5_l1": l_l1 / t_l1,
        "claim_boundary": [
            "Exact only for Q==0 rows; Q!=0 still uses the sealed residual leaf.",
            "Does not change TCFM5/Linear5 mapping. Relative backend speedup may stay similar.",
            "Not full encoder, not ASIC PPA, not energy.",
        ],
    }

    source_dir = args.result_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in args.source:
        target = source_dir / source.name
        shutil.copy2(source, target)
        files[source.name] = sha256(source)

    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = [
        "# Local5 Query-Silent exact score cascade",
        "",
        "> 证据：`[rtl]+[profile-qualified-trace]+[real-checkpoint-int8]`。状态：PASS。",
        "",
        "## 机制",
        "",
        "Q==0 时 AXNOR raw score 精确退化为 `32-popcount(K)`，可逆地跳过 residual XOR 扫描；",
        "Q!=0 仍走原 leaf。Shiftmax5 与 Acc32 保持 hardware-order bit-exact。",
        "",
        "## 结果",
        "",
        f"- leaf miter：checked={report['miter']['checked']}，Q==0 rows={report['miter']['q0_rows']}",
        f"- TCFM5 L1：{configs['tcfm5_l1']['baseline_cycles']:.0f} → {configs['tcfm5_l1']['cycles']} "
        f"（{float(configs['tcfm5_l1']['speedup_vs_residual']):.4f}x，周期 "
        f"{float(configs['tcfm5_l1']['cycle_reduction'])*100:.2f}%）",
        f"- Linear5 L1：{configs['linear5_l1']['baseline_cycles']:.0f} → {configs['linear5_l1']['cycles']} "
        f"（{float(configs['linear5_l1']['speedup_vs_residual']):.4f}x）",
        f"- TCFM5 L2：{configs['tcfm5_l2']['baseline_cycles']:.0f} → {configs['tcfm5_l2']['cycles']} "
        f"（{float(configs['tcfm5_l2']['speedup_vs_residual']):.4f}x）",
        f"- 同约束 TCFM5/Linear5 L1 仍为 {l_l1/t_l1:.4f}x",
        "",
        "## 边界",
        "",
        "- 只对 Q==0 行走快路径；不是 full encoder 或 ASIC PPA。",
        "- 相对 Linear5 的加速比不会自动变大，因为公共前端两边一起变短。",
        "",
    ]
    (args.result_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    complete = {"schema": "local5_qsilent_score_complete_v1", "status": "SEALED", "files": files}
    (args.result_dir / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "PASS Local5 Q-silent report "
        f"tcfm5_l1={configs['tcfm5_l1']['speedup_vs_residual']:.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
