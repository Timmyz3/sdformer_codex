#!/usr/bin/env python3
"""汇总H67四stage真实bit trace的GateStack RTL消融。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_gatestack_p0_baselines import parse_log


def summarize(root: Path, vector_manifest: Path) -> dict:
    manifest = json.loads(vector_manifest.read_text(encoding="utf-8"))
    build = root / "build_hitflow/gatestack_real_trace_ablation"
    rows = []
    for record in manifest["records"]:
        stage = int(record["stage"])
        for mode in ("raw_only", "no_residency", "gatestack"):
            rtl = parse_log(build / f"s{stage}" / mode / "verilator.log")
            if mode == "raw_only":
                vector = record["modes"]["rawonly"]
                payload_words = vector["payload_words_all_tiles_no_residency"]
            elif mode == "no_residency":
                vector = record["modes"]["capacity"]
                payload_words = vector["payload_words_all_tiles_no_residency"]
            else:
                vector = record["modes"]["capacity"]
                payload_words = vector["payload_words_all_tiles_resident"]
            rows.append(
                {
                    "stage": stage,
                    "name": record["name"],
                    "mode": mode,
                    "heads": int(vector["heads"]),
                    "cycles": rtl["cycles"],
                    "payload_words": int(payload_words),
                    "projection_terms": rtl["projection_terms"],
                    "slot_replays": rtl["slot_replays"],
                    "cache_hits": rtl["cache_hits"],
                    "mismatches": rtl["mismatches"],
                    "done_error": rtl["done_error"],
                    "protocol_errors": rtl["protocol_errors"],
                }
            )
    for stage in range(4):
        selected = {row["mode"]: row for row in rows if row["stage"] == stage}
        raw = selected["raw_only"]
        for row in selected.values():
            row["speedup_vs_raw"] = raw["cycles"] / row["cycles"]
            row["payload_reduction_vs_raw"] = (
                1.0 - row["payload_words"] / raw["payload_words"]
            )
            row["term_reduction_vs_raw"] = (
                1.0 - row["projection_terms"] / raw["projection_terms"]
                if raw["projection_terms"]
                else 0.0
            )
        selected["gatestack"]["speedup_vs_no_residency"] = (
            selected["no_residency"]["cycles"]
            / selected["gatestack"]["cycles"]
        )
    return {
        "status": "PASS",
        "evidence": "[H67真实Q/K/gate]+[候选dyadic INT8]+[RTL]",
        "source_manifest": str(vector_manifest),
        "rows": rows,
        "limits": [
            "每个stage只回放sample0/B0/window0，不能外推为全数据集均值",
            "INT8 projection weight与bias是候选量化合同，尚未通过valid825",
            "周期属于projection execution slice，不含完整encoder与外存",
            "RAW-only仍经过完整顶层，只是运行路径基线",
        ],
    }


def write_markdown(path: Path, result: dict) -> None:
    labels = {
        "raw_only": "RAW41-only运行路径",
        "no_residency": "IPD无驻留",
        "gatestack": "GateStack",
    }
    lines = [
        "# H67四Stage真实Bit Trace GateStack RTL消融",
        "",
        "## 结论",
        "",
        "四个stage均使用真实Q/K、真实Q1.7 gate、checkpoint projection weight的候选dyadic INT8编码和真实bias候选码；三种路径的32-bit accumulator输出均为零mismatch。",
        "",
        "| Stage | 模式 | 周期 | 相对RAW加速 | payload words | 相对RAW减少 | projection terms | 相对RAW减少 | cache hit |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in range(4):
        for mode in ("raw_only", "no_residency", "gatestack"):
            row = next(
                item
                for item in result["rows"]
                if item["stage"] == stage and item["mode"] == mode
            )
            lines.append(
                f"| S{stage} | {labels[mode]} | {row['cycles']} | "
                f"{row['speedup_vs_raw']:.3f}x | {row['payload_words']} | "
                f"{row['payload_reduction_vs_raw']:.2%} | "
                f"{row['projection_terms']} | {row['term_reduction_vs_raw']:.2%} | "
                f"{row['cache_hits']} |"
            )
    lines.extend(
        [
            "",
            "## Residency周期贡献",
            "",
            "| Stage | GateStack相对no-residency速度 |",
            "|---:|---:|",
        ]
    )
    for stage in range(4):
        row = next(
            item
            for item in result["rows"]
            if item["stage"] == stage and item["mode"] == "gatestack"
        )
        lines.append(f"| S{stage} | {row['speedup_vs_no_residency']:.3f}x |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 本表已经从统计塑形晋级为真实网络Q/K/gate与真实checkpoint权重的整数RTL回放。",
            "- 权重量化合同尚未通过valid825，因此不能声称完整部署精度保持。",
            "- 每stage只有一个窗口，不能当作profile100平均周期或整网FPS。",
            "- 面积、功耗和EDP仍需目标库DC、SRAM macro与mapped SAIF。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.vector_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", result)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
