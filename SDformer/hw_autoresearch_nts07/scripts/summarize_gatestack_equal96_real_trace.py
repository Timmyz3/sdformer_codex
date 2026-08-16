#!/usr/bin/env python3
"""汇总Central96、3xIndependent32和DCTF96的同边界真实回放。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(rows: list[dict]) -> dict[int, dict]:
    return {int(row["stage"]): row for row in rows}


def destination_profile(record: dict) -> dict:
    vector_dir = Path(record["vector_dir"])
    counts = [
        int(line, 16)
        for line in (vector_dir / "term_destination_counts.memh")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    supertiles = int(record["logical_supertiles"])
    logical_terms = len(counts) * supertiles
    destinations = sum(counts) * supertiles
    event_beats = sum((count + 3) // 4 for count in counts) * supertiles
    return {
        "logical_terms": logical_terms,
        "destinations": destinations,
        "event_beats": event_beats,
        "destinations_per_term": (
            destinations / logical_terms if logical_terms else 0.0
        ),
        "term_one_destination": sum(count == 1 for count in counts) * supertiles,
        "term_ge8_destinations": sum(count >= 8 for count in counts) * supertiles,
        "max_destinations_per_term": max(counts, default=0),
    }


def build_report(
    dctf_path: Path,
    central_path: Path,
    independent_path: Path,
    vector_manifest_path: Path,
    mapping_path: Path | None = None,
) -> dict:
    dctf = indexed(load_json(dctf_path)["Icarus"])
    central = indexed(load_json(central_path)["Icarus"])
    independent = indexed(load_json(independent_path)["Icarus"])
    vector_manifest = load_json(vector_manifest_path)
    records = {int(row["stage"]): row for row in vector_manifest["records"]}

    rows = []
    for stage in range(4):
        profile = destination_profile(records[stage])
        dctf_cycles = int(dctf[stage]["cycles"])
        central_cycles = int(central[stage]["cycles"])
        independent_cycles = int(independent[stage]["cycles"])
        rows.append({
            "stage": stage,
            "heads": int(records[stage]["heads"]),
            **profile,
            "cycles": {
                "central96": central_cycles,
                "independent32x3": independent_cycles,
                "dctf96": dctf_cycles,
            },
            "dctf_speedup_vs_central96": central_cycles / dctf_cycles,
            "dctf_speedup_vs_independent32x3": independent_cycles / dctf_cycles,
        })

    totals = {
        name: sum(row["cycles"][name] for row in rows)
        for name in ("central96", "independent32x3", "dctf96")
    }
    total_terms = sum(row["logical_terms"] for row in rows)
    one_cycle_per_term_counterfactual = totals["dctf96"] - total_terms
    summary = {
        "cycles": totals,
        "dctf_speedup_vs_central96": totals["central96"] / totals["dctf96"],
        "dctf_speedup_vs_independent32x3": (
            totals["independent32x3"] / totals["dctf96"]
        ),
        "dctf_slowdown_vs_central96_pct": (
            totals["dctf96"] / totals["central96"] - 1.0
        ) * 100.0,
        "dctf_slowdown_vs_independent32x3_pct": (
            totals["dctf96"] / totals["independent32x3"] - 1.0
        ) * 100.0,
        "s3_fraction_of_dctf_cycles": rows[3]["cycles"]["dctf96"] / totals["dctf96"],
        "total_logical_terms": total_terms,
        "one_cycle_per_term_counterfactual_cycles": one_cycle_per_term_counterfactual,
        "counterfactual_speedup_vs_central96": (
            totals["central96"] / one_cycle_per_term_counterfactual
        ),
    }

    mapping = None
    if mapping_path is not None and mapping_path.is_file():
        mapping = load_json(mapping_path)
        area = {row["name"]: row["logic_area"] for row in mapping["rows"]}
        summary["open_logic_area"] = area
        summary["area_normalized_throughput_vs_central96"] = {
            name: (totals["central96"] * area["central96_term"]) /
                  (totals[name] * area[mapping_name])
            for name, mapping_name in (
                ("central96", "central96_term"),
                ("independent32x3", "independent32x3_term"),
                ("dctf96", "dctf96_term"),
            )
        }

    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "boundary": "H67 sample0/window0，term/event到projection final",
        "rows": rows,
        "summary": summary,
        "mapping": mapping,
        "decision": (
            "当前DCTF96不晋级吞吐主线；先实现双上下文原子验证前端并用同一向量复测"
        ),
        "limits": [
            "只覆盖sample0/window0，不代表100帧或全数据分布",
            "固定一拍行为存储模型，不是SRAM宏时序",
            "final全ready，未覆盖系统sink拥塞分布",
            "INT8仍是候选量化合同，不替代valid825部署精度冻结",
            "一拍/term反事实仅是优化上界模型，不是RTL实测",
            "开放逻辑面积不含$mem_v2、SDC、STA和SAIF，不能形成EDP结论",
        ],
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# 三种96-Lane架构真实四阶段公平对照",
        "",
        "本报告在同一H67 sample0/window0 term/event输入、固定一拍weight/bias响应和相同最终acc32语义下，对比Central96、3xIndependent32与DCTF96。",
        "",
        "## Stage分账",
        "",
        "| Stage | Heads | 逻辑term | destination | 平均dest/term | Central | Independent | DCTF | DCTF/Central加速 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        c = row["cycles"]
        lines.append(
            f"| S{row['stage']} | {row['heads']} | {row['logical_terms']} | "
            f"{row['destinations']} | {row['destinations_per_term']:.3f} | "
            f"{c['central96']} | {c['independent32x3']} | {c['dctf96']} | "
            f"{row['dctf_speedup_vs_central96']:.3f}x |"
        )
    totals = summary["cycles"]
    lines += [
        "",
        "## 总体结论",
        "",
        f"- Central96累计{totals['central96']}周期，3xIndependent32累计{totals['independent32x3']}周期，DCTF96累计{totals['dctf96']}周期；",
        f"- DCTF96相对Central96慢{summary['dctf_slowdown_vs_central96_pct']:.3f}%，相对Independent慢{summary['dctf_slowdown_vs_independent32x3_pct']:.3f}%；",
        f"- S3占DCTF总周期{summary['s3_fraction_of_dctf_cycles'] * 100:.3f}%，其高term数量使前端不可重叠气泡反转了S0/S2收益；",
        f"- 若只作为反事实上界扣除每个逻辑term一拍，共{summary['total_logical_terms']}拍，DCTF为{summary['one_cycle_per_term_counterfactual_cycles']}周期、相对Central为{summary['counterfactual_speedup_vs_central96']:.3f}x；该值不是RTL结果；",
        "- 因此当前DCTF96不满足吞吐晋级条件。下一版本必须用双上下文实现“当前term发射/下一term收集验证”重叠，并重新跑同一完整回归。",
        "",
    ]
    if "open_logic_area" in summary:
        norm = summary["area_normalized_throughput_vs_central96"]
        lines += [
            "## 开放逻辑面积归一吞吐",
            "",
            "以下只组合无约束Nangate45逻辑面积与RTL周期，不含SRAM、频率和功耗。Central96归一为1。",
            "",
            "| Central96 | 3xIndependent32 | DCTF96 |",
            "|---:|---:|---:|",
            f"| {norm['central96']:.3f} | {norm['independent32x3']:.3f} | {norm['dctf96']:.3f} |",
            "",
        ]
    lines += [
        "## 证据边界",
        "",
        *[f"- {item}；" for item in report["limits"]],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dctf", type=Path, required=True)
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.dctf, args.central, args.independent, args.vector_manifest,
        args.mapping,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
