#!/usr/bin/env python3
"""用checkpoint-bound公平RTL行数据评估Motion class cache强基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "PASS" or report.get("schema") != "h67_rqtb_strong_baseline_v1":
        raise ValueError("输入不是PASS的H67 RQTB强基线报告")
    identity = report.get("input_identity")
    if not isinstance(identity, dict) or not identity.get("vector_sha256"):
        raise ValueError("输入报告未绑定checkpoint/vector身份")
    rows = report.get("rows_2s")
    if not isinstance(rows, list) or len(rows) != 138:
        raise ValueError("输入报告不是138行Fixed2S/RQTB2S结果")

    class_transactions = 0
    fixed_active_descriptors = 0
    rqtb_active_descriptors = 0
    fixed_cycles = 0
    rqtb_cycles = 0
    per_row: list[dict[str, int]] = []
    for row in rows:
        classes = int(row["fixed_exp"]) - int(row["active"])
        fixed_active = int(row["active"])
        rqtb_active = int(row["rqtb_exp"]) - classes
        if (
            classes <= 0
            or fixed_active < 0
            or rqtb_active < 0
            or rqtb_active > fixed_active
            or int(row["fixed_exp"]) != classes + fixed_active
            or int(row["rqtb_exp"]) != classes + rqtb_active
        ):
            raise ValueError(f"row={row.get('row')}的class/descriptor闭式不成立")
        class_transactions += classes
        fixed_active_descriptors += fixed_active
        rqtb_active_descriptors += rqtb_active
        fixed_cycles += int(row["fixed_cycles"])
        rqtb_cycles += int(row["rqtb_cycles"])
        per_row.append(
            {
                "row": int(row["row"]),
                "classes": classes,
                "fixed_active_descriptors": fixed_active,
                "rqtb_active_descriptors": rqtb_active,
            }
        )

    fixed_exp = class_transactions + fixed_active_descriptors
    rqtb_exp = class_transactions + rqtb_active_descriptors
    if (
        fixed_exp != int(report["work"]["fixed_exp"])
        or rqtb_exp != int(report["work"]["rqtb_exp"])
    ):
        raise ValueError("行级exp分账与报告总数不一致")

    max_score = 162
    exp_value_bits_min = 9
    exp_interface_bits = 16
    gate_bits = 9
    direct_entries = max_score + 1
    gate_cache_serialized_fixed = fixed_cycles + class_transactions
    gate_cache_serialized_rqtb = rqtb_cycles + class_transactions
    return {
        "schema": "h67_class_gate_cache_strong_baseline_v1",
        "status": "PASS_CACHE_REMOVES_EXP_ONLY_CLAIM_NOT_RQTB_CYCLE_GAIN",
        "evidence": "[rtl派生模型]",
        "scope": report["scope"],
        "identity": identity,
        "baseline_counts": {
            "rows": len(rows),
            "unique_class_transactions": class_transactions,
            "fixed_active_descriptors": fixed_active_descriptors,
            "rqtb_active_descriptors": rqtb_active_descriptors,
            "active_descriptor_reduction": (
                1.0 - rqtb_active_descriptors / fixed_active_descriptors
            ),
            "current_fixed_exp_evaluations": fixed_exp,
            "current_rqtb_exp_evaluations": rqtb_exp,
            "current_exp_reduction": 1.0 - rqtb_exp / fixed_exp,
        },
        "class_exp_cache": {
            "direct_index_entries": direct_entries,
            "value_bits_min": exp_value_bits_min,
            "rtl_interface_bits": exp_interface_bits,
            "compact_data_bits": direct_entries * exp_value_bits_min,
            "rtl_interface_data_bits": direct_entries * exp_interface_bits,
            "valid_bits": direct_entries,
            "compact_storage_bits": direct_entries * (exp_value_bits_min + 1),
            "rtl_interface_storage_bits": direct_entries * (exp_interface_bits + 1),
            "fixed_exp_lut_evaluations": class_transactions,
            "rqtb_exp_lut_evaluations": class_transactions,
            "rqtb_exp_lut_reduction_vs_fixed": 0.0,
            "remaining_descriptor_gate_quant": {
                "fixed": fixed_active_descriptors,
                "rqtb": rqtb_active_descriptors,
                "reduction": 1.0 - rqtb_active_descriptors / fixed_active_descriptors,
            },
        },
        "class_gate_cache": {
            "direct_index_entries": direct_entries,
            "data_bits": direct_entries * gate_bits,
            "valid_bits": direct_entries,
            "storage_bits": direct_entries * (gate_bits + 1),
            "fixed_gate_quant_evaluations": class_transactions,
            "rqtb_gate_quant_evaluations": class_transactions,
            "rqtb_gate_quant_reduction_vs_fixed": 0.0,
            "descriptor_cache_lookups": {
                "fixed": fixed_active_descriptors,
                "rqtb": rqtb_active_descriptors,
            },
            "serialized_second_class_pass_cycle_sensitivity": {
                "extra_cycles_both": class_transactions,
                "fixed_cycles": gate_cache_serialized_fixed,
                "rqtb_cycles": gate_cache_serialized_rqtb,
                "speedup": gate_cache_serialized_fixed / gate_cache_serialized_rqtb,
                "boundary": "只把第二次class scan串行相加；不是cache RTL实测",
            },
        },
        "decision": {
            "exp_activity_claim": "REJECT_AS_STANDALONE_NOVELTY",
            "gate_cache_rtl": "DO_NOT_IMPLEMENT_BEFORE_SAIF_OR_TIMING_BOTTLENECK",
            "rqtb_cycle_claim": "RETAINS_EXISTING_RTL_EVIDENCE",
            "reason": (
                "理想direct-index class cache模型是强通用基线，可消除Fixed/RQTB之间的exp/gate"
                "计算次数差；RQTB仍减少slot、active descriptor、FIFO压力和实测周期"
            ),
        },
        "boundaries": [
            "cache storage bit不是综合面积或SRAM宏面积。",
            "exp/gate evaluation数不是cycle、energy或SAIF功耗。",
            "serialized second-pass只作周期灵敏度，不是cache RTL。",
            "该模型不改变ep35公平RTL的1.1865x周期证据。",
        ],
        "rows": per_row,
    }


def render(report: dict[str, Any]) -> str:
    counts = report["baseline_counts"]
    exp_cache = report["class_exp_cache"]
    gate_cache = report["class_gate_cache"]
    serial = gate_cache["serialized_second_class_pass_cycle_sensitivity"]
    return "\n".join(
        [
            "# Motion Class-Exp/Gate Cache 强基线裁决",
            "",
            "> 证据：`[rtl派生模型]`；不是cache RTL、energy或PPA。",
            "",
            "## 结论",
            "",
            "- `exp transaction -22.22%`不能单独列为架构创新：163-entry direct-index class cache可使Fixed/RQTB的exp LUT求值都退化为唯一class数。",
            "- RQTB的公平RTL周期收益仍成立，因为它还减少slot、active descriptor和FIFO/issue压力。",
            "- 在SAIF证明exp/gate是能量瓶颈前，不实现class cache RTL。",
            "",
            "## 分账",
            "",
            "| 指标 | Fixed2S | RQTB2S |",
            "|---|---:|---:|",
            f"| 当前exp逻辑求值 | {counts['current_fixed_exp_evaluations']:,} | {counts['current_rqtb_exp_evaluations']:,} |",
            f"| active descriptor | {counts['fixed_active_descriptors']:,} | {counts['rqtb_active_descriptors']:,} |",
            f"| exp-cache后exp LUT求值 | {exp_cache['fixed_exp_lut_evaluations']:,} | {exp_cache['rqtb_exp_lut_evaluations']:,} |",
            f"| gate-cache后gate quant求值 | {gate_cache['fixed_gate_quant_evaluations']:,} | {gate_cache['rqtb_gate_quant_evaluations']:,} |",
            "",
            f"- 唯一class事务：{counts['unique_class_transactions']:,}。",
            f"- active descriptor减少：{counts['active_descriptor_reduction']:.3%}。",
            f"- exp cache精确紧凑下界为{exp_cache['compact_storage_bits']:,} bit，保持16-bit RTL接口时为{exp_cache['rtl_interface_storage_bits']:,} bit；gate cache为{gate_cache['storage_bits']:,} bit；均未综合。",
            f"- 若gate cache必须串行增加第二次class pass，灵敏度速度为{serial['speedup']:.4f}x；不是RTL结果。",
            "",
            "## 投稿边界",
            "",
        ]
        + [f"- {item}" for item in report["boundaries"]]
        + [""]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input_report.resolve()
    source = json.loads(input_path.read_text(encoding="utf-8"))
    report = analyze(source)
    report["input_report"] = str(input_path)
    report["input_report_sha256"] = sha256(input_path)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    snapshot = output / Path(__file__).name
    snapshot.write_bytes(Path(__file__).read_bytes())
    test_source = Path(__file__).with_name(
        "test_analyze_h67_class_gate_cache_baseline_v1.py"
    )
    test_snapshot = output / test_source.name
    test_snapshot.write_bytes(test_source.read_bytes())
    repo_root = Path(__file__).resolve().parents[1]
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(repo_root) + os.pathsep + test_env.get("PYTHONPATH", "")
    test_run = subprocess.run(
        [sys.executable, str(test_snapshot)],
        cwd=repo_root,
        env=test_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    test_receipt = output / "unit_tests.log"
    test_receipt.write_text(test_run.stdout, encoding="utf-8")
    if test_run.returncode != 0 or "Ran 3 tests" not in test_run.stdout:
        raise RuntimeError("class cache强基线单元测试未通过")
    report["source_sha256"] = sha256(snapshot)
    report["test_source_sha256"] = sha256(test_snapshot)
    report["test_receipt_sha256"] = sha256(test_receipt)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render(report), encoding="utf-8")
    write_json(
        output / "complete.json",
        {
            "schema": "h67_class_gate_cache_strong_baseline_complete_v1",
            "status": report["status"],
            "report_sha256": sha256(output / "report.json"),
            "markdown_sha256": sha256(output / "report.md"),
            "source_sha256": sha256(snapshot),
            "test_source_sha256": sha256(test_snapshot),
            "test_receipt_sha256": sha256(test_receipt),
            "unit_tests": "3/3 PASS",
        },
    )
    print(output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
