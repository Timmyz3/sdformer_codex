#!/usr/bin/env python3
"""用真实H67/H68行统计评估占用分数类扫描的周期收益。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ROWS_PER_STAGE = {"0": 2640, "1": 1440, "2": 2160, "3": 480}
N_TOKENS = 162
CONTROL_CYCLES = 3
FREQUENCY_HZ = 500_000_000

CASES = {
    "H67": {
        "classes": 35,
        "class_cycles": 2.0,
        "profile": REPO
        / "neuron_experiments/H9_bipolar_self_attention/results"
        / "h67_ep19_true_ttb_profile100_20260712/nts11_hardware_p0_profile.json",
    },
    "H68": {
        "classes": 3,
        "class_cycles": 1.0,
        "profile": REPO
        / "neuron_experiments/H9_bipolar_self_attention/results"
        / "h68_ep19_true_ttb_profile100_20260713/nts11_hardware_p0_profile.json",
    },
}


def row_cycles(active: float, class_cycles: float) -> float:
    return N_TOKENS + max(active, 1.0) + class_cycles + active + CONTROL_CYCLES


def evaluate(name: str, case: dict[str, object]) -> dict[str, object]:
    data = json.loads(Path(case["profile"]).read_text(encoding="utf-8"))
    stages = {item["group"]: item for item in data["summary"]["h60_by_stage"]}
    total_old = 0.0
    total_sparse = 0.0
    rows = 0
    details = []
    for stage, stage_rows in ROWS_PER_STAGE.items():
        item = stages[stage]
        active = float(item["zaf_active_entries_mean"])
        occupied = float(item["zaf_fold_classes_mean"])
        old = row_cycles(active, float(case["classes"]))
        # H67每类两拍以切断35路优先编码到乘加的长路径；H68仅3类，编译期特化为单拍。
        sparse = row_cycles(active, float(case["class_cycles"]) * occupied)
        total_old += stage_rows * old
        total_sparse += stage_rows * sparse
        rows += stage_rows
        details.append(
            {
                "阶段": stage,
                "行数": stage_rows,
                "活动项均值": active,
                "占用类均值": occupied,
                "固定扫描周期每行": old,
                "占用扫描周期每行": sparse,
                "周期下降": (old - sparse) / old,
            }
        )
    return {
        "设计": name,
        "行数每帧": rows,
        "固定扫描总周期每帧": total_old,
        "占用扫描总周期每帧": total_sparse,
        "周期下降": (total_old - total_sparse) / total_old,
        "固定扫描帧率500MHz": FREQUENCY_HZ / total_old,
        "占用扫描帧率500MHz": FREQUENCY_HZ / total_sparse,
        "阶段": details,
    }


def main() -> int:
    results = [evaluate(name, case) for name, case in CASES.items()]
    output_json = ROOT / "results/h67_h68_score_class_scan_cycle_model.json"
    output_md = output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"状态": "通过", "结果": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# H67/H68占用分数类扫描周期模型",
        "",
        "## 结果",
        "",
        "| 设计 | 固定扫描周期/帧 | 占用扫描周期/帧 | 周期下降 | 500MHz固定扫描帧率 | 500MHz占用扫描帧率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['设计']} | {item['固定扫描总周期每帧']:.0f} | "
            f"{item['占用扫描总周期每帧']:.0f} | {100*item['周期下降']:.2f}% | "
            f"{item['固定扫描帧率500MHz']:.2f} | {item['占用扫描帧率500MHz']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 分阶段明细",
            "",
            "| 设计 | 阶段 | 行数/帧 | 活动项/行 | 占用类/行 | 固定扫描周期/行 | 占用扫描周期/行 | 周期下降 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        for stage in item["阶段"]:
            lines.append(
                f"| {item['设计']} | {stage['阶段']} | {stage['行数']} | "
                f"{stage['活动项均值']:.2f} | {stage['占用类均值']:.2f} | "
                f"{stage['固定扫描周期每行']:.2f} | {stage['占用扫描周期每行']:.2f} | "
                f"{100*stage['周期下降']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## 口径与边界",
            "",
            "周期模型使用每帧6720行、每行162个token和profile100实测活动项/占用类均值。固定扫描指旧RTL逐项访问全部合法类；占用扫描指新RTL用位图只访问非空类。H67每类使用查找/读计数与指数乘加两拍流水，H68因仅有3类而在编译期特化为单拍。两者都计算完全相同的指数项和Shiftmax分母，不做近似。",
            "",
            "500MHz帧率只表示单个无外部停顿的注意力行核，不含Q/K投影、ATLIF、残差、SRAM同步读等待、数据搬运或decoder，不能写成SDformer端到端帧率。动态功耗收益仍需真实活动文件和目标工艺库确认。",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
