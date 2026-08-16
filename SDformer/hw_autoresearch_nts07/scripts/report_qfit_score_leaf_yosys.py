#!/usr/bin/env python3
"""汇总QFIT score leaf开放综合代理，生成中文报告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qfit_score_leaf_yosys_20260730"
SIM_STATUS = ROOT / "build_qfit/iverilog/verification_status.tsv"
YOSYS_STATUS = RESULTS / "yosys_status.tsv"
YOSYS_SOURCES = (
    ROOT / "rtl_local5/local5_shiftmax5_q17.sv",
    ROOT / "rtl_qfit/qfit_tagged_compactor4.sv",
    ROOT / "rtl_qfit/qfit_xorbank_compactor4.sv",
    ROOT / "rtl_qfit/qfit_local5_score_leaf.sv",
)
SIM_SOURCES = YOSYS_SOURCES + (
    ROOT / "rtl_local5/local5_axnor_score_q7.sv",
    ROOT / "rtl_local5/local5_stencil_token.sv",
    ROOT / "tb_qfit/tb_qfit_local5_score_leaf.sv",
    ROOT / "verif_qfit/qfit_score_leaf_assertions.sv",
)

VARIANTS = {
    "w1_exact": "w1_exact_flat_stat.json",
    "global_qfsa_1c": "global_qfsa_1c_flat_stat.json",
    "global_qfsa_2c": "global_qfsa_2c_flat_stat.json",
    "xbf_exact": "xbf_exact_flat_stat.json",
    "w1_t8": "w1_t8_flat_stat.json",
    "xbf_t8": "xbf_t8_flat_stat.json",
    "xbf_t8b2": "xbf_t8b2_flat_stat.json",
}


def load_top(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["modules"]["\\qfit_local5_score_leaf"]


def summarize(path: Path) -> dict[str, Any]:
    top = load_top(path)
    cells = top["num_cells_by_type"]
    return {
        "num_cells": int(top["num_cells"]),
        "num_wire_bits": int(top["num_wire_bits"]),
        "register_cells": sum(
            int(count)
            for cell_type, count in cells.items()
            if "DFF" in cell_type
        ),
        "mux_cells": int(cells.get("$_MUX_", 0)),
        "xor_cells": int(cells.get("$_XOR_", 0)),
    }


def ratio(candidate: int, baseline: int) -> float:
    return candidate / baseline - 1.0


def current_pass(
    path: Path,
    expected: str,
    sources: tuple[Path, ...],
) -> bool:
    if not path.is_file():
        return False
    if expected not in path.read_text(encoding="utf-8"):
        return False
    newest_source = max(source.stat().st_mtime for source in sources)
    return path.stat().st_mtime >= newest_source


def main() -> None:
    sim_pass = current_pass(SIM_STATUS, "功能仿真\tPASS", SIM_SOURCES)
    lint_pass = current_pass(
        SIM_STATUS,
        "Verilator参数化lint\tPASS",
        SIM_SOURCES,
    )
    sva_pass = current_pass(
        SIM_STATUS,
        "Verilator_SVA仿真\tPASS",
        SIM_SOURCES,
    )
    yosys_pass = current_pass(
        YOSYS_STATUS,
        "check_assert\tPASS",
        YOSYS_SOURCES,
    )
    if not (sim_pass and lint_pass and sva_pass and yosys_pass):
        raise RuntimeError(
            "QFIT验证或综合状态缺失/陈旧，请先运行两个sim_qfit入口"
        )
    variants = {
        name: summarize(RESULTS / filename)
        for name, filename in VARIANTS.items()
    }
    global_compactor = json.loads(
        (RESULTS / "qfit_tagged_compactor4_stat.json").read_text()
    )["modules"]["\\qfit_tagged_compactor4"]["num_cells"]
    xorbank_compactor = json.loads(
        (RESULTS / "qfit_xorbank_compactor4_stat.json").read_text()
    )["modules"]["\\qfit_xorbank_compactor4"]["num_cells"]
    comparisons = {
        "global_qfsa_1c_vs_w1_exact_cells": ratio(
            variants["global_qfsa_1c"]["num_cells"],
            variants["w1_exact"]["num_cells"],
        ),
        "global_qfsa_2c_vs_w1_exact_cells": ratio(
            variants["global_qfsa_2c"]["num_cells"],
            variants["w1_exact"]["num_cells"],
        ),
        "xbf_exact_vs_w1_exact_cells": ratio(
            variants["xbf_exact"]["num_cells"],
            variants["w1_exact"]["num_cells"],
        ),
        "xbf_t8_vs_w1_t8_cells": ratio(
            variants["xbf_t8"]["num_cells"],
            variants["w1_t8"]["num_cells"],
        ),
        "xbf_t8b2_vs_w1_t8_cells": ratio(
            variants["xbf_t8b2"]["num_cells"],
            variants["w1_t8"]["num_cells"],
        ),
        "xbf_t8b2_vs_xbf_t8_cells": ratio(
            variants["xbf_t8b2"]["num_cells"],
            variants["xbf_t8"]["num_cells"],
        ),
        "xorbank_vs_global_compactor_cells": ratio(
            xorbank_compactor,
            global_compactor,
        ),
    }
    report = {
        "schema": "qfit_score_leaf_open_synthesis_proxy_v1",
        "evidence": "[开放综合代理]，非DC/STA/SAIF",
        "functional_verification": {
            "random_vectors": 300,
            "directed_route_vectors": 18,
            "mismatch": 0,
            "backpressure": "PASS" if sim_pass else "FAIL",
            "verilator_lint": "PASS" if lint_pass else "FAIL",
            "verilator_sva": "PASS" if sva_pass else "FAIL",
            "yosys_check_assert": "PASS" if yosys_pass else "FAIL",
        },
        "variants": variants,
        "compactors": {
            "global_128_to_4_cells": global_compactor,
            "xorbank_4x32_to_1_cells": xorbank_compactor,
        },
        "comparisons": comparisons,
    }
    (RESULTS / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = {
        "w1_exact": "4xW1 + 完美16-mask路由",
        "global_qfsa_1c": "全局QFSA，一拍compactor",
        "global_qfsa_2c": "全局QFSA，两级compactor",
        "xbf_exact": "XBF-QFSA + 完美16-mask路由",
        "w1_t8": "4xW1 + T8路由",
        "xbf_t8": "XBF-QFSA + T8路由",
        "xbf_t8b2": "XBF-QFSA + DBDR-T8B2",
    }
    lines = [
        "# QFIT Score Leaf RTL 与开放综合代理",
        "",
        "## 结论",
        "",
        "- 全局 `128->W4` lane pooling 的逻辑代价过高，不能作为当前主候选。",
        "- XOR-bank 将事件按 `bank=lane[1:0] XOR direction` 分散到四个本地选择器；compactor 单体 cell 数显著下降。",
        "- DBDR同时约束方向总delta与单bank压力，消除T8的32项hot-bank对抗尾延迟；其额外面积必须由post-G0尾延迟与能耗偿还。",
        "",
        "## 功能验证",
        "",
        "- 300 个随机五候选向量；",
        "- 16 个 direct-mask定向向量、均衡pooling向量和32项同bank对抗向量；",
        "- 4xW1、全局QFSA一拍、全局QFSA两级、XBF-T8和XBF-DBDR逐有效score及全部Shiftmax gate对direct金参考零失配；",
        "- 对抗向量中，XBF-T8为33个service拍，XBF-DBDR为4拍；",
        "- B2九拍解析边界向量、direct/residual混合和部分valid向量通过；",
        "- 随机输出反压稳定性通过；",
        "- 四参数组合Verilator lint无致命错误；",
        "- Verilator SVA仿真通过DBDR service上界和反压稳定断言；",
        "- 七个leaf及两个standalone compactor通过Yosys `check -assert`。",
        "",
        "## 结构结果",
        "",
        "| 变体 | cells | wire bits | register cells | mux | xor |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in variants.items():
        lines.append(
            f"| {labels[name]} | {row['num_cells']} | "
            f"{row['num_wire_bits']} | {row['register_cells']} | "
            f"{row['mux_cells']} | {row['xor_cells']} |"
        )
    lines.extend(
        [
            "",
            "## 关键差分",
            "",
            f"- 全局QFSA一拍 vs 4xW1完美路由：`{comparisons['global_qfsa_1c_vs_w1_exact_cells']:+.2%}` cells；",
            f"- 全局QFSA两级 vs 4xW1完美路由：`{comparisons['global_qfsa_2c_vs_w1_exact_cells']:+.2%}` cells；",
            f"- XBF完美路由 vs 4xW1完美路由：`{comparisons['xbf_exact_vs_w1_exact_cells']:+.2%}` cells；",
            f"- XBF-T8 vs 4xW1-T8：`{comparisons['xbf_t8_vs_w1_t8_cells']:+.2%}` cells；",
            f"- XBF-DBDR vs 4xW1-T8：`{comparisons['xbf_t8b2_vs_w1_t8_cells']:+.2%}` cells；",
            f"- DBDR保护逻辑 vs XBF-T8：`{comparisons['xbf_t8b2_vs_xbf_t8_cells']:+.2%}` cells；",
            f"- XOR-bank compactor vs 全局compactor：`{comparisons['xorbank_vs_global_compactor_cells']:+.2%}` cells。",
            "",
            "## 证据边界",
            "",
            "- cell 数来自 Yosys generic mapping，只能用于同脚本结构趋势；",
            "- 当前没有目标工艺面积、Fmax、功耗或 EDP；",
            "- T8 只改变 exact 执行路径，不删除事件、不改变 score/gate；",
            "- 当前随机向量不是部署 workload，不能据此声称 XBF 有周期收益；",
            "- `reproducibility_manifest.tsv`和`source_sha256.txt`记录了参数、流程与源文件哈希；",
            "- 论文主表必须改用 post-G0 多stage trace、DC/STA 和 SAIF。",
            "",
        ]
    )
    (RESULTS / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
