#!/usr/bin/env python3
"""汇总 Local5 带 Tag T450 数值作业引擎的可审计证据。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qfit_local5_tagged_t450_job_engine_20260809"
SEEDS = (1, 17717, 44257, 48879)
PASS_RE = re.compile(
    r"PASS Local5 tagged T450 seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) jobs=(?P<jobs>\d+) "
    r"token=(?P<token>\d+) weight=(?P<weight>\d+) "
    r"result=(?P<result>\d+) done_stall=(?P<done_stall>\d+)"
)
ORACLE_RE = re.compile(
    r"seed=(?P<seed>\d+) out_dim=(?P<out_dim>\d+) "
    r"inputs=(?P<inputs>\d+) acc32=(?P<acc32>\d+) "
    r"terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)


def parse_run(path: Path) -> dict[str, int]:
    match = PASS_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"缺少 tagged T450 PASS 账本: {path}")
    row = {key: int(value) for key, value in match.groupdict().items()}
    expected = {"jobs": 2, "token": 900, "weight": 128, "result": 1800}
    for key, value in expected.items():
        if row[key] != value:
            raise RuntimeError(f"{path}: {key}={row[key]}，期望 {value}")
    if row["done_stall"] == 0:
        raise RuntimeError(f"{path}: job_done 反压覆盖为0")
    return row


def parse_oracle(path: Path) -> dict[str, int]:
    match = ORACLE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"缺少 Python oracle 账本: {path}")
    return {key: int(value) for key, value in match.groupdict().items()}


def yosys_cells(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    modules = payload.get("modules", {})
    top = modules.get("\\qfit_local5_tagged_t450_job_engine")
    if top is None:
        top = modules.get("qfit_local5_tagged_t450_job_engine")
    if top is None:
        raise RuntimeError(f"Yosys JSON 缺少顶层: {path}")
    return int(top["num_cells"])


def main() -> None:
    runs: dict[str, list[dict[str, int]]] = {"iverilog": [], "verilator_sva": []}
    for seed in SEEDS:
        iv = parse_run(OUT / f"main_seed_{seed}_iverilog.log")
        vl = parse_run(OUT / f"main_seed_{seed}_verilator_sva.log")
        if iv != vl:
            raise RuntimeError(f"seed={seed} 的 Icarus/Verilator 账本不一致")
        runs["iverilog"].append(iv)
        runs["verilator_sva"].append(vl)

    for mode in range(12):
        error_pass = f"PASS Local5 tagged T450 error_mode={mode} fail-closed"
        for simulator in ("iverilog", "verilator_sva"):
            name = f"error_mode_{mode}_{simulator}.log"
            if error_pass not in (OUT / name).read_text(encoding="utf-8"):
                raise RuntimeError(f"缺少 response 故障 PASS: {name}")

    status = []
    for line in (OUT / "status.tsv").read_text(encoding="utf-8").splitlines():
        item, result = line.split("\t")
        if result != "PASS":
            raise RuntimeError(f"签核项未通过: {line}")
        status.append({"item": item, "result": result})

    oracle = parse_oracle(OUT / "oracle_generation.log")
    report = {
        "evidence": "[rtl]",
        "scope": "单 input-head/单 output-tile 的带 Tag Local5 T450 数值作业；两次权重上下文",
        "oracle": oracle,
        "semantic_counts_per_two_jobs": {
            "jobs": 2,
            "token_requests_and_responses": 900,
            "weight_requests_and_responses": 128,
            "acc32_results": 1800,
        },
        "service_seeds": list(SEEDS),
        "runs": runs,
        "yosys_proxy": {
            "hierarchical_generic_cells": yosys_cells(OUT / "hier_stat.json"),
            "flat_generic_cells": yosys_cells(OUT / "flat_stat.json"),
            "not_asic_ppa": True,
        },
        "status": status,
        "response_fault_modes": {
            "token": ["tag", "head", "token_id", "explicit_error"],
            "weight": [
                "tag", "head", "output_tile", "lane", "out",
                "explicit_error", "duplicate_response",
            ],
            "unsolicited": ["token_response"],
        },
        "limitations": [
            "当前作业是单 input-head 对单 output-tile 的 partial projection，不含跨 input-head Acc32 归约。",
            "尚未连接 12-block frame scheduler、relation memo、bias/BN/residual 和最终 encoder 输出。",
            "服务延迟由 TB 生成，不是真实 SRAM macro 延迟；周期不是部署 FPS。",
            "Yosys generic cell 只证明综合可读，不是 DC/STA/SAIF PPA。",
        ],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = runs["iverilog"]
    md = [
        "# Local5 带 Tag T450 数值作业引擎 RTL 签核报告",
        "",
        "## 结论",
        "",
        "- `[rtl]` 两个连续 T450 作业完成 900 次 Q/K Token 服务、128 次权重服务和 1800 个 Acc32 ready/valid 结果，逐项匹配独立 Python oracle。",
        "- `[rtl]` 第二个作业加载与首作业符号相反的权重，全部 Acc32 结果也精确取反，证明 weight `load/use/release/reload` 上下文真实生效。",
        "- `[rtl]` 四种服务时序种子下 Icarus 与 Verilator/SVA 周期和账本完全一致，并实际覆盖 job_done 反压。",
        "- `[rtl]` 12 类 Token/weight response 身份、显式 error、unsolicited/duplicate 故障均被 fail-closed 捕获，错误后不再发请求或结果。",
        "- 本结果仍不是 12-block 全 encoder：它只闭合单 input-head/单 output-tile 的真实数值作业。",
        "",
        "## Python oracle",
        "",
        f"固定数据种子 `{oracle['seed']}` 生成 {oracle['inputs']} 个 T450 输入、{oracle['terms']} 个 source-major term、{oracle['updates']} 次 destination update 和 {oracle['acc32']} 个单上下文 Acc32 结果。",
        "",
        "## 跨模拟器回归",
        "",
        "| 服务种子 | Icarus 周期 | Verilator/SVA 周期 | Token | Weight | Acc32 result | done stall |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row, vrow in zip(rows, runs["verilator_sva"], strict=True):
        md.append(
            f"| {row['seed']} | {row['cycles']} | {vrow['cycles']} | "
            f"{row['token']} | {row['weight']} | {row['result']} | "
            f"{row['done_stall']} |"
        )
    md.extend(
        [
            "",
            "周期包含 TB 的 1--4 周期 Token/weight response latency 和随机 ready，不代表 SRAM macro、目标频率或整帧吞吐。",
            "",
            "故障矩阵覆盖 Token tag/head/id/error，weight tag/head/tile/lane/out/error，以及 unsolicited Token 和 duplicate weight response。",
            "",
            "## 开放综合代理",
            "",
            f"- `[rtl]` Yosys 层次 generic cells：{report['yosys_proxy']['hierarchical_generic_cells']}。",
            f"- `[rtl]` Yosys flatten generic cells：{report['yosys_proxy']['flat_generic_cells']}。",
            "- 上述值不进入 DATE ASIC PPA 主表。",
            "",
            "## 未闭合边界",
            "",
            "尚缺 scheduler 到数值引擎的真实连接、跨 input-head partial-sum 合并、relation memo 生命周期、bias/BN/residual、完整 12-block 数值回放，以及 DC/STA/SAIF。",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
