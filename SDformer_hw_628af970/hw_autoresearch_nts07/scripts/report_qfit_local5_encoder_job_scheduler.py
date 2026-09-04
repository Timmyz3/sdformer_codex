#!/usr/bin/env python3
"""汇总 Local5 12-block 分层调度器的可审计 RTL 证据。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qfit_local5_encoder_job_scheduler_20260809"
SEEDS = (1, 44257, 48879)
PASS_RE = re.compile(
    r"PASS Local5 encoder scheduler seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) groups=(?P<groups>\d+) "
    r"tiles=(?P<tiles>\d+) replays=(?P<replays>\d+) "
    r"decode=(?P<decode>\d+) release=(?P<release>\d+)"
)


def parse_run(path: Path) -> dict[str, int]:
    match = PASS_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"缺少整帧 PASS 账本: {path}")
    row = {key: int(value) for key, value in match.groupdict().items()}
    expected = {
        "groups": 1320,
        "tiles": 6720,
        "replays": 54000,
        "decode": 6720,
        "release": 6720,
    }
    for key, value in expected.items():
        if row[key] != value:
            raise RuntimeError(f"{path}: {key}={row[key]}，期望 {value}")
    return row


def yosys_cells(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    modules = payload.get("modules", {})
    top = modules.get("\\qfit_local5_encoder_job_scheduler")
    if top is None:
        top = modules.get("qfit_local5_encoder_job_scheduler")
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

    for name in ("error_iverilog.log", "error_verilator_sva.log"):
        text = (OUT / name).read_text(encoding="utf-8")
        if (
            "PASS Local5 encoder scheduler fail-closed wrong-tag and double-start"
            not in text
        ):
            raise RuntimeError(f"缺少 fail-closed PASS: {name}")

    status = []
    for line in (OUT / "status.tsv").read_text(encoding="utf-8").splitlines():
        item, result = line.split("\t")
        if result != "PASS":
            raise RuntimeError(f"签核项未通过: {line}")
        status.append({"item": item, "result": result})

    report = {
        "evidence": "[rtl]",
        "scope": "Local5 12-block 分层调度与作业协议；不含 token 级 SRAM 和完整数值数据通路",
        "semantic_counts": {
            "window_groups": 1320,
            "attention_decode_intent_jobs": 6720,
            "output_tile_requests": 6720,
            "input_head_job_replay_requests": 54000,
            "decode_intent_jobs": 6720,
            "cache_release_intent_jobs": 6720,
        },
        "seeds": list(SEEDS),
        "runs": runs,
        "yosys_proxy": {
            "hierarchical_generic_cells": yosys_cells(OUT / "hier_stat.json"),
            "flat_generic_cells": yosys_cells(OUT / "flat_stat.json"),
            "not_asic_ppa": True,
        },
        "status": status,
        "limitations": [
            "当前延迟和反压施加在 tile/head 作业接口，不代表真实 SRAM 周期。",
            "调度器尚未连接 T450 token 请求/响应、权重 reload、跨头归约和最终输出流。",
            "Yosys generic cell 仅证明可综合读取，不是 DC 面积、频率、功耗或 PPA。",
        ],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = runs["iverilog"]
    md = [
        "# Local5 12-block 分层调度器 RTL 签核报告",
        "",
        "## 结论",
        "",
        "- `[rtl]` 12-block 整帧调度账本闭合：1320 个窗口语义组、6720 个输出 tile request、54000 次 input-head job/replay request；另有 6720 个 decode intent。",
        "- `[rtl]` 首输出 tile 发出 decode/cache intent，末输出 tile 发出 release intent；Icarus 与 Verilator/SVA 在三组固定随机反压下逐项一致。",
        "- `[rtl]` 错误 head completion tag 与重复 start 被 fail-closed 捕获，错误后不再发出新 tile/head 作业。",
        "- 本结果只闭合控制与协议，不代表 Local5 全 encoder 数值数据通路已经闭环。",
        "",
        "## 分层工作量口径",
        "",
        "| 层次 | 每帧次数 | 硬件含义 |",
        "|---|---:|---|",
        "| 窗口语义组 | 1320 | `{stage, block, window}` |",
        "| attention decode intent | 6720 | 每组各 input head 的首访意图 |",
        "| 输出 tile request | 6720 | 每个组的 head 数决定输出 tile 数 |",
        "| input-head job/replay request | 54000 | 每个输出 tile 请求遍历全部输入 head |",
        "| decode intent 作业 | 6720 | 仅每个输入 head 的首输出 tile |",
        "| cache release intent 作业 | 6720 | 仅每个输入 head 的末输出 tile |",
        "",
        "## 整帧回归",
        "",
        "| 种子 | Icarus 周期 | Verilator/SVA 周期 | 账本 |",
        "|---:|---:|---:|---|",
    ]
    for row, vrow in zip(rows, runs["verilator_sva"], strict=True):
        md.append(
            f"| {row['seed']} | {row['cycles']} | {vrow['cycles']} | "
            "1320 / 6720 / 54000 / 6720 / 6720 |"
        )
    md.extend(
        [
            "",
            "周期只表示固定 TB 服务延迟与随机反压下的协议压力，不可作为部署吞吐。",
            "",
            "## 静态与综合可读检查",
            "",
            f"- `[rtl]` Yosys 层次 generic cells：{report['yosys_proxy']['hierarchical_generic_cells']}。",
            f"- `[rtl]` Yosys flatten 后 generic cells：{report['yosys_proxy']['flat_generic_cells']}。",
            "- 以上不是 ASIC PPA，不用于 DATE 主表面积、频率或功耗声明。",
            "",
            "## 已证明与未证明",
            "",
            "已证明：descriptor 顺序、stage/block/window 边界、输出 tile/输入 head 嵌套、tag 唯一性、decode/release intent、随机反压稳定，以及错误 tag/重复 start 后停止新发射。",
            "",
            "未证明：T450 token 级 SRAM 时序、真实权重切换、跨输入头 Acc32 归约、bias/BN/residual、整帧数值 bit-exact、目标工艺 PPA。",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
