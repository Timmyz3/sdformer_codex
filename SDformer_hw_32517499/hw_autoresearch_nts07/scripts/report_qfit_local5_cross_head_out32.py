#!/usr/bin/env python3
"""汇总 Local5 三输入头 OUT32 数值集成 RTL 证据。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qfit_local5_cross_head_out32_20260809"
SEEDS = (17717, 44257, 48879)
RUN_RE = re.compile(
    r"PASS Local5 cross-head OUT32 seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) heads=(?P<heads>\d+) "
    r"partial=(?P<partial>\d+) final=(?P<final>\d+) "
    r"result_stall=(?P<result_stall>\d+) "
    r"group_stall=(?P<group_stall>\d+)"
)


def parse_run(path: Path) -> dict[str, int]:
    match = RUN_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"无法解析 {path}")
    return {key: int(value) for key, value in match.groupdict().items()}


def design_stat(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload["design"]
    return {
        "cells": int(stats.get("num_cells", 0)),
        "mem_v2_cells": int(stats.get("num_cells_by_type", {}).get("$mem_v2", 0)),
        "wires": int(stats.get("num_wires", 0)),
    }


def main() -> int:
    iverilog = [parse_run(OUT / f"main_seed_{seed}_iverilog.log") for seed in SEEDS]
    verilator = [
        parse_run(OUT / f"main_seed_{seed}_verilator_sva.log") for seed in SEEDS
    ]
    for lhs, rhs in zip(iverilog, verilator, strict=True):
        if lhs != rhs:
            raise RuntimeError(f"跨模拟器不一致: {lhs} != {rhs}")
        if lhs["heads"] != 3 or lhs["partial"] != 43200 or lhs["final"] != 14400:
            raise RuntimeError(f"业务账本错误: {lhs}")
        if lhs["result_stall"] == 0:
            raise RuntimeError(f"未覆盖 final result 反压: {lhs}")

    for simulator in ("iverilog", "verilator_sva"):
        for mode in (0, 1):
            text = (OUT / f"error_mode_{mode}_{simulator}.log").read_text(
                encoding="utf-8"
            )
            if f"PASS Local5 cross-head invalid head fail-closed mode={mode}" not in text:
                raise RuntimeError(f"故障模式未通过: {simulator}/{mode}")

    executor_stat = design_stat(OUT / "executor_flat_stat.json")
    shell_stat = design_stat(OUT / "shell_flat_stat.json")
    report = {
        "evidence": "[rtl]",
        "scope": "真实 scheduler 核驱动的单 output-tile、三 input-head、T450、OUT32 数值归约",
        "runs": {"iverilog": iverilog, "verilator_sva": verilator},
        "ledger": {
            "tiles": 1,
            "heads": 3,
            "token_requests": 1350,
            "weight_requests": 3072,
            "partial_acc32": 43200,
            "accumulator_writes": 43200,
            "final_acc32": 14400,
        },
        "fault_modes": [
            "wrong_head_job_tag_no_partial_write",
            "wrong_head_job_id_no_partial_write",
        ],
        "yosys_proxy": {
            "executor": executor_stat,
            "encoder_shell": shell_stat,
            "cross_head_accumulator_contract_bits": 450 * 32 * 32,
            "not_asic_ppa": True,
        },
        "limitations": [
            "数值回放覆盖一个 stage0 三头 output tile；12-block shell 仅完成结构、lint、SVA 和综合可读签核。",
            "尚未集成 relation memo，因此后续 output tile 仍会重算 relation，数值正确但不是目标复用性能。",
            "尚未包含跨头 bias、no-running BN、requant、residual 和 decoder。",
            "服务延迟来自 TB；Yosys generic 统计不是 DC/STA/SAIF PPA。",
        ],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = "\n".join(
        "| {seed} | {cycles} | {heads} | {partial} | {final} | {result_stall} | {group_stall} |".format(**run)
        for run in iverilog
    )
    markdown = f"""# Local5 三输入头 OUT32 跨头数值集成 RTL 报告

## 结论

- `[rtl]` 真实 output-tile scheduler 核已直接驱动三个 `head_job`，同一 T450 数值引擎按 head 时间复用。
- `[rtl]` 三个独立 Python oracle head 共完成 1350 次 Token、3072 次权重、43200 个 partial Acc32；共享 1RW 累加空间完成 43200 次 exactly-once 写入。
- `[rtl]` 最后一个 head 的 `head_done` 被 scheduler 接受后才开始排空 14400 个 OUT32 最终 Acc32；全部结果接受后才发 `tile_done`。
- `[rtl]` 三种随机服务/输出反压种子在 Icarus 与 Verilator/SVA 中周期和账本完全一致。
- `[rtl]` 错 head-job tag 与错 head id 均在任何 Token/weight/partial write 前 fail-closed。

## 跨模拟器回归

| 种子 | 周期 | head | partial | final | result stall | group stall |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

周期包含合成的 1--4 拍 Token/weight 服务延迟、final-result 反压和 1RW 跨头 RMW，不能解释为目标 SRAM、整帧吞吐或 FPS。

## 开放综合代理

| 范围 | flatten generic cells | `$mem_v2` cell |
|---|---:|---:|
| 跨头 tile executor | {executor_stat['cells']} | {executor_stat['mem_v2_cells']} |
| 12-block encoder numeric shell | {shell_stat['cells']} | {shell_stat['mem_v2_cells']} |

跨头 Acc32 的 RTL 存储合同为 `450 x 32 x 32 = 460800 bit`。Yosys 把它保留为 `$mem_v2`，没有映射到 SRAM macro。这些值只证明 hierarchy、控制和存储合同可被开放工具读取，不是 ASIC 面积、频率、功耗或 EDP。

## 未闭合边界

本报告把 Local5 的 scheduler、T450 score/Shiftmax5、relation transpose、source-major term、TCFM5 和跨头 Acc32 接成了同一生产宽度路径，但只数值回放一个三头 tile。12-block shell 尚未做全帧数值回放；relation memo、bias/BN/requant/residual、真实 SRAM latency 和 DC/STA/SAIF 均未闭合。
"""
    (OUT / "report.md").write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
