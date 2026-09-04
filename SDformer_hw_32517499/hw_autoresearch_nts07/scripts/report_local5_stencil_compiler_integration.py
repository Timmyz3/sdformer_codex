#!/usr/bin/env python3
"""Fail-closed report for generated Local5 retirement-rule integration."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local5_stencil_retirement_compiler_20260814"
BASELINE = (
    ROOT
    / "results/local5_qsilent_rolling_composition_20260814"
    / "rolling_q1_g100_verilator_assert.log"
)
RULE_CANDIDATE = OUT / "compiled_fcsr_q1_g100_verilator_assert.log"
RULE_COMPILE_LOG = OUT / "compiled_fcsr_q1_compile.log"
CONTRACT_CANDIDATE = OUT / "compiled_contract_q1_g100_verilator_assert.log"
CONTRACT_COMPILE_LOG = OUT / "compiled_contract_q1_compile.log"
MODE4_SKIP0_LOG = OUT / "mode4_skip0_contract_run.log"
MODE4_SKIP0_EXITCODE = OUT / "mode4_skip0_contract_exitcode.txt"
MANIFEST = (
    ROOT
    / "tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813/manifest.json"
)
BAD_RE = re.compile(
    r"%Error|\bERROR:|\bFATAL:|Assertion failed|MISMATCH|\$fatal|\bFAIL\b|"
    r"\$readmem file not found"
)
PASS_RE = re.compile(
    r"^PASS Local5 score-to-projection .* groups=100 total_cycles=(\d+)$"
)
DILATION_PASS_RE = re.compile(
    r"^PASS dilation_miter d=(?P<dilation>\d+) seed=(?P<seed>\d+) "
    r"retire=(?P<retire>\d+) stalls=(?P<stalls>\d+) pending=(?P<pending>\d+)$"
)


def parse_yosys(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    cells = re.findall(r"Number of cells:\s+(\d+)", text)
    areas = re.findall(r"Chip area for module.*?:\s+([0-9.]+)", text)
    if (
        not cells
        or not areas
        or "Found and reported 0 problems" not in text
        or "Area for cell type" in text
    ):
        raise ValueError(f"incomplete or memory-unknown Yosys log: {path}")
    return {"cells": int(cells[-1]), "area_proxy": float(areas[-1])}


def parse_sta(path: Path) -> dict[str, float | str]:
    text = path.read_text()
    arrivals = re.findall(r"^\s*([0-9.]+)\s+data arrival time$", text, re.M)
    slacks = re.findall(r"^\s*(-?[0-9.]+)\s+slack \((MET|VIOLATED)\)$", text, re.M)
    if not arrivals or not slacks or "Error:" in text:
        raise ValueError(f"incomplete STA log: {path}")
    worst_slack = min(slacks, key=lambda item: float(item[0]))
    return {
        "arrival_ns": max(float(value) for value in arrivals),
        "slack_ns": float(worst_slack[0]),
        "timing": worst_slack[1],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ledger(path: Path) -> tuple[list[str], int]:
    text = path.read_text()
    if BAD_RE.search(text):
        raise ValueError(f"bad marker in {path}")
    lines = [line for line in text.splitlines() if line.startswith("GROUP ")]
    if len(lines) != 100:
        raise ValueError(f"expected 100 GROUP rows in {path}, got {len(lines)}")
    passes = [
        int(match.group(1))
        for line in text.splitlines()
        if (match := PASS_RE.match(line))
    ]
    if passes != [155_791]:
        raise ValueError(f"unexpected PASS ledger in {path}: {passes}")
    return lines, passes[0]


def dilation_pass(path: Path) -> dict[str, int]:
    text = path.read_text()
    if BAD_RE.search(text):
        raise ValueError(f"bad marker in {path}")
    rows = [
        {key: int(value) for key, value in match.groupdict().items()}
        for line in text.splitlines()
        if (match := DILATION_PASS_RE.match(line))
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one dilation PASS in {path}, got {rows}")
    return rows[0]


def main() -> None:
    compiler = json.loads((OUT / "report.json").read_text())
    if compiler["status"] != "CONDITIONAL_AS_208_CONTRACT_EVIDENCE":
        raise ValueError("compiler screen status drift")
    for topology in ("cross_r1", "cross_r2", "asym5"):
        verification = compiler["topologies"][topology]["verification"]
        if verification != {
            **verification,
            "iverilog": "PASS",
            "verilator_assert": "PASS",
            "yosys_check": "PASS",
            "verilator_warning_count": 0,
        }:
            raise ValueError(f"generated RTL verification drift for {topology}")
        scheduler_verification = compiler["topologies"][topology][
            "scheduler_verification"
        ]
        if (
            scheduler_verification["ordered_retirement"] != "PASS"
            or scheduler_verification["sparse_candidate_filter"] != "PASS"
            or scheduler_verification["long_backpressure"] != "PASS"
            or scheduler_verification["yosys_check"] != "PASS"
            or scheduler_verification["verilator_warning_count"] != 0
        ):
            raise ValueError(f"generated scheduler verification drift for {topology}")

    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest.get("selection", {}).get("groups") != 100
        or manifest.get("shape", {}).get("out_dim") != 2
    ):
        raise ValueError("production vector identity drift")

    baseline_rows, baseline_cycles = ledger(BASELINE)
    rule_rows, rule_cycles = ledger(RULE_CANDIDATE)
    contract_rows, contract_cycles = ledger(CONTRACT_CANDIDATE)
    for label, candidate_rows in (
        ("generated-rule", rule_rows),
        ("generated-contract", contract_rows),
    ):
        if candidate_rows != baseline_rows:
            for index, (baseline, candidate) in enumerate(
                zip(baseline_rows, candidate_rows, strict=True)
            ):
                if baseline != candidate:
                    raise ValueError(
                        f"{label}/manual FCSR mismatch group {index}:\n"
                        f"manual={baseline}\ngenerated={candidate}"
                    )
            raise ValueError(f"{label}/manual FCSR ledger length drift")

    rule_compile_text = RULE_COMPILE_LOG.read_text()
    rule_verfiles = (
        ROOT
        / "build_qfit/local5_qsilent_rolling/compiled_fcsr_q1_obj/"
        "Vtb_qfit_local5_score_projection_postg0__verFiles.dat"
    ).read_text()
    if (
        BAD_RE.search(rule_compile_text)
        or "QFIT_USE_COMPILED_CROSS_R1" not in rule_verfiles
    ):
        raise ValueError("generated-rule FCSR build provenance mismatch")

    contract_compile_text = CONTRACT_COMPILE_LOG.read_text()
    default_lint = OUT / "default_build_compat_lint.log"
    default_lint_text = default_lint.read_text()
    contract_verfiles = (
        ROOT
        / "build_qfit/local5_qsilent_rolling/compiled_contract_q1_obj/"
        "Vtb_qfit_local5_score_projection_postg0__verFiles.dat"
    ).read_text()
    required_contract_markers = (
        "QFIT_ROLLING_SCHED_MODE=4",
        "qfit_compiled_retirement_shell.sv",
        "generated_cross_r1_retirement_scheduler.sv",
    )
    if BAD_RE.search(contract_compile_text) or BAD_RE.search(default_lint_text) or any(
        marker not in contract_verfiles for marker in required_contract_markers
    ):
        raise ValueError("generated-contract build provenance mismatch")
    mode4_skip0_text = MODE4_SKIP0_LOG.read_text()
    if (
        MODE4_SKIP0_EXITCODE.read_text().strip() != "1"
        or "compiled cross_r1 scheduler requires SKIP_ZERO_K=1"
        not in mode4_skip0_text
    ):
        raise ValueError("MODE4/SKIP_ZERO_K parameter guard did not fail closed")

    cross_r2_logs = sorted(OUT.glob("cross_r2_full_scheduler_seed*.log"))
    cross_r2_rows = [dilation_pass(path) for path in cross_r2_logs]
    if [row["seed"] for row in cross_r2_rows] != [1, 17, 33, 57, 99]:
        raise ValueError(f"cross_r2 seed population drift: {cross_r2_rows}")
    if any(row["dilation"] != 2 for row in cross_r2_rows):
        raise ValueError("cross_r2 dilation drift")
    if cross_r2_rows[-1]["retire"] != 4:
        raise ValueError("cross_r2 sparse-gap seed did not retain four sources")
    cross_r2_iverilog = OUT / "cross_r2_full_scheduler_iverilog_seed17.log"
    cross_r2_iverilog_row = dilation_pass(cross_r2_iverilog)
    if (
        cross_r2_iverilog_row["dilation"] != 2
        or cross_r2_iverilog_row["seed"] != 17
        or cross_r2_iverilog_row["retire"] != cross_r2_rows[1]["retire"]
    ):
        raise ValueError("cross_r2 Icarus/Verilator population drift")
    cross_r2_compile = OUT / "cross_r2_full_scheduler_compile.log"
    if "%Error" in cross_r2_compile.read_text():
        raise ValueError("cross_r2 Verilator compile failed")
    cross_r2_compiled_yosys = (
        OUT / "qfit_crossr2_compiled_scheduler_openproxy_yosys.log"
    )
    cross_r2_banked_yosys = (
        OUT / "qfit_crossr2_banked_dynamic_flop_scheduler_openproxy_yosys.log"
    )
    cross_r2_compiled_sta = (
        OUT / "qfit_crossr2_compiled_scheduler_openproxy_sta.log"
    )
    cross_r2_banked_sta = (
        OUT / "qfit_crossr2_banked_dynamic_flop_scheduler_openproxy_sta.log"
    )
    compiled_proxy = {
        **parse_yosys(cross_r2_compiled_yosys),
        **parse_sta(cross_r2_compiled_sta),
    }
    banked_proxy = {
        **parse_yosys(cross_r2_banked_yosys),
        **parse_sta(cross_r2_banked_sta),
    }
    if compiled_proxy["timing"] != "MET" or banked_proxy["timing"] != "VIOLATED":
        raise ValueError("unexpected cross_r2 3ns timing classification")

    report = {
        "schema": "local5_stencil_compiler_production_integration_v1",
        "status": "CONDITIONAL_AS_208_CONTRACT_EVIDENCE",
        "evidence": "[rtl] + [有限网格穷举回归]",
        "scope": (
            "100 sample-disjoint population-stage-weighted real raw-Q/K and "
            "checkpoint-weight groups; score/Shiftmax5 through relation/TCFM5 "
            "to Acc32; OUT_DIM=2 tile; not encoder"
        ),
        "compiler": {
            "topologies": ["cross_r1", "cross_r2", "asym5"],
            "exhaustive_destination_cases": compiler[
                "exhaustive_destination_cases"
            ],
            "finite_grid_mismatches": 0,
            "generated_rtl_dual_simulator": "PASS",
            "generated_rtl_yosys_check": "PASS",
            "generated_scheduler_dual_simulator": "PASS",
            "generated_scheduler_order_backpressure": "PASS",
        },
        "production_cross_r1": {
            "manual_fcsr_cycles": baseline_cycles,
            "generated_rule_cycles": rule_cycles,
            "generated_contract_cycles": contract_cycles,
            "groups": 100,
            "three_way_per_group_ledger_exact": True,
            "acc32_mismatch": 0,
            "verilator_assert": "PASS",
            "parameter_contract": "SCHED_MODE=4 requires SKIP_ZERO_K=1",
            "invalid_parameter_negative_test": "PASS_EXPECTED_FATAL",
            "default_mode_without_generated_module_lint": "PASS",
        },
        "synthetic_cross_r2_scheduler": {
            "scope": (
                "15x15, two planes, sparse active-source masks, random input "
                "gaps, random output backpressure, and one forced long stall"
            ),
            "compared": [
                "runtime flat Dynamic counters",
                "five-bank Dynamic counters",
                "compiled static rules plus generic pending shell",
            ],
            "verilator_assert_seeds": [row["seed"] for row in cross_r2_rows],
            "retired_sources_by_seed": [row["retire"] for row in cross_r2_rows],
            "cycle_exact_ready_valid_payload": True,
            "duplicate_or_inactive_retirements": 0,
            "icarus_seed17_population_match": True,
            "seed99_same_ring_entry_generation_reuse": True,
            "runtime_counter_state_bits_model": compiler["topologies"][
                "cross_r2"
            ]["dynamic_tracker_state_bits_15x15_model"],
            "open_logic_proxy": {
                "compiled_static": compiled_proxy,
                "banked_dynamic_flop_mapped": banked_proxy,
                "banked_to_compiled_area_ratio": (
                    banked_proxy["area_proxy"] / compiled_proxy["area_proxy"]
                ),
                "banked_to_compiled_delay_ratio": (
                    banked_proxy["arrival_ns"] / compiled_proxy["arrival_ns"]
                ),
            },
        },
        "architectural_reading": (
            "The fixed offset set and raster order compile into last-consumer "
            "rules, live row span, an affine bank coloring, and ordered event "
            "vectors consumed by a topology-independent scheduler shell. In the "
            "trained cross_r1 production tile, generated rules and the generic "
            "ordered pending/backpressure shell replace the handwritten FCSR "
            "scheduler while retaining the same relation storage, term builder, "
            "and TCFM5 backend."
        ),
        "claim_boundary": [
            "cross_r2 demonstrates schedule/compiler generality only; it is not a trained accuracy result",
            "asym5 is a synthetic non-cross contract test, not a trained accuracy or production-tile result",
            "cross_r2 reaches the complete scheduler pending/backpressure boundary, not relation SRAM or TCFM5",
            "production evidence is OUT_DIM=2 tile, not a 12-block encoder",
            "equal cycles are an equivalence result, not a new speedup",
            "no DC, signoff STA, SAIF, PTPX, SRAM macro energy, or ASIC PPA",
            "cross_r2 open mapping is scheduler-control-only and maps Dynamic counters to flops; it is not an SRAM/RF-matched PPA result",
            "does not modify docs/359 frozen columns",
        ],
        "sha256": {
            "manual_fcsr_log": sha256(BASELINE),
            "generated_rule_log": sha256(RULE_CANDIDATE),
            "generated_rule_compile_log": sha256(RULE_COMPILE_LOG),
            "generated_contract_log": sha256(CONTRACT_CANDIDATE),
            "generated_contract_compile_log": sha256(CONTRACT_COMPILE_LOG),
            "compiler_report": sha256(OUT / "report.json"),
            "vector_manifest": sha256(MANIFEST),
            "scheduler_rtl": sha256(ROOT / "rtl_qfit/qfit_retirement_scheduler.sv"),
            "relation_leaf_rtl": sha256(
                ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
            ),
            "compiled_scheduler_shell_rtl": sha256(
                ROOT / "rtl_qfit/qfit_compiled_retirement_shell.sv"
            ),
            "cross_r1_rules": sha256(
                OUT / "generated_cross_r1_retirement_rules.sv"
            ),
            "cross_r1_scheduler": sha256(
                OUT / "generated_cross_r1_retirement_scheduler.sv"
            ),
            "generated_contract_verfiles": sha256(
                ROOT
                / "build_qfit/local5_qsilent_rolling/compiled_contract_q1_obj/"
                "Vtb_qfit_local5_score_projection_postg0__verFiles.dat"
            ),
            "generated_contract_binary": sha256(
                ROOT
                / "build_qfit/local5_qsilent_rolling/compiled_contract_q1_obj/"
                "Vtb_qfit_local5_score_projection_postg0"
            ),
            "default_build_compat_lint": sha256(default_lint),
            "mode4_skip0_negative_log": sha256(MODE4_SKIP0_LOG),
            "mode4_skip0_exitcode": sha256(MODE4_SKIP0_EXITCODE),
            "banked_dynamic_rtl": sha256(
                ROOT / "rtl_qfit/qfit_banked_dynamic_retirement_scheduler.sv"
            ),
            "asym5_rules": sha256(OUT / "generated_asym5_retirement_rules.sv"),
            "asym5_scheduler": sha256(
                OUT / "generated_asym5_retirement_scheduler.sv"
            ),
            "asym5_iverilog_seed17": sha256(
                OUT / "asym5_scheduler_iverilog_seed17.log"
            ),
            "asym5_verilator_seed17": sha256(
                OUT / "asym5_scheduler_verilator_seed17.log"
            ),
            "cross_r2_verilator_compile": sha256(cross_r2_compile),
            "cross_r2_iverilog_seed17": sha256(cross_r2_iverilog),
            "cross_r2_compiled_yosys": sha256(cross_r2_compiled_yosys),
            "cross_r2_banked_flop_yosys": sha256(cross_r2_banked_yosys),
            "cross_r2_compiled_sta": sha256(cross_r2_compiled_sta),
            "cross_r2_banked_flop_sta": sha256(cross_r2_banked_sta),
        },
    }
    report["sha256"].update(
        {
            f"cross_r2_seed{row['seed']}": sha256(path)
            for path, row in zip(cross_r2_logs, cross_r2_rows, strict=True)
        }
    )
    (OUT / "integration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    markdown = f"""# Local5 固定拓扑编译器生产边界验证

- 裁决：`{report['status']}`，证据 `{report['evidence']}`。
- 规格到 RTL：`cross_r1/cross_r2/asym5` 共枚举 `{report['compiler']['exhaustive_destination_cases']}` 个 destination case，mismatch `0`；三套生成 rule RTL 与通用 ordered shell 的 Icarus/Verilator/Yosys 检查均 PASS。
- 生产边界：`cross_r1` 生成规则与通用 ordered pending/backpressure shell 已替换手写 FCSR scheduler；relation ring、term builder 与 TCFM5 保持不变。
- 真实百组：手写 FCSR、仅生成规则、完整 generated-contract 三路均为 `{baseline_cycles}` cycles，100/100 逐组完整 ledger 相同，Acc32 mismatch `0`，Verilator `--assert` PASS。
- 第二拓扑完整调度边界：`cross_r2` 在 `15x15` 双 plane 上，以 Dynamic、Banked-Dynamic、compiled-static 三方逐拍比较；5 个 Verilator seed（含稀疏 generation-gap seed99）和 Icarus seed17 均 PASS，ready/valid/retire payload 一致，无重复或 inactive retirement。
- `cross_r2` 开放逻辑代理：compiled-static `{compiled_proxy['cells']}` cells / `{compiled_proxy['area_proxy']:.3f}` / `{compiled_proxy['arrival_ns']:.6f} ns` / 3ns `{compiled_proxy['timing']}`；flop-mapped Banked-Dynamic `{banked_proxy['cells']}` / `{banked_proxy['area_proxy']:.3f}` / `{banked_proxy['arrival_ns']:.6f} ns` / `{banked_proxy['timing']}`。面积/路径倍率 `{banked_proxy['area_proxy'] / compiled_proxy['area_proxy']:.2f}x/{banked_proxy['arrival_ns'] / compiled_proxy['arrival_ns']:.2f}x`，仅是开放 scheduler-control 结构代理。

## 架构含义

可辩护的增量不是“又做了一个 FCSR”，而是把固定 source-relative offset 和 raster order 编译为：最后消费者规则、所需 live row span、搜索得到的 affine bank coloring，以及可直接由通用 ordered pending/backpressure shell 消费的事件向量。现行三事件 FCSR 是 `cross_r1` 的一个生成实例；`cross_r2` 改变 row span，`asym5` 改变邻域对称性、调度序列、active-rule 数和最大 burst，二者均不需要手写 role dispatch。该编译合同是 `docs/208` 数据流的控制与生命周期核心，不单列为第四项贡献。

## 边界

`cross_r2/asym5` 到达通用 scheduler 的 pending/backpressure 边界，但没有 relation SRAM、term builder、TCFM5 或训练网络结果。cross-r2 的 Dynamic counter 映射为 flop，尚未做同端口 SRAM/RF；上述面积和时序不是 DC/ASIC PPA。生产回放只有 `cross_r1`，且是 `SCHED_MODE=4 + SKIP_ZERO_K=1`、`OUT_DIM=2` tile、不是 encoder；等周期只证明 generated-contract 没有偷吞吐，不是新加速。没有 SAIF/PTPX、真实 SRAM 宏或 full encoder，且不改 `docs/359` 封存列。
"""
    (OUT / "integration_report.md").write_text(markdown)


if __name__ == "__main__":
    main()
