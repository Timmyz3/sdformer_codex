#!/usr/bin/env python3
"""Seal the topology-aware banked-Dynamic strong-baseline evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local5_banked_dynamic_baseline_20260814"
PHYS = ROOT / "results/local5_relation_scheduler_openproxy_20260814/report.json"
COMPOSITION = (
    ROOT / "results/local5_qsilent_rolling_composition_20260814/report.json"
)
RTL = ROOT / "rtl_qfit/qfit_banked_dynamic_retirement_scheduler.sv"
TB = ROOT / "tb_qfit/tb_qfit_banked_dynamic_retirement_miter.sv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pass(path: Path, simulator: str) -> dict[str, int | str]:
    text = path.read_text()
    match = re.search(
        r"PASS banked_dynamic_miter seed=(\d+) retire=(\d+) "
        r"stalls=(\d+) pending=(\d+)",
        text,
    )
    if not match or "fatal" in text.lower() or "error" in text.lower():
        raise ValueError(f"incomplete {simulator} miter log: {path}")
    return {
        "simulator": simulator,
        "seed": int(match.group(1)),
        "retired_sources": int(match.group(2)),
        "producer_stalls": int(match.group(3)),
        "max_pending": int(match.group(4)),
        "log": str(path.relative_to(ROOT)),
    }


def main() -> None:
    runs = []
    for seed in (1, 2, 3, 4, 5, 17, 29, 97):
        runs.append(parse_pass(OUT / f"miter_seed{seed}.log", "Icarus"))
    for seed in (1, 17, 97):
        runs.append(parse_pass(OUT / f"verilator_seed{seed}.log", "Verilator--assert"))
    epoch_gap = parse_pass(
        OUT / "epoch_gap_h8_seed99_verilator_assert.log",
        "Verilator--assert-epoch-gap",
    )

    physical = json.loads(PHYS.read_text())
    composition = json.loads(COMPOSITION.read_text())
    banked = physical["schedulers"]["banked_dynamic"]
    fcsr = physical["schedulers"]["fcsr"]
    delta = physical["fcsr_vs_banked_dynamic"]
    report = {
        "schema": "local5_banked_dynamic_strong_baseline_v1",
        "status": "ADMIT_AS_STRONG_BASELINE_NOT_CONTRIBUTION",
        "evidence": ["[rtl-miter]", "[开放逻辑映射代理]", "[开放网表STA代理]"],
        "contract": {
            "bank_map": "bank=(x+2*y) mod 5",
            "property": "the five Local5 cross-stencil candidates occupy distinct banks",
            "state": "5x9x(3-bit count + 3-bit entry generation) + 3x3-bit row generation = 279 bit",
            "retirement": "runtime consumer-count completion; not FCSR closed-form",
        },
        "miter": {
            "reference": "qfit_retirement_scheduler MODE_DYNAMIC",
            "run_count": len(runs),
            "runs": runs,
            "epoch_gap_run": epoch_gap,
            "result": "cycle-exact ready/valid/retire payload under source activity, bubbles, random backpressure, and one 25-cycle stall",
        },
        "production_tile": {
            "scope": composition["scope"],
            "cycles": composition["scheduler_ablation_under_qsilent"]
                ["t450_to_banked_dynamic"]["candidate_cycles"],
            "cycle_exact_to_dynamic_and_fcsr": composition["verification"]
                ["banked_dynamic_production_tile_exact"],
            "acc32_mismatch": composition["workload"]["acc32_mismatch"],
        },
        "open_proxy": {
            "scope": physical["scope"],
            "fcsr": fcsr,
            "banked_dynamic": banked,
            "banked_dynamic_over_fcsr": delta,
        },
        "source_sha256": {
            str(RTL.relative_to(ROOT)): sha256(RTL),
            str(TB.relative_to(ROOT)): sha256(TB),
        },
        "claim_boundary": [
            "this baseline strengthens the FCSR comparison and is not a proposed mechanism",
            "the physical numbers are open-library proxies, not DC, ASIC PPA, or signoff",
            "the miter is scheduler-level and does not replace end-to-end score-to-Acc32 evidence",
            "does not modify docs/359 frozen columns",
        ],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (OUT / "report.md").write_text(
        f"""# Local5 Banked-Dynamic 强基线

- 裁决：`{report['status']}`。
- 合同：`bank=(x+2y) mod 5`，五邻域候选落入五个不同 counter bank；三行循环状态用 epoch 复用。
- 状态：`279 bit` consumer count/generation；仍是运行时完成计数，不使用 FCSR 闭式退休。3-bit generation 避免三行环形状态在跨两次稀疏空洞后发生 1-bit epoch ABA 别名。
- miter：`{len(runs)}` 次通过，其中 8 个 Icarus seed、3 个 Verilator `--assert` seed；含 source active/inactive、输入气泡、随机反压和一次 25-cycle 连续阻塞。
- 稀疏代际回归：8-row、seed99 跨两次复用空洞，retire `{epoch_gap['retired_sources']}`，Verilator `--assert` PASS。
- 生产 tile：100 个 sample-disjoint group、真实 raw Q/K 与 checkpoint 权重、`OUT_DIM=2`，Banked-Dynamic `{report['production_tile']['cycles']}` cycles，与 flat Dynamic/FCSR 逐组相同，Acc32 mismatch `0`；不是 encoder。
- 开放代理：FCSR `{fcsr['area_proxy']:.3f}` / `{fcsr['arrival_ns']:.6f} ns`，Banked-Dynamic `{banked['area_proxy']:.3f}` / `{banked['arrival_ns']:.6f} ns`。
- Banked-Dynamic/FCSR：逻辑面积代理 `{delta['area_ratio']:.2f}x`，路径代理 `{delta['delay_ratio']:.2f}x`。

## 边界

该实现只用于把原 flat Dynamic 收紧成公平强基线，不是新贡献。映射结果是 Nangate45 开放代理，不是 DC、ASIC PPA 或 signoff；scheduler miter 不替代 score-to-Acc32 端到端验证。`docs/359` 不更新。
"""
    )


if __name__ == "__main__":
    main()
