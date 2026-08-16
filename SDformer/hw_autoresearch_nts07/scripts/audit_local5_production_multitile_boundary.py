#!/usr/bin/env python3
"""Audit whether Local5 multi-tile term reuse creates a new execution object."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TERM_TAPE_WIDTH = 29
SOURCE_DESCRIPTOR_WIDTH = 92
TOKENS = 450
ACC_W = 32
OUT2 = 2
OUT32 = 32
MIN_STANDALONE_SPEEDUP = 1.05


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(source: dict, out32: dict, memo0: dict, memo25: dict) -> dict:
    if source.get("status") != "PASS_EXISTING_EXECUTION_OBJECT_BOUND_TO_PRODUCTION_RTL":
        raise ValueError("source-owned production report is not admitted")
    if out32.get("status") != "PASS":
        raise ValueError("OUT32 production report is not admitted")
    if any(
        report.get("decision")
        != "NO_GO_AS_STANDALONE_DATE_CONTRIBUTION_KEEP_AS_COMPLETENESS_EVIDENCE"
        for report in (memo0, memo25)
    ):
        raise ValueError("Relation Memo reports do not carry the sealed decision")

    totals = source["independent_reconstruction"]["totals"]
    groups = 100
    terms = totals["terms"]
    descriptors = totals["active"]
    term_tape_bits = terms * TERM_TAPE_WIDTH
    descriptor_tape_bits = descriptors * SOURCE_DESCRIPTOR_WIDTH
    if terms <= descriptors or descriptor_tape_bits <= 0:
        raise ValueError("term/descriptor population contract is inconsistent")

    out2_acc_bits = TOKENS * OUT2 * ACC_W
    out32_acc_bits = TOKENS * OUT32 * ACC_W
    if out32["physical_width"]["accumulator_payload_bits"] != out32_acc_bits:
        raise ValueError("OUT32 accumulator payload identity mismatch")
    spatial_context_ratio = out32_acc_bits / out2_acc_bits
    memo_speedups = [memo0["comparison"]["speedup"], memo25["comparison"]["speedup"]]
    if max(memo_speedups) >= MIN_STANDALONE_SPEEDUP:
        raise ValueError("Relation Memo unexpectedly passes the standalone speedup gate")

    return {
        "schema": "local5_production_multitile_boundary_audit_v1",
        "status": "NO_GO_AS_NEW_ARCHITECTURE_KEEP_EXISTING_COMPLETENESS_PATHS",
        "evidence": "[rtl-bound-profile]",
        "population": {
            "groups": groups,
            "source_owned_terms": terms,
            "active_source_descriptors": descriptors,
        },
        "materialization": {
            "expanded_term_tape": {
                "width_bits": TERM_TAPE_WIDTH,
                "population_bits": term_tape_bits,
                "mean_bits_per_group": term_tape_bits / groups,
            },
            "factorized_source_descriptor_tape": {
                "width_bits": SOURCE_DESCRIPTOR_WIDTH,
                "population_bits": descriptor_tape_bits,
                "mean_bits_per_group": descriptor_tape_bits / groups,
            },
            "expanded_over_factorized_ratio": term_tape_bits / descriptor_tape_bits,
            "decision": (
                "An expanded term tape is dominated by the existing factorized "
                "source descriptor/Relation Memo object. Refactorizing the term tape "
                "recovers that existing object."
            ),
        },
        "accumulator_context": {
            "out2_payload_bits": out2_acc_bits,
            "out32_payload_bits": out32_acc_bits,
            "context_ratio": spatial_context_ratio,
            "out32_rtl_cycle_invariant": out32["cycles"][
                "out2_out32_busy_cycle_invariant"
            ],
            "decision": (
                "Holding each term while visiting all output tiles requires all tile "
                "accumulator contexts; the existing packed OUT32 path already represents "
                "this spatial-width tradeoff."
            ),
        },
        "fixed_area_replay": {
            "sample0_speedup": memo_speedups[0],
            "sample25_speedup": memo_speedups[1],
            "standalone_gate": MIN_STANDALONE_SPEEDUP,
            "decision": (
                "With one accumulator tile, cross-tile reuse requires descriptor replay "
                "or recomputation. The existing real-window Relation Memo evidence fails "
                "the standalone performance gate."
            ),
        },
        "trilemma": [
            "spatial output contexts: existing packed OUT32, area scales with tile count",
            "fixed-area stored replay: existing factorized Relation Memo object",
            "fixed-area no replay state: recompute the production front per output tile",
        ],
        "claim_boundary": [
            "This is a structural go/no-go audit, not a new RTL speedup or energy result.",
            "The width accounting excludes control, SRAM granularity, and routing, which favors the rejected term tape.",
            "No result modifies docs/359 or creates a new Local5 contribution name.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-owned", type=Path, required=True)
    parser.add_argument("--out32", type=Path, required=True)
    parser.add_argument("--memo-sample0", type=Path, required=True)
    parser.add_argument("--memo-sample25", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "source_owned": args.source_owned,
        "out32": args.out32,
        "memo_sample0": args.memo_sample0,
        "memo_sample25": args.memo_sample25,
    }
    result = analyze(*(load(path) for path in inputs.values()))
    result["provenance"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in inputs.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    ratio = result["materialization"]["expanded_over_factorized_ratio"]
    memo = result["fixed_area_replay"]
    markdown = f"""# Local5 生产多 output-tile 执行边界审计

> 裁决：`{result['status']}`。证据：`[rtl-bound-profile]`。

## 三选一

1. term 保持、一次访问所有输出 tile：必须同时驻留全部输出 Acc 上下文；OUT32
   payload 是 OUT2 的 `{result['accumulator_context']['context_ratio']:.0f}x`，对应现有 packed OUT32 路径。
2. 固定 OUT2 Acc、先存后重放：29-bit expanded term tape 在 100-group 上为
   `{result['materialization']['expanded_term_tape']['population_bits']:,}` bit，反而是现有
   92-bit factorized source descriptor tape 的 `{ratio:.3f}x`；重新因子化后就是 Relation Memo。
3. 固定 OUT2 Acc 且不存 replay 对象：只能按输出 tile 重算生产前端。

## 强基线

真实多 tile Relation Memo 的两个窗口 speedup 为 `{memo['sample0_speedup']:.6f}x` 和
`{memo['sample25_speedup']:.6f}x`，未过 `{memo['standalone_gate']:.2f}x` 独立贡献门槛。

因此不新增 term-stationary、term-tape 或多 tile RTL 名称。现有 packed OUT32 和 Memo
分别保留为空间宽度与固定面积 replay 的完整度证据，不提高 Local5 创新分，也不修改
`docs/359`。
"""
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(
        "PASS Local5 multitile boundary NO-GO "
        f"expanded_over_factorized={ratio:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
