#!/usr/bin/env python3
"""Reconcile Local5 optimized-front and integrated HxH full-width evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
TOKENS = 450
OUT_DIM = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    optimized_path: Path,
    stage0_path: Path,
    stage_smoke_paths: dict[int, Path],
) -> dict[str, Any]:
    optimized = load(optimized_path)
    if (
        optimized.get("schema") != "local5_out32_population_sensitivity_rtl_v1"
        or optimized.get("status") != "PASS"
        or optimized.get("correctness", {}).get("acc32_mismatch") != 0
        or optimized.get("correctness", {}).get("acc32_values_checked")
        != 100 * TOKENS * OUT_DIM
        or optimized.get("correctness", {}).get("icarus_verilator_per_group_exact")
        is not True
    ):
        raise ValueError("optimized OUT32 population receipt differs")

    stage0 = load(stage0_path)
    stage0_scalars = STAGE_HEADS[0] * TOKENS * OUT_DIM
    if (
        stage0.get("schema") != "local5_erep_integrated_cross_head_merge_v1"
        or stage0.get("status") != "PASS_INTEGRATED_CROSS_HEAD_CANARY_NOT_G0"
        or stage0.get("formal_g0") != "DENY"
        or stage0.get("scalar_count") != stage0_scalars
        or len(stage0.get("simulators", [])) != 2
        or any(
            run.get("scalar_count") != stage0_scalars
            or run.get("mismatch_count") != 0
            or run.get("max_abs_error") != 0
            for run in stage0["simulators"]
        )
    ):
        raise ValueError("stage0 integrated HxH receipt differs")

    integrated: dict[str, Any] = {
        "0": {
            "heads": STAGE_HEADS[0],
            "output_tiles": STAGE_HEADS[0],
            "final_acc32_checked": stage0_scalars,
            "mismatch": 0,
            "simulators": [run["simulator"] for run in stage0["simulators"]],
            "evidence": "[rtl]+[software-integer-golden]+[recompute-front]",
        }
    }
    total_scalars = stage0_scalars
    for stage in (1, 2, 3):
        path = stage_smoke_paths[stage]
        smoke = load(path)
        expected = STAGE_HEADS[stage] * TOKENS * OUT_DIM
        if (
            smoke.get("schema") != "local5_integrated_stage_smoke_v1"
            or smoke.get("status") != "PASS_SMOKE_NOT_G0"
            or smoke.get("formal_g0") != "DENY"
            or smoke.get("heads") != STAGE_HEADS[stage]
            or smoke.get("scalars") != expected
            or smoke.get("mismatch") != 0
            or smoke.get("max_abs_error") != 0
        ):
            raise ValueError(f"stage{stage} integrated HxH smoke receipt differs")
        integrated[str(stage)] = {
            "heads": STAGE_HEADS[stage],
            "output_tiles": STAGE_HEADS[stage],
            "final_acc32_checked": expected,
            "mismatch": 0,
            "simulators": ["verilator"],
            "evidence": "[rtl-smoke]+[software-integer-golden]+[recompute-front]",
        }
        total_scalars += expected

    return {
        "schema": "local5_fullwidth_evidence_matrix_v1",
        "status": "PASS_EVIDENCE_RECONCILED_NOT_UNIFIED_DUT",
        "optimized_front_population": {
            "groups": optimized["population"]["groups"],
            "stage_counts": optimized["population"]["stage_counts"],
            "output_tiles_per_group": 1,
            "out_dim": OUT_DIM,
            "acc32_checked": optimized["correctness"]["acc32_values_checked"],
            "mismatch": 0,
            "simulators": ["icarus", "verilator_assert"],
            "evidence": optimized["evidence"],
            "dataflow": "current Query-Silent plus T450/rolling packed score-to-Acc",
        },
        "integrated_hxh_recompute_canaries": {
            "per_stage": integrated,
            "stages": 4,
            "final_acc32_checked": total_scalars,
            "mismatch": 0,
            "dataflow": (
                "real checkpoint HxH scheduler/cross-head path with recompute front; "
                "one selected window per stage"
            ),
        },
        "unclosed_intersection": [
            "current Query-Silent plus rolling front is not yet the front end of the integrated HxH cross-head DUT",
            "optimized-front population checks one output tile per group and has no cross-head accumulation/final drain",
            "integrated HxH canaries use one selected real window per stage and are not population or full-encoder runs",
            "formal G0 remains DENY; these receipts are simulation and smoke evidence, not universal formal proof",
            "bias, no-running BN, requant, residual, decoder, DC, STA, SAIF, and PTPX remain outside both chains",
        ],
        "claim_boundary": (
            "The matrix demonstrates complementary full-width evidence. It explicitly "
            "does not combine the two chains into a single optimized HxH or encoder claim."
        ),
        "provenance": {
            str(path.resolve()): sha256(path)
            for path in (
                optimized_path,
                stage0_path,
                stage_smoke_paths[1],
                stage_smoke_paths[2],
                stage_smoke_paths[3],
            )
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    optimized = report["optimized_front_population"]
    integrated = report["integrated_hxh_recompute_canaries"]
    lines = [
        "# Local5 Full-Width Evidence Matrix",
        "",
        "## Complementary Chains",
        "",
        f"- Current optimized front: `{optimized['groups']}` OUT32 population groups, `{optimized['acc32_checked']:,}` Acc32, mismatch `0`, one output tile per group.",
        f"- Integrated HxH recompute canaries: all four stages, `{integrated['final_acc32_checked']:,}` final Acc32, mismatch `0`, one selected window per stage.",
        "",
        "| Stage | Heads/output tiles | Final Acc32 | Simulators |",
        "|---:|---:|---:|---|",
    ]
    for stage, row in integrated["per_stage"].items():
        lines.append(
            f"| {stage} | {row['heads']} | {row['final_acc32_checked']:,} | "
            f"{', '.join(row['simulators'])} |"
        )
    lines.extend(["", "## Unclosed Intersection", ""])
    lines.extend(f"- {item}" for item in report["unclosed_intersection"])
    lines.extend(["", report["claim_boundary"]])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.optimized,
        args.stage0,
        {1: args.stage1, 2: args.stage2, 3: args.stage3},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    write_markdown(args.output_dir / "report.md", report)
    print(
        "PASS Local5 full-width evidence matrix "
        f"optimized={report['optimized_front_population']['acc32_checked']} "
        f"integrated={report['integrated_hxh_recompute_canaries']['final_acc32_checked']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
