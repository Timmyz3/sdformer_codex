#!/usr/bin/env python3
"""Publish leftover per-stage table. Decision stays REJECT_WRITTEN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    stages = data["stages"]
    leftover = int(data["leftover_total"])
    almost = int(data["almost_4of5_total"])
    decision = data["decision"]
    if decision not in {"REJECT_WRITTEN", "PROMOTE_NEAR_IDENT_K"}:
        raise ValueError(f"unsupported leftover decision {decision!r}")
    score_tie = sum(int(s["equal_score_despite_diff_k"]) for s in stages)
    h1 = sum(int(s["min_hamming_1"]) for s in stages)
    if sum(int(s["leftover_qnz_not_identk"]) for s in stages) != leftover:
        raise ValueError("leftover_total does not match stage sum")
    if sum(int(s["almost_4of5_k_equal"]) for s in stages) != almost:
        raise ValueError("almost_4of5_total does not match stage sum")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "local5_leftover_stage_table_v1",
        "status": "PASS",
        "decision": decision,
        "leftover_total": leftover,
        "almost_4of5_total": almost,
        "equal_score_despite_diff_k_total": score_tie,
        "min_hamming_1_total": h1,
        "why_not_ident_score": (
            "Score ties among leftover dests are observed after the residual "
            "AXNOR walk. Detecting them requires the scores the leaf already "
            "computes, so an ident-score sidecar has no cycle contract."
        ),
        "stages": stages,
        "claim_boundary": [
            "Does not promote a third exact path.",
            "Window-local sample0 twelve-block statistic.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Local5 leftover 十二块分阶段表",
        "",
        f"> 决策继承输入审计：**{decision}**。本表不重写决策。",
        "",
        f"- leftover Q≠0 非 ident-K：**{leftover}**",
        f"- 4/5 K 相同：{almost}（{almost / leftover:.1%}）" if leftover else "- 4/5 K 相同：0",
        f"- leftover 内分数全相同（K 不全同）：**{score_tie}**"
        + (f"（{score_tie / leftover:.1%}）" if leftover else ""),
        f"- 最小 Hamming=1：{h1}",
        "",
        "ident-score 不是第三条 exact 路径：发现分数相同必须先走完残留 AXNOR。",
        "",
        "| block | leftover | 4/5 K | score-tie | ham=1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in stages:
        name = Path(str(stage["vector_dir"])).name.replace(
            "local5_", ""
        ).replace("_window_proj_20260813", "")
        lines.append(
            f"| {name} | {stage['leftover_qnz_not_identk']} | "
            f"{stage['almost_4of5_k_equal']} | "
            f"{stage['equal_score_despite_diff_k']} | "
            f"{stage['min_hamming_1']} |"
        )
    lines.extend(["", f"**Decision: {decision}**", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"PASS leftover table leftover={leftover} score_tie={score_tie} "
        f"decision={decision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
