#!/usr/bin/env python3
"""DATE dual-track table that refuses to mix cycle columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEALED = {
    "motion_fair_ep35": {"fixed": 112589, "rqtb": 94891, "label": "1.1865x"},
    "motion_fair_ep30": {"fixed": 111807, "rqtb": 94348, "label": "1.1850x held-out"},
    "local5_q0_only": {"residual": 324605, "fast": 191424, "label": "1.6957x sealed"},
}


FORBIDDEN = (
    "1.47x as main",
    "ANT 0.90 as chip area",
    "average 1.1865 with 1.1850",
    "21600-group RTL",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fair-merge-json",
        type=Path,
        default=Path("results/h67_fair_merge_population_20260813/report.json"),
    )
    parser.add_argument(
        "--local-overlap-json",
        type=Path,
        default=Path("results/local5_qsilent_overlap_ablation_20260813/report.json"),
    )
    parser.add_argument(
        "--local-all12-json",
        type=Path,
        default=Path("results/local5_sample0_all12_identk100_20260813/report.json"),
    )
    args = parser.parse_args()
    merge = json.loads(args.fair_merge_json.read_text(encoding="utf-8"))
    cycles = merge.get("cycles") or {}
    if int(cycles.get("fixed2s", 0)) != 112589 or int(cycles.get("rqtb2s", 0)) != 94891:
        raise SystemExit("fair merge json cycles are not the sealed 112589/94891")
    if int(merge["slots"]["fixed"]) != 62100 or int(merge["slots"]["rqtb"]) != 34099:
        raise SystemExit("fair merge json slots are not 62100/34099")
    if int(merge["equal"]) != 28001:
        raise SystemExit("fair merge json equal count is not 28001")
    if abs(merge["slot_reduction"] - 0.4509) > 5e-4:
        raise SystemExit("slot reduction drifted off 45.09%")

    local = json.loads(args.local_overlap_json.read_text(encoding="utf-8"))
    if local.get("status") != "PASS" or local.get("schema") != "local5_qsilent_overlap_ablation_v1":
        raise SystemExit("Local5 overlap ablation is not a PASS v1 artifact")
    configurations = local["configurations"]
    expected_local = {
        "residual": 324605,
        "q0_serial": 191424,
        "q0_ident_serial": 184632,
    }
    for key, expected in expected_local.items():
        actual = int(configurations[key]["total_cycles"])
        if actual != expected:
            raise SystemExit(f"Local5 {key} drifted: {actual} != {expected}")

    all12 = json.loads(args.local_all12_json.read_text(encoding="utf-8"))
    if all12.get("status") != "PASS":
        raise SystemExit("Local5 all-12 artifact is not PASS")
    if int(all12["sample0_total_residual"]) != 482520:
        raise SystemExit("Local5 all-12 residual total drifted")
    if int(all12["sample0_total_qsilent"]) != 272624:
        raise SystemExit("Local5 all-12 Q-silent total drifted")
    table = {
        "schema": "date_dual_table_v1",
        "motion": {
            "main": "Fixed2S->RQTB2S 112589/94891 = 1.1865x [rtl]",
            "heldout": "ep30 111807/94348 = 1.1850x [rtl]",
            "slot_mechanism": (
                f"fair-package slots {merge['slots']['rqtb']}/"
                f"{merge['slots']['fixed']} = -{merge['slot_reduction']:.2%}"
            ),
            "heldout_slot": "ep30 34052/62100 = -45.17% (do not average into -45.09%)",
            "not_main": [
                "skip vs RQTB 1.090x",
                "dense-only slot -43.61%",
                "active-only equality population",
                "shared-backend overlap side paths",
                "ANT 0.90",
                "real-weight Acc32 as a cycle column",
            ],
        },
        "local5": {
            "sealed_slice": "100-group Q==0-only TCFM5 1.6957x",
            "identk_slice": "100-group overlap-disabled cascade 1.7581x (does not replace 1.6957x)",
            "window": "sample0 all-12 482520/272624 = 1.7699x OUT_DIM=2 tile",
            "not_a_model": "1.8256x/1.9328x 21600 clone deleted",
        },
        "forbidden": list(FORBIDDEN),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# DATE 双线主表（拒混列）",
        "",
        "| 线 | 主列 | 禁止写进主列 |",
        "|---|---|---|",
        "| Motion | **1.1865×**（112589→94891） | 1.356× / 1.47× / ANT / 与 1.1850× 平均 |",
        "| Motion held-out | **1.1850×** | 替换主锚点 |",
        "| Motion 机制 | slot **−45.09%**（34099/62100） | 密行 −43.61% 冒充 45.09% |",
        "| Local5 切片 | **1.6957×** Q==0-only | 1.7581× 覆盖封存 |",
        "| Local5 窗 | sample0 十二块 **1.770×** | 21600-group RTL |",
        "",
        "脚本在公平包周期偏离 112589/94891 或 slot 偏离 45.09% 时失败。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("PASS DATE dual table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
