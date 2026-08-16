#!/usr/bin/env python3
"""Model Motion length-aware dual-workspace issue from sealed RQTB2S rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def pack_order(cycles: list[int]) -> int:
    total = 0
    for index in range(0, len(cycles), 2):
        chunk = cycles[index:index + 2]
        total += max(chunk)
    return total


def pack_lpt(cycles: list[int], width: int) -> int:
    bins = [0] * width
    for value in sorted(cycles, reverse=True):
        slot = bins.index(min(bins))
        bins[slot] += value
    return max(bins)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rqtb-report",
        type=Path,
        default=Path(
            "results/h67_rqtb_strong_baseline_2s_ep35_t450_v2_final_20260813/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/h67_laws_dual_workspace_model_20260813"),
    )
    args = parser.parse_args()

    report = json.loads(args.rqtb_report.read_text(encoding="utf-8"))
    rows = report["rows_2s"]
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["stage"]), int(row["block"]))].append(row)

    sequential = 0
    order2 = 0
    lpt2 = 0
    lpt3 = 0
    detail = []
    for key, members in sorted(groups.items()):
        cycles = [int(item["rqtb_cycles"]) for item in members]
        sequential += sum(cycles)
        order2 += pack_order(cycles)
        lpt2 += pack_lpt(cycles, 2)
        lpt3 += pack_lpt(cycles, 3)
        detail.append(
            {
                "stage": key[0],
                "block": key[1],
                "heads": len(cycles),
                "sequential": sum(cycles),
                "inorder_dual": pack_order(cycles),
                "lpt_dual": pack_lpt(cycles, 2),
            }
        )

    payload = {
        "schema": "h67_laws_dual_workspace_model_v1",
        "evidence": "[rtl校准模型]+[ep35-sample0-window0]",
        "scope": (
            "Independent heads inside one stage/block can occupy a second "
            "RQTB2S workspace. This is a scheduler model, not shared-backend RTL."
        ),
        "sequential_rqtb2s_cycles": sequential,
        "inorder_dual_cycles": order2,
        "lpt_dual_cycles": lpt2,
        "lpt_triple_cycles": lpt3,
        "inorder_dual_speedup": sequential / order2,
        "lpt_dual_speedup": sequential / lpt2,
        "area_normalized_if_duplicated_core": (sequential / order2) / 2.0,
        "groups": detail,
        "go_nogo": {
            "naive_dual_core": "NO-GO as DATE contribution if the whole engine is copied; ANT<1.0",
            "shared_backend_row_pipeline": (
                "KEEP as candidate: encode+histogram of row n+1 while emitting row n, "
                "sharing encoder/Shiftmax. Requires dual directory+K store RTL."
            ),
        },
        "claim_boundary": [
            "Uses already-measured per-row RQTB2S wall times as independent jobs.",
            "Does not prove overlap of encode and emit inside one engine.",
            "Not energy or ASIC PPA.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion LAWS dual-workspace model",
        "",
        "> 证据：`[rtl校准模型]`。不是新的 RTL 周期。",
        "",
        f"- 顺序 RQTB2S：{sequential} cycle",
        f"- 同块按原顺序双发射：{order2} cycle，{sequential/order2:.4f}x",
        f"- LPT 双工作区：{lpt2} cycle，{sequential/lpt2:.4f}x",
        f"- 若整核复制，面积归一吞吐约 {(sequential/order2)/2:.3f}",
        "",
        "结论：整核双复制不作为 DATE 贡献。保留共享 encoder/Shiftmax 的",
        "row-pipeline 双 directory/K-store 作为下一档 RTL 候选。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        "PASS Motion LAWS model "
        f"inorder={sequential/order2:.4f}x ANT={((sequential/order2)/2):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
