#!/usr/bin/env python3
"""Seal Motion encoder merge ledger. Does not touch 1.1865x."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUM_RE = re.compile(
    r"SB_SUM rows=(?P<rows>\d+) wall=(?P<wall>\d+) skip=(?P<skip>\d+)"
    r" tokens=(?P<tokens>\d+) pairs=(?P<pairs>\d+) empty_pairs=(?P<empty>\d+)"
    r" sequential=(?P<seq>\d+) slots=(?P<slots>\d+) equal=(?P<equal>\d+)"
)
SEAL_RE = re.compile(
    r"SB_SEAL row=(?P<row>\d+) pairs=(?P<pairs>\d+) slots=(?P<slots>\d+) "
    r"equal=(?P<equal>\d+)"
)
FAIR_RE = re.compile(
    r"FAIR_SUM rows=(?P<rows>\d+) skip=(?P<skip>\d+) "
    r"fixed=(?P<fixed>\d+) rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+)"
)


def parse_shared(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if "PASS tb_h67_laws_shared_backend_2s" not in text:
        raise ValueError(f"missing PASS {path}")
    summary = SUM_RE.search(text)
    if summary is None:
        raise ValueError(f"missing SB_SUM {path}")
    seals = [m.groupdict() for m in SEAL_RE.finditer(text)]
    pairs = int(summary["pairs"])
    slots = int(summary["slots"])
    equal = int(summary["equal"])
    if pairs and (slots + equal) != (pairs * 2):
        raise ValueError(
            f"slot identity broken slots+equal={slots + equal} != 2*pairs={pairs * 2}"
        )
    for item in seals:
        row_pairs = int(item["pairs"])
        row_slots = int(item["slots"])
        row_equal = int(item["equal"])
        if row_pairs and (row_slots + row_equal) != (row_pairs * 2):
            raise ValueError(f"row {item['row']} slot identity broken")
        if row_pairs and row_pairs != 225:
            raise ValueError(f"row {item['row']} pairs={row_pairs} != 225")
    naive = pairs * 2
    return {
        "log": str(path),
        "rows": int(summary["rows"]),
        "wall": int(summary["wall"]),
        "skip": int(summary["skip"]),
        "tokens": int(summary["tokens"]),
        "pairs": pairs,
        "empty_pairs": int(summary["empty"]),
        "slots": slots,
        "equal": equal,
        "naive_two_slot": naive,
        "slot_reduction": (1.0 - slots / naive) if naive else 0.0,
        "equal_frac": equal / pairs if pairs else 0.0,
        "empty_frac": int(summary["empty"]) / pairs if pairs else 0.0,
        "sealed_rows": len(seals),
    }


def parse_fair(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if "PASS tb_h67_laws_fair_lfsr_threeway_2s" not in text:
        raise ValueError(f"missing PASS {path}")
    match = FAIR_RE.search(text)
    if match is None:
        raise ValueError(f"missing FAIR_SUM {path}")
    fixed = int(match["fixed"])
    rqtb = int(match["rqtb"])
    shared = int(match["shared"])
    return {
        "log": str(path),
        "rows": int(match["rows"]),
        "skip": int(match["skip"]),
        "fixed": fixed,
        "rqtb": rqtb,
        "shared": shared,
        "fixed_over_rqtb": fixed / rqtb,
        "rqtb_over_shared": rqtb / shared,
        "fixed_over_shared": fixed / shared,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--ep35-log", type=Path, required=True)
    parser.add_argument("--ep30-fair-log", type=Path, required=True)
    args = parser.parse_args()
    ep35 = parse_shared(args.ep35_log)
    ep30 = parse_fair(args.ep30_fair_log)
    report = {
        "schema": "h67_encoder_merge_ledger_v2",
        "status": "PASS",
        "evidence": "[rtl]+[seal-latched-encoder]+[ep30-fair-heldout]",
        "ep35_shared_overlap": ep35,
        "ep30_fair_heldout": ep30,
        "claim_boundary": [
            "ep30 Fixed/RQTB is a held-out LFSR fair package, not averaged into 1.1865x.",
            "Shared+skip merge ledger is occupancy-gated (empty rows issue 0 pairs).",
            "slots+equal==2*pairs is the TESC identity, not a cycle claim.",
            "empty_pairs still enter directory; dropping them is illegal.",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion encoder merge ledger + ep30 fair held-out",
        "",
        "> 证据：`[rtl]`。不改 1.1865×，不把 overlap 1.356× 写进主表。",
        "",
        "## ep35 共享后端合并账本（seal 锁存，重叠安全）",
        "",
        f"- 138 行 wall **{ep35['wall']}**，skip {ep35['skip']}，token {ep35['tokens']}",
        f"- 密行发行 pair **{ep35['pairs']}**，空 pair {ep35['empty_pairs']} "
        f"({ep35['empty_frac']:.1%})",
        f"- equal-score 合并 **{ep35['equal']}** / {ep35['pairs']} "
        f"= {ep35['equal_frac']:.1%}",
        f"- slot {ep35['slots']} vs 朴素双 slot {ep35['naive_two_slot']} "
        f"= **{ep35['slot_reduction']:.2%} slot 减少**",
        "- 恒等式 `slots + equal == 2 * pairs` 在每一密行和总计成立",
        "",
        "## ep30 LFSR 公平三路（held-out，禁止并入 1.1865×）",
        "",
        f"- Fixed2S **{ep30['fixed']}** → RQTB2S **{ep30['rqtb']}** = "
        f"**{ep30['fixed_over_rqtb']:.4f}×**",
        f"- Shared+Skip **{ep30['shared']}**（skip {ep30['skip']}）",
        f"- vs RQTB {ep30['rqtb_over_shared']:.3f}×，vs Fixed "
        f"{ep30['fixed_over_shared']:.3f}×",
        "- 主锚点仍是 ep35 112589/94891 = 1.1865×",
        "",
    ]
    (args.result_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"PASS merge ledger slots={ep35['slots']} equal={ep35['equal']} "
        f"ep30={ep30['fixed_over_rqtb']:.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
