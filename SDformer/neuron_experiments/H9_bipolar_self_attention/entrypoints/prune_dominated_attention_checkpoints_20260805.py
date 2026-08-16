#!/usr/bin/env python3
"""Remove explicitly dominated retired attention checkpoints with an audit receipt."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = ROOT / (
    "neuron_autoresearch/cleanup_audits/"
    "dominated_attention_checkpoints_20260805.json"
)
TARGETS = {
    "h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30_bs8_full30_20260712_setsid": {
        "remove": 29,
        "keep": 19,
    },
    "h66f_allbinary_all12_local5_tp_w720_fastlr_full30_bs8_full30_20260723_setsid": {
        "remove": 29,
        "keep": 19,
    },
    "h71_allbinary_all12_window_context_ttx_w720_fastlr_full30_bs8_full30_20260711_setsid": {
        "remove": 29,
        "keep": 19,
    },
}


def free_bytes() -> int:
    stat = os.statvfs(ROOT)
    return int(stat.f_bavail * stat.f_frsize)


def rank1_epoch(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"cannot parse rank-1 epoch: {path}")


def active_cmdlines() -> str:
    rows = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            rows.append((process / "cmdline").read_bytes().replace(b"\0", b" ").decode())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    active = active_cmdlines()
    candidates = []
    retained = []
    for run_name, epochs in TARGETS.items():
        run = RESULTS / run_name
        ranking = run / "profile_ranking_valid825.md"
        keep = run / f"checkpoint_epoch{epochs['keep']}.pth"
        remove = run / f"checkpoint_epoch{epochs['remove']}.pth"
        if rank1_epoch(ranking) != epochs["keep"]:
            raise RuntimeError(f"rank-1 changed for {run_name}")
        if not keep.is_file():
            raise FileNotFoundError(keep)
        retained.append(str(keep.relative_to(ROOT)))
        if not remove.is_file():
            continue
        stat = remove.stat()
        if stat.st_nlink != 1:
            raise RuntimeError(f"refuse shared inode: {remove}")
        if str(remove.resolve()) in active:
            raise RuntimeError(f"refuse active checkpoint: {remove}")
        candidates.append(
            {
                "path": str(remove.relative_to(ROOT)),
                "size_bytes": int(stat.st_size),
                "blocks_bytes": int(stat.st_blocks * 512),
                "inode": int(stat.st_ino),
                "reason": (
                    f"ep{epochs['remove']} is dominated by retained rank-1 ep{epochs['keep']} "
                    "in AEE, AAE, and total_spikes"
                ),
            }
        )

    before = free_bytes()
    if args.execute:
        for row in candidates:
            (ROOT / row["path"]).unlink()
            row["exists_after"] = (ROOT / row["path"]).exists()
        for relative in retained:
            if not (ROOT / relative).is_file():
                raise RuntimeError(f"retained rank-1 disappeared: {relative}")
    after = free_bytes()
    report = {
        "schema": "dominated_attention_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "scope": "three retired crop-training attention candidates only",
        "protected_scope": (
            "NB0, NTS/TTX anchors, H67 lineage, Local5 source/current run, optimizer states, "
            "configs, logs, rankings, profiles, and hardware evidence"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(row["size_bytes"]) for row in candidates),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "retained_rank1": retained,
        "deleted": candidates if args.execute else [],
        "candidates": [] if args.execute else candidates,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "executed",
                    "candidate_count",
                    "candidate_bytes",
                    "free_bytes_before",
                    "free_bytes_after",
                    "observed_free_bytes_delta",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
