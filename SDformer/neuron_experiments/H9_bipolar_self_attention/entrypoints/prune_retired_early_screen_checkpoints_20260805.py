#!/usr/bin/env python3
"""Remove three retired early-screen model files with an audit receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / (
    "neuron_autoresearch/cleanup_audits/"
    "retired_early_screen_checkpoints_20260805.json"
)
TARGETS = {
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "h56a_autopilot_20260528_123518/phase1_lambda_sweep/"
    "h56a_swp_l10_slowbb_tr05_steps360/checkpoint_epoch0.pth": (
        "retired H56A threshold-rate screen; later H56/TTX results supersede it"
    ),
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "ntx03_tx_refine_short_20260602_023426/runs/"
    "ntx03a_tx_m04_s005_steps360/checkpoint_epoch0.pth": (
        "retired NTX03 short TX-refinement screen; not a retained DATE anchor"
    ),
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "nsc08_h58_effective_short_20260603_132641/runs/"
    "nsc08g_h58_all_mu010_l03_lr1e5_steps360_steps360/checkpoint_epoch0.pth": (
        "retired NSC08 low-LR screen recorded as no improvement"
    ),
}


def free_bytes() -> int:
    stat = os.statvfs(ROOT)
    return int(stat.f_bavail * stat.f_frsize)


def active_cmdlines() -> str:
    rows: list[str] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            rows.append(
                (process / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return "\n".join(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    active = active_cmdlines()
    candidates = []
    for relative, reason in TARGETS.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        stat = path.stat()
        if stat.st_nlink != 1:
            raise RuntimeError(f"refuse shared inode: {path}")
        if str(path.resolve()) in active:
            raise RuntimeError(f"refuse active checkpoint: {path}")
        candidates.append(
            {
                "path": relative,
                "size_bytes": int(stat.st_size),
                "blocks_bytes": int(stat.st_blocks * 512),
                "inode": int(stat.st_ino),
                "sha256": sha256(path),
                "reason": reason,
            }
        )

    before = free_bytes()
    if args.execute:
        for row in candidates:
            path = ROOT / row["path"]
            path.unlink()
            row["exists_after"] = path.exists()
    after = free_bytes()
    report = {
        "schema": "retired_early_screen_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "scope": "three explicitly retired early-screen model files only",
        "protected_scope": (
            "NB0, NTX/NTS table anchors, TTX/BTTX, H67, Local5, full-resolution runs, "
            "configs, logs, metrics, rankings, profiles, and RTL evidence"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(row["size_bytes"]) for row in candidates),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "deleted": candidates if args.execute else [],
        "candidates": [] if args.execute else candidates,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
