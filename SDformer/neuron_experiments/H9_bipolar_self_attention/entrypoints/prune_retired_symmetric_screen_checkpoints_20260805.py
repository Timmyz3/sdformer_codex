#!/usr/bin/env python3
"""Prune retired symmetric two-neuron screen weights while preserving rank-1s."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = REPO / "neuron_autoresearch/cleanup_audits/retired_symmetric_screens_20260805.json"
ROOTS = (
    RESULTS / "nts11_two_neuron_20260611_203636",
    RESULTS / "nts11_phase2_20260611_230130",
    RESULTS / "nts11bc_short_20260613_152906",
)
PROTECTED = {
    "nts11_two_neuron_20260611_203636/runs/nts11c_hw_h60_s23_two_neuron_fastlr_s1224_steps1224/checkpoint_epoch0.pth",
    "nts11_phase2_20260611_230130/runs/nts11j_hw_h60_s23_two_neuron_vanilla_decoder_s1224_steps1224/checkpoint_epoch0.pth",
}


def relative(path: Path) -> str:
    return str(path.relative_to(RESULTS))


def active_command_lines() -> str:
    lines = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            lines.append((entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    for root in ROOTS:
        if not (root / "summary.md").is_file() or not (root / "summary.csv").is_file():
            raise RuntimeError(f"summary evidence missing: {root}")

    candidates = sorted(path for root in ROOTS for path in root.rglob("checkpoint_epoch*.pth"))
    preserved = [path for path in candidates if relative(path) in PROTECTED]
    deleted = [path for path in candidates if relative(path) not in PROTECTED]
    if {relative(path) for path in preserved} != PROTECTED:
        raise RuntimeError("protected rank-1 checkpoint set is incomplete")

    active = active_command_lines()
    referenced = [str(path) for path in deleted if str(path.resolve()) in active]
    if referenced:
        raise RuntimeError(f"candidate checkpoint is active: {referenced}")

    entries = [
        {
            "path": str(path.resolve()),
            "relative_to_results": relative(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in deleted
    ]
    report = {
        "schema": "retired_symmetric_screen_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PLANNED" if not args.execute else "EXECUTING",
        "reason": (
            "Retired symmetric two-neuron valid10 screens failed accuracy gates and are unrelated "
            "to the one-sided binary Local-5/H67/NB0 final chain. Preserve each multi-candidate "
            "screen rank-1 plus every config, log, summary and profile."
        ),
        "protected_rank1_checkpoints": sorted(PROTECTED),
        "deleted_files": entries,
        "deleted_count": len(entries),
        "reclaimed_bytes": sum(item["size_bytes"] for item in entries),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0

    for path in deleted:
        path.unlink()
    report["status"] = "COMPLETE"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["postconditions"] = {
        "all_candidates_removed": all(not path.exists() for path in deleted),
        "all_rank1_checkpoints_preserved": all(path.is_file() for path in preserved),
        "all_summaries_preserved": all((root / "summary.md").is_file() for root in ROOTS),
    }
    if not all(report["postconditions"].values()):
        raise RuntimeError(f"cleanup postcondition failed: {report['postconditions']}")
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "deleted_count": len(entries), "reclaimed_bytes": report["reclaimed_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
