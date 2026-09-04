#!/usr/bin/env python3
"""Remove optimizer-state checkpoints from retired H66-H81 crop runs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = ROOT / (
    "neuron_autoresearch/cleanup_audits/"
    "retired_h66_h81_optimizer_states_20260805.json"
)

RUNS = (
    "h66c_allbinary_all12_tp_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid",
    "h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30_bs8_full30_20260712_setsid",
    "h66f_allbinary_all12_local5_tp_w720_fastlr_full30_bs8_full30_20260723_setsid",
    "h66g_allbinary_all12_local5_motion_w720_fastlr_full30_bs8_full30_20260723_setsid",
    "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid",
    "h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_bs8_full30_20260711_setsid",
    "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_bs8_full30_20260711_setsid",
    "h71_allbinary_all12_window_context_ttx_w720_fastlr_full30_bs8_full30_20260711_setsid",
    "h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30_bs8_full30_20260717_setsid",
)


def active_commands() -> list[str]:
    commands = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command:
            commands.append(command)
    return commands


def free_bytes() -> int:
    stat = os.statvfs(ROOT)
    return int(stat.f_bavail * stat.f_frsize)


def describe(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(stat.st_size),
        "blocks_bytes": int(stat.st_blocks * 512),
        "inode": int(stat.st_ino),
        "links": int(stat.st_nlink),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    commands = active_commands()
    candidates: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []

    for run in RUNS:
        directory = RESULTS / run
        ranking = directory / "profile_ranking_valid825.md"
        if not ranking.is_file():
            raise RuntimeError(f"missing valid825 ranking: {ranking}")
        ranking_text = ranking.read_text(encoding="utf-8")
        if "| 1 | 19 |" not in ranking_text:
            raise RuntimeError(f"rank-1 is no longer ep19: {ranking}")

        for epoch in (19, 29):
            model = directory / f"checkpoint_epoch{epoch}.pth"
            state = directory / f"checkpoint_epoch{epoch}_state_dict.pth"
            if not model.is_file():
                raise RuntimeError(f"missing retained model anchor: {model}")
            retained.append(describe(model))
            if state.is_file():
                absolute = str(state.resolve())
                relative = str(state.relative_to(ROOT))
                if any(absolute in command or relative in command for command in commands):
                    raise RuntimeError(f"candidate referenced by active process: {state}")
                candidates.append(describe(state))

    before = free_bytes()
    deleted: list[dict[str, object]] = []
    if args.execute:
        for record in candidates:
            path = ROOT / str(record["path"])
            path.unlink()
            deleted.append(record)
    after = free_bytes()

    report = {
        "schema": "retired_h66_h81_optimizer_state_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "policy": (
            "Retain rank-1 ep19 and final ep29 model checkpoints, rankings, profiles, "
            "configs, and logs. Remove only paired optimizer/scheduler/scaler states "
            "from retired crop-resolution runs with no active-process reference."
        ),
        "runs": list(RUNS),
        "candidate_count": len(candidates),
        "candidate_size_bytes": sum(int(item["size_bytes"]) for item in candidates),
        "deleted_count": len(deleted),
        "deleted_size_bytes": sum(int(item["size_bytes"]) for item in deleted),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "free_bytes_delta": after - before,
        "candidates": candidates,
        "deleted": deleted,
        "retained_model_anchors": retained,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "executed", "candidate_count", "candidate_size_bytes", "deleted_count",
        "deleted_size_bytes", "free_bytes_delta",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
