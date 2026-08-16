#!/usr/bin/env python3
"""Prune non-anchor checkpoints from six retired full-model fine-tunes."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = ROOT / "neuron_autoresearch/cleanup_audits/retired_ft_intermediates_20260805.json"
CHECKPOINT_RE = re.compile(r"checkpoint_epoch(\d+)(_state_dict)?\.pth$")

# Every run has a completed standard-valid825 ranking with epoch 2 at rank 1.
# Keep that paper anchor and the final epoch, including both optimizer states.
POLICY = {
    "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid": {2, 7},
    "date11full_all_binary_atlif_h60_mu0_txonly_stdlr_ft_ep19_ft5_bs8_20260628_152115_setsid": {2, 4},
    "date11full_all_binary_atlif_faps_all12_nokmag_s64_z1_p6_sc0p015625_nosplit_stdlr_ft_txep19_ft5_bs8_20260627_234802_setsid": {2, 4},
    "date11full_all_binary_atlif_faps_all12_nokmag_s32_z1_p3_sc0p03125_nosplit_stdlr_ft_txep19_ft5_bs8_20260628_024839_setsid": {2, 4},
    "date11full_all_binary_atlif_drtx_b050_stdlr_ft_txep19_ft5_bs8_20260626_005642_setsid": {2, 4},
    "date11full_all_binary_atlif_original_stdlr_ft_ep29_ft5_bs8_20260620_015804_setsid": {2, 4},
}


def free_bytes() -> int:
    stats = os.statvfs(ROOT)
    return int(stats.f_bavail * stats.f_frsize)


def active_commands() -> list[str]:
    commands: list[str] = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command:
            commands.append(command)
    return commands


def describe(path: Path, reason: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(stat.st_size),
        "blocks_bytes": int(stat.st_blocks * 512),
        "inode": int(stat.st_ino),
        "links": int(stat.st_nlink),
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    running = active_commands()
    candidates: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []

    for run_name, keep_epochs in POLICY.items():
        run_dir = RESULTS / run_name
        ranking = run_dir / "profile_ranking_valid825.md"
        if not ranking.is_file():
            raise RuntimeError(f"missing standard-valid825 ranking: {ranking}")
        ranking_text = ranking.read_text(encoding="utf-8")
        if not re.search(r"\|\s*1\s*\|\s*2\s*\|", ranking_text):
            raise RuntimeError(f"epoch 2 is not rank 1: {ranking}")
        if any(str(run_dir) in command for command in running):
            raise RuntimeError(f"cleanup target is referenced by a running process: {run_dir}")

        files = sorted(run_dir.glob("checkpoint_epoch*.pth"))
        if not files:
            raise RuntimeError(f"no checkpoints found: {run_dir}")
        seen_keep: set[tuple[int, bool]] = set()
        for path in files:
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            is_state = match.group(2) is not None
            item = describe(path, "retired fine-tune intermediate; retain rank-1 and final anchors")
            item.update({"epoch": epoch, "training_state": is_state})
            if epoch in keep_epochs:
                retained.append(item)
                seen_keep.add((epoch, is_state))
            else:
                candidates.append(item)

        expected_keep = {(epoch, is_state) for epoch in keep_epochs for is_state in (False, True)}
        if seen_keep != expected_keep:
            raise RuntimeError(
                f"missing model/state anchor in {run_dir}: {sorted(expected_keep - seen_keep)}"
            )

    shared = [item for item in candidates if int(item["links"]) != 1]
    if shared:
        raise RuntimeError(f"refuse to unlink shared checkpoint inodes: {shared}")

    before = free_bytes()
    if args.execute:
        for item in candidates:
            (ROOT / str(item["path"])).unlink()
        for item in candidates:
            item["exists_after"] = (ROOT / str(item["path"])).exists()
            if item["exists_after"]:
                raise RuntimeError(f"checkpoint survived deletion: {item['path']}")
        for item in retained:
            item["exists_after"] = (ROOT / str(item["path"])).is_file()
            if not item["exists_after"]:
                raise RuntimeError(f"retained checkpoint disappeared: {item['path']}")
    after = free_bytes()

    report = {
        "schema": "retired_ft_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "protected_scope": (
            "all current Local5/H67/NB0 runs, every standard-valid825 rank-1 epoch, "
            "every final epoch, and paired optimizer/scheduler/scaler states for retained anchors"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(item["size_bytes"]) for item in candidates),
        "candidate_blocks_bytes": sum(int(item["blocks_bytes"]) for item in candidates),
        "retained_count": len(retained),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "deleted": candidates if args.execute else [],
        "candidates": [] if args.execute else candidates,
        "retained": retained,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "executed", "candidate_count", "candidate_bytes", "candidate_blocks_bytes",
        "retained_count", "free_bytes_before", "free_bytes_after", "observed_free_bytes_delta",
    )}, indent=2))
    print(f"audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
