#!/usr/bin/env python3
"""Prune explicit paper-irrelevant H9 intermediates with an audit trail."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = ROOT / "neuron_autoresearch/cleanup_audits/paper_irrelevant_intermediates_20260805.json"
CHECKPOINT_RE = re.compile(r"checkpoint_epoch(\d+)\.pth$")

# Every retained epoch is either a valid825 rank-1, a final checkpoint, or a
# standard paper milestone. Current NB0/H67/Local5 runs are outside this list.
POLICY: dict[str, dict[str, object]] = {
    "nts11u_hw_h60_s23_scope_ffn_all_ternary_scope_full30_bs8_20260612_130819_setsid": {
        "keep": {19, 24, 29},
        "reason": "retain standard valid825 milestones and rank-1/final",
    },
    "nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_bs8_20260612_065413_setsid": {
        "keep": {19, 24, 29},
        "reason": "retain standard valid825 milestones and rank-1/final",
    },
    "nts11bd_u12_dsffn2_w720_fastlr_full30_20260613_212628_bs8_20260613_212628_setsid": {
        "keep": {19, 24, 26, 29},
        "reason": "retain standard valid825 milestones, recorded rank-1, and final",
    },
    "nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid": {
        "keep": {19, 24, 29},
        "reason": "retain standard valid825 milestones and rank-1/final",
    },
    "date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid": {
        "keep": {2, 4},
        "reason": "retain valid825 rank-1 ep2 and final ep4",
    },
    "date11full_all_binary_atlif_faps_s2only_nokmag_stdlr_ft_txep19_ft5_bs8_20260626_162831_setsid": {
        "keep": {2, 4},
        "reason": "retain valid825 rank-1 ep2 and final ep4",
    },
}

# These short-screen checkpoints have completed full-run successors and are not
# resume sources for the current paper mainlines. Their configs/logs stay intact.
DROP_ALL = {
    "date11allbin_faps_s2only_nokmag_stdlr_s360_bs8_20260626_160617":
        "superseded standard-LR short screen",
    "date11allbin_faps_s2only_nokmag_fastlr_s360_bs8_20260626_161714":
        "superseded fast-LR short screen",
    "nts11_phase5_short_20260613_064549": "superseded NTS11 phase-5 short screens",
    "nts11_phase5_short_20260613_053952": "superseded NTS11 phase-5 short screens",
    "nts11_hw_friendly_short_20260613_022912": "superseded NTS11 hardware-friendly short screens",
}


def free_bytes() -> int:
    stats = os.statvfs(ROOT)
    return int(stats.f_bavail * stats.f_frsize)


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

    retained: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    selected: set[Path] = set()

    for run_name, policy in POLICY.items():
        run_dir = RESULTS / run_name
        if not run_dir.is_dir():
            continue
        keep = policy["keep"]
        assert isinstance(keep, set)
        for path in sorted(run_dir.glob("checkpoint_epoch*.pth")):
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            item = describe(path, str(policy["reason"]))
            item["epoch"] = epoch
            if epoch in keep:
                retained.append(item)
            else:
                candidates.append(item)
                selected.add(path)

    for relative, reason in DROP_ALL.items():
        run_dir = RESULTS / relative
        if not run_dir.is_dir():
            continue
        for path in sorted(run_dir.rglob("checkpoint_epoch*.pth")):
            if path in selected or CHECKPOINT_RE.fullmatch(path.name) is None:
                continue
            candidates.append(describe(path, reason))
            selected.add(path)

    linked = [item for item in candidates if int(item["links"]) != 1]
    if linked:
        raise RuntimeError(f"refuse to unlink shared checkpoint inodes: {linked}")

    before = free_bytes()
    if args.execute:
        for item in candidates:
            path = ROOT / str(item["path"])
            path.unlink()
            item["exists_after"] = path.exists()
        for item in retained:
            path = ROOT / str(item["path"])
            item["exists_after"] = path.exists()
            if not path.is_file():
                raise RuntimeError(f"retained checkpoint disappeared: {path}")
    after = free_bytes()

    report = {
        "schema": "paper_irrelevant_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "protected_scope": (
            "all current NB0/H67/Local5 checkpoints and optimizer states; all configs, logs, "
            "metrics, valid825 profiles, rank-1 checkpoints, final checkpoints, and RTL artifacts"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(item["size_bytes"]) for item in candidates),
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
        "executed", "candidate_count", "candidate_bytes", "retained_count",
        "free_bytes_before", "free_bytes_after", "observed_free_bytes_delta",
    )}, indent=2))
    print(f"audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
