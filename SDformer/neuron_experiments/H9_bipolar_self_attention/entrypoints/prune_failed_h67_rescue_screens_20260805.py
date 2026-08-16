#!/usr/bin/env python3
"""Remove two superseded H67 rescue-screen models with an audit trail."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = ROOT / "neuron_autoresearch/cleanup_audits/failed_h67_rescue_screens_20260805.json"
TARGETS = {
    RESULTS / "dsec_fullres_w15_rescue_screen_20260801/H67_nb0full_bb2e5/checkpoint_epoch0.pth": (
        "invalid architecture overlay (0 loaded, 210 missing); superseded rescue screen"
    ),
    RESULTS / "dsec_fullres_w15_rescue_screen_20260801/H67_crop_bb2e5/checkpoint_epoch0.pth": (
        "inferior low-LR rescue screen; superseded by selected bb1e4 H67 lineage"
    ),
}
PROTECTED = {
    RESULTS / "dsec_fullres_w15_rescue_screen_20260801/H67_crop_bb1e4/checkpoint_epoch0.pth",
    RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch19.pth",
    RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume15_20260803/checkpoint_epoch15.pth",
    RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30.pth",
}


def free_bytes() -> int:
    stat = os.statvfs(ROOT)
    return int(stat.f_bavail * stat.f_frsize)


def active_commands() -> list[str]:
    commands = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
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
    candidates = []
    for path, reason in TARGETS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if any(str(path) in command for command in running):
            raise RuntimeError(f"cleanup target is referenced by a running process: {path}")
        item = describe(path, reason)
        if item["links"] != 1:
            raise RuntimeError(f"refuse to unlink shared checkpoint inode: {item}")
        candidates.append(item)

    retained = []
    for path in PROTECTED:
        if not path.is_file():
            raise FileNotFoundError(path)
        retained.append(describe(path, "current H67 crop/fullres lineage anchor"))

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
                raise RuntimeError(f"protected checkpoint disappeared: {item['path']}")
    after = free_bytes()

    report = {
        "schema": "failed_h67_rescue_screen_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "protected_scope": (
            "selected H67 crop/fullres lineage, H67/NB0 equal+10, current Local-5, all logs, "
            "configs, valid825 profiles/rankings, and hardware evidence"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(item["size_bytes"]) for item in candidates),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "deleted": candidates if args.execute else [],
        "candidates": [] if args.execute else candidates,
        "retained": retained,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_keys = (
        "executed",
        "candidate_count",
        "candidate_bytes",
        "free_bytes_before",
        "free_bytes_after",
        "observed_free_bytes_delta",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    print(f"audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
