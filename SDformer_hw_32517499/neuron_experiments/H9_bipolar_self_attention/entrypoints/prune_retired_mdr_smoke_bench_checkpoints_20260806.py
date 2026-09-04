#!/usr/bin/env python3
"""Remove retired MDR smoke/throughput checkpoints while retaining measurements."""

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
    "retired_mdr_smoke_bench_checkpoints_20260806.json"
)
RETIRED_ROOTS = (
    "mdr_baseline_speed_bench_20260630_153001",
    "mdr_ttx_full_cupy_forkserver_20260630_161902/smoke_ckpts",
    "mdr_valid_resume_smoke_ckpts_20260625",
    "ttx_mdr_backend_bench_20260630_152443",
    "ttx_mdr_bs16_cupy_smoke1_ckpts",
    "ttx_mdr_forkserver_cupy_bench80_20260630_155032",
    "ttx_mdr_forkserver_cupy_bench80_codex_fixed_20260630_161513",
    "ttx_mdr_forkserver_preload_bench80_20260630_154148",
    "ttx_mdr_forkserver_preload_bench_20260630_153747",
    "ttx_mdr_full_cupy_from_ep47_20260630_162339/smoke_ckpts",
    "ttx_mdr_speed_bench_20260630_151531",
    "ttx_mdr_w6_cpuvoxel_bench80_resume_ep0_20260630_200423",
)
PROTECTED_ANCHORS = (
    "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30.pth",
    "dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728/checkpoint_epoch29.pth",
    "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth",
    "ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956/local_ckpts/checkpoint_epoch43.pth",
)


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


def describe(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(stat.st_size),
        "blocks_bytes": int(stat.st_blocks * 512),
        "inode": int(stat.st_ino),
        "links": int(stat.st_nlink),
        "reason": (
            "retired MDR loader/backend/smoke throughput artifact; logs and "
            "measurements retained; unrelated to the active DATE DSEC lineage"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    missing_anchors = [
        RESULTS / relative
        for relative in PROTECTED_ANCHORS
        if not (RESULTS / relative).is_file()
    ]
    if missing_anchors:
        raise FileNotFoundError(f"protected anchors missing: {missing_anchors}")

    active = active_cmdlines()
    candidates: list[dict[str, object]] = []
    retained_measurements: list[dict[str, object]] = []
    for relative in RETIRED_ROOTS:
        root = RESULTS / relative
        if not root.is_dir():
            raise FileNotFoundError(f"retired benchmark root missing: {root}")
        checkpoints = sorted(root.rglob("*.pth"))
        if not checkpoints:
            raise RuntimeError(f"no checkpoint candidates found in {root}")
        for checkpoint in checkpoints:
            if checkpoint.stat().st_nlink != 1:
                raise RuntimeError(f"refuse shared checkpoint inode: {checkpoint}")
            if str(checkpoint.resolve()) in active:
                raise RuntimeError(f"refuse active checkpoint: {checkpoint}")
            candidates.append(describe(checkpoint))
        retained = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix != ".pth"
        )
        retained_measurements.append(
            {
                "root": str(root.relative_to(ROOT)),
                "file_count": len(retained),
                "files": [str(path.relative_to(ROOT)) for path in retained],
            }
        )

    if len(candidates) != 30:
        raise RuntimeError(f"expected 30 checkpoint files, found {len(candidates)}")

    before = free_bytes()
    if args.execute:
        for row in candidates:
            path = ROOT / str(row["path"])
            path.unlink()
            row["exists_after"] = path.exists()
            if row["exists_after"]:
                raise RuntimeError(f"checkpoint survived deletion: {path}")
    after = free_bytes()

    report = {
        "schema": "retired_mdr_smoke_bench_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "scope": (
            "12 retired MDR smoke/loader/backend/throughput roots; delete only "
            "their 30 unique .pth artifacts"
        ),
        "protected_scope": (
            "all active DSEC NB0/TTX/H67/Local5 checkpoints and formal MDR ep43 "
            "continuation anchor; all configs, logs, timings, and reports"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(row["size_bytes"]) for row in candidates),
        "candidate_blocks_bytes": sum(
            int(row["blocks_bytes"]) for row in candidates
        ),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "deleted": candidates if args.execute else [],
        "candidates": [] if args.execute else candidates,
        "retained_measurements": retained_measurements,
        "protected_anchors": [
            str((RESULTS / relative).relative_to(ROOT)) for relative in PROTECTED_ANCHORS
        ],
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
                    "candidate_blocks_bytes",
                    "free_bytes_before",
                    "free_bytes_after",
                    "observed_free_bytes_delta",
                )
            },
            indent=2,
        )
    )
    print(f"audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
