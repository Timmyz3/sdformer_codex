#!/usr/bin/env python3
"""Prune old optimizer states and explicit retired non-rank model checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = REPO / (
    "neuron_autoresearch/cleanup_audits/"
    "retired_june_states_and_nonrank_models_20260806.json"
)

# All seven runs are concluded crop-resolution algorithm candidates. Their
# standard valid825 ranking names ep19 as rank-1; ep19 and all scalar/profile
# evidence remain protected.
NONRANK_EP29_RUNS = (
    "h66c_allbinary_all12_tp_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid",
    "h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30_bs8_full30_20260712_setsid",
    "h66g_allbinary_all12_local5_motion_w720_fastlr_full30_bs8_full30_20260723_setsid",
    "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid",
    "h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_bs8_full30_20260711_setsid",
    "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_bs8_full30_20260711_setsid",
    "h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30_bs8_full30_20260717_setsid",
)


def free_bytes() -> int:
    value = os.statvfs(REPO)
    return int(value.f_bavail * value.f_frsize)


def describe(path: Path, reason: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(REPO)),
        "size_bytes": int(stat.st_size),
        "blocks_bytes": int(stat.st_blocks * 512),
        "inode": int(stat.st_ino),
        "links": int(stat.st_nlink),
        "reason": reason,
    }


def active_command_lines() -> str:
    values: list[str] = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            values.append(path.read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return "\n".join(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    candidates: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []

    # Directory names carrying 202606 are concluded June experiments. Remove
    # only training-resume state, never a model checkpoint or evaluation file.
    for path in sorted(RESULTS.rglob("checkpoint_epoch*_state_dict.pth")):
        relative = str(path.relative_to(RESULTS))
        if "202606" not in relative or "mdr" in relative.lower():
            continue
        candidates.append(
            describe(path, "concluded June optimizer/scheduler/scaler resume state")
        )

    for run_name in NONRANK_EP29_RUNS:
        run = RESULTS / run_name
        ranking = run / "profile_ranking_valid825.md"
        rank1 = run / "checkpoint_epoch19.pth"
        nonrank = run / "checkpoint_epoch29.pth"
        if not ranking.is_file() or "| 1 | 19 |" not in ranking.read_text(encoding="utf-8"):
            raise RuntimeError(f"ep19 rank-1 contract missing: {ranking}")
        if not rank1.is_file():
            raise FileNotFoundError(rank1)
        retained.append(describe(rank1, "standard-valid825 rank-1 ep19 protected"))
        if nonrank.is_file():
            candidates.append(
                describe(nonrank, "retired crop candidate ep29 ranked below protected ep19")
            )

    duplicate_paths = {
        item["path"]
        for item in candidates
        if sum(other["path"] == item["path"] for other in candidates) != 1
    }
    if duplicate_paths:
        raise RuntimeError(f"duplicate cleanup candidates: {sorted(duplicate_paths)}")
    linked = [item["path"] for item in candidates if int(item["links"]) != 1]
    if linked:
        raise RuntimeError(f"refuse linked cleanup candidates: {linked}")

    commands = active_command_lines()
    active = [item["path"] for item in candidates if str(item["path"]) in commands]
    if active:
        raise RuntimeError(f"refuse active-process cleanup candidates: {active}")

    before = free_bytes()
    if args.execute:
        for item in candidates:
            path = REPO / str(item["path"])
            path.unlink()
            item["exists_after"] = path.exists()
        for item in retained:
            path = REPO / str(item["path"])
            if not path.is_file():
                raise RuntimeError(f"protected rank-1 disappeared: {path}")
            item["exists_after"] = True
    after = free_bytes()

    report = {
        "schema": "retired_june_states_and_nonrank_models_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "protected_scope": (
            "all NB0, H67, Local5, MDR, current queues, rank-1 model checkpoints, "
            "configs, logs, metrics, valid825 profiles, deployment profiles, and RTL evidence"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(item["size_bytes"]) for item in candidates),
        "candidate_blocks_bytes": sum(int(item["blocks_bytes"]) for item in candidates),
        "retained_rank1_count": len(retained),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "deleted": candidates if args.execute else [],
        "candidates": [] if args.execute else candidates,
        "retained_rank1": retained,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "executed",
                    "candidate_count",
                    "candidate_bytes",
                    "candidate_blocks_bytes",
                    "retained_rank1_count",
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
