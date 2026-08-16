#!/usr/bin/env python3
"""Remove retired one-epoch idea-screen weights while retaining evidence."""

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
    "retired_one_epoch_idea_screens_20260806.json"
)
CANDIDATES = (
    "date11_bttx_a4_txonly_from_ttx_20260710_200703/runs/"
    "date11full_bttx_a4_txonly_all12_s360_steps120/checkpoint_epoch0.pth",
    "date11_faps_txratio_integer_txep19_20260627_231503/runs/"
    "date11full_all_binary_atlif_faps_all12_nokmag_s16_z1_p2_"
    "sc0p0625_nosplit_txep19_s360_steps360/checkpoint_epoch0.pth",
    "date11_faps_txratio_integer_txep19_20260627_231503/runs/"
    "date11full_all_binary_atlif_faps_all12_nokmag_s32_z1_p3_"
    "sc0p03125_nosplit_txep19_s360_steps360/checkpoint_epoch0.pth",
    "date11_faps_txratio_integer_txep19_20260627_231503/runs/"
    "date11full_all_binary_atlif_faps_all12_nokmag_s64_z1_p6_"
    "sc0p015625_nosplit_txep19_s360_steps360/checkpoint_epoch0.pth",
    "h66a_accuracy_first_20260711_151351/runs/"
    "h66a_allbinary_all12_axnor_matrix_shiftmax_s120_steps120/"
    "checkpoint_epoch0.pth",
    "h66a_accuracy360_20260711_151943/runs/"
    "h66a_allbinary_all12_axnor_matrix_shiftmax_s120_steps360/"
    "checkpoint_epoch0.pth",
    "h65_all105_symtern_signed_hamming_recovery20_20260711_145334/runs/"
    "h65_all105_symtern_signed_hamming_s20_steps20/checkpoint_epoch0.pth",
)
SUMMARY_ROOTS = (
    "date11_bttx_a4_txonly_from_ttx_20260710_200703",
    "date11_faps_txratio_integer_txep19_20260627_231503",
    "h66a_accuracy_first_20260711_151351",
    "h66a_accuracy360_20260711_151943",
    "h65_all105_symtern_signed_hamming_recovery20_20260711_145334",
)
PROTECTED_ANCHORS = (
    "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30.pth",
    "dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728/checkpoint_epoch29.pth",
    "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/"
    "checkpoint_epoch29.pth",
    "nts11_two_neuron_20260611_203636/runs/"
    "nts11c_hw_h60_s23_two_neuron_fastlr_s1224_steps1224/checkpoint_epoch0.pth",
    "nts11_phase2_20260611_230130/runs/"
    "nts11j_hw_h60_s23_two_neuron_vanilla_decoder_s1224_steps1224/"
    "checkpoint_epoch0.pth",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protected = [RESULTS / relative for relative in PROTECTED_ANCHORS]
    missing_protected = [path for path in protected if not path.is_file()]
    if missing_protected:
        raise FileNotFoundError(f"protected anchors missing: {missing_protected}")

    for relative in SUMMARY_ROOTS:
        root = RESULTS / relative
        if not (root / "summary.csv").is_file() or not (root / "summary.md").is_file():
            raise FileNotFoundError(f"screen summary evidence missing: {root}")

    candidates = [RESULTS / relative for relative in CANDIDATES]
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"candidate checkpoint missing: {missing}")

    active = active_cmdlines()
    rows: list[dict[str, object]] = []
    for path in candidates:
        stat = path.stat()
        if stat.st_nlink != 1:
            raise RuntimeError(f"refuse shared checkpoint inode: {path}")
        if str(path.resolve()) in active:
            raise RuntimeError(f"refuse active checkpoint: {path}")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": int(stat.st_size),
                "blocks_bytes": int(stat.st_blocks * 512),
                "inode": int(stat.st_ino),
                "reason": (
                    "retired one-epoch idea screen; summary/config/log evidence retained; "
                    "not referenced by the active DSEC or hardware execution chain"
                ),
            }
        )

    before = free_bytes()
    if args.execute:
        for path in candidates:
            path.unlink()
    after = free_bytes()

    report = {
        "schema": "retired_one_epoch_idea_screen_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "scope": "seven retired one-epoch BTTX/FAPS/H66a/H65 screening weights",
        "candidate_count": len(rows),
        "candidate_bytes": sum(int(row["size_bytes"]) for row in rows),
        "candidate_blocks_bytes": sum(int(row["blocks_bytes"]) for row in rows),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "all_candidates_removed": all(not path.exists() for path in candidates)
        if args.execute
        else False,
        "deleted": rows if args.execute else [],
        "candidates": [] if args.execute else rows,
        "retained_summary_roots": [
            str((RESULTS / relative).relative_to(ROOT)) for relative in SUMMARY_ROOTS
        ],
        "protected_anchors": [str(path.relative_to(ROOT)) for path in protected],
    }
    if args.execute and not report["all_candidates_removed"]:
        raise RuntimeError("one or more retired screen checkpoints survived deletion")
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
