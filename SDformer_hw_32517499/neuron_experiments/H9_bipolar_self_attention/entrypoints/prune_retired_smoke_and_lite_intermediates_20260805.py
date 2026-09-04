#!/usr/bin/env python3
"""Prune retired smoke and lite checkpoints while retaining paper anchors."""

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
    "retired_smoke_lite_intermediates_20260805.json"
)
EXECUTED_RECEIPT = ROOT / (
    "neuron_autoresearch/cleanup_audits/"
    "retired_smoke_lite_intermediates_20260805.executed.json"
)
CHECKPOINT_RE = re.compile(r"checkpoint_epoch(\d+)(?:_state_dict)?\.pth$")

# These runs have complete valid825 rankings. Preserve the best/final anchors;
# configs, logs, profiles, and rankings are never cleanup candidates.
RANKED_POLICIES: dict[str, dict[str, object]] = {
    "nts11bj_u12_ds_w720_stdlr_ftbd19_ft5_bs8_20260614_233224_setsid": {
        "keep": {2, 4},
        "reason": "retain valid825 rank-1 ep2 and final ep4",
    },
    "nts11bl_u12_ds_w720_fastlr_w360_ftbd19_ft5_bs8_20260615_024301_setsid": {
        "keep": {2, 4},
        "reason": "retain valid825 rank-1/final ep4 and second-best ep2",
    },
    "nts11lite_u12_qkonly_w720_fastlr_full30_bs8_20260615_052324_setsid": {
        "keep": {24, 29},
        "reason": "retain best-AEE ep24 and ranking/final ep29",
    },
    "nts11lite_u12_qkds_w720_fastlr_full30_bs8_20260615_024930_setsid": {
        "keep": {24, 29},
        "reason": "retain best-AEE ep24 and ranking/final ep29",
    },
}

# Smoke outputs were superseded by completed full MDR runs. Their logs and
# configs remain, but no checkpoint is an inference, resume, or paper anchor.
DROP_ALL = {
    "mdr_ttx_bs16_from_best_baseline_20260630_144907/smoke_ckpts":
        "superseded MDR batch-16 smoke checkpoints",
    "mdr_allbinary_tx_smoke_ckpts_20260627_121258":
        "superseded all-binary TX MDR smoke checkpoints",
    "ttx_mdr_forkserver_cupy_bench80_codex_clean_20260630_161105/smoke_ckpts":
        "superseded 80-batch loader benchmark checkpoints",
}

# The current fair runs supersede these initialization snapshots. Preserve the
# completed ep29 model and paired state in every directory.
OLD_FULLRES_EP0 = {
    "dsec_fullres_paper_w15_h67_motion_ep19_ft30_bs2_20260728",
    "dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728",
    "dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728",
}

# The first successful execution was followed by an empty idempotence dry-run
# that overwrote the canonical audit. This immutable recovery ledger records the
# exact paths/sizes printed and inspected before that dry-run. No files are
# recreated or removed by --restore-executed-receipt.
RECOVERY_GROUPS: tuple[tuple[str, str, tuple[tuple[str, int], ...]], ...] = (
    (
        "nts11bj_u12_ds_w720_stdlr_ftbd19_ft5_bs8_20260614_233224_setsid",
        "non-anchor valid825-ranked intermediate; retain ep2/ep4",
        (("checkpoint_epoch0.pth", 738323429), ("checkpoint_epoch1.pth", 738323429),
         ("checkpoint_epoch3.pth", 738323429)),
    ),
    (
        "nts11bl_u12_ds_w720_fastlr_w360_ftbd19_ft5_bs8_20260615_024301_setsid",
        "non-anchor valid825-ranked intermediate; retain ep2/ep4",
        (("checkpoint_epoch0.pth", 738323429), ("checkpoint_epoch1.pth", 738323429),
         ("checkpoint_epoch3.pth", 738323429)),
    ),
    (
        "nts11lite_u12_qkonly_w720_fastlr_full30_bs8_20260615_052324_setsid",
        "retired lite intermediate; retain best-AEE ep24 and final ep29",
        (("checkpoint_epoch9.pth", 486069434), ("checkpoint_epoch14.pth", 486070258),
         ("checkpoint_epoch19.pth", 486070258), ("checkpoint_epoch28.pth", 486070258)),
    ),
    (
        "nts11lite_u12_qkds_w720_fastlr_full30_bs8_20260615_024930_setsid",
        "retired lite intermediate; retain best-AEE ep24 and final ep29",
        (("checkpoint_epoch14.pth", 486074243), ("checkpoint_epoch19.pth", 486074243),
         ("checkpoint_epoch28.pth", 486074243)),
    ),
    (
        "mdr_ttx_bs16_from_best_baseline_20260630_144907/smoke_ckpts",
        "superseded MDR batch-16 smoke checkpoints",
        (("checkpoint_epoch0.pth", 219804706), ("checkpoint_epoch0_state_dict.pth", 438577446),
         ("checkpoint_epoch1.pth", 219804706), ("checkpoint_epoch1_state_dict.pth", 438577446),
         ("checkpoint_epoch2.pth", 219804706), ("checkpoint_epoch2_state_dict.pth", 438577446)),
    ),
    (
        "mdr_allbinary_tx_smoke_ckpts_20260627_121258",
        "superseded all-binary TX MDR smoke checkpoints",
        (("checkpoint_epoch0.pth", 219804706), ("checkpoint_epoch0_state_dict.pth", 438609174),
         ("checkpoint_epoch1.pth", 219804706), ("checkpoint_epoch1_state_dict.pth", 438609174)),
    ),
    (
        "ttx_mdr_forkserver_cupy_bench80_codex_clean_20260630_161105/smoke_ckpts",
        "superseded 80-batch loader benchmark checkpoints",
        (("checkpoint_epoch0.pth", 219804706), ("checkpoint_epoch0_state_dict.pth", 438577446),
         ("checkpoint_epoch1.pth", 219804706), ("checkpoint_epoch1_state_dict.pth", 438577446)),
    ),
    (
        "dsec_fullres_paper_w15_h67_motion_ep19_ft30_bs2_20260728",
        "superseded fullres initialization snapshot",
        (("checkpoint_epoch0.pth", 591166821),),
    ),
    (
        "dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728",
        "superseded fullres initialization snapshot",
        (("checkpoint_epoch0.pth", 591166629),),
    ),
    (
        "dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728",
        "superseded fullres initialization snapshot",
        (("checkpoint_epoch0.pth", 411943098),),
    ),
)


def recovery_records() -> list[dict[str, object]]:
    records = []
    for directory, reason, files in RECOVERY_GROUPS:
        for filename, size in files:
            records.append({
                "path": str(Path("neuron_experiments/H9_bipolar_self_attention/results")
                            / directory / filename),
                "size_bytes": size,
                "blocks_bytes": None,
                "inode": None,
                "links": None,
                "reason": reason,
            })
    return records


def active_commands() -> list[str]:
    commands = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if text:
            commands.append(text)
    return commands


def free_bytes() -> int:
    stat = os.statvfs(ROOT)
    return int(stat.f_bavail * stat.f_frsize)


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
    parser.add_argument("--restore-executed-receipt", action="store_true")
    args = parser.parse_args()
    if args.execute and args.restore_executed_receipt:
        raise RuntimeError("--execute and --restore-executed-receipt are mutually exclusive")

    running = active_commands()
    candidates: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []
    selected: set[Path] = set()

    protected = {
        "dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805",
        "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804",
        "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805",
        "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805",
    }

    for name, policy in RANKED_POLICIES.items():
        directory = RESULTS / name
        ranking = directory / "profile_ranking_valid825.md"
        if not ranking.is_file():
            raise RuntimeError(f"missing valid825 ranking: {ranking}")
        keep = policy["keep"]
        assert isinstance(keep, set)
        files = sorted(directory.glob("checkpoint_epoch*.pth"))
        observed = set()
        for path in files:
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            observed.add(epoch)
            record = describe(path, str(policy["reason"]))
            record["epoch"] = epoch
            if epoch in keep:
                retained.append(record)
            else:
                candidates.append(record)
                selected.add(path)
        if not keep.issubset(observed):
            raise RuntimeError(f"missing retained anchors in {directory}: {keep - observed}")

    for relative, reason in DROP_ALL.items():
        directory = RESULTS / relative
        if not directory.is_dir():
            raise RuntimeError(f"missing smoke directory: {directory}")
        for path in sorted(directory.glob("checkpoint_epoch*.pth")):
            if path in selected:
                continue
            candidates.append(describe(path, reason))
            selected.add(path)

    for name in OLD_FULLRES_EP0:
        directory = RESULTS / name
        final_model = directory / "checkpoint_epoch29.pth"
        final_state = directory / "checkpoint_epoch29_state_dict.pth"
        if not final_model.is_file() or not final_state.is_file():
            raise RuntimeError(f"missing old fullres final anchors: {directory}")
        retained.extend([
            describe(final_model, "retain completed ep29 model anchor"),
            describe(final_state, "retain paired ep29 optimizer state"),
        ])
        ep0 = directory / "checkpoint_epoch0.pth"
        if ep0.is_file():
            candidates.append(describe(ep0, "superseded fullres initialization snapshot"))
            selected.add(ep0)

    if args.restore_executed_receipt:
        deleted = recovery_records()
        if len(deleted) != 30 or sum(int(item["size_bytes"]) for item in deleted) != 14035458579:
            raise RuntimeError("recovery ledger count/byte total drift")
        survivors = [item["path"] for item in deleted if (ROOT / str(item["path"])).exists()]
        if survivors:
            raise RuntimeError(f"recovery ledger contains surviving paths: {survivors}")
        for record in retained:
            if not (ROOT / str(record["path"])).is_file():
                raise RuntimeError(f"retained anchor missing during receipt recovery: {record['path']}")
        report = {
            "schema": "retired_smoke_lite_checkpoint_cleanup_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "executed": True,
            "receipt_recovery": (
                "canonical executed receipt was overwritten by a post-execution empty dry-run; "
                "restored from the exact pre-execution path/size ledger and command-reported "
                "free-space values, with all deleted/retained paths revalidated"
            ),
            "candidate_count": 30,
            "candidate_bytes": 14035458579,
            "retained_count": len(retained),
            "free_bytes_before": 253912637440,
            "free_bytes_after": 267948093440,
            "observed_free_bytes_delta": 14035456000,
            "deleted": deleted,
            "candidates": [],
            "retained": retained,
        }
        payload = json.dumps(report, indent=2) + "\n"
        AUDIT.write_text(payload, encoding="utf-8")
        EXECUTED_RECEIPT.write_text(payload, encoding="utf-8")
        print(json.dumps({
            "restored": True,
            "deleted_count": len(deleted),
            "deleted_bytes": report["candidate_bytes"],
            "receipt": str(EXECUTED_RECEIPT),
        }, indent=2))
        return 0

    for record in candidates:
        if int(record["links"]) != 1:
            raise RuntimeError(f"refuse shared inode: {record}")
        path_text = str(ROOT / str(record["path"]))
        if any(path_text in command for command in running):
            raise RuntimeError(f"cleanup candidate is referenced by a running process: {path_text}")
        if any(part in protected for part in Path(str(record["path"])).parts):
            raise RuntimeError(f"candidate entered protected current run: {record}")

    before = free_bytes()
    if args.execute:
        for record in candidates:
            (ROOT / str(record["path"])).unlink()
        for record in candidates:
            if (ROOT / str(record["path"])).exists():
                raise RuntimeError(f"checkpoint survived deletion: {record['path']}")
        for record in retained:
            if not (ROOT / str(record["path"])).is_file():
                raise RuntimeError(f"retained anchor disappeared: {record['path']}")
    after = free_bytes()

    report = {
        "schema": "retired_smoke_lite_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "protected_current_runs": sorted(protected),
        "policy": (
            "remove only superseded smoke, non-anchor ranked lite/FT, and old fullres ep0 "
            "models; retain best/final/resume anchors plus all configs, logs, metrics, "
            "valid825 rankings, profiles, and RTL artifacts"
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

    # Once an immutable executed receipt exists, an empty idempotence dry-run
    # must not replace the canonical execution evidence.
    if not args.execute and not candidates and EXECUTED_RECEIPT.is_file():
        existing = json.loads(EXECUTED_RECEIPT.read_text(encoding="utf-8"))
        if existing.get("executed") is not True:
            raise RuntimeError(f"invalid immutable execution receipt: {EXECUTED_RECEIPT}")
        AUDIT.write_text(EXECUTED_RECEIPT.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps({
            "executed": False,
            "candidate_count": 0,
            "candidate_bytes": 0,
            "retained_count": len(retained),
            "preserved_executed_receipt": str(EXECUTED_RECEIPT),
        }, indent=2))
        return 0
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        EXECUTED_RECEIPT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    keys = (
        "executed", "candidate_count", "candidate_bytes", "retained_count",
        "free_bytes_before", "free_bytes_after", "observed_free_bytes_delta",
    )
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    print(f"audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
