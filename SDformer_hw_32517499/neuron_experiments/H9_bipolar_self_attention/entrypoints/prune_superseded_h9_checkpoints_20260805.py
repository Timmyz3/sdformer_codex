"""Prune explicit superseded H9 checkpoints while retaining paper anchors."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = REPO_ROOT / "neuron_autoresearch/cleanup_audits/h9_superseded_20260805.json"
CHECKPOINT_RE = re.compile(r"checkpoint_epoch(\d+)\.pth$")

# These are concluded June routes. Current NB0/H67/Local5 and their staged
# convergence checkpoints are intentionally outside this allowlist.
POLICY = {
    "nts05d_hw_mu0075_mis000_sap0025_w720_s360_auto_full_bs6_20260608_020722_setsid": {
        "keep": set(),
        "reason": "failed early route; scalar results are already recorded",
    },
    "nts06a_hw_mu005_mis000_sap000_w720_s360_auto_full_bs6_20260608_031001_setsid": {
        "keep": set(),
        "reason": "failed early route; scalar results are already recorded",
    },
    "nts10d_hw_h60_s23_freeze1224_s1224_steps1224_auto_full_bs6_20260610_151207_setsid": {
        "keep": {19, 24, 29},
        "reason": "retain every standard-valid825 milestone; prune unevaluated intermediates",
    },
    "nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15_bs8_20260612_194020_setsid": {
        "keep": {0, 14},
        "reason": "retain rank-1/resume anchor ep0 and final ep14",
    },
    "nts11aq_hw_h60_s23_ds_w720_fastlr_ftaa19_full_20260613_070741_bs8_20260613_070741_setsid": {
        "keep": {2},
        "reason": "retain rank-1 and NTS11aqa continuation source ep2",
    },
    "nts11aqa_hw_h60_s23_ds_w720_fastlr_ftaq2_ft5_bs8_20260613_125039_setsid": {
        "keep": {5, 7},
        "reason": "retain best-AEE ep5 and best-AAE/final ep7",
    },
}


def disk_free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return int(stats.f_bavail * stats.f_frsize)


def record(path: Path, reason: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "links": int(stat.st_nlink),
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    delete: list[dict[str, object]] = []
    retain: list[dict[str, object]] = []
    for run_name, policy in POLICY.items():
        run_dir = RESULTS / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for path in sorted(run_dir.glob("checkpoint_epoch*.pth")):
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match is None:
                # Paired optimizer states are never selected by this cleanup.
                continue
            epoch = int(match.group(1))
            item = record(path, str(policy["reason"]))
            item["epoch"] = epoch
            if epoch in policy["keep"]:
                retain.append(item)
            else:
                if int(item["links"]) != 1:
                    raise RuntimeError(f"refuse linked checkpoint deletion: {path}")
                delete.append(item)

    before = disk_free_bytes(REPO_ROOT)
    if args.execute:
        for item in delete:
            path = REPO_ROOT / str(item["path"])
            path.unlink()
            item["exists_after"] = path.exists()
        for item in retain:
            path = REPO_ROOT / str(item["path"])
            item["exists_after"] = path.exists()
            if not path.is_file():
                raise RuntimeError(f"retained checkpoint disappeared: {path}")
    after = disk_free_bytes(REPO_ROOT)
    report = {
        "schema": "h9_superseded_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "scope": "explicit concluded June H9 model checkpoints only",
        "protected_scope": "all NB0, H67, Local5, current queues, paired optimizer states, configs, logs, metrics, and RTL/profile artifacts",
        "policy": {
            name: {"keep_epochs": sorted(value["keep"]), "reason": value["reason"]}
            for name, value in POLICY.items()
        },
        "delete_count": len(delete),
        "delete_bytes": sum(int(item["size_bytes"]) for item in delete),
        "retain_count": len(retain),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "deleted": delete,
        "retained": retain,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "executed", "delete_count", "delete_bytes", "retain_count",
        "free_bytes_before", "free_bytes_after", "observed_free_bytes_delta",
    )}, indent=2))
    print(f"audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
