#!/usr/bin/env python3
"""Prune unreferenced historical SDformerFlow intermediates, retaining epoch 59."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "third_party/SDformerFlow/results"
AUDIT = (
    ROOT
    / "neuron_autoresearch/cleanup_audits/third_party_sdformerflow_20260805.json"
)
KEEP = {"checkpoint_epoch59.pth", "checkpoint_epoch59_state_dict.pth"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    checkpoints = sorted(RESULT_DIR.glob("checkpoint_epoch*.pth"))
    retained = [path for path in checkpoints if path.name in KEEP]
    candidates = [path for path in checkpoints if path.name not in KEEP]
    if {path.name for path in retained} != KEEP:
        raise RuntimeError("historical epoch59 retention set is incomplete")

    audit = {
        "schema": "sdformerflow_checkpoint_cleanup_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "root": str(RESULT_DIR),
        "policy": (
            "retain historical epoch59 model/state; prune unreferenced intermediate "
            "checkpoints from superseded May 2026 third-party runs"
        ),
        "retained": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in retained
        ],
        "deleted": [],
        "candidate_count": len(candidates),
        "candidate_bytes": sum(path.stat().st_size for path in candidates),
    }
    if args.execute:
        for path in candidates:
            size = path.stat().st_size
            audit["deleted"].append({"path": str(path), "bytes": size})
            path.unlink()

    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
