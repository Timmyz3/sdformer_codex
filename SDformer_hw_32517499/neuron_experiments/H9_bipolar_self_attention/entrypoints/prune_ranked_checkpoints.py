"""Prune evaluated intermediate checkpoints while retaining reproducible results."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


RANK_RE = re.compile(r"\|\s*1\s*\|\s*(\d+)\s*\|")
EPOCH_RE = re.compile(r"checkpoint_epoch(\d+)(?:_state_dict)?\.pth$")


def ranked_epochs(ranking: Path) -> tuple[int, set[int]]:
    text = ranking.read_text(encoding="utf-8")
    match = RANK_RE.search(text)
    if match is None:
        raise RuntimeError(f"cannot parse best epoch from {ranking}")
    epochs = {
        int(value)
        for value in re.findall(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|", text, flags=re.MULTILINE)
    }
    return int(match.group(1)), epochs


def prune(run_dir: Path, dry_run: bool) -> dict:
    ranking = run_dir / "profile_ranking_valid825.md"
    if not ranking.exists():
        return {"run_dir": str(run_dir), "status": "skipped_no_ranking"}

    best, evaluated = ranked_epochs(ranking)
    final = max(evaluated)
    missing_profiles = [
        epoch
        for epoch in evaluated
        if not (run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json").exists()
    ]
    if missing_profiles:
        raise RuntimeError(f"refusing to prune {run_dir}: missing profiles {missing_profiles}")
    if not (run_dir / f"checkpoint_epoch{best}.pth").exists():
        raise RuntimeError(f"refusing to prune {run_dir}: best checkpoint epoch{best} is missing")
    if not (run_dir / f"checkpoint_epoch{final}.pth").exists():
        raise RuntimeError(f"refusing to prune {run_dir}: final checkpoint epoch{final} is missing")

    keep_epochs = {best, final}
    removed: list[dict] = []
    for checkpoint in sorted(run_dir.glob("checkpoint_epoch*.pth")):
        match = EPOCH_RE.fullmatch(checkpoint.name)
        if match is None or int(match.group(1)) in keep_epochs:
            continue
        size = checkpoint.stat().st_size
        removed.append({"path": checkpoint.name, "bytes": size})
        if not dry_run:
            checkpoint.unlink()

    audit = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "status": "dry_run" if dry_run else "pruned",
        "best_epoch": best,
        "final_epoch": final,
        "kept_epochs": sorted(keep_epochs),
        "removed": removed,
        "reclaimed_bytes": sum(item["bytes"] for item in removed),
    }
    if not dry_run:
        (run_dir / "checkpoint_prune_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        print(json.dumps(prune(run_dir.resolve(), args.dry_run), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
