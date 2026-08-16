#!/usr/bin/env python3
"""Prune non-best June checkpoints only when ranking and lineage are explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = REPO / (
    "neuron_autoresearch/cleanup_audits/"
    "ranked_june_nonbest_models_20260806.json"
)
DRY_RUN_AUDIT = AUDIT.with_name("ranked_june_nonbest_models_20260806_dry_run.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranking(path: Path) -> dict[int, int]:
    rows: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].isdigit():
            continue
        rank = int(cells[0])
        match = re.search(r"checkpoint_epoch(\d+)\.pth", cells[1])
        if match:
            epoch = int(match.group(1))
        elif cells[1].isdigit():
            epoch = int(cells[1])
        else:
            continue
        rows[epoch] = rank
    if not rows:
        raise RuntimeError(f"no ranking rows in {path}")
    return rows


def external_reference_index(paths: list[Path]) -> dict[str, list[str]]:
    index = {str(path.resolve()): [] for path in paths}
    if not paths:
        return index
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as patterns:
        patterns.write("\n".join(index) + "\n")
        patterns.flush()
        command = [
            "rg",
            "--json",
            "-F",
            "-f",
            patterns.name,
            str(REPO),
            "--glob",
            "!*.pth",
            "--glob",
            "!*.pyc",
            "--glob",
            "!ranked_june_nonbest_models_20260806.json",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip())
    own_dirs = {key: Path(key).parent.resolve() for key in index}
    for line in result.stdout.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        source = Path(data["path"]["text"]).resolve()
        for submatch in data.get("submatches", []):
            matched = submatch["match"]["text"]
            if matched in index and source.parent != own_dirs[matched]:
                index[matched].append(str(source))
    return {key: sorted(set(value)) for key, value in index.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    pending: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []
    for rank_file in sorted(RESULTS.rglob("profile_ranking_valid825.md")):
        route = rank_file.parent
        if "202606" not in route.name:
            continue
        models = sorted(
            path
            for path in route.glob("checkpoint_epoch*.pth")
            if "state_dict" not in path.name
        )
        if len(models) < 2:
            continue
        ranks = ranking(rank_file)
        ranked_models = []
        for model in models:
            match = re.fullmatch(r"checkpoint_epoch(\d+)\.pth", model.name)
            if match and int(match.group(1)) in ranks:
                ranked_models.append((ranks[int(match.group(1))], int(match.group(1)), model))
        if len(ranked_models) < 2:
            continue
        ranked_models.sort()
        best_rank, best_epoch, best_model = ranked_models[0]
        retained.append(
            {
                "path": str(best_model.resolve()),
                "reason": "best_rank_among_preserved_models",
                "rank": best_rank,
                "epoch": best_epoch,
            }
        )
        for rank, epoch, model in ranked_models[1:]:
            record = {
                "path": str(model.resolve()),
                "rank": rank,
                "epoch": epoch,
                "best_preserved_path": str(best_model.resolve()),
                "best_preserved_rank": best_rank,
                "size_bytes": model.stat().st_size,
                "allocated_bytes": model.stat().st_blocks * 512,
                "link_count": model.stat().st_nlink,
            }
            pending.append(record)

    reference_index = external_reference_index(
        [Path(str(item["path"])) for item in pending]
    )
    for record in pending:
        refs = reference_index[str(record["path"])]
        record["external_references"] = refs
        if refs:
            record["reason"] = "retained_external_lineage_reference"
            retained.append(record)
        else:
            record["reason"] = "ranked_nonbest_no_external_lineage_reference"
            candidates.append(record)

    if args.execute:
        for item in candidates:
            path = Path(str(item["path"]))
            if not path.is_file():
                raise RuntimeError(f"candidate disappeared before deletion: {path}")
            item["sha256"] = sha256(path)
        for item in retained:
            if item.get("reason") == "best_rank_among_preserved_models":
                path = Path(str(item["path"]))
                item["sha256"] = sha256(path)
        for item in candidates:
            Path(str(item["path"])).unlink()

    audit = {
        "schema": "ranked_june_nonbest_models_cleanup_v1",
        "status": "EXECUTED" if args.execute else "DRY_RUN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "scope": "202606 result directories with standard valid825 ranking",
            "keep": "best-ranked checkpoint among models still present",
            "lineage_guard": "retain any candidate named outside its own result directory",
            "protected_families": "all current NB0/H67/Local5/MDR paths are outside this scope",
        },
        "deleted": candidates if args.execute else [],
        "candidates": candidates,
        "retained": retained,
        "candidate_count": len(candidates),
        "logical_bytes": sum(int(item["size_bytes"]) for item in candidates),
        "allocated_bytes_upper_bound": sum(
            int(item["allocated_bytes"]) for item in candidates
        ),
        "post_checks": {
            "all_candidates_absent": (
                all(not Path(str(item["path"])).exists() for item in candidates)
                if args.execute
                else None
            ),
            "all_best_models_present": all(
                Path(str(item["path"])).is_file()
                for item in retained
                if item.get("reason") == "best_rank_among_preserved_models"
            ),
        },
    }
    output = AUDIT
    if not args.execute and AUDIT.is_file():
        previous = json.loads(AUDIT.read_text(encoding="utf-8"))
        if previous.get("status") == "EXECUTED":
            output = DRY_RUN_AUDIT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "candidate_count": audit["candidate_count"],
                "logical_gib": audit["logical_bytes"] / 2**30,
                "audit": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
