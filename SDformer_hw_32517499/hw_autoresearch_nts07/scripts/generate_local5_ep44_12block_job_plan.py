#!/usr/bin/env python3
"""Build a coverage-seeking 12-block job plan from the sealed ep44 vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


CHECKPOINT_SHA256 = (
    "19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57"
)
BLOCKS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 0),
    (3, 1),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(rows: list[dict[str, object]]) -> list[tuple[int, dict[str, object]]]:
    by_block: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
        defaultdict(list)
    )
    for index, row in enumerate(rows):
        by_block[(int(row["stage"]), int(row["block"]))].append((index, row))
    if set(by_block) != set(BLOCKS):
        raise ValueError("vector population does not cover the exact 12-block set")

    selected = []
    for block in BLOCKS:
        candidates = by_block[block]
        nonempty = [item for item in candidates if not bool(item[1]["empty"])]
        selected.append((nonempty or candidates)[0])
    return selected


def write_memh(path: Path, values: list[int], width: int) -> dict[str, object]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text(
        "".join(f"{value & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )
    return {
        "file": path.name,
        "entries": len(values),
        "width": width,
        "sha256": sha256(path),
    }


def decode_output_pair(row: dict[str, object]) -> tuple[list[int], int, int]:
    channels = [int(value) for value in row["projection_output_channels"]]
    if (
        len(channels) != 2
        or channels[1] != channels[0] + 1
        or channels[0] // 32 != channels[1] // 32
    ):
        raise ValueError("projection output channels are not one OUT_DIM=2 pair")
    return channels, channels[0] // 32, channels[0] % 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "local5_score_projection_vectors_v1"
        or manifest.get("checkpoint_sha256") not in (None, CHECKPOINT_SHA256)
        or manifest.get("weight_mode")
        != "checkpoint_theta_folded_dyadic_int8_head_slice"
        or int((manifest.get("selection") or {}).get("groups", 0)) != 100
    ):
        raise ValueError("input is not the sealed ep44 100-group vector population")
    source_manifest_path = Path(str(manifest.get("source_manifest", "")))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("source manifest is not bound to Local5 ep44")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_rows(manifest["selection"]["rows"])
    rows = []
    for ordinal, (group_index, row) in enumerate(selected):
        try:
            channels, output_tile, output_channel_offset = decode_output_pair(row)
        except ValueError as error:
            raise ValueError(f"group {group_index}: {error}") from error
        rows.append(
            {
                "ordinal": ordinal,
                "group_index": group_index,
                "stage": int(row["stage"]),
                "block": int(row["block"]),
                "sample": int(row["sample"]),
                "window": int(row["window"]),
                "head": int(row["head"]),
                "output_tile": output_tile,
                "output_channel_offset": output_channel_offset,
                "output_channels": channels,
                "empty": bool(row["empty"]),
                "ordered_item_sha256": row["ordered_item_sha256"],
            }
        )

    artifacts = {
        "group_index": write_memh(
            args.output_dir / "selected_group.memh",
            [row["group_index"] for row in rows],
            7,
        ),
        "stage": write_memh(
            args.output_dir / "selected_stage.memh",
            [row["stage"] for row in rows],
            2,
        ),
        "block": write_memh(
            args.output_dir / "selected_block.memh",
            [row["block"] for row in rows],
            3,
        ),
        "head": write_memh(
            args.output_dir / "selected_head.memh",
            [row["head"] for row in rows],
            5,
        ),
        "output_tile": write_memh(
            args.output_dir / "selected_output_tile.memh",
            [row["output_tile"] for row in rows],
            5,
        ),
        "empty": write_memh(
            args.output_dir / "selected_empty.memh",
            [int(row["empty"]) for row in rows],
            1,
        ),
    }
    result = {
        "schema": "local5_ep44_12block_job_plan_v1",
        "status": "PASS",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selection": (
            "first nonempty group per block, else first group; coverage-seeking "
            "verification selection, forbidden for performance statistics"
        ),
        "source_vector_manifest": str(manifest_path.resolve()),
        "source_vector_manifest_sha256": sha256(manifest_path),
        "jobs": len(rows),
        "nonempty_jobs": sum(not row["empty"] for row in rows),
        "rows": rows,
        "artifacts": artifacts,
        "claim_boundary": [
            "one real ep44 OUT_DIM=2 job per Local5 block",
            "not a same-window cross-head cohort",
            "not the 1320-window encoder schedule",
            "cycles are verification runtime and not paper performance",
        ],
    }
    output_path = args.output_dir / "plan.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "jobs": len(rows),
                "nonempty_jobs": result["nonempty_jobs"],
                "output": str(output_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
