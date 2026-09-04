#!/usr/bin/python3.12
"""Emit a deterministic inventory of every immediate repository-root node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def node_record(path: Path) -> dict[str, object]:
    info = path.lstat()
    mode = info.st_mode
    record: dict[str, object] = {
        "name": path.name,
        "mode_octal": format(stat.S_IMODE(mode), "04o"),
    }
    if stat.S_ISREG(mode):
        record.update(node_type="regular", size_bytes=info.st_size, sha256=digest(path))
    elif stat.S_ISDIR(mode):
        record.update(node_type="directory")
    elif stat.S_ISLNK(mode):
        record.update(node_type="symlink", target=os.readlink(path))
    elif stat.S_ISFIFO(mode):
        record.update(node_type="fifo")
    elif stat.S_ISSOCK(mode):
        record.update(node_type="socket")
    elif stat.S_ISBLK(mode):
        record.update(node_type="block_device", device=info.st_rdev)
    elif stat.S_ISCHR(mode):
        record.update(node_type="character_device", device=info.st_rdev)
    else:
        record.update(node_type="unknown", raw_mode=mode)
    return record


def inventory(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"root is not a real directory: {root}")
    nodes = [node_record(path) for path in sorted(root.iterdir(), key=lambda p: p.name)]
    return {
        "schema": "m2153_repo_root_inventory_r1_v1",
        "root": str(root),
        "node_count": len(nodes),
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = inventory(args.root)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PASS_M2153_REPO_ROOT_ALL_NODE_INVENTORY")
    print(f"node_count={payload['node_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
