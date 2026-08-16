#!/usr/bin/env python3
"""Archive tool-input identities and scope for a Synopsys handoff run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


PATH_VARS = (
    "LIB_DB",
    "RTL_FILELIST",
    "MAPPED_NETLIST",
    "MAPPED_SDC",
    "SPEF_FILE",
    "SAIF_FILE",
    "SAIF_MANIFEST",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dc", "formality", "ptsta", "ptpx"), required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths: dict[str, dict[str, str]] = {}
    for name in PATH_VARS:
        raw = os.environ.get(name, "")
        if not raw:
            continue
        path = Path(raw).resolve()
        paths[name] = {
            "path": str(path),
            "sha256": sha256(path) if path.is_file() else "MISSING",
        }
    macro_paths = []
    for raw in os.environ.get("MACRO_DBS", "").split(":"):
        if not raw:
            continue
        path = Path(raw).resolve()
        macro_paths.append(
            {"path": str(path), "sha256": sha256(path) if path.is_file() else "MISSING"}
        )

    tool_name = {
        "dc": "dc_shell",
        "formality": "fm_shell",
        "ptsta": "pt_shell",
        "ptpx": "pt_shell",
    }[args.mode]
    commit = git_value(args.root, "rev-parse", "HEAD")
    dirty = git_value(args.root, "status", "--porcelain", "--untracked-files=no")
    result = {
        "mode": args.mode,
        "design_name": args.design,
        "tool_executable": shutil.which(tool_name),
        "git_commit": commit,
        "git_tracked_dirty": bool(dirty) if dirty is not None else None,
        "operating_condition": os.environ.get("OPERATING_CONDITION", ""),
        "corner_role": os.environ.get("CORNER_ROLE", ""),
        "voltage_v": os.environ.get("VOLTAGE_V", ""),
        "temperature_c": os.environ.get("TEMPERATURE_C", ""),
        "ppa_admission": os.environ.get("PPA_ADMISSION", "0"),
        "expected_macro_refs": os.environ.get("EXPECTED_MACRO_REFS", ""),
        "paths": paths,
        "macro_dbs": macro_paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
