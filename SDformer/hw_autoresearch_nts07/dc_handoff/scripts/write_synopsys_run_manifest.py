#!/usr/bin/env python3
"""Archive tool-input identities and scope for a Synopsys handoff run."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional


PATH_VARS = (
    "LIB_DB",
    "MIN_LIB_DB",
    "RTL_FILELIST",
    "SDC_FILE",
    "MAPPED_NETLIST",
    "MAPPED_SDC",
    "MAPPED_SDC_SOURCE",
    "RTL_GATE_MAP_TCL",
    "SPEF_FILE",
    "SAIF_FILE",
    "SAIF_MANIFEST",
)

CLOCK_UNCERTAINTY_RE = re.compile(
    r"^\s*set_clock_uncertainty\s+-(setup|hold)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\b",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_clock_uncertainties(path: Path) -> Dict[str, str]:
    """Return the explicit setup/hold uncertainty contract in nanoseconds."""
    if not path.is_file():
        return {}
    result = {}  # type: Dict[str, str]
    for role, value in CLOCK_UNCERTAINTY_RE.findall(
        path.read_text(encoding="utf-8", errors="replace")
    ):
        result[role] = value
    return result


def git_value(root: Path, *args: str) -> Optional[str]:
    result = subprocess.run(
        ["git", *args], cwd=str(root), universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dc", "formality", "ptsta", "ptpx"), required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {}  # type: Dict[str, dict]
    for name in PATH_VARS:
        raw = os.environ.get(name, "")
        if not raw:
            continue
        path = Path(raw).resolve()
        entry = {
            "path": str(path),
            "sha256": sha256(path) if path.is_file() else "MISSING",
        }
        if name in {"SDC_FILE", "MAPPED_SDC", "MAPPED_SDC_SOURCE"}:
            entry["clock_uncertainty_ns"] = parse_clock_uncertainties(path)
        paths[name] = entry
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
    effective_sdc = paths.get("MAPPED_SDC", paths.get("SDC_FILE", {}))
    result = {
        "mode": args.mode,
        "design_name": args.design,
        "tool_executable": shutil.which(tool_name),
        "git_commit": commit,
        "git_tracked_dirty": bool(dirty) if dirty is not None else None,
        "operating_condition": os.environ.get("OPERATING_CONDITION", ""),
        "clock_period_ns": os.environ.get("CLOCK_PERIOD_NS", ""),
        "elab_parameters": os.environ.get("ELAB_PARAMETERS", ""),
        "dc_hold_uncertainty_ns": os.environ.get("DC_HOLD_UNCERTAINTY_NS", ""),
        "dc_hold_report_uncertainty_ns": os.environ.get(
            "DC_HOLD_REPORT_UNCERTAINTY_NS", ""
        ),
        "effective_clock_uncertainty_ns": effective_sdc.get(
            "clock_uncertainty_ns", {}
        ),
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
