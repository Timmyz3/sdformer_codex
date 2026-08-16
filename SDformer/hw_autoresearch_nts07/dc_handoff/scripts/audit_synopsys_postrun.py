#!/usr/bin/env python3
"""Check Synopsys post-run artifacts without claiming QoR signoff."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODE_FILES = {
    "ptsta": [
        "ptsta.log",
        "ptsta_run_manifest.json",
        "reports/ptsta_scope.rpt",
        "reports/ptsta_annotated_parasitics.rpt",
        "reports/ptsta_check_timing.rpt",
        "reports/ptsta_analysis_coverage.rpt",
        "reports/ptsta_global_timing.rpt",
        "reports/ptsta_timing_setup.rpt",
        "reports/ptsta_timing_hold.rpt",
        "reports/ptsta_constraint_violators.rpt",
    ],
    "ptpx": [
        "ptpx.log",
        "ptpx_run_manifest.json",
        "reports/ptpx_scope.rpt",
        "reports/ptpx_annotated_parasitics.rpt",
        "reports/ptpx_check_timing.rpt",
        "reports/ptpx_check_power.rpt",
        "reports/ptpx_unannotated.rpt",
        "reports/ptpx_switching_summary.rpt",
        "reports/ptpx_power_hierarchy.rpt",
        "reports/ptpx_power.rpt",
        "reports/ptpx_timing_setup.rpt",
    ],
}


def switching_coverage(text: str) -> float | None:
    matches = re.findall(
        r"(?im)^\s*SAIF\s+annotation\s+coverage\s*[:=]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*$",
        text,
    )
    return float(matches[0]) if len(matches) == 1 else None


def unannotated_object_count(text: str) -> int | None:
    """Parse only explicit PrimeTime totals; unknown formats fail closed."""
    patterns = (
        r"(?im)^\s*Total\s+number\s+of\s+unannotated\s+\S+\s*[:=]\s*(\d+)\s*$",
        r"(?im)^\s*Number\s+of\s+\S+\s+not\s+annotated\s*[:=]\s*(\d+)\s*$",
    )
    values: list[int] = []
    for pattern in patterns:
        values.extend(int(value) for value in re.findall(pattern, text))
    if values:
        return sum(values)
    if re.search(r"(?im)^\s*No\s+unannotated\s+(?:objects|nets|pins|cells)\.?\s*$", text):
        return 0
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_FILES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-saif-coverage-pct", type=float)
    args = parser.parse_args()
    checks = {
        name: (args.run_dir / name).is_file() and (args.run_dir / name).stat().st_size > 0
        for name in MODE_FILES[args.mode]
    }
    logs = []
    for name in MODE_FILES[args.mode]:
        path = args.run_dir / name
        if path.is_file():
            logs.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(logs)
    fatal_patterns = {
        "synopsys_error": bool(re.search(r"(^|\n)\s*Error:", text)),
        "fatal": bool(re.search(r"(^|\n)\s*FATAL", text, re.IGNORECASE)),
    }
    coverage_text = ""
    if args.mode == "ptpx":
        coverage_path = args.run_dir / "reports/ptpx_switching_summary.rpt"
        if coverage_path.is_file():
            coverage_text = coverage_path.read_text(encoding="utf-8", errors="replace")
    coverage = switching_coverage(coverage_text) if args.mode == "ptpx" else None
    unannotated = None
    if args.mode == "ptpx":
        unannotated_path = args.run_dir / "reports/ptpx_unannotated.rpt"
        if unannotated_path.is_file():
            unannotated = unannotated_object_count(
                unannotated_path.read_text(encoding="utf-8", errors="replace")
            )
    coverage_ok = True
    if args.min_saif_coverage_pct is not None:
        coverage_ok = coverage is not None and coverage >= args.min_saif_coverage_pct
    artifacts_ok = all(checks.values())
    unannotated_ok = args.mode != "ptpx" or unannotated == 0
    clean_exit = (
        artifacts_ok
        and not any(fatal_patterns.values())
        and coverage_ok
        and unannotated_ok
    )
    result = {
        "mode": args.mode,
        "status": "ARTIFACTS_PRESENT_REVIEW_REQUIRED" if clean_exit else "FAIL",
        "artifacts": checks,
        "fatal_patterns": fatal_patterns,
        "saif_annotation_coverage_pct": coverage,
        "minimum_saif_annotation_coverage_pct": args.min_saif_coverage_pct,
        "saif_coverage_pass": coverage_ok,
        "unannotated_object_count": unannotated,
        "unannotated_objects_pass": unannotated_ok,
        "boundary": (
            "Artifact presence and clean tool exit only. WNS/TNS, unconstrained paths, "
            "violations, library/PVT, and SPEF scope still require review. When requested, "
            "SAIF coverage and an explicit zero-unannotated-object total are fail-closed "
            "admission checks rather than a power-signoff claim."
        ),
    }
    output = args.run_dir / f"{args.mode}_artifact_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if clean_exit else 1


if __name__ == "__main__":
    raise SystemExit(main())
