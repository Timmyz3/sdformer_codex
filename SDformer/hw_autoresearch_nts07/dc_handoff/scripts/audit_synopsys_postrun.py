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


SWITCHING_COLUMNS = (
    "from_activity_file",
    "from_ssa",
    "from_ssa_force_annotated",
    "from_ssa_force_implied",
    "from_sca",
    "from_clock",
    "default",
    "propagated",
    "implied",
    "not_annotated",
)


def switching_activity_rows(text: str) -> dict[str, dict[str, object]]:
    """Parse the W-2024.09 switching-activity overview, not a made-up summary.

    PrimeTime prints two tables with the same shape.  Only the first table is
    transition activity; the later static-probability table must not be folded
    into admission.  Unknown or changed formats return no rows and fail closed.
    """
    start = text.find("Switching Activity Overview Statistics")
    end = text.find("Static Probability Overview Statistics")
    if start < 0 or end <= start:
        return {}
    table = text[start:end]
    row_pattern = re.compile(
        r"(?m)^\s*(Nets|Primary Input|Tri-State|Black Box|Sequential|"
        r"Combinational|Memory|Clock Gate)\s+(.+)$"
    )
    value_pattern = re.compile(r"(\d+)\(\s*([0-9]+(?:\.[0-9]+)?)%?\s*\)")
    rows: dict[str, dict[str, object]] = {}
    for match in row_pattern.finditer(table):
        values = value_pattern.findall(match.group(2))
        if len(values) != len(SWITCHING_COLUMNS):
            continue
        tail = match.group(2)[match.group(2).rfind(")") + 1 :]
        total_match = re.fullmatch(r"\s*(\d+)\s*", tail)
        if total_match is None:
            continue
        rows[match.group(1)] = {
            name: {"count": int(count), "pct": float(pct)}
            for name, (count, pct) in zip(SWITCHING_COLUMNS, values)
        }
        rows[match.group(1)]["total"] = int(total_match.group(1))
    return rows


def switching_coverage(text: str) -> float | None:
    rows = switching_activity_rows(text)
    nets = rows.get("Nets")
    if nets is not None:
        # Source-derived includes direct SAIF, clock and propagation, while
        # excluding PrimeTime defaults and explicitly unannotated nets.
        return 100.0 - float(nets["default"]["pct"]) - float(
            nets["not_annotated"]["pct"]
        )
    # Retain support for older explicitly generated summaries.  Ambiguous
    # percentages are still rejected.
    matches = re.findall(
        r"(?im)^\s*SAIF\s+annotation\s+coverage\s*[:=]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*$",
        text,
    )
    return float(matches[0]) if len(matches) == 1 else None


def unannotated_object_count(text: str) -> int | None:
    """Parse only explicit PrimeTime totals; unknown formats fail closed."""
    rows = switching_activity_rows(text)
    if "Nets" in rows:
        return int(rows["Nets"]["not_annotated"]["count"])
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
    activity_rows = (
        switching_activity_rows(coverage_text) if args.mode == "ptpx" else {}
    )
    coverage = switching_coverage(coverage_text) if args.mode == "ptpx" else None
    unannotated = None
    if args.mode == "ptpx":
        unannotated_path = args.run_dir / "reports/ptpx_unannotated.rpt"
        if unannotated_path.is_file():
            unannotated_text = unannotated_path.read_text(
                encoding="utf-8", errors="replace"
            )
            unannotated = unannotated_object_count(unannotated_text)
            if unannotated is None:
                unannotated = unannotated_object_count(coverage_text)
    coverage_ok = True
    if args.min_saif_coverage_pct is not None:
        coverage_ok = coverage is not None and coverage >= args.min_saif_coverage_pct
    component_coverage: dict[str, float | None] = {
        "primary_input_from_activity_file_pct": None,
        "sequential_from_activity_file_pct": None,
        "combinational_direct_or_propagated_pct": None,
        "nets_default_pct": None,
        "nets_not_annotated_pct": None,
    }
    component_coverage_ok = coverage_ok
    if activity_rows:
        required_rows = ("Nets", "Primary Input", "Sequential", "Combinational")
        if all(name in activity_rows for name in required_rows):
            nets = activity_rows["Nets"]
            primary = activity_rows["Primary Input"]
            sequential = activity_rows["Sequential"]
            combinational = activity_rows["Combinational"]
            component_coverage = {
                "primary_input_from_activity_file_pct": float(
                    primary["from_activity_file"]["pct"]
                ),
                "sequential_from_activity_file_pct": float(
                    sequential["from_activity_file"]["pct"]
                ),
                "combinational_direct_or_propagated_pct": float(
                    combinational["from_activity_file"]["pct"]
                ) + float(combinational["propagated"]["pct"]),
                "nets_default_pct": float(nets["default"]["pct"]),
                "nets_not_annotated_pct": float(nets["not_annotated"]["pct"]),
            }
            if args.min_saif_coverage_pct is not None:
                floor = args.min_saif_coverage_pct
                component_coverage_ok = (
                    component_coverage["primary_input_from_activity_file_pct"] >= floor
                    and component_coverage["sequential_from_activity_file_pct"] >= floor
                    and component_coverage[
                        "combinational_direct_or_propagated_pct"
                    ] >= floor
                    and component_coverage["nets_default_pct"] <= 100.0 - floor
                    and component_coverage["nets_not_annotated_pct"] == 0.0
                )
        else:
            component_coverage_ok = False
    artifacts_ok = all(checks.values())
    unannotated_ok = args.mode != "ptpx" or unannotated == 0
    clean_exit = (
        artifacts_ok
        and not any(fatal_patterns.values())
        and coverage_ok
        and component_coverage_ok
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
        "switching_activity_rows": activity_rows,
        "component_activity_coverage": component_coverage,
        "component_activity_coverage_pass": component_coverage_ok,
        "unannotated_object_count": unannotated,
        "unannotated_objects_pass": unannotated_ok,
        "boundary": (
            "Artifact presence and clean tool exit only. WNS/TNS, unconstrained paths, "
            "violations, library/PVT, and SPEF scope still require review. When requested, "
            "SAIF admission separately checks primary-input direct annotation, sequential "
            "direct annotation, combinational direct-or-propagated activity, defaults, and "
            "unannotated nets. These remain fail-closed checks, not a power-signoff claim."
        ),
    }
    output = args.run_dir / f"{args.mode}_artifact_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if clean_exit else 1


if __name__ == "__main__":
    raise SystemExit(main())
