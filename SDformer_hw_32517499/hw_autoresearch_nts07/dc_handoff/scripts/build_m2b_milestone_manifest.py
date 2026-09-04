#!/usr/bin/env python3
"""Build a fail-closed evidence manifest for the M2B banked multi-source milestone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ISSUE_WIDTHS = (1, 2, 4, 8)
EXPECTED_CASES = 20_000
EXPECTED_SOURCES = 334_542


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, label_root: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty artifact missing: {path}")
    label = str(path.relative_to(label_root)) if label_root and path.is_relative_to(label_root) else str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_sha256_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"malformed checksum line in {path}: {line!r}")
        expected, raw_target = match.groups()
        target = Path(raw_target)
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"checksum mismatch from {path}: {target}")


def git(args: list[str], root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.rstrip("\n")


def require_vcs_pass(log: Path, issue_width: int, out_lanes: int) -> dict[str, int]:
    text = log.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"PASS M2B banked multi-source issue_width={issue_width} "
        rf"out_lanes={out_lanes} "
        r"real_cases=(\d+) sources=(\d+) issue_beats=(\d+) "
        r"ideal_ready_cycles=(\d+) latency_cycles=(\d+) full_wall_cycles=(\d+) "
        r"empty=(\d+) protocol_injections=(\d+) directed_reset_cases=(\d+) wrap_cases=(\d+)"
    )
    match = pattern.search(text)
    if not match or re.search(r"Assertion failed|\bFAIL\b|Fatal:", text):
        raise ValueError(f"clean VCS PASS marker absent: {log}")
    cases, sources, beats, ideal, latency, full_wall, empty, injections, resets, wraps = map(
        int, match.groups()
    )
    if (
        cases != EXPECTED_CASES
        or sources != EXPECTED_SOURCES
        or injections < 2
        or resets < 2
        or wraps < 1
    ):
        raise ValueError(f"unexpected VCS accounting in {log}")
    return {
        "real_cases": cases,
        "sources": sources,
        "issue_beats": beats,
        "ideal_ready_cycles": ideal,
        "command_to_output_valid_cycles": latency,
        "command_to_output_fire_cycles": full_wall,
        "empty_cases": empty,
        "protocol_injections": injections,
        "directed_reset_cases": resets,
        "wrap_cases": wraps,
    }


def require_dc_pass(run_dir: Path) -> dict[str, float]:
    qor = (run_dir / "reports/qor.rpt").read_text(encoding="utf-8", errors="replace")
    area = (run_dir / "reports/area.rpt").read_text(encoding="utf-8", errors="replace")

    def value(label: str) -> float:
        match = re.search(rf"{re.escape(label)}:\s+(-?\d+(?:\.\d+)?)", qor)
        if not match:
            raise ValueError(f"missing {label} in {run_dir / 'reports/qor.rpt'}")
        return float(match.group(1))

    metrics = {
        "logic_levels": value("Levels of Logic"),
        "critical_path_ns": value("Critical Path Length"),
        "slack_ns": value("Critical Path Slack"),
        "cell_area_um2": value("Cell Area"),
        "violating_paths": value("No. of Violating Paths"),
    }
    macro_match = re.search(r"Number of macros/black boxes:\s+(\d+)", area)
    if not macro_match or int(macro_match.group(1)) != 0:
        raise ValueError(f"premacro DC unexpectedly contains a macro/black box: {run_dir}")
    metrics["macro_black_boxes"] = 0.0
    if metrics["slack_ns"] < 0 or metrics["violating_paths"] != 0:
        raise ValueError(f"DC timing is not clean: {run_dir}")
    timing_check = (run_dir / "reports/check_timing_postcompile.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    if re.search(r"Warning:|Error:", timing_check):
        raise ValueError(f"postcompile timing checks contain warnings/errors: {run_dir}")
    design_check = (run_dir / "reports/check_design_postcompile.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    if re.search(r"Warning:|Error:", design_check):
        raise ValueError(f"postcompile design checks contain warnings/errors: {run_dir}")
    constraints = (run_dir / "reports/constraint_violators.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    if re.search(r"VIOLATED|\bError:|max_leakage_power", constraints):
        raise ValueError(f"DC timing/design-rule constraint report is not clean: {run_dir}")
    run_manifest = json.loads((run_dir / "dc_run_manifest.json").read_text(encoding="utf-8"))
    sdc = run_manifest.get("paths", {}).get("SDC_FILE", {})
    library = run_manifest.get("paths", {}).get("LIB_DB", {})
    min_library = run_manifest.get("paths", {}).get("MIN_LIB_DB", {})
    if (
        run_manifest.get("mode") != "dc"
        or run_manifest.get("operating_condition") != "ssg0p9v125c"
        or float(run_manifest.get("clock_period_ns", 0)) != 3.0
        or float(run_manifest.get("dc_hold_uncertainty_ns", 0)) != 0.1
        or float(run_manifest.get("dc_hold_report_uncertainty_ns", 0)) != 0.09
        or run_manifest.get("ppa_admission") != "0"
        or run_manifest.get("macro_dbs") != []
        or Path(sdc.get("path", "")).name != "date_dual_core.sdc"
        or not re.fullmatch(r"[0-9a-f]{64}", sdc.get("sha256", ""))
        or Path(library.get("path", "")).name != "tcbn28hpcplusbwp35p140ssg0p9v125c.db"
        or library.get("sha256") != "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"
        or Path(min_library.get("path", "")).name != "tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
        or min_library.get("sha256") != "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a"
    ):
        raise ValueError(f"DC input identity/scope is not the admitted premacro contract: {run_dir}")
    return metrics


def require_formality_pass(run_dir: Path, design_name: str) -> int:
    status = (run_dir / "reports/formality_status.txt").read_text(encoding="utf-8").strip()
    log = (run_dir / "formality.log").read_text(encoding="utf-8", errors="replace")
    unmatched = (run_dir / "reports/formality_unmatched.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    failing = (run_dir / "reports/formality_verify.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    passing_match = re.search(r"(\d+) Passing compare points", log)
    run_manifest = json.loads((run_dir / "formality_run_manifest.json").read_text(encoding="utf-8"))
    netlist = run_manifest.get("paths", {}).get("MAPPED_NETLIST", {})
    if (
        status != "PASS"
        or "Verification SUCCEEDED" not in log
        or not passing_match
        or "No unmatched points." not in unmatched
        or "No failing compare points." not in failing
        or run_manifest.get("mode") != "formality"
        or run_manifest.get("design_name") != design_name
        or run_manifest.get("operating_condition") != "ssg0p9v125c"
        or run_manifest.get("macro_dbs") != []
        or netlist.get("sha256") != sha256(run_dir / f"netlist/{design_name}_mapped.v")
    ):
        raise ValueError(f"Formality evidence is not clean: {run_dir}")
    return int(passing_match.group(1))


def require_ptsta_pass(
    run_dir: Path, *, design_name: str, operating_condition: str, corner_role: str,
    library_sha256: str,
) -> dict[str, float]:
    audit = json.loads((run_dir / "ptsta_artifact_audit.json").read_text(encoding="utf-8"))
    scope = (run_dir / "reports/ptsta_scope.rpt").read_text(encoding="utf-8")
    global_timing = (run_dir / "reports/ptsta_global_timing.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    constraints = (run_dir / "reports/ptsta_constraint_violators.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    setup = (run_dir / "reports/ptsta_timing_setup.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    hold = (run_dir / "reports/ptsta_timing_hold.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    run_manifest = json.loads((run_dir / "ptsta_run_manifest.json").read_text(encoding="utf-8"))
    paths = run_manifest.get("paths", {})
    netlist_path = Path(paths.get("MAPPED_NETLIST", {}).get("path", ""))
    sdc_path = Path(paths.get("MAPPED_SDC", {}).get("path", ""))

    def worst_slack(text: str, label: str) -> float:
        values = [float(value) for value in re.findall(r"slack \(MET\)\s+(-?\d+(?:\.\d+)?)", text)]
        if not values:
            raise ValueError(f"PrimeTime {label} report contains no met path: {run_dir}")
        return min(values)

    if (
        audit.get("status") != "ARTIFACTS_PRESENT_REVIEW_REQUIRED"
        or any(audit.get("fatal_patterns", {}).values())
        or "prelayout_no_spef" not in scope
        or f"operating_condition={operating_condition}" not in scope
        or f"corner_role={corner_role}" not in scope
        or "No setup violations found." not in global_timing
        or "No hold violations found." not in global_timing
        or re.search(r"VIOLATED|\bError:", constraints)
        or run_manifest.get("mode") != "ptsta"
        or run_manifest.get("design_name") != design_name
        or run_manifest.get("operating_condition") != operating_condition
        or run_manifest.get("corner_role") != corner_role
        or run_manifest.get("macro_dbs") != []
        or paths.get("LIB_DB", {}).get("sha256") != library_sha256
        or not netlist_path.is_file()
        or paths.get("MAPPED_NETLIST", {}).get("sha256") != sha256(netlist_path)
        or not sdc_path.is_file()
        or paths.get("MAPPED_SDC", {}).get("sha256") != sha256(sdc_path)
    ):
        raise ValueError(f"PrimeTime prelayout STA evidence is not clean: {run_dir}")
    return {
        "setup_slack_ns": worst_slack(setup, "setup"),
        "hold_slack_ns": worst_slack(hold, "hold"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--out-lanes", type=int, choices=(16, 96), default=16)
    args = parser.parse_args()
    repo = args.repo.resolve()
    run_root = args.run_root.resolve()

    out_lanes = args.out_lanes
    lane_suffix = "" if out_lanes == 16 else "_l96"
    source_paths = [
        "hw_autoresearch_nts07/rtl_qfit/qfit_local_banked_multisource_engine.sv",
        (
            "hw_autoresearch_nts07/rtl_qfit/qfit_local_banked_multisource_dc_tops.sv"
            if out_lanes == 16
            else "hw_autoresearch_nts07/rtl_qfit/qfit_local_banked_multisource_l96_dc_tops.sv"
        ),
        "hw_autoresearch_nts07/tb_qfit/tb_qfit_local_banked_multisource_engine.sv",
        "hw_autoresearch_nts07/verif_qfit/qfit_local_banked_multisource_engine_assertions.sv",
        (
            "hw_autoresearch_nts07/dc_handoff/filelists/date_local_banked_multisource.f"
            if out_lanes == 16
            else "hw_autoresearch_nts07/dc_handoff/filelists/date_local_banked_multisource_l96.f"
        ),
        "hw_autoresearch_nts07/dc_handoff/constraints/date_dual_core.sdc",
        "hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_local_banked_multisource_sva.sh",
        "hw_autoresearch_nts07/dc_handoff/run_dc.sh",
        "hw_autoresearch_nts07/dc_handoff/scripts/run_dc.tcl",
        "hw_autoresearch_nts07/dc_handoff/run_formality.sh",
        "hw_autoresearch_nts07/dc_handoff/scripts/run_formality.tcl",
        "hw_autoresearch_nts07/dc_handoff/scripts/write_synopsys_run_manifest.py",
        "hw_autoresearch_nts07/dc_handoff/scripts/audit_dc_artifacts.py",
        "hw_autoresearch_nts07/dc_handoff/scripts/build_m2b_milestone_manifest.py",
        "hw_autoresearch_nts07/system_simulator/scripts/build_m2b_real_tile_vectors.py",
        "hw_autoresearch_nts07/system_simulator/scripts/build_dual_line_tile_memory_trace.py",
        "hw_autoresearch_nts07/system_simulator/tests/test_m2b_real_tile_vectors.py",
    ]
    if out_lanes == 96:
        source_paths.extend(
            [
                "hw_autoresearch_nts07/system_simulator/scripts/analyze_m2c_bank_remap.py",
                "hw_autoresearch_nts07/system_simulator/tests/test_m2c_bank_remap.py",
                "hw_autoresearch_nts07/dc_handoff/constraints/date_dual_pt.sdc",
                "hw_autoresearch_nts07/dc_handoff/run_ptsta.sh",
                "hw_autoresearch_nts07/dc_handoff/scripts/run_ptsta.tcl",
                "hw_autoresearch_nts07/dc_handoff/scripts/audit_synopsys_postrun.py",
                "hw_autoresearch_nts07/dc_handoff/scripts/report_dc_hold_guard.tcl",
                "hw_autoresearch_nts07/dc_handoff/scripts/rerun_dc_hold_reports.sh",
                "hw_autoresearch_nts07/docs/454_M2C_96Lane与PrimeTime收口_20260821.md",
            ]
        )
    sources = [artifact(repo / relative, repo) for relative in source_paths]
    source_set_sha = hashlib.sha256(
        json.dumps(
            [(item["path"], item["sha256"]) for item in sources],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    vector_dir = run_root / "dual_line_m2b_real_vectors_v2_20260821"
    vector_manifest = json.loads((vector_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("current_tiles.hex", "case_index.csv"):
        if sha256(vector_dir / name) != vector_manifest["sha256"][name]:
            raise ValueError(f"vector digest mismatch: {name}")

    evidence = [artifact(vector_dir / name) for name in ("manifest.json", "current_tiles.hex", "case_index.csv")]
    remap_summary: dict[str, Any] | None = None
    if out_lanes == 96:
        remap_dir = run_root / "m2c_bank_remap_all_tiles_20260821"
        remap_summary = json.loads((remap_dir / "bank_remap_dse.json").read_text(encoding="utf-8"))
        if remap_summary.get("status") != "PASS_REMAP_BELOW_SURVIVAL_GATE":
            raise ValueError("M2C remap DSE did not close with the expected negative result")
        evidence.extend(
            [artifact(remap_dir / "bank_remap_dse.json"), artifact(remap_dir / "REPORT.md")]
        )
    variants: dict[str, Any] = {}
    for issue_width in ISSUE_WIDTHS:
        vcs_dir = run_root / f"local_banked_multisource_p{issue_width}{lane_suffix}_vcs_sva_20260821"
        dc_dir = run_root / f"local_banked_multisource_p{issue_width}{lane_suffix}_dc_3ns_20260821"
        vcs_metrics = require_vcs_pass(vcs_dir / "simulation.log", issue_width, out_lanes)
        verify_sha256_file(vcs_dir / "evidence.sha256")
        dc_metrics = require_dc_pass(dc_dir)
        design_name = f"qfit_local_banked_multisource_p{issue_width}{lane_suffix}_top"
        passing_points = require_formality_pass(dc_dir, design_name)
        ptsta_metrics: dict[str, float] | None = None
        if out_lanes == 96:
            ptsta_dir = run_root / f"local_banked_multisource_p{issue_width}_l96_ptsta_3ns_20260821"
            ptsta_metrics = require_ptsta_pass(
                ptsta_dir, design_name=design_name, operating_condition="ssg0p9v125c",
                corner_role="setup",
                library_sha256="79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
            )
            hold_dir = run_root / f"local_banked_multisource_p{issue_width}_l96_ptsta_fast_hold_20260821"
            fast_hold_metrics = require_ptsta_pass(
                hold_dir, design_name=design_name, operating_condition="ffg1p05vm40c",
                corner_role="hold",
                library_sha256="a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
            )
            ptsta_metrics["fast_corner_hold_slack_ns"] = fast_hold_metrics["hold_slack_ns"]
        variants[f"p{issue_width}"] = {
            "vcs": vcs_metrics,
            "dc": dc_metrics,
            "formality": {
                "passing_compare_points": passing_points,
                "failing_compare_points": 0,
                "unmatched_compare_points": 0,
            },
            "ptsta": ptsta_metrics,
        }
        for path in (
            vcs_dir / "compile.log",
            vcs_dir / "simulation.log",
            vcs_dir / "evidence.sha256",
            dc_dir / "dc.log",
            dc_dir / "dc_run_manifest.json",
            dc_dir / "reports/qor.rpt",
            dc_dir / "reports/area.rpt",
            dc_dir / "reports/check_timing_postcompile.rpt",
            dc_dir / "reports/check_design_postcompile.rpt",
            dc_dir / "reports/constraint_violators.rpt",
            dc_dir / "reports/references.rpt",
            dc_dir / f"netlist/qfit_local_banked_multisource_p{issue_width}{lane_suffix}_top_mapped.v",
            dc_dir / f"netlist/qfit_local_banked_multisource_p{issue_width}{lane_suffix}_top_mapped.sdc",
            dc_dir / f"netlist/qfit_local_banked_multisource_p{issue_width}{lane_suffix}_top.svf",
            dc_dir / "formality.log",
            dc_dir / "formality_run_manifest.json",
            dc_dir / "reports/formality_status.txt",
            dc_dir / "reports/formality_unmatched.rpt",
            dc_dir / "reports/formality_verify.rpt",
        ):
            evidence.append(artifact(path))
        if out_lanes == 96:
            for path in (
                dc_dir / "dc_hold_report_refresh.log",
                dc_dir / "reports/timing_setup.rpt",
                dc_dir / "reports/timing_hold.rpt",
            ):
                evidence.append(artifact(path))
            for path in (
                ptsta_dir / "ptsta.log",
                ptsta_dir / "ptsta_run_manifest.json",
                ptsta_dir / "ptsta_artifact_audit.json",
                ptsta_dir / "reports/ptsta_scope.rpt",
                ptsta_dir / "reports/ptsta_annotated_parasitics.rpt",
                ptsta_dir / "reports/ptsta_check_timing.rpt",
                ptsta_dir / "reports/ptsta_analysis_coverage.rpt",
                ptsta_dir / "reports/ptsta_global_timing.rpt",
                ptsta_dir / "reports/ptsta_timing_setup.rpt",
                ptsta_dir / "reports/ptsta_timing_hold.rpt",
                ptsta_dir / "reports/ptsta_constraint_violators.rpt",
                ptsta_dir / f"netlist/qfit_local_banked_multisource_p{issue_width}_l96_top.sdf",
            ):
                evidence.append(artifact(path))
            for path in (
                hold_dir / "ptsta.log",
                hold_dir / "ptsta_run_manifest.json",
                hold_dir / "ptsta_artifact_audit.json",
                hold_dir / "reports/ptsta_scope.rpt",
                hold_dir / "reports/ptsta_annotated_parasitics.rpt",
                hold_dir / "reports/ptsta_check_timing.rpt",
                hold_dir / "reports/ptsta_analysis_coverage.rpt",
                hold_dir / "reports/ptsta_global_timing.rpt",
                hold_dir / "reports/ptsta_timing_setup.rpt",
                hold_dir / "reports/ptsta_timing_hold.rpt",
                hold_dir / "reports/ptsta_constraint_violators.rpt",
                hold_dir / f"netlist/qfit_local_banked_multisource_p{issue_width}_l96_top.sdf",
            ):
                evidence.append(artifact(path))

    p1_beats = variants["p1"]["vcs"]["issue_beats"]
    p1_ideal = variants["p1"]["vcs"]["ideal_ready_cycles"]
    p1_latency = variants["p1"]["vcs"]["command_to_output_valid_cycles"]
    p1_full_wall = variants["p1"]["vcs"]["command_to_output_fire_cycles"]
    p1_area = variants["p1"]["dc"]["cell_area_um2"]
    for issue_width in ISSUE_WIDTHS:
        variant = variants[f"p{issue_width}"]
        variant["derived"] = {
            "issue_speedup_vs_p1": p1_beats / variant["vcs"]["issue_beats"],
            "ideal_ready_speedup_vs_p1": p1_ideal / variant["vcs"]["ideal_ready_cycles"],
            "command_to_output_valid_speedup_vs_p1": (
                p1_latency / variant["vcs"]["command_to_output_valid_cycles"]
            ),
            "command_to_output_fire_speedup_vs_p1": (
                p1_full_wall / variant["vcs"]["command_to_output_fire_cycles"]
            ),
            "bank_utilization": p1_beats / (issue_width * variant["vcs"]["issue_beats"]),
            "area_ratio_vs_p1": variant["dc"]["cell_area_um2"] / p1_area,
            "logic_issue_per_area_vs_p1": (
                p1_beats / variant["vcs"]["issue_beats"]
            ) / (variant["dc"]["cell_area_um2"] / p1_area),
        }

    payload = {
        "schema": (
            "dual_line_m2b_evidence_manifest_v1"
            if out_lanes == 16
            else "dual_line_m2c_l96_evidence_manifest_v1"
        ),
        "status": (
            "PASS_M2B_REAL_BITMAP_VCS_DC_FM_PREMACRO"
            if out_lanes == 16
            else "PASS_M2C_LOCAL_L96_REAL_BITMAP_VCS_DC_FM_PTSTA_PREMACRO"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": {
            "verified": (
                "real-bitmap source conservation and bank-conflict issue ratios with a deterministic "
                f"{out_lanes}-lane weight miter; Synopsys VCS/SVA; premacro 3 ns DC; "
                "RTL-to-gate Formality"
                + ("; prelayout PrimeTime STA without SPEF" if out_lanes == 96 else "")
            ),
            "not_verified": (
                "checkpoint-weight bit exactness, SRAM macro PPA, post-layout timing, full-network "
                "latency/energy, or system-level acceleration"
            ),
        },
        "selected_candidate": "p4" if out_lanes == 16 else None,
        "pareto_candidates": [] if out_lanes == 16 else ["p4", "p8"],
        "candidate_gate": (
            None
            if out_lanes == 16
            else "pending matched SRAM bandwidth, PTPX energy, and full-system objective"
        ),
        "out_lanes": out_lanes,
        "remap_dse": remap_summary,
        "git": {
            "head": git(["rev-parse", "HEAD"], repo),
            "branch": git(["branch", "--show-current"], repo),
            "dirty": bool(git(["status", "--porcelain"], repo)),
            "source_set_sha256": source_set_sha,
        },
        "vector_identity": vector_manifest,
        "variants": variants,
        "sources": sources,
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"PASS: wrote {args.output} "
        f"({len(sources)} sources, {len(evidence)} evidence files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
