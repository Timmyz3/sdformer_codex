#!/usr/bin/env python3
"""Build a fail-closed, content-addressed manifest for the dual-line M1 milestone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_MARKERS = {
    "selector": "PASS dual-line selector requests=",
    "streamer": "PASS dual-line source streamer commands=",
    "executor": "PASS dual-line tile executor commands=",
    "stateful": "PASS stateful dual-line tile commands=",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty artifact missing: {path}")
    label = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256(path)}


def git(args: list[str], root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.rstrip("\n")


def require_log_marker(path: Path, marker: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if marker not in text or re.search(r"Assertion failed|\bFAIL\b|Fatal:", text):
        raise ValueError(f"PASS marker absent or failure marker present: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-formality", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    run_root = args.run_root.resolve()

    source_paths = [
        "hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_tile_selector.sv",
        "hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_source_streamer.sv",
        "hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_tile_executor.sv",
        "hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_stateful_tile_top.sv",
        "hw_autoresearch_nts07/verif_qfit/qfit_dual_line_tile_selector_assertions.sv",
        "hw_autoresearch_nts07/verif_qfit/qfit_dual_line_source_streamer_assertions.sv",
        "hw_autoresearch_nts07/verif_qfit/qfit_dual_line_tile_executor_assertions.sv",
        "hw_autoresearch_nts07/verif_qfit/qfit_dual_line_stateful_tile_assertions.sv",
        "hw_autoresearch_nts07/tb_qfit/tb_qfit_dual_line_tile_selector.sv",
        "hw_autoresearch_nts07/tb_qfit/tb_qfit_dual_line_source_streamer.sv",
        "hw_autoresearch_nts07/tb_qfit/tb_qfit_dual_line_tile_executor.sv",
        "hw_autoresearch_nts07/tb_qfit/tb_qfit_dual_line_stateful_tile.sv",
        "hw_autoresearch_nts07/dc_handoff/filelists/date_dual_line_stateful_tile.f",
        "hw_autoresearch_nts07/dc_handoff/run_dc.sh",
        "hw_autoresearch_nts07/dc_handoff/run_formality.sh",
    ]
    evidence: list[dict[str, Any]] = []
    for name, marker in PASS_MARKERS.items():
        log = run_root / f"dual_line_{'stateful_tile' if name == 'stateful' else ('source_streamer' if name == 'streamer' else ('tile_executor' if name == 'executor' else 'tile_selector'))}_vcs_sva_20260821" / "simulation.log"
        require_log_marker(log, marker)
        evidence.append(artifact(log))

    dc_dir = run_root / "dual_line_stateful_tile_dc_3ns_20260821"
    for relative in [
        "dc.log",
        "dc_run_manifest.json",
        "reports/qor.rpt",
        "reports/area.rpt",
        "reports/check_timing_postcompile.rpt",
        "reports/constraint_violators.rpt",
        "netlist/qfit_dual_line_stateful_tile_top_mapped.v",
        "netlist/qfit_dual_line_stateful_tile_top_mapped.sdc",
        "netlist/qfit_dual_line_stateful_tile_top.svf",
    ]:
        evidence.append(artifact(dc_dir / relative))

    qor = (dc_dir / "reports/qor.rpt").read_text(encoding="utf-8", errors="replace")
    if not re.search(r"Critical Path Slack:\s+(?:0\.00|[1-9]\d*\.\d+)", qor):
        raise ValueError("DC QoR does not contain a nonnegative critical-path slack")
    if not re.search(r"No\. of Violating Paths:\s+0\.00", qor):
        raise ValueError("DC QoR contains timing violations")

    formality_status = dc_dir / "reports/formality_status.txt"
    if args.require_formality:
        if formality_status.read_text(encoding="utf-8").strip() != "PASS":
            raise ValueError("Formality status is not PASS")
        for relative in [
            "formality.log",
            "formality_run_manifest.json",
            "reports/formality_status.txt",
            "reports/formality_unmatched.rpt",
            "reports/formality_verify.rpt",
        ]:
            evidence.append(artifact(dc_dir / relative))

    for relative in [
        "dual_line_source_issue_envelope_s10_20260821/source_issue_envelope.json",
        "dual_line_source_issue_envelope_s10_20260821/REPORT.md",
        "dual_line_full_network_trace_s1_s10_analysis_20260821/analysis.json",
        "dual_line_full_network_trace_s1_s10_analysis_20260821/operator_comparison.csv",
    ]:
        evidence.append(artifact(run_root / relative))

    sources = [artifact(repo / path, repo) for path in source_paths]
    source_set_sha256 = hashlib.sha256(
        json.dumps(
            [(entry["path"], entry["sha256"]) for entry in sources],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": "dual_line_m1_evidence_manifest_v1",
        "status": "PASS_M1_SYNOPYS_RTL_DC_FM" if args.require_formality else "PASS_M1_VCS_DC_FM_PENDING",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": {
            "verified": (
                "RTL/SVA behavior, RTL-to-gate equivalence, premacro 3 ns synthesis, source-issue envelope"
                if args.require_formality else
                "RTL/SVA behavior, premacro 3 ns synthesis, source-issue envelope; Formality pending"
            ),
            "not_verified": "SRAM/DRAM timing, post-layout timing, system latency, or measured system energy",
        },
        "git": {
            "head": git(["rev-parse", "HEAD"], repo),
            "branch": git(["branch", "--show-current"], repo),
            "dirty": bool(git(["status", "--porcelain"], repo)),
            # This digest is useful for tracked modifications, but an untracked
            # source has no git diff.  source_set_sha256 below is the actual
            # fail-closed content identity for every listed source.
            "tracked_source_diff_sha256": hashlib.sha256(
                git(["diff", "--binary", "--", *source_paths], repo).encode("utf-8")
            ).hexdigest(),
            "source_set_sha256": source_set_sha256,
        },
        "sources": sources,
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: wrote {args.output} ({len(payload['sources'])} sources, {len(evidence)} evidence files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
