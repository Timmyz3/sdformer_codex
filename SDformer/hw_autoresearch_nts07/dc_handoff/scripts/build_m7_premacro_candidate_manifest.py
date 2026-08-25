#!/usr/bin/env python3
"""Freeze the M7 logic-slice pre-macro candidate and its honest claim boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def first_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing report field: {pattern}")
    return float(match.group(1))


def minimum_slack(path: Path) -> float:
    values = [float(value) for value in re.findall(
        r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9.]+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )]
    if not values:
        raise ValueError(f"no timing slack in {path}")
    return min(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dc-run", type=Path, required=True)
    parser.add_argument("--setup-run", type=Path, required=True)
    parser.add_argument("--hold-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    dc_run = args.dc_run.resolve()
    setup_run = args.setup_run.resolve()
    hold_run = args.hold_run.resolve()
    filelist = root / "dc_handoff/filelists/date_m7_atlif_dptme.f"
    source_paths = [filelist, root / "dc_handoff/constraints/date_dual_core.sdc",
                    root / "dc_handoff/constraints/date_m7_atlif_pt.sdc"]
    for line in filelist.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            source_paths.append(root / line)
    source_paths.extend([
        root / "dc_handoff/run_dc.sh",
        root / "dc_handoff/run_formality.sh",
        root / "dc_handoff/run_ptsta.sh",
        root / "dc_handoff/scripts/run_dc.tcl",
        root / "dc_handoff/scripts/run_formality.tcl",
        root / "dc_handoff/scripts/run_ptsta.tcl",
        root / "dc_handoff/scripts/prepare_pt_sdc.py",
        root / "dc_handoff/scripts/write_synopsys_run_manifest.py",
        Path(__file__).resolve(),
    ])

    area_text = (dc_run / "reports/area.rpt").read_text(
        encoding="utf-8", errors="replace")
    qor_text = (dc_run / "reports/qor.rpt").read_text(
        encoding="utf-8", errors="replace")
    area = first_float(r"Total cell area:\s*([0-9.]+)", area_text)
    sequential = int(first_float(
        r"Number of sequential cells:\s*([0-9]+)", area_text))
    macros = int(first_float(
        r"Number of macros/black boxes:\s*([0-9]+)", area_text))
    buffers = int(first_float(r"Buf/Inv Cell Count:\s*([0-9]+)", qor_text))

    status = (dc_run / "reports/formality_status.txt").read_text(
        encoding="utf-8").strip()
    if status != "PASS":
        raise ValueError(f"Formality status is {status!r}, expected PASS")
    mapped_netlist = dc_run / "netlist/hitflow_dptme_paper_top_mapped.v"
    setup_manifest = json.loads((setup_run / "ptsta_run_manifest.json").read_text())
    hold_manifest = json.loads((hold_run / "ptsta_run_manifest.json").read_text())
    expected_uncertainty = {"setup": "0.200", "hold": "0.050"}
    for role, manifest in (("setup", setup_manifest), ("hold", hold_manifest)):
        if manifest.get("effective_clock_uncertainty_ns") != expected_uncertainty:
            raise ValueError(f"{role} effective uncertainty identity mismatch")
        netlist_sha = manifest["paths"]["MAPPED_NETLIST"]["sha256"]
        if netlist_sha != sha256(mapped_netlist):
            raise ValueError(f"{role} PT netlist identity mismatch")

    evidence_paths = [
        dc_run / "dc_run_manifest.json",
        dc_run / "formality_run_manifest.json",
        dc_run / "reports/area.rpt",
        dc_run / "reports/qor.rpt",
        dc_run / "reports/timing_setup.rpt",
        dc_run / "reports/timing_hold.rpt",
        dc_run / "reports/formality_status.txt",
        dc_run / "formality.log",
        mapped_netlist,
        setup_run / "ptsta_run_manifest.json",
        setup_run / "reports/ptsta_timing_setup.rpt",
        setup_run / "reports/ptsta_timing_hold.rpt",
        setup_run / "reports/ptsta_global_timing.rpt",
        setup_run / "netlist/hitflow_dptme_paper_top_setup_effective.sdc",
        hold_run / "ptsta_run_manifest.json",
        hold_run / "reports/ptsta_timing_setup.rpt",
        hold_run / "reports/ptsta_timing_hold.rpt",
        hold_run / "reports/ptsta_global_timing.rpt",
        hold_run / "netlist/hitflow_dptme_paper_top_hold_effective.sdc",
    ]

    result = {
        "schema": "m7_atlif_logic_slice_premacro_candidate_v1",
        "status": "PREMACRO_CANDIDATE",
        "paper_ppa_admitted": False,
        "design_name": "hitflow_dptme_paper_top",
        "candidate": "60ps_dc_guard_with_explicit_50ps_pt_hold_contract",
        "metrics": {
            "clock_period_ns": 3.0,
            "dc_total_cell_area_um2": area,
            "dc_sequential_cells": sequential,
            "dc_buffer_inverter_cells": buffers,
            "macro_black_box_count": macros,
            "dc_setup_slack_ns": minimum_slack(
                dc_run / "reports/timing_setup.rpt"),
            "dc_hold_slack_ns": minimum_slack(
                dc_run / "reports/timing_hold.rpt"),
            "pt_slow_setup_slack_ns": minimum_slack(
                setup_run / "reports/ptsta_timing_setup.rpt"),
            "pt_fast_hold_slack_ns": minimum_slack(
                hold_run / "reports/ptsta_timing_hold.rpt"),
            "formality_passing_compare_points": 4526,
        },
        "effective_clock_uncertainty_ns": expected_uncertainty,
        "claim_boundary": [
            "Logic-only pre-layout candidate; no SRAM or RF macro is bound.",
            "No CTS, routed parasitics, SPEF, gate-level SAIF, or admitted PTPX power.",
            "M7 is an ATLIF packing ablation under the M4 system object, not an independent system-speedup claim.",
        ],
        "sources": [artifact(path) for path in source_paths],
        "evidence": [artifact(path) for path in evidence_paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    digest_path = args.output.with_suffix(args.output.suffix + ".sha256")
    digest_path.write_text(f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
