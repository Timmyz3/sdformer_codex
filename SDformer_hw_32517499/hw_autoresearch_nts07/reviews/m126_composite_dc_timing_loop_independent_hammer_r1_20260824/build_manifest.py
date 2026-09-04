#!/usr/bin/env python3
import hashlib
import pathlib


review = pathlib.Path(__file__).resolve().parent
relative_files = [
    "README.md",
    "RUN_COMPLETE.txt",
    "audit_m126_timing_loop_review.py",
    "build_m126_registered_fault_barrier_delta.py",
    "build_manifest.py",
    "check_timing_only.tcl",
    "delta_dc.f",
    "delta_dc/check_design.rpt",
    "delta_dc/check_timing.rpt",
    "delta_dc/dc.raw.log",
    "delta_dc/dc.rc",
    "delta_dc/resources.rpt",
    "delta_generation.log",
    "delta_vcs.f",
    "delta_vcs/assert.report",
    "delta_vcs/compile.raw.log",
    "delta_vcs/compile.rc",
    "delta_vcs/sim.raw.log",
    "delta_vcs/sim.rc",
    "m125_busy_boolean_equivalence_exhaustive.json",
    "m125_registered_state_busy_delta.sv",
    "m126_composite_dc_timing_loop_independent_audit.json",
    "m126_registered_fault_barrier_delta.sv",
    "original_dc.f",
    "original_dc/check_design.rpt",
    "original_dc/check_timing.rpt",
    "original_dc/dc.raw.log",
    "original_dc/dc.rc",
    "original_dc/resources.rpt",
    "preflight_sha_checks.txt",
    "reproduction_outputs.sha256",
    "run_m126_loop_reproduction.sh",
    "source_evidence.sha256",
]
lines = []
for relative in relative_files:
    path = review / relative
    if not path.is_file():
        raise SystemExit("missing manifest input: " + relative)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(digest + "  " + relative)
(review / "manifest.sha256").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
print("PASS manifest entries=%d" % len(lines))
