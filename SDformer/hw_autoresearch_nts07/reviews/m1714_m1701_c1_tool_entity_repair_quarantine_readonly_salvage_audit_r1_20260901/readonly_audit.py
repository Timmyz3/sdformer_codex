#!/usr/bin/env python3
"""Read-only M1714 audit of the sealed M1701 DC quarantine."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
Q = HW / "dc_handoff/runs/m1701_m1695_c1_tool_entity_repair_dc_r1_20260901.failed_or_incomplete.2502881.quarantine"
REFERENCE = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901/k8/dc.log"
MANIFEST_SHA = "f132ca694a747e2da51708fb03f2ba6c84360606b4d38d2cc2e97998f9f3a022"
OUTER_SHA = "a65f2901b4ab4339a94bb032b9412b652a77afc50d1c72b403c8bd44d15f55a6"
GUI_ERROR = "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
AREA_BASELINE = 152898.625984
AREA_CEILING = 168188.4885824


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal():
    need(Q.is_dir() and not Q.is_symlink(), "quarantine directory identity")
    manifest = Q / "SHA256SUMS"
    outer = Q / "SHA256SUMS.seal.sha256"
    need(sha(manifest) == MANIFEST_SHA, "manifest SHA drift")
    need(sha(outer) == OUTER_SHA, "outer file SHA drift")
    need(outer.read_text().split() == [MANIFEST_SHA, "SHA256SUMS"],
         "outer seal content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest member")
        path = Q / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "manifest member drift: " + name)
        listed.add(name)
    actual = set()
    for path in Q.rglob("*"):
        need(not path.is_symlink(), "symlink in quarantine")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(Q).as_posix())
    need(actual == listed, "sealed population drift")


def kv(path):
    result = {}
    for row in Path(path).read_text().splitlines():
        if "=" in row:
            key, value = row.split("=", 1)
            need(key not in result, "duplicate key: " + key)
            result[key] = value
    return result


def timing(name):
    value = kv(Q / "reports" / name)
    need(value["status"] == "MET", name + " not MET")
    need(int(value["violating_paths"]) == 0, name + " violating paths")
    wns, tns = float(value["wns_ns"]), float(value["tns_ns"])
    need(math.isfinite(wns) and math.isfinite(tns) and wns >= 0 and tns == 0,
         name + " numeric failure")
    return {"wns_ns": wns, "tns_ns": tns,
            "violating_paths": int(value["violating_paths"]),
            "status": value["status"]}


def main():
    verify_seal()
    need((Q / "dc.rc").read_text() == "0\n", "dc return code")
    terminal = kv(Q / "TCL_INTERNAL_COMPLETE.txt")
    need(terminal.get("status") == "M1695_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED",
         "Tcl terminal marker")
    need(terminal.get("functional_rtl_modified") == "false", "functional identity")
    need(terminal.get("mapped_identity_modified") == "true", "mapped identity")
    need(terminal.get("formality_required") == "true", "Formality gate")
    need(terminal.get("independent_pt_required") == "true", "PT gate")

    summaries = {
        "setup_prehold": timing("setup_prehold_summary_machine.txt"),
        "hold_prehold": timing("hold_prehold_summary_machine.txt"),
        "setup_posthold": timing("setup_posthold_summary_machine.txt"),
        "hold_posthold": timing("hold_posthold_summary_machine.txt"),
    }
    area_text = (Q / "reports/area_posthold.rpt").read_text()
    match = re.search(r"Total cell area:\s*([0-9.]+)", area_text)
    need(match, "area absent")
    area = float(match.group(1))
    need(0 < area <= AREA_CEILING, "area ceiling")

    macro = kv(Q / "reports/macro_binding_audit.txt")
    need(macro.get("status") == "PASS_M1695_RESOLVED_LIBRARY_MACRO_STRUCTURE",
         "macro status")
    need(macro.get("macro_count_pre") == "9" and macro.get("macro_count_post") == "9",
         "macro count")
    mapped = next((Q / "netlist").glob("*_mapped.v"))
    need(mapped.read_text(errors="replace").count("TS1N28HPCPHVTB128X128M4S") == 9,
         "mapped macro population")

    qor = (Q / "reports/qor_posthold.rpt").read_text()
    drc = re.search(r"Nets With Violations:\s*([0-9.]+)", qor)
    need(drc and int(float(drc.group(1))) == 0, "QoR DRC")
    constraints = (Q / "reports/constraint_design_rules_posthold.rpt").read_text()
    need(constraints.count("This design has no violated constraints.") == 5,
         "design-rule report")
    for name in ("constraint_setup_posthold_all.rpt", "constraint_hold_posthold_all.rpt"):
        need("This design has no violated constraints." in (Q / "reports" / name).read_text(),
             name + " violation")

    log = (Q / "dc.log").read_text(errors="replace")
    need(log.count(GUI_ERROR) == 1, "GUI init signature cardinality")
    need(REFERENCE.is_file() and GUI_ERROR in REFERENCE.read_text(errors="replace"),
         "known Synopsys environment signature reference")
    need("Memory usage for this session" in log and log.rstrip().endswith("Thank you..."),
         "normal dc_shell termination")
    # The exact M1701 fatal regex has only the GUI initialization false positive.
    gate = re.compile(r"(^|\s)(Error|Fatal):|LINK-[0-9]+|unresolved (reference|design|cell)|unable to resolve|combinational[ _-]*loop|timing[ _-]*loop|\((TIM-209|OPT-150)\)", re.I | re.M)
    gate_matches = [match.group(0).strip() for match in gate.finditer(log)]
    need(len(gate_matches) == 1 and "Error:" in gate_matches[0],
         "unexpected M1701 fatal-regex match")
    # Broader keywords are present only in echoed guard source and informational
    # `Checking loops`; no guard fired because Tcl reached its terminal marker.
    echoed_guards = [row for row in log.splitlines()
                     if re.match(r"\s*(?:if .*\{|error \"|set link_status|redirect .*link\.rpt)", row)]
    need(any('error "M1695_FAIL' in row for row in echoed_guards), "Tcl error guard echo")
    need(any("link_status" in row for row in echoed_guards), "Tcl link guard echo")
    need("Fatal:" not in log and not re.search(r"LINK-[0-9]+", log), "true fatal/link id")
    need(not re.search(r"unable to resolve|combinational[ _-]*loop|timing[ _-]*loop", log, re.I),
         "true unresolved/loop diagnostic")
    link = (Q / "reports/link.rpt").read_text(errors="replace")
    need(not re.search(r"unresolved|unable to resolve|LINK-[0-9]+|Fatal:|Error:", link, re.I),
         "link report diagnostic")

    print(json.dumps({
        "status": "PASS_SALVAGE_CANDIDATE_ONLY",
        "quarantine_manifest_sha256": MANIFEST_SHA,
        "quarantine_outer_file_sha256": OUTER_SHA,
        "dc_rc": 0,
        "tcl_internal_complete": True,
        "timing": summaries,
        "area_um2": area,
        "area_baseline_um2": AREA_BASELINE,
        "area_overhead_percent": (area / AREA_BASELINE - 1.0) * 100.0,
        "area_ceiling_um2": AREA_CEILING,
        "macro_count": 9,
        "drc_violating_nets": 0,
        "fatal_regex_matches": 1,
        "fatal_regex_match_class": "fixed_synopsys_gui_init_environment_false_positive",
        "echoed_tcl_guard_lines": len(echoed_guards),
        "true_unresolved_loop_fatal": False,
        "formality_required": True,
        "independent_pt_required": True,
        "quarantine_modified_or_promoted": False,
        "eda_runs": 0,
        "paper_citable_now": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
