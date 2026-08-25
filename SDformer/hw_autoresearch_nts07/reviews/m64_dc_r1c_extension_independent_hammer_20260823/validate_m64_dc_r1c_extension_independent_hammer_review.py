#!/usr/bin/env python3
"""Fail-closed validator for the independent M64 r1c DC/STA extension review.

The validator does not invoke Synopsys tools and never writes the producer run.
It independently verifies both sealed manifests, parses the timing/area reports
and mapped netlist, enforces claim boundaries, and checks the review receipt.
"""

from __future__ import print_function

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUN = HW / "dc_handoff/runs/m64_parent_selector_dc_3p000ns_r1c_20260823"
REVIEW = HERE / "m64_dc_r1c_extension_independent_hammer_review.json"
RECEIPT = HERE / "m64_dc_r1c_extension_independent_hammer_validation_receipt.json"

EXPECTED = {
    "snapshot_manifest": "47ebf8417337a498cbeddde267ab037f409968cdb95d50f06efb620cd989abc2",
    "output_manifest": "d9d3469f8bf3a0cf3dcc2dcce29d4bc2fbf8443f25d62ab1f464c243ebcf631b",
    "tool_binary_record": "dd3008862088d0ad88e2102dda5232aa26f80fae0bd107c273fc4df709b36ab1",
    "run_complete": "c40a68b6fe593d32cfc1d20518877d68f6ed10a6161ebdb61d58b3fd97394b0c",
    "dc_internal": "949d917e110afb8e42bd2b09a20d4848cabded2e1e27567d27f67a9a33ef86a2",
    "dc_raw": "5bfb7398a684e43b8086c2c8d3950cb59ebf748d8291cabdf4c1dadc1f1c097e",
    "mapped_v": "2c7d23d0d605bdc920f5fb26d6d3fb9d83f141f31d9a4de76ff9c87c2220bcb3",
    "mapped_sdc": "1fd1fe7e2f8e94b118da5eac24e932b4a312870c6b3f096adde815d1f465742c",
    "area": "561c41c1d9c6f1f2ccf6dc6ba7d1b53314055c31b26ed2d4932541011ca7699e",
    "qor": "a150bd87b127d1902b56b47d095f1bfb8ebe5ebd4501541106442b835fbf44cb",
    "setup": "d4da64321293542c0b404c4f2ed92709a72d9a9a784fd33bccc1ec10b8a52a1b",
    "hold": "23ac1316248c4ba46f19c304c87eaf3157e73d20cd62196b07bd686286f206b0",
    "check_timing_post": "94bf7675a38cf2f7e109edbc8dc51e1112bf6d2b3427362c2e14197d434eaf38",
    "references": "32cc59beab21ef74eabef99e065cd9b168c8e8e33d3da56999ade4eccc953194",
    "resources": "98c9eade23a6f82dcb07d574eaa662b9d086bbfe39c38523b1d101697796b355",
    "violators": "a1c6472efd015f45680a3a4118474d01aeceb414387a643f594a6893983ef2a5",
}

PATHS = {
    "snapshot_manifest": RUN / "snapshot.sha256",
    "output_manifest": RUN / "output.sha256",
    "tool_binary_record": RUN / "dc.binary.sha256",
    "run_complete": RUN / "RUN_COMPLETE.txt",
    "dc_internal": RUN / "DC_INTERNAL_COMPLETE.txt",
    "dc_raw": RUN / "dc.raw.log",
    "mapped_v": RUN / "netlist/qfit_adaptive_parent_selector_p256_mapped.v",
    "mapped_sdc": RUN / "netlist/qfit_adaptive_parent_selector_p256_mapped.sdc",
    "area": RUN / "reports/area.rpt",
    "qor": RUN / "reports/qor.rpt",
    "setup": RUN / "reports/timing_setup.rpt",
    "hold": RUN / "reports/timing_hold.rpt",
    "check_timing_pre": RUN / "reports/check_timing_precompile.rpt",
    "check_timing_post": RUN / "reports/check_timing_postcompile.rpt",
    "check_design_post": RUN / "reports/check_design_postcompile.rpt",
    "references": RUN / "reports/references_postcompile.rpt",
    "resources": RUN / "reports/resources_postcompile.rpt",
    "violators": RUN / "reports/constraint_violators.rpt",
    "snapshot_sdc": RUN / "snapshot/hw_autoresearch_nts07/dc_handoff/constraints/date_m64_parent_selector_3ns.sdc",
    "snapshot_tcl": RUN / "snapshot/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m64_parent_selector_exact_snapshot.tcl",
    "dc_rc": RUN / "dc.rc",
    "version_rc": RUN / "dc.version.rc",
    "version_log": RUN / "dc.version.raw.log",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + value)))


def text(path):
    return Path(path).read_text(encoding="utf-8", errors="strict")


def exact_float(body, label):
    match = re.search(r"^" + re.escape(label) + r":\s+([0-9.]+)\s*$",
                      body, flags=re.MULTILINE)
    require(match is not None, "missing numeric report field: " + label)
    return float(match.group(1))


def validate_pinned_hashes():
    for name, expected in EXPECTED.items():
        observed = sha256_path(PATHS[name])
        require(observed == expected,
                "{} SHA drift {} != {}".format(name, observed, expected))


def validate_manifest(manifest, root, minimum_lines, required):
    lines = text(manifest).splitlines()
    require(len(lines) >= minimum_lines, "manifest unexpectedly short")
    root_resolved = str(root.resolve()) + "/"
    seen = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (\./.+)", line)
        require(match is not None, "malformed manifest line: " + line[:80])
        expected, relative = match.groups()
        target = (root / relative[2:]).resolve()
        require(str(target).startswith(root_resolved),
                "manifest path escapes root: " + relative)
        require(relative not in seen, "duplicate manifest path: " + relative)
        seen.add(relative)
        require(target.is_file(), "manifest target missing: " + relative)
        require(sha256_path(target) == expected,
                "manifest target SHA drift: " + relative)
    for item in required:
        require(item in seen, "required manifest path absent: " + item)
    return len(lines)


def validate_manifests():
    snapshot_count = validate_manifest(
        PATHS["snapshot_manifest"], RUN / "snapshot", 6,
        {
            "./hw_autoresearch_nts07/dc_handoff/constraints/date_m64_parent_selector_3ns.sdc",
            "./hw_autoresearch_nts07/dc_handoff/filelists/date_m64_parent_selector_dc.f",
            "./hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m64_parent_selector_exact_snapshot.tcl",
            "./hw_autoresearch_nts07/rtl_m64/qfit_adaptive_parent_selector_p256.sv",
            "./library/tcbn28hpcplusbwp35p140ffg1p05vm40c.db",
            "./library/tcbn28hpcplusbwp35p140ssg0p9v125c.db",
        })
    output_count = validate_manifest(
        PATHS["output_manifest"], RUN, 40,
        {
            "./RUN_COMPLETE.txt", "./DC_INTERNAL_COMPLETE.txt",
            "./dc.raw.log", "./dc.rc", "./dc.version.raw.log",
            "./dc.version.rc", "./snapshot.sha256",
            "./netlist/qfit_adaptive_parent_selector_p256_mapped.v",
            "./netlist/qfit_adaptive_parent_selector_p256_mapped.sdc",
            "./reports/area.rpt", "./reports/qor.rpt",
            "./reports/timing_setup.rpt", "./reports/timing_hold.rpt",
            "./reports/check_timing_postcompile.rpt",
        })
    require(snapshot_count == 6, "snapshot manifest file count drift")
    require(output_count == 44, "output manifest file count drift")


def validate_terminal_and_tool():
    terminal = text(PATHS["run_complete"])
    for line in (
            "status=PASS_EXACT_SNAPSHOT_M64_DC_STA",
            "scope=standalone_online_parent_selector_p256",
            "clock_period_ns=3.000",
            "physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO",
            "paper_ppa_ready=false", "system_speedup_admitted=false",
            "power_or_energy_admitted=false"):
        require(line in terminal, "terminal boundary missing: " + line)
    require(text(PATHS["dc_rc"]).strip() == "0", "DC return code is not zero")
    require(text(PATHS["version_rc"]).strip() == "1",
            "version-probe RC drift; review explicitly records RC=1")
    require("V-2023.12-SP3" in text(PATHS["version_log"]),
            "tool version absent")
    binary_record = text(PATHS["tool_binary_record"]).strip()
    require(binary_record ==
            "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2  "
            "/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell",
            "tool binary identity drift")


def validate_constraints():
    sdc = text(PATHS["snapshot_sdc"])
    mapped = text(PATHS["mapped_sdc"])
    tcl = text(PATHS["snapshot_tcl"])
    raw = text(PATHS["dc_raw"])
    require("# rst_core is synchronous" in sdc,
            "synchronous reset contract missing")
    require("[remove_from_collection [all_inputs] [get_ports clk_core]]" in sdc,
            "all data inputs are not constrained")
    require("set_input_delay 0.250 -clock core_clk $data_inputs" in sdc,
            "snapshot input delay drift")
    require(re.search(r"set_input_delay -clock core_clk\s+0\.25\s+\[get_ports rst_core\]",
                      mapped) is not None,
            "mapped SDC lacks rst_core input delay")
    require("set_input_transition -max 0.1  [get_ports rst_core]" in mapped,
            "mapped SDC lacks rst_core max transition")
    require("set_input_transition -min 0.1  [get_ports rst_core]" in mapped,
            "mapped SDC lacks rst_core min transition")
    require("set_min_library $lib_db -min_version $min_lib_db" in tcl,
            "min-library pair absent")
    require("set_wire_load_model -name ZeroWireload" in tcl,
            "ZeroWireload contract absent")
    require("set_clock_uncertainty -hold 0.090" in tcl,
            "final hold uncertainty drift")
    require("tcbn28hpcplusbwp35p140ffg1p05vm40c.db" in raw,
            "ffg min library was not loaded")
    require("set_min_library" in raw, "set_min_library not echoed in DC log")
    forbidden = (
        "no clock-relative input delay",
        "is not constrained for maximum delay",
        "unconstrained endpoint",
    )
    for name in ("check_timing_pre", "check_timing_post"):
        report = text(PATHS[name])
        for token in forbidden:
            require(token.lower() not in report.lower(),
                    "{} contains {}".format(name, token))
        require("Checking no_input_delay" in report,
                name + " lacks no_input_delay audit")
        require("Checking unconstrained_endpoints" in report,
                name + " lacks unconstrained-endpoint audit")
    post = text(PATHS["check_timing_post"])
    require("Warning:" not in post and "Error:" not in post,
            "postcompile check_timing is not warning-clean")
    violators = text(PATHS["violators"])
    require(violators.count("This design has no violated constraints.") == 5,
            "constraint violator report is not clean for all five classes")


def validate_reports():
    area = text(PATHS["area"])
    qor = text(PATHS["qor"])
    setup = text(PATHS["setup"])
    hold = text(PATHS["hold"])
    expected_area = {
        "Number of ports": 1652.0,
        "Number of nets": 11984.0,
        "Number of cells": 9939.0,
        "Number of combinational cells": 8255.0,
        "Number of sequential cells": 1684.0,
        "Number of macros/black boxes": 0.0,
        "Number of buf/inv": 3553.0,
        "Number of references": 45.0,
        "Combinational area": 7454.033879,
        "Noncombinational area": 3395.448055,
        "Macro/Black Box area": 0.0,
        "Total cell area": 10849.481934,
    }
    for label, expected in expected_area.items():
        observed = exact_float(area, label)
        require(abs(observed - expected) < 1e-9,
                "{} drift {} != {}".format(label, observed, expected))
    require("Net Interconnect area:      undefined" in area,
            "undefined interconnect area not preserved")
    for token in ("Leaf Cell Count:               9939",
                  "Levels of Logic:              45.00",
                  "Critical Path Length:          1.88",
                  "Total Negative Slack:          0.00",
                  "No. of Violating Paths:        0.00",
                  "Macro Count:                      0"):
        require(token in qor, "QoR token missing: " + token.strip())
    setup_slacks = [float(value) for value in
                    re.findall(r"slack \(MET\)\s+([0-9.]+)", setup)]
    hold_slacks = [float(value) for value in
                   re.findall(r"slack \(MET\)\s+([0-9.]+)", hold)]
    require(len(setup_slacks) == 100, "setup path count drift")
    require(len(hold_slacks) == 100, "hold path count drift")
    require(min(setup_slacks) == 0.8967, "worst setup slack drift")
    require(min(hold_slacks) == 0.0104, "worst hold slack drift")
    for report, path_type in ((setup, "max"), (hold, "min")):
        require("Path Type: " + path_type in report,
                "timing path type drift")
        require("ZeroWireload" in report, "timing WLM drift")
        require("clock network delay (ideal)" in report,
                "timing clock is not explicitly ideal")
        require("Operating Conditions: ssg0p9v125c" in report,
                "active operating-condition header drift")
    require("data arrival time                                                1.8796" in setup,
            "critical setup data arrival drift")
    require("data arrival time                                                0.1046" in hold,
            "critical hold data arrival drift")


def validate_netlist():
    mapped = text(PATHS["mapped_v"])
    require(len(re.findall(r"^module\s+", mapped, flags=re.MULTILINE)) == 1,
            "mapped netlist module count drift")
    require(len(re.findall(r"^endmodule\s*$", mapped, flags=re.MULTILINE)) == 1,
            "mapped netlist endmodule count drift")
    instances = re.findall(
        r"^\s*([A-Za-z0-9_]+BWP35P140)\s+([^\s(]+)\s*\(",
        mapped, flags=re.MULTILINE)
    cells = Counter(cell for cell, unused_name in instances)
    require(sum(cells.values()) == 9939,
            "independent mapped instance count drift")
    require(len(cells) == 45, "independent mapped cell-type count drift")
    require(cells["DFKCNQD1BWP35P140"] == 1683,
            "DFKCN count drift")
    require(cells["DFKCSND1BWP35P140"] == 1,
            "DFKCSN count drift")
    require(sum(count for cell, count in cells.items()
                if cell.startswith("DF")) == 1684,
            "independent sequential count drift")
    require(cells["FA1D0BWP35P140"] == 966, "full-adder count drift")
    require(cells["CKBD1BWP35P140"] == 1649, "buffer count drift")
    require(cells["MUX2ND0BWP35P140"] == 789, "mux count drift")
    for token in ("DW_", "GTECH_", "DP_OP"):
        require(token not in mapped, "unmapped token in mapped netlist: " + token)
    require(text(PATHS["check_design_post"]).strip() == "1",
            "postcompile check_design is not clean")


def validate_warnings():
    raw = text(PATHS["dc_raw"])
    warnings = re.findall(r"^Warning: .*\(([A-Z]+-[0-9]+)\)$",
                          raw, flags=re.MULTILINE)
    require(Counter(warnings) == Counter({"UISN-40": 4, "TIM-134": 4}),
            "unexpected DC raw warning inventory: " + repr(Counter(warnings)))
    require("Error:" not in raw and "Fatal:" not in raw,
            "DC raw log contains Error/Fatal")
    require(raw.count("(PWR-24)") == 3, "PWR-24 inventory drift")


def validate_review_and_attacks():
    review = strict_json(REVIEW)
    require(review["review_scope"]["admitted_run"] ==
            "dc_handoff/runs/m64_parent_selector_dc_3p000ns_r1c_20260823",
            "review run binding drift")
    require(review["scores"]["dc_sta_extension_quality_score"] == 82,
            "extension score drift")
    require(sum(review["scores"]["extension_subscores"].values()) == 82,
            "extension subscore arithmetic drift")
    require(review["scores"]["combined_m64_date_prosperity_phi_completeness_score"] == 55,
            "combined DATE score drift")
    require(sum(review["scores"]["combined_subscores"].values()) == 55,
            "combined subscore arithmetic drift")
    require(len(review["issues"]["P0"]) == 0, "P0 count drift")
    require(len(review["issues"]["P1"]) == 5, "P1 count drift")
    require(len(review["issues"]["P2"]) == 6, "P2 count drift")
    boundary = review["claim_boundary"]
    for field in ("paper_ppa_ready", "system_speedup_admitted",
                  "power_or_energy_admitted", "formality_admitted"):
        require(boundary[field] is False, "false claim promotion: " + field)
    require(review["area_and_structure"]["macro_blackbox_count"] == 0,
            "macro count drift")
    require(review["constraint_audit"]["r1c_timing_admission"] ==
            "PASS_WITHIN_EXACT_STANDALONE_LOGIC_ONLY_CONTRACT",
            "timing scope promotion")
    attacks = []
    for name, mutate in (
        ("system_speedup_promotion",
         lambda data: data["claim_boundary"].__setitem__(
             "system_speedup_admitted", True)),
        ("paper_ppa_promotion",
         lambda data: data["claim_boundary"].__setitem__(
             "paper_ppa_ready", True)),
        ("formality_promotion",
         lambda data: data["claim_boundary"].__setitem__(
             "formality_admitted", True)),
        ("extension_score_promotion",
         lambda data: data["scores"].__setitem__(
             "dc_sta_extension_quality_score", 92)),
        ("macro_false_promotion",
         lambda data: data["area_and_structure"].__setitem__(
             "macro_blackbox_count", 1)),
    ):
        attacked = copy.deepcopy(review)
        mutate(attacked)
        rejected = False
        try:
            require(attacked["scores"]["dc_sta_extension_quality_score"] == 82,
                    "score")
            require(attacked["claim_boundary"]["system_speedup_admitted"] is False,
                    "speedup")
            require(attacked["claim_boundary"]["paper_ppa_ready"] is False,
                    "ppa")
            require(attacked["claim_boundary"]["formality_admitted"] is False,
                    "formality")
            require(attacked["area_and_structure"]["macro_blackbox_count"] == 0,
                    "macro")
        except ValueError:
            rejected = True
        require(rejected, "negative attack was not rejected: " + name)
        attacks.append({"name": name, "rejected": True})
    return review, attacks


def validate_receipt(review, attacks):
    receipt = strict_json(RECEIPT)
    require(receipt["status"] ==
            "PASS_M64_DC_R1C_EXTENSION_INDEPENDENT_HAMMER_VALIDATED",
            "validation receipt status drift")
    require(receipt["review_sha256"] == sha256_path(REVIEW),
            "receipt review SHA drift")
    require(receipt["validator_sha256"] == sha256_path(Path(__file__)),
            "receipt validator SHA drift")
    require(receipt["producer_snapshot_manifest_sha256"] ==
            EXPECTED["snapshot_manifest"], "receipt snapshot SHA drift")
    require(receipt["producer_output_manifest_sha256"] ==
            EXPECTED["output_manifest"], "receipt output SHA drift")
    require(receipt["scores"]["dc_sta_extension_quality"] == 82,
            "receipt extension score drift")
    require(receipt["scores"]["combined_m64_date_completeness"] == 55,
            "receipt DATE score drift")
    require(receipt["severity_counts"] == {"P0": 0, "P1": 5, "P2": 6},
            "receipt severity counts drift")
    require(receipt["negative_attacks"] == attacks,
            "receipt negative attack inventory drift")
    require(receipt["admission"] == {
        "standalone_logic_only_dc_sta": True,
        "paper_ppa": False,
        "formality": False,
        "power_or_energy": False,
        "seed_sram_parent_bandwidth": False,
        "m57_all10_system_speedup": False,
    }, "receipt admission boundary drift")


def main():
    validate_pinned_hashes()
    validate_manifests()
    validate_terminal_and_tool()
    validate_constraints()
    validate_reports()
    validate_netlist()
    validate_warnings()
    review, attacks = validate_review_and_attacks()
    validate_receipt(review, attacks)
    print("PASS_M64_DC_R1C_EXTENSION_INDEPENDENT_HAMMER_VALIDATED")
    print("r1c_snapshot_sha256=" + EXPECTED["snapshot_manifest"])
    print("r1c_output_sha256=" + EXPECTED["output_manifest"])
    print("area_um2=10849.481934 setup_slack_ns=0.8967 hold_slack_ns=0.0104")
    print("scores=82/55 severities=P0:0,P1:5,P2:6")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("FAIL_M64_DC_R1C_EXTENSION_INDEPENDENT_HAMMER: " + str(exc),
              file=sys.stderr)
        sys.exit(1)
