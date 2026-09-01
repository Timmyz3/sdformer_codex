#!/usr/bin/env python3
"""Different-author, compile-free mutation hammer for the M1614 C1 source."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
TCL = HW / "dc_handoff/scripts/run_dc_m1614_m993_c1_hold_only_incremental_candidate.tcl"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1614_m993_c1_hold_only_incremental_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1614_m993_c1_hold_only_incremental_dc_source_contract_r1_20260901.json"
AUTHOR_TEST = HW / "system_simulator/tests/test_m1614_c1_hold_only_incremental_dc_source.py"
AUTHOR = HW / "reviews/m1614_m993_c1_hold_only_incremental_dc_source_author_handoff_r1_20260901"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
ORIGINAL = M993 / "original_quarantine"
M1006 = HW / "reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829"
M1612 = HW / "reviews/m1612_m993_c1_hold_closure_first_principles_readonly_review_r1_20260901"
INPUT_DDC = ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc"
INPUT_SDC = ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc"
INPUT_V = ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v"
INPUT_SVF = ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf"
RTL = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
MACRO_RTL = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "dc_handoff/runs/m1614_m993_c1_macro_aware_hold_only_incremental_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1614_m993_c1_macro_aware_hold_only_incremental_dc_attempt_consumed"
RELEASE = HW / "contracts/m1616_m1615_m1614_c1_hold_only_incremental_dc_launch_release_r1_20260901.json"

STD_SLOW = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
STD_FAST = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
MACRO_ROOT = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821")
MACRO_SLOW = MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
MACRO_FAST = MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"
MACRO_MANIFEST = MACRO_ROOT / "SHA256SUMS"
DC_SHELL = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_ACTUAL = Path("/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE = Path("/opt/synopsys/Synopsys.dat")

PINS = {
    TCL: "82cc53ac5f07162143a9ca99170daff9d64f03da3843abe7e0b4d830d24c9659",
    RUNNER: "c21fed97d28ec06b898548c4b406eeab1e1880f9f59d813c61bc8619357119dc",
    CONTRACT: "41be99940cb272a8ae1e040da9ea5c65a211953b6e3e67293f758b5ce513247d",
    AUTHOR_TEST: "3ec43da95360a41809ca5175159a2b43b46bb709cf0eef578f13b320240b54a8",
    AUTHOR / "review.json": "76ee2933b41925810e9bcb5e8e5e6cad84b8bcb257e046eef760def6ab3487e6",
    INPUT_DDC: "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56",
    INPUT_SDC: "cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5",
    INPUT_V: "9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf",
    INPUT_SVF: "8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7",
    RTL: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    MACRO_RTL: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M1612 / "review.json": "7baba71a21be61842be8c76bddfa40abf8d2c0b0736e06aa44a80d53556cef72",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    STD_SLOW: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    STD_FAST: "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
    MACRO_SLOW: "cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf",
    MACRO_FAST: "8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f",
    MACRO_MANIFEST: "c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f",
    DC_SHELL: "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    DC_ACTUAL: "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    LICENSE: "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490",
}

FORBIDDEN = ("set_false_path", "set_multicycle_path", "set_min_delay",
             "set_max_delay", "set_disable_timing", "set_case_analysis")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             AssertionError("nonfinite JSON: " + token)))


def command_text(text):
    return "\n".join(row.split("#", 1)[0] for row in text.splitlines())


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest absent " + str(root))
    require(outer.is_file() and not outer.is_symlink(), "outer absent " + str(root))
    listed = {}
    for row in manifest.read_text(encoding="ascii").splitlines():
        if not row.strip():
            continue
        digest, name = row.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
                "unsafe manifest row " + name)
        listed[name] = digest
    require(outer.read_text(encoding="ascii").split() == [sha256(manifest), "SHA256SUMS"],
            "outer seal mismatch " + str(root))
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        bp = Path(base)
        dirs[:] = [name for name in dirs if not (bp / name).is_symlink()]
        for name in files:
            path = bp / name
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            require(stat.S_ISREG(path.lstat().st_mode), "nonregular member " + str(path))
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(listed), "manifest topology mismatch " + str(root))
    for name, digest in listed.items():
        require(sha256(root / name) == digest, "manifest member drift " + name)


def verify_file_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(manifest.read_text(encoding="ascii").split() == [sha256(path), path.name],
            "file manifest mismatch")
    require(outer.read_text(encoding="ascii").split() == [sha256(manifest), manifest.name],
            "file outer seal mismatch")


def audit_tcl(tcl, sdc):
    t = command_text(tcl)
    s = command_text(sdc)
    require(len(re.findall(r"(?m)^\s*read_ddc\b", t)) == 1, "read_ddc count")
    require(len(re.findall(r"(?m)^\s*read_sdc\b", t)) == 1, "read_sdc count")
    require(len(re.findall(r"(?m)^\s*set_fix_hold\s+\[get_clocks core_clk\]\s*$", t)) == 1,
            "set_fix_hold identity/count")
    require(len(re.findall(r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$", t)) == 1,
            "hold-only optimizer count")
    require(len(re.findall(r"(?m)^\s*compile\b", t)) == 1, "all compile count")
    require(not re.search(r"(?m)^\s*compile_ultra\b", t), "compile_ultra present")
    for term in FORBIDDEN:
        require(not re.search(r"(?m)^\s*" + term + r"\b", t + "\n" + s),
                "timing concealment " + term)
    require("set_clock_period" not in t and "set_clock_uncertainty" not in t,
            "Tcl rewrites clock/uncertainty")
    require(len(re.findall(r"(?m)^\s*create_clock\b", s)) == 1, "clock population")
    require(re.search(r"create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)", s), "3ns drift")
    require(re.search(r"set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b", s),
            "setup uncertainty drift")
    require(re.search(r"set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b", s),
            "hold uncertainty drift")
    require(not re.search(r"(?m)^\s*set_propagated_clock\b", s), "clock no longer ideal")
    exact = (
        "set macro_cell TS1N28HPCPHVTB128X128M4S",
        "set expected_macro_count 9",
        "set area_baseline_um2 147246.392090",
        "set area_ceiling_um2 154608.7116945",
        "set_min_library $std_slow_db -min_version $std_fast_db",
        "set_min_library $macro_slow_db -min_version $macro_fast_db",
        "set_dont_touch $macro_cells_pre true",
        "set_wire_load_model -name ZeroWireload [current_design]",
        "set_operating_conditions ssg0p9v125c",
    )
    for token in exact:
        require(tcl.count(token) == 1, "Tcl physical token drift: " + token)
    require(tcl.count("if {$macro_count_pre != $expected_macro_count}") == 1,
            "pre macro gate")
    require(tcl.count("if {$macro_count_post != $expected_macro_count}") == 1,
            "post macro gate")
    require("set_app_var target_library [list $std_slow_db]" in tcl,
            "target library drift")
    require("$macro_slow_db $macro_fast_db]" in tcl, "macro link views absent")
    reports = (
        "link.rpt", "flow_contract.rpt", "macro_binding_audit.txt",
        "check_design_prehold.rpt", "check_timing_prehold.rpt",
        "qor_prehold.rpt", "area_prehold.rpt", "clocks_prehold.rpt",
        "references_prehold.rpt", "timing_setup_prehold_top100.rpt",
        "timing_hold_prehold_top100.rpt", "constraint_setup_prehold_all.rpt",
        "constraint_hold_prehold_all.rpt", "setup_prehold_summary_machine.txt",
        "hold_prehold_summary_machine.txt", "qor_posthold.rpt",
        "area_posthold.rpt", "hierarchy_posthold.rpt", "resources_posthold.rpt",
        "references_posthold.rpt", "clocks_posthold.rpt",
        "timing_setup_posthold_top100.rpt", "timing_hold_posthold_top100.rpt",
        "constraint_setup_posthold_all.rpt", "constraint_hold_posthold_all.rpt",
        "constraint_design_rules_posthold.rpt", "check_design_posthold.rpt",
        "check_timing_posthold.rpt", "setup_posthold_summary_machine.txt",
        "hold_posthold_summary_machine.txt",
    )
    for token in reports:
        require(token in tcl, "missing Tcl report " + token)
    for token in ("_m1614_hold_repaired_mapped.v", "_m1614_hold_repaired_mapped.sdc",
                  "_m1614_hold_repaired.ddc", "_m1614_hold_repaired.svf"):
        require(token in tcl, "missing output " + token)


def audit_runner(runner):
    r = command_text(runner)
    pins = (
        "sha_exact d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56 \"${INPUT_DDC}\"",
        "sha_exact cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5 \"${INPUT_SDC}\"",
        "sha_exact 9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf \"${INPUT_MAPPED_V}\"",
        "sha_exact 8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7 \"${INPUT_SVF}\"",
        "sha_tool 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2 \"${DC_SHELL}\"",
        "sha_tool bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391 \"${DC_ACTUAL}\"",
        "sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af \"${STD_SLOW}\"",
        "sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a \"${STD_FAST}\"",
        "sha_exact cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf \"${MACRO_SLOW}\"",
        "sha_exact 8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f \"${MACRO_FAST}\"",
        "sha_exact c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f \"${MACRO_MANIFEST}\"",
    )
    for token in pins:
        require(runner.count(token) == 1, "runner pin drift: " + token)
    require(runner.count('"${DC_SHELL}" -f "${TCL}"') == 1, "DC process budget")
    for tool in ("pt_shell", "fm_shell", "vcs ", "simv ", "ptpx"):
        require(tool not in r.lower(), "other EDA command present: " + tool)
    release = runner.index('verify_dir_seal "${HAMMER_DIR}"')
    collision = runner.index("blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}")
    attempt = runner.index('mkdir -- "${ATTEMPT}"')
    tool = runner.index('"${DC_SHELL}" -f "${TCL}"')
    require(release < collision < attempt < tool, "release/collision/attempt/tool order")
    require("if not p.name.isdigit() or int(p.name) in ancestry: continue" in runner,
            "ancestry exclusion absent")
    require("if p.stat().st_uid != os.getuid(): continue" in runner,
            "same-UID gate absent")
    require("same-UID DC collision" in runner, "same-UID collision stop absent")
    require('mkdir -- "${LOCK}" || exit 4' in runner, "exclusive lock absent")
    require("M1614_EXPECTED_DC_RUNNER_SHA256" in runner and
            "M1614_EXPECTED_DC_RELEASE_SHA256" in runner, "caller pins absent")
    require("retry=false" in runner and "rm -rf" not in runner,
            "retry/destructive policy drift")
    gate_tokens = (
        "met != (out['wns_ns']>=0 and out['tns_ns']==0 and out['violating_paths']==0)",
        "timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')",
        "area=float(m.group(1)); baseline=147246.392090; ceiling=154608.7116945",
        "macro_ok=(int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9)",
        "positive=timing_ok and area_ok and macro_ok and drc_count==0",
        "if not math.isfinite(area) or area<=0",
        "SEALED_NEGATIVE_M1614_C1_HOLD_OR_AREA_GATE_FAILED__NO_RETRY",
    )
    for token in gate_tokens:
        require(token in runner, "success gate drift: " + token)
    for term in FORBIDDEN:
        require("forbidden output SDC" in runner and term in runner,
                "output SDC exception audit drift " + term)
    for token in ("-period\\s+3", "-setup\\s+0?\\.2", "-hold\\s+0?\\.05"):
        require(token in runner, "output SDC identity gate absent " + token)
    required = (
        "setup_prehold_summary_machine.txt", "hold_prehold_summary_machine.txt",
        "setup_posthold_summary_machine.txt", "hold_posthold_summary_machine.txt",
        "constraint_design_rules_posthold.rpt", "macro_binding_audit.txt",
        "m1614_hold_repaired_mapped.v", "m1614_hold_repaired_mapped.sdc",
        "m1614_hold_repaired.ddc", "m1614_hold_repaired.svf",
    )
    for token in required:
        require(token in runner, "runner artifact gate absent " + token)
    require("macro_count_pre=9" in runner and "macro_count_post=9" in runner,
            "runner macro gate absent")


def audit_contract(value, tcl, runner, author_test):
    require(value["status"] == "SOURCE_ONLY_M1614_C1_HOLD_PACKAGE__NO_EDA_AUTHORIZED",
            "contract status")
    require(value["authorization"] == {
        "dc_runs_now": 0, "future_dc_runs_max": 1, "all_other_eda_runs": 0,
        "vcs_runs": 0, "pt_runs": 0, "formality_runs": 0, "ptpx_runs": 0,
        "gpu_runs": 0, "remote_runs": 0, "retry": False}, "authorization drift")
    require(value["identity"]["tcl_sha256"] == hashlib.sha256(tcl.encode()).hexdigest(),
            "contract Tcl binding")
    require(value["identity"]["runner_sha256"] == hashlib.sha256(runner.encode()).hexdigest(),
            "contract runner binding")
    require(value["identity"]["author_test_sha256"] ==
            hashlib.sha256(author_test.encode()).hexdigest(), "contract author test binding")
    phy = value["frozen_physical_point"]
    require(phy["clock_period_ns"] == 3.0 and phy["setup_uncertainty_ns"] == 0.2
            and phy["hold_uncertainty_ns"] == 0.05, "contract timing identity")
    require(phy["ideal_clock"] is True and phy["wireload"] == "ZeroWireload",
            "contract model identity")
    require(phy["macro_count_exact"] == 9 and
            phy["positive_area_ceiling_um2"] == 154608.7116945,
            "contract macro/area gate")
    require(all(count == 0 for count in phy["timing_exception_counts"].values()),
            "contract exception count")
    require(value["compile_contract"]["all_compile_command_count"] == 1 and
            value["compile_contract"]["hold_only_incremental_mapping_count"] == 1 and
            value["compile_contract"]["set_fix_hold_count"] == 1,
            "contract command count")
    require(value["future_release_chain"]["present_at_source_authoring"] is False,
            "premature release")
    for key in ("launch_authorized", "launch_executed", "hold_closed", "formality",
                "prime_time", "power", "energy", "cycle_speedup", "system_speedup",
                "paper_ppa_ready", "paper_citable", "headline"):
        require(value["claim_boundary"][key] is False, "claim boundary " + key)


def changed(text, old, new):
    require(text.count(old) >= 1, "mutation source token absent: " + old)
    return text.replace(old, new, 1)


def expect_reject(label, fn, *args):
    try:
        fn(*args)
    except (AssertionError, KeyError, ValueError, TypeError):
        return label
    raise AssertionError("mutation survived: " + label)


def build():
    for path, digest in PINS.items():
        require(path.is_file(), "identity nonregular " + str(path))
        if path != DC_SHELL:
            require(not path.is_symlink(), "unexpected symlink identity " + str(path))
        require(sha256(path) == digest, "identity SHA mismatch " + str(path))
    verify_file_seal(CONTRACT)
    for tree in (M993, ORIGINAL, M1006, M1612, AUTHOR):
        verify_tree(tree)
    require(sha256(M993 / "SHA256SUMS") ==
            "8aeda1372387692201badb90a7d81eb7d908f803c6cd652aab22dace5043d093",
            "M993 manifest identity")
    require(sha256(ORIGINAL / "SHA256SUMS") ==
            "9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe",
            "original manifest identity")
    require(sha256(AUTHOR / "SHA256SUMS") ==
            "126bcbe3e2d3b4d1ac6c23852a5800f2d5ebb8c34925d6ca4e82577a154aaf6a",
            "author handoff manifest identity")
    require(not RESULT.exists() and not ATTEMPT.exists() and not RELEASE.exists(),
            "source-only result/attempt/release boundary consumed")

    tcl = TCL.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    sdc = INPUT_SDC.read_text(encoding="utf-8")
    author_test = AUTHOR_TEST.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "review.json")
    require(author["status"] ==
            "PASS_M1614_SOURCE_AUTHOR_TESTS__READY_FOR_M1615_DIFFERENT_AUTHOR_HAMMER__NO_EDA_AUTHORIZED",
            "author handoff status")
    audit_tcl(tcl, sdc)
    audit_runner(runner)
    audit_contract(contract, tcl, runner, author_test)

    attacks = []
    # Tcl optimizer and constraint attacks.
    tcl_mutations = [
        ("duplicate_hold_compile", tcl + "\ncompile -incremental_mapping -only_hold_time\n", sdc),
        ("generic_incremental", tcl + "\ncompile -incremental_mapping\n", sdc),
        ("compile_ultra", tcl + "\ncompile_ultra -incremental\n", sdc),
        ("delete_set_fix_hold", changed(tcl, "set_fix_hold [get_clocks core_clk]", ""), sdc),
        ("macro_count_8", changed(tcl, "set expected_macro_count 9", "set expected_macro_count 8"), sdc),
        ("delete_macro_dont_touch", changed(tcl, "set_dont_touch $macro_cells_pre true", ""), sdc),
        ("delete_std_min_library", changed(tcl, "set_min_library $std_slow_db -min_version $std_fast_db", ""), sdc),
        ("delete_macro_min_library", changed(tcl, "set_min_library $macro_slow_db -min_version $macro_fast_db", ""), sdc),
        ("wireload_drift", changed(tcl, "set_wire_load_model -name ZeroWireload [current_design]",
                                   "set_wire_load_model -name Enclosed [current_design]"), sdc),
        ("area_ceiling_relaxed", changed(tcl, "set area_ceiling_um2 154608.7116945",
                                         "set area_ceiling_um2 170000.0"), sdc),
        ("missing_post_hold_report", changed(tcl, "hold_posthold_summary_machine.txt",
                                             "hold_posthold_summary_removed.txt"), sdc),
    ]
    for term in FORBIDDEN:
        tcl_mutations.append(("tcl_" + term, tcl + "\n" + term + " [get_clocks core_clk]\n", sdc))
    sdc_mutations = [
        ("sdc_period_4ns", changed(sdc, "-period 3  -waveform", "-period 4  -waveform")),
        ("sdc_setup_uncertainty", changed(sdc, "-setup 0.2", "-setup 0.1")),
        ("sdc_hold_uncertainty", changed(sdc, "-hold 0.05", "-hold 0.00")),
        ("sdc_propagated_clock", sdc + "\nset_propagated_clock [get_clocks core_clk]\n"),
    ]
    for term in FORBIDDEN:
        sdc_mutations.append(("sdc_" + term, sdc + "\n" + term + " [get_clocks core_clk]\n"))
    for label, mtcl, msdc in tcl_mutations:
        attacks.append(expect_reject(label, audit_tcl, mtcl, msdc))
    for label, msdc in sdc_mutations:
        attacks.append(expect_reject(label, audit_tcl, tcl, msdc))

    # Runner identity, success predicate and one-shot control attacks.
    runner_mutations = [
        ("runner_area_relaxed", changed(runner, "ceiling=154608.7116945", "ceiling=170000.0")),
        ("ignore_setup", changed(runner,
            "timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')",
            "timing_ok=(post_hold['status']=='MET')")),
        ("ignore_hold", changed(runner,
            "timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')",
            "timing_ok=(post_setup['status']=='MET')")),
        ("ignore_wns", changed(runner,
            "out['wns_ns']>=0 and out['tns_ns']==0 and out['violating_paths']==0",
            "out['tns_ns']==0 and out['violating_paths']==0")),
        ("ignore_tns", changed(runner,
            "out['wns_ns']>=0 and out['tns_ns']==0 and out['violating_paths']==0",
            "out['wns_ns']>=0 and out['violating_paths']==0")),
        ("ignore_violations", changed(runner,
            "out['wns_ns']>=0 and out['tns_ns']==0 and out['violating_paths']==0",
            "out['wns_ns']>=0 and out['tns_ns']==0")),
        ("ignore_drc", changed(runner,
            "positive=timing_ok and area_ok and macro_ok and drc_count==0",
            "positive=timing_ok and area_ok and macro_ok")),
        ("attempt_after_tool", changed(runner, 'mkdir -- "${ATTEMPT}"',
            'true # moved attempt after tool',) + '\nmkdir -- "${ATTEMPT}"\n'),
        ("other_eda_pt", runner + "\n/opt/synopsys/pt/bin/pt_shell\n"),
        ("same_uid_removed", changed(runner,
            "if p.stat().st_uid != os.getuid(): continue", "continue")),
        ("ancestry_removed", changed(runner,
            "if not p.name.isdigit() or int(p.name) in ancestry: continue",
            "if not p.name.isdigit(): continue")),
        ("dc_actual_collision_removed", changed(runner,
            "blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}",
            "blocked={'dc_shell','dc_shell-t'}")),
        ("missing_release_gate", changed(runner, 'verify_dir_seal "${HAMMER_DIR}"', "true")),
        ("retry_true", runner.replace("retry=false", "retry=true")),
        ("missing_macro_result_gate", changed(runner, "macro_count_post=9", "macro_count_post=8")),
        ("missing_drc_artifact", changed(runner, "constraint_design_rules_posthold.rpt",
                                        "constraint_design_rules_removed.rpt")),
    ]
    pin_attacks = [
        ("input_ddc_sha", "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56"),
        ("input_sdc_sha", "cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5"),
        ("input_v_sha", "9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf"),
        ("input_svf_sha", "8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7"),
        ("dc_shell_sha", "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"),
        ("dc_actual_sha", "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391"),
        ("std_slow_sha", "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"),
        ("std_fast_sha", "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a"),
        ("macro_slow_sha", "cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf"),
        ("macro_fast_sha", "8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f"),
        ("macro_manifest_sha", "c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f"),
    ]
    for label, digest in pin_attacks:
        runner_mutations.append((label, changed(runner, digest, "0" * 64)))
    for label, mutant in runner_mutations:
        attacks.append(expect_reject(label, audit_runner, mutant))

    # Contract identity and authority attacks.
    contract_mutations = []
    for label, mutate in (
        ("contract_macro_8", lambda v: v["frozen_physical_point"].__setitem__("macro_count_exact", 8)),
        ("contract_period_4", lambda v: v["frozen_physical_point"].__setitem__("clock_period_ns", 4.0)),
        ("contract_area_relaxed", lambda v: v["frozen_physical_point"].__setitem__("positive_area_ceiling_um2", 170000.0)),
        ("contract_dc_now", lambda v: v["authorization"].__setitem__("dc_runs_now", 1)),
        ("contract_retry", lambda v: v["authorization"].__setitem__("retry", True)),
        ("contract_claim_hold", lambda v: v["claim_boundary"].__setitem__("hold_closed", True)),
    ):
        value = json.loads(json.dumps(contract))
        mutate(value)
        contract_mutations.append((label, value))
    for label, value in contract_mutations:
        attacks.append(expect_reject(label, audit_contract, value, tcl, runner, author_test))

    require(len(attacks) >= 55, "mutation population too small")
    return {
        "schema": "m1615_m1614_c1_hold_only_incremental_dc_source_hammer_audit_r1_v1",
        "status": "PASS_M1615_M1614_C1_HOLD_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT",
        "date": "2026-09-01",
        "score": 98,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 1,
        "identity": {
            "tcl_sha256": PINS[TCL], "runner_sha256": PINS[RUNNER],
            "contract_sha256": PINS[CONTRACT], "author_test_sha256": PINS[AUTHOR_TEST],
            "author_handoff_review_sha256": PINS[AUTHOR / "review.json"],
            "author_handoff_manifest_sha256": sha256(AUTHOR / "SHA256SUMS"),
            "author_handoff_outer_seal_file_sha256": sha256(AUTHOR / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": PINS[DOCS359]
        },
        "static_audit": {
            "mutation_attacks": len(attacks), "mutation_rejections": len(attacks),
            "tcl_and_runner_semantic_audit": True, "sealed_tree_audit": True,
            "same_uid_collision_gate": True, "ancestry_excluded": True,
            "author_source_test_execution_recorded_separately": True
        },
        "p2": {
            "id": "P2_PUBLICATION_USES_EARLY_FRESHNESS_CHECK_PLUS_MV",
            "finding": "The runner has an exclusive lock and consumed attempt, but final publication uses an early freshness check followed by mv without a post-move exact-topology assertion.",
            "impact": "Operational hardening only under the same-UID single-run contract; it does not change Tcl, timing gates, retry=false, or this one-future-attempt authorization. The result hammer must reject nested or unexpected canonical topology."
        },
        "authorization": {
            "m1616_release_authoring": True,
            "future_dc_attempts_max_after_sealed_release": 1,
            "dc_now": 0, "all_eda_now": 0,
            "next": "Author M1616 release bound to this review, exact runner and source contract; only then one DC attempt may be launched."
        },
        "claim_boundary": {
            "source_hammer_only": True, "release_created": False,
            "dc": False, "hold_closed": False, "timing": False, "area_result": False,
            "formality": False, "prime_time": False, "power": False,
            "energy": False, "speedup": False, "paper_citable": False
        },
        "attacks": attacks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-frozen")
    args = parser.parse_args()
    value = build()
    if args.check_frozen:
        require(strict_json(Path(args.check_frozen)) == value, "frozen audit mismatch")
        print("PASS_M1615_FROZEN_AUDIT_MATCH attacks=%d" % len(value["attacks"]))
    else:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
