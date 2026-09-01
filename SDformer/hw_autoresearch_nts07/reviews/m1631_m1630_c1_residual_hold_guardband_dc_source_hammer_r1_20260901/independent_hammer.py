#!/usr/bin/env python3
"""Different-author static/mutation hammer for inert M1630 C1 DC sources."""
from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
TCL = HW / "dc_handoff/scripts/run_dc_m1630_m993_c1_residual_hold_guardband_candidate.tcl"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1630_m993_c1_residual_hold_guardband_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1630_m993_c1_residual_hold_guardband_dc_source_contract_r1_20260901.json"
AUTHOR_TEST = HW / "system_simulator/tests/test_m1630_c1_residual_hold_guardband_dc_source.py"
AUTHOR = HW / "reviews/m1630_m993_c1_residual_hold_guardband_dc_source_author_handoff_r1_20260901"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
ORIGINAL = M993 / "original_quarantine"
M1006 = HW / "reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829"
M1614 = HW / "dc_handoff/runs/m1614_m993_c1_macro_aware_hold_only_incremental_dc_r1_20260901.failed_or_incomplete.4065447.quarantine"
INPUT_SDC = ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc"
RESULT = HW / "dc_handoff/runs/m1630_m993_c1_residual_hold_guardband_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1630_m993_c1_residual_hold_guardband_dc_attempt_consumed"
RELEASE = HW / "contracts/m1632_m1631_m1630_c1_residual_hold_guardband_dc_launch_release_r1_20260901.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    TCL: "e1a138284a4e558c339d987fd4da8e248b0c2ea81858fe255e812c0a4ec592c1",
    RUNNER: "ceb4dcdf1a90b9285e10b06a6ff9e91faf8f840373dc6d2095c50004e7493b81",
    CONTRACT: "7fd0d5792dfcbfeb9f7c715ec7b0264e5f4da9eeabcd5ca660d07eca1d38dc0a",
    AUTHOR_TEST: "c52e1ec09514d9e33cb4cfca5121d72c04519f472e67c0b745fd2c7be6d852e2",
    AUTHOR / "review.json": "6007cae2eb4dea6b1c23c3803d7c381852ca6f3b119bf9d6e044a8e22d83469e",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc":
        "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56",
    INPUT_SDC: "cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v":
        "9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf":
        "8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7",
    M1006 / "review.json": "d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea",
    M1614 / "reports/setup_posthold_summary_machine.txt":
        "ff8f206815233c222c781a507d3ce504571148885ff7b113c5f94cb8824f639b",
    M1614 / "reports/hold_posthold_summary_machine.txt":
        "fb12725e8bc76cce0c9f8198cb8b915dc91cf8c0fbb4e185b4fae2da2414da8c",
    M1614 / "reports/area_posthold.rpt":
        "aa109ac641cbee88d6617e4b4f3008f6669a91d51e055bb151d7b3c324ec655e",
    Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"):
        "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    Path("/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec"):
        "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391",
}

FORBIDDEN = ("set_false_path", "set_multicycle_path", "set_min_delay",
             "set_max_delay", "set_disable_timing", "set_case_analysis")


def require(value, message):
    if not value:
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
            require(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             AssertionError("nonfinite JSON " + token)))


def commands(text):
    return "\n".join(row.split("#", 1)[0] for row in text.splitlines())


def verify_file_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(manifest.read_text(encoding="ascii").split() == [sha256(path), path.name],
            "file manifest mismatch " + str(path))
    require(outer.read_text(encoding="ascii").split() == [sha256(manifest), manifest.name],
            "file outer mismatch " + str(path))


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest absent")
    require(outer.is_file() and not outer.is_symlink(), "outer absent")
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
            "outer tree mismatch " + str(root))
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        bp = Path(base)
        dirs[:] = [name for name in dirs if not (bp / name).is_symlink()]
        for name in files:
            path = bp / name
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
                    "nonregular tree member " + str(path))
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(listed), "tree topology mismatch " + str(root))
    for name, digest in listed.items():
        require(sha256(root / name) == digest, "tree member drift " + name)


def audit_tcl(tcl, sdc):
    text = commands(tcl)
    source_sdc = commands(sdc)
    require(len(re.findall(r"(?m)^\s*read_ddc\b", text)) == 1, "read_ddc count")
    require(len(re.findall(r"(?m)^\s*read_sdc\b", text)) == 1, "read_sdc count")
    require("set design_collection [current_design]" in text, "design collection")
    require("set active_design [get_object_name $design_collection]" in text,
            "current design object name")
    require("if {$active_design ne $design_name}" in text, "design identity gate")
    require(len(re.findall(r"(?m)^\s*set_fix_hold\s+\$core_clock\s*$", text)) == 1,
            "set_fix_hold count")
    require(len(re.findall(r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$",
                           text)) == 1, "hold-only compile count")
    require(len(re.findall(r"(?m)^\s*compile\b", text)) == 1, "all compile count")
    require(not re.search(r"(?m)^\s*compile_ultra\b", text), "compile_ultra")
    require(tcl.count("set optimization_hold_guardband_ns 0.051") == 1, "guardband")
    require(tcl.count("set reported_hold_uncertainty_ns 0.050") == 1, "reported hold")
    guard = tcl.index("set_clock_uncertainty -hold $optimization_hold_guardband_ns $core_clock")
    compile_at = tcl.index("compile -incremental_mapping -only_hold_time")
    restore = tcl.index("set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock")
    post = tcl.index('report_qor > "$output_dir/reports/qor_posthold.rpt"')
    require(guard < compile_at < restore < post, "guardband/compile/restore order")
    for token in FORBIDDEN:
        require(not re.search(r"(?m)^\s*" + token + r"\b", text + "\n" + source_sdc),
                "timing concealment " + token)
    require(len(re.findall(r"(?m)^\s*create_clock\b", source_sdc)) == 1,
            "clock population")
    require(re.search(r"create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)", source_sdc),
            "period drift")
    require(re.search(r"set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b", source_sdc),
            "setup uncertainty drift")
    require(re.search(r"set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b", source_sdc),
            "hold uncertainty drift")
    require(not re.search(r"(?m)^\s*set_propagated_clock\b", source_sdc),
            "propagated clock")
    exact = (
        "set expected_macro_count 9",
        "set area_baseline_um2 147246.392090",
        "set area_ceiling_um2 154608.7116945",
        "set_min_library $std_slow_db -min_version $std_fast_db",
        "set_min_library $macro_slow_db -min_version $macro_fast_db",
        "set_operating_conditions ssg0p9v125c",
        "set_wire_load_model -name ZeroWireload [current_design]",
        "set_dont_touch $macro_cells_pre true",
        "if {$macro_count_pre != $expected_macro_count}",
        "if {$macro_count_post != $expected_macro_count}",
    )
    for token in exact:
        require(tcl.count(token) == 1, "Tcl physical token " + token)
    require(tcl.count("failed_m1614_output_used=false") == 2,
            "failed M1614 input boundary")
    require(tcl.count("input_generation=original_m993_m1006_admitted_ddc") == 2,
            "original M993 input boundary")
    for token in ("link.rpt", "check_design_prehold.rpt", "check_timing_prehold.rpt",
                  "check_design_posthold.rpt", "check_timing_posthold.rpt",
                  "constraint_design_rules_posthold.rpt", "macro_binding_audit.txt",
                  "setup_posthold_summary_machine.txt", "hold_posthold_summary_machine.txt",
                  "_m1630_residual_hold_closed_mapped.v",
                  "_m1630_residual_hold_closed_mapped.sdc",
                  "_m1630_residual_hold_closed.ddc", "_m1630_residual_hold_closed.svf"):
        require(token in tcl, "missing report/output " + token)


def audit_runner(runner):
    text = commands(runner)
    require('INPUT_DDC="${M993_ORIGINAL}/netlist/' in runner, "original input DDC")
    require(not re.search(r"(?m)^INPUT_DDC=.*m1614", runner), "M1614 input DDC")
    for token in (
        'sha_exact d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56 "${INPUT_DDC}"',
        'sha_exact cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5 "${INPUT_SDC}"',
        'sha_exact 9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf "${INPUT_MAPPED_V}"',
        'sha_exact 8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7 "${INPUT_SVF}"',
        'sha_exact cef7b0bb2cbcfbc0e723068e54018fbca5acf708f3cb0850e3d2a59677875d13 "${M1614_NEGATIVE}/SHA256SUMS"',
        'sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOC359}"',
        'sha_tool 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2 "${DC_SHELL}"',
        'sha_tool bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391 "${DC_ACTUAL}"',
    ):
        require(runner.count(token) == 1, "runner pin " + token)
    require(runner.count('"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"') == 1,
            "DC invocation/budget")
    require('DC_ACTUAL="/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec"'
            in runner, "actual DC executable identity")
    require(not re.search(r"(?m)^\s*(?:export\s+)?HOME=", text), "HOME repurpose")
    for tool in ("pt_shell", "fm_shell", "vcs ", "simv ", "ptpx"):
        require(tool not in text.lower(), "other EDA " + tool)
    review = runner.index('verify_dir_seal "${HAMMER_DIR}"')
    release = runner.index('verify_file_seal "${RELEASE}"')
    collision = runner.index("blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}")
    attempt = runner.index('mkdir -- "${ATTEMPT}"')
    tool = runner.index('"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"')
    require(review < release < collision < attempt < tool, "review/release/attempt/tool order")
    require("M1630_EXPECTED_DC_RUNNER_SHA256" in runner, "runner caller pin")
    require("M1630_EXPECTED_DC_RELEASE_SHA256" in runner, "release caller pin")
    require("if p.stat().st_uid != os.getuid(): continue" in runner, "same UID gate")
    require("int(p.name) in ancestry: continue" in runner, "ancestry exclusion")
    require("common_shell_exec" in runner and "same-UID DC collision" in runner,
            "actual DC collision gate")
    require('mkdir -- "${LOCK}" || fail "launch lock collision"' in runner,
            "launch lock")
    require('mkdir -- "${ATTEMPT}"' in runner and runner.index('mkdir -- "${ATTEMPT}"') < tool,
            "attempt before tool")
    require("retry=false" in runner and "rm -rf" not in runner, "retry/destructive")
    for token in ("(Error|Fatal):", "LINK-[0-9]+", "unresolved (reference|design|cell)",
                  "combinational[ _-]*loop", "timing[ _-]*loop", "(TIM-209|OPT-150)"):
        require(token in runner, "fatal scan " + token)
    for token in ("reports/link.rpt", "reports/check_design_prehold.rpt",
                  "reports/check_timing_prehold.rpt", "reports/check_design_posthold.rpt",
                  "reports/check_timing_posthold.rpt"):
        require(token in runner, "fatal report " + token)
    gate_tokens = (
        "met != (out['wns_ns']>=0 and out['tns_ns']==0 and out['violating_paths']==0)",
        "timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')",
        "area=float(m.group(1)); baseline=147246.392090; ceiling=154608.7116945",
        "area_ok=area<=ceiling",
        "macro_ok=(int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9)",
        "positive=timing_ok and area_ok and macro_ok and drc_count==0",
        "if not math.isfinite(area) or area<=0",
        "SEALED_NEGATIVE_M1630_C1_RESIDUAL_HOLD_OR_AREA_GATE_FAILED__NO_RETRY",
        "optimization guardband leaked into output SDC",
    )
    for token in gate_tokens:
        require(token in runner, "positive gate " + token)
    for token in ("macro_count_pre=9", "macro_count_post=9",
                  "optimization_hold_guardband_ns=0.051",
                  "reported_hold_uncertainty_ns=0.050",
                  "hold_only_incremental_mapping_count=1",
                  "all_compile_command_count=1", "compile_ultra_incremental_count=0",
                  "generic_incremental_mapping_count=0"):
        require(token in runner, "flow gate " + token)
    for term in FORBIDDEN:
        require(term in runner and "forbidden output SDC" in runner,
                "output exception audit " + term)


def audit_contract(value, tcl, runner, author_test):
    require(value["status"] ==
            "SOURCE_ONLY_M1630_C1_RESIDUAL_HOLD_GUARDBAND__NO_EDA_AUTHORIZED",
            "contract status")
    auth = value["authorization"]
    require(auth["dc_runs_now"] == 0 and auth["future_dc_runs_max"] == 1 and
            auth["all_other_eda_runs"] == 0 and auth["retry"] is False,
            "contract authority")
    for key in ("vcs_runs", "pt_runs", "formality_runs", "ptpx_runs",
                "gpu_runs", "remote_runs", "attempts_created_now"):
        require(auth[key] == 0, "contract authority " + key)
    require(value["identity"]["tcl_sha256"] == hashlib.sha256(tcl.encode()).hexdigest(),
            "contract Tcl binding")
    require(value["identity"]["runner_sha256"] == hashlib.sha256(runner.encode()).hexdigest(),
            "contract runner binding")
    require(value["identity"]["author_test_sha256"] ==
            hashlib.sha256(author_test.encode()).hexdigest(), "author-test binding")
    require(value["input_policy"]["only_original_m993_ddc"] is True and
            value["input_policy"]["failed_m1614_output_is_input"] is False,
            "input policy")
    phy = value["frozen_reported_point"]
    require((phy["clock_period_ns"], phy["setup_uncertainty_ns"],
             phy["reported_hold_uncertainty_ns"],
             phy["optimization_hold_guardband_ns"]) == (3.0, 0.2, 0.05, 0.051),
            "timing point")
    require(phy["macro_count_exact"] == 9 and phy["wireload"] == "ZeroWireload" and
            phy["ideal_clock"] is True and
            phy["positive_area_ceiling_um2"] == 154608.7116945,
            "physical point")
    require(all(item == 0 for item in phy["timing_exception_counts"].values()),
            "exception population")
    cc = value["compile_contract"]
    require(cc["set_fix_hold_count"] == cc["hold_only_incremental_mapping_count"] ==
            cc["all_compile_command_count"] == 1, "compile contract")
    require(cc["compile_ultra_count"] == 0 and
            cc["generic_incremental_mapping_count"] == 0 and
            cc["second_hold_only_pass"] is False and cc["frequency_change"] is False,
            "optimizer prohibition")
    require(value["future_release_chain"]["present_at_source_authoring"] is False,
            "premature release")
    for key in ("launch_authorized", "launch_executed", "hold_closed", "formality",
                "prime_time", "power", "energy", "cycle_speedup", "system_speedup",
                "paper_ppa_ready", "paper_citable", "headline"):
        require(value["claim_boundary"][key] is False, "claim boundary " + key)


def mutate(text, old, new):
    require(old in text, "mutation token absent " + old)
    return text.replace(old, new, 1)


def rejected(label, fn, *args):
    try:
        fn(*args)
    except (AssertionError, KeyError, ValueError, TypeError):
        return label
    raise AssertionError("mutation survived " + label)


def build():
    for path, digest in PINS.items():
        require(path.is_file(), "identity absent " + str(path))
        require(sha256(path) == digest, "identity SHA " + str(path))
    for tree in (M993, ORIGINAL, M1006, M1614, AUTHOR):
        verify_tree(tree)
    verify_file_seal(CONTRACT)
    require(sha256(M993 / "SHA256SUMS") ==
            "8aeda1372387692201badb90a7d81eb7d908f803c6cd652aab22dace5043d093",
            "M993 seal identity")
    require(sha256(ORIGINAL / "SHA256SUMS") ==
            "9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe",
            "original seal identity")
    require(sha256(M1614 / "SHA256SUMS") ==
            "cef7b0bb2cbcfbc0e723068e54018fbca5acf708f3cb0850e3d2a59677875d13",
            "M1614 negative identity")
    require(sha256(AUTHOR / "SHA256SUMS") ==
            "93cb45dcd295d95a8aaadcc17fbf0534238a5aa543f884a533ed21a041b61163",
            "author handoff identity")
    require(not RESULT.exists() and not ATTEMPT.exists() and not RELEASE.exists(),
            "source-only namespace already consumed")

    tcl = TCL.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    sdc = INPUT_SDC.read_text(encoding="utf-8")
    author_test = AUTHOR_TEST.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "review.json")
    require(author["status"] ==
            "PASS_SOURCE_ONLY_M1630_C1_RESIDUAL_HOLD_GUARDBAND_AUTHOR_HANDOFF__M1631_REVIEW_REQUIRED",
            "author handoff status")
    audit_tcl(tcl, sdc)
    audit_runner(runner)
    audit_contract(contract, tcl, runner, author_test)

    outcomes = []
    tcl_mutations = [
        ("remove_read_ddc", "read_ddc $input_ddc", "set x $input_ddc"),
        ("second_read_ddc", "read_ddc $input_ddc", "read_ddc $input_ddc\nread_ddc $input_ddc"),
        ("direct_current_design_compare", "set active_design [get_object_name $design_collection]", "set active_design [current_design]"),
        ("remove_design_identity", "if {$active_design ne $design_name}", "if {0}"),
        ("remove_set_fix_hold", "set_fix_hold $core_clock", "set x $core_clock"),
        ("second_set_fix_hold", "set_fix_hold $core_clock", "set_fix_hold $core_clock\nset_fix_hold $core_clock"),
        ("generic_compile", "compile -incremental_mapping -only_hold_time", "compile -incremental_mapping"),
        ("second_compile", "compile -incremental_mapping -only_hold_time", "compile -incremental_mapping -only_hold_time\ncompile -incremental_mapping -only_hold_time"),
        ("compile_ultra", "compile -incremental_mapping -only_hold_time", "compile_ultra -incremental"),
        ("guardband_zero", "set optimization_hold_guardband_ns 0.051", "set optimization_hold_guardband_ns 0.050"),
        ("guardband_large", "set optimization_hold_guardband_ns 0.051", "set optimization_hold_guardband_ns 0.100"),
        ("reported_hold_relaxed", "set reported_hold_uncertainty_ns 0.050", "set reported_hold_uncertainty_ns 0.049"),
        ("restore_before_compile", "set_fix_hold $core_clock\ncompile -incremental_mapping -only_hold_time\nset_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock", "set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock\nset_fix_hold $core_clock\ncompile -incremental_mapping -only_hold_time"),
        ("macro_count_eight", "set expected_macro_count 9", "set expected_macro_count 8"),
        ("drop_pre_macro_gate", "if {$macro_count_pre != $expected_macro_count}", "if {0}"),
        ("drop_post_macro_gate", "if {$macro_count_post != $expected_macro_count}", "if {0}"),
        ("drop_macro_dont_touch", "set_dont_touch $macro_cells_pre true", "set x $macro_cells_pre"),
        ("drop_std_min_pair", "set_min_library $std_slow_db -min_version $std_fast_db", "set x $std_fast_db"),
        ("drop_macro_min_pair", "set_min_library $macro_slow_db -min_version $macro_fast_db", "set x $macro_fast_db"),
        ("wrong_operating_corner", "set_operating_conditions ssg0p9v125c", "set_operating_conditions ffg1p05vm40c"),
        ("wireload_removed", "set_wire_load_model -name ZeroWireload [current_design]", "set x ZeroWireload"),
        ("area_ceiling_relaxed", "set area_ceiling_um2 154608.7116945", "set area_ceiling_um2 160000.0"),
        ("missing_link_report", "link.rpt", "link_missing.rpt"),
        ("missing_setup_summary", "setup_posthold_summary_machine.txt", "setup_posthold_missing.txt"),
        ("missing_hold_summary", "hold_posthold_summary_machine.txt", "hold_posthold_missing.txt"),
        ("missing_drc_report", "constraint_design_rules_posthold.rpt", "constraint_rules_missing.rpt"),
        ("missing_output_ddc", "_m1630_residual_hold_closed.ddc", "_m1630_missing.ddc"),
    ]
    for label, old, new in tcl_mutations:
        outcomes.append(rejected(label, audit_tcl, mutate(tcl, old, new), sdc))
    for term in FORBIDDEN:
        outcomes.append(rejected("tcl_" + term, audit_tcl,
                                 tcl + "\n" + term + " [current_design]\n", sdc))
        outcomes.append(rejected("sdc_" + term, audit_tcl,
                                 tcl, sdc + "\n" + term + " [current_design]\n"))
    for label, old, new in (
        ("period_relaxed", "-period 3 ", "-period 3.2 "),
        ("setup_uncertainty_relaxed", "-setup 0.2 ", "-setup 0.1 "),
        ("hold_uncertainty_relaxed", "-hold 0.05 ", "-hold 0.04 "),
    ):
        outcomes.append(rejected(label, audit_tcl, tcl, mutate(sdc, old, new)))
    outcomes.append(rejected("propagated_clock", audit_tcl, tcl,
                             sdc + "\nset_propagated_clock [all_clocks]\n"))

    runner_mutations = [
        ("failed_ddc_as_input", 'INPUT_DDC="${M993_ORIGINAL}/netlist/', 'INPUT_DDC="${M1614_NEGATIVE}/netlist/'),
        ("drop_ddc_pin", "sha_exact d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56", "sha_exact 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_sdc_pin", "sha_exact cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5", "sha_exact 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_v_pin", "sha_exact 9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf", "sha_exact 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_svf_pin", "sha_exact 8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7", "sha_exact 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_negative_pin", "sha_exact cef7b0bb2cbcfbc0e723068e54018fbca5acf708f3cb0850e3d2a59677875d13", "sha_exact 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_docs359_pin", "sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4", "sha_exact 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_dc_shell_pin", "sha_tool 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2", "sha_tool 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_dc_actual_pin", "sha_tool bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391", "sha_tool 0000000000000000000000000000000000000000000000000000000000000000"),
        ("drop_no_home", '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}" \\\n+  >"${WORK}/dc.log"', '"${DC_SHELL}" -no_local_init -no_gui -f "${TCL}" \\\n+  >"${WORK}/dc.log"'),
        ("drop_no_local", '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}" \\\n+  >"${WORK}/dc.log"', '"${DC_SHELL}" -no_home_init -no_gui -f "${TCL}" \\\n+  >"${WORK}/dc.log"'),
        ("drop_no_gui", '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}" \\\n+  >"${WORK}/dc.log"', '"${DC_SHELL}" -no_home_init -no_local_init -f "${TCL}" \\\n+  >"${WORK}/dc.log"'),
        ("home_repurpose", "umask 002", "umask 002\nHOME=/tmp"),
        ("delete_review_gate", 'verify_dir_seal "${HAMMER_DIR}"', 'true "${HAMMER_DIR}"'),
        ("delete_release_gate", 'verify_file_seal "${RELEASE}"', 'true "${RELEASE}"'),
        ("delete_runner_pin", "M1630_EXPECTED_DC_RUNNER_SHA256", "M1630_UNPINNED_RUNNER"),
        ("delete_release_pin", "M1630_EXPECTED_DC_RELEASE_SHA256", "M1630_UNPINNED_RELEASE"),
        ("delete_uid_gate", "if p.stat().st_uid != os.getuid(): continue", "if False: continue"),
        ("delete_ancestry_gate", "int(p.name) in ancestry: continue", "False: continue"),
        ("delete_common_shell", "common_shell_exec", "unlisted_shell_exec"),
        ("delete_launch_lock", 'mkdir -- "${LOCK}" || fail "launch lock collision"', "true"),
        ("retry_true", "retry=false", "retry=true"),
        ("destructive_cleanup", "set -euo pipefail", "set -euo pipefail\nrm -rf /tmp/m1630"),
        ("insert_pt", "umask 002", "umask 002\npt_shell -f x.tcl"),
        ("insert_vcs", "umask 002", "umask 002\nvcs foo.sv"),
        ("drop_error_scan", "(Error|Fatal):", "(Notice|Info):"),
        ("drop_link_scan", "LINK-[0-9]+", "NO_LINK_SCAN"),
        ("drop_loop_scan", "combinational[ _-]*loop", "NO_COMB_LOOP_SCAN"),
        ("drop_timing_predicate", "timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')", "timing_ok=True"),
        ("drop_tns_predicate", "out['wns_ns']>=0 and out['tns_ns']==0 and out['violating_paths']==0", "out['wns_ns']>=0"),
        ("drop_area_gate", "area_ok=area<=ceiling", "area_ok=True"),
        ("drop_macro_gate", "macro_ok=(int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9)", "macro_ok=True"),
        ("drop_drc_gate", "positive=timing_ok and area_ok and macro_ok and drc_count==0", "positive=timing_ok and area_ok and macro_ok"),
        ("drop_finite_area", "if not math.isfinite(area) or area<=0", "if area<=0"),
        ("drop_guardband_leak_gate", "optimization guardband leaked into output SDC", "guardband not checked"),
        ("drop_negative_status", "SEALED_NEGATIVE_M1630_C1_RESIDUAL_HOLD_OR_AREA_GATE_FAILED__NO_RETRY", "PASS_RAW_M1630_C1_RESIDUAL_HOLD_CLOSED_DC_CANDIDATE_PENDING_FORMALITY_PT_RESULT_HAMMER"),
    ]
    for label, old, new in runner_mutations:
        if label == "drop_no_home":
            candidate = runner.replace("-no_home_init ", "")
        elif label == "drop_no_local":
            candidate = runner.replace("-no_local_init ", "")
        elif label == "drop_no_gui":
            candidate = runner.replace("-no_gui ", "")
        elif label in ("delete_runner_pin", "delete_release_pin", "retry_true",
                       "drop_error_scan", "drop_link_scan", "drop_loop_scan"):
            require(old in runner, "mutation token absent " + old)
            candidate = runner.replace(old, new)
        else:
            candidate = mutate(runner, old, new)
        outcomes.append(rejected(label, audit_runner, candidate))

    contract_mutations = []
    for label, path, value in (
        ("contract_dc_now", ("authorization", "dc_runs_now"), 1),
        ("contract_two_future", ("authorization", "future_dc_runs_max"), 2),
        ("contract_retry", ("authorization", "retry"), True),
        ("contract_m1614_input", ("input_policy", "failed_m1614_output_is_input"), True),
        ("contract_period", ("frozen_reported_point", "clock_period_ns"), 3.2),
        ("contract_setup_uncertainty", ("frozen_reported_point", "setup_uncertainty_ns"), 0.1),
        ("contract_hold_uncertainty", ("frozen_reported_point", "reported_hold_uncertainty_ns"), 0.04),
        ("contract_guardband", ("frozen_reported_point", "optimization_hold_guardband_ns"), 0.05),
        ("contract_macro_count", ("frozen_reported_point", "macro_count_exact"), 8),
        ("contract_area_ceiling", ("frozen_reported_point", "positive_area_ceiling_um2"), 160000.0),
        ("contract_second_compile", ("compile_contract", "all_compile_command_count"), 2),
        ("contract_ultra", ("compile_contract", "compile_ultra_count"), 1),
        ("contract_release_present", ("future_release_chain", "present_at_source_authoring"), True),
        ("contract_hold_claim", ("claim_boundary", "hold_closed"), True),
        ("contract_paper_claim", ("claim_boundary", "paper_citable"), True),
        ("contract_headline", ("claim_boundary", "headline"), True),
    ):
        clone = json.loads(json.dumps(contract))
        clone[path[0]][path[1]] = value
        contract_mutations.append((label, clone))
    for label, clone in contract_mutations:
        outcomes.append(rejected(label, audit_contract, clone, tcl, runner, author_test))

    require(len(set(outcomes)) == len(outcomes), "duplicate mutation labels")
    return {
        "schema": "m1631_m1630_c1_residual_hold_guardband_source_independent_hammer_v1",
        "status": "PASS",
        "attacks": len(outcomes),
        "rejections": len(outcomes),
        "outcomes": outcomes,
        "identity": {"tcl_sha256": sha256(TCL), "runner_sha256": sha256(RUNNER),
                     "contract_sha256": sha256(CONTRACT),
                     "author_test_sha256": sha256(AUTHOR_TEST),
                     "docs359_sha256": sha256(DOC359)},
        "execution": {"dc": 0, "vcs": 0, "pt": 0, "formality": 0,
                      "ptpx": 0, "gpu": 0, "remote": 0,
                      "attempt_created": False, "result_created": False},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    value = build()
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
