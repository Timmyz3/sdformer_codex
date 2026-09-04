#!/usr/bin/env python3
"""Independent, source-only M2134 hammer for the M2133 ICC2 P&R package.

This checker never launches ICC2, a license utility, another EDA tool, or a GPU
workload.  It reads the frozen sources and exercises the Python admission path
with independent temporary fixtures and mutations.
"""

from __future__ import print_function

import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent

TCL = HW / "dc_handoff/scripts/run_icc2_m2133_m2029_m2018_matched_macrofree_axis.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2133_m2134_m2029_m2018_matched_macrofree_icc2_pnr_one_shot.sh"
PARSER = HW / "system_simulator/scripts/parse_m2133_m2029_m2018_matched_macrofree_icc2_pnr.py"
CANON = HW / "system_simulator/scripts/canonicalize_m2133_icc2_corner_spef.py"
CONTRACT = HW / "contracts/m2133_m2029_m2018_matched_macrofree_icc2_pnr_source_contract_r1_20260904.json"
CONTRACT_SUM = CONTRACT.with_name(CONTRACT.name + ".sha256")
CONTRACT_SEAL = CONTRACT_SUM.with_name(CONTRACT_SUM.name + ".seal.sha256")
AUTHOR = HW / "reviews/m2133_m2029_m2018_matched_macrofree_icc2_pnr_source_author_receipt_r1_20260904"
M2130 = HW / "reviews/m2130_m2129_m2029_m2018_matched_macrofree_icc2_pnr_source_hammer_r1_20260904"
MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ICC2_DOC = Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat2/write_parasitics.2")

EXPECTED = {
    "tcl": "0df08207da8c5601c0b23b21bff9ee84e73594101ec654ee4a7071a191ca1e5b",
    "runner": "3cde47d675728007782e34020356ff0196df2e82bdc9cefe456e2ed86ae542d8",
    "parser": "950eaac4ff9842a08c0485858391a987b6242c063293fa318b01b4d0f63e987f",
    "canonicalizer": "2b70cf3a087d67f73f3e63f5ca0c00351719e0118cf7ff476194faba8914e0cc",
    "contract": "cf513c2b879d2668296680a655f3fd8c37ea32ef825b2929514ab6b89b1b1b7a",
    "manifest": "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
    "m2130_review": "e95c5d4f90ac5a11c558eb86134ed33bca4124d57af703c0a235a716b18f873a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "icc2_doc": "3826117f39a9f3eb8ad7be947dc9f86653bb251a3ba328086b8cce8e56ad429b",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_checksum_line(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise AssertionError("checksum sidecar must contain one line: %s" % path)
    match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", lines[0])
    if not match:
        raise AssertionError("malformed checksum sidecar: %s" % path)
    return match.group(1), match.group(2)


def verify_double_sealed_dir(path):
    if not path.is_dir() or path.is_symlink():
        raise AssertionError("not a real sealed directory: %s" % path)
    sums = path / "SHA256SUMS"
    seal = path / "SHA256SUMS.seal.sha256"
    sum_hash, sum_name = parse_checksum_line(seal)
    if sum_name != "SHA256SUMS" or sum_hash != sha(sums):
        raise AssertionError("bad outer seal: %s" % path)
    listed = {}
    for raw in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        if not match or match.group(2) in listed:
            raise AssertionError("malformed/duplicate inner manifest in %s" % path)
        listed[match.group(2)] = match.group(1)
    actual = sorted(
        str(p.relative_to(path)) for p in path.rglob("*")
        if p.is_file() and not p.is_symlink()
        and p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")
    )
    symlinks = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_symlink()]
    if symlinks or sorted(listed) != actual:
        raise AssertionError("non-exhaustive or symlinked seal in %s" % path)
    for name, expected in listed.items():
        if sha(path / name) != expected:
            raise AssertionError("inner seal mismatch: %s/%s" % (path, name))
    return len(actual)


checks = {}


def need(condition, name):
    if name in checks:
        raise AssertionError("duplicate check name %s" % name)
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError("FAILED CHECK: %s" % name)


def reject(callable_, name, contains=None):
    try:
        callable_()
    except Exception as exc:
        if contains is not None and contains not in str(exc):
            raise AssertionError("%s rejected for wrong reason: %r" % (name, exc))
        need(True, name)
        return
    need(False, name)


def make_canonical_dirs(root):
    raw = root / "raw_parasitics"
    output = root / "output"
    raw.mkdir(parents=True)
    output.mkdir()
    return raw, output, root / "receipt.json"


def make_axis(parser_mod, root, axis, area=260000.0, setup=0.003, hold=0.001):
    reports = root / "reports"
    output = root / "output"
    reports.mkdir(parents=True)
    output.mkdir()
    ports = "".join("port_%04d\n" % i for i in range(4551))
    (reports / "ports_sorted.txt").write_text(ports, encoding="utf-8")
    report_text = {
        "actual_floorplan.txt": "die_boundary={{0 0} {800 0} {800 800} {0 800}}\ncore_bbox={{40 40} {760 760}}\n",
        "actual_routing_layers.rpt": "Minimum routing layer M2\nMaximum routing layer M8\n",
        "actual_cts_cells.txt": "CKBD1\nCKND1\n",
        "actual_hold_cells.txt": "BUFF1\nDEL1\nINV1\n",
        "actual_scenarios.rpt": "func_ss_setup setup\nfunc_ff_hold hold\nfunc_tt_power power\n",
    }
    for name, text in report_text.items():
        (reports / name).write_text(text, encoding="utf-8")
    policy_blob = b"".join((reports / n).read_bytes() for n in (
        "actual_routing_layers.rpt", "actual_cts_cells.txt", "actual_hold_cells.txt"))
    facts = {
        "status": parser_mod.PASS_TOKEN,
        "axis": axis,
        "top": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend",
        "public_port_count": "4551",
        "input_master_count": "94",
        "tt_master_coverage": "94/94",
        "ss_master_coverage": "94/94",
        "ff_master_coverage": "94/94",
        "physical_master_coverage": "94/94",
        "unresolved_reference_count": "0",
        "direct_unbound_cell_count": "0",
        "direct_unmapped_cell_count": "0",
        "direct_black_box_cell_count": "0",
        "direct_black_box_ref_name_count": "0",
        "not_repaired_mismatch_count": "0",
        "accepted_mismatch_count": "0",
        "logical_physical_mismatch_count": "0",
        "routing_layer_gate_count": "9",
        "via_layer_gate_count": "8",
        "route_check_return": "1",
        "route_open_net_count": "0",
        "route_drc_violation_count": "0",
        "pre_placement_check_return": "1",
        "pre_clock_check_return": "1",
        "pre_route_check_return": "1",
        "die_bbox_um": "0,0,800,800",
        "core_bbox_um": "40,40,760,760",
        "die_boundary_actual": "{{0 0} {800 0} {800 800} {0 800}}",
        "core_bbox_actual": "{{40 40} {760 760}}",
        "floorplan_policy": "fixed_die_core_800_720um_v1",
        "pin_policy": "sorted_four_side_round_robin_exact_location_v1",
        "route_layers": "M2:M8",
        "cts_cell_policy": "CKBD_and_CKND_only_v1",
        "hold_cell_policy": "DEL_BUFF_INV_only_v1",
        "clock_period_ns": "3.000",
        "setup_uncertainty_ns": "0.200",
        "hold_uncertainty_ns": "0.050",
        "parasitic_tech": "n28_1p9m_6x1z1u_typ",
        "parasitic_corner_scope": "same_typical_rc_on_ss_ff_tt",
        "setup_scenario_actual": "func_ss_setup",
        "hold_scenario_actual": "func_ff_hold",
        "power_scenario_actual": "func_tt_power",
        "common_external_sram_bytes": "294912",
        "common_external_sram_integrated": "false",
        "propagated_clock": "true",
        "macro_instances": "0",
        "physical_sdc_sha256": "a" * 64,
        "flow_tcl_sha256": EXPECTED["tcl"],
        "floorplan_actual_sha256": sha(reports / "actual_floorplan.txt"),
        "routing_policy_sha256": hashlib.sha256(policy_blob).hexdigest(),
        "scenario_policy_sha256": sha(reports / "actual_scenarios.rpt"),
        "port_inventory_sha256": sha(reports / "ports_sorted.txt"),
        "setup_wns_ns": "%.6f" % setup,
        "hold_wns_ns": "%.6f" % hold,
        "routed_standard_cell_area_um2": "%.6f" % area,
        "routed_leaf_cell_count": "270000",
        "routed_sequential_cell_count": "74460",
        "clock_like_cell_count": "900",
        "hold_like_cell_count": "2000",
    }
    (root / "machine_facts.txt").write_text(
        "".join("%s=%s\n" % item for item in facts.items()), encoding="utf-8")
    (root / "RUN_COMPLETE.txt").write_text(parser_mod.PASS_TOKEN + "\n", encoding="utf-8")
    receipt = {
        "schema": "m2133_icc2_corner_spef_canonicalization_r1_v1",
        "status": "PASS_M2133_UNIQUE_TT_CORNER_SPEF_CANONICALIZED",
        "raw_name": "routed.n28_1p9m_6x1z1u_typ_25.spef",
        "canonical_name": "routed.spef",
        "parasitic_technology": "n28_1p9m_6x1z1u_typ",
        "corner": "tt_power",
        "temperature_c": 25.0,
        "candidate_count_before_rename": 1,
        "scenario_metadata_is_not_spef": True,
        "atomic_rename": True,
    }
    (root / "spef_canonicalization_receipt.json").write_text(
        json.dumps(receipt) + "\n", encoding="utf-8")
    census_keys = (
        "direct_unbound_cell_count", "direct_unmapped_cell_count",
        "direct_black_box_cell_count", "direct_black_box_ref_name_count",
        "not_repaired_mismatch_count", "accepted_mismatch_count",
        "logical_physical_mismatch_count",
    )
    (reports / "postlink_reference_census.txt").write_text(
        "status=PASS_M2133_POSTLINK_DIRECT_CELL_REFERENCE_CENSUS\n" +
        "".join("%s=%s\n" % (key, facts[key]) for key in census_keys), encoding="utf-8")
    standard_reports = (
        "reference_libraries.rpt", "design_library.rpt", "design_mismatch.rpt",
        "pre_placement_check.rpt", "pre_clock_check.rpt", "pre_route_check.rpt",
        "qor.rpt", "clock_qor.rpt", "congestion.rpt", "wirelength.rpt",
        "final_design.rpt", "vectorless_power_diagnostic.rpt",
    )
    for name in standard_reports:
        (reports / name).write_text("independent fixture\n", encoding="utf-8")
    (reports / "route_check.rpt").write_text(
        "Total number of open nets = 0\nTotal number of DRC violations = 0\n", encoding="utf-8")
    (reports / "timing_setup.rpt").write_text(
        "slack (MET) %.6f\n" % setup, encoding="utf-8")
    (reports / "timing_hold.rpt").write_text(
        "slack (MET) %.6f\n" % hold, encoding="utf-8")
    metric_keys = (
        "setup_wns_ns", "hold_wns_ns", "routed_standard_cell_area_um2",
        "routed_leaf_cell_count", "routed_sequential_cell_count",
        "clock_like_cell_count", "hold_like_cell_count",
    )
    (reports / "postroute_metrics.txt").write_text(
        "status=PASS_M2133_POSTROUTE_METRICS_FROM_LIVE_QUERIES\n" +
        "".join("%s=%s\n" % (key, facts[key]) for key in metric_keys), encoding="utf-8")
    pins = "\n".join(
        "- port_%04d + NET port_%04d + DIRECTION INPUT + USE SIGNAL "
        "+ LAYER M3 ( 0 0 ) ( 10 10 ) + FIXED ( %d 0 ) N ;" % (i, i, i)
        for i in range(4551)
    )
    (output / "routed.def").write_text(
        "VERSION 5.8 ;\nUNITS DISTANCE MICRONS 1000 ;\n"
        "DIEAREA ( 0 0 ) ( 800000 800000 ) ;\nPINS 4551 ;\n" + pins +
        "\nEND PINS\nEND DESIGN\n", encoding="utf-8")
    for name in ("routed.v", "routed.sdc", "routed.spef"):
        (output / name).write_text("independent fixture\n", encoding="utf-8")
    return root


def pair_fixture(parser_mod, temp):
    root = Path(temp)
    ordinary = make_axis(parser_mod, root / "ordinary", "ordinary_lru4", area=260000.0)
    tsbg = make_axis(parser_mod, root / "tsbg", "tsbg_b4", area=260100.0)
    return ordinary, tsbg


def run():
    tcl = TCL.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    parser_text = PARSER.read_text(encoding="utf-8")
    canon_text = CANON.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parser_mod = load_module("m2134_parser_target", PARSER)
    canon_mod = load_module("m2134_canonicalizer_target", CANON)

    for key, path in (("tcl", TCL), ("runner", RUNNER), ("parser", PARSER),
                      ("canonicalizer", CANON), ("contract", CONTRACT),
                      ("manifest", MANIFEST), ("docs359", DOCS359),
                      ("icc2_doc", ICC2_DOC)):
        need(path.is_file() and not path.is_symlink() and sha(path) == EXPECTED[key],
             "exact_sha_%s" % key)
    need(sha(M2130 / "review.json") == EXPECTED["m2130_review"], "exact_sha_m2130_review")

    contract_hash, contract_name = parse_checksum_line(CONTRACT_SUM)
    need(contract_hash == EXPECTED["contract"] and contract_name == CONTRACT.name,
         "contract_inner_checksum_exact")
    seal_hash, seal_name = parse_checksum_line(CONTRACT_SEAL)
    need(seal_hash == sha(CONTRACT_SUM) and seal_name == CONTRACT_SUM.name,
         "contract_outer_checksum_exact")
    need(verify_double_sealed_dir(AUTHOR) == 4, "author_receipt_exhaustive_double_seal")
    need(verify_double_sealed_dir(M2130) == 5, "m2130_failure_review_exhaustive_double_seal")

    # The installed catman page uses both underline ("_\bX") and bold
    # ("X\bX") overstrikes.  Removing the character before each backspace
    # yields the actual command-reference text in both cases.
    doc_plain = re.sub(r".\x08", "", ICC2_DOC.read_bytes().decode("latin-1"))
    need("<file_name>.<parasitic_tech_name>_<temperature>" in doc_plain,
         "installed_doc_confirms_output_prefix_suffix_rule")
    need("-corner corner_name" in doc_plain and "single corner" in doc_plain,
         "installed_doc_confirms_single_corner_option")
    need("XX.spef_scenario" in doc_plain, "installed_doc_confirms_scenario_sidecar")

    need('file mkdir "$axis_dir/reports" "$axis_dir/output" "$axis_dir/raw_parasitics"' in tcl,
         "tcl_creates_dedicated_raw_parasitics_directory")
    exact_write = 'write_parasitics -output "$axis_dir/raw_parasitics/routed" -format spef -corner tt_power'
    need(tcl.count(exact_write) == 1, "tcl_exact_single_tt_raw_prefix_write")
    need("-compress" not in tcl, "tcl_does_not_request_gzip")
    need(tcl.index("create_corner tt_power") < tcl.index(exact_write), "tt_corner_created_before_write")
    need(tcl.index("-early_temperature 25") < tcl.index(exact_write) and
         tcl.index("-late_temperature 25") < tcl.index(exact_write),
         "tt_25c_parasitic_parameters_precede_write")
    need('write_parasitics -output "$axis_dir/output/routed.spef"' not in tcl,
         "m2130_broken_spef_prefix_not_regressed")

    link_at = tcl.index("link_block -force -verbose")
    nxt_at = tcl.index("read_parasitic_tech -tlup")
    direct_tokens = (
        'get_cells -hierarchical -quiet -filter "is_unbound == true"',
        'get_cells -hierarchical -quiet -filter "is_unmapped == true"',
        'get_cells -hierarchical -quiet -filter "is_black_box == true"',
        "direct_black_box_ref_name_count",
    )
    for token in direct_tokens:
        need(link_at < tcl.index(token) < nxt_at, "direct_postlink_query_%s" % re.sub(r"\W+", "_", token)[:60])
    need("not_repaired_mismatch_count" in tcl and "accepted_mismatch_count" in tcl and
         "logical_physical_mismatch_count" in tcl, "separate_mismatch_census_preserved")
    for view in ("tt_covered", "ss_covered", "ff_covered", "physical_covered"):
        need(("%s != 94" % view) in tcl, "coverage_94_gate_%s" % view)
    need("routing_layer_gate_count" in tcl and "via_layer_gate_count" in tcl,
         "routing_and_via_layer_gates_preserved")
    for report in ("design_library.rpt", "final_design.rpt", "vectorless_power_diagnostic.rpt",
                   "postlink_reference_census.txt", "timing_setup.rpt", "timing_hold.rpt",
                   "postroute_metrics.txt"):
        need(report in tcl and report in parser_text, "required_report_%s" % report.replace(".", "_"))
    need(tcl.count("-significant_digits 6") == 2, "timing_reports_have_six_significant_digits")
    need("m2133_parse_route_report" in tcl and "route_open_count" in tcl and "route_drc_count" in tcl,
         "route_counts_bound_to_machine_facts")
    need("PASS_M2133_POSTROUTE_METRICS_FROM_LIVE_QUERIES" in tcl and
         "PASS_M2133_POSTROUTE_METRICS_FROM_LIVE_QUERIES" in parser_text,
         "live_query_metrics_bound")
    need("write_def" in tcl and "write_verilog" in tcl and "write_sdc" in tcl,
         "routed_def_verilog_sdc_outputs_present")

    need(r'RAW_NAME = re.compile(r"^routed\.n28_1p9m_6x1z1u_typ_(25(?:\.0+)?)\.spef$")' in canon_text,
         "canonicalizer_exact_raw_name_regex")
    need('p.name.endswith(".spef")' in canon_text and "len(candidates) != 1" in canon_text,
         "canonicalizer_enumerates_exactly_one_real_spef_shape")
    need("source.is_symlink()" in canon_text and "source.stat().st_size <= 0" in canon_text,
         "canonicalizer_rejects_symlink_and_empty")
    need("canonical.exists() or canonical.is_symlink()" in canon_text,
         "canonicalizer_rejects_preexisting_canonical")
    need('os.replace(str(source), str(canonical))' in canon_text,
         "canonicalizer_atomic_same_filesystem_rename")
    need('CANONICAL_NAME = "routed.spef"' in canon_text, "canonicalizer_literal_output_name")
    need("routed.spef_scenario" in canon_text and "endswith(\".spef\")" in canon_text,
         "scenario_metadata_excluded_from_candidates")

    canonical_cases = []
    with tempfile.TemporaryDirectory() as temp:
        raw, output, receipt = make_canonical_dirs(Path(temp))
        src = raw / "routed.n28_1p9m_6x1z1u_typ_25.000.spef"
        src.write_text("*SPEF independent\n", encoding="utf-8")
        (raw / "routed.spef_scenario").write_text("tt_power\n", encoding="utf-8")
        result = canon_mod.canonicalize(raw, output, receipt)
        need(result["corner"] == "tt_power" and result["temperature_c"] == 25.0,
             "canonicalizer_valid_tt25_identity")
        need(not src.exists() and (output / "routed.spef").read_text() == "*SPEF independent\n",
             "canonicalizer_valid_atomic_move_effect")
        need(json.loads(receipt.read_text())["raw_name"].endswith("25.000.spef"),
             "canonicalizer_receipt_records_raw_name")
    for case in ("none", "scenario_only", "multiple", "wrong_temp", "wrong_tech",
                 "empty", "symlink", "preexisting", "raw_gzip_only"):
        with tempfile.TemporaryDirectory() as temp:
            raw, output, receipt = make_canonical_dirs(Path(temp))
            valid = raw / "routed.n28_1p9m_6x1z1u_typ_25.spef"
            if case == "scenario_only":
                (raw / "routed.spef_scenario").write_text("metadata\n")
            elif case == "multiple":
                valid.write_text("tt\n")
                (raw / "routed.n28_1p9m_6x1z1u_typ_125.spef").write_text("ss\n")
            elif case == "wrong_temp":
                (raw / "routed.n28_1p9m_6x1z1u_typ_125.spef").write_text("ss\n")
            elif case == "wrong_tech":
                (raw / "routed.wrong_25.spef").write_text("wrong\n")
            elif case == "empty":
                valid.write_text("")
            elif case == "symlink":
                target = raw / "payload"
                target.write_text("tt\n")
                valid.symlink_to(target)
            elif case == "preexisting":
                valid.write_text("tt\n")
                (output / "routed.spef").write_text("stale\n")
            elif case == "raw_gzip_only":
                (raw / "routed.n28_1p9m_6x1z1u_typ_25.spef.gz").write_text("gzip-shaped\n")
            reject(lambda r=raw, o=output, p=receipt: canon_mod.canonicalize(r, o, p),
                   "canonicalizer_rejects_%s" % case)

    with tempfile.TemporaryDirectory() as temp:
        ordinary, tsbg = pair_fixture(parser_mod, temp)
        result = parser_mod.parse_pair(ordinary, tsbg)
        need(result["status"].startswith("PASS_RAW_M2135"), "parser_accepts_independent_clean_pair")

    parser_mutations = (
        "missing_canonical", "gzip_substitution", "scenario_substitution",
        "canonical_symlink", "wrong_receipt_corner", "wrong_receipt_temp",
        "wrong_receipt_raw_name", "route_open_999", "route_drc_777",
        "direct_unbound", "missing_design_library", "violated_minus999",
        "live_area_mismatch", "def_pin_mismatch", "routing_policy_mismatch",
    )
    for mutation in parser_mutations:
        with tempfile.TemporaryDirectory() as temp:
            ordinary, tsbg = pair_fixture(parser_mod, temp)
            spef = tsbg / "output/routed.spef"
            if mutation == "missing_canonical":
                spef.unlink()
            elif mutation == "gzip_substitution":
                spef.rename(tsbg / "output/routed.spef.gz")
            elif mutation == "scenario_substitution":
                spef.unlink()
                (tsbg / "output/routed.spef_scenario").write_text("metadata\n")
            elif mutation == "canonical_symlink":
                spef.unlink()
                target = tsbg / "output/payload"
                target.write_text("payload\n")
                spef.symlink_to(target)
            elif mutation.startswith("wrong_receipt_"):
                rp = tsbg / "spef_canonicalization_receipt.json"
                data = json.loads(rp.read_text())
                if mutation == "wrong_receipt_corner":
                    data["corner"] = "ss_setup"
                elif mutation == "wrong_receipt_temp":
                    data["temperature_c"] = 125.0
                else:
                    data["raw_name"] = "routed.wrong_25.spef"
                rp.write_text(json.dumps(data) + "\n")
            elif mutation in ("route_open_999", "route_drc_777"):
                rp = tsbg / "reports/route_check.rpt"
                text = rp.read_text()
                text = text.replace("open nets = 0", "open nets = 999") if mutation.endswith("999") \
                    else text.replace("DRC violations = 0", "DRC violations = 777")
                rp.write_text(text)
            elif mutation == "direct_unbound":
                fp = tsbg / "machine_facts.txt"
                fp.write_text(fp.read_text().replace("direct_unbound_cell_count=0", "direct_unbound_cell_count=1"))
            elif mutation == "missing_design_library":
                (tsbg / "reports/design_library.rpt").unlink()
            elif mutation == "violated_minus999":
                (tsbg / "reports/timing_setup.rpt").write_text("slack (VIOLATED) -999.000000\n")
            elif mutation == "live_area_mismatch":
                fp = tsbg / "reports/postroute_metrics.txt"
                fp.write_text(fp.read_text().replace("routed_standard_cell_area_um2=260100.000000",
                                                     "routed_standard_cell_area_um2=1.000000"))
            elif mutation == "def_pin_mismatch":
                fp = tsbg / "output/routed.def"
                fp.write_text(fp.read_text().replace("FIXED ( 42 0 ) N", "FIXED ( 43 0 ) N", 1))
            elif mutation == "routing_policy_mismatch":
                fp = tsbg / "reports/actual_routing_layers.rpt"
                fp.write_text(fp.read_text().replace("M8", "M9"))
            reject(lambda o=ordinary, t=tsbg: parser_mod.parse_pair(o, t),
                   "parser_rejects_%s" % mutation)

    need('CANONICALIZER="${HW_ROOT}/system_simulator/scripts/canonicalize_m2133_icc2_corner_spef.py"' in runner,
         "runner_pins_canonicalizer_path")
    for key in ("runner_sha256", "tcl_sha256", "parser_sha256", "canonicalizer_sha256", "contract_sha256"):
        need(key in runner, "runner_requires_m2134_identity_%s" % key)
    need("verify_dir_seal \"${M2134}\"" in runner, "runner_verifies_m2134_double_seal")
    need("score_over_100'] >= 95" in runner and
         "{'p0': 0, 'p1': 0, 'p2': 0}" in runner,
         "runner_enforces_m2134_score_and_zero_severities")
    need("'license_queries': 1, 'icc2_shell_runs': 2" in runner and
         "'all_other_eda_runs': 0, 'automatic_retry': False" in runner,
         "runner_enforces_exact_m2135_budget")
    need(runner.count('"${LMUTIL}" lmstat') == 1, "runner_has_exactly_one_license_query_site")
    need(runner.count('"${ICC2}" -f "${TCL}"') == 1 and
         "for index in 0 1" in runner, "runner_has_two_sequential_icc2_axes")
    need('automatic_retry' in runner and 'False' in runner and "retry=false" in runner,
         "runner_no_retry_contract_and_attempt_marker")
    need('[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]]' in runner,
         "runner_requires_pristine_result_attempt_work_lock")
    need(runner.index('mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"') <
         runner.index('"${LMUTIL}" lmstat'), "attempt_consumed_before_license_query")
    icc2_pos = runner.index('"${ICC2}" -f "${TCL}"')
    canon_pos = runner.index('/usr/libexec/platform-python3.6 -I "${CANONICALIZER}"')
    parser_pos = runner.index('"${PARSER}" --ordinary-dir')
    need(icc2_pos < canon_pos < parser_pos, "canonicalizer_runs_after_axis_and_before_pair_parser")
    need('[[ -s "${axis_dir}/output/routed.spef" && ! -L "${axis_dir}/output/routed.spef" ]]' in runner,
         "runner_requires_literal_nonempty_nonsymlink_canonical_spef")
    need("spef_canonicalization_receipt.json" in runner and
         "PASS_M2133_UNIQUE_TT_CORNER_SPEF_CANONICALIZED" in runner,
         "runner_requires_canonicalization_receipt_and_pass_token")
    need(runner.count("cmp -s --") == 6, "runner_matches_six_physical_policy_artifacts")
    need("(operating_conditions, wireload_headers, wireload_continuations) == (1, 1, 1)" in runner,
         "runner_sdc_normalization_cardinality_fail_closed")
    need("same-UID EDA collision" in runner and "MemAvailable:" in runner and "Committed_AS:" in runner,
         "runner_collision_and_memory_gates_before_execution")
    need("FAILED_OR_INCOMPLETE_DO_NOT_CITE" in runner and "failed_or_incomplete" in runner,
         "runner_failure_quarantine_and_do_not_cite")
    need("mv -T -- \"${WORK}\" \"${RESULT}\"" in runner,
         "runner_atomic_result_publish_after_seal")

    need(contract["authorization"] == {
        "license_queries": 0, "icc2_shell_runs": 0, "all_other_eda_runs": 0,
        "gpu_runs": 0, "automatic_retry": False,
        "release_condition": contract["authorization"]["release_condition"],
    }, "source_contract_authorization_is_zero")
    need(contract["admission"]["strict_spef_name_regex"] == "^routed[.]spef$",
         "contract_operational_canonical_spef_is_literal")
    need(contract["m2130_spef_repair"]["raw_name_regex"] ==
         "^routed[.]n28_1p9m_6x1z1u_typ_25([.]0+)?[.]spef$",
         "contract_raw_regex_exact")
    need(contract["m2130_spef_repair"]["identity_gate"] == {
        "parasitic_technology": "n28_1p9m_6x1z1u_typ",
        "corner": "tt_power", "temperature_c": 25.0,
    }, "contract_tt_corner_identity_exact")
    need(contract["protected"]["docs359_sha256"] == EXPECTED["docs359"],
         "contract_protected_docs359_identity_exact")

    # The inherited prose still mentions an optional gzip name.  It is not an
    # executable admission rule: the newer strict regex, canonicalizer, runner,
    # and parser all require the literal uncompressed routed.spef.  Record this
    # as a non-severity editorial observation rather than weakening the gate.
    need("routed.spef.gz" in contract["inherited_m2111_repairs"]["spef_admission"],
         "observed_stale_inherited_gzip_phrase")
    need("routed.spef.gz" not in contract["admission"]["strict_spef_name_regex"],
         "operative_contract_does_not_admit_gzip")

    true_count = sum(1 for value in checks.values() if value)
    if true_count != len(checks):
        raise AssertionError("not all mechanical checks passed")
    summary = {
        "schema": "m2134_m2133_source_hammer_mechanical_checks_r1_v1",
        "status": "PASS_M2134_INDEPENDENT_MECHANICAL_CHECKS",
        "checks_total": len(checks),
        "checks_passed": true_count,
        "checks": checks,
        "eda_invoked": False,
        "license_query_invoked": False,
        "gpu_invoked": False,
        "installed_write_parasitics_doc_sha256": sha(ICC2_DOC),
        "source_identity": {
            "runner_sha256": sha(RUNNER),
            "tcl_sha256": sha(TCL),
            "parser_sha256": sha(PARSER),
            "canonicalizer_sha256": sha(CANON),
            "contract_sha256": sha(CONTRACT),
            "docs359_sha256": sha(DOCS359),
        },
    }
    (OUT / "mechanical_checks.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "PASS_M2134_INDEPENDENT_MECHANICAL_CHECKS",
        "checks_total=%d" % len(checks),
        "checks_passed=%d" % true_count,
        "eda_invoked=false",
        "license_query_invoked=false",
        "gpu_invoked=false",
    ]
    lines.extend("PASS %s" % name for name in sorted(checks))
    (OUT / "mechanical_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PASS_M2134_INDEPENDENT_MECHANICAL_CHECKS %d/%d" % (true_count, len(checks)))


if __name__ == "__main__":
    run()
