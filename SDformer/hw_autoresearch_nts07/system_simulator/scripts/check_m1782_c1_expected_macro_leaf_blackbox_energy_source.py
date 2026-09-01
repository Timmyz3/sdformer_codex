#!/usr/bin/env python3
"""Fail-closed source/runtime checker for the M1782 C1 PTPX successor.

M1782 changes only the post-link black-box admission gate.  It accepts exactly
the nine linked TS1N28 SRAM Liberty leaves and rejects every missing, extra,
hierarchical, or wrong-reference black box.  It deliberately requests fresh
mapped compile, simulation, SAIF and PTPX evidence rather than recycling the
unsealed M1772 private build.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
OLD_CHECKER = HW / "system_simulator/scripts/check_m1772_c1_m1701_two_bank_public_warmup_energy_source.py"
OLD_SPEC = importlib.util.spec_from_file_location("m1772_checker_for_m1782", str(OLD_CHECKER))
if OLD_SPEC is None or OLD_SPEC.loader is None:
    raise RuntimeError("M1772 predecessor checker unavailable")
OLD = importlib.util.module_from_spec(OLD_SPEC)
OLD_SPEC.loader.exec_module(OLD)

DESIGN = OLD.DESIGN
NET = OLD.NET
SDC = OLD.SDC
STD_TT = OLD.STD_TT
STD_SS = OLD.STD_SS
MACRO_DB = OLD.MACRO_DB
TOP = OLD.TOP
SAIF_SCOPE = OLD.SAIF_SCOPE
TB = OLD.TB
FILELIST = OLD.FILELIST
UCLI = OLD.UCLI
DOC359 = OLD.DOC359

PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1782_c1_m1701_expected_macro_leaf_blackbox_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1782_m1772_c1_expected_macro_leaf_blackbox_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1782_c1_expected_macro_leaf_blackbox_energy_source.py"
CONTRACT = HW / "contracts/m1782_m1772_c1_expected_macro_leaf_blackbox_energy_source_contract_r1_20260902.json"

M1772_FAILURE = HW / "results/m1772_c1_two_bank_public_warmup_energy_r1_20260902.failed_or_incomplete.quarantine"
M1772_PRIVATE = HW / "results/m1772_c1_two_bank_public_warmup_energy_r1_20260902.private_build.unsealed_do_not_cite"
M1772_FAILURE_JSON = M1772_FAILURE / "failure.json"
M1772_SIM_LOG = M1772_PRIVATE / "candidate/mapped_sim.log"
M1772_SAIF = M1772_PRIVATE / "candidate/m1772_c1_directed_component.saif"
M1772_PT_LOG = M1772_PRIVATE / "candidate/ptpx/ptpx.log"
M1772_COMPILE_LOG = M1772_PRIVATE / "build/compile.log"

MACRO_REF = "TS1N28HPCPHVTB128X128M4S"
EXPECTED_MACRO_NAMES = tuple(
    "u_parent_scratch/g_slice_%d__u_parent_sram" % index
    for index in range(9))

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    TB: "21ead36213d89a425a170fce85823994562e8410c9bd24b338b7cf29f02a750d",
    FILELIST: "9da54a6a3b60a05602adbb0bb4440d0ac95c035c73a1b69d6589dab2f8664906",
    UCLI: "beaa724867c28198d600840b2b8fe7dcbe665ad7cf6ee9449c92be6ccafccef7",
    M1772_FAILURE_JSON: "303f429913606bca3d32cc39f3941357ae07ccf1b09cd2a236888ce26ec96083",
    M1772_FAILURE / "SHA256SUMS": "14b11af0ebb7eb73147cd1f6dd3a995714bfeea154e2f655958d7ba2b5822c0f",
    M1772_FAILURE / "SHA256SUMS.seal.sha256": "3bdd311e4bae71e1b49c7396ed5900f93f74edb4cf2fb4af9a5181363251109c",
    M1772_SIM_LOG: "39b900d0f7d54be396e520866dddc6b0622214c7d92027b5558949c993b21167",
    M1772_SAIF: "0e397565e6a2141cb29bca924b0353f85caf1c8328323ac35a827e08f9854220",
    M1772_PT_LOG: "c5115cb32a75765b0f9302e77b07ef3874bb87a265756ce507084ad5c547c3e0",
    M1772_COMPILE_LOG: "9049e76a7094d775e582784f20d18ca5110b121fe85e0c8e30a2d08b1d9dfbce",
}

CLAIMS = dict((key, False) for key in (
    "launch_authorized", "launch_executed", "mapped_vcs", "production_saif",
    "ptpx", "logic_power", "component_energy", "total_c1_energy",
    "energy_per_frame", "performance", "system_speedup", "paper_ppa_ready",
    "headline"))


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "JSON absent/nonregular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest member")
        need(sha(root / rel) == digest, "manifest drift " + name)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink in sealed root")
        if path.is_file() and path.name not in {
                "SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    need(actual == listed, "sealed population drift")


def strip_tcl_comments(text):
    return "\n".join(row for row in text.splitlines()
                     if not row.lstrip().startswith("#"))


def validate_m1772_failure():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed M1772 forensic identity drift " + str(path))
    verify_seal(M1772_FAILURE)
    failure = strict_json(M1772_FAILURE_JSON)
    need(failure == {
        "attempt_consumed": True, "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 1, "saif_files": 1, "simv_runs": 1,
                   "vcs_compiles": 1},
        "error": "Failure", "phase": "PTPX",
        "status": "FAILED_OR_INCOMPLETE"},
        "M1772 failure disposition drift")
    runtime = OLD.validate_runtime(M1772_SIM_LOG)
    need(runtime == {
        "status": "PASS_M1772_PUBLIC_PORT_RUNTIME",
        "measurement_cycles": 253, "issue_accepts": 96,
        "parent_edges": 48, "macro_reads": 46, "macro_writes": 34,
        "forwards": 2, "dead_write_elisions": 30,
        "log_sha256": FIXED[M1772_SIM_LOG]},
        "M1772 runtime forensics drift")
    saif = OLD.validate_saif(M1772_SAIF, 253)
    need(saif.get("activity_forms_per_tag") == 117690
         and saif.get("tx_nonzero") == 0
         and saif.get("duration_ns") == 759.0,
         "M1772 SAIF forensics drift")
    pt = M1772_PT_LOG.read_text(errors="strict")
    need(pt.count("Error: M1750_FAIL_BLACK_BOX_AFTER_LINK") == 1
         and "Design '" + DESIGN + "' was successfully linked." in pt
         and "There are 113081 leaf cells, ports, hiers and 117690 nets" in pt
         and "read_saif -strip_path" not in pt
         and "PASS_M1750_C1_M1701_PUBLIC_PORT_MAPPED_WHOLE_COMPONENT_PTPX_TOOL_COMPLETE" not in pt,
         "M1772 PT failure forensics drift")
    return {"failure_phase": "PTPX_post_link_pre_SAIF",
            "vcs_compile": "PASS", "mapped_sim": "PASS",
            "measurement_cycles": 253, "saif_activity_forms_per_tag": 117690,
            "saif_tx_nonzero": 0, "ptpx_power_result": False,
            "automatic_retry": False}


def validate_sources():
    predecessor = OLD.validate_sources()
    need(predecessor.get("status") ==
         "PASS_M1772_TWO_BANK_PUBLIC_WARMUP_SOURCE_ONLY_NO_EDA",
         "M1772 predecessor source no longer valid")
    failure = validate_m1772_failure()
    pt = PT_TCL.read_text(errors="strict")
    active_pt = strip_tcl_comments(pt)
    for token in (
            "is_black_box==true", "ref_name == $macro_cell",
            "expected_macro_count 9", "black_box_count=$black_box_count",
            "M1782_FAIL_BLACK_BOX_COUNT_", "M1782_FAIL_MACRO_COUNT_",
            "M1782_FAIL_EXPECTED_MACRO_BLACK_BOX_MISSING_",
            "M1782_FAIL_UNEXPECTED_BLACK_BOX_",
            "M1782_FAIL_BLACK_BOX_WRONG_REF_",
            "M1782_FAIL_BLACK_BOX_NOT_LEAF_",
            "M1782_FAIL_EXPECTED_BLACK_BOX_ATTRIBUTE_",
            "black_box_inventory_machine.rpt",
            "read_saif -strip_path $::env(M1782_SAIF_INSTANCE)",
            "M1782_FAIL_EXACT_NET_ANNOTATION_GATE",
            "M1782_FAIL_EXACT_LEAF_ANNOTATION_GATE",
            "PASS_M1782_C1_M1701_EXPECTED_MACRO_LEAF_BLACKBOX_PTPX_TOOL_COMPLETE"):
        need(token in pt, "M1782 PT Tcl omits " + token)
    need('sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0' not in active_pt,
         "M1782 retains predecessor broad black-box rejection")
    need("remove_from_collection $black_boxes" not in active_pt
         and "remove_from_collection $macro_cells" not in active_pt,
         "M1782 silently subtracts a black-box/macro collection")
    need("report_power $macro_cells" not in active_pt
         and "ptpx_nine_parent_macros" not in active_pt,
         "M1782 changes whole-component accounting")
    for index, name in enumerate(EXPECTED_MACRO_NAMES):
        need(name == "u_parent_scratch/g_slice_%d__u_parent_sram" % index,
             "expected macro name construction")

    runner = RUNNER.read_text(errors="strict")
    for token in ("COUNTS = {\"vcs_compiles\": 1, \"simv_runs\": 1,",
                  "PT_TCL = CHECK.PT_TCL",
                  "validate_black_box_inventory(",
                  "PASS_M1782_C1_EXPECTED_MACRO_LEAF_BLACKBOX_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER"):
        need(token in runner, "M1782 runner omits " + token)
    for forbidden in ("m1772_c1_two_bank_public_warmup_energy_r1_20260902.private_build",
                      "m1772_c1_directed_component.saif\"),",
                      "+notimingcheck", "+no_notifier", "+nospecify",
                      "+initreg", "ignore" + "_black_box"):
        need(forbidden not in runner, "M1782 runner reuses/bypasses " + forbidden)
    need(runner.count('"+define+UNIT_DELAY"') == 1,
         "M1782 fresh compile must have exactly one UNIT_DELAY define")

    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1782_m1772_c1_expected_macro_leaf_blackbox_energy_source_contract_r1_v1",
         "M1782 contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1772_FAILURE_BOUND__EXACT_9_LINKED_SRAM_LEAF_BLACKBOX_ALLOWLIST__FRESH_1VCS_1SIM_1SAIF_1PTPX__M1783_REVIEW_AND_M1784_RELEASE_REQUIRED__NO_EDA",
         "M1782 contract status")
    need(contract.get("claim_boundary") == CLAIMS,
         "M1782 source contract promotes claims")
    gate = contract.get("black_box_gate", {})
    need(gate.get("expected_count") == 9
         and gate.get("expected_ref") == MACRO_REF
         and gate.get("expected_names") == list(EXPECTED_MACRO_NAMES)
         and gate.get("is_hierarchical") is False
         and gate.get("is_black_box") is True
         and gate.get("missing_fails") is True
         and gate.get("extra_fails") is True
         and gate.get("wrong_ref_fails") is True
         and gate.get("unresolved_nonmacro_allowed") is False,
         "M1782 exact macro black-box gate contract drift")
    budget = contract.get("fresh_execution_budget", {})
    need(budget == {"ptpx_runs": 1, "saif_files": 1, "simv_runs": 1,
                    "vcs_compiles": 1, "reuse_m1772_private_build": False},
         "M1782 fresh execution budget drift")
    rows = contract.get("source_files")
    need(isinstance(rows, list), "M1782 source inventory")
    mapping = dict((row.get("path"), row.get("sha256")) for row in rows)
    expected = (TB, FILELIST, UCLI, PT_TCL, RUNNER, CHECKER, TEST)
    need(len(mapping) == len(rows) and set(mapping) == set(
        path.relative_to(HW).as_posix() for path in expected),
        "M1782 source inventory paths")
    for path in expected:
        need(mapping[path.relative_to(HW).as_posix()] == sha(path),
             "M1782 source inventory SHA " + str(path))
    return {"schema": "m1782_c1_energy_source_check_r1_v1",
            "status": "PASS_M1782_EXACT_EXPECTED_MACRO_LEAF_BLACKBOX_SOURCE_ONLY_NO_EDA",
            "m1772_failure": failure,
            "black_box_policy": "exact_9_expected_linked_sram_liberty_leaves_only",
            "fresh_execution_budget": budget,
            "claim_boundary": CLAIMS}


INVENTORY_ROW = re.compile(
    r"^name=(\S+) ref=(\S+) is_hierarchical=(\S+) is_black_box=(\S+)$")


def validate_black_box_inventory(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "black-box inventory absent")
    rows = path.read_text(errors="strict").splitlines()
    need(rows[:2] == ["black_box_count=9", "expected_macro_count=9"],
         "black-box inventory header")
    observed = {}
    for row in rows[2:]:
        hit = INVENTORY_ROW.fullmatch(row)
        need(hit is not None, "black-box inventory row syntax")
        name, ref_name, is_hier, is_bb = hit.groups()
        need(name not in observed, "duplicate black-box inventory name")
        need(ref_name == MACRO_REF, "wrong black-box ref " + name)
        need(is_hier == "false", "black box is hierarchical " + name)
        need(is_bb == "true", "black-box attribute false " + name)
        observed[name] = ref_name
    need(set(observed) == set(EXPECTED_MACRO_NAMES)
         and len(observed) == 9,
         "black-box inventory set mismatch")
    return {"status": "PASS_M1782_EXACT_9_EXPECTED_LINKED_SRAM_LEAF_BLACKBOX_INVENTORY",
            "count": 9, "ref_name": MACRO_REF,
            "names": list(EXPECTED_MACRO_NAMES),
            "unexpected_black_boxes": 0, "missing_expected_macros": 0}


def validate_runtime(path):
    return OLD.validate_runtime(path)


def validate_saif(path, cycles, expected_activity_forms=117690):
    return OLD.validate_saif(path, cycles, expected_activity_forms)


def whole_component_power(path, cycles, reads, writes):
    return OLD.whole_component_power(path, cycles, reads, writes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "runtime", "saif",
                                           "blackbox", "power"), required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--top-power", type=Path)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--reads", type=int)
    parser.add_argument("--writes", type=int)
    args = parser.parse_args()
    if args.mode == "source":
        value = validate_sources()
    elif args.mode == "runtime":
        need(args.log is not None, "runtime log absent")
        value = validate_runtime(args.log)
    elif args.mode == "saif":
        need(args.log is not None and args.cycles is not None,
             "SAIF path/cycles absent")
        value = validate_saif(args.log, args.cycles)
    elif args.mode == "blackbox":
        need(args.log is not None, "black-box inventory absent")
        value = validate_black_box_inventory(args.log)
    else:
        need(None not in (args.top_power, args.cycles, args.reads, args.writes),
             "power inputs absent")
        value = whole_component_power(args.top_power, args.cycles,
                                      args.reads, args.writes)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
