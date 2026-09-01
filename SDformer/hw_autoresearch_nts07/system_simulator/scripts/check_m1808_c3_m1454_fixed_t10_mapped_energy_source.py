#!/usr/bin/env python3
"""Fail-closed source/runtime checker for additive M1808; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1798_CHECKER = HW / "system_simulator/scripts/check_m1798_c3_m1454_fixed_t10_mapped_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1798_checker_for_m1808",
                                              str(M1798_CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1798 predecessor checker unavailable")
M1798 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1798)
M1790 = M1798.BASE

DOC359 = M1798.DOC359
NET = M1798.NET
SDC = M1798.SDC
CELL_V = M1798.CELL_V
TT_DB = M1798.TT_DB
M1798_CONTRACT = M1798.CONTRACT
M1798_ATTEMPT = HW / "results/.m1798_c3_mapped_energy_attempt_consumed"
M1798_FAILURE = HW / "results/m1798_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
M1807 = HW / "reviews/m1807_m1798_c3_unique_production_failure_m1806_xz_diagnostic_hammer_r1_20260902"
M1456 = HW / "dc_handoff/runs/m1456_m1454_c3_hold_repair_prelayout_ptsta_r1_20260831"

TB = HW / "dc_handoff/tb/tb_m1808_c3_m1454_fixed_t10_mapped_energy_reset_settling.sv"
TB_TAG = HW / "dc_handoff/tb/tb_m1808_c3_m1454_fixed_t10_mapped_energy_tag_scoreboard.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1808_c3_m1454_fixed_t10_mapped_energy.f"
UCLI = HW / "dc_handoff/scripts/m1808_c3_m1454_fixed_t10_mapped_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1808_c3_m1454_fixed_t10_mapped_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1808_c3_m1454_fixed_t10_mapped_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1808_c3_m1454_fixed_t10_mapped_energy_source.py"
CONTRACT = HW / "contracts/m1808_m1807_c3_m1454_fixed_t10_mapped_energy_reset_settling_source_contract_r1_20260902.json"

TOP = "tb_m1808_c3_m1454_fixed_t10_mapped_energy"
SAIF_SCOPE = TOP + ".dut"
CLAIMS = dict(M1798.CLAIMS)

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    NET: "7c01af42322b8feed904df2862aac6e21cbe165b988f1b248f2e94d23f23a7a7",
    SDC: "bb3697e833cb987e4a85ab2a62b4f40946a8c3d6b7eaba08504570f5a862f23f",
    CELL_V: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    TT_DB: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
    M1798_CHECKER: "6c7866522055e0dd4324225ac25741c1f20be269b1c6a182a2c26ac8f2dcc535",
    M1798_CONTRACT: "f53e60760219c3f1c623e2666a863172dd3964e2b36768a23292ff2275f5dac3",
    Path(str(M1798_CONTRACT) + ".sha256"): "cd64a2811d626841502273cd92b53d605552af7f395b3cddc3ee395f329a1b8d",
    Path(str(M1798_CONTRACT) + ".sha256.seal.sha256"): "03b8278d5cf6e6135b9113354b1d7146d0cee52ababda46c904b51c22932db17",
    M1798_ATTEMPT / "attempt.json": "5dcef0780709dd52f95c571bd8302da70206a76ee82fe4c7e934ce62281aee24",
    M1798_ATTEMPT / "SHA256SUMS": "360100f3431dc29b29e22f545b8ad5bce9372b1632a42b67224c9e963ac1e8bd",
    M1798_ATTEMPT / "SHA256SUMS.seal.sha256": "a1d6cb83b90d82a1eda38d823eea04659989a54247ea3cb07b173b100793c356",
    M1798_FAILURE / "failure.json": "aea36bcd319f89be78ba7a6b26f0ec02acf17f095601ad227ac194759f063427",
    M1798_FAILURE / "SHA256SUMS": "ff0eaefd6ac92f539cbcd5d01f05592e73da568731f56584c17e06dc358ad2e2",
    M1798_FAILURE / "SHA256SUMS.seal.sha256": "bda296e738e3a6e8ad8791217c9ad3ed2706e53221796eab87fbd98086312b1a",
    M1807 / "review.json": "787c25bf07acdc9965d81933de23360ab70e9f33f867e4dfb03461bf59b9c75e",
    M1807 / "SHA256SUMS": "10c21946f092a895eb5a508bcd657bb7e7027ce9bd8f1ee3d1eb076f16f82c67",
    M1807 / "SHA256SUMS.seal.sha256": "0c80eaa4aa9b629fb8f5ddc5c9825bcd3685c88b360d421438f37782de26d46e",
    M1456 / "SHA256SUMS": "35d0ae3802dd98e25b78b1927dd1e865bf11b51f8b84544ebcb01475e8eb4f6c",
}

RELEASE_BINDING_TOKENS = (
    "M1808_EXPECTED_RUNNER_SHA256",
    "M1808_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1808_EXPECTED_M1815_MANIFEST_SHA256",
    "M1808_EXPECTED_M1815_OUTER_FILE_SHA256",
    "M1808_EXPECTED_M1815_REVIEW_SHA256",
    "M1808_EXPECTED_M1816_RELEASE_SHA256",
    "M1808_EXPECTED_M1816_SIDECAR_SHA256",
    "M1808_EXPECTED_M1816_OUTER_FILE_SHA256",
    "m1816_m1815_m1808_c3_m1454_fixed_t10_mapped_energy_launch_release_r1_v1",
    "AUTHORIZE_ONE_FRESH_M1808_C3_MAPPED_ENERGY_CAMPAIGN",
    "PASS_M1815_M1808_C3_MAPPED_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_M1808_CAMPAIGN",
    "m1798_attempt_json_sha256",
    "m1798_failure_json_sha256",
    "m1807_review_json_sha256",
    "m1456_ptsta_manifest_sha256",
    "release.get(\"identity\") != expected_identity",
    "release.get(\"prelaunch_claim_boundary\") != CHECK.CLAIMS",
    "release.get(\"measurement_boundary\") != RELEASE_BOUNDARY",
    "release.get(\"attempt_uniqueness\") != ATTEMPT_UNIQUENESS",
    "verify_review_member(M1815)",
    "verify_file_double_seal(M1816",
)


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
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        need((root / rel).is_file() and not (root / rel).is_symlink()
             and sha(root / rel) == fields[0], "manifest drift " + name)
        listed.add(name)


def active_lines(text):
    return [raw.split("#", 1)[0].strip() for raw in text.splitlines()
            if raw.split("#", 1)[0].strip()]


def strip_sv_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def validate_semantics(texts):
    tb = texts[TB]
    tag = texts[TB_TAG]
    active_sv = (strip_sv_comments(tb) + "\n" + strip_sv_comments(tag)).lower()
    runner = texts[RUNNER]
    active_all = active_sv + "\n" + runner.lower()
    for forbidden in ("force ", "$root", "dut.", "+notimingcheck",
                      "+no_notifier", "+nospecify", "+initreg", "deposit(",
                      "vpi_handle_by_name", "ignorex", "ignore_x"):
        need(forbidden not in active_all, "forbidden bypass " + forbidden)

    for token in (
            "module tb_m1808_c3_m1454_fixed_t10_mapped_energy",
            "post_reset_settle_cycles",
            "full_public_check_enabled",
            "post_reset_settle_cycles = post_reset_settle_cycles + 1",
            "post_reset_settle_cycles == 3",
            "post_reset_settle_cycles > 3",
            "M1808 architectural/control output contains X/Z",
            "M1808 activity during reset-settling",
            "M1808 debug counter X/Z at settling boundary",
            "M1808 debug counter nonzero at settling boundary",
            "M1808 full public output contains X/Z",
            "M1808_RESET_SETTLING_GATE cycles=3 debug=11 binary=1 zero=1",
            "repeat (8) @(posedge clk_core)",
            "repeat (3) @(posedge clk_core)",
            "if (!full_public_check_enabled || post_reset_settle_cycles != 3)",
            "one full tile warm the mapped state outside SAIF",
            "for (tile = 0; tile < MEASURE_TILES; tile = tile + 1)",
            "debug_stage1_issues-base_issues != 17*MEASURE_TILES",
            "debug_product_pushes-base_pushes != 5*MEASURE_TILES",
            "result_stall_cycles == 0 || raw_stall_cycles == 0",
            "context_retire_cycles != expected_retire",
            "PASS_M1808_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY"):
        need(token in tb, "TB omits " + token)
    need(tb.count("debug_config_beats, debug_raw_beats") >= 2,
         "debug aggregate not restored")
    need(tb.count("$isunknown({config_ready, config_accept") >= 2,
         "architectural immediate/full gates absent")
    need(tb.count("debug_context_cycles} != 0") == 1,
         "debug zero boundary absent/duplicated")

    for token in (
            "EXPECTED_TOTAL_TAGS = 9", "EXPECTED_MEASURED_TAGS = 8",
            "sampled_raw_tag !== directed_tag(expected_write)",
            "sampled_tile_done_tag !==",
            "expected_tile_done_tag[expected_read]",
            "expected_read >= expected_write",
            "expected_write != EXPECTED_TOTAL_TAGS",
            "raw_stall_cycles == 0", "result_stall_cycles == 0",
            "PASS_M1808_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD",
            "bind tb_m1808_c3_m1454_fixed_t10_mapped_energy"):
        need(token in tag, "tag scoreboard omits " + token)

    expected_filelist = [str(CELL_V), str(NET), str(TB), str(TB_TAG)]
    need(active_lines(texts[FILELIST]) == expected_filelist,
         "filelist/order drift")
    need(active_lines(texts[UCLI]) == [
        "power -gate_level all mda sv", "power " + SAIF_SCOPE, "run",
        "power -enable", "run", "power -disable",
        "power -report $::env(M1808_SAIF_FILE) 1e-9 " + SAIF_SCOPE,
        "quit"], "UCLI scope/order drift")

    pt = texts[PT_TCL]
    for token in ("M1808_TT_LIB_DB", "M1808_MAPPED_NETLIST",
                  "M1808_MAPPED_SDC", "M1808_GATE_SAIF",
                  "M1808_OUTPUT_DIR", "M1808_SAIF_INSTANCE",
                  "M1808_MEASUREMENT_CYCLES", "M1808_SAIF_DURATION_NS",
                  str(M1798.M1790_PT_TCL)):
        need(token in pt, "PTPX wrapper omits " + token)

    for token in (
            "results/.m1808_c3_mapped_energy_attempt_consumed",
            "date_dual_synopsys_same_uid_eda_queue.lock",
            "collision_gate()", "automatic_retry\": False",
            "reuse_prior_simv_saif_ptpx\": False",
            "+M1808_UCLI_SAIF", "CHECK.validate_runtime(sim_log)",
            "CHECK.validate_saif(saif", "CHECK.component_power(",
            "vcs_compiles\": 1", "simv_runs\": 1", "saif_files\": 1",
            "ptpx_runs\": 1", "publish_no_replace(STAGE, RESULT)",
            "\"mode\": \"zero_delay_mapped_functional\""):
        need(token in runner, "runner omits " + token)
    need("+define+UNIT_DELAY" not in runner, "misleading UNIT_DELAY label retained")
    for token in RELEASE_BINDING_TOKENS:
        need(token in runner, "release binding omits " + token)
    need(runner.count("state[\"vcs_compiles\"] += 1") == 1
         and runner.count("state[\"simv_runs\"] += 1") == 1
         and runner.count("state[\"saif_files\"] += 1") == 1
         and runner.count("state[\"ptpx_runs\"] += 1") == 1,
         "runner execution budget drift")


def validate_sources():
    M1798.validate_sources()
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    for root in (M1798_ATTEMPT, M1798_FAILURE, M1807, M1456):
        verify_seal(root)

    failure = strict_json(M1798_FAILURE / "failure.json")
    need(failure.get("status") == "FAILED_OR_INCOMPLETE"
         and failure.get("phase") == "MAPPED_SIM_SAIF"
         and failure.get("attempt_consumed") is True
         and failure.get("counts") == {
             "vcs_compiles": 1, "simv_runs": 1,
             "saif_files": 0, "ptpx_runs": 0}
         and failure.get("automatic_retry") is False
         and failure.get("canonical_result") is False,
         "M1798 unique failure drift")
    review = strict_json(M1807 / "review.json")
    need(review.get("status") ==
         "FAIL_CLOSED_M1807_M1798_UNIQUE_PRODUCTION_FAILURE_AND_M1806_READ_ONLY_DIAGNOSTIC__P0_0_P1_1__ADDITIVE_RESET_SETTLING_TB_SUCCESSOR_REQUIRED__NO_EDA",
         "M1807 status")
    need(review.get("repair_decision", {}).get("preferred") ==
         "A__BOUNDED_LEGAL_RESET_RELEASE_SETTLING_TB",
         "M1807 repair decision")

    source_paths = (TB, TB_TAG, FILELIST, UCLI, PT_TCL, RUNNER, CHECKER, TEST)
    for path in source_paths:
        need(path.is_file() and not path.is_symlink(), "source absent " + str(path))
    texts = dict((path, path.read_text()) for path in source_paths)
    validate_semantics(texts)

    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1808_m1807_c3_m1454_fixed_t10_mapped_energy_reset_settling_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1807_P1_REPAIRED__M1815_REVIEW_AND_M1816_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "source claim promotion")
    need(contract.get("execution_budget") == dict(
        vcs_compiles=1, simv_runs=1, saif_files=1, ptpx_runs=1,
        automatic_retry=False, reuse_prior_simv_saif_ptpx=False),
        "contract budget")
    need(contract.get("launch_governance", {}).get("exact_release_required") == "M1816"
         and contract.get("launch_governance", {}).get(
             "different_author_source_hammer_required") == "M1815"
         and contract.get("launch_governance", {}).get("double_seal") is True,
         "contract governance")
    mapping = dict((row.get("path"), row.get("sha256"))
                   for row in contract.get("source_files", []))
    need(len(mapping) == len(source_paths), "source inventory cardinality")
    for path in source_paths:
        need(mapping.get(str(path.relative_to(HW))) == sha(path),
             "source inventory drift " + str(path))
    return {"status": "PASS_M1808_RESET_SETTLING_SOURCE_STATIC",
            "source_files": len(source_paths),
            "predecessor_m1798_failure": "BOUND",
            "m1807_repair": "BOUND", "checks": 1}


def validate_runtime(path):
    text = Path(path).read_text(errors="strict")
    need(text.count(
        "PASS_M1808_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY") == 1,
        "runtime PASS count")
    need(text.count("PASS_M1808_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD") == 1,
         "tag PASS count")
    need(text.count(
        "M1808_RESET_SETTLING_GATE cycles=3 debug=11 binary=1 zero=1") == 1,
        "reset-settling gate")
    need("Error-" not in text and "$fatal" not in text
         and "Assertion failed" not in text, "runtime failure signature")
    window = re.findall(r"M1808_SAIF_WINDOW_STOP cycles=([0-9]+)", text)
    need(len(window) == 1 and int(window[0]) > 0, "runtime window")
    result = re.findall(
        r"M1808_PUBLIC_RESULT_CHECK tiles=([0-9]+) beats=([0-9]+) mismatches=([0-9]+) xz=([0-9]+)",
        text)
    need(result == [("8", "40", "0", "0")], "runtime result checker")
    counters = re.findall(
        r"M1808_PUBLIC_COUNTER_DELTAS raw_beats=([0-9]+) tiles=([0-9]+) issues=([0-9]+) done=([0-9]+) pushes=([0-9]+) departures=([0-9]+)",
        text)
    need(counters == [("40", "8", "136", "8", "40", "40")],
         "runtime conservation")
    cover = re.findall(
        r"M1808_PUBLIC_COVERAGE result_stall_cycles=([0-9]+) raw_stall_cycles=([0-9]+) retire_cycles=([0-9]+)",
        text)
    need(len(cover) == 1 and all(int(value) > 0 for value in cover[0]),
         "runtime cover")
    tags = re.findall(
        r"M1808_TILE_DONE_TAG_CHECK total=([0-9]+) warmup=([0-9]+) measured=([0-9]+) mismatches=([0-9]+) raw_stall=([0-9]+) result_stall=([0-9]+)",
        text)
    need(len(tags) == 1 and tags[0][0:4] == ("9", "1", "8", "0")
         and int(tags[0][4]) > 0 and int(tags[0][5]) > 0,
         "ordered tag checker")
    return {"status": "PASS_M1808_RESET_SETTLING_PUBLIC_RUNTIME",
            "measurement_cycles": int(window[0]), "measured_tiles": 8,
            "result_beats": 40, "tile_done_tags_checked": 9,
            "reset_settling_cycles": 3, "debug_counters_checked": 11,
            "result_stall_cycles": int(cover[0][0]),
            "raw_stall_cycles": int(cover[0][1]),
            "retire_cycles": int(cover[0][2]),
            "numeric_mismatches": 0, "public_xz": 0}


def validate_saif(path, cycles):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
         "SAIF input")
    root, comment_count = M1790.parse_saif(path.read_text(errors="strict"))
    duration = M1790.forms(root, "DURATION")
    need(len(duration) == 1 and len(duration[0]) == 2, "SAIF duration")
    duration_ns = float(duration[0][1])
    need(math.isfinite(duration_ns) and abs(duration_ns-cycles*3.0) <= 1e-6,
         "SAIF duration/cycle mismatch")
    top = M1790.direct_instance(root, TOP)
    dut = M1790.direct_instance(top, "dut")
    groups = dict((tag, M1790.all_forms(dut, tag))
                  for tag in ("T0", "T1", "TX", "TC", "IG"))
    count = len(groups["T0"])
    need(count > 0 and all(len(value) == count for value in groups.values()),
         "SAIF form count")
    need(all(len(item) == 2 and float(item[1]) == 0.0
             for item in groups["TX"]), "SAIF contains TX")
    for t0, t1, tx in zip(groups["T0"], groups["T1"], groups["TX"]):
        need(all(len(item) == 2 for item in (t0, t1, tx))
             and abs(float(t0[1])+float(t1[1])+float(tx[1])-duration_ns) <= 1e-6,
             "SAIF activity conservation")
    need(any(float(item[1]) > 0 for item in groups["TC"] if len(item) == 2),
         "SAIF has no toggles")
    return {"status": "PASS_M1808_DUT_ONLY_SAIF", "cycles": cycles,
            "duration_ns": duration_ns, "activity_forms_per_tag": count,
            "tx_nonzero": 0, "saif_scope": SAIF_SCOPE,
            "block_comments_skipped": comment_count, "saif_sha256": sha(path)}


def component_power(path, cycles):
    result = dict(M1790.component_power(path, cycles))
    result["status"] = "PASS_M1808_COMPONENT_METRIC_PENDING_RESULT_HAMMER"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    need(args.static, "only --static is allowed")
    print(json.dumps(validate_sources(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
