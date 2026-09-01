#!/usr/bin/env python3
"""Independent, zero-EDA M1773 hammer for the M1772 C1 energy source.

This checker derives its expectations from the frozen protocol and claim
boundary, not from the M1772 author's recorded PASS output.  It may execute
Python source tests and construct temporary SAIF fixtures, but it must never
query a license or launch VCS, simv, PrimeTime, DC, or Formality.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
CONTRACT = HW / "contracts/m1772_m1701_c1_two_bank_public_warmup_energy_source_contract_r1_20260902.json"
TB = HW / "dc_handoff/tb/tb_m1772_c1_m1701_two_bank_public_warmup_energy.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1772_c1_m1701_two_bank_public_warmup_energy.f"
UCLI = HW / "dc_handoff/scripts/m1772_c1_m1701_two_bank_public_warmup_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1750_c1_m1701_public_port_mapped_whole_component_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1772_m1701_c1_two_bank_public_warmup_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1772_c1_m1701_two_bank_public_warmup_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1772_c1_m1701_two_bank_public_warmup_energy_source.py"
RTL = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
NET = HW / ("dc_handoff/runs/m1701_m1695_c1_tool_entity_repair_dc_r1_20260901."
            "failed_or_incomplete.2502881.quarantine/netlist/"
            "m935_m912_three_stage_exact_parent_match_product_capture_island_"
            "m1695_fastmin_hold_closed_mapped.v")
AUTHOR = HW / "reviews/m1772_m1701_c1_two_bank_public_warmup_energy_source_author_receipt_r1_20260902"
ATTEMPT = HW / "results/.m1772_c1_two_bank_public_warmup_energy_attempt_consumed"
RESULT = HW / "results/m1772_c1_two_bank_public_warmup_energy_r1_20260902"
FAILURE = HW / "results/m1772_c1_two_bank_public_warmup_energy_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1772_c1_two_bank_public_warmup_energy_r1_20260902.private_build.unsealed_do_not_cite"
RELEASE = HW / "contracts/m1774_m1773_m1772_m1701_c1_two_bank_public_warmup_energy_launch_release_r1_20260902.json"

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    CONTRACT: "b38ecd2839cf26ab29fa4280816f18d93e0646d6a87fd4aefc8e23a0cd6d0f48",
    TB: "21ead36213d89a425a170fce85823994562e8410c9bd24b338b7cf29f02a750d",
    FILELIST: "9da54a6a3b60a05602adbb0bb4440d0ac95c035c73a1b69d6589dab2f8664906",
    UCLI: "beaa724867c28198d600840b2b8fe7dcbe665ad7cf6ee9449c92be6ccafccef7",
    PT_TCL: "1b9fdb335290e2e7dc14b3cdc1a0cbf3dbe63ed0ca691226762b037726a184c6",
    RUNNER: "ca4c10be47ffc8d95869714ca8905a554ea7e2239d61466395b82e52690def11",
    CHECKER: "c847e65d09e5bdc1c7c9512d5d7dae42773af1386c51aa3c545034c540cdc486",
    TEST: "0a36226a1d842a981d0a6bfb227fd8c5fc61ac6389f28bd71a090f36cfe7a668",
    RTL: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    NET: "d990bb416370fd07a1c241849e2fa494b94a179b47687a1a3ff2b1ab92c255e8",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "manifest absent")
    need(outer.is_file() and not outer.is_symlink(), "outer seal absent")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal mismatch")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest member")
        member = root / rel
        need(member.is_file() and not member.is_symlink()
             and sha(member) == fields[0], "manifest member mismatch " + name)
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed population mismatch")


def active_sv(text):
    """Strip comments while retaining strings; sufficient for drive audit."""
    out = []
    index = 0
    state = "code"
    while index < len(text):
        pair = text[index:index + 2]
        char = text[index]
        if state == "code" and pair == "//":
            state = "line"
            index += 2
        elif state == "code" and pair == "/*":
            state = "block"
            index += 2
        elif state == "line" and char == "\n":
            out.append(char)
            state = "code"
            index += 1
        elif state == "block" and pair == "*/":
            state = "code"
            index += 2
        elif state == "code" and char == '"':
            out.append(char)
            state = "string"
            index += 1
        elif state == "string":
            out.append(char)
            if char == "\\" and index + 1 < len(text):
                index += 1
                out.append(text[index])
            elif char == '"':
                state = "code"
            index += 1
        elif state == "code":
            out.append(char)
            index += 1
        else:
            index += 1
    need(state in {"code", "line"}, "unterminated SV comment/string")
    return "".join(out)


def run_python(interpreter):
    completed = subprocess.run(
        [interpreter, str(TEST), "-v"], cwd=str(HW.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=120, check=False)
    need(completed.returncode == 0 and "Ran 9 tests" in completed.stdout
         and completed.stdout.rstrip().endswith("OK"),
         "author tests fail under " + interpreter + "\n" + completed.stdout)
    return {"interpreter": interpreter,
            "version": subprocess.check_output(
                [interpreter, "--version"], stderr=subprocess.STDOUT,
                universal_newlines=True).strip(),
            "tests_run": 9, "failures": 0, "errors": 0,
            "output_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")).hexdigest()}


def load_checker():
    spec = importlib.util.spec_from_file_location("m1772_subject", str(CHECKER))
    need(spec is not None and spec.loader is not None, "checker import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_reject(action, label):
    try:
        action()
    except (RuntimeError, ValueError):
        return
    raise RuntimeError("mutation accepted " + label)


def saif_mutation_hammer(subject):
    template = """/** outside **/
(SAIFILE
 (VENDOR \"literal /* string */\")
 (DURATION {duration})
 (INSTANCE tb_m1772_c1_m1701_two_bank_public_warmup_energy
  (INSTANCE dut
   (NET (n0 (T0 {t0}) (T1 3) (TX {tx}) (TC 2) (IG 0)))
   (INSTANCE u_parent_scratch))))
"""
    rejected = 0
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "fixture.saif"
        path.write_text(template.format(duration=6, t0=3, tx=0))
        result = subject.validate_saif(path, 2, expected_activity_forms=1)
        need(result["duration_ns"] == 6.0 and result["tx_nonzero"] == 0
             and result["block_comments_skipped_outside_strings"] == 1,
             "valid SAIF fixture rejected")
        mutations = [
            (template.format(duration=6, t0=2, tx=1), "nonzero TX"),
            (template.format(duration=3, t0=0, tx=0), "duration"),
            (template.format(duration=6, t0=2, tx=0), "conservation"),
            ("/* unterminated", "unterminated comment"),
            ('(SAIFILE (VENDOR "unterminated))', "unterminated string"),
            ('(SAIFILE) trailing', "trailing token"),
            ('prefix (SAIFILE)', "leading token"),
            ('(SAIFILE (DURATION 6)', "unterminated list"),
        ]
        for payload, label in mutations:
            path.write_text(payload)
            expect_reject(lambda p=path: subject.validate_saif(
                p, 2, expected_activity_forms=1), label)
            rejected += 1
    return {"valid_fixtures": 1, "mutations_rejected": rejected,
            "comment_outside_string_skipped": True,
            "comment_delimiters_inside_string_preserved": True}


def full_cardinality_saif_hammer(subject):
    """Exercise the production cardinality rather than trusting a token."""
    expected = 117690
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "full_cardinality.saif"
        with path.open("w") as handle:
            handle.write("/** independent full-cardinality fixture **/\n")
            handle.write("(SAIFILE (DURATION 3) (INSTANCE ")
            handle.write("tb_m1772_c1_m1701_two_bank_public_warmup_energy ")
            handle.write("(INSTANCE dut\n")
            for index in range(expected):
                handle.write("(NET (n%d (T0 2) (T1 1) (TX 0) "
                             "(TC %d) (IG 0)))\n" %
                             (index, 1 if index == 0 else 0))
            handle.write("(INSTANCE u_parent_scratch))))\n")
        value = subject.validate_saif(path, 1)
    need(value.get("activity_forms_per_tag") == expected
         and value.get("tx_nonzero") == 0
         and value.get("duration_ns") == 3.0,
         "full-cardinality SAIF gate")
    return {"forms_per_tag": expected, "required_tags": 5,
            "total_activity_forms": expected * 5, "tx_nonzero": 0,
            "duration_ns": 3.0, "measurement_cycles": 1}


def main():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "identity drift " + str(path))
    verify_seal(AUTHOR)

    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1772_m1701_c1_two_bank_public_warmup_energy_source_contract_r1_v1",
         "contract schema")
    warmup = contract.get("two_bank_public_warmup", {})
    need(warmup.get("epochs") == [5943, 5944, 5945]
         and warmup.get("same_64_masks_for_all_three_tasks") is True
         and warmup.get("warmup_inside_saif") is False
         and warmup.get("public_sink_backpressure_only") is True
         and warmup.get("hierarchical_drive") is False
         and warmup.get("hierarchical_state_read") is False
         and warmup.get("per_task_counters_clear_at_execution_start") is True,
         "contract warmup geometry")
    need(all(value is False for value in contract.get("claim_boundary", {}).values()),
         "source claim promotion")
    need(contract.get("author_execution") == {
        "vcs_runs": 0, "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0,
        "license_queries": 0, "eda_attempts_created": 0,
        "results_created": 0, "release_created": False},
        "author execution boundary")

    tb = TB.read_text()
    active = active_sv(tb).lower()
    for forbidden in ("force ", "release ", "dut.", "$deposit", "$readmem"):
        need(forbidden not in active, "hierarchical/init bypass " + forbidden)
    ordered = (
        "load_public_task(WARMUP0_EPOCH, 1'b0)", "wait (execute_busy)",
        "psum_write_ready = 1'b0", "row_complete_ready = 1'b0",
        "load_public_task(WARMUP1_EPOCH, 1'b0)",
        "psum_write_ready = 1'b1", "row_complete_ready = 1'b1",
        "load_public_task(TEST_EPOCH, 1'b1)")
    cursor = 0
    for token in ordered:
        position = tb.find(token, cursor)
        need(position >= 0, "warmup order " + token)
        cursor = position + len(token)
    for token in ("WARMUP0_EPOCH = 16'd5943",
                  "WARMUP1_EPOCH = 16'd5944", "TEST_EPOCH = 16'd5945",
                  "if (measurement_open)",
                  "count_psum_commits != 64", "count_row_completions != 64",
                  "committed_rows !== 64'hffff_ffff_ffff_ffff",
                  "count_macro_reads + count_forwards"):
        need(token in tb, "TB invariant absent " + token)
    need(tb.count("if ($test$plusargs(\"M1772_UCLI_SAIF\")) $stop;") == 2,
         "UCLI window stop count")

    rtl = RTL.read_text()
    exec_start = rtl.index("if (!exec_active_q && ready_bank_valid_w)")
    exec_body = rtl[exec_start:rtl.index("if (exec_active_q)", exec_start)]
    for counter in ("count_issue_accepts", "count_parent_edges",
                    "count_dead_write_elisions", "count_macro_reads",
                    "count_macro_writes", "count_forwards",
                    "count_psum_commits", "count_row_completions"):
        need(counter + " <= '0;" in exec_body,
             "counter not cleared at task execution start " + counter)
    need(rtl.count("task_done_valid <= 1'b1;") == 1,
         "task_done producer ambiguity")

    filelist = [row.strip() for row in FILELIST.read_text().splitlines()
                if row.strip() and not row.lstrip().startswith("#")]
    need(len(filelist) == 4 and filelist[-2] == str(NET)
         and filelist[-1] == str(TB), "fresh filelist/order")
    need(len(re.findall(r"\bTS1N28HPCPHVTB128X128M4S\s+", NET.read_text())) == 9,
         "mapped macro count")
    need(UCLI.read_text().splitlines()[-8:] == [
        "power -gate_level all mda sv",
        "power tb_m1772_c1_m1701_two_bank_public_warmup_energy.dut",
        "run", "power -enable", "run", "power -disable",
        "power -report $::env(M1772_SAIF_FILE) 1e-9 tb_m1772_c1_m1701_two_bank_public_warmup_energy.dut",
        "quit"], "UCLI measured window/scope")

    runner = RUNNER.read_text()
    need(runner.count('"+define+UNIT_DELAY"') == 1,
         "UNIT_DELAY compile cardinality")
    for key in ("vcs_compiles", "simv_runs", "saif_files", "ptpx_runs"):
        need(runner.count('state["' + key + '"] += 1') == 1,
             "one-shot counter cardinality " + key)
    for forbidden in ("+notimingcheck", "+no_notifier", "+nospecify",
                      "+initreg", "+define+no_warning",
                      "+define+NO_INPUT_FLOATING_CHECK", "force ",
                      "ignore_tx", "ignoreTX"):
        need(forbidden not in runner, "runner bypass " + forbidden)
    for token in ("verify_authority()", "CHECK.validate_sources()",
                  "namespaces_fresh()", "ATTEMPT.mkdir()",
                  "automatic_retry\": False", "publish_no_replace",
                  "verify_seal(M1773", "exact(M1774",
                  "CHECK.validate_saif(", "CHECK.whole_component_power("):
        need(token in runner, "runner fail-closed invariant " + token)
    need(runner.index("verify_authority()") < runner.index("ATTEMPT.mkdir()")
         and runner.index("CHECK.validate_sources()") < runner.index("ATTEMPT.mkdir()"),
         "attempt before authorities/source check")
    need(not any(os.path.lexists(path) for path in
                 (ATTEMPT, RESULT, FAILURE, PRIVATE, RELEASE)),
         "fresh namespace/release violated during source review")

    pt = PT_TCL.read_text()
    for token in ("set expected_macro_count 9",
                  "M1750_FAIL_EXACT_NET_ANNOTATION_GATE",
                  "M1750_FAIL_EXACT_LEAF_ANNOTATION_GATE",
                  "$annotated_nets != $total_nets",
                  "$annotated_leaf_cells != $total_leaf_cells",
                  "$annotated_percent != 100.0",
                  "$annotated_leaf_percent != 100.0",
                  "ptpx_whole_mapped_c1_including_9macro_liberty.rpt",
                  "top_minus_macro=false",
                  "ptpx_plus_datasheet_sram_combined=false"):
        need(token in pt, "PTPX hard gate absent " + token)

    subject = load_checker()
    saif_hammer = saif_mutation_hammer(subject)
    full_saif_hammer = full_cardinality_saif_hammer(subject)
    source_result = subject.validate_sources()
    need(source_result.get("status") ==
         "PASS_M1772_TWO_BANK_PUBLIC_WARMUP_SOURCE_ONLY_NO_EDA",
         "subject source self-check")
    interpreters = {
        "cpython36": run_python("/usr/bin/python3.6"),
        "cpython310": run_python(
            "/opt/anaconda3/envs/pytorch310/bin/python3.10")}

    value = {
        "schema": "m1773_independent_hammer_output_r1_v1",
        "status": "PASS_M1773_ZERO_EDA_SOURCE_HAMMER",
        "identity_count": len(FIXED),
        "two_bank_public_warmup": True,
        "warmup_epochs": [5943, 5944],
        "measurement_epoch": 5945,
        "measurement_task_counter_clear": True,
        "public_port_only": True,
        "force_init_ignore_tx": False,
        "unit_delay_define_count": 1,
        "mapped_macro_count": 9,
        "saif_expected_forms_per_tag": 117690,
        "saif_all_tx_zero_gate": True,
        "saif_duration_rule": "measurement_cycles*3ns",
        "saif_mutation_hammer": saif_hammer,
        "saif_full_cardinality_hammer": full_saif_hammer,
        "ptpx_net_annotation_gate_percent": 100.0,
        "ptpx_leaf_annotation_gate_percent": 100.0,
        "interpreters": interpreters,
        "fresh_namespaces": True,
        "eda_or_license_actions": 0,
        "docs359_sha256": sha(DOC359)}
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
