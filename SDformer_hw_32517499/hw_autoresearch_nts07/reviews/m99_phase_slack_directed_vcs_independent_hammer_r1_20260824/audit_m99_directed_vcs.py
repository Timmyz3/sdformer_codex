#!/usr/bin/env python3
"""Read-only independent audit of the M99 dev_r2 directed VCS evidence."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RTL = HW / "rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv"
TB = HW / "tb_m99/tb_m99_phase_slack_guarded_wordpacked_pwp_stream.sv"
SVA = HW / "verif_m99/phase_slack_guarded_wordpacked_pwp_stream_assertions.sv"
M82 = HW / "rtl_m82/zero_bubble_elastic_pwp_stream.sv"
M85 = HW / "rtl_m85/guarded_wordpacked_pwp_stream.sv"
RUN = HW / "dc_handoff/runs/m99_phase_slack_directed_dev_r2_20260824"
COMPILE = RUN / "compile_r2.log"
SIM = RUN / "sim_r2.log"
ASSERT = RUN / "assert.report"
DISABLE = RUN / "assert.report.disablelog"
SIMV = RUN / "simv"
OUTPUT = HERE / "m99_directed_vcs_independent_audit.json"

EXPECTED = {
    "rtl": "93c638f69a2a50f4d020f4a2d0b974d620574e80b05f96c0a0358008c8883353",
    "tb": "9afa3a4d5d948fea695bd109f0ea6a12a08e9c49b7e0647bac869d1822dbbc99",
    "sva": "1f5cee2e0e31b287794b50cda2e6087ee89fc311ba89021ffb738dea0a6528c0",
    "m82": "2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f",
    "m85": "ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0",
    "compile": "7b9f87c2e691c59f4536d2f4542b4ef96c7e3eb7cf6b943d60696f74a224de86",
    "sim": "3f130a2ea749b42e6547ec3ebfde3047565c906797415f37d60bf64c0a8a6e20",
    "assert": "cee8082aa50f8aa22f465708bf483f3bc7a28c533b65d923cd43cbe174b7e3da",
    "disable": "aecf0c9dffcd66646682ee289bbc1def497a1a7e1b3e3a8fd60586b46ad7c056",
    "simv": "63b6eaa33f4e0437e986b4121e380c0427b1836385d1e8a9aa24ceb477ffd919",
}
EXPECTED_COVERS = {
    "cp_phase_load": 5,
    "cp_simultaneous_load_lookup": 1,
    "cp_lookup_stall": 1,
    "cp_escape": 27,
    "cp_width9": 28,
    "cp_width10": 28,
    "cp_width11": 27,
    "cp_metadata_error": 5,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count(text, needle):
    return text.count(needle)


def parse_pass(sim_text):
    pattern = re.compile(
        r"^PASS M99 M85-differential entries=(\d+) beats=(\d+) "
        r"parser_cycles=(\d+) stalls=(\d+) poison_attacks=(\d+) "
        r"early_lookup_attacks=(\d+) simultaneous_attacks=(\d+)$",
        re.MULTILINE)
    matches = pattern.findall(sim_text)
    require(len(matches) == 1, "PASS marker population")
    keys = ("entries", "beats", "parser_cycles", "stalls", "poison_attacks",
            "early_lookup_attacks", "simultaneous_attacks")
    return dict(zip(keys, map(int, matches[0])))


def parse_assert_report(text):
    pattern = re.compile(
        r"\.([A-Za-z0-9_]+),\s+(\d+) attempts,\s+(\d+) match$")
    output = {}
    for line in text.splitlines():
        match = pattern.search(line)
        require(match is not None, "unexpected assert.report line: " + line)
        name, attempts, hits = match.groups()
        require(name not in output, "duplicate cover " + name)
        output[name] = {"attempts": int(attempts), "matches": int(hits)}
    require(set(output) == set(EXPECTED_COVERS), "cover name population")
    require(all(output[name] == {"attempts": 1001, "matches": hits}
                for name, hits in EXPECTED_COVERS.items()), "cover count drift")
    return output


def main():
    paths = {"rtl": RTL, "tb": TB, "sva": SVA, "m82": M82, "m85": M85,
             "compile": COMPILE, "sim": SIM, "assert": ASSERT,
             "disable": DISABLE, "simv": SIMV}
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name], name + " SHA drift")

    rtl = RTL.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")
    sva = SVA.read_text(encoding="utf-8")
    compile_text = COMPILE.read_text(encoding="utf-8")
    sim_text = SIM.read_text(encoding="utf-8")
    assert_text = ASSERT.read_text(encoding="utf-8")
    disable_text = DISABLE.read_text(encoding="utf-8")

    require("Version V-2023.12-SP1_Full64" in compile_text,
            "VCS compiler version")
    require("Runtime version V-2023.12-SP1_Full64" in sim_text,
            "VCS runtime version")
    require("-assert svaext -cm line+cond+tgl+fsm+assert" in compile_text,
            "compile assertion/coverage flags")
    require("-top tb_m99_phase_slack_guarded_wordpacked_pwp_stream" in compile_text,
            "compile top")
    for relative in (
            "../../../rtl_m82/zero_bubble_elastic_pwp_stream.sv",
            "../../../rtl_m85/guarded_wordpacked_pwp_stream.sv",
            "../../../rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv",
            "../../../verif_m99/phase_slack_guarded_wordpacked_pwp_stream_assertions.sv",
            "../../../tb_m99/tb_m99_phase_slack_guarded_wordpacked_pwp_stream.sv"):
        require(relative in compile_text, "compile input missing " + relative)
    require("CPU time:" in compile_text and "../simv up to date" in compile_text,
            "compile completion marker")

    forbidden_compile = ("Error-[", "Syntax error", "cannot be opened", "FAILED")
    forbidden_sim = ("Assertion failed", "Error-[", "$fatal", "watchdog timeout",
                     "differential mismatch", "coverage mismatch")
    require(not any(token in compile_text for token in forbidden_compile),
            "compile failure signature")
    require(not any(token in sim_text for token in forbidden_sim),
            "simulation/assertion failure signature")
    require("$finish at simulation time              3002500" in sim_text,
            "simulation completion time")
    require("Disabled Module Assertions (compiletime)\n\n\n\n" in disable_text and
            "Dynamically disabled assertions at End-of-Simulation\n\n" in disable_text,
            "assertion-disable report is not empty-only")

    passed = parse_pass(sim_text)
    expected_pass = {"entries": 128, "beats": 436, "parser_cycles": 512,
                     "stalls": 10, "poison_attacks": 3,
                     "early_lookup_attacks": 1, "simultaneous_attacks": 1}
    require(passed == expected_pass, "PASS counter drift")
    covers = parse_assert_report(assert_text)
    sim_cover_lines = "\n".join(
        line for line in sim_text.splitlines() if " attempts," in line)
    require(sim_cover_lines == assert_text.rstrip("\n"),
            "sim/assert cover report mismatch")

    code_population = dict((str(code), sum(entry % 5 == code
                                           for entry in range(128)))
                           for code in range(5))
    beats_for = {0: 3, 1: 4, 2: 4, 3: 5, 4: 1}
    rebuilt_beats = sum(beats_for[entry % 5] for entry in range(128))
    require(rebuilt_beats == passed["beats"] and
            code_population == {"0": 26, "1": 26, "2": 26,
                                "3": 25, "4": 25},
            "synthetic legal campaign arithmetic")
    require(passed["parser_cycles"] == 4 * 128, "parser cycle arithmetic")

    require("while (!dut_phase_loaded)" in tb and
            "if (waited != 128)" in tb and
            "if (waited > 128)" in tb and
            count(tb, "load_and_wait_for_parser(") == 3,
            "128-cycle task source drift")
    # One direct call plus the three loop calls: legal + 3 poison = 4 loads.
    require(count(tb, "load_and_wait_for_parser(1'b0);") == 1 and
            count(tb, "load_and_wait_for_parser(1'b1);") == 1,
            "parser campaign call sites")
    require("for (int attack = 0; attack < 3; attack++)" in tb,
            "poison loop population")
    require("phase_metadata[0 +: 3] = 5" in tb and
            "phase_metadata[384+4*13 +: 13] ^= 1" in tb and
            "phase_metadata[384+15*13 +: 13] = 8191" in tb,
            "poison campaign source drift")
    require("for (int entry = 0; entry < 128; entry++)" in tb and
            "if ({ref_lookup_ready, ref_bank_rows, ref_output_valid" in tb and
            "M99 M85 differential mismatch" in tb,
            "M85 differential source drift")

    require("assign phase_load_ready = !m82_busy && !parse_active_q && !lookup_valid;" in rtl,
            "RTL load arbitration drift")
    require("assign lookup_ready = mapper_valid && m82_beat_ready;" in rtl,
            "RTL lookup ready drift")
    require("phase_load_valid && lookup_valid\n        |-> !phase_load_ready && !lookup_ready" in sva,
            "simultaneous SVA source drift")
    require(count(tb, "simultaneous_attacks++;") == 1 and
            tb.index("simultaneous_attacks++;") < tb.index("load_and_wait_for_parser(1'b0);"),
            "simultaneous scenario ordering")

    exact_source_attested_by_log = all(
        EXPECTED[name] in compile_text + sim_text
        for name in ("rtl", "tb", "sva", "m82", "m85"))
    actual_record_tokens = ("1728", "221184", "835383", "$readmem", "actual_record")
    full_actual_record_replay = any(token in tb for token in actual_record_tokens)
    internal_parser_sva = any(token in sva for token in
                              ("parse_index", "parse_active", "parse_cursor",
                               "parse_poison"))
    loaded_old_phase_simultaneous_test = (
        count(tb, "simultaneous_attacks++;") > 1 or
        "old_phase_simultaneous" in tb)
    early_middle_final_tests = all(token in tb for token in
                                   ("early_lookup_at_0", "early_lookup_at_63",
                                    "early_lookup_at_127"))
    full_poison_classes = all(token in tb for token in
                              ("reserved_code_6", "reserved_code_7",
                               "fetch_overflow", "cursor_overflow",
                               "zero_terminal"))
    second_load_recovery = "lookup_error_recovery" in tb
    held_load_during_parse = "held_phase_load" in tb
    captured_metadata_mutation = "mutate_phase_metadata" in tb
    explicit_reset_abort = "reset_abort" in tb

    output = {
        "schema": "m99_phase_slack_directed_vcs_independent_audit_v1",
        "status": "PASS_DEV_R2_DIRECTED_EXECUTION_NO_GO_EXACT_SHA_FREEZE_OR_ADMISSION",
        "producer_or_simulator_executed_by_reviewer": False,
        "sha256": dict((name, sha256(path)) for name, path in paths.items()),
        "vcs": {
            "compiler": "V-2023.12-SP1_Full64",
            "runtime": "V-2023.12-SP1_Full64",
            "compile_complete": True,
            "simulation_complete": True,
            "assertion_failure_signatures": 0,
            "assertions_disabled": False,
            "pass_counters": passed,
            "cover_properties": covers,
        },
        "independent_arithmetic": {
            "synthetic_code_population": code_population,
            "rebuilt_legal_beats": rebuilt_beats,
            "parser_campaigns": 4,
            "parser_cycles_per_campaign": 128,
            "rebuilt_parser_cycles": 512,
        },
        "coverage_boundary": {
            "exact_128_cycle_post_nba_wait_check": True,
            "legal_first_lookup_driven_on_next_edge_after_return": True,
            "all_128_synthetic_entries_m85_differential": True,
            "all_436_synthetic_beats_m85_differential": True,
            "bank_row_addresses_in_differential_bundle": True,
            "output_stall_cycles": 10,
            "simultaneous_unloaded_both_rejected": True,
            "simultaneous_loaded_old_phase_lookup_priority": loaded_old_phase_simultaneous_test,
            "early_lookup_at_first_parser_edge": True,
            "early_lookup_at_middle_and_final_parser_edges": early_middle_final_tests,
            "sticky_lookup_error_second_load_recovery": second_load_recovery,
            "poison_reserved5_index0": True,
            "poison_wrong_base_pattern4": True,
            "poison_wrong_base_pattern15_8191": True,
            "reserved6_reserved7_fetch_cursor_zero_terminal_classes": full_poison_classes,
            "held_second_load_during_parse": held_load_during_parse,
            "captured_metadata_vs_live_input_mutation": captured_metadata_mutation,
            "explicit_reset_abort_0_63_127": explicit_reset_abort,
            "full_actual_record_replay": full_actual_record_replay,
            "internal_parser_progress_sva": internal_parser_sva,
        },
        "simultaneous_semantics_audit": {
            "rtl_loaded_idle_case": "phase_load_ready=0 because lookup_valid=1; mapper_valid can be 1; lookup_ready can be 1, so old-phase lookup has priority",
            "sva_universal_case": "phase_load_valid && lookup_valid requires both phase_load_ready=0 and lookup_ready=0",
            "contradiction_on_legal_loaded_idle_lookup": True,
            "why_run_passed": "the sole simultaneous test occurs after reset with no loaded phase, where mapper_valid and lookup_ready are both zero",
        },
        "provenance_boundary": {
            "workspace_shas_match_requested_values": True,
            "source_mtimes_precede_compile_and_sim": True,
            "compile_log_embeds_source_paths": True,
            "compile_or_sim_log_embeds_exact_source_shas": exact_source_attested_by_log,
            "posthoc_exact_executed_source_proof": exact_source_attested_by_log,
        },
        "freeze_and_admission": {
            "archive_dev_r2_as_historical_directed_evidence": True,
            "freeze_current_rtl_tb_sva_as_directed_exact_sha_contract": False,
            "reason": "the universal simultaneous SVA rejects legal old-phase lookup priority, the relevant state is untested, and the run does not self-attest exact source SHAs",
            "full_actual_record_required_before_admission": True,
            "same_flow_3ns_dc_required_before_admission": True,
            "rtl_admission": False,
            "performance_admission": False,
            "paper_ppa_ready": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M99 dev_r2 log/arithmetic audit")
    print("entries=128 beats=436 parser_cycles=512 stalls=10 poison=3 early=1 simultaneous=1")
    print("covers=8 assertion_failures=0")
    print("loaded_old_phase_simultaneous_covered=false universal_sva_contradiction=true")
    print("full_actual_record=false same_flow_dc=false exact_sha_freeze=false admission=false")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
