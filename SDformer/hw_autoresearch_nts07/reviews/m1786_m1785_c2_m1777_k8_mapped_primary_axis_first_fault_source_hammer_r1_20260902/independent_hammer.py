#!/usr/bin/env python3
"""Different-author, CPU-only M1786 hammer for the inert M1785 diagnostic."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source.py"
TEST = HW / "system_simulator/tests/test_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_one_shot.py"
TB = HW / "dc_handoff/tb/tb_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic.f"
CONTRACT = HW / "contracts/m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1785_m1777_c2_k8_mapped_primary_axis_first_fault_source_author_receipt_r1_20260902"
FAILURE = HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902.failed_or_incomplete.quarantine"
NETLIST = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901/k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
M1334_ASSERT = HW / "dc_handoff/tb/m1334_c2_production_activity_assertions.sv"
M1684_ASSERT = HW / "dc_handoff/tb/m1684_c2_m1609_production_binary_fault_assertions.sv"
M1684_WRAPPER = HW / "dc_handoff/tb/tb_m1684_c2_m1609_fresh_mapped_production_energy.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    CHECKER: "d525a61d76708faa8a8b0c8825464119dac941eb3e3339dce5a64a3f4e6272f6",
    TEST: "b118b244980cd02c8e7c7d6cf9843994adaeb5ea7393bc5b68cb6c9c4def8bea",
    RUNNER: "6c44d8333cca47aa137093f20656f92156f6e858a4234411e32ef0773e395615",
    TB: "a4332e9209d2fed9415adc18de917e8da41e4015b1a4f810ba02ff8fc540edc0",
    FILELIST: "76c8cb2f872822cf9faac75f31a66bbbe134c5a029be43edf7eab51d94354eab",
    CONTRACT: "9a60cbf1398768512196d464e011af62d908e3c608fc9d0c9e08be5ee5798f3c",
    NETLIST: "6c62d99b444ba25f8eb3f1e491479b44f5613b0323e032af8150e81c84f393c4",
    M1334_ASSERT: "86be3fa541bf65afa6ada99aa3e2bd494ed689594fece18cfea135b91420c32a",
    M1684_ASSERT: "39fdc0f47628272a6f1a7b6887da52fdbf4d71f1f5fe6557d4a7022f06bc62b1",
    M1684_WRAPPER: "034934d1cdb6dc683ffa51811bd363fadd02673a1311a5b715b5c4b0e3cb5a2e",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

SPEC = importlib.util.spec_from_file_location("m1786_live_checker", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("checker loader unavailable")
SPEC.loader.exec_module(M)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=pairs,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_sealed_dir(path, manifest_sha, outer_sha):
    path = Path(path)
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    need(path.is_dir() and not path.is_symlink(), "sealed directory missing")
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
         "sealed directory identity")
    need(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
         "outer seal content")
    listed = set()
    for row in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = row.split("  ", 1)
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts
             and name not in listed, "unsafe/duplicate manifest row")
        member = path / rel
        need(member.is_file() and not member.is_symlink()
             and sha(member) == digest, "manifest member drift: " + name)
        listed.add(name)
    actual = set(member.relative_to(path).as_posix()
                 for member in path.rglob("*")
                 if member.is_file() and member.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed directory population")


def rejected(function, value):
    try:
        function(value)
    except Exception:
        return
    raise RuntimeError("mutation accepted")


def runtime_log(code, include_public=True, exact=True, settled=4):
    rows = []
    if code is not None:
        rows.append("M1785_FIRST_UNKNOWN code=%d class=%s field=f%d time_ps=25501" %
                    (code, "DIAGNOSTIC_TAP" if code == 37 else "FAULT", code))
    if include_public and code == 37:
        rows.append("M1785_FIRST_UNKNOWN code=1 class=FAULT field=protocol_error time_ps=25501")
    if exact:
        rows.append("M1684 mapped fault vector contains X/Z")
    rows.append("M1785_FINAL first_unknown_seen=1 first_unknown_code=%d first_unknown_time_ps=25501 settled_samples=%d exact_m1684_assertion_preserved=1 initreg=0 force=0 ignore_x=0" %
                (code if code is not None else 1, settled))
    return "\n".join(rows)


def main():
    for path, expected in PINS.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == expected,
             "pinned identity drift: " + str(path))
    verify_sealed_dir(AUTHOR,
        "1053caddf46b7214243c4f969334994427a18b8e673ae66009ff7b9e81330eab",
        "6ed202f019800e5fb3f572155655e5c2d24611640629c4d1f3e6dd3cc50e411c")
    verify_sealed_dir(FAILURE,
        "c99303b4362f62b4e6dde5c628e69cefb64344a8e22f349942e37a6e09d5bc1f",
        "f52dba4e28abe803c7811e8be2d2d74f3189944ec599e2b14102dbfb15591b33")

    failure = strict_json(FAILURE / "failure.json")
    need(failure["status"] == "FAILED_OR_INCOMPLETE"
         and failure["phase"] == "SIM_k8_0"
         and failure["counts"] == {"ptpx_runs": 0, "saif_files": 0,
                                   "simv_runs": 1, "vcs_compiles": 1}
         and failure["attempt_consumed"] is True
         and failure["automatic_retry"] is False
         and failure["partial_axis_citable"] is False,
         "M1777 sealed failure boundary")
    author = strict_json(AUTHOR / "author_receipt.json")
    need(author["status"] ==
         "PASS_M1785_SOURCE_AUTHORING__READY_FOR_DIFFERENT_AUTHOR_M1786_HAMMER__NO_EDA",
         "author status")
    need(author["p0_count"] == 0 and author["p1_count"] == 0,
         "author unresolved severity")
    need(author["future_budget_after_m1786"] == {
        "attempts": 1, "automatic_retry": False, "ptpx_runs": 0,
        "saif_files": 0, "simv_runs": 1, "vcs_compiles": 1},
        "author future budget")

    contract = strict_json(CONTRACT)
    need(contract["future_execution"]["authorized_now"] is False,
         "source contract self-authorized")
    need(contract["future_execution"]["attempts"] == 1
         and contract["future_execution"]["vcs_compiles"] == 1
         and contract["future_execution"]["simv_runs"] == 1
         and contract["future_execution"]["saif_files"] == 0
         and contract["future_execution"]["ptpx_runs"] == 0
         and contract["future_execution"]["automatic_retry"] is False,
         "contract future budget")
    need(contract["claim_boundary"]["diagnostic_only"] is True
         and not any(contract["claim_boundary"][key] for key in
                     ("mapped_functionality", "power", "energy", "performance",
                      "paper_citable", "system_speedup", "headline")),
         "claim boundary")

    value = M.main()
    need(value["status"] == "PASS_M1785_SOURCE_ONLY_NO_EDA_NO_ATTEMPT",
         "live source check")
    tb = TB.read_text(encoding="utf-8")
    filelist = FILELIST.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    M.audit_tb(tb)
    M.audit_filelist(filelist)
    M.audit_runner(runner)

    rows = [row.strip() for row in filelist.splitlines() if row.strip()]
    need(rows.count(str(M1334_ASSERT)) == 1
         and rows.count(str(M1684_ASSERT)) == 1
         and rows.count(str(M1684_WRAPPER)) == 1,
         "exact M1334/M1684 chain not unique")
    wrapper = M1684_WRAPPER.read_text(encoding="utf-8")
    need(wrapper.count("m1334_c2_production_activity_assertions production_checks") == 1
         and wrapper.count("m1684_c2_m1609_production_binary_fault_assertions fault_checks") == 1,
         "original monitors not instantiated exactly once")
    m1684 = M1684_ASSERT.read_text(encoding="utf-8")
    need(m1684.count("check_fault_vector();") == 2
         and "always @(posedge clk_core)" in m1684
         and "always @(negedge clk_core)" in m1684,
         "M1684 both-phase exact assertion drift")

    lowered = "\n".join(re.sub(r"//.*$", "", line)
                         for line in (tb + "\n" + runner + "\n" + filelist).splitlines()).lower()
    for token in ("+vcs+initreg", "ignore_x=1", "ignorex", "coerce_x",
                  "assert disable", "noassert", "notimingcheck", "nospecify"):
        need(token not in lowered, "forbidden X/assertion mechanism: " + token)
    need(not re.search(r"(?im)(?:^|[;{])\s*(?:force|release)\s+", lowered),
         "active force/release")
    need(runner.count("subprocess.run(compile_command") == 1
         and runner.count("subprocess.run(sim_command") == 1
         and "ATTEMPT.mkdir()" in runner
         and "ATTEMPT.mkdir(exist_ok=True)" not in runner
         and runner.count('"automatic_retry": False') == 2
         and '"automatic_retry": True' not in runner,
         "one-attempt compile/sim gate")
    for tap in (
            "g_k8_implementation_core_frontend_m202_protocol_error",
            "g_k8_implementation_core_frontend_paired_sink_fault_q",
            "g_k8_implementation_core_adapter_fault_q",
            "g_k8_implementation_core_g_k8_service_fault_q",
            "g_k8_implementation_memory_adapter_fault_q",
            "g_k8_implementation_memory_adapter_stale_q"):
        need(tap in NETLIST.read_text(encoding="utf-8"), "mapped tap absent: " + tap)

    tb_mutations = [
        tb.replace("#1ps;", "", 1),
        tb.replace("tb_m1684_c2_m1609_fresh_mapped_production_energy sealed();", "", 1),
        tb.replace("ignore_x=0", "ignore_x=1", 1),
        tb.replace("flag_unknown(1, \"FAULT\", \"protocol_error\")", "", 1),
        tb.replace("flag_unknown(37, \"DIAGNOSTIC_TAP\"", "flag_unknown(1, \"DIAGNOSTIC_TAP\"", 1),
        tb + "\nforce sealed.core.protocol_error = 1'b0;\n",
    ]
    for mutant in tb_mutations:
        rejected(M.audit_tb, mutant)
    filelist_mutations = [
        filelist.replace(str(M1334_ASSERT), "", 1),
        filelist.replace(str(M1684_ASSERT), "", 1),
        filelist.replace(str(M1684_WRAPPER), "", 1),
        filelist.replace("+define+SVA_RUNTIME_ENABLED\n", "", 1),
        filelist + "rtl_m803/injected.sv\n",
    ]
    for mutant in filelist_mutations:
        rejected(M.audit_filelist, mutant)
    runner_mutations = [
        runner.replace("ATTEMPT.mkdir()", "ATTEMPT.mkdir(exist_ok=True)", 1),
        runner.replace('"vcs_compiles": 1', '"vcs_compiles": 2', 1),
        runner.replace('"simv_runs": 1', '"simv_runs": 2', 1),
        runner.replace("subprocess.run(compile_command", "subprocess.run(other_command", 1),
        runner.replace("subprocess.run(sim_command", "subprocess.run(other_command", 1),
        runner + "\n+vcs+initreg+random\n",
        runner + "\nforce dut.fault 0\n",
    ]
    for mutant in runner_mutations:
        rejected(M.audit_runner, mutant)

    runtime_passes = 0
    for code in range(1, 37):
        localized = M.check_runtime_text(runtime_log(code))
        need(localized["first_unknown_code"] == code, "runtime field class")
        runtime_passes += 1
    private_then_public = M.check_runtime_text(runtime_log(37, include_public=True))
    need(private_then_public["first_unknown_code"] == 37
         and 1 in private_then_public["reported_unknown_codes"],
         "supplemental-before-public trace")
    for bad in (runtime_log(1, exact=False), runtime_log(37, include_public=False),
                runtime_log(1, settled=0), runtime_log(None)):
        rejected(M.check_runtime_text, bad)

    attempt = HW / "results/.m1785_c2_m1777_k8_mapped_primary_axis_first_fault_attempt_consumed"
    result = HW / "results/m1785_c2_m1777_k8_mapped_primary_axis_first_fault_r1_20260902"
    need(not os.path.lexists(str(attempt)) and not os.path.lexists(str(result)),
         "attempt/result namespace already consumed")

    output = {
        "schema": "m1786_m1785_c2_k8_first_fault_independent_hammer_output_r1_v1",
        "status": "PASS_M1786_M1785_C2_K8_MAPPED_FIRST_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "m1777_failure_double_seal": True,
        "author_receipt_double_seal": True,
        "exact_m1334_assertion_preserved": True,
        "exact_m1684_assertion_preserved": True,
        "m1684_both_phase_checks": True,
        "public_field_codes_exercised": runtime_passes,
        "supplemental_private_then_public_case_exercised": True,
        "tb_mutations_rejected": len(tb_mutations),
        "filelist_mutations_rejected": len(filelist_mutations),
        "runner_mutations_rejected": len(runner_mutations),
        "runtime_mutations_rejected": 4,
        "total_mutations_rejected": len(tb_mutations) + len(filelist_mutations)
                                    + len(runner_mutations) + 4,
        "future_budget": {"attempts": 1, "vcs_compiles": 1,
                          "simv_runs": 1, "saif_files": 0,
                          "ptpx_runs": 0, "automatic_retry": False},
        "eda_runs": 0,
        "attempts_created": 0,
        "results_created": 0,
        "docs359_sha256": PINS[DOCS359],
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
