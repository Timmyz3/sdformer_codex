#!/usr/bin/env python3
"""Static-only independent QA for the M1587 reserved-word repair.

This checker deliberately has no simulator or EDA execution path.  It compares
the repaired testbench with the frozen M1578 source, audits the already
consumed M1586 compile log, and runs in-memory robustness mutations.
"""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent
HW = ROOT / "hw_autoresearch_nts07"
OLD_COMMIT = "23be26d73606a26933bdc62f5c88236e5abbf8a0"
SOURCE_COMMIT = "842da3aa7e9aa2441e23953b643a836198d568d0"
TB_REL = "hw_autoresearch_nts07/dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv"
FILELIST_REL = "hw_autoresearch_nts07/dc_handoff/filelists/date_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.f"
CONTRACT_REL = "hw_autoresearch_nts07/contracts/m1587_m1578_c2_reserved_word_repair_source_contract_r1_20260901.json"
LOG_REL = "hw_autoresearch_nts07/results/m1586_c2_rtl_mapped_k8_case0_first_fault_r1_20260901/compile.log"
ATTEMPT_REL = "hw_autoresearch_nts07/results/.m1586_c2_rtl_mapped_k8_case0_first_fault_attempt_consumed"
TOP = "tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault"
EXPECTED = {
    FILELIST_REL: "09166d29aedc0a03266f9726ec006ac96efdd396c5290edb423ae303ad2548f1",
    TB_REL: "4a2ef4c40037274aadd936db8dbe38258aa39fa14a7e0322741f92acd958c435",
    CONTRACT_REL: "b5feba8a0a00318c1c767dda21892e26ee62b0f105112adb6da204161ea1a52b",
    LOG_REL: "21fa1fb0021acfbcf585989f0b0ccf98a8495a41010b252309f0f72c60448eaa",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def strict_json_bytes(data):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicates)


def git_show(commit, relative):
    return subprocess.check_output(
        ["git", "show", commit + ":SDformer/" + relative],
        cwd=str(REPO))


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def parse_repaired_tb(text, old_text):
    active = strip_comments(text)
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", active)
    require("tri" not in identifiers,
            "reserved word tri remains as an identifier")
    require(len(re.findall(r"\btri_state_char\b", active)) == 24,
            "tri_state_char identifier population drift")
    require(len(re.findall(r"\btri_state_char\s*\(", active)) == 21,
            "function declaration/call population drift")
    require(len(re.findall(r"\btri_state_char\s*=", active)) == 3,
            "four-state function assignment population drift")
    require(re.search(
        r"function\s+automatic\s+\[7:0\]\s+tri_state_char\s*"
        r"\(input\s+logic\s+value\)\s*;\s*"
        r"if\s*\(value\s*===\s*1'b0\)\s*tri_state_char\s*=\s*\"0\"\s*;\s*"
        r"else\s+if\s*\(value\s*===\s*1'b1\)\s*tri_state_char\s*=\s*\"1\"\s*;\s*"
        r"else\s+tri_state_char\s*=\s*\"X\"\s*;\s*endfunction",
        active, flags=re.S), "repaired four-state function does not parse")
    require(text.replace("tri_state_char", "tri") == old_text,
            "TB delta is not identifier-only tri -> tri_state_char")
    require(active.count("!==") == strip_comments(old_text).count("!=="),
            "four-state inequality population drift")
    require(active.count("===") == strip_comments(old_text).count("==="),
            "four-state equality population drift")
    require(active.count("$isunknown") ==
            strip_comments(old_text).count("$isunknown"),
            "unknown-state handling population drift")
    return {
        "repaired_identifier_occurrences": 24,
        "declaration_or_call_occurrences": 21,
        "function_assignments": 3,
        "canonicalized_bytes_equal_old": True,
    }


def check_compile_log(text):
    require(text.count("Error-[SE] Syntax error") == 1,
            "M1586 must contain exactly one syntax-error diagnostic")
    require(len(re.findall(r"^Error-\[", text, flags=re.M)) == 1,
            "M1586 contains another compiler error class")
    require("token is 'tri'" in text and
            'tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv", 350:' in text,
            "M1586 first parse failure identity drift")
    require(re.search(r"^1 warning\s*$", text, flags=re.M) is not None and
            re.search(r"^1 error\s*$", text, flags=re.M) is not None,
            "M1586 diagnostic totals drift")
    require(text.count("Warning-[") == 1 and
            "Warning-[LCA_FEATURES_ENABLED]" in text,
            "M1586 non-error diagnostic population drift")
    require("-top " + TOP in text and
            "-f dc_handoff/filelists/date_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.f" in text,
            "M1586 command identity drift")
    require("Top Level Modules:" not in text,
            "M1586 unexpectedly reached top elaboration")
    return {"parse_errors": 1, "other_errors": 0,
            "warnings": 1, "failed_token": "tri", "failed_line": 350}


def check_contract(contract):
    require(contract["schema"] ==
            "m1587_m1578_c2_reserved_word_repair_source_contract_r1_v1",
            "contract schema drift")
    require(contract["execution"] == {
        "vcs_compiles": 0, "simv_runs": 0, "ucli": False,
        "initreg": False, "saif": False, "ptpx": False,
        "attempt_consumed": False}, "source execution boundary drift")
    future = contract["future_execution"]
    require(future["authorized_now"] is False and
            future["different_author_hammer_required"] is True and
            future["budget"] == {"vcs_compiles": 1, "simv_runs": 1,
                                 "cases": ["k8_case0"], "saif": 0,
                                 "ptpx": 0}, "future budget drift")
    require(contract["sole_delta"] == {
        "before": "function automatic [7:0] tri(input logic value)",
        "after": "function automatic [7:0] tri_state_char(input logic value)",
        "call_sites_renamed_only": True,
        "dut_sources_changed": False,
        "stimulus_changed": False,
        "memory_changed": False,
        "fault_or_event_semantics_changed": False}, "sole-delta claim drift")


def check_all(tb_text, old_tb_text, filelist_text, log_text, contract):
    parsed = parse_repaired_tb(tb_text, old_tb_text)
    compile_result = check_compile_log(log_text)
    check_contract(contract)
    require(sha256_bytes(filelist_text.encode("utf-8")) ==
            EXPECTED[FILELIST_REL], "filelist identity drift")
    entries = [line.strip() for line in filelist_text.splitlines()
               if line.strip()]
    require(len(entries) == 16 and entries[-1] == TB_REL.split("/", 1)[1],
            "filelist ordering/top source drift")
    return parsed, compile_result, entries


def mutation_result(name, callable_):
    try:
        callable_()
    except Exception:
        return {"name": name, "rejected": True}
    return {"name": name, "rejected": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    current_commit_blob = subprocess.check_output(
        ["git", "rev-parse", SOURCE_COMMIT], cwd=str(REPO),
        universal_newlines=True).strip()
    require(current_commit_blob == SOURCE_COMMIT, "source commit missing")

    for relative, expected in EXPECTED.items():
        require(sha256(ROOT / relative) == expected,
                "frozen SHA drift: " + relative)
    tb_text = (ROOT / TB_REL).read_text(encoding="utf-8")
    old_tb_text = git_show(OLD_COMMIT, TB_REL).decode("utf-8")
    filelist_text = (ROOT / FILELIST_REL).read_text(encoding="utf-8")
    old_filelist = git_show(OLD_COMMIT, FILELIST_REL).decode("utf-8")
    require(filelist_text == old_filelist, "filelist changed since M1578")
    log_text = (ROOT / LOG_REL).read_text(encoding="utf-8", errors="replace")
    contract = strict_json_bytes((ROOT / CONTRACT_REL).read_bytes())
    parsed, compile_result, entries = check_all(
        tb_text, old_tb_text, filelist_text, log_text, contract)

    current_tb_blob = git_show(SOURCE_COMMIT, TB_REL)
    require(current_tb_blob == (ROOT / TB_REL).read_bytes(),
            "repaired TB worktree differs from source commit")
    unchanged_versioned = []
    for entry in entries[1:-1]:
        relative = "hw_autoresearch_nts07/" + entry
        require(git_show(OLD_COMMIT, relative) ==
                git_show(SOURCE_COMMIT, relative),
                "versioned DUT/memory source changed: " + relative)
        unchanged_versioned.append(relative)
    require((ROOT / ATTEMPT_REL).is_dir(), "M1586 attempt marker missing")
    result_dir = ROOT / "hw_autoresearch_nts07/results/m1586_c2_rtl_mapped_k8_case0_first_fault_r1_20260901"
    require(not (result_dir / "simv").exists() and
            not (result_dir / "sim.log").exists(),
            "M1586 unexpectedly contains simulation output")

    def full(tb=tb_text, filelist=filelist_text, log=log_text, obj=contract):
        return check_all(tb, old_tb_text, filelist, log, obj)

    mutations = [
        mutation_result("reserved_identifier_restored",
                        lambda: full(tb=tb_text.replace("tri_state_char", "tri"))),
        mutation_result("one_call_site_not_renamed",
                        lambda: full(tb=tb_text.replace("tri_state_char(", "old_name(", 1))),
        mutation_result("x_render_coerced_to_zero",
                        lambda: full(tb=tb_text.replace('tri_state_char = "X"',
                                                        'tri_state_char = "0"'))),
        mutation_result("case_inequality_weakened",
                        lambda: full(tb=tb_text.replace("!==", "!=", 1))),
        mutation_result("unknown_detector_removed",
                        lambda: full(tb=tb_text.replace("$isunknown", "$removed", 1))),
        mutation_result("filelist_top_removed",
                        lambda: full(filelist="\n".join(entries[:-1]) + "\n")),
        mutation_result("compile_error_duplicated",
                        lambda: full(log=log_text + "\nError-[SE] Syntax error\n")),
        mutation_result("failed_token_changed",
                        lambda: full(log=log_text.replace("token is 'tri'",
                                                         "token is 'wire'"))),
        mutation_result("compile_top_changed",
                        lambda: full(log=log_text.replace("-top " + TOP,
                                                         "-top wrong_top"))),
        mutation_result("future_preauthorized",
                        lambda: (contract["future_execution"].__setitem__(
                            "authorized_now", True), check_contract(contract))),
    ]
    contract["future_execution"]["authorized_now"] = False
    require(all(row["rejected"] for row in mutations),
            "a robustness mutation was accepted")

    result = {
        "schema": "m1588_m1587_reserved_word_repair_incremental_qa_r1_v1",
        "status": "PASS_M1588_INCREMENTAL_SOURCE_QA__AUTHORIZE_ONE_NEW_IDENTITY_COMPILE_AND_ONE_CASE0_SIM__NO_TOOL_RUN",
        "runtime": {"implementation": platform.python_implementation(),
                    "version": platform.python_version()},
        "source_commit": SOURCE_COMMIT,
        "m1586_compile_log": compile_result,
        "repair": parsed,
        "unchanged": {
            "filelist": True,
            "versioned_dut_mapped_memory_sources": len(unchanged_versioned),
            "stimulus_except_identifier_spelling": True,
            "four_state_semantics": True,
        },
        "robustness_mutations": {"passed": len(mutations),
                                 "count": len(mutations), "rows": mutations},
        "execution_by_m1588": {"vcs_compiles": 0, "simv_runs": 0,
                               "eda_runs": 0, "attempt_consumed": False},
        "authorization_recommendation": {
            "new_identity_required": True,
            "vcs_compiles": 1, "simv_runs": 1,
            "cases": ["k8_case0"], "saif": 0, "ptpx": 0,
            "required_top": TOP,
            "required_filelist_sha256": EXPECTED[FILELIST_REL],
        },
        "claim_boundary": {"source_qa_only": True, "rtl_pass": False,
                           "mapped_pass": False, "paper_citable": False},
    }
    encoded = json.dumps(result, indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
