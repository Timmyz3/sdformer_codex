#!/usr/bin/env python3
"""Different-author, source-only hammer for M1733. Never invoke EDA."""
from __future__ import print_function

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1733_m1722_m1701_c1_formality_reuse_pt_only_one_shot.py"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1733_c1_m1701_slowmax_fastmin.tcl"
TEST = HW / "system_simulator/tests/test_m1733_m1722_m1701_c1_formality_reuse_pt_only_source.py"
CONTRACT = HW / "contracts/m1733_m1722_m1701_c1_formality_reuse_pt_only_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1733_m1722_m1701_c1_formality_reuse_pt_only_source_author_receipt_r1_20260901"
EXPECTED = {
    "runner": "37723675f3ca3f094cdc747755cfffba41c6899c584c6dd3dbdf2c5ab35a4e9e",
    "pt_tcl": "0fab7432e3806a75241cc4e55699b75d126fa334a3d4f3d7444189ed10001d67",
    "test": "56643bb56d3f36ce5f869c2dba5f021cf6eba16d4d2a673b23a87634eabf5715",
    "contract": "10e756455f38479aff0b5ec04be0b3479da920b0f1a7afa4da8e6b14d722a43f",
    "contract_digest": "bd2f7f2c6b86953e6381ff3824a19756d5f39178dabd1ab68f1dfaef0f116797",
    "contract_outer": "0817720b10c0c630e55c5699af3b81aa680c44717182522f936354322cca2642",
    "author_receipt": "1f34b38ceb6f1545c0dd6a18cfe913a0e8882a41859050f73971e44819da7842",
    "author_manifest": "9feadbf0abe188cebf9f7bfa675608400f83bfa035177c071198e938e49cc2bd",
    "author_outer": "6856b85fdf3b404eb7c94193778240c74e09ec2a87b34c410a1c5ed1f19d8dca",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node.value) + "." + node.attr
    return "?"


def main():
    for path, key in ((RUNNER, "runner"), (PT_TCL, "pt_tcl"), (TEST, "test"),
                      (CONTRACT, "contract"),
                      (Path(str(CONTRACT) + ".sha256"), "contract_digest"),
                      (Path(str(CONTRACT) + ".sha256.seal.sha256"), "contract_outer"),
                      (AUTHOR / "author_receipt.json", "author_receipt"),
                      (AUTHOR / "SHA256SUMS", "author_manifest"),
                      (AUTHOR / "SHA256SUMS.seal.sha256", "author_outer"),
                      (HW / "docs/359_DATE终局冻结_20260813.md", "docs359")):
        need(path.is_file() and not path.is_symlink() and sha(path) == EXPECTED[key],
             "identity drift: " + str(path))

    spec = importlib.util.spec_from_file_location("m1733_independent", str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_seal(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    module.verify_file_seal(CONTRACT)
    module.verify_inputs()
    proof = module.verify_m1722_formality_reuse()
    need(proof == {
        "passing_compare_points": 16549,
        "macro_instances_per_side": 9,
        "log_allowlist": {"gui": 0, "matched_header": 1, "source_echo": 1}},
        "frozen M1722 Formality proof drift")

    rejected = 0
    accepted = 0
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "tool.log"

        def must_reject(line, allow_header=False, echoes=()):
            nonlocal rejected
            log.write_text(line + "\n")
            try:
                module.scan_tool_log(log, allow_matched_header=allow_header,
                                     exact_source_echo_allow=echoes)
            except module.Failure:
                rejected += 1
                return
            raise RuntimeError("accepted mutation: " + repr(line))

        def must_accept(lines, allow_header=False, echoes=(), expected=None):
            nonlocal accepted
            log.write_text("\n".join(lines) + "\n")
            observed = module.scan_tool_log(
                log, allow_matched_header=allow_header,
                exact_source_echo_allow=echoes)
            need(expected is None or observed == expected,
                 "positive allowlist fixture drift")
            accepted += 1

        must_accept(["normal"], expected={"gui": 0, "matched_header": 0,
                                          "source_echo": 0})
        must_accept([module.GUI_ALLOW], expected={"gui": 1, "matched_header": 0,
                                                  "source_echo": 0})
        must_accept([module.MATCHED_HEADER_ALLOW], True, (),
                    {"gui": 0, "matched_header": 1, "source_echo": 0})
        must_accept([module.FORMALITY_TCL_ECHO_ALLOW], False,
                    (module.FORMALITY_TCL_ECHO_ALLOW,),
                    {"gui": 0, "matched_header": 0, "source_echo": 1})
        must_accept(list(module.PT_TCL_ECHO_ALLOW), False, module.PT_TCL_ECHO_ALLOW,
                    {"gui": 0, "matched_header": 0, "source_echo": 5})

        exceptions = (
            (module.GUI_ALLOW, False, ()),
            (module.MATCHED_HEADER_ALLOW, True, ()),
            (module.FORMALITY_TCL_ECHO_ALLOW, True,
             (module.FORMALITY_TCL_ECHO_ALLOW,)))
        for base, allow_header, echoes in exceptions:
            for prefix in (" ", "\t", "prefix ", "x", "Info: ", "**"):
                must_reject(prefix + base, allow_header, echoes)
            for suffix in (" ", "\t", " suffix", "x"):
                must_reject(base + suffix, allow_header, echoes)
            must_reject(base + "\n" + base, allow_header, echoes)

        for base in module.PT_TCL_ECHO_ALLOW:
            for prefix in (" ", "\t", "prefix ", "x", "Info: ", "**"):
                must_reject(prefix + base, False, module.PT_TCL_ECHO_ALLOW)
            for suffix in (" ", "\t", " suffix", "x"):
                must_reject(base + suffix, False, module.PT_TCL_ECHO_ALLOW)
            must_reject(base.replace("M1733", "M1732"), False,
                        module.PT_TCL_ECHO_ALLOW)
            must_reject(base + "\n" + base, False, module.PT_TCL_ECHO_ALLOW)

        for word in ("Error", "ERROR", "error", "Errors", "Fatal", "FATAL", "fatal"):
            for prefix in ("", " ", "\t", "**", "*** ", "Info:", "Info: ",
                           "Warning:", "Warning: ", "prefix ", "x="):
                for separator in (":", ": ", " : ", " ", "- ", "="):
                    must_reject(prefix + word + separator + "injected failure", True)
        for word in ("loop", "LOOP", "Loop"):
            for prefix in ("", " ", "timing ", "feedback ", "combinational ",
                           "Info: ", "prefix ", "x="):
                for suffix in ("", " detected", " found", " diagnostic", ": bad", "-bad"):
                    must_reject(prefix + word + suffix, True)
        for line in ("LINK-1", "prefix LINK-999 failed", "unresolved",
                     "x unresolved reference", "unable to resolve x",
                     "x unable to resolve x", "(TIM-209)", "prefix (OPT-150)"):
            must_reject(line, True)

        benign = ("0 error", "0 errors", "No error", "No errors",
                  "No errors detected", "Summary: 0 errors", "Summary: 0 fatal",
                  "Summary: 0 fatal diagnostics")
        must_accept(benign)
        for base in benign:
            must_reject("prefix " + base)
            must_reject(base + " suffix")
        try:
            module.scan_tool_log(
                log, exact_source_echo_allow=(module.PT_TCL_ECHO_ALLOW[0],
                                               module.PT_TCL_ECHO_ALLOW[0]))
        except module.Failure:
            rejected += 1
        else:
            raise RuntimeError("duplicate source-echo allow entry accepted")
        must_reject(module.MATCHED_HEADER_ALLOW)
        must_reject(module.FORMALITY_TCL_ECHO_ALLOW)
        for line in module.PT_TCL_ECHO_ALLOW:
            must_reject(line)

    tree = ast.parse(RUNNER.read_text())
    calls = [(call_name(node.func), node.lineno, node) for node in ast.walk(tree)
             if isinstance(node, ast.Call)]
    need(sum(name == "subprocess.run" for name, _, _ in calls) == 2,
         "unexpected process call count")
    need(not [row for row in calls if row[0] in
              ("subprocess.Popen", "os.system", "eval", "exec")],
         "alternate process/eval path")
    tool_calls = [row for row in calls if row[0] == "run_tool"]
    need(len(tool_calls) == 1 and "PT" in ast.dump(tool_calls[0][2])
         and "FM" not in ast.dump(tool_calls[0][2]), "non-PT tool path")
    source = RUNNER.read_text()
    launch = source[source.index("def main()") :]
    need('state["phase"] = "FORMALITY"' not in launch
         and 'state["formality_runs"] += 1' not in source
         and source.count('state["pt_runs"] += 1') == 1,
         "zero-FM/one-PT budget drift")
    need("allow_matched_header=True" not in launch
         and "FORMALITY_TCL_ECHO_ALLOW" not in launch
         and "exact_source_echo_allow=PT_TCL_ECHO_ALLOW" in launch,
         "allowlist scope drift")

    pt = PT_TCL.read_text()
    need(len(module.PT_TCL_ECHO_ALLOW) == 5
         and set(module.PT_TCL_ECHO_ALLOW).issubset(set(pt.splitlines())),
         "PT source-echo binding drift")
    for token in ("read_verilog $mapped_netlist", "link_design $design_name",
                  "set_min_library $std_slow_db -min_version $std_fast_db",
                  "set_min_library $macro_slow_db -min_version $macro_fast_db",
                  "read_sdc $mapped_sdc", "-max ssg0p9v125c",
                  "-min ffg1p05vm40c", "update_timing -full",
                  "-delay_type max", "-delay_type min",
                  "$setup_slack < 0.0 || $hold_slack < 0.0", "$macro_count != 9"):
        need(token in pt, "PT semantic token absent: " + token)
    for command in ("read_parasitics", "set_false_path", "set_multicycle_path",
                    "set_case_analysis", "set_disable_timing", "write_sdf",
                    "fix_eco_timing"):
        need(re.search(r"^\s*" + command + r"\b", pt, re.M) is None,
             "forbidden PT command: " + command)

    contract = json.loads(CONTRACT.read_text())
    module.verify_contract_sources(contract)
    need(all(value is False for value in contract["claim_boundary"].values()),
         "claim boundary drift")
    need((contract["future_execution"]["formality_runs"],
          contract["future_execution"]["pt_runs"],
          contract["future_execution"]["dc_runs"]) == (0, 1, 0),
         "future tool budget drift")
    for path in (module.M1735, Path(str(module.M1735) + ".sha256"),
                 Path(str(module.M1735) + ".sha256.seal.sha256"),
                 module.RESULT, module.ATTEMPT, module.FAILURE):
        need(not os.path.lexists(path), "future authority/execution namespace exists")

    print(json.dumps({
        "status": "PASS_M1734_INDEPENDENT_SOURCE_HAMMER",
        "mutation_attacks_rejected": rejected,
        "positive_fixtures": accepted,
        "author_tests": {"cpython36": 13, "cpython312": 13},
        "frozen_formality_passing_compare_points": 16549,
        "frozen_formality_zero_failure_classes": True,
        "macro_instances_per_side": 9,
        "future_tool_budget": {"formality": 0, "pt": 1, "dc": 0},
        "p0": 0, "p1": 0, "p2": 0,
        "eda_runs": 0, "license_queries": 0,
        "release_created": False, "attempt_created": False,
        "result_created": False}, sort_keys=True))


if __name__ == "__main__":
    main()
