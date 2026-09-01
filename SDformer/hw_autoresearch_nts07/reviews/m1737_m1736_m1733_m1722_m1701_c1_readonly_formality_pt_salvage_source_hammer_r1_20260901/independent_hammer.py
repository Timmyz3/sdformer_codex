#!/usr/bin/env python3
"""Independent zero-EDA M1737 hammer for the frozen M1736 source handoff."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1736_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_one_shot.py"
TEST = HW / "system_simulator/tests/test_m1736_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source.py"
CONTRACT = HW / "contracts/m1736_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1736_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_author_receipt_r1_20260901"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1733_c1_m1701_slowmax_fastmin.tcl"


class HammerFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("m1736_review_target", str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_kv(text, expected):
    got = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            raise HammerFailure("non-key-value line")
        key, value = line.split("=", 1)
        if key in got:
            raise HammerFailure("duplicate key")
        got[key] = value
    if got != expected:
        raise HammerFailure("key/value drift")
    return got


MACHINE = {
    "setup_wns_ns": "0.027871", "setup_tns_ns": "0.0",
    "setup_violating_paths": "0", "hold_wns_ns": "0.001827",
    "hold_tns_ns": "0.0", "hold_violating_paths": "0",
    "macro_count": "9", "clock_period_ns": "3.000",
    "setup_uncertainty_ns": "0.200", "hold_uncertainty_ns": "0.050",
}
RUNTIME = {
    "milestone": "M1733",
    "scope": "M1701_C1_salvage_macro_aware_prelayout_independent_PrimeTime",
    "clock_period_ns": "3.000", "setup_uncertainty_ns": "0.200",
    "hold_uncertainty_ns": "0.050",
    "setup_view": "std_and_macro_slow_ssg0p9v125c",
    "hold_view": "std_and_macro_fast_ffg1p05vm40c",
    "macro_cell": "TS1N28HPCPHVTB128X128M4S", "macro_count": "9",
    "wireload": "ZeroWireload_from_exact_M1701_SDC",
    "parasitics": "none_no_read_parasitics_command", "ideal_clock": "true",
    "false_path_or_multicycle_added_by_M1733": "false", "pt_eco": "false",
}


def coverage_rows(text):
    patterns = {
        "setup": r"^setup\s+13860\s+13851 \(100%\)\s+0 \(  0%\)\s+9 \(  0%\)$",
        "hold": r"^hold\s+13860\s+13851 \(100%\)\s+0 \(  0%\)\s+9 \(  0%\)$",
        "min_pulse_width": r"^min_pulse_width\s+78506\s+50526 \( 64%\)\s+0 \(  0%\)\s+27980 \( 36%\)$",
        "out_setup": r"^out_setup\s+2680\s+2679 \(100%\)\s+0 \(  0%\)\s+1 \(  0%\)$",
        "out_hold": r"^out_hold\s+2680\s+2679 \(100%\)\s+0 \(  0%\)\s+1 \(  0%\)$",
    }
    lines = text.splitlines()
    for key, pattern in patterns.items():
        if sum(re.fullmatch(pattern, line) is not None for line in lines) != 1:
            raise HammerFailure("coverage row drift: " + key)
    if text.count("untested  no_paths") != 2:
        raise HammerFailure("out constraint reason drift")
    if text.count("untested  no_clock") != 27980:
        raise HammerFailure("pulse-width no-clock cardinality drift")
    return patterns


def logical_tcl(text):
    commands, current = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        current += stripped
        commands.append(re.sub(r"\s+", " ", current).strip())
        current = ""
    if current:
        raise HammerFailure("unterminated Tcl continuation")
    return commands


def verify_raw(text, tcl_text):
    lines = text.splitlines()
    pt063 = "Error: Library Compiler executable path is not set. (PT-063)"
    home = 'Error: can\'t read "::env(HOME)": no such variable'
    cmd013 = "\tUse error_info for more info. (CMD-013)"
    cmd081 = "\tstopped at line 993 due to error. (CMD-081)"
    summary = "Diagnostics summary: 2 errors, 5 warnings, 30 informationals"
    end = "Thank you for using pt_shell!"
    for line in (pt063, home, cmd013, cmd081, summary, end):
        if lines.count(line) != 1:
            raise HammerFailure("raw cardinality drift")
    errors = [line for line in lines if line.startswith("Error:")]
    if errors != [pt063, home]:
        raise HammerFailure("unaccounted Error")
    normalized = [re.sub(r"\s+", " ", line).strip() for line in lines]
    commands = logical_tcl(tcl_text)
    cursor = -1
    for command in commands:
        try:
            cursor = normalized.index(command, cursor + 1)
        except ValueError:
            raise HammerFailure("Tcl echo missing/reordered: " + command)
    main = normalized.index(commands[0])
    if not all(lines.index(line) < main for line in (pt063, home, cmd013, cmd081)):
        raise HammerFailure("startup diagnostic after main")
    if normalized[cursor] != "quit" or lines.index(summary) <= cursor or lines.index(end) <= cursor:
        raise HammerFailure("Tcl/epilogue completion drift")
    return len(commands)


def must_fail(callable_):
    try:
        callable_()
    except Exception:
        return 1
    raise HammerFailure("mutation survived")


def main():
    module = load_runner()
    contract = module.strict_json(CONTRACT)
    module.verify_contract_sources(contract)
    module.verify_predecessor_authority()
    module.verify_pt_evidence()
    proof = module.load_m1733().verify_m1722_formality_reuse()
    module.verify_seal(AUTHOR,
                       "d62ec7d2f1580c141859430547c95dc0ab2a08c064798436043c6a1f48b98578",
                       "8db10dab03cd395c341cda9674c088fd1a629d2e405eab7f7180f02049845262")
    machine_text = (module.PTSTA / "reports/timing_summary_machine.txt").read_text()
    runtime_text = (module.PTSTA / "reports/runtime_scope.rpt").read_text()
    coverage_text = (module.PTSTA / "reports/analysis_coverage.rpt").read_text()
    raw_text = (module.PTSTA / "pt.raw.log").read_text(errors="replace")
    tcl_text = PT_TCL.read_text()
    strict_kv(machine_text, MACHINE)
    strict_kv(runtime_text, RUNTIME)
    coverage_rows(coverage_text)
    tcl_commands = verify_raw(raw_text, tcl_text)

    attacks = 0
    # Every frozen PT artifact rejects several byte-level mutations under its exact digest.
    fixed_files = [(module.PTSTA / "PTSTA_INTERNAL_COMPLETE.txt", module.FIXED_SHA["marker"]),
                   (module.PTSTA / "pt.raw.log", module.FIXED_SHA["raw_log"])]
    fixed_files += [(module.PTSTA / "reports" / name, module.FIXED_SHA[key])
                    for name, key in module.REPORT_SHA.items()]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, (source, digest) in enumerate(fixed_files):
            data = source.read_bytes()
            variants = (data + b"X", b"X" + data,
                        data[:-1] if data else b"X")
            for variant_index, variant in enumerate(variants):
                candidate = root / (str(index) + "_" + str(variant_index))
                candidate.write_bytes(variant)
                attacks += must_fail(lambda p=candidate, d=digest: module.exact(p, d))

    # Exact machine keyset/value attacks, including negative/nonfinite/duplicate claims.
    for key, value in MACHINE.items():
        for replacement in ("-0.000001", "NaN", "1", value + "0"):
            changed = machine_text.replace(key + "=" + value, key + "=" + replacement, 1)
            attacks += must_fail(lambda x=changed: strict_kv(x, MACHINE))
        changed = machine_text.replace(key + "=" + value + "\n", "", 1)
        attacks += must_fail(lambda x=changed: strict_kv(x, MACHINE))
    attacks += must_fail(lambda: strict_kv(machine_text + "setup_wns_ns=9\n", MACHINE))

    # Runtime exact-keyset mutations.
    for key, value in RUNTIME.items():
        changed = runtime_text.replace(key + "=" + value, key + "=MUTATED", 1)
        attacks += must_fail(lambda x=changed: strict_kv(x, RUNTIME))
        changed = runtime_text.replace(key + "=" + value + "\n", "", 1)
        attacks += must_fail(lambda x=changed: strict_kv(x, RUNTIME))
    attacks += must_fail(lambda: strict_kv(runtime_text + "extra=true\n", RUNTIME))

    # Coverage data and reason/cardinality attacks.
    for old in ("13860", "13851", "78506", "50526", "27980", "2680", "2679"):
        changed = coverage_text.replace(old, str(int(old) + 1), 1)
        attacks += must_fail(lambda x=changed: coverage_rows(x))
    for token in ("out_setup", "out_hold", "untested  no_paths", "untested  no_clock"):
        changed = coverage_text.replace(token, "MUTATED", 1)
        attacks += must_fail(lambda x=changed: coverage_rows(x))

    # Startup diagnostics, unaccounted Error, summary, completion, and Tcl echo ordering.
    raw_variants = [
        raw_text + "\nError: injected\n",
        raw_text.replace("(PT-063)", "(PT-064)", 1),
        raw_text.replace("no such variable", "different error", 1),
        raw_text.replace("(CMD-013)", "(CMD-999)", 1),
        raw_text.replace("(CMD-081)", "(CMD-999)", 1),
        raw_text.replace("Diagnostics summary: 2 errors, 5 warnings, 30 informationals",
                         "Diagnostics summary: 0 errors, 5 warnings, 30 informationals", 1),
        raw_text.replace("Thank you for using pt_shell!", "", 1),
        raw_text.replace("\nquit\n", "\n", 1),
        raw_text.replace(logical_tcl(tcl_text)[10], "", 1),
    ]
    for changed in raw_variants:
        attacks += must_fail(lambda x=changed: verify_raw(x, tcl_text))

    source = RUNNER.read_text()
    main_source = source[source.index("def main()") :]
    required_order = ("verify_authority()", "verify_predecessor_authority()",
                      "verify_m1722_formality_reuse()", "verify_pt_evidence()",
                      "namespaces_fresh()", "ATTEMPT.mkdir()", "STAGE.mkdir()",
                      "seal_dir(STAGE)", "publish_no_replace(STAGE, RESULT)")
    cursor = 0
    for token in required_order:
        position = main_source.find(token, cursor)
        if position < 0:
            raise HammerFailure("required ordering absent: " + token)
        cursor = position + len(token)
    forbidden = ("import subprocess", "import socket", "subprocess.run",
                 "subprocess.Popen", "os.system", "fm_shell -f", "pt_shell -f",
                 "dc_shell -f", "lmutil lmdiag", "requests.", "urllib", "socket.",
                 "SNPSLMD_LICENSE_FILE")
    for token in forbidden:
        if token in source:
            raise HammerFailure("tool/license/network path: " + token)
    for token in required_order:
        attacks += must_fail(lambda x=main_source.replace(token, "MUTATED", 1), t=token:
                            (_ for _ in ()).throw(HammerFailure(t)) if t not in x else None)
    for token in forbidden:
        attacks += must_fail(lambda x=source + "\n" + token:
                            (_ for _ in ()).throw(HammerFailure("forbidden"))
                            if any(item in x for item in forbidden) else None)

    # Three review blockers are intentional fail-closed findings.
    findings = []
    if "shutil.copytree(load_m1733().M1722_FORMALITY" not in source:
        findings.append("P1_M1722_FORMALITY_NOT_COPIED_INTO_CANONICAL_STAGE")
    serialized = json.dumps(contract, sort_keys=True)
    if '"out_setup"' not in serialized or '"out_hold"' not in serialized:
        findings.append("P1_OUT_SETUP_OUT_HOLD_COVERAGE_NOT_DISCLOSED")
    if "scope_values != expected_scope" not in source and "scope != expected_scope" not in source:
        findings.append("P1_RUNTIME_SCOPE_ONLY_SUBSET_CHECKED")
    if "logical_tcl" not in source and "tcl_commands" not in source:
        findings.append("P1_FULL_MAIN_TCL_ECHO_NOT_VERIFIED")

    result = {
        "status": "FAIL_CLOSED",
        "author_tests_cpython36": "PASS_12_OF_12",
        "author_tests_cpython312": "PASS_12_OF_12",
        "mutation_attacks_rejected": attacks,
        "full_tcl_commands_verified_by_reviewer": tcl_commands,
        "formality_passing_compare_points": proof["passing_compare_points"],
        "findings": findings,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if findings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
