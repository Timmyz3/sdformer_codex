#!/usr/bin/env python3
"""Different-author, source-only hammer for M1722. Never launches EDA."""
from __future__ import print_function

import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1722_m1701_c1_salvage_formality_pt_one_shot.py"
FM_TCL = HW / "dc_handoff/scripts/run_formality_m1722_c1_m1665_to_m1701_gate_to_gate.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1722_c1_m1701_slowmax_fastmin.tcl"
CONTRACT = HW / "contracts/m1722_m1701_c1_salvage_formality_pt_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1722_m1701_c1_salvage_formality_pt_source_author_receipt_r1_20260901"


def load(path):
    spec = importlib.util.spec_from_file_location("m1722_independent_runner", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load(RUNNER)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_file_double_seal(path):
    path = Path(path)
    digest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(digest.read_text() == sha(path) + "  " + path.name + "\n",
         "payload digest seal")
    need(outer.read_text() == sha(digest) + "  " + digest.name + "\n",
         "payload outer seal")


def verify_dir_seal(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text() == sha(manifest) + "  SHA256SUMS\n", "directory outer seal")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        need(name not in listed, "duplicate manifest member")
        need(sha(root / name) == digest, "manifest member drift: " + name)
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed directory population drift")


def main():
    checks = []
    attacks = []

    contract = M.strict_json(CONTRACT)
    M.verify_contract_sources(contract)
    verify_file_double_seal(CONTRACT)
    verify_dir_seal(AUTHOR)
    checks.append("source_contract_and_author_receipt_double_seals")

    area = M.verify_inputs()
    need(abs(area - 166514.312080) < 1e-9, "area identity")
    need(sha(HW / "docs/359_DATE终局冻结_20260813.md") == M.FIXED_SHA["docs359"],
         "docs359 drift")
    checks.append("m1701_m1665_m1714_and_docs359_frozen_identity")

    fm = FM_TCL.read_text()
    need(fm.index("read_verilog -r $reference_netlist") <
         fm.index("read_verilog -i $implementation_netlist"), "Formality direction")
    for token in ("M1722_M1665_REFERENCE_NETLIST", "M1722_M1701_IMPLEMENTATION_NETLIST",
                  "set verification_succeeded [verify]", "report_unmatched_points",
                  "report_failing_points", "report_aborted_points",
                  "report_unverified_points"):
        need(token in fm, "Formality token absent: " + token)
    checks.append("gate_to_gate_m1665_reference_to_m1701_implementation")

    pt = PT_TCL.read_text()
    for token in ("set_min_library $std_slow_db -min_version $std_fast_db",
                  "set_min_library $macro_slow_db -min_version $macro_fast_db",
                  "-max ssg0p9v125c", "-min ffg1p05vm40c", "read_sdc $mapped_sdc",
                  "-delay_type max", "-delay_type min", "$macro_count != 9",
                  "abs($clock_period - 3.000)", "setup_uncertainty_ns=0.200",
                  "hold_uncertainty_ns=0.050"):
        need(token in pt, "PrimeTime token absent: " + token)
    checks.append("independent_pt_slowmax_fastmin_3ns_uncertainty_nine_macro")

    prefixes = ("", "X", "prefix ", "Info:", "**", "[tool] ", "\t")
    kinds = ("Error", "error", "ERROR", "Fatal", "fatal", "FATAL")
    spaces = ("", " ", "  ", "\t")
    payloads = [prefix + kind + space + ": boom"
                for prefix, kind, space in itertools.product(prefixes, kinds, spaces)]
    diagnostic_prefixes = ("", "X ", "prefix ", "Info: ", "** ", "[tool] ", "\t")
    for prefix in diagnostic_prefixes:
        payloads.extend(prefix + item for item in (
            "LINK-1 failed", "link-999 failed", "unresolved reference",
            "UNRESOLVED module", "unable to resolve x", "timing loop found",
            "LOOP detected", "feedback loop detected", "(TIM-209)", "(OPT-150)"))
    payloads.extend(("X" + M.GUI_ALLOW, "prefix " + M.GUI_ALLOW,
                     M.GUI_ALLOW + " suffix", " " + M.GUI_ALLOW,
                     "\t" + M.GUI_ALLOW, M.GUI_ALLOW + " ", M.GUI_ALLOW + "\t"))
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "tool.log"
        for number, payload in enumerate(payloads):
            log.write_text(payload + "\n")
            try:
                M.scan_tool_log(log)
            except M.Failure:
                attacks.append("fatal_mutation_%03d" % number)
            else:
                raise RuntimeError("fatal mutation accepted: " + repr(payload))
        log.write_text(M.GUI_ALLOW + "\n" + M.GUI_ALLOW + "\n")
        try:
            M.scan_tool_log(log)
        except M.Failure:
            attacks.append("double_exact_allowlist")
        else:
            raise RuntimeError("double GUI allowlist accepted")
        log.write_text("normal\n" + M.GUI_ALLOW + "\nnormal\n")
        need(M.scan_tool_log(log) == 1, "single exact GUI line rejected")
        log.write_text("normal\n")
        need(M.scan_tool_log(log) == 0, "benign log rejected")
    checks.append("exact_gui_only_and_fatal_mutation_fuzz")

    source = RUNNER.read_text()
    main = source[source.index("def main()") :]
    ordered = ("verify_authority()", "verify_inputs()", "namespaces_fresh()",
               "collision_gate()",
               "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
               "collision_gate()", "resource_gate()", "namespaces_fresh()",
               'state["phase"] = "LICENSE_PREFLIGHT"',
               'state["phase"] = "ATTEMPT_CONSUME"', "ATTEMPT.mkdir()",
               'state["phase"] = "FORMALITY"', "run_tool([str(FM)",
               'state["phase"] = "PRIMETIME"', "run_tool([str(PT)")
    cursor = 0
    for token in ordered:
        position = main.find(token, cursor)
        need(position >= 0, "execution order token absent: " + token)
        cursor = position + len(token)
    run_tool = source[source.index("def run_tool(") : source.index("def read_machine")]
    need(run_tool.index("collision_gate()") < run_tool.index("subprocess.run("),
         "per-tool collision gate order")
    need(source.count('state["formality_runs"] += 1') == 1, "Formality budget")
    need(source.count('state["pt_runs"] += 1') == 1, "PT budget")
    need("dc_shell" not in main, "DC present in launch path")
    checks.append("shared_lock_collision_resource_license_attempt_and_tool_order")

    for path in (M.M1724, Path(str(M.M1724) + ".sha256"),
                 Path(str(M.M1724) + ".sha256.seal.sha256"), M.RESULT,
                 M.ATTEMPT, M.FAILURE, M.WORK, M.STAGE):
        need(not os.path.lexists(path), "release/execution namespace exists: " + str(path))
    checks.append("fresh_m1722_execution_and_m1724_release_namespaces")

    need(contract["author_execution"] == {
        "source_only": True, "license_queries": 0, "formality_runs": 0,
        "pt_runs": 0, "dc_runs": 0, "attempts_created": 0,
        "results_created": 0, "release_created": False,
        "quarantine_writes_or_moves": 0, "commit_or_push": False},
        "author execution boundary")
    need(all(value is False for value in M.CLAIMS.values()), "claim boundary")
    checks.append("source_only_and_all_claims_false")

    print(json.dumps({
        "status": "PASS_M1723_INDEPENDENT_SOURCE_HAMMER",
        "checks": checks,
        "mutation_attacks_rejected": len(attacks),
        "python_tests": {"cpython36": 12, "cpython312": 12},
        "p0": 0, "p1": 0, "p2": 0,
        "eda_runs": 0, "license_queries": 0,
        "attempt_created": False, "result_created": False,
        "release_created": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
