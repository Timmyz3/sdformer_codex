#!/usr/bin/env python3
"""Different-author, source-only hammer for M1710. Never launches EDA."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1710_m1684_m1661_c2_runtime_bound_shared_eda_queue_production_energy_one_shot.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load("m1710_independent_checker", CHECKER)
R = load("m1710_independent_runner", RUNNER)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect_failure(call, label):
    try:
        call()
    except Exception:
        return
    raise RuntimeError("attack accepted: " + label)


def main():
    checks = []
    attacks = []

    # The author checker quite properly requires the future review namespace
    # to be absent.  During the review it is necessarily our current directory,
    # so redirect only that absence probe to a definitely absent sibling while
    # leaving every reviewed source and predecessor path untouched.
    original_checker_m1711 = C.M1711
    C.M1711 = HERE.parent / ".m1711_absence_probe"
    try:
        source = C.validate_sources()
    finally:
        C.M1711 = original_checker_m1711
    need(source["status"] == "PASS_M1710_SOURCE_ONLY_NO_EDA", "source check")
    need(source["runtime_bound_execution_sources"] == 6, "six-source claim")
    checks.append("author_source_checker_reproduced")

    # Exercise the launch-capable runtime binder itself, while replacing only
    # the validators with recording wrappers. No subprocess or tool is called.
    exact_seen = []
    force_seen = []
    original_exact = R.exact
    original_force = R.CHECK.active_force_present
    try:
        def record_exact(path, digest):
            original_exact(path, digest)
            exact_seen.append(Path(path).resolve())

        def record_force(path, text=None):
            result = original_force(path, text)
            force_seen.append(Path(path).resolve())
            return result

        R.exact = record_exact
        R.CHECK.active_force_present = record_force
        R.runtime_bind_execution_sources()
    finally:
        R.exact = original_exact
        R.CHECK.active_force_present = original_force
    expected = {Path(R.HW / rel).resolve() for rel in R.DIRECT_EXECUTION_PATHS}
    need(expected.issubset(set(exact_seen)), "runtime exact-SHA coverage")
    need(expected == set(force_seen), "runtime active-force coverage")
    need(len(expected) == 6, "runtime inventory cardinality")
    checks.append("launch_capable_runtime_binder_exercised_six_of_six")

    # M1686 and M1700 payload/digest/outer names must reject regular files and
    # dangling symlinks. Patch only the namespace roots into a temporary tree.
    old_1686, old_1700 = R.M1686_FORBIDDEN, R.M1700_FORBIDDEN
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            R.M1686_FORBIDDEN = root / "m1686.json"
            R.M1700_FORBIDDEN = root / "m1700.json"
            for base_name, base in (("m1686", R.M1686_FORBIDDEN),
                                    ("m1700", R.M1700_FORBIDDEN)):
                for suffix in ("", ".sha256", ".sha256.seal.sha256"):
                    path = Path(str(base) + suffix)
                    path.write_text("forbidden\n")
                    expect_failure(R.forbidden_release_namespaces_absent,
                                   base_name + suffix + "_regular")
                    attacks.append(base_name + suffix + "_regular")
                    path.unlink()
                    os.symlink(root / "missing-target", path)
                    expect_failure(R.forbidden_release_namespaces_absent,
                                   base_name + suffix + "_dangling_symlink")
                    attacks.append(base_name + suffix + "_dangling_symlink")
                    path.unlink()
            R.forbidden_release_namespaces_absent()
    finally:
        R.M1686_FORBIDDEN, R.M1700_FORBIDDEN = old_1686, old_1700
    checks.append("lexists_regular_and_dangling_namespaces_rejected")

    # Re-run the originally requested Tcl attacks. A quoted Tcl command
    # substitution is recorded separately as a non-blocking scanner limitation:
    # the exact-SHA runtime binder remains the enforcing security boundary.
    tcl_attacks = {
        "brace_inline": "if {1} { force dut/q 0 }\n",
        "semicolon_inline": "run; force dut/q 0\n",
        "nested_bracket": "puts [if {1} {force dut/q 0}]\n",
    }
    for label, text in tcl_attacks.items():
        need(C.active_force_present(Path("attack.tcl"), text),
             "Tcl attack missed: " + label)
        attacks.append("tcl_" + label)
    need(not C.active_force_present(Path("clean.tcl"),
                                    '# force ignored\nset x "force ignored"\nrun\n'),
         "Tcl clean fixture false positive")
    quoted_substitution_detected = C.active_force_present(
        Path("quoted.tcl"), 'set x "[force dut/q 0]"\n')
    checks.append("requested_inline_semicolon_brace_tcl_attacks_rejected")

    text = RUNNER.read_text()
    main_text = text[text.index("def main()") :]
    ordered_tokens = (
        "verify_authority()",
        "verify_predecessors_and_inputs()",
        "namespaces_fresh()",
        "collision_gate()",
        "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "collision_gate()",
        "resource_gate()",
        "namespaces_fresh()",
        'state["phase"] = "LICENSE_PREFLIGHT"',
        'state["phase"] = "RUNTIME_REBIND"',
        "runtime_bind_execution_sources()",
        "forbidden_release_namespaces_absent()",
        'state["phase"] = "ATTEMPT_CONSUME"',
        "ATTEMPT.mkdir()",
        'for axis in ("k8", "k1x8"):',
    )
    cursor = 0
    for token in ordered_tokens:
        position = main_text.find(token, cursor)
        need(position >= 0, "ordered token absent: " + token)
        cursor = position + len(token)
    run_text = text[text.index("def run(") : text.index("def result_identity")]
    need(run_text.index("collision_gate()") < run_text.index("subprocess.run("),
         "per-tool collision order")
    need('automatic_retry": False' in text, "no-retry receipt")
    need(text.index("ATTEMPT =") < text.index("RESULT ="), "fresh namespaces")
    checks.append("shared_lock_runtime_rebind_attempt_and_tool_order")

    # Source-only authority must remain absent and no campaign namespace may
    # have been created by this hammer.
    for path in (R.M1711, R.M1712, Path(str(R.M1712) + ".sha256"),
                 Path(str(R.M1712) + ".sha256.seal.sha256"),
                 R.ATTEMPT, R.RESULT, R.FAILURE, R.PRIVATE):
        # M1711 is this review directory, so only its future authority files are
        # prohibited; the directory itself necessarily exists while reviewing.
        if path == R.M1711:
            continue
        need(not os.path.lexists(path), "execution/release namespace exists: " + str(path))
    checks.append("no_release_attempt_result_or_eda")

    print(json.dumps({
        "status": "PASS_M1711_INDEPENDENT_SOURCE_HAMMER_WITH_ONE_NONBLOCKING_TCL_SCANNER_LIMITATION",
        "checks": checks,
        "mutation_attacks_rejected": len(attacks),
        "mutation_attack_labels": attacks,
        "runtime_exact_sources": len(expected),
        "runtime_force_scanned_sources": len(set(force_seen)),
        "quoted_tcl_command_substitution_detected": quoted_substitution_detected,
        "p0": 0,
        "p1": 0,
        "p2": 1 if not quoted_substitution_detected else 0,
        "eda_runs": 0,
        "release_created": False,
        "attempt_created": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
