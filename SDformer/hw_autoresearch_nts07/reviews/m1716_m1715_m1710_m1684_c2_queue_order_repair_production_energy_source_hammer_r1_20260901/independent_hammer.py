#!/usr/bin/env python3
"""Different-author, source-only hammer for M1715. Never launches EDA."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / (
    "system_simulator/scripts/check_m1715_m1710_m1684_c2_"
    "queue_order_repair_production_energy_source.py")
RUNNER = HW / (
    "dc_handoff/scripts/run_m1715_m1710_m1684_m1661_c2_"
    "queue_order_repair_production_energy_one_shot.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load("m1716_independent_checker", CHECKER)
R = load("m1716_independent_runner", RUNNER)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect_failure(call, label, attacks):
    try:
        call()
    except Exception:
        attacks.append(label)
        return
    raise RuntimeError("attack accepted: " + label)


def main():
    checks = []
    attacks = []

    # The author checker expects the future hammer directory to be absent.
    # Redirect only that one absence probe now that this review directory exists.
    original_m1716 = C.M1716
    C.M1716 = HERE.parent / ".m1716_absence_probe"
    try:
        source = C.validate_sources()
    finally:
        C.M1716 = original_m1716
    need(source["status"] == "PASS_M1715_SOURCE_ONLY_NO_EDA",
         "author source checker")
    need(source["runtime_bound_execution_sources"] == 6,
         "six-source source check")
    checks.append("author_source_checker_reproduced")

    failed_checker = C.verify_m1710_failure()
    R.verify_m1710_pre_attempt_failure()
    need(failed_checker == {
        "attempt_consumed": False,
        "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0,
                   "simv_runs": 0, "vcs_compiles": 0},
        "error": "Failure",
        "partial_axis_citable": False,
        "phase": "SOURCE_CHAIN",
        "status": "FAILED_OR_INCOMPLETE",
    }, "M1710 sealed failure semantics")
    checks.append("m1710_failure_double_seal_semantics_and_no_retry")

    # The M1715 source must be inert without a future exact release.
    expect_failure(R.verify_authority, "future_authority_absent", attacks)

    text = RUNNER.read_text()
    main_text = text[text.index("def main("):]
    blocking = "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)"
    lock = main_text.index(blocking)
    first_collision = main_text.index("collision_gate()", lock)
    first_rebind = main_text.index("runtime_bind_execution_sources()",
                                   first_collision)
    first_lexists = main_text.index("forbidden_release_namespaces_absent()",
                                    first_rebind)
    second_collision = main_text.index("collision_gate()", first_collision + 1)
    second_rebind = main_text.index("runtime_bind_execution_sources()",
                                    second_collision)
    second_lexists = main_text.index("forbidden_release_namespaces_absent()",
                                     second_rebind)
    attempt = main_text.index("ATTEMPT.mkdir()")
    need("LOCK_NB" not in text, "LOCK_NB present")
    need("collision_gate()" not in main_text[:lock],
         "pre-lock collision present")
    need(lock < first_collision < first_rebind < first_lexists <
         second_collision < second_rebind < second_lexists < attempt,
         "lock/collision/rebind/lexists/attempt order")
    run_text = text[text.index("def run("):text.index("def result_identity")]
    need(run_text.index("collision_gate()") < run_text.index("subprocess.run("),
         "per-tool collision rescan order")
    checks.append("blocking_lock_and_two_postlock_runtime_gates_ordered")

    # Mutation hammer for the queue order and all four execution budgets.
    counts_literal = ('COUNTS = {"vcs_compiles": 2, "simv_runs": 10,\n'
                      '          "saif_files": 10, "ptpx_runs": 10}')

    def validate_candidate(changed):
        C.validate_queue_source(changed)
        need(counts_literal in changed, "execution budget mutation")

    mutations = {
        "lock_nb": text.replace(blocking, blocking + " | fcntl.LOCK_NB", 1),
        "prelock_collision": text.replace(
            blocking, "collision_gate()\n        " + blocking, 1),
        "remove_first_collision": text.replace(
            'state["phase"] = "POST_LOCK_COLLISION"\n        collision_gate()\n',
            'state["phase"] = "POST_LOCK_COLLISION"\n', 1),
        "remove_first_rebind": text.replace(
            "runtime_bind_execution_sources()\n        forbidden_release_namespaces_absent()",
            "forbidden_release_namespaces_absent()", 1),
        "remove_first_lexists": text.replace(
            "forbidden_release_namespaces_absent()\n        resource_gate()",
            "resource_gate()", 1),
        "budget_vcs": text.replace(counts_literal,
                                   counts_literal.replace(
                                       '"vcs_compiles": 2',
                                       '"vcs_compiles": 3'), 1),
        "budget_simv": text.replace(counts_literal,
                                    counts_literal.replace(
                                        '"simv_runs": 10',
                                        '"simv_runs": 11'), 1),
        "budget_saif": text.replace(counts_literal,
                                    counts_literal.replace(
                                        '"saif_files": 10',
                                        '"saif_files": 11'), 1),
        "budget_ptpx": text.replace(counts_literal,
                                    counts_literal.replace(
                                        '"ptpx_runs": 10',
                                        '"ptpx_runs": 11'), 1),
    }
    for label, changed in mutations.items():
        expect_failure(lambda changed=changed: validate_candidate(changed),
                       label, attacks)
    checks.append("queue_order_and_budget_mutations_rejected")

    # Exercise the launch-capable six-source binder itself. Only the checking
    # functions are wrapped; no subprocess, license query or EDA path is called.
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
    expected = {Path(R.HW / rel).resolve()
                for rel in R.DIRECT_EXECUTION_PATHS}
    need(len(expected) == 6 and set(exact_seen) == expected,
         "six-source exact-SHA coverage")
    need(set(force_seen) == expected, "six-source active-force coverage")
    checks.append("runtime_exact_and_force_scans_exercised_six_of_six")

    # All six forbidden predecessor release namespaces reject both regular
    # files and dangling links because the gate intentionally uses lexists.
    old_1686, old_1700 = R.M1686_FORBIDDEN, R.M1700_FORBIDDEN
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            R.M1686_FORBIDDEN = root / "m1686.json"
            R.M1700_FORBIDDEN = root / "m1700.json"
            for name, base in (("m1686", R.M1686_FORBIDDEN),
                               ("m1700", R.M1700_FORBIDDEN)):
                for suffix in ("", ".sha256", ".sha256.seal.sha256"):
                    path = Path(str(base) + suffix)
                    path.write_text("forbidden\n")
                    expect_failure(R.forbidden_release_namespaces_absent,
                                   name + suffix + "_regular", attacks)
                    path.unlink()
                    os.symlink(root / "missing-target", path)
                    expect_failure(R.forbidden_release_namespaces_absent,
                                   name + suffix + "_dangling", attacks)
                    path.unlink()
            R.forbidden_release_namespaces_absent()
    finally:
        R.M1686_FORBIDDEN, R.M1700_FORBIDDEN = old_1686, old_1700
    checks.append("forbidden_release_lexists_regular_and_dangling_rejected")

    # Repeat the same regular/dangling-link attack against all three M1710
    # residue namespaces; no predecessor attempt may ever be revived.
    old_retry = R.M1710_ATTEMPT, R.M1710_RESULT, R.M1710_PRIVATE
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            R.M1710_ATTEMPT = root / "attempt"
            R.M1710_RESULT = root / "result"
            R.M1710_PRIVATE = root / "private"
            for name, path in (("attempt", R.M1710_ATTEMPT),
                               ("result", R.M1710_RESULT),
                               ("private", R.M1710_PRIVATE)):
                path.write_text("forbidden\n")
                expect_failure(R.verify_m1710_pre_attempt_failure,
                               "m1710_" + name + "_regular", attacks)
                path.unlink()
                os.symlink(root / "missing-target", path)
                expect_failure(R.verify_m1710_pre_attempt_failure,
                               "m1710_" + name + "_dangling", attacks)
                path.unlink()
            R.verify_m1710_pre_attempt_failure()
    finally:
        R.M1710_ATTEMPT, R.M1710_RESULT, R.M1710_PRIVATE = old_retry
    checks.append("m1710_retry_residue_regular_and_dangling_rejected")

    # Carry forward the already disclosed scanner limitation. Exact SHA is the
    # authority, so this remains non-blocking for this frozen six-source run.
    quoted_detected = C.active_force_present(
        Path("quoted.tcl"), 'set x "[force dut/q 0]"\n')

    need(R.COUNTS == {"vcs_compiles": 2, "simv_runs": 10,
                      "saif_files": 10, "ptpx_runs": 10},
         "execution budget drift")
    for path in (R.M1717, Path(str(R.M1717) + ".sha256"),
                 Path(str(R.M1717) + ".sha256.seal.sha256"),
                 R.ATTEMPT, R.RESULT, R.FAILURE, R.PRIVATE):
        need(not os.path.lexists(path),
             "release/execution namespace exists: " + str(path))
    checks.append("no_m1717_attempt_result_or_eda")

    print(json.dumps({
        "status": "PASS_M1716_INDEPENDENT_SOURCE_HAMMER",
        "checks": checks,
        "mutation_attacks_rejected": len(attacks),
        "mutation_attack_labels": attacks,
        "runtime_exact_sources": len(set(exact_seen)),
        "runtime_force_scanned_sources": len(set(force_seen)),
        "m1710_attempt_consumed": False,
        "m1710_execution_counts": failed_checker["counts"],
        "future_budget": R.COUNTS,
        "quoted_tcl_command_substitution_detected": quoted_detected,
        "p0": 0,
        "p1": 0,
        "p2": 1 if not quoted_detected else 0,
        "license_queries": 0,
        "eda_runs": 0,
        "release_created": False,
        "attempt_created": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
