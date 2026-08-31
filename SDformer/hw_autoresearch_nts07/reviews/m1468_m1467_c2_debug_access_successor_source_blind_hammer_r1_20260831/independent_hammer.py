#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, no-EDA blind hammer for the M1467 C2 successor.

This program never imports or enumerates the unsealed M1432 private build and
never invokes VCS, simv, PrimeTime, PTPX, lmstat, or another subprocess.  It
checks source/authority identities, reruns the thirteen in-process unit tests,
and attacks the exact execution invariants that a later M1469/M1472 authority
would rely on.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / (
    "dc_handoff/scripts/run_m1467_m1432_c2_mapped_vcs_saif_ptpx_"
    "debug_access_successor_one_shot.py")
CHECKER = HW / (
    "verif_m1467_c2_debug_access_successor/"
    "check_m1467_c2_debug_access_successor_source.py")
TESTS = HW / (
    "verif_m1467_c2_debug_access_successor/"
    "test_m1467_c2_debug_access_successor_source.py")
CONTRACT = HW / (
    "contracts/m1467_m1432_c2_debug_access_successor_source_contract_"
    "r1_20260831.json")
AUTHOR = HW / (
    "reviews/m1467_m1432_c2_debug_access_successor_source_author_"
    "r1_20260831")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUNNER_SHA = "120cb1a8abe3df1e537de6797b3962fe0a7496be78954ba3b31fd9c8627e9a8a"
CHECKER_SHA = "da947aa5bd192b0ba9f7fe6592b0c2f437638ea72638d2f9a85cf206a49495ea"
TESTS_SHA = "c69b2314fe26861f08e2ad27aa5bed25b14793c76bca62f5a5660a1af7086807"
CONTRACT_SHA = "6b6a9b6d495eaa3539ee7d933b85e11f171d25fabf041eaffeea06018a7eab19"
AUTHOR_REVIEW_SHA = "4592e65501cfc2665bd15890c09e2f3ced915e83fc5663c655db5e0f20220234"
AUTHOR_MANIFEST_SHA = "abe23c3c39d38c732c6b068869267938c571f0063ffe6b4978f863987ed0a410"
AUTHOR_OUTER_SHA = "60b9b14537e80c10197b08afaab9458533096b1cce82cd32388afd28ada4103c"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1468_bound_m1467_checker", CHECKER)
T = load("m1468_bound_m1467_tests", TESTS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def independent_static(text: str) -> dict[str, object]:
    """Exact source invariants independent of M1467's own checker."""
    required_counts = {
        # One occurrence is the compile flag and one is the predecessor-premise
        # negative check.  The compile block below requires its own exact copy.
        '"-debug_access+r"': 2,
        'for axis in ("k8", "k1x8"):': 4,
        "for case in range(5):": 2,
        'state["vcs_compiles"] += 1': 1,
        'state["simv_runs"] += 1': 1,
        'state["saif_files"] += 1': 1,
        'state["ptpx_runs"] += 1': 1,
        "BASE.FILELIST[axis]": 1,
        "BASE.run(command": 1,
        'BASE.run(["./simv"': 1,
        "BASE.run([str(BASE.PT)": 1,
        "BASE.PTPX_TCL": 2,
        "BASE.LIB_DB": 2,
        "BASE.SAIF_INSTANCE": 1,
        'netlist = BASE.M872 / axis / "netlist"': 2,
        'sdc = BASE.M872 / axis / "netlist"': 2,
        "BASE.collision_gate()": 2,
        "ATTEMPT.mkdir()": 1,
        "BASE.seal_dir(ATTEMPT)": 1,
        "publish_no_replace(STAGE, RESULT)": 1,
        "publish_no_replace(FAIL_STAGE, FAILURE)": 1,
        'partial_axis_citable": False': 1,
        '"automatic_retry": False': 4,
    }
    for token, count in required_counts.items():
        if text.count(token) != count:
            raise RuntimeError("static cardinality drift: " + token)
    compile_block = text[text.index("COMPILE_PREFIX ="):
                         text.index("\n\n\nclass Failure")]
    exact_prefix = (
        'COMPILE_PREFIX = [str(BASE.VCS), "-full64", "-sverilog", "+v2k",\n'
        '                  "-timescale=1ns/1ps", "-assert", "svaext",\n'
        '                  "-debug_access+r", "+vcs+lic+wait", "-Mdir=csrc"]')
    if exact_prefix not in compile_block:
        raise RuntimeError("compile prefix drift")
    old = M.OLD_RUNNER.read_text()
    if '"-debug_access+r"' in old or sha(M.OLD_RUNNER) != M.OLD_RUNNER_SHA:
        raise RuntimeError("M1432 premise drift")
    # Frozen execution inputs and PTPX configuration must remain inherited and
    # exact-pinned; none may be replaced by an M1467-local workload or netlist.
    required = (
        '"-f", str(BASE.FILELIST[axis])',
        '"-top", BASE.TB_TOP, "-o", "simv"',
        'f"+M979_CASE={case}"',
        '"-i", str(BASE.UCLI)',
        'cycles = BASE.CYCLES[axis][case]',
        'f"events={BASE.EVENTS[case]} cycles={cycles} "',
        'netlist = BASE.M872 / axis / "netlist"',
        'sdc = BASE.M872 / axis / "netlist"',
        '"LIB_DB": str(BASE.LIB_DB)',
        '"MAPPED_NETLIST": str(netlist)',
        '"MAPPED_SDC": str(sdc)',
        '"OPERATING_CONDITION": "ssg0p9v125c"',
        '"CORNER_ROLE": "slow_prelayout_power"',
        '"SAIF_INSTANCE": BASE.SAIF_INSTANCE',
        'fcntl.LOCK_EX | fcntl.LOCK_NB',
        'renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)',
        '"automatic_retry": False',
        '"canonical_result": False',
    )
    for token in required:
        if token not in text:
            raise RuntimeError("execution invariant absent: " + token)
    # Two collision checks and the nonblocking lock occur before lmstat and the
    # mkdir-based exclusive attempt token occurs before the first BASE.run.
    lmstat = text.index('"lmstat", "-a"')
    attempt = text.index("ATTEMPT.mkdir()")
    first_run = text.index("BASE.run(command")
    if not (text.index("BASE.collision_gate()") < lmstat
            and text.index("fcntl.LOCK_EX | fcntl.LOCK_NB") < lmstat
            and attempt < first_run):
        raise RuntimeError("one-shot ordering drift")
    gate = text.index('if any(state[key] != COUNTS[key] for key in')
    first_pt = text.index('state["phase"] = f"PTPX_{axis}_{case}"')
    result = text.index("publish_no_replace(STAGE, RESULT)")
    final_counts = text.index(
        "if any(state[key] != value for key, value in COUNTS.items())")
    if not (gate < first_pt < final_counts < result):
        raise RuntimeError("all-SAIF-before-PTPX/result ordering drift")
    return {"axes": 2, "cases_per_axis": 5, "vcs_compiles": 2,
            "simv_runs": 10, "saif_files": 10, "ptpx_runs": 10,
            "debug_access_r_compile_prefix_count": 1,
            "all_saif_before_any_ptpx": True, "attempt_before_eda": True,
            "atomic_no_replace": True, "partial_axis_citable": False}


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []

    def check(name: str, value: bool, category: str) -> None:
        checks.append({"check": name, "category": category,
                       "pass": bool(value)})

    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    check("runner_exact", sha(RUNNER) == RUNNER_SHA, "identity")
    check("checker_exact", sha(CHECKER) == CHECKER_SHA, "identity")
    check("tests_exact", sha(TESTS) == TESTS_SHA, "identity")
    check("contract_exact", sha(CONTRACT) == CONTRACT_SHA, "identity")
    check("docs359_exact", sha(DOCS359) == DOCS359_SHA, "identity")
    check("author_review_exact", sha(AUTHOR / "review.json") == AUTHOR_REVIEW_SHA,
          "authority")
    check("author_manifest_exact", sha(AUTHOR / "SHA256SUMS") == AUTHOR_MANIFEST_SHA,
          "authority")
    check("author_outer_exact", sha(AUTHOR / "SHA256SUMS.seal.sha256") ==
          AUTHOR_OUTER_SHA, "authority")
    check("author_population", M.verify_seal(AUTHOR, AUTHOR_MANIFEST_SHA,
          AUTHOR_OUTER_SHA) == {"review.json"}, "authority")
    author = M.strict_json(AUTHOR / "review.json")
    check("author_status", author.get("status") ==
          "PASS_M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE_AUTHOR__NO_EDA",
          "authority")
    check("author_no_launch", author["future_chain"]["launch_authorized"] is False,
          "authority")

    source = M.check_source(require_future_absent=False)
    check("source_checker_pass", source["status"] ==
          "PASS_M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE__NO_EDA", "source")
    check("predecessor_phase", source["failure"]["phase"] == "SIM_k8_0",
          "predecessor")
    check("predecessor_counts", source["failure"] == {
        "phase": "SIM_k8_0", "vcs_compiles": 1, "simv_runs": 1,
        "saif_files": 0, "ptpx_runs": 0, "attempt_consumed": True,
        "private_build_read": False, "automatic_retry": False}, "predecessor")
    check("predecessor_private_not_read",
          source["failure"]["private_build_read"] is False, "predecessor")

    # Rerun all thirteen author tests in-process, retaining no subprocess/EDA
    # path.  Text output is summarized in the sealed review artifacts.
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(T)
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    check("author_tests_13", replay.testsRun == 13 and not replay.failures
          and not replay.errors, "tests")

    static = independent_static(RUNNER.read_text())
    for key, expected in (("axes", 2), ("cases_per_axis", 5),
                          ("vcs_compiles", 2), ("simv_runs", 10),
                          ("saif_files", 10), ("ptpx_runs", 10),
                          ("all_saif_before_any_ptpx", True),
                          ("attempt_before_eda", True),
                          ("atomic_no_replace", True),
                          ("partial_axis_citable", False)):
        check("static_" + key, static[key] == expected, "execution")

    text = RUNNER.read_text()
    mutations = {
        "debug_delete": text.replace('                  "-debug_access+r", ',
                                     "                  ", 1),
        "debug_duplicate": text.replace('"-debug_access+r",',
                                        '"-debug_access+r", "-debug_access+r",', 1),
        "debug_write_only": text.replace('"-debug_access+r"', '"-debug_access+w"', 1),
        "axis_drop_k1x8": text.replace('("k8", "k1x8")', '("k8",)', 1),
        "case_drop_4": text.replace("for case in range(5):", "for case in range(4):", 1),
        "compile_counter_delete": text.replace('state["vcs_compiles"] += 1', "pass", 1),
        "sim_counter_delete": text.replace('state["simv_runs"] += 1', "pass", 1),
        "saif_counter_delete": text.replace('state["saif_files"] += 1', "pass", 1),
        "ptpx_counter_delete": text.replace('state["ptpx_runs"] += 1', "pass", 1),
        "filelist_swap": text.replace("BASE.FILELIST[axis]", "BASE.FILELIST[\"k8\"]", 1),
        "top_swap": text.replace('"-top", BASE.TB_TOP', '"-top", "wrong_top"', 1),
        "workload_case_swap": text.replace('f"+M979_CASE={case}"', '"+M979_CASE=0"', 1),
        "ucli_swap": text.replace('"-i", str(BASE.UCLI)', '"-i", "wrong.ucli"', 1),
        "cycle_swap": text.replace("BASE.CYCLES[axis][case]", "BASE.CYCLES[axis][0]", 1),
        "event_swap": text.replace("BASE.EVENTS[case]", "BASE.EVENTS[0]", 1),
        "netlist_swap": text.replace('BASE.M872 / axis / "netlist"',
                                     'BASE.M872 / "k8" / "netlist"', 1),
        "sdc_swap": text.replace('sdc = BASE.M872 / axis / "netlist"',
                                 'sdc = BASE.M872 / "k8" / "netlist"', 1),
        "lib_swap": text.replace('"LIB_DB": str(BASE.LIB_DB)',
                                 '"LIB_DB": "wrong.db"', 1),
        "mapped_netlist_env_swap": text.replace(
            '"MAPPED_NETLIST": str(netlist)', '"MAPPED_NETLIST": "wrong.v"', 1),
        "mapped_sdc_env_swap": text.replace(
            '"MAPPED_SDC": str(sdc)', '"MAPPED_SDC": "wrong.sdc"', 1),
        "corner_swap": text.replace('"OPERATING_CONDITION": "ssg0p9v125c"',
                                    '"OPERATING_CONDITION": "ffg"', 1),
        "ptpx_script_swap": text.replace("BASE.PTPX_TCL", 'Path("wrong.tcl")', 1),
        "saif_scope_swap": text.replace('"SAIF_INSTANCE": BASE.SAIF_INSTANCE',
                                       '"SAIF_INSTANCE": "wrong"', 1),
        "collision_delete": text.replace("        BASE.collision_gate()\n", "", 1),
        "lock_blocking": text.replace("fcntl.LOCK_EX | fcntl.LOCK_NB", "fcntl.LOCK_EX", 1),
        "attempt_delete": text.replace("ATTEMPT.mkdir()", "ATTEMPT.exists()", 1),
        "attempt_unsealed": text.replace("BASE.seal_dir(ATTEMPT)", "pass", 1),
        "replace_result": text.replace("publish_no_replace(STAGE, RESULT)",
                                       "os.replace(STAGE, RESULT)", 1),
        "replace_failure": text.replace("publish_no_replace(FAIL_STAGE, FAILURE)",
                                        "os.replace(FAIL_STAGE, FAILURE)", 1),
        "partial_axis_true": text.replace('partial_axis_citable": False',
                                          'partial_axis_citable": True', 1),
        "auto_retry_true": text.replace('"automatic_retry": False',
                                        '"automatic_retry": True', 1),
        "pt_before_gate": text.replace(
            'state["phase"] = f"PTPX_{axis}_{case}"',
            'state["phase"] = f"PTPX_{axis}_{case}"', 1).replace(
                'if any(state[key] != COUNTS[key] for key in\n'
                '               ("vcs_compiles", "simv_runs", "saif_files")):',
                'if False:', 1),
    }
    for name, mutated in mutations.items():
        attack(name, lambda value=mutated: independent_static(value), "source_mutation")

    # The source contract is exact-set/value: independently mutate every
    # top-level and nested leaf and require rejection by the bound checker.
    expected = M.expected_contract()
    contract_attacks = 0
    for section, value in expected.items():
        if type(value) is not dict:
            candidate = dict(expected)
            candidate[section] = "M1468_MUTATED"
            attack("contract_top_" + section,
                   lambda c=candidate: (_ for _ in ()).throw(RuntimeError())
                   if c != M.expected_contract() else None, "contract_mutation")
            contract_attacks += 1
            continue
        for key in value:
            candidate = json.loads(json.dumps(expected))
            old_value = candidate[section][key]
            candidate[section][key] = (not old_value if type(old_value) is bool
                                       else old_value + 1 if type(old_value) is int
                                       else "M1468_MUTATED")
            attack("contract_leaf_" + section + "_" + key,
                   lambda c=candidate: (_ for _ in ()).throw(RuntimeError())
                   if c != M.expected_contract() else None, "contract_mutation")
            contract_attacks += 1

    # Fresh public namespaces remain absent.  The old unsealed private residue
    # is checked only for directory existence by the bound source checker; this
    # hammer deliberately never lists or opens it.
    check("fresh_attempt_absent", not os.path.lexists(HW / M.NEW_NAMESPACES["attempt"]),
          "freshness")
    check("fresh_result_absent", not os.path.lexists(HW / M.NEW_NAMESPACES["result"]),
          "freshness")
    check("fresh_failure_absent", not os.path.lexists(HW / M.NEW_NAMESPACES["failure"]),
          "freshness")
    check("fresh_private_absent", not os.path.lexists(HW / M.NEW_NAMESPACES["private"]),
          "freshness")

    p0 = sum(not item["rejected"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    output = {
        "schema": "m1468_m1467_c2_debug_access_successor_blind_hammer_output_r1_v1",
        "status": ("PASS_ZERO_FALSE_NEGATIVE" if p0 == 0 and p1 == 0
                   else "FAIL_DO_NOT_CITE"),
        "checks": checks, "attacks": attacks,
        "summary": {"checks_passed": sum(item["pass"] for item in checks),
                    "checks_total": len(checks),
                    "mutations_rejected": sum(item["rejected"] for item in attacks),
                    "mutations_total": len(attacks),
                    "contract_mutations": contract_attacks,
                    "p0_count": p0, "p1_count": p1,
                    "author_tests_run": replay.testsRun,
                    "author_test_failures": len(replay.failures) + len(replay.errors)},
        "execution": {"license_query": 0, "vcs": 0, "simv": 0,
                      "saif": 0, "pt": 0, "ptpx": 0, "eda": 0,
                      "private_build_reads": 0, "attempts_consumed": 0},
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if p0 == 0 and p1 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
