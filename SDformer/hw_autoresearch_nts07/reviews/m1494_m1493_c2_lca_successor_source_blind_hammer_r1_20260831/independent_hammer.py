#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, no-EDA blind hammer for the M1493 C2 successor."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / (
    "dc_handoff/scripts/run_m1493_m1467_c2_mapped_vcs_saif_ptpx_"
    "lca_successor_one_shot.py")
CHECKER = HW / (
    "verif_m1493_c2_lca_successor/"
    "check_m1493_c2_lca_successor_source.py")
TESTS = HW / (
    "verif_m1493_c2_lca_successor/"
    "test_m1493_c2_lca_successor_source.py")
CONTRACT = HW / (
    "contracts/m1493_m1467_c2_lca_successor_source_contract_"
    "r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUNNER_SHA = "8d93d55ca600620eb903a7328f4cc38e0720ae45ce24d8128fac5924d2902677"
CHECKER_SHA = "747313aae818407ce134fb4f10b561a9cbf7d20e70025b44237429c8dca8b32c"
TESTS_SHA = "a2526e19464ac7d30fecd63cb1153429fa1c065bc96107ebce9207a9ea92dcb0"
CONTRACT_SHA = "efa9e6339564f2ec3c8294b7977c81782fafe6ac38f6e4fed5e61c89642da177"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1494_bound_m1493_checker", CHECKER)
T = load("m1494_bound_m1493_tests", TESTS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def independent_static(text: str) -> dict[str, object]:
    """Check execution invariants without trusting M1493's own checker."""
    counts = {
        '"-debug_access+r"': 2,
        '"-lca"': 2,
        'for axis in ("k8", "k1x8"):': 4,
        "for case in range(5):": 2,
        'state["vcs_compiles"] += 1': 1,
        'state["simv_runs"] += 1': 1,
        'state["saif_files"] += 1': 1,
        'state["ptpx_runs"] += 1': 1,
        "BASE.BASE.run(command": 1,
        'BASE.BASE.run(["./simv"': 1,
        "BASE.BASE.run([str(BASE.BASE.PT)": 1,
        "BASE.BASE.collision_gate()": 2,
        "ATTEMPT.mkdir()": 1,
        "BASE.BASE.seal_dir(ATTEMPT)": 1,
        "publish_no_replace(STAGE, RESULT)": 1,
        "publish_no_replace(FAIL_STAGE, FAILURE)": 1,
        'partial_axis_citable": False': 1,
        '"automatic_retry": False': 4,
    }
    for token, expected in counts.items():
        if text.count(token) != expected:
            raise RuntimeError("static cardinality drift: " + token)
    prefix = text[text.index("COMPILE_PREFIX ="):
                  text.index("\n\n\nclass Failure")]
    expected_prefix = (
        'COMPILE_PREFIX = [str(BASE.BASE.VCS), "-full64", "-sverilog", "+v2k",\n'
        '                  "-timescale=1ns/1ps", "-assert", "svaext",\n'
        '                  "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc"]')
    if expected_prefix not in prefix:
        raise RuntimeError("compile prefix drift")
    predecessor = M.OLD_RUNNER.read_text()
    predecessor_prefix = predecessor[
        predecessor.index("COMPILE_PREFIX ="):
        predecessor.index("\n\n\nclass Failure")]
    if (predecessor_prefix.count('"-debug_access+r"') != 1
            or '"-lca"' in predecessor_prefix
            or sha(M.OLD_RUNNER) != M.OLD_RUNNER_SHA):
        raise RuntimeError("M1467 premise drift")
    required = (
        '"-f", str(BASE.BASE.FILELIST[axis])',
        '"-top", BASE.BASE.TB_TOP, "-o", "simv"',
        'f"+M979_CASE={case}"',
        '"-i", str(BASE.BASE.UCLI)',
        "BASE.BASE.CYCLES[axis][case]",
        "BASE.BASE.EVENTS[case]",
        'netlist = BASE.BASE.M872 / axis / "netlist"',
        'sdc = BASE.BASE.M872 / axis / "netlist"',
        '"LIB_DB": str(BASE.BASE.LIB_DB)',
        '"MAPPED_NETLIST": str(netlist)',
        '"MAPPED_SDC": str(sdc)',
        '"OPERATING_CONDITION": "ssg0p9v125c"',
        '"CORNER_ROLE": "slow_prelayout_power"',
        '"SAIF_INSTANCE": BASE.BASE.SAIF_INSTANCE',
        "fcntl.LOCK_EX | fcntl.LOCK_NB",
        "renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)",
        '"canonical_result": False',
    )
    for token in required:
        if token not in text:
            raise RuntimeError("execution invariant absent: " + token)
    lmstat = text.index('"lmstat", "-a"')
    attempt = text.index("ATTEMPT.mkdir()")
    first_eda = text.index("BASE.BASE.run(command")
    if not (text.index("BASE.BASE.collision_gate()") < lmstat
            and text.index("fcntl.LOCK_EX | fcntl.LOCK_NB") < lmstat
            and attempt < first_eda):
        raise RuntimeError("attempt/collision ordering drift")
    all_saif_gate = text.index(
        'if any(state[key] != COUNTS[key] for key in\n'
        '               ("vcs_compiles", "simv_runs", "saif_files")):')
    first_pt = text.index('state["phase"] = f"PTPX_{axis}_{case}"')
    final_gate = text.index(
        "if any(state[key] != value for key, value in COUNTS.items())")
    publish = text.index("publish_no_replace(STAGE, RESULT)")
    if not all_saif_gate < first_pt < final_gate < publish:
        raise RuntimeError("all-SAIF/PTPX/publication ordering drift")
    return {"axes": 2, "cases_per_axis": 5, "vcs_compiles": 2,
            "simv_runs": 10, "saif_files": 10, "ptpx_runs": 10,
            "attempt_before_eda": True, "all_saif_before_ptpx": True,
            "atomic_publication": True}


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1494_MUTATED"
    if type(value) is list:
        return list(value) + ["M1494_MUTATED"]
    if type(value) is dict:
        result = copy.deepcopy(value)
        result["m1494_mutated"] = True
        return result
    raise TypeError(type(value))


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []

    def check(name: str, value: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(value)})

    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    check("runner_exact", sha(RUNNER) == RUNNER_SHA, "identity")
    check("checker_exact", sha(CHECKER) == CHECKER_SHA, "identity")
    check("tests_exact", sha(TESTS) == TESTS_SHA, "identity")
    check("contract_exact", sha(CONTRACT) == CONTRACT_SHA, "identity")
    check("docs359_exact", sha(DOCS359) == DOCS359_SHA, "identity")
    source = M.check_source(require_future_absent=False)
    check("native_source_check", source.get("status") ==
          "PASS_M1493_C2_LCA_SUCCESSOR_SOURCE__NO_EDA", "source")
    predecessor = M.check_predecessor()
    check("predecessor_exact", predecessor == {
        "phase": "SIM_k8_0", "vcs_compiles": 1, "simv_runs": 1,
        "saif_files": 0, "ptpx_runs": 0, "required_option": "-lca",
        "attempt_consumed": True, "automatic_retry": False}, "predecessor")

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(T)
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    check("author_tests_14", replay.testsRun == 14 and not replay.failures
          and not replay.errors, "tests")
    static = independent_static(RUNNER.read_text())
    for key, expected in (("axes", 2), ("cases_per_axis", 5),
                          ("vcs_compiles", 2), ("simv_runs", 10),
                          ("saif_files", 10), ("ptpx_runs", 10),
                          ("attempt_before_eda", True),
                          ("all_saif_before_ptpx", True),
                          ("atomic_publication", True)):
        check("static_" + key, static[key] == expected, "execution")

    text = RUNNER.read_text()
    mutations = {
        "lca_delete": text.replace('"-lca", ', "", 1),
        "lca_duplicate": text.replace('"-lca",', '"-lca", "-lca",', 1),
        "lca_wrong": text.replace('"-lca"', '"-lca_wrong"', 1),
        "debug_delete": text.replace('"-debug_access+r", ', "", 1),
        "debug_write_only": text.replace('"-debug_access+r"', '"-debug_access+w"', 1),
        "axis_drop": text.replace('(\"k8\", \"k1x8\")', '(\"k8\",)', 1),
        "case_drop": text.replace("for case in range(5):", "for case in range(4):", 1),
        "compile_count_drop": text.replace('state["vcs_compiles"] += 1', "pass", 1),
        "sim_count_drop": text.replace('state["simv_runs"] += 1', "pass", 1),
        "saif_count_drop": text.replace('state["saif_files"] += 1', "pass", 1),
        "ptpx_count_drop": text.replace('state["ptpx_runs"] += 1', "pass", 1),
        "filelist_swap": text.replace("BASE.BASE.FILELIST[axis]", 'BASE.BASE.FILELIST["k8"]', 1),
        "top_swap": text.replace('"-top", BASE.BASE.TB_TOP', '"-top", "wrong"', 1),
        "case_swap": text.replace('f"+M979_CASE={case}"', '"+M979_CASE=0"', 1),
        "ucli_swap": text.replace('"-i", str(BASE.BASE.UCLI)', '"-i", "wrong"', 1),
        "cycle_swap": text.replace("BASE.BASE.CYCLES[axis][case]", "BASE.BASE.CYCLES[axis][0]", 1),
        "event_swap": text.replace("BASE.BASE.EVENTS[case]", "BASE.BASE.EVENTS[0]", 1),
        "netlist_swap": text.replace('BASE.BASE.M872 / axis / "netlist"', 'BASE.BASE.M872 / "k8" / "netlist"'),
        "sdc_swap": text.replace('sdc = BASE.BASE.M872 / axis / "netlist"', 'sdc = BASE.BASE.M872 / "k8" / "netlist"'),
        "lib_swap": text.replace('"LIB_DB": str(BASE.BASE.LIB_DB)', '"LIB_DB": "wrong"', 1),
        "mapped_v_swap": text.replace('"MAPPED_NETLIST": str(netlist)', '"MAPPED_NETLIST": "wrong"', 1),
        "mapped_sdc_swap": text.replace('"MAPPED_SDC": str(sdc)', '"MAPPED_SDC": "wrong"', 1),
        "corner_swap": text.replace('"OPERATING_CONDITION": "ssg0p9v125c"', '"OPERATING_CONDITION": "ffg"', 1),
        "scope_swap": text.replace('"SAIF_INSTANCE": BASE.BASE.SAIF_INSTANCE', '"SAIF_INSTANCE": "wrong"', 1),
        "collision_drop": text.replace("        BASE.BASE.collision_gate()\n", "", 1),
        "blocking_lock": text.replace("fcntl.LOCK_EX | fcntl.LOCK_NB", "fcntl.LOCK_EX", 1),
        "attempt_nonexclusive": text.replace("ATTEMPT.mkdir()", "ATTEMPT.mkdir(exist_ok=True)", 1),
        "attempt_unsealed": text.replace("BASE.BASE.seal_dir(ATTEMPT)", "pass", 1),
        "result_replace": text.replace("publish_no_replace(STAGE, RESULT)", "os.replace(STAGE, RESULT)", 1),
        "failure_replace": text.replace("publish_no_replace(FAIL_STAGE, FAILURE)", "os.replace(FAIL_STAGE, FAILURE)", 1),
        "partial_cite": text.replace('partial_axis_citable": False', 'partial_axis_citable": True', 1),
        "auto_retry": text.replace('"automatic_retry": False', '"automatic_retry": True', 1),
        "saif_gate_removed": text.replace(
            'if any(state[key] != COUNTS[key] for key in\n'
            '               ("vcs_compiles", "simv_runs", "saif_files")):',
            "if False:", 1),
    }
    for name, value in mutations.items():
        attack(name, lambda v=value: independent_static(v), "source_mutation")

    expected = M.expected_contract()
    contract_mutations = 0
    for section, value in expected.items():
        if type(value) is not dict:
            candidate = copy.deepcopy(expected)
            candidate[section] = changed(value)
            attack("contract_top_" + section,
                   lambda c=candidate: (_ for _ in ()).throw(RuntimeError())
                   if c != expected else None, "contract_mutation")
            contract_mutations += 1
        else:
            for key, leaf in value.items():
                candidate = copy.deepcopy(expected)
                candidate[section][key] = changed(leaf)
                attack("contract_leaf_" + section + "_" + key,
                       lambda c=candidate: (_ for _ in ()).throw(RuntimeError())
                       if c != expected else None, "contract_mutation")
                contract_mutations += 1

    for name, rel in M.NEW_NAMESPACES.items():
        check("fresh_" + name, not os.path.lexists(HW / rel), "freshness")
    p0 = sum(not item["rejected"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    output = {
        "schema": "m1494_m1493_c2_lca_successor_blind_hammer_output_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE" if p0 == 0 and p1 == 0
                  else "FAIL_DO_NOT_CITE",
        "checks": checks, "attacks": attacks,
        "summary": {"checks_passed": sum(item["pass"] for item in checks),
                    "checks_total": len(checks),
                    "mutations_rejected": sum(item["rejected"] for item in attacks),
                    "mutations_total": len(attacks),
                    "contract_mutations": contract_mutations,
                    "p0_count": p0, "p1_count": p1,
                    "author_tests_run": replay.testsRun,
                    "author_test_failures": len(replay.failures) + len(replay.errors)},
        "execution": {"license_query": 0, "vcs": 0, "simv": 0,
                      "saif": 0, "pt": 0, "ptpx": 0, "eda": 0,
                      "attempts_consumed": 0},
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if p0 == 0 and p1 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
