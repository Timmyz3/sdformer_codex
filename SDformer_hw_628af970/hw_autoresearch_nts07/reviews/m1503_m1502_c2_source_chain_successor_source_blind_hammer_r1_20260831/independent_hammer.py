#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author, no-EDA blind hammer for M1502 C2 source."""
from __future__ import annotations

import copy
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
    "dc_handoff/scripts/run_m1502_m1493_c2_source_chain_successor_"
    "one_shot.py")
CHECKER = HW / (
    "verif_m1502_c2_source_chain_successor/"
    "check_m1502_c2_source_chain_successor_source.py")
TESTS = HW / (
    "verif_m1502_c2_source_chain_successor/"
    "test_m1502_c2_source_chain_successor_source.py")
CONTRACT = HW / (
    "contracts/m1502_m1493_c2_source_chain_successor_source_contract_"
    "r1_20260831.json")
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT_SIDECAR) + ".seal.sha256")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
UCLI_KEY = ROOT / "ucli.key"
FUTURE = (
    HW / "reviews/m1503_m1502_c2_source_chain_successor_source_blind_hammer_r1_20260831",
    HW / "contracts/m1504_m1503_m1502_c2_source_chain_successor_launch_release_r1_20260831.json",
    HW / "reviews/m1505_m1504_m1502_c2_source_chain_successor_final_launch_hammer_r1_20260831",
)

RUNNER_SHA = "91fc6a8867a138098b660e4d450eda50f5bd1850f9127bc349c2a303aac36df1"
CHECKER_SHA = "7535c11d878d0582c47b9247ef8be7b2b5e7104f5197ca031de8772ab24cfba1"
TESTS_SHA = "9fa4aa08e9033cd3d913bddc6932affc65377a5b1e8c504085306f32b8fe619a"
CONTRACT_SHA = "8ee9286fc59a536ef8e61d19b6111102933ea167eb40e910cf7fa3c17b7e0eb6"
CONTRACT_SIDECAR_SHA = "a113601324228f43470fc1951f910bfb47ff06df5e8b049e51289b33712efaf4"
CONTRACT_OUTER_SHA = "d157fd31abfe2eb48d01dd1d54fdd363059cec22fb05ba29596fa76c926abea1"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
UCLI_KEY_SHA = "1107aa2b8d30b14e7e4f9237ff461fb058ae4e07c8a5bed30bef3ad3eb9c30ac"
M1493_FAILURE_SHA = {
    "payload": "43497b8701400b6c7c5d3f0cc29a2a41955a135fff4be6720968cbeb736cc5e7",
    "manifest": "53e77670cd0f07ea457dc35f041e3885f7d73b304149c8d52e116fd06d6a5f88",
    "outer": "8cb2e41374f9b827c118b949e1a37b66baeec5bef578d81ee68a0d95a90d4a7e",
}
M1494_SHA = {
    "review": "65435aca804c486d50d8332774c70e87083d66d5c2e7acc30485dc84ba458340",
    "manifest": "b2ff59fd22bd0bd6463ae9ac9aa31ee82d77099d40ea4890fd99600255b9811b",
    "outer": "329ed4435761eb7d00be969d43ac05221c837cc3f79cedefd03d557034c432f7",
}
M1495_SHA = "838ea0f3714167c43c6f4e40829c2d1a59d1b84ee7468758798c82f21114eb94"
M1495_SIDECAR_SHA = "e6cf168d790f890824936be1555fc7fa22d4c0ea8faf27ff071f66af0a5f5fb9"
M1495_OUTER_SHA = "dcadd0a39357f61b91cd3221cdf54ea68fb6c18685e9aefd60bfbc09deecfdd7"
M1496_SHA = {
    "review": "ef0af9fbf0ab094f40052de8fc552b7b97e2519dd5db88c6f3c2bf7505acb810",
    "manifest": "72da922a5b652bf07eecc2ecc75ade847c7950c1c3a056299cca613bc1a19049",
    "outer": "2c8a99c7a9f0d2f56d6b77583f09cdc9ade265ba55b47c721e0ff44680d98e79",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return value + "__M1503_MUTATION"
    if type(value) is list:
        return value + ["__M1503_MUTATION"]
    raise TypeError(type(value).__name__)


def main() -> int:
    exact = {
        "runner": (RUNNER, RUNNER_SHA),
        "checker": (CHECKER, CHECKER_SHA),
        "tests": (TESTS, TESTS_SHA),
        "contract": (CONTRACT, CONTRACT_SHA),
        "contract_sidecar": (CONTRACT_SIDECAR, CONTRACT_SIDECAR_SHA),
        "contract_outer": (CONTRACT_OUTER, CONTRACT_OUTER_SHA),
        "docs359": (DOCS359, DOCS359_SHA),
        "ucli_key": (UCLI_KEY, UCLI_KEY_SHA),
    }
    checks = []
    for name, (path, digest) in exact.items():
        checks.append({"check": name + "_exact", "pass": sha(path) == digest})
    if not all(item["pass"] for item in checks):
        raise RuntimeError("identity gate")
    if CONTRACT_SIDECAR.read_text().split() != [CONTRACT_SHA, CONTRACT.name]:
        raise RuntimeError("M1502 contract sidecar content")
    if CONTRACT_OUTER.read_text().split() != [CONTRACT_SIDECAR_SHA,
                                               CONTRACT_SIDECAR.name]:
        raise RuntimeError("M1502 contract outer content")
    if any(os.path.lexists(path) for path in FUTURE):
        raise RuntimeError("M1503/M1504/M1505 not fresh")

    C = load("m1503_bound_m1502_checker", CHECKER)
    T = load("m1503_bound_m1502_tests", TESTS)
    R = C.R

    source = C.check_source(True)
    checks.append({"check": "native_source_checker", "pass":
                   source.get("status") ==
                   "PASS_M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE__NO_EDA"})

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(T)
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    checks.append({"check": "author_tests_17", "pass":
                   replay.testsRun == 17 and not replay.failures and
                   not replay.errors})

    # Independently rebind the sealed pre-attempt SOURCE_CHAIN failure and
    # the complete M1494/M1495/M1496 authority chain.
    R.verify_predecessor_failure()
    failure_members = R.AUTH.verify_seal(
        R.OLD_FAILURE, M1493_FAILURE_SHA["manifest"],
        M1493_FAILURE_SHA["outer"])
    checks.append({"check": "m1493_failure_members", "pass":
                   failure_members == {"failure.json"}})
    checks.append({"check": "m1493_failure_payload_exact", "pass":
                   sha(R.OLD_FAILURE / "failure.json") ==
                   M1493_FAILURE_SHA["payload"]})
    blind = R.AUTH.verify_authority(
        R.M1494, M1494_SHA["review"], M1494_SHA["manifest"],
        M1494_SHA["outer"])
    checks.append({"check": "m1494_authority", "pass":
                   blind.get("status") ==
                   "PASS_M1494_M1493_C2_LCA_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE"})
    if (sha(R.M1495) != M1495_SHA
            or sha(Path(str(R.M1495) + ".sha256")) != M1495_SIDECAR_SHA
            or sha(Path(str(R.M1495) + ".sha256.seal.sha256")) !=
            M1495_OUTER_SHA):
        raise RuntimeError("M1495 exact authority drift")
    R.AUTH.verify_sidecars(R.M1495)
    release = R.strict_json(R.M1495)
    checks.append({"check": "m1495_authority", "pass":
                   release.get("status") ==
                   "RELEASE_M1493_C2_LCA_SUCCESSOR__FRESH_M1496_REQUIRED__NO_LAUNCH"})
    final = R.AUTH.verify_authority(
        R.M1496, M1496_SHA["review"], M1496_SHA["manifest"],
        M1496_SHA["outer"])
    checks.append({"check": "m1496_authority", "pass":
                   final.get("status") ==
                   "PASS_M1496_AUTHORIZE_ONE_M1493_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"})

    # Exercise the corrected production entrypoint, not a string surrogate.
    saved = {name: os.environ.pop(name, None) for name in R.ENV_PINS}
    callpath_error = None
    try:
        try:
            R.verify_frozen_execution_inputs()
        except BaseException as error:
            callpath_error = error
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    callpath_ok = (type(callpath_error) is R.Failure and
                   str(callpath_error) ==
                   "M1502 authority absent: required exact SHA environment" and
                   not isinstance(callpath_error, AttributeError))
    checks.append({"check": "real_corrected_callpath", "pass": callpath_ok})

    source_text = RUNNER.read_text()
    attacks = []

    def attack(name, thunk, category):
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    source_mutations = {
        "restore_bad_method_call": source_text.replace(
            "    verify_predecessor_failure()\n    verify_new_authority()",
            "    EXEC.verify_predecessor_failure()\n"
            "    verify_predecessor_failure()\n    verify_new_authority()", 1),
        "drop_axis": source_text.replace(
            'for axis in ("k8", "k1x8"):', 'for axis in ("k8",):', 1),
        "drop_case": source_text.replace(
            "for case in range(5):", "for case in range(4):", 1),
        "drop_vcs_counter": source_text.replace(
            'state["vcs_compiles"] += 1', "pass", 1),
        "drop_simv_counter": source_text.replace(
            'state["simv_runs"] += 1', "pass", 1),
        "drop_saif_counter": source_text.replace(
            'state["saif_files"] += 1', "pass", 1),
        "drop_ptpx_counter": source_text.replace(
            'state["ptpx_runs"] += 1', "pass", 1),
    }
    for name, mutation in source_mutations.items():
        if mutation == source_text:
            raise RuntimeError("mutation not applied: " + name)
        attack(name, lambda text=mutation: C.check_execution_text(text),
               "source")

    for flag, name in (("-debug_access+r", "delete_debug_access_r"),
                       ("-lca", "delete_lca")):
        def invoke_flag(f=flag):
            mutated_prefix = [item for item in R.COMPILE_PREFIX if item != f]
            with mock.patch.object(R, "COMPILE_PREFIX", mutated_prefix):
                C.check_execution_text(source_text)
        attack(name, invoke_flag, "source")

    expected = C.expected_contract()

    def contract_attack(name, mutate):
        candidate = copy.deepcopy(expected)
        mutate(candidate)
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "contract.json"
            path.write_text(json.dumps(candidate, allow_nan=False) + "\n")
            def invoke():
                with mock.patch.object(C, "CONTRACT", path):
                    C.check_contract()
            attack(name, invoke, "contract")

    contract_attack("axes", lambda value:
                    value["preserved_execution"].__setitem__("axes", ["k8"]))
    contract_attack("cases", lambda value:
                    value["preserved_execution"].__setitem__("cases", [0, 1, 2, 3]))
    for key in ("vcs_compiles", "simv_runs", "production_saif_files",
                "ptpx_runs"):
        contract_attack("counter_" + key, lambda value, k=key:
                        value["preserved_execution"].__setitem__(
                            k, value["preserved_execution"][k] + 1))
    for key in tuple(expected["claim_boundary"]):
        contract_attack("claim_" + key, lambda value, k=key:
                        value["claim_boundary"].__setitem__(k, True))
    for key in tuple(expected["preserved_execution"]["fresh_namespaces"]):
        contract_attack("namespace_" + key, lambda value, k=key:
                        value["preserved_execution"]["fresh_namespaces"].__setitem__(
                            k, value["preserved_execution"]["fresh_namespaces"][k]
                            + ".mutation"))
    for key in tuple(expected["future_authority"]):
        contract_attack("future_authority_" + key, lambda value, k=key:
                        value["future_authority"].__setitem__(
                            k, changed(value["future_authority"][k])))

    with tempfile.TemporaryDirectory() as temp_name:
        duplicate = Path(temp_name) / "duplicate.json"
        duplicate.write_text('{"schema":1,"schema":2}\n')
        attack("duplicate_json", lambda: C.strict_json(duplicate), "json")
        nonfinite = Path(temp_name) / "nonfinite.json"
        nonfinite.write_text('{"schema":NaN}\n')
        attack("nonfinite_json", lambda: C.strict_json(nonfinite), "json")

    if not all(item["pass"] for item in checks):
        raise RuntimeError("independent check failed")
    if not attacks or any(item["false_negative"] for item in attacks):
        raise RuntimeError("mutation false negative")
    categories = {}
    for item in attacks:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    output = {
        "schema": "m1503_m1502_c2_source_chain_successor_mechanical_checks_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE",
        "author_tests": {"passed": 17, "total": 17},
        "source_checker": "PASS_M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE__NO_EDA",
        "corrected_callpath": {
            "called": "verify_frozen_execution_inputs",
            "terminal": "M1502 authority absent: required exact SHA environment",
            "attribute_error": False,
        },
        "independent_checks": {
            "passed": len(checks), "total": len(checks), "details": checks},
        "mutation_campaign": {
            "rejected": len(attacks), "total": len(attacks),
            "false_negatives": 0, "categories": categories,
            "details": attacks,
        },
        "authority_chain": {
            "m1493_sealed_source_chain_failure": True,
            "m1494_blind_hammer": True,
            "m1495_release": True,
            "m1496_final_hammer": True,
        },
        "fresh_namespaces_before_review_publication": {
            "m1503": True, "m1504": True, "m1505": True},
        "execution": {
            "license_query": 0, "vcs": 0, "simv": 0, "saif": 0,
            "pt": 0, "ptpx": 0, "eda": 0, "ssh": 0, "gpu": 0,
            "attempts_consumed": 0,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
