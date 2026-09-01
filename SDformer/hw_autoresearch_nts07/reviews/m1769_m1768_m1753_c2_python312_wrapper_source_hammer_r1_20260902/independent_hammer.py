#!/usr/bin/env python3
"""Independent zero-execution hammer for M1768 Python-3.12 C2 wrapper."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
WRAPPER = HW / "dc_handoff/scripts/run_m1768_m1753_c2_python312_wrapper_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1768_m1753_c2_python312_wrapper_source.py"
TEST = HW / "system_simulator/tests/test_m1768_m1753_c2_python312_wrapper_source.py"
CONTRACT = HW / "contracts/m1768_m1767_m1753_c2_python312_wrapper_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1768_m1767_m1753_c2_python312_wrapper_source_author_receipt_r1_20260902"
M1767 = HW / "reviews/m1767_m1761_m1753_c2_python36_preparse_failure_receipt_r1_20260902"
M1753 = HW / "dc_handoff/scripts/run_m1753_m1715_c2_three_axis_mapped_directed_component_energy_one_shot.py"
M1753_CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
M1760 = HW / "reviews/m1760_m1753_c2_three_axis_mapped_directed_component_energy_source_hammer_r1_20260901"
M1761 = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"
PYTHON312 = Path("/usr/bin/python3.12")
M1770 = HW / "contracts/m1770_m1769_m1768_m1753_c2_python312_wrapper_launch_release_r1_20260902.json"
NAMESPACES = (
    HW / "results/.m1768_m1753_c2_python312_wrapper_attempt_consumed",
    HW / "results/.m1753_c2_three_axis_mapped_directed_component_energy_attempt_consumed",
    HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901",
    HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.failed_or_incomplete.quarantine",
    HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.private_build.unsealed_do_not_cite")


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
            need(key not in value, "duplicate key " + key); value[key] = item
        return value
    path = Path(path); need(path.is_file() and not path.is_symlink(), "JSON nonregular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON")))
    need(type(value) is dict, "JSON root"); return value


def verify_file(path, payload, sidecar, outer):
    s = Path(str(path) + ".sha256"); o = Path(str(path) + ".sha256.seal.sha256")
    need(sha(path) == payload and sha(s) == sidecar and sha(o) == outer,
         "file triple " + str(path))
    need(s.read_text().split() == [payload, path.name]
         and o.read_text().split() == [sidecar, s.name], "file seal content")


def verify_dir(root, primary, primary_sha, manifest_sha, outer_sha):
    manifest = root / "SHA256SUMS"; outer = root / "SHA256SUMS.seal.sha256"
    need(sha(root / primary) == primary_sha and sha(manifest) == manifest_sha
         and sha(outer) == outer_sha, "directory triple " + str(root))
    need(outer.read_text().split() == [manifest_sha, "SHA256SUMS"], "outer")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1); need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*")); name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == fields[0],
             "member drift " + name); listed.add(name)


def load_checker():
    spec = importlib.util.spec_from_file_location("m1769_target", str(CHECKER))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def must_fail(function):
    try:
        function()
    except Exception:
        return 1
    raise RuntimeError("negative mutation survived")


def wrapper_policy(text):
    need(text.startswith("#!/usr/bin/python3.12\n"), "shebang")
    need(text.count("os.execve(") == 1, "execve count")
    need("subprocess" not in text and "os.system(" not in text, "alternate launch")
    need("while True" not in text and "for attempt" not in text,
         "retry construct")
    need("renameat2" in text and
         "os.fsencode(destination), 1" in text, "RENAME_NOREPLACE")
    main = text[text.index("def main():"):]
    tokens = ("verify_interpreter()", "verify_authority()", "namespaces_fresh()",
              "stage.mkdir()", "seal_dir(stage)",
              "publish_no_replace(stage, ATTEMPT)", "os.execve(")
    locations = [main.index(token) for token in tokens]
    need(locations == sorted(locations) and len(set(locations)) == len(locations),
         "execution ordering")
    need('os.execve(str(PYTHON312), [str(PYTHON312), str(M1753)], dict(os.environ))'
         in main, "exact execve")
    for token in ("M1768_EXPECTED_WRAPPER_SHA256",
                  "M1768_EXPECTED_SOURCE_CONTRACT_SHA256",
                  "M1768_EXPECTED_M1769_REVIEW_SHA256",
                  "M1768_EXPECTED_M1769_MANIFEST_SHA256",
                  "M1768_EXPECTED_M1769_OUTER_FILE_SHA256",
                  "M1768_EXPECTED_M1770_RELEASE_SHA256",
                  "M1753_EXPECTED_RUNNER_SHA256",
                  "M1753_EXPECTED_M1760_REVIEW_SHA256",
                  "M1753_EXPECTED_M1761_RELEASE_SHA256"):
        need(token in text, "authority pin " + token)
    return True


def release_policy(value, wrapper_sha, contract_sha, review_sha):
    need(value.get("status") ==
         "AUTHORIZE_ONE_M1768_C2_PYTHON312_WRAPPER_ATTEMPT", "release status")
    need(value.get("identity") == {"wrapper_sha256": wrapper_sha,
        "source_contract_sha256": contract_sha,
        "m1769_review_sha256": review_sha}, "release identity")
    need(value.get("authorization") == {
        "future_m1768_wrapper_attempts": 1, "automatic_retry": False,
        "underlying_m1753_campaigns": 1}, "release budget")


def main():
    verify_file(CONTRACT,
        "1a5268690144aa0b61813132f292d50e48fa5a94fa808f2a4bf7c6ea5fe70f8e",
        "c192596b6b33ad8371ee03756d3df00bed7ea3f863d40a70832da339699dfa55",
        "11843f5ce3c4dde3d73c860065e213dad637bc47beb7a3a987cbc1ac0a542b68")
    verify_dir(AUTHOR, "author_receipt.json",
        "80663202ba318eebb1449e5032495445eb9c8181f7763c5829c16d443428d564",
        "52237cce7410819c2b49faf1f799b41d35ee3f94302cc36b7afb89d1a6a5d1f1",
        "3118d731fd61d19f2e94ee0156be76158abe93c49728193aa45636074e76db2f")
    verify_dir(M1767, "receipt.json",
        "330e533d0f545439b7b0539a0c4816e8e77a6c89330620571162ec060c6b3729",
        "ed9fbb6e5a3b30e77b81f74ee64861231336576ce562517a7a999f518c26d474",
        "80aaf88a542ed1fb9e754172d722ec8cdd7741bfe399ce02be212c30c60f2b71")
    verify_file(M1753_CONTRACT,
        "39f864a254aa3314ab2b4939997674958c7ae7cc5966273629c94d53ecbe0e21",
        "ec8dcccf92d8979b674008ca83edff4ae98f87e127e3212a979801853ac27092",
        "2b7510d270632a1989366870abdb68e1bcb3470e665c486b89be6d4e3f50b8d9")
    verify_dir(M1760, "review.json",
        "987fccddbad6281bb31aa128987118ef4942e210d47201c528ab9be50055329c",
        "e8921f4612f9b0b8532b43f441ccd2b93c2600e5dca861cefcb6ef293601afcf",
        "55caca70cf9670ee8e361c062f4c73e1272c399c990eb4b1e27771008f00830e")
    verify_file(M1761,
        "bb5b32ead4bd2ff682abfbcedf242b645c20c89d71db0b7eeadc7c18f5191f5e",
        "71b353b92b87c559b8c6501e4b8834e4c78383c5e29d1a367b1ae277423f7e3d",
        "47df6661b49128152093bce9cfaacf9868e4288d4647011f1b654f4783095074")
    need(sha(WRAPPER) ==
         "3a19b42593c22d5b756e9584e5cfd6a94fab9e03e614735f6db230fa1da4443c",
         "wrapper drift")
    need(sha(M1753) ==
         "adb24c20746bc95340952426dbcba1c5fde3400dce7763d73320f303d3a64d9e",
         "M1753 drift")
    need(sha(PYTHON312) ==
         "0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814",
         "Python 3.12 binary drift")

    failure = strict_json(M1767 / "receipt.json")
    forensic = strict_json(M1767 / "forensic_checks.json")
    author = strict_json(AUTHOR / "author_receipt.json")
    need(failure["status"] ==
         "SEALED_M1767_OPERATOR_ENVIRONMENT_FAILURE__M1753_BODY_NOT_ENTERED__EDA_LICENSE_ATTEMPT_RESULT_ZERO__AUTHORIZE_SOURCE_ONLY_M1768_WRAPPER",
         "M1767 status")
    counts = failure["execution_counts"]
    need(counts == {"m1753_module_body_entries": 0, "m1753_main_entries": 0,
        "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
        "saif_files": 0, "ptpx_runs": 0, "m1753_attempts_created": 0,
        "m1753_results_created": 0}, "M1767 execution counts")
    need(failure["operator_observation"]["syntax_error_line"] == 13
         and failure["operator_observation"]["syntax_error_source"] ==
             "from __future__ import annotations"
         and failure["operator_observation"]["m1753_module_body_entered"] is False
         and forensic["target_line_13"] == "from __future__ import annotations"
         and forensic["compile_only_result"] ==
             "SyntaxError: future feature annotations is not defined"
         and forensic["m1753_execution_namespaces_absent"] == 4
         and forensic["same_uid_eda_or_license_processes_observed_for_m1753"] == 0,
         "M1767 preparse evidence")
    need(author["author_execution"] == {"wrapper_runs": 0, "m1753_runs": 0,
        "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
        "saif_files": 0, "ptpx_runs": 0, "wrapper_attempts_created": 0,
        "m1753_attempts_created": 0, "results_created": 0},
         "M1768 author execution")
    need(not os.path.lexists(str(M1770)) and
         not any(os.path.lexists(str(path)) for path in NAMESPACES),
         "premature release/namespace")

    checker = load_checker(); checker.validate_sources()
    wrapper = WRAPPER.read_text(); wrapper_policy(wrapper)
    contract = strict_json(CONTRACT); checker.validate_contract(contract)
    source_mutations = 0
    mutations = (
        wrapper.replace("#!/usr/bin/python3.12", "#!/usr/bin/python3"),
        wrapper + "\nos.execve('/bin/false', [], {})\n",
        wrapper.replace("verify_interpreter()\n    verify_authority()",
                        "verify_authority()\n    verify_interpreter()"),
        wrapper.replace("verify_authority()\n    namespaces_fresh()",
                        "namespaces_fresh()\n    verify_authority()"),
        wrapper.replace("publish_no_replace(stage, ATTEMPT)\n    os.execve(",
                        "os.execve("),
        wrapper.replace("os.fsencode(destination), 1", "os.fsencode(destination), 0"),
        wrapper.replace("os.execve(str(PYTHON312)", "subprocess.run(str(PYTHON312)"),
        wrapper + "\nwhile True:\n    pass\n")
    for mutant in mutations:
        source_mutations += must_fail(lambda value=mutant: wrapper_policy(value))

    contract_mutations = 0
    for coordinate, changed in (
            (("interpreter_identity", "version"), "3.12.12"),
            (("interpreter_identity", "binary_sha256"), "0" * 64),
            (("bound_authority", "m1767_receipt_sha256"), "0" * 64),
            (("bound_authority", "m1761_release_sha256"), "0" * 64),
            (("future_budget", "m1768_wrapper_attempts"), 2),
            (("future_budget", "underlying_m1753_campaigns"), 2),
            (("future_budget", "automatic_retry"), True),
            (("claim_boundary", "mapped_vcs"), True)):
        mutant = copy.deepcopy(contract); mutant[coordinate[0]][coordinate[1]] = changed
        contract_mutations += must_fail(lambda value=mutant: checker.validate_contract(value))

    release_mutations = 0
    good_release = {"status": "AUTHORIZE_ONE_M1768_C2_PYTHON312_WRAPPER_ATTEMPT",
        "identity": {"wrapper_sha256": sha(WRAPPER),
            "source_contract_sha256": sha(CONTRACT),
            "m1769_review_sha256": "1" * 64},
        "authorization": {"future_m1768_wrapper_attempts": 1,
            "automatic_retry": False, "underlying_m1753_campaigns": 1}}
    release_policy(good_release, sha(WRAPPER), sha(CONTRACT), "1" * 64)
    for section, key, changed in (("authorization", "future_m1768_wrapper_attempts", 2),
                                  ("authorization", "automatic_retry", True),
                                  ("authorization", "underlying_m1753_campaigns", 2),
                                  ("identity", "wrapper_sha256", "0" * 64)):
        mutant = copy.deepcopy(good_release); mutant[section][key] = changed
        release_mutations += must_fail(lambda value=mutant:
            release_policy(value, sha(WRAPPER), sha(CONTRACT), "1" * 64))

    source_text = M1753.read_text()
    preparse = "NOT_APPLICABLE"
    if sys.version_info[:2] == (3, 6):
        try:
            compile(source_text, str(M1753), "exec")
        except SyntaxError as error:
            need(error.lineno == 13 and "future feature annotations" in str(error),
                 "Python3.6 preparse signature")
            preparse = "PASS_EXACT_M1753_SYNTAXERROR_LINE13_BODY_ZERO"
        else:
            raise RuntimeError("Python3.6 unexpectedly parsed M1753")
    elif sys.version_info[:2] == (3, 12):
        need(Path(sys.executable) == PYTHON312
             and Path(sys.executable).resolve() == PYTHON312
             and platform.python_implementation() == "CPython"
             and platform.python_version() == "3.12.13",
             "live Python3.12 path/version")
        compile(source_text, str(M1753), "exec")
        preparse = "PASS_EXACT_M1753_PARSE_ONLY_NO_EXECUTION"
    need(not any(os.path.lexists(str(path)) for path in NAMESPACES),
         "source hammer created namespace")
    need((source_mutations, contract_mutations, release_mutations) == (8, 8, 4),
         "mutation count")

    result = {"schema": "m1769_m1768_c2_python312_wrapper_source_hammer_output_r1_v1",
        "status": "PASS_M1769_M1768_C2_PYTHON312_WRAPPER_SOURCE_HAMMER__AUTHORIZE_ONE_WRAPPER_ATTEMPT",
        "python": sys.version.split()[0], "live_interpreter_check":
            sys.version_info[:2] == (3, 12),
        "python312_sha256": sha(PYTHON312), "m1753_preparse": preparse,
        "m1767_body_attempt_eda_license_result_all_zero": True,
        "ordered_protocol": ["live_interpreter", "m1767_m1753_m1760_m1761_m1769_m1770_authority",
            "fresh_namespace", "atomic_sealed_m1768_attempt", "execve_exact_m1753"],
        "exact_unchanged_m1753": True, "direct_retry": False,
        "automatic_retry": False, "m1761_budget_bypass": False,
        "mutations_rejected": {"source_order_atomic_exec": source_mutations,
            "contract_identity_budget": contract_mutations,
            "future_release_budget": release_mutations},
        "wrapper_runs": 0, "m1753_runs": 0, "license_queries": 0,
        "eda_runs": 0, "attempts_created": 0, "results_created": 0,
        "p0_count": 0, "p1_count": 0, "p2_count": 0}
    output = HERE / ("cpython" + str(sys.version_info[0]) +
                     str(sys.version_info[1]) + "_hammer.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
