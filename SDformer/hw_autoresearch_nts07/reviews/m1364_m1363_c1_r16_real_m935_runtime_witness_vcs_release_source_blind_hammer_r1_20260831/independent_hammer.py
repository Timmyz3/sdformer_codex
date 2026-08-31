#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent source-only hammer for the M1363 C1/R16 launch source.

This program never executes the runner or any EDA/license command.  It checks
the frozen source, attacks every M1355 false-negative family, and audits the
transition from source-absent authoring to runtime-present launch.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKER = HW / "verif_m1363_c1_r16_vcs_release_exact/check_m1363_c1_r16_vcs_release_exact_source.py"
TESTS = HW / "verif_m1363_c1_r16_vcs_release_exact/test_m1363_c1_r16_vcs_release_exact_source.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("m1364_bound_m1363_checker", CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("checker import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_checker()


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1364_MUTATED"
    if type(value) is dict:
        result = dict(value)
        result["m1364_extra"] = True
        return result
    raise TypeError(type(value))


def prior_false_negative_cases():
    base = M.expected_contract()
    cases = [
        ("contract_extra_top_level", lambda d: d.__setitem__("m1364_extra", True)),
        ("contract_date_changed", lambda d: d.__setitem__("date", "2099-01-01")),
        ("contract_future_execution_removed", lambda d: d.pop("future_execution")),
        ("contract_future_execution_extra", lambda d:
         d["future_execution"].__setitem__("m1364_extra", True)),
    ]
    cases.extend(("future_execution_" + key, lambda d, key=key:
                  d["future_execution"].__setitem__(key, changed(d["future_execution"][key])))
                 for key in base["future_execution"])
    cases.extend([
        ("author_execution_extra", lambda d:
         d["author_execution"].__setitem__("m1364_extra", False)),
        ("claim_boundary_extra", lambda d:
         d["claim_boundary"].__setitem__("m1364_extra", False)),
    ])
    if len(cases) != 16:
        raise AssertionError(len(cases))
    return cases


def main() -> int:
    runner = M.RUNNER.read_text(encoding="utf-8")
    tests = TESTS.read_text(encoding="utf-8")
    checks: dict[str, object] = {}

    # Canonical exact-byte/source authority, including predecessor seals.
    common = M.validate_common(skip_author=False)
    checks["exact_byte_members"] = common["exact_byte_members"]
    checks["sealed_authorities"] = common["sealed_authorities"]
    checks["contract_exact_set_value"] = M.strict_json(M.CONTRACT) == M.expected_contract()

    # All sixteen previously accepted mutations must now fail closed.
    outcomes = []
    for name, mutate in prior_false_negative_cases():
        candidate = copy.deepcopy(M.expected_contract())
        mutate(candidate)
        rejected = False
        try:
            M.check_contract_dict(candidate)
        except AssertionError:
            rejected = True
        outcomes.append({"attack": name, "rejected": rejected})
    false_negatives = [row["attack"] for row in outcomes if not row["rejected"]]
    checks["m1355_attack_count"] = len(outcomes)
    checks["m1355_rejected_count"] = len(outcomes) - len(false_negatives)
    checks["m1355_false_negative_count"] = len(false_negatives)

    # The eight future external digests are all mandatory lowercase SHA-256.
    env_names = (
        "M1363_EXPECTED_RUNNER_SHA256", "M1363_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
        "M1363_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256",
        "M1363_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
        "M1363_EXPECTED_LAUNCH_RELEASE_SHA256", "M1363_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
        "M1363_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256",
        "M1363_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256")
    good = {name: "a" * 64 for name in env_names}
    env_attacks = 0
    env_false_negatives = 0
    for name in env_names:
        for value in (None, "a" * 63, "A" * 64):
            mutant = dict(good)
            if value is None:
                mutant.pop(name)
            else:
                mutant[name] = value
            env_attacks += 1
            if M.env_gate(mutant):
                env_false_negatives += 1
    checks["external_digest_pin_attacks"] = env_attacks
    checks["external_digest_pin_false_negatives"] = env_false_negatives

    # Independently audit the one-shot ordering and failure containment.
    compile_call = runner.index('"${VCS_BIN}" -full64')
    simulate_call = runner.index("./simv -no_save")
    attempt_publish = runner.index('publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"')
    license_export = runner.index('export SNPSLMD_LICENSE_FILE=')
    resource_phase = runner.index('phase="RESOURCE_PREFLIGHT"')
    attempt_phase = runner.index('phase="ATTEMPT_CONSUME"')
    checks["vcs_compile_cardinality"] = runner.count('"${VCS_BIN}" -full64')
    checks["simv_run_cardinality"] = runner.count("./simv -no_save")
    checks["timeout_cardinality"] = runner.count(
        "/usr/bin/timeout --signal=TERM --kill-after=30s")
    checks["attempt_publish_before_license_and_tools"] = (
        attempt_publish < license_export < compile_call < simulate_call)
    checks["collision_gates_before_attempt"] = runner.count(
        "collision_gate\n", resource_phase, attempt_phase)
    checks["failure_isolation_recursive_seal"] = all(anchor in runner for anchor in (
        'trap on_exit EXIT', 'mv -- "${WORK}" "${FAILURE_STAGE}/private_build"',
        'seal_dir "${FAILURE_STAGE}"',
        'publish_no_replace "${FAILURE_STAGE}" "${QUARANTINE}"'))
    checks["no_automatic_retry_true"] = "automatic_retry=true" not in runner
    checks["no_destructive_remove"] = re.search(r"(^|[;&|\s])rm(\s|$)", runner) is None
    checks["runner_sha256"] = hashlib.sha256(M.RUNNER.read_bytes()).hexdigest()
    checks["runner_sha_matches_pin"] = checks["runner_sha256"] == M.RUNNER_SHA256

    # P0: launch requires future-present, then unconditionally executes a test
    # whose first assertion requires future-absent.  This makes the authorized
    # runtime state unable to reach ATTEMPT_CONSUME or VCS.
    runtime_checker_call = '"${PYTHON_BIN}" -I "${SOURCE_CHECKER}" --mode runtime_present >/dev/null'
    source_tests_call = 'PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -I "${SOURCE_TESTS}" >/dev/null'
    checks["runtime_checker_before_source_tests"] = (
        runner.index(runtime_checker_call) < runner.index(source_tests_call) < attempt_phase)
    checks["tests_require_source_absent"] = (
        'M.validate_future("source_absent")' in tests and
        'self.assertTrue(M.validate_future("source_absent")["future_absent"])' in tests)
    checks["runtime_launch_reachable"] = not (
        checks["runtime_checker_before_source_tests"] and checks["tests_require_source_absent"])

    p0 = []
    if not checks["runtime_launch_reachable"]:
        p0.append("runtime_present_then_source_absent_test_deadlock")
    if false_negatives:
        p0.append("m1355_false_negatives_remain")
    if env_false_negatives:
        p0.append("external_digest_pin_false_negative")
    required = {
        "vcs_compile_cardinality": 1,
        "simv_run_cardinality": 1,
        "timeout_cardinality": 2,
        "collision_gates_before_attempt": 2,
        "attempt_publish_before_license_and_tools": True,
        "failure_isolation_recursive_seal": True,
        "no_automatic_retry_true": True,
        "no_destructive_remove": True,
        "runner_sha_matches_pin": True,
        "contract_exact_set_value": True,
    }
    for key, expected in required.items():
        if checks.get(key) != expected:
            p0.append("static_gate_" + key)

    output = {
        "schema": "m1364_m1363_c1_r16_vcs_release_source_blind_hammer_output_r1_v1",
        "status": "PASS" if not p0 else "FAIL_DO_NOT_CITE",
        "p0": p0,
        "checks": checks,
        "m1355_outcomes": outcomes,
        "authorization": {
            "launch_release": False,
            "vcs_compiles": 0,
            "simv_runs": 0,
            "all_other_eda_runs": 0,
            "automatic_retry": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not p0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
