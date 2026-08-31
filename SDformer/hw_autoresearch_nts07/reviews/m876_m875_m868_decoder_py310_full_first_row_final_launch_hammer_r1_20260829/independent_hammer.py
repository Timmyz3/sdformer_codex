"""Independent bounded final-launch hammer for M875/M868.

This reviewer-owned program never invokes the no-argument M868 runner, never
enumerates the full first row, and never writes in the canonical results
namespace.  All publication attacks use an automatically removed temporary
namespace below reviews.
"""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True


HW = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DRIVER = HW / "system_simulator/scripts/execute_m868_m861_decoder_py310_full_first_row_diagnostic.py"
RUNNER = HW / "system_simulator/scripts/run_m868_m861_decoder_py310_full_first_row_one_shot.sh"
TESTS = HW / "system_simulator/tests/test_m868_m861_decoder_py310_full_first_row_diagnostic.py"
CANDIDATE = HW / "contracts/m868_m861_decoder_py310_full_first_row_diagnostic_candidate_r1_20260829.json"
RELEASE = HW / "contracts/m875_m868_decoder_py310_full_first_row_diagnostic_true_release_r1_20260829.json"
M869_DIR = HW / "reviews/m869_m868_decoder_py310_full_first_row_source_hammer_r1_20260829"
M869_REVIEW = M869_DIR / "review.json"
M869_HAMMER = M869_DIR / "independent_hammer.py"
M875_HANDOFF = HW / "reviews/m875_m868_decoder_py310_full_first_row_true_release_author_handoff_r1_20260829"
M876_REQUEST = HW / "reviews/m876_m875_m868_decoder_py310_full_first_row_final_launch_hammer_REQUEST_r1_20260829"
M868_HANDOFF = HW / "reviews/m868_m861_decoder_py310_full_first_row_source_author_handoff_r1_20260829"
M865_DIR = HW / "reviews/m865_m861_decoder_streaming_event_sweep_source_hammer_r1_20260829"
M857_DIR = HW / "reviews/m857_m836_decoder_controlled_scalability_failure_hammer_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "driver": "128fb2686d400593f59ed99390d0acf8c60d6c992f5daa951daa0ed4f6b0efbd",
    "runner": "fb1042474073c98d5dbc81a79f604de5d257f3953fb4bc40a3dfe840e303fe5a",
    "tests": "c27a048dcd4deb576f0b6cc2196b31d4c618b5b6ee6408c903190c3586d09867",
    "candidate": "2bcf8aeaf22cbf9c5178a9a030d72ee52372e78bdeec2c94e7361947d09d57d3",
    "release": "4e781456574ac6240a2303fe1d2104b1e7b517745f0a5d80db9b2322feeef85f",
    "m869_review": "38650a4a37e09a7ac4ae0d8d96a3838c433a2191fcfa368018f57292ab55cad5",
    "m869_manifest": "cc277cde39344880c3af3dd59e5583e02c93b30119b3df9e6bcfb7e8561f2f83",
    "m869_outer": "d827e0c24c62bdf05649bb1065267472c2c8799fcb82a280ec672bcd2d59452a",
    "m865_review": "68ac2981629250346fb7ec30c376b2d1707de5f3d0cde2d7badf1431be4737fa",
    "m857_review": "c2b244e4d6d56af6d81c028aa0cfe000517161e67e2866cc1ca782c9fd58e75a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

AUTHORIZATION = {
    "launch_now": True,
    "run_one_nonproduction_full_first_row_diagnostic": True,
    "formal_runner_invocations": 1,
    "max_attempts": 1,
    "run_full_population": False,
    "run_production": False,
    "run_vcs": False,
    "run_dc": False,
    "run_formality": False,
    "run_pt": False,
    "run_ptpx": False,
    "query_license": False,
    "run_gpu": False,
    "run_remote": False,
    "network_or_remote_jobs": 0,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=reject_duplicates,
                       parse_constant=reject_constant)
    return value


def exact_typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(actual) == set(expected) and
                all(exact_typed_equal(actual[key], expected[key])
                    for key in expected))
    if isinstance(expected, list):
        return (len(actual) == len(expected) and
                all(exact_typed_equal(left, right)
                    for left, right in zip(actual, expected)))
    if isinstance(expected, float):
        return math.isfinite(actual) and actual == expected
    return actual == expected


def require_exact_authorization(value):
    if not exact_typed_equal(value, AUTHORIZATION):
        raise ValueError("authorization is not exact-key typed-equal")
    return True


def load_exact(path, expected, name):
    assert sha256(path) == expected
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load_exact(DRIVER, EXPECTED["driver"], "m876_exact_m868")
M869 = load_exact(M869_HAMMER, sha256(M869_HAMMER), "m876_exact_m869_hammer")


def sealed(directory):
    identity = D.verify_sealed_directory(Path(directory)) if hasattr(
        D, "verify_sealed_directory") else D.verify_sealed(Path(directory))
    return identity


def verify_fixed_identities():
    assert sha256(PYTHON) == EXPECTED["python"]
    assert sha256(DRIVER) == EXPECTED["driver"]
    assert sha256(RUNNER) == EXPECTED["runner"]
    assert sha256(TESTS) == EXPECTED["tests"]
    assert sha256(CANDIDATE) == EXPECTED["candidate"]
    assert sha256(RELEASE) == EXPECTED["release"]
    assert sha256(M869_REVIEW) == EXPECTED["m869_review"]
    assert sha256(DOCS359) == EXPECTED["docs359"]
    assert sealed(M869_DIR) == {
        "manifest_sha256": EXPECTED["m869_manifest"],
        "outer_seal_file_sha256": EXPECTED["m869_outer"],
    }
    for directory in (M875_HANDOFF, M876_REQUEST, M868_HANDOFF,
                      M865_DIR, M857_DIR):
        identity = sealed(directory)
        assert len(identity["manifest_sha256"]) == 64
        assert len(identity["outer_seal_file_sha256"]) == 64
    assert sha256(M865_DIR / "review.json") == EXPECTED["m865_review"]
    assert sha256(M857_DIR / "review.json") == EXPECTED["m857_review"]
    return {
        "python_sha256": EXPECTED["python"],
        "driver_sha256": EXPECTED["driver"],
        "runner_sha256": EXPECTED["runner"],
        "tests_sha256": EXPECTED["tests"],
        "candidate_sha256": EXPECTED["candidate"],
        "release_sha256": EXPECTED["release"],
        "m869_review_sha256": EXPECTED["m869_review"],
        "m869_outer_seal_file_sha256": EXPECTED["m869_outer"],
        "docs359_sha256": EXPECTED["docs359"],
        "all_applicable_double_seals_recomputed": True,
    }


def verify_release_semantics():
    release = strict_json(RELEASE)
    request = strict_json(M876_REQUEST / "request.json")
    handoff = strict_json(M875_HANDOFF / "handoff.json")
    m869 = strict_json(M869_REVIEW)
    m865 = strict_json(M865_DIR / "review.json")
    m857 = strict_json(M857_DIR / "review.json")
    candidate = strict_json(CANDIDATE)
    assert release["schema"] == "m875_m868_decoder_py310_full_first_row_diagnostic_true_release_v1"
    assert release["status"] == "INERT_TRUE_RELEASE_AFTER_M869_PASS100__PENDING_DIFFERENT_M876_FINAL_HAMMER__ONE_NONPRODUCTION_ROW_ONLY"
    assert release["release"] is True
    assert release["launch_now"] is False and release["effective_now"] is False
    assert type(release["max_attempts"]) is int and release["max_attempts"] == 1
    assert release["candidate_binding"]["sha256"] == EXPECTED["candidate"]
    assert release["source_identity"]["runner"]["sha256"] == EXPECTED["runner"]
    assert release["source_identity"]["driver"]["sha256"] == EXPECTED["driver"]
    assert release["source_identity"]["tests"]["sha256"] == EXPECTED["tests"]
    assert release["interpreter"] == {
        "absolute_path": str(PYTHON), "sha256": EXPECTED["python"],
        "version": "3.10.18", "ambient_python3_allowed": False,
        "python_shebang_allowed": False, "path_fallback_allowed": False,
    }
    assert release["m869_source_hammer"]["review_json_sha256"] == EXPECTED["m869_review"]
    assert release["m869_source_hammer"]["outer_seal_file_sha256"] == EXPECTED["m869_outer"]
    assert release["m869_source_hammer"]["score"] == 100
    assert release["m869_source_hammer"]["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
    assert release["diagnostic_identity"] == {
        "label": "M854_FIRST_D0_A1_T0",
        "population": "M686_ZURICH_CITY_09_A_S10",
        "record_ordinal": 0, "module_index": 0, "sample_id": 0,
        "configuration": "A1_OSG", "timestep": 0, "rows": 1,
        "expected_compressed_transaction_count_gate": 9582057,
        "expected_expanded_request_count_gate": 38672612,
        "counts_are_cycles": False, "full_population_rows": 0,
        "production_rows": 0,
    }
    require_exact_authorization(
        release["future_final_authorization"]["authorization"])
    assert release["future_final_authorization"]["key_count"] == 15
    assert release["future_final_authorization"]["comparison"] == "EXACT_KEY_SET_VALUE_AND_PYTHON_TYPE_EQUALITY"
    assert release["release_effective_only_after"]["required_status"] == "PASS100_M868_PY310_FULL_FIRST_ROW_FINAL_LAUNCH__ONE_NONPRODUCTION_DIAGNOSTIC_AUTHORIZED"
    assert release["release_effective_only_after"]["different_reviewer_required"] is True
    assert release["one_way_authorization"]["formal_runner_invocations"] == 1
    assert release["one_way_authorization"]["full_population_runs"] == 0
    assert release["one_way_authorization"]["production_runs"] == 0
    assert release["one_way_authorization"]["release_reuse"] is False
    assert release["claim_boundary"]["paper_citable"] is False
    assert release["claim_boundary"]["production_cycles"] is False
    assert release["claim_boundary"]["production_speedup"] is False
    assert request["review_target"]["release_sha256"] == EXPECTED["release"]
    assert request["required_final_authorization"]["key_count"] == 15
    require_exact_authorization(request["required_final_authorization"]["authorization"])
    assert handoff["release"]["sha256"] == EXPECTED["release"]
    assert handoff["release"]["effective_before_final_hammer"] is False
    assert m869["status"] == "PASS100_M868_PY310_FULL_FIRST_ROW_SOURCE__AUTHORIZE_EXACTLY_ONE_NONPRODUCTION_DIAGNOSTIC"
    assert m865["status"].startswith("NO_GO_M861_FULL_FIRST_ROW_GATE__P1_1")
    assert m865["finding"]["m861_scheduling_semantics_implicated"] is False
    assert m857["status"].startswith("PASS100_CONTROLLED_SCALABILITY_FAILURE_AUDIT")
    assert m857["attempt_audit"]["permanently_consumed"] is True
    assert candidate["workload"]["expected_compressed_transaction_count"] == 9582057
    assert candidate["workload"]["expected_expanded_request_count"] == 38672612
    assert candidate["claim_boundary"]["production_cycles"] is False
    return {
        "release_is_inert_before_final_hammer": True,
        "release_exact_typed_authorization_key_count": 15,
        "m869_pass100_bound": True,
        "m865_failure_preserved": True,
        "m857_m836_consumed_failure_preserved": True,
        "cardinality_gates_only": [9582057, 38672612],
        "cycles_speedup_paper_citable": False,
    }


def malformed_json_and_typed_attacks(tmp):
    attacks = []
    for label, text in (
            ("duplicate", '{"a":1,"a":2}\n'),
            ("nan", '{"a":NaN}\n'),
            ("infinity", '{"a":Infinity}\n')):
        path = tmp / (label + ".json")
        path.write_text(text, encoding="utf-8")
        try:
            strict_json(path)
            raise AssertionError(label + " strict JSON accepted")
        except ValueError:
            attacks.append(label)
    variants = []
    missing = dict(AUTHORIZATION)
    missing.pop("run_remote")
    variants.append(("missing", missing))
    extra = dict(AUTHORIZATION)
    extra["extra"] = False
    variants.append(("extra", extra))
    bool_for_int = dict(AUTHORIZATION)
    bool_for_int["formal_runner_invocations"] = True
    variants.append(("bool_for_int", bool_for_int))
    int_for_bool = dict(AUTHORIZATION)
    int_for_bool["launch_now"] = 1
    variants.append(("int_for_bool", int_for_bool))
    wrong = dict(AUTHORIZATION)
    wrong["run_production"] = True
    variants.append(("wrong_value", wrong))
    rejected = []
    for label, value in variants:
        try:
            require_exact_authorization(value)
            raise AssertionError(label + " authorization accepted")
        except ValueError:
            rejected.append(label)
    assert require_exact_authorization(dict(AUTHORIZATION))
    return {"strict_json_rejected": attacks,
            "typed_authorization_rejected": rejected,
            "positive_exact_authorization": True}


def run_command(command, env=None):
    return subprocess.run(command, text=True, capture_output=True,
                          check=False, env=env)


def runtime_and_no_work_attacks(tmp):
    exact = run_command([str(PYTHON), "-c",
                         "import platform,sys;print(sys.executable);print(platform.python_version())"])
    assert exact.returncode == 0
    assert exact.stdout.splitlines() == [str(PYTHON), "3.10.18"]
    assert not DRIVER.read_bytes().startswith(b"#!")
    base = ["/bin/bash", "--noprofile", "--norc", str(RUNNER),
            "--dry-run-no-work"]
    clean = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    attacks = []
    for label, additions in (
            ("missing_pins", {}),
            ("malformed_runner", {"M868_EXPECTED_RUNNER_SHA256": "bad",
                                  "M868_EXPECTED_CANDIDATE_SHA256": EXPECTED["candidate"]}),
            ("wrong_runner", {"M868_EXPECTED_RUNNER_SHA256": "0" * 64,
                              "M868_EXPECTED_CANDIDATE_SHA256": EXPECTED["candidate"]}),
            ("wrong_candidate", {"M868_EXPECTED_RUNNER_SHA256": EXPECTED["runner"],
                                 "M868_EXPECTED_CANDIDATE_SHA256": "0" * 64})):
        env = dict(clean)
        env.update(additions)
        completed = run_command(base, env)
        assert completed.returncode != 0
        attacks.append({"label": label, "return_code": completed.returncode})
    copied = tmp / "copied_runner.sh"
    shutil.copyfile(RUNNER, copied)
    copied.chmod(0o700)
    copied_env = dict(clean)
    copied_env.update({"M868_EXPECTED_RUNNER_SHA256": sha256(copied),
                       "M868_EXPECTED_CANDIDATE_SHA256": EXPECTED["candidate"]})
    completed = run_command(["/bin/bash", "--noprofile", "--norc",
                             str(copied), "--dry-run-no-work"], copied_env)
    assert completed.returncode != 0
    attacks.append({"label": "noncanonical_runner", "return_code": completed.returncode})
    positive_env = dict(clean)
    positive_env.update({"M868_EXPECTED_RUNNER_SHA256": EXPECTED["runner"],
                         "M868_EXPECTED_CANDIDATE_SHA256": EXPECTED["candidate"]})
    positive = run_command(base, positive_env)
    assert positive.returncode == 0
    assert "PASS_M868_NO_WORK_DRY_RUN__NO_FILES_CREATED__NO_ATTEMPT" in positive.stdout
    return {"python_path": str(PYTHON), "python_version": "3.10.18",
            "python_sha256": EXPECTED["python"], "negative_attacks": attacks,
            "no_work_dry_run": "PASS", "request_enumerated": False}


def fake_and_wrong_hammer_pin_attacks(tmp):
    saved = D.HAMMER_DIR
    directory, review_sha, identity = M869.fake_hammer(tmp / "positive")
    D.HAMMER_DIR = directory
    try:
        assert D.validate_hammer(review_sha, identity["outer_seal_file_sha256"])
        failures = []
        for label, rsha, outer in (
                ("wrong_review_pin", "0" * 64, identity["outer_seal_file_sha256"]),
                ("wrong_outer_pin", review_sha, "0" * 64),
                ("malformed_review_pin", "bad", identity["outer_seal_file_sha256"]),
                ("malformed_outer_pin", review_sha, "bad")):
            try:
                D.validate_hammer(rsha, outer)
                raise AssertionError(label + " accepted")
            except D.Failure:
                failures.append(label)
        bad_dir, bad_review, bad_identity = M869.fake_hammer(tmp / "bad", "FAKE_PASS")
        D.HAMMER_DIR = bad_dir
        try:
            D.validate_hammer(bad_review, bad_identity["outer_seal_file_sha256"])
            raise AssertionError("fake status accepted")
        except D.Failure:
            failures.append("fake_status")
    finally:
        D.HAMMER_DIR = saved
    return {"positive_sealed_synthetic_hammer": True,
            "rejected": failures}


def isolated_publication_and_collision_attacks(tmp):
    collision_root = tmp / "collision"
    collision_root.mkdir()
    collision = M869.attack_collision_namespace(collision_root)
    publication = M869.attack_temporary_attempt_publication_failure(
        tmp / "publication")
    return {"collision": collision, "publication": publication,
            "canonical_namespace_touched": False}


def static_runner_audit():
    source = RUNNER.read_text(encoding="utf-8")
    preflight = source.index("--validate-formal-preflight")
    resource = source.index("m868_resource_gate", preflight)
    consume_phase = source.index('m868_phase="CONSUME_ONE_WAY_ATTEMPT"')
    consume = source.index("--consume-attempt", consume_phase)
    assert preflight < resource < consume_phase < consume
    assert "2097152" in source and source.count("100663296") >= 2
    assert source.index("mv -T --no-clobber") < source.index("--write-failure-receipt")
    assert "run-production" not in source
    assert source.count("RUN_EXACT_ONE_FULL_FIRST_ROW_DIAGNOSTIC") == 1
    assert source.count("PUBLISH_CANONICAL_NOREPLACE") == 1
    assert "renameat2" in DRIVER.read_text(encoding="utf-8")
    return {
        "formal_preflight_before_resource_before_attempt": True,
        "disk_floor_kib": 2097152,
        "mem_available_floor_kib": 100663296,
        "commit_headroom_floor_kib": 100663296,
        "same_directory_double_seal_noreplace": True,
        "failure_partial_move_before_receipt": True,
        "full_population_or_production_mode_present": False,
    }


def canonical_population():
    result = HW / "results/m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829"
    attempt = HW / "results/.m868_m861_decoder_py310_full_first_row_diagnostic_r1_attempt_consumed"
    failure_prefix = result.name + ".failed_or_incomplete."
    names = []
    for entry in os.scandir(result.parent):
        if (entry.name == result.name or entry.name == attempt.name or
                entry.name.startswith(result.name + ".stage.") or
                entry.name.startswith(attempt.name + ".stage.") or
                entry.name.startswith(failure_prefix) or
                entry.name.startswith(result.name + ".driver_stdout.") or
                entry.name.startswith(result.name + ".driver_stderr.")):
            names.append(entry.name)
    return sorted(names)


def main():
    before = canonical_population()
    assert before == []
    with tempfile.TemporaryDirectory(prefix="m876_final_hammer.",
                                     dir=str(HW / "reviews")) as name:
        tmp = Path(name)
        result = {
            "schema": "m876_independent_final_launch_bounded_attacks_v1",
            "identities": verify_fixed_identities(),
            "release_semantics": verify_release_semantics(),
            "json_and_authorization_attacks": malformed_json_and_typed_attacks(tmp),
            "runtime_and_no_work": runtime_and_no_work_attacks(tmp),
            "hammer_pin_attacks": fake_and_wrong_hammer_pin_attacks(tmp),
            "isolated_state_attacks": isolated_publication_and_collision_attacks(tmp),
            "runner_static_audit": static_runner_audit(),
            "canonical_population_before": before,
            "canonical_population_after": canonical_population(),
            "full_first_row_invoked": False,
            "full_population_invoked": False,
            "production_invoked": False,
            "vcs_dc_pt_fm_eda_license_gpu_remote_network_invoked": False,
            "status": "PASS_INDEPENDENT_M876_FINAL_LAUNCH_BOUNDED_ATTACKS",
        }
        assert result["canonical_population_after"] == []
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
