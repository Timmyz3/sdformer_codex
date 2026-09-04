"""Fresh independent bounded attacks for the exact M868 source identity.

This reviewer-owned harness is Python-3.10-only.  It never invokes the full
first row, consumes the canonical attempt, creates a canonical result or
quarantine, or enters a production/EDA/remote path.  Publication-state attacks
are confined to an automatically removed directory below ``reviews``.
"""

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


HW = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DRIVER = HW / "system_simulator/scripts/execute_m868_m861_decoder_py310_full_first_row_diagnostic.py"
RUNNER = HW / "system_simulator/scripts/run_m868_m861_decoder_py310_full_first_row_one_shot.sh"
CANDIDATE = HW / "contracts/m868_m861_decoder_py310_full_first_row_diagnostic_candidate_r1_20260829.json"
M861_PATH = HW / "system_simulator/scripts/analyze_m861_decoder_streaming_event_sweep.py"

EXPECTED = {
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "driver": "128fb2686d400593f59ed99390d0acf8c60d6c992f5daa951daa0ed4f6b0efbd",
    "runner": "fb1042474073c98d5dbc81a79f604de5d257f3953fb4bc40a3dfe840e303fe5a",
    "candidate": "2bcf8aeaf22cbf9c5178a9a030d72ee52372e78bdeec2c94e7361947d09d57d3",
    "m861": "f72ed3b820051d624699152b784c05fa674106556ab73f452a2cf96a9f72d7a4",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exact(path, expected, name):
    assert sha256(path) == expected
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load_exact(DRIVER, EXPECTED["driver"], "m869_exact_m868")
M = load_exact(M861_PATH, EXPECTED["m861"], "m869_exact_m861")


def run(command, env=None):
    return subprocess.run(command, text=True, capture_output=True,
                          check=False, env=env)


def expect_fail(command, label, env=None):
    completed = run(command, env=env)
    assert completed.returncode != 0, label + " unexpectedly passed"
    return {
        "label": label,
        "return_code": completed.returncode,
        "failed_closed": True,
    }


def attack_runtime_and_runner_pins(tmp):
    assert sha256(PYTHON) == EXPECTED["python"]
    exact = run([str(PYTHON), "-c",
                 "import platform,sys; print(sys.executable); print(platform.python_version())"])
    assert exact.returncode == 0
    assert exact.stdout.splitlines() == [str(PYTHON), "3.10.18"]
    assert not DRIVER.read_bytes().startswith(b"#!")
    assert Path(os.readlink("/proc/{}/exe".format(os.getpid()))).resolve() == PYTHON.resolve()

    attacks = []
    attacks.append(expect_fail(
        ["/usr/bin/python3", str(DRIVER), "--validate-candidate"],
        "ambient_python3"))
    attacks.append(expect_fail(
        ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "python3",
         str(DRIVER), "--validate-candidate"], "PATH_fallback"))

    base = ["/bin/bash", "--noprofile", "--norc", str(RUNNER),
            "--dry-run-no-work"]
    clean = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    attacks.append(expect_fail(base, "missing_caller_pins", clean))
    malformed = dict(clean)
    malformed.update({
        "M868_EXPECTED_RUNNER_SHA256": "bad",
        "M868_EXPECTED_CANDIDATE_SHA256": EXPECTED["candidate"],
    })
    attacks.append(expect_fail(base, "malformed_runner_pin", malformed))
    wrong = dict(clean)
    wrong.update({
        "M868_EXPECTED_RUNNER_SHA256": "0" * 64,
        "M868_EXPECTED_CANDIDATE_SHA256": EXPECTED["candidate"],
    })
    attacks.append(expect_fail(base, "wrong_runner_pin", wrong))
    wrong_candidate = dict(clean)
    wrong_candidate.update({
        "M868_EXPECTED_RUNNER_SHA256": EXPECTED["runner"],
        "M868_EXPECTED_CANDIDATE_SHA256": "0" * 64,
    })
    attacks.append(expect_fail(base, "wrong_candidate_pin", wrong_candidate))

    copied_runner = tmp / "copied_runner.sh"
    shutil.copyfile(RUNNER, copied_runner)
    copied_runner.chmod(0o700)
    copied_env = dict(clean)
    copied_env.update({
        "M868_EXPECTED_RUNNER_SHA256": sha256(copied_runner),
        "M868_EXPECTED_CANDIDATE_SHA256": EXPECTED["candidate"],
    })
    attacks.append(expect_fail(
        ["/bin/bash", "--noprofile", "--norc", str(copied_runner),
         "--dry-run-no-work"], "noncanonical_runner", copied_env))
    return {"interpreter_version": "3.10.18",
            "interpreter_sha256": sha256(PYTHON), "attacks": attacks}


def attack_candidate_source_and_json(tmp):
    original_candidate = D.CANDIDATE
    original = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    try:
        status_drift = tmp / "status_drift.json"
        mutated = json.loads(json.dumps(original))
        mutated["status"] = "FAKE"
        status_drift.write_text(json.dumps(mutated), encoding="utf-8")
        D.CANDIDATE = status_drift
        try:
            D.validate_candidate(status_drift)
            raise AssertionError("candidate status drift accepted")
        except D.Failure:
            pass

        source_drift = tmp / "source_drift.json"
        mutated = json.loads(json.dumps(original))
        mutated["source_identity"]["m861_analyzer"]["sha256"] = "0" * 64
        source_drift.write_text(json.dumps(mutated), encoding="utf-8")
        D.CANDIDATE = source_drift
        try:
            D.validate_candidate(source_drift)
            raise AssertionError("source drift accepted")
        except D.Failure:
            pass
    finally:
        D.CANDIDATE = original_candidate

    duplicate = tmp / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    nonfinite = tmp / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    for path in (duplicate, nonfinite):
        try:
            D.strict_json(path)
            raise AssertionError("malformed strict JSON accepted")
        except D.M785.Failure:
            pass
    return {"candidate_drift_rejected": True,
            "source_identity_drift_rejected": True,
            "duplicate_key_rejected": True,
            "nonfinite_rejected": True}


def fake_hammer(tmp, status=D.HAMMER_STATUS):
    directory = tmp / "hammer"
    directory.mkdir(parents=True)
    review = {
        "schema": "m869_m868_decoder_py310_full_first_row_source_hammer_v1",
        "status": status,
        "score": 100,
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "decision": {
            "exactly_one_nonproduction_full_first_row_diagnostic_authorized": True,
            "full_population_authorized": False,
            "cycles_or_speedup_citable": False,
        },
    }
    path = directory / "review.json"
    path.write_text(json.dumps(review, sort_keys=True) + "\n", encoding="utf-8")
    identity = D.seal_directory(directory, ("review.json",))
    return directory, sha256(path), identity


def attack_collision_namespace(tmp):
    saved = (D.RESULT, D.ATTEMPT, D.FAILURE_PREFIX)
    results = tmp / "collision_results"
    results.mkdir()
    D.RESULT = results / "canonical"
    D.ATTEMPT = results / ".attempt_consumed"
    D.FAILURE_PREFIX = "canonical.failed_or_incomplete."
    cases = (
        D.RESULT.name,
        D.ATTEMPT.name,
        D.RESULT.name + ".stage.attack",
        D.ATTEMPT.name + ".stage.attack",
        D.FAILURE_PREFIX + "attack",
    )
    observed = []
    try:
        for name in cases:
            path = results / name
            path.mkdir()
            collision = D.scan_collisions()
            assert name in collision
            observed.append(name)
            path.rmdir()
        assert D.scan_collisions() == ()
    finally:
        D.RESULT, D.ATTEMPT, D.FAILURE_PREFIX = saved
    return {"collision_classes": observed,
            "all_detected": True}


def attack_temporary_attempt_publication_failure(tmp):
    saved = (D.RESULT, D.ATTEMPT, D.FAILURE_PREFIX, D.CANDIDATE,
             D.HAMMER_DIR)
    work = tmp / "publication"
    results = work / "results"
    results.mkdir(parents=True)
    D.RESULT = results / "diagnostic"
    D.ATTEMPT = results / ".attempt_consumed"
    D.FAILURE_PREFIX = "diagnostic.failed_or_incomplete."
    hammer_dir, review_sha, hammer_identity = fake_hammer(work)
    D.HAMMER_DIR = hammer_dir

    candidate_data = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    candidate_data["canonical"] = D._candidate_paths()
    candidate = work / "candidate.json"
    candidate.write_text(json.dumps(candidate_data, indent=2) + "\n",
                         encoding="utf-8")
    D.CANDIDATE = candidate
    try:
        # A fake-but-well-sealed hammer with the wrong admission status is
        # rejected before the positive temporary admission is exercised.
        bad_dir, bad_review_sha, bad_identity = fake_hammer(work / "bad", "FAKE_PASS")
        D.HAMMER_DIR = bad_dir
        try:
            D.validate_hammer(bad_review_sha,
                              bad_identity["outer_seal_file_sha256"])
            raise AssertionError("fake hammer status accepted")
        except D.Failure:
            pass
        D.HAMMER_DIR = hammer_dir

        attempt = D.consume_attempt(
            candidate, RUNNER, EXPECTED["runner"], review_sha,
            hammer_identity["outer_seal_file_sha256"],
            D.ATTEMPT.name + ".stage.first")
        assert attempt["status"].startswith("CONSUMED_IMMEDIATELY")
        first_identity = D.verify_sealed(D.ATTEMPT)
        try:
            D.consume_attempt(
                candidate, RUNNER, EXPECTED["runner"], review_sha,
                hammer_identity["outer_seal_file_sha256"],
                D.ATTEMPT.name + ".stage.retry")
            raise AssertionError("retry restored/consumed attempt")
        except D.Failure:
            pass
        assert D.verify_sealed(D.ATTEMPT) == first_identity

        stdout = work / "stdout.log"
        stderr = work / "stderr.log"
        stdout.write_text("bounded synthetic failure\n", encoding="utf-8")
        stderr.write_text("no canonical work\n", encoding="utf-8")
        private = results / (D.RESULT.name + ".stage.partial")
        private.mkdir()
        (private / "diagnostic.json").write_text("{}\n", encoding="utf-8")
        partial = results / (D.FAILURE_PREFIX + "one.partial_artifact")
        os.rename(private, partial)
        quarantine = results / (D.FAILURE_PREFIX + "one")
        failure = D.write_failure_receipt(
            candidate, RUNNER, EXPECTED["runner"], review_sha,
            hammer_identity["outer_seal_file_sha256"], stdout, stderr,
            quarantine, 77, "SYNTHETIC_POST_ATTEMPT_FAILURE", str(partial))
        assert failure["status"].startswith("FAILED_OR_INCOMPLETE")
        assert D.verify_sealed(quarantine) == {
            "manifest_sha256": failure["manifest_sha256"],
            "outer_seal_file_sha256": failure["outer_seal_file_sha256"],
        }
        assert partial.is_dir() and not D.RESULT.exists()

        stage = results / (D.RESULT.name + ".stage.publish")
        stage.mkdir()
        (stage / "diagnostic.json").write_text("{}\n", encoding="utf-8")
        D.seal_directory(stage, ("diagnostic.json",))
        publication = D.publish_no_replace(stage, D.RESULT)
        assert publication["status"] == "PASS_M868_CANONICAL_NOREPLACE_PUBLICATION"
        replacement = results / (D.RESULT.name + ".stage.replacement")
        replacement.mkdir()
        (replacement / "diagnostic.json").write_text("{}\n", encoding="utf-8")
        D.seal_directory(replacement, ("diagnostic.json",))
        try:
            D.publish_no_replace(replacement, D.RESULT)
            raise AssertionError("canonical replacement accepted")
        except D.Failure:
            pass
        assert replacement.is_dir()
        return {
            "fake_hammer_rejected": True,
            "attempt_consumed_once": True,
            "retry_refused_and_attempt_not_restored": True,
            "partial_artifact_moved": True,
            "failure_receipt_double_sealed": True,
            "canonical_publication_renameat2_noreplace": True,
            "canonical_replacement_rejected": True,
        }
    finally:
        D.RESULT, D.ATTEMPT, D.FAILURE_PREFIX, D.CANDIDATE, D.HAMMER_DIR = saved


def audit_runner_order_and_resource_gate():
    source = RUNNER.read_text(encoding="utf-8")
    preflight = source.index("--validate-formal-preflight")
    resource = source.index("m868_resource_gate", preflight)
    consume_phase = source.index('m868_phase="CONSUME_ONE_WAY_ATTEMPT"')
    consume = source.index("--consume-attempt", consume_phase)
    assert preflight < resource < consume_phase < consume
    assert "free_kib" in source and "2097152" in source
    assert "mem_available" in source and "100663296" in source
    assert "commit_headroom" in source
    assert source.index("mv -T --no-clobber") < source.index("--write-failure-receipt")
    assert source.count("SHA256SUMS.seal.sha256") >= 3
    assert "renam" in inspect.getsource(D._rename_noreplace).lower()
    return {
        "preflight_then_resource_then_attempt": True,
        "disk_floor_kib": 2097152,
        "mem_available_floor_kib": 100663296,
        "commit_headroom_floor_kib": 100663296,
        "resource_failure_precedes_attempt": True,
        "partial_move_precedes_failure_receipt": True,
    }


def inspect_first_row_without_enumeration():
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    row = candidate["workload"]
    assert row == {
        "identity": "M854_FIRST_D0_A1_T0",
        "population": "M686_ZURICH_CITY_09_A_S10",
        "record_ordinal": 0,
        "module_index": 0,
        "sample_id": 0,
        "configuration": "A1_OSG",
        "timestep": 0,
        "expected_compressed_transaction_count": 9582057,
        "expected_expanded_request_count": 38672612,
        "rows_authorized": 1,
        "population_rows_authorized": 0,
    }
    contract = D.strict_json(HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")
    entry = contract["inputs"]["primary_m686"]
    payload_root = HW / entry["directory"]
    manifest = D.strict_json(payload_root / "manifest.json")
    records = D.M785.normalized_population_records(
        manifest, "M686_ZURICH_CITY_09_A_S10")
    first = records[0]
    assert int(first["module_index"]) == 0
    assert int(first["sample_id"]) == 0
    source = inspect.getsource(D.full_first_row_requests)
    assert 'records[0]' in source and '"A1_OSG", 0' in source
    assert "yield from" in source
    return {
        "population": row["population"],
        "record_ordinal": 0,
        "module_index": 0,
        "sample_id": 0,
        "configuration": "A1_OSG",
        "timestep": 0,
        "expected_compressed_gate": 9582057,
        "expected_expanded_gate": 38672612,
        "full_row_enumerated": False,
    }


def bounded_m861_replay():
    manual = M.manual_endpoint_priority_miter()
    random_miter = M.exact_old_new_miter(M.deterministic_random_dag(
        777, seed=869))
    real = M.run_real_prefix(100000, miter_limit=1000)
    assert manual["status"] == "PASS_MANUAL_E_D_I_R_PRIORITY_MITER"
    assert random_miter["status"] == "PASS_EXACT_OLD_NEW_MITER"
    assert real["prefix_requests"] == 100000
    assert real["detail_retained"] is False
    assert real["old_new_miter"]["status"] == "PASS_EXACT_OLD_NEW_MITER"
    return {
        "manual_six_priority_classes": manual["cycle_classes"],
        "random_dag_requests": 777,
        "random_all_11_fields_and_tokens_equal": True,
        "real_prefix_requests": 100000,
        "real_prefix_detail_retained": False,
        "separate_real_old_new_miter_requests": 1000,
        "real_old_new_miter_pass": True,
        "full_first_row_enumerated": False,
    }


def main():
    assert sha256(CANDIDATE) == EXPECTED["candidate"]
    assert sha256(HW / "docs/359_DATE终局冻结_20260813.md") == EXPECTED["docs359"]
    with tempfile.TemporaryDirectory(prefix="m869_hammer.",
                                     dir=str(HW / "reviews")) as name:
        tmp = Path(name)
        output = {
            "schema": "m869_independent_bounded_attacks_v1",
            "runtime": attack_runtime_and_runner_pins(tmp),
            "candidate_source_json": attack_candidate_source_and_json(tmp),
            "collisions": attack_collision_namespace(tmp),
            "temporary_publication": attack_temporary_attempt_publication_failure(tmp),
            "runner_resource_order": audit_runner_order_and_resource_gate(),
            "first_row_identity": inspect_first_row_without_enumeration(),
            "bounded_m861": bounded_m861_replay(),
            "canonical_attempt_consumed": False,
            "canonical_result_or_quarantine_created": False,
            "full_first_row_invoked": False,
            "full_population_invoked": False,
            "production_invoked": False,
            "vcs_dc_pt_fm_eda_license_gpu_remote_training_invoked": False,
            "status": "PASS_INDEPENDENT_M869_BOUNDED_SOURCE_ATTACKS",
        }
        print(json.dumps(output, indent=2, sort_keys=True,
                         allow_nan=False))


if __name__ == "__main__":
    main()
