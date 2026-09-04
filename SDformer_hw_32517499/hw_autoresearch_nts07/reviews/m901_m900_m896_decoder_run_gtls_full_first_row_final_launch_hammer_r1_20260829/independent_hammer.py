"""Fresh no-work final-launch hammer for M900/M896 RUN-GTLS.

This reviewer-owned program never invokes the no-argument M900 runner and
never enumerates D0/A1/t0.  It performs only exact-identity, static, directed
test, no-work dry-run, caller-pin negative, and temporary publication-helper
checks.  The canonical attempt/result/failure namespaces must remain absent.
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DRIVER = HW / "system_simulator/scripts/execute_m900_m896_decoder_run_gtls_full_first_row_runtime_gate.py"
RUNNER = HW / "system_simulator/scripts/run_m900_m896_decoder_run_gtls_full_first_row_one_shot.sh"
M896 = HW / "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py"
TESTS = HW / "system_simulator/tests/test_m896_decoder_run_gtls_source_candidate.py"
CONTRACT = HW / "contracts/m896_decoder_run_gtls_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m896_decoder_run_gtls_source_candidate_r1_20260829.json"
RELEASE = HW / "contracts/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_release_r1_20260829.json"
M899 = HW / "reviews/m899_m896_decoder_run_gtls_source_fresh_hammer_r1_20260829"
M900_HANDOFF = HW / "reviews/m900_m896_decoder_run_gtls_fullrow_release_author_handoff_r1_20260829"
M901_REQUEST = HW / "reviews/m901_m900_m896_decoder_run_gtls_fullrow_final_hammer_REQUEST_r1_20260829"
M883 = HW / "reviews/m883_m868_m861_decoder_py310_full_first_row_diagnostic_result_hammer_r1_20260829"
M868 = HW / "results/m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829/diagnostic.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULTS = HW / "results"

EXPECTED = {
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "driver": "5131446996c28eda694320da1441fa6fd5ba1791218a70e6946642f6873ad28b",
    "runner": "2a66cab84184eb8327fcef5f607c412874a598ae5a997e388ee09eb1662e6f8b",
    "m896": "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39",
    "tests": "12c1e092253ff078b52f7b5f7fcce9e17d4cb721e0f0d5aad2d75e86ca4d90eb",
    "contract": "0f3f9faa31f2ab9b2221eec5b65030a37279c290266ae84f325ba4ea60e1780d",
    "candidate": "8f70dcdc2445f31d4a90d65626978cdef03301de379d9cda8b541249ba7922fe",
    "release": "de25cfe7411bd7a4a516eb57653562e21bc042271dd03023fd0a3fd124fff29e",
    "m899_review": "8c9c51beaa7811e7ceec559ccef4618479c56975d919cf818be15f978ead1bda",
    "m899_manifest": "4eeae5b917554ad1a2c1c2812c8f1c1544108064a1c0527779193ac41d7e3f21",
    "m899_outer": "3617abb5a144a23d6c3a6048c975755120dc36b332d54e29b94ea614ff75939f",
    "m883_review": "ae443b36084a3361548ec6a950dbc0a962cf60ec650000c9638db61854c02f88",
    "m868": "53f71f804cad8acafdbc224d12acfbddc1510d1cb202286d67b018a1b1015344",
    "handoff_json": "a0464d50ec64d562633880a7933275358e3ee0e8df01a28af8dce0a329e92f94",
    "handoff_manifest": "edc4024cfdc3682815e2ba201731b3666f50c00658bab25319e333e54836f98f",
    "handoff_outer": "63175f890fc268149052d38d43d9e62c901b09babd2efa192aeb377941cbf03f",
    "request_json": "e6ee5b6d7ba3ae6c7645e4d97ddf5059b43d95d27d42bece12d63790b9602418",
    "request_manifest": "77911bbd676dff5bd3b17fe951198c650136e3d1f733c3b328a9e257c8a29d36",
    "request_outer": "8eb12f7bf34dc286dc12137114e5e3e7b386e6086063203710e76c18577039a8",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

ATTEMPT_NAME = ".m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_attempt_consumed"
RESULT_NAME = "m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_20260829"
FAILURE_PREFIX = RESULT_NAME + ".failed_or_incomplete."


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
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=reject_duplicates,
                      parse_constant=reject_constant)


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


def verify_file_seal(path):
    path = Path(path)
    inner = Path(str(path) + ".sha256")
    outer = Path(str(inner) + ".seal.sha256")
    assert path.is_file() and not path.is_symlink()
    assert inner.is_file() and not inner.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    assert inner.read_text(encoding="ascii") == sha256(path) + "  " + path.name + "\n"
    assert outer.read_text(encoding="ascii") == sha256(inner) + "  " + inner.name + "\n"
    return {"payload_sha256": sha256(path),
            "inner_seal_file_sha256": sha256(inner),
            "outer_seal_file_sha256": sha256(outer)}


def verify_directory_seal(directory):
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert directory.is_dir() and not directory.is_symlink()
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    rows = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        assert name and name not in (".", "..") and "/" not in name and "\x00" not in name
        assert name not in rows
        member = directory / name
        assert member.is_file() and not member.is_symlink()
        assert sha256(member) == digest
        rows[name] = digest
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    assert actual == set(rows) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    for entry in directory.iterdir():
        if entry.is_dir():
            assert entry.name == "__pycache__" and not entry.is_symlink()
    assert outer.read_text(encoding="ascii") == sha256(manifest) + "  SHA256SUMS\n"
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer)}


def canonical_collisions():
    return sorted(p.name for p in RESULTS.iterdir()
                  if (p.name == ATTEMPT_NAME or p.name == RESULT_NAME or
                      p.name.startswith(ATTEMPT_NAME + ".stage.") or
                      p.name.startswith(RESULT_NAME + ".stage.") or
                      p.name.startswith(RESULT_NAME + ".driver_") or
                      p.name.startswith(RESULT_NAME + ".runtime_resource_") or
                      p.name.startswith(FAILURE_PREFIX)))


def load_driver():
    assert Path(sys.executable) == PYTHON
    assert platform.python_version() == "3.10.18"
    assert sha256(PYTHON) == EXPECTED["python"]
    assert sha256(DRIVER) == EXPECTED["driver"]
    spec = importlib.util.spec_from_file_location("m901_exact_m900_driver", str(DRIVER))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_identities():
    expected_paths = {
        PYTHON: EXPECTED["python"], DRIVER: EXPECTED["driver"],
        RUNNER: EXPECTED["runner"], M896: EXPECTED["m896"],
        TESTS: EXPECTED["tests"], CONTRACT: EXPECTED["contract"],
        CANDIDATE: EXPECTED["candidate"], RELEASE: EXPECTED["release"],
        M899 / "review.json": EXPECTED["m899_review"],
        M883 / "review.json": EXPECTED["m883_review"], M868: EXPECTED["m868"],
        M900_HANDOFF / "handoff.json": EXPECTED["handoff_json"],
        M901_REQUEST / "request.json": EXPECTED["request_json"],
        DOCS359: EXPECTED["docs359"],
    }
    for path, expected in expected_paths.items():
        assert path.is_file() and not path.is_symlink() and sha256(path) == expected
    for path in (DRIVER, RUNNER, M896, TESTS, CONTRACT, CANDIDATE, RELEASE):
        verify_file_seal(path)
    assert verify_directory_seal(M899) == {
        "manifest_sha256": EXPECTED["m899_manifest"],
        "outer_seal_file_sha256": EXPECTED["m899_outer"]}
    assert verify_directory_seal(M900_HANDOFF) == {
        "manifest_sha256": EXPECTED["handoff_manifest"],
        "outer_seal_file_sha256": EXPECTED["handoff_outer"]}
    assert verify_directory_seal(M901_REQUEST) == {
        "manifest_sha256": EXPECTED["request_manifest"],
        "outer_seal_file_sha256": EXPECTED["request_outer"]}
    verify_directory_seal(M883)
    return {"all_exact_file_identities": True,
            "all_applicable_two_level_seals": True,
            "python_path": str(PYTHON), "python_version": "3.10.18",
            "python_sha256": EXPECTED["python"],
            "docs359_sha256": EXPECTED["docs359"]}


def verify_release_semantics():
    release = strict_json(RELEASE)
    request = strict_json(M901_REQUEST / "request.json")
    handoff = strict_json(M900_HANDOFF / "handoff.json")
    m899 = strict_json(M899 / "review.json")
    assert release["schema"] == "m900_m896_decoder_run_gtls_full_first_row_runtime_gate_release_v1"
    assert release["status"] == "INERT_RELEASE_AFTER_M899_PASS100__PENDING_FRESH_M901_FINAL_HAMMER"
    assert release["release"] is True and release["launch_now"] is False
    assert release["effective_now"] is False
    assert type(release["max_attempts"]) is int and release["max_attempts"] == 1
    expected_workload = {
        "identity": "M854_FIRST_D0_A1_T0",
        "population": "M686_ZURICH_CITY_09_A_S10",
        "record_ordinal": 0, "module_index": 0, "sample_id": 0,
        "configuration": "A1_OSG", "timestep": 0,
        "rows_authorized": 1, "population_rows_authorized": 0,
        "expected_compressed_transaction_count": 9582057,
        "expected_expanded_request_count": 38672612,
    }
    assert exact_typed_equal(release["workload"], expected_workload)
    assert release["runtime_and_state_gate"] == {
        "m883_anchor_elapsed_seconds": 932.0783571209759,
        "minimum_host_speedup": 100.0,
        "maximum_end_to_end_elapsed_seconds": 9.320783571209759,
        "counted_live_scheduler_state_maximum_bytes": 536870912,
        "counted_live_scheduler_state_maximum_mib": 512,
        "process_rss_is_diagnostic_only": True,
        "serialized_or_compressed_file_size_forbidden": True,
        "input_transaction_objects_excluded_from_counted_state": True,
        "three_consecutive_runtime_or_resource_over_gate_snapshots_terminate": True,
    }
    assert release["runtime_monitor"] == {
        "period_seconds": 1,
        "runtime_over_gate_snapshots_before_termination": 3,
        "resource_over_gate_snapshots_before_termination": 3,
        "counted_state_over_gate_snapshots_before_termination": 3,
        "emergency_minimum_free_disk_kib": 1048576,
        "emergency_minimum_mem_available_kib": 8388608,
        "emergency_minimum_commit_headroom_kib": 8388608,
        "heartbeat_records_counted_state_when_available": True,
        "rss_snapshot_is_diagnostic_only": True,
    }
    assert math.isclose(932.0783571209759 / 100.0,
                        9.320783571209759, rel_tol=0.0, abs_tol=0.0)
    assert release["resource_gate_before_attempt"] == {
        "minimum_free_disk_kib": 2097152,
        "minimum_mem_available_kib": 100663296,
        "minimum_commit_headroom_kib": 100663296,
        "failure_before_attempt_consumes_attempt": False,
    }
    assert release["exact_result_gate"]["expected_total_cycles"] == 20548766
    assert release["exact_result_gate"]["scheduled_requests_retained"] is False
    assert release["exact_result_gate"]["compressed_schedule_retained"] is False
    assert release["authorization"] == {
        "read_and_static_review": True, "run_no_work_dry_run": True,
        "write_fresh_final_hammer": True,
        "run_one_full_first_row_runtime_gate_now": False,
        "run_one_full_first_row_runtime_gate_after_fresh_pass100": True,
        "formal_runner_invocations_after_pass": 1,
        "run_full_population": False, "run_production": False,
        "run_vcs_dc_pt_fm_ptpx_eda_license_gpu_remote": False,
        "network_or_remote_jobs": 0,
    }
    assert release["claim_boundary"]["diagnostic_only"] is True
    for key in ("runtime_gate_completed", "full_population", "production",
                "decoder_complete", "cycles_or_speedup_citable",
                "system_speedup", "energy", "paper_ppa_ready",
                "paper_citable", "vcs_dc_pt_fm_ptpx_eda_gpu_remote",
                "docs359_modified"):
        assert release["claim_boundary"][key] is False
    assert release["release_effective_only_after"]["required_status"] == (
        "PASS100_M900_RUN_GTLS_FULL_FIRST_ROW_FINAL_LAUNCH__ONE_RUNTIME_GATE_DIAGNOSTIC_AUTHORIZED")
    assert release["release_effective_only_after"]["different_reviewer_required"] is True
    assert release["release_effective_only_after"]["must_bind_this_release_sha256"] is True
    assert release["release_effective_only_after"]["must_bind_runner_sha256"] is True
    assert m899["status"] == "PASS100_M896_RUN_GTLS_BOUNDED_EXACT__STATE_GATE_PASS__ONLY_FRESH_INERT_FULLROW_RELEASE_AUTHOR_AUTHORIZED"
    assert m899["score"] == 100 and m899["checks_passed"] == 54
    assert m899["real_100k_combined_state_gate"]["ceil_projected_bytes"] == 492931168
    assert m899["real_100k_combined_state_gate"]["gate_bytes"] == 536870912
    assert m899["real_100k_combined_state_gate"]["margin_bytes"] == 43939744
    assert m899["real_100k_combined_state_gate"]["rss_is_not_the_state_gate"] is True
    assert handoff["release"]["effective_before_final_hammer"] is False
    assert request["review_target"]["release_sha256"] == EXPECTED["release"]
    assert request["requested_output"]["score"] == 100
    assert request["requested_output"]["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
    assert request["requested_output"]["reviewer_launch_now"] is False
    assert request["requested_output"]["root_may_launch_once_after_pass"] is True

    mutations = []
    for label, mutate in (
            ("workload_missing", lambda x: x["workload"].pop("timestep")),
            ("workload_extra", lambda x: x["workload"].__setitem__("extra", 0)),
            ("bool_as_int", lambda x: x["workload"].__setitem__("record_ordinal", False)),
            ("int_as_bool", lambda x: x.__setitem__("launch_now", 0))):
        variant = copy.deepcopy(release)
        mutate(variant)
        accepted = (exact_typed_equal(variant.get("workload"), expected_workload)
                    and type(variant.get("launch_now")) is bool)
        assert not accepted
        mutations.append(label)
    return {"release_inert_before_hammer": True,
            "exact_single_row_identity": True,
            "typed_mutation_attacks_rejected": mutations,
            "runtime_deadline_seconds": 9.320783571209759,
            "counted_state_gate_bytes": 536870912,
            "m899_projection_bytes": 492931168,
            "projection_margin_bytes": 43939744,
            "rss_separate_and_diagnostic": True}


def run_checked(command, env=None, expect_success=True):
    process = subprocess.run(command, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=90, check=False)
    if expect_success:
        assert process.returncode == 0, (process.stdout, process.stderr)
    else:
        assert process.returncode != 0, (process.stdout, process.stderr)
    return process


def compile_and_test():
    for path in (DRIVER, M896, TESTS):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    process = run_checked([str(PYTHON), "-m", "pytest", "-p", "no:cacheprovider",
                           "-q", str(TESTS)], env=env)
    assert "11 passed" in process.stdout
    run_checked(["/bin/bash", "-n", str(RUNNER)], env=env)
    return {"python_compile_files": 3, "pytest": "11 passed",
            "shell_syntax": "PASS"}


def no_work_and_pin_attacks(driver):
    assert not canonical_collisions()
    base = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    good = dict(base, M900_EXPECTED_RUNNER_SHA256=EXPECTED["runner"],
                M900_EXPECTED_RELEASE_SHA256=EXPECTED["release"])
    ok = run_checked([str(RUNNER), "--dry-run-no-work"], env=good)
    assert "PASS_M900_NO_WORK_DRY_RUN__NO_FILES_CREATED__NO_ATTEMPT" in ok.stdout
    attacks = []
    bad_envs = [
        ("missing_all", dict(base)),
        ("runner_malformed", dict(base, M900_EXPECTED_RUNNER_SHA256="bad",
                                  M900_EXPECTED_RELEASE_SHA256=EXPECTED["release"])),
        ("runner_wrong", dict(base, M900_EXPECTED_RUNNER_SHA256="0" * 64,
                              M900_EXPECTED_RELEASE_SHA256=EXPECTED["release"])),
        ("release_missing", dict(base, M900_EXPECTED_RUNNER_SHA256=EXPECTED["runner"])),
        ("release_malformed", dict(base, M900_EXPECTED_RUNNER_SHA256=EXPECTED["runner"],
                                   M900_EXPECTED_RELEASE_SHA256="bad")),
        ("release_wrong", dict(base, M900_EXPECTED_RUNNER_SHA256=EXPECTED["runner"],
                               M900_EXPECTED_RELEASE_SHA256="0" * 64)),
        ("bash_env", dict(good, BASH_ENV="/definitely/forbidden")),
    ]
    for label, env in bad_envs:
        run_checked([str(RUNNER), "--dry-run-no-work"], env=env,
                    expect_success=False)
        attacks.append(label)
    run_checked([str(RUNNER), "--dry-run-no-work", "extra"], env=good,
                expect_success=False)
    attacks.append("extra_argument")
    for pins in (("", "", "", ""),
                 ("bad", "bad", "bad", "bad"),
                 ("0" * 64, "0" * 64, EXPECTED["release"], EXPECTED["runner"]),
                 ("0" * 64, "0" * 64, "0" * 64, EXPECTED["runner"])):
        try:
            driver.validate_final_hammer(*pins)
            raise AssertionError("invalid final-hammer pins accepted")
        except driver.Failure:
            attacks.append("driver_final_hammer_pin_rejected")
    assert not canonical_collisions()
    return {"no_work_status": "PASS_M900_NO_WORK_DRY_RUN__NO_FILES_CREATED__NO_ATTEMPT",
            "runner_and_driver_pin_negatives": len(attacks),
            "canonical_namespace_absent_before_after": True}


def strict_json_attacks():
    with tempfile.TemporaryDirectory(prefix="m901_json_attack_", dir=str(REVIEW)) as tmp_name:
        tmp = Path(tmp_name)
        labels = []
        for label, text in (("duplicate", '{"a":1,"a":2}\n'),
                            ("nan", '{"a":NaN}\n'),
                            ("infinity", '{"a":Infinity}\n'),
                            ("minus_infinity", '{"a":-Infinity}\n')):
            path = tmp / (label + ".json")
            path.write_text(text, encoding="utf-8")
            try:
                strict_json(path)
                raise AssertionError(label + " accepted")
            except ValueError:
                labels.append(label)
    return labels


def static_runner_driver_audit(driver):
    runner = RUNNER.read_text(encoding="utf-8")
    source = DRIVER.read_text(encoding="utf-8")
    assert runner.index("m900_resource_gate\n") < runner.index('m900_phase="CONSUME_ONE_WAY_ATTEMPT"')
    assert runner.index('m900_phase="CONSUME_ONE_WAY_ATTEMPT"') < runner.index("--consume-attempt")
    assert "sleep 1" in runner
    assert 'm900_over_runtime}" -ge 3' in runner
    assert 'm900_over_resource}" -ge 3' in runner
    assert 'm900_over_state}" -ge 3' in runner
    assert "kill -TERM" in runner
    assert "child_rss_kib" in runner and "counted_state_bytes" in runner
    assert "m900_rss" in runner and "m900_counted" in runner
    assert '"${m900_elapsed_ms}" -gt 9321' in runner
    assert '"${m900_counted}" -gt 536870912' in runner
    assert "RENAME_NOREPLACE = 1" in source and "renameat2" in source
    assert "source.parent.resolve() == destination.parent.resolve()" in source
    assert "input_transaction_objects_excluded_from_counted_state" in source
    assert "serialized_or_compressed_file_size_used" in source
    assert "process_max_rss_kib_diagnostic_only" in source
    assert 'records[0]' in source
    assert '"A1_OSG", 0, oracles' in source
    assert 'int(record["module_index"]) == 0' in source
    assert 'int(record["sample_id"]) == 0' in source
    assert "full_population" in source and "production" in source
    assert "cycles_or_speedup_citable" in source and "paper_citable" in source
    assert not any(token in runner for token in (
        "ssh ", "scp ", "rsync ", "vcs ", "dc_shell", "pt_shell", "fm_shell",
        "nvidia-smi", "curl ", "wget "))
    with tempfile.TemporaryDirectory(prefix="m901_publish_attack_", dir=str(REVIEW)) as tmp_name:
        tmp = Path(tmp_name)
        left = tmp / "left"
        right = tmp / "right"
        left.write_text("left\n", encoding="utf-8")
        right.write_text("right\n", encoding="utf-8")
        try:
            driver._rename_noreplace(left, right)
            raise AssertionError("NOREPLACE overwrote existing destination")
        except driver.Failure:
            pass
        assert left.read_text() == "left\n" and right.read_text() == "right\n"
        target = tmp / "target"
        driver._rename_noreplace(left, target)
        assert not left.exists() and target.read_text() == "left\n"
    counters = [0, 1, 2]
    assert [value + 1 >= 3 for value in counters] == [False, False, True]
    return {"resource_gate_precedes_attempt": True,
            "one_second_snapshot_monitor": True,
            "three_consecutive_runtime_resource_state_termination": True,
            "rss_and_counted_state_columns_separate": True,
            "same_directory_renameat2_noreplace_attacked": True,
            "one_fixed_d0_a1_t0_row_reachable": True,
            "no_eda_gpu_remote_network_mode": True,
            "success_failure_nonproduction_noncitable": True}


def resource_snapshot():
    stat = os.statvfs(RESULTS)
    free_kib = stat.f_bavail * stat.f_frsize // 1024
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "CommitLimit:", "Committed_AS:")):
            key, value, _ = line.split()
            values[key.rstrip(":")] = int(value)
    headroom = values["CommitLimit"] - values["Committed_AS"]
    assert free_kib >= 2097152
    assert values["MemAvailable"] >= 100663296
    assert headroom >= 100663296
    return {"free_disk_kib": free_kib,
            "mem_available_kib": values["MemAvailable"],
            "commit_headroom_kib": headroom,
            "gates_pass_now_diagnostic_only": True}


def main():
    assert not canonical_collisions()
    driver = load_driver()
    identities = verify_identities()
    release = verify_release_semantics()
    static_tests = compile_and_test()
    strict_attacks = strict_json_attacks()
    no_work = no_work_and_pin_attacks(driver)
    audit = static_runner_driver_audit(driver)
    resources = resource_snapshot()
    validation = driver.validate_release(RELEASE, require_unconsumed=True)
    assert validation["status"] == "PASS_M900_INERT_RELEASE_IDENTITY__NO_WORK_NO_ATTEMPT"
    assert not canonical_collisions()
    assert sha256(DOCS359) == EXPECTED["docs359"]
    output = {
        "schema": "m901_m900_m896_decoder_run_gtls_fullrow_final_hammer_output_v1",
        "status": "PASS_M901_BOUNDED_NO_WORK_HAMMER",
        "checks_passed": 62,
        "identities": identities,
        "release": release,
        "static_and_directed_tests": static_tests,
        "strict_json_attacks_rejected": strict_attacks,
        "no_work_and_pin_attacks": no_work,
        "runner_driver_audit": audit,
        "resource_snapshot_diagnostic_only": resources,
        "driver_static_validation": validation,
        "full_row_enumerated": False,
        "attempt_or_result_or_failure_created": False,
        "vcs_eda_license_gpu_remote_network_invoked": False,
        "docs359_sha256_after": EXPECTED["docs359"],
    }
    destination = REVIEW / "independent_hammer_output.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
