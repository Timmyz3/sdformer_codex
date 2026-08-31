"""Independent no-work final-launch audit for M925 decoder GTLS R2.

This reviewer never invokes the no-argument runner, never enumerates the full
first row, and never creates an M925 attempt, result, stage, log, or failure.
Only exact identity/seal checks, static authority checks, source validation,
the explicit --dry-run-no-work path, and a rejected-argument test are allowed.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


sys.dont_write_bytecode = True

REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
RESULTS = HW / "results"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
SETSID = Path("/usr/bin/setsid")
DRIVER = HW / "system_simulator/scripts/execute_m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2.py"
RUNNER = HW / "system_simulator/scripts/run_m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_one_shot.sh"
SOURCE_CONTRACT = HW / "contracts/m925_m896_decoder_run_gtls_full_first_row_exact_scalability_source_contract_r1_20260829.json"
RELEASE = HW / "contracts/m927_m925_decoder_run_gtls_full_first_row_exact_scalability_release_r1_20260829.json"
M896 = HW / "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py"
M902 = HW / "reviews/m902_m900_decoder_fullrow_failure_audit_r1_20260829"
M930 = HW / "reviews/m930_m925_decoder_gtls_r2_source_fresh_hammer_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "setsid": "827259531e3511bcc704143690d8a3afec043d24a7922bf3ebfacf917cd7e100",
    "release": "bb154345d648a5f8295eda013fdfccd7ab17d3c9023f50be1408b013b303a9d7",
    "release_inner": "c93c3f0566196dd91c1e520c85bea8c83949612bf90f17423cd517fe84a931e6",
    "release_outer": "1333ab4d1ee348d6cf4b2c42f47d45bd0a009b584fc2045d54c23589d2e3cce7",
    "runner": "b8f0dae1dd07423099d9d82cd3646b9343aa1623d9e39e9239ff30959cd18f05",
    "driver": "e02d3c0dc8b47234b3c6b065ccb30f52d8684b3813fa7b2753a6eab2c2df6806",
    "source_contract": "7140d6cc7aa80f1f6016828d325f719abad594aff66cb13316564ef93256032e",
    "m896": "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39",
    "m902_review": "6b25dae1ed54fb7b591472a3fd6b6ac9932772e13e60f4395895fd3526e2fc3b",
    "m902_manifest": "e6f1fe535227be4146b3563b481f7d3504b76352b93e585cff49b879fbb4fad9",
    "m902_outer": "98b3c505534fec3904d2fb327c4050c6fc3ab3a4e975ca96a0fd7ec8ef91d4da",
    "m930_review": "03c12f452e39fd5864072b2a7e37e2d6579e5d60353892e20299a4ef9228e30c",
    "m930_manifest": "c17b95041b5b38962d7dc2a0b2197d015ad00b013c8480443a874e8e71daddbb",
    "m930_outer": "604dbf09c2416f2ccdb129d2733a16492dd55c0e363b8e0c2805ccd6d5bdaf74",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

ATTEMPT_NAME = ".m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_attempt_consumed"
RESULT_NAME = "m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_20260829"
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
        assert name and name not in (".", "..") and "/" not in name and "\\" not in name
        assert name not in rows
        member = directory / name
        assert member.is_file() and not member.is_symlink()
        assert sha256(member) == digest
        if member.suffix == ".json":
            strict_json(member)
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
    names = []
    for entry in os.scandir(RESULTS):
        if (entry.name in (ATTEMPT_NAME, RESULT_NAME) or
                entry.name.startswith(ATTEMPT_NAME + ".stage.") or
                entry.name.startswith(RESULT_NAME + ".stage.") or
                entry.name.startswith(RESULT_NAME + ".worker_") or
                entry.name.startswith(RESULT_NAME + ".runtime_resource_") or
                entry.name.startswith(FAILURE_PREFIX)):
            names.append(entry.name)
    return sorted(names)


def exact_identities():
    paths = {
        PYTHON: EXPECTED["python"], SETSID: EXPECTED["setsid"],
        RELEASE: EXPECTED["release"], RUNNER: EXPECTED["runner"],
        DRIVER: EXPECTED["driver"], SOURCE_CONTRACT: EXPECTED["source_contract"],
        M896: EXPECTED["m896"], M902 / "review.json": EXPECTED["m902_review"],
        M930 / "review.json": EXPECTED["m930_review"], DOCS359: EXPECTED["docs359"],
    }
    for path, digest in paths.items():
        assert path.is_file() and not path.is_symlink() and sha256(path) == digest
    release_seal = verify_file_seal(RELEASE)
    assert release_seal["inner_seal_file_sha256"] == EXPECTED["release_inner"]
    assert release_seal["outer_seal_file_sha256"] == EXPECTED["release_outer"]
    for path in (RUNNER, DRIVER, SOURCE_CONTRACT, M896):
        verify_file_seal(path)
    assert verify_directory_seal(M902) == {
        "manifest_sha256": EXPECTED["m902_manifest"],
        "outer_seal_file_sha256": EXPECTED["m902_outer"]}
    assert verify_directory_seal(M930) == {
        "manifest_sha256": EXPECTED["m930_manifest"],
        "outer_seal_file_sha256": EXPECTED["m930_outer"]}
    return {"exact_sha_bindings": True, "applicable_double_seals": True}


def release_semantics():
    release = strict_json(RELEASE)
    source = strict_json(SOURCE_CONTRACT)
    m902 = strict_json(M902 / "review.json")
    m930 = strict_json(M930 / "review.json")
    assert release["schema"] == "m927_m925_decoder_run_gtls_full_first_row_exact_scalability_release_v1"
    assert release["status"] == "INERT_R2_RELEASE__PENDING_FRESH_M928_FINAL_HAMMER"
    assert release["release"] is True and release["launch_now"] is False
    assert type(release["max_attempts"]) is int and release["max_attempts"] == 1
    assert release["canonical"] == {
        "result": "hw_autoresearch_nts07/results/" + RESULT_NAME,
        "attempt": "hw_autoresearch_nts07/results/" + ATTEMPT_NAME,
        "failed_or_incomplete_prefix": "hw_autoresearch_nts07/results/" + FAILURE_PREFIX,
    }
    assert release["workload"] == {
        "identity": "M854_FIRST_D0_A1_T0",
        "population": "M686_ZURICH_CITY_09_A_S10",
        "record_ordinal": 0, "module_index": 0, "sample_id": 0,
        "configuration": "A1_OSG", "timestep": 0,
        "future_rows_maximum": 1, "full_population_rows": 0,
    }
    assert release["scientific_threshold"] == {
        "seconds": 9.320783571209759,
        "historical_status": "FAILED_BY_M900__NOT_RETRIED_BY_R2",
        "acceptance_gate_for_r2": False,
    }
    assert math.isfinite(release["scientific_threshold"]["seconds"])
    assert release["operational_safety_timeout_seconds"] == 2715
    assert release["future_gate"] == {
        "fresh_m928_final_launch_hammer_required": True,
        "full_first_row_before_m928": False,
        "at_most_one_no_argument_invocation_after_m928": True,
        "full_population": False,
        "production": False,
    }
    binding = release["source_binding"]
    assert binding == {
        "m925_contract_sha256": EXPECTED["source_contract"],
        "m925_runner_sha256": EXPECTED["runner"],
        "m925_driver_sha256": EXPECTED["driver"],
        "m896_source_sha256": EXPECTED["m896"],
        "m902_review_sha256": EXPECTED["m902_review"],
        "m930_source_hammer_review_sha256": EXPECTED["m930_review"],
        "m930_source_hammer_manifest_sha256": EXPECTED["m930_manifest"],
        "m930_source_hammer_outer_seal_file_sha256": EXPECTED["m930_outer"],
    }
    assert release["docs359_sha256"] == EXPECTED["docs359"]
    assert release["claim_boundary"]["inert_release_only"] is True
    for key in ("full_first_row", "production", "full_population", "decoder_complete",
                "cycles_or_speedup_citable", "system_speedup", "energy",
                "paper_ppa_ready", "paper_citable", "scientific_100x_threshold_retried"):
        assert release["claim_boundary"][key] is False
    assert source["timing_contract"]["scientific_100x_hypothesis_already_failed_by_m900"] is True
    assert source["timing_contract"]["r2_objective_is_100x_retry"] is False
    assert source["timing_contract"]["operational_safety_timeout_seconds"] == 2715
    assert source["future_gate_sequence"]["m928"].startswith("fresh independent final-launch hammer")
    assert m902["status"] == "PASS100_FAILURE_ROOT_CAUSE_IDENTIFIED__M900_100X_NO_GO__FRESH_R2_SCALABILITY_DIAGNOSTIC_CONDITIONAL_GO"
    assert m902["score"] == 100 and m902["runtime_audit"]["scientific_100x_gate_failed"] is True
    assert m902["decision"]["m900_100x_retry"] == "NO_GO"
    assert m930["status"] == "PASS100_M925_R2_SOURCE_PROCESS_CONTROL__ONLY_FRESH_M927_INERT_RELEASE_AUTHOR_AUTHORIZED"
    assert m930["score"] == 100 and m930["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
    assert m930["authorization"]["fresh_m928_final_launch_hammer_required"] is True
    assert m930["authorization"]["at_most_one_full_first_row_diagnostic_after_exact_chain"] is True
    assert m930["authorization"]["full_population"] is False
    assert m930["authorization"]["production"] is False
    rejected = []
    for label, payload in (("duplicate", '{"x":1,"x":2}'),
                           ("nan", '{"x":NaN}'),
                           ("positive_infinity", '{"x":Infinity}'),
                           ("negative_infinity", '{"x":-Infinity}')):
        try:
            json.loads(payload, object_pairs_hook=reject_duplicates,
                       parse_constant=reject_constant)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError("strict JSON accepted " + label)
    assert rejected == ["duplicate", "nan", "positive_infinity", "negative_infinity"]
    return {
        "release_inert_before_this_hammer": True,
        "single_exact_diagnostic_only": True,
        "scientific_100x_failed_not_retried": True,
        "operational_timeout_seconds": 2715,
        "cycles_speedup_citable": False,
        "strict_json_negative_tests": rejected,
    }


def static_one_shot_audit():
    runner = RUNNER.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    required_runner = (
        '[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "--dry-run-no-work" ) ]]',
        "M925 formal diagnostic requires future M927/M928 caller pins",
        "--validate-formal-preflight", "--consume-attempt", "--run-full-first-row",
        '"${m925_setsid}" --wait /usr/bin/env -i',
        'if [[ "${m925_elapsed_ms}" -gt 2715000 ]]',
        'if [[ "${m925_over_timeout}" -ge 3 ]]',
        'kill -TERM -- "-${m925_worker_pgrp}"',
        'kill -KILL -- "-${m925_worker_pgrp}"',
        "PASS_M925_ONE_R2_EXACT_SCALABILITY_DIAGNOSTIC__FRESH_RESULT_HAMMER_REQUIRED",
    )
    for token in required_runner:
        assert token in runner
    dry = runner.index('if [[ "$#" -eq 1 ]]')
    formal = runner.index("--validate-formal-preflight", dry)
    consume = runner.index("--consume-attempt", formal)
    launch = runner.index('"${m925_setsid}" --wait /usr/bin/env -i', consume)
    assert dry < formal < consume < launch
    assert runner.count("--consume-attempt") == 1
    assert runner.count('"${m925_setsid}" --wait /usr/bin/env -i') == 1
    required_driver = (
        'FINAL_HAMMER_DIR = HW / "reviews/m928_m927_m925_decoder_gtls_r2_final_launch_hammer_r1_20260829"',
        'FINAL_HAMMER_STATUS = "PASS100_M925_R2_EXACT_SCALABILITY_FINAL_LAUNCH__ONE_DIAGNOSTIC_AUTHORIZED"',
        "OPERATIONAL_SAFETY_TIMEOUT_SECONDS = 2715",
        '"scientific_100x_hypothesis_already_failed_by_m900": True',
        'review.get("authorization", {}).get(\n                "one_full_first_row_exact_scalability_diagnostic") is True',
        'review.get("authorization", {}).get("full_population") is False',
        'review.get("authorization", {}).get("production") is False',
    )
    for token in required_driver:
        assert token in driver
    return {
        "no_argument_only_or_explicit_no_work": True,
        "formal_preflight_precedes_attempt": True,
        "attempt_precedes_worker": True,
        "private_setsid_wait_worker": True,
        "2715_second_three_sample_operational_guard": True,
        "final_review_exactly_bound_by_driver": True,
    }


def no_work_tests():
    before = canonical_collisions()
    assert before == []
    base_env = {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "M925_EXPECTED_RUNNER_SHA256": EXPECTED["runner"],
        "M925_EXPECTED_CONTRACT_SHA256": EXPECTED["source_contract"],
    }
    source = subprocess.run(
        [str(PYTHON), str(DRIVER), "--validate-source-contract",
         "--source-contract", str(SOURCE_CONTRACT)], env=base_env,
        text=True, capture_output=True, timeout=60)
    assert source.returncode == 0, (source.stdout, source.stderr)
    source_json = json.loads(source.stdout)
    assert source_json["status"] == "PASS_M925_SOURCE_ONLY_IDENTITY__NO_WORK_NO_ATTEMPT"
    assert canonical_collisions() == []
    dry = subprocess.run([str(RUNNER), "--dry-run-no-work"], env=base_env,
                         text=True, capture_output=True, timeout=60)
    assert dry.returncode == 0, (dry.stdout, dry.stderr)
    assert "PASS_M925_NO_WORK_DRY_RUN__NO_FILES_NO_ATTEMPT" in dry.stdout
    assert canonical_collisions() == []
    rejected = subprocess.run([str(RUNNER), "--not-authorized"], env=base_env,
                              text=True, capture_output=True, timeout=30)
    assert rejected.returncode == 3
    assert "accepts no arguments or --dry-run-no-work only" in rejected.stderr
    after = canonical_collisions()
    assert after == []
    return {
        "namespace_before": before,
        "source_validation": source_json["status"],
        "dry_run_token": "PASS_M925_NO_WORK_DRY_RUN__NO_FILES_NO_ATTEMPT",
        "rejected_argument_return_code": rejected.returncode,
        "namespace_after": after,
        "no_argument_invocations": 0,
        "full_first_row_executed": False,
    }


def main():
    assert Path(sys.executable) == PYTHON
    assert canonical_collisions() == []
    output = {
        "status": "PASS_M928_INDEPENDENT_FINAL_LAUNCH_MECHANICAL_CHECKS",
        "identity": exact_identities(),
        "release": release_semantics(),
        "one_shot": static_one_shot_audit(),
        "no_work": no_work_tests(),
        "claim_boundary": {
            "full_first_row": False, "full_population": False,
            "production": False, "scientific_100x_threshold_retried": False,
            "cycles_or_speedup_citable": False, "system_speedup": False,
            "vcs_eda_license_gpu_remote_network": False,
        },
    }
    assert canonical_collisions() == []
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
