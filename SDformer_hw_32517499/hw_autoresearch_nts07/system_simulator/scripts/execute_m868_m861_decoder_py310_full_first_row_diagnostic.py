"""M868 Python-3.10-only, one-shot M861 full-first-row diagnostic.

This additive wrapper does not change M861, M785, or M768 semantics.  It
exists only to bind one future D0/A1/t0 aggregate diagnostic to the exact
Python interpreter under which M861 has been demonstrated.  The diagnostic
is nonproduction and its cycle count and any derived speedup are noncitable.

There is intentionally no shebang.  Every invocation must name the pinned
absolute Python 3.10 interpreter explicitly.
"""

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import resource as process_resource
import shutil
import stat
import sys
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple


PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
PYTHON_VERSION = "3.10.18"


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_runtime_early() -> None:
    executable = Path(sys.executable)
    require(executable == PYTHON_PATH,
            "M868 forbids ambient python3, shebang, or PATH fallback")
    require(PYTHON_PATH.is_file() and not PYTHON_PATH.is_symlink(),
            "M868 Python interpreter is absent, nonregular, or symlinked")
    require(sha256(PYTHON_PATH) == PYTHON_SHA256,
            "M868 Python interpreter SHA drift")
    require(platform.python_version() == PYTHON_VERSION,
            "M868 Python interpreter version drift")


_verify_runtime_early()

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M861_PATH = HERE / "analyze_m861_decoder_streaming_event_sweep.py"
M861_SHA256 = "f72ed3b820051d624699152b784c05fa674106556ab73f452a2cf96a9f72d7a4"
M861_TESTS = HW / "system_simulator/tests/test_m861_decoder_streaming_event_sweep.py"
M861_TESTS_SHA256 = "cd9cb5ac05c982511d0d2a51843e7ea4f4d93b752eebf82cd60e9b011dfa76ab"
M861_CONTRACT = HW / "contracts/m861_decoder_streaming_event_sweep_candidate_r1_20260829.json"
M861_CONTRACT_SHA256 = "5ca88752677ea82557ebf62032b373de086dde202614df3949a3f11f79a1e2f2"
M865_DIR = HW / "reviews/m865_m861_decoder_streaming_event_sweep_source_hammer_r1_20260829"
M865_REVIEW_SHA256 = "68ac2981629250346fb7ec30c376b2d1707de5f3d0cde2d7badf1431be4737fa"
M865_MANIFEST_SHA256 = "7bc53c9cb0b8186df89ec7a87c0e069f2fc0680e52d334db639467b54ae80726"
M865_OUTER_SEAL_FILE_SHA256 = "e5e3d7a7d90fc541f673955e1070a47c2cc147261812235a92e9267d71b54c8b"
M861_HANDOFF_DIR = HW / "reviews/m861_decoder_streaming_event_sweep_source_author_handoff_r1_20260829"
M861_HANDOFF_MANIFEST_SHA256 = "377ee3346d703efa5e829cb69efbe342f4f7023d918fc3c122ad18c5bbf976fc"
M861_HANDOFF_OUTER_SHA256 = "5eba82bd8d9138667d988a294d493305d6ab1feb41b14409c2275681f81381ee"
M862_REQUEST_DIR = HW / "reviews/m862_m861_decoder_streaming_event_sweep_source_fresh_hammer_REQUEST_r1_20260829"
M862_REQUEST_MANIFEST_SHA256 = "3970fc65210762abcd350f9b65113486e7e31b47f3250638c1616f2bb25924d7"
M862_REQUEST_OUTER_SHA256 = "86b949de2aede7685d62e886351c1028f0cf1c7987b09bb5636b57f09d0a44b5"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CANDIDATE = HW / "contracts/m868_m861_decoder_py310_full_first_row_diagnostic_candidate_r1_20260829.json"
CANDIDATE_SCHEMA = "m868_m861_decoder_py310_full_first_row_diagnostic_candidate_v1"
HAMMER_DIR = HW / "reviews/m869_m868_decoder_py310_full_first_row_source_hammer_r1_20260829"
HAMMER_STATUS = "PASS100_M868_PY310_FULL_FIRST_ROW_SOURCE__AUTHORIZE_EXACTLY_ONE_NONPRODUCTION_DIAGNOSTIC"
RESULT = HW / "results/m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829"
ATTEMPT = HW / "results/.m868_m861_decoder_py310_full_first_row_diagnostic_r1_attempt_consumed"
FAILURE_PREFIX = "m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829.failed_or_incomplete."
EXPECTED_COMPRESSED = 9582057
EXPECTED_EXPANDED = 38672612
RENAME_NOREPLACE = 1


def _load_exact(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink(), name + " absent")
    require(sha256(path) == expected, name + " SHA drift")
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M861 = _load_exact(M861_PATH, M861_SHA256, "m868_frozen_m861")
M785 = M861.M785


def strict_json(path: Path) -> object:
    return M785.strict_json(Path(path))


def verify_sealed(path: Path) -> Dict[str, str]:
    try:
        return M785.verify_sealed_directory(Path(path))
    except M785.Failure as error:
        raise Failure(str(error)) from error


def regular_exact(path: Path, expected: str, label: str) -> None:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), label + " absent")
    require(sha256(path) == expected, label + " SHA drift")


def _verify_authority(directory: Path, manifest_sha: str,
                      outer_sha: str, label: str) -> Dict[str, str]:
    identity = verify_sealed(directory)
    require(identity == {
        "manifest_sha256": manifest_sha,
        "outer_seal_file_sha256": outer_sha,
    }, label + " sealed identity drift")
    return identity


def _canonical_relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _candidate_paths() -> Dict[str, str]:
    return {
        "result": _canonical_relative(RESULT),
        "attempt": _canonical_relative(ATTEMPT),
        "failed_or_incomplete_prefix":
            _canonical_relative(RESULT.parent) + "/" + FAILURE_PREFIX,
    }


def validate_candidate(candidate_path: Path,
                       require_unconsumed: bool = True) -> Dict[str, object]:
    candidate_path = Path(candidate_path)
    require(candidate_path.resolve() == CANDIDATE.resolve(),
            "M868 candidate canonical path drift")
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict) and
            candidate.get("schema") == CANDIDATE_SCHEMA,
            "M868 candidate schema drift")
    require(candidate.get("status") ==
            "SOURCE_ONLY_PY310_FULL_FIRST_ROW_DIAGNOSTIC__FRESH_PASS100_REQUIRED" and
            candidate.get("launch_now") is False and
            candidate.get("max_attempts") == 1,
            "M868 candidate authority drift")
    require(candidate.get("canonical") == _candidate_paths(),
            "M868 canonical namespace drift")
    require(candidate.get("interpreter") == {
        "absolute_path": str(PYTHON_PATH),
        "sha256": PYTHON_SHA256,
        "version": PYTHON_VERSION,
        "ambient_python3_allowed": False,
        "python_shebang_allowed": False,
        "path_fallback_allowed": False,
    }, "M868 interpreter contract drift")
    require(candidate.get("workload") == {
        "identity": "M854_FIRST_D0_A1_T0",
        "population": "M686_ZURICH_CITY_09_A_S10",
        "record_ordinal": 0,
        "module_index": 0,
        "sample_id": 0,
        "configuration": "A1_OSG",
        "timestep": 0,
        "expected_compressed_transaction_count": EXPECTED_COMPRESSED,
        "expected_expanded_request_count": EXPECTED_EXPANDED,
        "rows_authorized": 1,
        "population_rows_authorized": 0,
    }, "M868 first-row identity drift")
    for label, entry in candidate["source_identity"].items():
        regular_exact(HW / entry["path"], entry["sha256"], label)
    require(candidate["source_identity"]["m861_analyzer"]["sha256"] ==
            M861_SHA256, "M861 analyzer identity drift")
    require(candidate["source_identity"]["m861_tests"]["sha256"] ==
            M861_TESTS_SHA256, "M861 tests identity drift")
    require(candidate["source_identity"]["m861_contract"]["sha256"] ==
            M861_CONTRACT_SHA256, "M861 contract identity drift")
    _verify_authority(M861_HANDOFF_DIR, M861_HANDOFF_MANIFEST_SHA256,
                      M861_HANDOFF_OUTER_SHA256, "M861 handoff")
    _verify_authority(M862_REQUEST_DIR, M862_REQUEST_MANIFEST_SHA256,
                      M862_REQUEST_OUTER_SHA256, "M862 request")
    _verify_authority(M865_DIR, M865_MANIFEST_SHA256,
                      M865_OUTER_SEAL_FILE_SHA256, "M865 failure hammer")
    regular_exact(M865_DIR / "review.json", M865_REVIEW_SHA256,
                  "M865 failure review")
    m865 = strict_json(M865_DIR / "review.json")
    require(m865["status"] ==
            "NO_GO_M861_FULL_FIRST_ROW_GATE__P1_1__ADDITIVE_EXPLICIT_PY310_ONLY_IDENTITY_REQUIRED" and
            m865["severity_counts"] == {"p0": 0, "p1": 1, "p2": 0} and
            m865["finding"]["m861_scheduling_semantics_implicated"] is False,
            "M865 successor authority drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    require(candidate["claim_boundary"] == {
        "diagnostic_only": True,
        "full_first_row_completed": False,
        "full_population": False,
        "production_cycles": False,
        "production_speedup": False,
        "decoder_complete": False,
        "table_a": False,
        "paper_citable": False,
        "vcs_dc_pt_fm_eda_license_gpu_remote_training": False,
        "docs359_modified": False,
    }, "M868 claim boundary drift")
    if require_unconsumed:
        collisions = scan_collisions()
        require(not collisions, "M868 canonical/collision namespace occupied: " +
                repr(collisions))
    return {
        "status": "PASS_M868_SOURCE_CANDIDATE__NO_WORK_NO_ATTEMPT",
        "candidate_sha256": sha256(candidate_path),
        "interpreter_sha256": sha256(PYTHON_PATH),
        "interpreter_version": platform.python_version(),
        "full_first_row_run": False,
        "attempt_consumed": False,
    }


def scan_collisions() -> Tuple[str, ...]:
    results = RESULT.parent
    require(results.is_dir() and not results.is_symlink(),
            "results directory is absent or symlinked")
    names = []
    for entry in os.scandir(results):
        if (entry.name == RESULT.name or entry.name == ATTEMPT.name or
                entry.name.startswith(RESULT.name + ".stage.") or
                entry.name.startswith(ATTEMPT.name + ".stage.") or
                entry.name.startswith(FAILURE_PREFIX)):
            names.append(entry.name)
    return tuple(sorted(names))


def validate_hammer(expected_review_sha256: str,
                    expected_outer_sha256: str) -> Dict[str, object]:
    require(len(expected_review_sha256) == 64 and
            len(expected_outer_sha256) == 64,
            "M868 hammer caller pins malformed")
    identity = verify_sealed(HAMMER_DIR)
    require(identity["outer_seal_file_sha256"] == expected_outer_sha256,
            "M868 hammer outer-seal caller pin drift")
    review_path = HAMMER_DIR / "review.json"
    regular_exact(review_path, expected_review_sha256, "M868 hammer review")
    review = strict_json(review_path)
    require(review.get("status") == HAMMER_STATUS and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
            "M868 hammer did not grant the exact one-row gate")
    require(review.get("decision", {}).get(
        "exactly_one_nonproduction_full_first_row_diagnostic_authorized") is True and
        review.get("decision", {}).get("full_population_authorized") is False and
        review.get("decision", {}).get("cycles_or_speedup_citable") is False,
        "M868 hammer decision boundary drift")
    return {
        "review_sha256": expected_review_sha256,
        "manifest_sha256": identity["manifest_sha256"],
        "outer_seal_file_sha256": expected_outer_sha256,
    }


def _safe_basename(name: str, label: str) -> None:
    require(name and name not in (".", "..") and "/" not in name and
            "\x00" not in name, label + " basename malformed")


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(source.parent.resolve() == destination.parent.resolve(),
            "M868 publication must remain in one directory")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    parent_fd = os.open(str(source.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        rc = renameat2(parent_fd, os.fsencode(source.name), parent_fd,
                       os.fsencode(destination.name), RENAME_NOREPLACE)
    finally:
        os.close(parent_fd)
    if rc != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise Failure("M868 no-replace publication collision")
        raise Failure("M868 renameat2 failed: " + os.strerror(number))


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    fd = os.open(str(path), flags | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def seal_directory(directory: Path, members: Sequence[str]) -> Dict[str, str]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "M868 seal target is not a regular directory")
    require("SHA256SUMS" not in members and
            "SHA256SUMS.seal.sha256" not in members,
            "M868 seal members include seal files")
    lines = []
    for name in sorted(members):
        _safe_basename(name, "sealed member")
        path = directory / name
        require(path.is_file() and not path.is_symlink(),
                "M868 seal member absent: " + name)
        lines.append(sha256(path) + "  " + name + "\n")
    _write_exclusive(directory / "SHA256SUMS", "".join(lines).encode("ascii"))
    outer = sha256(directory / "SHA256SUMS") + "  SHA256SUMS\n"
    _write_exclusive(directory / "SHA256SUMS.seal.sha256",
                     outer.encode("ascii"))
    return verify_sealed(directory)


def consume_attempt(candidate_path: Path, runner_path: Path,
                    expected_runner_sha256: str,
                    hammer_review_sha256: str,
                    hammer_outer_sha256: str,
                    stage_basename: str) -> Dict[str, object]:
    candidate = validate_candidate(candidate_path, require_unconsumed=True)
    runner_path = Path(runner_path)
    regular_exact(runner_path, expected_runner_sha256, "M868 runner")
    hammer = validate_hammer(hammer_review_sha256, hammer_outer_sha256)
    _safe_basename(stage_basename, "attempt stage")
    require(stage_basename.startswith(ATTEMPT.name + ".stage."),
            "M868 attempt stage namespace drift")
    stage = ATTEMPT.parent / stage_basename
    require(not stage.exists() and not stage.is_symlink(),
            "M868 attempt stage collision")
    os.mkdir(stage, 0o700)
    published = False
    try:
        receipt = {
            "schema": "m868_m861_decoder_full_first_row_attempt_v1",
            "status": "CONSUMED_IMMEDIATELY_BEFORE_M868_FULL_FIRST_ROW_DIAGNOSTIC",
            "max_attempts": 1,
            "candidate_sha256": candidate["candidate_sha256"],
            "runner_sha256": expected_runner_sha256,
            "hammer": hammer,
            "interpreter_path": str(PYTHON_PATH),
            "interpreter_sha256": PYTHON_SHA256,
            "interpreter_version": PYTHON_VERSION,
            "workload_identity": "M854_FIRST_D0_A1_T0",
            "production_authorized": False,
            "cycles_or_speedup_citable": False,
        }
        _write_exclusive(stage / "attempt.json", (json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        identity = seal_directory(stage, ("attempt.json",))
        _rename_noreplace(stage, ATTEMPT)
        published = True
        require(verify_sealed(ATTEMPT) == identity,
                "M868 published attempt seal drift")
        return {
            "status": receipt["status"],
            "attempt_manifest_sha256": identity["manifest_sha256"],
            "attempt_outer_seal_file_sha256":
                identity["outer_seal_file_sha256"],
        }
    finally:
        if not published and stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)


def validate_attempt(candidate_path: Path, runner_path: Path,
                     expected_runner_sha256: str,
                     hammer_review_sha256: str,
                     hammer_outer_sha256: str) -> Dict[str, object]:
    candidate = validate_candidate(candidate_path, require_unconsumed=False)
    regular_exact(runner_path, expected_runner_sha256, "M868 runner")
    hammer = validate_hammer(hammer_review_sha256, hammer_outer_sha256)
    identity = verify_sealed(ATTEMPT)
    receipt = strict_json(ATTEMPT / "attempt.json")
    require(receipt.get("schema") ==
            "m868_m861_decoder_full_first_row_attempt_v1" and
            receipt.get("status") ==
            "CONSUMED_IMMEDIATELY_BEFORE_M868_FULL_FIRST_ROW_DIAGNOSTIC" and
            receipt.get("max_attempts") == 1 and
            receipt.get("candidate_sha256") == candidate["candidate_sha256"] and
            receipt.get("runner_sha256") == expected_runner_sha256 and
            receipt.get("hammer") == hammer,
            "M868 consumed attempt identity drift")
    return identity


def full_first_row_requests():
    contract = strict_json(M861.M785.HERE.parent.parent /
                           "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")
    entry = contract["inputs"]["primary_m686"]
    payload_root = HW / entry["directory"]
    manifest = strict_json(payload_root / "manifest.json")
    records = M785.normalized_population_records(
        manifest, "M686_ZURICH_CITY_09_A_S10")
    record = records[0]
    require(int(record["module_index"]) == 0 and
            int(record["sample_id"]) == 0,
            "M868 first-row record identity drift")
    mapper_row = contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(HW / mapper_row["path"],
                                     mapper_row["sha256"],
                                     "m868_m672_mapper")
    m712 = contract["inputs"]["m712_oracle"]
    m722 = contract["inputs"]["m722r2_oracle"]
    storage = contract["inputs"]["m785_storage_oracle"]
    oracles = M785.load_pinned_oracles(
        HW / m712["path"], m712["sha256"],
        HW / m722["path"], m722["sha256"],
        HW / storage["path"], storage["sha256"])
    compressed = M785.iter_record_transactions(
        mapper, record, payload_root, "M686_ZURICH_CITY_09_A_S10",
        "A1_OSG", 0, oracles)
    yield from M785.expand_transactions(compressed)


def run_full_first_row(candidate_path: Path, runner_path: Path,
                       expected_runner_sha256: str,
                       hammer_review_sha256: str,
                       hammer_outer_sha256: str,
                       output: Path) -> Dict[str, object]:
    attempt_identity = validate_attempt(
        candidate_path, runner_path, expected_runner_sha256,
        hammer_review_sha256, hammer_outer_sha256)
    output = Path(output)
    require(output.parent.resolve() == RESULT.parent.resolve() and
            output.name.startswith(RESULT.name + ".stage."),
            "M868 private result stage path drift")
    require(not output.exists() and not output.is_symlink(),
            "M868 private result stage collision")
    os.mkdir(output, 0o700)
    started = time.monotonic()
    contract = strict_json(HW /
        "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")
    scheduler = M861.StreamingAddressTimedScheduler(
        M785.resource_from_contract(contract))
    summary = scheduler.schedule(full_first_row_requests(),
                                 retain_details=False)
    elapsed = time.monotonic() - started
    require(summary["expanded_request_count"] == EXPECTED_EXPANDED,
            "M868 full-first-row expanded cardinality drift")
    require(summary["compressed_transaction_count"] == EXPECTED_COMPRESSED,
            "M868 full-first-row compressed cardinality drift")
    require(summary["population_ids"] == ["M686_ZURICH_CITY_09_A_S10"] and
            summary["configs"] == ["A1_OSG"] and
            summary["detail_retained"] is False and
            "scheduled_requests" not in summary and
            "compressed_schedule" not in summary,
            "M868 full-first-row aggregate boundary drift")
    result = {
        "schema": "m868_m861_decoder_py310_full_first_row_diagnostic_v1",
        "status": "PASS_M868_FULL_FIRST_ROW_DIAGNOSTIC__NONPRODUCTION__FRESH_RESULT_HAMMER_REQUIRED",
        "identity": {
            "label": "M854_FIRST_D0_A1_T0",
            "population": "M686_ZURICH_CITY_09_A_S10",
            "record_ordinal": 0,
            "module_index": 0,
            "sample_id": 0,
            "configuration": "A1_OSG",
            "timestep": 0,
        },
        "aggregate": {
            "expanded_request_count": summary["expanded_request_count"],
            "compressed_transaction_count":
                summary["compressed_transaction_count"],
            "total_cycles_diagnostic_only": summary["total_cycles"],
            "transaction_address_sha256":
                summary["transaction_address_sha256"],
            "commit_sequence_sha256": summary["commit_sequence_sha256"],
            "cycle_classes_diagnostic_only": summary["cycle_classes"],
            "event_sweep_diagnostics": summary["event_sweep_diagnostics"],
            "detail_retained": False,
        },
        "runtime": {
            "elapsed_seconds_diagnostic_only": elapsed,
            "process_max_rss_kib": int(process_resource.getrusage(
                process_resource.RUSAGE_SELF).ru_maxrss),
            "token_ready_entries": len(scheduler.token_ready),
            "next_port_cycle_entries": len(scheduler.next_port_cycle),
            "outstanding_return_entries": sum(
                len(values) for values in scheduler.outstanding_returns.values()),
        },
        "attempt": attempt_identity,
        "claim_boundary": {
            "one_full_first_row_diagnostic_completed": True,
            "full_population": False,
            "production_cycles": False,
            "production_speedup": False,
            "decoder_complete": False,
            "table_a": False,
            "paper_citable": False,
            "fresh_result_hammer_required": True,
        },
    }
    _write_exclusive(output / "diagnostic.json", (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return result


def publish_no_replace(stage: Path, destination: Path) -> Dict[str, object]:
    stage, destination = Path(stage), Path(destination)
    require(destination.resolve(strict=False) == RESULT.resolve(strict=False),
            "M868 canonical result destination drift")
    identity = verify_sealed(stage)
    population = {path.name for path in stage.iterdir()}
    require(population == {"diagnostic.json", "SHA256SUMS",
                           "SHA256SUMS.seal.sha256"},
            "M868 result stage population drift")
    _rename_noreplace(stage, destination)
    require(verify_sealed(destination) == identity and not stage.exists(),
            "M868 canonical publication identity drift")
    return {"status": "PASS_M868_CANONICAL_NOREPLACE_PUBLICATION",
            **identity}


def write_failure_receipt(candidate_path: Path, runner_path: Path,
                          expected_runner_sha256: str,
                          hammer_review_sha256: str,
                          hammer_outer_sha256: str,
                          stdout_log: Path, stderr_log: Path,
                          output: Path, return_code: int, phase: str,
                          partial_artifact: str) -> Dict[str, object]:
    validate_attempt(candidate_path, runner_path, expected_runner_sha256,
                     hammer_review_sha256, hammer_outer_sha256)
    require(return_code != 0 and phase,
            "M868 failure receipt identity malformed")
    output = Path(output)
    require(output.parent.resolve() == RESULT.parent.resolve() and
            output.name.startswith(FAILURE_PREFIX) and
            not output.exists() and not output.is_symlink(),
            "M868 failure quarantine path drift")
    os.mkdir(output, 0o700)
    members = []
    receipt = {
        "schema": "m868_m861_decoder_full_first_row_failure_receipt_v1",
        "status": "FAILED_OR_INCOMPLETE__NO_CYCLES_OR_SPEEDUP_CITABLE",
        "return_code": int(return_code),
        "phase": str(phase),
        "attempt_path": _canonical_relative(ATTEMPT),
        "runner_sha256": expected_runner_sha256,
        "hammer_review_sha256": hammer_review_sha256,
        "hammer_outer_seal_file_sha256": hammer_outer_sha256,
        "partial_artifact": str(partial_artifact),
        "canonical_result_absent": not RESULT.exists() and
            not RESULT.is_symlink(),
        "full_population": False,
        "production_cycles": False,
        "production_speedup": False,
        "paper_citable": False,
    }
    _write_exclusive(output / "failure.json", (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    members.append("failure.json")
    for source, name in ((Path(stdout_log), "driver.stdout.log"),
                         (Path(stderr_log), "driver.stderr.log")):
        require(source.is_file() and not source.is_symlink(),
                "M868 failure log absent")
        _write_exclusive(output / name, source.read_bytes())
        members.append(name)
    identity = seal_directory(output, tuple(members))
    return {"status": receipt["status"], **identity}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-candidate", action="store_true")
    parser.add_argument("--dry-run-no-work", action="store_true")
    parser.add_argument("--validate-formal-preflight", action="store_true")
    parser.add_argument("--consume-attempt", action="store_true")
    parser.add_argument("--validate-attempt", action="store_true")
    parser.add_argument("--run-full-first-row", action="store_true")
    parser.add_argument("--publish-no-replace", action="store_true")
    parser.add_argument("--write-failure-receipt", action="store_true")
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--expected-runner-sha256", default="")
    parser.add_argument("--hammer-review-sha256", default="")
    parser.add_argument("--hammer-outer-sha256", default="")
    parser.add_argument("--stage-basename", default="")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-to", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--return-code", type=int)
    parser.add_argument("--phase", default="")
    parser.add_argument("--partial-artifact", default="")
    args = parser.parse_args(argv)
    modes = (args.validate_candidate, args.dry_run_no_work,
             args.validate_formal_preflight, args.consume_attempt,
             args.validate_attempt, args.run_full_first_row,
             args.publish_no_replace, args.write_failure_receipt)
    require(sum(bool(value) for value in modes) == 1,
            "select exactly one M868 mode")
    if args.validate_candidate or args.dry_run_no_work:
        value = validate_candidate(args.candidate, require_unconsumed=True)
        value["status"] = ("PASS_M868_NO_WORK_DRY_RUN__NO_FILES_CREATED__NO_ATTEMPT"
                           if args.dry_run_no_work else value["status"])
    elif args.validate_formal_preflight:
        value = validate_candidate(args.candidate, require_unconsumed=True)
        value["hammer"] = validate_hammer(
            args.hammer_review_sha256, args.hammer_outer_sha256)
        value["status"] = "PASS_M868_FORMAL_PREFLIGHT__UNCONSUMED"
    else:
        require(args.runner is not None and args.expected_runner_sha256,
                "M868 runner identity is required")
        if args.consume_attempt:
            value = consume_attempt(
                args.candidate, args.runner, args.expected_runner_sha256,
                args.hammer_review_sha256, args.hammer_outer_sha256,
                args.stage_basename)
        elif args.validate_attempt:
            value = {"status": "PASS_M868_CONSUMED_ATTEMPT",
                     **validate_attempt(
                         args.candidate, args.runner,
                         args.expected_runner_sha256,
                         args.hammer_review_sha256,
                         args.hammer_outer_sha256)}
        elif args.run_full_first_row:
            require(args.output is not None,
                    "M868 private output stage is required")
            value = run_full_first_row(
                args.candidate, args.runner, args.expected_runner_sha256,
                args.hammer_review_sha256, args.hammer_outer_sha256,
                args.output)
        elif args.publish_no_replace:
            require(args.output is not None and args.publish_to is not None,
                    "M868 stage and publication target are required")
            value = publish_no_replace(args.output, args.publish_to)
        else:
            require(all((args.stdout_log, args.stderr_log, args.output,
                         args.return_code is not None, args.phase)),
                    "M868 failure receipt inputs incomplete")
            value = write_failure_receipt(
                args.candidate, args.runner, args.expected_runner_sha256,
                args.hammer_review_sha256, args.hammer_outer_sha256,
                args.stdout_log, args.stderr_log, args.output,
                args.return_code, args.phase, args.partial_artifact)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
