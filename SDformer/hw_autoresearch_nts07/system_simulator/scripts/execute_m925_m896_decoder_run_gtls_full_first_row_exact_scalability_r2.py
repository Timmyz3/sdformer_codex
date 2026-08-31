"""M925 source-only full-first-row exact/scalability diagnostic R2.

This additive successor repairs only M900 process-control and publication
boundaries.  It imports the frozen M896 RUN-GTLS scheduler and preserves the
same D0/A1/t0 exact aggregate.  M900's 9.320783571-second 100x hypothesis is a
historical failed scientific threshold, never an acceptance gate here.  A
future, separately reviewed M927/M928 release may authorize one R2 diagnostic
with a 2715-second operational safety timeout.

There is intentionally no shebang.  The absolute sealed Python 3.10 binary
must be named by the runner.  This source alone cannot launch a full row.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource as process_resource
import shutil
import sys
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
PYTHON_VERSION = "3.10.18"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M896_SHA256 = "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39"
M902_REVIEW_SHA256 = "6b25dae1ed54fb7b591472a3fd6b6ac9932772e13e60f4395895fd3526e2fc3b"
M902_MANIFEST_SHA256 = "e6f1fe535227be4146b3563b481f7d3504b76352b93e585cff49b879fbb4fad9"
M902_OUTER_SHA256 = "98b3c505534fec3904d2fb327c4050c6fc3ab3a4e975ca96a0fd7ec8ef91d4da"
M900_ATTEMPT_MANIFEST_SHA256 = "4584af13bfa85033aafb9d6ffd9881a1ce20f37e46c45e544454fa39128dff7b"
M900_ATTEMPT_OUTER_SHA256 = "01f582fefec1e4ba2f079c4e0057e02e108f934bde3b6863af4b46cd77eccedd"
M900_FAILURE_MANIFEST_SHA256 = "ed3cff05817be659093f2546b25180b75d0f5ac7432e16f27f643b23ed98206a"
M900_FAILURE_OUTER_SHA256 = "f36d2335e2dff5a2102e8c89a5b6c6b61181540519ccb2088ba3baf53d9d94c2"

EXPECTED_COMPRESSED = 9582057
EXPECTED_EXPANDED = 38672612
EXPECTED_CYCLES = 20548766
EXPECTED_ADDRESS_SHA256 = "78b90d378956948fc3eab3d7a1bd6f88c8bcf4d32871e971641c9b1a62dfaa6e"
EXPECTED_COMMIT_SHA256 = "aa69b355efd62b428e2909ee4c1dbecdf34ec3e1e8681b0c78ace19a444ff861"
EXPECTED_CYCLE_CLASSES = {
    "active_service": 18502452,
    "compute": 1,
    "dependency_completion": 2046313,
    "memory": 0,
    "psum_bank": 0,
    "weight_bank": 0,
}
M883_ANCHOR_SECONDS = 932.0783571209759
SCIENTIFIC_100X_THRESHOLD_SECONDS = 9.320783571209759
OPERATIONAL_SAFETY_TIMEOUT_SECONDS = 2715
STATE_OBSERVATION_BYTES = 512 * 1024 * 1024
RENAME_NOREPLACE = 1


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
    require(Path(sys.executable) == PYTHON_PATH,
            "M925 forbids ambient Python, shebang and PATH fallback")
    require(PYTHON_PATH.is_file() and not PYTHON_PATH.is_symlink(),
            "M925 Python identity is nonregular")
    require(sha256(PYTHON_PATH) == PYTHON_SHA256,
            "M925 Python SHA drift")
    require(platform.python_version() == PYTHON_VERSION,
            "M925 Python version drift")


_verify_runtime_early()

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M896_PATH = HERE / "analyze_m896_decoder_run_gtls_source_candidate.py"
SOURCE_CONTRACT = HW / "contracts/m925_m896_decoder_run_gtls_full_first_row_exact_scalability_source_contract_r1_20260829.json"
FUTURE_RELEASE = HW / "contracts/m927_m925_decoder_run_gtls_full_first_row_exact_scalability_release_r1_20260829.json"
M902_DIR = HW / "reviews/m902_m900_decoder_fullrow_failure_audit_r1_20260829"
M900_ATTEMPT = HW / "results/.m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_attempt_consumed"
M900_FAILURE = HW / "results/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_20260829.failed_or_incomplete.3773893.17022.27057"
FINAL_HAMMER_DIR = HW / "reviews/m928_m927_m925_decoder_gtls_r2_final_launch_hammer_r1_20260829"
FINAL_HAMMER_STATUS = "PASS100_M925_R2_EXACT_SCALABILITY_FINAL_LAUNCH__ONE_DIAGNOSTIC_AUTHORIZED"
RESULT = HW / "results/m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_20260829"
ATTEMPT = HW / "results/.m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_attempt_consumed"
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
SOURCE_SCHEMA = "m925_m896_decoder_run_gtls_full_first_row_exact_scalability_source_v1"
RELEASE_SCHEMA = "m927_m925_decoder_run_gtls_full_first_row_exact_scalability_release_v1"


def _load_exact(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink(), name + " absent")
    require(sha256(path) == expected, name + " SHA drift")
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M896 = _load_exact(M896_PATH, M896_SHA256, "m925_frozen_m896")
M785 = M896.M785


def strict_json(path: Path) -> object:
    return M785.strict_json(Path(path))


def _safe_basename(name: str, label: str) -> None:
    require(name and name not in (".", "..") and "/" not in name and
            "\\" not in name and "\x00" not in name, label + " malformed")


def verify_sealed_directory(path: Path) -> Dict[str, str]:
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(),
            "sealed directory absent/nonregular: " + str(path))
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "double seal absent: " + str(path))
    listed = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed manifest row")
        digest, name = fields
        _safe_basename(name, "manifest member")
        require(name not in listed, "duplicate manifest member")
        member = path / name
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == digest, "sealed member drift: " + name)
        if member.suffix == ".json":
            strict_json(member)
        listed[name] = digest
    actual = {entry.name for entry in path.iterdir() if entry.is_file()}
    require(actual == set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "sealed regular-file population drift")
    for entry in path.iterdir():
        if entry.is_dir():
            require(entry.name == "__pycache__" and not entry.is_symlink(),
                    "unsealed directory member")
    require(outer.read_text(encoding="ascii") ==
            sha256(manifest) + "  SHA256SUMS\n", "outer seal content drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer)}


def regular_exact(path: Path, expected: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), label + " absent")
    require(sha256(path) == expected, label + " SHA drift")


def _canonical_relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def canonical_paths() -> Dict[str, str]:
    return {
        "result": _canonical_relative(RESULT),
        "attempt": _canonical_relative(ATTEMPT),
        "failed_or_incomplete_prefix":
            _canonical_relative(RESULT.parent) + "/" + FAILURE_PREFIX,
    }


def scan_collisions() -> Tuple[str, ...]:
    require(RESULT.parent.is_dir() and not RESULT.parent.is_symlink(),
            "M925 result parent unavailable")
    names = []
    for entry in os.scandir(RESULT.parent):
        if (entry.name in (RESULT.name, ATTEMPT.name) or
                entry.name.startswith(RESULT.name + ".stage.") or
                entry.name.startswith(ATTEMPT.name + ".stage.") or
                entry.name.startswith(RESULT.name + ".worker_") or
                entry.name.startswith(RESULT.name + ".runtime_resource_") or
                entry.name.startswith(FAILURE_PREFIX)):
            names.append(entry.name)
    return tuple(sorted(names))


def validate_source_contract(contract_path: Path = SOURCE_CONTRACT,
                             require_unconsumed: bool = True) -> Dict[str, object]:
    contract_path = Path(contract_path)
    require(contract_path.resolve() == SOURCE_CONTRACT.resolve(),
            "M925 source contract canonical path drift")
    contract = strict_json(contract_path)
    require(isinstance(contract, dict) and contract.get("schema") == SOURCE_SCHEMA,
            "M925 source contract schema drift")
    require(contract.get("status") ==
            "SOURCE_ONLY_R2_PROCESS_CONTROL_REPAIR__FRESH_M926_HAMMER_REQUIRED" and
            contract.get("launch_now") is False and
            contract.get("full_first_row") is False and
            contract.get("full_population") is False and
            contract.get("production") is False,
            "M925 source contract authority drift")
    require(contract.get("canonical") == canonical_paths(),
            "M925 canonical namespace drift")
    timing = contract.get("timing_contract")
    require(timing == {
        "m883_anchor_seconds": M883_ANCHOR_SECONDS,
        "scientific_100x_threshold_seconds": SCIENTIFIC_100X_THRESHOLD_SECONDS,
        "scientific_100x_hypothesis_already_failed_by_m900": True,
        "r2_objective_is_100x_retry": False,
        "bounded_100k_seconds_used": 3.51,
        "maximum_population_ratio_used": 386.72612,
        "linear_scaled_seconds": 1357.4086812,
        "operational_safety_factor": 2.0,
        "operational_safety_timeout_seconds": OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
        "monitor_period_seconds": 1,
        "consecutive_over_timeout_samples": 3,
    }, "M925 scientific/operational timing separation drift")
    for label, row in contract["exact_files"].items():
        regular_exact(HW / row["path"], row["sha256"], label)
    m902 = verify_sealed_directory(M902_DIR)
    require(m902 == {"manifest_sha256": M902_MANIFEST_SHA256,
                     "outer_seal_file_sha256": M902_OUTER_SHA256},
            "M902 sealed identity drift")
    regular_exact(M902_DIR / "review.json", M902_REVIEW_SHA256, "M902 review")
    attempt = verify_sealed_directory(M900_ATTEMPT)
    require(attempt == {"manifest_sha256": M900_ATTEMPT_MANIFEST_SHA256,
                        "outer_seal_file_sha256": M900_ATTEMPT_OUTER_SHA256},
            "M900 consumed attempt drift")
    failure = verify_sealed_directory(M900_FAILURE)
    require(failure == {"manifest_sha256": M900_FAILURE_MANIFEST_SHA256,
                        "outer_seal_file_sha256": M900_FAILURE_OUTER_SHA256},
            "M900 failure receipt drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    if require_unconsumed:
        require(not scan_collisions(),
                "M925 source namespace occupied: " + repr(scan_collisions()))
    return {
        "status": "PASS_M925_SOURCE_ONLY_IDENTITY__NO_WORK_NO_ATTEMPT",
        "contract_sha256": sha256(contract_path),
        "scientific_100x_hypothesis_already_failed": True,
        "operational_safety_timeout_seconds": OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
        "launch_now": False,
        "full_first_row": False,
        "full_population": False,
    }


def validate_future_release(release_path: Path, runner_path: Path,
                            expected_release_sha256: str,
                            expected_runner_sha256: str,
                            hammer_review_sha256: str,
                            hammer_outer_sha256: str,
                            require_unconsumed: bool) -> Dict[str, object]:
    release_path, runner_path = Path(release_path), Path(runner_path)
    require(release_path.resolve() == FUTURE_RELEASE.resolve(),
            "M925 future release canonical path drift")
    regular_exact(release_path, expected_release_sha256, "M927 release")
    regular_exact(runner_path, expected_runner_sha256, "M925 runner")
    release = strict_json(release_path)
    require(isinstance(release, dict) and release.get("schema") == RELEASE_SCHEMA and
            release.get("status") ==
            "INERT_R2_RELEASE__PENDING_FRESH_M928_FINAL_HAMMER" and
            release.get("release") is True and release.get("launch_now") is False and
            release.get("max_attempts") == 1,
            "M927 release authority drift")
    require(release.get("canonical") == canonical_paths(),
            "M927 canonical namespace drift")
    require(release.get("scientific_threshold") == {
        "seconds": SCIENTIFIC_100X_THRESHOLD_SECONDS,
        "historical_status": "FAILED_BY_M900__NOT_RETRIED_BY_R2",
        "acceptance_gate_for_r2": False,
    }, "M927 scientific threshold drift")
    require(release.get("operational_safety_timeout_seconds") ==
            OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
            "M927 operational timeout drift")
    claims = release.get("claim_boundary", {})
    for key in ("production", "full_population", "decoder_complete",
                "cycles_or_speedup_citable", "system_speedup", "energy",
                "paper_ppa_ready", "paper_citable"):
        require(claims.get(key) is False, "M927 claim drift: " + key)
    require(release.get("source_binding", {}).get("m925_contract_sha256") ==
            sha256(SOURCE_CONTRACT) and
            release.get("source_binding", {}).get("m925_runner_sha256") ==
            expected_runner_sha256 and
            release.get("source_binding", {}).get("m925_driver_sha256") ==
            sha256(Path(__file__).resolve()) and
            release.get("source_binding", {}).get("m896_source_sha256") ==
            M896_SHA256 and
            release.get("source_binding", {}).get("m902_review_sha256") ==
            M902_REVIEW_SHA256,
            "M927 source binding drift")
    hammer_identity = verify_sealed_directory(FINAL_HAMMER_DIR)
    require(hammer_identity["outer_seal_file_sha256"] == hammer_outer_sha256,
            "M928 outer seal caller pin drift")
    regular_exact(FINAL_HAMMER_DIR / "review.json", hammer_review_sha256,
                  "M928 final review")
    review = strict_json(FINAL_HAMMER_DIR / "review.json")
    require(review.get("status") == FINAL_HAMMER_STATUS and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
            "M928 final authority drift")
    binding = review.get("reviewed_identity", {})
    require(binding.get("release_sha256") == expected_release_sha256 and
            binding.get("runner_sha256") == expected_runner_sha256 and
            binding.get("driver_sha256") == sha256(Path(__file__).resolve()) and
            binding.get("source_contract_sha256") == sha256(SOURCE_CONTRACT) and
            review.get("authorization", {}).get(
                "one_full_first_row_exact_scalability_diagnostic") is True and
            review.get("authorization", {}).get("full_population") is False and
            review.get("authorization", {}).get("production") is False,
            "M928 exact binding/authorization drift")
    if require_unconsumed:
        require(not scan_collisions(),
                "M925 one-shot namespace occupied: " + repr(scan_collisions()))
    return {"release_sha256": expected_release_sha256,
            "runner_sha256": expected_runner_sha256,
            "hammer_review_sha256": hammer_review_sha256,
            "hammer_manifest_sha256": hammer_identity["manifest_sha256"],
            "hammer_outer_seal_file_sha256": hammer_outer_sha256}


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o600)
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


def _write_atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    require(not temporary.exists() and not temporary.is_symlink(),
            "M925 heartbeat temporary collision")
    _write_exclusive(temporary, payload)
    os.replace(str(temporary), str(path))


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(source.parent.resolve() == destination.parent.resolve(),
            "M925 publication must stay in one directory")
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
    if rc:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise Failure("M925 no-replace collision")
        raise Failure("M925 renameat2 failed: " + os.strerror(number))


def seal_directory(directory: Path, members: Sequence[str]) -> Dict[str, str]:
    lines = []
    for name in sorted(members):
        _safe_basename(name, "sealed member")
        member = directory / name
        require(member.is_file() and not member.is_symlink(),
                "M925 seal member absent: " + name)
        lines.append(sha256(member) + "  " + name + "\n")
    _write_exclusive(directory / "SHA256SUMS", "".join(lines).encode("ascii"))
    _write_exclusive(directory / "SHA256SUMS.seal.sha256",
                     (sha256(directory / "SHA256SUMS") +
                      "  SHA256SUMS\n").encode("ascii"))
    return verify_sealed_directory(directory)


def consume_attempt(release_path: Path, runner_path: Path,
                    expected_release_sha256: str,
                    expected_runner_sha256: str,
                    hammer_review_sha256: str,
                    hammer_outer_sha256: str,
                    stage_basename: str) -> Dict[str, object]:
    authority = validate_future_release(
        release_path, runner_path, expected_release_sha256,
        expected_runner_sha256, hammer_review_sha256,
        hammer_outer_sha256, require_unconsumed=True)
    _safe_basename(stage_basename, "attempt stage")
    require(stage_basename.startswith(ATTEMPT.name + ".stage."),
            "M925 attempt stage namespace drift")
    stage = ATTEMPT.parent / stage_basename
    require(not stage.exists() and not stage.is_symlink(),
            "M925 attempt stage collision")
    os.mkdir(stage, 0o700)
    published = False
    try:
        receipt = {
            "schema": "m925_decoder_gtls_r2_attempt_v1",
            "status": "CONSUMED_BEFORE_ONE_EXACT_SCALABILITY_DIAGNOSTIC",
            "max_attempts": 1,
            "authority": authority,
            "workload": "M854_FIRST_D0_A1_T0",
            "m900_attempt_restored_reused_or_deleted": False,
            "scientific_100x_hypothesis_already_failed": True,
            "operational_safety_timeout_seconds": OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
            "full_population": False,
            "production": False,
            "cycles_or_speedup_citable": False,
        }
        _write_exclusive(stage / "attempt.json", (json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False) +
            "\n").encode("utf-8"))
        identity = seal_directory(stage, ("attempt.json",))
        _rename_noreplace(stage, ATTEMPT)
        published = True
        require(verify_sealed_directory(ATTEMPT) == identity,
                "M925 published attempt drift")
        return {"status": receipt["status"], **identity}
    finally:
        if not published and stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)


def validate_attempt(release_path: Path, runner_path: Path,
                     expected_release_sha256: str,
                     expected_runner_sha256: str,
                     hammer_review_sha256: str,
                     hammer_outer_sha256: str) -> Dict[str, str]:
    authority = validate_future_release(
        release_path, runner_path, expected_release_sha256,
        expected_runner_sha256, hammer_review_sha256,
        hammer_outer_sha256, require_unconsumed=False)
    identity = verify_sealed_directory(ATTEMPT)
    receipt = strict_json(ATTEMPT / "attempt.json")
    require(receipt.get("schema") == "m925_decoder_gtls_r2_attempt_v1" and
            receipt.get("status") ==
            "CONSUMED_BEFORE_ONE_EXACT_SCALABILITY_DIAGNOSTIC" and
            receipt.get("max_attempts") == 1 and
            receipt.get("authority") == authority and
            receipt.get("scientific_100x_hypothesis_already_failed") is True and
            receipt.get("operational_safety_timeout_seconds") ==
            OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
            "M925 consumed attempt identity drift")
    return identity


def _full_row_transactions():
    contract = strict_json(HW /
        "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")
    entry = contract["inputs"]["primary_m686"]
    payload_root = HW / entry["directory"]
    manifest = strict_json(payload_root / "manifest.json")
    records = M785.normalized_population_records(
        manifest, "M686_ZURICH_CITY_09_A_S10")
    record = records[0]
    require(int(record["module_index"]) == 0 and int(record["sample_id"]) == 0,
            "M925 first row identity drift")
    mapper_row = contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(HW / mapper_row["path"],
                                     mapper_row["sha256"], "m925_mapper")
    m712, m722, storage = (contract["inputs"][name] for name in
                           ("m712_oracle", "m722r2_oracle",
                            "m785_storage_oracle"))
    oracles = M785.load_pinned_oracles(
        HW / m712["path"], m712["sha256"],
        HW / m722["path"], m722["sha256"],
        HW / storage["path"], storage["sha256"])
    return contract, M785.iter_record_transactions(
        mapper, record, payload_root, "M686_ZURICH_CITY_09_A_S10",
        "A1_OSG", 0, oracles)


def _heartbeat(path: Path, started: float, phase: str,
               *, compressed: int = 0, expanded: int = 0,
               counted_state_bytes: Optional[int] = None) -> None:
    payload = {
        "schema": "m925_r2_runtime_heartbeat_v1",
        "phase": phase,
        "elapsed_seconds": time.monotonic() - started,
        "compressed_transactions_observed": int(compressed),
        "expanded_requests_observed": int(expanded),
        "counted_live_scheduler_state_bytes_diagnostic_only": counted_state_bytes,
        "process_max_rss_kib_diagnostic_only": int(process_resource.getrusage(
            process_resource.RUSAGE_SELF).ru_maxrss),
        "scientific_100x_threshold_seconds_historical_failed":
            SCIENTIFIC_100X_THRESHOLD_SECONDS,
        "operational_safety_timeout_seconds": OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
    }
    _write_atomic_replace(path, (json.dumps(
        payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def run_full_row(release_path: Path, runner_path: Path,
                 expected_release_sha256: str,
                 expected_runner_sha256: str,
                 hammer_review_sha256: str,
                 hammer_outer_sha256: str,
                 output: Path) -> Dict[str, object]:
    attempt = validate_attempt(
        release_path, runner_path, expected_release_sha256,
        expected_runner_sha256, hammer_review_sha256, hammer_outer_sha256)
    output = Path(output)
    require(output.parent.resolve() == RESULT.parent.resolve() and
            output.name.startswith(RESULT.name + ".stage."),
            "M925 private result stage path drift")
    require(not output.exists() and not output.is_symlink(),
            "M925 private result stage collision")
    os.mkdir(output, 0o700)
    heartbeat = output / "runtime_heartbeat.json"
    started = time.monotonic()
    _heartbeat(heartbeat, started, "LOAD_FROZEN_D0_A1_T0")
    contract, stream = _full_row_transactions()
    transactions = []
    expanded = 0
    for tx in stream:
        transactions.append(tx)
        expanded += int(tx.count)
        if len(transactions) % 65536 == 0:
            _heartbeat(heartbeat, started, "BUILD_COMPRESSED_RUN_IR",
                       compressed=len(transactions), expanded=expanded)
    require(len(transactions) == EXPECTED_COMPRESSED and
            expanded == EXPECTED_EXPANDED,
            "M925 full-row cardinality drift before schedule")
    _heartbeat(heartbeat, started, "CONSTRUCT_RUN_GTLS_LIVENESS",
               compressed=len(transactions), expanded=expanded)
    ir = M896.RunGroupIR(transactions,
                         ("M686_ZURICH_CITY_09_A_S10", "A1_OSG", 0, 0, 0))
    del transactions
    _heartbeat(heartbeat, started, "SCHEDULE_RUN_GTLS",
               compressed=EXPECTED_COMPRESSED, expanded=EXPECTED_EXPANDED)
    scheduler = M896.RUNGTLSScheduler(M785.resource_from_contract(contract))
    summary = scheduler.schedule(
        ir, retain_details=False, retain_expanded_address_sha=True,
        retain_terminal_audit=False)
    elapsed = time.monotonic() - started
    counted_state = int(summary["combined_live_event_state_bytes"])
    _heartbeat(heartbeat, started, "VERIFY_EXACT_AND_REPORT_DIAGNOSTICS",
               compressed=EXPECTED_COMPRESSED, expanded=EXPECTED_EXPANDED,
               counted_state_bytes=counted_state)
    require(elapsed <= OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
            "M925 exceeded operational safety timeout")
    require(summary["compressed_transaction_count"] == EXPECTED_COMPRESSED and
            summary["expanded_request_count"] == EXPECTED_EXPANDED and
            summary["total_cycles"] == EXPECTED_CYCLES and
            summary["transaction_address_sha256"] == EXPECTED_ADDRESS_SHA256 and
            summary["commit_sequence_sha256"] == EXPECTED_COMMIT_SHA256 and
            summary["cycle_classes"] == EXPECTED_CYCLE_CLASSES and
            summary["population_ids"] == ["M686_ZURICH_CITY_09_A_S10"] and
            summary["configs"] == ["A1_OSG"] and
            summary["detail_retained"] is False and
            "scheduled_requests" not in summary and
            "compressed_schedule" not in summary,
            "M925 full-row exact anchor mismatch")
    result = {
        "schema": "m925_decoder_gtls_full_first_row_exact_scalability_diagnostic_r2_v1",
        "status": "PASS_M925_ONE_FULL_ROW_EXACT_SCALABILITY_DIAGNOSTIC__FRESH_RESULT_HAMMER_REQUIRED",
        "identity": {
            "label": "M854_FIRST_D0_A1_T0",
            "population": "M686_ZURICH_CITY_09_A_S10",
            "record_ordinal": 0,
            "module_index": 0,
            "sample_id": 0,
            "configuration": "A1_OSG",
            "timestep": 0,
        },
        "exact_aggregate": {
            "expanded_request_count": summary["expanded_request_count"],
            "compressed_transaction_count": summary["compressed_transaction_count"],
            "total_cycles_diagnostic_only": summary["total_cycles"],
            "transaction_address_sha256": summary["transaction_address_sha256"],
            "commit_sequence_sha256": summary["commit_sequence_sha256"],
            "cycle_classes_diagnostic_only": summary["cycle_classes"],
            "event_run_counts": summary["event_run_counts"],
            "live_token_peak": summary["live_token_peak"],
        },
        "host_scalability_diagnostic": {
            "m883_anchor_elapsed_seconds": M883_ANCHOR_SECONDS,
            "scientific_100x_threshold_seconds": SCIENTIFIC_100X_THRESHOLD_SECONDS,
            "scientific_100x_hypothesis_already_failed_by_m900": True,
            "r2_is_100x_retry": False,
            "operational_safety_timeout_seconds": OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
            "measured_end_to_end_elapsed_seconds": elapsed,
            "measured_host_speedup_vs_m883_diagnostic_only":
                M883_ANCHOR_SECONDS / elapsed,
            "counted_live_scheduler_state_bytes_diagnostic_only": counted_state,
            "prior_projected_512mib_observation_bytes": STATE_OBSERVATION_BYTES,
            "process_max_rss_kib_diagnostic_only": int(process_resource.getrusage(
                process_resource.RUSAGE_SELF).ru_maxrss),
            "serialized_or_compressed_file_size_used": False,
        },
        "attempt": attempt,
        "claim_boundary": {
            "one_full_row_exact_scalability_diagnostic_completed": True,
            "production": False,
            "full_population": False,
            "decoder_complete": False,
            "cycles_or_speedup_citable": False,
            "system_speedup": False,
            "energy": False,
            "paper_ppa_ready": False,
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
            "M925 canonical destination drift")
    identity = verify_sealed_directory(stage)
    require({path.name for path in stage.iterdir()} == {
        "diagnostic.json", "runtime_heartbeat.json",
        "runtime_resource_snapshots.tsv", "worker_identity.txt",
        "job_tree_drain_receipt.txt", "SHA256SUMS",
        "SHA256SUMS.seal.sha256"}, "M925 result population drift")
    _rename_noreplace(stage, destination)
    require(verify_sealed_directory(destination) == identity and
            not stage.exists(), "M925 canonical publication drift")
    return {"status": "PASS_M925_CANONICAL_NOREPLACE_PUBLICATION", **identity}


def write_failure_receipt(release_path: Path, runner_path: Path,
                          expected_release_sha256: str,
                          expected_runner_sha256: str,
                          hammer_review_sha256: str,
                          hammer_outer_sha256: str,
                          stdout_log: Path, stderr_log: Path,
                          snapshot_log: Path, worker_identity: Path,
                          drain_receipt: Path,
                          output: Path, return_code: int, phase: str,
                          partial_artifact: str) -> Dict[str, object]:
    validate_attempt(release_path, runner_path, expected_release_sha256,
                     expected_runner_sha256, hammer_review_sha256,
                     hammer_outer_sha256)
    require(return_code != 0 and phase,
            "M925 failure receipt identity malformed")
    output = Path(output)
    require(output.parent.resolve() == RESULT.parent.resolve() and
            output.name.startswith(FAILURE_PREFIX) and
            not output.exists() and not output.is_symlink(),
            "M925 failure quarantine path drift")
    os.mkdir(output, 0o700)
    receipt = {
        "schema": "m925_decoder_gtls_r2_failure_receipt_v1",
        "status": "FAILED_OR_INCOMPLETE__R2_DIAGNOSTIC_NONCITABLE",
        "return_code": int(return_code),
        "phase": str(phase),
        "attempt_path": _canonical_relative(ATTEMPT),
        "release_sha256": expected_release_sha256,
        "runner_sha256": expected_runner_sha256,
        "hammer_review_sha256": hammer_review_sha256,
        "hammer_outer_seal_file_sha256": hammer_outer_sha256,
        "partial_artifact": str(partial_artifact),
        "scientific_100x_threshold_seconds_historical_failed":
            SCIENTIFIC_100X_THRESHOLD_SECONDS,
        "operational_safety_timeout_seconds": OPERATIONAL_SAFETY_TIMEOUT_SECONDS,
        "worker_group_drained_before_quarantine": True,
        "m900_attempt_restored_reused_or_deleted": False,
        "canonical_result_absent": not RESULT.exists() and not RESULT.is_symlink(),
        "production": False,
        "full_population": False,
        "decoder_complete": False,
        "cycles_or_speedup_citable": False,
        "system_speedup": False,
        "paper_citable": False,
    }
    _write_exclusive(output / "failure.json", (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    members = ["failure.json"]
    for source, name in ((stdout_log, "worker.stdout.log"),
                         (stderr_log, "worker.stderr.log"),
                         (snapshot_log, "runtime_resource_snapshots.tsv"),
                         (worker_identity, "worker_identity.txt"),
                         (drain_receipt, "job_tree_drain_receipt.txt")):
        source = Path(source)
        require(source.is_file() and not source.is_symlink(),
                "M925 failure evidence absent: " + str(source))
        _write_exclusive(output / name, source.read_bytes())
        members.append(name)
    return {"status": receipt["status"],
            **seal_directory(output, tuple(members))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-source-contract", action="store_true")
    parser.add_argument("--dry-run-no-work", action="store_true")
    parser.add_argument("--validate-formal-preflight", action="store_true")
    parser.add_argument("--consume-attempt", action="store_true")
    parser.add_argument("--validate-attempt", action="store_true")
    parser.add_argument("--run-full-first-row", action="store_true")
    parser.add_argument("--publish-no-replace", action="store_true")
    parser.add_argument("--write-failure-receipt", action="store_true")
    parser.add_argument("--source-contract", type=Path, default=SOURCE_CONTRACT)
    parser.add_argument("--release", type=Path, default=FUTURE_RELEASE)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--expected-release-sha256", default="")
    parser.add_argument("--expected-runner-sha256", default="")
    parser.add_argument("--hammer-review-sha256", default="")
    parser.add_argument("--hammer-outer-sha256", default="")
    parser.add_argument("--stage-basename", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-to", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--snapshot-log", type=Path)
    parser.add_argument("--worker-identity", type=Path)
    parser.add_argument("--drain-receipt", type=Path)
    parser.add_argument("--return-code", type=int)
    parser.add_argument("--phase", default="")
    parser.add_argument("--partial-artifact", default="")
    args = parser.parse_args(argv)
    modes = (args.validate_source_contract, args.dry_run_no_work,
             args.validate_formal_preflight, args.consume_attempt,
             args.validate_attempt, args.run_full_first_row,
             args.publish_no_replace, args.write_failure_receipt)
    require(sum(bool(value) for value in modes) == 1,
            "select exactly one M925 mode")
    if args.validate_source_contract or args.dry_run_no_work:
        value = validate_source_contract(args.source_contract,
                                         require_unconsumed=True)
        if args.dry_run_no_work:
            value["status"] = "PASS_M925_NO_WORK_DRY_RUN__NO_FILES_NO_ATTEMPT"
    else:
        require(args.runner is not None and args.expected_release_sha256 and
                args.expected_runner_sha256 and args.hammer_review_sha256 and
                args.hammer_outer_sha256,
                "M925 future release/runner/hammer identities required")
        if args.validate_formal_preflight:
            value = {"status": "PASS_M925_FORMAL_PREFLIGHT__UNCONSUMED",
                     "authority": validate_future_release(
                         args.release, args.runner,
                         args.expected_release_sha256,
                         args.expected_runner_sha256,
                         args.hammer_review_sha256,
                         args.hammer_outer_sha256,
                         require_unconsumed=True)}
        elif args.consume_attempt:
            value = consume_attempt(
                args.release, args.runner, args.expected_release_sha256,
                args.expected_runner_sha256, args.hammer_review_sha256,
                args.hammer_outer_sha256, args.stage_basename)
        elif args.validate_attempt:
            value = {"status": "PASS_M925_CONSUMED_ATTEMPT",
                     **validate_attempt(
                         args.release, args.runner,
                         args.expected_release_sha256,
                         args.expected_runner_sha256,
                         args.hammer_review_sha256,
                         args.hammer_outer_sha256)}
        elif args.run_full_first_row:
            require(args.output is not None, "M925 private stage required")
            value = run_full_row(
                args.release, args.runner, args.expected_release_sha256,
                args.expected_runner_sha256, args.hammer_review_sha256,
                args.hammer_outer_sha256, args.output)
        elif args.publish_no_replace:
            require(args.output is not None and args.publish_to is not None,
                    "M925 stage and destination required")
            value = publish_no_replace(args.output, args.publish_to)
        else:
            require(all((args.stdout_log, args.stderr_log, args.snapshot_log,
                         args.worker_identity, args.drain_receipt, args.output,
                         args.return_code is not None, args.phase)),
                    "M925 failure receipt inputs incomplete")
            value = write_failure_receipt(
                args.release, args.runner, args.expected_release_sha256,
                args.expected_runner_sha256, args.hammer_review_sha256,
                args.hammer_outer_sha256, args.stdout_log, args.stderr_log,
                args.snapshot_log, args.worker_identity, args.drain_receipt,
                args.output,
                args.return_code, args.phase, args.partial_artifact)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
