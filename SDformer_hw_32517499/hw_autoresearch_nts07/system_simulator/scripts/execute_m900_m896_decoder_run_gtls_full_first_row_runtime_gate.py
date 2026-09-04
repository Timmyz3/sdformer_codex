"""M900 one-shot RUN-GTLS full-first-row host-runtime diagnostic.

This additive executor imports the exact M896 source admitted by M899 and is
the only place where one future D0/A1/t0 full row may be scheduled.  The run is
diagnostic and nonproduction.  It must finish before the frozen 100x host-time
deadline and keep M896's counted live scheduler state at or below 512 MiB.
Process RSS is recorded separately and is never substituted for that state
metric.

There is intentionally no shebang.  The pinned absolute Python 3.10 binary
must be named by the runner.
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
import sys
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple


PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
PYTHON_VERSION = "3.10.18"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M896_SHA256 = "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39"
M899_REVIEW_SHA256 = "8c9c51beaa7811e7ceec559ccef4618479c56975d919cf818be15f978ead1bda"
M899_MANIFEST_SHA256 = "4eeae5b917554ad1a2c1c2812c8f1c1544108064a1c0527779193ac41d7e3f21"
M899_OUTER_SHA256 = "3617abb5a144a23d6c3a6048c975755120dc36b332d54e29b94ea614ff75939f"
M883_REVIEW_SHA256 = "ae443b36084a3361548ec6a950dbc0a962cf60ec650000c9638db61854c02f88"
M883_MANIFEST_SHA256 = "3cdd7be9cde8177e4cce6dfd16fc42dda5a84ba729757c92638eb242fe6fed0d"
M883_OUTER_SHA256 = "4ddece71698ee0b83c18d039eb34205a0f2c93b4e5b95fd349f011686ab8d5a1"
M868_RESULT_SHA256 = "53f71f804cad8acafdbc224d12acfbddc1510d1cb202286d67b018a1b1015344"

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
STATE_GATE_BYTES = 512 * 1024 * 1024
M883_ANCHOR_SECONDS = 932.0783571209759
RUNTIME_GATE_SECONDS = M883_ANCHOR_SECONDS / 100.0
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
            "M900 forbids ambient python, shebang, and PATH fallback")
    require(PYTHON_PATH.is_file() and not PYTHON_PATH.is_symlink(),
            "M900 Python interpreter identity is nonregular")
    require(sha256(PYTHON_PATH) == PYTHON_SHA256,
            "M900 Python interpreter SHA drift")
    require(platform.python_version() == PYTHON_VERSION,
            "M900 Python interpreter version drift")


_verify_runtime_early()

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M896_PATH = HERE / "analyze_m896_decoder_run_gtls_source_candidate.py"
RELEASE = HW / "contracts/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_release_r1_20260829.json"
RELEASE_SCHEMA = "m900_m896_decoder_run_gtls_full_first_row_runtime_gate_release_v1"
M899_DIR = HW / "reviews/m899_m896_decoder_run_gtls_source_fresh_hammer_r1_20260829"
M883_DIR = HW / "reviews/m883_m868_m861_decoder_py310_full_first_row_diagnostic_result_hammer_r1_20260829"
M868_RESULT = HW / "results/m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829/diagnostic.json"
FINAL_HAMMER_DIR = HW / "reviews/m901_m900_m896_decoder_run_gtls_full_first_row_final_launch_hammer_r1_20260829"
FINAL_HAMMER_STATUS = "PASS100_M900_RUN_GTLS_FULL_FIRST_ROW_FINAL_LAUNCH__ONE_RUNTIME_GATE_DIAGNOSTIC_AUTHORIZED"
RESULT = HW / "results/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_20260829"
ATTEMPT = HW / "results/.m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_attempt_consumed"
FAILURE_PREFIX = "m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_20260829.failed_or_incomplete."


def _load_exact(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink(), name + " absent")
    require(sha256(path) == expected, name + " SHA drift")
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M896 = _load_exact(M896_PATH, M896_SHA256, "m900_frozen_m896")
M785 = M896.M785


def strict_json(path: Path) -> object:
    return M785.strict_json(Path(path))


def verify_sealed(path: Path) -> Dict[str, str]:
    """Verify the two-level file seal, tolerating only Python cache dirs.

    M899 was independently sealed before a later read-only Python import left
    an unsealed ``__pycache__`` directory beside its files.  Cache bytes carry
    no authority and are not loaded here.  Every regular file remains subject
    to exact population and digest checks; all other extra names are rejected.
    """
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(),
            "sealed directory absent or symlinked")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "double seal absent or nonregular")
    rows: Dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64 and
                all(character in "0123456789abcdef" for character in parts[0]),
                "manifest row malformed")
        name = parts[1]
        _safe_basename(name, "manifest member")
        require(name not in rows, "duplicate manifest member")
        member = path / name
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == parts[0], "sealed member drift: " + name)
        rows[name] = parts[0]
    actual_files = {entry.name for entry in path.iterdir() if entry.is_file()}
    require(actual_files == set(rows) | {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"},
        "sealed regular-file population mismatch")
    for entry in path.iterdir():
        if entry.is_dir():
            require(entry.name == "__pycache__" and not entry.is_symlink(),
                    "unsealed directory population mismatch")
    expected_outer = sha256(manifest) + "  SHA256SUMS\n"
    require(outer.read_text(encoding="ascii") == expected_outer,
            "outer seal drift")
    return {
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
    }


def regular_exact(path: Path, expected: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), label + " absent")
    require(sha256(path) == expected, label + " SHA drift")


def _verify_sealed_identity(directory: Path, manifest: str,
                            outer: str, label: str) -> None:
    identity = verify_sealed(directory)
    require(identity == {"manifest_sha256": manifest,
                         "outer_seal_file_sha256": outer},
            label + " sealed identity drift")


def _canonical_relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _canonical_paths() -> Dict[str, str]:
    return {
        "result": _canonical_relative(RESULT),
        "attempt": _canonical_relative(ATTEMPT),
        "failed_or_incomplete_prefix":
            _canonical_relative(RESULT.parent) + "/" + FAILURE_PREFIX,
    }


def scan_collisions() -> Tuple[str, ...]:
    require(RESULT.parent.is_dir() and not RESULT.parent.is_symlink(),
            "M900 result parent unavailable")
    names = []
    for entry in os.scandir(RESULT.parent):
        if (entry.name in (RESULT.name, ATTEMPT.name) or
                entry.name.startswith(RESULT.name + ".stage.") or
                entry.name.startswith(ATTEMPT.name + ".stage.") or
                entry.name.startswith(FAILURE_PREFIX)):
            names.append(entry.name)
    return tuple(sorted(names))


def validate_release(release_path: Path = RELEASE,
                     require_unconsumed: bool = True) -> Dict[str, object]:
    release_path = Path(release_path)
    require(release_path.resolve() == RELEASE.resolve(),
            "M900 release canonical path drift")
    release = strict_json(release_path)
    require(isinstance(release, dict) and
            release.get("schema") == RELEASE_SCHEMA,
            "M900 release schema drift")
    require(release.get("status") ==
            "INERT_RELEASE_AFTER_M899_PASS100__PENDING_FRESH_M901_FINAL_HAMMER" and
            release.get("launch_now") is False and
            release.get("effective_now") is False and
            release.get("max_attempts") == 1,
            "M900 release authority drift")
    require(release.get("canonical") == _canonical_paths(),
            "M900 canonical namespace drift")
    require(release.get("workload") == {
        "identity": "M854_FIRST_D0_A1_T0",
        "population": "M686_ZURICH_CITY_09_A_S10",
        "record_ordinal": 0,
        "module_index": 0,
        "sample_id": 0,
        "configuration": "A1_OSG",
        "timestep": 0,
        "rows_authorized": 1,
        "population_rows_authorized": 0,
        "expected_compressed_transaction_count": EXPECTED_COMPRESSED,
        "expected_expanded_request_count": EXPECTED_EXPANDED,
    }, "M900 workload identity drift")
    require(release.get("runtime_and_state_gate") == {
        "m883_anchor_elapsed_seconds": M883_ANCHOR_SECONDS,
        "minimum_host_speedup": 100.0,
        "maximum_end_to_end_elapsed_seconds": RUNTIME_GATE_SECONDS,
        "counted_live_scheduler_state_maximum_bytes": STATE_GATE_BYTES,
        "counted_live_scheduler_state_maximum_mib": 512,
        "process_rss_is_diagnostic_only": True,
        "serialized_or_compressed_file_size_forbidden": True,
        "input_transaction_objects_excluded_from_counted_state": True,
        "three_consecutive_runtime_or_resource_over_gate_snapshots_terminate": True,
    }, "M900 runtime/state gate drift")
    require(release.get("claim_boundary") == {
        "diagnostic_only": True,
        "runtime_gate_completed": False,
        "full_population": False,
        "production": False,
        "decoder_complete": False,
        "cycles_or_speedup_citable": False,
        "system_speedup": False,
        "energy": False,
        "paper_ppa_ready": False,
        "paper_citable": False,
        "vcs_dc_pt_fm_ptpx_eda_gpu_remote": False,
        "docs359_modified": False,
    }, "M900 claim boundary drift")
    for label, row in release["source_identity"].items():
        regular_exact(HW / row["path"], row["sha256"], label)
    _verify_sealed_identity(M899_DIR, M899_MANIFEST_SHA256,
                            M899_OUTER_SHA256, "M899")
    regular_exact(M899_DIR / "review.json", M899_REVIEW_SHA256,
                  "M899 review")
    m899 = strict_json(M899_DIR / "review.json")
    require(m899.get("status") ==
            "PASS100_M896_RUN_GTLS_BOUNDED_EXACT__STATE_GATE_PASS__ONLY_FRESH_INERT_FULLROW_RELEASE_AUTHOR_AUTHORIZED" and
            m899.get("score") == 100 and
            m899.get("checks_passed") == 54,
            "M899 authority drift")
    _verify_sealed_identity(M883_DIR, M883_MANIFEST_SHA256,
                            M883_OUTER_SHA256, "M883")
    regular_exact(M883_DIR / "review.json", M883_REVIEW_SHA256,
                  "M883 review")
    regular_exact(M868_RESULT, M868_RESULT_SHA256, "M868 anchor result")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    if require_unconsumed:
        require(not scan_collisions(),
                "M900 one-shot namespace occupied: " + repr(scan_collisions()))
    return {
        "status": "PASS_M900_INERT_RELEASE_IDENTITY__NO_WORK_NO_ATTEMPT",
        "release_sha256": sha256(release_path),
        "launch_now": False,
        "runtime_gate_executed": False,
        "attempt_consumed": False,
    }


def validate_final_hammer(expected_review_sha256: str,
                          expected_outer_sha256: str,
                          expected_release_sha256: str,
                          expected_runner_sha256: str) -> Dict[str, object]:
    require(all(len(value) == 64 for value in (
        expected_review_sha256, expected_outer_sha256,
        expected_release_sha256, expected_runner_sha256)),
        "M900 final-hammer caller pins malformed")
    regular_exact(RELEASE, expected_release_sha256, "M900 release")
    identity = verify_sealed(FINAL_HAMMER_DIR)
    require(identity["outer_seal_file_sha256"] == expected_outer_sha256,
            "M901 outer seal caller pin drift")
    regular_exact(FINAL_HAMMER_DIR / "review.json",
                  expected_review_sha256, "M901 final review")
    review = strict_json(FINAL_HAMMER_DIR / "review.json")
    require(review.get("status") == FINAL_HAMMER_STATUS and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
            "M901 did not grant exact one-shot authority")
    binding = review.get("reviewed_identity", {})
    require(binding.get("release_sha256") == expected_release_sha256 and
            binding.get("runner_sha256") == expected_runner_sha256 and
            binding.get("m896_source_sha256") == M896_SHA256 and
            review.get("authorization", {}).get(
                "one_full_first_row_runtime_gate_diagnostic") is True and
            review.get("authorization", {}).get("full_population") is False and
            review.get("authorization", {}).get("production") is False,
            "M901 exact binding/authorization drift")
    return {
        "review_sha256": expected_review_sha256,
        "manifest_sha256": identity["manifest_sha256"],
        "outer_seal_file_sha256": expected_outer_sha256,
    }


def _safe_basename(name: str, label: str) -> None:
    require(name and name not in (".", "..") and "/" not in name and
            "\x00" not in name, label + " malformed")


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
            "M900 heartbeat temporary collision")
    _write_exclusive(temporary, payload)
    os.replace(str(temporary), str(path))


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(source.parent.resolve() == destination.parent.resolve(),
            "M900 publication must remain in one directory")
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
            raise Failure("M900 no-replace collision")
        raise Failure("M900 renameat2 failed: " + os.strerror(number))


def seal_directory(directory: Path, members: Sequence[str]) -> Dict[str, str]:
    lines = []
    for name in sorted(members):
        _safe_basename(name, "sealed member")
        member = directory / name
        require(member.is_file() and not member.is_symlink(),
                "M900 seal member absent: " + name)
        lines.append(sha256(member) + "  " + name + "\n")
    _write_exclusive(directory / "SHA256SUMS", "".join(lines).encode("ascii"))
    _write_exclusive(directory / "SHA256SUMS.seal.sha256",
                     (sha256(directory / "SHA256SUMS") +
                      "  SHA256SUMS\n").encode("ascii"))
    return verify_sealed(directory)


def consume_attempt(release_path: Path, runner_path: Path,
                    expected_release_sha256: str,
                    expected_runner_sha256: str,
                    hammer_review_sha256: str,
                    hammer_outer_sha256: str,
                    stage_basename: str) -> Dict[str, object]:
    validate_release(release_path, require_unconsumed=True)
    regular_exact(release_path, expected_release_sha256, "M900 release")
    regular_exact(runner_path, expected_runner_sha256, "M900 runner")
    hammer = validate_final_hammer(
        hammer_review_sha256, hammer_outer_sha256,
        expected_release_sha256, expected_runner_sha256)
    _safe_basename(stage_basename, "attempt stage")
    require(stage_basename.startswith(ATTEMPT.name + ".stage."),
            "M900 attempt stage namespace drift")
    stage = ATTEMPT.parent / stage_basename
    require(not stage.exists() and not stage.is_symlink(),
            "M900 attempt stage collision")
    os.mkdir(stage, 0o700)
    published = False
    try:
        receipt = {
            "schema": "m900_m896_decoder_run_gtls_attempt_v1",
            "status": "CONSUMED_IMMEDIATELY_BEFORE_ONE_FULL_ROW_RUNTIME_GATE",
            "max_attempts": 1,
            "release_sha256": expected_release_sha256,
            "runner_sha256": expected_runner_sha256,
            "hammer": hammer,
            "workload": "M854_FIRST_D0_A1_T0",
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
        require(verify_sealed(ATTEMPT) == identity,
                "M900 published attempt drift")
        return {"status": receipt["status"], **identity}
    finally:
        if not published and stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)


def validate_attempt(release_path: Path, runner_path: Path,
                     expected_release_sha256: str,
                     expected_runner_sha256: str,
                     hammer_review_sha256: str,
                     hammer_outer_sha256: str) -> Dict[str, str]:
    validate_release(release_path, require_unconsumed=False)
    regular_exact(release_path, expected_release_sha256, "M900 release")
    regular_exact(runner_path, expected_runner_sha256, "M900 runner")
    hammer = validate_final_hammer(
        hammer_review_sha256, hammer_outer_sha256,
        expected_release_sha256, expected_runner_sha256)
    identity = verify_sealed(ATTEMPT)
    receipt = strict_json(ATTEMPT / "attempt.json")
    require(receipt.get("schema") == "m900_m896_decoder_run_gtls_attempt_v1" and
            receipt.get("status") ==
            "CONSUMED_IMMEDIATELY_BEFORE_ONE_FULL_ROW_RUNTIME_GATE" and
            receipt.get("max_attempts") == 1 and
            receipt.get("release_sha256") == expected_release_sha256 and
            receipt.get("runner_sha256") == expected_runner_sha256 and
            receipt.get("hammer") == hammer,
            "M900 consumed attempt identity drift")
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
    require(int(record["module_index"]) == 0 and
            int(record["sample_id"]) == 0,
            "M900 first row identity drift")
    mapper_row = contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(HW / mapper_row["path"],
                                     mapper_row["sha256"], "m900_mapper")
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
        "schema": "m900_runtime_heartbeat_v1",
        "phase": phase,
        "elapsed_seconds": time.monotonic() - started,
        "compressed_transactions_observed": int(compressed),
        "expanded_requests_observed": int(expanded),
        "counted_live_scheduler_state_bytes": counted_state_bytes,
        "counted_state_gate_bytes": STATE_GATE_BYTES,
        "process_max_rss_kib_diagnostic_only": int(process_resource.getrusage(
            process_resource.RUSAGE_SELF).ru_maxrss),
        "rss_is_not_the_counted_state_gate": True,
    }
    _write_atomic_replace(path, (json.dumps(
        payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def run_full_row_runtime_gate(release_path: Path, runner_path: Path,
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
            "M900 private result stage path drift")
    require(not output.exists() and not output.is_symlink(),
            "M900 private result stage collision")
    os.mkdir(output, 0o700)
    heartbeat_path = output / "runtime_heartbeat.json"
    started = time.monotonic()
    _heartbeat(heartbeat_path, started, "LOAD_FROZEN_D0_A1_T0")
    contract, stream = _full_row_transactions()
    transactions = []
    expanded = 0
    for tx in stream:
        transactions.append(tx)
        expanded += int(tx.count)
        if len(transactions) % 65536 == 0:
            _heartbeat(heartbeat_path, started, "BUILD_COMPRESSED_RUN_IR",
                       compressed=len(transactions), expanded=expanded)
    require(len(transactions) == EXPECTED_COMPRESSED and
            expanded == EXPECTED_EXPANDED,
            "M900 full-row cardinality drift before schedule")
    _heartbeat(heartbeat_path, started, "CONSTRUCT_RUN_GTLS_LIVENESS",
               compressed=len(transactions), expanded=expanded)
    ir = M896.RunGroupIR(transactions,
                         ("M686_ZURICH_CITY_09_A_S10", "A1_OSG", 0, 0, 0))
    del transactions
    _heartbeat(heartbeat_path, started, "SCHEDULE_RUN_GTLS",
               compressed=EXPECTED_COMPRESSED, expanded=EXPECTED_EXPANDED)
    scheduler = M896.RUNGTLSScheduler(M785.resource_from_contract(contract))
    summary = scheduler.schedule(
        ir, retain_details=False, retain_expanded_address_sha=True,
        retain_terminal_audit=False)
    elapsed = time.monotonic() - started
    counted_state = int(summary["combined_live_event_state_bytes"])
    _heartbeat(heartbeat_path, started, "VERIFY_EXACT_AND_GATES",
               compressed=EXPECTED_COMPRESSED, expanded=EXPECTED_EXPANDED,
               counted_state_bytes=counted_state)
    require(elapsed <= RUNTIME_GATE_SECONDS,
            "M900 full-row end-to-end host runtime failed the 100x gate")
    require(counted_state <= STATE_GATE_BYTES,
            "M900 counted live scheduler state exceeded 512 MiB")
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
            "M900 full-row exact anchor mismatch")
    result = {
        "schema": "m900_m896_decoder_run_gtls_full_first_row_runtime_gate_diagnostic_v1",
        "status": "PASS_M900_ONE_FULL_ROW_RUNTIME_GATE__NONPRODUCTION__FRESH_RESULT_HAMMER_REQUIRED",
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
            "closed_form_transactions": summary["closed_form_transactions"],
            "fallback_transactions": summary["fallback_transactions"],
        },
        "runtime_and_state_gate": {
            "m883_anchor_elapsed_seconds": M883_ANCHOR_SECONDS,
            "minimum_host_speedup": 100.0,
            "maximum_elapsed_seconds": RUNTIME_GATE_SECONDS,
            "measured_elapsed_seconds": elapsed,
            "host_speedup_diagnostic_only": M883_ANCHOR_SECONDS / elapsed,
            "counted_live_scheduler_state_bytes": counted_state,
            "counted_state_gate_bytes": STATE_GATE_BYTES,
            "counted_state_gate_passed": True,
            "process_max_rss_kib_diagnostic_only": int(process_resource.getrusage(
                process_resource.RUSAGE_SELF).ru_maxrss),
            "rss_is_not_the_counted_state_gate": True,
            "serialized_or_compressed_file_size_used": False,
            "input_transaction_objects_excluded_from_counted_state": True,
        },
        "attempt": attempt,
        "claim_boundary": {
            "one_full_row_runtime_gate_completed": True,
            "full_population": False,
            "production": False,
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
            "M900 canonical destination drift")
    identity = verify_sealed(stage)
    require({path.name for path in stage.iterdir()} == {
        "diagnostic.json", "runtime_heartbeat.json",
        "runtime_resource_snapshots.tsv", "SHA256SUMS",
        "SHA256SUMS.seal.sha256"}, "M900 result population drift")
    _rename_noreplace(stage, destination)
    require(verify_sealed(destination) == identity and not stage.exists(),
            "M900 canonical publication drift")
    return {"status": "PASS_M900_CANONICAL_NOREPLACE_PUBLICATION", **identity}


def write_failure_receipt(release_path: Path, runner_path: Path,
                          expected_release_sha256: str,
                          expected_runner_sha256: str,
                          hammer_review_sha256: str,
                          hammer_outer_sha256: str,
                          stdout_log: Path, stderr_log: Path,
                          snapshot_log: Path, output: Path,
                          return_code: int, phase: str,
                          partial_artifact: str) -> Dict[str, object]:
    validate_attempt(release_path, runner_path, expected_release_sha256,
                     expected_runner_sha256, hammer_review_sha256,
                     hammer_outer_sha256)
    require(return_code != 0 and phase,
            "M900 failure receipt identity malformed")
    output = Path(output)
    require(output.parent.resolve() == RESULT.parent.resolve() and
            output.name.startswith(FAILURE_PREFIX) and
            not output.exists() and not output.is_symlink(),
            "M900 failure quarantine path drift")
    os.mkdir(output, 0o700)
    receipt = {
        "schema": "m900_m896_decoder_run_gtls_full_row_failure_receipt_v1",
        "status": "FAILED_OR_INCOMPLETE__NO_DECODER_CYCLES_OR_SPEEDUP_CITABLE",
        "return_code": int(return_code),
        "phase": str(phase),
        "attempt_path": _canonical_relative(ATTEMPT),
        "release_sha256": expected_release_sha256,
        "runner_sha256": expected_runner_sha256,
        "hammer_review_sha256": hammer_review_sha256,
        "hammer_outer_seal_file_sha256": hammer_outer_sha256,
        "partial_artifact": str(partial_artifact),
        "runtime_gate_seconds": RUNTIME_GATE_SECONDS,
        "counted_state_gate_bytes": STATE_GATE_BYTES,
        "rss_is_diagnostic_only": True,
        "canonical_result_absent": not RESULT.exists() and not RESULT.is_symlink(),
        "full_population": False,
        "production": False,
        "decoder_complete": False,
        "cycles_or_speedup_citable": False,
        "paper_citable": False,
    }
    _write_exclusive(output / "failure.json", (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    members = ["failure.json"]
    for source, name in ((stdout_log, "driver.stdout.log"),
                         (stderr_log, "driver.stderr.log"),
                         (snapshot_log, "runtime_resource_snapshots.tsv")):
        source = Path(source)
        require(source.is_file() and not source.is_symlink(),
                "M900 failure evidence absent: " + str(source))
        _write_exclusive(output / name, source.read_bytes())
        members.append(name)
    return {"status": receipt["status"],
            **seal_directory(output, tuple(members))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-release", action="store_true")
    parser.add_argument("--dry-run-no-work", action="store_true")
    parser.add_argument("--validate-formal-preflight", action="store_true")
    parser.add_argument("--consume-attempt", action="store_true")
    parser.add_argument("--validate-attempt", action="store_true")
    parser.add_argument("--run-full-first-row", action="store_true")
    parser.add_argument("--publish-no-replace", action="store_true")
    parser.add_argument("--write-failure-receipt", action="store_true")
    parser.add_argument("--release", type=Path, default=RELEASE)
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
    parser.add_argument("--return-code", type=int)
    parser.add_argument("--phase", default="")
    parser.add_argument("--partial-artifact", default="")
    args = parser.parse_args(argv)
    modes = (args.validate_release, args.dry_run_no_work,
             args.validate_formal_preflight, args.consume_attempt,
             args.validate_attempt, args.run_full_first_row,
             args.publish_no_replace, args.write_failure_receipt)
    require(sum(bool(value) for value in modes) == 1,
            "select exactly one M900 mode")
    if args.validate_release or args.dry_run_no_work:
        value = validate_release(args.release, require_unconsumed=True)
        if args.dry_run_no_work:
            value["status"] = "PASS_M900_NO_WORK_DRY_RUN__NO_FILES_CREATED__NO_ATTEMPT"
    elif args.validate_formal_preflight:
        value = validate_release(args.release, require_unconsumed=True)
        value["hammer"] = validate_final_hammer(
            args.hammer_review_sha256, args.hammer_outer_sha256,
            args.expected_release_sha256, args.expected_runner_sha256)
        value["status"] = "PASS_M900_FORMAL_PREFLIGHT__UNCONSUMED"
    else:
        require(args.runner is not None and args.expected_release_sha256 and
                args.expected_runner_sha256,
                "M900 release/runner identities required")
        if args.consume_attempt:
            value = consume_attempt(
                args.release, args.runner, args.expected_release_sha256,
                args.expected_runner_sha256, args.hammer_review_sha256,
                args.hammer_outer_sha256, args.stage_basename)
        elif args.validate_attempt:
            value = {"status": "PASS_M900_CONSUMED_ATTEMPT",
                     **validate_attempt(
                         args.release, args.runner,
                         args.expected_release_sha256,
                         args.expected_runner_sha256,
                         args.hammer_review_sha256,
                         args.hammer_outer_sha256)}
        elif args.run_full_first_row:
            require(args.output is not None, "M900 private stage required")
            value = run_full_row_runtime_gate(
                args.release, args.runner, args.expected_release_sha256,
                args.expected_runner_sha256, args.hammer_review_sha256,
                args.hammer_outer_sha256, args.output)
        elif args.publish_no_replace:
            require(args.output is not None and args.publish_to is not None,
                    "M900 stage and destination required")
            value = publish_no_replace(args.output, args.publish_to)
        else:
            require(all((args.stdout_log, args.stderr_log, args.snapshot_log,
                         args.output, args.return_code is not None, args.phase)),
                    "M900 failure receipt inputs incomplete")
            value = write_failure_receipt(
                args.release, args.runner, args.expected_release_sha256,
                args.expected_runner_sha256, args.hammer_review_sha256,
                args.hammer_outer_sha256, args.stdout_log, args.stderr_log,
                args.snapshot_log, args.output, args.return_code, args.phase,
                args.partial_artifact)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
