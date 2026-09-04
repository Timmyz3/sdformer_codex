#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Immutable one-shot launcher for the M1146CA production digest compiler.

The author path is bounded-only.  A different-author hammer must approve this
source before root invokes its zero-argument production entry exactly once.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import stat
import sys
import tempfile
import time
import traceback
from types import ModuleType
from typing import Any, Mapping

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HERE = SOURCE_FILE.parent
HW = HERE.parent.parent
RESULTS = HW / "results"
COMPILER_SOURCE = HERE / "build_m1146ca_c1_independent_expected_digest_compiler_source.py"
COMPILER_SOURCE_SHA = "7b1f5cd2cd4c4bb0a771d0360f8be924d075215e8dd660728a8decac0c886e73"
COMPILER_AUTHOR = HW / "reviews/m1146ca_c1_independent_expected_digest_compiler_author_receipt_r1_20260830"
COMPILER_AUTHOR_ID = (
    "83a962a93c7fc53273340bfbd6b364ffd1add1ebe661728acaf108f21967e4ac",
    "f3b29f34dc18eff9486d594411445279d7c1232b3ab745097e04e16708bc24cd",
    "9aa612c53b3d4064f4fb80ac057f936459624cc7a211373664a9fd04c3650414",
)
M1147 = HW / "reviews/m1147ca_m1146ca_c1_independent_expected_digest_compiler_hammer_r1_20260830"
M1147_ID = (
    "99ca348c322cd215ad46062cf26306ecf5ce885b512278a5992039affdae36cd",
    "080fb37d73bd16cd561ee2457cbe772eb13caf02d5f51feb482b17065a953674",
    "b18cfb733ae43eb7c07ebf7725b4f0a3de028100b51c51adfa15a0b227072de9",
)
CONTRACT = HW / "contracts/m1148ca_c1_production_expected_digest_compiler_launcher_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "50ff6a357a4497aa9ee1950ecc7dbebce325e7ec4258d38a7293cd5266164b10",
    "5ddb4a7634a88d972bb44f21ffef756c836483603b19a0b0a2a866e1876122fd",
    "6543696b35d014879cf89c20e9559d60b6cc7945c6a169ef4ab3283b8a7ad554",
)
M1141 = RESULTS / "m1141ca_c1_production_schedule_release_r1_20260830"
M1141_RECORDS = M1141 / "m1141ca_per_task_schedule_records.jsonl"
M1141_RECORDS_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
M1141_RECORDS_BYTES = 836_268_740
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_UID = 1913
EXPECTED_RECORDS = 2_436_480
EXPECTED_EVENTS = 212_559_552
MIN_CPUS = 4
MIN_MEM_AVAILABLE = 4 * (1 << 30)
MIN_COMMIT_HEADROOM = 8 * (1 << 30)
MIN_DISK_FREE = 2 * (1 << 30)
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str, expected_uid: int | None = EXPECTED_UID) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected and
            (expected_uid is None or value.st_uid == expected_uid),
            "identity/owner drift: " + str(path))


def strict_json_bytes(payload: bytes) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def strict_json(path: Path) -> Any:
    return strict_json_bytes(path.read_bytes())


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content drift")


def verify_tree(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink() and
            directory.stat().st_uid == EXPECTED_UID, "authority directory drift")
    review = directory / "review.json"; manifest = directory / MANIFEST
    outer = directory / OUTER
    verify_regular(review, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], MANIFEST],
            "authority outer content drift")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "authority manifest row")
        name = fields[1].lstrip("*"); relative = Path(name)
        require(name not in listed and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "authority manifest member")
        listed[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "authority symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "authority special member")
    require(actual == set(listed), "authority exact member set drift")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return strict_json(review)


def load_compiler() -> ModuleType:
    verify_regular(COMPILER_SOURCE, COMPILER_SOURCE_SHA)
    name = "m1146ca_frozen_compiler_for_m1148ca"
    spec = importlib.util.spec_from_file_location(name, COMPILER_SOURCE)
    require(spec is not None and spec.loader is not None, "compiler import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _namespace_paths() -> tuple[Path, ...]:
    fixed = (RESULT, ATTEMPT, LOCK)
    variable = tuple(RESULTS.glob(WORK_PREFIX + "*")) + tuple(RESULTS.glob(FAILURE_PREFIX + "*"))
    return tuple(path for path in fixed + variable if path.exists() or path.is_symlink())


def _meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"(MemAvailable|CommitLimit|Committed_AS):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    require(set(values) == {"MemAvailable", "CommitLimit", "Committed_AS"},
            "meminfo fields absent")
    return values


def _conflicting_processes() -> tuple[int, ...]:
    conflicts = []
    tokens = {str(SOURCE_FILE), SOURCE_FILE.name}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            uid = int(re.search(r"^Uid:\s+(\d+)",
                                (entry / "status").read_text(encoding="utf-8"), re.M).group(1))
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            text = {item.decode("utf-8", "replace") for item in argv if item}
            if uid == EXPECTED_UID and text.intersection(tokens):
                conflicts.append(int(entry.name))
        except (FileNotFoundError, PermissionError, AttributeError):
            continue
    return tuple(sorted(conflicts))


def resource_preflight() -> dict[str, int]:
    require(os.getuid() == EXPECTED_UID and RESULTS.stat().st_uid == EXPECTED_UID,
            "uid drift")
    cpus = len(os.sched_getaffinity(0)); memory = _meminfo()
    free = os.statvfs(RESULTS).f_bavail * os.statvfs(RESULTS).f_frsize
    commit_headroom = memory["CommitLimit"] - memory["Committed_AS"]
    conflicts = _conflicting_processes()
    require(cpus >= MIN_CPUS and memory["MemAvailable"] >= MIN_MEM_AVAILABLE and
            commit_headroom >= MIN_COMMIT_HEADROOM and free >= MIN_DISK_FREE and
            conflicts == (), "resource/process preflight failed")
    return {"cpus": cpus, "mem_available_bytes": memory["MemAvailable"],
            "commit_headroom_bytes": commit_headroom, "disk_free_bytes": free,
            "same_uid_conflicts": len(conflicts)}


def source_preflight(require_fresh_namespace: bool = True) -> dict[str, Any]:
    verify_double(CONTRACT, CONTRACT_ID)
    verify_regular(DOCS359, DOCS359_SHA)
    author = verify_tree(COMPILER_AUTHOR, COMPILER_AUTHOR_ID)
    hammer = verify_tree(M1147, M1147_ID)
    require(author["subject"]["source_sha256"] == COMPILER_SOURCE_SHA and
            author["authorization"]["production_digest_compiler_execution"] is False and
            hammer["status"] ==
                "PASS_M1147CA_DIFFERENT_AUTHOR_BOUNDED_DIGEST_COMPILER_HAMMER__PRODUCTION_LAUNCHER_SOURCE_ONLY_NEXT" and
            hammer["authorization"]["one_shot_production_digest_compiler_launcher_source_next"] is True and
            hammer["authorization"]["production_digest_compiler_execution_by_this_hammer"] is False,
            "M1146CA/M1147CA authorization drift")
    module = load_compiler()
    compiler_preflight = module.source_preflight()
    require(compiler_preflight["production_schedule_records_opened"] is False and
            compiler_preflight["production_events_compiled"] == 0,
            "compiler source preflight boundary drift")
    value = M1141_RECORDS.lstat()
    require(stat.S_ISREG(value.st_mode) and not M1141_RECORDS.is_symlink() and
            value.st_uid == EXPECTED_UID and value.st_size == M1141_RECORDS_BYTES,
            "sealed schedule file metadata drift")
    if require_fresh_namespace:
        require(_namespace_paths() == (), "production namespace is not fresh")
    return {"status": "PASS_M1148CA_SOURCE_PREFLIGHT__NO_PRODUCTION_JSONL_OPEN",
            "compiler_source_sha256": COMPILER_SOURCE_SHA,
            "compiler_author_outer_sha256": COMPILER_AUTHOR_ID[2],
            "m1147ca_outer_sha256": M1147_ID[2],
            "m1141_records_sha256_expected": M1141_RECORDS_SHA,
            "production_jsonl_opened": False, "production_events_compiled": 0}


def _compile_stream(path: Path, expected_sha: str, expected_records: int,
                    geometry: Any, module: ModuleType) -> dict[str, Any]:
    compiler = module.IndependentExpectedDigestCompiler(geometry)
    digest = hashlib.sha256(); count = 0; byte_count = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "schedule fd is not regular")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for raw in stream:
                require(raw.endswith(b"\n") and not raw.endswith(b"\r\n") and
                        1 < len(raw) <= 65536, "schedule JSONL framing drift")
                digest.update(raw); byte_count += len(raw)
                mapping = strict_json_bytes(raw[:-1])
                record = module.ScheduleRecord.from_mapping(mapping, geometry)
                compiler.consume_schedule_record(record)
                count += 1
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                "schedule fd changed during stream")
    finally:
        os.close(fd)
    require(count == expected_records and byte_count == before.st_size and
            digest.hexdigest() == expected_sha,
            "schedule count/byte/SHA identity drift")
    accepted = []
    authority = compiler.finalize(accepted.append)
    require(len(accepted) == 1 and accepted[0] is authority,
            "terminal authority sink cardinality drift")
    return {"authority": authority, "schedule_records": count,
            "schedule_bytes": byte_count, "schedule_sha256": digest.hexdigest()}


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n").encode("utf-8"))


def _seal_tree(directory: Path) -> tuple[str, str]:
    members = []
    for member in directory.rglob("*"):
        if member.name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "result symlink")
        if stat.S_ISREG(mode):
            members.append(member)
        else:
            require(stat.S_ISDIR(mode), "result special member")
    lines = [f"{sha256(member)}  {member.relative_to(directory).as_posix()}"
             for member in sorted(members, key=lambda item: item.relative_to(directory).as_posix())]
    _write_exclusive(directory / MANIFEST, ("\n".join(lines) + "\n").encode("utf-8"))
    manifest_sha = sha256(directory / MANIFEST)
    _write_exclusive(directory / OUTER,
                     f"{manifest_sha}  {MANIFEST}\n".encode("utf-8"))
    return manifest_sha, sha256(directory / OUTER)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


RESULT = RESULTS / "m1148ca_c1_production_expected_digest_compiler_r1_20260830"
ATTEMPT = RESULTS / ".m1148ca_c1_production_expected_digest_compiler_attempt_consumed"
LOCK = Path("/tmp/m1148ca_c1_production_expected_digest_compiler.lock")
WORK_PREFIX = ".m1148ca_c1_production_expected_digest_compiler_work."
FAILURE_PREFIX = "m1148ca_c1_production_expected_digest_compiler_r1_20260830.failed_or_incomplete."


def source_bounded_self_test() -> dict[str, Any]:
    before = _namespace_paths()
    preflight = source_preflight(True)
    module = load_compiler()
    rows = []
    for record in module.bounded_schedule_records():
        mapping = {field: getattr(record, field) for field in module.SCHEDULE_FIELDS}
        rows.append((json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n").encode("utf-8"))
    payload = b"".join(rows)
    with tempfile.TemporaryDirectory(prefix="m1148ca_bounded_") as temporary:
        path = Path(temporary) / "bounded_schedule.jsonl"
        path.write_bytes(payload)
        compiled = _compile_stream(path, hashlib.sha256(payload).hexdigest(), 9,
                                   module.BOUNDED_GEOMETRY, module)
    require(compiled["authority"]["expected_digest_by_axis"] ==
            module.BOUNDED_GOLDEN_DIGESTS and
            compiled["authority"]["expected_count_by_axis"] ==
                {axis: 8 for axis in module.AXES} and
            before == () and _namespace_paths() == (),
            "bounded launcher/compiler oracle or namespace drift")
    duplicate_rejected = False
    try:
        strict_json_bytes(b'{"axis":"candidate","axis":"candidate"}')
    except Failure:
        duplicate_rejected = True
    require(duplicate_rejected, "duplicate JSON key accepted")
    return {"schema": "m1148ca_c1_production_digest_launcher_bounded_oracle_v1",
            "status": "PASS_BOUNDED_STREAM_SHA_STRICT_JSON_COMPILER__PRODUCTION_STOP",
            "preflight": preflight, "records": 9, "events": 24,
            "expected_digest_by_axis": compiled["authority"]["expected_digest_by_axis"],
            "duplicate_key_attack_rejected": True,
            "production_jsonl_opened": False, "production_events_compiled": 0,
            "production_namespace_mutated": False, "automatic_retry": False}


def production_main() -> dict[str, Any]:
    source_preflight(True)
    resources_before = resource_preflight()
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    attempt_consumed = False
    work = RESULTS / (WORK_PREFIX + f"{os.getpid()}.{time.time_ns()}")
    failure = RESULTS / (FAILURE_PREFIX + f"{os.getpid()}.{time.time_ns()}.quarantine")
    phase = "LOCKED_PREFLIGHT"
    started_wall = time.monotonic(); started_cpu = time.process_time()
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(tuple(path for path in _namespace_paths() if path != LOCK) == (),
                "namespace changed under lock")
        resources_locked = resource_preflight()
        phase = "CONSUME_SINGLE_ATTEMPT_BEFORE_INPUT_OPEN"
        ATTEMPT.mkdir(mode=0o700)
        _write_json(ATTEMPT / "attempt.json", {
            "schema": "m1148ca_c1_production_expected_digest_attempt_r1_v1",
            "status": "M1148CA_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY",
            "source_sha256": sha256(SOURCE_FILE),
            "compiler_source_sha256": COMPILER_SOURCE_SHA,
            "compiler_author_outer_sha256": COMPILER_AUTHOR_ID[2],
            "m1147ca_outer_sha256": M1147_ID[2],
            "m1141_records_sha256_expected": M1141_RECORDS_SHA,
            "schedule_opened_before_attempt": False,
            "automatic_retry": False,
        })
        _seal_tree(ATTEMPT); _fsync_dir(RESULTS); attempt_consumed = True
        work.mkdir(mode=0o700)
        phase = "STREAM_SEALED_SCHEDULE_AND_COMPILE_EXPECTED_DIGEST"
        module = load_compiler()
        compiled = _compile_stream(M1141_RECORDS, M1141_RECORDS_SHA,
                                   EXPECTED_RECORDS, module.PRODUCTION_GEOMETRY, module)
        authority = compiled["authority"]
        require(sum(authority["expected_count_by_axis"].values()) == EXPECTED_EVENTS and
                authority["retained_event_row_or_key_history"] is False,
                "terminal production authority geometry/history drift")
        elapsed_wall = time.monotonic() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        usage = resource.getrusage(resource.RUSAGE_SELF)
        phase = "WRITE_SMALL_AUTHORITY_AND_RESOURCE_RECEIPT"
        _write_json(work / "expected_digest_authority.json", authority)
        _write_json(work / "runtime_resources.json", {
            "schema": "m1148ca_c1_production_digest_runtime_resources_r1_v1",
            "wall_seconds": elapsed_wall, "cpu_seconds": elapsed_cpu,
            "max_rss_kib": usage.ru_maxrss,
            "input_bytes_streamed": compiled["schedule_bytes"],
            "input_records_streamed": compiled["schedule_records"],
            "events_compiled": EXPECTED_EVENTS,
            "state_complexity": "O(axes + axes*24) plus one bounded JSON line and one ScheduleRecord",
            "retained_event_row_or_key_history": False,
        })
        _write_json(work / "receipt.json", {
            "schema": "m1148ca_c1_production_expected_digest_compiler_receipt_r1_v1",
            "status": "PASS_M1148CA_PRODUCTION_EXPECTED_DIGEST_COMPILED__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
            "source_sha256": sha256(SOURCE_FILE),
            "compiler": {"source_sha256": COMPILER_SOURCE_SHA,
                         "author_outer_sha256": COMPILER_AUTHOR_ID[2],
                         "hammer_outer_sha256": M1147_ID[2]},
            "sealed_input": {"records_sha256_expected": M1141_RECORDS_SHA,
                             "records_sha256_observed": compiled["schedule_sha256"],
                             "records": compiled["schedule_records"],
                             "bytes": compiled["schedule_bytes"]},
            "authority": {"authority_id_sha256": authority["authority_id_sha256"],
                          "expected_count_by_axis": authority["expected_count_by_axis"]},
            "resources_before": resources_before,
            "resources_under_lock": resources_locked,
            "attempt_consumed": True, "automatic_retry": False,
            "event_output_written": False,
            "claim_boundary": {"different_author_result_hammer_required": True,
                               "real_producer_replay": False,
                               "traffic_cycles_energy_speedup": False,
                               "paper_citable_performance": False},
        })
        _write_exclusive(work / "RUN_COMPLETE.txt",
                         b"PASS_M1148CA_PRODUCTION_EXPECTED_DIGEST_COMPILED__RESULT_HAMMER_REQUIRED\n")
        manifest_sha, outer_sha = _seal_tree(work)
        _rename_noreplace(work, RESULT); _fsync_dir(RESULTS)
        require(not tuple(RESULTS.glob(FAILURE_PREFIX + "*")),
                "result/failure mutual exclusion drift")
        return {"status": "PASS_M1148CA_PRODUCTION_EXPECTED_DIGEST_COMPILED__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
                "result": str(RESULT), "authority_id_sha256": authority["authority_id_sha256"],
                "manifest_sha256": manifest_sha, "outer_seal_file_sha256": outer_sha,
                "events_compiled": EXPECTED_EVENTS, "automatic_retry": False}
    except BaseException:
        reason = traceback.format_exc()
        if attempt_consumed:
            try:
                work.mkdir(mode=0o700, exist_ok=True)
                _write_json(work / "failure.json", {
                    "schema": "m1148ca_c1_production_expected_digest_failure_r1_v1",
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "phase": phase, "traceback": reason,
                    "attempt_consumed": True, "automatic_retry": False,
                })
                _seal_tree(work); _rename_noreplace(work, failure); _fsync_dir(RESULTS)
                require(not RESULT.exists(), "failure/result mutual exclusion drift")
            except BaseException:
                pass
        raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    require(len(sys.argv) == 1, "M1148CA accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
