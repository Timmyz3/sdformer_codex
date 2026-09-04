#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1199 one-shot production consumer for the hammered M1169 recurrence.

The zero-argument entry is production-sized and MUST NOT be invoked until a
fresh different-author source hammer authorizes its single attempt.  Author
testing calls only ``bounded_source_self_test`` and never opens M1141's 836 MB
schedule.  A successful future run emits only O(axes) terminal accounting.
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
from typing import Any, BinaryIO, Mapping

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HERE = SOURCE_FILE.parent
HW = HERE.parent.parent
RESULTS = HW / "results"

DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

M1169_SOURCE = HERE / "build_m1169_c1_ii2_service_aware_interval_replay_source.py"
M1169_SOURCE_SHA = "bd243ca34760757cadbf9c1104049480197f1fb77bf6ad6ec1071870250ebc4f"
M1169_CONTRACT = HW / "contracts/m1169_c1_ii2_service_aware_interval_replay_source_contract_r1_20260830.json"
M1169_CONTRACT_SHA = "275214c40e1a53b922c1db448dcedff8792f5232124fc1ea5d474360ded861dc"
M1169_AUTHOR = HW / "reviews/m1169_c1_ii2_service_aware_interval_replay_source_author_receipt_r1_20260830"
M1169_AUTHOR_OUTER_FILE_SHA = "19472c48ddfe53b2c5aa2ef9ad647a5b0d378c6d0c7143d789505895412b270b"

M1170 = HW / "reviews/m1170_m1169_c1_ii2_service_aware_interval_replay_source_hammer_r1_20260830"
M1170_RESULT_SHA = "c52c7bb2086e2ad638b7b91656c9c21c1fe517d81fa032a158973a2867f57f16"
M1170_MANIFEST_SHA = "5a3d7a821190c39d4b1213517e81f240ec2cd8e1a1e557832d6c404c74291af0"
M1170_OUTER_FILE_SHA = "0e1cf625aee653b734b2e949a459fe9d8ac3c9b95d830c772a9682b5e7c3bebd"

M1161_RESULT = RESULTS / "m1161ca_c1_production_real_replay_r1_20260830"
M1161_TERMINAL_SHA = "e681c65f25a42b7960b2a68f0709fff2b4c2bfe7d4ac7e69cccf689b9723add8"
M1161_RECEIPT_SHA = "2e6d5ae223f4057e66916ee46c483b523ec233d4a621a070e1438e50b559c751"
M1161_MANIFEST_SHA = "b6c2be64d8cb32fcf0c31ae44070b5efdcb10d0db2661dddb0ec2c4cc3733198"
M1161_OUTER_FILE_SHA = "7bb4ff9dc40a9764d9312c1639a022756305c0170c483854a84c02d2a6cf5b5c"

M1196 = HW / "reviews/m1196_m1161ca_c1_production_real_replay_result_hammer_r1_20260830"
M1196_REVIEW_SHA = "7b1a8b4fa8f1e2a6c361817c65ba198f76e332f5ed09a5199b96c699e241a65e"
M1196_MANIFEST_SHA = "174dee393c022db03dc315266e0d90f4ba45892147d4d69b01b970ffb1f16092"
M1196_OUTER_FILE_SHA = "8b919a0ad6e6ba6638ba6c21a5fbe993dfde0097fddc327001b5c4c5543a8dd0"

M1141 = RESULTS / "m1141ca_c1_production_schedule_release_r1_20260830"
M1141_RECORDS = M1141 / "m1141ca_per_task_schedule_records.jsonl"
M1141_RELEASE = M1141 / "m1141ca_schedule_release.json"
M1141_RECORDS_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
M1141_RECORDS_BYTES = 836_268_740
M1141_RELEASE_SHA = "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c"
M1141_MANIFEST_SHA = "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace"
M1141_OUTER_FILE_SHA = "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a"

CONTRACT = HW / "contracts/m1199_c1_ii2_service_aware_production_consumer_source_contract_r1_20260830.json"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
TASKS = 812_160
EVENTS_PER_AXIS = 70_853_184
EXPECTED_RECORDS = TASKS * len(AXES)
EXPECTED_UID = 1913

RESULT = RESULTS / "m1199_c1_ii2_service_aware_production_replay_r1_20260830"
ATTEMPT = RESULTS / ".m1199_c1_ii2_service_aware_production_replay_attempt_consumed"
LOCK = Path("/tmp/m1199_c1_ii2_service_aware_production_replay.lock")
WORK_PREFIX = ".m1199_c1_ii2_service_aware_production_replay_work."
FAILURE_PREFIX = "m1199_c1_ii2_service_aware_production_replay_r1_20260830.failed_or_incomplete."
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"

MIN_CPUS = 4
MIN_MEM_AVAILABLE = 2 * (1 << 30)
MIN_COMMIT_HEADROOM = 8 * (1 << 30)
MIN_DISK_FREE = 1 * (1 << 30)


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


def verify_regular(path: Path, expected: str,
                   expected_uid: int | None = EXPECTED_UID) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected and
            (expected_uid is None or value.st_uid == expected_uid),
            "identity/owner drift: " + str(path))


def strict_json_bytes(payload: bytes) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def strict_json(path: Path) -> Any:
    return strict_json_bytes(path.read_bytes())


def _manifest_rows(directory: Path, manifest_sha: str,
                   outer_file_sha: str) -> dict[str, str]:
    manifest = directory / MANIFEST
    outer = directory / OUTER
    verify_regular(manifest, manifest_sha)
    verify_regular(outer, outer_file_sha)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, MANIFEST], "outer seal content drift")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "manifest row drift")
        name = fields[1].lstrip("*")
        relative = Path(name)
        require(name not in listed and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "manifest member drift")
        listed[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(listed), "sealed exact member set drift")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return listed


def verify_m1161_and_m1196() -> tuple[dict[str, Any], dict[str, Any]]:
    listed = _manifest_rows(M1161_RESULT, M1161_MANIFEST_SHA,
                            M1161_OUTER_FILE_SHA)
    require(listed.get("producer_replay_terminal.json") == M1161_TERMINAL_SHA and
            listed.get("receipt.json") == M1161_RECEIPT_SHA,
            "M1161 result member identity drift")
    terminal = strict_json(M1161_RESULT / "producer_replay_terminal.json")
    review_rows = _manifest_rows(M1196, M1196_MANIFEST_SHA,
                                 M1196_OUTER_FILE_SHA)
    require(review_rows.get("review.json") == M1196_REVIEW_SHA,
            "M1196 review identity drift")
    review = strict_json(M1196 / "review.json")
    require(terminal["status"].startswith("PASS_REAL_M1137") and
            terminal["sealed_schedule"] == {
                "bytes": M1141_RECORDS_BYTES,
                "records": EXPECTED_RECORDS,
                "sha256": M1141_RECORDS_SHA,
            } and
            review["status"].startswith("PASS_M1196_M1161CA") and
            review["sealed_chain"]["result_outer_seal_file_sha256"] ==
                M1161_OUTER_FILE_SHA and
            review["production_evidence"]["rows_per_axis"] == EVENTS_PER_AXIS,
            "M1161/M1196 admission drift")
    return terminal, review


def verify_m1169_and_m1170() -> ModuleType:
    verify_regular(M1169_SOURCE, M1169_SOURCE_SHA)
    verify_regular(M1169_CONTRACT, M1169_CONTRACT_SHA)
    author_outer = M1169_AUTHOR / OUTER
    verify_regular(author_outer, M1169_AUTHOR_OUTER_FILE_SHA)
    rows = _manifest_rows(M1170, M1170_MANIFEST_SHA, M1170_OUTER_FILE_SHA)
    require(rows.get("hammer_result.json") == M1170_RESULT_SHA,
            "M1170 result identity drift")
    review = strict_json(M1170 / "hammer_result.json")
    require(review["status"].startswith("PASS_M1170_M1169") and
            review["identity"]["m1169_source_sha256"] == M1169_SOURCE_SHA and
            review["production_geometry_proof"]["beats_per_axis"] ==
                EVENTS_PER_AXIS,
            "M1170 admission drift")
    spec = importlib.util.spec_from_file_location("m1199_frozen_m1169", M1169_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1169")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    require((tuple(module.AXES), module.TASKS, module.EVENTS_PER_AXIS) ==
            (AXES, TASKS, EVENTS_PER_AXIS), "M1169 geometry drift")
    return module


def verify_m1141_metadata_only() -> dict[str, Any]:
    verify_regular(M1141_RELEASE, M1141_RELEASE_SHA)
    verify_regular(M1141 / MANIFEST, M1141_MANIFEST_SHA)
    verify_regular(M1141 / OUTER, M1141_OUTER_FILE_SHA)
    rows = {}
    for line in (M1141 / MANIFEST).read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "M1141 manifest row drift")
        rows[fields[1].lstrip("*")] = fields[0]
    require(rows.get(M1141_RECORDS.name) == M1141_RECORDS_SHA and
            rows.get(M1141_RELEASE.name) == M1141_RELEASE_SHA,
            "M1141 member identity drift")
    value = M1141_RECORDS.lstat()
    require(stat.S_ISREG(value.st_mode) and not M1141_RECORDS.is_symlink() and
            value.st_uid == EXPECTED_UID and value.st_size == M1141_RECORDS_BYTES,
            "M1141 records metadata drift")
    release = strict_json(M1141_RELEASE)
    require(release["records"]["count"] == EXPECTED_RECORDS and
            release["records"]["sha256"] == M1141_RECORDS_SHA and
            release["geometry"]["tasks"] == TASKS and
            tuple(release["geometry"]["axes"]) == AXES,
            "M1141 release geometry drift")
    return release


def _namespace_paths() -> tuple[Path, ...]:
    paths = []
    for path in (RESULT, ATTEMPT, LOCK):
        if path.exists() or path.is_symlink():
            paths.append(path)
    paths.extend(sorted(RESULTS.glob(WORK_PREFIX + "*")))
    paths.extend(sorted(RESULTS.glob(FAILURE_PREFIX + "*")))
    return tuple(paths)


_M1169: ModuleType | None = None


def source_preflight(require_fresh_namespace: bool = True) -> dict[str, Any]:
    global _M1169
    verify_regular(DOCS359, DOCS359_SHA)
    terminal, result_hammer = verify_m1161_and_m1196()
    _M1169 = verify_m1169_and_m1170()
    release = verify_m1141_metadata_only()
    contract = strict_json(CONTRACT)
    require(contract["schema"] ==
            "m1199_c1_ii2_service_aware_production_consumer_source_contract_r1_v1",
            "M1199 contract schema drift")
    if require_fresh_namespace:
        require(_namespace_paths() == (), "M1199 production namespace is not fresh")
    return {
        "status": "PASS_M1199_SOURCE_PREFLIGHT__NO_PRODUCTION_SCHEDULE_OPEN",
        "m1161_result_outer_seal_file_sha256": M1161_OUTER_FILE_SHA,
        "m1196_outer_seal_file_sha256": M1196_OUTER_FILE_SHA,
        "m1169_source_sha256": M1169_SOURCE_SHA,
        "m1170_outer_seal_file_sha256": M1170_OUTER_FILE_SHA,
        "m1141_records_sha256_expected": M1141_RECORDS_SHA,
        "m1141_records": release["records"]["count"],
        "m1161_original_weight_schedule_makespans": {
            axis: terminal["row_terminal"]["axes"][axis]
            ["weight_service_makespan_coordinate"] for axis in AXES},
        "m1196_score": result_hammer["score"],
        "production_schedule_opened": False,
        "production_records_consumed": 0,
        "production_execution_authorized_by_author_milestone": False,
    }


def _parse_and_replay(stream: BinaryIO, expected_records: int,
                      expected_bytes: int, expected_sha: str,
                      replay: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    for raw in stream:
        require(raw.endswith(b"\n") and not raw.endswith(b"\r\n") and
                1 < len(raw) <= 65_536, "schedule JSONL framing drift")
        digest.update(raw)
        byte_count += len(raw)
        mapping = strict_json_bytes(raw[:-1])
        record = _M1169.ScheduleRecord.from_mapping(mapping)
        replay.consume_interval(record.axis, record.task_sequence_ordinal,
                                record.requested_cycle_first)
        count += 1
    require(count == expected_records and byte_count == expected_bytes and
            digest.hexdigest() == expected_sha,
            "schedule terminal count/byte/SHA drift")
    return {"records": count, "bytes": byte_count,
            "sha256": digest.hexdigest(), "terminal": replay.finalize()}


def _bounded_payload(m1169: ModuleType) -> bytes:
    rows = []
    starts = {
        "candidate": (0, 5),
        "strongest_zero": (0, 11),
        "same_coordinate_bit": (0, 8),
    }
    for task in range(2):
        source = hashlib.sha256(f"m1199-fixture:{task}".encode()).hexdigest()
        for axis in AXES:
            requested = starts[axis][task]
            mapping = {
                "axis": axis, "chunk": 0, "operator": 0,
                "partition": task, "requested_cycle_first": requested,
                "sample": 0, "source_task_provenance_sha256": source,
                "task_sequence_ordinal": task,
            }
            mapping["schedule_record_provenance_sha256"] = m1169.record_provenance(
                axis, task, 0, 0, 0, task, requested, source)
            rows.append((json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n").encode())
    return b"".join(rows)


def bounded_source_self_test() -> dict[str, Any]:
    before = _namespace_paths()
    preflight = source_preflight(True)
    m1169 = _M1169
    payload = _bounded_payload(m1169)
    replay = m1169.IntervalReplay(2, 7, AXES)
    with tempfile.TemporaryDirectory(prefix="m1199_bounded_") as temporary:
        path = Path(temporary) / "fixture.jsonl"
        path.write_bytes(payload)
        with path.open("rb") as stream:
            value = _parse_and_replay(stream, 6, len(payload),
                                      hashlib.sha256(payload).hexdigest(), replay)
    terminal = value["terminal"]
    require(terminal["axes"]["candidate"]["beats"] == 7 and
            terminal["component_schedule_ratios"]
                ["strongest_zero_over_candidate"]["ratio_decimal"] > 1.0 and
            terminal["expanded_beats"] == 0 and before == _namespace_paths(),
            "bounded terminal/namespace drift")

    attacks = 0
    for mutation in ("drop", "duplicate", "reorder", "sha", "bytes"):
        attacked = payload
        lines = payload.splitlines(keepends=True)
        if mutation == "drop":
            attacked = b"".join(lines[:-1])
        elif mutation == "duplicate":
            attacked = b"".join(lines + [lines[-1]])
        elif mutation == "reorder":
            lines[0], lines[1] = lines[1], lines[0]
            attacked = b"".join(lines)
        elif mutation == "sha":
            attacked = payload.replace(b'"requested_cycle_first":0',
                                       b'"requested_cycle_first":1', 1)
        else:
            attacked = payload + b" "
        try:
            trial = m1169.IntervalReplay(2, 7, AXES)
            with tempfile.TemporaryDirectory(prefix="m1199_attack_") as temporary:
                path = Path(temporary) / "attack.jsonl"
                path.write_bytes(attacked)
                with path.open("rb") as stream:
                    _parse_and_replay(stream, 6, len(payload),
                                      hashlib.sha256(payload).hexdigest(), trial)
        except (Failure, m1169.Failure, json.JSONDecodeError):
            attacks += 1
    require(attacks == 5, "bounded stream attack escaped")
    return {
        "schema": "m1199_c1_ii2_production_consumer_bounded_oracle_v1",
        "status": "PASS_M1199_BOUNDED_CONSUMER_ORACLE__PRODUCTION_STOP",
        "preflight": preflight,
        "records": value["records"],
        "beats_per_axis": 7,
        "terminal": terminal,
        "attacks_rejected": attacks,
        "production_schedule_opened": False,
        "production_records_consumed": 0,
        "production_namespace_mutated": False,
        "component_schedule_only": True,
        "rtl_or_system_speedup": False,
    }


def _meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(
            r"(MemAvailable|CommitLimit|Committed_AS):\s+(\d+)\s+kB", line)
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
            status = (entry / "status").read_text(encoding="utf-8")
            match = re.search(r"^Uid:\s+(\d+)", status, re.M)
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            text = {item.decode("utf-8", "replace") for item in argv if item}
            if (match is not None and int(match.group(1)) == EXPECTED_UID and
                    text.intersection(tokens)):
                conflicts.append(int(entry.name))
        except (FileNotFoundError, PermissionError):
            continue
    return tuple(sorted(conflicts))


def resource_preflight() -> dict[str, int]:
    require(os.getuid() == EXPECTED_UID and RESULTS.stat().st_uid == EXPECTED_UID,
            "uid drift")
    cpus = len(os.sched_getaffinity(0))
    memory = _meminfo()
    statvfs = os.statvfs(RESULTS)
    disk_free = statvfs.f_bavail * statvfs.f_frsize
    commit_headroom = memory["CommitLimit"] - memory["Committed_AS"]
    conflicts = _conflicting_processes()
    require(cpus >= MIN_CPUS and memory["MemAvailable"] >= MIN_MEM_AVAILABLE and
            commit_headroom >= MIN_COMMIT_HEADROOM and disk_free >= MIN_DISK_FREE and
            conflicts == (), "resource/process preflight failed")
    return {"cpus": cpus, "mem_available_bytes": memory["MemAvailable"],
            "commit_headroom_bytes": commit_headroom,
            "disk_free_bytes": disk_free,
            "same_uid_conflicts": len(conflicts)}


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n").encode())


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
             for member in sorted(
                 members, key=lambda item: item.relative_to(directory).as_posix())]
    _write_exclusive(directory / MANIFEST, ("\n".join(lines) + "\n").encode())
    manifest_sha = sha256(directory / MANIFEST)
    _write_exclusive(directory / OUTER,
                     f"{manifest_sha}  {MANIFEST}\n".encode())
    return manifest_sha, sha256(directory / OUTER)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(not destination.exists() and not destination.is_symlink(),
            "publish collision")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable")
    result = renameat2(-100, os.fsencode(source), -100,
                       os.fsencode(destination), 1)
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def production_main() -> dict[str, Any]:
    """Consume the sole attempt and run the O(tasks) replay; no retry."""
    preflight = source_preflight(True)
    resources_before = resource_preflight()
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                      os.O_NOFOLLOW, 0o600)
    attempt_consumed = False
    work = RESULTS / (WORK_PREFIX + f"{os.getpid()}.{time.time_ns()}")
    failure = RESULTS / (FAILURE_PREFIX +
                         f"{os.getpid()}.{time.time_ns()}.quarantine")
    phase = "LOCKED_PREFLIGHT"
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(tuple(path for path in _namespace_paths() if path != LOCK) == (),
                "namespace changed under lock")
        resources_locked = resource_preflight()
        phase = "CONSUME_PERSISTENT_ATTEMPT_BEFORE_SCHEDULE_OPEN"
        ATTEMPT.mkdir(mode=0o700)
        _write_json(ATTEMPT / "attempt.json", {
            "schema": "m1199_c1_ii2_production_attempt_r1_v1",
            "status": "M1199_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY",
            "source_sha256": sha256(SOURCE_FILE),
            "m1161_result_outer_seal_file_sha256": M1161_OUTER_FILE_SHA,
            "m1196_outer_seal_file_sha256": M1196_OUTER_FILE_SHA,
            "m1169_source_sha256": M1169_SOURCE_SHA,
            "m1170_outer_seal_file_sha256": M1170_OUTER_FILE_SHA,
            "m1141_records_sha256_expected": M1141_RECORDS_SHA,
            "schedule_opened_before_attempt": False,
            "automatic_retry": False,
        })
        _seal_tree(ATTEMPT)
        _fsync_dir(RESULTS)
        attempt_consumed = True
        work.mkdir(mode=0o700)

        phase = "STREAM_M1141_TASK_INTERVALS_TO_M1169_RECURRENCE"
        replay = _M1169.IntervalReplay(TASKS, EVENTS_PER_AXIS, AXES)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(M1141_RECORDS, flags)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and
                    before.st_size == M1141_RECORDS_BYTES,
                    "production schedule fd metadata drift")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                result = _parse_and_replay(
                    stream, EXPECTED_RECORDS, M1141_RECORDS_BYTES,
                    M1141_RECORDS_SHA, replay)
            after = os.fstat(fd)
            require((before.st_dev, before.st_ino, before.st_size,
                     before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size,
                     after.st_mtime_ns), "production schedule changed during stream")
        finally:
            os.close(fd)
        terminal = result["terminal"]
        require(all(terminal["axes"][axis]["records"] == TASKS and
                    terminal["axes"][axis]["beats"] == EVENTS_PER_AXIS
                    for axis in AXES), "production terminal conservation drift")

        elapsed_wall = time.monotonic() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        usage = resource.getrusage(resource.RUSAGE_SELF)
        phase = "WRITE_SMALL_TERMINAL_RECEIPTS"
        _write_json(work / "ii2_service_aware_terminal.json", {
            "schema": "m1199_c1_ii2_service_aware_production_terminal_r1_v1",
            "status": "PASS_M1199_EXACT_II2_COMPONENT_SCHEDULE__RESULT_HAMMER_REQUIRED",
            "sealed_schedule": {"records": result["records"],
                                "bytes": result["bytes"],
                                "sha256": result["sha256"]},
            "m1161_result_outer_seal_file_sha256": M1161_OUTER_FILE_SHA,
            "m1196_outer_seal_file_sha256": M1196_OUTER_FILE_SHA,
            "m1169_source_sha256": M1169_SOURCE_SHA,
            "m1170_outer_seal_file_sha256": M1170_OUTER_FILE_SHA,
            "service_terminal": terminal,
            "per_event_output_written": False,
            "retained_schedule_record_or_event_history": False,
            "claim_boundary": {
                "component_weight_service_schedule_only": True,
                "rtl_cycles_or_system_speedup": False,
                "traffic_energy_or_paper_ppa": False,
                "different_author_result_hammer_required": True,
            },
        })
        _write_json(work / "runtime_resources.json", {
            "schema": "m1199_c1_ii2_service_aware_resources_r1_v1",
            "wall_seconds": elapsed_wall,
            "cpu_seconds": elapsed_cpu,
            "max_rss_kib": usage.ru_maxrss,
            "input_bytes_streamed": result["bytes"],
            "input_records_streamed": result["records"],
            "events_expanded": 0,
            "state_complexity": "O(axes) plus one JSON line",
            "retained_schedule_record_or_event_history": False,
        })
        _write_json(work / "receipt.json", {
            "schema": "m1199_c1_ii2_service_aware_receipt_r1_v1",
            "status": "PASS_M1199_II2_PRODUCTION_CONSUMER__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
            "source_sha256": sha256(SOURCE_FILE),
            "source_preflight": preflight,
            "resources_before": resources_before,
            "resources_under_lock": resources_locked,
            "attempt_consumed": True,
            "automatic_retry": False,
            "production_schedule_opened_after_attempt": True,
            "per_event_output_written": False,
            "component_schedule_only": True,
            "rtl_or_system_speedup": False,
        })
        _write_exclusive(work / "RUN_COMPLETE.txt",
                         b"PASS_M1199_II2_PRODUCTION_CONSUMER__RESULT_HAMMER_REQUIRED\n")
        manifest_sha, outer_sha = _seal_tree(work)
        _rename_noreplace(work, RESULT)
        _fsync_dir(RESULTS)
        require(not tuple(RESULTS.glob(FAILURE_PREFIX + "*")),
                "result/failure mutual exclusion drift")
        return {
            "status": "PASS_M1199_II2_PRODUCTION_CONSUMER__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
            "result": str(RESULT),
            "records": result["records"],
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha,
            "automatic_retry": False,
            "component_schedule_only": True,
            "rtl_or_system_speedup": False,
        }
    except BaseException:
        reason = traceback.format_exc()
        if attempt_consumed:
            try:
                work.mkdir(mode=0o700, exist_ok=True)
                _write_json(work / "failure.json", {
                    "schema": "m1199_c1_ii2_service_aware_failure_r1_v1",
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "phase": phase,
                    "traceback": reason,
                    "attempt_consumed": True,
                    "automatic_retry": False,
                })
                _seal_tree(work)
                _rename_noreplace(work, failure)
                _fsync_dir(RESULTS)
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
    require(len(sys.argv) == 1, "M1199 accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
