#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1141CA zero-argument production schedule-release source.

SOURCE ONLY until a different-author hammer authorizes execution.  Importing
this module performs no canonical open and creates no production namespace.
The production entry has zero arguments and is hard-bound to frozen M410.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
import traceback
from typing import Any, Mapping

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HERE = SOURCE_FILE.parent
HW = HERE.parent.parent
M1007_SOURCE = HERE / "m1007_c1_matched_common_charge_address_replay_source.py"
M1007_SOURCE_SHA = "150f22eaa11d219bfa20561b91a38049f14abbc541a6b40db04bd73533ec3442"
M1016_SOURCE = HERE / "run_m1016_c1_full_matched_address_replay.py"
M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1137_SOURCE = HERE / "build_m1137c_c1_real_per_task_weight_beat_hook_source.py"
M1137_SOURCE_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
M1139_SOURCE = HERE / "build_m1139ca_c1_independent_per_task_schedule_authority_source.py"
M1139_SOURCE_SHA = "d18137661517538a8273b696b5f2ada09aff9847c16da0d3a723037e901153a9"
M1140CA = HW / "reviews/m1140ca_m1139ca_c1_independent_per_task_schedule_authority_hammer_r1_20260830"
M1140CA_ID = (
    "486e9fd733a7ce656c8c97538a9597a87ace8b3d8a89643765cc00cccf61a242",
    "1ac3566dabc7965615485e192163accb41e64972344ff3c1335ebb22d2ec1289",
    "f73cafa73ed047abd59730749bf48fcb3f463fca77609aec6085f5b3389fa352",
)
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
ROWS_BYTES = 466_560_000
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

RESULT = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
WORK_PREFIX = ".m1141ca_c1_production_schedule_release_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
RECORDS_NAME = "m1141ca_per_task_schedule_records.jsonl"
RELEASE_NAME = "m1141ca_schedule_release.json"
MANIFEST_NAME = "SHA256SUMS"
OUTER_NAME = "SHA256SUMS.seal.sha256"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
BYTES_PER_LINE = 9


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            _sha_path(path) == expected, "identity drift: " + str(path))


def _verify_flat(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    review = directory / "review.json"
    manifest = directory / MANIFEST_NAME
    outer = directory / OUTER_NAME
    _verify_regular(review, identity[0])
    _verify_regular(manifest, identity[1])
    _verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], MANIFEST_NAME], "M1140CA outer content drift")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        relative = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "M1140CA manifest member drift")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST_NAME, OUTER_NAME}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "M1140CA symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "M1140CA special member")
    require(actual == set(expected), "M1140CA exact member set drift")
    for name, digest in expected.items():
        _verify_regular(directory / name, digest)
    return json.loads(review.read_text(encoding="utf-8"))


def _load_m1007():
    _verify_regular(M1007_SOURCE, M1007_SOURCE_SHA)
    spec = importlib.util.spec_from_file_location("m1141ca_frozen_m1007", M1007_SOURCE)
    require(spec is not None and spec.loader is not None, "M1007 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Geometry:
    samples: int
    operators: int
    partitions: int
    rows_per_phase: int
    row_tile: int
    blocks: int
    commit_cycles: int
    weight_events: int
    dma_events: int

    @property
    def chunks(self) -> int:
        return math.ceil(self.rows_per_phase / self.row_tile)

    @property
    def tasks_per_sample(self) -> int:
        return self.operators * self.chunks * self.partitions

    @property
    def total_tasks(self) -> int:
        return self.samples * self.tasks_per_sample

    @property
    def total_records(self) -> int:
        return self.total_tasks * len(AXES)

    @property
    def raw_rows(self) -> int:
        return self.samples * self.operators * self.partitions * self.rows_per_phase

    def validate(self) -> None:
        require(all(type(value) is int and value > 0 for value in (
                    self.samples, self.operators, self.partitions,
                    self.rows_per_phase, self.row_tile, self.blocks)) and
                type(self.commit_cycles) is int and self.commit_cycles >= 0 and
                type(self.weight_events) is int and self.weight_events >= 0 and
                type(self.dma_events) is int and self.dma_events >= 0,
                "geometry drift")


PRODUCTION_GEOMETRY = Geometry(10, 4, 432, 3000, 64, 8, 96_000,
                               70_853_184, 1_476_108)
require(PRODUCTION_GEOMETRY.total_tasks == 812_160 and
        PRODUCTION_GEOMETRY.total_records == 2_436_480 and
        PRODUCTION_GEOMETRY.raw_rows * BYTES_PER_LINE == ROWS_BYTES,
        "frozen production geometry drift")


def _u64(value: int) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), "u64 drift")
    return struct.pack(">Q", value)


def _task_id(geometry: Geometry, sample: int, operator: int,
             chunk: int, partition: int) -> int:
    geometry.validate()
    require(0 <= sample < geometry.samples and 0 <= operator < geometry.operators and
            0 <= chunk < geometry.chunks and 0 <= partition < geometry.partitions,
            "task coordinate drift")
    return (((sample * geometry.operators + operator) * geometry.chunks + chunk) *
            geometry.partitions + partition)


def _phase_index(geometry: Geometry, sample: int, operator: int,
                 partition: int) -> int:
    return (sample * geometry.operators + operator) * geometry.partitions + partition


def _quota(total: int, index: int, population: int) -> int:
    require(0 <= index < population and total >= 0, "quota coordinate drift")
    return ((index + 1) * total) // population - (index * total) // population


def _raw_task_provenance(task: int, sample: int, operator: int, chunk: int,
                         partition: int, preprocess: Mapping[str, int],
                         work: Mapping[str, int], raw_sha: str) -> str:
    require(tuple(preprocess) == AXES and tuple(work) == AXES and
            re.fullmatch(r"[0-9a-f]{64}", raw_sha) is not None,
            "raw task provenance input drift")
    payload = [b"M1139CA_PRIOR_TASK\x00\x01", bytes.fromhex(M1016_SOURCE_SHA),
               bytes.fromhex(M1102_SOURCE_SHA), _u64(task), _u64(sample),
               _u64(operator), _u64(chunk), _u64(partition), bytes.fromhex(raw_sha)]
    for axis in AXES:
        payload.extend((_u64(preprocess[axis]), _u64(work[axis])))
    return hashlib.sha256(b"".join(payload)).hexdigest()


def _record_provenance(axis: str, task: int, sample: int, operator: int,
                       chunk: int, partition: int, requested: int,
                       source_task_provenance: str) -> str:
    require(axis in AXES and re.fullmatch(r"[0-9a-f]{64}",
                                         source_task_provenance) is not None,
            "record provenance input drift")
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(M1016_SOURCE_SHA),
        bytes.fromhex(M1102_SOURCE_SHA), bytes.fromhex(M1137_SOURCE_SHA),
        struct.pack(">B", AXES.index(axis)), _u64(task), _u64(sample),
        _u64(operator), _u64(chunk), _u64(partition), _u64(requested),
        bytes.fromhex(source_task_provenance),
    ))
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScheduleRecord:
    axis: str
    task_sequence_ordinal: int
    sample: int
    operator: int
    chunk: int
    partition: int
    requested_cycle_first: int
    source_task_provenance_sha256: str
    schedule_record_provenance_sha256: str

    def validate(self) -> None:
        require(self.axis in AXES and all(type(value) is int and value >= 0 for value in (
                    self.task_sequence_ordinal, self.sample, self.operator,
                    self.chunk, self.partition, self.requested_cycle_first)) and
                self.schedule_record_provenance_sha256 == _record_provenance(
                    self.axis, self.task_sequence_ordinal, self.sample, self.operator,
                    self.chunk, self.partition, self.requested_cycle_first,
                    self.source_task_provenance_sha256),
                "schedule record schema/provenance drift")


@dataclass
class _AxisState:
    previous_start: int | None = None
    previous_work: int = 0
    sample_offset: int = 0
    last_requested: int | None = None
    records: int = 0


class ExactScheduleRecurrence:
    """Independent M1016 recurrence with O(axes) retained state."""
    def __init__(self, geometry: Geometry, sink):
        geometry.validate()
        require(callable(sink), "schedule sink must be callable")
        self._geometry = geometry
        self._sink = sink
        self._axis = {axis: _AxisState() for axis in AXES}
        self._next_task = 0

    def consume(self, task: int, sample: int, operator: int, chunk: int,
                partition: int, preprocess: Mapping[str, int],
                work: Mapping[str, int], raw_sha: str) -> None:
        require(task == self._next_task == _task_id(
                    self._geometry, sample, operator, chunk, partition),
                "task missing, duplicate, or out of order")
        require(tuple(preprocess) == AXES and tuple(work) == AXES,
                "axis map/order drift")
        source_provenance = _raw_task_provenance(
            task, sample, operator, chunk, partition, preprocess, work, raw_sha)
        for axis in AXES:
            state = self._axis[axis]
            pre = preprocess[axis]
            amount = work[axis]
            require(type(pre) is int and pre >= 0 and type(amount) is int and
                    amount >= 0 and amount % self._geometry.blocks == 0,
                    "preprocess/work drift")
            start = (pre if state.previous_start is None else
                     state.previous_start + max(state.previous_work, pre) + 2)
            requested = state.sample_offset + start - pre
            require(state.last_requested is None or requested >= state.last_requested,
                    "requested cycle regressed")
            record = ScheduleRecord(
                axis, task, sample, operator, chunk, partition, requested,
                source_provenance,
                _record_provenance(axis, task, sample, operator, chunk, partition,
                                   requested, source_provenance))
            self._sink(record)
            state.previous_start = start
            state.previous_work = amount
            state.last_requested = requested
            state.records += 1
        self._next_task += 1
        if self._next_task % self._geometry.tasks_per_sample == 0:
            for axis in AXES:
                state = self._axis[axis]
                require(state.previous_start is not None, "empty sample state")
                state.sample_offset += (state.previous_start + state.previous_work + 2 +
                                        self._geometry.commit_cycles)
                state.previous_start = None
                state.previous_work = 0

    def finalize(self) -> dict[str, Any]:
        require(self._next_task == self._geometry.total_tasks and
                all(state.records == self._geometry.total_tasks
                    for state in self._axis.values()),
                "terminal task/axis record conservation mismatch")
        return {
            "tasks": self._next_task,
            "records_by_axis": {axis: self._axis[axis].records for axis in AXES},
            "last_requested_cycle_by_axis": {
                axis: self._axis[axis].last_requested for axis in AXES},
            "state_complexity": "O(axes)",
            "retained_record_or_key_history": False,
        }


class _StreamingRecordSink:
    def __init__(self, stream):
        self._stream = stream
        self.count = 0
        self.axis_counts = {axis: 0 for axis in AXES}
        self.stream_digest = hashlib.sha256()
        self.provenance_digest = hashlib.sha256()

    def __call__(self, record: ScheduleRecord) -> None:
        require(type(record) is ScheduleRecord, "exact schedule record required")
        record.validate()
        expected_task = self.count // len(AXES)
        expected_axis = AXES[self.count % len(AXES)]
        require(record.task_sequence_ordinal == expected_task and
                record.axis == expected_axis, "record missing, duplicate, or out of order")
        encoded = (json.dumps(record.__dict__, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n").encode()
        self._stream.write(encoded)
        self.stream_digest.update(encoded)
        self.provenance_digest.update(bytes.fromhex(
            record.schedule_record_provenance_sha256))
        self.count += 1
        self.axis_counts[record.axis] += 1

    def finalize(self, total_tasks: int) -> None:
        require(type(total_tasks) is int and total_tasks >= 0 and
                self.count == total_tasks * len(AXES) and
                self.axis_counts == {axis: total_tasks for axis in AXES},
                "stream record conservation mismatch")


def _preprocess_and_work(m1007, geometry: Geometry, task: int,
                         masks: list[int]) -> tuple[dict[str, int], dict[str, int]]:
    rows = len(masks)
    require(0 < rows <= geometry.row_tile and
            all(type(mask) is int and 0 <= mask <= 0xffff for mask in masks),
            "task mask drift")
    weight = _quota(geometry.weight_events, task, geometry.total_tasks)
    dma = _quota(geometry.dma_events, task, geometry.total_tasks)
    common = max(math.ceil(rows / 64), weight, dma, geometry.blocks * 2)
    capture = math.ceil(rows / 8)
    search_rows = sum(mask.bit_count() > 1 for mask in masks)
    frontend = {
        "candidate": capture + search_rows * math.ceil(rows / 64) + 17 * capture + 2,
        "strongest_zero": rows + 5,
        "same_coordinate_bit": math.ceil(rows / 8) + 2,
    }
    preprocess = {axis: max(frontend[axis], common) for axis in AXES}
    trace = list(m1007.parent_cycle_trace(masks))
    candidate_summary = m1007.parent_summary(trace)
    input_nnz = sum(mask.bit_count() for mask in masks)
    work = {
        "candidate": int(candidate_summary["cycles"]) * geometry.blocks,
        "strongest_zero": input_nnz * geometry.blocks,
        "same_coordinate_bit": input_nnz * geometry.blocks,
    }
    return preprocess, work


def _open_canonical_nofollow(path: Path, expected_bytes: int) -> tuple[int, os.stat_result]:
    require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "canonical path is not regular")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and opened.st_size == expected_bytes,
                "canonical opened identity/size drift")
        path_stat = path.stat(follow_symlinks=False)
        require((opened.st_dev, opened.st_ino) == (path_stat.st_dev, path_stat.st_ino),
                "canonical path replacement before open")
        return fd, opened
    except BaseException:
        os.close(fd)
        raise


def _fd_hash(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(fd, 1 << 20, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def _verify_open_identity(path: Path, fd: int, opened: os.stat_result,
                          expected_sha: str) -> None:
    after = os.fstat(fd)
    path_after = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(path_after.st_mode) and not path.is_symlink() and
            (after.st_dev, after.st_ino, after.st_size) ==
            (opened.st_dev, opened.st_ino, opened.st_size) and
            (path_after.st_dev, path_after.st_ino) == (opened.st_dev, opened.st_ino),
            "canonical path replacement/identity drift")
    require(_fd_hash(fd) == expected_sha, "canonical content changed during replay")


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(not destination.exists() and not destination.is_symlink(),
            "publish/quarantine collision")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        number = ctypes.get_errno()
        raise Failure("renameat2 noreplace failed: " + os.strerror(number))


def _seal_directory(stage: Path) -> tuple[str, str]:
    members = sorted(path for path in stage.iterdir()
                     if path.name not in {MANIFEST_NAME, OUTER_NAME})
    require(members and all(path.is_file() and not path.is_symlink() for path in members),
            "seal member drift")
    manifest_payload = "".join(f"{_sha_path(path)}  {path.name}\n" for path in members)
    _write_exclusive(stage / MANIFEST_NAME, manifest_payload.encode())
    manifest_sha = _sha_path(stage / MANIFEST_NAME)
    _write_exclusive(stage / OUTER_NAME,
                     f"{manifest_sha}  {MANIFEST_NAME}\n".encode())
    _fsync_dir(stage)
    return manifest_sha, _sha_path(stage / OUTER_NAME)


def _source_preflight() -> dict[str, Any]:
    _verify_regular(SOURCE_FILE, _sha_path(SOURCE_FILE))
    _verify_regular(M1016_SOURCE, M1016_SOURCE_SHA)
    _verify_regular(M1102_SOURCE, M1102_SOURCE_SHA)
    _verify_regular(M1137_SOURCE, M1137_SOURCE_SHA)
    _verify_regular(M1139_SOURCE, M1139_SOURCE_SHA)
    _verify_regular(DOCS359, DOCS359_SHA)
    review = _verify_flat(M1140CA, M1140CA_ID)
    require(review["status"] ==
            "PASS_M1140CA_INDEPENDENT_BOUNDED_SCHEDULE_HAMMER__AUTHOR_PRODUCTION_RELEASE_SOURCE_ONLY" and
            review["authorization"]["production_schedule_execution"] is False,
            "M1140CA authorization drift")
    return {"m1140ca_outer_seal_file_sha256": M1140CA_ID[2],
            "m410_opened": False, "production_executed": False}


def _execute_release(rows: Path, expected_sha: str, expected_bytes: int,
                     geometry: Geometry, result: Path) -> dict[str, Any]:
    """Fixture-capable core. Only zero-argument production_main binds real paths."""
    geometry.validate()
    require(expected_bytes == geometry.raw_rows * BYTES_PER_LINE,
            "canonical byte geometry drift")
    require(result.parent.is_dir() and not result.exists() and not result.is_symlink(),
            "result namespace collision")
    stage = result.parent / ("." + result.name + ".private_staging.%d.%d" %
                             (os.getpid(), time.time_ns()))
    quarantine = result.parent / (result.name + ".failed_or_incomplete.%d.%d.quarantine" %
                                  (os.getpid(), time.time_ns()))
    require(not stage.exists() and not quarantine.exists(), "private namespace collision")
    stage.mkdir(mode=0o700)
    fd = -1
    phase = "OPEN_NOFOLLOW"
    try:
        fd, opened = _open_canonical_nofollow(rows, expected_bytes)
        phase = "VERIFY_FROZEN_IDENTITY"
        require(_fd_hash(fd) == expected_sha, "canonical SHA-256 identity drift")
        m1007 = _load_m1007()
        phase = "STREAM_EXACT_SCHEDULE"
        records_path = stage / RECORDS_NAME
        records_fd = os.open(records_path,
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600)
        try:
            with os.fdopen(records_fd, "wb", closefd=False) as stream:
                sink = _StreamingRecordSink(stream)
                recurrence = ExactScheduleRecurrence(geometry, sink)
                for sample in range(geometry.samples):
                    for operator in range(geometry.operators):
                        for chunk in range(geometry.chunks):
                            count = min(geometry.row_tile,
                                        geometry.rows_per_phase - chunk * geometry.row_tile)
                            for partition in range(geometry.partitions):
                                task = _task_id(geometry, sample, operator, chunk, partition)
                                phase_index = _phase_index(geometry, sample, operator, partition)
                                offset = ((phase_index * geometry.rows_per_phase +
                                           chunk * geometry.row_tile) * BYTES_PER_LINE)
                                raw = os.pread(fd, count * BYTES_PER_LINE, offset)
                                require(len(raw) == count * BYTES_PER_LINE,
                                        "short canonical task read")
                                lines = raw.splitlines()
                                require(len(lines) == count and all(
                                    re.fullmatch(rb"[0-9a-f]{8}", line) is not None
                                    for line in lines), "canonical row parse drift")
                                masks = [int(line, 16) & 0xffff for line in lines]
                                preprocess, work = _preprocess_and_work(
                                    m1007, geometry, task, masks)
                                recurrence.consume(
                                    task, sample, operator, chunk, partition,
                                    preprocess, work, hashlib.sha256(raw).hexdigest())
                terminal = recurrence.finalize()
                sink.finalize(geometry.total_tasks)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(records_fd)
        phase = "REVERIFY_OPEN_FILE_IDENTITY"
        _verify_open_identity(rows, fd, opened, expected_sha)
        require(_sha_path(records_path) == sink.stream_digest.hexdigest(),
                "stream file SHA mismatch")
        release = {
            "schema": "m1141ca_c1_production_schedule_release_r1_v1",
            "status": "PASS_EXACT_PRODUCTION_SCHEDULE_RELEASE__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
            "source_rows": {"sha256": expected_sha, "bytes": expected_bytes,
                            "no_follow_single_fd": True,
                            "identity_reverified_after_stream": True},
            "geometry": {"samples": geometry.samples, "operators": geometry.operators,
                         "partitions": geometry.partitions, "chunks": geometry.chunks,
                         "tasks": geometry.total_tasks, "axes": list(AXES),
                         "records": geometry.total_records},
            "records": {"file": RECORDS_NAME, "count": sink.count,
                        "sha256": sink.stream_digest.hexdigest(),
                        "schedule_provenance_sha256": sink.provenance_digest.hexdigest(),
                        "axis_counts": sink.axis_counts,
                        "axis_order_within_each_task": list(AXES)},
            "terminal": terminal,
            "state_complexity": "O(axes) plus one bounded row tile",
            "retained_record_or_key_history": False,
            "authority": {"m1016_source_sha256": M1016_SOURCE_SHA,
                          "m1139ca_source_sha256": M1139_SOURCE_SHA,
                          "m1140ca_outer_seal_file_sha256": M1140CA_ID[2]},
            "claim_boundary": {"digest_compiler": False, "real_driver": False,
                               "full_replay": False, "eda": False,
                               "traffic_cycles_energy_speedup": False,
                               "paper_citable": False},
        }
        _write_exclusive(stage / RELEASE_NAME,
                         (json.dumps(release, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n").encode())
        phase = "ATOMIC_SEAL_AND_PUBLISH"
        manifest_sha, outer_file_sha = _seal_directory(stage)
        _rename_noreplace(stage, result)
        _fsync_dir(result.parent)
        return {"status": release["status"], "result": str(result),
                "records": sink.count, "manifest_sha256": manifest_sha,
                "outer_seal_file_sha256": outer_file_sha}
    except BaseException:
        failure_text = traceback.format_exc()
        if stage.exists():
            try:
                _write_exclusive(stage / "failure.json", (json.dumps({
                    "schema": "m1141ca_c1_schedule_release_failure_r1_v1",
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "phase": phase, "automatic_retry": False,
                }, sort_keys=True) + "\n").encode())
                _write_exclusive(stage / "failure.txt", failure_text.encode())
                _seal_directory(stage)
                _rename_noreplace(stage, quarantine)
                _fsync_dir(result.parent)
            except BaseException:
                pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def source_static_self_test() -> dict[str, Any]:
    preflight = _source_preflight()
    return {
        "status": "PASS_M1141CA_SOURCE_PREFLIGHT__NO_M410_OPEN_NO_PRODUCTION",
        "preflight": preflight,
        "zero_argument_production_entry": True,
        "production_geometry": {"tasks": PRODUCTION_GEOMETRY.total_tasks,
                                "records": PRODUCTION_GEOMETRY.total_records,
                                "axes": list(AXES)},
        "canonical_opened": False, "production_records": 0,
        "production_result_created": False, "digest_compiler": False,
        "real_driver": False, "full_replay": False, "eda": False,
    }


def production_main() -> dict[str, Any]:
    _source_preflight()
    require(not RESULT.exists() and not RESULT.is_symlink(),
            "production result already exists")
    return _execute_release(ROWS, ROWS_SHA, ROWS_BYTES, PRODUCTION_GEOMETRY, RESULT)


def main() -> int:
    require(len(sys.argv) == 1, "M1141CA accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
