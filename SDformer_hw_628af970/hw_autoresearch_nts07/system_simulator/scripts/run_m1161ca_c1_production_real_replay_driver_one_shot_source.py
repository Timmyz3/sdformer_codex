#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot C1 real producer replay driver over the sealed task schedule.

The normal zero-argument entry is production-sized and MUST NOT be invoked
until a fresh different-author hammer authorizes it.  The author milestone
imports this module and calls ``source_bounded_self_test`` only; that path uses
two controlled tasks and never opens the production schedule.

The resulting cycle fields, if production is later authorized, describe the
M1135C 1RW weight-service schedule only.  They are neither RTL cycles nor a
system speedup measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
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
import struct
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

M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1135_SOURCE = HERE / "build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
M1135_SOURCE_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
M1137_SOURCE = HERE / "build_m1137c_c1_real_per_task_weight_beat_hook_source.py"
M1137_SOURCE_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
M1138 = HW / "reviews/m1138c_m1137c_c1_real_per_task_weight_beat_hook_hammer_r1_20260830"
M1138_ID = (
    "83356f85ce1d7a3be950d50fc226dd193b1c19e537c6764d94bd07cb6d9fe41a",
    "67bb65e27418fb83657e815cc4ef95d190d9e09c69d2d86cb1306bae4e9c2c39",
    "f55db3e6daed3f10c44e60caea81e419af36db08f71ca164b076eac7baea72fc",
)

M1141 = RESULTS / "m1141ca_c1_production_schedule_release_r1_20260830"
M1141_RECORDS = M1141 / "m1141ca_per_task_schedule_records.jsonl"
M1141_RELEASE = M1141 / "m1141ca_schedule_release.json"
M1141_RECORDS_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
M1141_RECORDS_BYTES = 836_268_740
M1141_RELEASE_SHA = "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c"
M1141_MANIFEST_SHA = "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace"
M1141_OUTER_SHA = "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a"

M1145 = HW / "reviews/m1145ca_m1143ca_c1_production_result_hammer_r1_20260830"
M1145_ID = (
    "cfe7bf030743c4bcc098d267c69422ae1e76238696902e8dc601ea8143ee208d",
    "7dbc93256d915962a4f83e860aff9aac0bb3b62b1c76113509daef74b852eb4c",
    "8dcc8e84ec8c6273f155c418078fac92b96ef851768ec6cb2066c64ab3d3423e",
)

M1148 = RESULTS / "m1148ca_c1_production_expected_digest_compiler_r1_20260830"
M1148_AUTHORITY = M1148 / "expected_digest_authority.json"
M1148_AUTHORITY_SHA = "c45fd835db7fddca268a8891051a5d24bf9492806c6e3610b8e52b8730e705b2"
M1148_MANIFEST_SHA = "6fc0048c84409cc7afc114f540ad17c83a2a00d0d1db19b0684881d8f2dadf5f"
M1148_OUTER_SHA = "98d69e2799af300b2babe72ac3cceb97f3ecc9a435ac7d12c6c7b8fdd13979d1"
AUTHORITY_ID = "a53f0141ff9f01b32ed8920c0c3fc10a2d70848773e9b99e02b8905ea05a6fbf"

M1157 = HW / "reviews/m1157ca_m1148ca_c1_production_expected_digest_result_hammer_r1_20260830"
M1157_ID = (
    "495fcca0bc853a993eb413d64d64b169838928b9e571291b7d4906e343150417",
    "d5a5b568134cc9bba013b6e501e48c04aecb25bf988b229ebeda0509c14c3280",
    "0dde25832d4af29f983bf6e9aa4573de55835677f668541f2076a531e2b913ee",
)

CONTRACT = HW / "contracts/m1161ca_c1_production_real_replay_driver_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "93471a51d5f9d9270ece1629688b10b0cf88047abed9a5e7b6e71048cd63ef63",
    "89345e94816a72f3672920d4eb9c984afa085789fc47213ef8c981b824f437ea",
    "5c7fdc73e9a69211fea340fa6c9862d19531df551176aa0351f6c914a2f12272",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
SCHEDULE_FIELDS = (
    "axis", "chunk", "operator", "partition", "requested_cycle_first",
    "sample", "schedule_record_provenance_sha256",
    "source_task_provenance_sha256", "task_sequence_ordinal",
)
SAMPLES = 10
OPERATORS = 4
CHUNKS = 47
PARTITIONS = 432
TASKS = 812_160
EXPECTED_RECORDS = 2_436_480
EVENTS_PER_AXIS = 70_853_184
EXPECTED_EVENTS = 212_559_552
EXPECTED_UID = 1913

MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
RESULT = RESULTS / "m1161ca_c1_production_real_replay_r1_20260830"
ATTEMPT = RESULTS / ".m1161ca_c1_production_real_replay_attempt_consumed"
LOCK = Path("/tmp/m1161ca_c1_production_real_replay.lock")
WORK_PREFIX = ".m1161ca_c1_production_real_replay_work."
FAILURE_PREFIX = "m1161ca_c1_production_real_replay_r1_20260830.failed_or_incomplete."
MIN_CPUS = 4
MIN_MEM_AVAILABLE = 4 * (1 << 30)
MIN_COMMIT_HEADROOM = 12 * (1 << 30)
MIN_DISK_FREE = 2 * (1 << 30)


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


def verify_double_contract() -> None:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    verify_regular(CONTRACT, CONTRACT_ID[0])
    verify_regular(side, CONTRACT_ID[1])
    verify_regular(outer, CONTRACT_ID[2])
    require(side.read_text(encoding="utf-8").split() ==
            [CONTRACT_ID[0], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [CONTRACT_ID[1], side.name],
            "contract double-seal content drift")


def _manifest_rows(directory: Path, manifest_sha: str,
                   outer_sha: str) -> dict[str, str]:
    manifest = directory / MANIFEST
    outer = directory / OUTER
    verify_regular(manifest, manifest_sha)
    verify_regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, MANIFEST],
            "outer seal content drift")
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
    return listed


def verify_tree(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink() and
            directory.stat().st_uid == EXPECTED_UID, "authority directory drift")
    listed = _manifest_rows(directory, identity[1], identity[2])
    require(listed.get("review.json") == identity[0], "review manifest drift")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return strict_json(directory / "review.json")


def verify_m1141_metadata_only() -> dict[str, Any]:
    require(M1141.is_dir() and not M1141.is_symlink(), "M1141 directory drift")
    listed = _manifest_rows(M1141, M1141_MANIFEST_SHA, M1141_OUTER_SHA)
    require(listed.get(M1141_RECORDS.name) == M1141_RECORDS_SHA and
            listed.get(M1141_RELEASE.name) == M1141_RELEASE_SHA,
            "M1141 manifest identity drift")
    for name, digest in listed.items():
        if name != M1141_RECORDS.name:
            verify_regular(M1141 / name, digest)
    value = M1141_RECORDS.lstat()
    require(stat.S_ISREG(value.st_mode) and not M1141_RECORDS.is_symlink() and
            value.st_uid == EXPECTED_UID and value.st_size == M1141_RECORDS_BYTES,
            "M1141 schedule metadata drift")
    return strict_json(M1141_RELEASE)


def verify_m1148() -> dict[str, Any]:
    listed = _manifest_rows(M1148, M1148_MANIFEST_SHA, M1148_OUTER_SHA)
    require(listed.get(M1148_AUTHORITY.name) == M1148_AUTHORITY_SHA,
            "M1148 authority manifest drift")
    for name, digest in listed.items():
        verify_regular(M1148 / name, digest)
    return strict_json(M1148_AUTHORITY)


def load_frozen(path: Path, expected: str, name: str) -> ModuleType:
    verify_regular(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_M1137: ModuleType | None = None


def load_m1137() -> ModuleType:
    global _M1137
    if _M1137 is None:
        _M1137 = load_frozen(M1137_SOURCE, M1137_SOURCE_SHA,
                             "m1161ca_frozen_m1137")
    return _M1137


def _hex64(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _u64(value: int) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), "u64 drift")
    return struct.pack(">Q", value)


def task_index(sample: int, operator: int, chunk: int, partition: int) -> int:
    require(all(type(value) is int for value in
                (sample, operator, chunk, partition)) and
            0 <= sample < SAMPLES and 0 <= operator < OPERATORS and
            0 <= chunk < CHUNKS and 0 <= partition < PARTITIONS,
            "task coordinate drift")
    return (((sample * OPERATORS + operator) * CHUNKS + chunk) *
            PARTITIONS + partition)


def record_provenance(axis: str, task: int, sample: int, operator: int,
                      chunk: int, partition: int, requested: int,
                      source_task_provenance: str) -> str:
    require(axis in AXES and _hex64(source_task_provenance),
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
    chunk: int
    operator: int
    partition: int
    requested_cycle_first: int
    sample: int
    schedule_record_provenance_sha256: str
    source_task_provenance_sha256: str
    task_sequence_ordinal: int

    @classmethod
    def from_mapping(cls, mapping: Any) -> "ScheduleRecord":
        require(type(mapping) is dict and set(mapping) == set(SCHEDULE_FIELDS),
                "schedule record exact field set drift")
        record = cls(**mapping)
        record.validate()
        return record

    def validate(self) -> None:
        require(self.axis in AXES and all(type(value) is int and value >= 0 for value in (
                    self.chunk, self.operator, self.partition,
                    self.requested_cycle_first, self.sample,
                    self.task_sequence_ordinal)) and
                _hex64(self.source_task_provenance_sha256) and
                self.task_sequence_ordinal == task_index(
                    self.sample, self.operator, self.chunk, self.partition) and
                self.schedule_record_provenance_sha256 == record_provenance(
                    self.axis, self.task_sequence_ordinal, self.sample,
                    self.operator, self.chunk, self.partition,
                    self.requested_cycle_first,
                    self.source_task_provenance_sha256),
                "schedule record schema/coordinate/provenance drift")


@dataclass
class _RowAxisState:
    count: int = 0
    bytes: int = 0
    native_activations: int = 0
    stall_cycles: int = 0
    stalled_transactions: int = 0
    first_requested_cycle: int | None = None
    last_requested_cycle: int | None = None
    first_scheduled_cycle: int | None = None
    last_scheduled_cycle: int | None = None
    max_scheduled_cycle: int | None = None
    digest: Any = None

    def __post_init__(self) -> None:
        self.digest = hashlib.sha256()


class OAxesRowReceiptSink:
    """Validate and summarize addressed rows with fixed O(axes) state."""
    def __init__(self) -> None:
        self._axis = {axis: _RowAxisState() for axis in AXES}

    def __call__(self, row: Any) -> None:
        row.validate()
        require(row.axis in AXES, "row axis drift")
        state = self._axis[row.axis]
        require(row.service_beat_ordinal == state.count and
                row.store_transaction_ordinal == state.count,
                "row ordinal discontinuity")
        payload = json.dumps({
            "axis": row.axis,
            "requested_cycle": row.requested_cycle,
            "scheduled_cycle": row.cycle,
            "stall_cycles": row.stall_cycles,
            "logical_bank": row.logical_bank,
            "half_slot": row.half_slot,
            "logical_row": row.logical_row,
            "local_row": row.local_row,
            "native_slices": list(row.native_slices),
            "bytes": row.bytes,
            "native_macro_activations": row.native_macro_activations,
            "service_beat_ordinal": row.service_beat_ordinal,
            "store_transaction_ordinal": row.store_transaction_ordinal,
            "source_task_id": row.source_task_id,
            "source_local_ordinal": row.source_local_ordinal,
            "source_row_provenance_sha256": row.source_row_provenance_sha256,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        candidate = state.digest.copy(); candidate.update(payload)
        if state.last_requested_cycle is not None:
            require(row.requested_cycle >= state.last_requested_cycle,
                    "row requested cycle regressed")
        state.digest = candidate
        state.count += 1
        state.bytes += row.bytes
        state.native_activations += row.native_macro_activations
        state.stall_cycles += row.stall_cycles
        state.stalled_transactions += int(row.stall_cycles > 0)
        if state.first_requested_cycle is None:
            state.first_requested_cycle = row.requested_cycle
            state.first_scheduled_cycle = row.cycle
        state.last_requested_cycle = row.requested_cycle
        state.last_scheduled_cycle = row.cycle
        state.max_scheduled_cycle = (row.cycle if state.max_scheduled_cycle is None
                                     else max(state.max_scheduled_cycle, row.cycle))

    def finalize(self, expected_count: int) -> dict[str, Any]:
        axes = {}
        for axis in AXES:
            state = self._axis[axis]
            require(state.count == expected_count and
                    state.first_scheduled_cycle is not None and
                    state.last_scheduled_cycle is not None and
                    state.max_scheduled_cycle is not None,
                    "row sink terminal count/cycle drift")
            axes[axis] = {
                "rows": state.count,
                "bytes": state.bytes,
                "native_activations": state.native_activations,
                "stall_cycles": state.stall_cycles,
                "stalled_transactions": state.stalled_transactions,
                "first_requested_cycle": state.first_requested_cycle,
                "last_requested_cycle": state.last_requested_cycle,
                "first_scheduled_cycle": state.first_scheduled_cycle,
                "last_scheduled_cycle": state.last_scheduled_cycle,
                "max_scheduled_cycle": state.max_scheduled_cycle,
                "weight_service_makespan_coordinate": state.max_scheduled_cycle + 1,
                "row_digest_sha256": state.digest.hexdigest(),
            }
        return {
            "schema": "m1161ca_oaxes_row_sink_terminal_v1",
            "status": "PASS_O_AXES_O1_ROW_COUNTS_DIGESTS_AND_CYCLES",
            "axes": axes,
            "state_complexity": "O(axes)",
            "retained_schedule_event_row_or_key_history": False,
            "cycle_claim": "weight-service schedule coordinates only; not RTL or system cycles",
        }


class ScheduleReplayDriver:
    """Consume task-major/axis-minor schedule records into the live hook."""
    def __init__(self, hook: Any, scope: str, expected_tasks: int):
        require(scope in ("production", "bounded_synthetic") and
                type(expected_tasks) is int and expected_tasks > 0,
                "replay scope/geometry drift")
        self._hook = hook
        self._scope = scope
        self._expected_tasks = expected_tasks
        self._records = 0
        self._axis_records = {axis: 0 for axis in AXES}
        self._last_requested = {axis: None for axis in AXES}

    def consume(self, record: ScheduleRecord) -> int:
        record.validate()
        expected_task = self._records // len(AXES)
        expected_axis = AXES[self._records % len(AXES)]
        require(expected_task < self._expected_tasks and
                record.task_sequence_ordinal == expected_task and
                record.axis == expected_axis,
                "schedule missing, duplicate, or reordered record")
        previous = self._last_requested[record.axis]
        require(previous is None or record.requested_cycle_first >= previous,
                "per-axis requested cycle regressed")
        if self._scope == "production":
            emitted = self._hook.stream_production_task(
                axis=record.axis, sample=record.sample,
                operator=record.operator, chunk=record.chunk,
                partition=record.partition,
                requested_cycle_first=record.requested_cycle_first)
        else:
            emitted = self._hook.stream_bounded_task(
                axis=record.axis, task_id=record.task_sequence_ordinal,
                requested_cycle_first=record.requested_cycle_first)
        require(type(emitted) is int and emitted > 0,
                "live producer emitted no beats")
        self._last_requested[record.axis] = record.requested_cycle_first
        self._records += 1
        self._axis_records[record.axis] += 1
        return emitted

    def finalize(self) -> dict[str, Any]:
        require(self._records == self._expected_tasks * len(AXES) and
                all(self._axis_records[axis] == self._expected_tasks for axis in AXES),
                "terminal schedule record conservation mismatch")
        terminal = self._hook.finalize()
        return {
            "schema": "m1161ca_schedule_to_real_producer_terminal_v1",
            "status": "PASS_SCHEDULE_RECORDS_TO_M1137C_LIVE_PRODUCER",
            "scope": self._scope,
            "tasks_per_axis": self._expected_tasks,
            "records": self._records,
            "records_per_axis": self._axis_records,
            "m1137c_terminal": terminal,
            "state_complexity": "O(axes + axes*24)",
            "retained_schedule_event_row_or_key_history": False,
        }


def _parse_and_replay(stream: BinaryIO, expected_records: int,
                      expected_bytes: int, expected_sha: str,
                      driver: ScheduleReplayDriver) -> dict[str, Any]:
    digest = hashlib.sha256(); count = 0; byte_count = 0; emitted = 0
    for raw in stream:
        require(raw.endswith(b"\n") and not raw.endswith(b"\r\n") and
                1 < len(raw) <= 65_536, "schedule JSONL framing drift")
        digest.update(raw); byte_count += len(raw)
        record = ScheduleRecord.from_mapping(strict_json_bytes(raw[:-1]))
        emitted += driver.consume(record)
        count += 1
    require(count == expected_records and byte_count == expected_bytes and
            digest.hexdigest() == expected_sha,
            "schedule terminal count/byte/SHA drift")
    return {"records": count, "bytes": byte_count,
            "sha256": digest.hexdigest(), "events_emitted": emitted,
            "driver_terminal": driver.finalize()}


def _authority_from_sealed_json(m1137: ModuleType,
                                mapping: Mapping[str, Any]) -> Any:
    require(type(mapping) is dict and set(mapping) == {
                "authority_id_sha256", "axes", "expected_count_by_axis",
                "expected_digest_by_axis", "retained_event_row_or_key_history",
                "schema", "state_complexity", "status"} and
            mapping["schema"] == "m1146ca_independent_expected_digest_authority_v1" and
            mapping["status"] == "PASS_INDEPENDENT_EXPECTED_DIGEST_COMPILATION" and
            mapping["authority_id_sha256"] == AUTHORITY_ID and
            mapping["retained_event_row_or_key_history"] is False,
            "sealed authority schema/status drift")
    counts = mapping["expected_count_by_axis"]
    digests = mapping["expected_digest_by_axis"]
    require(type(counts) is dict and type(digests) is dict and
            set(counts) == set(AXES) and set(digests) == set(AXES) and
            all(type(counts[axis]) is int and counts[axis] == EVENTS_PER_AXIS and
                _hex64(digests[axis]) for axis in AXES),
            "sealed authority axis drift")
    m1135 = m1137.load_m1135()
    return m1135.ExpectedDigestAuthority(
        "production", AUTHORITY_ID,
        {axis: counts[axis] for axis in AXES},
        {axis: digests[axis] for axis in AXES})


def _namespace_paths() -> tuple[Path, ...]:
    fixed = (RESULT, ATTEMPT, LOCK)
    variable = (tuple(RESULTS.glob(WORK_PREFIX + "*")) +
                tuple(RESULTS.glob(FAILURE_PREFIX + "*")))
    return tuple(path for path in fixed + variable if path.exists() or path.is_symlink())


def source_preflight(require_fresh_namespace: bool = True) -> dict[str, Any]:
    verify_double_contract()
    verify_regular(DOCS359, DOCS359_SHA)
    verify_regular(M1135_SOURCE, M1135_SOURCE_SHA)
    verify_regular(M1137_SOURCE, M1137_SOURCE_SHA)
    m1138 = verify_tree(M1138, M1138_ID)
    release = verify_m1141_metadata_only()
    m1145 = verify_tree(M1145, M1145_ID)
    authority_mapping = verify_m1148()
    m1157 = verify_tree(M1157, M1157_ID)
    require(m1138["status"] ==
                "PASS_M1138C_M1137C_REAL_PER_TASK_BEAT_HOOK_HAMMER__AUTHOR_PRODUCTION_EXPECTED_DIGEST_AUTHORITY_CAPTURE_SOURCE_ONLY" and
            m1145["status"] ==
                "PASS_M1145CA_INDEPENDENT_PRODUCTION_RESULT_HAMMER__DIGEST_COMPILER_SOURCE_AUTHORING_ONLY_NEXT" and
            m1157["status"] ==
                "PASS_M1157CA_DIFFERENT_AUTHOR_RESULT_HAMMER__EXPECTED_DIGEST_AUTHORITY_ONLY" and
            m1157["authorization"]["expected_digest_authority_may_be_consumed_by_successor"] is True and
            m1157["authorization"]["real_producer_replay_still_required_for_producer_claim"] is True,
            "sealed hammer authorization drift")
    require(release["geometry"] == {
                "axes": list(AXES), "chunks": CHUNKS, "operators": OPERATORS,
                "partitions": PARTITIONS, "records": EXPECTED_RECORDS,
                "samples": SAMPLES, "tasks": TASKS} and
            release["records"]["sha256"] == M1141_RECORDS_SHA and
            release["claim_boundary"]["real_driver"] is False,
            "sealed schedule release geometry/boundary drift")
    m1137 = load_m1137()
    hook_preflight = m1137.source_preflight()
    require(hook_preflight["real_production_driver_integrated"] is False and
            hook_preflight["canonical_events"] == 0 and
            m1137.PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256 is None,
            "M1137 frozen stop boundary drift")
    _authority_from_sealed_json(m1137, authority_mapping).validate()
    if require_fresh_namespace:
        require(_namespace_paths() == (), "production namespace is not fresh")
    return {
        "status": "PASS_M1161CA_SOURCE_PREFLIGHT__NO_PRODUCTION_SCHEDULE_OPEN",
        "m1137_source_sha256": M1137_SOURCE_SHA,
        "m1138_outer_seal_file_sha256": M1138_ID[2],
        "m1141_records_sha256_expected": M1141_RECORDS_SHA,
        "m1145_outer_seal_file_sha256": M1145_ID[2],
        "authority_id_sha256": AUTHORITY_ID,
        "m1148_outer_seal_file_sha256": M1148_OUTER_SHA,
        "m1157_outer_seal_file_sha256": M1157_ID[2],
        "production_schedule_opened": False,
        "production_events_replayed": 0,
        "production_execution_authorized_by_source_milestone": False,
    }


def _fixture_payload() -> bytes:
    rows = []
    for task in range(2):
        source_provenance = hashlib.sha256(
            f"m1161ca-bounded-task:{task}".encode()).hexdigest()
        for axis in AXES:
            requested = 5 + task
            mapping = {
                "axis": axis, "chunk": 0, "operator": 0,
                "partition": task, "requested_cycle_first": requested,
                "sample": 0,
                "schedule_record_provenance_sha256": record_provenance(
                    axis, task, 0, 0, 0, task, requested, source_provenance),
                "source_task_provenance_sha256": source_provenance,
                "task_sequence_ordinal": task,
            }
            rows.append((json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n").encode("utf-8"))
    return b"".join(rows)


def source_bounded_self_test() -> dict[str, Any]:
    before = _namespace_paths()
    preflight = source_preflight(True)
    m1137 = load_m1137()
    require(m1137.PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256 is None,
            "bounded test cannot inject production authority")
    sink = OAxesRowReceiptSink()
    hook = m1137.M1016SuccessorPerTaskWeightBeatHook(
        m1137.bounded_authority(), sink)
    driver = ScheduleReplayDriver(hook, "bounded_synthetic", 2)
    payload = _fixture_payload()
    with tempfile.TemporaryDirectory(prefix="m1161ca_bounded_") as temporary:
        path = Path(temporary) / "bounded_schedule.jsonl"
        path.write_bytes(payload)
        with path.open("rb") as stream:
            replay = _parse_and_replay(
                stream, 6, len(payload), hashlib.sha256(payload).hexdigest(), driver)
    row_terminal = sink.finalize(4)
    hook_terminal = replay["driver_terminal"]["m1137c_terminal"]
    require(replay["events_emitted"] == 12 and
            hook_terminal["events_per_axis"] == {axis: 4 for axis in AXES} and
            all(row_terminal["axes"][axis]["rows"] == 4 and
                row_terminal["axes"][axis]["stall_cycles"] == 2
                for axis in AXES) and before == () and _namespace_paths() == (),
            "bounded replay conservation/namespace drift")

    attacks = 0
    for payload_attack in (
        b'{"axis":"candidate","axis":"candidate"}',
        b'{"axis":NaN}',
    ):
        try:
            strict_json_bytes(payload_attack)
        except (Failure, json.JSONDecodeError):
            attacks += 1
    mapping = strict_json_bytes(_fixture_payload().splitlines()[0])
    for mutation in ("extra", "provenance", "task"):
        attacked = dict(mapping)
        if mutation == "extra":
            attacked["unexpected"] = 1
        elif mutation == "provenance":
            attacked["schedule_record_provenance_sha256"] = "0" * 64
        else:
            attacked["task_sequence_ordinal"] = 1
        try:
            ScheduleRecord.from_mapping(attacked)
        except Failure:
            attacks += 1
    require(attacks == 5, "bounded parser attacks escaped")
    return {
        "schema": "m1161ca_c1_real_replay_driver_bounded_oracle_v1",
        "status": "PASS_BOUNDED_TWO_TASK_LIVE_M1137_REPLAY__PRODUCTION_STOP",
        "preflight": preflight,
        "records": replay["records"],
        "events": replay["events_emitted"],
        "driver_terminal": replay["driver_terminal"],
        "row_terminal": row_terminal,
        "attacks_rejected": attacks,
        "production_schedule_opened": False,
        "production_events_replayed": 0,
        "production_namespace_mutated": False,
        "full_replay": False,
        "eda_rtl_gpu_remote": False,
        "cycle_claim": "bounded weight-service schedule coordinates only",
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
            if match is not None and int(match.group(1)) == EXPECTED_UID and text.intersection(tokens):
                conflicts.append(int(entry.name))
        except (FileNotFoundError, PermissionError):
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
             for member in sorted(members,
                                  key=lambda item: item.relative_to(directory).as_posix())]
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
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100,
                            os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def production_main() -> dict[str, Any]:
    """Consume the sole attempt and run the full replay; zero arguments only."""
    preflight = source_preflight(True)
    resources_before = resource_preflight()
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    attempt_consumed = False
    work = RESULTS / (WORK_PREFIX + f"{os.getpid()}.{time.time_ns()}")
    failure = RESULTS / (FAILURE_PREFIX + f"{os.getpid()}.{time.time_ns()}.quarantine")
    phase = "LOCKED_PREFLIGHT"
    started_wall = time.monotonic(); started_cpu = time.process_time()
    m1137 = load_m1137()
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(tuple(path for path in _namespace_paths() if path != LOCK) == (),
                "namespace changed under lock")
        resources_locked = resource_preflight()
        phase = "CONSUME_SINGLE_ATTEMPT_BEFORE_SCHEDULE_OPEN"
        ATTEMPT.mkdir(mode=0o700)
        _write_json(ATTEMPT / "attempt.json", {
            "schema": "m1161ca_c1_production_real_replay_attempt_r1_v1",
            "status": "M1161CA_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY",
            "source_sha256": sha256(SOURCE_FILE),
            "m1137_source_sha256": M1137_SOURCE_SHA,
            "m1141_records_sha256_expected": M1141_RECORDS_SHA,
            "authority_id_sha256": AUTHORITY_ID,
            "schedule_opened_before_attempt": False,
            "automatic_retry": False,
        })
        _seal_tree(ATTEMPT); _fsync_dir(RESULTS); attempt_consumed = True
        work.mkdir(mode=0o700)
        phase = "STREAM_SCHEDULE_TO_LIVE_PRODUCER_AND_VALIDATOR"
        authority_mapping = strict_json(M1148_AUTHORITY)
        authority = _authority_from_sealed_json(m1137, authority_mapping)
        require(m1137.PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256 is None,
                "M1137 production authority was already injected")
        m1137.PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256 = AUTHORITY_ID
        sink = OAxesRowReceiptSink()
        hook = m1137.M1016SuccessorPerTaskWeightBeatHook(authority, sink)
        driver = ScheduleReplayDriver(hook, "production", TASKS)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(M1141_RECORDS, flags)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and
                    before.st_size == M1141_RECORDS_BYTES,
                    "production schedule fd metadata drift")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                replay = _parse_and_replay(
                    stream, EXPECTED_RECORDS, M1141_RECORDS_BYTES,
                    M1141_RECORDS_SHA, driver)
            after = os.fstat(fd)
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                    "production schedule changed during stream")
        finally:
            os.close(fd)
        row_terminal = sink.finalize(EVENTS_PER_AXIS)
        require(replay["events_emitted"] == EXPECTED_EVENTS and
                all(row_terminal["axes"][axis]["stall_cycles"] == 0
                    for axis in AXES),
                "production event/stall terminal drift")
        elapsed_wall = time.monotonic() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        usage = resource.getrusage(resource.RUSAGE_SELF)
        phase = "WRITE_SMALL_TERMINAL_RECEIPTS"
        _write_json(work / "producer_replay_terminal.json", {
            "schema": "m1161ca_c1_production_real_replay_terminal_r1_v1",
            "status": "PASS_REAL_M1137_PRODUCER_TO_M1135_VALIDATOR_REPLAY__RESULT_HAMMER_REQUIRED",
            "sealed_schedule": {"records": replay["records"],
                                "bytes": replay["bytes"],
                                "sha256": replay["sha256"]},
            "events_emitted": replay["events_emitted"],
            "driver_terminal": replay["driver_terminal"],
            "row_terminal": row_terminal,
            "retained_schedule_event_row_or_key_history": False,
            "per_event_output_written": False,
            "claim_boundary": {
                "real_producer_replay": True,
                "weight_service_schedule_cycles": True,
                "rtl_cycle_or_system_speedup": False,
                "traffic_energy_or_paper_ppa": False,
                "different_author_result_hammer_required": True,
            },
        })
        _write_json(work / "runtime_resources.json", {
            "schema": "m1161ca_c1_production_real_replay_resources_r1_v1",
            "wall_seconds": elapsed_wall, "cpu_seconds": elapsed_cpu,
            "max_rss_kib": usage.ru_maxrss,
            "input_bytes_streamed": replay["bytes"],
            "input_records_streamed": replay["records"],
            "events_replayed": replay["events_emitted"],
            "state_complexity": "O(axes + axes*24) plus one JSON line and one row",
            "retained_schedule_event_row_or_key_history": False,
        })
        _write_json(work / "receipt.json", {
            "schema": "m1161ca_c1_production_real_replay_receipt_r1_v1",
            "status": "PASS_M1161CA_REAL_PRODUCER_REPLAY__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
            "source_sha256": sha256(SOURCE_FILE),
            "source_preflight": preflight,
            "resources_before": resources_before,
            "resources_under_lock": resources_locked,
            "attempt_consumed": True, "automatic_retry": False,
            "event_output_written": False,
            "claim_boundary": {
                "producer_replay_and_1rw_schedule_receipt": True,
                "rtl_cycle_or_system_speedup": False,
                "paper_citable_performance": False,
            },
        })
        _write_exclusive(work / "RUN_COMPLETE.txt",
                         b"PASS_M1161CA_REAL_PRODUCER_REPLAY__RESULT_HAMMER_REQUIRED\n")
        manifest_sha, outer_sha = _seal_tree(work)
        _rename_noreplace(work, RESULT); _fsync_dir(RESULTS)
        require(not tuple(RESULTS.glob(FAILURE_PREFIX + "*")),
                "result/failure mutual exclusion drift")
        return {
            "status": "PASS_M1161CA_REAL_PRODUCER_REPLAY__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
            "result": str(RESULT), "events": replay["events_emitted"],
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha,
            "automatic_retry": False,
            "rtl_cycle_or_system_speedup": False,
        }
    except BaseException:
        reason = traceback.format_exc()
        if attempt_consumed:
            try:
                work.mkdir(mode=0o700, exist_ok=True)
                _write_json(work / "failure.json", {
                    "schema": "m1161ca_c1_production_real_replay_failure_r1_v1",
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
        m1137.PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256 = None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    require(len(sys.argv) == 1, "M1161CA accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
