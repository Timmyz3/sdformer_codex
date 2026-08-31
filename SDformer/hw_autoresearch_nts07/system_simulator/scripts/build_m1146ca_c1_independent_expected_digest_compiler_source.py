#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1135C-compatible expected-digest compiler source.

This source binds frozen identities but imports and calls none of the M1137C,
M1135C, M1130C, or M1132C subject runtime.  Its runnable path is a bounded
three-task oracle.  Production is stopped before the sealed schedule JSONL is
opened until a successor authorization is pinned.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Callable, Iterator, Mapping

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

M1016_SOURCE = HERE / "run_m1016_c1_full_matched_address_replay.py"
M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1130_SOURCE = HERE / "build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
M1130_SOURCE_SHA = "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf"
M1132_SOURCE = HERE / "build_m1132c_c1_upstream_weight_event_producer_source.py"
M1132_SOURCE_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
M1135_SOURCE = HERE / "build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
M1135_SOURCE_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
M1137_SOURCE = HERE / "build_m1137c_c1_real_per_task_weight_beat_hook_source.py"
M1137_SOURCE_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
M1139_SOURCE = HERE / "build_m1139ca_c1_independent_per_task_schedule_authority_source.py"
M1139_SOURCE_SHA = "d18137661517538a8273b696b5f2ada09aff9847c16da0d3a723037e901153a9"

M1141 = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
M1141_RELEASE = M1141 / "m1141ca_schedule_release.json"
M1141_RECORDS = M1141 / "m1141ca_per_task_schedule_records.jsonl"
M1141_ID = (
    "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c",
    "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace",
    "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a",
)
M1141_RECORDS_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
M1141_SCHEDULE_PROVENANCE_SHA = "d8289ede1ec668cd86b9ea2c561c76f62738cbd5aa361d9c21642f900e3fa1b9"
M1145 = HW / "reviews/m1145ca_m1143ca_c1_production_result_hammer_r1_20260830"
M1145_ID = (
    "cfe7bf030743c4bcc098d267c69422ae1e76238696902e8dc601ea8143ee208d",
    "7dbc93256d915962a4f83e860aff9aac0bb3b62b1c76113509daef74b852eb4c",
    "8dcc8e84ec8c6273f155c418078fac92b96ef851768ec6cb2066c64ab3d3423e",
)
CONTRACT = HW / "contracts/m1146ca_c1_independent_expected_digest_compiler_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "5f36b42c088e0143ab61b90098d55610c7e4bb555f318b416968759c45a33a2f",
    "854d2d1c0b1162e8618198d44c2fe1f7bb272672735474e6a7489e405f4bc02c",
    "60fae54229f5c6f802127ea10906ed7bec3c42e8d2b6bb50b6b323e8c4e42b13",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
AXIS_CODE = {axis: index for index, axis in enumerate(AXES)}
EVENT_FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)
SCHEDULE_FIELDS = (
    "axis", "task_sequence_ordinal", "sample", "operator", "chunk",
    "partition", "requested_cycle_first", "source_task_provenance_sha256",
    "schedule_record_provenance_sha256",
)
PRODUCTION_TASKS = 812_160
PRODUCTION_EVENTS_PER_AXIS = 70_853_184
PRODUCTION_EVENTS_TOTAL = PRODUCTION_EVENTS_PER_AXIS * len(AXES)
PRODUCTION_COMPILER_EXECUTION_AUTHORIZATION_SHA256: str | None = None
BOUNDED_GOLDEN_DIGESTS = {
    "candidate": "ab87d9d8da38d28a54d6048dc75cb7ac749aebba7807f855cac69165b9fa5644",
    "strongest_zero": "eb2dd17d2d0aa43e19d2f66b9d079760f7495c1f9b4653d206831605e1b44717",
    "same_coordinate_bit": "18a4e643ee4a606b5ec8e646fbd76aa155ffe324213a4f8bb36925c6fb678d7a",
}


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


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


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


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.name], "contract outer content")


def verify_small_flat(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    review = directory / "review.json"; manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(review, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "authority outer content")
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
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "authority symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "authority special member")
    require(actual == set(listed), "authority exact member set")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return strict_json(review)


def verify_schedule_release_without_records_open() -> dict[str, Any]:
    """Verify the small seal metadata; deliberately do not open/hash JSONL."""
    manifest = M1141 / "SHA256SUMS"; outer = M1141 / "SHA256SUMS.seal.sha256"
    verify_regular(M1141_RELEASE, M1141_ID[0]); verify_regular(manifest, M1141_ID[1])
    verify_regular(outer, M1141_ID[2])
    require(outer.read_text(encoding="utf-8").split() == [M1141_ID[1], "SHA256SUMS"],
            "M1141 outer content")
    expected_rows = {
        M1141_RELEASE.name: M1141_ID[0],
        M1141_RECORDS.name: M1141_RECORDS_SHA,
    }
    observed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in observed and re.fullmatch(r"[0-9a-f]{64}", digest),
                "M1141 manifest row")
        observed[name] = digest
    require(observed == expected_rows and M1141_RECORDS.exists() and
            stat.S_ISREG(M1141_RECORDS.lstat().st_mode) and not M1141_RECORDS.is_symlink(),
            "M1141 member identity set drift")
    release = strict_json(M1141_RELEASE)
    require(release["status"] ==
            "PASS_EXACT_PRODUCTION_SCHEDULE_RELEASE__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED" and
            release["geometry"]["tasks"] == PRODUCTION_TASKS and
            release["geometry"]["records"] == PRODUCTION_TASKS * len(AXES) and
            release["records"]["sha256"] == M1141_RECORDS_SHA and
            release["records"]["schedule_provenance_sha256"] ==
                M1141_SCHEDULE_PROVENANCE_SHA and
            tuple(release["records"]["axis_order_within_each_task"]) == AXES,
            "M1141 release schema/geometry drift")
    return release


def source_preflight() -> dict[str, Any]:
    for path, digest in (
        (M1016_SOURCE, M1016_SOURCE_SHA), (M1102_SOURCE, M1102_SOURCE_SHA),
        (M1130_SOURCE, M1130_SOURCE_SHA), (M1132_SOURCE, M1132_SOURCE_SHA),
        (M1135_SOURCE, M1135_SOURCE_SHA), (M1137_SOURCE, M1137_SOURCE_SHA),
        (M1139_SOURCE, M1139_SOURCE_SHA), (DOCS359, DOCS359_SHA),
    ):
        verify_regular(path, digest)
    verify_double(CONTRACT, CONTRACT_ID)
    release = verify_schedule_release_without_records_open()
    review = verify_small_flat(M1145, M1145_ID)
    require(review["status"] ==
            "PASS_M1145CA_INDEPENDENT_PRODUCTION_RESULT_HAMMER__DIGEST_COMPILER_SOURCE_AUTHORING_ONLY_NEXT" and
            review["authorization"]["digest_compiler_source_authoring_only_next"] is True and
            review["authorization"]["digest_compiler_execution"] is False and
            review["stream"]["records_sha256"] == M1141_RECORDS_SHA,
            "M1145CA authorization drift")
    return {
        "status": "STOP_DIGEST_COMPILER_EXECUTION_NOT_AUTHORIZED__SOURCE_BOUNDED_ONLY",
        "m1145ca_source_authoring_authorized": True,
        "sealed_schedule_release_verified_without_records_open": True,
        "production_schedule_records_opened": False,
        "production_events_compiled": 0,
        "production_target_events": PRODUCTION_EVENTS_TOTAL,
        "release_status": release["status"],
    }


def _hex64(value: Any) -> bool:
    return (type(value) is str and
            re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _u64(value: int, label: str) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), label + " outside u64")
    return struct.pack(">Q", value)


@dataclass(frozen=True)
class Geometry:
    samples: int
    operators: int
    chunks: int
    partitions: int
    tasks: int
    events_per_axis: int

    def validate(self) -> None:
        require(all(type(value) is int and value > 0 for value in (
                    self.samples, self.operators, self.chunks, self.partitions,
                    self.tasks, self.events_per_axis)) and
                self.tasks == self.samples * self.operators * self.chunks * self.partitions and
                self.events_per_axis >= self.tasks,
                "geometry drift")


PRODUCTION_GEOMETRY = Geometry(10, 4, 47, 432, PRODUCTION_TASKS,
                               PRODUCTION_EVENTS_PER_AXIS)
BOUNDED_GEOMETRY = Geometry(1, 1, 1, 3, 3, 8)


def task_coordinates(geometry: Geometry, task_id: int) -> tuple[int, int, int, int]:
    geometry.validate(); require(0 <= task_id < geometry.tasks, "task id range")
    partition = task_id % geometry.partitions; quotient = task_id // geometry.partitions
    chunk = quotient % geometry.chunks; quotient //= geometry.chunks
    operator = quotient % geometry.operators; sample = quotient // geometry.operators
    return sample, operator, chunk, partition


def schedule_record_provenance(axis: str, task_id: int, sample: int,
                               operator: int, chunk: int, partition: int,
                               requested_cycle_first: int,
                               source_task_provenance_sha256: str) -> str:
    require(axis in AXES and _hex64(source_task_provenance_sha256),
            "schedule provenance input")
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(M1016_SOURCE_SHA),
        bytes.fromhex(M1102_SOURCE_SHA), bytes.fromhex(M1137_SOURCE_SHA),
        struct.pack(">B", AXIS_CODE[axis]), _u64(task_id, "task"),
        _u64(sample, "sample"), _u64(operator, "operator"),
        _u64(chunk, "chunk"), _u64(partition, "partition"),
        _u64(requested_cycle_first, "requested cycle first"),
        bytes.fromhex(source_task_provenance_sha256),
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

    def validate(self, geometry: Geometry) -> None:
        require(self.axis in AXES and all(type(value) is int and value >= 0 for value in (
                    self.task_sequence_ordinal, self.sample, self.operator,
                    self.chunk, self.partition, self.requested_cycle_first)) and
                (self.sample, self.operator, self.chunk, self.partition) ==
                    task_coordinates(geometry, self.task_sequence_ordinal) and
                _hex64(self.source_task_provenance_sha256) and
                self.schedule_record_provenance_sha256 == schedule_record_provenance(
                    self.axis, self.task_sequence_ordinal, self.sample, self.operator,
                    self.chunk, self.partition, self.requested_cycle_first,
                    self.source_task_provenance_sha256),
                "schedule record coordinate/provenance drift")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], geometry: Geometry) -> "ScheduleRecord":
        require(type(value) is dict and tuple(sorted(value)) == tuple(sorted(SCHEDULE_FIELDS)),
                "schedule record exact key set drift")
        record = cls(*(value[field] for field in SCHEDULE_FIELDS))
        record.validate(geometry)
        return record


def exact_once_id(axis: str, task_id: int, local_ordinal: int,
                  beat_ordinal: int, transaction_ordinal: int) -> str:
    require(axis in AXES, "exact ID axis")
    payload = (f"m1130c:{axis}:{task_id}:{local_ordinal}:"
               f"{beat_ordinal}:{transaction_ordinal}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_row_provenance(axis: str, sample: int, operator: int, chunk: int,
                          partition: int, task_id: int, local_ordinal: int,
                          global_beat_ordinal: int, requested_cycle: int,
                          half_slot: int, logical_row: int,
                          native_slices: tuple[int, ...]) -> str:
    require(axis in AXES and half_slot in (0, 1) and 0 <= logical_row < 16 and
            len(native_slices) == 8, "source provenance mapping")
    payload = b"".join((
        b"M1137C_REAL_BEAT\x00\x01", bytes.fromhex(M1016_SOURCE_SHA),
        bytes.fromhex(M1102_SOURCE_SHA), bytes.fromhex(M1135_SOURCE_SHA),
        struct.pack(">B", AXIS_CODE[axis]), _u64(sample, "sample"),
        _u64(operator, "operator"), _u64(chunk, "chunk"),
        _u64(partition, "partition"), _u64(task_id, "task"),
        _u64(local_ordinal, "local"), _u64(global_beat_ordinal, "global beat"),
        _u64(requested_cycle, "requested cycle"), struct.pack(">B", half_slot),
        struct.pack(">B", logical_row), struct.pack(">B", len(native_slices)),
        bytes(native_slices),
    ))
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class IndependentWeightEvent:
    axis: str
    task_id: int
    source_local_ordinal: int
    requested_cycle: int
    op: str
    logical_bank: int
    half_slot: int
    logical_row: int
    local_row: int
    native_slices: tuple[int, ...]
    bytes: int
    byte_enable_per_slice: tuple[int, ...]
    native_macro_activations: int
    service_beat_ordinal: int
    store_transaction_ordinal: int
    service_event_exact_once_id: str
    source_row_provenance_sha256: str

    def validate(self) -> None:
        require(self.axis in AXES and self.op == "WRITE" and
                all(type(value) is int and value >= 0 for value in (
                    self.task_id, self.source_local_ordinal, self.requested_cycle,
                    self.service_beat_ordinal, self.store_transaction_ordinal)) and
                self.half_slot in (0, 1) and self.logical_bank == self.half_slot and
                0 <= self.logical_row < 16 and
                self.local_row == self.half_slot * 16 + self.logical_row and
                self.native_slices == tuple(range(min(self.native_slices),
                                                  min(self.native_slices) + 8)) and
                min(self.native_slices) % 8 == 0 and max(self.native_slices) < 24 and
                self.bytes == 128 and self.byte_enable_per_slice == (0xffff,) * 8 and
                self.native_macro_activations == 8 and
                self.service_event_exact_once_id == exact_once_id(
                    self.axis, self.task_id, self.source_local_ordinal,
                    self.service_beat_ordinal, self.store_transaction_ordinal) and
                _hex64(self.source_row_provenance_sha256),
                "independent 17-field event mapping drift")


def reconstruct_event(record: ScheduleRecord, global_beat: int,
                      interval_begin: int) -> IndependentWeightEvent:
    local = global_beat - interval_begin
    requested = record.requested_cycle_first + local
    half_slot = record.task_sequence_ordinal & 1
    logical_row = global_beat % 16
    slice_base = ((global_beat // 16) % 3) * 8
    native_slices = tuple(range(slice_base, slice_base + 8))
    event = IndependentWeightEvent(
        record.axis, record.task_sequence_ordinal, local, requested, "WRITE",
        half_slot, half_slot, logical_row, half_slot * 16 + logical_row,
        native_slices, 128, (0xffff,) * 8, 8, global_beat, global_beat,
        exact_once_id(record.axis, record.task_sequence_ordinal, local,
                      global_beat, global_beat),
        source_row_provenance(
            record.axis, record.sample, record.operator, record.chunk,
            record.partition, record.task_sequence_ordinal, local, global_beat,
            requested, half_slot, logical_row, native_slices))
    event.validate()
    return event


def canonical_event_bytes(event: IndependentWeightEvent, sequence_ordinal: int,
                          scheduled_cycle: int, stall_cycles: int) -> bytes:
    """Independent exact rewrite of the frozen M1135C byte serialization."""
    event.validate()
    pieces = [
        b"M1135C\x00\x01", struct.pack(">B", AXIS_CODE[event.axis]),
        _u64(event.task_id, "task_id"),
        _u64(event.source_local_ordinal, "source_local_ordinal"),
        _u64(event.requested_cycle, "requested_cycle"), b"W",
        struct.pack(">B", event.logical_bank), struct.pack(">B", event.half_slot),
        struct.pack(">B", event.logical_row), struct.pack(">B", event.local_row),
        struct.pack(">B", len(event.native_slices)), bytes(event.native_slices),
        _u64(event.bytes, "bytes"),
        struct.pack(">B", len(event.byte_enable_per_slice)),
        b"".join(struct.pack(">H", item) for item in event.byte_enable_per_slice),
        _u64(event.native_macro_activations, "native_macro_activations"),
        _u64(event.service_beat_ordinal, "service_beat_ordinal"),
        _u64(event.store_transaction_ordinal, "store_transaction_ordinal"),
        bytes.fromhex(event.service_event_exact_once_id),
        bytes.fromhex(event.source_row_provenance_sha256),
        _u64(sequence_ordinal, "sequence_ordinal"),
        _u64(scheduled_cycle, "scheduled_cycle"),
        _u64(stall_cycles, "stall_cycles"),
    ]
    return b"".join(pieces)


@dataclass
class _AxisState:
    events: int
    records: int
    bytes: int
    native_activations: int
    stalled_transactions: int
    stall_cycles: int
    last_requested_cycle_first: int | None
    last_scheduler_key: tuple[int, int, int, int] | None
    digest: Any
    next_free_cycle: list[int]


class IndependentExpectedDigestCompiler:
    """Streaming compiler with three SHA states and exactly 3x24 cycle slots."""
    def __init__(self, geometry: Geometry):
        geometry.validate()
        self._geometry = geometry
        self._axis = {
            axis: _AxisState(0, 0, 0, 0, 0, 0, None, None,
                             hashlib.sha256(), [0] * 24)
            for axis in AXES
        }
        self._next_record = 0
        self._finalized = False

    def consume_schedule_record(self, record: ScheduleRecord) -> int:
        require(not self._finalized and type(record) is ScheduleRecord,
                "exact schedule record required before finalize")
        record.validate(self._geometry)
        expected_task = self._next_record // len(AXES)
        expected_axis = AXES[self._next_record % len(AXES)]
        require(record.task_sequence_ordinal == expected_task and
                record.axis == expected_axis,
                "schedule missing, duplicate, reorder, task, or axis drift")
        state = self._axis[record.axis]
        require(state.last_requested_cycle_first is None or
                record.requested_cycle_first >= state.last_requested_cycle_first,
                "schedule requested-cycle regression")
        begin = (record.task_sequence_ordinal * self._geometry.events_per_axis) // self._geometry.tasks
        end = ((record.task_sequence_ordinal + 1) * self._geometry.events_per_axis) // self._geometry.tasks
        require(state.events == begin and begin < end,
                "event interval/count conservation drift")

        candidate_digest = state.digest.copy()
        candidate_next_free = list(state.next_free_cycle)
        candidate_stalled = state.stalled_transactions
        candidate_stall_cycles = state.stall_cycles
        candidate_last_key = state.last_scheduler_key
        for global_beat in range(begin, end):
            event = reconstruct_event(record, global_beat, begin)
            scheduler_key = (event.requested_cycle, event.task_id,
                             event.source_local_ordinal,
                             event.store_transaction_ordinal)
            require(candidate_last_key is None or scheduler_key >= candidate_last_key,
                    "event scheduler key regression")
            scheduled = max([event.requested_cycle] +
                            [candidate_next_free[index] for index in event.native_slices])
            stall = scheduled - event.requested_cycle
            candidate_digest.update(canonical_event_bytes(
                event, global_beat, scheduled, stall))
            for index in event.native_slices:
                candidate_next_free[index] = scheduled + 1
            candidate_stalled += int(stall > 0)
            candidate_stall_cycles += stall
            candidate_last_key = scheduler_key

        # One schedule record is the transaction boundary.  Nothing above is
        # committed until its complete variable-length interval succeeds.
        beats = end - begin
        state.digest = candidate_digest
        state.next_free_cycle = candidate_next_free
        state.stalled_transactions = candidate_stalled
        state.stall_cycles = candidate_stall_cycles
        state.last_scheduler_key = candidate_last_key
        state.last_requested_cycle_first = record.requested_cycle_first
        state.events = end; state.records += 1
        state.bytes += beats * 128; state.native_activations += beats * 8
        self._next_record += 1
        return beats

    def snapshot(self) -> dict[str, Any]:
        return {
            "next_record": self._next_record, "finalized": self._finalized,
            "axes": {axis: {
                "events": state.events, "records": state.records,
                "bytes": state.bytes, "native_activations": state.native_activations,
                "stalled_transactions": state.stalled_transactions,
                "stall_cycles": state.stall_cycles,
                "last_requested_cycle_first": state.last_requested_cycle_first,
                "last_scheduler_key": state.last_scheduler_key,
                "digest": state.digest.hexdigest(),
                "next_free_cycle": tuple(state.next_free_cycle),
            } for axis, state in self._axis.items()},
            "state_complexity": "O(axes + axes*24)",
        }

    def _candidate_authority(self) -> dict[str, Any]:
        require(self._next_record == self._geometry.tasks * len(AXES) and
                all(state.records == self._geometry.tasks and
                    state.events == self._geometry.events_per_axis
                    for state in self._axis.values()),
                "partial output or terminal count conservation mismatch")
        counts = {axis: self._axis[axis].events for axis in AXES}
        digests = {axis: self._axis[axis].digest.hexdigest() for axis in AXES}
        identity_payload = json.dumps({
            "schema": "m1146ca_expected_digest_authority_identity_v1",
            "counts": counts, "digests": digests,
            "m1141_records_sha256": M1141_RECORDS_SHA,
            "m1135_semantics_sha256": M1135_SOURCE_SHA,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return {
            "schema": "m1146ca_independent_expected_digest_authority_v1",
            "status": "PASS_INDEPENDENT_EXPECTED_DIGEST_COMPILATION",
            "authority_id_sha256": hashlib.sha256(identity_payload).hexdigest(),
            "expected_count_by_axis": counts,
            "expected_digest_by_axis": digests,
            "axes": {axis: {
                "records": self._axis[axis].records,
                "events": self._axis[axis].events,
                "bytes": self._axis[axis].bytes,
                "native_activations": self._axis[axis].native_activations,
                "stalled_transactions": self._axis[axis].stalled_transactions,
                "stall_cycles": self._axis[axis].stall_cycles,
            } for axis in AXES},
            "state_complexity": "O(axes + axes*24)",
            "retained_event_row_or_key_history": False,
        }

    def finalize(self, authority_sink: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
        require(not self._finalized and callable(authority_sink),
                "one callable terminal authority sink required")
        authority = self._candidate_authority()
        authority_sink(authority)  # A failure leaves compiler terminal state uncommitted.
        self._finalized = True
        return authority


def bounded_schedule_records() -> Iterator[ScheduleRecord]:
    requested = {
        "candidate": (5, 6, 8),
        "strongest_zero": (7, 8, 10),
        "same_coordinate_bit": (11, 12, 14),
    }
    for task_id in range(BOUNDED_GEOMETRY.tasks):
        sample, operator, chunk, partition = task_coordinates(BOUNDED_GEOMETRY, task_id)
        source_task = hashlib.sha256(
            f"m1146ca-bounded-task:{task_id}".encode("utf-8")).hexdigest()
        for axis in AXES:
            first = requested[axis][task_id]
            yield ScheduleRecord(
                axis, task_id, sample, operator, chunk, partition, first,
                source_task, schedule_record_provenance(
                    axis, task_id, sample, operator, chunk, partition, first,
                    source_task))


def compile_production_expected_digest_authority() -> dict[str, Any]:
    """Future production entry; the current source stops before JSONL open."""
    require(PRODUCTION_COMPILER_EXECUTION_AUTHORIZATION_SHA256 is not None,
            "STOP: production digest compiler execution is not authorized")
    # A successor must pin an execution authorization, reverify M1141/M1145,
    # then stream strict JSONL records through IndependentExpectedDigestCompiler.
    raise Failure("STOP: production record opener belongs to an authorized successor")


def source_small_oracle() -> dict[str, Any]:
    preflight = source_preflight()
    compiler = IndependentExpectedDigestCompiler(BOUNDED_GEOMETRY)
    beat_counts = []
    for record in bounded_schedule_records():
        beat_counts.append(compiler.consume_schedule_record(record))
    accepted: list[Mapping[str, Any]] = []
    authority = compiler.finalize(accepted.append)
    require(len(accepted) == 1 and
            authority["expected_count_by_axis"] == {axis: 8 for axis in AXES} and
            authority["expected_digest_by_axis"] == BOUNDED_GOLDEN_DIGESTS and
            beat_counts == [2, 2, 2, 3, 3, 3, 3, 3, 3],
            "bounded independent golden/count drift")
    stopped = False
    try:
        compile_production_expected_digest_authority()
    except Failure:
        stopped = True
    require(stopped, "production compiler escaped authorization gate")
    return {
        "schema": "m1146ca_independent_expected_digest_compiler_small_oracle_v1",
        "status": "PASS_BOUNDED_3_TASK_VARIABLE_BEAT_3_AXIS_DIGEST__PRODUCTION_STOP",
        "preflight": preflight,
        "bounded": {
            "tasks": 3, "axes": 3, "schedule_records": 9,
            "events": 24, "beats_per_task": [2, 3, 3],
            "authority": authority,
        },
        "production_schedule_records_opened": False,
        "production_events_compiled": 0,
        "production_target_events": PRODUCTION_EVENTS_TOTAL,
        "full_replay_eda": False,
    }


def main() -> None:
    require(sys.argv[1:] in ([], ["--self-test"]), "bounded self-test only")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
