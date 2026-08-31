#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent per-task schedule authority source; bounded authoring only.

It never imports M1137C and never reads producer events or results.  Production
row access is gated off before any open/hash until a future sealed release.
"""
from __future__ import annotations

from dataclasses import dataclass
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
from typing import Any, Callable, Iterator, Mapping

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1016_SOURCE = HERE / "run_m1016_c1_full_matched_address_replay.py"
M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1016_CONTRACT = HW / "contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
M1016_CONTRACT_SHA = "b980f51017778b1958845547601de5d343ba5a1f3db1b046963afa7549644c90"
M1102_SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1102_CONTRACT = HW / "contracts/m1102_c1_legal_work8_exact_1rw_additive_source_contract_r1_20260830.json"
M1102_CONTRACT_ID = (
    "fad9c381fc1e55fc78d6cf4b95ad0959b5a7089989a7acce1ccfafa73714db6e",
    "e6754574c804a7ed2cfd39e5a99c991db38402389901fef570359decf43e3607",
    "b17774b1b3fad06f104081b2ab2b0de4b3b539c72fd9e6adcb2171a46d55770c",
)
M1007_SOURCE = HERE / "m1007_c1_matched_common_charge_address_replay_source.py"
M1007_SOURCE_SHA = "150f22eaa11d219bfa20561b91a38049f14abbc541a6b40db04bd73533ec3442"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
ROWS_BYTES = 466_560_000
M1137_SOURCE = HERE / "build_m1137c_c1_real_per_task_weight_beat_hook_source.py"
M1137_SOURCE_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
M1137_CONTRACT = HW / "contracts/m1137c_c1_real_per_task_weight_beat_hook_source_contract_r1_20260830.json"
M1137_CONTRACT_OUTER_SHA = "865dac0d7bf89f1a57777f5eafbc6b6fef8b8cbc78403c1822ba5191adfc349d"
M1138 = HW / "reviews/m1138c_m1137c_c1_real_per_task_weight_beat_hook_hammer_r1_20260830"
M1138_ID = (
    "83356f85ce1d7a3be950d50fc226dd193b1c19e537c6764d94bd07cb6d9fe41a",
    "67bb65e27418fb83657e815cc4ef95d190d9e09c69d2d86cb1306bae4e9c2c39",
    "f55db3e6daed3f10c44e60caea81e419af36db08f71ca164b076eac7baea72fc",
)
CONTRACT = HW / "contracts/m1139ca_c1_independent_per_task_schedule_authority_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "8c92bdd9b7e3668b47b97d2d8a85a0f1977980961470e3dabf7bb2c22d5d9973",
    "b9d12cb80136b674d9fe38794e7cc226eb9f318a3b3c00f866ba7114e15a751d",
    "b8feb4f8394ddbe5445692efc14485ad818cbff719104c8d77aca29ea7a32d0b",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
PRODUCTION_RELEASE_OUTER_SEAL_FILE_SHA256: str | None = None
PRODUCTION_TASKS = 812_160
PRODUCTION_TASKS_PER_SAMPLE = 81_216
PRODUCTION_RECORDS = PRODUCTION_TASKS * len(AXES)
PRODUCTION_COMMIT_CYCLES = 96_000
PRODUCTION_WEIGHT_EVENTS = 70_853_184
PRODUCTION_DMA = 1_476_108


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


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content")


def strict_json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    review = directory / "review.json"; manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(review, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "flat outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and name not in expected and
                name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
                "flat manifest member")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "flat symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "flat special member")
    require(actual == set(expected), "flat exact member set")
    for name, digest in expected.items(): verify_regular(directory / name, digest)
    return strict_json(review)


def source_preflight() -> dict[str, Any]:
    verify_regular(M1016_SOURCE, M1016_SOURCE_SHA)
    verify_regular(M1016_CONTRACT, M1016_CONTRACT_SHA)
    verify_regular(M1102_SOURCE, M1102_SOURCE_SHA)
    verify_double(M1102_CONTRACT, M1102_CONTRACT_ID)
    verify_regular(M1007_SOURCE, M1007_SOURCE_SHA)
    verify_regular(M1137_SOURCE, M1137_SOURCE_SHA)
    verify_regular(Path(str(M1137_CONTRACT) + ".sha256.seal.sha256"),
                   M1137_CONTRACT_OUTER_SHA)
    verify_double(CONTRACT, CONTRACT_ID)
    review = verify_flat(M1138, M1138_ID)
    verify_regular(DOCS359, DOCS359_SHA)
    require(review["status"] ==
            "PASS_M1138C_M1137C_REAL_PER_TASK_BEAT_HOOK_HAMMER__AUTHOR_PRODUCTION_EXPECTED_DIGEST_AUTHORITY_CAPTURE_SOURCE_ONLY" and
            review["authorization"]["production_expected_digest_authority_execution"] is False,
            "M1138 authorization drift")
    return {
        "status": "STOP_PRODUCTION_SCHEDULE_RELEASE_ABSENT__BOUNDED_SOURCE_ONLY",
        "requested_cycle_derivable_from_m1016_raw_primitives": True,
        "m1102_shared_preprocess_alone_sufficient": False,
        "production_release_integrated": False,
        "production_rows_opened": False,
        "production_records": 0,
        "digest_compiler": False,
        "real_driver": False,
    }


@dataclass(frozen=True)
class Geometry:
    samples: int
    operators: int
    chunks: int
    partitions: int
    tasks_per_sample: int
    total_tasks: int
    commit_cycles: int

    def validate(self) -> None:
        require(all(type(value) is int and value > 0 for value in (
                    self.samples, self.operators, self.chunks, self.partitions,
                    self.tasks_per_sample, self.total_tasks)) and
                type(self.commit_cycles) is int and self.commit_cycles >= 0 and
                self.tasks_per_sample == self.operators * self.chunks * self.partitions and
                self.total_tasks == self.samples * self.tasks_per_sample,
                "geometry drift")


PRODUCTION_GEOMETRY = Geometry(10, 4, 47, 432, PRODUCTION_TASKS_PER_SAMPLE,
                               PRODUCTION_TASKS, PRODUCTION_COMMIT_CYCLES)
BOUNDED_GEOMETRY = Geometry(1, 1, 1, 2, 2, 2, 7)


def task_id_for(geometry: Geometry, sample: int, operator: int,
                chunk: int, partition: int) -> int:
    geometry.validate()
    require(0 <= sample < geometry.samples and 0 <= operator < geometry.operators and
            0 <= chunk < geometry.chunks and 0 <= partition < geometry.partitions,
            "task coordinate drift")
    return (((sample * geometry.operators + operator) * geometry.chunks + chunk) *
            geometry.partitions + partition)


def _u64(value: int) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), "u64 drift")
    return struct.pack(">Q", value)


def prior_task_provenance(task_id: int, sample: int, operator: int, chunk: int,
                          partition: int, preprocess_by_axis: Mapping[str, int],
                          work_by_axis: Mapping[str, int], raw_sha256: str) -> str:
    require(tuple(preprocess_by_axis) == AXES and tuple(work_by_axis) == AXES and
            re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is not None,
            "prior provenance input drift")
    payload = [b"M1139CA_PRIOR_TASK\x00\x01", bytes.fromhex(M1016_SOURCE_SHA),
               bytes.fromhex(M1102_SOURCE_SHA), _u64(task_id), _u64(sample),
               _u64(operator), _u64(chunk), _u64(partition), bytes.fromhex(raw_sha256)]
    for axis in AXES:
        payload.extend((_u64(preprocess_by_axis[axis]), _u64(work_by_axis[axis])))
    return hashlib.sha256(b"".join(payload)).hexdigest()


@dataclass(frozen=True)
class PriorTaskTimingPrimitive:
    task_id: int
    sample: int
    operator: int
    chunk: int
    partition: int
    preprocess_by_axis: Mapping[str, int]
    work_by_axis: Mapping[str, int]
    source_raw_sha256: str
    source_task_provenance_sha256: str

    def validate(self, geometry: Geometry) -> None:
        require(type(self.task_id) is int and self.task_id == task_id_for(
                    geometry, self.sample, self.operator, self.chunk, self.partition) and
                tuple(self.preprocess_by_axis) == AXES and tuple(self.work_by_axis) == AXES,
                "prior task coordinate/axis order drift")
        require(re.fullmatch(r"[0-9a-f]{64}", self.source_raw_sha256) is not None,
                "prior raw provenance drift")
        for axis in AXES:
            preprocess = self.preprocess_by_axis[axis]; work = self.work_by_axis[axis]
            require(type(preprocess) is int and preprocess >= 0 and
                    type(work) is int and work >= 0 and work % 8 == 0,
                    "prior timing value drift")
        expected = prior_task_provenance(
            self.task_id, self.sample, self.operator, self.chunk, self.partition,
            self.preprocess_by_axis, self.work_by_axis,
            self.source_raw_sha256)
        require(self.source_task_provenance_sha256 == expected,
                "prior task provenance mismatch")


def schedule_record_provenance(axis: str, task_sequence_ordinal: int,
                               sample: int, operator: int, chunk: int,
                               partition: int, requested_cycle_first: int,
                               source_task_provenance_sha256: str) -> str:
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(M1016_SOURCE_SHA),
        bytes.fromhex(M1102_SOURCE_SHA), bytes.fromhex(M1137_SOURCE_SHA),
        struct.pack(">B", AXES.index(axis)), _u64(task_sequence_ordinal),
        _u64(sample), _u64(operator), _u64(chunk), _u64(partition),
        _u64(requested_cycle_first), bytes.fromhex(source_task_provenance_sha256),
    ))
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PerTaskScheduleAuthorityRecord:
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
                    self.task_sequence_ordinal, self.sample, self.operator, self.chunk,
                    self.partition, self.requested_cycle_first)) and
                self.schedule_record_provenance_sha256 == schedule_record_provenance(
                    self.axis, self.task_sequence_ordinal, self.sample, self.operator,
                    self.chunk, self.partition, self.requested_cycle_first,
                    self.source_task_provenance_sha256),
                "schedule record schema/provenance drift")


@dataclass
class _AxisState:
    previous_start: int | None = None
    previous_work: int = 0
    sample_global_offset: int = 0
    last_requested_cycle: int | None = None
    records: int = 0


class IndependentPerTaskScheduleAuthority:
    """Emit exact axis-ordered task schedule records with O(axes) state."""
    def __init__(self, geometry: Geometry,
                 record_sink: Callable[[PerTaskScheduleAuthorityRecord], None]):
        geometry.validate(); require(callable(record_sink), "record sink must be callable")
        self._geometry = geometry
        self._sink = record_sink
        self._axis = {axis: _AxisState() for axis in AXES}
        self._next_task_id = 0
        self._active_signature: tuple[Any, ...] | None = None
        self._next_axis_index = 0

    def consume_task(self, primitive: PriorTaskTimingPrimitive) -> int:
        require(type(primitive) is PriorTaskTimingPrimitive,
                "exact prior task primitive required")
        primitive.validate(self._geometry)
        signature = (primitive.task_id, primitive.sample, primitive.operator,
                     primitive.chunk, primitive.partition,
                     tuple(primitive.preprocess_by_axis.items()),
                     tuple(primitive.work_by_axis.items()),
                     primitive.source_raw_sha256,
                     primitive.source_task_provenance_sha256)
        if self._active_signature is None:
            require(primitive.task_id == self._next_task_id,
                    "task missing, duplicate, or out of order")
            start_axis = 0
        else:
            require(signature == self._active_signature,
                    "only interrupted task may resume")
            start_axis = self._next_axis_index
        emitted = 0
        for axis_index in range(start_axis, len(AXES)):
            axis = AXES[axis_index]; state = self._axis[axis]
            preprocess = primitive.preprocess_by_axis[axis]
            work = primitive.work_by_axis[axis]
            start = (preprocess if state.previous_start is None else
                     state.previous_start + max(state.previous_work, preprocess) + 2)
            requested = state.sample_global_offset + start - preprocess
            require(state.last_requested_cycle is None or
                    requested >= state.last_requested_cycle, "requested cycle regressed")
            record = PerTaskScheduleAuthorityRecord(
                axis, primitive.task_id, primitive.sample, primitive.operator,
                primitive.chunk, primitive.partition, requested,
                primitive.source_task_provenance_sha256,
                schedule_record_provenance(
                    axis, primitive.task_id, primitive.sample, primitive.operator,
                    primitive.chunk, primitive.partition, requested,
                    primitive.source_task_provenance_sha256))
            record.validate()
            self._sink(record)
            state.previous_start = start; state.previous_work = work
            state.last_requested_cycle = requested; state.records += 1
            emitted += 1
            self._active_signature = signature
            self._next_axis_index = axis_index + 1
        self._active_signature = None; self._next_axis_index = 0
        self._next_task_id += 1
        if self._next_task_id % self._geometry.tasks_per_sample == 0:
            for axis in AXES:
                state = self._axis[axis]
                require(state.previous_start is not None, "empty sample state")
                state.sample_global_offset += (
                    state.previous_start + state.previous_work + 2 +
                    self._geometry.commit_cycles)
                state.previous_start = None; state.previous_work = 0
        return emitted

    def snapshot(self) -> dict[str, Any]:
        return {
            "next_task_id": self._next_task_id,
            "active_signature": self._active_signature,
            "next_axis_index": self._next_axis_index,
            "axes": {axis: {
                "previous_start": state.previous_start,
                "previous_work": state.previous_work,
                "sample_global_offset": state.sample_global_offset,
                "last_requested_cycle": state.last_requested_cycle,
                "records": state.records,
            } for axis, state in self._axis.items()},
            "state_complexity": "O(axes)",
        }

    def finalize(self) -> dict[str, Any]:
        require(self._next_task_id == self._geometry.total_tasks and
                self._active_signature is None and self._next_axis_index == 0 and
                all(state.records == self._geometry.total_tasks
                    for state in self._axis.values()),
                "terminal task/axis record conservation mismatch")
        return {
            "schema": "m1139ca_independent_per_task_schedule_terminal_v1",
            "status": "PASS_INDEPENDENT_TASK_SCHEDULE_STREAM",
            "tasks": self._next_task_id,
            "records_by_axis": {axis: self._axis[axis].records for axis in AXES},
            "last_requested_cycle_by_axis": {
                axis: self._axis[axis].last_requested_cycle for axis in AXES},
            "state_complexity": "O(axes)",
            "retained_record_or_key_history": False,
        }


class CountingDigestSink:
    def __init__(self):
        self.count = 0
        self.digest = hashlib.sha256()
        self.last_axis_index = -1
        self.last_task = -1

    def __call__(self, record: PerTaskScheduleAuthorityRecord) -> None:
        record.validate()
        expected_axis_index = self.count % len(AXES)
        expected_task = self.count // len(AXES)
        require(record.axis == AXES[expected_axis_index] and
                record.task_sequence_ordinal == expected_task,
                "axis/task stream order drift")
        self.digest.update(json.dumps(record.__dict__, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode())
        self.last_axis_index = expected_axis_index; self.last_task = expected_task
        self.count += 1


def bounded_primitives() -> Iterator[PriorTaskTimingPrimitive]:
    rows = (
        ({"candidate": 10, "strongest_zero": 10, "same_coordinate_bit": 10},
         {"candidate": 16, "strongest_zero": 0, "same_coordinate_bit": 8}),
        ({"candidate": 6, "strongest_zero": 6, "same_coordinate_bit": 6},
         {"candidate": 8, "strongest_zero": 16, "same_coordinate_bit": 0}),
    )
    for task_id, (preprocess, work) in enumerate(rows):
        raw = hashlib.sha256(("bounded:" + str(task_id)).encode()).hexdigest()
        yield PriorTaskTimingPrimitive(
            task_id, 0, 0, 0, task_id, preprocess, work,
            raw,
            prior_task_provenance(task_id, 0, 0, 0, task_id,
                                  preprocess, work, raw))


def requested_cycle_derivability_audit() -> dict[str, Any]:
    return {
        "status": "PASS_DERIVABLE_FROM_M1016_RAW_TASK_PRIMITIVES__M1102_ALONE_INSUFFICIENT",
        "m1016_exact_inputs": ["raw task masks", "design-specific preprocess",
                                 "design work", "pipeline recurrence", "sample offset"],
        "m1102_retained_preprocess": "shared maximum only",
        "production_rows_needed": True,
        "production_rows_opened_now": False,
        "invented_requested_cycles": False,
    }


def source_small_oracle() -> dict[str, Any]:
    preflight = source_preflight(); sink = CountingDigestSink()
    authority = IndependentPerTaskScheduleAuthority(BOUNDED_GEOMETRY, sink)
    requested = {axis: [] for axis in AXES}

    class Observe:
        def __call__(self, record):
            requested[record.axis].append(record.requested_cycle_first)
            sink(record)

    authority._sink = Observe()
    for primitive in bounded_primitives():
        require(authority.consume_task(primitive) == 3, "three axes per task")
    terminal = authority.finalize()
    require(requested == {"candidate": [0, 22], "strongest_zero": [0, 12],
                          "same_coordinate_bit": [0, 14]} and sink.count == 6,
            "bounded requested-cycle oracle drift")
    stopped = False
    try:
        next(iter_production_schedule_authority_records())
    except Failure:
        stopped = True
    require(stopped, "production iterator escaped absent release")
    return {
        "schema": "m1139ca_independent_per_task_schedule_small_oracle_v1",
        "status": "PASS_BOUNDED_2_TASK_3_AXIS_SCHEDULE__PRODUCTION_STOP",
        "preflight": preflight,
        "derivability": requested_cycle_derivability_audit(),
        "requested_cycles": requested, "records": sink.count,
        "record_digest": sink.digest.hexdigest(), "terminal": terminal,
        "production_rows_opened": False, "production_records": 0,
        "digest_compiler": False, "real_driver": False,
    }


def iter_production_schedule_authority_records() -> Iterator[Any]:
    preflight = source_preflight()
    require(PRODUCTION_RELEASE_OUTER_SEAL_FILE_SHA256 is not None and
            preflight["production_release_integrated"] is True,
            "STOP: production schedule release is absent")
    # The future released successor must independently read/hash frozen M410
    # bytes, re-derive design-specific preprocess/work, and feed this class.
    # This source stage deliberately opens no canonical row and emits nothing.
    if False:  # pragma: no cover
        yield None


def main() -> None:
    require(sys.argv[1:] in ([], ["--self-test"]), "bounded self-test only")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
