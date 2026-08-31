#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive per-task weight-beat creation hook into the M1135C stream sink.

The production driver and production digest authority are intentionally absent.
The bounded oracle creates each event in the same live beat loop used by the
production method and retains no event, row, or key history.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Callable, Iterator

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1016_SOURCE = HERE / "run_m1016_c1_full_matched_address_replay.py"
M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1132_SOURCE = HERE / "build_m1132c_c1_upstream_weight_event_producer_source.py"
M1132_SOURCE_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
M1135_SOURCE = HERE / "build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
M1135_SOURCE_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
M1135_CONTRACT = HW / "contracts/m1135c_c1_oaxes_streaming_weight_validator_sink_source_contract_r1_20260830.json"
M1135_CONTRACT_ID = (
    "6d6fcdcd414e020c6aa456d4e162a63e85d4f70cd37d849abbe292bc7ce9c41f",
    "8532cbf2f9d69852593536d1900ab95a225d2484bd46de8f82c050e34cd5a67b",
    "310608b91bc36f48fd7a82024ef84e2843f802878cf9ed6e11ee799823bda0d6",
)
M1136 = HW / "reviews/m1136c_m1135c_c1_oaxes_streaming_validator_sink_hammer_r1_20260830"
M1136_ID = (
    "35559fdec20ddee27f29ef6f2cf1841c55258f067c8cbc8dbc16b2159548cb81",
    "45056ee2a2e2e79eebfd2b438899c64bef98bece0238ce1c93a8a4ee1a8d74f0",
    "fe766b8810c74489f058f0cc38275951e335c9e369ef096e608e3fe82d1a198d",
)
CONTRACT = HW / "contracts/m1137c_c1_real_per_task_weight_beat_hook_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "51e9370e43830ba10075c994d73da665e8b7d559697f54ebb38ad93a13128acc",
    "01c888e5477133d716ad0db499107ff77eb21b2b1e17688784df3a2716e45e61",
    "865dac0d7bf89f1a57777f5eafbc6b6fef8b8cbc78403c1822ba5191adfc349d",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
TASKS = 812_160
PRODUCTION_EVENTS_PER_AXIS = 70_853_184
PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256: str | None = None
BOUNDED_TASKS_PER_AXIS = 2
BOUNDED_BEATS_PER_TASK = 2
EVENT_FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)
BOUNDED_EXPECTED_DIGESTS = {
    "candidate": "49facfeb00bb3b388d1ac1145a9a099602f54a625875ed34d14cfa5125edc749",
    "strongest_zero": "47950bf0e7f5187e3516aa9fd87e605e75789972663bb1772522fc298aecad4b",
    "same_coordinate_bit": "605be1f2dfc3443850bf4f2a7bee0f7e8c7fb2d992862d50f5a8c143fd0a63d9",
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


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
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
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "sealed authority directory drift")
    verify_regular(review, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "sealed authority outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "sealed manifest row")
        name = fields[1].lstrip("*"); relative = Path(name)
        require(name not in expected and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "sealed manifest member")
        expected[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member set")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)
    return strict_json(review)


def load_frozen(path: Path, expected: str, name: str):
    verify_regular(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_M1016 = None
_M1135 = None


def load_m1016():
    global _M1016
    if _M1016 is None:
        _M1016 = load_frozen(M1016_SOURCE, M1016_SOURCE_SHA, "m1137c_frozen_m1016")
    return _M1016


def load_m1135():
    global _M1135
    if _M1135 is None:
        _M1135 = load_frozen(M1135_SOURCE, M1135_SOURCE_SHA, "m1137c_frozen_m1135")
    return _M1135


def source_preflight() -> dict[str, Any]:
    verify_regular(M1016_SOURCE, M1016_SOURCE_SHA)
    verify_regular(M1102_SOURCE, M1102_SOURCE_SHA)
    verify_regular(M1132_SOURCE, M1132_SOURCE_SHA)
    verify_regular(M1135_SOURCE, M1135_SOURCE_SHA)
    verify_double(M1135_CONTRACT, M1135_CONTRACT_ID)
    verify_double(CONTRACT, CONTRACT_ID)
    review = verify_flat(M1136, M1136_ID)
    verify_regular(DOCS359, DOCS359_SHA)
    require(review["status"] ==
            "PASS_M1136C_M1135C_O_AXES_STREAMING_HAMMER__AUTHOR_ADDITIVE_REAL_PRODUCER_HOOK_SOURCE_ONLY" and
            review["authorization"]["additive_real_producer_hook_source_authoring"] is True and
            review["authorization"]["real_hook_execution"] is False,
            "M1136C authorization drift")
    m1016 = load_m1016(); m1135 = load_m1135()
    require((m1016.TASKS, m1016.EXPECTED_SERVICE_COUNTS["weight"],
             tuple(m1016.DESIGNS)) == (TASKS, PRODUCTION_EVENTS_PER_AXIS, AXES),
            "frozen M1016 geometry drift")
    require(tuple(m1135.EVENT_FIELDS) == EVENT_FIELDS and
            m1135.EXPECTED_PRODUCTION_EVENTS_PER_AXIS == PRODUCTION_EVENTS_PER_AXIS,
            "frozen M1135 schema/scale drift")
    return {
        "status": "STOP_NO_PRODUCTION_DIGEST_AUTHORITY_OR_REAL_DRIVER",
        "successor_source_exists": True,
        "bounded_hook_ready": True,
        "production_expected_digest_authority_integrated": False,
        "real_production_driver_integrated": False,
        "canonical_rows": 0,
        "canonical_events": 0,
    }


def _u64(value: int, label: str) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), label + " outside u64")
    return struct.pack(">Q", value)


def successor_exact_once_id(axis: str, task_id: int, local_ordinal: int,
                            beat_ordinal: int, transaction_ordinal: int) -> str:
    payload = (f"m1130c:{axis}:{task_id}:{local_ordinal}:"
               f"{beat_ordinal}:{transaction_ordinal}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def beat_provenance(axis: str, sample: int, operator: int, chunk: int,
                    partition: int, task_id: int, local_ordinal: int,
                    global_beat_ordinal: int, requested_cycle: int,
                    half_slot: int, logical_row: int,
                    native_slices: tuple[int, ...]) -> str:
    """Independent fixed-endian identity made at the live beat creation point."""
    axis_code = AXES.index(axis)
    payload = b"".join((
        b"M1137C_REAL_BEAT\x00\x01",
        bytes.fromhex(M1016_SOURCE_SHA), bytes.fromhex(M1102_SOURCE_SHA),
        bytes.fromhex(M1135_SOURCE_SHA), struct.pack(">B", axis_code),
        _u64(sample, "sample"), _u64(operator, "operator"),
        _u64(chunk, "chunk"), _u64(partition, "partition"),
        _u64(task_id, "task_id"), _u64(local_ordinal, "local ordinal"),
        _u64(global_beat_ordinal, "global beat"),
        _u64(requested_cycle, "requested cycle"),
        struct.pack(">B", half_slot), struct.pack(">B", logical_row),
        struct.pack(">B", len(native_slices)), bytes(native_slices),
    ))
    return hashlib.sha256(payload).hexdigest()


@dataclass
class _TaskCursor:
    next_task_id: int = 0
    active_signature: tuple[int, ...] | None = None
    next_global_beat: int | None = None
    emitted: int = 0


class M1016SuccessorPerTaskWeightBeatHook:
    """Create and immediately consume each weight beat with O(axes) state."""
    def __init__(self, authority: Any, row_sink: Callable[[Any], None]):
        m1135 = load_m1135()
        require(type(authority) is m1135.ExpectedDigestAuthority,
                "exact M1135C expected-digest authority required")
        authority.validate()
        if authority.scope == "production":
            require(PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256 is not None and
                    authority.authority_id_sha256 ==
                        PRODUCTION_EXPECTED_DIGEST_AUTHORITY_ID_SHA256,
                    "STOP: sealed production digest authority is absent")
        require(callable(row_sink), "row sink must be callable")
        self._authority_scope = authority.scope
        self._validator = m1135.OAxesStreamingWeightValidatorSink(authority, row_sink)
        self._cursor = {axis: _TaskCursor() for axis in AXES}

    def _stream_task_interval(self, *, axis: str, sample: int, operator: int,
                              chunk: int, partition: int, task_id: int,
                              global_beat_begin: int, global_beat_end: int,
                              requested_cycle_first: int) -> int:
        require(axis in AXES and all(type(value) is int and value >= 0 for value in
                (sample, operator, chunk, partition, task_id, global_beat_begin,
                 global_beat_end, requested_cycle_first)), "task/interval schema")
        require(global_beat_begin < global_beat_end, "task beat interval must be nonempty")
        state = self._cursor[axis]
        signature = (sample, operator, chunk, partition, task_id,
                     global_beat_begin, global_beat_end, requested_cycle_first)
        if state.active_signature is None:
            require(task_id == state.next_task_id, "per-axis task id is not contiguous")
            current = global_beat_begin
        else:
            require(state.active_signature == signature and
                    state.next_global_beat is not None,
                    "only the interrupted active task may resume")
            current = state.next_global_beat
        emitted_now = 0
        m1135 = load_m1135(); m1130 = m1135.load_m1130()
        while current < global_beat_end:
            local_ordinal = current - global_beat_begin
            requested_cycle = requested_cycle_first + local_ordinal
            half_slot = task_id & 1
            logical_row = current % 16
            slice_base = ((current // 16) % 3) * 8
            native_slices = tuple(range(slice_base, slice_base + 8))
            service_ordinal = state.emitted
            exact_id = successor_exact_once_id(
                axis, task_id, local_ordinal, service_ordinal, service_ordinal)
            provenance = beat_provenance(
                axis, sample, operator, chunk, partition, task_id,
                local_ordinal, current, requested_cycle, half_slot,
                logical_row, native_slices)
            event = m1130.InternalWeightServiceRefillEvent(
                axis, task_id, local_ordinal, requested_cycle, "WRITE",
                half_slot, half_slot, logical_row, half_slot * 16 + logical_row,
                native_slices, 128, (0xffff,) * 8, 8, service_ordinal,
                service_ordinal, exact_id, provenance)
            event.validate()
            # The frozen M1135C sink commits only after its downstream sink
            # succeeds.  This successor advances its cursor only after that.
            self._validator(event)
            current += 1
            state.emitted += 1
            emitted_now += 1
            if current < global_beat_end:
                state.active_signature = signature
                state.next_global_beat = current
            else:
                state.active_signature = None
                state.next_global_beat = None
                state.next_task_id += 1
        return emitted_now

    def stream_production_task(self, *, axis: str, sample: int, operator: int,
                               chunk: int, partition: int,
                               requested_cycle_first: int) -> int:
        require(self._authority_scope == "production",
                "production task requires production digest authority")
        m1016 = load_m1016()
        task_id = m1016.task_index(sample, operator, chunk, partition)
        begin = (task_id * PRODUCTION_EVENTS_PER_AXIS) // TASKS
        end = ((task_id + 1) * PRODUCTION_EVENTS_PER_AXIS) // TASKS
        return self._stream_task_interval(
            axis=axis, sample=sample, operator=operator, chunk=chunk,
            partition=partition, task_id=task_id, global_beat_begin=begin,
            global_beat_end=end, requested_cycle_first=requested_cycle_first)

    def stream_bounded_task(self, *, axis: str, task_id: int,
                            requested_cycle_first: int) -> int:
        require(self._authority_scope == "bounded_synthetic" and
                0 <= task_id < BOUNDED_TASKS_PER_AXIS,
                "bounded task scope/id drift")
        begin = task_id * BOUNDED_BEATS_PER_TASK
        return self._stream_task_interval(
            axis=axis, sample=0, operator=0, chunk=0, partition=task_id,
            task_id=task_id, global_beat_begin=begin,
            global_beat_end=begin + BOUNDED_BEATS_PER_TASK,
            requested_cycle_first=requested_cycle_first)

    def snapshot(self) -> dict[str, Any]:
        return {
            "successor": {
                axis: {"next_task_id": state.next_task_id,
                       "active_signature": state.active_signature,
                       "next_global_beat": state.next_global_beat,
                       "emitted": state.emitted}
                for axis, state in self._cursor.items()
            },
            "validator": self._validator.snapshot(),
            "state_complexity": "O(axes + axes*24)",
        }

    def finalize(self) -> dict[str, Any]:
        expected_tasks = TASKS if self._authority_scope == "production" else BOUNDED_TASKS_PER_AXIS
        require(all(state.next_task_id == expected_tasks and
                    state.active_signature is None and state.next_global_beat is None
                    for state in self._cursor.values()),
                "terminal task/cursor conservation mismatch")
        terminal = self._validator.finalize()
        return {
            "schema": "m1137c_real_per_task_weight_beat_hook_terminal_v1",
            "status": "PASS_TASK_BEAT_CREATION_TO_M1135C_STREAM",
            "authority_scope": self._authority_scope,
            "tasks_per_axis": expected_tasks,
            "events_per_axis": {
                axis: self._cursor[axis].emitted for axis in AXES},
            "m1135c_terminal": terminal,
            "state_complexity": "O(axes + axes*24)",
            "retained_rows_events_or_key_history": False,
        }


class CountingDigestRowSink:
    """O(1) bounded observer; no row collection."""
    def __init__(self):
        self.count = 0
        self.digest = hashlib.sha256()

    def __call__(self, row: Any) -> None:
        row.validate()
        payload = json.dumps({
            "axis": row.axis, "requested_cycle": row.requested_cycle,
            "cycle": row.cycle, "stall_cycles": row.stall_cycles,
            "logical_bank": row.logical_bank, "logical_row": row.logical_row,
            "native_slices": list(row.native_slices), "bytes": row.bytes,
            "service_beat_ordinal": row.service_beat_ordinal,
            "store_transaction_ordinal": row.store_transaction_ordinal,
            "task_id": row.source_task_id,
            "source_local_ordinal": row.source_local_ordinal,
            "source_row_provenance_sha256": row.source_row_provenance_sha256,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.digest.update(payload); self.count += 1


def bounded_authority():
    m1135 = load_m1135()
    return m1135.ExpectedDigestAuthority(
        "bounded_synthetic", hashlib.sha256(b"m1137c-bounded-authority-v1").hexdigest(),
        {axis: BOUNDED_TASKS_PER_AXIS * BOUNDED_BEATS_PER_TASK for axis in AXES},
        BOUNDED_EXPECTED_DIGESTS)


def _run_bounded(finalize: bool) -> tuple[M1016SuccessorPerTaskWeightBeatHook,
                                          CountingDigestRowSink, Any]:
    sink = CountingDigestRowSink()
    hook = M1016SuccessorPerTaskWeightBeatHook(bounded_authority(), sink)
    for axis in AXES:
        hook.stream_bounded_task(axis=axis, task_id=0, requested_cycle_first=5)
        hook.stream_bounded_task(axis=axis, task_id=1, requested_cycle_first=6)
    terminal = hook.finalize() if finalize else None
    return hook, sink, terminal


def source_small_oracle() -> dict[str, Any]:
    preflight = source_preflight()
    hook, sink, terminal = _run_bounded(finalize=True)
    snapshot = hook.snapshot()
    require(sink.count == 12 and terminal["events_per_axis"] == {
                axis: 4 for axis in AXES} and
            all(terminal["m1135c_terminal"]["axes"][axis]["events"] == 4
                for axis in AXES), "bounded hook conservation drift")
    stopped = False
    try:
        next(iter_canonical_real_per_task_weight_beats())
    except Failure:
        stopped = True
    require(stopped, "canonical source escaped missing authority/driver")
    return {
        "schema": "m1137c_real_per_task_weight_beat_hook_small_oracle_v1",
        "status": "PASS_BOUNDED_2_TASK_3_AXIS_REAL_CREATION_HOOK__CANONICAL_STOP",
        "preflight": preflight, "terminal": terminal,
        "row_sink_count": sink.count, "row_sink_digest": sink.digest.hexdigest(),
        "state_shape": {
            "axes": len(snapshot["successor"]),
            "next_free_per_axis": {
                axis: len(snapshot["validator"][axis]["next_free_cycle"])
                for axis in AXES},
            "retained_rows_events_or_key_history": False,
        },
        "canonical_rows": 0, "canonical_events": 0,
        "full_replay": False, "eda_gpu_remote": False,
    }


def iter_canonical_real_per_task_weight_beats() -> Iterator[Any]:
    preflight = source_preflight()
    require(preflight["production_expected_digest_authority_integrated"] is True and
            preflight["real_production_driver_integrated"] is True,
            "STOP: production digest authority and real driver are absent")
    if False:  # pragma: no cover
        yield None


def main() -> None:
    require(sys.argv[1:] in ([], ["--self-test"]), "bounded self-test only")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
