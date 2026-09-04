#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive O(axes) streaming weight-event validator/scheduler sink source.

No real producer hook or production expected-digest authority is integrated.
The bounded oracle uses six synthetic events.  Production remains fail-closed.
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
from typing import Any, Callable, Iterator, Mapping

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1130_SOURCE = HERE / "build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
M1130_SOURCE_SHA = "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf"
M1132_SOURCE = HERE / "build_m1132c_c1_upstream_weight_event_producer_source.py"
M1132_SOURCE_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
M1134 = HW / "reviews/m1134c_m1132c_production_scale_first_principles_audit_r1_20260830"
M1134_OUTER = "8522bc2b5b271a1b9e55a420ac4d82c221c8455175910a803919243de9ffdf11"
CONTRACT = HW / "contracts/m1135c_c1_oaxes_streaming_weight_validator_sink_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "6d6fcdcd414e020c6aa456d4e162a63e85d4f70cd37d849abbe292bc7ce9c41f",
    "8532cbf2f9d69852593536d1900ab95a225d2484bd46de8f82c050e34cd5a67b",
    "310608b91bc36f48fd7a82024ef84e2843f802878cf9ed6e11ee799823bda0d6",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
AXIS_CODE = {axis: index for index, axis in enumerate(AXES)}
NATIVE_SLICES = 24
EXPECTED_PRODUCTION_EVENTS_PER_AXIS = 70_853_184
EVENT_FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)
SYNTHETIC_EXPECTED_DIGESTS = {
    "candidate": "f4e5a19127c3310ecfe1b538c9f1cc295a5a8b6f83488e28fbfcb44acae891c7",
    "strongest_zero": "f502037e7b6de7dd55105f5db435e6ab60312962a8ebbe7b864e6c3b3c06e8a3",
    "same_coordinate_bit": "4c05281142fa41fe9c2bf98024862658bebe5480c03ffc4de6bca68bd662e435",
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
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.name], "double seal content")


def verify_flat_outer(directory: Path, expected_outer: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    require(directory.is_dir() and not directory.is_symlink() and
            outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "flat outer")


_M1130 = None
_M1128 = None


def load_m1130():
    global _M1130
    if _M1130 is None:
        verify_regular(M1130_SOURCE, M1130_SOURCE_SHA)
        spec = importlib.util.spec_from_file_location("m1135c_frozen_m1130", M1130_SOURCE)
        require(spec is not None and spec.loader is not None, "cannot load M1130C")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _M1130 = module
    return _M1130


def load_m1128():
    """Load the frozen addressed-row type once, independent of stream length."""
    global _M1128
    if _M1128 is None:
        m1130 = load_m1130()
        verify_regular(m1130.M1128_SOURCE, m1130.M1128_SOURCE_SHA)
        spec = importlib.util.spec_from_file_location(
            "m1135c_frozen_m1128", m1130.M1128_SOURCE)
        require(spec is not None and spec.loader is not None,
                "cannot load M1128C")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _M1128 = module
    return _M1128


def source_preflight() -> dict[str, Any]:
    verify_regular(M1130_SOURCE, M1130_SOURCE_SHA)
    verify_regular(M1132_SOURCE, M1132_SOURCE_SHA)
    verify_flat_outer(M1134, M1134_OUTER)
    verify_double(CONTRACT, CONTRACT_ID)
    verify_regular(DOCS359, DOCS359_SHA)
    return {
        "status": "STOP_NO_REAL_HOOK_OR_PRODUCTION_EXPECTED_DIGEST_AUTHORITY",
        "m1134_oaxes_source_authorized": True,
        "real_hook_integrated": False,
        "production_expected_digest_authority_integrated": False,
        "canonical_ready": False,
        "canonical_rows": 0,
        "canonical_events": 0,
    }


def _hex64(value: str) -> bool:
    return (type(value) is str and len(value) == 64 and
            re.fullmatch(r"[0-9a-f]{64}", value) is not None)


@dataclass(frozen=True)
class ExpectedDigestAuthority:
    scope: str
    authority_id_sha256: str
    expected_count_by_axis: Mapping[str, int]
    expected_digest_by_axis: Mapping[str, str]

    def validate(self) -> None:
        require(self.scope in ("production", "bounded_synthetic") and
                _hex64(self.authority_id_sha256) and
                tuple(self.expected_count_by_axis) == AXES and
                tuple(self.expected_digest_by_axis) == AXES,
                "expected digest authority schema drift")
        for axis in AXES:
            count = self.expected_count_by_axis[axis]
            require(type(count) is int and count > 0 and
                    _hex64(self.expected_digest_by_axis[axis]),
                    "expected digest authority axis value drift")
        if self.scope == "production":
            require(all(self.expected_count_by_axis[axis] ==
                        EXPECTED_PRODUCTION_EVENTS_PER_AXIS for axis in AXES),
                    "production expected count must be 70,853,184 per axis")
        else:
            require(all(self.expected_count_by_axis[axis] <= 64 for axis in AXES),
                    "bounded synthetic authority exceeds 64 events per axis")


@dataclass
class _AxisState:
    next_beat: int
    next_transaction: int
    event_count: int
    bytes: int
    native_activations: int
    stalled_transactions: int
    stall_cycles: int
    first_beat: int | None
    first_transaction: int | None
    last_beat: int | None
    last_transaction: int | None
    last_scheduler_key: tuple[int, int, int, int] | None
    digest: Any


def recompute_exact_once_id(axis: str, task_id: int, source_local_ordinal: int,
                            service_beat_ordinal: int,
                            store_transaction_ordinal: int) -> str:
    """Independent copy of the frozen exact-ID byte contract."""
    payload = (f"m1130c:{axis}:{task_id}:{source_local_ordinal}:"
               f"{service_beat_ordinal}:{store_transaction_ordinal}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _u64(value: int, label: str) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64),
            label + " outside canonical u64")
    return struct.pack(">Q", value)


def canonical_event_bytes(event: Any, sequence_ordinal: int,
                          scheduled_cycle: int, stall_cycles: int) -> bytes:
    """Unambiguous fixed-endian encoding of all 17 fields plus schedule."""
    require(type(event.axis) is str and event.axis in AXIS_CODE, "axis encoding")
    require(event.op == "WRITE", "streaming sink accepts WRITE only")
    pieces = [
        b"M1135C\x00\x01", struct.pack(">B", AXIS_CODE[event.axis]),
        _u64(event.task_id, "task_id"),
        _u64(event.source_local_ordinal, "source_local_ordinal"),
        _u64(event.requested_cycle, "requested_cycle"), b"W",
        struct.pack(">B", event.logical_bank),
        struct.pack(">B", event.half_slot),
        struct.pack(">B", event.logical_row),
        struct.pack(">B", event.local_row),
        struct.pack(">B", len(event.native_slices)),
        bytes(event.native_slices),
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


class OAxesStreamingWeightValidatorSink:
    """Validate, schedule, sink and commit one event using fixed-size state."""
    def __init__(self, authority: ExpectedDigestAuthority,
                 scheduled_sink: Callable[[Any], None]):
        authority.validate()
        require(callable(scheduled_sink), "scheduled sink must be callable")
        self._authority = authority
        self._sink = scheduled_sink
        self._state = {
            axis: _AxisState(0, 0, 0, 0, 0, 0, 0, None, None, None, None,
                             None, hashlib.sha256())
            for axis in AXES
        }
        self._next_free_cycle = {axis: [0] * NATIVE_SLICES for axis in AXES}
        self._finalized = False

    def __call__(self, event: Any):
        require(not self._finalized, "stream already finalized")
        m1130 = load_m1130()
        require(type(event) is m1130.InternalWeightServiceRefillEvent,
                "exact M1130C event type required")
        event.validate()
        require(event.op == "WRITE", "stream accepts weight WRITE events only")
        axis = event.axis
        state = self._state[axis]
        expected_count = self._authority.expected_count_by_axis[axis]
        require(state.event_count < expected_count, "event count exceeds authority")
        require(event.service_beat_ordinal == state.next_beat,
                "per-axis service beat is not globally contiguous")
        require(event.store_transaction_ordinal == state.next_transaction,
                "per-axis transaction is not globally contiguous")
        scheduler_key = (event.requested_cycle, event.task_id,
                         event.source_local_ordinal,
                         event.store_transaction_ordinal)
        require(state.last_scheduler_key is None or
                scheduler_key >= state.last_scheduler_key,
                "per-axis frozen scheduler key regressed")
        independent_id = recompute_exact_once_id(
            axis, event.task_id, event.source_local_ordinal,
            int(event.service_beat_ordinal), event.store_transaction_ordinal)
        require(event.service_event_exact_once_id == independent_id,
                "independently recomputed exact-once ID mismatch")
        next_free = self._next_free_cycle[axis]
        scheduled_cycle = max(
            [event.requested_cycle] +
            [next_free[native_slice] for native_slice in event.native_slices])
        stall_cycles = scheduled_cycle - event.requested_cycle
        row = load_m1128().AddressedWeightTransaction(
                event.axis, event.requested_cycle, scheduled_cycle, stall_cycles,
                event.op, event.logical_bank, event.half_slot,
                event.logical_row, event.local_row, event.native_slices,
                event.bytes, event.byte_enable_per_slice,
                event.native_macro_activations, event.service_beat_ordinal,
                event.store_transaction_ordinal, event.task_id,
                event.source_local_ordinal, event.source_row_provenance_sha256)
        row.validate()
        candidate_digest = state.digest.copy()
        candidate_digest.update(canonical_event_bytes(
            event, state.event_count, scheduled_cycle, stall_cycles))
        # No validator, digest, counter or scheduler state changes before this
        # call.  Any exception propagates and leaves the event retryable.
        self._sink(row)
        state.digest = candidate_digest
        state.event_count += 1
        state.next_beat += 1
        state.next_transaction += 1
        state.bytes += event.bytes
        state.native_activations += event.native_macro_activations
        state.stalled_transactions += int(stall_cycles > 0)
        state.stall_cycles += stall_cycles
        if state.first_beat is None:
            state.first_beat = int(event.service_beat_ordinal)
            state.first_transaction = event.store_transaction_ordinal
        state.last_beat = int(event.service_beat_ordinal)
        state.last_transaction = event.store_transaction_ordinal
        state.last_scheduler_key = scheduler_key
        for native_slice in event.native_slices:
            next_free[native_slice] = scheduled_cycle + 1
        return row

    def snapshot(self) -> dict[str, Any]:
        return {
            axis: {
                "next_beat": state.next_beat,
                "next_transaction": state.next_transaction,
                "event_count": state.event_count,
                "bytes": state.bytes,
                "native_activations": state.native_activations,
                "stalled_transactions": state.stalled_transactions,
                "stall_cycles": state.stall_cycles,
                "first_beat": state.first_beat,
                "first_transaction": state.first_transaction,
                "last_beat": state.last_beat,
                "last_transaction": state.last_transaction,
                "last_scheduler_key": state.last_scheduler_key,
                "digest": state.digest.hexdigest(),
                "next_free_cycle": tuple(self._next_free_cycle[axis]),
            }
            for axis, state in self._state.items()
        }

    def finalize(self) -> dict[str, Any]:
        require(not self._finalized, "stream already finalized")
        axes = {}
        for axis in AXES:
            state = self._state[axis]
            expected_count = self._authority.expected_count_by_axis[axis]
            require(state.event_count == expected_count and
                    state.next_beat == expected_count and
                    state.next_transaction == expected_count and
                    state.first_beat == 0 and state.first_transaction == 0 and
                    state.last_beat == expected_count - 1 and
                    state.last_transaction == expected_count - 1,
                    "terminal per-axis count/ordinal conservation mismatch")
            digest = state.digest.hexdigest()
            require(digest == self._authority.expected_digest_by_axis[axis],
                    "terminal independently authorized digest mismatch")
            axes[axis] = {
                "events": state.event_count, "first_beat": state.first_beat,
                "last_beat": state.last_beat,
                "first_transaction": state.first_transaction,
                "last_transaction": state.last_transaction,
                "digest": digest, "bytes": state.bytes,
                "native_activations": state.native_activations,
                "stalled_transactions": state.stalled_transactions,
                "stall_cycles": state.stall_cycles,
            }
        self._finalized = True
        return {
            "schema": "m1135c_oaxes_streaming_terminal_receipt_v1",
            "status": "PASS_STREAMING_COUNTS_ORDINALS_DIGESTS_AND_1RW_SCHEDULE",
            "authority_scope": self._authority.scope,
            "authority_id_sha256": self._authority.authority_id_sha256,
            "axes": axes, "state_complexity": "O(axes + axes*24)",
        }


def _synthetic_event(axis: str, axis_index: int, ordinal: int):
    m1130 = load_m1130()
    provenance = hashlib.sha256(
        f"m1135c:{axis}:{ordinal}".encode("utf-8")).hexdigest()
    return m1130.InternalWeightServiceRefillEvent(
        axis, axis_index, ordinal, 5, "WRITE", 0, 0, axis_index,
        axis_index, tuple(range(8)), 128, (0xffff,) * 8, 8, ordinal,
        ordinal, recompute_exact_once_id(axis, axis_index, ordinal,
                                         ordinal, ordinal), provenance)


def iter_bounded_synthetic_events() -> Iterator[Any]:
    for axis_index, axis in enumerate(AXES):
        for ordinal in range(2):
            yield _synthetic_event(axis, axis_index, ordinal)


def iter_canonical_oaxes_streaming_weight_events() -> Iterator[Any]:
    preflight = source_preflight()
    require(preflight["canonical_ready"] is True,
            "STOP: no real hook or production expected-digest authority")
    if False:  # pragma: no cover
        yield None


def source_small_oracle() -> dict[str, Any]:
    preflight = source_preflight()
    authority = ExpectedDigestAuthority(
        "bounded_synthetic",
        hashlib.sha256(b"m1135c-bounded-authority-v1").hexdigest(),
        {axis: 2 for axis in AXES}, SYNTHETIC_EXPECTED_DIGESTS)
    rows = []
    validator = OAxesStreamingWeightValidatorSink(authority, rows.append)
    for event in iter_bounded_synthetic_events():
        validator(event)
    terminal = validator.finalize()
    require(len(rows) == 6 and
            all(terminal["axes"][axis]["events"] == 2 and
                terminal["axes"][axis]["stalled_transactions"] == 1 and
                terminal["axes"][axis]["stall_cycles"] == 1
                for axis in AXES), "bounded streaming oracle drift")
    stopped = False
    try:
        next(iter_canonical_oaxes_streaming_weight_events())
    except Failure:
        stopped = True
    require(stopped, "canonical iterator escaped absent authority/hook")
    return {
        "schema": "m1135c_oaxes_streaming_weight_validator_sink_small_oracle_v1",
        "status": "PASS_BOUNDED_O_AXES_STREAMING__CANONICAL_ZERO_STOP",
        "preflight": preflight, "terminal": terminal,
        "bounded_rows": len(rows), "canonical_rows": 0, "canonical_events": 0,
        "real_hook": False, "full_51840000": False,
        "eda_gpu_remote": False,
    }


def main() -> None:
    require(sys.argv[1:] in ([], ["--self-test"]), "bounded self-test only")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
