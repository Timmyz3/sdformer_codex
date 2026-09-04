#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1130C additive interface for real internal weight service/refill events.

Frozen M1102/M1016 has no per-beat addressed weight event object.  Therefore
the canonical zero-argument iterator stops before opening any row.  This file
defines the exact producer-supplied event interface and a bounded synthetic
exercise only; aggregate count/first-beat/interval data is never expanded.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

M1128_SOURCE = HERE / "build_m1128c_c1_weight_service_addressed_ledger_source.py"
M1128_SOURCE_SHA = "d25f9e4fdfda62f56e7efb120fe0c8f6108a4b23ba4eee712e3ec471b5fa493e"
M1128_CONTRACT = HW / "contracts/m1128c_c1_weight_service_addressed_ledger_source_contract_r1_20260830.json"
M1128_CONTRACT_SHA = "69bcc952953a23d102ac021e2b67375ef0d539b47bf88c347081200fae1b9102"
M1128_CONTRACT_OUTER_SHA = "bb8eca6f7dd02546a9d8aed009e44212c89ed9fe90376ce83306128133786166"
M1128_AUTHOR = HW / "reviews/m1128c_c1_weight_service_addressed_ledger_author_receipt_r1_20260830"
M1128_AUTHOR_ID = (
    "b4bd360904e99dc8b0457d3d07ead95ad0f529e96e73d2f3d3f1bc2fd8dc0300",
    "248f908a0f9662dc5836cce8e447cbd6758dbd41d82e2c6704cc65d98be49b9d",
    "ccb5ac5836271577c95021d2afee63aa8300a873771a2e49eafabb0e439babd0",
)
M1129 = HW / "reviews/m1129c_m1128c_c1_weight_service_addressed_ledger_static_hammer_r1_20260830"
M1129_ID = (
    "39a08dda6c6f0d33576e68a1b86b96e0f7ed371f6f3586559dcb11321aef1712",
    "5a67e10512ad05710775a194acc9f38c6d69f2af2d943da74156cbb821a04300",
    "0c02afad88261a3bfd09d191741d168629b20e370d93e67944ad9ff8add79f31",
)
M1102_SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1102_RESULT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_RESULT_SHA = "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91"
M1102_RESULT_OUTER = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/.m1102_atomic_seal/SHA256SUMS.seal.sha256"
M1102_RESULT_OUTER_SHA = "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f"
M1016_SOURCE = HERE / "run_m1016_c1_full_matched_address_replay.py"
M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1056_SOURCE = HERE / "run_m1056_c1_exact_1rw_arbitration_replay_source.py"
M1056_SOURCE_SHA = "95e276a7afe7a049faa2b967bed1431762c72a5e0b521c3e9857121ece5c816f"
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1000_ID = (
    "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
    "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
    "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5",
)
CONTRACT = HW / "contracts/m1130c_c1_internal_weight_service_refill_instrumentation_source_contract_r1_20260830.json"
CONTRACT_SHA = "20ff9026f8dbc25ad0e9813107a6e97a96f1e379244dcb26ffb51d3a972bcfab"
CONTRACT_OUTER_SHA = "efc4bc08d3634531b99c1e45d1ce20c362bb5ca74249d9f2a6877b857af9352a"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
NATIVE_SLICES = 24
NATIVE_DEPTH = 128
SLICE_BYTES = 16
ROWS_PER_HALF = 16


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == digest,
            "regular-file identity drift: " + str(path))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed authority identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        require(relative not in listed and member.is_file() and not member.is_symlink() and
                sha256(member) == expected, "sealed member drift: " + relative)
        listed.add(relative)
    expected, relative = outer.read_text(encoding="utf-8").split()
    require(relative == "SHA256SUMS" and expected == sha256(manifest),
            "outer seal drift: " + directory.name)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    node = next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    require(node is not None, "missing function: " + name)
    return node


def method_node(source: str, class_name: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    klass = next((item for item in tree.body
                  if isinstance(item, ast.ClassDef) and item.name == class_name), None)
    require(klass is not None, "missing class: " + class_name)
    node = next((item for item in klass.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    require(node is not None, "missing method: " + class_name + "." + name)
    return node


def audit_frozen_internal_event_point() -> dict[str, Any]:
    """Inspect producer code only; do not create a canonical row reader."""
    verify_regular(M1128_SOURCE, M1128_SOURCE_SHA)
    verify_regular(M1128_CONTRACT, M1128_CONTRACT_SHA)
    verify_regular(Path(str(M1128_CONTRACT) + ".sha256.seal.sha256"),
                   M1128_CONTRACT_OUTER_SHA)
    verify_flat(M1128_AUTHOR, M1128_AUTHOR_ID)
    verify_flat(M1129, M1129_ID)
    verify_regular(M1102_SOURCE, M1102_SOURCE_SHA)
    verify_regular(M1102_RESULT, M1102_RESULT_SHA)
    verify_regular(M1102_RESULT_OUTER, M1102_RESULT_OUTER_SHA)
    verify_regular(M1016_SOURCE, M1016_SOURCE_SHA)
    verify_regular(M1056_SOURCE, M1056_SOURCE_SHA)
    verify_flat(M1000, M1000_ID)
    verify_regular(CONTRACT, CONTRACT_SHA)
    verify_regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), CONTRACT_OUTER_SHA)
    verify_regular(DOCS359, DOCS359_SHA)

    m1016_text = M1016_SOURCE.read_text(encoding="utf-8")
    common = function_node(m1016_text, "common_receipt")
    weight = method_node(m1016_text, "PackingAudit", "weight_task")
    run_full = function_node(m1016_text, "run_full")
    common_text = ast.get_source_segment(m1016_text, common) or ast.unparse(common)
    weight_text = ast.get_source_segment(m1016_text, weight) or ast.unparse(weight)
    run_text = ast.get_source_segment(m1016_text, run_full) or ast.unparse(run_full)
    require([item.arg for item in weight.args.args] == ["self", "start", "beats", "half_slot"] and
            "self.weight_runs.append((start, end, half_slot))" in weight_text and
            'receipt["counts"]["weight"]' in run_text and "index & 1" in run_text,
            "M1016 aggregate weight interval point drift")
    forbidden = ("native_slices", "logical_bank", "local_row", "byte_enable",
                 "native_macro_activations", "service_event_exact_once_id")
    require(not any(token in common_text or token in weight_text for token in forbidden),
            "M1016 unexpectedly has native event fields")
    m1056_text = M1056_SOURCE.read_text(encoding="utf-8")
    nominal = function_node(m1056_text, "nominal_task_events")
    nominal_text = ast.get_source_segment(m1056_text, nominal) or ast.unparse(nominal)
    require("packed_address" in nominal_text and "for bank in range(BLOCKS)" in nominal_text and
            "weight" not in nominal_text.lower(),
            "M1056 native event scope is no longer psum-only")
    return {
        "schema": "m1130c_frozen_internal_weight_event_point_audit_v1",
        "status": "STOP_UPSTREAM_HAS_AGGREGATE_WEIGHT_INTERVAL_NOT_PER_BEAT_ADDRESSED_EVENT",
        "m1016_weight_task_arguments": ["start", "beats", "half_slot"],
        "m1016_weight_task_state": ["weight_runs(start,end,half_slot)",
                                    "aggregate_overlap_count",
                                    "aggregate_cross_half_overlap_count"],
        "m1056_m1102_port_events": "psum only",
        "real_internal_weight_service_refill_event_available": False,
        "minimum_missing_upstream_fields": [
            "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
            "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
            "bytes", "byte_enable_per_slice", "native_macro_activations",
            "service_beat_ordinal", "store_transaction_ordinal",
            "service_event_exact_once_id", "source_row_provenance_sha256",
        ],
        "aggregate_expansion_allowed": False,
        "canonical_row_reader_opened": False,
        "canonical_rows_read": 0,
        "canonical_events_emitted": 0,
        "canonical_ready": False,
    }


def exact_once_id(axis: str, task_id: int, source_local_ordinal: int,
                  service_beat_ordinal: int, store_transaction_ordinal: int) -> str:
    payload = (f"m1130c:{axis}:{task_id}:{source_local_ordinal}:"
               f"{service_beat_ordinal}:{store_transaction_ordinal}")
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class InternalWeightServiceRefillEvent:
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
    service_beat_ordinal: int | None
    store_transaction_ordinal: int
    service_event_exact_once_id: str | None
    source_row_provenance_sha256: str

    def validate(self) -> None:
        require(self.axis in AXES and type(self.task_id) is int and self.task_id >= 0 and
                type(self.source_local_ordinal) is int and self.source_local_ordinal >= 0 and
                type(self.requested_cycle) is int and self.requested_cycle >= 0 and
                self.op in ("READ", "WRITE") and
                self.logical_bank == self.half_slot and self.half_slot in (0, 1) and
                type(self.logical_row) is int and 0 <= self.logical_row < ROWS_PER_HALF and
                self.local_row == self.half_slot * ROWS_PER_HALF + self.logical_row and
                self.native_slices == tuple(sorted(set(self.native_slices))) and
                all(0 <= value < NATIVE_SLICES for value in self.native_slices) and
                self.bytes == len(self.native_slices) * SLICE_BYTES and
                self.byte_enable_per_slice == (0xffff,) * len(self.native_slices) and
                self.native_macro_activations == len(self.native_slices) and
                type(self.store_transaction_ordinal) is int and self.store_transaction_ordinal >= 0 and
                len(self.source_row_provenance_sha256) == 64 and
                all(char in "0123456789abcdef" for char in self.source_row_provenance_sha256),
                "internal weight event schema/mapping drift")
        if self.op == "WRITE":
            require(type(self.service_beat_ordinal) is int and self.service_beat_ordinal >= 0 and
                    len(self.native_slices) == 8 and self.bytes == 128 and
                    min(self.native_slices) % 8 == 0 and
                    self.native_slices == tuple(range(min(self.native_slices),
                                                      min(self.native_slices) + 8)) and
                    self.service_event_exact_once_id == exact_once_id(
                        self.axis, self.task_id, self.source_local_ordinal,
                        self.service_beat_ordinal, self.store_transaction_ordinal),
                    "internal refill event exact-once identity drift")
        else:
            require(self.service_beat_ordinal is None and
                    self.service_event_exact_once_id is None and
                    self.native_slices == tuple(range(24)) and self.bytes == 384,
                    "internal full-read event drift")


def instrument_real_event_inputs(
        events: Iterable[InternalWeightServiceRefillEvent]) -> list[Any]:
    """Consume producer-supplied events; never synthesize from aggregate receipts."""
    m1128 = load_module(M1128_SOURCE, "m1130c_frozen_m1128")
    rows = []
    seen_exact_ids = set()
    expected_beats = {axis: set() for axis in AXES}
    seen_transaction_ordinals = set()
    for event in events:
        require(type(event) is InternalWeightServiceRefillEvent,
                "exact internal event type required")
        event.validate()
        transaction_identity = (event.axis, event.task_id, event.source_local_ordinal,
                                event.store_transaction_ordinal)
        require(transaction_identity not in seen_transaction_ordinals,
                "duplicate producer transaction identity")
        seen_transaction_ordinals.add(transaction_identity)
        if event.op == "WRITE":
            require(event.service_event_exact_once_id not in seen_exact_ids,
                    "duplicate producer service exact-once identity")
            seen_exact_ids.add(event.service_event_exact_once_id)
            require(event.service_beat_ordinal not in expected_beats[event.axis],
                    "duplicate producer service beat")
            expected_beats[event.axis].add(int(event.service_beat_ordinal))
        rows.append(m1128.AddressedWeightTransaction(
            event.axis, event.requested_cycle, event.requested_cycle, 0, event.op,
            event.logical_bank, event.half_slot, event.logical_row, event.local_row,
            event.native_slices, event.bytes, event.byte_enable_per_slice,
            event.native_macro_activations, event.service_beat_ordinal,
            event.store_transaction_ordinal, event.task_id, event.source_local_ordinal,
            event.source_row_provenance_sha256))
    require(rows, "empty internal event stream")
    scheduled = m1128.schedule_native_one_rw(rows)
    summary = m1128.validate_exact_once_and_conflicts(scheduled, expected_beats)
    require(summary["service_beats_expected"] == len(seen_exact_ids),
            "producer exact-once ID and service beat conservation drift")
    return scheduled


def iter_canonical_internal_weight_service_refill_events() -> Iterator[dict[str, Any]]:
    """Future canonical interface; STOP before rows because producer hook is absent."""
    audit = audit_frozen_internal_event_point()
    require(audit["canonical_ready"] is True,
            "STOP: real M1102/M1016 lacks per-beat addressed weight event producer hook")
    if False:  # pragma: no cover
        yield {}


def source_small_oracle() -> dict[str, Any]:
    audit = audit_frozen_internal_event_point()
    digest = "0" * 64
    events = []
    for axis_id, axis in enumerate(AXES):
        base = axis_id * 3
        beat0, beat1 = axis_id * 2, axis_id * 2 + 1
        events.extend((
            InternalWeightServiceRefillEvent(
                axis, 0, 0, 5, "WRITE", 0, 0, 0, 0, tuple(range(8)), 128,
                (0xffff,) * 8, 8, beat0, base,
                exact_once_id(axis, 0, 0, beat0, base), digest),
            InternalWeightServiceRefillEvent(
                axis, 0, 1, 5, "WRITE", 1, 1, 0, 16, tuple(range(8)), 128,
                (0xffff,) * 8, 8, beat1, base + 1,
                exact_once_id(axis, 0, 1, beat1, base + 1), digest),
            InternalWeightServiceRefillEvent(
                axis, 0, 2, 7, "READ", 0, 0, 0, 0, tuple(range(24)), 384,
                (0xffff,) * 24, 24, None, base + 2, None, digest),
        ))
    scheduled = instrument_real_event_inputs(events)
    summary = {
        "events": len(scheduled),
        "writes": sum(row.op == "WRITE" for row in scheduled),
        "reads": sum(row.op == "READ" for row in scheduled),
        "unique_exact_once_write_ids": len({event.service_event_exact_once_id
                                             for event in events if event.op == "WRITE"}),
        "explicitly_stalled_transactions": sum(row.stall_cycles > 0 for row in scheduled),
        "final_native_1rw_conflicts": 0,
        "final_weight_half_slot_overlaps": 0,
    }
    require(summary == {
        "events": 9, "writes": 6, "reads": 3, "unique_exact_once_write_ids": 6,
        "explicitly_stalled_transactions": 3,
        "final_native_1rw_conflicts": 0, "final_weight_half_slot_overlaps": 0,
    }, "bounded direct-event instrumentation drift")
    stopped = False
    try:
        next(iter_canonical_internal_weight_service_refill_events())
    except Failure:
        stopped = True
    require(stopped, "canonical iterator escaped absent producer hook")
    return {
        "schema": "m1130c_internal_weight_service_refill_instrumentation_small_oracle_v1",
        "status": "PASS_SYNTHETIC_DIRECT_EVENT_INSTRUMENTATION__CANONICAL_STOP",
        "frozen_event_point_audit": audit,
        "synthetic": summary,
        "canonical_iterator_stopped_before_row_open": True,
        "full_51840000_replayed": False,
        "eda_rtl_gpu_remote_executed": False,
    }


def main() -> None:
    require(sys.argv == [str(Path(sys.argv[0]))] or sys.argv[1:] == ["--self-test"],
            "only zero-argument or --self-test bounded source is allowed")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
